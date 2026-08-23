import unittest
from tinygrad import Device
from tinygrad.helpers import colored, DEBUG
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.amd.dsl import s, OPERANDS
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna3.enum as e3
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import TARGET_TO_ARCH, llvm_disasm, get_mattr, capture_runs, capture_emu, sram_scope
from test.amd.helpers import times_of, execs_of, insts_of

assert "MOCK" not in type(Device["AMD"].iface).__name__, "needs real hardware, the emulator is under test"
assert TARGET_TO_ARCH[Device["AMD"].arch] == "rdna3", "only rdna3"

# Each pipe (SALU, VALU, LDS, VMEM) is a FIFO queue feeding a worker.
#   dispatch  the wave put the instruction in the queue
#   exec      the worker took it out and started on it

def _kernel(name:str, insts:list):
  def fxn(A:UOp) -> UOp:
    threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
    sink = UOp.sink(A.flatten().base, threads, wg, arg=KernelInfo(name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  return fxn

SWEEP_REPEATS, SWEEP_BLOCKS = 36, 2
assert SWEEP_REPEATS + 1 <= 64, "kernel would outrun the prefetcher"
_SWEEP_SRC, _SWEEP_DST = 4, 8
assert _SWEEP_DST + 2*SWEEP_REPEATS <= 104, f"SWEEP_REPEATS={SWEEP_REPEATS} needs more sgprs than exist" 

_UNSAFE = ("PC", "SAVEEXEC", "SETREG", "GETREG", "BRANCH", "CALL", "RFE", "ENDPGM", "TRAP",
           "SENDMSG", "SLEEP", "BARRIER", "WAITCNT", "NOP", "HALT", "PRIO", "ICACHE", "TTRACE",
           "WAKEUP", "PERFLEVEL", "VERSION", "CLAUSE", "DELAY", "WAIT", "MSG")

def _sweep_inst(op, i:int):
  kwargs, sreg = {}, _SWEEP_SRC
  for field, (_fmt, width, kind) in OPERANDS[op].items():
    if field == "simm16": kwargs[field] = 1
    elif kind.name == "OPR_SDST":
      kwargs[field] = s[_SWEEP_DST+2*i:_SWEEP_DST+1+2*i] if width == 64 else s[_SWEEP_DST+i]
    else:
      kwargs[field] = s[sreg:sreg+1] if width == 64 else s[sreg]
      sreg += 2 if width == 64 else 1
  return getattr(r3, op.name.lower())(**kwargs)

def _sweep_ok(op, target:str) -> bool:
  if OPERANDS.get(op) is None or any(u in op.name for u in _UNSAFE): return False
  if not hasattr(r3, name:=op.name.lower()): return False
  try:
    inst = _sweep_inst(op, 0)
    if repr(decode_inst(inst.to_bytes(), "rdna3")) != repr(inst): return False
    return llvm_disasm(inst.to_bytes(), target, get_mattr("rdna3"))[0].split()[0] == name
  except (TypeError, ValueError, KeyError, IndexError): return False  # opcode we cannot build or decode

def sweep_ops(target:str, en) -> list: return sorted((m for m in en if _sweep_ok(m, target)), key=lambda m: m.name)

def _diff_row(vals:list, ref:list) -> str:
  return "[" + ", ".join(colored(str(v), "green" if i < len(ref) and ref[i] == v else "red") for i, v in enumerate(vals)) + "]"

class TestSIMDModel(unittest.TestCase):
  def _sweep(self, en):
    fails = []
    for op in sweep_ops(Device["AMD"].arch, en):
      name = op.name.lower()
      kname = f"custom_salu_{name}"
      block = [_sweep_inst(op, i) for i in range(SWEEP_REPEATS)] + [s_endpgm()]
      projs, lib, arch = capture_runs(_kernel(kname, block), SWEEP_BLOCKS)
      try: emu, err = sram_scope(capture_emu(block), lib, arch, 0), None
      except Exception as e: emu, err = None, repr(e)
      was = len(fails)
      if any(p != projs[0] for p in projs[1:]): fails.append(f"{name}: hardware trials disagree with each other")
      if emu is None: fails.append(f"{name}: emulator produced no trace, {err}")
      elif insts_of(emu) != insts_of(projs[0]): fails.append(f"{name}: emulator executed different instructions")
      elif (times_of(emu), execs_of(emu)) != (times_of(projs[0]), execs_of(projs[0])):
        fails.append(f"{name}: emulator timing differs from hardware")
      if (ok := len(fails) == was) and DEBUG < 1: continue
      print(f"\n  **** {name}")
      if emu is None: print("    " + colored(f"emu  no trace: {err}", "red"))
      ref_rows = {}
      for b, proj in enumerate(projs + ([emu] if emu else [])):
        label = "emu" if b == len(projs) else f"#{b}"
        blk = [(t, e) for t, e, _, _op in proj]
        disp = [y[0]-x[0] for x, y in zip(blk, blk[1:])]
        gaps = [y[1]-x[1] for x, y in zip(blk, blk[1:]) if x[1] is not None and y[1] is not None]
        t0 = blk[0][0]
        rows = {"dispatch": [t - t0 for t, _ in blk], "exec": [None if e is None else e - t0 for _, e in blk],
                "dispatch_to_exec": [None if e is None else e - t for t, e in blk],
                "dispatch gaps": disp, "exec gaps": gaps}
        if b == 0: ref_rows = rows
        print(f"    {label if label != 'emu' else colored(label, 'green' if ok else 'red')}")
        for k, v in rows.items(): print(f"        {k:<16} " + (str(v) if label != "emu" else _diff_row(v, ref_rows[k])))
    self.assertFalse(fails, f"{len(fails)} opcodes disagree with the emulator or with themselves:\n" + "\n".join(fails))

  def test_sop1(self): self._sweep(e3.SOP1Op)
  def test_sop2(self): self._sweep(e3.SOP2Op)
  def test_sopc(self): self._sweep(e3.SOPCOp)
  def test_sopk(self): self._sweep(e3.SOPKOp)

if __name__ == "__main__":
  unittest.main()

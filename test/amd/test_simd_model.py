import unittest
from tinygrad import Device
from tinygrad.helpers import colored, DEBUG
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.amd.dsl import s, v, M0, OPERANDS
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna3.enum as e3
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import TARGET_TO_ARCH, SIMDS, capture_runs, sram_scope, llvm_disasm, get_mattr
import ctypes, test.mockgpu.amd.emu as emu

assert "MOCK" not in type(Device["AMD"].iface).__name__, "needs real hardware, the emulator is under test"
assert TARGET_TO_ARCH[Device["AMD"].arch] == "rdna3", "only rdna3"
_MCPU, _MATTR = Device["AMD"].arch, get_mattr(TARGET_TO_ARCH[Device["AMD"].arch])

# Each pipe (SALU, VALU, LDS, VMEM) is a FIFO queue feeding a worker.
#   dispatch  the wave put the instruction in the queue
#   exec      the worker took it out and started on it

def _kernel(name:str, insts:list):
  def fxn(A:UOp) -> UOp:
    threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
    sink = UOp.sink(A.flatten().base, threads, wg, arg=KernelInfo(name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  return fxn

SWEEP_REPEATS = 30
PREFETCH_BYTES = 256  # the prefetcher runs 3 cache lines ahead of the current one, past that the sweep stalls mid kernel
_SWEEP_SRC, _SWEEP_DST, _VSWEEP_SRC, _VSWEEP_DST = 4, 8, 0, 4
assert _SWEEP_DST + 2*SWEEP_REPEATS <= 104, f"SWEEP_REPEATS={SWEEP_REPEATS} needs more sgprs than exist"
assert _VSWEEP_DST + 4*SWEEP_REPEATS <= 256, f"SWEEP_REPEATS={SWEEP_REPEATS} needs more vgprs than exist"

_UNSAFE = ("PC", "SAVEEXEC", "SETREG", "GETREG", "BRANCH", "CALL", "RFE", "ENDPGM", "TRAP",
           "SENDMSG", "SLEEP", "BARRIER", "WAITCNT", "NOP", "HALT", "PRIO", "ICACHE", "TTRACE",
           "WAKEUP", "PERFLEVEL", "VERSION", "CLAUSE", "DELAY", "WAIT", "MSG")
_UNSAFE_SALU = _UNSAFE + ("F16", "F32", "F64")  # gfx1102 has no scalar float unit, these hang the queue

def _regs(bank, base:int, width:int): return bank[base] if width <= 32 else bank[base:base+width//32-1]

def _sweep_inst(op, i:int, vec:bool=False):
  kwargs, sreg, vreg = {}, _SWEEP_SRC, _VSWEEP_SRC
  for field, (_fmt, width, kind) in OPERANDS[op].items():
    name = "src0" if field == "vsrc0" else field
    if field == "simm16": kwargs[field] = 1
    elif vec and kind.name == "OPR_SREG": kwargs[name] = s[_SWEEP_DST+2*i]  # vop3 encodes one index, the pair is implicit
    elif kind.name == "OPR_SDST":
      kwargs[field] = s[_SWEEP_DST+2*i:_SWEEP_DST+1+2*i] if width == 64 else s[_SWEEP_DST+i]
    elif vec and field.endswith("dst"): kwargs[field] = _regs(v, _VSWEEP_DST+4*i, width)
    elif vec:
      kwargs[name] = _regs(v, vreg, width)
      vreg += max(1, width//32)
    else:
      kwargs[field] = _regs(s, sreg, width)
      sreg += max(1, width//32)
  return getattr(r3, op.name.lower())(**kwargs)

def _sweep_ok(op, vec:bool=False) -> bool:
  if (ops := OPERANDS.get(op)) is None or any(u in op.name for u in (_UNSAFE if vec else _UNSAFE_SALU)): return False
  if any(kind.name == "OPR_EXEC" for _f, (_x, _w, kind) in ops.items()): return False
  inst = _sweep_inst(op, 0, vec)
  return repr(decode_inst(code:=inst.to_bytes(), "rdna3")) == repr(inst) and len(llvm_disasm(code, _MCPU, _MATTR)) == 1

def _diff_row(vals:list, ref:list) -> str:
  return "[" + ", ".join(colored(str(v), "green" if i < len(ref) and ref[i] == v else "red") for i, v in enumerate(vals)) + "]"

class TestSIMDModel(unittest.TestCase):
  def _sweep(self, en, vec:bool=False):
    fails = []
    for op in sorted((m for m in en if _sweep_ok(m, vec)), key=lambda m: m.name):
      name = op.name.lower()
      kname = f"custom_{'valu' if vec else 'salu'}_{name}"
      # m0 feeds the movrel index, leaving it uninitialised would index a vgpr outside the allocation
      block = ([s_mov_b32(M0, 0)] if vec else []) + [_sweep_inst(op, i, vec) for i in range(SWEEP_REPEATS)] + [s_endpgm()]
      assert (nb := sum(i.size() for i in block)) <= PREFETCH_BYTES, f"{name}: {nb} byte kernel outruns the prefetcher"
      projs, lib, arch = capture_runs(_kernel(kname, block))
      code = b"".join(i.to_bytes() for i in block)
      buf, args = (ctypes.c_char * len(code)).from_buffer_copy(code), (ctypes.c_uint64 * 1)(0)
      emu.sqtt_traces.clear()
      assert emu.run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, 32, 1, 1, ctypes.addressof(args)) == 0, "emulator rejected the kernel"
      assert emu.sqtt_traces, "emulator produced no SQTT trace, is PROFILE=1 set?"
      emu_proj = sram_scope(emu.sqtt_traces[0], lib, arch)
      was = len(fails)
      if any(p != projs[0] for p in projs[1:]): fails.append(f"{name}: the two simds disagree with each other")
      if emu_proj != projs[0]:
        what = "instructions" if [r[2:] for r in emu_proj] != [r[2:] for r in projs[0]] else "timing"
        fails.append(f"{name}: emulator {what} differs from hardware")
      if (ok := len(fails) == was) and DEBUG < 1: continue
      print(f"\n  **** {name}")
      for b, proj in enumerate(projs + [emu_proj]):
        is_emu = b == len(projs)
        blk = [(t, e) for t, e, _, _op in proj]
        disp = [y[0]-x[0] for x, y in zip(blk, blk[1:])]
        gaps = [y[1]-x[1] for x, y in zip(blk, blk[1:]) if x[1] is not None and y[1] is not None]
        t0 = blk[0][0]
        rows = {"dispatch": [t - t0 for t, _ in blk], "exec": [None if e is None else e - t0 for _, e in blk],
                "dispatch_to_exec": [None if e is None else e - t for t, e in blk],
                "dispatch gaps": disp, "exec gaps": gaps}
        if b == 0: ref_rows = rows
        print("    " + (colored("emu", "green" if ok else "red") if is_emu else f"simd{SIMDS[b]}"))
        for k, v in rows.items(): print(f"        {k:<16} " + (_diff_row(v, ref_rows[k]) if is_emu else str(v)))
    self.assertFalse(fails, f"{len(fails)} opcodes disagree with the emulator or with themselves:\n" + "\n".join(fails))

  def test_sop1(self): self._sweep(e3.SOP1Op)
  def test_sop2(self): self._sweep(e3.SOP2Op)
  def test_sopc(self): self._sweep(e3.SOPCOp)
  def test_sopk(self): self._sweep(e3.SOPKOp)

  # vopc is swept through its vop3 encoding, the e32 form writes vcc implicitly and so serialises on itself
  def test_vop1(self): self._sweep(e3.VOP1Op, vec=True)
  def test_vop2(self): self._sweep(e3.VOP2Op, vec=True)
  def test_vop3(self): self._sweep(e3.VOP3Op, vec=True)

if __name__ == "__main__":
  unittest.main()

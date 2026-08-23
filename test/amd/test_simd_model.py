# Tests of our mental model of one RDNA3 SIMD against real hardware.
#
# These are not regression tests for tinygrad. The sweep runs one kernel per opcode on hardware and
# on the emulator and requires the two traces to be identical, so the model of the SIMD lives in the
# emulator rather than being restated here. They need a real AMD GPU and skip everywhere else, so
# they never run in CI.
#
# A failure here means the emulator or the model is wrong, not that the test is. Do not loosen an
# assertion to make it pass: fix the emulator, or change the model and record what refuted it.
#
# The model: each pipe (SALU, VALU, LDS, VMEM) is a FIFO queue feeding a worker.
#   dispatch  the wave put the instruction in the queue      SQTT INST / VALUINST packet
#   exec      the worker took it out and started on it       SQTT ALUEXEC / VMEMEXEC packet
# There is no completion event, so nothing here can measure how long an instruction runs for.
# The scope is deliberately narrow: one opcode per kernel, repeated, nothing else in flight. Kernel
# shapes the sweep cannot reach (a lone instruction, an idle queue mid-wave, mixed opcodes) are not
# covered and want their own kernels when the time comes.
import unittest
from tinygrad import Device
from tinygrad.helpers import colored
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.amd.dsl import s, OPERANDS
from tinygrad.renderer.amd.sqtt import map_insts, ALUEXEC
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna3.enum as e3
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import TARGET_TO_ARCH, llvm_disasm, get_mattr, capture_runs, capture_emu, sram_scope
from test.amd.helpers import times_of, execs_of, insts_of

assert "MOCK" not in type(Device["AMD"].iface).__name__, "needs real hardware, the emulator is under test"
assert TARGET_TO_ARCH[Device["AMD"].arch] == "rdna3", "only rdna3"

# dispatch_to_exec is the cycles from an instruction being enqueued to its ALUEXEC packet.
# dispatch_to_exec and initiation interval are composed from properties of the opcode, not tabulated
# per opcode. measured on gfx1102 over every SALU opcode the device implements:
#   dispatch_to_exec = 2, +2 if the instruction needs a register read that is not a plain operand
#              (its own destination, or an M0 indexed source), +1 if it multiplies
#   interval         = 2 if it multiplies, or if it reads two sgpr sources that each fit in one register;
#                      otherwise 1
# the s_pack family fixes what "fits in one register" means: its sources are 16 bit fields, but each
# still costs a whole register read, and it measures 2 like the 32 bit two source opcodes.
# the 64 bit siblings are the surprise: s_and_b64 sustains 1/cycle where s_and_b32 sustains 1 per 2,
# and the same holds for or/xor/nand/nor/xnor/lshl/lshr/ashr/bfe/cmp/cselect. s_bfm_b64 identifies
# the mechanism: 64 bit destination but 32 bit sources, and it reads 2 like the 32 bit ops. so the
# cost is in the source reads, not the width of the result.
# s_movk_i32 is the control for the +2 dispatch_to_exec term: it writes sdst without reading it, and is the
# only SOPK opcode at dispatch_to_exec 2.
# OPERANDS marks sdst the same whether it is read or written, so the extra-read set is listed.
_EXTRA_READ = ("s_cmpk_", "s_addk_", "s_mulk_", "s_cmovk_", "s_cmov_", "s_bitset0_", "s_bitset1_", "s_movrels")

def salu_timing(op) -> tuple[int, int]:
  srcs = [w for _f, (_fmt, w, k) in OPERANDS[op].items() if k.name == "OPR_SSRC"]
  extra, mul = (n:=op.name.lower()).startswith(_EXTRA_READ), "_mul" in n
  interval = 2 if (mul or (len(srcs) == 2 and max(srcs) <= 32)) else 1
  return 2 + 2*extra + mul, interval


def _kernel(name:str, insts:list):
  def fxn(A:UOp) -> UOp:
    threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
    sink = UOp.sink(A.flatten().base, threads, wg, arg=KernelInfo(name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  return fxn

# ── SALU sweep ───────────────────────────────────────────────────────────────────────────────────
# One kernel per instruction, holding nothing but that instruction SWEEP_REPEATS times, each writing
# a different destination so the repeats are independent. The wave starts with an empty queue, so the
# first one measures dispatch_to_exec; the spacing between their execs measures how fast the worker takes new
# work. No drain is needed: the kernel is the block.
# a wave prefetches at most 3 instruction cache lines (3*64 bytes) ahead of the pc plus the line it
# is on, so 256 bytes = 64 instructions of straight line code run before it catches the prefetcher
# and stalls ~235 cycles for a real fetch. keep the whole kernel under that and the stall never
# happens, instead of landing mid block and corrupting whichever repeat it hits.
SWEEP_REPEATS, SWEEP_BLOCKS, SWEEP_RAMP = 36, 2, 10
assert SWEEP_REPEATS + 1 <= 64, "kernel would outrun the prefetcher"
# sources sit below the destinations and are shared by every repeat. putting them above instead
# makes the base scale with SWEEP_REPEATS and silently run off the end of the sgpr file. an opcode
# reads at most two 64 bit sources, so four registers cover every one of them, and they start above
# s[0:1] to leave the kernarg pointer alone. 64 bit destinations take two registers each, and
# s[0]..s[105] is the whole file, of which the top pair is left free for vcc.
_SWEEP_SRC, _SWEEP_DST = 4, 8
assert _SWEEP_DST + 2*SWEEP_REPEATS <= 104, f"SWEEP_REPEATS={SWEEP_REPEATS} needs more sgprs than exist" 

# every scalar ALU opcode except the ones that would not come back: anything touching the PC, EXEC,
# hardware registers, or wave state. SOPP is control flow only and emits no exec packet.
_UNSAFE = ("PC", "SAVEEXEC", "SETREG", "GETREG", "BRANCH", "CALL", "RFE", "ENDPGM", "TRAP",
           "SENDMSG", "SLEEP", "BARRIER", "WAITCNT", "NOP", "HALT", "PRIO", "ICACHE", "TTRACE",
           "WAKEUP", "PERFLEVEL", "VERSION", "CLAUSE", "DELAY", "WAIT", "MSG")
# destinations walk upward from _SWEEP_DST so the repeats do not depend on each other
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

# keep an opcode only if this device really implements it. tinygrad's rdna3 enum is a union over
# gfx11.0 and gfx11.5, so it contains scalar float ops (s_add_f32, s_cvt_*, ...) that gfx1102 has no
# unit for: executing one raises sq_intr ILLEGAL_INST and hangs the queue. LLVM knows per target.
# also require tinygrad to decode it back, since amd_decode must disassemble the whole kernel.
def _sweep_ok(op, target:str) -> bool:
  if OPERANDS.get(op) is None or any(u in op.name for u in _UNSAFE): return False
  if not hasattr(r3, name:=op.name.lower()): return False
  try:
    inst = _sweep_inst(op, 0)
    if repr(decode_inst(inst.to_bytes(), "rdna3")) != repr(inst): return False
    return llvm_disasm(inst.to_bytes(), target, get_mattr("rdna3"))[0].split()[0] == name
  except (TypeError, ValueError, KeyError, IndexError): return False  # opcode we cannot build or decode

def sweep_ops(target:str, en) -> list: return sorted((m for m in en if _sweep_ok(m, target)), key=lambda m: m.name)

def _sweep_block(op) -> list: return [_sweep_inst(op, i) for i in range(SWEEP_REPEATS)]

# one row of the emulator's trace against hardware's, green where they agree
def _diff_row(vals:list, ref:list) -> str:
  return "[" + ", ".join(colored(str(v), "green" if i < len(ref) and ref[i] == v else "red") for i, v in enumerate(vals)) + "]"

class TestSIMDModel(unittest.TestCase):
  # every SALU instruction should show the same dispatch_to_exec on an idle queue and the same
  # initiation interval back to back. anything new is a discovery.
  # refuted by: an opcode whose dispatch_to_exec or interval is not what salu_timing() says
  def _sweep(self, en):
    fails = []
    for op in sweep_ops(Device["AMD"].arch, en):
      name = op.name.lower()
      # one kernel holding one block, dispatched SWEEP_BLOCKS times, so every trial runs the same
      # code at the same offset. the block is short enough that the prefetcher never runs dry.
      # every trial is kept. a handful of opcodes read dispatch_to_exec 4 on trial 0 and 2 on the rest, and
      # which ones they are changes between runs, so the first dispatch is not reproducible the way
      # everything else here is. that is a finding, not noise to drop: do not add a warmup dispatch
      # to hide it.
      # TODO: explain it. a cold first dispatch would slow every opcode, not five of them, so the
      # cause is something that varies per run. compare the full packet stream of trial 0 against
      # trial 1 for one affected opcode.
      kname, block = f"custom_salu_{name}", _sweep_block(op) + [s_endpgm()]
      projs, raw, lib, arch, simd = capture_runs(_kernel(kname, block), kname, SWEEP_BLOCKS)
      n_exec = sum(isinstance(p, ALUEXEC) for p, _ in map_insts(raw[0][0], lib, arch, simd))
      print(f"\n  **** {name}" + (f"   <- {n_exec} ALUEXEC for {SWEEP_REPEATS} instructions, pairing unreliable"
                                   if n_exec != SWEEP_REPEATS else ""))
      # the emulator runs the same instructions in this process and joins as the last trial. it is
      # where the model lives, so the sweep does not restate it here: hardware and emulator either
      # emit the same trace or they do not.
      try: emu = sram_scope(capture_emu(block), lib, arch, 0)
      except Exception as e:  # no pcode for this opcode, or it faulted
        emu, err = None, repr(e)
        print("    " + colored(f"emu  no trace: {err}", "red"))
      seen, want, ref_rows = set(), salu_timing(op), {}
      for b, proj in enumerate(projs + ([emu] if emu else [])):
        label = "emu" if b == len(projs) else f"#{b}"
        blk = [(t, e) for t, e, _, _op in proj]
        # the emulator's own count is checked below, against hardware, not against the model
        if label != "emu": self.assertEqual(len(blk), SWEEP_REPEATS, f"{name} {label}: unexpected instruction count")
        dispatch_to_exec = None if blk[0][1] is None else blk[0][1] - blk[0][0]
        disp = [y[0]-x[0] for x, y in zip(blk, blk[1:])]
        gaps = [y[1]-x[1] for x, y in zip(blk, blk[1:]) if x[1] is not None and y[1] is not None]
        # some opcodes run their first few at the normal rate before settling, so read the interval
        # off the tail rather than off the ramp.
        tail = gaps[SWEEP_RAMP:]
        interval = max(set(tail), key=tail.count) if tail else None
        if label != "emu": seen.add((dispatch_to_exec, interval))
        # absolute cycles first, then the gaps between them. dispatch_to_exec is the vertical
        # distance between the two rows, so it is the first exec time once both are anchored on 0.
        t0 = blk[0][0]
        rows = {"dispatch": [t - t0 for t, _ in blk], "exec": [None if e is None else e - t0 for _, e in blk],
                "dispatch_to_exec": [None if e is None else e - t for t, e in blk],
                "dispatch gaps": disp, "exec gaps": gaps}
        # hardware trial 0 is the reference the emulator has to reproduce, so its rows print plain
        # and the emulator's print green where they agree and red where they do not
        if b == 0: ref_rows = rows
        hit = label != "emu" or all(rows[k] == ref_rows.get(k) for k in rows)
        print(f"    {label if label != 'emu' else colored(label, 'green' if hit else 'red')}")
        for k, v in rows.items(): print(f"        {k:<16} " + (str(v) if label != "emu" else _diff_row(v, ref_rows.get(k, []))))
      if len(seen) > 1: fails.append(f"{name}: unstable across trials, {sorted(seen)}")
      elif seen != {want}: fails.append(f"{name}: {seen.pop()}, expected {want}")
      # trial 0 is the reference: whatever hardware did, the emulator has to reproduce exactly
      if emu is None: fails.append(f"{name}: emulator produced no trace, {err}")
      elif insts_of(emu) != insts_of(projs[0]): fails.append(f"{name}: emulator executed different instructions")
      elif (times_of(emu), execs_of(emu)) != (times_of(projs[0]), execs_of(projs[0])):
        fails.append(f"{name}: emulator timing differs from hardware")
    self.assertFalse(fails, f"{len(fails)} opcodes disagree with salu_timing() or with the emulator:\n" + "\n".join(fails))

  def test_sop1(self): self._sweep(e3.SOP1Op)
  def test_sop2(self): self._sweep(e3.SOP2Op)
  def test_sopc(self): self._sweep(e3.SOPCOp)
  def test_sopk(self): self._sweep(e3.SOPKOp)

if __name__ == "__main__":
  unittest.main()

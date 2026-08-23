# Tests of our mental model of one RDNA3 SIMD against real hardware.
#
# These are not regression tests for tinygrad. Each test states one property of the model in
# RDNA3_representation.md and runs a kernel whose SQTT trace proves or refutes it. They need a real
# AMD GPU and skip everywhere else, so they never run in CI.
#
# A failure here means the model is wrong, not that the code is. Do not loosen an assertion to make
# it pass: change the model, and record what refuted it.
#
# The model under test: each pipe (SALU, VALU, LDS, VMEM) is a FIFO queue feeding a worker.
#   dispatch  the wave put the instruction in the queue      SQTT INST / VALUINST packet
#   exec      the worker took it out and started on it       SQTT ALUEXEC / VMEMEXEC packet
# There is no completion event, so nothing here can measure how long an instruction runs for.
import unittest, os, pickle
from tinygrad import Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.amd.dsl import s, OPERANDS
from tinygrad.renderer.amd.sqtt import map_insts, ALUEXEC
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna3.enum as e3
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import TARGET_TO_ARCH, llvm_disasm, get_mattr, capture_runs, project, times_of, execs_of, insts_of

# dispatch->exec with an empty queue and an idle worker is not one number: the first instruction of
# a wave costs more than a later one. measured on gfx1102 with s_add_i32.
FIRST_INST_CYCLES = 4  # first instruction of the wave
# transit and initiation interval are composed from properties of the opcode, not tabulated per
# opcode. measured on gfx1102 over every SALU opcode the device implements:
#   transit  = 2, +2 if the instruction needs a register read that is not a plain operand (its own
#              destination, or an M0 indexed source), +1 if it multiplies
#   interval = 2 if it multiplies, or if it reads exactly two 32 bit sgpr sources; 8 for the wrexec
#              family; otherwise 1
# the 64 bit siblings are the surprise: s_and_b64 sustains 1/cycle where s_and_b32 sustains 1 per 2,
# and the same holds for or/xor/nand/nor/xnor/lshl/lshr/ashr/bfe/cmp/cselect. s_bfm_b64 identifies
# the mechanism: 64 bit destination but 32 bit sources, and it reads 2 like the 32 bit ops. so the
# cost is in the source reads, not the width of the result.
# s_movk_i32 is the control for the +2 transit term: it writes sdst without reading it, and is the
# only SOPK opcode at transit 2.
# OPERANDS marks sdst the same whether it is read or written, so the extra-read set is listed.
_EXTRA_READ = ("s_cmpk_", "s_addk_", "s_mulk_", "s_cmovk_", "s_cmov_", "s_bitset0_", "s_bitset1_", "s_movrels")

def salu_timing(name:str) -> tuple[int, int]:
  srcs = [w for _f, (_fmt, w, k) in OPERANDS[_SOP_OPS[name]].items() if k.name == "OPR_SSRC"]
  extra, mul = name.startswith(_EXTRA_READ), "_mul" in name
  interval = 8 if "wrexec" in name else 2 if (mul or srcs == [32, 32]) else 1
  return 2 + 2*extra + mul, interval

# the sgpr file is banked and two reads landing in the same bank cost one extra cycle. these read
# their own walking destination alongside the fixed source, so once the destination reaches a
# register congruent to the source they collide, periodically, every 16 registers. that is a
# property of the register allocation in this sweep, not of the opcode, so skip them here.
# TODO: measure the bank count instead of inferring it. 16 is the number that makes the observed
# period come out right for both widths at once, but it was never measured directly: fix the
# destination and sweep the source across s[4]..s[20], which should show the extra cycle at exactly
# one source position per bank. that also settles whether the collision is dest against src or dest
# against something fixed.
# TODO: only these six can hit it, and it is worth understanding why the rest cannot rather than
# only that they do not. the SOPK ones (s_cmpk_, s_addk_, s_mulk_) read sdst but pair it with
# simm16, so there is a single sgpr read and no pair to collide. s_movrels_ reads M0, which is not
# in the banked file. everything with two fixed sources reads s[4] and s[5], never congruent. the
# case this sweep cannot reach: two *sources* 16 apart, which would say whether the conflict is
# about reading two sgprs at all or specifically about reading the destination.
_BANK_CONFLICT = ("s_cmov_b32", "s_cmov_b64", "s_bitset0_b32", "s_bitset0_b64", "s_bitset1_b32", "s_bitset1_b64")

REF = "/tmp/tinygrad_sqtt_ref.pkl"  # hardware traces captured here, replayed by test_cycle_accurate_emu
KERNELS: dict = {}  # name -> builder, so the emulator test can rerun exactly what hardware ran

# register=False for kernels that only make sense on hardware, so test_cycle_accurate_emu does not
# try to replay them: the sweep covers opcodes the emulator has no pcode for
def _kernel(name:str, insts:list, register:bool=True):
  def fxn(A:UOp) -> UOp:
    threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
    sink = UOp.sink(A.flatten().base, threads, wg, arg=KernelInfo(name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  if register: KERNELS[name] = fxn
  return fxn

# one SALU instruction, nothing before it. the queue is empty because the wave just started, so the
# measured dispatch->exec gap is the queue transit floor -- plus any wave launch cost baked in.
custom_salu_single = _kernel("custom_salu_single", [
  s_add_i32(s[10], s[10], 1),
  s_endpgm(),
])

# the same measurement made a second time mid-kernel. s_nop occupies no pipe (it is an IMMEDIATE
# packet with no exec), so by the time the second s_add is dispatched the SALU queue has long
# drained. if this one also shows QUEUE_CYCLES then the gap is queue transit, not wave launch.
custom_salu_after_idle = _kernel("custom_salu_after_idle", [
  s_add_i32(s[10], s[10], 1),
  *[s_nop(0) for _ in range(8)],
  s_add_i32(s[11], s[11], 1),
  s_endpgm(),
])

# ── SALU sweep ───────────────────────────────────────────────────────────────────────────────────
# One kernel per instruction, holding nothing but that instruction SWEEP_REPEATS times, each writing
# a different destination so the repeats are independent. The wave starts with an empty queue, so the
# first one measures transit; the spacing between their execs measures how fast the worker takes new
# work. No drain is needed: the kernel is the block.
# a wave prefetches at most 3 instruction cache lines (3*64 bytes) ahead of the pc plus the line it
# is on, so 256 bytes = 64 instructions of straight line code run before it catches the prefetcher
# and stalls ~235 cycles for a real fetch. keep the whole kernel under that and the stall never
# happens, instead of landing mid block and corrupting whichever repeat it hits.
SWEEP_REPEATS, SWEEP_BLOCKS, SWEEP_RAMP = 48, 3, 10
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
_SOP_OPS = {m.name.lower(): m for en in (e3.SOP1Op, e3.SOP2Op, e3.SOPCOp, e3.SOPKOp) for m in en}

# destinations walk upward from _SWEEP_DST so the repeats do not depend on each other
def _sweep_inst(name:str, i:int):
  kwargs, sreg = {}, _SWEEP_SRC
  for field, (_fmt, width, kind) in OPERANDS[_SOP_OPS[name]].items():
    if field == "simm16": kwargs[field] = 1
    elif kind.name == "OPR_SDST":
      kwargs[field] = s[_SWEEP_DST+2*i:_SWEEP_DST+1+2*i] if width == 64 else s[_SWEEP_DST+i]
    else:
      kwargs[field] = s[sreg:sreg+1] if width == 64 else s[sreg]
      sreg += 2 if width == 64 else 1
  return getattr(r3, name)(**kwargs)

# keep an opcode only if this device really implements it. tinygrad's rdna3 enum is a union over
# gfx11.0 and gfx11.5, so it contains scalar float ops (s_add_f32, s_cvt_*, ...) that gfx1102 has no
# unit for: executing one raises sq_intr ILLEGAL_INST and hangs the queue. LLVM knows per target.
# also require tinygrad to decode it back, since amd_decode must disassemble the whole kernel.
def _sweep_ok(name:str, target:str) -> bool:
  if OPERANDS.get(_SOP_OPS[name]) is None or any(u in name.upper() for u in _UNSAFE): return False
  if name in _BANK_CONFLICT: return False
  if not hasattr(r3, name): return False
  try:
    inst = _sweep_inst(name, 0)
    if repr(decode_inst(inst.to_bytes(), "rdna3")) != repr(inst): return False
    return llvm_disasm(inst.to_bytes(), target, get_mattr("rdna3"))[0].split()[0] == name
  except (TypeError, ValueError, KeyError, IndexError): return False  # opcode we cannot build or decode

def sweep_ops(target:str) -> list[str]: return sorted(n for n in _SOP_OPS if _sweep_ok(n, target))

def _sweep_block(name:str) -> list: return [_sweep_inst(name, i) for i in range(SWEEP_REPEATS)]

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestSIMDModel(unittest.TestCase):
  # open the device once: a failed open leaves its flock held, so retrying per test buries the real
  # error under a lock error from the next attempt
  @classmethod
  def setUpClass(cls):
    if "MOCK" in type(Device["AMD"].iface).__name__: raise unittest.SkipTest("needs real hardware, not the emulator")
    if TARGET_TO_ARCH[Device["AMD"].arch] != "rdna3": raise unittest.SkipTest("only rdna3")
    if os.path.exists(REF): os.remove(REF)  # a run starts a fresh set of references

  # capture on hardware, record the raw trace for the emulator test, and return the projection
  def _capture(self, fxn, kname):
    projs, raw, lib, arch, simd = capture_runs(fxn, kname)
    ref = {}
    if os.path.exists(REF):
      with open(REF, "rb") as f: ref = pickle.load(f)
    ref[kname] = {"runs":raw, "lib":lib, "arch":arch, "simd_sel":simd}
    with open(REF, "wb") as f: pickle.dump(ref, f)
    self.raw = (raw[0][0], lib, arch, simd)
    print(f"\n  {kname}: " + ", ".join(f"{op}@{t}->{e}" for t, e, _, op in projs[0]))
    return projs[0]

  def _salu_gaps(self, fxn, kname):
    return [(t, e) for t, e, _, op in self._capture(fxn, kname) if op.startswith("S_ADD")]

  # refuted by: a gap that is not FIRST_INST_CYCLES
  def test_first_instruction_transit(self):
    gaps = self._salu_gaps(custom_salu_single, "custom_salu_single")
    self.assertEqual(len(gaps), 1)
    t, e = gaps[0]
    self.assertIsNotNone(e, "no ALUEXEC packet, cannot measure queue transit")
    self.assertEqual(e - t, FIRST_INST_CYCLES)

  # the second s_add is dispatched long after the SALU queue drained, so it measures transit without
  # whatever the first instruction of a wave pays for.
  # refuted by: the second gap equalling the first (transit would then be one constant), or the two
  # gaps differing from these values at all
  def test_steady_state_transit(self):
    gaps = self._salu_gaps(custom_salu_after_idle, "custom_salu_after_idle")
    self.assertEqual(len(gaps), 2)
    for i, (t, e) in enumerate(gaps):
      self.assertIsNotNone(e, f"s_add {i} has no ALUEXEC packet")
    self.assertEqual(gaps[0][1] - gaps[0][0], FIRST_INST_CYCLES, "first s_add")
    self.assertEqual(gaps[1][1] - gaps[1][0], QUEUE_CYCLES, "second s_add, queue already drained")

  # every SALU instruction should show the same transit on an idle queue and the same initiation
  # interval back to back. exceptions are listed, and anything new is a discovery.
  # refuted by: an opcode whose transit or interval is not what SALU_TIMING says
  def test_salu_sweep(self):
    fails = []
    for name in sweep_ops(Device["AMD"].arch):
      # one kernel holding one block, dispatched SWEEP_BLOCKS times, so every trial runs the same
      # code at the same offset. the block is short enough that the prefetcher never runs dry.
      kname = f"custom_salu_{name}"
      kernel = _kernel(kname, _sweep_block(name) + [s_endpgm()], register=False)
      projs, raw, lib, arch, simd = capture_runs(kernel, kname, SWEEP_BLOCKS)
      n_exec = sum(isinstance(p, ALUEXEC) for p, _ in map_insts(raw[0][0], lib, arch, simd))
      print(f"\n  **** {name}" + (f"   <- {n_exec} ALUEXEC for {SWEEP_REPEATS} instructions, pairing unreliable"
                                   if n_exec != SWEEP_REPEATS else ""))
      seen = set()
      for b, proj in enumerate(projs):
        blk = [(t, e) for t, e, _, _op in proj]
        self.assertEqual(len(blk), SWEEP_REPEATS, f"{name} trial {b}: unexpected instruction count")
        transit = None if blk[0][1] is None else blk[0][1] - blk[0][0]
        disp = [y[0]-x[0] for x, y in zip(blk, blk[1:])]
        gaps = [y[1]-x[1] for x, y in zip(blk, blk[1:]) if x[1] is not None and y[1] is not None]
        # the wrexec family runs its first few at the normal rate before settling, so read the
        # interval off the tail rather than off the ramp.
        tail = gaps[SWEEP_RAMP:]
        interval = max(set(tail), key=tail.count) if tail else None
        seen.add((transit, interval))
        print(f"    #{b} transit={str(transit):>4} interval={str(interval):>4}")
        print(f"        dispatch {disp}")
        print(f"        exec     {gaps}")
      want = salu_timing(name)
      if len(seen) > 1: fails.append(f"{name}: unstable across trials, {sorted(seen)}")
      elif seen != {want}: fails.append(f"{name}: {seen.pop()}, expected {want}")
    self.assertFalse(fails, f"{len(fails)} opcodes disagree with salu_timing():\n" + "\n".join(fails))

if __name__ == "__main__":
  unittest.main()

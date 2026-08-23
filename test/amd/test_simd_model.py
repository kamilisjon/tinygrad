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
# (transit, initiation interval) per opcode, None is the default. multiply is its own class.
SALU_TIMING: dict = {None: (2, 1), "s_mul_i32": (3, 2), "s_mul_hi_u32": (3, 2)}
QUEUE_CYCLES = 2       # any later instruction
# open: an 8 instruction chain starting with s_mov_b32 showed 2, not 4, for its first instruction, so
# the extra cost is not paid by every opcode. unexplained.
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
# One block per instruction: drain the queue with nops, then dispatch the instruction 5 times to
# different destinations so the repeats are independent. The first of the 5 lands on an idle queue
# and measures transit; the spacing between their execs measures how fast the worker takes new work.
SWEEP_REPEATS, SWEEP_DRAIN, SWEEP_BLOCKS = 50, 20, 3
# 64 bit destinations take two sgprs each, so sources have to start above all of them or the later
# repeats overwrite their own inputs and become a dependent chain
_SWEEP_SRC = 10 + 2*SWEEP_REPEATS

# every scalar ALU opcode except the ones that would not come back: anything touching the PC, EXEC,
# hardware registers, or wave state. SOPP is control flow only and emits no exec packet.
_UNSAFE = ("PC", "SAVEEXEC", "SETREG", "GETREG", "BRANCH", "CALL", "RFE", "ENDPGM", "TRAP",
           "SENDMSG", "SLEEP", "BARRIER", "WAITCNT", "NOP", "HALT", "PRIO", "ICACHE", "TTRACE",
           "WAKEUP", "PERFLEVEL", "VERSION", "CLAUSE", "DELAY", "WAIT", "MSG")
_SOP_OPS = {m.name.lower(): m for en in (e3.SOP1Op, e3.SOP2Op, e3.SOPCOp, e3.SOPKOp) for m in en}

# destinations walk upward from s[10] so the repeats do not depend on each other
def _sweep_inst(name:str, i:int):
  kwargs, sreg = {}, _SWEEP_SRC
  for field, (_fmt, width, kind) in OPERANDS[_SOP_OPS[name]].items():
    if field == "simm16": kwargs[field] = 1
    elif kind.name == "OPR_SDST": kwargs[field] = s[10+2*i:11+2*i] if width == 64 else s[10+i]
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
  if not hasattr(r3, name): return False
  try:
    inst = _sweep_inst(name, 0)
    if repr(decode_inst(inst.to_bytes(), "rdna3")) != repr(inst): return False
    return llvm_disasm(inst.to_bytes(), target, get_mattr("rdna3"))[0].split()[0] == name
  except Exception: return False

def sweep_ops(target:str) -> list[str]: return sorted(n for n in _SOP_OPS if _sweep_ok(n, target))

def _sweep_block(name:str) -> list:
  return [s_nop(0) for _ in range(SWEEP_DRAIN)] + [_sweep_inst(name, i) for i in range(SWEEP_REPEATS)]

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
    ops_names = sweep_ops(Device["AMD"].arch)
    insts = [x for n in ops_names for _ in range(SWEEP_BLOCKS) for x in _sweep_block(n)]
    kernel = _kernel("custom_salu_sweep", insts + [s_endpgm()], register=False)
    proj = self._capture(kernel, "custom_salu_sweep")
    ops = [(t, e) for t, e, _, op in proj if op != "S_NOP"]
    self.assertEqual(len(ops), len(ops_names)*SWEEP_REPEATS*SWEEP_BLOCKS, "unexpected instruction count")
    # sram_scope pairs one ALUEXEC per dispatch. if the hardware emits more than one per instruction
    # (a 64 bit op retiring in two halves would) that pairing silently shifts and every transit and
    # interval below is wrong, so count the packets before trusting any of it.
    n_exec = sum(isinstance(p, ALUEXEC) for p, _ in map_insts(*self.raw))
    print(f"\n  {n_exec} ALUEXEC packets for {len(ops)} SALU instructions"
          f"{'  <- pairing is unreliable' if n_exec != len(ops) else ''}")
    fails = []
    for i, name in enumerate(ops_names):
      print(f"\n  **** {name}")
      seen = set()
      for b in range(SWEEP_BLOCKS):
        blk = ops[(i*SWEEP_BLOCKS + b)*SWEEP_REPEATS:][:SWEEP_REPEATS]
        transit = None if blk[0][1] is None else blk[0][1] - blk[0][0]
        disp = [y[0]-x[0] for x, y in zip(blk, blk[1:])]
        gaps = [y[1]-x[1] for x, y in zip(blk, blk[1:]) if x[1] is not None and y[1] is not None]
        # the interval is the mode: a kernel this long takes icache misses that stall the whole wave
        # for ~250 cycles, and those show up as one huge gap among otherwise equal ones
        interval = max(set(gaps), key=gaps.count) if gaps else None
        seen.add((transit, interval))
        # a 64 bit op may retire as two exec packets; sram_scope pairs them FIFO one per dispatch, so
        # unpaired is nonzero when the pairing (and therefore transit and interval) is unreliable
        paired = sum(1 for x in blk if x[1] is not None)
        print(f"    #{b} transit={str(transit):>4} interval={str(interval):>4} paired={paired}/{len(blk)}")
        print(f"        dispatch {disp}")
        print(f"        exec     {gaps}")
      want = SALU_TIMING.get(name, SALU_TIMING[None])
      if len(seen) > 1: fails.append(f"{name}: unstable across blocks, {sorted(seen)}")
      elif seen != {want}: fails.append(f"{name}: {seen.pop()}, expected {want}")
    self.assertFalse(fails, f"{len(fails)} opcodes disagree with SALU_TIMING:\n" + "\n".join(fails))

if __name__ == "__main__":
  unittest.main()

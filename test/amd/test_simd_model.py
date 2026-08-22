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
from tinygrad.renderer.amd.dsl import s
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import TARGET_TO_ARCH, capture_runs, project, times_of, execs_of, insts_of

# dispatch->exec with an empty queue and an idle worker is not one number: the first instruction of
# a wave costs more than a later one. measured on gfx1102 with s_add_i32.
FIRST_INST_CYCLES = 4  # first instruction of the wave
QUEUE_CYCLES = 2       # any later instruction
# open: an 8 instruction chain starting with s_mov_b32 showed 2, not 4, for its first instruction, so
# the extra cost is not paid by every opcode. unexplained.
REF = "/tmp/tinygrad_sqtt_ref.pkl"  # hardware traces captured here, replayed by test_cycle_accurate_emu
KERNELS: dict = {}  # name -> builder, so the emulator test can rerun exactly what hardware ran

def _kernel(name:str, insts:list):
  def fxn(A:UOp) -> UOp:
    threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
    sink = UOp.sink(A.flatten().base, threads, wg, arg=KernelInfo(name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  KERNELS[name] = fxn
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

if __name__ == "__main__":
  unittest.main()

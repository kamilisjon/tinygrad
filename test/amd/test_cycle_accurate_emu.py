import unittest, os, pickle
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import Context
from tinygrad.renderer.amd.dsl import s, v
from tinygrad.renderer.amd.sqtt import map_insts, INST, VALUINST, ALUEXEC, VMEMEXEC
from tinygrad.runtime.autogen.amd.rdna3.ins import *
import tinygrad.runtime.ops_amd  # noqa: F401  registers the SQTT_* ContextVars

REF = "/tmp/tinygrad_sqtt_ref.pkl"
KNAME, N_RUNS, N_SALU = "custom_sram_kernel", 2, 8

# SRAM-only kernels for making the emulator's SQTT output cycle accurate.

def custom_sram_kernel(A:UOp) -> UOp:
  A = A.flatten()
  threads = UOp.special(32, "lidx0")
  wg = UOp.special(1, "gidx0")
  # SALU only: no VALU, no branches, no memory ops. A is bound but never touched.
  # every s_add reads the s[1] the one above wrote, so the whole chain is serialized on one sgpr.
  # writing to a different sgpr each time would make them independent and measure issue rate instead.
  insts = [
    s_mov_b32(s[1], 1),
    *[s_add_i32(s[1], s[1], 1) for _ in range(N_SALU)],
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo(KNAME))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

# helper functions

# link a trace to the exact binary that produced it via ProfileSQTTEvent.kern -> ProfileProgramEvent.tag,
# the way viz and test_sqttmap do. matching on name alone can pick a stale build of an edited kernel.
def _lib_for(kern:int|None) -> bytes:
  prgs = {e.tag:e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
  if kern is not None:
    assert (e:=prgs.get(kern)) is not None and e.lib, f"no ProfileProgramEvent tagged {kern}"
    return e.lib
  # the emulator emits no ProfileSQTTEvent, so there is no kern to link from
  assert (c:=[e for e in prgs.values() if e.name == KNAME and e.lib]), f"no ProfileProgramEvent for {KNAME}, is PROFILE=1 set?"
  return c[-1].lib

def _pkt_hist(blobs:list[bytes]) -> dict[str, int]:
  from tinygrad.renderer.amd.sqtt import decode
  import collections
  c: collections.Counter = collections.Counter()
  for b in blobs: c.update(type(x).__name__ for x in decode(b))
  return dict(c)

def capture(n_runs:int, simd_sel:int=0) -> tuple[list[list[bytes]], bytes, str]:
  import test.mockgpu.amd.emu as emu
  on_hw = "MOCK" not in type(Device["AMD"].iface).__name__
  a = Tensor.empty(32, dtype=dtypes.float32).contiguous().realize()
  runs, kern = [], None
  for _ in range(n_runs):
    start = len(Compiled.profile_events)
    emu.sqtt_traces.clear()
    with Context(SQTT_LIMIT_SE=1, SQTT_ITRACE_SE_MASK=1, SQTT_SIMD_SEL=simd_sel):
      Tensor.custom_kernel(a, fxn=custom_sram_kernel)[0].realize()
    Device[Device.DEFAULT].synchronize()
    if on_hw:
      evs = [e for e in Compiled.profile_events[start:] if type(e).__name__ == "ProfileSQTTEvent" and e.itrace]
      assert evs, "hardware produced no instruction-traced SQTT events, is SQTT=1 set?"
      kern = evs[0].kern
      runs.append([e.blob for e in evs])
    else:
      assert emu.sqtt_traces, "emulator produced no SQTT trace, is PROFILE=1 set?"
      runs.append(list(emu.sqtt_traces))
  return runs, _lib_for(kern), Device["AMD"].arch

def _project(blobs:list[bytes], lib:bytes, arch:str, simd:int):
  for b in blobs:
    try: p = sram_scope(b, lib, arch, simd)
    except KeyError: continue
    if p: return p
  return None

# only one simd per se is instruction traced, and the dispatcher does not always put the wave on it:
# even with a single CU enabled the two simds of that CU alternate between dispatches. so find the
# traced simd once, then retry dispatches until n_runs of them actually landed on it.
def capture_runs(n_runs:int, max_dispatch:int=40):
  sel, projs, raw, lib, arch, seen = None, [], [], None, None, {}
  for _ in range(max_dispatch):
    if len(projs) == n_runs: break
    for simd_sel in (range(4) if sel is None else [sel]):
      blobs, lib, arch = capture(1, simd_sel)
      if (pr:=_project(blobs[0], lib, arch, simd_sel)) is not None:
        sel = simd_sel
        projs.append(pr)
        raw.append(blobs[0])
        break
      seen[simd_sel] = _pkt_hist(blobs[0])
  assert len(projs) == n_runs, \
    f"only {len(projs)}/{n_runs} dispatches landed on a traced simd in {max_dispatch} tries; last packets seen: {seen}"
  return projs, raw, lib, arch, sel

# maps a dispatch packet's op category to the exec queue it will be retired from, same table viz uses
_DISPATCH_TO_EXEC = {"WMMA":"VALU", "VALU":"VALU", "VALU1":"VALU", "VALUT":"VALU", "VALUB":"VALU", "VALUINST":"VALU", "VINTERP":"VALU",
                     "SGMEM":"VMEM", "FLAT":"VMEM", "LDS":"LDS", "SALU":"SALU", "SMEM":"SALU", "VMEM":"VMEM"}

# project a raw blob down to SRAM scope: one entry per executed instruction on the traced SIMD, as
# (dispatch time, exec time, pc, op name). dispatch is when the wave issued it, exec is when the pipe
# started it, and exec is None for ops with no exec packet (branches). exec packets carry no wave or
# pc, they are matched to dispatches in issue order per exec queue.
# WAVEEND carries a synthetic s_endpgm and is dropped; s_delay_alu/s_wait_alu emit no token at all.
# pc and both times are made relative to the first entry: the absolute pc depends on where the elf
# put .text, and the absolute time depends on when tracing armed relative to dispatch.
def sram_scope(blob:bytes, lib:bytes, arch:str, simd:int=0) -> list[tuple[int, int|None, int, str]]:
  out: list[list] = []
  pending: dict[str, list[int]] = {}
  for p, info in map_insts(blob, lib, arch, simd):
    if isinstance(p, (ALUEXEC, VMEMEXEC)):
      for q in (["VALU", "SALU"] if (n:=p.src.name) == "VALU_SALU" else [n]):
        if pending.get(q): out[pending[q].pop(0)][1] = p._time
      continue
    if info is None or info.inst.op_name == "S_ENDPGM": continue
    if isinstance(p, (INST, VALUINST)):
      name = p.op.name if isinstance(p, INST) else "VALUINST"
      if (et:=_DISPATCH_TO_EXEC.get(name.replace("OTHER_", "").split("_")[0])) is not None:
        pending.setdefault(et, []).append(len(out))
    out.append([p._time, None, info.pc, info.inst.op_name])
  if not out: return []
  t0, pc0 = out[0][0], out[0][2]
  return [(t - t0, None if e is None else e - t0, pc - pc0, op) for t, e, pc, op in out]

def insts_of(proj): return [(pc, op) for _, _, pc, op in proj]
def times_of(proj): return [t for t, _, _, _ in proj]
def execs_of(proj): return [e for _, e, _, _ in proj]


class TestSQTTCapture(unittest.TestCase):
  def setUp(self):
    if Device.DEFAULT != "AMD" or "MOCK" in type(Device["AMD"].iface).__name__: self.skipTest("needs real AMD hardware")

  def test_capture_is_reproducible(self):
    if os.path.exists(REF): os.remove(REF)  # never leave a stale reference for TestSQTTEmu
    projs, raw, lib, arch, simd_sel = capture_runs(N_RUNS)
    print(f"\n  traced SIMD {simd_sel}, {len(projs[0])} instructions/run")
    for i, pr in enumerate(projs[1:], 1):
      self.assertEqual(insts_of(pr), insts_of(projs[0]), f"run {i} executed a different instruction sequence")
      self.assertEqual(times_of(pr), times_of(projs[0]), f"run {i} timing differs from run 0")
    os.makedirs(os.path.dirname(REF) or ".", exist_ok=True)
    with open(REF, "wb") as f: pickle.dump({"runs":raw, "lib":lib, "arch":arch, "simd_sel":simd_sel}, f)

class TestSQTTEmu(unittest.TestCase):
  def setUp(self):
    if Device.DEFAULT != "AMD" or "MOCK" not in type(Device["AMD"].iface).__name__: self.skipTest("needs the emulator")
    if not os.path.exists(REF): self.skipTest("no reference, run TestSQTTCapture on hardware first")
    with open(REF, "rb") as f: self.ref = pickle.load(f)
    runs, lib, arch = capture(1)
    self.emu = sram_scope(runs[0][0], lib, arch)
    self.hw = _project(self.ref["runs"][0], self.ref["lib"], self.ref["arch"], self.ref["simd_sel"])

  def test_emu_matches_hw_instructions(self):
    self.assertEqual(insts_of(self.emu), insts_of(self.hw))

  def test_emu_matches_hw_timing(self):
    ht, et = times_of(self.hw), times_of(self.emu)
    self.assertEqual(len(et), len(ht), "emulator and hardware executed a different number of instructions")
    hx, ex = execs_of(self.hw), execs_of(self.emu)
    f = lambda x: "-" if x is None else str(x)
    print(f"\n  {'#':>3} {'pc':>4}  {'instruction':<18} {'hw':>4} {'hw ex':>6} {'hw dly':>7} {'emu':>5} {'emu ex':>7} {'hw dt':>6} {'emu dt':>7}")
    for i in range(len(ht)):
      hdt, edt = (ht[i+1]-ht[i] if i+1 < len(ht) else 0), (et[i+1]-et[i] if i+1 < len(et) else 0)
      dly = "-" if hx[i] is None else str(hx[i]-ht[i])
      print(f"  {i:>3} {self.hw[i][2]:>4}  {self.hw[i][3]:<18} {ht[i]:>4} {f(hx[i]):>6} {dly:>7} {et[i]:>5} {f(ex[i]):>7} {hdt:>6} {edt:>7}"
            f"{'' if hdt == edt else '  <-'}")
    self.assertEqual(et, ht)

if __name__ == "__main__":
  unittest.main()

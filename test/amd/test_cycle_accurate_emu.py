# Compares the emulator's SQTT output against traces captured from real hardware.
#
# The references come from test_simd_model.py, which runs the same kernels on a GPU and writes them
# to REF. This half needs no GPU: it reruns each kernel on the emulator and diffs the projections.
import unittest, os, pickle
from tinygrad import Device
from test.amd.helpers import capture, project, sram_scope, dump_trace, insts_of, times_of, execs_of
from test.amd.test_simd_model import REF, KERNELS

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestSQTTEmu(unittest.TestCase):
  def setUp(self):
    if "MOCK" not in type(Device["AMD"].iface).__name__: self.skipTest("needs the emulator")
    if not os.path.exists(REF): self.skipTest(f"no references at {REF}, run test_simd_model.py on hardware first")
    with open(REF, "rb") as f: self.ref = pickle.load(f)

  # one (hw, emu) projection pair per captured kernel
  def _pairs(self):
    for kname, r in sorted(self.ref.items()):
      hw_raw = (r["runs"][0][0], r["lib"], r["arch"], r["simd_sel"])
      runs, lib, arch = capture(KERNELS[kname], kname)
      yield kname, project(r["runs"][0], r["lib"], r["arch"], r["simd_sel"]), sram_scope(runs[0][0], lib, arch), hw_raw, (runs[0][0], lib, arch, 0)

  def test_emu_matches_hw_instructions(self):
    for kname, hw, emu, _, _ in self._pairs():
      with self.subTest(kernel=kname): self.assertEqual(insts_of(emu), insts_of(hw))

  def test_emu_matches_hw_timing(self):
    for kname, hw, emu, hw_raw, emu_raw in self._pairs():
      with self.subTest(kernel=kname):
        ht, et, hx, ex = times_of(hw), times_of(emu), execs_of(hw), execs_of(emu)
        print(f"\n  ##### {kname} #####")
        dump_trace("hw", hw_raw)
        dump_trace("emu", emu_raw)
        self.assertEqual(len(et), len(ht), "emulator and hardware executed a different number of instructions")
        f = lambda x: "-" if x is None else str(x)
        print(f"\n  {'#':>3} {'pc':>4}  {'instruction':<18} {'hw':>4} {'hw ex':>6} {'hw dly':>7} {'emu':>5} {'emu ex':>7} {'hw dt':>6} {'emu dt':>7}")
        for i in range(len(ht)):
          hdt, edt = (ht[i+1]-ht[i] if i+1 < len(ht) else 0), (et[i+1]-et[i] if i+1 < len(et) else 0)
          dly = "-" if hx[i] is None else str(hx[i]-ht[i])
          print(f"  {i:>3} {hw[i][2]:>4}  {hw[i][3]:<18} {ht[i]:>4} {f(hx[i]):>6} {dly:>7} {et[i]:>5} {f(ex[i]):>7} {hdt:>6} {edt:>7}"
                f"{'' if hdt == edt else '  <-'}")
        self.assertEqual(et, ht)

if __name__ == "__main__":
  unittest.main()

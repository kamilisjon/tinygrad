"""Shared test helpers for AMD tests."""
import ctypes
from typing import Callable
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled
from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import map_insts, INST, VALUINST, ALUEXEC, VMEMEXEC
import tinygrad.runtime.ops_amd  # noqa: F401  registers the SQTT_* ContextVars
from tinygrad.helpers import unwrap
from tinygrad.runtime.autogen import llvm
from tinygrad.runtime.support.elf import elf_loader

ARCH_TO_TARGET:dict[str, list[str]] = {
  "rdna3":["gfx1100", "gfx1102", "gfx1151"],
  "rdna4":["gfx1200", "gfx1201"],
  "cdna":["gfx950", "gfx942"],
}

TARGET_TO_ARCH:dict[str, str] = {t:arch for arch,targets in ARCH_TO_TARGET.items() for t in targets}

_DPP16_RANGE_OPS = {0x100: "row_shl", 0x110: "row_shr", 0x120: "row_ror", 0x150: "row_newbcast", 0x160: "row_share", 0x170: "row_xmask"}
_DPP16_EXACT_OPS = {0x130: ("wave_shl", 1), 0x134: ("wave_rol", 1), 0x138: ("wave_shr", 1), 0x13c: ("wave_ror", 1),
                    0x140: ("row_mirror", 0), 0x141: ("row_half_mirror", 0), 0x142: ("row_bcast", 15), 0x143: ("row_bcast", 31)}

def get_target(arch:str) -> str: return ARCH_TO_TARGET[arch][0]

def decode_dpp16(dpp: int) -> tuple[str, int | tuple[int, int, int, int]]:
  """Decode a DPP16 control word into a symbolic operation and argument."""
  if dpp < 0x100: return "quad_perm", ((dpp >> 0) & 0x3, (dpp >> 2) & 0x3, (dpp >> 4) & 0x3, (dpp >> 6) & 0x3)
  if dpp in _DPP16_EXACT_OPS: return _DPP16_EXACT_OPS[dpp]
  if (base := dpp & 0x1f0) in _DPP16_RANGE_OPS: return _DPP16_RANGE_OPS[base], dpp & 0xf
  return "dpp", dpp

def get_mattr(arch:str) -> str:
  return {"rdna3":"+real-true16,+wavefrontsize32", "rdna4":"+real-true16,+wavefrontsize32", "cdna":"+wavefrontsize64"}[arch]

# LLVM in-process assembler/disassembler (replaces llvm-mc and llvm-objdump subprocesses)
_SENTINEL = b'\xde\xad\xbe\xef'
_SENTINEL_ASM = '.byte 0xde, 0xad, 0xbe, 0xef'

def _cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))
def _expect(x, err, ret=None):
  if x: raise RuntimeError(unwrap(ctypes.cast(err.contents, ctypes.c_char_p).value).decode() if not isinstance(err, str) else err)
  return ret

def _init_llvm():
  for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter', 'Disassembler']:
    getattr(llvm, f'LLVMInitializeAMDGPU{component}')()

def _create_target_machine(mcpu:str, mattr:str) -> llvm.LLVMTargetMachineRef:
  target = _expect(llvm.LLVMGetTargetFromTriple(b'amdgcn-amd-amdhsa', ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=_cerr()), err, tgt)
  return llvm.LLVMCreateTargetMachine(target, b'amdgcn-amd-amdhsa', mcpu.encode(), mattr.encode(),
                                      llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocDefault, llvm.LLVMCodeModelDefault)

def _emit_obj(asm_text:str, mcpu:str, mattr:str, diag_errors:list[str]|None=None) -> bytes:
  """Assemble raw asm text into an ELF object using LLVM in-process."""
  _init_llvm()
  tm = _create_target_machine(mcpu, mattr)
  ctx = llvm.LLVMContextCreate()
  try:
    errors = diag_errors if diag_errors is not None else []
    @llvm.LLVMDiagnosticHandler
    def handle_diag(diag_ref, _arg):
      if llvm.LLVMGetDiagInfoSeverity(diag_ref) == llvm.LLVMDSError:
        errors.append(ctypes.string_at(llvm.LLVMGetDiagInfoDescription(diag_ref)).decode())
    llvm.LLVMContextSetDiagnosticHandler(ctx, handle_diag, None)
    mod = llvm.LLVMModuleCreateWithNameInContext(b'asm', ctx)
    llvm.LLVMSetTarget(mod, b'amdgcn-amd-amdhsa')
    asm_bytes = asm_text.encode()
    llvm.LLVMSetModuleInlineAsm2(mod, asm_bytes, len(asm_bytes))
    buf = llvm.LLVMMemoryBufferRef()
    _expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, llvm.LLVMObjectFile, err:=_cerr(), ctypes.pointer(buf)), err)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(buf), llvm.LLVMGetBufferSize(buf))
    llvm.LLVMDisposeMemoryBuffer(buf)
    llvm.LLVMDisposeModule(mod)
    return obj
  finally:
    llvm.LLVMContextDispose(ctx)
    llvm.LLVMDisposeTargetMachine(tm)

def _extract_text(obj:bytes) -> bytes:
  """Extract .text section from ELF object bytes."""
  return next(s.content for s in elf_loader(obj)[1] if s.name == ".text")

def llvm_assemble(instrs:list[str], mcpu:str, mattr:str) -> list[bytes]:
  """Assemble instructions in one LLVM emission, return per-instruction bytes."""
  if not instrs: return []
  parts = []
  for instr in instrs:
    parts.append(instr)
    parts.append(_SENTINEL_ASM)
  text = _extract_text(_emit_obj('.text\n' + '\n'.join(parts) + '\n', mcpu, mattr))
  results, start = [], 0
  for _ in instrs:
    idx = text.find(_SENTINEL, start)
    assert idx != -1, "sentinel not found in .text section"
    results.append(bytes(text[start:idx]))
    start = idx + len(_SENTINEL)
  return results

def llvm_disasm(code:bytes, mcpu:str, mattr:str) -> list[str]:
  """Disassemble raw bytes into instruction strings using LLVM."""
  _init_llvm()
  dc = llvm.LLVMCreateDisasmCPUFeatures(b'amdgcn-amd-amdhsa', mcpu.encode(), mattr.encode(), None, 0,
                                         llvm.LLVMOpInfoCallback(0), llvm.LLVMSymbolLookupCallback(0))
  if not dc: raise RuntimeError(f"failed to create disasm context for {mcpu}")
  llvm.LLVMSetDisasmOptions(dc, 2 | 4)  # PrintImmHex | AsmPrinterVariant
  try:
    buf = ctypes.create_string_buffer(256)
    arr = (ctypes.c_uint8 * len(code)).from_buffer_copy(code)
    results, offset = [], 0
    while offset < len(code):
      size = llvm.LLVMDisasmInstruction(dc, ctypes.cast(ctypes.addressof(arr) + offset, ctypes.POINTER(ctypes.c_uint8)),
                                        len(code) - offset, 0, buf, 256)
      if size == 0: break
      results.append(buf.value.decode().strip())
      offset += size
    return results
  finally:
    llvm.LLVMDisasmDispose(dc)

def llvm_filter_valid_asm(tests:list[tuple[str, bytes]], mcpu:str, mattr:str) -> list[tuple[str, bytes]]:
  """Filter out tests where original ASM isn't valid on target, and where LLVM roundtrip doesn't match."""
  if not tests: return []
  # Assemble all instructions at once with sentinels and diagnostic handler to detect failures
  parts, diag_errors = [], []  # type: ignore[var-annotated]
  for asm, _ in tests:
    parts.append(asm)
    parts.append(_SENTINEL_ASM)
  text = _extract_text(_emit_obj('.text\n' + '\n'.join(parts) + '\n', mcpu, mattr, diag_errors))
  results, start = [], 0
  for _ in tests:
    idx = text.find(_SENTINEL, start)
    assert idx != -1, "sentinel not found in .text section"
    results.append(bytes(text[start:idx]))
    start = idx + len(_SENTINEL)
  # Invalid instructions produce 0 bytes; also filter where LLVM roundtrip doesn't match original
  return [(asm, data) for (asm, data), chunk in zip(tests, results) if len(chunk) > 0 and chunk == data]

def capture_runs(fxn:Callable, n_runs:int=1, max_dispatch:int=40):
  a = Tensor.empty(32, dtype=dtypes.float32).contiguous().realize()
  arch, sel, projs, lib = Device["AMD"].arch, None, [], None
  for _ in range(max_dispatch):
    if len(projs) == n_runs: break
    for simd_sel in (range(4) if sel is None else [sel]):
      start = len(Compiled.profile_events)
      with Context(SQTT_LIMIT_SE=1, SQTT_ITRACE_SE_MASK=1, SQTT_SIMD_SEL=simd_sel):
        Tensor.custom_kernel(a, fxn=fxn)[0].realize()
      Device[Device.DEFAULT].synchronize()
      evs = [e for e in Compiled.profile_events[start:] if type(e).__name__ == "ProfileSQTTEvent" and e.itrace]
      assert evs, "hardware produced no instruction-traced SQTT events, is SQTT=1 set?"
      prgs = {e.tag:e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
      assert (prg:=prgs.get(evs[0].kern)) is not None and prg.lib, f"no ProfileProgramEvent tagged {evs[0].kern}, is PROFILE=1 set?"
      lib = prg.lib
      if pr:=next((p for e in evs if (p:=sram_scope(e.blob, lib, arch))), None):
        sel = simd_sel
        projs.append(pr)
        break
  assert len(projs) == n_runs, f"only {len(projs)}/{n_runs} dispatches landed on a traced simd in {max_dispatch} tries"
  return projs, lib, arch

def sram_scope(blob:bytes, lib:bytes, arch:str) -> list[tuple[int, int|None, int, str]]:
  from test.mockgpu.amd.sqtt_enc import pipe_of
  out: list[list] = []
  pending: dict[str, list[int]] = {}
  for p, info in map_insts(blob, lib, arch):
    if isinstance(p, (ALUEXEC, VMEMEXEC)):
      for q in (["VALU", "SALU"] if (n:=p.src.name) == "VALU_SALU" else [n]):
        if pending.get(q): out[pending[q].pop(0)][1] = p._time
      continue
    if info is None or info.inst.op_name == "S_ENDPGM": continue
    if isinstance(p, (INST, VALUINST)):
      name = p.op.name if isinstance(p, INST) else "VALUINST"
      if (et:=pipe_of(name)[0]) is not None:
        pending.setdefault(et, []).append(len(out))
    out.append([p._time, None, info.pc, info.inst.op_name])
  if not out: return []
  t0, pc0 = out[0][0], out[0][2]
  return [(t - t0, None if e is None else e - t0, pc - pc0, op) for t, e, pc, op in out]


def insts_of(proj): return [(pc, op) for _, _, pc, op in proj]
def times_of(proj): return [t for t, _, _, _ in proj]
def execs_of(proj): return [e for _, e, _, _ in proj]


def capture_emu(insts:list, n_lanes:int=32) -> bytes:
  import test.mockgpu.amd.emu as emu
  code = b"".join(i.to_bytes() for i in insts)
  buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  args = (ctypes.c_uint64 * 1)(0)
  emu.sqtt_traces.clear()
  assert emu.run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, n_lanes, 1, 1, ctypes.addressof(args)) == 0, "emulator rejected the kernel"
  assert emu.sqtt_traces, "emulator produced no SQTT trace, is PROFILE=1 set?"
  return emu.sqtt_traces[0]

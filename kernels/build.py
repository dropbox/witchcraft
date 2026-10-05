#!/usr/bin/env python3
"""Build transformer kernels with Neso.

Generates TTIR from @triton.jit, writes per-platform ninja files, runs ninja.
Output: out/{metal,metal_nosimd}/*.metallib, out/hlsl/*.hlsl

    python build.py                    # all platforms
    python build.py metal              # Apple Silicon metallibs only
    python build.py metal_nosimd       # Intel Mac metallibs only
    python build.py hlsl               # HLSL + DXIL only
    python build.py metal hlsl         # multiple platforms
"""
import io
import tarfile
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists():
        try:
            if path.read_text() == content:
                return False
        except Exception:
            pass
    path.write_text(content)
    return True

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "out"
NESO = Path(os.environ.get("NESO_DIR", SCRIPT_DIR.parent.parent / "neso"))
os.environ.setdefault("TRITON_CACHE_DIR", str(OUT / "cache"))
os.environ.setdefault("CLANG_MODULE_CACHE_PATH", str(OUT / "cache" / "clang"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / "cache"))
_VENV = NESO / "env"
_BIN = _VENV / "Scripts" if sys.platform == "win32" else _VENV / "bin"
PYTHON = str(_BIN / "python")
NINJA = str(_BIN / "ninja") if (_BIN / "ninja").exists() else (shutil.which("ninja") or "ninja")
COMPILE_STEP = str(SCRIPT_DIR / "compile_step.py")
DXC = os.environ.get("DXC_PATH", str(SCRIPT_DIR.parent.parent / "directxshadercompiler" / "build-release" / "bin" / "dxc"))


def gen_ttir():
    """Generate TTIR for every kernel config."""
    sys.path.insert(0, str(NESO / "src"))
    sys.path.insert(0, str(SCRIPT_DIR))

    import neso.aot_compile as neso_aot
    # Source/IR generation is separated from platform compilation below.  Avoid
    # asking Neso to invoke the Metal toolchain once per specialization here.
    neso_aot.compile_msl = lambda _source: None
    compile_kernel = neso_aot.compile_kernel
    import encoder_kernels as K
    from kernel_configs import METAL_KERNELS, HLSL_EXTRA_KERNELS

    ttir_dir = OUT / "ttir"
    ttir_dir.mkdir(parents=True, exist_ok=True)

    configs = {}
    for cfg in METAL_KERNELS:
        configs[cfg[0]] = cfg
    for cfg in HLSL_EXTRA_KERNELS:
        configs.setdefault(cfg[0], cfg)

    print(f"Generating TTIR for {len(configs)} kernels...")
    t0 = time.time()
    ok = 0

    for name, cfg in sorted(configs.items()):
        func_name, sig, nw, grid = cfg[1], cfg[2], cfg[3], cfg[4]
        opts = cfg[5] if len(cfg) > 5 else {}
        fn = getattr(K, func_name, None)
        if fn is None:
            print(f"  {name}: SKIP (no {func_name})")
            continue
        try:
            r = compile_kernel(fn=fn, signature=sig, num_warps=nw, grid=grid)
            ir = r.ttgir_text or r.ttir_text
            write_if_changed(ttir_dir / f"{name}.ttir", ir)
            serializable_constants = {}
            for k, v in r.constants.items():
                if hasattr(v, '__name__'):
                    serializable_constants[k] = v.__name__
                elif v is None:
                    serializable_constants[k] = None
                else:
                    serializable_constants[k] = v
            write_if_changed(ttir_dir / f"{name}.json", json.dumps({
                "kernel_name": r.kernel_name,
                "params": r.params,
                "constants": serializable_constants,
                "threadgroup_size": r.threadgroup_size,
                "grid": grid,
                "force_acc_fp16": opts.get("force_acc_fp16", False),
            }, indent=2))
            ok += 1
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback; traceback.print_exc()

    print(f"TTIR: {ok}/{len(configs)} in {time.time()-t0:.1f}s\n")
    if ok != len(configs):
        raise RuntimeError(f"Neso compilation failed for {len(configs) - ok} kernel(s)")


def _ninja_preamble():
    codegen_dir = NESO / "src" / "neso" / "backend" / "codegen"
    compiler_deps = " ".join(str(p) for p in sorted(codegen_dir.glob("*.py")))
    implicit = f"| {compiler_deps} {COMPILE_STEP}"
    return [
        "# Auto-generated — do not edit",
        f"python = {PYTHON}",
        f"step = {COMPILE_STEP}",
        f"module_cache = {OUT / 'cache' / 'clang'}",
        "",
    ], implicit


def gen_ninja_metal():
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS

    ttir = OUT / "ttir"
    metal = OUT / "metal"
    metal.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal\n  command = $python $step msl_metal $in $out\n  restat = 1\n  description = MSL(metal) $out")
    w.append("rule metallib_metal\n  command = xcrun metal -fmodules-cache-path=$module_cache -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal) $out")
    w.append("")

    libs = []
    for cfg in METAL_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        am = metal / f"{name}.metal"
        al = metal / f"{name}.metallib"
        w.append(f"build {am}: msl_metal {t} {implicit}")
        w.append(f"build {al}: metallib_metal {am}")
        libs.append(str(al))

    w.append("")
    w.append(f"build metal: phony {' '.join(libs)}")
    w.append("default metal")
    w.append("")

    write_if_changed(OUT / "build_metal.ninja", "\n".join(w))
    print(f"build_metal.ninja: {len(libs)} metallibs")


def gen_ninja_metal_nosimd():
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS

    ttir = OUT / "ttir"
    metal_nosimd = OUT / "metal_nosimd"
    metal_nosimd.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal_nosimd\n  command = $python $step msl_metal_nosimd $in $out\n  restat = 1\n  description = MSL(metal_nosimd) $out")
    w.append("rule metallib_metal_nosimd\n  command = xcrun metal -fmodules-cache-path=$module_cache -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal_nosimd) $out")
    w.append("")

    libs = []
    for cfg in METAL_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        im = metal_nosimd / f"{name}.metal"
        il = metal_nosimd / f"{name}.metallib"
        w.append(f"build {im}: msl_metal_nosimd {t} {implicit}")
        w.append(f"build {il}: metallib_metal_nosimd {im}")
        libs.append(str(il))

    w.append("")
    w.append(f"build metal_nosimd: phony {' '.join(libs)}")
    w.append("default metal_nosimd")
    w.append("")

    write_if_changed(OUT / "build_metal_nosimd.ninja", "\n".join(w))
    print(f"build_metal_nosimd.ninja: {len(libs)} metallibs")


def gen_ninja_hlsl():
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import get_hlsl_kernels

    ttir = OUT / "ttir"
    hlsl = OUT / "hlsl"
    dxil = OUT / "dxil"
    hlsl.mkdir(parents=True, exist_ok=True)
    dxil.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append(f"dxc = {DXC}")
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out")
    w.append("rule dxil\n  command = $dxc -T cs_6_6 -E $entry -enable-16bit-types -O3 -Fo $out $in\n  description = DXIL $out")
    w.append("")

    dxil_files = []
    hlsl_seen = set()
    for cfg in get_hlsl_kernels():
        name = cfg[0]
        if name in hlsl_seen:
            continue
        hlsl_seen.add(name)
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        h = hlsl / f"{name}.hlsl"
        d = dxil / f"{name}.dxil"
        metadata = json.loads((ttir / f"{name}.json").read_text())
        w.append(f"build {h}: hlsl {t} {implicit}")
        w.append(f"build {d}: dxil {h}")
        w.append(f"  entry = {metadata['kernel_name']}")
        dxil_files.append(str(d))

    w.append("")
    w.append(f"build hlsl_all: phony {' '.join(dxil_files)}")
    w.append("default hlsl_all")
    w.append("")

    write_if_changed(OUT / "build_hlsl.ninja", "\n".join(w))
    print(f"build_hlsl.ninja: {len(dxil_files)} dxil")


def run_ninja(platform):
    ninja_file = OUT / f"build_{platform}.ninja"
    if not ninja_file.exists():
        print(f"ninja: no {ninja_file.name}, skipping")
        return True
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-C", str(OUT), "-f", ninja_file.name])
    dt = time.time() - t0
    print(f"ninja({platform}): {dt:.1f}s (exit={r.returncode})")
    return r.returncode == 0


def gen_rust():
    from gen_rust import main as gen_rust_main
    gen_rust_main()


def pack_kernels(platform):
    if platform == "metal":
        return  # Runtime uses the scalar lowering on both Mac architectures.
    from kernel_configs import METAL_KERNELS, get_hlsl_kernels
    if platform == "metal_nosimd":
        configs, directory, extension = METAL_KERNELS, "metal_nosimd", "metallib"
        loader, filename = "neso_metal_gen.rs", "kernels_metal_nosimd.tar.zst"
    else:
        configs, directory, extension = get_hlsl_kernels(), "dxil", "dxil"
        loader, filename = "neso_d3d12_gen.rs", "kernels_dxil.tar.zst"
    paths = [OUT / directory / f"{name}.{extension}" for name in sorted({cfg[0] for cfg in configs})]
    paths.append(OUT / "generated" / loader)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in paths:
            data = path.read_bytes()
            if not data:
                raise ValueError(f"Empty kernel cache input: {path}")
            info = tarfile.TarInfo(path.name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    compressed = subprocess.run(["zstd", "-19", "--stdout"], input=buffer.getvalue(),
                                stdout=subprocess.PIPE, check=True).stdout
    destination = OUT / filename
    if not destination.exists() or destination.read_bytes() != compressed:
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(compressed)
        temporary.replace(destination)
    print(f"Cached {len(paths) - 1} kernels in {filename} ({len(compressed)} bytes)")


VALID_PLATFORMS = ("metal", "metal_nosimd", "hlsl")

if __name__ == "__main__":
    platforms = sys.argv[1:] or list(VALID_PLATFORMS)
    for p in platforms:
        if p not in VALID_PLATFORMS:
            print(f"Unknown platform: {p} (valid: {', '.join(VALID_PLATFORMS)})")
            sys.exit(1)

    gen_ttir()
    for p in platforms:
        {"metal": gen_ninja_metal, "metal_nosimd": gen_ninja_metal_nosimd, "hlsl": gen_ninja_hlsl}[p]()
    for p in platforms:
        if not run_ninja(p):
            sys.exit(1)
    gen_rust()
    for p in platforms:
        pack_kernels(p)

#!/usr/bin/env python3
"""Build T5 Triton kernels.

Generates TTIR from @triton.jit, writes per-platform ninja files, runs ninja.
Output: out/{metal,metal_nosimd}/*.metallib, out/hlsl/*.hlsl

    python build.py                    # all platforms
    python build.py metal              # Apple Silicon metallibs only
    python build.py metal_nosimd       # Intel Mac metallibs only
    python build.py hlsl               # HLSL + DXIL only
    python build.py metal hlsl         # multiple platforms
"""
import json
import os
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
TRITON = SCRIPT_DIR.parent.parent.parent / "triton"
TRITON_METAL = TRITON / "third_party" / "metal"
_VENV = TRITON / "env"
_BIN = _VENV / "Scripts" if sys.platform == "win32" else _VENV / "bin"
PYTHON = str(_BIN / "python")
NINJA = str(_BIN / "ninja")
COMPILE_STEP = str(SCRIPT_DIR / "compile_step.py")


def gen_ttir():
    """Generate TTIR for every kernel config."""
    sys.path.insert(0, str(TRITON_METAL))
    sys.path.insert(0, str(SCRIPT_DIR))

    from aot_compile import compile_kernel
    import t5_kernels as K
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


def _ninja_preamble():
    codegen_dir = TRITON_METAL / "backend" / "codegen"
    compiler_deps = " ".join(str(p) for p in sorted(codegen_dir.glob("*.py")))
    implicit = f"| {compiler_deps} {COMPILE_STEP}"
    return [
        "# Auto-generated — do not edit",
        f"python = {PYTHON}",
        f"step = {COMPILE_STEP}",
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
    w.append("rule metallib_metal\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal) $out")
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
    w.append("rule metallib_metal_nosimd\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal_nosimd) $out")
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
    hlsl.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out")
    w.append("")

    hlsl_files = []
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
        w.append(f"build {h}: hlsl {t} {implicit}")
        hlsl_files.append(str(h))

    w.append("")
    w.append(f"build hlsl_all: phony {' '.join(hlsl_files)}")
    w.append("default hlsl_all")
    w.append("")

    write_if_changed(OUT / "build_hlsl.ninja", "\n".join(w))
    print(f"build_hlsl.ninja: {len(hlsl_files)} hlsl")


def run_ninja(platform):
    ninja_file = OUT / f"build_{platform}.ninja"
    if not ninja_file.exists():
        print(f"ninja: no {ninja_file.name}, skipping")
        return True
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-f", str(ninja_file)])
    dt = time.time() - t0
    print(f"ninja({platform}): {dt:.1f}s (exit={r.returncode})")
    return r.returncode == 0


def gen_rust():
    from gen_rust import main as gen_rust_main
    gen_rust_main()


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

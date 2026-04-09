#!/usr/bin/env python3
"""Per-file compilation step for ninja.

Usage:
    python compile_step.py msl_metal         INPUT.ttir  OUTPUT.metal
    python compile_step.py msl_metal_nosimd  INPUT.ttir  OUTPUT.metal
    python compile_step.py hlsl              INPUT.ttir  OUTPUT.hlsl
"""
import json
import re
import sys
from pathlib import Path

TRITON_METAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "triton" / "third_party" / "metal"
sys.path.insert(0, str(TRITON_METAL_DIR))


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists():
        try:
            if path.read_text() == content:
                return False
        except Exception:
            pass
    path.write_text(content)
    return True


def _parse_num_warps(ttir_text: str) -> int:
    """Extract num-warps from TTIR module attributes. Returns 0 if not found."""
    m = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', ttir_text)
    return int(m.group(1)) if m else 0


def main():
    cmd, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]

    if cmd == "msl_metal":
        from backend.codegen import ttir_to_msl_with_metadata
        ttir_text = Path(inp).read_text()
        num_warps = _parse_num_warps(ttir_text)
        max_threads = num_warps * 32 if num_warps > 0 else 0
        msl, _, _, _ = ttir_to_msl_with_metadata(
            ttir_text, block_size=256, use_simdgroup=True,
            max_threads=max_threads)
        write_if_changed(Path(out), msl)

    elif cmd == "msl_metal_nosimd":
        from backend.codegen import ttir_to_msl_with_metadata
        ttir_text = Path(inp).read_text()
        num_warps = _parse_num_warps(ttir_text)
        max_threads = num_warps * 32 if num_warps > 0 else 0
        msl, _, _, _ = ttir_to_msl_with_metadata(
            ttir_text, block_size=256, use_simdgroup=False,
            max_threads=max_threads)
        write_if_changed(Path(out), msl)

    elif cmd == "hlsl":
        from backend.codegen import ttir_to_hlsl_with_metadata
        meta_path = Path(inp).with_suffix(".json")
        force_fp16 = False
        if meta_path.exists():
            force_fp16 = json.loads(meta_path.read_text()).get("force_acc_fp16", False)
        hlsl, name, _, threads, half4_args = ttir_to_hlsl_with_metadata(
            Path(inp).read_text(), block_size=256, force_acc_fp16=force_fp16)
        write_if_changed(Path(out), hlsl)

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

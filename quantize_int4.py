#!/usr/bin/env python3
"""
Quantize the OpenVINO T5 encoder model to INT4 using NNCF.

INT4 provides even better compression than INT8 (8x vs 4x) with minimal
additional accuracy loss when combined with proper layer exclusion.
"""

import openvino as ov
import numpy as np
from pathlib import Path
import nncf
import zstandard as zstd
import os

# Configuration
MODEL_DIR = Path("openvino_model")
MODEL_XML = MODEL_DIR / "xtr-ov.xml"
MODEL_BIN = MODEL_DIR / "xtr-ov.bin"
OUTPUT_DIR = MODEL_DIR / "int4"
OUTPUT_XML = OUTPUT_DIR / "xtr-ov-int4.xml"
ASSETS_DIR = Path("assets")

# Quantization parameters
NUM_SAMPLES = 300


class ProgressReader:
    def __init__(self, fileobj, label="", report_every_mb=1):
        self.fileobj = fileobj
        self.label = label
        self.total_read = 0
        self.report_every = report_every_mb * 1024 * 1024
        self.next_report = self.report_every

    def read(self, size=-1):
        chunk = self.fileobj.read(size)
        self.total_read += len(chunk)
        if self.total_read >= self.next_report:
            print(f"[{self.label}] Compressed {self.total_read / (1024*1024):.1f} MB...")
            self.next_report += self.report_every
        return chunk


def compress_file(in_path: str, out_path: str, level: int = 19):
    """Compress a file using zstandard."""
    print(f"Compressing {in_path} -> {out_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, "rb") as src_file:
        reader = ProgressReader(src_file, label=os.path.basename(in_path))
        cctx = zstd.ZstdCompressor(level=level)
        with open(out_path, "wb") as dst_file:
            cctx.copy_stream(reader, dst_file)


def create_calibration_dataset():
    """Create calibration dataset with random tokens."""
    print(f"Creating calibration dataset with {NUM_SAMPLES} samples...")
    seq_lengths = [32, 64, 128, 256, 512]
    calibration_data = []
    for i in range(NUM_SAMPLES):
        seq_len = seq_lengths[i % len(seq_lengths)]
        input_ids = np.random.randint(2, 32128, size=(1, seq_len), dtype=np.int64)
        calibration_data.append({"input_ids": input_ids})
    return calibration_data


def identify_layernorm_operations(model):
    """Find all LayerNorm-related operations to exclude."""
    layernorm_names = []
    for op in model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if "layer_norm" in op_name:
            layernorm_names.append(op_name)
    return layernorm_names


def quantize_model_int4():
    """Quantize the FP16 model to INT4."""
    print(f"Loading model from {MODEL_XML}...")

    if not MODEL_XML.exists():
        print(f"Error: Model file not found: {MODEL_XML}")
        return False

    core = ov.Core()
    model = core.read_model(MODEL_XML)

    print(f"Model loaded. Input: {model.input().partial_shape}, Output: {model.output().partial_shape}")

    # Find LayerNorm operations to exclude
    print("\nIdentifying LayerNorm operations to keep in higher precision...")
    layernorm_names = identify_layernorm_operations(model)
    print(f"Found {len(layernorm_names)} LayerNorm operations")

    # Create ignored scope
    ignored_scope = nncf.IgnoredScope(names=layernorm_names)

    # Create calibration dataset
    calibration_data = create_calibration_dataset()
    calibration_dataset = nncf.Dataset(calibration_data, lambda x: x)

    print(f"\nQuantizing model to INT4 (excluding LayerNorm)...")
    print(f"This may take several minutes...")

    # Quantize weights to INT4
    # Using INT4_SYM for better performance on CPU
    quantized_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ignored_scope=ignored_scope,
        dataset=calibration_dataset,
        subset_size=NUM_SAMPLES,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving quantized model to {OUTPUT_XML}...")
    ov.save_model(quantized_model, OUTPUT_XML)

    # File sizes
    fp16_size = MODEL_BIN.stat().st_size / (1024 * 1024)
    int4_bin = OUTPUT_DIR / "xtr-ov-int4.bin"
    int4_xml = OUTPUT_DIR / "xtr-ov-int4.xml"
    int4_size = int4_bin.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"INT4 Quantization complete!")
    print(f"{'='*60}")
    print(f"FP16: {fp16_size:.2f} MB")
    print(f"INT4: {int4_size:.2f} MB")
    print(f"Compression: {fp16_size/int4_size:.2f}x")
    print(f"Size reduction: {((fp16_size - int4_size) / fp16_size * 100):.1f}%")

    # Compress and save to assets/
    print(f"\n{'='*60}")
    print(f"Saving compressed model to assets/...")
    print(f"{'='*60}")
    ASSETS_DIR.mkdir(exist_ok=True)

    compress_file(str(int4_xml), str(ASSETS_DIR / "xtr-ov-int4.xml.zst"))
    compress_file(str(int4_bin), str(ASSETS_DIR / "xtr-ov-int4.bin.zst"))

    # Check compressed sizes
    compressed_xml_size = (ASSETS_DIR / "xtr-ov-int4.xml.zst").stat().st_size / (1024 * 1024)
    compressed_bin_size = (ASSETS_DIR / "xtr-ov-int4.bin.zst").stat().st_size / (1024 * 1024)
    total_compressed = compressed_xml_size + compressed_bin_size

    print(f"\nAssets saved:")
    print(f"  {ASSETS_DIR / 'xtr-ov-int4.xml.zst'} ({compressed_xml_size:.2f} MB)")
    print(f"  {ASSETS_DIR / 'xtr-ov-int4.bin.zst'} ({compressed_bin_size:.2f} MB)")
    print(f"  Total compressed: {total_compressed:.2f} MB")
    print(f"\nTo use the INT4 model:")
    print(f"  OPENVINO_MODEL_PATH=openvino_model/int4 sh run.sh")

    return True


def test_quantized_model():
    """Test the INT4 quantized model."""
    print(f"\n{'='*60}")
    print(f"Testing INT4 quantized model...")
    print(f"{'='*60}")

    core = ov.Core()
    fp16_model = core.read_model(MODEL_XML)
    int4_model = core.read_model(OUTPUT_XML)

    fp16_compiled = core.compile_model(fp16_model, "CPU")
    int4_compiled = core.compile_model(int4_model, "CPU")

    # Try to load INT8 for comparison
    int8_path = MODEL_DIR / "int8" / "xtr-ov-int8.xml"
    int8_no_ln_path = MODEL_DIR / "int8_no_layernorm" / "xtr-ov-int8-no-ln.xml"

    has_int8 = int8_path.exists()
    has_int8_no_ln = int8_no_ln_path.exists()

    if has_int8:
        int8_model = core.read_model(int8_path)
        int8_compiled = core.compile_model(int8_model, "CPU")
    if has_int8_no_ln:
        int8_no_ln_model = core.read_model(int8_no_ln_path)
        int8_no_ln_compiled = core.compile_model(int8_no_ln_model, "CPU")

    similarities_int4 = []
    similarities_int8 = []
    similarities_int8_no_ln = []

    print(f"Running 20 test samples...")
    for i in range(20):
        seq_len = np.random.choice([64, 128, 256])
        test_input = np.random.randint(2, 32128, size=(1, seq_len), dtype=np.int64)

        fp16_out = fp16_compiled.infer_new_request({"input_ids": test_input})
        int4_out = int4_compiled.infer_new_request({"input_ids": test_input})

        fp16_emb = fp16_out[fp16_compiled.output()]
        int4_emb = int4_out[int4_compiled.output()]

        sim = np.dot(fp16_emb.flatten(), int4_emb.flatten()) / (
            np.linalg.norm(fp16_emb) * np.linalg.norm(int4_emb)
        )
        similarities_int4.append(sim)

        if has_int8:
            int8_out = int8_compiled.infer_new_request({"input_ids": test_input})
            int8_emb = int8_out[int8_compiled.output()]
            int8_sim = np.dot(fp16_emb.flatten(), int8_emb.flatten()) / (
                np.linalg.norm(fp16_emb) * np.linalg.norm(int8_emb)
            )
            similarities_int8.append(int8_sim)

        if has_int8_no_ln:
            int8_no_ln_out = int8_no_ln_compiled.infer_new_request({"input_ids": test_input})
            int8_no_ln_emb = int8_no_ln_out[int8_no_ln_compiled.output()]
            int8_no_ln_sim = np.dot(fp16_emb.flatten(), int8_no_ln_emb.flatten()) / (
                np.linalg.norm(fp16_emb) * np.linalg.norm(int8_no_ln_emb)
            )
            similarities_int8_no_ln.append(int8_no_ln_sim)

    avg_int4 = np.mean(similarities_int4)
    min_int4 = np.min(similarities_int4)

    print(f"\nAccuracy (FP16 vs INT4):")
    print(f"  Average: {avg_int4:.6f}")
    print(f"  Minimum: {min_int4:.6f}")

    if has_int8:
        avg_int8 = np.mean(similarities_int8)
        print(f"\nComparison with INT8:")
        print(f"  INT8 (full): {avg_int8:.6f}")

    if has_int8_no_ln:
        avg_int8_no_ln = np.mean(similarities_int8_no_ln)
        print(f"  INT8 (no LayerNorm): {avg_int8_no_ln:.6f}")

    print(f"  INT4 (no LayerNorm): {avg_int4:.6f}")

    if has_int8_no_ln:
        diff = avg_int4 - avg_int8_no_ln
        print(f"\n  Difference from INT8 (no LayerNorm): {diff:+.6f} ({diff*100:+.2f}%)")

    if avg_int4 > 0.95:
        print(f"\n[OK] Excellent quality (>95%)")
    elif avg_int4 > 0.90:
        print(f"\n[OK] Good quality (>90%)")
    elif avg_int4 > 0.85:
        print(f"\n[NOTE] Acceptable for many use cases (~{avg_int4*100:.1f}%)")
    else:
        print(f"\n[!] Significant accuracy loss (<85%)")

    return True


if __name__ == "__main__":
    try:
        if quantize_model_int4():
            test_quantized_model()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

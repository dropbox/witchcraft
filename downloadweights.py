import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file
import zstandard as zstd
import os


snapshot_download(repo_id="google/xtr-base-en", local_dir="xtr-base-en", local_dir_use_symlinks=False, revision="main")

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
    print("compressing", in_path, "->", out_path, "...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, "rb") as src_file:
        reader = ProgressReader(src_file, label=os.path.basename(in_path))
        cctx = zstd.ZstdCompressor(level=level)
        with open(out_path, "wb") as dst_file:
            cctx.copy_stream(reader, dst_file)

def export_to_openvino(model, output_dir="openvino_model"):
    """Convert PyTorch model directly to OpenVINO IR (no ONNX intermediate)."""
    import openvino as ov

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    dummy_input = torch.randint(0, 32128, (1, 512), dtype=torch.long)

    print("Converting PyTorch model directly to OpenVINO IR...")
    # Test forward pass to verify output shape
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Model output shape: {test_output.shape} (expected: [1, 512, 128])")

        # Direct PyTorch to OpenVINO conversion (no ONNX intermediate)
        ov_model = ov.convert_model(
            model,
            example_input=dummy_input,
            input=[("input_ids", [1, -1], torch.int64)]  # Dynamic sequence length
        )

    xml_path = os.path.join(output_dir, "xtr-ov.xml")

    print("Saving FP32 base model (will be quantized to INT4 by quantize_int4.py)...")
    ov.save_model(ov_model, xml_path)

    bin_path = xml_path.replace(".xml", ".bin")
    if not os.path.exists(xml_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(f"OpenVINO conversion failed - files not found: {xml_path}, {bin_path}")

    print(f"Saved OpenVINO model: {xml_path}")
    return xml_path, bin_path

class XTR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("google/xtr-base-en", use_safetensors=True).encoder
        self.linear = torch.nn.Linear(768, 128, bias=False)

        to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")
        state = torch.load(to_dense_path)

        other = {}
        other["weight"] = state["linear.weight"]
        self.linear.load_state_dict(other)

    def forward(self, input_ids):
        # T5 encoder forward pass
        encoder_output = self.encoder(input_ids)
        # Get last_hidden_state from encoder output
        hidden_states = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output[0]
        # Apply final linear projection 768 -> 128
        embeddings = self.linear(hidden_states)
        return embeddings

snapshot_download(repo_id="google/xtr-base-en", local_dir="xtr-base-en", local_dir_use_symlinks=False, revision="main")

xtr = XTR()
fp16_state_dict = {k: v.half().cpu() for k, v in xtr.state_dict().items()}
save_file(fp16_state_dict, "xtr.safetensors")

compress_file("xtr-base-en/config.json", "assets/config.json.zst")
compress_file("xtr-base-en/tokenizer.json", "assets/tokenizer.json.zst")

try:
    model_xml_path, model_bin_path = export_to_openvino(xtr)
    print("\n" + "="*60)
    print("OpenVINO FP32 base model export successful!")
    print("="*60)

    # Get model size
    model_size = os.path.getsize(model_bin_path) / (1024 * 1024)
    print(f"FP32 base model size: {model_size:.2f} MB")
    print(f"Model files: {model_xml_path}, {model_bin_path}")

    print("\nNOTE: This is the uncompressed FP32 base model.")
    print("To create INT4 quantized model (~66 MB), run:")
    print("  python quantize_int4.py")
    print("\n" + "="*60)

except Exception as e:
    print(f"\nERROR: OpenVINO export failed: {e}")
    print(f"\nMake sure you have OpenVINO and NNCF installed:")
    print(f"  pip install openvino nncf")
    import traceback
    traceback.print_exc()
    exit(1)

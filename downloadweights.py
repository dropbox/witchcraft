import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging
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

class XTRLinear(torch.nn.Module):
    def __init__(self, in_features=768, out_features=128, bias=False):
        super().__init__()

    def forward(self, x):
        return self.linear(x)

class XTR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("google/xtr-base-en", dtype=torch.float16, use_safetensors=True).encoder
        self.encoder.linear = torch.nn.Linear(768, 128, bias=False)

        to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")
        state = torch.load(to_dense_path)

        other = {}
        other["weight"] = state["linear.weight"]
        self.encoder.linear.load_state_dict(other)

snapshot_download(repo_id="google/xtr-base-en", local_dir="xtr-base-en", local_dir_use_symlinks=False, revision="main")

xtr = XTR()
fp16_state_dict = {k: v.half().cpu() for k, v in xtr.state_dict().items()}
save_file(fp16_state_dict, "xtr.safetensors")

compress_file("xtr-base-en/config.json", "assets/config.json.zst")
compress_file("xtr-base-en/tokenizer.json", "assets/tokenizer.json.zst")
#compress_file("xtr.safetensors", "assets/xtr.safetensors.zst")

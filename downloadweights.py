import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file
import os
import shutil

from transformers import T5EncoderModel

snapshot_download(repo_id="google/xtr-base-en", local_dir="xtr-base-en",
                  local_dir_use_symlinks=False, revision="main")

class XTR(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_model = T5EncoderModel.from_pretrained(
            "google/xtr-base-en",
            use_safetensors=True,
        )
        self.encoder = encoder_model.encoder

        self.linear = nn.Linear(768, 128, bias=False)
        to_dense_path = hf_hub_download(
            repo_id="google/xtr-base-en",
            filename="2_Dense/pytorch_model.bin",
        )
        state = torch.load(to_dense_path, map_location="cpu", weights_only=True)
        self.linear.load_state_dict({"weight": state["linear.weight"]})

    def forward(self, input_ids):
        encoder_output = self.encoder(input_ids)
        hidden_states = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output[0]
        return self.linear(hidden_states)


xtr = XTR()
fp16_state_dict = {k: v.half().cpu() for k, v in xtr.state_dict().items()}
save_file(fp16_state_dict, "xtr.safetensors")
print(f"Saved xtr.safetensors with {len(fp16_state_dict)} tensors")

shutil.copy("xtr-base-en/config.json", "assets/config.json")
shutil.copy("xtr-base-en/tokenizer.json", "assets/tokenizer.json")

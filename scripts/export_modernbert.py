#!/usr/bin/env python3
"""Export ModernBERT-based XTRBert checkpoint to f16 safetensors for rust-warp.

Works for any ModernBERT variant (ColModernVBert, mmBERT-small, granite, etc.).
Infers most config values from checkpoint tensor shapes, reads num_attention_heads
from HF config (checkpoint dir, or --model flag).

Usage: export_modernbert.py <checkpoint_dir> [output_dir] [--model MODEL_NAME]
"""
import argparse
import json
import shutil
import torch
from safetensors.torch import save_file
from pathlib import Path

SKIP_PREFIXES = ("vision_model.", "connector.", "hash_embedding.")


def load_hf_config(ckpt_dir: Path, model_name: str | None) -> dict | None:
    """Try to load HF config from checkpoint dir, then from HuggingFace."""
    # 1. Check checkpoint dir and parent
    for d in [ckpt_dir, ckpt_dir.parent]:
        p = d / "config.json"
        if p.exists():
            with open(p) as f:
                cfg = json.load(f)
            # Distinguish HF config from our export config by checking for
            # HF-specific keys
            if "num_attention_heads" in cfg or "model_type" in cfg:
                print(f"  using HF config from {p}")
                return cfg
    # 2. Download from HuggingFace
    if model_name:
        from transformers import AutoConfig
        print(f"  loading config from HuggingFace: {model_name}")
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export ModernBERT checkpoint to f16 safetensors for rust-warp")
    parser.add_argument("checkpoint", help="Path to XTRBert checkpoint directory")
    parser.add_argument("output", nargs="?", default="assets-modernbert",
                        help="Output directory (default: assets-modernbert)")
    parser.add_argument("--model", help="HuggingFace model name for config lookup")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(ckpt_dir / "xtrbert.pt", map_location="cpu", weights_only=False)

    # Keep text encoder + projection, strip vision/connector/hash keys
    tensors = {}
    for k, v in ckpt.items():
        if any(k.startswith(p) for p in SKIP_PREFIXES):
            continue
        if v.ndim == 0:
            continue
        if v.dtype == torch.bfloat16:
            tensors[k] = v.to(torch.float32).contiguous()
        else:
            tensors[k] = v.contiguous()

    save_file(tensors, out_dir / "modernbert.safetensors")

    # Infer config from tensor shapes
    embed_w = tensors["encoder.embeddings.tok_embeddings.weight"]
    vocab_size, hidden_size = embed_w.shape
    wi_w = tensors["encoder.layers.0.mlp.Wi.weight"]
    intermediate_size = wi_w.shape[0] // 2  # GeGLU packs gate+up
    num_layers = sum(1 for k in tensors if k.endswith(".mlp.Wo.weight")
                     and k.startswith("encoder.layers."))

    # Detect MLP projection
    projection_mlp = None
    if "linear.0.weight" in tensors:
        projection_mlp = int(tensors["linear.0.weight"].shape[0])
        projection_dim = int(tensors["linear.2.weight"].shape[0])
    else:
        projection_dim = int(tensors["linear.weight"].shape[0])

    # Load HF config for values we can't infer from shapes
    hf_cfg = load_hf_config(ckpt_dir, args.model)

    rope_theta = 160000.0
    global_rope_theta = None
    norm_eps = 1e-5
    local_attention = 128
    global_attn_every_n = 3
    num_heads = hidden_size // 64  # fallback
    hidden_activation = "gelu"

    if hf_cfg:
        num_heads = hf_cfg.get("num_attention_heads", num_heads)
        # ModernBERT stores per-layer-type rope_theta in rope_parameters dict
        rope_params = hf_cfg.get("rope_parameters", {})
        if "sliding_attention" in rope_params:
            rope_theta = rope_params["sliding_attention"].get("rope_theta", rope_theta)
        elif "rope_theta" in hf_cfg:
            rope_theta = hf_cfg["rope_theta"]
        if "full_attention" in rope_params:
            gt = rope_params["full_attention"].get("rope_theta", rope_theta)
            if gt != rope_theta:
                global_rope_theta = gt
        norm_eps = hf_cfg.get("norm_eps",
                              hf_cfg.get("layer_norm_eps", norm_eps))
        local_attention = hf_cfg.get("local_attention", local_attention)
        global_attn_every_n = hf_cfg.get("global_attn_every_n_layers",
                                         global_attn_every_n)
        hidden_activation = hf_cfg.get("hidden_activation", hidden_activation)
    else:
        print(f"  WARNING: no HF config found, using head_dim=64 fallback "
              f"(num_heads={num_heads}). Pass --model to fix this.")

    config = {
        "hidden_size": int(hidden_size),
        "num_hidden_layers": num_layers,
        "num_attention_heads": int(num_heads),
        "intermediate_size": int(intermediate_size),
        "vocab_size": int(vocab_size),
        "rope_theta": rope_theta,
        **({"global_rope_theta": global_rope_theta} if global_rope_theta else {}),
        "norm_eps": norm_eps,
        "local_attention": local_attention,
        "global_attn_every_n_layers": global_attn_every_n,
        "hidden_activation": hidden_activation,
        "projection_dim": projection_dim,
    }
    if projection_mlp is not None:
        config["projection_mlp"] = projection_mlp
    if "token_gate.weight" in tensors and "token_gate_norm.weight" in tensors:
        config["token_gate"] = True

    (out_dir / "modernbert-config.json").write_text(json.dumps(config, indent=2))

    tok_src = ckpt_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy(tok_src, out_dir / "modernbert-tokenizer.json")
    else:
        print(f"WARNING: {tok_src} not found, copy modernbert-tokenizer.json manually")

    print(f"Exported to {out_dir}/")
    print(f"  config: {json.dumps(config, indent=2)}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

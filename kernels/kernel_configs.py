"""Kernel configurations for T5 Flash Attention.

Each config: (name, function_name, signature, num_warps, grid, options)
The function_name refers to an @triton.jit function in t5_kernels.py.
"""

# Q, K, V, O: fp16 pointers (aligned 16)
# Bias: fp32 pointer (aligned 16)
# seq_len, stride_h, stride_m, stride_o: i32 scalars
# sm_scale: fp32 scalar
# BM=32, BN=32, D=64: constexpr

METAL_KERNELS = [
    ("flash_attention_bias_32x32x64", "flash_attention_bias_fwd",
     "*fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp32:16, i32, i32, i32, i32, fp32, 32, 32, 64",
     4, ["cdiv(seq_len, 32)", "n_heads", "1"]),
    ("gelu_mul_1024", "gelu_mul_forward",
     "*fp16:16, *fp16:16, *fp16:16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("rms_norm_1024", "rms_norm_forward",
     "*fp32:16, *fp32:16, *fp32:16, i32, i32, fp32, 1024",
     4, ["n_rows", "1", "1"]),
]

HLSL_EXTRA_KERNELS = [
    ("flash_attention_bias_d64", "flash_attention_bias_fwd",
     "*fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp32:16, i32, i32, i32, i32, fp32, 32, 32, 64",
     4, ["cdiv(seq_len, 32)", "n_heads", "1"]),
]

KERNEL_METADATA = {
    "flash_attention_bias_32x32x64": {
        "alias": "flash_attention_bias",
        "group": "encoder",
        "tg_mem": 8192,
    },
    "gelu_mul_1024": {
        "alias": "gelu_mul",
        "group": "encoder",
    },
    "rms_norm_1024": {
        "alias": "rms_norm",
        "group": "encoder",
    },
    "flash_attention_bias_d64": {
        "alias": "flash_attention_bias_d64",
        "group": "encoder",
        "d3d12": True,
        "d3d12_only": True,
    },
}


def get_metal_kernels():
    return METAL_KERNELS


def get_hlsl_kernels():
    all_configs = []
    for cfg in METAL_KERNELS:
        all_configs.append(cfg if len(cfg) > 5 else (*cfg, {}))
    for cfg in HLSL_EXTRA_KERNELS:
        all_configs.append(cfg if len(cfg) > 5 else (*cfg, {}))
    return all_configs

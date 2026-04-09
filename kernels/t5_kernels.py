"""Triton kernels for T5 encoder acceleration.

Flash Attention 2 with additive position bias, adapted for the XTR T5 encoder:
  - head_dim = 64, num_heads = 8, max seq_len = 2048
  - No causal mask (bidirectional encoder self-attention)
  - Additive relative position bias [num_heads, q_len, kv_len] in FP32
"""
import triton
import triton.language as tl


@triton.jit
def flash_attention_bias_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, Bias_ptr,
    seq_len,
    stride_h,    # stride between heads (elements)
    stride_m,    # stride between rows for Q/K/V (elements)
    stride_o,    # stride between rows for O (elements)
    sm_scale,    # 1/sqrt(head_dim)
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    """Flash Attention 2 forward pass with additive position bias.

    Q, K, V: [n_heads, seq_len, D] fp16, contiguous.
    O:       [n_heads, seq_len, D] fp16, contiguous.
    Bias:    [n_heads, seq_len, seq_len] fp32.
    Grid:    (cdiv(seq_len, BM), n_heads, 1)

    Uses online softmax (Dao et al.) to avoid materializing the full
    [seq_len, seq_len] attention matrix. fp32 accumulator, fp16 output.
    """
    pid_m = tl.program_id(0)   # which BM-block of query rows
    pid_h = tl.program_id(1)   # which head

    off_m = pid_m * BM
    head_off = pid_h * stride_h
    bias_head_off = pid_h * seq_len * seq_len

    # Load Q tile [BM, D] — persists across all K/V iterations
    offs_m = off_m + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + head_off + offs_m[:, None] * stride_m + offs_d[None, :],
                mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize online softmax accumulators
    m_i = tl.full([BM], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    # Loop over all K/V blocks (no sliding window — full attention)
    for kv_start in range(0, seq_len, BN):
        offs_n = kv_start + tl.arange(0, BN)

        # Load K tile [BN, D]
        k = tl.load(K_ptr + head_off + offs_n[:, None] * stride_m + offs_d[None, :],
                    mask=offs_n[:, None] < seq_len, other=0.0)

        # QK = Q @ K^T : [BM, D] x [D, BN] -> [BM, BN]
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # Add position bias [BM, BN] from Bias[head, offs_m, offs_n]
        bias = tl.load(Bias_ptr + bias_head_off
                       + offs_m[:, None] * seq_len + offs_n[None, :],
                       mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
                       other=0.0)
        qk = qk + bias

        # Bounds mask: keys past seq_len get -inf
        qk = tl.where(offs_n[None, :] < seq_len, qk, -1e9)

        # Online softmax update
        m_ij = tl.max(qk, axis=1)            # [BM]
        m_new = tl.maximum(m_i, m_ij)        # [BM]
        alpha = tl.exp(m_i - m_new)          # [BM]
        p = tl.exp(qk - m_new[:, None])      # [BM, BN]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile [BN, D]
        v = tl.load(V_ptr + head_off + offs_n[:, None] * stride_m + offs_d[None, :],
                    mask=offs_n[:, None] < seq_len, other=0.0)

        # acc += P @ V : [BM, BN] x [BN, D] -> [BM, D]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Final normalization: O = acc / l
    acc = acc / l_i[:, None]

    # Store O [BM, D] as fp16
    tl.store(O_ptr + pid_h * stride_h + offs_m[:, None] * stride_o + offs_d[None, :],
             acc.to(tl.float16), mask=offs_m[:, None] < seq_len)


@triton.jit
def gelu_mul_forward(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused gated-GELU: out = GELU(gate) * up.

    gate, up: [n_elements] fp16 contiguous.
    out:      [n_elements] fp16 contiguous.
    Grid:     (cdiv(n_elements, BLOCK_SIZE), 1, 1)

    GELU computed in fp32 for precision (tanh approximation).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    # GELU (tanh approximation, matches PyTorch gelu_new / gelu_pytorch_tanh)
    cst = 0.7978845608028654  # sqrt(2/pi)
    inner = cst * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offsets, (gelu * up).to(tl.float16), mask=mask)


@triton.jit
def rms_norm_forward(
    x_ptr, weight_ptr, out_ptr,
    n_rows, dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """T5 RMS LayerNorm: out = x / sqrt(mean(x^2) + eps) * weight.

    x:      [n_rows, dim] fp32, contiguous.
    weight: [dim] fp32, contiguous.
    out:    [n_rows, dim] fp32, contiguous.
    Grid:   (n_rows, 1, 1)

    Each threadgroup handles one row. Reduction in fp32.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    # variance = mean(x^2)
    var = tl.sum(x * x, axis=0) / dim
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * w
    tl.store(out_ptr + row * dim + offs, out, mask=mask)

"""Triton kernels for T5 encoder acceleration.

Flash Attention 2 with additive position bias, adapted for the XTR T5 encoder:
  - head_dim = 64, num_heads = 8, max seq_len = 2048
  - No causal mask (bidirectional encoder self-attention)
  - Additive relative position bias [num_heads, q_len, kv_len] in FP32
"""
import triton
import triton.language as tl


# ─── Matmul FP16 — for all projections (QKV, O, FFN) ────────────────────────

@triton.jit
def matmul_fp16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N], all fp16, fp32 accumulate, fp16 output."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ─── Matmul Q8 — A(fp16) @ B(int8) with per-column scales ────────────────────

@triton.jit
def matmul_q8(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scales_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K](fp16) @ B[K,N](int8) * scales[N](fp16), fp32 accum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0).to(tl.float16)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply per-column scales
    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scales[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ─── Residual add (element-wise) ────────────────────────────────────────────

@triton.jit
def residual_add_forward(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = a + b, all fp16."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# ─── RMS Norm FP16 — fp16 in/out, fp32 accumulate ───────────────────────────

@triton.jit
def rms_norm_f16_forward(
    x_ptr, weight_ptr, out_ptr,
    n_rows, dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """T5 RMS LayerNorm on fp16 tensors: out = x / sqrt(mean(x^2) + eps) * weight.

    x:      [n_rows, dim] fp16, contiguous.
    weight: [dim] fp16, contiguous.
    out:    [n_rows, dim] fp16, contiguous.
    Grid:   (n_rows, 1, 1)

    Reduction in fp32 for numerical stability.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / dim
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * w
    tl.store(out_ptr + row * dim + offs, out.to(tl.float16), mask=mask)


# ─── Flash Attention 2 with position bias ────────────────────────────────────

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
    # Clamp before tanh: Metal's tanh produces NaN for |x| > ~45.
    # tanh saturates at ±1 for |x| > 10, so clamping is mathematically exact.
    inner = tl.minimum(tl.maximum(inner, -10.0), 10.0)
    gelu = 0.5 * gate * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offsets, (gelu * up).to(tl.float16), mask=mask)


# ─── Mixed-precision bridge kernels ─────────────────────────────────────────

@triton.jit
def rms_norm_f32_f16_forward(
    x_ptr, weight_ptr, out_ptr,
    n_rows, dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMS norm F32→F16: out = half(x / sqrt(mean(x^2) + eps) * weight).

    x:      [n_rows, dim] fp32, contiguous.
    weight: [dim] fp32, contiguous.
    out:    [n_rows, dim] fp16, contiguous.
    Grid:   (n_rows, 1, 1)
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim
    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / dim
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * w
    tl.store(out_ptr + row * dim + offs, out.to(tl.float16), mask=mask)


@triton.jit
def residual_add_f16_f32_forward(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Residual add: out = float(a) + b. a: fp16, b: fp32, out: fp32.

    b and out may alias for in-place update.
    Grid: (cdiv(n_elements, BLOCK_SIZE), 1, 1)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# ─── RMS Norm FP32 (legacy, used by existing CPU integration) ──────────────

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

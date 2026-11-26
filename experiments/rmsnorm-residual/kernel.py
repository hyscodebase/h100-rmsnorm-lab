"""Forward-only Triton residual addition and RMSNorm for width-4096 BF16 inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_rmsnorm(X, R, W, Updated, Normalized, WIDTH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, WIDTH)
    offsets = row * WIDTH + cols
    x = tl.load(X + offsets).to(tl.float32)
    residual = tl.load(R + offsets).to(tl.float32)
    updated = (x + residual).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Updated + offsets, updated)

    # RMSNorm consumes exactly the BF16 values returned to the caller.
    z = updated.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(z * z, axis=0) / WIDTH + 1e-6)
    weight = tl.load(W + cols).to(tl.float32)
    result = (z * inv_rms * weight).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Normalized + offsets, result)


def fused_residual_rmsnorm(x, residual, weight):
    """Return the rounded sum and its weighted RMSNorm as two new CUDA BF16 tensors."""
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 4096:
        raise ValueError("x must have shape [positive rows, 4096]")
    if residual.shape != x.shape or weight.shape != (4096,):
        raise ValueError("residual must match x and weight must have 4096 elements")
    if not x.is_cuda or residual.device != x.device or weight.device != x.device:
        raise ValueError("all inputs must be on the same CUDA device")
    inputs = (x, residual, weight)
    if any(t.dtype != torch.bfloat16 for t in inputs):
        raise TypeError("all inputs must use torch.bfloat16")
    if not all(t.is_contiguous() for t in inputs):
        raise ValueError("all inputs must be contiguous")
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        raise ValueError("this kernel supports forward inference only")

    with torch.cuda.device(x.device):
        updated = torch.empty_like(x)
        normalized = torch.empty_like(x)
        _fused_residual_rmsnorm[(x.shape[0],)](x, residual, weight, updated, normalized, 4096, num_warps=4)
    return updated, normalized

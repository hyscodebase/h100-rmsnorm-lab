"""A fixed Triton implementation of the vLLM 0.10.0 BF16 in-place contract."""
import triton
import triton.language as tl


@triton.jit
def fused(X, R, W, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    offsets = row * WIDTH + col
    original = tl.load(X + offsets).to(tl.float32)
    residual = tl.load(R + offsets).to(tl.float32)
    updated = (original + residual).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(R + offsets, updated)
    z = updated.to(tl.float32)
    scale = tl.rsqrt(tl.sum(z * z, axis=0) / WIDTH + 1e-6)
    scaled = (z * scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    tl.store(X + offsets, (scaled * weight).to(tl.bfloat16, fp_downcast_rounding="rtne"))


def fused_inplace(x, residual, weight):
    fused[(x.shape[0],)](x, residual, weight, x.shape[1], num_warps=4)

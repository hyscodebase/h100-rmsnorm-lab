"""Eight-warp variant with a fixed FP32 accumulation tree for width 4096."""
import triton
import triton.language as tl


@triton.jit
def _split_sum(values, BLOCKS: tl.constexpr, ENTRIES: tl.constexpr):
    left, right = tl.split(tl.reshape(values, (BLOCKS, ENTRIES // 2, 2)))
    return left + right


@triton.jit
def _group_square_sum(z, GROUPS: tl.constexpr = 256):
    left, right = tl.split(tl.reshape(z, (GROUPS, 4, 2)))
    pairs = left * left + right * right
    even, odd = tl.split(tl.reshape(pairs, (GROUPS, 2, 2)))
    pair0, pair2 = tl.split(even)
    pair1, pair3 = tl.split(odd)
    return ((pair0 + pair1) + pair2) + pair3


@triton.jit
def _block_sum(partials):
    warp = tl.reshape(partials, (8, 32))
    warp = _split_sum(warp, 8, 32)
    warp = _split_sum(warp, 8, 16)
    warp = _split_sum(warp, 8, 8)
    warp = _split_sum(warp, 8, 4)
    warp = _split_sum(warp, 8, 2)

    even, odd = tl.split(tl.reshape(warp, (2, 2, 2)))
    even0, even1 = tl.split(even)
    odd0, odd1 = tl.split(odd)
    w0, w4 = tl.split(even0)
    w2, w6 = tl.split(even1)
    w1, w5 = tl.split(odd0)
    w3, w7 = tl.split(odd1)
    return ((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7


@triton.jit
def _updated(X, R, addresses):
    x = tl.load(X + addresses).to(tl.float32)
    r = tl.load(R + addresses).to(tl.float32)
    updated = (x + r).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(R + addresses, updated)
    return updated.to(tl.float32)


@triton.jit
def _normalize(X, W, addresses, columns, z, scale):
    first_round = (z * scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    weight = tl.load(W + columns).to(tl.float32)
    tl.store(X + addresses, (first_round * weight).to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def grouped(X, R, W):
    row_offset = tl.program_id(0) * 4096
    columns = tl.arange(0, 256)[:, None] * 8 + tl.arange(0, 8)[None, :]
    z0 = _updated(X, R, row_offset + columns)
    z1 = _updated(X, R, row_offset + 2048 + columns)
    local_sum = _group_square_sum(z0) + _group_square_sum(z1)
    scale = tl.rsqrt(_block_sum(local_sum) / 4096.0 + 1e-6)
    _normalize(X, W, row_offset + columns, columns, z0, scale)
    _normalize(X, W, row_offset + 2048 + columns, 2048 + columns, z1, scale)


def fused_grouped(x, residual, weight):
    return grouped[(x.shape[0],)](x, residual, weight, num_warps=8, enable_fp_fusion=False)

"""Grouped BF16 residual RMSNorm with an explicit warp-level FP32 reduction."""
import triton
import triton.language as tl

from grouped_kernel import _group_square_sum, _normalize, _updated


@triton.jit
def _ordered_block_sum(partial, logical_thread):
    # One FP32 partial per physical lane. Every lane executes the barrier.
    return tl.inline_asm_elementwise(
        """{
        .shared .align 16 .b8 warp_sum[32];
        .reg .f32 total, peer, w0, w1, w2, w3, w4, w5, w6, w7;
        .reg .u32 lane, thread, warp;
        .reg .u64 base, offset, slot;
        .reg .pred leader, mismatch;

        mov.f32 total, $1;
        shfl.sync.bfly.b32 peer, total, 1, 31, -1;
        add.rn.f32 total, total, peer;
        shfl.sync.bfly.b32 peer, total, 2, 31, -1;
        add.rn.f32 total, total, peer;
        shfl.sync.bfly.b32 peer, total, 4, 31, -1;
        add.rn.f32 total, total, peer;
        shfl.sync.bfly.b32 peer, total, 8, 31, -1;
        add.rn.f32 total, total, peer;
        shfl.sync.bfly.b32 peer, total, 16, 31, -1;
        add.rn.f32 total, total, peer;

        mov.u32 lane, %laneid;
        mov.u32 thread, %tid.x;
        shr.u32 warp, thread, 5;
        mov.u64 base, warp_sum;
        mul.wide.u32 offset, warp, 4;
        add.u64 slot, base, offset;
        setp.eq.u32 leader, lane, 0;
        @leader st.shared.f32 [slot], total;
        bar.sync 0;

        ld.shared.v4.f32 {w0, w1, w2, w3}, [base];
        ld.shared.v4.f32 {w4, w5, w6, w7}, [base+16];
        add.rn.f32 total, w0, w1;
        add.rn.f32 total, total, w2;
        add.rn.f32 total, total, w3;
        add.rn.f32 total, total, w4;
        add.rn.f32 total, total, w5;
        add.rn.f32 total, total, w6;
        add.rn.f32 total, total, w7;
        setp.ne.u32 mismatch, thread, $2;
        @mismatch mov.b32 total, 0x7fc00000;
        mov.f32 $0, total;
        }""",
        constraints="=f,f,r",
        args=[partial, logical_thread],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def shuffled(X, R, W):
    row_offset = tl.program_id(0) * 4096
    thread = tl.arange(0, 256)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    z0 = _updated(X, R, row_offset + columns)
    z1 = _updated(X, R, row_offset + 2048 + columns)
    local_sum = _group_square_sum(z0) + _group_square_sum(z1)
    variance = _ordered_block_sum(local_sum, thread)
    scale = tl.rsqrt(variance / 4096.0 + 1e-6)[:, None]
    _normalize(X, W, row_offset + columns, columns, z0, scale)
    _normalize(X, W, row_offset + 2048 + columns, 2048 + columns, z1, scale)


def fused_shuffle(x, residual, weight):
    return shuffled[(x.shape[0],)](x, residual, weight, num_warps=8, enable_fp_fusion=False)

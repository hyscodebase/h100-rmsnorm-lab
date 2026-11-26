"""One vector per physical thread for the bit-matched 128-row RMSNorm."""
import triton
import triton.language as tl

from grouped_kernel import _group_square_sum, _normalize, _updated


@triton.jit
def _ordered_sixteen_warp_sum(partial, logical_thread):
    return tl.inline_asm_elementwise(
        """{
        .shared .align 16 .b8 physical_warp_sum[64];
        .reg .f32 total, peer, w0, w1, w2, w3;
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
        mov.u64 base, physical_warp_sum;
        mul.wide.u32 offset, warp, 4;
        add.u64 slot, base, offset;
        setp.eq.u32 leader, lane, 0;
        @leader st.shared.f32 [slot], total;
        bar.sync 0;

        // The warp totals retain the released vLLM order.
        ld.shared.v4.f32 {w0, w1, w2, w3}, [base];
        add.rn.f32 total, w0, w1;
        add.rn.f32 total, total, w2;
        add.rn.f32 total, total, w3;
        ld.shared.v4.f32 {w0, w1, w2, w3}, [base+16];
        add.rn.f32 total, total, w0;
        add.rn.f32 total, total, w1;
        add.rn.f32 total, total, w2;
        add.rn.f32 total, total, w3;
        ld.shared.v4.f32 {w0, w1, w2, w3}, [base+32];
        add.rn.f32 total, total, w0;
        add.rn.f32 total, total, w1;
        add.rn.f32 total, total, w2;
        add.rn.f32 total, total, w3;
        ld.shared.v4.f32 {w0, w1, w2, w3}, [base+48];
        add.rn.f32 total, total, w0;
        add.rn.f32 total, total, w1;
        add.rn.f32 total, total, w2;
        add.rn.f32 total, total, w3;
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
def sixteen_warp_rmsnorm(X, R, W):
    row_offset = tl.program_id(0) * 4096
    thread = tl.arange(0, 512)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    updated = _updated(X, R, row_offset + columns)
    partial = _group_square_sum(updated, 512)
    total = _ordered_sixteen_warp_sum(partial, thread)
    scale = tl.rsqrt(total / 4096.0 + 1e-6)[:, None]
    _normalize(X, W, row_offset + columns, columns, updated, scale)


def fused_sixteen_warps(x, residual, weight):
    return sixteen_warp_rmsnorm[(x.shape[0],)](x, residual, weight, num_warps=16, enable_fp_fusion=False)

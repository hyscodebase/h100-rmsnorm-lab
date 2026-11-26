"""One vector per physical thread for the bit-matched 128-row RMSNorm."""
import triton
import triton.language as tl

from grouped_kernel import _normalize, _updated


@triton.jit
def _square_sum_vectors(z):
    # Keep the older kernel and its compiled instructions fixed as a baseline.
    left, right = tl.split(tl.reshape(z, (512, 4, 2)))
    pairs = left * left + right * right
    even, odd = tl.split(tl.reshape(pairs, (512, 2, 2)))
    pair0, pair2 = tl.split(even)
    pair1, pair3 = tl.split(odd)
    return ((pair0 + pair1) + pair2) + pair3


@triton.jit
def _ordered_sixteen_warp_sum(partial, columns):
    # Match the assembly pack to the eight contiguous elements owned by one lane.
    return tl.inline_asm_elementwise(
        """{
        .shared .align 16 .b8 physical_warp_sum[64];
        .reg .f32 total, peer, w0, w1, w2, w3;
        .reg .u32 lane, thread, warp, column;
        .reg .u64 base, offset, slot;
        .reg .pred leader, mismatch;

        mov.f32 total, $8;
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
        shl.b32 column, thread, 3;
        setp.ne.u32 mismatch, column, $16;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $17, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $18, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $19, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $20, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $21, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $22, mismatch;
        add.u32 column, column, 1;
        setp.ne.or.u32 mismatch, column, $23, mismatch;
        @mismatch mov.b32 total, 0x7fc00000;
        mov.f32 $0, total;
        mov.f32 $1, total;
        mov.f32 $2, total;
        mov.f32 $3, total;
        mov.f32 $4, total;
        mov.f32 $5, total;
        mov.f32 $6, total;
        mov.f32 $7, total;
        }""",
        constraints=("=f,=f,=f,=f,=f,=f,=f,=f,"
                     "f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r"),
        args=[partial, columns],
        dtype=tl.float32,
        is_pure=False,
        pack=8,
    )


@triton.jit
def sixteen_warp_rmsnorm(X, R, W):
    row_offset = tl.program_id(0) * 4096
    thread = tl.arange(0, 512)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    updated = _updated(X, R, row_offset + columns)
    partial = _square_sum_vectors(updated)
    total = _ordered_sixteen_warp_sum(tl.broadcast_to(partial[:, None], (512, 8)), columns)
    scale = tl.rsqrt(total / 4096.0 + 1e-6)
    _normalize(X, W, row_offset + columns, columns, updated, scale)


def fused_sixteen_warps(x, residual, weight):
    return sixteen_warp_rmsnorm[(x.shape[0],)](x, residual, weight, num_warps=16, enable_fp_fusion=False)

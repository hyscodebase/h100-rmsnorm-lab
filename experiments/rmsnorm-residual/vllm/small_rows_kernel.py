"""Match the low-row vLLM reduction with 16 logical warps on eight physical warps."""
import triton
import triton.language as tl

from grouped_kernel import _group_square_sum, _normalize, _updated


@triton.jit
def _split_block_sum(low, high, logical_thread):
    # Each physical lane carries two independent logical warp values.
    return tl.inline_asm_elementwise(
        """{
        .shared .align 16 .b8 small_warp_sum[64];
        .reg .f32 lower, upper, peer0, peer1, total, w0, w1, w2, w3;
        .reg .u32 lane, thread, warp;
        .reg .u64 base, offset, slot;
        .reg .pred leader, mismatch;

        mov.f32 lower, $1;
        mov.f32 upper, $2;
        shfl.sync.bfly.b32 peer0, lower, 1, 31, -1;
        shfl.sync.bfly.b32 peer1, upper, 1, 31, -1;
        add.rn.f32 lower, lower, peer0;
        add.rn.f32 upper, upper, peer1;
        shfl.sync.bfly.b32 peer0, lower, 2, 31, -1;
        shfl.sync.bfly.b32 peer1, upper, 2, 31, -1;
        add.rn.f32 lower, lower, peer0;
        add.rn.f32 upper, upper, peer1;
        shfl.sync.bfly.b32 peer0, lower, 4, 31, -1;
        shfl.sync.bfly.b32 peer1, upper, 4, 31, -1;
        add.rn.f32 lower, lower, peer0;
        add.rn.f32 upper, upper, peer1;
        shfl.sync.bfly.b32 peer0, lower, 8, 31, -1;
        shfl.sync.bfly.b32 peer1, upper, 8, 31, -1;
        add.rn.f32 lower, lower, peer0;
        add.rn.f32 upper, upper, peer1;
        shfl.sync.bfly.b32 peer0, lower, 16, 31, -1;
        shfl.sync.bfly.b32 peer1, upper, 16, 31, -1;
        add.rn.f32 lower, lower, peer0;
        add.rn.f32 upper, upper, peer1;

        mov.u32 lane, %laneid;
        mov.u32 thread, %tid.x;
        shr.u32 warp, thread, 5;
        mov.u64 base, small_warp_sum;
        mul.wide.u32 offset, warp, 4;
        add.u64 slot, base, offset;
        setp.eq.u32 leader, lane, 0;
        @leader st.shared.f32 [slot], lower;
        @leader st.shared.f32 [slot+32], upper;
        bar.sync 0;

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
        setp.ne.u32 mismatch, thread, $3;
        @mismatch mov.b32 total, 0x7fc00000;
        mov.f32 $0, total;
        }""",
        constraints="=f,f,f,r",
        args=[low, high, logical_thread],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def split_warps(X, R, W):
    row_offset = tl.program_id(0) * 4096
    thread = tl.arange(0, 256)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    low = _updated(X, R, row_offset + columns)
    high = _updated(X, R, row_offset + 2048 + columns)
    variance = _split_block_sum(_group_square_sum(low), _group_square_sum(high), thread)
    scale = tl.rsqrt(variance / 4096.0 + 1e-6)[:, None]
    _normalize(X, W, row_offset + columns, columns, low, scale)
    _normalize(X, W, row_offset + 2048 + columns, 2048 + columns, high, scale)


def fused_split_warps(x, residual, weight):
    return split_warps[(x.shape[0],)](x, residual, weight, num_warps=8, enable_fp_fusion=False)

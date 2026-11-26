"""Assign vLLM's eight logical warp groups to four physical warps."""
import triton
import triton.language as tl

from eight_warp_outputs import _add_and_write
from grouped_kernel import _normalize
from sixteen_warp_kernel import validate_output_buffers


@triton.jit
def _sum_squares(z):
    left, right = tl.split(tl.reshape(z, (128, 4, 2)))
    pairs = left * left + right * right
    even, odd = tl.split(tl.reshape(pairs, (128, 2, 2)))
    pair0, pair2 = tl.split(even)
    pair1, pair3 = tl.split(odd)
    return ((pair0 + pair1) + pair2) + pair3


@triton.jit
def _ordered_sum(lower, upper, physical_thread):
    return tl.inline_asm_elementwise(
        """{
        .shared .align 16 .b8 warp_sum[32];
        .reg .f32 lo, hi, peer_lo, peer_hi, total, w0, w1, w2, w3, w4, w5, w6, w7;
        .reg .u32 lane, thread, warp;
        .reg .u64 base, offset, slot;
        .reg .pred leader, mismatch;

        mov.f32 lo, $1;
        mov.f32 hi, $2;
        shfl.sync.bfly.b32 peer_lo, lo, 1, 31, -1;
        shfl.sync.bfly.b32 peer_hi, hi, 1, 31, -1;
        add.rn.f32 lo, lo, peer_lo;
        add.rn.f32 hi, hi, peer_hi;
        shfl.sync.bfly.b32 peer_lo, lo, 2, 31, -1;
        shfl.sync.bfly.b32 peer_hi, hi, 2, 31, -1;
        add.rn.f32 lo, lo, peer_lo;
        add.rn.f32 hi, hi, peer_hi;
        shfl.sync.bfly.b32 peer_lo, lo, 4, 31, -1;
        shfl.sync.bfly.b32 peer_hi, hi, 4, 31, -1;
        add.rn.f32 lo, lo, peer_lo;
        add.rn.f32 hi, hi, peer_hi;
        shfl.sync.bfly.b32 peer_lo, lo, 8, 31, -1;
        shfl.sync.bfly.b32 peer_hi, hi, 8, 31, -1;
        add.rn.f32 lo, lo, peer_lo;
        add.rn.f32 hi, hi, peer_hi;
        shfl.sync.bfly.b32 peer_lo, lo, 16, 31, -1;
        shfl.sync.bfly.b32 peer_hi, hi, 16, 31, -1;
        add.rn.f32 lo, lo, peer_lo;
        add.rn.f32 hi, hi, peer_hi;

        mov.u32 lane, %laneid;
        mov.u32 thread, %tid.x;
        shr.u32 warp, thread, 5;
        mov.u64 base, warp_sum;
        mul.wide.u32 offset, warp, 4;
        add.u64 slot, base, offset;
        setp.eq.u32 leader, lane, 0;
        @leader st.shared.f32 [slot], lo;
        @leader st.shared.f32 [slot+16], hi;
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
        setp.ne.u32 mismatch, thread, $3;
        @mismatch mov.b32 total, 0x7fc00000;
        mov.f32 $0, total;
        }""",
        constraints="=f,f,f,r",
        args=[lower, upper, physical_thread],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def fused_outputs_four_warps(x, residual, weight, x_out, residual_out):
    row_start = tl.program_id(0) * 4096
    thread = tl.arange(0, 128)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    # The second logical lane is 128 lanes, or 1,024 columns, ahead.
    lower_first = _add_and_write(x, residual, residual_out, row_start + columns)
    lower_second = _add_and_write(x, residual, residual_out, row_start + 2048 + columns)
    upper_first = _add_and_write(x, residual, residual_out, row_start + 1024 + columns)
    upper_second = _add_and_write(x, residual, residual_out, row_start + 3072 + columns)
    lower = _sum_squares(lower_first) + _sum_squares(lower_second)
    upper = _sum_squares(upper_first) + _sum_squares(upper_second)
    variance = _ordered_sum(lower, upper, thread)
    scale = tl.rsqrt(variance / 4096.0 + 1e-6)[:, None]
    _normalize(x_out, weight, row_start + columns, columns, lower_first, scale)
    _normalize(x_out, weight, row_start + 2048 + columns, 2048 + columns, lower_second, scale)
    _normalize(x_out, weight, row_start + 1024 + columns, 1024 + columns, upper_first, scale)
    _normalize(x_out, weight, row_start + 3072 + columns, 3072 + columns, upper_second, scale)


def fused_out_of_place(x, residual, weight, x_out, residual_out):
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        return fused_outputs_four_warps[(x.shape[0],)](
            x, residual, weight, x_out, residual_out, num_warps=4, enable_fp_fusion=False,
        )

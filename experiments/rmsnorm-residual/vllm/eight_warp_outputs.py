"""Copy-free residual RMSNorm with the ordered 256-thread vLLM reduction."""
import triton
import triton.language as tl

from grouped_kernel import _group_square_sum, _normalize
from shuffle_kernel import _ordered_block_sum
from sixteen_warp_kernel import validate_output_buffers
from two_kernel import ADD_BLOCK, launch_residual_add


@triton.jit
def _row_indices():
    row_start = tl.program_id(0) * 4096
    thread = tl.arange(0, 256)
    columns = thread[:, None] * 8 + tl.arange(0, 8)[None, :]
    return row_start, thread, columns


@triton.jit
def _add_and_write(x, residual, residual_out, addresses):
    x_fp32 = tl.load(x + addresses).to(tl.float32)
    residual_fp32 = tl.load(residual + addresses).to(tl.float32)
    rounded = (x_fp32 + residual_fp32).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(residual_out + addresses, rounded)
    return rounded.to(tl.float32)


@triton.jit
def _normalize_pair(x_out, weight, row_start, thread, columns, low, high):
    local_sum = _group_square_sum(low) + _group_square_sum(high)
    variance = _ordered_block_sum(local_sum, thread)
    scale = tl.rsqrt(variance / 4096.0 + 1e-6)[:, None]
    _normalize(x_out, weight, row_start + columns, columns, low, scale)
    _normalize(x_out, weight, row_start + 2048 + columns, 2048 + columns, high, scale)


@triton.jit
def fused_outputs(x, residual, weight, x_out, residual_out):
    row_start, thread, columns = _row_indices()
    low = _add_and_write(x, residual, residual_out, row_start + columns)
    high = _add_and_write(x, residual, residual_out, row_start + 2048 + columns)
    _normalize_pair(x_out, weight, row_start, thread, columns, low, high)


@triton.jit
def normalize_written_residual(residual, weight, x_out):
    row_start, thread, columns = _row_indices()
    low = tl.load(residual + row_start + columns).to(tl.float32)
    high = tl.load(residual + row_start + 2048 + columns).to(tl.float32)
    _normalize_pair(x_out, weight, row_start, thread, columns, low, high)


def fused_out_of_place(x, residual, weight, x_out, residual_out):
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        return fused_outputs[(x.shape[0],)](
            x, residual, weight, x_out, residual_out, num_warps=8, enable_fp_fusion=False,
        )


def launch_normalization(residual_out, weight, x_out):
    return normalize_written_residual[(residual_out.shape[0],)](
        residual_out, weight, x_out, num_warps=8, enable_fp_fusion=False,
    )


def split_out_of_place(x, residual, weight, x_out, residual_out, *, tile=ADD_BLOCK):
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        add = launch_residual_add(x, residual, residual_out, tile=tile)
        norm = launch_normalization(residual_out, weight, x_out)
    return add, norm

"""Forward-only RMSNorm for contiguous, two-dimensional CUDA BF16 activations."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    offset = row * WIDTH + col
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + EPS)
    tl.store(Y + offset, x * inv_rms * weight, mask=mask)


@triton.jit
def _rmsnorm_rows_kernel(
    X, W, Y, rows, WIDTH: tl.constexpr, EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    first_row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    row = first_row + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_COLS)
    offset = row[:, None] * WIDTH + col[None, :]
    valid = (row[:, None] < rows) & (col[None, :] < WIDTH)
    x = tl.load(X + offset, mask=valid, other=0.0).to(tl.float32)
    weight = tl.load(W + col, mask=col < WIDTH, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / WIDTH + EPS)
    tl.store(Y + offset, x * inv_rms[:, None] * weight[None, :], mask=valid)


def _validate_rmsnorm_inputs(x, weight, output, eps, num_warps, rows_per_program):
    if x.ndim != 2 or x.shape[0] == 0 or not 1 <= x.shape[1] <= 16384:
        raise ValueError("expected x with positive rows and width in [1, 16384]")
    if weight.shape != (x.shape[1],):
        raise ValueError("weight must have one element per column")
    if not x.is_cuda or weight.device != x.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("x and weight must use torch.bfloat16")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if output.shape != x.shape or output.dtype != torch.bfloat16 or output.device != x.device:
        raise ValueError("output must match x shape, BF16 dtype, and device")
    if not output.is_contiguous() or output._base is not None:
        raise ValueError("output must be a contiguous base tensor")
    input_storage = x.untyped_storage().data_ptr()
    weight_storage = weight.untyped_storage().data_ptr()
    output_storage = output.untyped_storage().data_ptr()
    if output_storage in (input_storage, weight_storage):
        raise ValueError("output must not alias x or weight storage")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("this kernel supports forward inference only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if rows_per_program not in (1, 2, 4):
        raise ValueError("rows_per_program must be 1, 2 or 4")


def rmsnorm_into(x, weight, output, eps=1e-6, num_warps=4, rows_per_program=1):
    """Write BF16 RMSNorm into a distinct, caller-owned output buffer."""
    _validate_rmsnorm_inputs(x, weight, output, eps, num_warps, rows_per_program)
    rows, width = x.shape
    with torch.cuda.device(x.device):
        block_cols = triton.next_power_of_2(width)
        if rows_per_program == 1:
            _rmsnorm_kernel[(rows,)](
                x, weight, output, width, eps, block_cols, num_warps=num_warps
            )
        else:
            grid = (triton.cdiv(rows, rows_per_program),)
            _rmsnorm_rows_kernel[grid](
                x, weight, output, rows, width, eps, rows_per_program, block_cols,
                num_warps=num_warps,
            )
    return output


def rmsnorm(x, weight, eps=1e-6, num_warps=4, rows_per_program=1):
    """Normalize each row and return a newly allocated BF16 result tensor."""
    output = torch.empty_like(x)
    return rmsnorm_into(x, weight, output, eps, num_warps, rows_per_program)

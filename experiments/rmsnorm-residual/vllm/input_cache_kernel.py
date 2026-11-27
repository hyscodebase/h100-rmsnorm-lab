"""Eight-warp fused residual RMSNorm with L1-bypassing input loads."""
import triton
import triton.language as tl

from eight_warp_outputs import _normalize_pair, _row_indices
from sixteen_warp_kernel import validate_output_buffers


@triton.jit
def _add_and_write_cg(x, residual, residual_out, addresses):
    x_fp32 = tl.load(x + addresses, cache_modifier=".cg").to(tl.float32)
    residual_fp32 = tl.load(residual + addresses, cache_modifier=".cg").to(tl.float32)
    rounded = (x_fp32 + residual_fp32).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(residual_out + addresses, rounded)
    return rounded.to(tl.float32)


@triton.jit
def fused_outputs_cg_inputs(x, residual, weight, x_out, residual_out):
    row_start, thread, columns = _row_indices()
    low = _add_and_write_cg(x, residual, residual_out, row_start + columns)
    high = _add_and_write_cg(x, residual, residual_out, row_start + 2048 + columns)
    _normalize_pair(x_out, weight, row_start, thread, columns, low, high)


def fused_out_of_place(x, residual, weight, x_out, residual_out):
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        return fused_outputs_cg_inputs[(x.shape[0],)](
            x, residual, weight, x_out, residual_out, num_warps=8, enable_fp_fusion=False,
        )

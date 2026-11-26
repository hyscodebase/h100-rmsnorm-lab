"""Copy-free BF16 residual addition followed by the matched SM90 RMSNorm."""
import triton
import triton.language as tl

from sixteen_warp_kernel import compile_residual_normalization, validate_output_buffers

ADD_BLOCK, ADD_WARPS = 2048, 4


@triton.jit
def residual_add(x, residual, residual_out, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    x_fp32 = tl.load(x + offsets).to(tl.float32)
    residual_fp32 = tl.load(residual + offsets).to(tl.float32)
    rounded = (x_fp32 + residual_fp32).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(residual_out + offsets, rounded)


def launch_residual_add(x, residual, residual_out):
    return residual_add[(x.numel() // ADD_BLOCK,)](
        x, residual, residual_out, ADD_BLOCK, num_warps=ADD_WARPS, enable_fp_fusion=False,
    )


def split_out_of_place(x, residual, weight, x_out, residual_out):
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        add = launch_residual_add(x, residual, residual_out)
        norm = compile_residual_normalization()
        norm[(x.shape[0], 1, 1)](residual_out, weight, x_out)
    return add, norm

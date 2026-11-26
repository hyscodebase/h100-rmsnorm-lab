"""Launch the 16-warp RMSNorm with an explicit one-vector-per-lane layout."""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

TEMPLATE = Path(__file__).with_name("sixteen_warp.ttgir.in")

# One ordered collective per block. The shared slot is indexed by physical warp,
# and the logical-thread guard makes an unexpected lane mapping fail validation.
REDUCTION = r"""
{
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
        }
"""


def render_kernel(*, out_of_place=False):
    # MLIR string literals encode a newline as the byte escape \0A.
    asm = REDUCTION.strip().replace("\n", r"\0A")
    name = "sixteen_warp_rmsnorm"
    args, addresses = "", ""
    activation, residual = "%x_addr", "%r_addr"
    if out_of_place:
        name += "_out"
        args = (
            ",\n      %x_out: !tt.ptr<bf16> {tt.divisibility = 16 : i32}"
            ",\n      %r_out: !tt.ptr<bf16> {tt.divisibility = 16 : i32}"
        )
        addresses = (
            "    %x_out_base = tt.splat %x_out : !tt.ptr<bf16> -> tensor<512x8x!tt.ptr<bf16>, #tile>\n"
            "    %r_out_base = tt.splat %r_out : !tt.ptr<bf16> -> tensor<512x8x!tt.ptr<bf16>, #tile>\n"
            "    %x_out_addr = tt.addptr %x_out_base, %positions : tensor<512x8x!tt.ptr<bf16>, #tile>, tensor<512x8xi32, #tile>\n"
            "    %r_out_addr = tt.addptr %r_out_base, %positions : tensor<512x8x!tt.ptr<bf16>, #tile>, tensor<512x8xi32, #tile>\n"
        )
        activation, residual = "%x_out_addr", "%r_out_addr"
    ir = TEMPLATE.read_text()
    for placeholder, value in (
        ("KERNEL_NAME", "@" + name), ("ORDERED_SUM", asm), ("OUTPUT_ARGUMENTS", args),
        ("OUTPUT_ADDRESSES", addresses), ("ACTIVATION_DESTINATION", activation), ("RESIDUAL_DESTINATION", residual),
    ):
        ir = ir.replace(f"@{placeholder}@", value)
    return name, ir


def render_normalization():
    fused_name, ir = render_kernel(out_of_place=True)
    name = "sixteen_warp_rmsnorm_from_residual"
    module = ir[:ir.index(f"  tt.func public @{fused_name}(")]
    row_layout = ir[ir.index("    %width ="):ir.index("    %x_base =")]
    normalization = ir[ir.index("    // Sum adjacent squared pairs"):]
    signature = f"""  tt.func public @{name}(
      %residual: !tt.ptr<bf16> {{tt.divisibility = 16 : i32}},
      %weight: !tt.ptr<bf16> {{tt.divisibility = 16 : i32}},
      %x_out: !tt.ptr<bf16> {{tt.divisibility = 16 : i32}}) {{
"""
    load_residual = """    %r_base = tt.splat %residual : !tt.ptr<bf16> -> tensor<512x8x!tt.ptr<bf16>, #tile>
    %r_addr = tt.addptr %r_base, %positions : tensor<512x8x!tt.ptr<bf16>, #tile>, tensor<512x8xi32, #tile>
    %rounded = tt.load %r_addr : tensor<512x8x!tt.ptr<bf16>, #tile>
    %x_out_base = tt.splat %x_out : !tt.ptr<bf16> -> tensor<512x8x!tt.ptr<bf16>, #tile>
    %x_out_addr = tt.addptr %x_out_base, %positions : tensor<512x8x!tt.ptr<bf16>, #tile>, tensor<512x8xi32, #tile>

"""
    return name, module + signature + row_layout + load_residual + normalization


def compile_ttgir(name, ir):
    import triton
    from triton.backends.compiler import GPUTarget

    with TemporaryDirectory(prefix="rmsnorm-ttgir-") as directory:
        path = Path(directory) / f"{name}.ttgir"
        path.write_text(ir)
        return triton.compile(str(path), target=GPUTarget("cuda", 90, 32), options={"enable_fp_fusion": False})


@lru_cache(maxsize=2)
def compile_kernel(*, out_of_place=False):
    return compile_ttgir(*render_kernel(out_of_place=out_of_place))


@lru_cache(maxsize=1)
def compile_residual_normalization():
    return compile_ttgir(*render_normalization())


def fused_sixteen_warps(x, residual, weight):
    compiled = compile_kernel()
    compiled[(x.shape[0], 1, 1)](x, residual, weight)
    return compiled


def validate_output_buffers(x, residual, weight, x_out, residual_out):
    import torch

    sources, outputs = (x, residual, weight), (x_out, residual_out)
    if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 4096 or x.numel() > 2**31
            or any(t.shape != x.shape for t in (residual, *outputs)) or weight.shape != (4096,)
            or any(not t.is_cuda or t.device != x.device or t.dtype != torch.bfloat16
                   or not t.is_contiguous() or t.data_ptr() % 16 for t in (*sources, *outputs))):
        raise ValueError("Use aligned contiguous CUDA BF16 buffers at width 4096.")

    def span(tensor):
        start = tensor.data_ptr()
        return start, start + tensor.numel() * tensor.element_size()

    occupied = [span(t) for t in sources]
    for tensor in outputs:
        start, end = span(tensor)
        if any(start < previous_end and previous_start < end for previous_start, previous_end in occupied):
            raise ValueError("Each output must be disjoint from the inputs and the other output.")
        occupied.append((start, end))
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError("The explicit kernel targets SM90.")


def fused_out_of_place(x, residual, weight, x_out, residual_out):
    """Write contiguous BF16 outputs to disjoint buffers on an SM90 GPU."""
    import torch

    validate_output_buffers(x, residual, weight, x_out, residual_out)
    with torch.cuda.device(x.device):
        compiled = compile_kernel(out_of_place=True)
        compiled[(x.shape[0], 1, 1)](x, residual, weight, x_out, residual_out)
    return compiled

"""Launch the 16-warp RMSNorm with an explicit one-vector-per-lane layout."""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget

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


@lru_cache(maxsize=1)
def compile_kernel():
    # MLIR string literals encode a newline as the byte escape \0A.
    asm = REDUCTION.strip().replace("\n", r"\0A")
    ir = TEMPLATE.read_text().replace("@ORDERED_SUM@", asm)
    with TemporaryDirectory(prefix="rmsnorm-ttgir-") as directory:
        path = Path(directory) / "sixteen_warp_rmsnorm.ttgir"
        path.write_text(ir)
        return triton.compile(str(path), target=GPUTarget("cuda", 90, 32), options={"enable_fp_fusion": False})


def fused_sixteen_warps(x, residual, weight):
    compiled = compile_kernel()
    compiled[(x.shape[0], 1, 1)](x, residual, weight)
    return compiled

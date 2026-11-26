# Grouped FP32 reduction on H100

The grouped eight-warp Triton kernel reproduced the released vLLM operator's BF16 output bits on all five tested seeds. It cleared all **47** previously rejected coordinates after first reproducing them with the original four-warp kernel. The measured grouped version had **7.37–7.82% higher latency** than the released vLLM operator. The 5% improvement gate remains unmet. [Original boundary failures](REPORT.md) · [Frozen reduction plan](REDUCTION_PLAN.md) · [PR #13](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/13)

| Seed | vLLM 0.10.0 µs | Grouped Triton µs | Triton latency change |
| ---: | ---: | ---: | ---: |
| [118](grouped-results/s118.json) | 13.763 | 14.778 | +7.37% |
| [119](grouped-results/s119.json) | 13.714 | 14.786 | +7.82% |
| [120](grouped-results/s120.json) | 13.733 | 14.786 | +7.67% |

Each timing seed records 21 CUDA-event samples of 32 in-place calls. Both implementations run on disjoint sets of identical seeded buffer pairs. The measurement stream restores the inputs before recording the start event. Each graph output is checked before timing. [Timing and verification code](grouped_run.py) · [Attempt records](grouped-results/attempts)

## Correctness and compiler output

Seed 114 confirms the earlier 47 discrepancies and records the two Triton encodings and the vLLM encoding at each coordinate. A fresh seed 117 passed both exact BF16 rounding probes and the signed-zero and residual checks. Across all five seeds, 40 fixture checks and 96 captured-call checks covered **570,425,344 normalized elements**. All corresponding output bits agreed under the registered shape, distributions and environment. The FP32 bound accepted 1,349 elements beyond the strict tolerance with both kernels agreeing at those coordinates. [All grouped checkpoints](grouped-results) · [FP32 numerical rule](PRECISION_PLAN.md)

The generated PTX includes six vectorized global loads, 128 static shared-memory load instructions and 271 FP32 add instructions. The compiler reported eight warps, 32 registers per thread, 2,048 bytes of shared memory and zero spills. These counts describe emitted instructions. The next experiment tests whether explicit warp exchanges preserve the arithmetic order and reduce the instruction footprint and latency. [Codegen metadata](grouped-results/s114.json) · [Warp exchange plan](SHUFFLE_PLAN.md)

## Verify the result

Run the report from a checkout containing the recorded source revision. It verifies compiler-instruction hashes across checkpoints and recomputes every median from the original intervals.

```bash
python3 experiments/rmsnorm-residual/vllm/grouped_report.py
```

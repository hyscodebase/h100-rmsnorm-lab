# Bit-matched residual RMSNorm at 128 token rows

The 128 × 4,096 experiment reproduced all 238 prior bit disagreements and 126 registered numerical rejections from the fixed-order 256-thread Triton kernel. A split-warp Triton candidate matched every checked BF16 activation bit and preserved the exact updated residual. The three archived seeds and a separate held-out seed passed before timing. [Registered protocol](SMALL_ROWS.md) · [Frozen implementation](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/d998b0d) · [PR #19](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/19)

At width 4,096, the released vLLM version uses a 1,024-thread block below 256 rows. Of those, 512 threads each handle one eight-element vector. Each physical thread in the candidate still loads two vectors. It accumulates those vectors separately, applies the same five warp reductions to both, and sums the 16 logical warp totals in order. The emitted PTX contains ten butterfly shuffles, four vector shared loads, and a 64-byte static shared buffer. The kernel uses 32 registers per thread and records zero spills. [Released CUDA source](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.0/csrc/layernorm_kernels.cu) · [Kernel](small_rows_kernel.py)

## Latency and eligibility

| Seed | vLLM µs | Split-warp Triton µs | Latency reduction | 5% criterion |
| ---: | ---: | ---: | ---: | --- |
| 130 | 4.011 | 3.965 | 1.147% | Below threshold |
| 131 | 4.016 | 3.974 | 1.046% | Below threshold |
| 132 | 4.022 | 3.980 | 1.044% | Below threshold |

The candidate matched all 146,800,640 compared normalized elements across seven checkpoints. All three timing seeds showed a latency reduction of 1.044–1.147%, below the registered threshold. The next experiment keeps the same reduction tree and assigns one eight-element vector per physical lane in a 16-warp block. [Follow-up plan](SMALL_ROWS_16W.md)

Each process verified eight fixtures and 32 disjoint captured pairs. The three timing seeds record 126 original CUDA event intervals from 21 randomized replay rounds per backend. The archive includes a separate set of 23 whole-device sensor samples from the seven attempt windows. The operator result applies to this H100 and the released vLLM 0.10.0 CUDA extension. [Raw numerical checks, CUDA events and device samples](small-rows-results)

## Reproduce

```bash
python3 experiments/rmsnorm-residual/vllm/small_rows_report.py
```

The verifier checks the frozen Git file hashes, reproduces the archived error counts and recomputes the latency statistics from CUDA events. To repeat one numerical checkpoint on the same GPU and software, prepare the pinned vLLM release and run the frozen source with the archived input protocol. [Wheel preparation](REPORT.md#reproduce-the-numerical-checkpoint)

```bash
BENCH_REVISION=d998b0dc273d37b7543e153f76e0e020a731c69f \
  python3 experiments/rmsnorm-residual/vllm/small_rows_run.py \
    --seed 126 --asset-dir .lab/vllm-release
```

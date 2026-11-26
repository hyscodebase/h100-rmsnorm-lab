# Multi-row BF16 RMSNorm on H100

**Keep one row per Triton program at 1,024 token rows × 8,192 hidden units.** The fastest multi-row kernel used two rows and eight warps. Its median latency was 1.99% and 2.41% higher than the fastest one-row kernel in two independent runs. All 60 numerical checks passed. [Sprint decision rule](../../GOAL.md) · [Correctness results](correctness.json)

## Target shape

The decision required at least 5% lower median latency than **both** the fastest one-row Triton kernel and shape-specialized `torch.compile` in **each** run. The one-row, eight-warp reference took 8.463 and 8.386 µs. Compiled PyTorch took 8.726 and 8.628 µs. Each cell lists seed 42 followed by seed 43. A negative delta means the candidate took less time than its reference.

| Candidate (rows / warps) | Median (µs) | Latency Δ vs. best one-row | Latency Δ vs. compiled PyTorch | Qualifies |
| --- | ---: | ---: | ---: | --- |
| 2 / 4 | 9.116 / 9.084 | +7.72% / +8.32% | +4.47% / +5.29% | No |
| 2 / 8 | 8.631 / 8.588 | +1.99% / +2.41% | −1.09% / −0.46% | No |
| 4 / 4 | 20.659 / 20.742 | +144.11% / +147.34% | +136.75% / +140.40% | No |
| 4 / 8 | 9.527 / 9.513 | +12.57% / +13.44% | +9.18% / +10.26% | No |

## Control shapes

At 1,024 × 4,096, two rows and four warps won both runs, reducing latency by 1.61% and 1.94% relative to the best one-row kernel. At all six smaller controls a one-row kernel led. The table compares the fastest existing baseline (eager, compiled or one-row) with the fastest multi-row candidate in each run. All values are median microseconds in seed order 42 / 43.

| Rows × width | Fastest existing baseline | Baseline (µs) | Best multi-row (µs) |
| --- | --- | ---: | ---: |
| 1 × 4,096 | 1 row, 8 warps | 1.850 / 1.816 | 1.949 / 1.917 |
| 16 × 4,096 | 1 row, 8 warps | 1.916 / 1.886 | 2.153 / 2.122 |
| 128 × 4,096 | 1 row, 8 warps | 2.087 / 2.055 | 2.265 / 2.229 |
| 1,024 × 4,096 | 1 row, 4 warps | 4.910 / 4.887 | 4.831 / 4.792 |
| 1 × 8,192 | 1 row, 8 warps | 2.153 / 2.180 | 2.216 / 2.242 |
| 16 × 8,192 | 1 row, 8 warps | 2.305 / 2.312 | 2.703 / 2.718 |
| 128 × 8,192 | 1 row, 8 then 4 warps | 2.695 / 2.651 | 2.976 / 2.932 |

## Reproduction

Both processes ran revision `1a18288b740cd44a2508393119be7192bf27cae7` on one NVIDIA H100 80GB HBM3 with PyTorch 2.7.1+cu128, Triton 3.3.1 and driver 550.144.03. CUDA events timed 21 graph replays per backend and shape, each with 32 operations. Each round shuffled backend order. Inputs reused the same memory addresses without an explicit L2 flush. [Seed 42 timings](run-42.json) · [Seed 43 timings](run-43.json)

Run from the repository root in the GPU environment. The numerical suite uses seed 42, BF16 inputs and an FP32 reference with `eps=1e-6`, `rtol=0.005` and `atol=1e-6`. The ten cases include the target shape and a partial final tile at 7 × 3,073. The benchmark also checks every measured backend against FP32 before timing.

```bash
python check.py
for seed in 42 43; do
  BENCH_REVISION=$(git rev-parse HEAD) python benchmark.py --compiled \
    --rows 1 16 128 1024 --widths 4096 8192 --row-tiles 1 2 4 \
    --ops 32 --rounds 21 --seed "$seed" --output "replay-$seed.json"
done
```

The earlier four-backend runs ranked compiled PyTorch first on the target. In these eight-backend runs, one-row Triton led. The decision uses references measured in the same run as each candidate. The results apply to this single-operator graph-replay workload. End-to-end serving and FlashInfer require separate measurements. [Earlier first run](../h100.json) · [Earlier repeat](../h100-repeat.json)

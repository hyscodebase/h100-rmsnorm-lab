# Sustained BF16 prefill projection on H100

A large language model's feed-forward projection can turn thousands of hidden vectors into one dense matrix multiplication. This experiment uses synthetic BF16 inputs with hidden width 4,096 and projection width 11,008. It varies token rows across 2,048, 4,096, 8,192 and 16,384. Each checkpoint runs one shape and seed using a preallocated output and 45 seconds of measured work after warmup. [Experiment plan](../../plans/prefill-gemm.json)

The primary measure is completed dense floating-point work divided by elapsed wall time, in TFLOP/s. CUDA events give an additional throughput estimate for the timed batches. We sample GPU activity, memory and power once per second within the measured window. The compute work uses `2 × M × K × N` FLOPs per matrix product. User-visible GPU activity counts intervals with active kernels and does not directly measure Tensor Core efficiency. [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) · [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

A shape qualifies as a sustained high-load workload if at least four of its five seeds average 95% device activity or more. Following review of the first checkpoint, a seed counts toward that device-activity decision only if it has at least 75% of the planned one-second readings (rounded up) and no gap above five seconds, including both window edges. Sparse samples still yield valid throughput measurements. Rank qualifying shapes by median wall TFLOP/s across their five seeds. If the top two medians are within 3%, report the near tie. The first baseline uses `torch.mm`. A direct Triton comparison can use the same operands and test the strongest observed shape.

## Evidence archive

The [experiment log](LOG.md) links changes to their measurements, review findings and corrections.

[Per-run index (CSV)](evidence/run-index.csv) · [Data dictionary and coverage](evidence/README.md) · [Time-aligned supplementary readings](evidence/supplemental) · [Execution attempts](evidence/attempts)

## First measured checkpoint

With 8,192 token rows and seed 44, the H100 completed 41,664 BF16 matrix products in 45.136 seconds. This corresponds to **681.9 TFLOP/s** by wall time. All 39 in-window device samples reported 100% GPU activity. Mean power was **698.8 W**, and peak observed device memory was **1,036 MiB**. The 256 sampled output entries passed the FP32 comparison. The largest gap between usable readings and the window edges was 7.56 seconds, so this seed contributes throughput evidence only under the post-pilot coverage rule. The shape ranking requires all seeds and shapes. All seeds use the same shape order. Small cross-shape differences warrant a counterbalanced rerun. [Raw timings and telemetry](results/s44-m8192-k4096-n11008.json)

Generate the current comparison directly from the archived checkpoints with `python3 experiments/prefill-gemm/summarize.py`.

## Reproduce

From a Linux checkout with CUDA PyTorch and nvidia-smi available:

```bash
python experiments/prefill-gemm/benchmark.py --m 8192 --k 4096 --n 11008 --seed 44 --seconds 45
```

The JSON includes the full batch counts, durations, 16×16 sampled FP32 output checks over the full inner dimension, and in-window telemetry samples. These are single-projection measurements. Serving also includes attention, transfers and other model operations.

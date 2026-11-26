# Sustained BF16 prefill projection on H100

A large language model's feed-forward projection can turn thousands of hidden vectors into one dense matrix multiplication. This experiment uses synthetic BF16 inputs with hidden width 4,096 and projection width 11,008. It varies token rows across 2,048, 4,096, 8,192 and 16,384. Each checkpoint runs one shape and seed using a preallocated output and 45 seconds of measured work after warmup. [Experiment plan](../../plans/prefill-gemm.json)

The primary measure is completed dense floating-point work divided by elapsed wall time, in TFLOP/s. CUDA events give an additional throughput estimate for the timed batches. We sample GPU activity, memory and power once per second within the measured window. The compute work uses `2 × M × K × N` FLOPs per matrix product. User-visible GPU activity counts intervals with active kernels and does not directly measure Tensor Core efficiency. [NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) · [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

A shape qualifies as a sustained high-load workload if at least four of its five seeds average 95% device activity or more. Rank qualifying shapes by median wall TFLOP/s across their five seeds. If the top two medians are within 3%, report the near tie. The first baseline uses `torch.mm`. A direct Triton comparison can use the same operands and test the strongest observed shape.

From a Linux checkout with CUDA PyTorch and nvidia-smi available:

```bash
python experiments/prefill-gemm/benchmark.py --m 8192 --k 4096 --n 11008 --seed 44 --seconds 45
```

The JSON includes the full batch counts, durations, 16×16 sampled FP32 output checks over the full inner dimension, and in-window telemetry samples. These are single-projection measurements. Serving also includes attention, transfers and other model operations.

# H100 prefill observer comparison

On a single H100 the 8,192-row BF16 projection reached 680.10 median wall TFLOP/s in the previous campaign. Device-wide samples frequently had multi-second gaps. This follow-up asks whether a direct NVML reader produces more complete observations while the GPU performs the same arithmetic. The counters are device-wide. [Previous results](../prefill-gemm/REPORT.md) · [Frozen protocol and budget](../prefill-gemm/NEXT.md) · [Issue #5](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/5)

Test one seed and observer arm at a time. A uses one nvidia-smi process per poll, B calls the NVML shared library in a persistent worker, and C collects both at the same deadlines with independent workers. The benchmark matches both readers to the active CUDA device by its physical UUID. The archive retains only public device properties. Polls can time out or be skipped, and a completed GEMM with no usable samples still contributes a coverage failure. [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [Benchmark](benchmark.py) · [Observer implementation](sensors.py)

## Run one checkpoint

Use a Linux checkout with an H100, CUDA-enabled PyTorch, the NVIDIA driver library and nvidia-smi. The default local runner uses the current Python interpreter. The same controller can call a configured SSH runner. A fresh state directory keeps the prior experiment available.

```bash
python3 lab.py init --state-dir .lab/observer --plan plans/observer-coverage.json
python3 lab.py step --state-dir .lab/observer
python3 lab.py status --state-dir .lab/observer
```

The first step is read-only preflight. The following 18 steps measure one 45-second GEMM window each. To stop after a running checkpoint, use `python3 lab.py pause --state-dir .lab/observer`; use `resume` before the next step. The controller pins the numerical source revision, caps a unit at 90 seconds, reserves a full watchdog interval for an uncertain exit and limits extra compute attempts to two per campaign. Its 1,800-second allocation accounting covers preflight and active attempts. Time spent offline between attempts consumes no GPU allocation.

Each completed arm produces a source revision, seed, UTC start time, driver/runtime and device properties, sampled numerical reference, raw CUDA-event batches, and one record for each of the 45 scheduled polling slots per reader. Every attempt is journaled before dispatch. A stop or a failure preserves previous checkpoints. Each observer arm supplies its own monitoring data to avoid interference from an extra polling process.

The final comparison requires all 18 timed checkpoints. Adoption depends on coverage in standalone runs, within-seed throughput ratios, and agreement in simultaneous paired windows. [Analysis and decision rules](analyze.py)

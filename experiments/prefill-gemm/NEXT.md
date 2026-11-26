# Does direct NVML polling improve H100 activity coverage during BF16 GEMM?

**Preregistration.** Freeze this protocol and the eventual runner commit before collecting the first comparison. In the previous 8,192-row pilot, 39 in-window samples showed 100% GPU activity, yet the largest gap was 7.56 seconds. That run missed the five-second coverage criterion. The next experiment tests how often each observation method meets the same criterion during sustained compute. Both paths read device-wide NVIDIA driver counters. [Pilot record](results/s44-m8192-k4096-n11008.json) · [Existing coverage calculation](summarize.py)

## Workload and run order

Use one H100, with BF16 inputs shaped `A[8192,4096]`, `B[4096,11008]` and a preallocated BF16 output. Retain the current contiguous buffers, disabled TF32 and reduced-precision BF16 reduction, sampled FP32 correctness check, two-second warmup and CUDA-event batch calibration. Run `torch.mm` for at least 45 measured seconds per checkpoint and synchronize the final batch. Recreate inputs from the same seed for all three checkpoints in each block. [Benchmark implementation](benchmark.py)

| Arm | Device observations during the measured GEMM |
| --- | --- |
| A | Launch one `nvidia-smi` process per query with the current CSV fields, GPU selection and five-second timeout. |
| B | Initialize NVML once before timing and use a persistent reader for the same fields. |
| C | Start A and B together on the same GEMM window. Give each reader its own worker and the same target deadlines. |

| Seed | Arm order |
| ---: | --- |
| 51 | A, B, C |
| 52 | B, C, A |
| 53 | C, A, B |
| 54 | C, B, A |
| 55 | B, A, C |
| 56 | A, C, B |

Run seeds in the listed order. Each arm appears twice in each position. Compare standalone throughput within seed. Use C for same-window sample agreement and missing-poll counts.

## Sampling and evidence

Resolve the CUDA device to the same physical GPU for both readers using its UUID internally. Run in the existing unprivileged CUDA environment with read-only NVML calls. The implementation needs the driver-provided NVML library and a pinned Python binding or a versioned equivalent. Record the driver, library/binding, PyTorch and CUDA versions and benchmark source hash. Keep GPU identifiers outside the exported data.

Use the same host monotonic clock as the workload. Schedule 45 slots, numbered 0 through 44, at one-second offsets from the timed start. A worker still busy at a deadline skips that slot. Record each slot as a valid read, skipped, error, timeout, or crossing the window end. For each attempted read save start and end offsets. Give B a five-second deadline for the entire field group. If a call stalls, its supervisor can discard the timed-out reader and start a fresh one for the next available slot. C's two readers dispatch independently.

B reads GPU and memory activity using `nvmlDeviceGetUtilizationRates`, used memory using `nvmlDeviceGetMemoryInfo`, power using `nvmlDeviceGetPowerUsage`, and SM clock using `nvmlDeviceGetClockInfo`. Convert bytes to MiB and milliwatts to watts. Emit unsupported values as missing and retain successful fields. Count a GPU-activity sample only when both query endpoints land inside the timed GEMM window and the activity value is finite and in [0, 100]. Calculate gaps between successive valid query midpoints and include both window edges. Record full per-batch iterations and CUDA durations, actual wall time and wall TFLOP/s using `2 × M × K × N` FLOPs per product.

## Decision before deployment

1. A checkpoint meets activity coverage with **at least 34 valid activity samples** and a **maximum gap of 5.0 seconds**. Report each arm's count, largest gap, timeouts and errors per seed. A completed GEMM with zero valid activity readings records a coverage failure.
2. In C, pair results by seed and scheduled slot, retaining pairs with two valid readings whose midpoints are at most 0.5 seconds apart. Publish matched and unmatched counts separately. For seeds with at least 20 matched slots, compute the median absolute difference in GPU activity in percentage points.
3. Adopt B as the default observer if all three conditions hold. B passes coverage in at least five of six standalone seeds and at least two more seeds than A. The median of the six within-seed wall-throughput ratios B/A is at least 0.98. At least four C seeds have 20 matched slots, and every such seed's median absolute difference is at most five percentage points. Otherwise retain A and publish which conditions passed. If an execution failure leaves any planned checkpoint incomplete, report partial evidence and defer the decision.

Report the comparison for this host, driver and shape. The two readers share driver counters, and running them together may influence their query latencies. Use solo A/B results for the coverage and throughput gates.

## Pause, retries and compute budget

Use a fresh state directory and a task ID per seed and arm. One step runs one checkpoint and atomically saves correctness, telemetry outcomes and timing evidence. Check the pause flag before starting another unit. On restart, skip validated completed units and retry only a missing unit. Preserve failed attempts and their start/end times. A low coverage score counts as a completed measurement. The existing GEMM runner needs arm-aware checkpoints and a bounded reader supervisor before this protocol can run. [Current checkpoint runner](../../lab.py)

Budget 18 units × 45 measured seconds, **0.225 H100-hours** of timed compute. Cap each unit at 90 seconds from launch through child cleanup and allow at most two extra attempts across the entire experiment for interrupted or failed compute runs. The total allocation ceiling is **0.50 H100-hours**, including setup and retries. Apply the same global deadline to preflight and setup. Stop at the deadline or when all 18 checkpoints are recorded, whichever comes first.

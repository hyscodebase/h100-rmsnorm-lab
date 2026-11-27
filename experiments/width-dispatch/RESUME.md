# Continuing the registered width screen after the driver preflight

At seed 53 and width 4107, the benchmark stopped before tensor allocation because the driver query did not return a version. It saved zero latency intervals, sustained phases and fixture checks. On the same configured host, a subsequent read of both `nvidia-smi` and `/sys/module/nvidia/version` returned 550.144.03. The original query's failure mode is unknown.

## Evidence carried into the continuation

The archive at `results/width-dispatch/raw/archives/driver-preflight-d42481a/` preserves the two completed units, the stopped unit, the five attempt receipts, the source pin and the partial audit as published in commit `88b3930b4f0796a875e872e1cd4a965a3453541c`. Each audit checks those bytes against Git, validates the original receipt links and recomputes the two accepted measurements with their original pin. The archived failure remains visible throughout the campaign. The first submission under the new pin is the unfinished seed 53, width 4107.

The only change to the benchmark is in driver discovery before allocation. It records the native query's result, elapsed time, row count and available output lengths and hashes. After a five-second timeout, execution error or unusable response, it reads the running kernel module's version from sysfs. Either source must return a numeric driver version. Failing both reads stops the checkpoint before allocation. Public outputs retain the error category and errno or exit status when available.

The two benchmark revisions are named separately for each measurement. The kernel, numerical checks, CUDA graph and event timing code, telemetry, workload plan and remote runner match the first cohort. The audit permits the reviewed benchmark digest and requires the same H100 model, compute capability, memory size, SM count, driver, CUDA runtime, Python, PyTorch and Triton across accepted measurements. Performance comparisons use the same four backends within each checkpoint.

## Running and resuming

Run `python3 experiments/width-dispatch/step.py --status` for progress and `python3 experiments/width-dispatch/step.py` for one unit. Use `--pause` to stop at the next boundary and `--resume` to clear the local pause marker. The continuation pin assigns a fresh campaign ID to isolate the remote cache. Both campaigns share the existing three-submission limit for each unit. The old 4107 attempt consumes one submission. A new saved failure again stops further submissions. During protocol review the pilot may save one new unit. Promotion to the remaining matrix follows its evidence audit and review.

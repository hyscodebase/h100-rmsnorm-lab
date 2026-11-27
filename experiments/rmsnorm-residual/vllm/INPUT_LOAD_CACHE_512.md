# 512-row input-load cache-policy experiment

## Question

At 512 × 4,096 BF16, does bypassing L1 for the fused residual RMSNorm activation and residual loads improve fixed-work throughput while keeping both output tensors bitwise identical to the pinned vLLM implementation?

The only candidate change is Triton's .cg cache modifier on the activation and residual input loads. The existing eight-warp kernel is the baseline. .cg requests caching at the global level, bypassing L1 and retaining L2 caching, as documented by [Triton](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators).

## Qualification before timing

Seed 209 runs the eight registered numerical fixtures and captured graphs for vLLM, the baseline, and the candidate. Both BF16 outputs must match the vLLM result bitwise, and the input and weight fingerprints must remain unchanged. The baseline binary must match the archived seed-205 binary. The candidate must still use eight warps and emit four ld.global.cg input-load instructions while leaving the two ordinary weight loads unchanged. The record includes PTX load instructions, register count, spills, binary hashes, and all checks.

Each timing period repeats the numerical fixtures and graph checks in a separate process. The two periods for a timing seed must reproduce the same input and weight fingerprints. The preregistered backend order is baseline then candidate for seed 210, candidate then baseline for seed 211, and baseline then candidate for seed 212.

## Fixed-work measurement

Each period warms up for at least three seconds, then records 512 burst wall times and CUDA event times. Eight streams replay 32 independent BF16 pairs of the 512 × 4,096 workload per burst. Each period processes 2,147,483,648 normalized rows. Direct NVML resource samples and two snapshots around the period are saved alongside code and input identities.

Adopt the candidate only if both outputs remain bitwise identical and its median normalized rows per second is at least 5% higher in every timing seed. Otherwise retain the eight-warp baseline and use the recorded results to choose one follow-up.

## Run one saved unit at a time

On the configured H100 connection, inspect the queue with:

    python3 .lab/step-vllm-input-load-cache-512.py --status

Submit exactly the next qualification or timing period with:

    python3 .lab/step-vllm-input-load-cache-512.py

Each completed unit is atomically checkpointed before the next one can start. The shared pause marker is .lab/vllm/paused; creating it prevents further submissions. Do not move on to timing unless seed 209 passes all qualification gates.

The frozen machine-readable protocol is [input-load-cache-512-plan.json](input-load-cache-512-plan.json). Raw public checkpoints and derived results will be archived in input-load-cache-512-results/ after collection.

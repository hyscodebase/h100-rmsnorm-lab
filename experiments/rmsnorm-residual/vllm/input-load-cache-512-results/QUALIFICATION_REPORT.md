# Input-load cache policy: qualification

**Status:** seed 209 passed; no throughput result is available yet. The candidate is not adopted.

## Numerical checks

The H100 run used 512 × 4,096 BF16 inputs and the release-pinned vLLM 0.10.0 operator. All eight registered fixtures matched both vLLM outputs bitwise for the eight-warp baseline and the `.cg` candidate. Captured-graph outputs also matched bitwise across all eight streams, the input and weight fingerprints stayed unchanged, and the weight was not modified.

## Generated code

Both Triton kernels use eight warps, 32 registers per thread, and zero spills. The baseline binary reproduces the archived seed-205 binary (`7b38c6052f9842335a8be7794921eef913dc208e9939ba910096dba9eaffc947`). Its PTX has six ordinary global-load instructions. The candidate has six global-load instructions: four `ld.global.cg` loads for activation and residual, and two ordinary loads for weights. The candidate binary differs from the baseline.

## H100 record

The run used an NVIDIA H100 80GB HBM3 with CUDA 12.8, PyTorch 2.7.1+cu128, Triton 3.3.1, driver 550.144.03, and Python 3.12.16. Qualification took 18.71 seconds of runner wall time. PyTorch reported peak allocated memory of 6,720.02 MiB and peak reserved memory of 7,172 MiB. These are qualification-run memory statistics, not throughput-period utilization samples.

The complete source, compiler, fixture, graph, fingerprint, memory, and timestamp evidence is in [s209.json](s209.json). [attempts.json](attempts.json) retains the sanitized submission and recovery history, including the source fixes made before qualification passed.

## Next step

Run seed 210 period 0 with the eight-warp baseline, then period 1 with the `.cg` candidate. Each period repeats the numerical gates, warms up for at least three seconds, measures 512 bursts, and saves its own checkpoint. Seeds 211 and 212 reverse the backend order as preregistered. The candidate can only be adopted after all three seeds show at least 5% higher median throughput with exact outputs. See the [registered protocol](../input-load-cache-512-plan.json) and [tracking issue #58](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/58).

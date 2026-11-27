# H100 input-load cache policy at 512 × 4,096 BF16

**Decision: retain the eight-warp baseline.** The .cg candidate kept both outputs bitwise identical but did not reach the preregistered 5% median-throughput gain in any of three independent seeds.

## Method

The candidate changes only the activation and residual input-load cache hint. Each period repeats eight numerical fixtures and captured-graph checks against the pinned vLLM 0.10.0 operator, checks unchanged inputs and weights, warms up for at least three seconds, then records 512 bursts. Each period processes 2,147,483,648 normalized rows across eight streams and 32 pairs per stream.

Median burst throughput is the median of 4,194,304 rows divided by each burst's wall time. The registered backend order is baseline then candidate in seeds 210 and 212, and candidate then baseline in seed 211.

## Results

| Seed | Order | Baseline median rows/s | Candidate median rows/s | Candidate gain | Exact inputs | Passes 5% |
| ---: | --- | ---: | ---: | ---: | :---: | :---: |
| 210 | baseline → candidate | 91,547,406 | 91,548,119 | +0.000779% | Yes | No |
| 211 | candidate → baseline | 91,544,022 | 91,544,924 | +0.000985% | Yes | No |
| 212 | baseline → candidate | 91,535,863 | 91,542,773 | +0.007549% | Yes | No |

Every fixture and timing period passed both-output bitwise checks. Input and weight fingerprints matched within each seed. The candidate gained less than 0.01% in all three seeds, so this H100 workload provides no evidence to adopt .cg.

## Compilation and numerical gates

Seed 209 passed all eight fixtures and captured-graph checks. The eight-warp baseline reproduced the seed-205 anchor binary. Both kernels used eight warps and 32 registers per thread with zero spills. PTX contained six ordinary global-load instructions in the baseline, and four ld.global.cg input loads plus two ordinary weight loads in the candidate. The candidate binary differed from baseline.

## H100 resource use

Device and software: NVIDIA H100 80GB HBM3, compute capability 9.0, driver 550.144.03, CUDA 12.8, PyTorch 2.7.1+cu128, Triton 3.3.1, and Python 3.12.16.

The six measured periods processed 12,884,901,888 normalized rows in 140.792 seconds, plus 18.146 seconds of warm-up. The observer recorded 133 NVML samples. Median GPU busy and memory busy were both 100%; median VRAM use was 8,431.25 MiB. Per-period median power ranged from 514.843 to 518.510 W. The largest sample gap was 2.499 seconds.

Seed 209 qualification took 18.707 seconds of runner wall time. PyTorch peak allocated and reserved memory were 6,720.02 MiB and 7,172.00 MiB. These are qualification-run memory statistics; timed-period samples and auxiliary snapshots are stored in each raw checkpoint.

## Audit trail

Frozen source revision: 4721ed8ae8602debcdaacb440d6fee7e6889117d. Full compiler, fixture, fingerprint, burst, timing, resource, and snapshot records are in s209.json and s210-p0.json through s212-p1.json. Sanitized submission receipts are in attempts.json; derived rates and per-period resource summaries are in summary.json.

Run python3 experiments/rmsnorm-residual/vllm/summarize_input_load_cache.py to verify source hashes, checkpoint identity, PTX gates, exactness, fingerprints, fixed-work counts, and derived statistics.

The .cg load hint is described in the [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [NVIDIA PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators).

## Next hypothesis

Keep the eight-warp baseline. Revalidate the existing 4,104-wide two-row RMSNorm recommendation at widths 4,103, 4,104, 4,105, and 4,112 with fresh seeds to see how local the measured gain is. See [issue #60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60) and [GOAL.md](../../../../GOAL.md).

Tracking: [issue #58](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/58) and [draft PR #59](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/59).

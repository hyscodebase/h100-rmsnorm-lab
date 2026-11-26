# Fused BF16 residual update and RMSNorm on H100

**No compiled variant qualified at the target in any seed. The registered comparison cannot establish a Triton speedup.** The target was 1,024 rows at width 4,096. The rule required at least 5% lower candidate latency against the fastest correct compiled mode in each of three seeds. 0/3 target comparisons were eligible and 0/3 met the latency threshold. [Registered plan](PLAN.md)

## Numerical results

The four-warp Triton kernel passed all six fixtures and verification of both outputs from all 32 captured calls in 12/12 checkpoints. The explicit eager reference qualified in 12/12. Across two PyTorch spellings and two compilation modes, 0/48 compiled attempts qualified for timing.

The residual sum is rounded to BF16 before normalization. The oracle checks the updated residual bit for bit and the normalized output against an FP64 calculation from the rounded residual. Two fixtures also check BF16 bits at selected coordinates. The denominator fixture detects normalization with an unrounded denominator even when the numerator has been rounded. [CPU rounding example](evidence/cpu-rounding.json) · [Numerical tests](test_check.py)

The table groups exclusions by backend and recorded failure. A checkpoint with several failed fixtures contributes once to the combined reason.

| Excluded backend | Stage and recorded reason | Checkpoints |
|---|---|---:|
| `compiled_explicit_default` | numerical_fixtures · bf16_denominator_tie / rounding_sensitive_bits | 12 |
| `compiled_explicit_max-autotune-no-cudagraphs` | numerical_fixtures · bf16_denominator_tie / rounding_sensitive_bits | 12 |
| `compiled_native_default` | numerical_fixtures · bf16_denominator_tie / rounding_sensitive_bits | 12 |
| `compiled_native_max-autotune-no-cudagraphs` | numerical_fixtures · bf16_denominator_tie / rounding_sensitive_bits | 12 |

## Median latency per two-output call

Each latency is the median of 21 CUDA-event intervals divided by 32 calls, in microseconds. The timing order was shuffled each round. The graphs reused input addresses, kept both outputs live and ran without an explicit L2 flush. Eager latency is a separate measurement. Each seed links to raw event intervals and numerical checks.

| Rows | Seed | Eager µs | Triton µs | Fastest correct compiled mode | Compiled µs | Triton latency reduction vs compiled |
|---:|---:|---:|---:|---|---:|---:|
| 1,024 | [101](results/s101-r1024-w4096.json) | 115.8120 | 10.0680 | None qualified | — | — |
| 1,024 | [102](results/s102-r1024-w4096.json) | 115.6660 | 10.0750 | None qualified | — | — |
| 1,024 | [103](results/s103-r1024-w4096.json) | 115.8440 | 10.1130 | None qualified | — | — |
| 1 | [101](results/s101-r1-w4096.json) | 24.6950 | 2.3570 | None qualified | — | — |
| 1 | [102](results/s102-r1-w4096.json) | 24.6490 | 2.3180 | None qualified | — | — |
| 1 | [103](results/s103-r1-w4096.json) | 24.6390 | 2.3050 | None qualified | — | — |
| 16 | [101](results/s101-r16-w4096.json) | 33.1860 | 2.4530 | None qualified | — | — |
| 16 | [102](results/s102-r16-w4096.json) | 33.2490 | 2.3990 | None qualified | — | — |
| 16 | [103](results/s103-r16-w4096.json) | 33.2540 | 2.3950 | None qualified | — | — |
| 128 | [101](results/s101-r128-w4096.json) | 41.1940 | 2.8600 | None qualified | — | — |
| 128 | [102](results/s102-r128-w4096.json) | 41.1220 | 2.9390 | None qualified | — | — |
| 128 | [103](results/s103-r128-w4096.json) | 41.1160 | 2.9320 | None qualified | — | — |

The 1-, 16- and 128-row controls describe the same operator at smaller shapes. A dash marks an unavailable comparison or measurement. Positive reductions indicate lower Triton latency. Model-level token throughput was outside this benchmark.

## Reproduction and follow-up

The campaign used NVIDIA H100 80GB HBM3, driver 550.144.03, CUDA 12.8, PyTorch 2.7.1+cu128, Triton 3.3.1 and Python 3.12.16. Seeds 101, 102, 103 ran in fresh processes. The experiment hash is `c7192563adbd9a3397f100060ae58ae21e14b08aee91f5074f0cc66ac5a2a1f0`. [Frozen source](https://github.com/hyscodebase/h100-rmsnorm-lab/tree/455d378c6df97d40cbeb2d13e45b1966b835e8f6/experiments/rmsnorm-residual) · [Source SHA-256 hashes](results/manifest.json) · [Controller attempts](results/attempts)

```bash
python3 experiments/rmsnorm-residual/summarize.py experiments/rmsnorm-residual/results
```

In a separate H100 checkpoint at fresh seed 104, two isolated Python processes confirmed the precision-cast configuration was effectively OFF and ON. The default compiled expression passed the random control and first tie probe under both settings. At the denominator-sensitive probe it returned normalized values 0.99609375 and 1.9921875 in both processes. Eager and the Triton candidate returned the required 1.0 and 2.0. The observed mismatch persisted with the setting enabled. This experiment does not locate the compiler pass responsible. [Raw values, BF16 bits and source hashes](precision/checkpoint.json). [Precision diagnostic plan](precision/PLAN.md) · [Probe implementation](precision/probe.py)

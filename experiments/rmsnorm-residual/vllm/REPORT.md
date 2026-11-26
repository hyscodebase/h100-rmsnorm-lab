# vLLM in-place residual RMSNorm on H100

The official vLLM 0.10.0 CUDA extension loaded on the PyTorch 2.7.1 and CUDA 12.8 H100 environment, and both implementations passed the exact BF16 check for rounding before the weight multiplication. Both failed the registered tolerance on the cancellation fixture at seed 109, so the preflight stopped before latency measurement. [Plan](PLAN.md) · [Raw preflight](results/s109.json) · [Wheel and extension hashes](wheel-record.json)

## Cancellation diagnostic

After an independent replay of the same seeded fixture, the two kernels returned identical BF16 encodings at all **4,194,304 normalized elements**. Both returned the FP64 oracle's exact updated residual bits. Ten normalized elements exceeded the strict FP64 tolerance in both implementations. The ten elements came from row 699. Their FP64 pre-weight activations lay about **0.0000147 of one BF16 step** from the nearest pre-weight rounding midpoint. [Coordinate-level evidence](results/cancellation-diagnostic.json)

At column 41, for example, the pre-weight FP64 value was 1.62109363495937 and the midpoint was 1.62109375. Both GPU implementations returned −0.376953125 after weighting. The strict oracle's weighted reference was −0.37429046630859375 with a tolerance of 0.0018724523315429688. An independent PyTorch FP32 expression produced −0.375 at this coordinate and differed from the two kernels at twenty positions in the full fixture. The diagnostic localizes the disagreements to rounding boundaries. It does not identify the exact reduction or approximation instruction responsible at each coordinate. [Diagnostic implementation](diagnose.py)

## Fresh seeds and the FP32 interval

A CPU replay of the registered interval arithmetic covered all ten measured seed-109 witness coordinates. On fresh seed 113 both GPU implementations passed. On seeds 114–116 the boundary rule permitted only a valid adjacent encoded value and required both kernels to produce the same bits whenever the strict tolerance was exceeded. Each seed failed that rule, so all three raw timing fields are empty. [Preregistered rule](PRECISION_PLAN.md) · [Tracking issue #11](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/11)

| Seed | Elements beyond strict tolerance in either kernel | Accepted with matching bits | Rejected |
| ---: | ---: | ---: | ---: |
| [113](results/s113.json) | 0 | 0 | 0 |
| [114](results/s114.json) | 124 | 77 | 47 |
| [115](results/s115.json) | 42 | 6 | 36 |
| [116](results/s116.json) | 23 | 0 | 23 |

The seed-114 diagnostic independently evaluated both weighted BF16 endpoints at all 47 rejected coordinates. At every one, the two kernels individually matched an endpoint and selected different results. At row 286, column 35, vLLM returned −1.4921875 and Triton returned −1.5. Further investigation will test how the reduction grouping affects those choices. [Endpoint evidence](results/bounded-disagreements.json) · [Grouped reduction experiment](REDUCTION_PLAN.md)

The first run of seed 114 stopped at a missing Python import in the metadata lookup. Commit 084e88c fixed the import before any replay was timed. The failed and successful attempts remain in the journal. [Attempt records](results/attempts)

## Reproduce the numerical checkpoint

The runner uses the original extension extracted from the verified vLLM wheel. It needs the CUDA build of PyTorch 2.7.1 and Triton 3.3.1 on an H100.

```bash
python3 experiments/rmsnorm-residual/vllm/wheel.py .lab/vllm-release > wheel-record.json
BENCH_REVISION=$(git rev-parse bd5dac8) python3 experiments/rmsnorm-residual/vllm/run.py \
  --asset-dir .lab/vllm-release --wheel-record wheel-record.json --seed 109
```

Use a checkout at the commit named by the relevant checkpoint. Diagnostic inputs and the original published wheel remain the same. [Frozen preflight](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/bd5dac80b6a04cfa61eb461fbea55baca0acb99a) · [Frozen diagnostic](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/13f0aab586e47fd286e438f4e6e33d9f285caf69)

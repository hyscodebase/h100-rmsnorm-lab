# vLLM in-place residual RMSNorm on H100

The official vLLM 0.10.0 CUDA extension loaded on the PyTorch 2.7.1 and CUDA 12.8 H100 environment, and both implementations passed the exact BF16 check for rounding before the weight multiplication. Both failed the registered tolerance on the cancellation fixture at seed 109, so the preflight stopped before latency measurement. [Plan](PLAN.md) · [Raw preflight](results/s109.json) · [Wheel and extension hashes](wheel-record.json)

## Cancellation diagnostic

After an independent replay of the same seeded fixture, the two kernels returned identical BF16 encodings at all **4,194,304 normalized elements**. Both returned the FP64 oracle's exact updated residual bits. Ten normalized elements exceeded the strict FP64 tolerance in both implementations. The ten elements came from row 699. Their FP64 pre-weight activations lay about **0.0000147 of one BF16 step** from the nearest pre-weight rounding midpoint. [Coordinate-level evidence](results/cancellation-diagnostic.json)

At column 41, for example, the pre-weight FP64 value was 1.62109363495937 and the midpoint was 1.62109375. Both GPU implementations returned −0.376953125 after weighting. The strict oracle's weighted reference was −0.37429046630859375 with a tolerance of 0.0018724523315429688. An independent PyTorch FP32 expression produced −0.375 at this coordinate and differed from the two kernels at twenty positions in the full fixture. The diagnostic localizes the disagreements to rounding boundaries. It does not identify the exact reduction or approximation instruction responsible at each coordinate. [Diagnostic implementation](diagnose.py)

The follow-up registers a bounded FP32 acceptance test on fresh seeds. A CPU replay of the interval arithmetic covered all ten recorded coordinates. The next numerical gate also requires bitwise agreement of both kernels at any such boundary. This seed remains a failed preflight under its registered rule. [Fresh-seed protocol](PRECISION_PLAN.md) · [Tracking issue #11](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/11)

## Reproduce the numerical checkpoint

The runner uses the original extension extracted from the verified vLLM wheel. It needs the CUDA build of PyTorch 2.7.1 and Triton 3.3.1 on an H100.

```bash
python3 experiments/rmsnorm-residual/vllm/wheel.py .lab/vllm-release > wheel-record.json
BENCH_REVISION=$(git rev-parse bd5dac8) python3 experiments/rmsnorm-residual/vllm/run.py \
  --asset-dir .lab/vllm-release --wheel-record wheel-record.json --seed 109
```

Use a checkout at the commit named by the relevant checkpoint. Diagnostic inputs and the original published wheel remain the same. [Frozen preflight](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/bd5dac80b6a04cfa61eb461fbea55baca0acb99a) · [Frozen diagnostic](https://github.com/hyscodebase/h100-rmsnorm-lab/commit/13f0aab586e47fd286e438f4e6e33d9f285caf69)

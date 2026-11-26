# Verify the FP32 reduction contract at BF16 rounding boundaries

At seed 109, vLLM and Triton agreed bit for bit across 4,194,304 cancellation outputs. Both exceeded the original FP64 tolerance at ten elements in one row. Each FP64 pre-weight activation lay within 0.000015 of one BF16 step from a midpoint. The original preflight remains failed under its registered rule. [Measured result](REPORT.md) · [Ten measured coordinates](results/cancellation-diagnostic.json)

## New numerical gate

Use the same pinned vLLM and Triton implementations at 1,024 × 4,096. Keep the two required rounding steps, the exact residual check, signed zeros, and the exact weight-rounding probe. Apply the original elementwise tolerance throughout the tensor. When a normalized result falls outside that tolerance, require all of the following conditions.

1. The FP32 arithmetic interval crosses exactly one BF16 rounding boundary before multiplying the weight.
2. The observed normalized BF16 bits exactly equal the rounded weighted result from one of the two adjacent BF16 encodings at that boundary.
3. The two GPU implementations agree bitwise at the coordinate. Record counts, coordinates and maximum error for all such cases.

Use finite normal intermediates or zero. Bound the absolute value of each nonzero stored residual and weight to [2⁻⁴⁰, 32]. Products of two BF16 operands then fit inside normal FP32 and are exactly representable. Summing 4,096 nonnegative terms has relative error bounded by γ₄₀₉₆ = 4096u/(1 − 4096u) for u = 2⁻²⁴, even with sequential ordering. Dividing by 4,096 is exact. Allow one round for epsilon conversion and one for its addition. Bound the FP32 reciprocal square root and following multiplication with an additional relative margin of 2⁻²⁰, which exceeds the documented FP32 rsqrt approximation bound and a rounding step. Evaluate both interval endpoints in FP64. [CUDA 12.8 PTX reciprocal square root](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-rsqrt) · [Interval calculation](float32_gate.py)

## Repetition and decision

Run seed 113 for numerical preflight, using the same six distributions and two exact fixtures as before. Use new independent seeds 114–116 for timing after both implementations qualify. Recheck the new gate on the actual timed input, including both outputs of each restored graph invocation. Maintain the previous 32 independent buffer pairs, 21 shuffled rounds, untimed resets and 5% gate in every seed. [Previous timing protocol](PLAN.md)

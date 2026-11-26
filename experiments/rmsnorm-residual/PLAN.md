# Fused residual addition and RMSNorm at width 4,096

**Preregistration for the experiment after the observer comparison.** At 1,024 rows and width 4,096, the earlier standalone RMSNorm tile missed its 5% adoption threshold. The next question is whether a single Triton kernel can write an updated residual and normalize it faster than a compiler given the same two-output operation. Freeze the source, launch configuration and plan before timing. [Prior width sweep](../../results/width-sweep/REPORT.md) · [Existing Triton kernel](../../kernels.py)

## Operator contract

Use contiguous BF16 activations `x` and `residual` of shape `[rows, 4096]` and a nonconstant BF16 weight of length 4,096. The caller receives two distinct BF16 tensors. Inputs remain available for reuse. The following operations define rounding and reduction order at the tensor level.

```python
updated = (x.float() + residual.float()).to(torch.bfloat16)
z = updated.float()
inverse_rms = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + 1e-6)
normalized = (z * inverse_rms * weight.float()).to(torch.bfloat16)
return updated, normalized
```

Implement **one candidate**, one Triton program per row with four warps. Round the residual sum to BF16 before both the store and the FP32 RMS reduction. Retain both returned tensors for every captured invocation. In a verification replay, reduce each output separately and check both against the numerical reference outside the timed interval. Every backend follows the same output allocation and consumption contract.

## Comparisons and checks

- Compile the complete two-output function with `torch.compile(fullgraph=True, dynamic=False)`. Run the explicit reference above and a native PyTorch spelling using BF16 residual addition and `F.rms_norm(updated, (4096,), weight, eps=1e-6)`. Evaluate the default and `max-autotune-no-cudagraphs` modes for each supported spelling. For every shape and seed use the lowest median among correct, warmed compiled variants as the reference. Publish every variant's result or exclusion. At least one correct compiled variant is required for a decision. Include the uncompiled explicit reference for context.
- Verify the updated BF16 residual bit for bit. Compare normalized values in FP32 against an independent FP64 oracle computed from the **rounded** residual, with `rtol=0.005` and `atol=1e-6`. Cover random inputs, zero rows, cancellation, tiny values and BF16 ties. Freeze a rounding-sensitive fixture that distinguishes normalization of the rounded and unrounded sum. At discriminating entries away from BF16 output midpoints, also require the oracle's BF16 bits. Any candidate correctness failure ends the performance decision.
- Use one H100. The target is **1,024 × 4,096**. Controls use 1, 16 and 128 rows. Run each shape with fresh seeds 101, 102 and 103, one process and one saved checkpoint per shape and seed. Precompile and warm every backend, then record 21 CUDA-event samples, each a graph replay of 32 invocations on reused input addresses. Shuffle backend order each round. Save raw samples, medians, quartiles, GPU/software versions and source hashes. Exclude compilation, autotuning and verification from timings. [Existing graph timing implementation](../../benchmark.py)

## Adoption decision

Recommend the candidate for the target shape only if all numerical checks pass and its median latency is **at least 5% lower** than the fastest correct compiled reference in **each of the three seeds**, using unrounded timings. Publish control-shape wins and losses separately. Otherwise keep the compiled path for the target. Report the outcome as microseconds per two-output operator call on this H100. Model latency and tokens per second remain separate measurements.

An upstream comparison is optional follow-up work. First verify the external build, GPU/software versions, residual rounding, mutation contract and output materialization. Publish an upstream speed claim only after a matched run against that implementation.

# A correct compiled baseline for residual-add RMSNorm

The seed-104 H100 diagnostic observed compiled values 0.99609375 and 1.9921875 on the denominator tie with cast emulation both off and on. Eager PyTorch and the fused Triton candidate produced the required 1.0 and 2.0. This follow-up tests two ways to express the rounded operand in PyTorch. [Diagnostic results](../REPORT.md) · [Original contract](../PLAN.md)

## Implementations to freeze

Use the existing H100 environment, contiguous BF16 inputs, width 4,096, 1,024 token rows and epsilon 1e-6. Return two distinct BF16 tensors, the updated residual and its weighted RMSNorm. Preserve the input tensors. Normalize in FP32 using the updated residual for both the numerator and the mean square.

- **Materialized PyTorch.** Compile addition and normalization as two independent shape-specialized callables. In an ordinary Python caller, obtain `r = compiled_add(x, residual)`, pass that BF16 tensor to `compiled_norm(r, weight)` and return `(r, y)`. Keep the outer caller outside compilation. Capture and time both stages together.
- **Single-graph PyTorch.** Compile a tensor implementation of round-to-nearest, ties-to-even. Reinterpret the FP32 sum as 32-bit integer words. Extract unsigned halves `hi` and `lo` with masks. Increment `hi` when `lo > 0x8000`, or when `lo == 0x8000` and `hi` is odd. Use those 16 bits for the returned BF16 residual. Reconstruct the FP32 normalization operand from the same bits by placing them in the high half of a word and zeroing the low half. Keep the arithmetic on-device and inside one compiled graph. Preserve the sign bit of zero. Use finite inputs, finite FP32 sums and reductions and finite BF16 results. NaN and infinity propagation are outside this test.
- Retain the original explicit and native single-graph spellings, the eager explicit reference and the frozen four-warp fused Triton kernel. Test each PyTorch compiled spelling in default and `max-autotune-no-cudagraphs` modes with `fullgraph=True, dynamic=False`. Both materialized stages use the same mode. Fix cast emulation off and record its effective value.

## Numerical gate before timing

Freeze source hashes, versions and the plan. Run preflight with fresh seed **105**. Apply all six original fixtures, including both exact tie probes, against the independent FP64 oracle. Check the updated residual bit for bit, normalized values at `rtol=0.005, atol=1e-6`, both output sums, disjoint output storage and unchanged inputs. Also test the bit implementation on positive and negative values immediately below, at and above BF16 rounding midpoints of both parities. Check the two signed zeros by their BF16 encodings, including an operator row with two negative-zero addends. [Fixtures and output checks](../check.py)

Proceed only if eager, Triton, at least one materialized mode and at least one compiled single-graph mode pass. Record unsupported bit operations, compilation errors and numerical exclusions. If this gate fails, publish the diagnostic and stop before timing.

## Timing and decision

Run fresh seeds **106, 107 and 108** in separate checkpoints. Recheck every preflight survivor on each seed. Retain all 32 output pairs per graph, poison and verify every pair in an untimed replay, then record 21 shuffled rounds of 32 calls with the existing CUDA-event protocol. Exclude compilation and verification. Publish all raw timings and exclusions. [Timing protocol](../benchmark.py)

Recommend the frozen candidate at this shape only if the numerical gate holds in each seed and its median latency is at least 5% below the **fastest correct compiled comparator in that seed**, using unrounded timings. A gain against the split path alone cannot qualify when a faster correct single graph is present. Report this two-output operator's latency on the tested H100 and software stack. Serving throughput requires a separate measurement.

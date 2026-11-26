# Preserve the FP32 tree with warp shuffles

The grouped eight-warp implementation exactly reproduced the released vLLM output bits on the tested fixtures and timed inputs. Its emitted PTX contained 128 shared load instructions for the reduction. Its latency was 7.37–7.82% above vLLM. Test a single change to the reduction implementation. Use ordered intra-warp shuffles at distances 1, 2, 4, 8 and 16 and a serial sum of the eight warp totals. Keep the two groups of eight per lane, vectorized BF16 IO, normalized BF16 conversion and multiply by weight unchanged. [Evidence](GROUPED_REPORT.md) · [Grouped Triton source](grouped_kernel.py)

Verify the generated PTX and actual logical-to-physical lane mapping. The shuffle kernel qualifies for numerical work only if its emitted instruction count for shared loads is below the grouped version and the exact five shuffle stages are visible. Record the toolchain and binary hash.

Replay seed 114, with vLLM and the grouped candidate both as references. Require bitwise equality at all 47 formerly rejected coordinates. Check fresh seed 121, including the original eight fixtures, exact two-output rules and FP32 interval. Then measure seeds 122–124 with all three kernels. Re-check each graph result before timing. Fresh 32-call CUDA graphs with disjoint buffered inputs use the same untimed reset and 21 shuffled rounds.

Recommend the candidate only if it passes the numerical gate and reduces latency against the released vLLM comparison by at least 5% in every timing seed. Report the delta against the original grouped Triton candidate as well. The compiler and benchmark sources are fixed before the measurements.

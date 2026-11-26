# Probe the fusion threshold at 6,144 downstream columns

The square BF16 chain reached 1.256–1.359% higher throughput with fusion. At 12,288 downstream columns, the same fused residual/RMSNorm kernel led the tuned split by 0.362–0.536%. Test an intermediate width with the same 512 × 4,096 residual stage. Two projections at 6,144 downstream columns do 1.25 times the square chain's GEMM work per row, putting the simple estimate from the square chain close to 1%. [Square-chain result](PROJECTION_CHAIN_512_REPORT.md) · [Wide-chain result](PROJECTION_TAIL_12288_REPORT.md) · [Tracking issue](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/46)

Use a first projection with a 4,096 × 4,096 BF16 weight and a second projection with a 4,096 × 6,144 BF16 weight. Retain the fixed eight-warp fused kernel, 8,192-element split addition and matched normalization. Use identical preallocated buffers, PyTorch GEMM settings and projection weights within a seed, with 32 disjoint pairs on each of eight streams.

Use seed 185 to check both projections and both residual/RMSNorm outputs bitwise against the released vLLM chain. Verify disjoint buffers, all 19 input and weight digests and the captured residual kernel binaries. Then measure independent seeds 186–188 one at a time. Use two balanced periods per backend with at least one second of warm-up and six measured seconds per period, with 32 graph replays per stream in each burst.

Classify the 1% threshold outcome across all three exact seeds. Report absolute throughput and effective GEMM rate beside the two earlier widths. Preserve raw periods, graph counts, all four exactness checks, device samples, clock probes and one cache-only recovery.

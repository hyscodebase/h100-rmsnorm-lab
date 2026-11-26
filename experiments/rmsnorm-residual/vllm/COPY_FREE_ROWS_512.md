# Test matched eight-warp fusion at 512 rows

The 256-row experiment matched the released vLLM output bits by using the eight-warp FP32 reduction on both copy-free paths. Test whether the fusion gain persists at 512 rows with the same compiled kernels. The larger grid doubles the disjoint input and output footprint to about 4 GiB while staying in the same released vLLM thread-block regime. [256-row results](COPY_FREE_ROWS_256_REPORT.md) · [Released vLLM dispatch](https://github.com/vllm-project/vllm/blob/v0.10.0/csrc/layernorm_kernels.cu#L161-L199) · [Earlier 512-row in-place comparison](TRANSFER_REPORT.md)

Keep BF16 width 4,096, eight CUDA streams, 32 disjoint pairs per stream, immutable activation, residual and weight, and identical writable output addresses. Verify the canonical PTX and cubin hashes of all three Triton kernels against seed 157. The split path keeps the vectorized BF16 addition and ordered eight-warp normalization. Use released vLLM with both restoration copies as the accuracy and context reference.

Run seed 161 as the standalone numerical checkpoint. Verify both outputs bitwise with poisoned writable buffers, signed zero, BF16 weight rounding, the output alias guards and preserved input hashes. After qualification, run independent seeds 162–164. Give each backend three balanced periods with at least one second of warm-up and six seconds timed. Each coordinated burst replays 32 graphs per stream.

Extend the recommendation if fusion delivers at least 5% more median wall throughput than the matched two-kernel path in every timing seed, with bitwise matching outputs and the same source hashes after timing. Keep the original CUDA events, completed rows, both device observation streams, clock and binary provenance and cache-only recovery.

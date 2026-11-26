# Tune the split residual addition at 512 rows

The 512-row comparison uses 2,048 BF16 elements per four-warp addition block. Test whether a different addition tile strengthens the split baseline. Keep the adjacent add-then-normalize schedule and the fixed eight-warp normalization and fused binaries. [512-row fusion results](COPY_FREE_ROWS_512_REPORT.md) · [Scheduling experiment](SPLIT_SCHEDULE_512_REPORT.md)

At 512 × 4,096, test 1,024, 2,048, 4,096 and 8,192 elements per addition block with exactly four warps. Keep every BF16 rounding operation, eight CUDA streams, 32 disjoint pairs per stream, and the same input and output addresses. Verify emitted vector loads and stores, the unchanged 2,048-element baseline and normalization binaries, and both outputs bitwise against released vLLM. Check alias guards and the original input and weight hashes.

Use tuning seed 173 to screen each exact tile in four balanced periods per tile. Give each period at least one second of warm-up and four seconds of timing after qualification. Choose the highest median wall throughput among exact tiles and resolve exact ties in favor of the 2,048-element tile, then the smaller tile. Freeze that choice before opening the held-out seeds.

Use independent seeds 174–176 to compare the original 2,048-element split path, the selected split path and the original fused path. Use three balanced periods per backend when a new tile is selected and two when the original tile wins. Each period has at least one second of warm-up and six seconds of timing, with 32 replays per stream in each burst. A new tile earns the recommendation if it is at least 5% faster than the original split path in all independent seeds with exact outputs. Report the original fused kernel's gain over the stronger verified split path.

Save the original numerical results, selection order, vectorized instructions and binaries, completed work, CUDA intervals, device samples and the recovery checkpoint.

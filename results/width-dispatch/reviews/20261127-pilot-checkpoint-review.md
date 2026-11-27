# Active-source pilot checkpoint review — 2026-11-27

## PR review

- PR #67 head reviewed: `a4bb98aaa8b5d1a9e87fb85f8b011ec8392daff3`.
- A GitHub `COMMENT` review (ID `6089320605`) recorded no blocking issue in the runner recovery, source/cache identity, attempt accounting, memory gates, telemetry audit, or receipt validation changes. This is a review comment, not an approval.
- The independent protocol review and local static checks are recorded in `20261127-runner-review-followup.md`.

## Connection recovery

The first two submitted attempts ended before the server accepted the SSH session. `wireproxy --configtest` failed inside the default network-restricted command environment while resolving the configured WireGuard peer. Running the read-only check with the approved network-enabled command environment returned `Config OK`; a read-only SSH query then confirmed an NVIDIA H100 80GB HBM3. Neither failed attempt started a remote benchmark or used H100 time.

The third attempt completed checkpoint `s53-w4102` under source revision `d42481aef66b4be3f7bd2cdd71b93778274a2875`. The checkpoint retains three attempt receipts, including both transport failures and the successful result.

## Pilot result

Environment: H100 80GB HBM3, compute capability 9.0, driver 550.144.03, CUDA 12.8, PyTorch 2.7.1+cu128, Triton 3.3.1, Python 3.12.16.

All correctness fixtures passed; the checkpoint contains 1,604 randomized CUDA-event latency intervals and four 30-window sustained phases. The four sustained phases recorded 132.182 seconds of CUDA-event work in total (2.20 minutes). The 16 GiB minimum free-memory floor passed at allocation and timing checkpoints.

| Backend | Median latency (us/op) | Sustained logical rate (GB/s) |
| --- | ---: | ---: |
| Triton 4 warps / 1 row | 181.392 | 2,986.458 |
| Triton 8 warps / 1 row | 183.144 | 2,960.028 |
| Triton 4 warps / 4 rows | 184.304 | 2,938.154 |
| torch.compile | 196.968 | 2,751.639 |

At this one width/seed, the candidate is 1.62% below the fastest one-row baseline in sustained logical throughput and 6.78% above `torch.compile`. It does not qualify because it must beat both baselines by at least 5%. This point does not support a parity dispatch conclusion.

## Telemetry and continuation

The checkpoint saved 454 telemetry rows. GPU busy was 100% in the recorded median and peak samples; peak VRAM use was 5,064 MiB, median/peak power was 623.2/691.1 W, and peak temperature was 50 C. Query-timestamp gaps reached 2.775 seconds, over the preregistered 1.5-second limit, so all four phases are marked telemetry-ineligible. These readings are retained for diagnosis and are not treated as eligible activity evidence; the compute record remains usable.

The pilot found no code blocker. The source pin is promoted from `pilot` to `full_matrix`; the remaining 23 checkpoints continue under the same frozen source, and the bounded dispatch conclusion stays pending until the preregistered matrix and audit complete.

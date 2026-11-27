# Active matrix progress — 2026-11-27

This page preserves the state at the first driver failure. The resumed measurement passed its audit at commit `e86e267` with +11.93% over the fastest one-row kernel. The [current run table](../PROGRESS.md) is updated after each new unit.

## Current state

- Active source revision: `d42481aef66b4be3f7bd2cdd71b93778274a2875`.
- Source scope: `full_matrix`; matrix currently blocked at `s53-w4107`.
- Accepted checkpoints: 2/24; cumulative CUDA-event time: 264.444 seconds (4.41 minutes).
- Attempt receipts: five total; two successful checkpoints, two transport failures before remote benchmark start, and one saved preflight gate failure.

## Seed 53 results

| Width | 4w/1r GB/s | 8w/1r GB/s | 4w/4r GB/s | torch.compile GB/s | 4w/4r vs best 1-row | Qualifies |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4102 | 2,986.458 | 2,960.028 | 2,938.154 | 2,751.639 | -1.617% | No |
| 4106 | 2,985.877 | 2,944.201 | 2,939.480 | 2,756.218 | -1.554% | No |

At both registered even widths the 4w/4r candidate trails the strongest one-row baseline. It beats `torch.compile` by 6.78% and 6.65%, respectively, but the preregistered gate requires at least 5% over both references for all three seeds. These two seed-53 observations are not a parity conclusion.

## Resource observations

The two checkpoints saved 878 telemetry rows and 264.444 seconds of sustained CUDA-event work. GPU-busy median and peak readings are 100%; peak VRAM use is 5,064 MiB; aggregate median/peak power is 634.1/699.8 W; peak temperature is 53 C. Query timestamp gaps reached 4.548 seconds, above the 1.5-second eligibility limit, so 0/8 phases qualify as formal activity telemetry. Raw readings remain available for diagnosis; the separately audited compute results remain usable.

The initial two transport failures were caused by the default command sandbox lacking outbound network access. A network-enabled read-only WireGuard check succeeded, after which SSH confirmed the H100 and the third attempt completed the first checkpoint. Those failed attempts did not start a remote benchmark or use GPU time.

## Saved blocker: s53-w4107

The third checkpoint saved `status=gate_failed` at `preflight`: `Cannot establish required NVIDIA driver provenance before measurement.` It produced zero latency intervals, zero sustained phases, zero allocated/reserved CUDA bytes, and no correctness fixtures. The remote benchmark wall time in the checkpoint is 11.975 seconds; the local attempt lasted 22.284 seconds. It used the pinned source revision and is preserved with its attempt receipt and checkpoint hash in the partial evidence audit.

A later read-only SSH probe loaded the same configured remote environment: `nvidia-smi` was present and the driver-version query returned one valid row. This confirms that the query works at probe time, but it does not identify why the original preflight query failed. The original failure record does not retain whether the command was missing, timed out, returned nonzero, or produced empty output. The registered runner blocks later checkpoints on saved gate failures, so the matrix remains stopped at `s53-w4107`; no retry or bypass has been made.

The two accepted checkpoints remain useful compute/correctness evidence, but their 8 sustained telemetry phases are all formally ineligible due query timestamp gaps up to 4.548 seconds. No final dispatch decision is available until the saved failure is resolved and the full preregistered matrix passes its audit.

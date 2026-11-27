# Active matrix progress — 2026-11-27

## Current state

- Active source revision: `d42481aef66b4be3f7bd2cdd71b93778274a2875`.
- Source scope: `full_matrix`; next checkpoint: `s53-w4107`.
- Accepted checkpoints: 2/24; cumulative CUDA-event time: 264.444 seconds (4.41 minutes).
- Attempt receipts: four total; two successful checkpoints and two transport failures before remote benchmark start.

## Seed 53 results

| Width | 4w/1r GB/s | 8w/1r GB/s | 4w/4r GB/s | torch.compile GB/s | 4w/4r vs best 1-row | Qualifies |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4102 | 2,986.458 | 2,960.028 | 2,938.154 | 2,751.639 | -1.617% | No |
| 4106 | 2,985.877 | 2,944.201 | 2,939.480 | 2,756.218 | -1.554% | No |

At both registered even widths the 4w/4r candidate trails the strongest one-row baseline. It beats `torch.compile` by 6.78% and 6.65%, respectively, but the preregistered gate requires at least 5% over both references for all three seeds. These two seed-53 observations are not a parity conclusion.

## Resource observations

The two checkpoints saved 878 telemetry rows and 264.444 seconds of sustained CUDA-event work. GPU-busy median and peak readings are 100%; peak VRAM use is 5,064 MiB; aggregate median/peak power is 634.1/699.8 W; peak temperature is 53 C. Query timestamp gaps reached 4.548 seconds, above the 1.5-second eligibility limit, so 0/8 phases qualify as formal activity telemetry. Raw readings remain available for diagnosis; the separately audited compute results remain usable.

The initial two transport failures were caused by the default command sandbox lacking outbound network access. A network-enabled read-only WireGuard check succeeded, after which SSH confirmed the H100 and the third attempt completed the first checkpoint. Those failed attempts did not start a remote benchmark or use GPU time.

The next checkpoint is `s53-w4107`. No final dispatch decision is available until the full preregistered matrix passes its audit.

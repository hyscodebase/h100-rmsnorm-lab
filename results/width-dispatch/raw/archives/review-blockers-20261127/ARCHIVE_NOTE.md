# Review-blocked pilot evidence archive

The two checkpoints below were collected on 2026-11-27 under source revision `85bf889ffca8863f59e80f84e82ef9fcdc297cee`. Both passed that revision's partial evidence audit. A later independent review found protocol issues affecting timeout classification, telemetry window timing, execution-environment pooling, replay-count auditing, and recovery readiness. The results and receipts are preserved here but are excluded from the revised pinned matrix; the revised protocol will start a fresh pilot.

## Preserved usage summary

- Checkpoints: `s53-w4102`, `s53-w4106` (2 of 24 under the prior pin)
- CUDA-event time: 264.50 seconds (4.41 minutes); host benchmark wall time: 322.05 seconds
- Telemetry: 925 rows; 0 of 8 phases eligible for activity claims because timestamp gaps exceeded the gate
- Recorded GPU busy: 100% median and peak
- Peak reported VRAM: 5,064 MiB
- Power: 627.7 W median and 699.8 W peak
- Peak temperature: 54 C

Each raw checkpoint, its attempt receipt, the old source pin, and the old partial audit are retained alongside this note. These measurements remain part of the resource-use history but are not pooled with measurements collected under the revised protocol.

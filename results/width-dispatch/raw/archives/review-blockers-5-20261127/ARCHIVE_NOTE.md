# Fourth recovery-review pilot archived

The `s53-w4102` pilot completed under source revision `41e623a5772affc8f77ae3c77a882a19bac01b2e`. Its complete checkpoint passed the then-current source and evidence audit. A subsequent independent runner review found one P1 cache-recovery receipt defect and additional P2 audit/runner gaps. This result is preserved for protocol-development history and is not pooled with future runs.

- Attempt: `2069e7f640524dd6b4fa1350505a0cc8`; checkpoint status: `complete`
- CUDA-event work: 2.20 minutes; telemetry: 472 rows, 0/4 activity-eligible phases (gaps exceeded the 1.5-second threshold)
- GPU busy median/peak: 100% / 100%; VRAM peak: 5,064 MiB
- Power median/peak: 642.3 / 690.7 W; temperature peak: 50 C
- Three point-in-time GPU snapshots preserve two in-run samples and one post-run sample.

The checkpoint, attempt receipt, source pin, partial audit and monitoring snapshots are retained with SHA-256 digests in `manifest.json`.

# Superseded pilot evidence archive

The checkpoint `s53-w4102` completed under source revision `cb8cf24863194ada8534967b1ff3de463d0a1ae4` and passed the partial evidence audit under that pin. A subsequent review identified execution-recording and timeout-handling gaps, so this pilot is preserved for provenance but excluded from the replacement source pin. The code changes affect the runner and audit path; the kernel result is not being erased.

- Status: `complete`; attempt `5f9112b1310145c5a3692914e51d4cbf`
- CUDA-event time: 2.20 minutes; remote benchmark wall time: 160.08 seconds
- GPU busy median/peak: 100% / 100%
- VRAM peak: 5,064 MiB
- Power median/peak: 627.3 W / 696.8 W
- Temperature peak: 51 C
- Activity telemetry: 491 rows across 4 phases; 2 phases met the activity eligibility gate

All original checkpoint, attempt, source-pin and partial-audit files are retained here. `manifest.json` records each archived file's size and SHA-256. This historical resource-use record is not pooled into the fresh, revised pilot.

# Review-blocked pilot evidence archive

The fresh pilot `s53-w4102` completed on `d926b34a9bddb8eeec7c3e5db2905909b9300983` and passed the then-current one-checkpoint audit. Independent review later found cases where the audit could accept contradictory row counts, below-floor memory snapshots or negative event gaps, and where a local timeout or benchmark crash could be misclassified. The actual saved record has 32,768 rows and at least 74.15 GiB free before every measured phase; the result is preserved as historical evidence but excluded from the replacement source pin.

- Attempt: `072fee275bc5434cbc73b06cd147291a`; status: `complete`
- CUDA-event work: 2.20 minutes; remote benchmark wall: 171.44 seconds
- GPU busy median/peak: 100.0% / 100.0%
- VRAM used peak: 5064 MiB
- Power median/peak: 629.0 / 698.7 W
- Temperature peak: 50 C; in-window telemetry: 456 rows, 0/4 phases activity-eligible because each phase had a 2.31–2.53 second query gap
- Separate instantaneous H100 snapshots preserve one measurement during the run and one after it stopped.

All checkpoint, attempt, source-pin, partial-audit and monitoring files are preserved here. `manifest.json` records sizes and SHA-256 digests for the archived source files. These readings are not pooled into the replacement protocol.

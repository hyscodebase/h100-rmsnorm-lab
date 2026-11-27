# Pilot evidence archived after recovery-path review

The pilot `s53-w4102` completed under `90cc5ff2f78ec7b8ee7ac4e4c7af113c0ecf7b9d` and passed the complete-checkpoint audit. A follow-up independent review found two remaining issues only in the special telemetry-only partial-failure classifier: it did not require each accepted partial phase to satisfy the 16 GiB free-memory floor and could accept negative event gaps. These do not occur in this successful record, which shows 32,768 rows and at least 74.15 GiB free before each measured phase, but the data is archived so the next pin starts from a reviewed, consistent protocol.

- Attempt: `8af5253abab14d6698ec06ac5f67e6fe`; status: `complete`
- CUDA-event work: 2.20 minutes; remote wall time: 150.62 seconds
- GPU busy median/peak: 100.0% / 100.0%
- VRAM peak: 5064 MiB
- Power median/peak: 624.5 / 692.1 W
- Temperature peak: 50 C; 447 in-phase telemetry rows; 0/4 activity-eligible phases because all phase query gaps exceeded the 1.5-second threshold.
- Two time-stamped monitoring snapshots preserve one in-run sample and one after the worker stopped.

All checkpoint, attempt, source-pin, audit and monitoring data is retained. `manifest.json` contains each archived evidence file's size and SHA-256.

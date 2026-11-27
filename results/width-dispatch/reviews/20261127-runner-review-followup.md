# Runner review follow-up — 2026-11-27

## Scope

Reviewed the unstaged follow-up changes on top of source revision `a74c073264b3df26d7422e509fe1e8594548a376` for issue #66. The review covered producer/retrieval attempt linkage, timeout recovery, checkout-scoped cache identity, pre-latency and capture memory evidence, partial-checkpoint validation, frozen numeric gates, local-launch attempt accounting, driver provenance, receipt redaction, and status-137 classification.

## Review result

The independent GPT-6 Luna review found no remaining actionable P1, P2, or P3 defects in that scope. GPT-6 Luna was explicitly requested as the review target, but the tool did not expose its effective runtime model; model execution remains unverified.

The fixes preserve both the producing and retrieving attempt IDs, use a local persistent run ID bound to the full source pin for remote-cache isolation, leave remote result-transfer timeouts unresolved until a durable record is retrieved, and exclude confirmed local process-launch failures from the submitted-attempt limit. The audit now validates latency rows and calibrated replay data for telemetry-only partial records, rejects booleans as numeric measurements, recomputes capture-memory deltas from per-backend snapshots, and enforces frozen dtype, epsilon and correctness tolerances. The runner also records a free-memory snapshot immediately before latency timing, queries driver versions without assuming GPU index zero, and records exit 137 without inferring a timeout.

## Static checks

- `python3 -m py_compile experiments/width-dispatch/step.py experiments/width-dispatch/summarize.py experiments/width-dispatch/run.py` — passed.
- `bash -n experiments/width-dispatch/ssh_runner.sh` — passed.
- `python3 experiments/width-dispatch/summarize.py --check-plan` — passed.
- `git diff --check` — passed.

No tests were run and no H100 work was submitted during this static review. The nine previously saved checkpoints remain archived and excluded from the active experiment's analysis; the next eligible measurement must use the new frozen source pin.

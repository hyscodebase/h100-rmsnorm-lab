# Runner review log - source `41e623a5772affc8f77ae3c77a882a19bac01b2e`

## Scope and disposition

- Source was pinned for the `s53-w4102` H100 pilot. The checkpoint completed and passed the then-current audit; it is preserved in `../raw/archives/review-blockers-5-20261127/` and excluded from the next source pin's performance analysis.
- The subsequent Codex CLI review completed on 2026-11-27 and found one P1, nine P2 and one P3 issue.
- The independent GPT-6 Luna review found an additional P2 numeric-validation edge. Effective runtime model selection was not independently verified.
- No code was edited by the reviewing agent. The full 24-checkpoint matrix remains closed until the findings are addressed and a fresh pinned pilot passes review and evidence audit.

## Codex CLI findings

1. **P1 - Preserve the producing attempt ID through receipt audit.** A cache hit has a retrieval attempt and an original producing attempt. The checkpoint summary retained only the retrieval ID, so a legitimate cached checkpoint could fail receipt audit and block resume.
2. **P2 - Separate result-transfer timeouts from execution failures.** The outer remote deadline includes reading the already-published result. A stalled transfer can replace a successful remote checkpoint with a local terminal failure.
3. **P2 - Scope cached checkpoints to a recoverable run.** A fresh checkout can encounter a cache result whose producer receipt is absent locally, then consume retries fetching the same unusable payload.
4. **P2 - Check free memory immediately before latency measurement.** The 16 GiB guard was checked before allocation and sustained phases, but not after graph capture and before the latency rounds.
5. **P2 - Require compute evidence for telemetry-only failure classification.** A partial record could be labeled telemetry-only without validating calibrated replay counts and the completed phase's compute evidence.
6. **P2 - Freeze numerical plan settings.** The plan audit did not enforce the registered epsilon, BF16 dtype or strict/cross-check tolerances.
7. **P2 - Recompute capture-memory peak from saved snapshots.** The audit trusted the aggregate field and did not prove that it matched the per-backend capture snapshots.
8. **P2 - Exclude confirmed local launch failures from the remote attempt limit.** Failures before SSH submission could consume the three-submission allowance.
9. **P2 - Query the accessible GPU driver instead of physical GPU 0.** A job allocated a nonzero device could fail preflight when access to GPU 0 was denied.
10. **P2 - Keep local checkout paths out of published failure receipts.** A subprocess error string could include a local absolute runner path and username.
11. **P3 - Do not classify every SIGKILL as a benchmark timeout.** Exit 137 can result from an external kill such as host OOM; the record should keep the cause unconfirmed without separate deadline evidence.

## Independent Luna finding

- **P2 - Reject booleans in numeric evidence fields.** In Python, `bool` is a subclass of `int`, so timing checks using `isinstance(value, (int, float))` can accept `True` as a numeric duration. Numeric audit fields should explicitly reject booleans.

## H100 usage preserved with this review

- Checkpoint: `s53-w4102`; CUDA-event duration: 132.197 seconds (2.20 minutes); host benchmark wall time: 158.18 seconds.
- GPU busy median/peak: 100% / 100%; VRAM peak: 5,064 MiB; telemetry: 472 rows, 0/4 activity-eligible phases because query gaps exceeded the registered limit.
- Power median/peak: 642.3 / 690.7 W; temperature peak: 50 C.
- These are resource-use observations, not an accepted width-dispatch performance result. The full source pin and raw data are recorded in the linked archive.

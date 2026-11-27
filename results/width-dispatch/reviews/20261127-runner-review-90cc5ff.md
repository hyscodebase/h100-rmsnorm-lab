# Width-dispatch recovery-path review log — 2026-11-27

## Review and execution overlap

- Reviewed source: `90cc5ff2f78ec7b8ee7ac4e4c7af113c0ecf7b9d` (`codex/width-dispatch-issue66`).
- A Codex CLI review was started in parallel with the H100 pilot. It repeatedly hit model-refresh timeouts and stream disconnects; it was stopped without a completed Codex review result.
- The GPT-6 Luna review dispatch was requested in parallel. Effective runtime model selection remains unverified.
- The H100 pilot `s53-w4102` completed under this pin and passed the normal complete-checkpoint audit. It is retained in `../raw/archives/review-blockers-4-20261127/`, but excluded from the next source pin.

## Remaining Luna review findings

1. **P1 — partial telemetry-only failure audit omitted the per-phase free-memory floor.** The normal complete-checkpoint path checked it, but the special recovery classifier could still accept a failed partial record whose pre-phase `free_bytes` was below 16 GiB.
2. **P3 — the same partial-recovery path accepted negative event gaps.** It checked the upper bound only.

No other P1/P2/P3 finding was reported in that review. The actual archived successful record had 74.15 GiB or more free before each phase; the issue was a missing guard in a separate partial-failure path.

## Changes prepared for re-review

- Require the partial phase's memory snapshot timestamp and free-memory floor.
- Require every partial phase event duration to be finite and positive, each inter-window gap to be finite and nonnegative and within the registered maximum, and the sum of durations to match the recorded phase duration.
- Require the partial phase's compute gate to have passed before it can be classified as telemetry-only.
- Lock the eight correctness fixture names in the plan audit so plan edits cannot silently replace the audited fixture set.

These changes passed Python compilation, Bash syntax validation, the registered-plan audit, archive-manifest hash verification and `git diff --check`. A fresh source pin and re-review are still pending. The full matrix remains closed until the protocol and pilot source are reviewed.

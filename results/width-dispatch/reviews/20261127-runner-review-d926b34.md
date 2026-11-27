# Width-dispatch runner review log — 2026-11-27

## Review and execution overlap

- Reviewed source: `d926b34a9bddb8eeec7c3e5db2905909b9300983` (`codex/width-dispatch-issue66`).
- Codex PR review ran at the same time as one H100 checkpoint, `s53-w4102`.
- An independent GPT-6 Luna review was requested and reported no P1/P2 findings. The dispatch tool did not verify the effective runtime model, so Luna selection remains unconfirmed.
- The H100 checkpoint completed and passed its then-current partial audit. Its evidence is preserved in `../raw/archives/review-blockers-3-20261127/` and excluded from the replacement pin.

## Codex review findings

The review found two P1, five P2 and one P3 issue:

1. **P1 — per-phase free memory was not enforced or audited.** The actual archived pilot had at least 74.15 GiB free before each measured phase, but the audit could still accept a tampered low-memory snapshot.
2. **P1 — local SSH timeout was treated as proof of remote execution timeout.** A completed remote job whose result transfer stalled could be converted into a blocking failed checkpoint.
3. **P2 — non-timeout benchmark crashes were recorded as transport failures.** A known remote crash could be resubmitted instead of saved as a blocking execution failure.
4. **P2 — setup-script stdout could corrupt the JSON checkpoint stream.**
5. **P2 — cached measurements did not preserve the original producing attempt ID.**
6. **P2 — local preflight omitted Bash and Python executables required by the runner.**
7. **P2 — the audit did not bind the recorded row count to the registered shape.**
8. **P3 — negative CUDA event gaps passed the upper-bound-only check.**

## Changes prepared for re-review

- Enforce and audit the 16 GiB free-memory floor before every sustained phase.
- Keep local SSH timeouts unresolved as transport failures; preserve the remote producing attempt ID and link a later cache retrieval to its earlier receipt.
- Publish timeout and non-timeout benchmark exits as source-bound `execution_failed` records.
- Redirect setup-script stdout to stderr and check Bash/Python availability before creating an attempt receipt.
- Audit the exact registered row count and reject non-finite, negative or over-limit event gaps.

These changes passed Python compilation, Bash syntax validation, the registered-plan audit and `git diff --check`. The fixes still need a fresh PR review and a new source-pinned H100 pilot before full-matrix execution.

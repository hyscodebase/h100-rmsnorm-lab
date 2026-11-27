# H100 width dispatch measurements

5/24 registered units completed at 32,768 BF16 rows. Latest saved measurement (UTC) `2026-11-27T15:06:34.882394+00:00`.
The 8 submitted attempts include the two early SSH failures and the initial driver preflight failure at width 4107.
All completed units passed the numerical, replay and memory checks.

**11.02 minutes** of sustained CUDA-event measurements, 2,168 device samples, and **0/20** phases eligible for activity estimates.
Recorded samples have median/peak GPU busy 100.0/100.0%, median/peak power 594.2/699.8 W, peak VRAM 5,064 MiB and peak temperature 53 °C. Sparse traces are retained as diagnostics. Each formal activity claim requires the phase coverage audit.

## Paired measurements

Rates are logical input plus output GB/s. Gains compare the four-row candidate with each reference from the same run.

| Seed / width | 4-row GB/s | Best 1-row GB/s | compile GB/s | vs 1-row | vs compile | Both ≥5% |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| [53 / 4102](raw/archives/driver-preflight-d42481a/checkpoints/s53-w4102.json) | 2,938.15 | 2,986.46 | 2,751.64 | -1.62% | +6.78% | no |
| [53 / 4106](raw/archives/driver-preflight-d42481a/checkpoints/s53-w4106.json) | 2,939.48 | 2,985.88 | 2,756.22 | -1.55% | +6.65% | no |
| [53 / 4107](raw/checkpoints/s53-w4107.json) | 2,610.12 | 2,331.85 | 1,872.28 | +11.93% | +39.41% | yes |
| [53 / 4108](raw/checkpoints/s53-w4108.json) | 2,953.15 | 3,011.24 | 2,880.12 | -1.93% | +2.54% | no |
| [53 / 4109](raw/checkpoints/s53-w4109.json) | 2,610.22 | 2,330.70 | 1,872.20 | +11.99% | +39.42% | yes |

## Decision

Next registered unit: `s53-w4111`. A width qualifies after the candidate beats both references by at least 5% in all three seeds.

## Evidence

Accepted units by benchmark revision: `d42481aef66b` (2), `2a38f189c879` (3).
The first width-4107 attempt stopped before any measured work. Its original archive and the resumed measurement have separate source and run identities.
[Plan](../../experiments/width-dispatch/PLAN.md) · [Continuation and attempt rules](../../experiments/width-dispatch/RESUME.md) · [Original preflight failure](raw/archives/driver-preflight-d42481a/checkpoints/s53-w4107.json) · [Current audit](raw/partial-audit.json) · [PR #67](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/67)

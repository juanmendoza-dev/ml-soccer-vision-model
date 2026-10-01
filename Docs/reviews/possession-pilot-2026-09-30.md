# Learned possession: timing pilot (2026-09-30)

03 stage 8, "Timing plan". Workstation CPU (i5-11400F, `num_threads = 6`, 32 GB), `python -m vision.possession_train --model-id lgbm-v1-nested5x4-s1 --pilot`, in a fresh process. The decision uses time and memory only. No agreement, log loss or shot metric was looked at. Raw output: `data/models/possession/lgbm-v1-nested5x4-s1/pilot.json`.

## Decision: stride 1 (`lgbm-v1-nested5x4-s1`)

| check | bar | pilot |
|---|---|---|
| projected possession stage | ≤ 10 h | **3.19 h** (4.82 h if the determinism check fails) |
| peak RSS | ≤ 24 GiB | **6.8 GiB** |

Both pass at stride 1, so `-s2` and `-s4` aren't needed. The pilot's `fits/outer_0/` is the real `outer_0`. The driver reuses it only if its `fit.json` still matches the plan, the stride, the parameters, the feature contract, the input hashes and the model file.

## outer_0 (the largest fit)

| | |
|---|---|
| training matches | 52 (probe 44, early stopping 8) |
| refit rows | 1,660,373 eligible → 3,320,746 mirrored |
| best round | **2,000** (the cap: validation log loss was still falling, so early stopping never fired) |
| probe (design + train) | 1 + 371 s |
| refit (design + train) | 1 + 427 s |
| total | 801 s (13.3 min), about 0.21 s per refit round |
| peak RSS | 6.8 GiB (whole process, including all 52 training tables in memory) |
| pilot overhead | 10 s (loading the 52 tables, outside the fit's timings) |

03 had guessed that a probe stopping near the baseline's 113–216 rounds would take 2–4 min per fit, and that 1,000–2,000 rounds would take 10–25 min. The fit ran to the 2,000 cap, and 13.3 min sits inside that guess. The cap is the spec's and stays where it is. A fit still improving at 2,000 rounds isn't a timing problem, and per 03 this step decides nothing on accuracy.

## Largest match (10508: 234,874 native frames, 78,368 grid rows)

| step | time |
|---|---|
| extraction to a cache (`pf.build`) | 1.9 s |
| cache load with hash checks + 2D rule | 0.6 s |
| inference, q(x) and q(Mx), 2,000 trees | 3.7 s (about 24 µs per row for both predictions) |

## Training rows (eligible, stride 1) per training

| training | rows | | training | rows |
|---|---|---|---|---|
| outer_0 | 1,660,373 | | pair_0_1 | 1,242,429 |
| outer_1 | 1,604,276 | | pair_0_2 | 1,242,724 |
| outer_2 | 1,604,571 | | pair_0_3 | 1,249,446 |
| outer_3 | 1,611,293 | | pair_0_4 | 1,246,520 |
| outer_4 | 1,608,367 | | pair_1_2 | 1,186,627 |
| | | | pair_1_3 | 1,193,349 |
| | | | pair_1_4 | 1,190,423 |
| | | | pair_2_3 | 1,193,644 |
| | | | pair_2_4 | 1,190,718 |
| | | | pair_3_4 | 1,197,440 |

The pairs are 0.71–0.75× `outer_0`; 03 estimated 0.75×.

## Projection

| part | how | hours |
|---|---|---|
| extraction, 64 matches | 1.9 s × (all native frames / 10508's) | 0.03 (measured: 87 s for all 64) |
| 15 trainings + the determinism twin (`pair_3_4` again) | 801 s × Σ rows / `outer_0`'s rows | 2.87 |
| predictions, 25 logical fits (each match 5×: 1 test + 4 inner-OOF) | (0.6 + 3.7 s) × 5 × all frames / 10508's | 0.30 |
| **total** | | **3.19** |
| if the determinism check fails (25 trainings + twin) | | 4.82 |

Notes:
- Training time is scaled by rows, which assumes every fit runs to the same round count. `outer_0` hit the 2,000 cap, and no fit can go further, so this is close to an upper bound.
- 03's extraction formula multiplies the one-match time by 64 *and* by the frame ratio, which counts the matches twice. The table uses the frame ratio only. Extraction is cached already, either way.
- The predictions run when `prediction.possession` first fills a context's cache (the gate script or `prediction.cv`), not inside `vision.possession_train`. They're in the budget, but not in the overnight command.

## Also found while building

- **Mirror and Float32 differences.** `pf.mirror(pf.mirror(x)) != x` bitwise in all 12 `diff_*` columns on every cached match: the cache subtracts in Float64 and rounds once, and M recomputes in Float32. The model now reads every `diff_*` recomputed in Float32 (`possession_model.canonical`), the same way M recomputes it. M is then an exact involution and |p(x) + p(Mx) − 1| ≤ 1e−12 holds. Values move by at most 4e−6. 03 has the clarification, and the `pfeat-v1` cache and contract are unchanged.
- `all_estimated`, eligibility and the training-row selection match the resampler's `frames_10hz` on all 64 matches (`tests/test_possession_model.py`).

## Next

1. Overnight: `python -m vision.possession_train --model-id lgbm-v1-nested5x4-s1`. It runs the 4 other outer fits, `pair_3_4` and its determinism twin, then the other 9 pairs, and seals the manifest. About 3 h.
2. The gate: `possession_split.py --scored h5 --possession-model <id>`, still to build. It reads the concatenated `test` contexts through `prediction.possession.learned_state`.

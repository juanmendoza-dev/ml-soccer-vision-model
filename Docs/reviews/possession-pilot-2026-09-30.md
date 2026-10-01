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

## Checks before the overnight run

- The pilot's `outer_0` is reusable under the full run's inputs (`possession_train.reusable` returns its record).
- Real-data symmetry with that booster on two fold-0 matches (10505, 10517): max |p(x) + p(Mx) − 1| = 1.1e−16, and the hard labels swap on every row away from 0.5.
- The provider `features_v1_held` rebuilt with `cache=False` equals the cached file on 10502 and 10517. The old inferred caches (`*_pinf_e737054b5d`, `state_inferred_age_e737054b5d`) aren't on the workstation, so they can't be compared here.
- `vision.possession_train` now keeps Windows awake like `prediction.cv`.

## Overnight run (2026-09-30 22:35 to 2026-10-01 01:38)

`python -m vision.possession_train --model-id lgbm-v1-nested5x4-s1`, then `python -m prediction.possession --possession-model lgbm-v1-nested5x4-s1`. Both exited 0. Training wasn't looked at beyond what is shown here: no agreement, no log loss, no shot metric.

- **Sealed:** `data/models/possession/lgbm-v1-nested5x4-s1/manifest.json`, with 25 logical fits over 15 trainings. `check_manifest` passes, and every `model.txt` and `fit.json` matches its recorded SHA256.
- **Determinism:** `pair_3_4` trained twice from scratch gave the same `model.txt` SHA256. Pairs are deduplicated, as 03 planned.
- **Time:**
  - trainings: 10,117 s (2.81 h, `outer_0` from the pilot included), plus 597 s for the twin
  - learned states: 1,065 s (0.30 h)
  - whole stage: 3.27 h, against the 3.19 h projection and the 10 h budget
- **Peak RSS:** 7.2 GiB.

| training | matches | mirrored rows | rounds | s |
|---|---|---|---|---|
| outer_0 | 52 | 3,320,746 | 2,000 | 801 |
| outer_1 | 51 | 3,208,552 | 1,994 | 817 |
| outer_2 | 51 | 3,209,142 | 1,986 | 808 |
| outer_3 | 51 | 3,222,586 | 1,991 | 905 |
| outer_4 | 51 | 3,216,734 | 1,974 | 879 |
| pair_0_1 | 39 | 2,484,858 | 1,721 | 552 |
| pair_0_2 | 39 | 2,485,448 | 1,991 | 614 |
| pair_0_3 | 39 | 2,498,892 | 1,995 | 617 |
| pair_0_4 | 39 | 2,493,040 | 1,934 | 619 |
| pair_1_2 | 38 | 2,373,254 | 1,623 | 509 |
| pair_1_3 | 38 | 2,386,698 | 2,000 | 602 |
| pair_1_4 | 38 | 2,380,846 | 1,992 | 600 |
| pair_2_3 | 38 | 2,387,288 | 1,992 | 595 |
| pair_2_4 | 38 | 2,381,436 | 1,947 | 592 |
| pair_3_4 | 38 | 2,394,880 | 1,995 | 608 |

Most fits run to near the 2,000-round cap. Three stop clearly earlier (`pair_0_1` 1,721, `pair_1_2` 1,623, `pair_0_4` 1,934).

**Learned states, verified after the run:**
- All 320 (64 matches × 5 outer contexts) re-read through `prediction.possession.learned_state`, with every hash check, in 24 s.
- Every state has 03's columns and its context's `fit_id`, and its rows and native `frame_id` equal `frames_10hz`'s.
- `p_home` is non-null exactly where fallback is false, and possession there is home or away.
- **Fallback:** 1,570,118 of 3,912,924 outer-test grid rows (40.1%). These are exactly the `all_estimated` rows (1,570,118, all overlapping), where no player is visible. On the 1,977,379 H = 5 scored rows there are **0** fallbacks. That row count is also the gate's expected baseline.

## Next

1. ~~Overnight training and learned states~~: done, see above.
2. The gate: `possession_split.py --scored h5 --possession-model <id>`, still to build. It reads the concatenated `test` contexts, reproduces the baseline counts first (1,977,379 rows, 49,812 positives, 281,606 overall and 154,806 first-3-s disagreements; a mismatch stops it), then checks the five bars.

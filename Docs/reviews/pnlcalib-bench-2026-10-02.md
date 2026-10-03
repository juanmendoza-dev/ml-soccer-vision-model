# PnLCalib as stage 4 on the bench (2026-10-02)

The implementation of 03 Pitch calibration → PnLCalib, scored by `python -m vision.bench` on vb01–vb03. Earlier the same day `scripts/pnl_compare.py` put PnLCalib (WC14) at 89% within 2 m on every 5th frame, with no gate or filter (`Docs/reviews/pnlcalib-2026-10-02.md`). This is the real pipeline: gate, filter, every frame, replay check.

## What was built
- `vision/pnlcalib/`: upstream's inference code at commit `8c87391`, changed only in two import lines (README there).
- `PnLCalibCamera` (`vision/stages.py`, GPU): image → per-channel heatmap peaks. `vision/calib.py` (CPU): peaks → camera (upstream voting, a fresh calibrator per call) → per-call checks → ground homography in the TV frame. On a vb03 frame it gives upstream `inference`'s camera to 1e−14 m.
- The pipeline's stage 4 branches on `calib_backend`. The gate now takes a pass/fail pitch check: an accepted camera (PnLCalib) or `min_keypoints` (roboflow). The camera becomes a `Fit` on a 5 × 3 image grid, so `HomographyFilter` is unchanged.
- `camera.parquet` (peaks + voted camera per call); replay from cached cameras, `--revote` from the peaks, sparser cadence by skipping calls; `vision.run --calib-backend pnlcalib --pnl-weights-dir`. An old `run.json` replays as roboflow.
- Tests: `tests/test_vision_calib.py`, a synthetic 3D broadcast camera whose projected keypoints feed a fake nets stage (round trip, checks, pipeline, cache, exact replay and revote, blind calls, cadence, old runs). Full suite 619 passed before the default flip; vision + demo tests 156 passed after.

## Runs
Fresh `vision.run` on all three clips, same pre-roll and frame count as the reviewed roboflow runs, RTX 2060, `--calib-backend pnlcalib` with the then-defaults (window 3, jump 5 m). **3.0–3.4 fps**, against 4.0 for roboflow; 03 predicted ~3. vb01's log ends at 4.0 fps, since it spends 10% of its frames in `other`. The roboflow caches are kept as `<clip>-roboflow`. The replay check passes on all three, and `--revote` at the run's thresholds reproduces vb03's run exactly.

## Results

| | Roboflow (7.2) | PnLCalib, as run | PnLCalib, picked config |
|---|---|---|---|
| vb01 within 2 m / median | 20.9% / 2.43 m | 71.5% / 1.09 m | **86.2% / 0.70 m** |
| vb02 | 32.4% / 2.25 m | 76.0% / 0.97 m | **87.8% / 0.62 m** |
| vb03 | 15.3% / 3.10 m | 92.9% / 0.62 m | **94.4% / 0.51 m** |
| Pooled within 2 m / median | 22.1% / 2.59 m | 81.6% / 0.82 m | **90.0% / 0.59 m** |
| Geometry missing, pooled | 10.4% | 6.6% | **5.8%** (view `other` 5.3%, rejected 0.5%) |
| False live vb01 / vb02 | 1.5 s / 2.9 s | 2.6 / 1.7 s | **0.1 / 0.2 s** |
| Outfield team accuracy vb01 / vb02 / vb03 | – / 89.5% / 73.0% | | 96.6% / 97.6% / 98.6% |

Picked config: `homography_window` 1, `max_homography_jump_m` 10, `pnl_blind_kp` 4. These are now the defaults, and `calib_backend` defaults to `pnlcalib`.

**03's bar passes:** pooled ≥ 80% (90.0%), every clip ≥ 75%, geometry missing ≤ 10.4% (5.8%), false live not above now.

## What moved it
- **The window.** With 3 trailing homographies averaged, the camera lags every pan. vb01 and vb02 follow attacks, so their median sat at ~1.0 m against 0.6 m in `pnl_compare`. 07's sweep (`homography_window` 1–3 × `max_homography_jump_m` 2 / 5 / 10): window 1 is best on all three clips. Jump 10 vs 5 is within 0.2 pt with less rejected, so 07's tie-break takes 10. Jump 2 rejects 4.6% of match frames for nothing.
- **False live, and a rule 03 didn't have.** PnLCalib's gate leaves a close-up after the minimum `off_after_s` (0.5 s per mark on vb01/vb02, against 0.5–2.2 s for roboflow). But during those 0.5 s the held camera was still fresh, so close-up boxes got positions: 2.6 s on vb01, above roboflow's 1.5 s. Roboflow's lower number was its homography failing on the close-up, not its gate. New field `pnl_blind_kp` (4): a rejected call with fewer keypoints than that sees no pitch, and drops the camera instead of holding it. Close-ups have 0 keypoints on 97 of 98 sampled frames; vb03's broken cameras have 4–9, so they're still held. Cost 0.3 pt pooled (90.3 → 90.0%, all on vb03, 95.1 → 94.4%), inside 07's 0.5 pt tie window. 07's pick rule doesn't see false live, so this one was chosen for the bar, not by the rule. A run without the field replays with 0 (always hold).
- **`homography_max_age_s`** 0.5 / 1 / 2 barely matters at every 5th frame (≤ 0.3 pt). Kept at 1.
- The vb03 run near 281 s (03 asked): the view never left `match` (view `other` 0.0%), and rejected is 0.8% of match frames.

## Calibration rate (for live, 09)
Replay with every k-th call (approximate: the gate still saw every call), picked config:

| Stage 4 every | Calls a second | Within 2 m | Median / p90 |
|---|---|---|---|
| 5th frame | 6 | 90.0% | 0.59 / 1.41 m |
| 10th | 3 | 86.4% | 0.70 / 1.77 m |
| 15th | 2 | 80.4% | 0.82 / 2.26 m |
| 30th | 1 | 65.8% | 1.09 / 3.06 m |

Holding a camera between calls costs a lot once the camera pans: 1 Hz loses a quarter of the accuracy. A longer max age doesn't help. 03's live plan (1–2 Hz, hold the last camera) isn't enough. Live needs ~3 Hz or more, or something that carries the camera between calls (frame-to-frame tracking of the pitch, or a PTZ model with fixed position and smoothed pan/tilt/zoom).

## What's left
- **View `other` is now the largest loss:** 10.6% of vb01's scored frames, 6.7% of vb02's. It's the gate's 1 s `on_after_s` after each close-up (vb02 22.2–23.0 s is one), not geometry. A shorter `on_after_s` trades directly against false live. That's a gate sweep with fresh runs, since replay doesn't simulate the gate.
- **vb02 keepers 63.8%** (roboflow 61.3%): assignment by mean x, not geometry.
- **Live rate:** see above. 09's row now has numbers.
- **More stadiums:** the WC14 pick and these defaults are tuned on three clips at two stadiums. The geometry-only clips (2014/2018) are next.

## Gate `on_after_s` (2026-10-02, later)
The 1 s wait before going back to `match` after a close-up was the largest loss left. Fresh runs (replay doesn't simulate the gate) on vb01 and vb02 with `--set on_after_s=0.5` (new `vision.run --set`), scored by `vision.bench`.

| | `on_after_s` 1.0 | 0.5 |
|---|---|---|
| vb01 view `other` / within 2 m | 10.6% / 86.2% | **5.4% / 90.0%** |
| vb02 | 6.7% / 87.8% | **3.5% / 90.8%** |
| False live vb01 / vb02 | 0.1 / 0.2 s | 0.1 / 0.2 s |
| Back to `match` after each mark | 1.00–1.13 s | 0.50–0.63 s |

- **Per mark (`view.parquet`):** on all 8 close-ups the only match-view time inside the mark is the 0.47–0.63 s off-lag at its start, the same as at 1.0. No mark re-enters `match`, so no close-up gets through. All unmarked `other` time is now the on-lag: 2.6 s on vb01 (was 5.1), 1.6 s on vb02 (was 3.1).
- **Picked 0.5:** the lowest value allowed, and it passes the rule (false live ≤ 0.5 s per clip, no mark let through). 0.7 wasn't run: it can't beat 0.5 under the rule. 8 close-ups on 2 clips is a small sample, so it doesn't go lower. A close-up with a lucky camera would need 0.5 s of accepted probes (3 in a row at every 5th frame); 1 of 98 close-up frames had one.
- **Default now.** All three clips rerun on the new defaults (vb03 too, for the ball cache; it has no cuts and gives the same 94.4%). Pooled: **within 2 m 92.0%** (from 90.0%), median 0.59 m, geometry missing **3.2%** (view `other` 2.7%, rejected 0.4%). The bench now scores these runs without `--set` overrides.
- These runs also write `balls.parquet` (every ball candidate, 03 Diagnostics), so the ball work replays on CPU. The bench's replay check now covers the ball rows too, and passes on all three.
- **Run speed:** two runs at once on the 2060 filled its 6 GB, spilled to shared memory and took 67 min each (0.1–0.5 fps). One at a time: 3.1 fps. Run them one after another.
- `run.json`'s `git_commit` is stamped when the run closes, so vb01/vb02 show a commit from after they started. The pipeline code didn't change in between (only the bench and docs).

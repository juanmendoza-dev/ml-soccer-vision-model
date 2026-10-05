# Offline Demo (end to end) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One private demo clip: real broadcast footage of a World Cup 2022 goal, with the vision pipeline's output feeding the goal model and a danger meter (P(goal) in the next 5 s) drawn on the video. The real PFF shots and goals are drawn as markers, so the lead time is visible.

**Architecture:** Nothing new in the model, only the joins between the existing parts:
- `vision.run` → offline stage 8 fill (`vision/stage8.py`) → `prediction.infer` (in-memory resample, v1 features, the demo goal model, 05's inference mask) → `demo.video` (ball marker + danger meter + PFF truth ticker).
- The demo goal model is the CV's outer-fold LightGBM, refit and saved, so the clip's match was never trained on. It's paired with xG v1 and that fold's cross-fitted P(goal) map.
- The clip comes from 10517 (ARG–FRA), whose PFF tracking gives a reference curve to compare against.

**Tech Stack:** Python 3.11, polars, LightGBM, OpenCV, pytest. Run everything with `.venv/Scripts/python` on the workstation: a plain `uv run` may drop torch.

**Spec:** `Docs/Specs/05-prediction-model.md`, `Docs/Specs/08-demo-overlay.md`, `Docs/Specs/03-vision-pipeline.md` (stage 8), `Docs/Specs/02-game-state-schema.md`. Task 1 adds the sections this plan builds on. Roadmap: Phase 3, the first three items.

## Global Constraints

- **No leakage.** A prediction at frame t uses frames `<= t` only. The goal model never trained on the clip's match (10517 is in fold 0 of `data/splits/folds.json`). The P(goal) map never saw 10517's goal labels.
- **Coordinates.** Pitch meters once data leaves `vision/`; pixels only for display.
- **Packages.** `vision/` and `prediction/` share only the 02 schema:
  - `vision/stage8.py` is vision code.
  - `prediction/infer.py` reads the 02 game state only, never the vision cache.
  - `demo/` may read both, as it already does.
- **Schema.** 02 stays as it is (schema 0.6). Predictions are not a 02 table: they go to `data/predictions/<match_id>/<model_id>.parquet`.
- **Footage.**
  - Broadcast footage is private (08 "Footage"). It is never committed and never published.
  - Renders go outside the repo, in `C:/footage/renders/`.
  - Each review names the footage a render used.
- **Git.** Small commits, pushed one at a time. Plain human messages, no AI attribution. Signing stays on.

## Review Focus

1. **Inference mask.**
   - Stage 8's `ball_state` is about 45% null. The resampler's `eligible` (`ball_state == "alive"`) and `LGBMModel.predict` (`predictable()`) would null those rows and blank the meter.
   - Vision inference must use 05's Inference rule instead: possession set, `ball_state` not dead (null counts as not dead), not `all_estimated`.
   - Pinned by `test_null_ball_state_is_predicted_dead_is_not` (Task 5).
2. **Demo model leakage and exactness.**
   - The saved model must be the CV's outer-0 model: same `best_iter` and `es_matches` as `lgbm-held-2026-09-27`, and p on fold-0 rows within 1e-3 of the run's out-of-fold `p_h5`.
   - The map must come from the other folds only.
   - Pinned by `test_fold_map_ignores_the_folds_own_labels` (Task 3). The exactness is checked at fit time and written to the manifest.
3. **Direction, teams and sync.**
   - A 180° direction error, or a null `home_cluster`, gives a flat meter or no meter, and nothing fails.
   - Check the clip with `vision.bench` against PFF before predicting:
     - within 2 m close to vb01's (same broadcast);
     - best offset within ±0.1 s;
     - team accuracy high.
   - `vision.stage8` refuses a run with no teams (`test_no_teams_is_an_error`, Task 4).
4. **Video frame alignment.**
   - `demo.video` used to seek to the cache's `frame_id` and ignored `run.json`'s `video_start_s`. A clip run with `--start-s` would draw every overlay on the wrong frame.
   - Pinned by `test_frames_line_up_with_the_runs_start` (Task 6).
5. **The meter is causal.**
   - A video frame shows the latest grid row at or before its time, never a nearer later row.
   - Nothing is shown across a grid gap.
   - Pinned by `test_a_frame_between_rows_shows_the_earlier_one` and `test_a_grid_gap_shows_nothing` (Task 2).

## File Structure

| File | What it does |
|---|---|
| `Docs/Specs/05-prediction-model.md` | + "Offline demo model" and "Vision inference" sections (Task 1) |
| `Docs/Specs/08-demo-overlay.md` | + danger meter element definition, video alignment, PFF truth ticker (Task 1) |
| `Docs/Specs/03-vision-pipeline.md` | + "Offline stage 8 fill" under stage 8 (Task 1) |
| `demo/meter.py` (new) | Which P(goal) a frame shows (causal lookup), bar level and label, `draw()` |
| `demo/overlay.py` | + `meter_color`, `danger_meter` (drawing only) |
| `demo/render.py` | + `--pgoal`: meter column in the pitch view |
| `prediction/goal_model.py` (new) | Fit, check and save the demo goal model. `GoalModel` loads it and predicts |
| `vision/stage8.py` (new) | Fill stage 8's columns into a finished vision run's `frames.parquet` |
| `prediction/infer.py` (new) | Vision game state → 10 Hz P(shot), xG, P(goal) with 05's inference mask |
| `demo/video.py` | Start-offset fix, `--predictions` meter, `--truth-clip` PFF ticker |
| `data/splits/demo_clips.json` (new) | The demo clip in `vision.bench`'s manifest format (sync, marks, direction) |
| `scripts/demo_compare.py` (new) | Vision vs PFF-tracking P(goal) on the clip, for the review |
| `tests/vision_gs.py` (new) | Builds a small vision-format game state for tests |
| `tests/test_demo_meter.py`, `tests/test_goal_model.py`, `tests/test_vision_stage8.py`, `tests/test_infer.py` (new); `tests/test_demo_video.py` | Tests |

Order: Tasks 1–6 are code and need no footage. Task 7 needs the clip download, which the user can do any time before it. Task 8 runs everything and writes the review.

---

### Task 1: Spec additions (05, 08, 03)

CLAUDE.md: formats and behavior go into the specs before code.

**Files:**
- Modify: `Docs/Specs/05-prediction-model.md`: insert after the "Recalibration (added 2026-10-01)" block, before "## Class imbalance".
- Modify: `Docs/Specs/08-demo-overlay.md`: append to "## Element definitions".
- Modify: `Docs/Specs/03-vision-pipeline.md`: append after stage 8's "Known limits and later live check" paragraph (around line 222).

- [x] **Step 1: Add to 05** (text to insert verbatim)

```markdown
## Offline demo model (2026-10-05)
The first end-to-end clip (roadmap Phase 3) needs a saved goal model. A broadcast clip from PFF match M is scored by models that never saw M, so its PFF tracking stays a fair reference.
- **P(shot):** the CV's outer-fold model for M's fold k (`data/splits/folds.json`), refit and saved. Same LightGBM config as `prediction.cv` (`MODELS["lgbm"]`: v1 features, held ball, H = 5 only), trained on the matches outside fold k, loaded in sorted order like `prediction.cv`. The fit has to reproduce `lgbm-held-2026-09-27`'s fold k: equal `best_iter` and `es_matches`, the same null rows, and p on fold k's rows within 1e-3 of the run's out-of-fold `p_h5` (another machine can move LightGBM a little). Otherwise it stops and nothing is saved.
- **xG:** `xg-v1` unchanged (never saw World Cup 2022). Its file hash goes into the manifest and is checked on load.
- **P(goal) map:** fold k's cross-fitted map, refit from the run's `pgoal.parquet` on the other four folds' scored rows (05 Recalibration). It has to equal that fold's row of the `pgoal.md` table. The all-64 `pgoal_map.json` isn't used: it was fitted on M's goal labels.
- **Stored:** `data/models/goal/<model_id>/` with `model.txt` and `manifest.json`: model_id, fold, training match ids, `best_iter`, `es_matches`, features and version, ball source, horizon, map (a, b), base run, the reproduction check, the xG dir and hash, git commit.
- **Clean training.** Training with the sensitivity degradations on is the deployable choice (Vision sensitivity test), but the existing map was fitted on clean P(shot). A degraded model needs its own CV run and map first, so it's a follow-up.
- **Not for live or public use.** Live needs an all-64 fit, and the public demo runs on self-recorded footage (08).

## Vision inference (2026-10-05)
`python -m prediction.infer --match-id <id> --model <dir>` scores a vision run's game state (02) after stage 8 has filled possession and ball state (03, "Offline stage 8 fill").
- The match is resampled in memory with `resample.resample_match` (`data/processed` isn't written). Vision has no events, so the label columns are empty and unused.
- v1 features on the held ball with the grid's own `possession_team` and `flipped` (stage 8's possession).
- **Mask:** this section's Inference rule, not the training `eligible`. Predict where `possession_team` is set, `ball_state` isn't dead (null counts as not dead, so vision gaps don't blank the meter) and some player is visible. Elsewhere every p is null. The shares of grid rows predicted, with null ball state and with no possession go in the sidecar JSON.
- A run with no possession anywhere is an error (stage 8 wasn't run).
- **Output:** `data/predictions/<match_id>/<model_id>.parquet`: `match_id, period, t_s, frame_id, timestamp_s, possession_team, ball_state, predicted, p_shot_h5, xg, p_goal_h5`. A same-stem `.json` holds the model manifest's id, the game state's file hashes and the shares.
- Causal: resampling, features and stage 8 all are, and a test changes objects after t and checks every row `<= t`.
```

- [x] **Step 2: Add to 08** (append under "## Element definitions")

```markdown
- **Danger meter (2026-10-05):** calibrated P(goal within 5 s) from 05's "Offline demo model" (`p_goal_h5` from `prediction.infer`, or a CV run's `p_goal_cal_h5` in the pitch view).
  - **Which value:** a frame at period time t shows the latest grid row of its period with `t_s <= t`. Never a later or interpolated row. When that row is more than 0.1 s old (a grid gap) or its p is null, the bar is grey and the value reads `--`.
  - **Scale:** log, so the build-up shows (the calibrated top decile is about 2%). 0.1% is an empty bar, 50% a full one, with ticks at 1%, 5% and 20%. The value is a percentage, one decimal under 10%. One fill color for the whole bar: green at the bottom of the scale, amber halfway, red at the top. Title: "goal in 5 s".
  - Checked on the PFF pitch view (`demo.render --pgoal`) before any vision clip. The scale may change once there, recorded here.
  - **On video:** a panel at the right edge of the frame.
- **Video frame alignment:** a vision run's `frame_id` 0 is source frame `round(video_start_s × fps)` (`run.json`, vision.run's `--start-s`). `demo.video` skips to it with `grab()`, frame-exact like vision.run, never by seeking.
- **PFF truth ticker (clips from PFF matches):** the match's PFF shots and goals, mapped onto the run's frames through the clip's sync offset (`data/splits/demo_clips.json`, `vision.bench`'s format), in the ticker labelled "(PFF)". Each shows from its own frame for 4 s, like the event ticker, so the meter's lead time can be read off the video.
```

- [x] **Step 3: Add to 03** (after stage 8's "Known limits and later live check" paragraph)

```markdown
     **Offline stage 8 fill (2026-10-05).** `python -m vision.stage8 --match-id <id>` runs `vision.state.infer` (rule-only, default `StateConfig`) on a finished vision run and writes `ball_state`, `possession_team` and `ball_carrier_id` into its `frames.parquet`, then the 02 validator. It records its config and shares in the cache's `run.json` under `stage8`. It's idempotent: infer reads only objects and frame times. `vision.run` writes the three as null again, so the fill runs after every `vision.run`, and after `vision.bench` (the bench replays from the caches and doesn't need it). A run where no player has a team is refused: possession would be null everywhere. **Rule-only is a temporary deviation** from the v1h decision above: `vision.possession_model.resolve` / `predict_match` refuse a match outside the manifest, so v1h on a vision run needs its own entry path (a follow-up).
```

- [x] **Step 4: Commit and push each spec on its own**

```bash
git add Docs/Specs/05-prediction-model.md && git commit -m "05: offline demo model and vision inference sections" && git push
git add Docs/Specs/08-demo-overlay.md && git commit -m "08: danger meter definition, video alignment, pff truth ticker" && git push
git add Docs/Specs/03-vision-pipeline.md && git commit -m "03: offline stage 8 fill for vision runs, rule-only for now" && git push
```

---

### Task 2: Danger meter, tried on the PFF pitch view

**Files:**
- Create: `demo/meter.py`, `tests/test_demo_meter.py`
- Modify: `demo/overlay.py` (append after `event_marker`), `demo/render.py`

**Interfaces:**
- Produces:
  - `meter.Meter(preds: pl.DataFrame, col: str)` with `.value(period: int, t: float) -> float | None`
  - `meter.level(p) -> float | None`, `meter.label(p) -> str`
  - `meter.draw(img, box, p) -> None`, `meter.METER_W = 96`
  - `overlay.meter_color(level) -> tuple`, `overlay.danger_meter(img, box, level, value, ticks, title)`

- [x] **Step 1: Write the failing tests** (`tests/test_demo_meter.py`)

```python
"""08 danger meter: the causal lookup, the log scale and the drawing."""

import numpy as np
import polars as pl
import pytest

cv2 = pytest.importorskip("cv2")

from demo import meter
from demo import overlay as ov


def preds(ts, ps, period=1):
    return pl.DataFrame({"period": [period] * len(ts), "t_s": ts, "p": ps})


def test_a_frame_between_rows_shows_the_earlier_one():
    m = meter.Meter(preds([0.0, 0.1, 0.2], [0.01, 0.2, 0.3]), "p")
    assert m.value(1, 0.05) == 0.01
    assert m.value(1, 0.0999) == 0.01  # just before a row: still the older one
    assert m.value(1, 0.1) == 0.2  # exactly on a row: that row (its frame is <= t)


def test_before_the_first_row_or_another_period_shows_nothing():
    m = meter.Meter(preds([1.0, 1.1], [0.1, 0.1]), "p")
    assert m.value(1, 0.95) is None
    assert m.value(2, 1.05) is None


def test_a_grid_gap_shows_nothing():
    m = meter.Meter(preds([0.0, 0.1, 0.4], [0.1, 0.2, 0.3]), "p")
    assert m.value(1, 0.2) == 0.2  # 0.1 s after the last row: still fine
    assert m.value(1, 0.25) is None  # rows 0.2 and 0.3 were skipped
    assert m.value(1, 0.4) == 0.3


def test_a_null_p_shows_nothing():
    m = meter.Meter(preds([0.0, 0.1], [0.1, None]), "p")
    assert m.value(1, 0.15) is None


def test_level_is_a_clamped_log_scale():
    assert meter.level(meter.P_MIN) == 0.0
    assert meter.level(meter.P_MAX) == 1.0
    assert meter.level(1e-5) == 0.0 and meter.level(0.9) == 1.0
    assert meter.level(None) is None and meter.level(float("nan")) is None
    lv = [meter.level(p) for p in (0.002, 0.01, 0.05, 0.2)]
    assert lv == sorted(lv)


def test_label():
    assert meter.label(None) == "--"
    assert meter.label(0.034) == "3.4%"
    assert meter.label(0.27) == "27%"


def test_the_bar_fills_to_its_level():
    box = (0, 0, meter.METER_W, 300)
    hi = np.zeros((300, meter.METER_W, 3), np.uint8)
    lo = hi.copy()
    meter.draw(hi, box, 0.3)
    meter.draw(lo, box, 0.003)
    x = 14 + 13  # middle of the bar
    y = 300 - 34 - 5  # just above the bar's bottom
    assert tuple(int(v) for v in hi[y, x]) == ov.meter_color(meter.level(0.3))
    y_mid = (40 + 300 - 34) // 2
    assert tuple(int(v) for v in hi[y_mid, x]) != (70, 70, 70)  # 0.3 fills past halfway
    assert tuple(int(v) for v in lo[y_mid, x]) == (70, 70, 70)


def test_no_value_draws_a_grey_bar():
    img = np.zeros((300, meter.METER_W, 3), np.uint8)
    meter.draw(img, (0, 0, meter.METER_W, 300), None)
    assert tuple(int(v) for v in img[300 - 34 - 5, 27]) == (70, 70, 70)


def test_meter_color_runs_green_amber_red():
    assert ov.meter_color(0.0) == ov.METER_COLORS[0]
    assert ov.meter_color(0.5) == ov.METER_COLORS[1]
    assert ov.meter_color(1.0) == ov.METER_COLORS[2]
```

- [x] **Step 2: Run them, expect failures**

Run: `.venv/Scripts/python -m pytest tests/test_demo_meter.py -q`
Expected: FAIL, `ImportError: cannot import name 'meter' from 'demo'`

- [x] **Step 3: Add the drawing to `demo/overlay.py`** (append at the end)

```python
METER_COLORS = ((80, 200, 80), (0, 190, 255), (40, 40, 230))  # BGR green, amber, red


def meter_color(level: float) -> tuple[int, int, int]:
    """Green at the bottom of the scale, amber halfway, red at the top."""
    if level < 0.5:
        lo, hi, f = METER_COLORS[0], METER_COLORS[1], 2 * level
    else:
        lo, hi, f = METER_COLORS[1], METER_COLORS[2], 2 * level - 1
    return tuple(round(a + (b - a) * f) for a, b in zip(lo, hi))


def danger_meter(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    level: float | None,
    value: str,
    ticks: list[tuple[float, str]],
    title: str,
) -> None:
    """08 danger meter in box (x, y, w, h): a vertical bar filled to `level` (0-1; None
    leaves it grey), the value over it, `ticks` [(level, text)] beside it, the title under."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BG, -1)
    bx, bw = x + 14, 26
    top, bottom = y + 40, y + h - 34
    cv2.rectangle(img, (bx, top), (bx + bw, bottom), (70, 70, 70), -1)
    if level is not None:
        fill = bottom - round(level * (bottom - top))
        cv2.rectangle(img, (bx, fill), (bx + bw, bottom), meter_color(level), -1)
    for lv, s in ticks:
        ty = bottom - round(lv * (bottom - top))
        cv2.line(img, (bx + bw, ty), (bx + bw + 6, ty), MUTED, 1)
        text(img, s, (bx + bw + 9, ty + 4), 0.38, MUTED)
    text(img, value, (x + w // 2, y + 26), 0.7, TEXT if level is not None else MUTED, 2, "center")
    text(img, title, (x + w // 2, y + h - 12), 0.38, MUTED, align="center")
```

- [x] **Step 4: Create `demo/meter.py`**

```python
"""08 danger meter: which P(goal) a frame shows, and where it sits on the bar."""

import math

import numpy as np
import polars as pl

from demo import overlay as ov

P_MIN, P_MAX = 0.001, 0.5  # log scale: 0.1% is an empty bar, 50% a full one
TICKS = (0.01, 0.05, 0.2)
TITLE = "goal in 5 s"
METER_W = 96
STALE_S = 0.1 + 1e-6  # one grid step: an older row means the grid skipped rows here
EPS_S = 1e-6  # float slack, so a frame exactly on a grid time sees that row


def level(p: float | None) -> float | None:
    """Bar height 0-1 on the log scale, None without a value."""
    if p is None or not math.isfinite(p):
        return None
    lo, hi = math.log10(P_MIN), math.log10(P_MAX)
    return min(max((math.log10(max(p, P_MIN)) - lo) / (hi - lo), 0.0), 1.0)


def label(p: float | None) -> str:
    if p is None:
        return "--"
    return f"{100 * p:.1f}%" if p < 0.1 else f"{round(100 * p)}%"


def draw(img: np.ndarray, box: tuple[int, int, int, int], p: float | None) -> None:
    ticks = [(level(t), f"{round(100 * t)}%") for t in TICKS]
    ov.danger_meter(img, box, level(p), label(p), ticks, TITLE)


class Meter:
    """Predictions (period, t_s, col) looked up causally: a frame at period time t shows
    the latest grid row at or before t, never a later or interpolated one (08)."""

    def __init__(self, preds: pl.DataFrame, col: str):
        self.rows = {}
        for (period,), g in preds.sort("period", "t_s").partition_by("period", as_dict=True).items():
            p = g[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
            self.rows[period] = (g["t_s"].to_numpy(), p)

    def value(self, period: int, t: float) -> float | None:
        if period not in self.rows:
            return None
        ts, p = self.rows[period]
        i = int(np.searchsorted(ts, t + EPS_S, side="right")) - 1
        if i < 0 or t - ts[i] > STALE_S:
            return None
        return None if np.isnan(p[i]) else float(p[i])
```

- [x] **Step 5: Run the tests, expect a pass**

Run: `.venv/Scripts/python -m pytest tests/test_demo_meter.py -q`
Expected: PASS (9 tests)

- [x] **Step 6: Commit and push**

```bash
git add demo/meter.py demo/overlay.py tests/test_demo_meter.py
git commit -m "demo: danger meter, causal lookup on the 10 Hz rows and a log scale bar" && git push
```

- [x] **Step 7: Meter column in the pitch view (`demo/render.py`)**

Make these changes:
- `from demo import meter as mt` at the imports.
- `Scene.__init__(self, match_dir, first, last, meter=None)`: store `self.meter = meter`. Change the size to `self.size = (PITCH.size[0] + (mt.METER_W if meter else 0), HEADER + PITCH.size[1] + FOOTER)`.
- In `draw`: change `img[HEADER : HEADER + PITCH.size[1]] = PITCH.blank()` to `img[HEADER : HEADER + PITCH.size[1], : PITCH.size[0]] = PITCH.blank()`. Before the possession panel, add:

```python
        if self.meter is not None:
            p = self.meter.value(f["period"], f["timestamp_s"])
            mt.draw(img, (PITCH.size[0], HEADER, mt.METER_W, PITCH.size[1]), p)
```

- In `main`:

```python
    ap.add_argument("--pgoal", type=Path, help="a CV run's pgoal.parquet: draws the danger meter")
    ...
    meter = None
    if args.pgoal:
        rows = pl.read_parquet(args.pgoal).filter(pl.col("match_id") == args.match)
        meter = mt.Meter(rows, "p_goal_cal_h5")
    scene = Scene(d, first, last, meter)
```

The pitch view's meter is checked by eye in Step 9. The lookup and the drawing are already covered by the tests above. Run the whole demo test set: `.venv/Scripts/python -m pytest tests/test_demo_meter.py tests/test_demo_overlay.py tests/test_demo_video.py -q`. Expected: PASS.

- [x] **Step 8: Commit and push**

```bash
git add demo/render.py tests/test_demo_meter.py
git commit -m "render: --pgoal draws the danger meter next to the pitch" && git push
```

- [ ] **Step 9: Look at both open-play goals of 10517 on PFF (no vision yet)**

In frame order, 10517's goals are: 1 = 44867 (period 1 penalty), 2 = 68200 (period 1, open play, t 2121.1), 3 = 162106 (period 2 penalty), 4 = 164932 (period 2, open play, t 2158.6).

```bash
.venv/Scripts/python -m demo.render --match 10517 --goal 4 --before 20 --after 5 --pgoal data/runs/lgbm-held-2026-09-27/pgoal.parquet --out C:/footage/renders/pff-10517-goal4.mp4
.venv/Scripts/python -m demo.render --match 10517 --goal 2 --before 20 --after 5 --pgoal data/runs/lgbm-held-2026-09-27/pgoal.parquet --out C:/footage/renders/pff-10517-goal2.mp4
```

Also print the numbers the review will use:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "
import polars as pl
p = pl.read_parquet('data/runs/lgbm-held-2026-09-27/pgoal.parquet').filter(pl.col('match_id') == '10517')
for per, t in ((1, 2121.12), (2, 2158.59)):
    w = p.filter(pl.col('period') == per, pl.col('t_s').is_between(t - 10, t)).select('t_s', 'p_h5', 'xg', 'p_goal_cal_h5')
    print(per, t); print(w.gather_every(5))
"
```

Watch both renders. Write down whether the meter visibly rises before each goal, and how far ahead. Pick the clip's goal:
- Default: goal 4, Mbappé 81' (period 2). The vb01 sync is for period 2 of the same broadcast.
- Take goal 2 if goal 4's meter doesn't rise and goal 2's does.

If the bar is unreadable (flat or pinned), change `P_MIN` / `P_MAX` once, update the 08 scale line, and commit both together ("meter: scale from the pff renders"). Keep the notes for the review in Task 8.

---

### Task 3: The demo goal model (fit, check, save, load)

**Files:**
- Create: `prediction/goal_model.py`, `tests/test_goal_model.py`

**Interfaces:**
- Consumes: `prediction.cv.load_data`, `of`, `MODELS`, `DATA_KEYS`, `KEYS`; `prediction.pgoal.fit_map`, `apply_map`, `scored`, `xg_model`; `converters.common.sha256`, `git_commit`.
- Produces:
  - `goal_model.train_ids(folds: dict, fold: int) -> list[str]`
  - `goal_model.fold_map(pgoal: pl.DataFrame, fold: int) -> tuple[float, float]`
  - `goal_model.GoalModel(model_dir: Path)` with `.manifest: dict` and `.predict(feats: pl.DataFrame) -> pl.DataFrame`. `predict` returns the columns `p_shot_h5, xg, p_goal_h5`, unmasked; nulls where there's no held ball.

- [ ] **Step 1: Write the failing tests** (`tests/test_goal_model.py`)

```python
"""05 "Offline demo model": the fold map, and GoalModel's P(shot) x xG -> map."""

import json

import numpy as np
import polars as pl
import pytest

lgb = pytest.importorskip("lightgbm")

from converters.common import sha256
from prediction import goal_model as gm
from prediction.features import FEATURES
from prediction.pgoal import apply_map, fit_map
from prediction.xg import XG_FEATURES


def pgoal_rows(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    p = rng.uniform(1e-4, 0.2, n)
    return pl.DataFrame(
        {
            "fold": rng.integers(0, 5, n),
            "label_mask_h5": np.ones(n, bool),
            "all_estimated": np.zeros(n, bool),
            "p_goal_h5": p,
            "label_goal_h5": rng.random(n) < p * 1.8,
        }
    )


def test_fold_map_is_the_other_folds_fit():
    rows = pgoal_rows()
    other = rows.filter(pl.col("fold") != 0)
    want = fit_map(other["p_goal_h5"].to_numpy(), other["label_goal_h5"].to_numpy().astype(float))
    assert gm.fold_map(rows, 0) == pytest.approx(want)


def test_fold_map_ignores_the_folds_own_labels():
    rows = pgoal_rows()
    flipped = rows.with_columns(
        label_goal_h5=pl.when(pl.col("fold") == 0).then(~pl.col("label_goal_h5")).otherwise("label_goal_h5")
    )
    assert gm.fold_map(rows, 0) == gm.fold_map(flipped, 0)


def test_train_ids_leave_the_fold_out_sorted():
    folds = {"matches": [{"match_id": i, "fold": f} for i, f in (("3812", 1), ("10517", 0), ("10502", 2))]}
    assert gm.train_ids(folds, 0) == ["10502", "3812"]


def write_model(tmp_path, ab=(0.1, 0.8)):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, len(FEATURES)))
    y = (x[:, 3] + rng.normal(size=400) > 1).astype(float)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 20},
        lgb.Dataset(x, y, feature_name=list(FEATURES)),
        num_boost_round=5,
    )
    xg_dir = tmp_path / "xg"
    xg_dir.mkdir()
    (xg_dir / "manifest.json").write_text(json.dumps({"model": "logistic", "features": list(XG_FEATURES)}))
    k = len(XG_FEATURES)
    (xg_dir / "model.json").write_text(json.dumps({"w": [-1.0] + [0.0] * k, "mean": [0.0] * k, "std": [1.0] * k}))
    d = tmp_path / "goal"
    d.mkdir()
    booster.save_model(str(d / "model.txt"))
    man = {
        "model_id": "goal-test",
        "features": list(FEATURES),
        "map": {"a": ab[0], "b": ab[1]},
        "xg": {"dir": str(xg_dir), "file": "model.json", "sha256": sha256(xg_dir / "model.json")},
    }
    (d / "manifest.json").write_text(json.dumps(man))
    return d, booster


def feats(n=50, seed=2):
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({f: rng.normal(size=n).astype(np.float32) for f in FEATURES})
    no_ball = np.arange(n) % 7 == 0
    return df.with_columns(ball_dist=pl.Series(np.where(no_ball, np.nan, np.abs(df["ball_dist"].to_numpy())), dtype=pl.Float32)), no_ball


def test_predict_is_pshot_times_xg_through_the_map(tmp_path):
    d, booster = write_model(tmp_path)
    df, no_ball = feats()
    out = gm.GoalModel(d).predict(df)
    p_shot = booster.predict(df.select(pl.col(f).cast(pl.Float32) for f in FEATURES).to_numpy())
    xg = 1 / (1 + np.exp(1.0))  # the logistic xG above: intercept -1, zero weights
    want = apply_map(p_shot * xg, (0.1, 0.8))
    got = out["p_goal_h5"].to_numpy()
    assert np.allclose(got[~no_ball], want[~no_ball])
    assert out["p_goal_h5"].is_null().to_numpy()[no_ball].all()
    assert out["xg"].is_null().to_numpy()[no_ball].all()
    assert np.allclose(out["p_shot_h5"].to_numpy(), p_shot)  # p_shot needs no ball


def test_a_changed_xg_file_is_refused(tmp_path):
    d, _ = write_model(tmp_path)
    (tmp_path / "xg" / "model.json").write_text(json.dumps({"w": [0.0] * 9, "mean": [0.0] * 8, "std": [1.0] * 8}))
    with pytest.raises(ValueError, match="xG"):
        gm.GoalModel(d)
```

- [ ] **Step 2: Run them, expect failures**

Run: `.venv/Scripts/python -m pytest tests/test_goal_model.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'prediction.goal_model'`

- [ ] **Step 3: Implement `prediction/goal_model.py`**

```python
"""The demo's goal model (05 "Offline demo model"): LightGBM P(shot) refit exactly like CV
outer fold k, xG v1, and fold k's cross-fitted P(goal) map. A clip from a match in fold k
is then scored by models that never saw that match.

    python -m prediction.goal_model --fold 0 --model-id goal-f0-2026-10-05

The fit has to reproduce the base run's fold k (best_iter, early-stopping matches, p on the
fold's rows within MAX_DIFF), otherwise nothing is saved.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from converters.common import git_commit, sha256
from prediction.cv import DATA_KEYS, KEYS, MODELS, load_data, of
from prediction.pgoal import apply_map, fit_map, scored, xg_model
from prediction.xg import OUT_DIR as XG_DIR

H = "h5"
BASE_RUN = Path("data/runs/lgbm-held-2026-09-27")
MODELS_DIR = Path("data/models/goal")
FOLDS_PATH = Path("data/splits/folds.json")
PROCESSED_DIR = Path("data/processed")
GAMESTATE_DIR = Path("data/gamestate")
MAX_DIFF = 1e-3  # another machine can move LightGBM's p this little, not more


def train_ids(folds: dict, fold: int) -> list[str]:
    """Matches outside outer fold `fold`, sorted as prediction.cv loads them."""
    return sorted(m["match_id"] for m in folds["matches"] if m["fold"] != fold)


def fold_map(pgoal: pl.DataFrame, fold: int) -> tuple[float, float]:
    """Fold k's cross-fitted P(goal) map: fitted on the other folds' scored rows only."""
    sc = pgoal.filter(scored(H), pl.col("fold") != fold)
    p = sc[f"p_goal_{H}"].to_numpy().astype(float)
    return fit_map(p, sc[f"label_goal_{H}"].to_numpy().astype(float))


def fit_shot_model(data: pl.DataFrame):
    """The CV's LightGBM, built the way run_cv's make() builds it."""
    cls, config = MODELS["lgbm"]
    kw = {k: v for k, v in config.items() if k not in DATA_KEYS}
    return cls(**kw, features=config["features"]).fit(data, H)


def reproduction(model, data: pl.DataFrame, fold: int, fold_ids: list[str], base_run: Path) -> dict:
    """How close the refit is to the base run's outer fold `fold`."""
    want = json.loads((base_run / "run.json").read_text())["folds"][H][str(fold)]
    base = pl.read_parquet(base_run / "predictions.parquet").select(*KEYS, base=f"p_{H}")
    rows = of(data, fold_ids)
    got = rows.select(KEYS).with_columns(p=model.predict(rows))
    j = got.join(base, on=KEYS, how="left", validate="1:1")
    both = j.drop_nulls(["p", "base"])
    out = {
        "best_iter": int(model.best_iter),
        "base_best_iter": int(want["best_iter"]),
        "es_matches_equal": list(model.es_matches) == list(want["es_matches"]),
        "rows": both.height,
        "null_mismatch": int((j["p"].is_null() != j["base"].is_null()).sum()),
        "max_abs_diff": float((both["p"] - both["base"]).abs().max()),
    }
    out["ok"] = (
        out["best_iter"] == out["base_best_iter"]
        and out["es_matches_equal"]
        and out["null_mismatch"] == 0
        and out["max_abs_diff"] <= MAX_DIFF
    )
    return out


def xg_file(xg_dir: Path) -> str:
    man = json.loads((xg_dir / "manifest.json").read_text())
    return "model.txt" if man["model"] == "lgbm" else "model.json"


class GoalModel:
    """A saved goal model: P(shot), xG and P(goal) for feature rows. Unmasked: the caller
    applies 05's mask. No held ball (ball_dist NaN): no xG and no P(goal)."""

    def __init__(self, model_dir: Path):
        import lightgbm as lgb

        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.features = self.manifest["features"]
        xg = self.manifest["xg"]
        if sha256(Path(xg["dir"]) / xg["file"]) != xg["sha256"]:
            raise ValueError(f"{xg['dir']}: the xG model changed since {model_dir.name} was saved")
        _, self.xg = xg_model(Path(xg["dir"]))
        self.ab = (self.manifest["map"]["a"], self.manifest["map"]["b"])

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        x = feats.select(pl.col(f).cast(pl.Float32) for f in self.features).to_numpy()
        p_shot = self.booster.predict(x)
        has = feats["ball_dist"].cast(pl.Float64).fill_null(np.nan).is_not_nan().to_numpy()
        xg = np.full(feats.height, np.nan)
        p_goal = np.full(feats.height, np.nan)
        if has.any():
            xg[has] = self.xg(feats.filter(pl.Series(has)))
            p_goal[has] = apply_map(p_shot[has] * xg[has], self.ab)
        return pl.DataFrame({"p_shot_h5": p_shot, "xg": xg, "p_goal_h5": p_goal}).fill_nan(None)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="prediction.goal_model")
    ap.add_argument("--fold", type=int, required=True, help="outer fold of the clip's match")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--base-run", type=Path, default=BASE_RUN)
    ap.add_argument("--xg", type=Path, default=XG_DIR)
    ap.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    out = args.out_dir / args.model_id
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --model-id")
    folds = json.loads(FOLDS_PATH.read_text())
    ids = sorted(m["match_id"] for m in folds["matches"])
    train = train_ids(folds, args.fold)
    fold_ids = [i for i in ids if i not in train]
    data, _ = load_data(ids, PROCESSED_DIR, GAMESTATE_DIR, [H])
    model = fit_shot_model(of(data, train))
    check = reproduction(model, data, args.fold, fold_ids, args.base_run)
    print(json.dumps(check, indent=2))
    if not check["ok"]:
        print("the refit doesn't reproduce the base run's fold; nothing saved", file=sys.stderr)
        return 1
    ab = fold_map(pl.read_parquet(args.base_run / "pgoal.parquet"), args.fold)
    out.mkdir(parents=True)
    model.booster.save_model(str(out / "model.txt"))
    xf = xg_file(args.xg)
    man = {
        "model_id": args.model_id,
        "fold": args.fold,
        "train_ids": train,
        "best_iter": int(model.best_iter),
        "es_matches": list(model.es_matches),
        "features": list(model.features),
        "features_version": MODELS["lgbm"][1]["features_version"],
        "ball_source": "held",
        "horizon": H,
        "map": {"a": ab[0], "b": ab[1], "fitted_on": f"pgoal.parquet rows of folds != {args.fold}"},
        "base_run": str(args.base_run),
        "reproduction": check,
        "xg": {"dir": str(args.xg), "file": xf, "sha256": sha256(args.xg / xf)},
        "git_commit": git_commit(),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"saved {out}: map a {ab[0]:.4f} b {ab[1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Check that `converters.common` exports `git_commit` and `sha256`. `vision/writer.py` and `vision/run.py` import them from there, so it should.

- [ ] **Step 4: Run the tests, expect a pass**

Run: `.venv/Scripts/python -m pytest tests/test_goal_model.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit and push**

```bash
git add prediction/goal_model.py tests/test_goal_model.py
git commit -m "goal model: refit cv outer fold, check it against the base run, save with its fold map" && git push
```

- [ ] **Step 6: Fit the fold-0 model on the workstation**

Run: `.venv/Scripts/python -m prediction.goal_model --fold 0 --model-id goal-f0-2026-10-05`

Expected:
- the reproduction JSON shows `best_iter 112` (base 112), `es_matches_equal true`, `null_mismatch 0`, `max_abs_diff` ≤ 1e-3;
- then `saved ... map a 0.1509 b 0.8413` (`pgoal.md`, h5 fold 0, to 4 places).

If it fails, stop and look before going further:
- A `best_iter` or `es_matches` mismatch means the data or config differs from the base run (feature cache or resample changes since 2026-09-27).
- A `max_abs_diff` just over the limit with everything else equal is the machine. Report it and ask before raising `MAX_DIFF`.

`data/` is gitignored, so there's nothing to commit here.

---

### Task 4: Offline stage 8 fill for vision runs

**Files:**
- Create: `vision/stage8.py`, `tests/vision_gs.py`, `tests/test_vision_stage8.py`

**Interfaces:**
- Consumes: `vision.state.infer`, `StateConfig`; `gamestate.validate.validate_match`.
- Produces:
  - `stage8.fill(match_dir: Path, config: StateConfig | None = None) -> dict`, which rewrites `frames.parquet` and returns the shares;
  - `tests/vision_gs.write_match(d: Path, n=60, fps=10.0, teams=True, possession=None, ball_state=None) -> Path`, also used by Task 5.

- [ ] **Step 1: Write the test helper** (`tests/vision_gs.py`)

```python
"""A small game state in vision's format (vision.writer): one period, the ball carried
toward +x by home player h1, an away player nearby, three more players. For stage 8 and
inference tests."""

from pathlib import Path

import polars as pl

from converters.common import causal_velocities
from gamestate.schema import SCHEMA_VERSION

FRAME_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "period": pl.Int64,
    "timestamp_s": pl.Float64,
    "home_attacks_positive_x": pl.Boolean,
    "ball_state": pl.String,
    "possession_team": pl.String,
    "ball_carrier_id": pl.String,
    "view_polygon": pl.List(pl.Float64),
    "set_play_phase": pl.Boolean,
}
OBJECT_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "object_id": pl.String,
    "object_type": pl.String,
    "team": pl.String,
    "player_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "visible": pl.Boolean,
    "interpolated": pl.Boolean,
    "confidence": pl.Float64,
}


def ball_x(t: float) -> float:
    return 5.0 + 4.0 * t  # 4 m/s toward the +x goal


def write_match(d: Path, n=60, fps=10.0, teams=True, possession=None, ball_state=None) -> Path:
    """possession / ball_state: None leaves them null (as vision.run writes them), a str
    fills every frame, a list gives one value per frame."""
    d.mkdir(parents=True, exist_ok=True)
    per = lambda v: v if isinstance(v, list) else [v] * n  # noqa: E731
    frames = pl.DataFrame(
        {
            "match_id": ["m"] * n,
            "frame_id": list(range(n)),
            "period": [1] * n,
            "timestamp_s": [f / fps for f in range(n)],
            "home_attacks_positive_x": [True] * n,
            "ball_state": per(ball_state),
            "possession_team": per(possession),
            "ball_carrier_id": [None] * n,
            "view_polygon": [None] * n,
            "set_play_phase": [None] * n,
        },
        schema=FRAME_SCHEMA,
    )
    rows = []
    for f in range(n):
        t = f / fps
        bx = ball_x(t)
        people = [
            ("h1", "home", bx - 0.5, 0.3),
            ("h2", "home", bx - 12.0, 10.0),
            ("a1", "away", bx + 6.0, -2.0),
            ("a2", "away", bx + 15.0, 4.0),
            ("k1", "away", 50.0, 0.0),
        ]
        for oid, team, x, y in people:
            rows.append(
                ("m", f, oid, "goalkeeper" if oid == "k1" else "player", team if teams else None,
                 None, x, y, None, True, False, 0.9)
            )
        rows.append(("m", f, "ball", "ball", None, None, bx, 0.0, None, True, False, 0.8))
    objects = pl.DataFrame(rows, schema=OBJECT_SCHEMA, orient="row")
    objects = causal_velocities(objects, frames, fps=fps)
    match = pl.DataFrame(
        {
            "match_id": ["m"],
            "schema_version": [SCHEMA_VERSION],
            "source": ["vision"],
            "competition": [None],
            "season": [None],
            "date": [None],
            "home_team": ["Home"],
            "away_team": ["Away"],
            "native_fps": [fps],
        },
        schema_overrides={"competition": pl.String, "season": pl.String, "date": pl.Date},
    )
    events = pl.DataFrame(
        schema={
            "match_id": pl.String,
            "frame_id": pl.Int64,
            "event_type": pl.String,
            "team": pl.String,
            "player_id": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "outcome": pl.String,
            "set_piece": pl.String,
            "set_play_phase": pl.Boolean,
        }
    )
    players = pl.DataFrame(
        schema={
            "match_id": pl.String,
            "player_id": pl.String,
            "team": pl.String,
            "jersey_number": pl.Int64,
            "position": pl.String,
            "name": pl.String,
        }
    )
    for name, df in (("match", match), ("frames", frames), ("objects", objects), ("events", events), ("players", players)):
        df.write_parquet(d / f"{name}.parquet")
    return d
```

Check the helper first: `.venv/Scripts/python -c "from pathlib import Path; import tempfile; from tests.vision_gs import write_match; from gamestate.validate import validate_match; d = write_match(Path(tempfile.mkdtemp()) / 'm'); print(validate_match(d))"` must print `[]`. If the validator objects to something (a column type, `view_polygon` null), match `vision/writer.py`'s `close()` exactly. That's the format being imitated.

- [ ] **Step 2: Write the failing tests** (`tests/test_vision_stage8.py`)

```python
"""03 "Offline stage 8 fill": stage 8's columns written into a vision run's frames."""

import json

import polars as pl
import pytest

from gamestate.validate import validate_match
from tests.vision_gs import write_match
from vision import stage8


def test_fill_finds_the_carriers_team_and_still_validates(tmp_path):
    d = write_match(tmp_path / "m")
    stats = stage8.fill(d)
    frames = pl.read_parquet(d / "frames.parquet")
    late = frames.filter(pl.col("timestamp_s") >= 1.0)  # past carrier_min_s
    assert (late["possession_team"] == "home").all()
    assert validate_match(d) == []
    assert stats["possession_set_share"] > 0.8


def test_fill_is_idempotent(tmp_path):
    d = write_match(tmp_path / "m")
    stage8.fill(d)
    once = pl.read_parquet(d / "frames.parquet")
    stage8.fill(d)
    assert pl.read_parquet(d / "frames.parquet").equals(once)


def test_no_teams_is_an_error(tmp_path):
    d = write_match(tmp_path / "m", teams=False)
    with pytest.raises(ValueError, match="home-cluster"):
        stage8.fill(d)


def test_main_records_the_fill_in_run_json(tmp_path):
    gs, cache = tmp_path / "gs", tmp_path / "cache"
    write_match(gs / "m")
    (cache / "m").mkdir(parents=True)
    (cache / "m" / "run.json").write_text(json.dumps({"video_start_s": 0.0}))
    stage8.main(["--match-id", "m", "--gamestate-dir", str(gs), "--cache-dir", str(cache)])
    run = json.loads((cache / "m" / "run.json").read_text())
    assert run["video_start_s"] == 0.0  # the run's own keys stay
    assert "state_config" in run["stage8"]
    assert 0 < run["stage8"]["possession_set_share"] <= 1
```

- [ ] **Step 3: Run them, expect failures**

Run: `.venv/Scripts/python -m pytest tests/test_vision_stage8.py -q`
Expected: FAIL, `ImportError: cannot import name 'stage8' from 'vision'`

- [ ] **Step 4: Implement `vision/stage8.py`**

```python
"""Stage 8 on a finished vision run (03 "Offline stage 8 fill"): ball_state,
possession_team and ball_carrier_id filled into its frames.parquet by vision.state.infer,
then the 02 validator again.

    python -m vision.stage8 --match-id demo01-arg-fra-81

vision.run writes the three as null, so run this after every vision.run of the match.
Rule-only with the default StateConfig: the learned v1h model can't read a match outside
its manifest yet (03).
"""

import argparse
import json
from pathlib import Path

import polars as pl

from gamestate.validate import validate_match
from vision.state import PLAYER_TYPES, StateConfig, infer

STATE_COLS = ("ball_state", "possession_team", "ball_carrier_id")


def fill(match_dir: Path, config: StateConfig | None = None) -> dict:
    frames = pl.read_parquet(match_dir / "frames.parquet")
    objects = pl.scan_parquet(match_dir / "objects.parquet")
    with_team = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("team").is_not_null())
        .select(pl.len())
        .collect()
        .item()
    )
    if not with_team:
        raise ValueError(
            f"{match_dir.name}: no player has a team, so possession would be null everywhere; "
            "rerun vision.run with --home-cluster"
        )
    state = infer(frames, objects, config)
    out = (
        frames.drop(*STATE_COLS)
        .join(state, on="frame_id", how="left", validate="1:1")
        .select(frames.columns)
    )
    out.write_parquet(match_dir / "frames.parquet")
    errors = validate_match(match_dir)
    if errors:
        raise ValueError(f"{match_dir.name}: 02 validation failed after stage 8: {errors}")
    bs = out["ball_state"]
    return {
        "frames": out.height,
        "possession_set_share": round(float(out["possession_team"].is_not_null().mean()), 4),
        "alive_share": round(float((bs == "alive").fill_null(False).mean()), 4),
        "dead_share": round(float((bs == "dead").fill_null(False).mean()), 4),
        "ball_state_null_share": round(float(bs.is_null().mean()), 4),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.stage8")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    args = ap.parse_args(argv)
    config = StateConfig()
    stats = fill(args.gamestate_dir / args.match_id, config)
    run_path = args.cache_dir / args.match_id / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        run["stage8"] = {"state_config": config.to_dict(), **stats}
        run_path.write_text(json.dumps(run, indent=2, default=str))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests, expect a pass**

Run: `.venv/Scripts/python -m pytest tests/test_vision_stage8.py tests/test_state.py -q`
Expected: PASS. If the first test fails on `possession_team == "home"`, print the frames: the helper's h1 has to be within `carrier_radius_m` (1.5 m) of the ball for `carrier_min_s` (0.3 s). Fix the helper, not stage 8.

- [ ] **Step 6: Commit and push**

```bash
git add vision/stage8.py tests/vision_gs.py tests/test_vision_stage8.py
git commit -m "stage 8 fill for finished vision runs, refuses runs without teams" && git push
```

---

### Task 5: Vision inference (`prediction/infer.py`)

**Files:**
- Create: `prediction/infer.py`, `tests/test_infer.py`

**Interfaces:**
- Consumes: `prediction.resample.resample_match`; `prediction.features.match_features`; a model with `.manifest["model_id"]` and `.predict(feats) -> pl.DataFrame[p_shot_h5, xg, p_goal_h5]` (`GoalModel` from Task 3).
- Produces:
  - `infer.inference_mask() -> pl.Expr`
  - `infer.predict_match(match_dir: Path, model) -> pl.DataFrame` (columns in 05 "Vision inference")
  - CLI output `data/predictions/<match_id>/<model_id>.parquet` + `.json`

- [ ] **Step 1: Write the failing tests** (`tests/test_infer.py`)

```python
"""05 "Vision inference": the mask, the stage 8 guard, causality and the output files."""

import json

import numpy as np
import polars as pl
import pytest

from prediction import infer
from tests.vision_gs import write_match


class FakeModel:
    manifest = {"model_id": "fake"}

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        bx = feats["ball_x"].cast(pl.Float64).fill_nan(0.0).fill_null(0.0).to_numpy()
        p = 1 / (1 + np.exp(-(bx - 20) / 5))
        return pl.DataFrame({"p_shot_h5": p, "xg": np.full(len(p), 0.1), "p_goal_h5": p * 0.1})


def test_null_ball_state_is_predicted_dead_is_not(tmp_path):
    n = 60
    states = [None] * 20 + ["dead"] * 10 + ["alive"] * 30
    poss = ["home"] * 50 + [None] * 10
    d = write_match(tmp_path / "m", n=n, possession=poss, ball_state=states)
    out = infer.predict_match(d, FakeModel()).sort("frame_id")
    by = dict(zip(out["frame_id"].to_list(), out["predicted"].to_list()))
    assert all(by[f] for f in range(1, 20))  # null ball state, possession set: predicted
    assert not any(by[f] for f in range(20, 30))  # dead
    assert all(by[f] for f in range(30, 50))
    assert not any(by[f] for f in range(50, 60))  # no possession
    assert out.filter(~pl.col("predicted"))["p_goal_h5"].is_null().all()
    assert out.filter(pl.col("predicted"))["p_goal_h5"].is_not_null().all()


def test_no_possession_anywhere_means_stage_8_wasnt_run(tmp_path):
    d = write_match(tmp_path / "m")
    with pytest.raises(ValueError, match="stage8"):
        infer.predict_match(d, FakeModel())


def test_later_frames_dont_change_earlier_predictions(tmp_path):
    a = write_match(tmp_path / "a", possession="home", ball_state="alive")
    b = write_match(tmp_path / "b", possession="home", ball_state="alive")
    objs = pl.read_parquet(b / "objects.parquet")
    objs.with_columns(
        x=pl.when(pl.col("frame_id") > 30).then(pl.col("x") + 20.0).otherwise("x")
    ).write_parquet(b / "objects.parquet")
    pa = infer.predict_match(a, FakeModel()).filter(pl.col("frame_id") <= 30)
    pb = infer.predict_match(b, FakeModel()).filter(pl.col("frame_id") <= 30)
    assert pa.drop("match_id").equals(pb.drop("match_id"))


def test_main_writes_predictions_and_shares(tmp_path, monkeypatch):
    write_match(tmp_path / "gs" / "m", possession="home", ball_state=None)
    monkeypatch.setattr(infer, "GoalModel", lambda _: FakeModel())
    infer.main(["--match-id", "m", "--model", "unused", "--gamestate-dir", str(tmp_path / "gs"), "--out-dir", str(tmp_path / "pred")])
    out = pl.read_parquet(tmp_path / "pred" / "m" / "fake.parquet")
    side = json.loads((tmp_path / "pred" / "m" / "fake.json").read_text())
    assert out.columns == infer.COLUMNS
    assert side["model_id"] == "fake"
    assert side["ball_state_null_share"] == 1.0
    assert 0.9 < side["predicted_share"] <= 1.0
```

- [ ] **Step 2: Run them, expect failures**

Run: `.venv/Scripts/python -m pytest tests/test_infer.py -q`
Expected: FAIL, `ImportError: cannot import name 'infer' from 'prediction'`

- [ ] **Step 3: Implement `prediction/infer.py`**

```python
"""P(goal) on a vision run's game state (05 "Vision inference"): resampled to 10 Hz in
memory, v1 features on stage 8's possession, the demo goal model, and 05's Inference mask
(not the training `eligible`: stage 8's ball state is often null, and null isn't dead).

    python -m prediction.infer --match-id demo01-arg-fra-81 --model data/models/goal/goal-f0-2026-10-05

Reads the 02 game state only. Run vision.stage8 first.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from converters.common import sha256
from prediction.features import match_features
from prediction.goal_model import GoalModel
from prediction.resample import resample_match

PRED_DIR = Path("data/predictions")
P_COLS = ["p_shot_h5", "xg", "p_goal_h5"]
COLUMNS = [
    "match_id", "period", "t_s", "frame_id", "timestamp_s",
    "possession_team", "ball_state", "predicted", *P_COLS,
]


def inference_mask() -> pl.Expr:
    """05 Inference: a team in possession, the ball not dead (null counts as not dead, so
    vision gaps don't blank the meter), and some player visible."""
    return (
        pl.col("possession_team").is_not_null()
        & (pl.col("ball_state") != "dead").fill_null(True)
        & ~pl.col("all_estimated")
    )


def predict_match(match_dir: Path, model) -> pl.DataFrame:
    frames = pl.read_parquet(match_dir / "frames.parquet")
    if frames["possession_team"].is_null().all():
        raise ValueError(f"{match_dir.name}: no possession anywhere; run vision.stage8 first")
    fps = pl.read_parquet(match_dir / "match.parquet")["native_fps"].item()
    frames10, objects10, _ = resample_match(
        frames,
        pl.read_parquet(match_dir / "objects.parquet"),
        pl.read_parquet(match_dir / "events.parquet"),
        match_dir.name,
        fps,
    )
    frames10 = frames10.sort("period", "t_s")
    feats = match_features(
        frames10.select("period", "t_s", "possession_team", "flipped"), objects10.lazy(), "held", "1"
    )
    if not feats.select("period", "t_s").equals(frames10.select("period", "t_s")):
        raise ValueError(f"{match_dir.name}: feature rows don't line up with the grid")
    p = model.predict(feats)
    out = frames10.select(
        "match_id", "period", "t_s", "frame_id", "timestamp_s", "possession_team", "ball_state",
        predicted=inference_mask(),
    ).hstack(p)
    return out.with_columns(
        pl.when(pl.col("predicted")).then(pl.col(c)).otherwise(None).alias(c) for c in P_COLS
    ).select(COLUMNS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="prediction.infer")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--model", type=Path, required=True, help="data/models/goal/<model_id>")
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--out-dir", type=Path, default=PRED_DIR)
    args = ap.parse_args(argv)
    model = GoalModel(args.model)
    d = args.gamestate_dir / args.match_id
    out = predict_match(d, model)
    model_id = model.manifest["model_id"]
    dst = args.out_dir / args.match_id
    dst.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dst / f"{model_id}.parquet")
    side = {
        "model_id": model_id,
        "match_id": args.match_id,
        "frames_sha256": sha256(d / "frames.parquet"),
        "objects_sha256": sha256(d / "objects.parquet"),
        "grid_rows": out.height,
        "predicted_share": round(float(out["predicted"].mean()), 4),
        "ball_state_null_share": round(float(out["ball_state"].is_null().mean()), 4),
        "no_possession_share": round(float(out["possession_team"].is_null().mean()), 4),
    }
    (dst / f"{model_id}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()
```

Notes for the implementer:
- `resample_match` takes `events` with zero rows. If its label code fails on that, fix the empty case in `prediction/resample.py` with its own test in `tests/test_resample.py`. Don't build a fake event.
- `test_main_writes_predictions_and_shares` monkeypatches `infer.GoalModel`. That's why `main` calls it through the module global.

- [ ] **Step 4: Run the tests, expect a pass**

Run: `.venv/Scripts/python -m pytest tests/test_infer.py tests/test_resample.py tests/test_features.py -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add prediction/infer.py tests/test_infer.py
git commit -m "infer: p(goal) on a vision run, with 05's inference mask instead of eligible" && git push
```

---

### Task 6: `demo.video`: start offset, danger meter, PFF truth ticker

**Files:**
- Modify: `demo/video.py`, `tests/test_demo_video.py`

**Interfaces:**
- Consumes: `demo.meter.Meter`, `meter.draw`, `meter.METER_W`; `demo.tally.ticker`, `event_text`; `demo.overlay.ticker_strip`; `vision.bench.load_manifest`, `sync_offset`.
- Produces:
  - `video.start_frame(cache: Path, fps: float) -> int`
  - `video.skip_to(cap, n: int) -> None`
  - `video.truth_events(clip: dict, gamestate_dir: Path, video_start_s: float, fps: float) -> pl.DataFrame`
  - the CLI flags `--predictions PATH`, `--truth-clip ID`, `--clip-manifest PATH`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_demo_video.py`)

```python
import json  # at the top with the other imports
from demo import meter as mt  # likewise


def shaded_video(path, n):
    """Frame i is a flat gray of 20 + 10 i, so a frame's index can be read back."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_PX))
    for i in range(n):
        w.write(np.full((H_PX, W, 3), 20 + 10 * i, np.uint8))
    w.release()


def test_frames_line_up_with_the_runs_start(tmp_path):
    _, cache, gs = write_run(tmp_path, n_video=20, processed=10)
    (cache / "run.json").write_text(json.dumps({"video_start_s": 3 / FPS}))
    shaded = tmp_path / "shaded.mp4"
    shaded_video(shaded, 20)
    out = tmp_path / "out.mp4"
    video.main(["--video", str(shaded), "--match-id", "m", "--cache-dir", str(cache.parent),
                "--gamestate-dir", str(gs.parent), "--out", str(out)])
    cap = cv2.VideoCapture(str(out))
    ok, first = cap.read()
    assert ok
    assert abs(float(first[5:40, 5:40].mean()) - (20 + 10 * 3)) < 4  # source frame 3, not 0


def test_start_frame_without_run_json_is_zero(tmp_path):
    assert video.start_frame(tmp_path, 30.0) == 0


def test_truth_events_map_pff_time_to_run_frames(tmp_path):
    d = tmp_path / "gs" / "pff1"
    d.mkdir(parents=True)
    pl.DataFrame(
        {"frame_id": [1, 2, 3], "period": [2, 2, 1], "timestamp_s": [105.0, 101.0, 105.0]}
    ).write_parquet(d / "frames.parquet")
    pl.DataFrame(
        {
            "match_id": ["pff1"] * 3,
            "frame_id": [1, 2, 3],
            "event_type": ["goal", "shot", "goal"],
            "team": ["away", "away", "home"],
            "player_id": [None] * 3,
            "x": [0.0] * 3,
            "y": [0.0] * 3,
            "outcome": ["goal", "saved", "goal"],
            "set_piece": ["open_play"] * 3,
            "set_play_phase": [False] * 3,
        }
    ).write_parquet(d / "events.parquet")
    clip = {"match_id": "pff1", "period": 2, "sync": [{"video_s": 0.0, "timestamp_s": 100.0}]}
    ev = video.truth_events(clip, tmp_path / "gs", video_start_s=2.0, fps=FPS)
    # goal at PFF 105 s -> video 5 s -> run 3 s -> frame 30; the shot at 101 s is before
    # the run starts (frame -10) and is dropped, the period-1 goal too
    assert ev["frame_id"].to_list() == [30]
    assert ev["event_type"].to_list() == ["goal"]


def test_main_draws_the_meter_from_predictions(tmp_path):
    path, cache, gs = write_run(tmp_path, n_video=20, processed=15)
    pl.DataFrame(
        {"frame_id": list(range(15)), "period": [1] * 15, "timestamp_s": [f / FPS for f in range(15)]}
    ).write_parquet(gs / "frames.parquet")
    preds = tmp_path / "p.parquet"
    pl.DataFrame({"period": [1] * 15, "t_s": [f / FPS for f in range(15)], "p_goal_h5": [0.3] * 15}).write_parquet(preds)
    out = tmp_path / "out.mp4"
    video.main(["--video", str(path), "--match-id", "m", "--cache-dir", str(cache.parent),
                "--gamestate-dir", str(gs.parent), "--predictions", str(preds), "--out", str(out)])
    ok, img = cv2.VideoCapture(str(out)).read()
    assert ok
    x = W - 16 - mt.METER_W + 14 + 13  # middle of the meter's bar
    y = 80 + min(H_PX - 160, 420) - 34 - 5  # just above the bar's bottom
    want = np.array(ov.meter_color(mt.level(0.3)))
    assert np.abs(img[y, x].astype(int) - want).max() < 40  # mp4v blurs colors a little
```

- [ ] **Step 2: Run them, expect failures**

Run: `.venv/Scripts/python -m pytest tests/test_demo_video.py -q`
Expected: the four new tests FAIL (`start_frame` / `truth_events` missing, `--predictions` unknown, frame 0 drawn instead of frame 3). The old tests pass.

- [ ] **Step 3: Implement in `demo/video.py`**

Add the imports `import json`, `from demo import meter as mt`, `from demo import tally` and `from vision.bench import load_manifest, sync_offset`. Then add these functions above `main`:

```python
TICKER_H = 50
METER_TOP = 80


def start_frame(cache: Path, fps: float) -> int:
    """Source video frame of the run's frame_id 0: vision.run's --start-s, from run.json
    (0 for a run without one)."""
    run = cache / "run.json"
    if not run.exists():
        return 0
    return round(json.loads(run.read_text()).get("video_start_s", 0.0) * fps)


def skip_to(cap, n: int) -> None:
    """Frame-exact skip by grab(), as vision.run does: seeking can land on a keyframe."""
    for _ in range(n):
        if not cap.grab():
            raise SystemExit(f"the video ends before frame {n}")


def truth_events(clip: dict, gamestate_dir: Path, video_start_s: float, fps: float) -> pl.DataFrame:
    """The clip's PFF shots and goals as events on the run's frame_ids (08 "PFF truth
    ticker"): PFF period time -> source video time (the clip's sync offset) -> run frame.
    Display only."""
    offset = sync_offset(clip)
    d = gamestate_dir / clip["match_id"]
    times = pl.read_parquet(d / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    return (
        pl.read_parquet(d / "events.parquet")
        .filter(pl.col("event_type").is_in(["shot", "goal"]))
        .join(times, on="frame_id")
        .filter(pl.col("period") == clip["period"])
        .with_columns(
            frame_id=((pl.col("timestamp_s") - offset - video_start_s) * fps).round().cast(pl.Int64)
        )
        .filter(pl.col("frame_id") >= 0)
        .drop("period", "timestamp_s")
        .sort("frame_id")
    )
```

In `main`:
- Add the flags:

```python
    ap.add_argument("--predictions", type=Path, help="prediction.infer output: draws the danger meter")
    ap.add_argument("--truth-clip", help="clip_id in --clip-manifest: PFF shots and goals in the ticker")
    ap.add_argument("--clip-manifest", type=Path, default=Path("data/splits/demo_clips.json"))
```

- Replace `cap.set(cv2.CAP_PROP_POS_FRAMES, first)` with `skip_to(cap, start_frame(args.cache_dir / args.match_id, fps) + first)`.
- Before the loop:

```python
    gs = args.gamestate_dir / args.match_id
    meter, when = None, {}
    if args.predictions:
        meter = mt.Meter(pl.read_parquet(args.predictions), "p_goal_h5")
        times = pl.read_parquet(gs / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
        when = {r[0]: (r[1], r[2]) for r in times.iter_rows()}
    events, names = None, {}
    if args.truth_clip:
        clips = {c["clip_id"]: c for c in load_manifest(args.clip_manifest)}
        clip = clips[args.truth_clip]
        events = truth_events(clip, args.gamestate_dir, start_frame(args.cache_dir / args.match_id, fps) / fps, fps)
        m = pl.read_parquet(args.gamestate_dir / clip["match_id"] / "match.parquet").row(0, named=True)
        names = {"home": m["home_team"], "away": m["away_team"]}
```

- In the loop, replace `writer.write(bv.draw(image, frame_id))` with:

```python
        img = bv.draw(image, frame_id)
        h, w = img.shape[:2]
        if meter is not None:
            p = meter.value(*when[frame_id]) if frame_id in when else None
            mt.draw(img, (w - 16 - mt.METER_W, METER_TOP, mt.METER_W, min(h - 160, 420)), p)
        if events is not None:
            lines = [(f"{tally.event_text(e, names)} (PFF)", e["event_type"] == "goal")
                     for e in tally.ticker(events, frame_id, fps)]
            ov.ticker_strip(img, (0, h - TICKER_H, w, TICKER_H), lines, False)
        writer.write(img)
```

Update the module docstring's usage line to show the new flags and the alignment rule.

- [ ] **Step 4: Run the tests, expect a pass**

Run: `.venv/Scripts/python -m pytest tests/test_demo_video.py tests/test_demo_meter.py -q`
Expected: PASS. If the meter-color test is flaky because mp4v blurs colors, sample the bar's middle (`y = 80 + (min(H_PX-160,420)) // 2 + 20`). Keep the 40 tolerance. Don't drop the assertion.

- [ ] **Step 5: Commit and push**

```bash
git add demo/video.py tests/test_demo_video.py
git commit -m "demo.video: skip to the run's start frame, danger meter, pff truth ticker" && git push
```

---

### Task 7: The clip: footage, sync, vision run, checks (manual steps for the user)

The goal is from Task 2 Step 9 (default: goal 4, Mbappé, period 2 at PFF t 2158.59 s). The numbers below are for that goal.

**Files:**
- Create: `data/splits/demo_clips.json` (tracked, like `vision_benchmark.json`)
- Local only: `C:/footage/wc2022/13_argentina-vs-france-goal81.mp4`, an entry in `data/vision_bench/videos.json`

- [ ] **Step 1: Download the window (the user runs this)**

The 01 clip came from YouTube `RgqKdplLIk4` starting at 4465 s, and vb01's sync puts PFF period-2 time at video time + 1167.48 s. So PFF period-2 t ≈ YouTube s − 3297.5, and the goal at 2158.6 s is near YouTube 5456 s. Download 5366–5476 s (110 s). That's a 30 s pre-roll for the gate and the team warmup, then 60 s before the goal and 20 s after it.

Use the flags in the first lines of `C:/footage/wc2022/01_argentina-vs-france.log` (format `616+251`, re-encoded to H.264), with the section `*5366-5476`:

```bash
yt-dlp -f 616+251 --download-sections "*5366-5476" --force-keyframes-at-cuts --recode-video mp4 -o "C:/footage/wc2022/13_argentina-vs-france-goal81.%(ext)s" https://www.youtube.com/watch?v=RgqKdplLIk4
```

Check it's 1080p at 30 fps (`ffprobe`), and that the goal is near 90 s into the file. If the broadcast upload has edits and the goal isn't there, widen the window. Don't guess the offset.

- [ ] **Step 2: Two sync pairs inside the clip**

Estimated offset for this file: PFF t ≈ video_s + 2068.5. List the kicks around it:

```bash
PYTHONPATH=. .venv/Scripts/python scripts/kick_times.py --match-id 10517 --period 2 --from-s 2095 --to-s 2175 --offset 2068.5
```

Open the clip and find two kicks the list predicts: one early (around 10–30 s into the file) and the strike before the goal. Record each as `{"video_s": ..., "timestamp_s": ...}`. The two offsets must agree within 0.1 s (the bench's drift check). Otherwise mark them again.

- [ ] **Step 3: Marks and manifest**

Watch 28–105 s and mark every replay, close-up or other non-live stretch, as in vb01. The celebration after the goal will be one. Then write `data/splits/demo_clips.json`:

```json
{
 "version": 1,
 "clips": [
  {
   "clip_id": "demo01-arg-fra-81",
   "match_id": "10517",
   "period": 2,
   "video_sha256": "<sha256 of the file>",
   "video_start_s": 28.0,
   "video_end_s": 105.0,
   "home_attacks_tv_right_p1": true,
   "home_cluster": null,
   "sync_method": "kicks from scripts/kick_times.py, two pairs, offsets <a> and <b>",
   "sync": [{"video_s": 0.0, "timestamp_s": 0.0}, {"video_s": 0.0, "timestamp_s": 0.0}],
   "marks": [],
   "marks_checked_by_user": true
  }
 ]
}
```

Fill in the real sha256 (`sha256sum`), the sync pairs and the marks. `video_start_s` is 28, not 30: the bench allows at most `MAX_PREROLL_S` = 30 s of unscored pre-roll, so this stays clear of that limit. `home_attacks_tv_right_p1: true` is vb01's value for the same broadcast; Step 5 checks it. Add `"demo01-arg-fra-81": "C:/footage/wc2022/13_argentina-vs-france-goal81.mp4"` to `data/vision_bench/videos.json` (local, gitignored). Keep the clip out of `vision_benchmark.json`: adding it would shift the pooled bench numbers.

- [ ] **Step 4: Vision run (no home cluster yet)**

```bash
.venv/Scripts/python -m vision.run --video C:/footage/wc2022/13_argentina-vs-france-goal81.mp4 --weights-dir ../sports/examples/soccer/data --pnl-weights-dir C:/Users/superCookie/Desktop/PnLCalib --match-id demo01-arg-fra-81 --home Argentina --away France --period 2 --max-frames 3150
```

Expect about 3 fps on the 2060, so roughly 17 minutes. 3150 frames is 105 s at 30 fps.

- [ ] **Step 5: Bench check against PFF (direction, sync, teams)**

```bash
.venv/Scripts/python -m vision.bench --manifest data/splits/demo_clips.json --offset-check
```

Pass looks like:
- within 2 m in the range of vb01's 86% (anything under about 60% needs a look);
- best offset within ±0.1 s of the sync's;
- player team accuracy above 90%, with "home cluster picked by agreement". That cluster is the `--home-cluster` value.

If within 2 m is near zero, the direction is wrong. Set `home_attacks_tv_right_p1` to false, rerun Step 4 with `--home-attacks-left` and check again. If the offset is off, redo Step 2.

- [ ] **Step 6: Rerun with the home cluster, check again, then fill stage 8**

```bash
.venv/Scripts/python -m vision.run ... (as Step 4) --home-cluster <N>
.venv/Scripts/python -m vision.bench --manifest data/splits/demo_clips.json --offset-check
.venv/Scripts/python -m vision.stage8 --match-id demo01-arg-fra-81
```

The bench numbers should be the same as in Step 5, with the cluster now set rather than picked. From stage 8, write down `possession_set_share` and `ball_state_null_share` for the review.

- [ ] **Step 7: Commit and push the manifest**

```bash
git add data/splits/demo_clips.json
git commit -m "demo clip manifest: arg-fra 81st minute goal, sync and marks" && git push
```

---

### Task 8: Run it end to end, compare with PFF tracking, write it up

**Files:**
- Create: `scripts/demo_compare.py`, `Docs/reviews/offline-demo-<date>.md`
- Modify: `Docs/Specs/roadmap.md`

- [ ] **Step 1: Predict and render**

```bash
.venv/Scripts/python -m prediction.infer --match-id demo01-arg-fra-81 --model data/models/goal/goal-f0-2026-10-05
.venv/Scripts/python -m demo.video --video C:/footage/wc2022/13_argentina-vs-france-goal81.mp4 --match-id demo01-arg-fra-81 --predictions data/predictions/demo01-arg-fra-81/goal-f0-2026-10-05.parquet --truth-clip demo01-arg-fra-81 --out C:/footage/renders/demo01-arg-fra-81.mp4
```

Write down `predicted_share` and the null shares from infer. Watch the render all the way through.

- [ ] **Step 2: Comparison script** (`scripts/demo_compare.py`, analysis only, like the other scripts)

```python
"""Vision vs PFF-tracking P(goal) on a demo clip (roadmap Phase 3, "compare predictor
accuracy on vision vs dataset tracking", one clip).

    PYTHONPATH=. python scripts/demo_compare.py --clip demo01-arg-fra-81 --model-id goal-f0-2026-10-05

Vision's grid time t (period time from the run's start) is PFF period time
t + video_start_s(run.json) + the clip's sync offset. The PFF side is the base run's
out-of-fold p_goal_cal_h5, the same fold-0 model and map as the demo model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from vision.bench import load_manifest, sync_offset

BASE = Path("data/runs/lgbm-held-2026-09-27/pgoal.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/splits/demo_clips.json"))
    args = ap.parse_args()
    clip = {c["clip_id"]: c for c in load_manifest(args.manifest)}[args.clip]
    start = json.loads(Path(f"data/vision_cache/{args.clip}/run.json").read_text())["video_start_s"]
    shift = start + sync_offset(clip)
    vis = (
        pl.read_parquet(f"data/predictions/{args.clip}/{args.model_id}.parquet")
        .with_columns(k=((pl.col("t_s") + shift) * 10).round().cast(pl.Int64))
        .select("k", "predicted", vis_poss="possession_team", vis_p="p_goal_h5")
    )
    pff = pl.read_parquet(BASE).filter(
        pl.col("match_id") == clip["match_id"], pl.col("period") == clip["period"]
    ).select("k", pff_poss="possession_team", pff_p="p_goal_cal_h5")
    j = vis.join(pff, on="k", how="inner").sort("k")
    goals = pl.read_parquet(f"data/gamestate/{clip['match_id']}/events.parquet").filter(pl.col("event_type") == "goal")
    times = pl.read_parquet(f"data/gamestate/{clip['match_id']}/frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    goal_t = goals.join(times, on="frame_id").filter(pl.col("period") == clip["period"])["timestamp_s"]
    both = j.drop_nulls(["vis_p", "pff_p"])
    print(f"rows {j.height}, both predicted {both.height}")
    print(f"possession agreement {(j['vis_poss'] == j['pff_poss']).mean():.3f}")
    if both.height > 1:
        print(f"correlation of log p: {np.corrcoef(np.log(both['vis_p']), np.log(both['pff_p']))[0, 1]:.3f}")
    for g in goal_t:
        print(f"goal at {g:.1f} s")
        for before in (5, 3, 2, 1, 0.5):
            k = round((g - before) * 10)
            r = j.filter(pl.col("k") == k)
            v = r["vis_p"].to_list()[0] if r.height else None
            p = r["pff_p"].to_list()[0] if r.height else None
            print(f"  {before:>4} s before: vision {v}, pff {p}")
    j.with_columns(t=pl.col("k") / 10).select("t", "vis_poss", "pff_poss", "vis_p", "pff_p").gather_every(10).write_csv(
        f"data/predictions/{args.clip}/compare_1s.csv"
    )


if __name__ == "__main__":
    main()
```

Run: `PYTHONPATH=. .venv/Scripts/python scripts/demo_compare.py --clip demo01-arg-fra-81 --model-id goal-f0-2026-10-05`

- [ ] **Step 3: Commit and push the script**

```bash
git add scripts/demo_compare.py
git commit -m "script: vision vs pff tracking p(goal) on the demo clip" && git push
```

- [ ] **Step 4: Write `Docs/reviews/offline-demo-<date>.md`**

Write it in the style of the existing reviews: numbers first, short sections. Include:
- **Footage:** file, source, window, private only (08).
- **What ran:** commit, model id and its reproduction check, map, bench numbers, stage 8 shares, `predicted_share`.
- **The answer to 08's acceptance:** does the meter visibly rise before the goal, how far ahead, and its value at 5/3/2/1/0.5 s next to PFF tracking's.
- **Where vision and PFF disagree and why:** possession agreement, rows with no meter, ball gaps, cut-aways.
- **What to fix first, judged from this clip:** ball, possession (v1h on vision), lead time, direction/teams.
- **The Task 2 PFF pitch-view notes:** both goals.

- [ ] **Step 5: Roadmap**

In `Docs/Specs/roadmap.md`, Phase 3:
- Tick "Feed vision game state into trained predictor", with a pointer to the review.
- Under "Compare predictor accuracy on vision vs. dataset tracking", note that one clip is done (keep it open for more clips).
- Add a ticked sub-item under "Offline overlay renderer" for the danger meter.

Phase 2: tick "Plug in stage 8" as "offline fill done (rule-only), live still open". Keep it unticked if you'd rather tick only the live wiring, and say so in the line.

- [ ] **Step 6: Commit and push**

```bash
git add Docs/reviews/offline-demo-*.md && git commit -m "review: first offline demo clip, arg-fra goal with the meter" && git push
git add Docs/Specs/roadmap.md && git commit -m "roadmap: offline demo done, what's next from the review" && git push
```

---

## Follow-ups (not in this plan)

- **v1h possession on vision runs:** an entry path in `vision/possession_model.py` for a match outside the manifest (an explicit fit id, plus the native/grid hash checks on the run itself). It replaces the rule-only fill (03's decision).
- **Degraded training for the vision-facing model:** a CV run with `--degrade-arm both` at the target levels, then its own P(goal) map, then a new demo model.
- **Ball build** (`Docs/Specs/10-ball.md` §4, steps 1–5).
- **All-64 fit for live:** live mode, the small ball model, self-recorded footage for the public demo (08).

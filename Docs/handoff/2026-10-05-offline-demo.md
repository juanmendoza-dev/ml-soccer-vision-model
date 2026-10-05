Build the offline end-to-end demo by following `Docs/plans/2026-10-05-offline-demo.md`, task by task. Implement it yourself in this session (Native execution, using the superpowers:executing-plans skill). Don't spawn subagents per task. At the very end, one fresh reviewer on the most capable model reviews the whole branch.

## Read first, in this order
1. `CLAUDE.md` (project rules: no leakage, meters not pixels, `vision/` and `prediction/` share only the schema, specs before code).
2. `Docs/plans/2026-10-05-offline-demo.md`: the whole plan, including Global Constraints and Review Focus. It has the code, the tests, the commands and the commit messages for every step.
3. Skim the specs the plan touches: `Docs/Specs/05-prediction-model.md` (Inference, xG v1 build, Recalibration), `Docs/Specs/08-demo-overlay.md` (Element definitions), `Docs/Specs/03-vision-pipeline.md` (stage 8), `Docs/Specs/02-game-state-schema.md`.
4. Read every existing file before you change it: `demo/video.py`, `demo/render.py`, `demo/overlay.py`, `prediction/cv.py` (`load_data`, `run_cv`, `MODELS`), `prediction/pgoal.py`, `prediction/resample.py` (`resample_match`, `add_frame_flags`), `prediction/features.py` (`match_features`), `vision/state.py` (`infer`), `vision/writer.py` (`close`), `vision/bench.py` (`load_manifest`, `sync_offset`).

## What we're building
One private demo clip: the ARG–FRA World Cup final (PFF match 10517), the open-play goal in period 2 at PFF t 2158.59 s (Mbappé, 81').
- Path: broadcast video → `vision.run` → the new `vision.stage8` fill (possession and ball state) → the new `prediction.infer` with a demo goal model that never trained on 10517 (the CV's fold-0 LightGBM, xG v1, fold 0's cross-fitted P(goal) map) → `demo.video` with a danger meter and a PFF shot/goal ticker.
- Then compare it with the same model on PFF's own tracking, and write a review.

## Environment
- Windows workstation (RTX 2060). All data is local under `data/` (gitignored). Footage is in `C:/footage/wc2022/` and is never committed. Renders go to `C:/footage/renders/`.
- Run Python as `.venv/Scripts/python` (e.g. `.venv/Scripts/python -m pytest tests/test_x.py -q`). Don't use a plain `uv run`: it can sync without extras and drop torch.
- If Windows printing breaks on Unicode, set `PYTHONIOENCODING=utf-8`.
- Vision weights: `--weights-dir ../sports/examples/soccer/data --pnl-weights-dir C:/Users/superCookie/Desktop/PnLCalib`.

## How to work
- Go in order: Task 1 → 6, then 7, then 8. Inside each task, follow its steps exactly: write the failing tests, run them and see them fail, implement, run them and see them pass, commit, push.
- The plan's code is a strong draft, not gospel. If a test or the code doesn't work against the real codebase, fix the root cause, keeping the plan's intent and the Review Focus guarantees. Say what you changed. Never weaken or delete an assertion just to get a pass.
- Before marking a task done, run its tests plus the neighboring test files the plan names. Run the full suite (`.venv/Scripts/python -m pytest -q`) after Tasks 4, 5 and 6. The last count was 566 passed, 110 skipped.
- Tick the plan's checkboxes (`- [ ]` → `- [x]`) as you go, and commit that with the task's work.

## Git (from my global rules)
- Commit and push after every step that changes files, without asking. Small, logical commits: roughly one per plan step that has a commit. Push each one.
- Commit messages are casual and human, like the ones in the plan (`git log` shows the style). Never add Co-Authored-By or any AI attribution. Signing is already configured; never pass `--no-gpg-sign`.
- Work on `main` (single agent, no worktree). No force pushes and no hard resets without asking me.
- End every turn with a clean, pushed tree (a Stop hook enforces this).

## Hard stops: stop and tell me, don't work around
- **Task 3, Step 6:** the fold-0 refit must reproduce `data/runs/lgbm-held-2026-09-27` fold 0 (h5): best_iter 112, the same es_matches, 0 null mismatches, max abs diff ≤ 1e-3. The map must come out near a 0.1509, b 0.8413. If anything differs, report the numbers and stop. Don't loosen `MAX_DIFF` on your own.
- **Task 2, Step 9:** render both open-play goals of 10517 in the pitch view, then tell me whether the meter visibly rises before each one, with the numbers. Pick the goal per the plan (default goal 4), and change the meter scale at most once.
- **Task 7 needs me.**
  - I download the clip. Give me the exact `yt-dlp` command from the plan, adjusted to the flags in `C:/footage/wc2022/01_argentina-vs-france.log`, for me to run with `!`.
  - We mark the sync pairs and the replay/close-up marks together. You prepare the `scripts/kick_times.py` output and the `data/splits/demo_clips.json` skeleton.
  - Don't invent sync pairs or marks.
- **Task 7, Step 5:** if `vision.bench` shows within-2 m far below vb01's ~86%, a sync offset off by more than 0.1 s, or bad team accuracy, stop and show me before going further.
- Never add the demo clip to `data/splits/vision_benchmark.json`.

## Finish
After Task 8 (review written, roadmap updated, everything pushed):
1. Get one whole-branch review from a fresh reviewer on the most capable model (superpowers:requesting-code-review). Fix what it confirms, then commit and push.
2. Give me a short summary:
   - what got built;
   - the reproduction-check numbers;
   - the bench numbers for the clip;
   - whether the meter rose before the goal, and how early, next to PFF tracking;
   - the path of the render;
   - what the review says to fix first.

Start by reading the files above. Then begin Task 1.

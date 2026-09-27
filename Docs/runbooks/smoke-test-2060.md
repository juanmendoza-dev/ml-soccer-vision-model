# Smoke test on the RTX 2060 workstation

First real run of the vision pipeline (roadmap Phase 2, group 1). Until now it has only run on synthetic fakes, so any result is useful, including a crash. Don't fix or tune anything during this run: record what happens and report it.

For Claude Code on the workstation: work through the steps in order. Commands are PowerShell unless marked. Steps marked **(user)** need a person watching the video; ask them and wait. Run commands from the repo root.

## 0. Ask the user first
- Path to the clip: a short broadcast clip, about 30 s of normal wide play, ideally with a close-up or replay in it. It must be **outside** OneDrive or any synced folder (08: footage is private). Never copy it into the repo.
- Which period the clip is from (1 or 2).
- Which way the home team attacks on screen (TV left or right).

## 1. Update and check CUDA
```powershell
git pull
uv sync --extra dev --extra vision
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```
Expect `True` and a list containing `sm_75`. If it's `False`, stop and report: the CUDA torch build didn't install (pyproject sends Windows torch to the cu128 index).

## 2. Roboflow weights (skip if already there)
Check for `..\sports\examples\soccer\data\` containing `football-player-detection.pt`, `football-pitch-detection.pt` and `football-ball-detection.pt`. If missing:
```powershell
uv tool install gdown
git clone https://github.com/roboflow/sports ..\sports
```
Then in **Git Bash** (setup.sh needs bash):
```bash
cd ../sports/examples/soccer && bash setup.sh
```
Confirm the three `.pt` files exist afterwards.

## 3. First run
```powershell
uv run python -m vision.run --video <clip> --weights-dir ..\sports\examples\soccer\data --match-id smoke01 --device cuda --detect-every 1 --max-frames 750
```
- Add `--home-attacks-left` if home attacks TV left; add `--period 2` for a second-half clip.
- Save the full console output (progress lines show fps).
- It ends either cleanly or with `02 validation failed:` and a list. Both are results; outputs are written either way. A traceback is also a result: save it in full and skip to step 7.

## 4. Validator and debug video
```powershell
uv run python -m gamestate.validate data\gamestate\smoke01
uv run python -m demo.debug --video <clip> --cache data\vision_cache\smoke01 --out data\vision_cache\smoke01\debug.mp4 --frames 0-749
```
Always pass `--frames 0-749`; without it the renderer runs past the processed frames and the rest looks frozen. Keep the debug video under `data\` (gitignored).

## 5. Watch the debug video (user)
Ask the user to open `data\vision_cache\smoke01\debug.mp4` and note:
- **Orientation:** in the minimap, center spot in the middle, penalty spots near the goals; left/right and near/far correct?
- **Kit clusters:** rings are colored by cluster. Which cluster number is the home team (0 or 1)?
- **View switching:** do rings disappear on close-ups, replays, ads?
- **Ball:** found most of the time, or jumping to other things?
- **IDs:** do players keep their number, or does it change constantly?
Screenshots of anything wrong help.

## 6. Rerun with home/away
```powershell
uv run python -m vision.run --video <clip> --weights-dir ..\sports\examples\soccer\data --match-id smoke02 --device cuda --detect-every 1 --max-frames 750 --home-cluster <0 or 1>
```
Same direction/period flags as step 3. New match id, because an existing one gets overwritten.

## 7. Homography acceptance follow-up
The first smoke01 run predates homography acceptance (03 Homography acceptance) and put objects >15 m off the pitch. Before rerunning, look at what those old rows actually were:
```powershell
uv run python -m vision.offpitch --match-id smoke01
```
Then rerun with the current code, new match id (an existing one gets overwritten):
```powershell
uv run python -m vision.run --video <clip> --weights-dir ..\sports\examples\soccer\data --match-id smoke03 --device cuda --detect-every 1 --max-frames 750 --home-cluster <0 or 1>
```
Same direction/period flags as step 3.
```powershell
uv run python -m gamestate.validate data\gamestate\smoke03
uv run python -m vision.offpitch --match-id smoke03
```
Judge this run on `homography_ok` coverage and projections rejected by the pipeline, not just whether the validator passes — nulled positions on a bad fit are the new correct behavior, not a failure. Thresholds (`min_inliers`, `max_homography_err_m`, `max_homography_jump_m`, `homography_max_age_s`, `max_off_pitch_m`) are untuned guesses; note anything that looks obviously wrong but don't tune them in this session.

## 8. Write up the results
Create `Docs/reviews/smoke-test-<YYYY-MM-DD>.md` with:
- Commit hash (`git rev-parse HEAD`), GPU, `uv run python -c "import torch, ultralytics, supervision; print(torch.__version__, ultralytics.__version__, supervision.__version__)"`
- Clip length, resolution, fps, period, home direction (not the file path or name if it identifies private footage)
- fps from the progress lines, for smoke01, smoke02 and smoke03
- Validator output for all three runs, and any traceback in full
- The contents of `data\vision_cache\smoke01\run.json` (without the video path)
- The user's notes from step 5, and the chosen `home_cluster`
- The `vision.offpitch` output for smoke01 and smoke03, and what changed between them

Commit and push only that markdown file. Don't commit anything under `data\`, the clip, or the debug video. Don't change code in this session. Fixes happen after the results are reviewed.

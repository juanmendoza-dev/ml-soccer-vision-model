# 09 — Hardware

## Goal
Record what the workstation actually is and whether it can run the vision pipeline (03) live on a broadcast feed (08 Live mode). Training is secondary here.

## Workstation (as of 2026-09-27)
| Part | Spec |
|---|---|
| Board | MSI MS-7C51 |
| OS | Windows 11 Home, build 10.0.26200, x64 |
| CPU | Intel Core i5-11400F, 6 cores / 12 threads, 2.6 GHz base. No integrated GPU (the "F"), so no Quick Sync |
| GPU | NVIDIA GeForce RTX 2060, 6 GB VRAM, Turing (compute capability 7.5, `sm_75`) |
| GPU driver | 32.0.16.1088 (nvidia-smi 610.88), supports CUDA up to 13.3 |
| RAM | 32 GB (2x Corsair 16 GB @ 3200 MHz), ~23 GB free at idle |
| Storage | WD 500 GB SSD (WDS500G2B0C) |
| Python | 3.13.5, 64-bit (system install) |

The M1 MacBook Pro is covered in 01.

## Setup notes
- **Python version:** 01 pins Python 3.11 via `uv`. `uv` downloads its own 3.11, so the 3.13 system install doesn't matter. Don't build the venv on 3.13.
- **PyTorch + CUDA:** the pip wheels bundle their own CUDA runtime. The driver only has to be new enough for that runtime, and 610.88 is. Pick the CUDA wheel from the PyTorch install page and don't try to match "13.3". After installing, check:
  ```python
  import torch
  torch.cuda.is_available()          # True
  "sm_75" in torch.cuda.get_arch_list()  # True; Turing support
  ```
- **Video decode/encode:** no Quick Sync, so decoding runs on the CPU (6 cores handle 1080p30 H.264) or on the 2060's NVDEC. Encoding the overlay output, e.g. for OBS, should use NVENC so the CPU stays free.
- **Disk:** 500 GB is tight once SoccerNet downloads, raw match video (2–4 GB per 1080p match) and `data/vision_cache/` pile up. Keep raw video on external or homelab storage (01 open question).

## First vision run (workstation)
```
git clone <this repo> && cd <repo>
uv sync --extra dev --extra vision   # pyproject sends Windows torch to the cu128 index
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"   # True, includes sm_75

# roboflow/sports weights (their examples/soccer/setup.sh, needs gdown)
git clone https://github.com/roboflow/sports ../sports && bash ../sports/examples/soccer/setup.sh

uv run python -m vision.run --video <demo clip> --weights-dir ../sports/examples/soccer/data --match-id demo1 --max-frames 750
uv run python -m demo.debug --video <demo clip> --cache data/vision_cache/demo1 --out demo1_debug.mp4
```
- Watch `demo1_debug.mp4`: rings colored by kit cluster, IDs, minimap. Then rerun `vision.run` with `--home-cluster 0|1` to get home/away in the game state.
- The progress lines print fps. That's the first real number for the live budget below.
- Keep clips and outputs outside iCloud or other synced folders (08: footage is private).

## GNN runs (workstation runbook)
Code and tests come from the M1. The full CV of the frame GNN (05 model 2) runs here on the 2060. `data/` is gitignored, so the data travels as a tar made on the M1 by `scripts/pack_workstation_data.sh`. It holds each match's `frames_10hz`/`objects_10hz` parquet files and `resample_report.json`, its game state `match`/`events`/`frames`, the native `objects` of 10502, 10504 and 10505 (the resampler's real-data tests read them), and the baseline run `lgbm-held-2026-09-27`. That's 402 files, 2.5 GB, with a `.sha256` beside it. `folds.json` is in git, and feature and graph caches are rebuilt here. All commands below are PowerShell, run from the repo root.

**Environment (checked 2026-09-28 against `uv.lock`):**
- Every package in the `prediction` extra has a Windows wheel. On Windows, torch comes from the cu128 index as `2.11.0+cu128`; the Mac gets `2.14.0` from PyPI, so numbers won't match bit for bit across machines.
- torch-geometric and unravelsports resolved too, but they left the extra because the frame GNN builds its own graphs (05).
- `uv sync` removes every extra you don't name, so sync `--all-extras`. That keeps the vision tools working on this machine and matches the test command. Every locked package has a Windows wheel.
- Windows writes redirected output in cp1252, which can't encode the reports' τ, ≥ and Δ. `PYTHONUTF8=1` switches Python to UTF-8. The code also writes its markdown files as UTF-8 explicitly.
- Idle sleep would kill a multi-hour run. `prediction.cv` asks Windows to stay awake while it runs (`SetThreadExecutionState`), but Update restarts aren't covered by that.
- On macOS only, torch and LightGBM can't share a process: LightGBM then torch hangs, torch then LightGBM segfaults. That's why the GNN tests run in their own pytest process (`tests/test_gnn.py`). A GNN run never loads LightGBM.

**Once, before the runs**
```powershell
# keep the machine up: no sleep or hibernation on AC power, and pause Windows Update
# (Settings > Windows Update > Pause updates for 1 week). Undo afterwards (last step).
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

cd <repo>
git pull

# data from the USB drive (E: here): check the copy, then unpack at the repo root
$tar = "E:\workstation-data-2026-09-29.tar"
(Get-FileHash $tar -Algorithm SHA256).Hash -eq (Get-Content "$tar.sha256").Split(" ")[0]   # True
tar -xf $tar                                     # Windows' built-in bsdtar
(Get-ChildItem data\processed -Directory).Count  # 66 (64 PFF + 2 Metrica)

# every new PowerShell session
$env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

uv sync --all-extras
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'sm_75' in torch.cuda.get_arch_list())"
# 2.11.0+cu128 12.8 True NVIDIA GeForce RTX 2060 True

uv run python -m pytest -q                        # 390 passed, 110 skipped (see below)
uv run python scripts\gnn_smoke.py --device cuda  # ~2 min: loss goes down, peak GPU memory, rows/s
```
- The tests give 390 passed and 110 skipped (360 on the clean clone check, plus 30 data-free tests added with the temporal GNN, the degraded graphs, stage 8 and the inferred possession arm). The skips are the converter tests: `data/raw` isn't copied. The M1, which has the raw data, runs all 500 in about 3 minutes (2026-09-29, 2 min 58 s; the slowest are the PFF and Metrica converter setups, 54 s and 26 s, and the GNN cases, 46 s). The GNN cases run in their own process, inside `tests/test_gnn.py`.
- The smoke fit prints training rows/s, including the early-stopping passes. The M1 Pro did about 2,800 on MPS. Peak GPU memory should be well under 6,000 MB.
- Git Bash works too (it's what Claude Code uses on Windows). The same commands work there with forward slashes and `export PYTHONUTF8=1`.

**Time one fold, then decide.** The first run on a machine also builds the graph caches (about 1 s a match on the M1, clean or degraded) and reports that time separately.
```powershell
uv run python -m prediction.cv --model gnn --horizons h5 --one-fold 0 --run-id gnn-frame-timing-h5
```
- It prints the estimated full run for H = 5 (5 folds plus the final τ) and saves a run marked partial, with a "PARTIAL RUN" report.
- Its fold 0 PR-AUC may be put next to LightGBM's fold 0 (0.330 at H = 5) as a sanity check. It's one fold, not a result.
- H = 3 costs about the same as H = 5.

**Full runs, one horizon at a time** (the GPU can't share: start H = 3 once H = 5 has finished)
```powershell
$id = "gnn-frame-2026-09-29-h5"
Start-Process uv -ArgumentList "run","python","-m","prediction.cv","--model","gnn","--horizons","h5","--run-id",$id `
  -RedirectStandardOutput "data\runs\$id.log" -RedirectStandardError "data\runs\$id.err" -WindowStyle Hidden
Get-Content "data\runs\$id.log" -Wait -Tail 20   # follow; Ctrl+C stops following, not the run
nvidia-smi                                         # GPU busy, memory in use
```
- The log gets one line per epoch and per fold. The last line is `wrote data\runs\<id>\report.md`.
- The run is hidden and survives closing the window. To stop it: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like "*prediction.cv*" | ForEach-Object { Stop-Process -Id $_.ProcessId }`.
- Then the same with `h3` and `gnn-frame-2026-09-29-h3`.

**Compare,** pairing runs by horizon (each GNN run has one horizon; the baseline has both):
```powershell
$h5 = "data\runs\gnn-frame-2026-09-29-h5"; $h3 = "data\runs\gnn-frame-2026-09-29-h3"; $base = "data\runs\lgbm-held-2026-09-27"
uv run python -m evaluation.compare $h5 $base --out "$h5\compare.md"
uv run python -m evaluation.compare $h3 $base --out "$h3\compare.md"
uv run python scripts\lead_time.py $base $h5 | Out-File -Encoding utf8 "$h5\lead_time.md"
uv run python scripts\lead_time.py $base $h3 --horizon h3 | Out-File -Encoding utf8 "$h3\lead_time.md"
```
- The lead-time script's gain table shows zeros for the GNN, since it has no gain shares (05).

**Temporal GNN** (05 model 3), after both frame GNN runs. Same steps with `--model tgnn`. Each row reads 6 graphs, so time a fold first. It's about 6× the frame GNN per row, not measured yet:
```powershell
uv run python -m prediction.cv --model tgnn --horizons h5 --one-fold 0 --run-id gnn-temporal-timing-h5
```
- Watch peak GPU memory in the log and `nvidia-smi`. A batch is 128 windows × 6 graphs. If it's too slow or runs out of memory, shrink it without a code edit: `--gnn-param steps=4` (the last 1.5 s), then `--gnn-param stride=8`, or `--gnn-param batch_size=64` for memory. Record what was used: it goes into `run.json` either way.
- Full runs as above with `--model tgnn` and ids `gnn-temporal-<date>-h5` / `-h3`.
- The comparison that matters is temporal vs frame GNN (05): `evaluation.compare` with the temporal run first and the frame run second, then `scripts\lead_time.py <frame run> <temporal run>`. The comparison with LightGBM comes second.

**Sensitivity reruns on a GNN** (05, Vision sensitivity test), after its clean run. Same command plus `--degrade`, H = 5 only, e.g. `--model tgnn --horizons h5 --degrade id_fragment:2 --run-id tgnn-degrade-id_fragment-2`, and `--degrade target` for the combined arm. Degraded graphs are rebuilt every run (about 1 s a match, not cached), and `--degrade-arm test` holds a second graph store in RAM. Compare each with the clean run of the same model.

**Afterwards:** `powercfg /change standby-timeout-ac 30` (or whatever it was), and resume Windows Update.

## Live feasibility

### Frame budget
Broadcast runs at 25–30 fps, so each frame gets **33–40 ms**. With 08's detection on every 2nd–3rd frame (tracker fills the rest), each detection gets **66–120 ms**. Tracking, the game state writer and the overlay still run every frame.

### Per-stage estimates
These are unmeasured estimates for a 2060 with FP16 TensorRT exports. Plain PyTorch is roughly 2–3x slower. Replace them with the benchmark numbers (roadmap Phase 3).

| Stage (03) | Live approach | Rough cost |
|---|---|---|
| 0. View gate | Every frame, CPU; skips everything below on ads/studio/close-ups | ~1 ms |
| 1. Player detection | YOLOv8s at 640–960, every 2nd–3rd frame | ~5–10 ms |
| 5. Ball detection | Small model at higher resolution, or tiled crops near the last ball position | ~10–25 ms; the most expensive stage |
| 4. Pitch calibration (PnLCalib, 03) | ≥ 3 calls a second on its own worker, plus pitch tracking between calls: holding the last camera at 1–2 Hz drops within 2 m from 90% to 66–80% (bench review 2026-10-02) | Measured in PyTorch: 185 ms GPU per call in fp16 autocast plus ~60 ms CPU voting, so ≥ 55% of the GPU at 3 Hz. Needs TensorRT or a smaller input; every 5th frame (offline) would be ~1.5 s of GPU a second |
| 2. Tracking (ByteTrack) | Every frame, CPU | < 2 ms |
| 3. Team assignment | Fit during a warmup window, then classify only new tracks | Occasional SigLIP batch |
| 6. Jersey OCR | Separate async worker, low rate, outside the main loop | Doesn't count against the frame budget |
| 8. Game state inference | Rule: every frame, pitch coordinates only. Learned possession (03, proposed): 159 features and two LightGBM predictions per 10 Hz tick, on its own worker | Rule: negligible. Learned: not measured; budget p95 < 10 ms, p99 < 33 ms per tick (03) |
| Predictor (05) | 10 Hz, small GNN | Negligible |

VRAM isn't the limit: detector, ball model, keypoint model and SigLIP together need well under 6 GB in FP16 at batch size 1. The limit is GPU time per frame. The full offline setup (large models, every frame, full resolution) won't reach real time on this card. Expect single-digit fps for that.

### Verdict
**Live is feasible with the reduced setup 08 already assumes**: a smaller YOLO, frame skip, TensorRT FP16, keypoints and OCR at reduced rates. The benchmark has to confirm it before the live config is fixed. If the ball stage doesn't fit, lower its rate first (the tracker and the < 1 s gap rule cover short misses), then lower the processing resolution.

### Stages that aren't live-safe as specced
- **Ball gap interpolation (03 stage 5):** interpolating a gap of up to 1 s needs the frames after the gap. Live mode extrapolates forward instead and marks those frames `interpolated=True` (see Live app).
- **Homography smoothing (03 stage 4):** must use a trailing window only. A centered window leaks future frames, and live mode can't see them anyway.
- **Team assignment (03 stage 3):** the offline method clusters crops from the whole video. Live mode fits KMeans on the first ~N seconds and then assigns new tracks to the nearest centroid. Refit after half-time kit/side changes if needed.
- **Jersey OCR (03 stage 6):** voting over frames works incrementally. `player_id` shows up a few seconds after a track appears, and stays null until then.
- **The predictor is already fine:** the no-leakage rule (CLAUDE.md) means it only uses frames `<= t`, so it runs online without changes.

## Training (brief)
YOLOv8n/s/m fine-tuning at 640 fits in 6 GB at batch size ~8–16. A ball detector at 1280 needs batch size ~2–4 or Colab (01).

LightGBM goal-model runs stay on the M1. The learned possession fits (03 stage 8) are the exception: they run on the workstation CPU, `num_threads = 6`, because the 15 nested trainings need an overnight slot the M1 can't give. Runbook:
1. Pilot (03, Timing plan): extraction on one match, then `outer_0` probe + refit. Record time, best rounds and peak RSS. Don't look at agreement.
2. Project the whole possession stage (extraction × 64, 15 trainings by row count, 25 prediction contexts). Keep stride 1 if it's ≤ 10 h and ≤ 24 GiB peak RSS; otherwise stride 2, then stride 4, re-piloting each time; past stride 4, stop.
3. Run it overnight: `python -m vision.possession_train --model-id <id>`. Models go to `data/models/possession/<id>/`, features to `data/vision_cache/<match>/`, per-context state to `data/processed/<match>/`.
4. The scored-row gate and, only if it passes, 07 #6 run afterwards and aren't in the 10 h.

Measured on the inferred possession run (07 #6, 2026-09-29), 64 games and both horizons: 7 min 22 s in all. That's stage 8 over the native game state 12 s (cached per match after), features 7 s, loading 21 s, CV 400 s (about 35 s per fold per horizon).

## Live app
Live mode lives in its own repo, `soccer-live-overlay`, which will install this one as a dependency. It never trains or defines formats.
- **Input:** screen capture of the match already playing, via `dxcam` on the workstation. It captures a monitor region, so the player runs fullscreen. DRM-protected players can capture as black frames; check the streaming service first.
- **Output:** predictions over a local websocket to an OBS browser source.
- **Ball gaps:** extrapolated, never buffered. The overlay sits on the video, so ~1 s of delay would show.
- **Recording:** every live session is recorded (video, per-frame timing, predictions, shot/goal markers) for replays and backtests. Format in that repo's `Docs/Specs/02-session-recording.md`.

## Open questions
- The live config (model sizes, input resolution, per-stage rates) comes from the benchmark.

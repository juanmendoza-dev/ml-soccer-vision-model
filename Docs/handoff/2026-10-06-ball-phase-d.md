Continue the ball fix from Phase D. The plan is `Docs/plans/2026-10-05-ball.md`; Phases 0, A, B and C are done and ticked. Execute the rest with superpowers:executing-plans, inline: no subagent per task, one fresh reviewer on the most capable model at the very end. The original handoff, `Docs/handoff/2026-10-05-ball-fix.md`, still sets the rules: hard stops, git, environment and finish.

## Where things stand (2026-10-05, `main` at the commit that added this file)
- **Ledger:** `.superpowers/sdd/2026-10-05-ball/progress.md` (gitignored, on the workstation). Every task's completion line and every `Ruling:` line is there. Read it first. Collect the rulings into the final summary.
- **Results so far:** `Docs/reviews/ball-fix-2026-10-05.md` (Baseline, Phase A, Phase B, Phase C with the sweep and the user's decision). Append Phase D, E and F sections there.
- **Built:**
  - `vision.replay --variant TAG` (stage 5 rerun into `data/variants/TAG/`);
  - `python -m demo.check` (the demo regression table; `--rerun --variant TAG --set ...` runs replay, stage 8 and inference; `--render` writes to `C:/footage/renders/`);
  - `vision/ball_truth.py` (PFF reference, cached in `data/vision_bench/ball_truth/`);
  - the PFF score in `vision.bench`, plus the ball score on PFF-estimated (aerial) frames;
  - `vision/ball_assist.py` and `scripts/ball_click.py --assist/--flag`;
  - the tracker gate in `vision/ball.py`.
- **Verified labels:** all four clips (vb01–vb03 and `demo01-arg-fra-81`) in `data/splits/vision_ball_labels.json`. `python -m vision.ball_assist --status` shows 0 unresolved flags.
- **Tracker default since `9697e4c`, chosen by the user at the Phase C hard stop:**
  - `ball_picker=gate`, `ball_cand_margin_m=2`, `ball_size_lo=0.5`, `ball_size_hi=2`, `ball_max_speed_mps=40`;
  - `ball_max_gap_s=0.5` (was 1 s; 10-ball and 03 updated).
  - Old runs replay as "max". Pooled ball score went from 78.8 / 81.4% to 81.1 / 83.1%.
- **The cached runs (vb01–vb03, demo) are old runs:** they replay as "max". To see the new default on the demo, use a variant: `--rerun --variant all-gap05 --set ball_picker=gate --set ball_max_speed_mps=40 --set ball_max_gap_s=0.5 --set ball_cand_margin_m=2 --set ball_size_lo=0.5 --set ball_size_hi=2`. That variant already exists under `data/variants/all-gap05`.
- **The demo now, on that variant:**
  - meter at 5/3/2/1/0.5 s: 0.48 / 0.18 / 1.05 / 0.41 / 0.30% (PFF 0.17 / 0.17 / 1.55 / 2.24 / 2.62%);
  - the white boot at −0.36 s is 1.29% (was 4.84%);
  - false peaks: 5 rows, up to 8.0%. That's the far keeper's glove at 0.72 while the real ball is undetected, a Phase F matter;
  - after the goal: 21 rows over 2%, up to 23% (Phase E).

## What's left, in this order
1. **Phase E** (small, independent; the user may want it first): Task E1's diagnosis, then the rule in 03, then the code with a test. Check on the demo variant: no row with p > 2% after the PFF goal frame. `vision.state_check` agreement must not drop.
2. **Phase D, ball in the air:**
   - Task D1: the measurements (`scripts/ball_height.py`).
   - **Hard stop D2:** superpowers:brainstorming, then show the user the design, the measurements and the spec text before any code.
   - The aerial rows are frames 2613–2622 and 2667–2688 (−3.0 to −2.7 s and −1.2 to −0.5 s): the meter sits at 0.3–0.4% while PFF is at 2.2–2.6%. On PFF-ESTIMATED rows the vision ball is ~11 m off at the median.
   - PFF z on aerial frames is ESTIMATED (interpolated, ~44 px off), so treat it as a loose reference.
3. **Phase F, detector fine-tune** (added at the user's request): F1 sync check → F2 auto-labels (GPU inference) → F3 missed-ball labeling (**user**) → **F4 hard stop with the GPU time estimate** → F5 train → F6 bench.
4. **Finish:** the render, roadmap, 10-ball ticks, the whole-branch review (`git log 36b581a..HEAD`), and the summary from the original handoff.

## Gotchas found this session
- **PowerShell:** the user runs commands in PowerShell, so give them `$env:PYTHONPATH = "."` and `.venv\Scripts\python ...`, not bash-style `PYTHONPATH=. ...`.
- **The label tool saves after every key press.** The Stop hook then asks to commit `data/splits/vision_ball_labels.json`; commit it as the user's work in progress.
- **Pushes to GitHub sometimes return 500.** Retry once and they go through.
- **Ruff flags `vision/possession_train.py` PERF102.** It predates this work; leave it.
- **The demo's PFF score is withheld** (65% agreement, under 70%). It isn't sync: a time sweep peaks at the current offset. Score the demo on its verified labels.
- **Full suite:** 703 passed, 110 skipped at the end of Phase C.

Pick the ball fix back up from an open decision: the `ball_air_ratio` default. The plan is `Docs/plans/2026-10-05-ball.md`; the original handoff `Docs/handoff/2026-10-05-ball-fix.md` still sets the rules (hard stops, git, environment, finish).

## Where things stand (end of 2026-10-05)
- **Ledger:** `.superpowers/sdd/2026-10-05-ball/progress.md`, with every task line and `Ruling:` line. Read it first.
- **Phase E is done, with no code.** The real ball isn't detected after the shot. A goal-mouth rule gained nothing on 64 PFF games. 03 has a known-limit note, and the rows after the goal belong to Phase F.
- **Phase D is built** (10-ball 2g, `fd6a6cb`..`abfcaee`):
  - `ball_air_ratio`, with an airborne flag from the box size;
  - airborne rows are written `visible=False, interpolated=True`;
  - the bench prints the airborne share.
- **Checks:**
  - Scores are identical on all four clips; the replay check passes.
  - Demo aerial rows went from 0.3% to 1.45–4.53% (variant `air15`).
  - Full suite: 709 passed, 110 skipped.
- **Open: the default.** 1.5 makes vb01 and vb02 lose a usable ball on 23% and 29% of ball frames: no xG, so the meter blanks. Table: review doc, "Phase D: built". Options:
  1. keep 1.5;
  2. default `inf` until Phase F, demo as the `air15` variant (recommended);
  3. 1.8 or 2.0, which barely helps the demo.
- **Not yet done:** 10-ball §4 step 3b is unticked until the default is settled.

## Next
1. Apply the user's choice:
   - option 2: `ball_air_ratio = inf` in `VisionConfig`, and note it in 10-ball 2g and 03;
   - then the tests, a bench run, and the tick.
2. Phase F: F1 sync check → F2 auto-labels → F3 missed-ball labels (the user) → **F4 hard stop with the GPU estimate** → F5 → F6. F6 now also re-derives `ball_air_ratio` on the new boxes.
3. Finish: the render, roadmap, 10-ball ticks, whole-branch review (`git log 36b581a..HEAD`), summary.

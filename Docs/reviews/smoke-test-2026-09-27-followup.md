# Smoke test follow-up — homography acceptance and kit colors (2026-09-27)

Second and third runs on the same clip as `smoke-test-2026-09-27.md`, per step 7 of
`Docs/runbooks/smoke-test-2060.md`. Same 2060 workstation, same clip (Spain vs Japan,
period 1, home attacks TV left, first 750 frames of a 1920x1080 30 fps source).

## Runs
| run | commit | what changed | wall time | avg fps | validator |
| --- | --- | --- | --- | --- | --- |
| smoke01 | `ba8831a` | before acceptance | 120.2 s | 6.2 | FAIL, 36 off-pitch rows |
| smoke03 | `9edc3af` | homography acceptance + 3 s team warmup | 111.3 s | 6.7 | pass |
| smoke04 | `6708735` | kit colors on Lab chroma, k-means restarts | 122.7 s | 6.1 | pass |

smoke03 ran with `--home-cluster 1`, smoke04 without one (the cluster numbering was
expected to change with the new fit, so it was left for the debug video).

## What acceptance did — `vision.offpitch`
```
smoke01  frames 750, match 375, homography ok 375, objects 5940, off pitch 36, rejected 0
smoke03  frames 750, match 375, homography ok 310, objects 5818, off pitch  0, rejected 75
smoke04  frames 750, match 375, homography ok 310, objects 5818, off pitch  0, rejected 75
```
The 36 bad rows in smoke01 were three separate things:

- frame 153, one interpolated ball at (22.0, 53.4) — just off the touchline, on a good fit
  (11 keypoints, 0.8 m reprojection error). This is the extrapolation-off-the-pitch case.
- frame 348, one referee at (285.9, -1812.6) — nonsense, on a fit with **0** keypoints in the
  last sample and 5.1 m error, 318 frames after the last view switch: a stale matrix.
- frames 725–749, 34 rows, two players around y = -63 to -69 — one bad fit held across the
  end of the processed range (7 keypoints but 0.9 m error, right after a view switch).

Under acceptance all three classes go away: 0 rows off the pitch, 75 projections nulled by
the pipeline instead. Cost is `homography_ok` on 310 of 375 match frames (83%) instead of
375 — the 65 lost frames are the ones that were producing the garbage.

smoke03 and smoke04 are identical on every geometry number, as expected: the kit-color
change touches team assignment only, not projection.

## Still open
- Nobody has watched `smoke04/debug.mp4` yet. Needed: does the chroma fit actually split
  red Spain from blue Japan from the start of the match segment (warmup is now 3 s), and
  which cluster number is home?
- Thresholds (`min_inliers`, `max_homography_err_m`, `max_homography_jump_m`,
  `homography_max_age_s`, `max_off_pitch_m`) are still the untuned guesses. 83% coverage on
  one 25 s clip isn't enough to tune on — leave them until there are broadcast clips with
  more camera variety.
- The frame-153 ball case is the F8 item (reset ball motion after gaps and cuts); acceptance
  only nulls it, it doesn't fix the extrapolation.

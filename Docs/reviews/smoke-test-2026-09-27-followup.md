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

## Kit clusters in smoke04
The first `smoke04/debug.mp4` was only 61 frames (2 s) — the render was given a short frame
range, not the processed 0–749. Re-rendered over the full range; the cache itself always
covered all 750 frames.

Measured straight off the cache instead of by eye: for every tenth frame, take the torso
band of each detected player box, convert to Lab, and keep the most saturated 40% of pixels.

```
cluster 0: n=166 a*=+38.6 b*=+39.2 chroma=62.1 | red-ish 89%  blue-ish  1%  hue median 36 deg
cluster 1: n=175 a*=-0.5  b*=+6.8  chroma=37.7 | red-ish 13%  blue-ish 30%  hue median 119 deg
```
Cluster 0 is decisively Spain's red. Cluster 1 reads as grass-dominated in this measurement
(Japan's navy is dark and low-chroma, so the saturated-pixel filter picks up the pitch behind
the player) — it is clearly *not* red, but this measurement doesn't prove it is coherently
Japan rather than a leftovers bucket. The split is even (1,706 vs 1,773 player rows) and
clustering starts at frame 120 (4 s) with the 3 s warmup.

So **home is cluster 0 in smoke04**. Note smoke03 ran with `--home-cluster 1` under the old
fit: cluster numbering is only stable across refits *within* a run, not across runs, so the
number has to be re-checked per run until the live warmup prompt exists.

Watched afterwards on the full-length render: the two kits are separated and the
assignment doesn't switch during the match segment. Cluster 1 is Japan, not a leftovers
bucket. Kit colors are good enough on this clip.

## Still open
- Thresholds (`min_inliers`, `max_homography_err_m`, `max_homography_jump_m`,
  `homography_max_age_s`, `max_off_pitch_m`) are still the untuned guesses. 83% coverage on
  one 25 s clip isn't enough to tune on — leave them until there are broadcast clips with
  more camera variety.
- The frame-153 ball case is the F8 item (reset ball motion after gaps and cuts); acceptance
  only nulls it, it doesn't fix the extrapolation.

# Vision benchmark, first clip (2026-10-01)

First real run of the W0 benchmark (07 "Vision benchmark (W0)"). One clip, so this is a baseline and a check that the tooling works on real footage, not a result to tune on. Marks checked by the user 2026-10-01.

## Clip
`vb01-arg-fra`: PFF 10517 (Argentina v France, final), second half, video 1:00–2:00 (match clock 65:26–66:26), 1080p30. Vision ran from 0:52 (8 s pre-roll), roboflow weights, `--detect-every 1`, RTX 2060 at 5.0 fps.

- **Sync on cut edges:** five tight live close-ups. Their 10 edges give offsets of 1167.44–1167.52 s against PFF's switches to and from all-`ESTIMATED` frames, so it's the world feed and the sync is good to about a frame. The scoreboard clock agrees to 1.5 s. `--offset-check` puts the best offset 0.23 s earlier; worth a look on the next clips.
- **Direction:** with the right goal on screen, PFF's visible players are at x ≈ +45, so home attacks TV right in period 1.
- **Replay check:** the replay reproduced the run's detections cache (the bench refuses a clip otherwise).

## Baseline (current thresholds)
| | |
|---|---|
| Scored frames | 1,327 of 1,800 (the rest are close-ups) |
| Geometry missing | 24.5%: view `other` 16.1%, homography rejected 10.1% of match view |
| People within 2 m (misses count) | 14.0% of 23,155 PFF visible player-frames |
| Matched error, median / p90 | 2.81 m / 4.43 m |
| Within 2 m on frames with geometry | 17.8% |
| Unmatched vision rows per frame | 4.37 |
| Outfield team accuracy / coverage | 52.4% / 99.4% |
| Keeper team accuracy / coverage | 87.5% / 54.7% |
| False live | 1.5 s of marked close-up |

## What it shows
1. **Team clustering failed on this clip.** 98% of player rows (21,140 of 21,479 with a cluster) landed in cluster 1, both teams alike. So team accuracy is a coin flip. That's `KitColorTeams`, not the scorer: navy France against white and sky-blue Argentina should separate, so the fit is splitting on something else (grass, lighting, referees).
2. **Positions are about 2–3 m off, both ways.** Flipping x or y makes it worse, so orientation is right. Against the nearest PFF player, vision's error is about 2 m in each axis (median absolute), with a −1.1 m bias in y (toward the near touchline). Plots of four frames show it near the halfway line too, not only far out. Candidates: keypoint/homography accuracy on this stadium, the foot anchor, and what PFF's position means (centre of mass vs feet). One clip can't separate them.
3. **The view gate drops 16% of unmarked wide frames.** That's more than the rejected homographies, and no acceptance threshold moves it (07). Example: video 1:16, a wide shot with the halfway line and centre circle has no geometry.
4. The 17% target on rejected homographies is met here (10.1%), but within 2 m is far below the 90% target. **Threshold tuning isn't the bottleneck on this clip;** geometry accuracy and teams are.

## Diagnosis (same clip, after the user checked the marks)
**Teams.** `KitColorTeams` uses mean Lab chroma (a, b) only. Its own docstring says kits that differ mainly in lightness need L back, and navy vs white/sky blue is that case. On 350 torso crops matched to PFF players (≤ 2.5 m), a 2-means fit on standardized features:

| Feature | Agreement with PFF team |
|---|---|
| (a, b) mean (current) | 56.9% |
| (L, a, b) mean | 85.4% |
| (L, a, b) median | 85.4% |
| L histogram + (a, b) | 85.4% |

These are per crop; per-track votes come on top.

**Positions: the homography is wrong as a whole on each frame, not the detections.**
- Removing one shift per frame: median 2.80 → 1.95 m (within 2 m: 26% → 52%). Removing one affine per frame: 0.52 m (88%). So boxes and feet are fine, and each frame's mapping is off by a stretch.
- The per-frame affine from vision to PFF has median a_xx 0.87 and a_yy 0.945 (p10–p90: 0.78–1.02 and 0.88–1.05). Vision's picture is stretched about 13% along the pitch and 5% across, and the amount changes per frame. One global affine leaves 2.39 m, so it's not one constant (pitch size, PFF units).
- The fit's own check can't see it. Accepted fits have a 0.41 m mean inlier error (p90 0.65 m), but a homography has 8 parameters and these fits use 5–11 inliers (median 8). The fit can pass close to every keypoint and still be wrong elsewhere.
- Not the cause: inlier count or spread (error is flat across quartiles of the inlier hull area, 2.5–2.8 m), lag (`homography_window = 1`: 2.81 → 2.70 m), distance from the camera (flat across the width), the error threshold (`max_homography_err_m = 0.5` makes it worse).

## Next
- User checks the marks; then the other four 2022 clips and the three geometry-only clips, before reading anything into these numbers.
- If items 1 and 2 hold across clips, they go ahead of the threshold sweep: kit clustering, and a look at where the 2 m comes from. Tackle the y bias first, as the cheapest lead.

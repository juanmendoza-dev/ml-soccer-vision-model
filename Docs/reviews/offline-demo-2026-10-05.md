# Offline demo, first clip (2026-10-05)

Plan `Docs/plans/2026-10-05-offline-demo.md`, roadmap Phase 3. Broadcast video of the ARG–FRA final (PFF 10517) → `vision.run` → `vision.stage8` → `prediction.infer` with the fold-0 demo goal model → `demo.video` with the danger meter and the PFF ticker. Then compared with the same model on PFF's own tracking.

**Short answer:** in the open play before Mbappé's 81' goal, the vision meter rises about 2 s out, like PFF's, wherever vision has the real ball on the ground: 1.0% at −2.3 s to 3.3% at −1.5 s, against PFF's 1.5% to 2.9%. It doesn't hold up after that. The ball goes into the air for Thuram's headed lay-off, the ground homography puts it about 15 m too far up the pitch, and the meter falls to 0.3%. Its highest value before the goal, 4.8% at −0.36 s, comes from a false detection, and its highest value in the 10 s before the goal, 7.9% about 7 s out, from a phantom ball. **The ball is the thing to fix first.**

## Footage

- `C:/footage/wc2022/13_argentina-vs-france-goal81.mp4`: YouTube `RgqKdplLIk4`, 5366–5476 s (110 s), format 616 (1080p, 30 fps) video only, re-encoded to H.264. sha256 `f104da23…79084f5`, 3,300 frames, no gaps.
- Private (08 "Footage"). Not committed, not published. The render `C:/footage/renders/demo01-arg-fra-81.mp4` stays private too.
- The first download (formats 616+251) came out VP9 with the video track cut at 61 s (decoder errors on a 65 KiB/s connection), and a retry hit a 403 on the audio. Video only fixed both. Vision doesn't use audio.

## What ran

| | |
|---|---|
| Code | `main` at `3c26d29` (stage 8 fill fix included) |
| Clip manifest | `data/splits/demo_clips.json`, `demo01-arg-fra-81`: P2, video 28–105 s scored, sync offset 2068.529 s (pairs 41.700 ↔ 2110.243 and 80.000 ↔ 2148.515; all 9 cut edges in the clip agree to 0.06 s), 6 marks (5 close-ups, the post-goal wide shot as `other`) |
| Vision | `vision.run --period 2 --max-frames 3150 --home-cluster 0`, 5.9 fps on the 2060 (about 9 min) |
| Goal model | `goal-f0-2026-10-05`: `lgbm-held-2026-09-27` fold 0 refit. Reproduction: best_iter 112 (base 112), es_matches equal, 361,847 rows, 0 null mismatches, max abs diff 1.1e-16. Map a 0.1509, b 0.8413 (folds 1–4 only). xG v1 |
| Predictions | `data/predictions/demo01-arg-fra-81/goal-f0-2026-10-05.parquet` |

### Bench against PFF (`vision.bench --offset-check`)

| | demo01 | vb01 (same broadcast) |
|---|---|---|
| within 2 m | **86.5%** (with geometry 90.0%) | about 86% |
| median / p90 | 0.65 / 1.66 m | |
| geometry missing | 4.4% (all view `other`) | |
| best offset vs the sync | **−0.033 s** | |
| player team accuracy / coverage | **97.4% / 97.4%** | |
| goalkeeper team accuracy / coverage | 64.9% / 93.4% | |
| false live | close-up 0.3 s | |

Direction is right: a 180° error would put within 2 m near zero. Home is kit cluster 0, picked by agreement on the first run and set on the second, with identical numbers.

### Stage 8 and inference

| stage 8 (3,150 native frames) | | inference (1,050 grid rows) | |
|---|---|---|---|
| possession set | 43.3% | predicted | **18.0%** |
| ball state alive / dead / null | 45.7% / 4.0% / 50.4% | ball state null | 50.2% |
| teamless carriers dropped | 31 frames | no possession | 56.8% |

- **Why only 18% predicted:** 56% of grid rows are all-estimated (no player detected, so no player visible). Video 0–25 s and the close-ups are like that, and PFF is all-ESTIMATED over the same stretches. Stage 8 also finds no possession at all until 50 s: from 25 to 50 s the nearest player is within 1.5 m of the vision ball on only 13–18% of frames. In the scored part (28–90 s) 29% of rows are predicted; in 80–90 s, the goal's build-up, 94%.
- **Bug found on the real clip, fixed:** the first stage 8 fill failed 02 validation with 31 frames of a carrier but no possession. The carrier, `0-18`, is a player vision gave no team, at 27.8 s, before any possession existed. The fill now drops such a carrier and counts it (`carrier_without_team_dropped`, test `test_a_carrier_without_a_team_is_dropped`, 03 updated). `vision/state.py` is unchanged.

## 08's acceptance: does the meter rise before the goal?

Calibrated P(goal within 5 s) at grid rows before the goal (PFF t 2158.59, video 90.06 s). "Ball" is what vision's ball was on that row, checked on full-resolution frames with the detection box drawn.

| before the goal | vision | PFF tracking | vision's ball |
|---|---|---|---|
| 8 s | 0.21% | 0.16% | |
| 5 s | 0.26% | 0.17% | |
| 3 s | 0.18% | 0.17% | real, in the air (long ball to Mbappé) |
| 2.5 s | 0.57% | 0.17% | tracker only, near the ball |
| 2 s | 1.05% | 1.55% | real, on the ground |
| 1.7 s | 1.53% | 1.69% | real, on the ground |
| 1.5 s | **3.33%** | 2.87% | real, on the ground |
| 1.3 s | 1.02% | 2.43% | real, lifted by Thuram's header |
| 1 s | 0.41% | 2.24% | real, in the air: ground-projected ~15 m too far up-pitch |
| 0.5 s | 0.30% | 2.62% | same |
| 0.36 s | 4.84% | 2.54% | **false**: a white boot (frame 2691), then the tracker |
| 0.2 s | 2.38% | 1.12% | tracker from the false detection |

- **Yes on the ground, no in the air.** While vision has the real ball on the ground (−2.3 to −1.3 s), the meter rises like PFF's and to the same level: about 2 s ahead, 10× over the build-up's 0.2%. The real build-up is two aerial balls (the long ball in, then Thuram's header back to Mbappé's volley), and vision can't place a ball in the air.
- `scripts/demo_compare.py`'s fixed samples at 5/3/2/1/0.5 s land in that aerial dip (1 s 0.41% vs 2.24%, 0.5 s 0.30% vs 2.62%). Read them with the table above.
- **Two false peaks:**
  - **−7.9 to −6.7 s (video 82.2–83.4 s), up to 7.9%:** a phantom ball. At frame 2464 a 0.42-confidence detection pulled the ball from the touchline (17, −30) into the box, about 20 m in 0.1 s. The tracker then carried it toward goal for 1.3 s with no detection behind it (`tracked_only`). xG read 0.43–0.75. PFF there: 0.2%.
  - **After the goal (91.3–92.1 s), 3–17%:** the ball sits in the net with `ball_state` still alive. The PFF ticker has the goal 1 s earlier, so this reads as late, not as a lead.
- Elsewhere the meter stays low: over video 28–81 s the highest is 0.92%, the median 0.10%.

### Where vision and PFF disagree, over the whole clip

- **Agreement:** possession agrees on 88.8% of grid rows where both have one. 162 rows have both a vision and a PFF P(goal); the correlation of log p is 0.62.
- **Rows with no meter:** close-ups and pre-roll (all-estimated, 56% of rows), then 25–50 s with no possession because stage 8 never sees a carrier.
- **Ball:** three failure types in 10 s.
  - A ball in the air is projected to the ground: −3.0 to −2.7 s and −1.2 to −0.5 s.
  - A low-confidence false detection that the tracker then keeps alive: about −7 s and −0.36 s.
  - Short tracker-only stretches.
- **Goalkeeper:** Martínez is on screen only in the last 1.3 s, labelled home (correct) on 28 of 39 frames and away for the last 0.37 s. The keeper's kit is in neither outfield cluster, so his team is close to a coin flip (64.9% on the bench). The keeper features carry about 2% of the model's gain, so this nudges the meter more than it breaks it.
- **PFF ticker:** "GOAL France (PFF)" appears on the frame where Martínez is beaten, so the sync holds on the video.

## What to fix first, judged from this clip

1. **Ball height and ball outliers (10-ball).** Both lead-time failures come from the ball, not the model:
   - aerial balls projected to the ground;
   - a 0.42-confidence jump the tracker extrapolates for 1.3 s.

   The 10-ball build (§4 steps 1–5) comes first. A speed gate on the tracker would remove the −7 s phantom (that jump implies about 200 m/s).
2. **Ball state after a goal.** The ball in the net stays `alive` on vision, which lights the meter after the goal. Stage 8's out-of-play rule should see the ball past the goal line.
3. **Possession coverage on vision (v1h on vision runs).** No possession for 25 s of live play, and 31 teamless carrier frames. It's the follow-up already in 03: v1h needs an entry path for a match outside its manifest.
4. **Keeper team from position, not kit.** Assign a goalkeeper to the team whose goal he's in front of (the attacking direction is known). It's a small vision change, and it brings keeper accuracy from 65% to near 100%.
5. **Direction and teams:** fine on this clip (86.5% within 2 m, 97.4% teams). Nothing to fix here.

## Task 2: the PFF pitch view (`demo.render --pgoal`, both open-play goals of 10517)

Out-of-fold `p_goal_cal_h5` from `lgbm-held-2026-09-27`, the same fold-0 model and map.

| before the goal | −20 s | −10 s | −5 s | −3 s | −2 s | −1 s | −0.5 s | goal frame |
|---|---|---|---|---|---|---|---|---|
| goal 4, Mbappé 81' (P2 t 2158.6) | 0.4% | 0.0% | 0.2% | 0.2% | 1.5% | 2.2% | 2.6% | 1.3% |
| goal 2, P1 t 2121.1 | 1.1% | 0.4% | -- | 1.0% | 3.3% | 5.7% | 14% | 19% |

- Goal 4 rises from about 2 s out (t 2156.6), 10× over the build-up's 0.2%: green to amber, just under the 5% tick. Goal 2 rises from about 4 s out and reaches red. The `--` at −5 s is a null p.
- The scale (log, 0.1–50%, ticks at 1/5/20%) was readable on both, so it wasn't changed. Goal 4 kept as the clip's goal, because it rises and vb01's broadcast sync is for period 2.
- The H.264 encoder (`avc1`, openh264) fails on the workstation; `open_writer` falls back to `mp4v`. Renders play fine.

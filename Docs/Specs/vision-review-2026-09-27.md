# Vision review — 2026-09-27

**Phase A only; implementation awaits approval.** Reviewed commit `13d80d7fc214d7a2ba9dd23c0777228b7102e3f2`. This is a separate review, not a replacement for numbered specs. No source, test, dependency, schema, or roadmap files were changed during this review.

The pipeline has working synthetic plumbing, but it is not yet a validated source of broadcast game state. Several reproducible logic defects sit outside existing test coverage. The first real workstation run remains the first evidence-gathering task; it must precede accuracy claims and tuning. Stage 8 is also an unresolved Phase 1 dependency, not a component ready to plug in.

## 1. Scope and evidence

Read 00, 02, 03, 09, and the roadmap in the requested order, followed by all nine named implementation files and all five named test files. Also read `CLAUDE.md`, `gamestate/schema.py`, `gamestate/validate.py`, `pyproject.toml`, and the helper functions in `converters/common.py` used by vision. No prediction implementation was inspected, and no converter or dataset was modified.

Evidence labels below:

- **Reproduced:** exercised existing code with temporary, in-memory synthetic inputs; no new test files were written.
- **Code finding:** follows directly from the implementation, but the corresponding end-to-end scenario was not run.
- **Unverified risk:** needs actual model weights, footage, or labeled evaluation.
- **Decision:** the specs or intended operating policy need clarification before implementation.

The local checkout uses Python 3.11.16 on arm64 and lacks OpenCV, supervision, torch, and ultralytics. Its initial test result was **43 passed, 4 skipped**: all four vision test modules skipped at import. To exercise them, used an isolated `uv --no-project` environment, leaving the repo environment and dependencies unchanged:

```sh
uv run --no-project --python .venv/bin/python \
  --with opencv-python --with polars --with pytest --with supervision==0.27.0 \
  python -m pytest -q tests/test_vision_pipeline.py tests/test_vision_pitch.py \
  tests/test_vision_view_gate.py tests/test_vision_stages.py tests/test_validate.py
```

Result: **62 passed, no skips**, with OpenCV 5.0.0, Polars 1.44.2, and supervision 0.27.0. This includes the real ByteTrack wrapper on synthetic boxes; YOLO inference remains untested. The temporary environment is not evidence that the workstation's eventual dependency resolution works. No CUDA workstation was available here; no first workstation run was attempted.

### What is already consistent

- The writer emits all five schema-v0.6 tables, including correctly shaped empty events and players tables. Null player IDs before OCR are permitted.
- Pixel anchors are projected directly to meters with the 105 × 68 template. The global 180° direction conversion matches 03. No vision/prediction cross-import was found.
- Player filling and ball extrapolation use past state. Homography averaging uses only preceding fits. No centered position smoother or future-position interpolation was found.
- On an actual transition to `other`, the pipeline emits no objects and a null polygon; on return it resets tracks, ball state, and homography, and prefixes IDs with a new segment.
- Unknown teams and unimplemented possession fields remain null. This is honest missingness, although it does not fulfill stage 8's acceptance criteria.
- The synthetic tests establish geometry against their own constructed template, basic team assignment, ad hysteresis, cached output, and rendering. They do not establish real keypoint ordering, ball accuracy, or broadcast team accuracy.

## 2. Confirmed defects and contract gaps

File references identify the reviewed revision; line numbers will move after implementation.

### F1 — Velocity window uses future timestamps

**Reproduced; causality violation on irregular input.** `vision/writer.py:142` calls `converters/common.py:17` at close. The helper picks its backward-difference window from the median positive timestamp interval across the entire run (`common.py:25`). Thus future timestamps affect an earlier output even though the position differences themselves look backward.

Probe: positions `x = frame_id²`, `y = 0`, one continuous track, period 1. With timestamps `[0, .1, .2, .3]`, frame 3 has `vx = 40 m/s`. Append timestamps `[.31, .32, .33, .34, .35, .36]` and corresponding later positions: the same frame 3 becomes **30 m/s**. This violates 02's prefix-causality rule. The current CLI synthesizes constant-rate timestamps, so this probe does not show changing velocities on an ordinary fixed-rate CLI run.

Proposed fix within scope: calculate vision velocities using configured native cadence or a strictly trailing time-based window, with track-gap and period resets. Do not modify the shared converter helper under this task. Add a prefix-invariance regression test with irregular timestamps. Computing at close is not intrinsically leakage; the future-dependent window selection is.

### F2 — Sparse keypoint failures never switch off a green close-up

**Reproduced.** `vision/view_gate.py:31–45` treats an unsampled keypoint count (`None`) as passing and resets the failure streak. `vision/pipeline.py:125` samples every fifth match frame by default. Consequently, repeated failures are separated by passing frames and cannot accumulate 0.5 seconds.

Probe: six seconds of green frames at 25 fps, default config, keypoint model returning zero confident points throughout. The first second is `other`; all remaining **125 frames are `match`**, including the final frame. The existing gate test supplies bad keypoint counts on every frame and misses this integration failure.

Entry also uses grass alone: the first keypoint result is obtained after `gate.update()` has switched to `match`, and is not fed back into that update. A remedy must define causal sample retention/expiry and recovery probing. See Q1: demanding keypoints to recover while forbidding all keypoint work in `other` would otherwise deadlock.

### F3 — Skipped-frame motion is measured from an already extrapolated box

**Reproduced.** `_update_tracks()` (`vision/pipeline.py:205`) subtracts the preceding frame's filled box, then divides by the full detection interval. This underestimates motion after the first fill.

Probe: `detect_every=2`, detections at frames 0/2/4/6 with box x positions 0/2/4/6. Filled frame 3 correctly has x=3; frame 5 has **4.5 instead of 5**, and frame 7 has **6.75 instead of 7**. Existing tests use stationary players. Keep the last observed box and observation time separate from predicted state; handle missing observations and reacquisition explicitly. ByteTrack is only updated on detection frames, so the fill is custom extrapolation, not ByteTrack predicting on every video frame.

### F4 — Refitting can silently reverse home and away

**Reproduced at the assigner; pipeline consequence is a code finding.** `KitColorTeams.reset()` clears the fit but neither aligns new centroids to old identities nor invalidates `config.home_cluster`. `_enter_match()` retains that mapping after a long break. Cluster IDs are arbitrary.

Probe: repeatedly reset and fit the same ordered red/blue samples on the same assigner. Red/blue labels change from **[0, 1] to [1, 0]** on the next fit. With `home_cluster=0`, team identity reverses. Resetting the random seed alone would not solve changed sample ordering or distributions. Preserve verified kit identity across refits, or clear the mapping and require a fresh selection. Do not silently reuse the old cluster number.

### F5 — Writer has valid edge cases it cannot serialize successfully

**Reproduced.** These are separate fixes, not a reason to weaken validation generally.

| Input | Actual result | Contract/consumer consequence |
|---|---|---|
| Two frames with an object, `period=5` | Validator: `frames: 2 shootout (period 5) frames not marked dead` | Writer always writes null ball state (`writer.py:53`); 02 requires dead throughout period 5. |
| Two `other` frames and no detections | Validator: `objects: no rows` | Stage 0 can legitimately suppress every frame of a short clip, but the current validator requires a nonempty objects table. Proposed contract clarification in section 5. |
| Same empty-detection run | `detections.parquet` has **zero columns**; renderer's partition by frame raises `ColumnNotFoundError` | Cache frames must retain their specified column types even with zero rows (`writer.py:195`, `demo/debug.py:111`). |
| One frame containing an object | `TypeError` during velocity computation: division by `None` | No positive timestamp interval exists from which the shared helper can estimate FPS. A track's first velocity should be null. |

`close()` does not invoke the validator, so CLI success does not establish conformity. Zero decoded frames also produce invalid empty frames output. Define a clear no-input failure separately from a nonempty video with no usable objects.

### F6 — Period time and transitions are not implemented as specified

**Code finding.** `vision/run.py:99` supplies `frame_id / fps` as seconds since period start. That is clip-relative time for a mid-period excerpt, not 02's period clock. There is no period-start offset flag despite 03 saying the period start is configurable. A full-match input is assigned one period for its entire duration.

Mutating `config.period` externally does not reset the pipeline's tracks, ball velocity, homography, or gate timers. Resetting `t` at half-time can make old timestamps newer than the current frame and contaminate freshness checks. The writer's velocity helper does break by recorded period, but that does not reset upstream ball or tracking state. Specify either an explicit transition/reset API or single-period runs with a supplied offset and a supported assembly path. Do not silently merge files or restart frame IDs within one match. See Q2.

### F7 — Output bounds and visibility are not enforced

**Code finding; real frequency unverified.** `box_frac` divides raw/extrapolated box coordinates by image dimensions without clipping (`pipeline.py:191,310,332`). A player moving off screen therefore violates the display-only **0–1** box contract. `writer.py:102` writes `visible=True` for every row, including extrapolated positions demonstrably outside the camera footprint.

Do not equate every detector miss with off-camera status: occluded or skipped detections can remain in view. Determine visibility conservatively from valid geometry, or omit uncertain rows under 02's existing missing-player rule. Preserve `interpolated=True` for estimates. The fractional display-box correction needs no new game-state field. The raw pixel box remains cache-internal; live consumers should use only the fractional display box. Reject implausible projections rather than clipping all meter positions to the pitch: genuine out-of-bounds ball positions are needed by stage 8.

### F8 — Ball history can survive invalid geometry or elapsed gaps

**Reproduced helper behavior; policy partly undecided.** `_ball_object()` (`pipeline.py:286`) emits its cached meter-space extrapolation without checking current homography validity. A missing detection with an invalid mapping callback still returns the old position. With defaults, a ball observed near the end of a homography's valid lifetime can outlive that homography. This contradicts `VisionObject.x`'s null-without-valid-homography comment, but preserving a brief world-space estimate could be intentional. Resolve Q4 before changing that policy.

The expiry check occurs only in the no-detection branch. If calls jump from t=0 to t=3 with a new detected ball, a one-second gap limit still allows velocity to be calculated across the three-second gap; the next missing frame extrapolates from that velocity. Repeated detections that cannot be projected can also bypass history expiry. Expire/reset motion before processing reacquisition, and reset at period/camera discontinuities. The displayed box for an extrapolated ball remains the last observed box, even while its meter-space position moves.

### F9 — Configuration and provenance are incomplete

**Code finding.** `detect_every=0` is accepted by the CLI and later divides by zero; period values are unrestricted integers. FPS uses `reported_fps or 25`, which does not reject negative or nonfinite values. The fallback can also silently replace unknown timing. Validate cadence, durations, window sizes, period, and timestamps at the vision boundary.

The config says every threshold lives there, but model confidence is independently fixed at 0.3, both people and ball input sizes default to 1280, pitch input size is 640, RANSAC tolerance is fixed at 2 m, and track team votes are fixed at five. Lowering `min_det_conf` cannot recover detections already removed by the wrapper. The CLI cannot select separate image sizes or most tuning thresholds. `run.json` omits those effective model settings and dependency versions; it records total wall time, **not the per-stage wall times required by 03**.

## 3. Missing behavior, spec drift, and first-run risks

### Implementation gaps, distinct from defects

| Area | What exists | What remains |
|---|---|---|
| Team assignment | Lab shirt-color features and two-means; warmup and per-track votes | 03 still specifies SigLIP → UMAP → KMeans. This appears to be an intentional cheaper baseline, also named in the roadmap, but its >95% player-frame target is unmeasured. Goalkeepers use only mean **x**, not full 2D centroid distance. Decide/document before replacing it. |
| Ball detector | Optional dedicated weights; otherwise people-model ball detections; maximum-confidence ball chosen | Both detectors run at the same 1280 input size and shared cadence. No temporal candidate association, plausibility rejection, or separate high-resolution/tiled strategy. Interpolation must mean causal forward estimates, never later-frame gap filling. |
| Stage 7/live output | Velocities added only when the writer closes | `VisionFrame` has no velocity or inferred-state output. Streaming frame processing exists, but complete live game-state production still needs a design consistent with 02/03. |
| Stage 8 | Polygon projection only; other three fields always null | Phase 1 possession-rule implementation and provider comparisons are unchecked. No implementation was found in the inspected vision/shared/evaluation modules. No claim is made about uninspected prediction code; nothing there should be imported into vision. This is blocked on a confirmed shared, schema-only interface and owner. |
| OCR | Null IDs and empty players table | Number reading, causal voting, roster input/linking, and uncertainty handling are absent. This is planned and valid under 0.6. |
| Cache/replay | Final tracked detections, view rows, run metadata | Raw detections (including unmatched candidates), keypoints, transforms, and team-fit state are not cached. No cache-reader execution path reruns later stages. Therefore 03's “every stage can cache” requirement is not implemented. Any new cache shapes must be specified in 03 first; they are vision diagnostics, not new game-state fields. |
| Full-video writing | All rows accumulated as Python dictionaries until close | Unbounded memory growth and loss of unwritten output on interruption. At 25 fps, 90 minutes and 23 objects mean about 3.1 million object rows plus comparable cache rows before dataframe conversion. This is a sizing illustration, not a measured RAM benchmark. Bound memory before full-video/live use. |
| Debug renderer | Track/cluster rings, ball marker, projected object minimap | No pitch-keypoint overlay, projected landmark residuals, cluster-number legend, homography-health indication, or distinction for extrapolated ball markers. `--max-frames 750` inference followed by debug without `--frames` renders the rest of a longer source as apparently paused. Limit debug to the processed range. |

03's output header still says **v0.4** and omits `events.parquet`; the implementation emits all five tables at v0.6. Its description of points 31/32 as “on the halfway line” is geometrically wrong: `(-9.15,0)` and `(9.15,0)` lie on the longitudinal center axis; points 15/16 are on x=0. The numeric coordinates and current template agree. The roadmap's unchecked “Homography → pitch meters → game state writer” mixes implemented synthetic functionality with unverified real correctness; split those statuses after approval instead of checking it off wholesale. 09's suggestion that offline interpolation can use later frames conflicts with the global causal-only rule; all production paths must remain causal.

### First real run: most likely failure modes

1. **Weights/runtime integration before accuracy.** Actual class-name metadata, pitch tensor shape/order, confidence availability, CUDA/torchvision compatibility, and MPS support have never been exercised here. `YoloDetector` requires exact recognized class names; pitch fitting assumes 32 entries; absent keypoint confidence becomes all ones. The dedicated ball file is optional and silently changes detector behavior when absent. Check active filenames, hashes, classes, keypoint shapes/confidences, device, and memory before trusting a successful CLI exit. The upstream setup script requires `gdown` and Bash; Windows setup must supply those. The filenames match the [upstream setup script](https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/setup.sh).
2. **Plausible-looking but wrong pitch positions.** The [upstream vertex list](https://raw.githubusercontent.com/roboflow/sports/main/sports/configs/soccer.py) agrees with the code's landmark ordering after geometric normalization; this does not validate the weights' actual predictions or TV orientation. Four high-confidence points can be poorly distributed. `fit_homography()` ignores the RANSAC inlier mask for acceptance, imposes no error/conditioning/finite-output threshold, and averages matrix coefficients across camera motion. An accepted bad fit can therefore create large jumps, projection singularities, or objects more than the validator's 15 m margin outside the pitch. A small residual on fitted landmarks alone is not proof of unseen-position accuracy.
3. **Broadcast camera cuts that stay green.** Tracker/transform resets happen only after a gate transition. A direct wide-shot cut to another pitch camera, a brief close-up, or a replay may leave the gate in `match`, mixing previous fits and identities with the new view. The sparse-keypoint bug worsens this. Even grass-poor ads still receive detection during the intended 0.5-second grace period: the synthetic test explicitly calls the detector on ad frames 40, 42, and 44. “Detection never runs during the ad” is only true once the gate becomes `other`. Keypoint GPU work also occurs before the gate update on scheduled frames.
4. **Kit ambiguity and goalkeeper errors.** Green kits may be removed as grass; similar colors, shadows, multicolor shirts, partial crops, or only one team in warmup can produce an uninformative two-cluster fit. Five early votes then freeze a track. Nearest-team mean x is unreliable during corners or crowded penalty-area play. None of these failures is represented by solid red/blue synthetic rectangles. Long-break cluster reversal is already confirmed by F4.
5. **Ball misses, false positives, and airborne projection.** A distant ball can be a few pixels; maximum confidence can select a white advertisement or another ball. Homography assumes the ball lies on the ground, so airborne detections can project incorrectly while z remains unknown. Constant-confidence extrapolation has no speed or acceleration check. The [upstream ball example](https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/main.py) uses tiled inference and temporal ball handling; the current wrapper is not an equivalent reference implementation. Whether fine-tuning is needed remains an empirical question.
6. **Tracker lifetime and timing.** With supervision 0.27.0, the adapter's `lost_track_s=1` at 25 fps becomes only **20 tracker updates**. The [upstream constructor](https://raw.githubusercontent.com/roboflow/supervision/0.27.0/supervision/tracker/byte_tracker/core.py) scales the supplied buffer by fps/30, while the adapter already scaled it by fps. Detection skipping further changes elapsed wall time because updates run only on detection frames. The 0.3 confidence filter also removes the low-confidence candidates used in ByteTrack's second association pass. Measure ID continuity and missed-track duration, then migrate with equivalent time-based behavior; do not simply lift the `<0.31` bound.
7. **Throughput and durability.** Three models, two at input size 1280, may not meet either M1 short-clip comfort or the 2060 live budget. No GPU memory or per-stage timing was measured. Full-run in-memory writing and post-run rendering add CPU/RAM load. 09's performance/VRAM statements remain estimates, not acceptance evidence.

## 4. Proposed priority order for Phase B

The first workstation run remains first. A failed smoke run is useful evidence; fix only its demonstrated execution blockers before repeating it. Deterministic logic fixes can proceed on the M1 after this review is approved if workstation results are delayed, but real-data tuning must wait for those results. Each row may take several small commits; couple behavioral changes with their regression tests, sign and push each logical commit, and update 03/roadmap when behavior or status changes.

| Order | Concrete task and dependencies | Done when / evidence |
|---|---|---|
| 1 — user, RTX 2060 | Run current `vision.run` and `demo.debug` on a short real broadcast clip with Roboflow weights. Start with `detect_every=1`. Record commit, dependency versions, active model hashes, class/keypoint metadata, clip time/period, actual home direction, FPS and peak memory. | Save all five tables, both caches, `run.json`, validator output, logs, and a bounded debug excerpt. A failure is recorded as a failure; no quality checkbox is inferred from process exit. |
| 2 — same run | Check orientation and pick `home_cluster`. Inspect more than one shot, including midfield and a penalty area. Rerun with the chosen mapping and a new match/run ID. | Independently clicked center spot maps near `(0,0)`; penalty spots near `(-41.5,0)`/`(41.5,0)` in TV coordinates; verify near/far y sign, left/right x sign and the optional 180° turn. Identify which actual club each cluster represents. Check post-cut behavior and note that long-refit mapping remains unsafe until F4 is fixed. |
| 3 — M1, deterministic fixes | Address F1/F5 first: strictly causal vision velocities, first-frame handling, period-5 semantics, typed empty caches, validation/failure reporting. Apply the no-objects policy only after section 5/Q5 is approved. | Prefix output stays unchanged when future timestamps/positions are appended; starts/gaps/period boundaries have null velocities; one-frame, zero-input, all-other and all-no-homography cases have explicit tested outcomes. Existing suite remains green. |
| 4 — M1, pipeline state | Fix F2/F3/F4/F6/F7/F8: sparse keypoint evidence, observed-versus-filled motion, refit identity mapping, period/clock transitions, bounded display boxes/visibility, and ball history reset. Resolve Q1–Q4 first where required. Add rejection checks for invalid config and malformed/nonfinite stage output. | Fake-stage tests cover moving tracks at skips 1/2/3, actual sparse keypoint cadence, long breaks, reordered kit samples, off-screen boxes, irregular timestamps, stale homography, reacquisition, and transitions. No historical frame or object gets rewritten. |
| 5 — M1/2060, diagnostics and reference | Specify effective model settings and per-stage timings in 03, then expose/persist them. Add the minimum keypoint/transform evidence and cache replay needed to diagnose real failures. Run Roboflow/sports end to end on a SoccerNet sample in a separate environment, pinned to a recorded revision. | Compare same-frame boxes, ball candidates, pitch points, tracks and kits with this pipeline. Treat upstream's whole-video team fitting as an offline diagnostic reference only, never production behavior. Document coordinate differences instead of comparing raw radar numbers. Debug output clearly identifies cluster numbers and processed range. |
| 6 — real calibration clips | Harden homography acceptance and camera discontinuity handling; then tune stage 0 using labeled ads, studio, crowd, green close-ups and normal wide play. Use `view.parquet`; replay revised gate logic causally or rerun where changed sampling requires new keypoints. | Report false-match duration, false-other duration, switch delays, accepted-transform coverage, landmark errors and failures around cuts. Use separate matches for tuning and checking thresholds. Do not tune around F2 or accept a homography solely because it has four points. |
| 7 — tracker migration | Replace `sv.ByteTrack` behind the injected tracker interface, before relaxing the dependency pin. Select the replacement after comparing documented API behavior and the real reference clip. | Synthetic ID continuity, detection skips, wall-time lost-track expiry, class handling, reset/ID uniqueness and real cut/occlusion samples pass. Bound/refactor writer memory before advancing from short clips to full-video runs. Keep MPS/CPU tests small. |
| 8 — ball improvement | Measure ball precision/recall, localization error and gap lengths on labeled visible/occluded/off-camera/airborne segments. Add candidate association, realistic causal motion gating, separate resolution/cadence or tiling as justified by results. | Detected and extrapolated metrics are reported separately; short-gap estimates stop at the configured age and reset across discontinuities. Decide fine-tuning only after establishing whether misses arise from scale, association, geometry, or the detector. Train on the 2060; use Colab only if measured VRAM requires it. |
| 9 — stage 8 dependency | Confirm the Phase 1 owner's implementation/status and agree where reusable meter-only inference lives without a prediction import. Then integrate causal state updates at the specified 10 Hz/time threshold, not “three native frames.” | Provider comparisons meet the starting ≥90% possession and dead-or-null targets on approved data. Reset/unknown rules cover cuts, no homography, >2 s unseen ball, and period starts. SkillCorner/IDSSE availability and the untouched final IDSSE holdout policy must be resolved with the Phase 1 owner. No prediction or converter work is part of this review's implementation authorization. |
| 10 — OCR | Add optional roster input, low-rate number reading, causal multi-frame voting and confidence-based identity attachment after team/track stability. Keep M1 validation to small clips; async work must have a defined availability time. | Wrong/ambiguous numbers remain null; a later identity never backfills earlier output. Object IDs remain independent of player IDs; IDs refer to roster rows. Test synthetic delayed votes plus a labeled real jersey sample. |
| 11 — formal GSR evaluation | After orientation is established, assemble a small representative GSR validation sample early; run the full acceptance evaluation after the preceding correctness/quality changes. Split by match. | Report player position error distribution and the fraction within 2 m, coverage/missingness, team accuracy >95%, ID continuity and GS-HOTA secondary metric, with ball and camera-shot breakdowns. “Most” in the position target needs an agreed fraction before signoff. Show both all player-frames (null counts as unavailable/incorrect) and conditional labeled accuracy so warmup cannot hide poor coverage. |

Suggested workstation commands, **for the user to run**, after setup in 09 and availability of the clip/weights. Replace paths and direction/period arguments with known values. This command does not repair the current clip-relative timestamp limitation.

```sh
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
uv run python -m vision.run --video <clip.mp4> --weights-dir <weights-dir> --match-id vision_smoke_01 --device cuda --detect-every 1 --max-frames 750
uv run python -m gamestate.validate data/gamestate/vision_smoke_01
uv run python -m demo.debug --video <clip.mp4> --cache data/vision_cache/vision_smoke_01 --out vision_smoke_01_debug.mp4 --frames 0-749
```

Expect CUDA availability and `sm_75` support; report the actual result. Keep video and outputs outside synced folders per 09. Use a new output ID for subsequent parameter runs because the writer overwrites matching output paths. For later M1 real-model checks use `--device mps`, small models and short clips; current CLI image-size configuration remains a gap.

## 5. Proposed schema edit — not applied

Most fixes fit **v0.6 without new columns**: causal velocities, unknown state as null, missing object rows, interpolation flags, meters, and roster links already have representations. Do not add replay labels, detection pixels, velocity provenance, or homography-health columns to game state as part of a bug fix. Put operational evidence in the vision cache after specifying it in 03.

One contract clarification is recommended for the legitimate no-usable-objects run (F5). The table prose does not explicitly ban it, but `gamestate/schema.py` currently allows only events and players to be empty, so a versioned relaxation is preferable to silently changing validation. The following is the proposed text for **`Docs/Specs/02-game-state-schema.md`**, held here for approval to preserve Phase A's single-document scope:

> Under `objects.parquet`, before the column table:
>
> May be empty (0.7): a nonempty vision input can contain no usable match view or no valid projected detections. Preserve a `frames` row for each processed frame and write all required object columns with their declared types even when there are zero rows. Do not synthesize objects to satisfy a row-count check. An empty objects table does not establish vision quality or make the sample eligible for prediction/evaluation.
>
> Under `events.parquet`, before the column table:
>
> May be empty when the producer cannot annotate events, including vision. The requirement to represent shots and goals applies when event annotations are available; an empty vision events table is not evidence that no shots or goals occurred.
>
> Replace the final version declaration with `Schema version: 0.7` and prepend:
>
> `0.7: explicitly allows a typed empty objects table for nonempty vision runs with no usable projected detections, and clarifies empty vision events. No columns change; existing 0.5/0.6 data stays valid and needs no reconversion. Empty frames remain invalid.`

The events clarification aligns the document with the existing validator and tests; it does not introduce an event detector. If approved, edit **02 first**, then coordinate the minimal shared schema/validator change and tests. That shared-package change is outside a strict vision-only code boundary and needs explicit inclusion in Phase B scope. Alternative: retain v0.6, mark an all-empty run unsuccessful while preserving diagnostic caches, and do not claim it passes acceptance. Neither choice warrants fake object rows.

## 6. Decisions requested before implementation

1. **Gate recovery and camera cuts (Q1):** may stage 4 run a low-rate recovery probe while the gate is `other`, or must recovery remain grass-only until a provisional match entry? Is retaining detections through the specified 0.5-second off-delay intentional? Recommendation: allow a documented recovery probe and explicitly define what gets emitted during provisional entry/cut handling. Keep earlier emitted frames unchanged.
2. **Period scope (Q2):** should Phase B support period transitions within one pipeline, or explicitly limit the CLI to one known period with a required clip-time offset? Recommendation: implement an explicit clock/reset contract for streaming, with a simple offset for single-period CLI excerpts. Never infer a game's period clock from video elapsed time alone.
3. **Team policy (Q3):** is Lab color intended as the supported baseline despite 03 specifying SigLIP, and should an ambiguous refit invalidate the home mapping? Recommendation: document color as the baseline, preserve known kit identity only with adequate evidence, otherwise return null until reselected. Measure before adding a heavier model. Does 03's “nearest team centroid” mean full 2D distance, or is the current x-only distance intentional?
4. **Lost-ball policy (Q4):** should meter-space extrapolation continue briefly after the current homography becomes invalid, provided the last observation was valid, or must all such positions disappear immediately? Recommendation: withhold output across detected camera/period discontinuities; make any permitted same-view fallback explicit in 03 and keep stage 8 unknown when homography is invalid. Extrapolated display boxes must not imply fresh visual evidence.
5. **Empty-run contract/scope (Q5):** approve the proposed 0.7 relaxation and the minimal shared schema/validator follow-up, or keep such runs as explicit unsuccessful v0.6 outputs? This choice is needed before the corresponding acceptance test is changed.
6. **Stage 8 ownership and validation (Q6):** has the Phase 1 implementation landed somewhere outside the inspected scope, and who owns its shared interface and provider validation? The roadmap says both implementation and comparisons are still open. IDSSE's final-only evaluation rule also needs reconciliation with 03's instruction to tune/check inferred state against IDSSE; do not open the final holdout to settle this.
7. **Replay/product policy and evaluation target (Q7):** are replays temporarily acceptable only for diagnostics, or also for the live danger meter? What fraction does “most visible players within ~2 m” mean for acceptance? Recommendation: retain the current replay behavior only as an explicitly untrusted diagnostic baseline; require causal replay suppression/unknown handling before claiming live prediction quality. Agree the position coverage target before GSR evaluation.

### Opinions on 03's remaining open questions

- **Ball fine-tuning:** undecided until order 8's measured failure breakdown. Fix scale/tiling, temporal association, and geometry errors before attributing all misses to pretrained weights.
- **Replays:** not a leak from future input frames in the current video-processing sense, but semantically stale match action that can generate false live danger and contaminate track/possession state. A causal replay-transition detector or classifier is justified if replay intervals materially affect the intended demo. Offline labels can score it but must not be fed into production decisions ahead of their availability. Specify behavior in 03 before implementation.
- **Throw-ins/goal-kicks:** stage 8 does not exist here, so there is no current rule performance to endorse. First integrate and evaluate the specified meter-only rules, recording off-camera unknown coverage separately from correct dead-state decisions. A small classifier becomes a concrete task only if labeled restart segments show a material failure; do not substitute a confident guess for null.

**Approval boundary:** review this priority order and the decisions above. No Phase B source or test changes, workstation inference, schema bump, or roadmap completion marks have been made.

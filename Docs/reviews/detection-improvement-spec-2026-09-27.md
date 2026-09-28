# Detection improvement spec — from smoke test to reliable game state

Review date: 2026-09-27. Reviewed baseline: `2f297e43bd37e517b74cf97640146675b89391c6`. Status: proposed implementation plan; this change adds only this document. Recheck findings against the implementing branch: other agents are working concurrently, including a separate vision worktree. A task already fixed should be verified against its acceptance checks, not implemented twice.

**Concurrent-work update:** before this review landed, main advanced to `19acb1d`, adding `prediction/features.py`, `prediction/floor.py` and `prediction/cv.py`. Read those additions: the distance/angle logistic floor and grouped-CV driver now exist. The baseline inventory below describes the pinned revision; do not rebuild that floor. W9 can use the new implementation once its run/results are available. No new prediction-quality claim is made here.

**Recommendation:** build a measured, causal, broadcast-aware game-state pipeline before buying accuracy through larger models. The next milestone is reliable ball/player positions, teams, timing and possession on unseen broadcasts, with explicit missingness and a measured latency budget. Another attractive minimap, a clean validator result, or higher detector mAP alone does not establish that milestone.

## 1. End goal and present scope

The user's intended end goal is **anticipate goals, then use those predictions in a future Polymarket trading system**. The current numbered specs stop at a danger-meter demo and explicitly defer betting. Treat the user's clarification as the longer-term product direction; quant models, market selection, market data, execution and trading are outside this implementation plan.

That direction changes the order of vision work. A late, replay-derived or wrong-team observation can be worse than a missing observation. Useful outputs must describe current play, preserve when the information became available, and expose when the pipeline cannot tell. A probability of a shot or goal in the next five seconds is not itself a probability for an unspecified market contract. That mapping belongs to a later project phase.

The planned prediction path is `P(shot within H) × current-position xG`, with H = 3/5 seconds. This is an approximation described in 05, not a proved goal probability identity for all match situations: future shot locations, multiple shots and non-shot goals complicate it. Vision should support evaluation of that approximation without taking over the prediction model. Do not add goal recognition or market logic as a substitute for recovering good game state.

### How the repository fits together

```text
PFF / Metrica tracking + events ── converters ───────────┐
                                                      v
Broadcast ── gate ── people/ball detection ── tracking ── game state (02)
                       + teams + pitch mapping         |
                       + state inference [missing]     v
                                           causal 10 Hz data + labels
                                                      |
                               floor → LightGBM → GNNs + xG [planned]
                                                      |
                                   grouped evaluation → danger meter
                                                      |
                                     future market system [out of scope]
```

| Component | Implemented at the reviewed revision | Meaning for this plan |
|---|---|---|
| `gamestate/` | Five Parquet tables, schema 0.6, validator; 105 × 68 m, centered origin, fixed match coordinate frame | Preserve the common contract; pixels stay in vision diagnostics/display output. Validity is not accuracy. |
| `converters/` | PFF raw tracking/event parsing, visibility, causal position differences, extra-time exclusions, set-play normalization; Metrica games 1–2 | Useful development and provider-comparison inputs. PFF estimates and possession annotations are imperfect references, not independent optical truth. |
| `vision/` | Stateful frame API; grass/keypoint gate; YOLO wrappers; ByteTrack on detection frames; custom skipped-frame extrapolation; Lab kit clustering; homography acceptance; ball extrapolation; writer/caches | A functioning prototype with real smoke runs. No local detector-training pipeline or trained replacement weights were found. |
| `prediction/` | 10 Hz resampling and shot/goal labels | No logistic floor, LightGBM, GNN or xG training implementation yet. |
| `evaluation/` | Frozen match folds, inner splits, probability/alarm metrics, threshold selection, run storage and reports | Preserve this harness. Vision quality/abstention needs its own report before downstream model comparisons exist. |
| `profiles/`, `demo/` | Profiles placeholder; cached debug renderer | OCR/profile work can wait; debug needs better diagnostics. The prediction overlay is still planned. |

The current writer sets `ball_state`, `possession_team`, and `ball_carrier_id` to null on every frame. Thus current vision data has **no eligible training rows under 05**, even if every detected player is positioned correctly. Empty vision events also mean “unannotated,” not “no shots occurred.” Do not generate training negatives from an unannotated vision run after state inference is added.

## 2. Evidence and limits

Read the repository instructions, README/data guide, all numbered specs, roadmap, references, three previous reviews, smoke runbook, all implementation modules, and relevant test fixtures/coverage. Inspected the external primary references linked below. Did not run new real YOLO inference, train weights, access the RTX 2060, evaluate SoccerNet or re-watch the workstation footage. “100% understood” cannot establish accuracy on unseen broadcasts; the benchmark below is how to resolve those unknowns.

Verification during this review:

- Local full suite: `.venv/bin/python -m pytest -q` → **271 passed, 5 skipped**. This ran in the primary checkout while concurrent evaluation commits were landing; it is supporting evidence, not an immutable release benchmark.
- At the pinned review revision, vision/schema tests in an isolated environment → **113 passed**, no skips, one ByteTrack deprecation warning. Environment: OpenCV 5.0.0, supervision 0.30.0, Polars 1.44.2. The ordinary local environment lacks OpenCV; four vision modules otherwise skip entirely. These tests exercise fake models and real ByteTrack on synthetic boxes, not YOLO weights.
- In-memory probes reproduced the ball-history issue and a future-cadence dependency in resampling (section 3). No source or test changes were made to reproduce them.

Focused test command (use an available Python 3.11 interpreter):

```sh
uv run --no-project --python 3.11 \
  --with opencv-python --with polars --with pytest --with supervision==0.30.0 \
  python -m pytest -q -rs tests/test_vision_pipeline.py tests/test_vision_pitch.py \
  tests/test_vision_stages.py tests/test_vision_view_gate.py tests/test_validate.py
```

### Existing smoke evidence, correctly interpreted

The [first smoke review](smoke-test-2026-09-27.md) and [follow-up](smoke-test-2026-09-27-followup.md) describe the first 750 frames of a Spain–Japan clip: 25 seconds of 1080p30 video, with 375 frames classified as match view. The follow-up reports:

- Off-pitch rows beyond the validator margin fell from 36 to zero.
- Homography availability fell from 375/375 to 310/375 match frames; 75 projections were rejected.
- Kit separation improved on this clip; home was cluster 0 in smoke04, versus cluster 1 under the older fit.
- Runtime was 111.3–122.7 seconds for 25 seconds of video, about 6.1–6.7 input fps, including cheaper non-match frames.

This supports acceptance/rejection and kit-color fixes. It does **not** measure player recall, meter accuracy, ball recall, replay rejection or general team accuracy. The 83% homography figure uses the pipeline's own gate as denominator; the future benchmark must use independently labeled usable frames. Throughput is about 4.5–4.9 times slower than source time for these runs, not a live measurement.

Already addressed from the older review: retained sparse keypoint evidence and recovery probes, ByteTrack low-score input/buffer scaling, detection-to-detection box motion, centroid alignment on refit, FPS-based vision velocity window, typed empty caches, single-frame writing, config guardrails, display-box clipping and homography rejection. Keep their tests. Retain the current decision that all-empty output is an unsuccessful run; no schema 0.7 proposal is necessary for this plan.

## 3. Remaining findings that determine priority

“Code” means directly visible behavior; “reproduced” means exercised during this review; “risk” means real prevalence or accuracy is unmeasured. File/function references refer to the pinned revision.

| ID | Evidence and location | Consequence / required work |
|---|---|---|
| D1 | **Reproduced:** `VisionPipeline._ball_object` checks age only on the no-detection branch. Observations x=0 at t=0 and x=30 at t=3 establish vx=10 despite `ball_max_gap_s=1`; a missing observation at t=3.1 emits x=31 even with an invalid projection callback. | Reset before reacquisition and suppress meter output when geometry is invalid. Existing off-pitch rejection catches some symptoms, not this history error. W2. |
| D2 | **Code:** `step` picks the highest-confidence ball candidate. Both detectors default to 1280 and share `detect_every`; no candidate association or separate ball cadence. | High-confidence distractors can replace the ball, and increasing player skip also reduces ball observations. Actual frequency is unknown. W2/W7. |
| D3 | **Code:** segment/tracker resets occur on `other → match`, while `HomographyFilter` can accept a changed camera without notifying the tracker or clearing team/ball motion history. Replays with grass/keypoints pass. | A valid replacement transform does not establish identity continuity or live-play status. Detect cuts separately from view usability. W3. |
| D4 | **Code/risk:** homographies are coefficient-averaged; acceptance uses fit-landmark residuals and broad pitch bounds. Image corners are projected without polygon validity checks. | In-bounds positions can still be wrong; panning/zooming can create lag and false velocities. Four fitted points are not independent accuracy evidence. W4. |
| D5 | **Code:** `KitColorTeams.features` discards lightness, removes green pixels, and assigns any usable crop to a cluster. `_update_teams` freezes votes after five; `_goalkeeper_teams` uses nearest mean x. | Similar-chroma kits, green kits, one-team warmup, early mistakes and crowded goalmouths need explicit uncertainty handling. Red/blue smoke success does not cover them. W5. |
| D6 | **Code:** `GameStateWriter.add` writes `visible=True` for every object; skipped-frame boxes can move outside the image. `VisionFrame` has neither velocity nor possession outputs; velocity is computed only at close. | Visibility can disagree with 05's broadcast-view filtering, and streaming and saved state are incomplete/different interfaces. W6. |
| D7 | **Code:** CLI time is `frame_id/fps`, `config.period` is fixed, and direction is `period % 2 == 1`. PFF conversion instead handles a separate extra-time starting direction. | Mid-period clips have the wrong clock; period changes lack a reset contract; periods 3/4 cannot safely rely on odd/even direction. Explicit metadata is needed. W6. |
| D8 | **Reproduced:** `prediction.resample.native_interval_us/build_grid` derive allowed staleness from all timestamps. `[0,.04,.08,.30]` emits grid points `[0,.1,.3]`; append `.31,.32,…,.50` and the same prefix emits `[0,.3]`. | Future cadence changes whether a past row exists, despite the causal spec. Current regular-rate PFF results are not shown to be affected. Fix before reusing this path for irregular live input. W6 handoff. |
| D9 | **Code:** the writer accumulates Python lists until close; caches contain post-tracker rows, not raw rejected candidates/keypoints/transforms. `run.json` records total time, not stage timing/effective model settings. | Full-match memory is unbounded; failure diagnosis and cheap ablation are limited; throughput cannot be attributed to a stage. W1/W7. |
| D10 | **Code/spec:** stage 8 is absent, 05 permits unknown ball state at inference, and alarm nulls can hold an existing alarm with no expiry. | Unknown, stale and replay states need a consumer policy; “hold the display” must not become “emit a fresh signal.” Separate model evaluation from operational freshness. W3/W6/W9. |

Additional interface checks belong with W6: finite, strictly increasing stream timestamps within a period; explicit reset on seek/reconnect; no velocity across an elapsed-time gap merely because frame IDs are adjacent. The shared velocity helper breaks by frame-ID gaps/periods, which does not fully describe irregular capture gaps.

## 4. Benchmark first: define what improvement means (W0)

Create a **vision benchmark independent of the predictor's frozen folds**. Do not reshuffle `data/splits/folds.json`, tune on IDSSE, or split neighboring frames from one match into train and validation. If a later paired predictor experiment uses these matches, its outer held-out matches must also be excluded from vision tuning for that experiment, or the result must disclose that overlap.

Proposed initial size: 30–50 clips of 20–60 seconds from at least 8 matches, selected across multiple camera/kit conditions, plus at least one uninterrupted 10–15 minute stream for latency and recovery. This is a bootstrap evaluation set, not sufficient evidence for every competition. Split whole matches into development and a locked check set before tuning; keep source identifiers so duplicate highlights/replays cannot cross splits. Use official SoccerNet partitions where applicable and check source-match overlap where metadata permits. Already-inspected smoke footage belongs to development.

Include wide midfield, penalty-area attacks, tiny distant ball, fast passes/crosses, occlusion, crowded boxes, near/far touchlines, pan/zoom, direct green-to-green cuts, short close-ups, replay transitions and replay footage, ads, shadow/sun, dark/light and green kits, keeper kit clashes, missing ball and an out-of-play ball. Include ordinary non-shot play so false positives have a real denominator. Keep a separate stress subset for shot lead-ups; do not report its oversampled prevalence as ordinary broadcast prevalence.

Annotation and storage requirements:

1. A committed manifest, proposed `data/splits/vision_benchmark.json`: benchmark version, source/match/clip IDs, split, video hash, PTS/frame ranges, period/time mapping, scenario tags, annotation version and provenance. Follow existing data storage/access rules; keep footage and detailed annotations gitignored.
2. Independently label live/replay/other/uncertain intervals and cuts. A replay may be a geometrically excellent match view, so use separate labels for visual usability and live status.
3. For sampled frames: people boxes, class/team, foot anchor, visible/occluded/truncated state, pitch position where defensibly annotatable; ball center/box and visible/occluded/off-screen/airborne/uncertain state. Double-check a subset and record annotation disagreement.
4. For continuous bursts: identities, ball gaps, recovery and possession transitions. Sparse stills cannot score tracking continuity or time-to-recover. Ball position on the ground must not be invented from its airborne image location.
5. Independent pitch landmarks/line points beyond those used to fit a transform. Use SoccerNet pitch annotations for people where appropriate; its athlete GSR task does not replace a dedicated ball benchmark. Official GS-HOTA includes role, team and jersey identity, so report it as a secondary identity-aware metric while OCR is absent. [SoccerNet GSR documentation](https://github.com/SoccerNet/sn-gamestate#-about-gs-hota-the-evaluation-metric-for-game-state-reconstruction).

### Required scorecard

Compute metrics per clip/match and scenario, then aggregate with counts. Use match-level bootstrap intervals when sample size supports them; adjacent frames are not independent trials. Freeze matching tolerances and thresholds before the locked check. Pair every quality score with coverage and false positives so abstaining on everything cannot win.

| Layer | Metrics and denominator |
|---|---|
| View/live status | False-live seconds per hour of annotated non-live video; missed usable-live seconds; time to suppress/recover at each transition; total interval counts. Report replay separately. |
| People detection | Per-class AP/precision/recall; small/occluded-player recall; foot-anchor pixel error. Score keeper/referee confusion separately. |
| People in meters | One-to-one matched median/p90 error; **fraction of all annotated visible person-frames with a correct output within 2 m**, counting missing/rejected rows as failures; unmatched predictions separately. Also show conditional error and geometry availability on annotated usable frames. |
| Tracking | ID switches, fragmentation, IDF1/HOTA as available, reacquisition time, and 2–3 second window continuity. Score within camera segments; resetting identity at a cut is expected, not a tracking failure. |
| Teams | Accuracy on assigned player-frames, assigned coverage over all visible player-frames, correct-team coverage including misses/warmup, time to fit, and whole-team inversion count. Keep outfield and goalkeeper scores separate. |
| Ball | Visible-ball precision/recall and pixel-center error normalized by annotated ball size; false-ball time; ground-position error only with valid ground truth; gap-length distribution and recovery delay. Score detected and extrapolated output separately, including extrapolation error versus age. |
| Geometry/motion | Independent landmark error, accepted coverage, rejection reasons, pan/zoom lag, and apparent speed of stationary landmarks. Player velocity error only where reference cadence/positions justify it. |
| Inferred state | Possession confusion/coverage, turnover delay, carrier precision/coverage, alive/dead/unknown confusion and unknown duration. Provider agreement and independently labeled truth are separate reports. |
| Operations | Per-stage p50/p95/p99 time, usable-live-only throughput, frame age at output, dropped frames, queue age, peak RAM/VRAM and cache volume. |
| Goal-prediction relevance | Coverage and longest missing interval during 5–3, 3–1 and 1–0 seconds before a shot; usable 2–3 second history windows; errors involving carrier, closest defenders and keeper. All designated shot windows remain in the denominator. |

Proposed initial acceptance targets, to freeze after checking annotation uncertainty on **development only**: ≥90% of visible player-frames correctly localized within 2 m; >95% outfield team accuracy on assigned frames with ≥90% correct-team coverage overall; visible-ball precision ≥95% and recall ≥90% on annotated usable live frames. These are new engineering targets, **not achieved results or published standards**. Report keeper and hard-scenario results even if they miss them. Do not invent a meter target for airborne balls. Before live promotion, set a replay false-live/time-to-suppress budget using the benchmark; missing replay labels or an unmeasured budget means the live-status gate is unfinished.

## 5. Work packages other agents can implement

Each package must include a baseline comparison, meaningful regression tests for changed behavior, effective configuration, and a short results note under `Docs/reviews/`. Update the numbered spec before changing its public contract. New diagnostic formats below are proposals to specify in 03, not silent additions to 02.

### W1 — Record enough evidence to improve the models

Files: `vision/run.py`, `vision/config.py`, `vision/writer.py`, new vision diagnostic/cache helpers; `demo/debug.py` owned by the same integration lane or a separate non-overlapping follow-up.

- Persist active people/ball/pitch model hashes and class maps, separate image sizes/cadences, detector/NMS thresholds, tracker settings, precision/backend, dependency versions, seed, git revision/dirty flag, clip identity, source cadence and frame dimensions. State explicitly when dedicated ball weights are missing and the people model is being used.
- Add raw detections before tracking/selection, keypoint coordinates/confidences, fit/inlier diagnostics, accepted H and its age, camera segment, decision/rejection reasons and ball observation age. A per-frame diagnostic row must exist even when there are no tracked objects. Keep these vision-internal and version the cache.
- Cache model evidence for deterministic downstream reruns with a sequential replay reader. A new sampling schedule requiring uncached inference must fail clearly or rerun inference; it cannot silently reuse missing samples. Log team-fit provenance and home mapping.
- Time decode, each model, projection/tracking/teams, state, enqueue/write and end-to-end output. GPU timing must account for asynchronous work; report profiling overhead separately from normal operation.
- Debug rendering stops at the processed range and distinguishes detections from extrapolations, unknown teams, rejected geometry, replay/uncertain status and stale state. Add landmark overlays and a cluster-to-team legend.

Done when: an alternative ball association or geometry threshold can be replayed without rerunning unchanged models; results match a direct run under the same settings; all rejection/missingness cases remain inspectable. Persist both stage cost and output age, not just FPS.

### W2 — Make the ball a real temporal track

Files: new `vision/ball.py` (proposed), detector/config interfaces, integration into `VisionPipeline`. Depends on W0/W1 for measurements; D1 regression fixes can land first.

- Expire history **before** processing a new observation. Reset on expired observation, camera segment, period change and invalid geometry; require new evidence before estimating velocity after reset. A detected image-space ball may remain in diagnostics while pitch x/y are unavailable.
- Associate candidates using observation age, detector score and causal motion uncertainty. Gate obvious jumps, while allowing fast kicks and changing direction; do not enforce one rigid “maximum pixel speed” across zoom levels. Keep candidate rejection reasons and a full-frame reacquisition path so a bad region-of-interest cannot lock out the real ball.
- Start with a simple alpha-beta/Kalman-style motion baseline and compare against the existing max-score rule. Keep image-space and pitch-space uncertainty separate. Camera motion must not be interpreted as world-space ball velocity.
- Separate people and ball cadence/resolution. Compare full-frame detection, overlapping tiles and a predicted search region with periodic full-frame refresh. Tile coordinates must map back to original pixels, with duplicate suppression. The [upstream Roboflow example](https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/main.py) provides a tiled ball baseline and temporal candidate handling, not evidence it meets this project's live budget.
- Bound extrapolation by time and uncertainty. Store observation age/estimate quality in diagnostics; a reused detector score is not confidence in an extrapolated position. Mark estimates `interpolated=True`. Projecting an airborne ball through a ground-plane homography does not recover its ground location or height; retain z=null and avoid asserting a carrier from this alone.

Done when: long-gap reacquisition, distractor with higher score, shot-speed motion, occlusion, invalid H, cuts and off-pitch ball all have explicit causal outcomes. Same-clip ball recall/localization improves without hiding false balls or extending unsupported tracks. “Grounded versus airborne” remains uncertain unless supported by evidence, never inferred merely from a plausible pitch coordinate.

### W3 — Separate camera cuts, usable view and live play

Files: proposed `vision/scene.py`, `vision/view_gate.py`, coordinated pipeline integration.

- Keep the cheap grass/keypoint gate as a baseline. Add a causal cut signal using low-cost image change and available geometric evidence, evaluated against pans/zooms. Reset identity/motion state on confirmed cuts even when both sides remain green and usable.
- Do not treat a rejected fit as a confirmed cut: noisy keypoints and ordinary camera motion are different cases. Cut confirmation and geometry reacquisition must have distinct reasons and timers.
- Add live/replay/uncertain status in vision diagnostics/runtime metadata. Begin with labeled broadcaster transitions plus a causal appearance/motion baseline. A scorebug freeze or missing graphic alone is insufficient. Train a small classifier only if simpler evidence cannot meet the frozen benchmark budget.
- During uncertainty or replay, emit no fresh prediction-eligible state; preserve frame/time continuity and reset temporal state when returning to live. Any displayed held value must be marked stale with its age. Offline replay labels may score decisions but must never be fed ahead of the time production could know them.

Done when: tests cover green-to-green cuts, replay without a transition graphic, sub-0.5-second close-ups, false cut candidates and return to live. No ball velocity, team vote or carrier survives a confirmed discontinuity incorrectly. Measure suppression/recovery delay, including frames emitted before a transition can be recognized; do not erase those retrospectively.

### W4 — Improve geometry where attacks happen

Files: `vision/pitch.py`, keypoint diagnostics; coordinate pipeline integration through one owner.

- Retain current RANSAC/inlier/freshness/off-pitch checks. Add spatial support and degeneracy checks, finite homogeneous denominators, and valid projected-polygon checks; a finite 3×3 matrix or four confident landmarks alone is insufficient.
- Evaluate error at independent field points/lines and actual people anchors. Separate keypoint error, camera mapping error and bounding-box-foot error. Tightening the off-pitch bound cannot fix a wrong projection that stays on the field.
- Measure current coefficient averaging against no smoothing and a causal landmark/camera-parameter smoothing alternative. Select by held-out position error **and motion lag**, not visual smoothness. Test pan/zoom bursts and sudden cuts.
- Use shorter refresh intervals when supported camera-change evidence warrants them, with recorded latency cost. Preserve the 02 orientation, including near/far y, across allowed camera views. A reversed camera needs explicit orientation evidence or unavailable coordinates.
- If keypoints are insufficient, evaluate a line-based calibration reference in a separate pinned environment before integrating a heavier model. SoccerNet's baseline exposes several calibration alternatives; its availability is a comparison opportunity, not a requirement to replace the current pipeline. [SoccerNet baseline](https://github.com/SoccerNet/sn-gamestate).

Done when: p90 error and correct-position coverage improve together on the locked check; stationary landmarks do not acquire material apparent motion during camera movement; polygon failures produce null rather than malformed footprints. Keep legitimate near-boundary/out-of-play coordinates within the supported margin.

### W5 — Teams, keepers and player-track continuity

Files: `vision/stages.py`, proposed team/tracking helpers; coordinate edits with the pipeline owner.

- Keep Lab clustering as the measured cheap baseline. Compare robust color features including controlled lightness with an embedding approach only on demonstrated hard kits. Green masking must not delete an entire team's shirt evidence.
- Sample warmup across distinct tracks and quality crops; many correlated crops from one player are not evidence of two teams. Require cluster separation/support and allow unknown. Bound the sample reservoir if a fit never becomes possible.
- Replace permanently frozen five-vote labels with confidence-weighted, conservative updates and ambiguity rejection. Keep known cluster/home identity through a valid refit; invalidate an ambiguous mapping instead of silently swapping teams. Cross-run cluster numbers remain arbitrary.
- Keeper assignment needs explicit kit/direction/temporal evidence. Nearest outfield centroid, whether x-only or 2D, is a baseline to test, not sufficient truth during an attack. Measure keeper quality separately because xG depends on it.
- Retain ByteTrack's low-confidence association path; test player/referee/keeper class flicker, partial occlusion and 2–3 second identity continuity before changing trackers. The [ByteTrack paper](https://arxiv.org/abs/2110.06864) motivates associating low-score detections rather than discarding them. Migrate the deprecated adapter only behind equivalent reset, identity and elapsed-time tests.

Done when: dark/light, green, similar-color and single-team warmup cases are measured; wrong-team freezes and inversions are eliminated in deterministic tests; real team accuracy/coverage and ID continuity improve together. Do not introduce full-video clustering or backward relabeling.

### W6 — Complete the causal game-state handoff

Files: `vision/types.py`, `vision/pipeline.py`, `vision/writer.py`, `vision/run.py`; one shared meter-only state implementation agreed in 01/03; a separate coordinated change to `prediction/resample.py` and its tests for D8. Vision must not import prediction.

- Add an explicit period clock/offset and per-period attacking direction. Keep one known period per offline run initially; make seek/reconnect/period transitions explicit resets. Do not mutate config midway through a run as a substitute for a transition API. Periods 3/4 require known direction, not a parity assumption.
- Distinguish source PTS, period time, monotonic capture time, inference completion and emission time in the runtime/diagnostic envelope. 02 retains period time in seconds; do not mix wall-clock timestamps into it. Version 02 first if any shared table columns must change.
- Compute causal velocities and state incrementally once; have offline writing and live consumers use that same result. Bound history and reset after time gaps, geometry discontinuities, missing tracks and periods. Do not retroactively populate earlier outputs from later observations or OCR results.
- Implement stage 8 on meter-space schema inputs: carrier evidence sustained for a duration equivalent to roughly 0.3 s, possession persistence through short loose-ball intervals, explicit expiry/unknown handling, conservative ball-state inference. Three frames at 30 Hz are not three frames at 10 Hz. Freshness/usable-input policy belongs at the vision adapter boundary so the shared rules also run on dataset tracking.
- Tune on PFF/SkillCorner development matches and manually checked transitions. PFF has no reliable carrier ground truth and its possession sometimes changes late (06/07); distinguish annotation lag from inference failure. Reserve IDSSE for the final check, reconciling 03's broader tuning language with 07's holdout rule.
- Repair D8 using declared native cadence from match metadata or a documented strictly trailing estimator; handle single-frame input and irregular gaps. The converter velocity helper's whole-table cadence fallback also needs an audit before any irregular-stream reuse. Regular dataset behavior should remain covered by regression tests.
- Correct `visible` conservatively: skipped inference does not itself imply off-camera, but an off-screen extrapolated box cannot be unconditionally visible. Omit uncertain off-camera rows or mark them `visible=False, interpolated=True` per 02. Preserve nulls for unsupported state; do not manufacture alive/possession to unlock the predictor.

Done when: prefix-invariance tests alter future timestamps, images, detections and event labels while all earlier non-label outputs remain identical. One-frame inputs, dropped/repeated/out-of-order timestamps, cuts and periods have specified outcomes. Offline and streaming state agree. Possession meets the existing ≥90% provider-agreement target **with nulls counted as non-agreement on provider-known frames**, plus confusion/coverage/turnover-delay reports. The existing “dead or null ≥90%” target must also show dead recall and unknown share; all-null output is not success. Unknown state is never confused with a no-goal label.

### W7 — Fit the measured pipeline to the RTX 2060

Files: model/config adapters and bounded writer; live capture/queue integration remains in the separate app described in 09.

- Benchmark usable live play, not just an average dominated by ads/cutaways. Expose independent people/ball/keypoint resolutions and rates first, then compare smaller models and FP16/backend exports. Do not assume hardware estimates in 09 are measured capacity.
- Preserve the original source timing when dropping frames. Use bounded latest-frame queues in the live app; record dropped work and state age. Time-based expiry and motion prediction must handle that policy before it is enabled.
- Stream typed Parquet row groups/chunks and diagnostic output with bounded buffers. Distinguish incomplete/failed runs, finalize metadata safely, and refuse accidental overwrite unless explicitly requested. Match identity and run identity must not require pretending repeated experiments are different matches.
- Require an uninterrupted 10–15 minute target-cadence run with no growing backlog or monotonic memory growth after warmup. Report decode/capture-to-state p95/p99 latency, model invocation rates and quality changes at each setting. A proposed first local frame-age budget is p95 ≤200 ms; it must be measured and agreed in the live app, not presented as a guarantee of market usefulness.

Done when: the chosen config meets quality targets and the declared source-cadence/frame-age budget on the workstation. Broadcast delivery delay is a separate, currently unmeasured component; local processing speed cannot establish how early information is relative to the match or a future market feed.

### W8 — Fine-tune only the demonstrated detector failure

Files: proposed training/annotation scripts and dataset manifests; no detector replacement is chosen by this review.

- Use W0/W1 errors to choose the target: missed small people, class confusion, tiny/blurred balls, or poorly localized pitch features. Do not fine-tune the player model to fix a homography or ball-association error.
- Start with a few hundred diverse, independently selected frames for a pilot, plus dense bursts for temporal scoring. Add hard negatives such as boards, boots, white lines, crowd and spare balls. Audit annotation quality and duplicate-frame/match overlap before increasing volume.
- Compare frozen pretrained weights against one changed factor at a time: resolution/tiling, association, then fine-tuning. Track seeds, dataset/annotation hashes, training settings, validation selection and weight hashes. Augmentations must transform keypoints/boxes/identities correctly; unrestricted flips must not corrupt pitch keypoint semantics.
- Use a measured batch/resolution that fits 6 GB; only move training elsewhere if that job exceeds the workstation budget. No new large-model purchase or dependency stack is justified by this spec.

Done when: the new weights improve their targeted held-out scenario and the full pipeline scorecard without unacceptable ball false positives, team regressions or runtime cost. Failure to beat the frozen baseline is a recorded experiment, not a reason to silently change the benchmark.

### W9 — Prove readiness for prediction; defer identity until it helps

Dependencies: W0–W7; W8 only where needed. Prediction-model implementation remains a separate lane.

- Before a predictor exists, report usable history, ball freshness, team/geometry correctness and shot-window coverage. Do not claim predictive improvement from perception metrics alone.
- Once a floor/baseline exists, use the same held-out matches and aligned events to compare provider tracking, provider tracking with inferred state, measured vision-like corruption, and actual vision output where paired footage exists. Corruption tests measure sensitivity; they are not proof that actual vision matches the provider.
- Make corruption resemble measured failures: contiguous ball gaps, correlated camera drift, team mistakes and ID fragmentation, rather than only independent position noise. Audit the always-kept ball/carrier assumption in 05. Provider-estimated balls are not equivalent to fresh video detections, and their upstream causality is not established here; compare observed-ball coverage and explicit missing-ball behavior before claiming live transfer. The new CV driver already records estimated/missing-ball shares.
- Keep reference labels fixed. Do not relabel targets using the candidate's inferred possession, or compute the evaluation subset from whichever frames the candidate happened to recover. The existing prediction report rejects null probabilities on scored rows; define a separate operational coverage/miss report and a fixed paired scoring policy before using it for abstaining vision outputs. Never silently fill abstentions with zero or drop hard rows to gain AP.
- Future results must report calibration, missed shots, false alarms and **lead time at emission**, accounting for processing/queue delay. Keep probability-quality metrics separate from the operational policy for expiring a held alarm during replay/unknown/stale intervals. Update 05/07 explicitly if the policy changes.
- OCR comes after these gates unless an experiment proves identity is the bottleneck. Use optional rosters, delayed causal votes and null until confident; measure identity coverage and wrong-ID rate separately. A track ID is not a jersey number. Async OCR applies only when its result is available and never backfills past predictions. Profile value remains subject to 04's ablations.

Done when: a documented vision-to-prediction comparison explains the quality/coverage/latency cost on unseen matches, or clearly marks that check pending because the predictor/paired data is unavailable. No market/trading readiness claim follows from completing this vision milestone.

## 6. Execution order and coordination

| Milestone | Work and ordering | Exit artifact |
|---|---|---|
| A — Establish truth | W0 benchmark + W1 diagnostics; land D1 ball-history regression fix and clock/resample correctness work early | Versioned manifest, baseline scorecard, reproducible failing/passing synthetic cases |
| B — Recover correct state | W2 ball and W3 discontinuities, then W4 geometry and W5 teams/tracks; develop W6 meter-only rules on dataset tracking and integrate after inputs stabilize | Same locked-check scorecard, state coverage/confusion report, prefix/offline-live parity tests |
| C — Make it operational | W7 timing/bounded storage; W8 targeted training only where B's failures justify it | Workstation latency/memory/quality report and frozen runtime config |
| D — Establish predictive value | W9 with the concurrently added floor and later baseline; OCR only after its dependency/value is established | Paired downstream report, explicit remaining limitations |

For concurrent agents, use one worktree/branch per lane under the standing `gwt` rules. Suggested ownership: benchmark/scorer lane; scene/ball/pipeline lane; geometry lane; team/tracker lane; streaming writer/clock lane; meter-state/resampling lane. These are ownership suggestions, not instructions to run all lanes at once. Several touch `pipeline.py`, `config.py`, `types.py` and 03: assign one integrator to those files, land interfaces first, and have other lanes add isolated modules/tests. Push signed logical commits and land with `--no-ff`; never have two agents edit the shared integration files concurrently.

Before implementing a numbered task, compare its evidence against current main and existing vision-lane work. Claims in this document apply to the pinned baseline. Keep historical reviews intact and record new measurements in a new report rather than rewriting smoke evidence.

### Spec reconciliation in the relevant implementation commits

- **00/01:** record the clarified long-term goal while retaining the immediate perception/prediction scope; identify ownership of reusable meter-only state inference and the streaming state boundary.
- **02:** preserve 0.6 unless the shared contract actually changes. Clarify visibility and event-annotation semantics before relying on them downstream. Operational metadata does not automatically belong in game-state Parquet.
- **03:** fix the stale v0.4 output header/missing events table, document the actual Lab baseline, correct the prose for keypoints 31/32, add clock/cut/replay/cache/state contracts and measurable acceptance denominators.
- **05/07:** settle stale/null inference and operational alarm expiry; preserve leakage boundaries and held-out IDSSE. Specify coverage-aware vision comparison without weakening the current fixed-row model comparison.
- **09:** replace feasibility estimates with measured configs/timings; distinguish custom fill from ByteTrack updates and prohibit future-frame interpolation in production paths.
- **Roadmap:** promote benchmark, scene validity and complete live state ahead of optional OCR; distinguish implemented, synthetic-tested, real-clip-tested and held-out-validated status.

The first useful handoff is Milestone A plus the deterministic ball/clock fixes. The eventual success criterion is **fresh, correctly oriented, sufficiently complete state during the seconds before a shot, with quantified failures**, ready for the predictor to prove whether it adds useful warning time.

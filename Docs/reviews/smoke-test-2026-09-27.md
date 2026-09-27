# Smoke test — 2026-09-27

First real run of the vision pipeline on the RTX 2060 workstation, per `Docs/runbooks/smoke-test-2060.md`. Until now it had only run on synthetic fakes.

## Environment
- Commit: `f2b1f53` (smoke02 run; smoke01 ran one commit earlier at `ba8831a`, before the `--shape` flag was added to `demo.debug` — that change didn't touch `vision/`)
- GPU: NVIDIA GeForce RTX 2060, 6 GB VRAM (Turing, `sm_75`)
- Versions: `torch 2.11.0+cu128`, `ultralytics 8.4.164`, `supervision 0.30.5`

## Clip
- World Cup 2022, Spain vs Japan, period 1
- Home team (Spain) attacks TV left, wears red; Japan wears blue
- Source clip: 1920x1080, 30 fps, 3000 frames (~100 s) total; pipeline processed the first 750 frames (25 s) via `--max-frames 750`

## Runs

### smoke01 — no `--home-cluster`
- fps: started ~3.7, climbed to 6.2 as the view switched from "match" to "other" (view gate skipping heavier stages)
- Validator: **FAIL** — `objects: 36 rows more than 15 m off the pitch (pixels?)`
- Team cluster unset — pipeline printed `team is null until you pass --home-cluster (check the colors in demo.debug)`

### smoke02 — `--home-cluster 1`
- fps: started ~4.3, ended ~6.5 (same view-gate pattern as smoke01)
- Validator: **FAIL** — same error: `objects: 36 rows more than 15 m off the pitch (pixels?)`

No tracebacks in either run — both completed and wrote outputs.

## run.json (smoke01, video path removed)
```json
{
  "video_sha256": "d6f9238304a92df97e89bdcf0b55cba1e1e3e5b1fb91cd594b1a7fda3efb7885",
  "weights": {
    "football-ball-detection.pt": "678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912",
    "football-pitch-detection.pt": "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36",
    "football-player-detection.pt": "75b09c377fbf9d0791d23f6cfb689f5aed6eaa43a6818bd1fb884cf7507fffaf"
  },
  "device": "cuda",
  "config": {
    "min_grass": 0.3,
    "min_keypoints": 4,
    "off_after_s": 0.5,
    "on_after_s": 1.0,
    "refit_teams_after_s": 120.0,
    "detect_every": 1,
    "min_det_conf": 0.3,
    "track_min_conf": 0.1,
    "lost_track_s": 1.0,
    "team_warmup_s": 10.0,
    "team_min_crops": 60,
    "home_cluster": null,
    "keypoints_every": 5,
    "min_keypoint_conf": 0.5,
    "homography_window": 3,
    "homography_max_age_s": 1.0,
    "ball_max_gap_s": 1.0,
    "home_attacks_tv_right_p1": false,
    "period": 1
  },
  "git_commit": "ba8831a40510005a4f1efbda226a40e0400020fe",
  "wall_time_s": 120.23345255851746,
  "validation_errors": [
    "objects: 36 rows more than 15 m off the pitch (pixels?)"
  ]
}
```

## Debug video review (user notes)
Watched `data/vision_cache/smoke01/debug.mp4` (rings) and a `--shape box` render of the same cache.

- **Orientation:** correct — center spot centered, penalty spots at the right ends of the minimap.
- **Kit clusters:** rings/boxes were white/gray for nearly the whole 25 s clip, only picking up color in the last ~5 s. Once colored, Spain (red kit) rendered with the cluster-1 color (orange/blue in `CLUSTER_COLORS`), not the cluster-0 red — i.e. the display color didn't match the kit color, though the cluster assignment itself was consistent. `home_cluster=1` was used for smoke02 based on this.
  - Likely explanation, not confirmed: `team_warmup_s: 10.0` and `team_min_crops: 60` in the config mean clustering doesn't fit until 10 s of crops accumulate — on a 25-frame-processed / 25 s window that's a large fraction of the clip spent with `team_cluster` null (rendered as `UNKNOWN` gray).
- **View switching:** worked as expected — rings disappeared during non-match views (close-ups/replays/ads).
- **Ball:** tracked well, not jumping to other objects.
- **IDs:** stable (same id kept per player) but noted these are track ids, not jersey numbers — jersey OCR (spec 03 stage 6) isn't part of this pipeline run, so this is expected, not a bug.

## Open items for follow-up (not fixed this session)
- Validator failure "objects: 36 rows more than 15 m off the pitch" in both runs — needs investigation (homography or coordinate scale issue).
- Team clustering takes most of this short clip to lock in — worth checking `team_warmup_s`/`team_min_crops` against expected live clip lengths.

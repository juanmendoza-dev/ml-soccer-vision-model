# 01 — Architecture

## Pipeline
```
Video
  └─> [Vision] detection → tracking → team assignment → pitch homography → ball tracking → jersey OCR
        └─> Game state (Parquet, see 02)
              ├─> [Profiles] static stats + live tracking features (see 04)
              └─> [Prediction] graph builder → temporal GNN → P(shot) ; xG model → P(goal)
                    └─> [Demo] overlay renderer (see 08)
```

Tracking datasets (PFF, SkillCorner, IDSSE, Metrica) enter the pipeline directly as game state, skipping vision. This lets the prediction side be built first.

## Repo layout
```
<project root>/
├── CLAUDE.md
├── Docs/
│   └── Specs/            # these specs
├── pyproject.toml        # core deps; extras: converters, prediction, vision, dev
├── data/                 # gitignored; raw/, processed/, gamestate/, splits/, vision_cache/
├── gamestate/            # schema 02 as code + validator; the only thing vision/ and prediction/ share
├── vision/               # video → game state
├── converters/           # PFF/SkillCorner/IDSSE/Metrica → game state
├── profiles/             # player profile features
├── prediction/           # graph building, models, training
├── evaluation/
├── demo/                 # overlay rendering
├── notebooks/            # exploration only; no logic lives here
└── tests/
```

## Hardware split
Workstation specs and the live budget are in 09.

| Task | Machine | Why |
|---|---|---|
| Coding, prediction training | MacBook Pro M1 | Tracking data is small; CPU/MPS is enough |
| Vision inference (short clips) | M1 (`--device mps`) | Fine offline, slower than real time |
| YOLO fine-tuning | RTX 2060 workstation | CUDA; much faster than M1 |
| Full-video inference, live demo | RTX 2060 workstation | Near real time |
| sn-gamestate components | RTX 2060 workstation | CUDA-oriented; lower batch sizes for 6GB VRAM |
| Oversized jobs | Google Colab | Fallback |

## Tech stack
- Python 3.11 (venv via `uv`), PyTorch, PyTorch Geometric
- Ultralytics YOLO, supervision (Roboflow), ByteTrack
- kloppy (tracking data loading), unravelsports (graph conversion)
- Polars or pandas, Parquet
- OpenCV for video I/O and overlay

## Open questions
- Share data between machines via homelab storage, or sync processed files through git-lfs / rsync?

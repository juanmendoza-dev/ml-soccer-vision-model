"""P(goal) on a vision run's game state (05 "Vision inference"): resampled to 10 Hz in
memory, v1 features on stage 8's possession, the demo goal model, and 05's Inference mask
(not the training `eligible`: stage 8's ball state is often null, and null isn't dead).

    python -m prediction.infer --match-id demo01-arg-fra-81 --model data/models/goal/goal-f0-2026-10-05

Reads the 02 game state only. Run vision.stage8 first.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from converters.common import sha256
from prediction.features import match_features
from prediction.goal_model import GoalModel
from prediction.resample import resample_match

PRED_DIR = Path("data/predictions")
P_COLS = ["p_shot_h5", "xg", "p_goal_h5"]
COLUMNS = [
    "match_id", "period", "t_s", "frame_id", "timestamp_s",
    "possession_team", "ball_state", "predicted", *P_COLS,
]


def inference_mask() -> pl.Expr:
    """05 Inference: a team in possession, the ball not dead (null counts as not dead, so
    vision gaps don't blank the meter), and some player visible."""
    return (
        pl.col("possession_team").is_not_null()
        & (pl.col("ball_state") != "dead").fill_null(True)
        & ~pl.col("all_estimated")
    )


def predict_match(match_dir: Path, model) -> pl.DataFrame:
    frames = pl.read_parquet(match_dir / "frames.parquet")
    if frames["possession_team"].is_null().all():
        raise ValueError(f"{match_dir.name}: no possession anywhere; run vision.stage8 first")
    fps = pl.read_parquet(match_dir / "match.parquet")["native_fps"].item()
    frames10, objects10, _ = resample_match(
        frames,
        pl.read_parquet(match_dir / "objects.parquet"),
        pl.read_parquet(match_dir / "events.parquet"),
        match_dir.name,
        fps,
    )
    frames10 = frames10.sort("period", "t_s")
    feats = match_features(
        frames10.select("period", "t_s", "possession_team", "flipped"), objects10.lazy(), "held", "1"
    )
    if not feats.select("period", "t_s").equals(frames10.select("period", "t_s")):
        raise ValueError(f"{match_dir.name}: feature rows don't line up with the grid")
    p = model.predict(feats)
    out = frames10.select(
        "match_id", "period", "t_s", "frame_id", "timestamp_s", "possession_team", "ball_state",
        predicted=inference_mask(),
    ).hstack(p)
    return out.with_columns(
        pl.when(pl.col("predicted")).then(pl.col(c)).otherwise(None).alias(c) for c in P_COLS
    ).select(COLUMNS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="prediction.infer")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--model", type=Path, required=True, help="data/models/goal/<model_id>")
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--out-dir", type=Path, default=PRED_DIR)
    args = ap.parse_args(argv)
    model = GoalModel(args.model)
    d = args.gamestate_dir / args.match_id
    out = predict_match(d, model)
    model_id = model.manifest["model_id"]
    dst = args.out_dir / args.match_id
    dst.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dst / f"{model_id}.parquet")
    side = {
        "model_id": model_id,
        "match_id": args.match_id,
        "frames_sha256": sha256(d / "frames.parquet"),
        "objects_sha256": sha256(d / "objects.parquet"),
        "grid_rows": out.height,
        "predicted_share": round(float(out["predicted"].mean()), 4),
        "ball_state_null_share": round(float(out["ball_state"].is_null().mean()), 4),
        "no_possession_share": round(float(out["possession_team"].is_null().mean()), 4),
    }
    (dst / f"{model_id}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()

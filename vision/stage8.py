"""Stage 8 on a finished vision run (03 "Offline stage 8 fill"): ball_state,
possession_team and ball_carrier_id filled into its frames.parquet by vision.state.infer,
then the 02 validator again.

    python -m vision.stage8 --match-id demo01-arg-fra-81

vision.run writes the three as null, so run this after every vision.run of the match.
Rule-only with the default StateConfig: the learned v1h model can't read a match outside
its manifest yet (03).
"""

import argparse
import json
from pathlib import Path

import polars as pl

from gamestate.validate import validate_match
from vision.state import PLAYER_TYPES, StateConfig, infer

STATE_COLS = ("ball_state", "possession_team", "ball_carrier_id")


def fill(match_dir: Path, config: StateConfig | None = None) -> dict:
    frames = pl.read_parquet(match_dir / "frames.parquet")
    objects = pl.scan_parquet(match_dir / "objects.parquet")
    with_team = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("team").is_not_null())
        .select(pl.len())
        .collect()
        .item()
    )
    if not with_team:
        raise ValueError(
            f"{match_dir.name}: no player has a team, so possession would be null everywhere; "
            "rerun vision.run with --home-cluster"
        )
    state = infer(frames, objects, config)
    out = (
        frames.drop(*STATE_COLS)
        .join(state, on="frame_id", how="left", validate="1:1")
        .select(frames.columns)
    )
    out.write_parquet(match_dir / "frames.parquet")
    errors = validate_match(match_dir)
    if errors:
        raise ValueError(f"{match_dir.name}: 02 validation failed after stage 8: {errors}")
    bs = out["ball_state"]
    return {
        "frames": out.height,
        "possession_set_share": round(float(out["possession_team"].is_not_null().mean()), 4),
        "alive_share": round(float((bs == "alive").fill_null(False).mean()), 4),
        "dead_share": round(float((bs == "dead").fill_null(False).mean()), 4),
        "ball_state_null_share": round(float(bs.is_null().mean()), 4),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.stage8")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    args = ap.parse_args(argv)
    config = StateConfig()
    stats = fill(args.gamestate_dir / args.match_id, config)
    run_path = args.cache_dir / args.match_id / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        run["stage8"] = {"state_config": config.to_dict(), **stats}
        run_path.write_text(json.dumps(run, indent=2, default=str))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

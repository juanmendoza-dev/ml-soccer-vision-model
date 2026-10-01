"""P(goal) v1 (05 "xG v1 build"): P(goal within H)(t) = P(shot within H)(t) x xG(t).

    python -m prediction.pgoal data/runs/<run> [--xg data/models/xg/xg-v1]

P(shot) is the run's out-of-fold prediction; xG is the final xG v1 model, which never saw
World Cup 2022, applied to the same grid row's eight columns (held ball, VISIBLE players,
attacking frame). A row with no held ball has no xG and no P(goal). Writes pgoal.parquet and
pgoal.md into the run folder: PR-AUC and calibration against label_goal_*, next to xG and
P(shot) alone, and how far xG at the ball before a shot undershoots xG at the shot itself.
Descriptive only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.metrics import brier, calibration, pr_auc
from prediction.features import load_match
from prediction.floor import sigmoid
from prediction.xg import OUT_DIR, XG_FEATURES, table

HORIZONS = ("h5", "h3")
BEFORE_S = (5, 2, 1)


def xg_model(model_dir: Path):
    """The final xG v1 as a function of a table holding its input columns."""
    man = json.loads((model_dir / "manifest.json").read_text())
    feats = man["features"]
    if man["model"] == "lgbm":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

        def predict(df: pl.DataFrame) -> np.ndarray:
            return booster.predict(df.select(pl.col(f).cast(pl.Float32) for f in feats).to_numpy())
    else:
        c = json.loads((model_dir / "model.json").read_text())
        w, mean, std = np.array(c["w"]), np.array(c["mean"]), np.array(c["std"])

        def predict(df: pl.DataFrame) -> np.ndarray:
            x = df.select(feats).to_numpy().astype(float)
            return sigmoid(w[0] + ((x - mean) / std) @ w[1:])

    return man, predict


def tick() -> pl.Expr:
    return (pl.col("t_s") * 10).round().cast(pl.Int64)


def match_rows(match_id: str, processed_dir: Path, predict) -> pl.DataFrame:
    """One match's grid rows with labels, possession and xG (null without a held ball)."""
    g = load_match(match_id, processed_dir, list(HORIZONS)).with_columns(k=tick())
    goals = pl.read_parquet(processed_dir / match_id / "frames_10hz.parquet").select(
        "period", *(f"label_goal_{h}" for h in HORIZONS), k=tick()
    )
    keep = ["period", "k", "t_s", "possession_team", "all_estimated"]
    keep += [f"label_{x}_{h}" for h in HORIZONS for x in ("mask", "shot")]
    out = (
        g.select(*keep, *XG_FEATURES)
        .join(goals, on=["period", "k"], how="left", validate="1:1")
        .with_columns(match_id=pl.lit(match_id))
    )
    has = out["ball_dist"].is_not_null().to_numpy()
    xg = np.full(out.height, np.nan)
    xg[has] = predict(out.filter(pl.Series(has)))
    return out.with_columns(xg=pl.Series(xg).fill_nan(None)).drop(XG_FEATURES)


def shot_ratios(rows: pl.DataFrame, gamestate_dir: Path) -> pl.DataFrame:
    """For every PFF open-play shot outside set-play phases: xG at its own grid row and at
    the rows 5, 2 and 1 s before, while the shooter's team had the ball on that row."""
    out = []
    for (m,), r in rows.partition_by("match_id", as_dict=True).items():
        d = gamestate_dir / m
        ev = pl.read_parquet(d / "events.parquet").filter(
            pl.col("event_type") == "shot",
            pl.col("set_piece") == "open_play",
            ~pl.col("set_play_phase").fill_null(False),
        )
        if not ev.height:
            continue
        fr = pl.read_parquet(d / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
        shots = ev.join(fr, on="frame_id").select(
            "period", "team", goal=pl.col("outcome") == "goal",
            k=(pl.col("timestamp_s") * 10 + 1e-6).floor().cast(pl.Int64),
        ).with_row_index("shot")
        grid = r.select("period", "k", "possession_team", "xg").sort("period", "k")
        for lag in (0, *(10 * s for s in BEFORE_S)):
            got = (
                shots.with_columns(k=pl.col("k") - lag)
                .sort("period", "k")
                .join_asof(grid, on="k", by="period")
                .with_columns(
                    xg=pl.when(pl.col("possession_team") == pl.col("team")).then("xg"),
                    before_s=pl.lit(lag / 10),
                    match_id=pl.lit(m),
                )
            )
            out.append(got.select("match_id", "shot", "goal", "before_s", "xg"))
    long = pl.concat(out)
    at = long.filter(pl.col("before_s") == 0).select("match_id", "shot", xg_shot="xg")
    return long.filter(pl.col("before_s") > 0).join(at, on=["match_id", "shot"])


def render(run: Path, rows: pl.DataFrame, ratios: pl.DataFrame, man: dict) -> str:
    out = [f"# P(goal) v1 on {run.name}", "",
           f"xG model `{man['model_id']}` ({man['model']}), trained on {man['train_shots']} StatsBomb "
           "shots, never World Cup 2022. P(goal) = P(shot) x xG on rows with a held ball.", ""]
    for h in HORIZONS:
        sc = rows.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"))
        ok = sc.filter(pl.col(f"p_goal_{h}").is_not_null())
        y = ok[f"label_goal_{h}"].to_numpy().astype(float)
        lost = sc.height - ok.height
        lost_goals = int(sc.filter(pl.col(f"p_goal_{h}").is_null())[f"label_goal_{h}"].sum())
        out += [f"## {h}", "",
                f"{sc.height:,} scored rows, {ok.height:,} with a held ball ({lost:,} without, holding "
                f"{lost_goals:,} goal-positive rows). Goal-positive rows with a ball: {int(y.sum()):,} "
                f"(base rate {y.mean():.4f}).", ""]
        res = []
        for name, col in ((f"P(goal) = P(shot) x xG", f"p_goal_{h}"), ("P(shot) alone", f"p_{h}"),
                          ("xG alone", "xg")):
            p = ok[col].to_numpy().astype(float)
            res.append({"score": name, "PR-AUC vs goal": pr_auc(y, p), "Brier": brier(y, p),
                        "sum p": float(p.sum()), "goal rows": int(y.sum())})
        out += [table(pl.DataFrame(res)), "", "Calibration of P(goal), 10 quantile bins:", "",
                table(calibration(y, ok[f"p_goal_{h}"].to_numpy().astype(float), 10)), ""]
    r = ratios.filter(pl.col("xg").is_not_null(), pl.col("xg_shot").is_not_null(), pl.col("xg_shot") > 0)
    summary = (
        r.with_columns(ratio=pl.col("xg") / pl.col("xg_shot"))
        .group_by("before_s", "goal")
        .agg(shots=pl.len(), median_xg=pl.col("xg").median(), median_xg_at_shot=pl.col("xg_shot").median(),
             median_ratio=pl.col("ratio").median())
        .sort("before_s", "goal", descending=[True, False])
    )
    out += ["## The v1 underestimate: xG at the ball before a shot vs at the shot's own row", "",
            "Only rows where the shooter's team had the ball, both xG defined.", "", table(summary), ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--xg", type=Path, default=OUT_DIR)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gamestate", type=Path, default=Path("data/gamestate"))
    a = ap.parse_args(argv)
    man, predict = xg_model(a.xg)
    pred = pl.read_parquet(a.run / "predictions.parquet").with_columns(k=tick()).drop("t_s")
    rows = pl.concat(
        match_rows(m, a.processed, predict) for m in sorted(pred["match_id"].unique())
    ).join(pred, on=["match_id", "period", "k"], how="left")
    rows = rows.with_columns(
        (pl.col(f"p_{h}") * pl.col("xg")).alias(f"p_goal_{h}") for h in HORIZONS
    )
    ratios = shot_ratios(rows, a.gamestate)
    rows.write_parquet(a.run / "pgoal.parquet")
    text = render(a.run, rows, ratios, man)
    (a.run / "pgoal.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

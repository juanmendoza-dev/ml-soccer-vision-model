"""P(goal) v1 (05 "xG v1 build"): P(goal within H)(t) = P(shot within H)(t) x xG(t).

    python -m prediction.pgoal data/runs/<run> [--xg data/models/xg/xg-v1]

P(shot) is the run's out-of-fold prediction; xG is the final xG v1 model, which never saw
World Cup 2022, applied to the same grid row's eight columns (held ball, VISIBLE players,
attacking frame). A row with no held ball has no xG and no P(goal). Writes pgoal.parquet and
pgoal.md into the run folder: PR-AUC and calibration against label_goal_*, next to xG and
P(shot) alone, and how far xG at the ball before a shot undershoots xG at the shot itself.
Descriptive only.

Recalibration (05): p_goal_cal_<h> = sigmoid(a + b logit(p_goal)), a and b fitted on the
other outer folds' scored rows (cross-fitted), and pgoal_map.json holds the all-64 map
live uses.
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
CLIP = 1e-6
FOLDS_PATH = Path("data/splits/folds.json")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p) - np.log1p(-p)


def fit_map(p: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """(a, b) of P(goal) = sigmoid(a + b logit(p)) by plain log loss: Newton steps on two
    numbers, halved until the loss goes down. b <= 0 would reverse the order: an error."""
    X = np.column_stack([np.ones(len(p)), logit(p)])
    w = np.array([0.0, 1.0])

    def loss(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z)

    cur = loss(w)
    for _ in range(max_iter):
        q = sigmoid(X @ w)
        step = np.linalg.solve((X * (q * (1 - q))[:, None]).T @ X, X.T @ (q - y))
        for _ in range(40):
            new = loss(w - step)
            if new <= cur:
                break
            step /= 2
        w, cur = w - step, new
        if np.abs(step).max() < 1e-10:
            break
    if w[1] <= 0:
        raise ValueError(f"recalibration slope b = {w[1]:.4f} <= 0 would reverse P(goal)'s order")
    return float(w[0]), float(w[1])


def apply_map(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return sigmoid(ab[0] + ab[1] * logit(p))


def scored(h: str) -> pl.Expr:
    return pl.col(f"label_mask_{h}") & ~pl.col("all_estimated") & pl.col(f"p_goal_{h}").is_not_null()


def recalibrate(rows: pl.DataFrame, fold_of: dict[str, int]) -> tuple[pl.DataFrame, dict]:
    """p_goal_cal_<h> on every row with a p_goal: each outer fold's rows mapped by the fit on
    the other folds' scored rows. Returns the rows and {h: {"folds": {k: (a, b)}, "all":
    all-64 map, counts}}."""
    rows = rows.with_columns(fold=pl.col("match_id").replace_strict(fold_of, return_dtype=pl.Int64))
    maps = {}
    for h in HORIZONS:
        sc = rows.filter(scored(h))
        p, y, f = (sc[c].to_numpy().astype(float) for c in (f"p_goal_{h}", f"label_goal_{h}", "fold"))
        folds = {int(k): fit_map(p[f != k], y[f != k]) for k in sorted(set(f.astype(int)))}
        cal = np.full(rows.height, np.nan)
        raw = rows[f"p_goal_{h}"].fill_null(np.nan).to_numpy()
        rf = rows["fold"].to_numpy()
        for k, ab in folds.items():
            m = (rf == k) & ~np.isnan(raw)
            cal[m] = apply_map(raw[m], ab)
        rows = rows.with_columns(pl.Series(f"p_goal_cal_{h}", cal).fill_nan(None))
        maps[h] = {"folds": folds, "all": fit_map(p, y), "rows": len(y), "goal_rows": int(y.sum())}
    return rows, maps


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


def render(run: Path, rows: pl.DataFrame, ratios: pl.DataFrame, man: dict, maps: dict) -> str:
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
        for name, col in (("P(goal) recalibrated (cross-fitted)", f"p_goal_cal_{h}"),
                          ("P(goal) = P(shot) x xG", f"p_goal_{h}"), ("P(shot) alone", f"p_{h}"),
                          ("xG alone", "xg")):
            p = ok[col].to_numpy().astype(float)
            res.append({"score": name, "PR-AUC vs goal": pr_auc(y, p), "Brier": brier(y, p),
                        "sum p": float(p.sum()), "goal rows": int(y.sum())})
        per_fold = (
            ok.group_by("fold")
            .agg(goal_rows=pl.col(f"label_goal_{h}").sum(), sum_raw=pl.col(f"p_goal_{h}").sum(),
                 sum_cal=pl.col(f"p_goal_cal_{h}").sum())
            .sort("fold")
            .with_columns(
                a=pl.col("fold").replace_strict({k: v[0] for k, v in maps[h]["folds"].items()}),
                b=pl.col("fold").replace_strict({k: v[1] for k, v in maps[h]["folds"].items()}),
            )
        )
        a64, b64 = maps[h]["all"]
        out += [table(pl.DataFrame(res)), "",
                "Per outer fold: the map fitted on the other four folds, and the sums it gives:", "",
                table(per_fold), "", f"All-64 map for live: a = {a64:.4f}, b = {b64:.4f}.", "",
                "Calibration, 10 quantile bins, before:", "",
                table(calibration(y, ok[f"p_goal_{h}"].to_numpy().astype(float), 10)), "",
                "and after recalibration:", "",
                table(calibration(y, ok[f"p_goal_cal_{h}"].to_numpy().astype(float), 10)), ""]
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
    ap.add_argument("--folds", type=Path, default=FOLDS_PATH)
    a = ap.parse_args(argv)
    man, predict = xg_model(a.xg)
    pred = pl.read_parquet(a.run / "predictions.parquet").with_columns(k=tick()).drop("t_s")
    rows = pl.concat(
        match_rows(m, a.processed, predict) for m in sorted(pred["match_id"].unique())
    ).join(pred, on=["match_id", "period", "k"], how="left")
    rows = rows.with_columns(
        (pl.col(f"p_{h}") * pl.col("xg")).alias(f"p_goal_{h}") for h in HORIZONS
    )
    fold_of = {m["match_id"]: m["fold"] for m in json.loads(a.folds.read_text())["matches"]}
    rows, maps = recalibrate(rows, fold_of)
    ratios = shot_ratios(rows, a.gamestate)
    rows.write_parquet(a.run / "pgoal.parquet")
    live = {h: {"a": m["all"][0], "b": m["all"][1], "rows": m["rows"], "goal_rows": m["goal_rows"],
                "map": "sigmoid(a + b * logit(clip(P(shot) * xG, 1e-6, 1 - 1e-6)))", "xg_model": man["model_id"]}
            for h, m in maps.items()}
    (a.run / "pgoal_map.json").write_text(json.dumps(live, indent=2) + "\n")
    text = render(a.run, rows, ratios, man, maps)
    (a.run / "pgoal.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""xG v1 (05 "xG v1 build"): P(goal | a shot from here, now), on the shot model's own columns.

    python -m prediction.xg [--statsbomb data/raw/statsbomb] [--out data/models/xg/xg-v1]

Trains on StatsBomb 360 open-play shots outside set-play phases, men's competitions minus
World Cup 2022 (the PFF matches). Each shot becomes a one-row synthetic game state, so the
eight inputs come out of features.match_features itself: one geometry for training and for
the grid rows xG is applied to. 5-fold CV grouped by match picks LightGBM over the
distance + angle baseline only if it wins; the choice is refitted on every training shot and
checked, never retrained, on World Cup 2022 (StatsBomb 360) and PFF's game state.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.metrics import brier, calibration
from prediction.features import GOAL_X, load_match, match_features
from prediction.floor import LogisticFloor
from prediction.lgbm import LGBMModel

YD = 0.9144
XG_FEATURES = (
    "ball_dist",
    "ball_angle",
    "lane_defenders",
    "gk_in_lane",
    "gk_off_line",
    "gk_ball_dist",
    "press_dist",
    "def_within_5m",
)
BASELINE = ("ball_dist", "ball_angle")
H = "xg"  # the label suffix LGBMModel and LogisticFloor read: label_shot_xg = goal
SEED = 20260927
N_FOLDS = 5
PARAMS = {"num_leaves": 7, "min_data_in_leaf": 100}
MAX_ROUNDS, PATIENCE = 1000, 50
WC2022 = (43, 106)
SET_PLAY_S = 10.0
FINAL_THIRD_SB = 60 + 17.5 / YD  # 17.5 m past halfway, in StatsBomb yards
SB_DIR = Path("data/raw/statsbomb")
OUT_DIR = Path("data/models/xg/xg-v1")


def to_m(x, y):
    """StatsBomb yards (attacking +x, origin top-left) to 02 meters, anchored at the goal:
    the goal line at x = 52.5 and the goal centre at y = 0, so the posts stay 7.32 m apart."""
    return GOAL_X - (120.0 - np.asarray(x, float)) * YD, (40.0 - np.asarray(y, float)) * YD


def seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def restarts(events: list[dict]) -> dict[tuple, list[float]]:
    """(team, period) -> times of the restarts that open a set-play phase (02's proxy):
    corners anywhere, free kicks at least 17.5 m past halfway (or with no location)."""
    out = {}
    for e in events:
        kind = e.get("pass", {}).get("type", {}).get("name")
        if kind not in ("Corner", "Free Kick"):
            continue
        loc = e.get("location")
        if kind == "Free Kick" and loc is not None and loc[0] < FINAL_THIRD_SB:
            continue
        out.setdefault((e["team"]["id"], e["period"]), []).append(seconds(e["timestamp"]))
    return out


def match_shots(match_id: int, events: list[dict], frames: list[dict]) -> list[dict]:
    """Every shot of one match with its population flags and, where the 360 frame exists,
    the ball (shot location) and the defenders in meters."""
    by_uuid = {f["event_uuid"]: f for f in frames}
    starts = restarts(events)
    out = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        t = seconds(e["timestamp"])
        before = starts.get((e["team"]["id"], e["period"]), [])
        f = by_uuid.get(e["id"])
        x, y = to_m(*e["location"][:2])
        rec = {
            "match_id": str(match_id),
            "shot_id": e["id"],
            "period": e["period"],
            "goal": e["shot"]["outcome"]["name"] == "Goal",
            "statsbomb_xg": e["shot"].get("statsbomb_xg"),
            "open_play": e["shot"]["type"]["name"] == "Open Play",
            "set_play_phase": any(0 <= t - s <= SET_PLAY_S for s in before),
            "has_360": f is not None,
            "x": float(x),
            "y": float(y),
            "defenders": [],
        }
        if f is not None:
            for p in f["freeze_frame"]:
                if not p["teammate"]:
                    px, py = to_m(*p["location"][:2])
                    rec["defenders"].append((float(px), float(py), bool(p["keeper"])))
        out.append(rec)
    return out


def columns(shots: list[dict]) -> pl.DataFrame:
    """The eight xG columns for each shot, from features.match_features on a synthetic game
    state: one period per shot, one 0.1 s frame, home in possession attacking +x, the ball
    and a home shooter at the shot location and the defenders (keeper as goalkeeper) as
    VISIBLE away players. Order and length match `shots`."""
    if not shots:
        return pl.DataFrame({c: [] for c in XG_FEATURES}, schema=dict.fromkeys(XG_FEATURES, pl.Float64))
    n = len(shots)
    frames = pl.DataFrame(
        {"period": np.arange(1, n + 1), "t_s": np.zeros(n), "possession_team": ["home"] * n,
         "flipped": [False] * n}
    )
    rows = []
    for i, s in enumerate(shots, 1):
        common = {"period": i, "t_s": 0.0, "z": None, "visible": True}
        rows.append(common | {"object_id": "ball", "object_type": "ball", "team": None, "x": s["x"], "y": s["y"]})
        rows.append(common | {"object_id": "shooter", "object_type": "player", "team": "home", "x": s["x"], "y": s["y"]})
        for j, (px, py, keeper) in enumerate(s["defenders"]):
            kind = "goalkeeper" if keeper else "player"
            rows.append(common | {"object_id": f"d{j}", "object_type": kind, "team": "away", "x": px, "y": py})
    objects = pl.DataFrame(rows, schema_overrides={"z": pl.Float64}).lazy()
    out = match_features(frames, objects).sort("period")
    return out.select(pl.col(c).cast(pl.Float64) for c in XG_FEATURES)


def load_statsbomb(sb_dir: Path) -> tuple[pl.DataFrame, dict]:
    """Every shot of every men's 360 match on disk, with competition/season, the
    population flags, the eight columns (360 shots only) and per-file SHA256s."""
    comps = json.loads((sb_dir / "competitions.json").read_text(encoding="utf-8"))
    seasons = [(c["competition_id"], c["season_id"]) for c in comps
               if c.get("match_available_360") and c["competition_gender"] == "male"]
    shots, hashes = [], {}
    for comp, season in seasons:
        matches = json.loads((sb_dir / f"matches/{comp}/{season}.json").read_text(encoding="utf-8"))
        for m in matches:
            if m.get("match_status_360") != "available":
                continue
            mid = m["match_id"]
            ev_path, ff_path = sb_dir / f"events/{mid}.json", sb_dir / f"three-sixty/{mid}.json"
            ev, ff = ev_path.read_bytes(), ff_path.read_bytes()
            hashes[str(mid)] = hashlib.sha256(ev + ff).hexdigest()
            for s in match_shots(mid, json.loads(ev), json.loads(ff)):
                shots.append(s | {"competition_id": comp, "season_id": season})
    have = [s for s in shots if s["has_360"]]
    cols = columns(have)
    meta = pl.DataFrame([{k: v for k, v in s.items() if k != "defenders"} for s in shots])
    keyed = pl.DataFrame({"shot_id": [s["shot_id"] for s in have]}).with_columns(cols)
    return meta.join(keyed, on="shot_id", how="left"), hashes


def population(df: pl.DataFrame) -> pl.Expr:
    return pl.col("open_play") & ~pl.col("set_play_phase") & pl.col("has_360")


def is_wc2022() -> pl.Expr:
    return (pl.col("competition_id") == WC2022[0]) & (pl.col("season_id") == WC2022[1])


def as_rows(df: pl.DataFrame) -> pl.DataFrame:
    """The shot table under the column names LGBMModel and LogisticFloor read."""
    return df.with_columns(
        **{f"label_shot_{H}": pl.col("goal"), f"label_mask_{H}": pl.lit(True)},
        all_estimated=pl.lit(False),
        eligible=pl.lit(True),
    )


def folds(match_ids) -> dict[str, int]:
    ids = np.array(sorted(set(match_ids)))
    order = np.random.default_rng(SEED).permutation(len(ids))
    return {str(ids[j]): i % N_FOLDS for i, j in enumerate(order)}


def models() -> dict:
    return {
        "lgbm": lambda: LGBMModel(PARAMS, MAX_ROUNDS, PATIENCE, seed=SEED, features=XG_FEATURES),
        "baseline": lambda: LogisticFloor(),
    }


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    return {"shots": len(y), "goals": int(y.sum()), "sum_xg": float(p.sum()),
            "log_loss": float(log_loss(y, p)), "brier": float(brier(y, p))}


def cross_validate(train: pl.DataFrame) -> dict:
    """Out-of-fold xG for both models and per-fold scores."""
    fold_of = folds(train["match_id"])
    fold = train["match_id"].replace_strict(fold_of, return_dtype=pl.Int64)
    rows = as_rows(train).with_columns(fold=fold)
    oof = {name: np.full(rows.height, np.nan) for name in models()}
    per_fold, best = [], []
    for k in range(N_FOLDS):
        test = (rows["fold"] == k).to_numpy()
        tr, te = rows.filter(~pl.Series(test)), rows.filter(pl.Series(test))
        y = te["goal"].to_numpy().astype(float)
        rec = {"fold": k, "shots": te.height, "goals": int(y.sum())}
        for name, make in models().items():
            m = make().fit(tr, H)
            p = m.predict(te).to_numpy()
            oof[name][test] = p
            rec |= {f"{name}_log_loss": float(log_loss(y, p)), f"{name}_brier": float(brier(y, p))}
            if name == "lgbm":
                best.append(int(m.best_iter))
        per_fold.append(rec)
    return {"oof": oof, "per_fold": pl.DataFrame(per_fold), "best_iter": best, "folds": fold_of}


def choose(cv: dict, y: np.ndarray) -> tuple[str, dict]:
    """05's fixed rule: LightGBM if its pooled OOF log loss beats the baseline's and it
    wins on at least 4 of 5 folds."""
    pooled = {n: float(log_loss(y, p)) for n, p in cv["oof"].items()}
    wins = int((cv["per_fold"]["lgbm_log_loss"] < cv["per_fold"]["baseline_log_loss"]).sum())
    pick = "lgbm" if pooled["lgbm"] < pooled["baseline"] and wins >= 4 else "baseline"
    return pick, {"pooled_log_loss": pooled, "lgbm_fold_wins": wins}


def pff_shot_rows(processed_dir: Path, gamestate_dir: Path) -> pl.DataFrame:
    """PFF open-play shots outside set-play phases, each with the eight columns from the
    provider-possession v1 held cache at the latest grid row at or before the shot."""
    out = []
    for d in sorted(gamestate_dir.iterdir()):
        if not (d / "match.parquet").exists():
            continue
        if pl.read_parquet(d / "match.parquet")["source"][0] != "pff":
            continue
        ev = pl.read_parquet(d / "events.parquet").filter(
            pl.col("event_type") == "shot",
            pl.col("set_piece") == "open_play",
            ~pl.col("set_play_phase").fill_null(False),
        )
        if not ev.height:
            continue
        fr = pl.read_parquet(d / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
        shots = ev.join(fr, on="frame_id").select(
            "frame_id", "period", "timestamp_s", goal=pl.col("outcome") == "goal",
            k=(pl.col("timestamp_s") * 10 + 1e-6).floor().cast(pl.Int64),
        )
        g = load_match(d.name, processed_dir, ["h5"]).with_columns(
            k=(pl.col("t_s") * 10).round().cast(pl.Int64)
        )
        got = shots.sort("period", "k").join_asof(
            g.select("period", "k", *XG_FEATURES).sort("period", "k"), on="k", by="period"
        )
        out.append(got.with_columns(match_id=pl.lit(d.name)))
    return pl.concat(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--statsbomb", type=Path, default=SB_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gamestate", type=Path, default=Path("data/gamestate"))
    a = ap.parse_args(argv)
    if (a.out / "manifest.json").exists():
        raise SystemExit(f"{a.out} exists; a change needs a new model id")

    shots, hashes = load_statsbomb(a.statsbomb)
    counts = shots.group_by(wc=is_wc2022()).agg(
        shots=pl.len(), open_play=pl.col("open_play").sum(),
        open_not_set_play=(pl.col("open_play") & ~pl.col("set_play_phase")).sum(),
        in_population=population(shots).sum(),
        goals=(population(shots) & pl.col("goal")).sum(),
        keeper_on_camera=(population(shots) & pl.col("gk_off_line").is_not_null()).sum(),
        matches=pl.col("match_id").n_unique(),
    ).sort("wc")
    pop = shots.filter(population(shots))
    train, wc = pop.filter(~is_wc2022()), pop.filter(is_wc2022())

    cv = cross_validate(train)
    y = train["goal"].to_numpy().astype(float)
    pick, why = choose(cv, y)
    final = models()[pick]().fit(as_rows(train), H)

    def apply(df):
        return final.predict(as_rows(df)).to_numpy()

    wc_p = apply(wc)
    pff = pff_shot_rows(a.processed, a.gamestate)
    pff_ball = pff.filter(pl.col("ball_dist").is_not_null())
    pff_p = apply(pff_ball)

    a.out.mkdir(parents=True, exist_ok=True)
    if pick == "lgbm":
        final.booster.save_model(str(a.out / "model.txt"))
    else:
        (a.out / "model.json").write_text(json.dumps(
            {"w": final.w.tolist(), "mean": final.mean.tolist(), "std": final.std.tolist()}) + "\n")
    report = render(counts, cv, pick, why, y, wc, wc_p, pff, pff_ball, pff_p)
    (a.out / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "model_id": a.out.name,
        "model": pick,
        "features": list(XG_FEATURES if pick == "lgbm" else BASELINE),
        "params": final.params if pick == "lgbm" else {"l2": final.l2},
        "best_iter": int(final.best_iter) if pick == "lgbm" else None,
        "population": "shot.type Open Play, not within 10 s of the team's corner or final-third free kick, has a 360 frame",
        "excluded": "FIFA World Cup 2022 (43/106)",
        "train_shots": train.height,
        "train_goals": int(y.sum()),
        "train_shot_ids_sha256": hashlib.sha256("\n".join(sorted(train["shot_id"])).encode()).hexdigest(),
        "folds": cv["folds"],
        "cv": {"choice": why, "best_iter": cv["best_iter"]},
        "source_sha256": hashes,
        "git_commit": git_commit(),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report)
    return 0


def git_commit() -> str | None:
    import subprocess

    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def table(df: pl.DataFrame) -> str:
    with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, tbl_rows=50, tbl_cols=20, float_precision=4):
        return str(df)


def render(counts, cv, pick, why, y, wc, wc_p, pff, pff_ball, pff_p) -> str:
    wy = wc["goal"].to_numpy().astype(float)
    py = pff_ball["goal"].to_numpy().astype(float)
    sb_ref = wc["statsbomb_xg"].to_numpy().astype(float)
    out = ["# xG v1", "", "## Shots (StatsBomb 360, men's; wc = World Cup 2022, never trained on)", "",
           table(counts), "", "## 5-fold CV, grouped by match", "", table(cv["per_fold"]), "",
           f"Pooled OOF log loss: {why['pooled_log_loss']}; LightGBM wins {why['lgbm_fold_wins']}/5 folds. "
           f"**Chosen: {pick}.** LightGBM best rounds per fold: {cv['best_iter']}.", "",
           "OOF scores:", ""]
    out += [f"- {n}: {scores(p, y)}" for n, p in cv["oof"].items()]
    out += ["", "OOF calibration of the chosen model (10 bins):", "",
            table(calibration(y, cv["oof"][pick], 10)), "",
            "## World Cup 2022, StatsBomb 360 (held out)", "",
            f"- xG v1: {scores(wc_p, wy)}", f"- statsbomb_xg (reference): {scores(sb_ref, wy)}", "",
            table(calibration(wy, wc_p, 10)), "",
            "## PFF game state at the shot's grid row (held out)", "",
            f"{pff.height} open-play shots outside set-play phases, {pff_ball.height} with a held ball "
            f"({pff.height - pff_ball.height} without: no xG).", "",
            f"- xG v1: {scores(pff_p, py)}", "", table(calibration(py, pff_p, 5)), ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""The demo's goal model (05 "Offline demo model"): LightGBM P(shot) refit exactly like CV
outer fold k, xG v1, and fold k's cross-fitted P(goal) map. A clip from a match in fold k
is then scored by models that never saw that match.

    python -m prediction.goal_model --fold 0 --model-id goal-f0-2026-10-05

The fit has to reproduce the base run's fold k (best_iter, early-stopping matches, p on the
fold's rows within MAX_DIFF), otherwise nothing is saved.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from converters.common import git_commit, sha256
from prediction.cv import DATA_KEYS, KEYS, MODELS, load_data, of
from prediction.pgoal import apply_map, fit_map, scored, xg_model
from prediction.xg import OUT_DIR as XG_DIR

H = "h5"
BASE_RUN = Path("data/runs/lgbm-held-2026-09-27")
MODELS_DIR = Path("data/models/goal")
FOLDS_PATH = Path("data/splits/folds.json")
PROCESSED_DIR = Path("data/processed")
GAMESTATE_DIR = Path("data/gamestate")
MAX_DIFF = 1e-3  # another machine can move LightGBM's p this little, not more


def train_ids(folds: dict, fold: int) -> list[str]:
    """Matches outside outer fold `fold`, sorted as prediction.cv loads them."""
    return sorted(m["match_id"] for m in folds["matches"] if m["fold"] != fold)


def fold_map(pgoal: pl.DataFrame, fold: int) -> tuple[float, float]:
    """Fold k's cross-fitted P(goal) map: fitted on the other folds' scored rows only."""
    sc = pgoal.filter(scored(H), pl.col("fold") != fold)
    p = sc[f"p_goal_{H}"].to_numpy().astype(float)
    return fit_map(p, sc[f"label_goal_{H}"].to_numpy().astype(float))


def fit_shot_model(data: pl.DataFrame):
    """The CV's LightGBM, built the way run_cv's make() builds it."""
    cls, config = MODELS["lgbm"]
    kw = {k: v for k, v in config.items() if k not in DATA_KEYS}
    return cls(**kw, features=config["features"]).fit(data, H)


def reproduction(model, data: pl.DataFrame, fold: int, fold_ids: list[str], base_run: Path) -> dict:
    """How close the refit is to the base run's outer fold `fold`."""
    want = json.loads((base_run / "run.json").read_text())["folds"][H][str(fold)]
    base = pl.read_parquet(base_run / "predictions.parquet").select(*KEYS, base=f"p_{H}")
    rows = of(data, fold_ids)
    got = rows.select(KEYS).with_columns(p=model.predict(rows))
    j = got.join(base, on=KEYS, how="left", validate="1:1")
    both = j.drop_nulls(["p", "base"])
    out = {
        "best_iter": int(model.best_iter),
        "base_best_iter": int(want["best_iter"]),
        "es_matches_equal": list(model.es_matches) == list(want["es_matches"]),
        "rows": both.height,
        "null_mismatch": int((j["p"].is_null() != j["base"].is_null()).sum()),
        "max_abs_diff": float((both["p"] - both["base"]).abs().max()),
    }
    out["ok"] = (
        out["best_iter"] == out["base_best_iter"]
        and out["es_matches_equal"]
        and out["null_mismatch"] == 0
        and out["max_abs_diff"] <= MAX_DIFF
    )
    return out


def xg_file(xg_dir: Path) -> str:
    man = json.loads((xg_dir / "manifest.json").read_text())
    return "model.txt" if man["model"] == "lgbm" else "model.json"


class GoalModel:
    """A saved goal model: P(shot), xG and P(goal) for feature rows. Unmasked: the caller
    applies 05's mask. No held ball (ball_dist NaN): no xG and no P(goal)."""

    def __init__(self, model_dir: Path):
        import lightgbm as lgb

        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.features = self.manifest["features"]
        xg = self.manifest["xg"]
        if sha256(Path(xg["dir"]) / xg["file"]) != xg["sha256"]:
            raise ValueError(f"{xg['dir']}: the xG model changed since {model_dir.name} was saved")
        _, self.xg = xg_model(Path(xg["dir"]))
        self.ab = (self.manifest["map"]["a"], self.manifest["map"]["b"])

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        x = feats.select(pl.col(f).cast(pl.Float32) for f in self.features).to_numpy()
        p_shot = self.booster.predict(x)
        has = feats["ball_dist"].cast(pl.Float64).fill_null(np.nan).is_not_nan().to_numpy()
        xg = np.full(feats.height, np.nan)
        p_goal = np.full(feats.height, np.nan)
        if has.any():
            xg[has] = self.xg(feats.filter(pl.Series(has)))
            p_goal[has] = apply_map(p_shot[has] * xg[has], self.ab)
        return pl.DataFrame({"p_shot_h5": p_shot, "xg": xg, "p_goal_h5": p_goal}).fill_nan(None)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="prediction.goal_model")
    ap.add_argument("--fold", type=int, required=True, help="outer fold of the clip's match")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--base-run", type=Path, default=BASE_RUN)
    ap.add_argument("--xg", type=Path, default=XG_DIR)
    ap.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    out = args.out_dir / args.model_id
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --model-id")
    folds = json.loads(FOLDS_PATH.read_text())
    ids = sorted(m["match_id"] for m in folds["matches"])
    train = train_ids(folds, args.fold)
    fold_ids = [i for i in ids if i not in train]
    data, _ = load_data(ids, PROCESSED_DIR, GAMESTATE_DIR, [H])
    model = fit_shot_model(of(data, train))
    check = reproduction(model, data, args.fold, fold_ids, args.base_run)
    print(json.dumps(check, indent=2))
    if not check["ok"]:
        print("the refit doesn't reproduce the base run's fold; nothing saved", file=sys.stderr)
        return 1
    ab = fold_map(pl.read_parquet(args.base_run / "pgoal.parquet"), args.fold)
    out.mkdir(parents=True)
    model.booster.save_model(str(out / "model.txt"))
    xf = xg_file(args.xg)
    man = {
        "model_id": args.model_id,
        "fold": args.fold,
        "train_ids": train,
        "best_iter": int(model.best_iter),
        "es_matches": list(model.es_matches),
        "features": list(model.features),
        "features_version": MODELS["lgbm"][1]["features_version"],
        "ball_source": "held",
        "horizon": H,
        "map": {"a": ab[0], "b": ab[1], "fitted_on": f"pgoal.parquet rows of folds != {args.fold}"},
        "base_run": str(args.base_run),
        "reproduction": check,
        "xg": {"dir": str(args.xg), "file": xf, "sha256": sha256(args.xg / xf)},
        "git_commit": git_commit(),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"saved {out}: map a {ab[0]:.4f} b {ab[1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Offline nested-fit driver for learned possession (03 stage 8, "Strict nested fits").

    python -m vision.possession_train --model-id lgbm-v1-nested5x4-s1 [--pilot | --only outer_0 ...]

Trains the 5 outer fits and the 10 pair fits under data/models/possession/<model-id>/,
checks refit determinism on one pair (if two trainings of one set differ, every inner
logical ID gets its own training), then seals manifest.json: the 25 logical IDs, each
resolved to one stored training. A finished training whose fit.json still matches its
plan, inputs and model file is reused, so a run can resume and the stride-1 pilot is
outer_0. A sealed manifest is never rewritten: a change needs a new model ID.

--pilot trains (or reuses) outer_0 only, times extraction, the 2D rule and inference on
the largest match, and projects the whole stage against the overnight budget. It
decides on time and memory only and never looks at agreement.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

from vision import possession_features as pf
from vision import possession_model as pm
from vision.state import StateConfig

STRIDES = {f"lgbm-v1-nested5x4-s{s}": s for s in (1, 2, 4)}
FOLDS_PATH = Path("data/splits/folds.json")
MODELS_DIR = Path("data/models/possession")
FOLD_SIZES = [12, 13, 13, 13, 13]
DETERMINISM_PAIR = "pair_3_4"
BUDGET_H = 10.0
RSS_LIMIT = 24 * 2**30
MIRROR = (
    "M: negate X and VX, keep Y and VY, swap home/away, flip the label; "
    "p = (q(x) + 1 - q(Mx)) / 2; diff_* recomputed in Float32 (canonical)"
)
RULE_2D = "stage 8 default StateConfig on usable objects, every z null, no team_near_s"


def read_folds(path: Path = FOLDS_PATH) -> tuple[dict[str, int], str]:
    """The frozen PFF fold assignments {match_id: fold} and the file's SHA256."""
    raw = path.read_bytes()
    spec = json.loads(raw)
    folds = {m["match_id"]: int(m["fold"]) for m in spec["matches"] if m["source"] == "pff"}
    sizes = [sum(f == j for f in folds.values()) for j in range(pm.N_FOLDS)]
    if spec["n_folds"] != pm.N_FOLDS or sizes != FOLD_SIZES or "pff" not in spec["frozen"]:
        raise ValueError(f"{path}: not the frozen 64-match PFF folds (sizes {sizes})")
    return folds, hashlib.sha256(raw).hexdigest()


def load_tables(ids, gamestate_dir: Path, cache_dir: Path) -> tuple[dict, dict]:
    """Training rows only (train true) per match, and each match's recorded input hashes."""
    keep = ["k", "label", "train", *pf.COLUMNS]
    tables, inputs = {}, {}
    for m in sorted(ids):
        tables[m] = pm.match_rows(m, gamestate_dir, cache_dir).filter("train").select(keep)
        meta = json.loads(pf.cache_paths(m, cache_dir)[1].read_text())
        inputs[m] = {
            k: meta[k]
            for k in ("frames_sha256", "objects_sha256", "grid_sha256", "extractor_sha256")
        }
    return tables, inputs


def inputs_sha256(inputs: dict, ids) -> str:
    sub = {m: inputs[m] for m in sorted(ids)}
    return hashlib.sha256(json.dumps(sub, sort_keys=True).encode()).hexdigest()


def reusable(fit_dir: Path, ids: list[str], stride: int, inp: str) -> dict | None:
    """The fit record if fit_dir already holds this exact training, else None."""
    rec_path, model = fit_dir / "fit.json", fit_dir / "model.txt"
    if not rec_path.exists() or not model.exists():
        return None
    rec = json.loads(rec_path.read_text())
    same = (
        rec.get("training_ids") == ids
        and rec.get("stride") == stride
        and rec.get("params") == pm.PARAMS
        and rec.get("contract") == pf.CONTRACT
        and rec.get("columns_sha256") == pm.ids_sha256(pf.COLUMNS)
        and rec.get("inputs_sha256") == inp
        and rec.get("model_sha256") == pf.sha256_file(model)
    )
    return rec if same else None


def train(name, ids, model_dir, tables, inputs, stride, log=print) -> dict:
    fit_dir = model_dir / "fits" / name
    inp = inputs_sha256(inputs, ids)
    rec = reusable(fit_dir, ids, stride, inp)
    if rec is not None:
        log(f"{name}: reused ({rec['best_iteration']} rounds)")
        return rec
    t0 = time.perf_counter()
    rec = pm.fit(tables, ids, fit_dir, stride, extra={"name": name, "inputs_sha256": inp})
    log(
        f"{name}: {len(ids)} matches, {rec['rows']['refit']['mirrored']} mirrored rows, "
        f"{rec['best_iteration']} rounds, {time.perf_counter() - t0:.0f} s, "
        f"peak {rec['peak_rss_bytes'] / 2**30:.1f} GiB"
    )
    return rec


def entry(model_dir: Path, name: str, rec: dict) -> dict:
    fit_dir = model_dir / "fits" / name
    return {
        "dir": f"fits/{name}",
        "training_ids": rec["training_ids"],
        "training_sha256": rec["training_sha256"],
        "probe_ids": rec["probe_ids"],
        "es_ids": rec["es_ids"],
        "model_sha256": rec["model_sha256"],
        "fit_sha256": pf.sha256_file(fit_dir / "fit.json"),
        "best_iteration": rec["best_iteration"],
        "rows": rec["rows"],
        "timing_s": rec["timing_s"],
        "peak_rss_bytes": rec["peak_rss_bytes"],
    }


def manifest(model_id, folds, fold_sha, inputs, fits, stored, dedup, determinism) -> dict:
    stride = STRIDES[model_id]
    return {
        "model_id": model_id,
        "scheme": "nested5x4",
        "sealed": False,
        "contract": pf.CONTRACT,
        "columns": pf.COLUMNS,
        "stride": stride,
        "seed": pm.SEED,
        "params": pm.PARAMS,
        "max_rounds": pm.MAX_ROUNDS,
        "patience": pm.PATIENCE,
        "es_share": pm.ES_SHARE,
        "mirror": MIRROR,
        "rule_2d": RULE_2D,
        "rule_config": StateConfig().to_dict(),
        "threshold": "home if p >= 0.5; no usable player: 2D rule, p null, fallback",
        "folds": {"path": str(FOLDS_PATH), "sha256": fold_sha, "assignments": folds},
        "inputs": inputs,
        "dedup_pairs": dedup,
        "determinism": determinism,
        "trainings": stored,
        "logical": {
            lid: {k: f[k] for k in ("training", "role", "outer", "fold", "predicts")}
            | {"training_sha256": pm.ids_sha256(f["training_ids"])}
            for lid, f in fits.items()
        },
        "versions": pm.versions(),
        "git_commit": pm.git_commit(),
    }


def write_manifest(model_dir: Path, man: dict) -> None:
    (model_dir / pm.MANIFEST).write_text(json.dumps(man, indent=2) + "\n")


def run(
    model_id: str,
    only: list[str] | None = None,
    gamestate_dir: Path = pf.GAMESTATE_DIR,
    cache_dir: Path = pf.CACHE_DIR,
    models_dir: Path = MODELS_DIR,
    folds_path: Path = FOLDS_PATH,
    log=print,
) -> dict:
    """Train (or reuse) the planned trainings; without `only`, also check determinism
    and seal the manifest. Returns the manifest as written."""
    if model_id not in STRIDES:
        raise ValueError(f"unknown model id {model_id!r}; one of {sorted(STRIDES)}")
    stride = STRIDES[model_id]
    model_dir = models_dir / model_id
    path = model_dir / pm.MANIFEST
    if path.exists() and json.loads(path.read_text()).get("sealed"):
        raise ValueError(f"{path} is sealed; a change needs a new model id")
    folds, fold_sha = read_folds(folds_path)
    fits = pm.plan(folds)
    plan = pm.trainings(fits)
    names = only or list(plan)
    unknown = [n for n in names if n not in plan]
    if unknown:
        raise ValueError(f"not in the plan: {unknown}")
    need = sorted({m for n in names for m in plan[n]})
    tables, inputs = load_tables(need if only else sorted(folds), gamestate_dir, cache_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not only:  # assignments recorded before any fit
        write_manifest(model_dir, manifest(model_id, folds, fold_sha, inputs, fits, {}, None, None))
    records = {}
    if only:
        for name in names:
            records[name] = train(name, plan[name], model_dir, tables, inputs, stride, log)
        return {"trainings": records}
    # the outer fits, then one pair twice; the other pairs wait for that check
    for name in [f"outer_{k}" for k in range(pm.N_FOLDS)] + [DETERMINISM_PAIR]:
        records[name] = train(name, plan[name], model_dir, tables, inputs, stride, log)

    twin = model_dir / "determinism" / DETERMINISM_PAIR
    ids = plan[DETERMINISM_PAIR]
    again = pm.fit(tables, ids, twin, stride, extra={"name": f"{DETERMINISM_PAIR} (twin)"})
    same = again["model_sha256"] == records[DETERMINISM_PAIR]["model_sha256"]
    determinism = {
        "training": DETERMINISM_PAIR,
        "model_sha256": [records[DETERMINISM_PAIR]["model_sha256"], again["model_sha256"]],
        "same": same,
        "timing_s": again["timing_s"],
    }
    log(f"determinism {DETERMINISM_PAIR}: {'same' if same else 'DIFFERENT'} model.txt")
    if not same:  # one training per logical ID from here on
        fits = pm.plan(folds, dedup=False)
        plan = pm.trainings(fits)
    for name in plan:
        if name not in records:
            records[name] = train(name, plan[name], model_dir, tables, inputs, stride, log)
    stored = {n: entry(model_dir, n, records[n]) for n in plan}
    man = manifest(model_id, folds, fold_sha, inputs, fits, stored, same, determinism)
    man["sealed"] = True
    pm.check_manifest(man)
    write_manifest(model_dir, man)
    log(f"sealed {path}: {len(man['logical'])} logical fits over {len(stored)} trainings")
    return man


def native_frames(m: str, gamestate_dir: Path) -> int:
    return pl.scan_parquet(gamestate_dir / m / "frames.parquet").select(pl.len()).collect().item()


def pilot(
    model_id: str,
    gamestate_dir: Path = pf.GAMESTATE_DIR,
    cache_dir: Path = pf.CACHE_DIR,
    models_dir: Path = MODELS_DIR,
    folds_path: Path = FOLDS_PATH,
    log=print,
) -> dict:
    """03 "Timing plan": outer_0 (trained or reused), one match's extraction, 2D rule and
    inference on the largest match, and the projection of the whole stage."""
    import lightgbm as lgb

    stride = STRIDES[model_id]
    folds, _ = read_folds(folds_path)
    t0 = time.perf_counter()
    rec = run(model_id, ["outer_0"], gamestate_dir, cache_dir, models_dir, folds_path, log)
    rec = rec["trainings"]["outer_0"]
    pilot_overhead_s = time.perf_counter() - t0 - sum(rec["timing_s"].values())

    frames_n = {m: native_frames(m, gamestate_dir) for m in sorted(folds)}
    big = max(frames_n, key=frames_n.get)
    with tempfile.TemporaryDirectory() as tmp:
        t = time.perf_counter()
        pf.build(big, gamestate_dir, Path(tmp))
        extract_s = time.perf_counter() - t
    d = gamestate_dir / big
    fps = pl.read_parquet(d / "match.parquet")["native_fps"][0]
    t = time.perf_counter()
    feats = pf.load(big, gamestate_dir, cache_dir)
    frames = pl.read_parquet(d / "frames.parquet", columns=pm.FRAME_TIME_COLUMNS)
    pm.rule_grid(frames, pl.scan_parquet(d / "objects.parquet"), fps)
    load_rule_s = time.perf_counter() - t
    booster = lgb.Booster(model_file=str(models_dir / model_id / "fits/outer_0/model.txt"))
    t = time.perf_counter()
    pm.p_home(booster, feats)
    infer_s = time.perf_counter() - t

    # eligible rows per training at this stride, from the labels (not the 0.75x estimate)
    fits = pm.plan(folds)
    rows = {}
    for m in sorted(folds):
        lab = pm.match_rows(m, gamestate_dir, cache_dir).filter("train")
        rows[m] = lab.filter(pl.col("k") % stride == 0).height
    size = {n: sum(rows[m] for m in ids) for n, ids in pm.trainings(fits).items()}
    outer0_s = sum(rec["timing_s"].values())
    unit = outer0_s / size["outer_0"]
    train15_s = unit * (sum(size.values()) + size[DETERMINISM_PAIR])
    nodedup = pm.trainings(pm.plan(folds, dedup=False))
    train25_s = unit * (
        sum(sum(rows[m] for m in ids) for ids in nodedup.values()) + size[DETERMINISM_PAIR]
    )
    total_frames = sum(frames_n.values())
    extraction_s = extract_s * total_frames / frames_n[big]
    # 25 logical fits predict 5 x 64 matches: each match once per outer k
    predict_s = (load_rule_s + infer_s) * pm.N_FOLDS * total_frames / frames_n[big]
    total_h = (extraction_s + train15_s + predict_s) / 3600
    total25_h = (extraction_s + train25_s + predict_s) / 3600
    fits_ok = total_h <= BUDGET_H and rec["peak_rss_bytes"] <= RSS_LIMIT
    out = {
        "model_id": model_id,
        "stride": stride,
        "outer_0": {
            "best_iteration": rec["best_iteration"],
            "mirrored_rows": rec["rows"]["refit"]["mirrored"],
            "timing_s": rec["timing_s"],
            "total_s": outer0_s,
            "peak_rss_gib": rec["peak_rss_bytes"] / 2**30,
        },
        "largest_match": {
            "match_id": big,
            "native_frames": frames_n[big],
            "grid_rows": feats.height,
            "extract_s": extract_s,
            "load_and_rule_s": load_rule_s,
            "inference_s": infer_s,
        },
        "training_rows": size,
        "projection_h": {
            "extraction": extraction_s / 3600,
            "trainings_15_plus_twin": train15_s / 3600,
            "predictions_25_contexts": predict_s / 3600,
            "total": total_h,
            "total_if_not_deterministic": total25_h,
        },
        "pilot_overhead_s": pilot_overhead_s,
        "budget_h": BUDGET_H,
        "rss_limit_gib": RSS_LIMIT / 2**30,
        "fits_budget": fits_ok,
    }
    (models_dir / model_id / "pilot.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model-id", required=True, choices=sorted(STRIDES))
    ap.add_argument("--only", nargs="+", help="train just these (no determinism check, no seal)")
    ap.add_argument("--pilot", action="store_true", help="outer_0 plus timings and projection")
    ap.add_argument("--gamestate", type=Path, default=pf.GAMESTATE_DIR)
    ap.add_argument("--cache", type=Path, default=pf.CACHE_DIR)
    ap.add_argument("--models", type=Path, default=MODELS_DIR)
    ap.add_argument("--folds", type=Path, default=FOLDS_PATH)
    a = ap.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    if a.pilot:
        out = pilot(a.model_id, a.gamestate, a.cache, a.models, a.folds, log)
        print(json.dumps(out, indent=2))
    else:
        run(a.model_id, a.only, a.gamestate, a.cache, a.models, a.folds, log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

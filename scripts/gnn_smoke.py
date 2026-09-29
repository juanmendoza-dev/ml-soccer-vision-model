"""Smoke check for the frame GNN (05 model 2): a short fit on a few real matches.

    python scripts/gnn_smoke.py [--device auto|cuda|mps|cpu] [--minutes 2]
        [--train 10502 10503 10504] [--test 10505] [--horizon h5] [--layers N] [--graph-only]

Fits the full-size model on --train (one of those matches held out for early stopping)
for at most --minutes, printing the loss per epoch, then predicts --test: PR-AUC next to
its base rate, mean p next to the base rate, p before its shots, and training speed.

Not a result: a few matches can't say how the model compares with LightGBM. It shows
that the loss goes down, predictions are sane and the device works (09 runbook).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.folds import GAMESTATE_DIR
from evaluation.metrics import brier, pr_auc, roc_auc
from evaluation.report import PROCESSED_DIR
from prediction.cv import load_data
from prediction.features import FEATURES
from prediction.gnn import DEVICES, PARAMS, GNNModel
from prediction.graphs import GraphStore

OFFSETS = (5.0, 2.0, 1.0, 0.2)


def p_before(test: pl.DataFrame, p: np.ndarray, shots: pl.DataFrame, off: float) -> np.ndarray:
    """p on the grid row at or before each open-play shot minus off (same period)."""
    grid = test.select("period", k=(pl.col("t_s") * 10).round().cast(pl.Int64), p=pl.Series(p))
    k = shots.select("period", k=((pl.col("t_s") - off) * 10 + 1e-6).floor().cast(pl.Int64))
    return k.join(grid, on=["period", "k"], how="left")["p"].to_numpy()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=DEVICES)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--train", nargs="+", default=["10502", "10503", "10504"])
    ap.add_argument("--test", default="10505")
    ap.add_argument("--horizon", default="h5")
    ap.add_argument("--layers", type=int, help="default: the full model's")
    ap.add_argument("--graph-only", action="store_true", help="no global hand features")
    args = ap.parse_args(argv)
    h, ids = args.horizon, [*args.train, args.test]
    t0 = time.perf_counter()
    data, shots = load_data(ids, PROCESSED_DIR, GAMESTATE_DIR, [h])
    store = GraphStore.load(ids, PROCESSED_DIR, log=print)
    print(
        f"{len(ids)} matches, {data.height:,} rows, graphs {store.nodes.nbytes / 2**30:.2f} GB, "
        f"loaded in {time.perf_counter() - t0:.0f} s"
    )
    params = PARAMS if args.layers is None else {**PARAMS, "layers": args.layers}
    model = GNNModel(
        () if args.graph_only else FEATURES,
        params,
        es_share=1 / len(args.train),  # one training match for early stopping
        device=args.device,
        graphs=store,
        max_minutes=args.minutes,
        log=lambda s: print(s, flush=True),
    )
    train = data.filter(pl.col("match_id").is_in(args.train))
    model.fit(train, h)
    info = model.fit_info
    print(
        f"device {info['device']} ({info['device_name']}), {info['n_params']:,} parameters, "
        f"peak GPU memory {info['peak_mem_mb']} MB"
    )
    print(
        f"stopped by {info['stopped']} after {info['epochs_run']} epochs, best epoch "
        f"{info['best_epoch']} (early-stopping match {', '.join(info['es_matches'])}), "
        f"T {info['temperature']}, {info['rows_seen'] / info['fit_s']:,.0f} training rows/s "
        f"({info['fit_s']:.0f} s, early-stopping passes included)"
    )
    curve = np.array(info["curve"])
    print(
        f"train loss {curve[0, 1]:.4f} -> {curve[-1, 1]:.4f}, early-stopping loss "
        f"{curve[0, 2]:.4f} -> {np.nanmin(curve[:, 2]):.4f} (best)"
    )

    test = data.filter(pl.col("match_id") == args.test)
    t1 = time.perf_counter()
    p = model.predict(test).fill_null(np.nan).to_numpy()
    print(f"predicted {test.height:,} rows in {time.perf_counter() - t1:.1f} s")
    scored = (test[f"label_mask_{h}"] & ~test["all_estimated"]).to_numpy()
    y = test[f"label_shot_{h}"].fill_null(False).to_numpy()[scored]
    ps = p[scored]
    print(f"held-out match {args.test}: {scored.sum():,} scored rows, {int(y.sum()):,} positives")
    print(
        f"  PR-AUC {pr_auc(y, ps):.3f} (base rate {y.mean():.4f}), ROC-AUC {roc_auc(y, ps):.3f}, "
        f"Brier {brier(y, ps):.4f}"
    )
    print(f"  mean p {ps.mean():.4f} vs base rate {y.mean():.4f} (calibration in the large)")
    top = ps >= np.quantile(ps, 0.9)
    print(f"  top decile: mean p {ps[top].mean():.3f}, positive rate {y[top].mean():.3f}")
    op = shots.filter(pl.col("match_id") == args.test, "open_play")
    meds = [np.nanmedian(p_before(test, p, op, off)) for off in OFFSETS]
    print(
        f"  median p before its {op.height} open-play shots: "
        + ", ".join(f"{off:g} s {m:.3f}" for off, m in zip(OFFSETS, meds, strict=True))
        + f"; median p over all scored rows {np.median(ps):.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

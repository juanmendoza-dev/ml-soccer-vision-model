"""The frame GNN (05 model 2): message passing over one graph per grid row.

Same interface as the floor and LightGBM (fit / predict / coef / n_train / fit_info), so
prediction.cv runs it unchanged. Graphs come from a GraphStore lined up with the data's
`row` column (prediction.graphs). The hand features in `features` (v1 by default) ride
along as one global vector.

Plain log loss, no class weights (05, Class imbalance): p has to stay calibrated. Early
stopping holds out whole training matches, never the evaluation fold (07). The best
epoch is kept without a refit, then one temperature is fitted on those same matches.
torch is only imported when a model is fitted, so prediction.cv runs without it.
"""

import os
import platform
import time
import warnings

import numpy as np
import polars as pl

from prediction.features import tenths
from prediction.floor import predictable, training_rows
from prediction.graphs import NODE_FEATURES

# cuBLAS reads this when CUDA starts; without it deterministic algorithms can't be used
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PARAMS = {
    "d": 64,
    "layers": 3,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "max_epochs": 40,
    "patience": 4,
    # epoch e trains on the rows whose grid tenth k has (k + e) % stride == 0: 10 Hz rows
    # are near duplicates, and every `stride` epochs still see all of them
    "stride": 4,
    "grad_clip": 1.0,
    "temperature": True,
}
PREDICT_BATCH = 2048
DEVICES = ("auto", "cuda", "mps", "cpu")
CLIP = 5.0  # standardized global features are clipped to +-5


def resolve_device(name: str) -> str:
    """auto: cuda, then mps, then cpu. Asking for one that isn't there is an error."""
    import torch

    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {name!r}")
    cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
    if name == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if (name == "cuda" and not cuda) or (name == "mps" and not mps):
        raise RuntimeError(f"device {name!r} isn't available here")
    return name


def device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -z))


def log_loss(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T minimizing the log loss of sigmoid(logit / T), T in [1/8, 8]. The loss is convex
    in 1 / T, so a golden-section search on log T finds it."""
    lo, hi = np.log(1 / 8), np.log(8)
    r = (np.sqrt(5) - 1) / 2
    f = lambda u: log_loss(logits / np.exp(u), y)
    a, b = hi - r * (hi - lo), lo + r * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - r * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + r * (hi - lo)
            fb = f(b)
    return float(np.exp((lo + hi) / 2))


class GNNModel:
    def __init__(
        self,
        features=(),
        params: dict | None = None,
        es_share: float = 0.15,
        seed: int = 20260928,
        device: str = "auto",
        graphs=None,
        max_minutes: float | None = None,
        log=None,
    ):
        """features: global hand features (() for graph only). graphs: a GraphStore
        lined up with the `row` column of every frame passed to fit and predict.
        max_minutes stops training early (smoke runs only: it makes results depend on
        the machine's speed). log gets one line per epoch."""
        self.features = tuple(features)
        self.params = {**PARAMS, **(params or {})}
        self.es_share = es_share
        self.seed = seed
        self.device = device
        self.graphs = graphs
        self.max_minutes = max_minutes
        self.log = log

    def _raw_globals(self, df: pl.DataFrame) -> np.ndarray:
        if not self.features:
            return np.zeros((df.height, 0))
        # explicit list: nothing label-side (label_set_play_phase, masks) can slip in
        return (
            df.select(pl.col(f).cast(pl.Float64).fill_null(np.nan) for f in self.features)
            .to_numpy()
            .reshape(df.height, len(self.features))
        )

    def _globals(self, raw: np.ndarray) -> np.ndarray:
        z = (raw - self.g_mean) / self.g_std
        miss = np.isnan(z)
        z = np.clip(np.where(miss, 0.0, z), -CLIP, CLIP)
        return np.concatenate([z, miss], axis=1).astype(np.float32)

    def _inputs(self, idx: np.ndarray, glob: np.ndarray, ii: np.ndarray):
        import torch

        nodes, n = self.graphs.take(idx[ii])
        dev = self.dev
        return (
            torch.from_numpy(nodes).to(dev).float(),
            torch.from_numpy(n.astype(np.int64)).to(dev),
            torch.from_numpy(glob[ii]).to(dev),
        )

    def _logits(self, net, idx: np.ndarray, glob: np.ndarray, ii: np.ndarray) -> np.ndarray:
        import torch

        out = []
        net.eval()
        with torch.no_grad():
            for s in range(0, len(ii), PREDICT_BATCH):
                out.append(net(*self._inputs(idx, glob, ii[s : s + PREDICT_BATCH])).cpu())
        return torch.cat(out).double().numpy() if out else np.zeros(0)

    def fit(self, df: pl.DataFrame, h: str) -> "GNNModel":
        import torch

        from prediction.gnn_net import FrameGNN

        if self.graphs is None:
            raise ValueError("the GNN needs graphs: a GraphStore lined up with the data")
        rows = training_rows(df, h)
        if not rows.height:
            raise ValueError("no training rows")
        start = time.perf_counter()
        self.dev = dev = resolve_device(self.device)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        p = self.params
        idx = self.graphs.check(rows)
        y = rows[f"label_shot_{h}"].to_numpy().astype(np.float32)
        k = rows.select(tenths())["t_s"].to_numpy()
        raw = self._raw_globals(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            mean, std = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
        self.g_mean = np.nan_to_num(mean)
        self.g_std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        glob = self._globals(raw)

        # early stopping on whole training matches (as LightGBM's es_share)
        rng = np.random.default_rng(self.seed)
        matches = np.array(sorted(rows["match_id"].unique()))
        n_es = round(self.es_share * len(matches))
        es = []
        if n_es >= 1 and len(matches) - n_es >= 1:
            es = sorted(str(m) for m in rng.choice(matches, n_es, replace=False))
        hold = rows["match_id"].is_in(es).to_numpy()
        train_i, es_i = np.flatnonzero(~hold), np.flatnonzero(hold)

        net = FrameGNN(len(NODE_FEATURES), len(self.features), p["d"], p["layers"], p["dropout"])
        net = net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_loss, best_epoch, best_logits = None, np.inf, -1, None
        curve, stopped = [], "max_epochs"
        for epoch in range(p["max_epochs"]):
            net.train()
            pick = rng.permutation(train_i[(k[train_i] + epoch) % p["stride"] == 0])
            total = torch.zeros((), device=dev)
            for s in range(0, len(pick), p["batch_size"]):
                ii = pick[s : s + p["batch_size"]]
                loss = loss_fn(net(*self._inputs(idx, glob, ii)), torch.from_numpy(y[ii]).to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), p["grad_clip"])
                opt.step()
                total += loss.detach() * len(ii)
            train_loss = float(total) / max(len(pick), 1)
            es_loss, note = float("nan"), ""
            if len(es_i):
                logits = self._logits(net, idx, glob, es_i)
                es_loss = log_loss(logits, y[es_i])
                if es_loss < best_loss:
                    best_loss, best_epoch, best_logits = es_loss, epoch, logits
                    best = {n: t.detach().clone() for n, t in net.state_dict().items()}
                    note = " (best)"
            curve.append([epoch, round(train_loss, 5), round(es_loss, 5)])
            elapsed = time.perf_counter() - start
            if self.log:
                self.log(
                    f"  epoch {epoch}: train {train_loss:.5f}, early stop {es_loss:.5f}{note}, "
                    f"{len(pick):,} rows, {elapsed:.0f} s"
                )
            if len(es_i) and epoch - best_epoch >= p["patience"]:
                stopped = "patience"
                break
            if self.max_minutes is not None and elapsed > 60 * self.max_minutes:
                stopped = "time"
                break
        if best is not None:
            net.load_state_dict(best)
        else:  # too few matches to hold any out: the last epoch
            best_epoch = epoch
        self.temperature = 1.0
        # T needs both classes among the early-stopping rows (tiny test worlds may not
        # have a shot there); otherwise it runs off to a bound
        if p["temperature"] and best_logits is not None and 0 < y[es_i].sum() < len(es_i):
            self.temperature = fit_temperature(best_logits, y[es_i])
        self.net = net.eval()
        self.n_train = len(y)
        self._info = {
            "best_epoch": int(best_epoch),
            "epochs_run": len(curve),
            "stopped": stopped,
            "es_matches": es,
            "es_loss": None if best is None else round(float(best_loss), 6),
            "temperature": round(self.temperature, 4),
            "grad_rows": len(train_i),
            "fit_s": round(time.perf_counter() - start, 1),
            "device": dev,
            "device_name": device_name(dev),
            "peak_mem_mb": (
                round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None
            ),
            "n_params": int(sum(t.numel() for t in net.parameters())),
            "curve": curve,
        }
        return self

    @property
    def coef(self) -> dict[str, float]:
        """No gain shares for a network (05): an importance would need an ablation."""
        return {}

    @property
    def fit_info(self) -> dict:
        return self._info

    def predict(self, df: pl.DataFrame) -> pl.Series:
        """P(shot) per row, null where 05 says not to predict. Rows are independent: a
        row's p doesn't depend on which other rows come with it."""
        ok = df.select(predictable()).to_series().to_numpy()
        p = np.full(df.height, np.nan)
        if ok.any():
            part = df.filter(pl.Series(ok))
            idx = self.graphs.check(part)
            glob = self._globals(self._raw_globals(part))
            logits = self._logits(self.net, idx, glob, np.arange(part.height))
            p[ok] = sigmoid(logits / self.temperature)
        return pl.Series("p", p).fill_nan(None)

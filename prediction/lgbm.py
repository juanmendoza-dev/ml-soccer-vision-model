"""The LightGBM baseline (05 model 1): gradient boosting on the hand features.

Same interface as the floor (fit / predict / coef / n_train), so prediction.cv runs it
unchanged. Early stopping holds out whole matches from the rows it's fitted on, never
the evaluation fold (07: anything tuned is tuned inside the training matches). Then it
refits on all of them with the round count it found. No class weights: p has to stay
calibrated for the alarms and for P(goal) = P(shot) x xG.
"""

import numpy as np
import polars as pl

from prediction.features import FEATURES
from prediction.floor import predictable, training_rows

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    # 10 Hz rows are highly correlated, so a leaf needs many rows to mean anything
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "metric": "binary_logloss",
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


class LGBMModel:
    def __init__(
        self,
        params: dict | None = None,
        max_rounds: int = 2000,
        early_stopping: int = 100,
        es_share: float = 0.15,
        seed: int = 20260927,
        features=FEATURES,
    ):
        self.features = tuple(features)
        self.params = {**PARAMS, **(params or {}), "seed": seed}
        self.max_rounds = max_rounds
        self.early_stopping = early_stopping
        self.es_share = es_share
        self.seed = seed

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        # explicit list: nothing label-side (label_set_play_phase, masks) can slip in
        return df.select(pl.col(f).cast(pl.Float32) for f in self.features).to_numpy()

    def _train(self, x, y, rounds, valid=None):
        import lightgbm as lgb

        train = lgb.Dataset(x, y, feature_name=list(self.features), free_raw_data=False)
        kw = {}
        if valid is not None:
            kw["valid_sets"] = [lgb.Dataset(*valid, reference=train)]
            kw["callbacks"] = [lgb.early_stopping(self.early_stopping, verbose=False)]
        return lgb.train(self.params, train, num_boost_round=rounds, **kw)

    def fit(self, df: pl.DataFrame, h: str) -> "LGBMModel":
        rows = training_rows(df, h)
        if not rows.height:
            raise ValueError("no training rows")
        x = self._matrix(rows)
        y = rows[f"label_shot_{h}"].to_numpy().astype(float)
        matches = np.array(sorted(rows["match_id"].unique()))
        n_es = round(self.es_share * len(matches))
        if n_es >= 1 and len(matches) - n_es >= 1:
            rng = np.random.default_rng(self.seed)
            es = set(rng.choice(matches, n_es, replace=False))
            hold = rows["match_id"].is_in(list(es)).to_numpy()
            probe = self._train(x[~hold], y[~hold], self.max_rounds, valid=(x[hold], y[hold]))
            self.best_iter = max(1, probe.best_iteration)
            self.es_matches = sorted(es)
        else:  # too few matches to hold any out
            self.best_iter, self.es_matches = self.max_rounds, []
        self.booster = self._train(x, y, self.best_iter)
        self.n_train = len(y)
        return self

    @property
    def coef(self) -> dict[str, float]:
        """Share of total split gain per feature (sums to 1)."""
        gain = self.booster.feature_importance(importance_type="gain")
        total = gain.sum() or 1.0
        return {f: float(g / total) for f, g in zip(self.features, gain, strict=True)}

    @property
    def fit_info(self) -> dict:
        return {"best_iter": int(self.best_iter), "es_matches": list(self.es_matches)}

    def predict(self, df: pl.DataFrame) -> pl.Series:
        """P(shot) per row, null where 05 says not to predict. Missing features are NaN,
        which LightGBM routes on its own."""
        p = self.booster.predict(self._matrix(df))
        ok = df.select(predictable()).to_series().to_numpy()
        return pl.Series("p", np.where(ok, p, np.nan)).fill_nan(None)

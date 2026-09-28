"""The floor (05 model 0): logistic regression on ball distance + angle to goal.

Plain numpy (Newton steps on standardized features), so it doesn't need the
prediction extra. Everything it learns, including the standardization and the
missing-ball fallback, comes from the rows it's fitted on.
"""

import numpy as np
import polars as pl

FEATURES = ("ball_dist", "ball_angle")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def training_rows(df: pl.DataFrame, h: str) -> pl.DataFrame:
    """The rows a model trains on: label_mask true, not all-ESTIMATED (05). Same rows the
    report scores."""
    return df.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"))


def predictable() -> pl.Expr:
    """05: predict only with the ball not dead and a team in possession; all-ESTIMATED
    rows are scored as null."""
    return pl.col("eligible") & ~pl.col("all_estimated")


class LogisticFloor:
    def __init__(self, l2: float = 1e-6, max_iter: int = 50):
        self.l2 = l2
        self.max_iter = max_iter

    def fit(self, df: pl.DataFrame, h: str) -> "LogisticFloor":
        rows = training_rows(df, h)
        y_all = rows[f"label_shot_{h}"].to_numpy().astype(float)
        if not len(y_all):
            raise ValueError("no training rows")
        # rows without a ball get the training base rate at predict time
        self.base_rate = float(y_all.mean())
        has_ball = rows.select(pl.all_horizontal(pl.col(f).is_not_null() for f in FEATURES))
        rows = rows.filter(has_ball.to_series())
        x = rows.select(FEATURES).to_numpy().astype(float)
        y = rows[f"label_shot_{h}"].to_numpy().astype(float)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std == 0] = 1.0
        X = np.column_stack([np.ones(len(x)), (x - self.mean) / self.std])
        w = np.zeros(X.shape[1])
        w[0] = np.log(y.mean() / (1 - y.mean())) if 0 < y.mean() < 1 else 0.0
        reg = self.l2 * np.eye(len(w))
        reg[0, 0] = 0.0  # no penalty on the intercept
        for _ in range(self.max_iter):
            p = sigmoid(X @ w)
            grad = X.T @ (p - y) + reg @ w
            hess = (X * (p * (1 - p))[:, None]).T @ X + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        self.n_train = len(y_all)
        return self

    @property
    def coef(self) -> dict[str, float]:
        """Standardized coefficients: p should fall with distance and rise with angle."""
        return {"intercept": float(self.w[0])} | {
            f: float(c) for f, c in zip(FEATURES, self.w[1:], strict=True)
        }

    def predict(self, df: pl.DataFrame) -> pl.Series:
        """P(shot) per row: null where 05 says not to predict, the training base rate
        where the row has no ball."""
        x = df.select(FEATURES).to_numpy().astype(float)
        z = self.w[0] + ((x - self.mean) / self.std) @ self.w[1:]
        p = np.where(np.isnan(z), self.base_rate, sigmoid(z))
        ok = df.select(predictable()).to_series().to_numpy()
        return pl.Series("p", np.where(ok, p, np.nan)).fill_nan(None)

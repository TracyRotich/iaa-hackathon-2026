"""Benchmark (Tweedie GLM) and challenger (LightGBM Tweedie) per line.

Exposure handling: model the cost rate y / exposure with sample weight = exposure.
Under a compound Poisson-gamma model whose claim frequency is proportional to exposure,
this is equivalent to a log offset. Predictions are rate x exposure. Logged in decision_log.md.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .config import SEED, TWEEDIE_POWER


def _split_cols(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


class TweedieGLM:
    """Explainable benchmark: splines on numerics, one-hot categoricals, light ridge penalty."""

    def __init__(self, alpha=1e-3, power=TWEEDIE_POWER):
        self.alpha, self.power = alpha, power

    def fit(self, X, y, exposure):
        num, cat = _split_cols(X)
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()),
                              ("sp", SplineTransformer(n_knots=5, degree=3))]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
        ])
        self.pipe = Pipeline([
            ("pre", pre),
            ("glm", TweedieRegressor(power=self.power, link="log", alpha=self.alpha, max_iter=1000)),
        ])
        e = np.asarray(exposure, dtype=float)
        self.pipe.fit(X, np.asarray(y) / e, glm__sample_weight=e)
        return self

    def predict(self, X, exposure):
        return self.pipe.predict(X) * np.asarray(exposure, dtype=float)


class TweedieGBM:
    """Challenger. Monotone constraints and interaction limits can be added once features are known."""

    def __init__(self, power=TWEEDIE_POWER, **params):
        import lightgbm as lgb
        base = dict(objective="tweedie", tweedie_variance_power=power, n_estimators=600,
                    learning_rate=0.03, num_leaves=15, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=SEED, verbose=-1)
        base.update(params)
        self.model = lgb.LGBMRegressor(**base)

    def _prep(self, X):
        # Categories are fixed from training so test levels map to the same codes;
        # unseen levels become missing rather than silently shifting codes.
        X = X.copy()
        for c, levels in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=levels)
        return X

    def fit(self, X, y, exposure):
        _, cat = _split_cols(X)
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in cat}
        e = np.asarray(exposure, dtype=float)
        self.model.fit(self._prep(X), np.asarray(y) / e, sample_weight=e)
        return self

    def predict(self, X, exposure):
        return self.model.predict(self._prep(X)) * np.asarray(exposure, dtype=float)


def blend(p_glm, p_gbm, w_gbm=0.5):
    """Simple credibility-style blend; weight chosen on the time-split holdout, not on test."""
    return (1 - w_gbm) * np.asarray(p_glm) + w_gbm * np.asarray(p_gbm)

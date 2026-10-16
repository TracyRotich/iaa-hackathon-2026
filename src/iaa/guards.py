"""Checks that run before anything is trusted. Each one is also an automated test."""

import numpy as np
import pandas as pd

from .config import COLS, POST_LOSS_FIELD_HINTS


def leakage_suspects(feature_names, allow=()):
    """Feature names that look like post-loss information. Review each by hand; allow-list survivors."""
    hits = []
    for f in feature_names:
        low = f.lower()
        if f in allow:
            continue
        if any(h in low for h in POST_LOSS_FIELD_HINTS):
            hits.append(f)
    return hits


def check_exposure(df, col=COLS["exposure"]):
    """Exposure must be present, positive and at most one year per row (unless the data says otherwise)."""
    e = df[col]
    problems = {}
    if e.isna().any():
        problems["missing"] = int(e.isna().sum())
    if (e <= 0).any():
        problems["non_positive"] = int((e <= 0).sum())
    if (e > 1.0001).any():
        problems["above_one_year"] = int((e > 1.0001).sum())
    return problems


def check_predictions(pred):
    """Competition rule: finite and non-negative."""
    p = np.asarray(pred, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("Predictions contain NaN or inf")
    if (p < 0).any():
        raise ValueError("Predictions contain negative values")
    return True


def calibration_gap(actual_total, predicted_total):
    """Relative miss on the portfolio total. Flag anything beyond a few percent."""
    return predicted_total / actual_total - 1.0


def describe_target(df, col=COLS["target"]):
    """Quick shape of the target: share of zeros, mean, tail. Heavy zero mass supports Tweedie."""
    y = df[col]
    return pd.Series({
        "rows": len(y),
        "zero_share": float((y == 0).mean()),
        "mean": float(y.mean()),
        "p99": float(y.quantile(0.99)),
        "max": float(y.max()),
        "negatives": int((y < 0).sum()),
    })

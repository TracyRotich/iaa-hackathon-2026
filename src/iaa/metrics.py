"""Local copy of the competition metric, plus actual-versus-expected diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score

from .config import TWEEDIE_POWER


def d2(y_true, y_pred, sample_weight=None, power=TWEEDIE_POWER):
    """Tweedie D^2 against a constant (mean) baseline. 1 is perfect, 0 equals the baseline."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    return d2_tweedie_score(y_true, y_pred, sample_weight=sample_weight, power=power)


def ave_table(df, segment, actual="actual", expected="expected"):
    """Actual versus expected by segment: the reserving control applied to a pricing model."""
    g = df.groupby(segment, observed=True)[[actual, expected]].sum()
    g["n"] = df.groupby(segment, observed=True).size()
    g["a_over_e"] = g[actual] / g[expected].replace(0, np.nan)
    tot = pd.DataFrame(
        {actual: [df[actual].sum()], expected: [df[expected].sum()], "n": [len(df)]},
        index=["TOTAL"],
    )
    tot["a_over_e"] = tot[actual] / tot[expected]
    return pd.concat([g, tot])


def lift_table(y_true, y_pred, bins=10):
    """Decile lift: does the model rank risks? Mean actual and predicted by predicted decile."""
    d = pd.DataFrame({"actual": y_true, "pred": y_pred})
    d["decile"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False) + 1
    return d.groupby("decile")[["actual", "pred"]].mean()

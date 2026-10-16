"""Time-based validation: fit on earlier years, score on the latest year, mirroring 'predict 2025'."""

import pandas as pd

from .config import COLS
from .metrics import d2


def time_split(df, holdout_year=None, year_col=COLS["year"]):
    years = sorted(df[year_col].unique())
    holdout_year = holdout_year or years[-1]
    return df[df[year_col] < holdout_year], df[df[year_col] == holdout_year]


def evaluate(models, train, test, features, target=COLS["target"], exposure=COLS["exposure"]):
    """Fit each model on train, score D^2 and calibration on test. Returns a tidy table and predictions."""
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(train[features], train[target], train[exposure])
        p = model.predict(test[features], test[exposure])
        preds[name] = p
        rows.append({
            "model": name,
            "d2_tweedie_1.5": d2(test[target], p),
            "pred_total / actual_total": p.sum() / test[target].sum(),
        })
    return pd.DataFrame(rows).set_index("model"), preds

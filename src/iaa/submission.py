"""Build and validate the single combined submission file: row_id,prediction."""

import pandas as pd

from .guards import check_predictions


def build(parts):
    """parts: dict line -> DataFrame with columns row_id, prediction."""
    sub = pd.concat(parts.values(), ignore_index=True)[["row_id", "prediction"]]
    return sub


def validate(sub, expected_ids):
    """Fail loudly before upload rather than after."""
    check_predictions(sub["prediction"])
    if sub["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_ids: {int(sub['row_id'].duplicated().sum())}")
    got, want = set(sub["row_id"]), set(expected_ids)
    missing, extra = want - got, got - want
    if missing or extra:
        raise ValueError(f"ID mismatch: {len(missing)} missing, {len(extra)} unexpected")
    if list(sub.columns) != ["row_id", "prediction"]:
        raise ValueError(f"Columns must be row_id,prediction; got {list(sub.columns)}")
    return True

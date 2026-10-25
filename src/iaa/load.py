"""Load the Kaggle parquet files with identifiers as text, dates parsed and exposure repaired."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import ID_COLS

DATE_COLS = ("period_start", "period_end", "loss_date", "report_date", "date_of_birth")


def repair_exposure(df):
    """Recompute exposure from the row dates (days / 365).

    Found in the training files (21 Oct): about 0.5% of rows are on the wrong basis:
    some hold the number of days (e.g. 327.0), others use a 360-day year (1.0139 for a
    full year, 172 days -> 0.4778). The test files are all on days / 365.
    Returns the repaired frame and the number of rows changed.
    """
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    fixed = days / 365.0
    changed = ~np.isclose(df["exposure"], fixed, atol=0.003)
    df = df.copy()
    df["exposure_raw"] = df["exposure"]
    df["exposure"] = fixed
    return df, int(changed.sum())


def read(name, data_dir=None, fix_exposure=True):
    df = pd.read_parquet(Path(data_dir or config.DATA_DIR) / f"{name}.parquet")
    for c in df.columns:
        if c in ID_COLS:
            df[c] = df[c].astype(str)
        elif c in DATE_COLS:
            df[c] = pd.to_datetime(df[c])
    if "period_start" in df.columns:
        df["policy_year"] = df["period_start"].dt.year
        if fix_exposure:
            df, n = repair_exposure(df)
            df.attrs["exposure_rows_repaired"] = n
    return df

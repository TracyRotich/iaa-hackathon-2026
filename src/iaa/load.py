"""Load the Kaggle parquet files with identifiers as text and dates parsed."""

from pathlib import Path

import pandas as pd

from .config import ID_COLS

DATE_COLS = ("period_start", "period_end", "loss_date", "report_date", "date_of_birth")


def read(name, data_dir="data"):
    df = pd.read_parquet(Path(data_dir) / f"{name}.parquet")
    for c in df.columns:
        if c in ID_COLS:
            df[c] = df[c].astype(str)
        elif c in DATE_COLS:
            df[c] = pd.to_datetime(df[c])
    if "period_start" in df.columns:
        df["policy_year"] = df["period_start"].dt.year
    return df

"""Toy data with known structure, so the pipeline and tests run before the real data is loaded.
Never used for any submitted result."""

import numpy as np
import pandas as pd

from .config import COLS, SEED


def make(n=6000, years=(2021, 2022, 2023, 2024), seed=SEED, prefix="M"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        COLS["id"]: [f"{prefix}{i:06d}" for i in range(n)],
        COLS["year"]: rng.choice(years, n),
        COLS["exposure"]: rng.uniform(0.2, 1.0, n),
        "driver_age": rng.integers(18, 85, n),
        "region": rng.choice(list("ABCDE"), n),
    })
    rel = {"A": 0.8, "B": 1.0, "C": 1.1, "D": 1.3, "E": 1.6}
    young = np.where(df["driver_age"] < 25, 1.8, 1.0)
    trend = 1.05 ** (df[COLS["year"]] - years[0])
    freq = 0.12 * df[COLS["exposure"]] * young * df["region"].map(rel) * trend
    counts = rng.poisson(freq)
    df[COLS["target"]] = [rng.gamma(2.0, 1500.0, k).sum() if k else 0.0 for k in counts]
    return df

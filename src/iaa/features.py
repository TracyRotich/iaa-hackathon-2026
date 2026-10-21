"""Row-level feature frame for one line: policy row + customer + address + historical weather.

Weather: each address gets its 2021-2024 average and its prior-year value. Same-year weather
is NOT used, because 2025 weather is not known when the 2025 price is set (the file does contain
2025, which is the trap). The holdout test in run_part1.py measures what same-year weather would
have added, for the write-up.
"""

import numpy as np
import pandas as pd

from .config import EXCLUDE

CUSTOMER_KEEP = ["customer_id", "occupation", "marital_status", "household_size", "credit_band",
                 "prior_claims_5y", "licence_points", "years_at_address", "customer_since_year",
                 "previous_insurer"]
WEATHER_VARS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
                "wildfire_index", "rainfall_mm", "mean_temperature_c"]


def weather_features(weather, hist_years=(2021, 2022, 2023, 2024)):
    hist = (weather[weather["year"].isin(hist_years)].groupby("address_id")[WEATHER_VARS]
            .mean().add_prefix("wx_avg_"))
    return hist


def prior_year_weather(weather):
    w = weather.copy()
    w["year"] = w["year"] + 1  # value from year t-1 is joined to rows of year t
    return w.set_index(["address_id", "year"])[WEATHER_VARS].add_prefix("wx_prev_")


def build(rows, customers, addresses, wx_hist, wx_prev, same_year_weather=None):
    df = rows.merge(customers[CUSTOMER_KEEP], on="customer_id", how="left")
    df = df.merge(addresses, on="address_id", how="left")
    df = df.merge(wx_hist, left_on="address_id", right_index=True, how="left")
    df = df.merge(wx_prev, left_on=["address_id", "policy_year"], right_index=True, how="left")
    if same_year_weather is not None:  # experiment only, never in a submission
        sw = same_year_weather.set_index(["address_id", "year"])[WEATHER_VARS].add_prefix("wx_same_")
        df = df.merge(sw, left_on=["address_id", "policy_year"], right_index=True, how="left")
    df["customer_tenure"] = df["policy_year"] - df["customer_since_year"]
    df["property_age"] = df["policy_year"] - df["year_built"]
    df["start_month"] = df["period_start"].dt.month
    if "smoke_alarms" in df.columns:
        df["smoke_alarms"] = df["smoke_alarms"].map({True: "yes", False: "no"}).fillna("unknown")
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def feature_list(df, extra_exclude=()):
    drop = set(EXCLUDE) | set(extra_exclude) | {
        "exposure", "exposure_raw", "policy_year", "customer_since_year", "year_built",
        "claim_cost", "reported_cost", "n_claims", "completeness"}
    return [c for c in df.columns if c not in drop and not c.startswith("wx_same_")]

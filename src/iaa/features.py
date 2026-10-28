"""Row-level feature frame for one line: policy row + customer + address + historical weather.

Weather: each address gets its average over past years and its prior-year value. Same-year weather
is NOT used: a price for 2025 is set before 2025 weather is observed. The weather file does
contain 2025; excluding it is our information-timing assumption, not a stated competition rule,
and the organisers have not clarified it (review finding R8). For validation, time_safe_weather
rebuilds the averages from years before the scored year (finding R2).
"""

import numpy as np
import pandas as pd

from .config import EXCLUDE

CUSTOMER_KEEP = ["customer_id", "occupation", "marital_status", "household_size", "credit_band",
                 "prior_claims_5y", "licence_points", "years_at_address", "customer_since_year",
                 "previous_insurer", "gender"]  # gender: emerging only, excluded from features (config.EXCLUDE)
WEATHER_VARS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
                "wildfire_index", "rainfall_mm", "mean_temperature_c"]


def weather_features(weather, hist_years=(2021, 2022, 2023, 2024)):
    hist = (weather[weather["year"].isin(hist_years)].groupby("address_id")[WEATHER_VARS]
            .mean().add_prefix("wx_avg_"))
    return hist


def time_safe_weather(tr, te, weather, cutoff):
    """Review fix R2 (Codex, 23 Oct). Rebuild the weather averages from years before `cutoff`
    only, so a validation year never sees its own or later weather. For the 2025 submission
    (cutoff 2025) this equals the original 2021-2024 average.
    Also returns a feature filter that drops prior-year weather: it is missing only for the
    first policy year, so a model can use the missingness as a year flag."""
    wx = weather_features(weather, hist_years=tuple(y for y in sorted(weather["year"].unique()) if y < cutoff))
    tr, te = tr.copy(deep=False), te.copy(deep=False)
    for f in (tr, te):
        for c in wx.columns:
            f[c] = f["address_id"].map(wx[c])
    return tr, te, (lambda feats: [c for c in feats if not c.startswith("wx_prev_")])


def prior_year_weather(weather):
    w = weather.copy()
    w["year"] = w["year"] + 1  # value from year t-1 is joined to rows of year t
    return w.set_index(["address_id", "year"])[WEATHER_VARS].add_prefix("wx_prev_")


TEXT_FIELDS = ("occupation", "construction_type", "marital_status", "region", "soil_type", "property_type", "roof_type")


def build(rows, customers, addresses, wx_hist, wx_prev, same_year_weather=None, normalise_text=True):
    keep = [c for c in CUSTOMER_KEEP if c in customers.columns]  # markets differ (emerging: no credit_band)
    df = rows.merge(customers[keep], on="customer_id", how="left")
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
    if normalise_text:  # found 28 Oct: emerging occupation has 80 spellings of 40 jobs; developed
        for c in TEXT_FIELDS:  # construction_type has 11 spellings of 5 types (missed in Part 1)
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip().str.lower()
    return df


def feature_list(df, extra_exclude=()):
    drop = set(EXCLUDE) | set(extra_exclude) | {
        "exposure", "exposure_raw", "policy_year", "customer_since_year", "year_built",
        "claim_cost", "reported_cost", "n_claims", "completeness"}
    return [c for c in df.columns if c not in drop and not c.startswith("wx_same_")]

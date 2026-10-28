"""Portfolio level for 2025: expected cost per exposure year, by line and peril.

This is a burning-cost / frequency-severity projection, done separately from the row-level
models. The row models supply relativities (who is riskier); this module supplies the level
(how much in total), so immature 2024 claims and trend do not distort the level.

Method, per line and peril:
- Claims: duplicates removed, repudiated removed, value = incurred - recoveries, capped at
  LARGE_CAP (one motor bodily-injury case estimate of 11.9m is ten times the next largest
  claim and looks like a keying error; the cap removes its influence).
- Frequency: reported count / reporting completeness (from the peril's own delay pattern,
  observed on 2021 losses), divided by exposure of the same calendar year.
- Severity: mean capped value over base years where few claims are still open, trended to
  2025 at the line's severity trend (fitted on its short-tail perils, loss years 2021-2024).
- Frequency for 2025: exposure-weighted mean of 2022-2024 (centred on 2023), trended two years
  at the peril's fitted 2021-2024 frequency trend, capped at +/-3% a year (subsidence shows +12%,
  which is not credible to extrapolate from four points).
"""

import numpy as np
import pandas as pd

from .config import VALUATION_DATE

LARGE_CAP = 1_200_000
PROJECTION_YEAR = 2025
FREQ_TREND_CLIP = 0.03  # observed peril frequency trends are used, capped at +/-3% a year
BASE_YEARS_LONG_TAIL = {"bodily_injury": (2021, 2022), "subsidence": (2021, 2022)}
BASE_YEARS_DEFAULT = (2021, 2022, 2023)
TREND_YEARS = (2021, 2022, 2023, 2024)  # short-tail perils: 2024 open share <= 27%
SHORT_TAIL_FOR_TREND = {
    "motor": ("own_damage", "glass_hail", "third_party_prop", "theft"),
    "buildings": ("storm", "escape_of_water", "accidental_damage"),
    "contents": ("theft", "escape_of_water", "accidental_damage"),
}


def history_settings(years):
    """Year choices for the level, derived from the years in the data. For the developed market
    (2021-2024) this returns exactly the settings used in Part 1. For a shorter history (the
    emerging market has 2023-2024 only) the same rules are applied mechanically; those choices
    must be reviewed against the data before use (2 years cannot support a fitted trend)."""
    years = tuple(sorted(int(y) for y in years))
    if years == (2021, 2022, 2023, 2024):
        return {"years": years, "ref_year": 2021, "trend_years": TREND_YEARS, "base_default": BASE_YEARS_DEFAULT,
                "base_long_tail": BASE_YEARS_LONG_TAIL, "freq_years": (2022, 2023, 2024)}
    lt = years[:max(1, len(years) - 2)]
    return {"years": years, "ref_year": years[0], "trend_years": years, "base_default": years[:-1] or years,
            "base_long_tail": {"bodily_injury": lt, "subsidence": lt}, "freq_years": years[-3:]}


def peril_completeness(claims, years=(2021, 2022, 2023, 2024), ref_year=2021):
    """Share of each loss year's claims reported by the valuation date, by line and peril."""
    c = claims.assign(delay=(claims["report_date"] - claims["loss_date"]).dt.days)
    out = []
    for (line, peril), g in c.groupby(["line", "peril"]):
        ref = np.sort(g.loc[g["loss_date"].dt.year == ref_year, "delay"].to_numpy())
        for y in years:
            ages = (VALUATION_DATE - pd.date_range(f"{y}-01-01", f"{y}-12-31")).days.to_numpy()
            comp = (np.searchsorted(ref, ages, side="right") / len(ref)).mean()
            out.append((line, peril, y, comp))
    return pd.DataFrame(out, columns=["line", "peril", "loss_year", "completeness"])


def _trend(years, values, weights):
    """Annual trend from a weighted log-linear fit."""
    b = np.polyfit(np.asarray(years, float), np.log(values), 1, w=np.sqrt(weights))
    return float(np.exp(b[0]) - 1)


def project(claims, exposure_by_line_year, freq_trend=True, sev_trend_override=None,
            long_tail_base=None, cap=None, settings=None, completeness=None):
    """claims: cleaned claims (no duplicates). exposure_by_line_year: DataFrame line, year, exposure.

    Returns (peril_table, line_table). line_table has the 2025 cost per exposure year.
    The keyword arguments exist for sensitivity testing (scripts/uncertainty.py); defaults = base case.
    """
    cap = LARGE_CAP if cap is None else cap
    st = settings or history_settings(exposure_by_line_year["year"].unique())
    lt_base = st["base_long_tail"] if long_tail_base is None else long_tail_base
    c = claims[claims["status"] != "repudiated"].copy()
    c["value"] = (c["incurred_to_date"] - c["recoveries"]).clip(lower=0, upper=cap)
    c["loss_year"] = c["loss_date"].dt.year
    c = c[c["loss_year"].between(st["years"][0], st["years"][-1])]

    stats = c.groupby(["line", "peril", "loss_year"]).agg(
        n=("value", "size"), sev=("value", "mean"),
        open_share=("status", lambda s: (s == "open").mean())).reset_index()
    comp = completeness if completeness is not None else peril_completeness(claims, st["years"], st["ref_year"])
    stats = stats.merge(comp, on=["line", "peril", "loss_year"])
    stats = stats.merge(exposure_by_line_year.rename(columns={"year": "loss_year"}),
                        on=["line", "loss_year"])
    stats["n_ult"] = stats["n"] / stats["completeness"]
    stats["freq"] = stats["n_ult"] / stats["exposure"]

    sev_trend = {}
    for line, perils in (SHORT_TAIL_FOR_TREND.items() if len(st["trend_years"]) > 2 else []):
        s = stats[(stats["line"] == line) & stats["peril"].isin(perils)
                  & stats["loss_year"].isin(st["trend_years"])]
        # Chain a weighted index across perils: severity relative to each peril's own mean.
        s = s.assign(rel=s["sev"] / s.groupby("peril")["sev"].transform("mean"))
        yr = s.groupby("loss_year").apply(lambda g: np.average(g["rel"], weights=g["n"]))
        wt = s.groupby("loss_year")["n"].sum()
        sev_trend[line] = _trend(yr.index, yr.values, wt.values)
    if sev_trend_override:
        sev_trend.update(sev_trend_override)

    rows = []
    for (line, peril), g in stats.groupby(["line", "peril"]):
        base = lt_base.get(peril, st["base_default"])
        b = g[g["loss_year"].isin(base)]
        t = sev_trend[line]
        sev25 = np.average(b["sev"] * (1 + t) ** (PROJECTION_YEAR - b["loss_year"]), weights=b["n"])
        f = g[g["loss_year"].isin(st["freq_years"])]
        freq25 = np.average(f["freq"], weights=f["exposure"])
        ft = _trend(g["loss_year"], g["freq"], g["exposure"])
        ft_used = float(np.clip(ft, -FREQ_TREND_CLIP, FREQ_TREND_CLIP)) if freq_trend else 0.0
        if freq_trend:
            freq25 *= (1 + ft_used) ** (PROJECTION_YEAR - float(np.mean(st["freq_years"])))
        rows.append({"line": line, "peril": peril, "freq_2025": freq25, "freq_trend_obs": ft, "freq_trend_used": ft_used,
                     "sev_trend_used": t, "sev_2025": sev25, "base_years": str(base),
                     "cost_rate_2025": freq25 * sev25})
    perils = pd.DataFrame(rows)
    lines = perils.groupby("line")["cost_rate_2025"].sum().rename("cost_rate_2025").reset_index()
    return perils, lines, stats

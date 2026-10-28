"""Build the training target from the claims table.

Steps, each logged and each testable:
1. Match every claim to the policy row in force on its loss_date (same policy_id).
2. Drop repudiated (declined) claims.
3. Claim value = incurred_to_date - recoveries. incurred_to_date is already net of excess.
4. Sum to row level: reported cost per row.
5. Gross up for claims incurred but not yet reported (IBNR) at 31 Dec 2024, using the
   empirical reporting-delay distribution by line. This mainly affects late-2024 rows.
6. IBNER (development on open case estimates) cannot be measured from a single snapshot;
   the default factor is 1.0 and the sensitivity is reported, not hidden.
"""

import numpy as np
import pandas as pd

from .config import VALUATION_DATE


# --------------------------------------------------------------------------- data checks

def claims_data_checks(claims):
    """Anomalies to count and explain before trusting any claim value."""
    c = claims
    return pd.Series({
        "claims": len(c),
        "report_before_loss": int((c["report_date"] < c["loss_date"]).sum()),
        "loss_after_valuation": int((c["loss_date"] > VALUATION_DATE).sum()),
        "report_after_valuation": int((c["report_date"] > VALUATION_DATE).sum()),
        "incurred_below_paid": int((c["incurred_to_date"] < c["paid_to_date"] - 1e-6).sum()),
        "negative_paid": int((c["paid_to_date"] < 0).sum()),
        "repudiated": int((c["status"] == "repudiated").sum()),
        "repudiated_with_payment": int(((c["status"] == "repudiated") & (c["paid_to_date"] > 0)).sum()),
        "recoveries_exceed_incurred": int((c["recoveries"] > c["incurred_to_date"]).sum()),
        "duplicate_claim_id": int(c["claim_id"].duplicated().sum()),
    })


def drop_duplicate_claims(claims):
    """Remove records identical on every field except claim_id (found 16 Oct data: 604 pairs,
    one id of each pair usually 9 digits). Keeps the first record of each pair."""
    cols = [c for c in claims.columns if c != "claim_id"]
    dup = claims.duplicated(cols, keep="first")
    return claims[~dup].copy(), int(dup.sum())


# --------------------------------------------------------------------------- 1. matching

def match_claims(policies, claims):
    """Attach each claim to the policy row whose period contains its loss_date.

    Returns (matched, unmatched). A claim is unmatched if its loss date falls outside every
    row of its policy, e.g. after cancellation. Those are reported, never silently kept.
    Where rows touch at a change date, the later-starting row wins (merge_asof backward).
    """
    rows = policies[["row_id", "policy_id", "period_start", "period_end"]].sort_values("period_start")
    c = claims.sort_values("loss_date")
    m = pd.merge_asof(c, rows, left_on="loss_date", right_on="period_start",
                      by="policy_id", direction="backward")
    inside = m["row_id"].notna() & (m["loss_date"] <= m["period_end"])
    return m[inside].copy(), m[~inside].drop(columns=["row_id", "period_start", "period_end"])


# --------------------------------------------------------------------------- 2-4. value

def claim_value(matched, ibner_factor=1.0, cap=None):
    """Net claim value for non-repudiated claims. Open claims can carry an IBNER factor."""
    c = matched[matched["status"] != "repudiated"].copy()
    c["value"] = (c["incurred_to_date"] - c["recoveries"]).clip(lower=0, upper=cap)
    c.loc[c["status"] == "open", "value"] *= ibner_factor
    return c


def reported_cost_by_row(policies, valued_claims):
    agg = valued_claims.groupby("row_id")["value"].agg(reported_cost="sum", n_claims="size")
    out = policies[["row_id"]].merge(agg, on="row_id", how="left")
    return out.fillna({"reported_cost": 0.0, "n_claims": 0})


# --------------------------------------------------------------------------- 5. IBNR

def reporting_delay_cdf(claims, max_delay_days=730, valuation=VALUATION_DATE):
    """Empirical CDF of report delay (days), using only losses old enough for every delay up to
    max_delay_days to have been observed. Delays beyond the window are treated as reported by then
    (stated limitation: the tail beyond max_delay_days is not measured)."""
    eligible = claims[claims["loss_date"] <= valuation - pd.Timedelta(days=max_delay_days)]
    d = (eligible["report_date"] - eligible["loss_date"]).dt.days.clip(lower=0).to_numpy()
    if len(d) == 0:
        raise ValueError("No claims old enough to estimate the reporting delay distribution")
    grid = np.arange(max_delay_days + 1)
    cdf = np.searchsorted(np.sort(d), grid, side="right") / len(d)
    return grid, cdf


def row_completeness(policies, grid, cdf, n_points=24, valuation=VALUATION_DATE):
    """Expected share of a row's eventual claims already reported at valuation.
    Averages the delay CDF over n_points loss dates spread evenly across the row's period,
    which assumes losses are uniform in time within the row (seasonality ignored; noted)."""
    start = policies["period_start"].to_numpy("datetime64[D]")
    end = np.minimum(policies["period_end"].to_numpy("datetime64[D]"),
                     np.datetime64(valuation.date(), "D"))
    span = (end - start).astype(int)
    frac = (np.arange(n_points) + 0.5) / n_points
    loss_days = start[:, None] + (span[:, None] * frac[None, :]).astype("timedelta64[D]")
    age = (np.datetime64(valuation.date(), "D") - loss_days).astype(int)
    f = cdf[np.clip(age, 0, len(grid) - 1)]
    return f.mean(axis=1)


def build_target(policies, claims, ibner_factor=1.0, max_delay_days=730, min_completeness=0.2, cap=None, delay=None):
    """delay: optional (grid, cdf) reporting pattern to use instead of estimating it from `claims`.
    The emerging market has only 2023-24 losses, too recent to observe a 2-year delay pattern;
    its observable delays match the developed market's (28 Oct), so the developed pattern is borrowed."""
    """Full pipeline for one line. Returns a row-level frame plus a diagnostics dict."""
    claims, n_dup = drop_duplicate_claims(claims)
    matched, unmatched = match_claims(policies, claims)
    valued = claim_value(matched, ibner_factor, cap)
    rows = reported_cost_by_row(policies, valued)
    grid, cdf = delay if delay is not None else reporting_delay_cdf(claims, max_delay_days)
    comp = row_completeness(policies, grid, cdf)
    rows["completeness"] = np.clip(comp, min_completeness, 1.0)
    rows["claim_cost"] = rows["reported_cost"] / rows["completeness"]
    diag = {
        "duplicates_dropped": n_dup,
        "claims_in": len(claims),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "repudiated_dropped": int((matched["status"] == "repudiated").sum()),
        "open_share_of_value": float(valued.loc[valued["status"] == "open", "value"].sum()
                                     / max(valued["value"].sum(), 1e-9)),
        "ibnr_uplift_total": float(rows["claim_cost"].sum() / max(rows["reported_cost"].sum(), 1e-9)),
        "rows_completeness_floored": int((comp < min_completeness).sum()),
    }
    return rows, diag

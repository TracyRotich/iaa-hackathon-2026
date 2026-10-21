"""Level checks:
 1. Severity trend three ways (s.9.2): mean-based chain index (used), median (robust), and a
    mix-adjusted Gamma GLM with peril and rating factors (like-for-like, 'basket' idea).
 2. IBNER evidence (s.15): open vs closed severity, paid share of incurred on open claims, and
    cohort severity by loss year after removing trend.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import GammaRegressor
from iaa import load, target
from iaa.level import LARGE_CAP, SHORT_TAIL_FOR_TREND

claims, _ = target.drop_duplicate_claims(load.read("claims"))
c = claims[(claims.status != "repudiated")].copy()
c["value"] = (c.incurred_to_date - c.recoveries).clip(0, LARGE_CAP)
c["ly"] = c.loss_date.dt.year
c = c[c.ly.between(2021, 2024) & (c.value > 0)]
out = {"trend": {}, "ibner": {}}

ROW_FEATS = {"motor": ["cover_type", "vehicle_value_group", "driver_age", "region"],
             "buildings": ["region", "sum_insured", "property_type", "construction_type"],
             "contents": ["region", "sum_insured", "tenure", "security_level"]}
for line, perils in SHORT_TAIL_FOR_TREND.items():
    cl = c[(c.line == line) & c.peril.isin(perils)]
    # 1a median-based chained index, weighted by count
    rel_med = []
    for p, g in cl.groupby("peril"):
        med = g.groupby("ly").value.median(); rel_med.append((med / med.mean()).rename(p) )
    m = pd.concat(rel_med, axis=1); w = cl.groupby(["ly", "peril"]).size().unstack()
    idx = (m * w).sum(1) / w.sum(1)
    b = np.polyfit(idx.index.astype(float), np.log(idx.values), 1, w=np.sqrt(w.sum(1).values))
    med_trend = float(np.exp(b[0]) - 1)
    # 1b mix-adjusted Gamma GLM on individual claims: log E[value] = year*b + peril + rating factors
    pol = load.read(f"policies_{line}")
    matched, _ = target.match_claims(pol, cl)
    from iaa.features import build  # attach address/customer fields where needed
    rows = pol[["row_id", "region"] if "region" in pol.columns else ["row_id", "address_id"]]
    addr = load.read("addresses")[["address_id", "region", "property_type"]]
    pol2 = pol.merge(addr, on="address_id", how="left")
    mm = matched.merge(pol2[["row_id"] + [f for f in ROW_FEATS[line] if f in pol2.columns]], on="row_id")
    X = pd.get_dummies(mm[["peril"] + [f for f in ROW_FEATS[line] if f in mm.columns]].astype(
        {f: str for f in ["peril", "region", "cover_type", "property_type", "construction_type", "tenure", "security_level"] if f in mm.columns}),
        drop_first=True, dtype=float)
    for num in ("vehicle_value_group", "driver_age", "sum_insured"):
        if num in X.columns:
            X[num] = np.log(mm[num].clip(lower=1))
    X["year"] = mm.ly - 2021
    g = GammaRegressor(alpha=0, max_iter=500).fit(X.to_numpy(float), mm.value.to_numpy(float))
    glm_trend = float(np.exp(g.coef_[list(X.columns).index("year")]) - 1)
    lv = pd.read_csv("data/work/level_perils.csv")
    used = float(lv.loc[lv.line == line, "sev_trend_used"].iloc[0])
    out["trend"][line] = {"used_mean_chain": round(used, 4), "median_chain": round(med_trend, 4),
                          "mix_adjusted_glm": round(glm_trend, 4), "claims": int(len(mm))}

# 2 IBNER evidence by peril
for (line, peril), g in c.groupby(["line", "peril"]):
    r = {}
    for y, gy in g.groupby("ly"):
        op, cl_ = gy[gy.status == "open"], gy[gy.status == "closed"]
        r[int(y)] = {"open_share": round(len(op) / len(gy), 3),
                     "mean_open/closed": round(op.value.mean() / cl_.value.mean(), 2) if len(op) > 29 else None,
                     "paid/incurred_open": round(op.paid_to_date.sum() / op.incurred_to_date.sum(), 2) if len(op) > 29 else None}
    out["ibner"][f"{line}/{peril}"] = r
print(json.dumps(out, indent=1))

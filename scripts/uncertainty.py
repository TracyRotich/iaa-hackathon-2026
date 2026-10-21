"""Uncertainty in the 2025 level:
- Parameter uncertainty: bootstrap of claims within line/peril/loss-year (counts Poisson-resampled).
- Assumption uncertainty: severity trend method, frequency trend on/off, bodily-injury base years, cap.
- Model uncertainty: spread between the two submitted relativity models (reported separately).
- Process uncertainty: CV of each line's 2025 total under a compound Poisson model.
Writes data/work/results/uncertainty.json
"""
import json
import numpy as np, pandas as pd
from iaa import level, load, target

claims, _ = target.drop_duplicate_claims(load.read("claims"))
ex = []
for line in ("motor", "buildings", "contents"):
    d = pd.read_parquet(f"data/work/{line}_train.parquet", columns=["policy_year", "exposure"])
    e = d.groupby("policy_year")["exposure"].sum().reset_index(); e.columns = ["year", "exposure"]; e["line"] = line
    ex.append(e)
ex = pd.concat(ex)
base_lines = level.project(claims, ex)[1].set_index("line")["cost_rate_2025"]
res = {"base": base_lines.round(1).to_dict()}

# Assumption uncertainty
trend_alt = {  # from scripts/gi_pricing_level_checks.py
    "median_chain": {"motor": 0.0742, "buildings": 0.0668, "contents": 0.0449},
    "mix_adjusted_glm": {"motor": 0.0652, "buildings": 0.0561, "contents": 0.0266},
}
scen = {
    "sev_trend_median": dict(sev_trend_override=trend_alt["median_chain"]),
    "sev_trend_mix_adjusted": dict(sev_trend_override=trend_alt["mix_adjusted_glm"]),
    "sev_trend_minus_2pp": dict(sev_trend_override={k: v - 0.02 for k, v in
                                                    level.project(claims, ex)[0].groupby("line").sev_trend_used.first().items()}),
    "sev_trend_plus_2pp": dict(sev_trend_override={k: v + 0.02 for k, v in
                                                   level.project(claims, ex)[0].groupby("line").sev_trend_used.first().items()}),
    "no_freq_trend": dict(freq_trend=False),
    "bi_base_2021_only": dict(long_tail_base={"bodily_injury": (2021,), "subsidence": (2021, 2022)}),
    "bi_base_2021_2023": dict(long_tail_base={"bodily_injury": (2021, 2022, 2023), "subsidence": (2021, 2022, 2023)}),
    "cap_500k": dict(cap=500_000),
}
res["assumptions"] = {}
for name, kw in scen.items():
    v = level.project(claims, ex, **kw)[1].set_index("line")["cost_rate_2025"]
    res["assumptions"][name] = {k: round(100 * (v[k] / base_lines[k] - 1), 1) for k in base_lines.index}

# Parameter uncertainty: bootstrap
rng = np.random.default_rng(20261016)
boot = []
c_all = claims.copy()
grp = c_all.groupby(["line", "peril", c_all.loss_date.dt.year]).indices
for b in range(200):
    idx = []
    for k, ix in grp.items():
        n = rng.poisson(len(ix))
        idx.append(rng.choice(ix, size=n, replace=True))
    cb = c_all.iloc[np.concatenate(idx)]
    boot.append(level.project(cb, ex)[1].set_index("line")["cost_rate_2025"])
B = pd.DataFrame(boot)
res["parameter_bootstrap"] = {k: {"cv_pct": round(100 * B[k].std() / B[k].mean(), 1),
                                  "p5": round(B[k].quantile(.05), 1), "p95": round(B[k].quantile(.95), 1)} for k in B}

# Process uncertainty: compound Poisson, line total in 2025
perils = level.project(claims, ex)[0]
c = claims[claims.status != "repudiated"].copy()
c["v"] = (c.incurred_to_date - c.recoveries).clip(0, level.LARGE_CAP)
res["process_cv_pct"] = {}
for line in base_lines.index:
    te_exp = pd.read_parquet(f"data/work/{line}_test.parquet", columns=["exposure"]).exposure.sum()
    var = mean = 0.0
    for _, r in perils[perils.line == line].iterrows():
        v = c[(c.line == line) & (c.peril == r.peril)].v
        lam = r.freq_2025 * te_exp
        scale = r.sev_2025 / v.mean()
        mean += lam * r.sev_2025; var += lam * (v ** 2).mean() * scale ** 2
    res["process_cv_pct"][line] = round(100 * np.sqrt(var) / mean, 2)
json.dump(res, open("data/work/results/uncertainty.json", "w"), indent=1)
print(json.dumps(res, indent=1))

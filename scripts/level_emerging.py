"""2025 cost level for the emerging market, by line and peril (same method as level.py), with
the choices a two-year history forces, each borrowed from the developed market and stated:
- reporting completeness by peril: developed patterns (2021 losses) at the same ages, because
  2023-24 losses are too recent to show a 2-year delay tail; observable delays match (28 Oct);
- severity trend: developed 2021-24 short-tail trend by line (a 2-point emerging trend is noise);
  emerging 2023->24 changes are reported as a check;
- frequency: exposure-weighted 2023-24 mean, no trend (two points cannot support one);
- severity base years: both years for short-tail perils; 2023 only for bodily injury and
  subsidence (2024 too immature).
Output is the level at the 2023-24 portfolio mix; the 2025 mix (76% new business) is applied
afterwards through the relativity model (scripts/run_part2.py)."""
import pandas as pd
from iaa import level, load, target

dev = target.drop_duplicate_claims(load.read("claims", "data"))[0]
em = target.drop_duplicate_claims(load.read("claims", "data/emerging"))[0]
ex = []
for line in ("motor", "buildings", "contents"):
    d = pd.read_parquet(f"data/emerging/work/{line}_train.parquet", columns=["policy_year", "exposure"])
    e = d.groupby("policy_year")["exposure"].sum().reset_index(); e.columns = ["year", "exposure"]; e["line"] = line; ex.append(e)
ex = pd.concat(ex)
dev_ex = []
for line in ("motor", "buildings", "contents"):
    d = pd.read_parquet(f"data/work/{line}_train.parquet", columns=["policy_year", "exposure"])
    e = d.groupby("policy_year")["exposure"].sum().reset_index(); e.columns = ["year", "exposure"]; e["line"] = line; dev_ex.append(e)
dev_perils, _, _ = level.project(dev, pd.concat(dev_ex))
sev_trend = dev_perils.groupby("line")["sev_trend_used"].first().to_dict()
comp = level.peril_completeness(dev, years=(2023, 2024), ref_year=2021)
settings = {"years": (2023, 2024), "ref_year": 2023, "trend_years": (2023, 2024), "base_default": (2023, 2024),
            "base_long_tail": {"bodily_injury": (2023,), "subsidence": (2023,)}, "freq_years": (2023, 2024)}
perils, lines, stats = level.project(em, ex, freq_trend=False, sev_trend_override=sev_trend,
                                     settings=settings, completeness=comp)
# Severity credibility (shrinkage to a structured target). Benchmark severity = emerging median x developed mean/median ratio (the body
# from the emerging market, the tail shape from the developed market). Z = s_h^2 / (s_h^2 + se^2),
# se = CV / sqrt(n) of the emerging mean, s_h = spread across perils of the emerging/developed
# tail ratio. Few-claim heavy-tailed perils (fire: 70 claims) move towards the benchmark;
# high-volume perils keep their own severity (Z near 1).
import numpy as np
def sev_stats(c):
    c = c[(c.status != "repudiated") & c.loss_date.dt.year.isin([2023, 2024])]
    v = (c.incurred_to_date - c.recoveries).clip(lower=0); c = c.assign(v=v)[v > 0]
    return c.groupby(["line", "peril"]).v.agg(n="size", mean="mean", median="median", sd="std")
se, sd_ = sev_stats(em), sev_stats(dev)
t = se.join(sd_, rsuffix="_dev")
t["tail_em"], t["tail_dev"] = t["mean"] / t["median"], t.mean_dev / t.median_dev
t["ratio"] = t.tail_em / t.tail_dev                      # emerging tail relative to developed
s_h = float(t.ratio.std())                               # heterogeneity of tail shape across perils
t["se_rel"] = t.sd / t["mean"] / np.sqrt(t.n)            # relative standard error of emerging mean
t["Z"] = s_h ** 2 / (s_h ** 2 + t.se_rel ** 2)
t["bench_factor"] = 1 / t.ratio                          # benchmark severity / emerging severity
t["cred_factor"] = t.Z + (1 - t.Z) * t.bench_factor
perils = perils.merge(t[["Z", "cred_factor"]].reset_index(), on=["line", "peril"], how="left")
perils["sev_2025"] *= perils.cred_factor
perils["cost_rate_2025"] = perils.freq_2025 * perils.sev_2025
lines = perils.groupby("line")["cost_rate_2025"].sum().rename("cost_rate_2025").reset_index()
print("tail-shape heterogeneity s_h =", round(s_h, 3))
print(t[["n", "tail_em", "tail_dev", "se_rel", "Z", "cred_factor"]].round(3).to_string())
perils.to_csv("data/emerging/work/level_perils.csv", index=False)
lines.to_csv("data/emerging/work/level_lines.csv", index=False)
stats.to_csv("data/emerging/work/level_stats.csv", index=False)
print("severity trends (developed):", {k: round(v, 4) for k, v in sev_trend.items()})
print(perils[["line", "peril", "freq_2025", "sev_2025", "cost_rate_2025"]].round(4).to_string(index=False))
print(lines.round(2).to_string(index=False))

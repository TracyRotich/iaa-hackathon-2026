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
perils.to_csv("data/emerging/work/level_perils.csv", index=False)
lines.to_csv("data/emerging/work/level_lines.csv", index=False)
stats.to_csv("data/emerging/work/level_stats.csv", index=False)
print("severity trends (developed):", {k: round(v, 4) for k, v in sev_trend.items()})
print(perils[["line", "peril", "freq_2025", "sev_2025", "cost_rate_2025"]].round(4).to_string(index=False))
print(lines.round(2).to_string(index=False))

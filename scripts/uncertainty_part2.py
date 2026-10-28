"""Sensitivity of the emerging 2025 level (rate per exposure year at the 2023-24 mix) to the
assumptions a two-year history forces. Each row changes one assumption from the base case."""
import pandas as pd
from iaa import level, load, target

dev = target.drop_duplicate_claims(load.read("claims", "data"))[0]
em = target.drop_duplicate_claims(load.read("claims", "data/emerging"))[0]
ex = pd.concat([pd.read_parquet(f"data/emerging/work/{l}_train.parquet", columns=["policy_year", "exposure"])
                .groupby("policy_year").exposure.sum().reset_index().rename(columns={"policy_year": "year"}).assign(line=l)
                for l in ("motor", "buildings", "contents")])
base_trend = {"motor": 0.0656, "buildings": 0.0739, "contents": 0.0324}
comp_dev = level.peril_completeness(dev, years=(2023, 2024), ref_year=2021)
S = {"years": (2023, 2024), "ref_year": 2023, "trend_years": (2023, 2024), "base_default": (2023, 2024),
     "base_long_tail": {"bodily_injury": (2023,), "subsidence": (2023,)}, "freq_years": (2023, 2024)}
def run(trend=base_trend, settings=S, comp=comp_dev, ftrend=False, claims=em):
    return level.project(claims, ex, freq_trend=ftrend, sev_trend_override=trend, settings=settings, completeness=comp)[1].set_index("line").cost_rate_2025
base = run(); rows = {"base": base}
rows["severity trend -2pp"] = run({k: v - 0.02 for k, v in base_trend.items()})
rows["severity trend +2pp"] = run({k: v + 0.02 for k, v in base_trend.items()})
rows["BI & subsidence severity from 2023-24"] = run(settings={**S, "base_long_tail": {}})
rows["BI open claims +20% (IBNER)"] = run(claims=em.assign(incurred_to_date=em.incurred_to_date.where(~((em.peril == "bodily_injury") & (em.status == "open")), em.incurred_to_date * 1.2)))
rows["completeness from emerging 2023 (truncated)"] = run(comp=level.peril_completeness(em, years=(2023, 2024), ref_year=2023))
no_big_fire = em[~((em.peril == "fire") & ((em.incurred_to_date - em.recoveries) > 250_000))]
rows["fire claims > 250k removed"] = run(claims=no_big_fire)
t = pd.DataFrame(rows).T
pct = (t / t.loc["base"] - 1) * 100
print(t.round(1).to_string()); print("\n% change from base"); print(pct.round(1).to_string())

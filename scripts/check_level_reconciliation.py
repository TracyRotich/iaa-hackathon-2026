"""Review findings R6 and R7: reconcile the level with the row targets.
R7: the level counts claims that match no policy row (targets exclude them). Refit the level
    without them and report the change in the 2025 rate.
R6: row targets gross up with a line-level reporting-delay pattern, the level with peril-level
    patterns. Compare the implied IBNR uplift by loss year on both bases."""
import json
import numpy as np, pandas as pd
from iaa import level, load, target

claims, _ = target.drop_duplicate_claims(load.read("claims"))
ex, matched_ids, out = [], [], {}
for line in ("motor", "buildings", "contents"):
    d = pd.read_parquet(f"data/work/{line}_train.parquet", columns=["policy_year", "exposure"])
    e = d.groupby("policy_year")["exposure"].sum().reset_index(); e.columns = ["year", "exposure"]; e["line"] = line; ex.append(e)
    pol = load.read(f"policies_{line}")[["row_id", "policy_id", "period_start", "period_end"]]
    m, u = target.match_claims(pol, claims[claims.line == line])
    matched_ids.append(m.claim_id)
    u = u[u.status != "repudiated"]
    out[line] = {"unmatched_non_repudiated": len(u),
                 "unmatched_in_2021_24": int(u.loss_date.dt.year.between(2021, 2024).sum()),
                 "unmatched_value_2021_24": round(float((u.incurred_to_date - u.recoveries).clip(lower=0)[u.loss_date.dt.year.between(2021, 2024)].sum()))}
ex = pd.concat(ex)
_, base, _ = level.project(claims, ex)
_, r7, _ = level.project(claims[claims.claim_id.isin(pd.concat(matched_ids))], ex)
cmp = base.merge(r7, on="line", suffixes=("_base", "_matched_only"))
cmp["change_pct"] = 100 * (cmp.cost_rate_2025_matched_only / cmp.cost_rate_2025_base - 1)
for _, r in cmp.iterrows():
    out[r.line]["rate_base"] = round(r.cost_rate_2025_base, 2)
    out[r.line]["rate_matched_only"] = round(r.cost_rate_2025_matched_only, 2)
    out[r.line]["R7_change_pct"] = round(r.change_pct, 3)
# R6: completeness by loss year, line-level pattern vs claim-weighted peril-level patterns
pc = level.peril_completeness(claims)
for line in ("motor", "buildings", "contents"):
    c = claims[(claims.line == line) & (claims.status != "repudiated")]
    grid, cdf = target.reporting_delay_cdf(c)
    res = {}
    for y in (2022, 2023, 2024):
        ages = (target.VALUATION_DATE - pd.date_range(f"{y}-01-01", f"{y}-12-31")).days.to_numpy()
        line_comp = cdf[np.clip(ages, 0, len(grid) - 1)].mean()
        p = pc[(pc.line == line) & (pc.loss_year == y)].set_index("peril").completeness
        n = c[c.loss_date.dt.year == y].peril.value_counts()
        n = n.reindex(p.index).fillna(0)
        peril_comp = n.sum() / (n / p).sum()  # reported / ultimate across perils
        res[y] = {"line_pattern": round(float(line_comp), 4), "peril_patterns": round(float(peril_comp), 4)}
    out[line]["R6_completeness_by_loss_year"] = res
print(json.dumps(out, indent=1))

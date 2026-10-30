"""Diagnostics quoted in the write-up: calibration by predicted decile (A/E) and Gini on the holdouts
of the chosen models, and a gender proxy check on the emerging 2025 predictions (gender is excluded)."""
import json
import numpy as np, pandas as pd
from iaa import metrics

def deciles(y, p, e):
    rate = p / e
    d = pd.qcut(pd.Series(rate).rank(method="first"), 10, labels=False) + 1
    g = pd.DataFrame({"d": d, "A": y, "E": p}).groupby("d")[["A", "E"]].sum()
    return (g.A / g.E).round(2).tolist()

out = {"part1_2024_holdout_v5": {}, "part2_2024_holdout_chosen": {}}
for l in ["motor", "buildings", "contents"]:
    z = np.load(f"data/work/results/v5_{l}_2024.npz"); y, e = z["y"], z["exposure"]; p = z["v5"] * y.sum()
    out["part1_2024_holdout_v5"][l] = {"ae_by_decile": deciles(y, p, e), "gini": round(float(metrics.normalised_gini(y, e, p / e)), 3)}
    em = pd.read_parquet(f"data/emerging/work/{l}_train.parquet", columns=["policy_year", "exposure", "claim_cost"]); te = em[em.policy_year == 2024]
    zt = np.load(f"data/emerging/work/valT_{l}_2024.npz"); k = "prior_cap25k+freqsev@0.25" if l == "motor" else "prior"
    y2, e2 = te.claim_cost.to_numpy(), te.exposure.to_numpy(); p2 = zt[k] * y2.sum()
    out["part2_2024_holdout_chosen"][l] = {"model": k, "ae_by_decile": deciles(y2, p2, e2)}
# gender proxy check: emerging 2025 predictions (gender not used) vs observed 2023-24 experience by gender
sub = pd.read_csv("submissions/part2_sub03_si_fix.csv", dtype={"row_id": str}).set_index("row_id").prediction
gp = {}
for l in ["motor", "buildings", "contents"]:
    te = pd.read_parquet(f"data/emerging/work/{l}_test.parquet", columns=["row_id", "gender", "exposure"])
    tr = pd.read_parquet(f"data/emerging/work/{l}_train.parquet", columns=["gender", "exposure", "claim_cost"])
    te["pred"] = sub.loc[te.row_id.astype(str)].to_numpy()
    pr = te.groupby("gender").apply(lambda g: g.pred.sum() / g.exposure.sum())
    ob = tr.groupby("gender").apply(lambda g: g.claim_cost.sum() / g.exposure.sum())
    gp[l] = {"predicted_2025_F_over_M": round(float(pr["F"] / pr["M"]), 3), "observed_2023_24_F_over_M": round(float(ob["F"] / ob["M"]), 3)}
out["gender_proxy_check"] = gp
print(json.dumps(out, indent=1))
json.dump(out, open("docs/figures/diagnostics.json", "w"), indent=1)

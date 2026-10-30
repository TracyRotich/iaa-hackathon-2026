"""Illustrative office (technical) premium for the 20 emerging reference policies, using the
`expenses` table and the cost-plus formula in  Not part of the Kaggle CSV.
  premium = (risk premium x (1 + cat reinsurance, property only) + fixed expense) /
            (1 - variable expense - commission(channel) - premium tax - target profit)
Assumptions (stated, not given by the organisers): fixed expense charged once per policy row;
percentage loadings are shares of the premium including tax; emerging comparison-site commission
is missing in the table and is shown as not priced rather than set to zero."""
import numpy as np, pandas as pd

ex = pd.read_parquet("data/emerging/expenses.parquet").set_index("item")["emerging"]
ref = pd.read_parquet("data/emerging/reference_policies.parquet")
risk = pd.read_csv("submissions/part2_sub03_si_fix.csv", dtype={"row_id": str}).set_index("row_id").prediction
ref["risk_premium"] = risk.loc[ref.row_id.astype(str)].to_numpy()
comm = ref.channel.map({"direct": ex["commission_direct"], "broker": ex["commission_broker"],
                        "comparison_site": ex["commission_comparison_site"]})
prop = ref.line.isin(["buildings", "contents"])
cost = ref.risk_premium * np.where(prop, 1 + ex["cat_reinsurance"], 1) + ex["fixed_expense_per_policy"]
denom = 1 - ex["variable_expense"] - comm - ex["premium_tax"] - ex["target_profit_margin"]
ref["technical_premium"] = cost / denom
ref["actual_vs_technical"] = ref.written_premium / ref.technical_premium
cols = ["row_id", "line", "channel", "exposure", "new_business", "risk_premium", "technical_premium", "written_premium", "actual_vs_technical"]
t = ref[cols].round(3)
t.to_csv("docs/figures/reference_policy_premiums.csv", index=False)
print("loadings:", ex.to_dict())
print(t.to_string(index=False))
print("\nmedian written / technical by line:", t.groupby("line").actual_vs_technical.median().round(2).to_dict())

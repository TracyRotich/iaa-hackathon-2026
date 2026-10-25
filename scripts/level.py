"""Compute the 2025 cost level by line and peril into data/work/level_*.csv."""
import pandas as pd
from iaa import level, load, target
from iaa import config
WORK = config.WORK_DIR

claims, _ = target.drop_duplicate_claims(load.read("claims"))
ex = []
for line in ("motor", "buildings", "contents"):
    d = pd.read_parquet(f"{WORK}/{line}_train.parquet", columns=["policy_year", "exposure"])
    e = d.groupby("policy_year")["exposure"].sum().reset_index()
    e.columns = ["year", "exposure"]; e["line"] = line; ex.append(e)
perils, lines, stats = level.project(claims, pd.concat(ex))
perils.to_csv(f"{WORK}/level_perils.csv", index=False)
lines.to_csv(f"{WORK}/level_lines.csv", index=False)
stats.to_csv(f"{WORK}/level_stats.csv", index=False)
print(lines.round(2).to_string(index=False))

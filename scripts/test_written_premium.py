"""Holdout test: does the insurer's current written premium add predictive value?
Compares v3-style GBM and GLM with and without log(written_premium) as a feature.
Usage: test_written_premium.py <line>"""
import sys, json
import numpy as np, pandas as pd
from iaa import features, metrics, gbm
from iaa.glm import FreqSevGLM

line = sys.argv[1]
df = pd.read_parquet(f"data/work/{line}_train.parquet")
df["log_written_premium_rate"] = np.log((df.written_premium / df.exposure).clip(lower=1))
drop = ["occupation"] if line == "buildings" else []
base = [f for f in features.feature_list(df) if f not in drop and f != "log_written_premium_rate"]
withp = base + ["log_written_premium_rate"]
w = {"motor": 0.3, "buildings": 0.7, "contents": 0.5}[line]
out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)]; te = df[df.policy_year == test_year]
    y, e = te.claim_cost.values, te.exposure.values
    r = {}
    for name, f in (("base", base), ("with_premium", withp)):
        pg = gbm.fit_predict(line, tr, te, f, te_make_model=(line == "motor"))
        pl = FreqSevGLM(backfit=True).fit(tr[f], tr.n_claims / tr.completeness, tr.reported_cost, tr.exposure).predict(te[f], te.exposure)
        bl = (w * pg / pg.sum() + (1 - w) * pl / pl.sum()) * y.sum()
        r[name] = {k: round(metrics.scorecard(y, p, e)["d2_rescaled"], 4) for k, p in (("gbm", pg), ("glm", pl), ("blend", bl))}
    out[test_year] = r
print(line, json.dumps(out))

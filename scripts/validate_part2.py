"""Emerging market relativity models: train on 2023, score 2024 (the only holdout available).
Scores rescale predictions to the actual 2024 total (allocation only).
Usage: validate_part2.py [line ...]"""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import metrics, gbm, features
from iaa.glm import FreqSevGLM
from iaa.pipeline import norm, onlevel_sum_insured

W = "data/emerging/work"
lines = sys.argv[1:] or ["motor", "buildings", "contents"]
res = {}
for line in lines:
    df = pd.read_parquet(f"{W}/{line}_train.parquet")
    tr, te = df[df.policy_year == 2023], df[df.policy_year == 2024]
    tr, te, g = onlevel_sum_insured(tr, te)
    f = [c for c in features.feature_list(tr) if not c.startswith("wx_prev_")]
    y = te.claim_cost.to_numpy(); n = tr.n_claims / tr.completeness
    P = {}
    P["glm"] = norm(FreqSevGLM(backfit=True).fit(tr[f], n, tr.claim_cost, tr.exposure).predict(te[f], te.exposure))
    P["gbm_p1"] = norm(gbm.fit_predict(line, tr, te, f, te_make_model=line == "motor"))
    for k in ("shallow", "freqsev"):
        P[k] = norm(gbm.fit_predict_variant(tr, te, f, k, tr.claim_cost))
    if line == "motor":
        cap = tr.assign(claim_cost=tr.reported_cost.clip(upper=25_000) / tr.completeness)
        P["glm_cap25k"] = norm(FreqSevGLM(backfit=True).fit(cap[f], n, cap.claim_cost, cap.exposure).predict(te[f], te.exposure))
    if "gender" in tr:
        fg = f + ["gender"]
        P["glm_with_gender_TEST_ONLY"] = norm(FreqSevGLM(backfit=True).fit(tr[fg], n, tr.claim_cost, tr.exposure).predict(te[fg], te.exposure))
    for a in ("gbm_p1", "shallow", "freqsev"):
        for w in (0.25, 0.5, 0.75):
            P[f"glm+{a}@{w}"] = (1 - w) * P["glm"] + w * P[a]
    s = {k: round(metrics.d2(y, v * y.sum()), 5) for k, v in P.items()}
    res[line] = s
    np.savez_compressed(f"{W}/val_{line}_2024.npz", y=y, exposure=te.exposure.to_numpy(), **P)
    print(line, json.dumps(dict(sorted(s.items(), key=lambda kv: -kv[1]))), flush=True)
    del df, tr, te, P; gc.collect()

"""Holdout tests of on-levelled sums insured and credibility smoothing.
Usage: gi_pricing_experiments.py <line>
  onlevel  : sum insured on-levelled to a common year (s.10.3.2), estimated from same-policy growth
  backfit  : high-cardinality factors credibility-smoothed on A/E after a base GLM (s.28.1.2)
"""
import sys, json
import numpy as np, pandas as pd
from iaa import features, metrics, gbm
from iaa.glm import FreqSevGLM, CredibilityEncoder

line = sys.argv[1]
df = pd.read_parquet(f"data/work/{line}_train.parquet")
drop = ["occupation"] if line == "buildings" else []
feats = [f for f in features.feature_list(df) if f not in drop]


def si_index(tr):
    """Annual sum-insured indexation from the same policy in consecutive years (training only)."""
    p = tr.pivot_table(index="policy_id", columns="policy_year", values="sum_insured", aggfunc="first")
    yrs = sorted(p.columns)
    g = np.nanmedian(np.concatenate([(p[y] / p[x]).dropna().to_numpy() for x, y in zip(yrs, yrs[1:])]))
    return float(g)


def onlevel(frame, g, base_year=2021):
    f = frame.copy()
    f["sum_insured"] = f["sum_insured"] / g ** (f["policy_year"] - base_year)
    return f


def score(te, p):
    return {k: round(v, 4) for k, v in metrics.scorecard(te.claim_cost.values, p, te.exposure.values).items()
            if k in ("d2_rescaled", "gini_norm")}


out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)]; te = df[df.policy_year == test_year]
    n_tr = tr.n_claims / tr.completeness
    r = {}
    p_glm = FreqSevGLM().fit(tr[feats], n_tr, tr.reported_cost, tr.exposure).predict(te[feats], te.exposure)
    r["glm"] = score(te, p_glm)
    p_gbm = gbm.fit_predict(line, tr, te, feats, te_make_model=(line == "motor"))
    r["gbm"] = score(te, p_gbm)
    if "sum_insured" in feats:
        g = si_index(tr); r["si_growth"] = round(g, 4)
        tr2, te2 = onlevel(tr, g), onlevel(te, g)
        p = FreqSevGLM().fit(tr2[feats], n_tr, tr2.reported_cost, tr2.exposure).predict(te2[feats], te2.exposure)
        r["glm_onlevel"] = score(te, p)
        r["gbm_onlevel"] = score(te, gbm.fit_predict(line, tr2, te2, feats, te_make_model=(line == "motor")))
    m = FreqSevGLM(backfit=True).fit(tr[feats], n_tr, tr.reported_cost, tr.exposure)
    if m.design.te:
        p_bf = m.predict(te[feats], te.exposure)
        r["glm_backfit"] = score(te, p_bf)
        r["backfit_kappa(freq,sev)"] = m.kappas()
        # blend check with the backfitted GLM
        a = p_gbm / p_gbm.sum()
        for name, pg in (("blend_glm", p_glm), ("blend_glm_backfit", p_bf)):
            w = {"motor": 0.3, "contents": 0.5}.get(line, 0.7)
            r[name] = score(te, (w * a + (1 - w) * pg / pg.sum()) * te.claim_cost.sum())
    out[test_year] = r
print(line, json.dumps(out))

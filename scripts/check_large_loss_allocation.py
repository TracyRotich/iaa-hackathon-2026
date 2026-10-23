"""Review of submission 4 (motor basic-limit cap): how should the cost above the cap be allocated?
 A  proportional: capped relativities rescaled to the full total (what submission 4 does)
 B  capped relativities for the attritional share + large-loss load per unit of exposure (uniform)
 C  capped relativities for the attritional share + large-loss load in proportion to the uncapped model
 U  uncapped (submission 3 method)
Scores D² and actual/expected by cover type and age band on the 2023 and 2024 holdouts."""
import json
import numpy as np, pandas as pd
from iaa import features, metrics, gbm
from iaa.glm import FreqSevGLM

CAP, W = 25_000, 0.3
df = pd.read_parquet("data/work/motor_train.parquet")
feats = features.feature_list(df)


def blend(tr, te):
    pg = gbm.fit_predict("motor", tr, te, feats, te_make_model=True)
    pl = FreqSevGLM(backfit=True).fit(tr[feats], tr.n_claims / tr.completeness, tr.reported_cost, tr.exposure).predict(te[feats], te.exposure)
    return W * pg / pg.sum() + (1 - W) * pl / pl.sum()


out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)].copy(); te = df[df.policy_year == test_year].copy()
    rep_c = tr.reported_cost.clip(upper=CAP)
    tr_c = tr.assign(reported_cost=rep_c, claim_cost=rep_c / tr.completeness)
    share = rep_c.sum() / tr.reported_cost.sum()          # attritional share of cost, from training years
    u = blend(tr, te); c = blend(tr_c, te)
    y = te.claim_cost.to_numpy(); e = te.exposure.to_numpy(); T = y.sum()   # level-neutral comparison
    preds = {"U_uncapped": u * T,
             "A_proportional": c * T,
             "B_uniform_load": share * T * c + (1 - share) * T * e / e.sum(),
             "C_load_by_uncapped": share * T * c + (1 - share) * T * u}
    r = {"attritional_share": round(share, 4)}
    for k, p in preds.items():
        seg = {}
        for col, bands in (("cover_type", None), ("driver_age", [16, 21, 25, 30, 60, 100])):
            key = te[col] if bands is None else pd.cut(te[col], bands).astype(str)
            g = pd.DataFrame({"k": key.values, "y": y, "p": p}).groupby("k").sum()
            seg.update({f"{col}={i}": round(v, 3) for i, v in (g.y / g.p).items()})
        r[k] = {"d2": round(metrics.d2(y, p), 4), "AE": seg}
    out[test_year] = r
print(json.dumps(out))

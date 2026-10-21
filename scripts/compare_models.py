"""LightGBM vs GLM (Poisson x Gamma) vs blend, time holdouts. Usage: compare_models.py <line>"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from iaa import features, metrics
from iaa.glm import FreqSevGLM
from iaa.config import SEED, TWEEDIE_POWER

line = sys.argv[1]
df = pd.read_parquet(f"data/work/{line}_train.parquet")
feats = features.feature_list(df)
PARAMS = dict(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, learning_rate=0.03, num_leaves=31,
              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              cat_smooth=50, min_data_per_group=200, max_cat_to_onehot=8, verbose=-1, random_state=SEED)
TREES = {"motor": 130, "buildings": 160, "contents": 140}

def gbm(tr, te):
    cats = {c: sorted(tr[c].dropna().astype(str).unique()) for c in feats if not pd.api.types.is_numeric_dtype(tr[c])}
    def prep(X):
        X = X[feats].copy()
        for c, lv in cats.items():
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=lv)
        return X
    m = lgb.LGBMRegressor(n_estimators=TREES[line], **PARAMS).fit(prep(tr), tr.claim_cost / tr.exposure, sample_weight=tr.exposure)
    return m.predict(prep(te)) * te.exposure.values

out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)]; te = df[df.policy_year == test_year]
    t0 = time.time(); p_gbm = gbm(tr, te); t_gbm = time.time() - t0
    t0 = time.time()
    g = FreqSevGLM().fit(tr[feats], tr.n_claims / tr.completeness, tr.reported_cost, tr.exposure)
    p_glm = g.predict(te[feats], te.exposure); t_glm = time.time() - t0
    y = te.claim_cost.values; e = te.exposure.values
    a = p_gbm / p_gbm.sum(); b = p_glm / p_glm.sum()
    res = {"gbm": metrics.scorecard(y, p_gbm, e), "glm": metrics.scorecard(y, p_glm, e)}
    for w in (0.3, 0.5, 0.7):
        res[f"blend_gbm{w}"] = metrics.scorecard(y, (w * a + (1 - w) * b) * y.sum(), e)
    out[test_year] = {k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}
    out[test_year]["secs"] = {"gbm": round(t_gbm), "glm": round(t_glm)}
    out[test_year]["kappa(freq,sev)"] = g.kappas()
    out[test_year]["glm_columns"] = len(g.design.names_)
print(line, json.dumps(out, indent=1))

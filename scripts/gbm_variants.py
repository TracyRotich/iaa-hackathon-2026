"""GBM variants suggested by the credibility results. Usage: gbm_variants.py <line> <variant>
variants: base | drop_occupation | te_make_model"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from iaa import features, metrics
from iaa.glm import CredibilityEncoder
from iaa.config import SEED, TWEEDIE_POWER

line, variant = sys.argv[1], sys.argv[2]
df = pd.read_parquet(f"data/work/{line}_train.parquet")
feats = features.feature_list(df)
if variant == "drop_occupation":
    feats = [f for f in feats if f != "occupation"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, learning_rate=0.03, num_leaves=31,
              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              cat_smooth=50, min_data_per_group=200, max_cat_to_onehot=8, verbose=-1, random_state=SEED)
TREES = {"motor": 130, "buildings": 160, "contents": 140}
out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)].copy(); te = df[df.policy_year == test_year].copy()
    f = list(feats)
    if variant == "te_make_model":
        n = (tr.n_claims / tr.completeness).values; e = tr.exposure.values; c = tr.reported_cost.values
        for name, y, w, pv in [("mm_freq", n, e, n.sum() / e.sum()), ("mm_cost", c, e, None)]:
            if pv is None:  # cost rate: process variance estimated from the data
                pv = np.var(c / e, ddof=1) * np.mean(e)
            enc = CredibilityEncoder().fit(tr.make_model, y, w, pv)
            tr[name] = enc.transform_train(tr.make_model, y, w); te[name] = enc.transform(te.make_model)
            f.append(name)
        f.remove("make_model")
    cats = {c: sorted(tr[c].dropna().astype(str).unique()) for c in f if not pd.api.types.is_numeric_dtype(tr[c])}
    def prep(X):
        X = X[f].copy()
        for c, lv in cats.items():
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=lv)
        return X
    m = lgb.LGBMRegressor(n_estimators=TREES[line], **PARAMS).fit(prep(tr), tr.claim_cost / tr.exposure, sample_weight=tr.exposure)
    p = m.predict(prep(te)) * te.exposure.values
    out[test_year] = {k: round(v, 4) for k, v in metrics.scorecard(te.claim_cost.values, p, te.exposure.values).items()}
print(line, variant, json.dumps(out))

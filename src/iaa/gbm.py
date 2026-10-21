"""LightGBM Tweedie per line, with the credibility-encoded make_model option."""
import numpy as np, pandas as pd, lightgbm as lgb
from .config import SEED, TWEEDIE_POWER
from .glm import CredibilityEncoder

PARAMS = dict(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, learning_rate=0.03, num_leaves=31,
              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              cat_smooth=50, min_data_per_group=200, max_cat_to_onehot=8, verbose=-1, random_state=SEED)
TREES = {"motor": 130, "buildings": 160, "contents": 140}


def fit_predict(line, tr, te, feats, te_make_model=False):
    tr, te, f = tr.copy(), te.copy(), list(feats)
    if te_make_model and "make_model" in f:
        n = (tr.n_claims / tr.completeness).to_numpy(); e = tr.exposure.to_numpy(); c = tr.reported_cost.to_numpy()
        for name, y, w, pv in [("mm_freq", n, e, n.sum() / e.sum()), ("mm_cost", c, e, np.var(c / e, ddof=1) * np.mean(e))]:
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
    return m.predict(prep(te)) * te.exposure.to_numpy()

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


# --------------------------------------------------------------------------- review variants
# Ported from the Codex review (23 Oct), settings unchanged so v5 reproduces.
# Fix R3: for Tweedie p = 1.5, row deviance D(e*y, e*mu) = sqrt(e) * D(y, mu), so a rate model
# matches the competition's row-level metric with weight sqrt(exposure), not exposure.
VARIANT_PARAMS = dict(learning_rate=0.025, min_child_samples=700, reg_lambda=20, colsample_bytree=0.9,
                      subsample=0.9, subsample_freq=1, verbosity=-1, n_jobs=4, random_state=713,
                      cat_smooth=100, min_data_per_group=500, max_cat_to_onehot=8)


def fit_predict_variant(tr, te, feats, kind, cost):
    """kind: 'shallow' (7 leaves, 800 trees) or 'medium' (15 leaves, 500 trees) Tweedie on the
    rate with sqrt(exposure) weights; 'freqsev' = Poisson frequency x Gamma severity GBMs.
    cost: ultimate-basis row cost (same basis as the IBNR-adjusted counts)."""
    X, Z = tr[feats].copy(), te[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            lv = sorted(X[c].dropna().astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=lv)
            v = Z[c].astype(str).where(Z[c].notna())
            Z[c] = pd.Categorical(v.where(v.isin(lv)), categories=lv)
    e = tr.exposure.to_numpy(); n = (tr.n_claims / tr.completeness).to_numpy(); y = np.asarray(cost, float)
    p = dict(VARIANT_PARAMS, num_leaves=7 if kind == "shallow" else 15)
    if kind == "freqsev":
        f = lgb.LGBMRegressor(objective="poisson", n_estimators=500, **p).fit(X, n / e, sample_weight=e)
        pos = (n > 0) & (y > 0)
        s = lgb.LGBMRegressor(objective="gamma", n_estimators=180, **p).fit(X.loc[pos], y[pos] / n[pos], sample_weight=n[pos])
        return f.predict(Z) * s.predict(Z) * te.exposure.to_numpy()
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER,
                          n_estimators=800 if kind == "shallow" else 500, **p).fit(X, y / e, sample_weight=np.sqrt(e))
    return m.predict(Z) * te.exposure.to_numpy()

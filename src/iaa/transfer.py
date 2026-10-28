"""Transfer from the developed market to the emerging market (Part 2).

The emerging market has about 1/20 of the developed data (two years, ~1,000 property claims a
line), and models fitted on it alone overfit (28 Oct holdout: GLMs scored below the
constant-rate baseline on buildings and contents). The developed market is used as a prior:

1. Harmonise: lower-case text, map developed ages to the emerging age bands, and keep only
   features coded the same way in both markets (region, construction-type and occupation codes
   differ; prior-year weather is dropped, finding R2).
2. Prior: frequency (Poisson) x severity (Gamma) LightGBM fitted on all developed years.
3. Re-level: the prior is scaled to the emerging training frequency and severity (the markets
   differ in level: emerging motor severity is ~60% of developed).
4. Boost from the prior: a short LightGBM run on emerging data starting from log(prior)
   (init_score), so emerging data corrects the developed shape only where it has evidence.
This is the boosting analogue of credibility: the developed model is the collateral estimate,
and the number of emerging trees controls how far the emerging experience can move it.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import features
from .config import ID_COLS
from .gbm import VARIANT_PARAMS

AGE_BINS = [0, 25, 35, 50, 65, 200]
AGE_LABELS = ["17-24", "25-34", "35-49", "50-64", "65+"]
AGE_COL = {"motor": ("driver_age", "driver_age_band"), "buildings": ("policyholder_age", "policyholder_age_band"),
           "contents": ("policyholder_age", "policyholder_age_band")}
NOT_TRANSFERABLE = {"region", "construction_type", "occupation", "previous_insurer", "make_model"}


def harmonise(df, line):
    df = df.copy(deep=False)
    age, band = AGE_COL[line]
    if age in df and band not in df:
        df[band] = pd.cut(df[age], AGE_BINS, right=False, labels=AGE_LABELS).astype("string")
    for c in df.columns:
        if c in ID_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip().str.lower()
    return df


def transfer_features(dev, em):
    fd, fe = set(features.feature_list(dev)), features.feature_list(em)
    return [c for c in fe if c in fd and c not in NOT_TRANSFERABLE and not c.startswith("wx_prev_")]


def _cats(X, feats):
    return {c: sorted(X[c].dropna().astype(str).unique()) for c in feats if not pd.api.types.is_numeric_dtype(X[c])}


def _prep(X, feats, cats):
    X = X[feats].copy()
    for c, lv in cats.items():
        v = X[c].astype(str).where(X[c].notna())
        X[c] = pd.Categorical(v.where(v.isin(lv)), categories=lv)
    return X


def _counts_cost(df, cap=None):
    n = (df.n_claims / df.completeness).to_numpy()
    c = (df.reported_cost.clip(upper=cap) / df.completeness).to_numpy() if cap else df.claim_cost.to_numpy()
    return n, c


class DevPrior:
    def __init__(self, n_freq=500, n_sev=180, leaves=15, cap=None):
        self.n_freq, self.n_sev, self.leaves, self.cap = n_freq, n_sev, leaves, cap

    def fit(self, dev, feats):
        self.feats, self.cats = feats, _cats(dev, feats)
        X = _prep(dev, feats, self.cats); e = dev.exposure.to_numpy(); n, c = _counts_cost(dev, self.cap)
        p = dict(VARIANT_PARAMS, num_leaves=self.leaves)
        self.f = lgb.LGBMRegressor(objective="poisson", n_estimators=self.n_freq, **p).fit(X, n / e, sample_weight=e)
        pos = (n > 0) & (c > 0)
        self.s = lgb.LGBMRegressor(objective="gamma", n_estimators=self.n_sev, **p).fit(X.loc[pos], c[pos] / n[pos], sample_weight=n[pos])
        return self

    def predict(self, df):
        X = _prep(df, self.feats, self.cats)
        return self.f.predict(X), self.s.predict(X)  # frequency per exposure year, severity


def boost_from_prior(tr, te, feats, prior_tr, prior_te, n_freq=150, n_sev=60, lr=0.02, leaves=7, cap=None):
    """Emerging frequency and severity models started at the re-levelled developed prior.
    n_freq = n_sev = 0 returns the re-levelled prior itself."""
    e = tr.exposure.to_numpy(); n, c = _counts_cost(tr, cap)
    fq_tr, sv_tr = prior_tr; fq_te, sv_te = prior_te
    pos = (n > 0) & (c > 0)
    kf = n.sum() / (fq_tr * e).sum()                         # re-level frequency
    ks = c[pos].sum() / (sv_tr[pos] * n[pos]).sum()          # re-level severity (claim-weighted)
    fq_tr, fq_te, sv_tr, sv_te = fq_tr * kf, fq_te * kf, sv_tr * ks, sv_te * ks
    if n_freq == 0 and n_sev == 0:
        return fq_te * sv_te * te.exposure.to_numpy()
    cats = _cats(tr, feats); X, Z = _prep(tr, feats, cats), _prep(te, feats, cats)
    p = dict(VARIANT_PARAMS, num_leaves=leaves, learning_rate=lr, min_child_samples=300, min_data_per_group=200)
    f = lgb.LGBMRegressor(objective="poisson", n_estimators=n_freq, **p)
    f.fit(X, n / e, sample_weight=e, init_score=np.log(fq_tr))
    s = lgb.LGBMRegressor(objective="gamma", n_estimators=n_sev, **p)
    s.fit(X.loc[pos], c[pos] / n[pos], sample_weight=n[pos], init_score=np.log(sv_tr[pos]))
    # predict() excludes init_score: add the prior back on the log scale
    f_te = np.exp(np.log(fq_te) + f.predict(Z, raw_score=True))
    s_te = np.exp(np.log(sv_te) + s.predict(Z, raw_score=True))
    return f_te * s_te * te.exposure.to_numpy()

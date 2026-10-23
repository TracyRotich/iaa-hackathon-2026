"""GLM benchmark: Poisson frequency x Gamma severity.

Frequency x severity per line:
- Frequency: Poisson GLM, log link, rate = claims / exposure with weight = exposure
  (equivalent to a log(exposure) offset).
- Severity: Gamma GLM, log link, average claim per claiming row, weight = number of claims.
- Both with a ridge (L2) penalty, a shrinkage GLM.

Pre-processing, all fitted on training data only and frozen for test (no leakage):
- Numeric features with many values: clipped at the training 1st/99th percentiles,
  missing values set to the training median plus a missing flag, then natural-style cubic
  splines (knots at training quantiles, as ns(df=3) in R).
- Low-cardinality categoricals: one-hot, with missing as its own level.
- High-cardinality categoricals: actuarial target encoding (credibility-weighted towards the
  portfolio mean, leave-one-out on training rows). kappa is estimated by Buhlmann-Straub
  rather than fixed at 20, the fully data-driven choice.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.preprocessing import SplineTransformer

HIGH_CARD = 15  # categoricals with more levels than this are target-encoded
SPLINE_MIN_UNIQUE = 12
N_KNOTS = 4  # spline flexibility (tested 4 vs 6 on holdouts, 23 Oct)


# --------------------------------------------------------------------------- credibility

def buhlmann_kappa(sum_y, sum_w, process_var):
    """Buhlmann-Straub k = s^2 / a for group means of y per unit weight.

    sum_y, sum_w: per-group totals. process_var: expected within-group variance per unit
    weight (s^2). Returns k in weight units; inf when there is no between-group signal.
    """
    sum_y, sum_w = np.asarray(sum_y, float), np.asarray(sum_w, float)
    keep = sum_w > 0
    sum_y, sum_w = sum_y[keep], sum_w[keep]
    w = sum_w.sum()
    xbar_g = sum_y / sum_w
    xbar = sum_y.sum() / w
    G = len(sum_w)
    a = (np.sum(sum_w * (xbar_g - xbar) ** 2) - (G - 1) * process_var) / (w - np.sum(sum_w ** 2) / w)
    return float(process_var / a) if a > 0 else np.inf


class CredibilityEncoder:
    """Actuarial target encoding: Z * group mean + (1 - Z) * portfolio mean, Z = w / (w + k).
    Training rows use leave-one-out totals, so a row never sees its own outcome."""

    def __init__(self, kappa=None):
        self.kappa = kappa

    def fit(self, groups, y, w, process_var):
        g = pd.Series(groups).astype(str).fillna("NA").to_numpy()
        df = pd.DataFrame({"g": g, "y": y, "w": w})
        tot = df.groupby("g")[["y", "w"]].sum()
        self.mean_ = df["y"].sum() / df["w"].sum()
        k = self.kappa if self.kappa is not None else buhlmann_kappa(tot["y"], tot["w"], process_var)
        self.kappa_ = k
        z = tot["w"] / (tot["w"] + k) if np.isfinite(k) else 0 * tot["w"]
        mu = tot["y"] / tot["w"].where(tot["w"] > 0)
        self.table_ = (z * mu.fillna(self.mean_) + (1 - z) * self.mean_).to_dict()
        self._tot = tot
        return self

    def transform_train(self, groups, y, w):
        g = pd.Series(groups).astype(str).fillna("NA").to_numpy()
        ty = self._tot["y"].reindex(g).to_numpy() - y
        tw = self._tot["w"].reindex(g).to_numpy() - w
        k = self.kappa_
        z = tw / (tw + k) if np.isfinite(k) else np.zeros_like(tw)
        mu = np.where(tw > 0, ty / np.where(tw > 0, tw, 1), self.mean_)
        return z * mu + (1 - z) * self.mean_

    def transform(self, groups):
        g = pd.Series(groups).astype(str).fillna("NA")
        return g.map(self.table_).fillna(self.mean_).to_numpy(float)


# --------------------------------------------------------------------------- design matrix

class Design:
    """Fit on training frame, then build identical columns for any frame."""

    def fit(self, X):
        self.num, self.spl, self.ohe, self.te = [], [], {}, []
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                (self.spl if s.nunique() >= SPLINE_MIN_UNIQUE else self.num).append(c)
            else:
                lv = s.astype(str).where(s.notna(), "NA").value_counts()
                if len(lv) > HIGH_CARD:
                    self.te.append(c)
                else:
                    self.ohe[c] = [v for v in lv.index[1:]]  # drop most common level as base
        self.med = X[self.num + self.spl].median()
        self.lo = X[self.spl].quantile(0.01)
        self.hi = X[self.spl].quantile(0.99)
        self.missing = [c for c in self.num + self.spl if X[c].isna().any()]
        self.splines = {}
        for c in self.spl:
            v = X[c].fillna(self.med[c]).clip(self.lo[c], self.hi[c]).to_numpy()[:, None]
            self.splines[c] = SplineTransformer(n_knots=N_KNOTS, degree=3, knots="quantile",
                                                extrapolation="linear", include_bias=False).fit(v)
        self.num_mean = X[self.num].fillna(self.med[self.num]).mean()
        self.num_sd = X[self.num].fillna(self.med[self.num]).std().replace(0, 1)
        return self

    def transform(self, X, te_values=None):
        cols, names = [], []
        for c in self.num:
            cols.append(((X[c].fillna(self.med[c]) - self.num_mean[c]) / self.num_sd[c]).to_numpy())
            names.append(c)
        for c in self.missing:
            cols.append(X[c].isna().to_numpy(float)); names.append(f"{c}__missing")
        for c in self.spl:
            v = X[c].fillna(self.med[c]).clip(self.lo[c], self.hi[c]).to_numpy()[:, None]
            b = self.splines[c].transform(v)
            cols.extend(b.T); names.extend(f"{c}__s{j}" for j in range(b.shape[1]))
        for c, levels in self.ohe.items():
            s = X[c].astype(str).where(X[c].notna(), "NA").to_numpy()
            for lv in levels:
                cols.append((s == lv).astype(float)); names.append(f"{c}={lv}")
        for c in self.te:
            cols.append(np.log(te_values[c])); names.append(f"log_te_{c}")
        self.names_ = names
        return np.column_stack(cols)


# --------------------------------------------------------------------------- model

class FreqSevGLM:
    def __init__(self, alpha_freq=1e-4, alpha_sev=1e-3, kappa=None, backfit=False):
        """backfit=True: high-cardinality factors are credibility-smoothed on actual/expected
        relative to a first GLM fitted without them (backfitting), so their effect is not double-counted with correlated ordinary factors."""
        self.alpha_freq, self.alpha_sev, self.kappa, self.backfit = alpha_freq, alpha_sev, kappa, backfit

    def fit(self, X, n_claims, cost, exposure):
        n = np.asarray(n_claims, float); c = np.asarray(cost, float); e = np.asarray(exposure, float)
        self.design = Design().fit(X)
        pos = (n > 0) & (c > 0)  # rows whose claims net to zero after recoveries carry no severity information
        sev_mean = c[pos].sum() / n[pos].sum()
        sev_var = np.average((c[pos] / n[pos] - sev_mean) ** 2, weights=n[pos])
        self.enc_f, self.enc_s, tf, ts = {}, {}, {}, {}
        if self.backfit and self.design.te:
            return self._fit_backfit(X, n, c, e, pos)
        for col in self.design.te:
            ef = CredibilityEncoder(self.kappa).fit(X[col], n, e, process_var=n.sum() / e.sum())
            es = CredibilityEncoder(self.kappa).fit(X[col], c, n, process_var=sev_var)
            self.enc_f[col], self.enc_s[col] = ef, es
            tf[col] = np.clip(ef.transform_train(X[col], n, e), 1e-6, None)
            ts[col] = np.clip(es.transform_train(X[col], c, n), 1e-6, None)
        Xf = self.design.transform(X, tf)
        self.freq = PoissonRegressor(alpha=self.alpha_freq, max_iter=300).fit(Xf, n / e, sample_weight=e)
        Xs = self.design.transform(X[pos], {k: v[pos] for k, v in ts.items()})
        self.sev = GammaRegressor(alpha=self.alpha_sev, max_iter=300).fit(Xs, c[pos] / n[pos], sample_weight=n[pos])
        return self

    def _fit_backfit(self, X, n, c, e, pos):
        base = [col for col in X.columns if col not in self.design.te]
        self.base = FreqSevGLM(self.alpha_freq, self.alpha_sev).fit(X[base], n, c, e)
        Xb = X[base]
        mu_n = self.base.freq.predict(self.base.design.transform(Xb, {})) * e
        mu_s = self.base.sev.predict(self.base.design.transform(Xb, {}))
        ys = np.where(pos, c / mu_s, 0.0)  # cost in units of expected severity
        phi = np.sum(n[pos] * (c[pos] / (n[pos] * mu_s[pos]) - 1) ** 2) / n[pos].sum()
        tf, ts = {}, {}
        for col in self.design.te:
            ef = CredibilityEncoder(self.kappa).fit(X[col], n, mu_n, process_var=1.0)
            es = CredibilityEncoder(self.kappa).fit(X[col][pos], ys[pos], n[pos], process_var=phi)
            self.enc_f[col], self.enc_s[col] = ef, es
            tf[col] = np.clip(ef.transform_train(X[col], n, mu_n), 1e-6, None)
            s_tr = np.full(len(n), es.mean_)
            s_tr[pos] = es.transform_train(X[col][pos], ys[pos], n[pos])
            ts[col] = np.clip(s_tr, 1e-6, None)
        Xf = self.design.transform(X, tf)
        self.freq = PoissonRegressor(alpha=self.alpha_freq, max_iter=300).fit(Xf, n / e, sample_weight=e)
        Xs = self.design.transform(X[pos], {k: v[pos] for k, v in ts.items()})
        self.sev = GammaRegressor(alpha=self.alpha_sev, max_iter=300).fit(Xs, c[pos] / n[pos], sample_weight=n[pos])
        return self

    def predict(self, X, exposure):
        tf = {k: np.clip(v.transform(X[k]), 1e-6, None) for k, v in self.enc_f.items()}
        ts = {k: np.clip(v.transform(X[k]), 1e-6, None) for k, v in self.enc_s.items()}
        f = self.freq.predict(self.design.transform(X, tf))
        s = self.sev.predict(self.design.transform(X, ts))
        return f * s * np.asarray(exposure, float)

    def kappas(self):
        return {c: (round(self.enc_f[c].kappa_, 1), round(self.enc_s[c].kappa_, 1)) for c in self.enc_f}

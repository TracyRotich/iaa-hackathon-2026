"""Credibility-style frequency correction on top of the re-levelled developed prior.
log E[n] = log(prior freq x exposure) + beta' x, with x = one-hot emerging factors (region,
payment method, new business, age band, urban density, channel) and a ridge penalty alpha that
shrinks beta towards 0 (= trust the developed shape). Fitted on emerging 2023, scored on 2024.
Usage: validate_correction.py [line ...]"""
import sys, json, gc
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor
from iaa import metrics, transfer
from iaa.pipeline import norm

W = "data/emerging/work"
CAP = {"motor": 25_000}
FACTORS = ["region", "payment_method", "new_business", "urban_density", "channel"]


def onehot(df, cols, levels=None):
    X = pd.get_dummies(df[cols].astype(str), drop_first=False, dtype=float)
    return X.reindex(columns=levels, fill_value=0.0) if levels is not None else X


def correct(tr, te, ftr, fte, cols, alpha):
    """Return multiplicative frequency corrections for te rows."""
    e = tr.exposure.to_numpy(); n = (tr.n_claims / tr.completeness).to_numpy()
    mu = ftr * e
    X = onehot(tr, cols); Z = onehot(te, cols, X.columns)
    m = PoissonRegressor(alpha=alpha, max_iter=500).fit(X, n / mu, sample_weight=mu)
    return m.predict(Z) / m.predict(X).dot(mu) * mu.sum()   # keep the training total unchanged


if __name__ == "__main__":
    for line in sys.argv[1:] or ["motor", "buildings", "contents"]:
        dev = transfer.harmonise(pd.read_parquet(f"data/work/{line}_train.parquet"), line)
        em = transfer.harmonise(pd.read_parquet(f"{W}/{line}_train.parquet"), line)
        feats = transfer.transfer_features(dev, em)
        cap = CAP.get(line)
        prior = transfer.DevPrior(cap=cap).fit(dev, feats); del dev; gc.collect()
        tr, te = em[em.policy_year == 2023], em[em.policy_year == 2024]
        (ftr, str_), (fte, ste) = prior.predict(tr), prior.predict(te)
        base = transfer.boost_from_prior(tr, te, feats, (ftr, str_), (fte, ste), 0, 0, cap=cap)
        y = te.claim_cost.to_numpy(); band = transfer.AGE_COL[line][1]
        cols = FACTORS + [band]
        s = {"prior": metrics.d2(y, norm(base) * y.sum())}
        if line == "motor":
            fs = np.load(f"{W}/val_{line}_2024.npz")["freqsev"]
            s["prior+freqsev@0.25"] = metrics.d2(y, (0.75 * norm(base) + 0.25 * fs) * y.sum())
        for alpha in (1e-1, 1e-2, 3e-3, 1e-3):
            k = correct(tr, te, ftr, fte, cols, alpha)
            p = norm(base * k)
            s[f"corr_a{alpha}"] = metrics.d2(y, p * y.sum())
            if line == "motor":
                s[f"corr_a{alpha}+freqsev@0.25"] = metrics.d2(y, (0.75 * p + 0.25 * fs) * y.sum())
        # one factor at a time at a middle penalty, to see which corrections carry signal
        for c in cols:
            s[f"only_{c}"] = metrics.d2(y, norm(base * correct(tr, te, ftr, fte, [c], 3e-3)) * y.sum())
        print(line, json.dumps({k: round(v, 5) for k, v in sorted(s.items(), key=lambda kv: -kv[1])}), flush=True)
        del em, tr, te, prior; gc.collect()

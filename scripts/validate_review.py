"""Tests prompted by the ChatGPT review (28 Oct), on the emerging 2024 holdout (train emerging 2023).

Variants of the developed prior (all re-levelled to emerging 2023):
  old        : as submitted (saved predictions from validate_transfer.py)
  si_fix     : sums insured repaired (1,000x) and put in 2023 money in both markets
  si_fix_nb  : si_fix x credibility new-business factor estimated on emerging 2023 only
  prospective: si_fix with the prior fitted on developed years before 2024 only (strict time test)
Scores (allocation only, rescaled to the actual 2024 total):
  d2_all     : all 2024 rows
  d2_nb      : new-business rows only
  d2_nbw     : rows reweighted so new business is 77% of exposure, as in 2025
Paired bootstrap by policy (300 draws) of the deviance gain of si_fix over the emerging-only GBM."""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import metrics, transfer
from iaa.pipeline import norm

W = "data/emerging/work"
CAP = {"motor": 25_000}
NB_SHARE_2025 = 0.77


def dev15(y, mu):
    mu = np.maximum(mu, 1e-9)
    return 4 * (np.sqrt(mu) + y / np.sqrt(mu) - 2 * np.sqrt(y))


def scores(y, p, nb, e):
    p = p * y.sum()
    w = np.where(nb, NB_SHARE_2025 / e[nb].sum(), (1 - NB_SHARE_2025) / e[~nb].sum()) * e.sum()
    return {"d2_all": metrics.d2(y, p), "d2_nb": metrics.d2(y[nb], p[nb] * y[nb].sum() / p[nb].sum()),
            "d2_nbw": metrics.d2(y, p, sample_weight=w)}


out = {}
for line in sys.argv[1:] or ["contents", "buildings", "motor"]:
    cap = CAP.get(line)
    dev = transfer.harmonise(pd.read_parquet(f"data/work/{line}_train.parquet"), line)
    em = transfer.harmonise(pd.read_parquet(f"{W}/{line}_train.parquet"), line)
    feats = transfer.transfer_features(dev, em)
    tr, te = em[em.policy_year == 2023], em[em.policy_year == 2024]
    if "sum_insured" in dev:
        g_dev = transfer.si_growth(dev); g_em = transfer.si_growth(em)  # emerging growth from both years
        dev = transfer.onlevel_si(dev, g_dev); tr = transfer.onlevel_si(tr, g_em); te = transfer.onlevel_si(te, g_em)
    y = te.claim_cost.to_numpy(); e = te.exposure.to_numpy(); nb = te.new_business.astype(bool).to_numpy()
    old = dict(np.load(f"{W}/valT_{line}_2024.npz")); fs = dict(np.load(f"{W}/val_{line}_2024.npz"))["freqsev"]
    P = {"old": old["prior_cap25k" if cap else "prior"], "emerging_only_freqsev": fs}
    res = {}
    for tag, years in [("si_fix", None), ("prospective", 2024)]:
        d = dev if years is None else dev[dev.policy_year < years]
        prior = transfer.DevPrior(cap=cap).fit(d, feats)
        (ftr, s_tr), pte = prior.predict(tr), prior.predict(te)
        base = transfer.boost_from_prior(tr, te, feats, (ftr, s_tr), pte, 0, 0, cap=cap)
        P[tag] = norm(base)
        if tag == "si_fix":
            f, info = transfer.nb_credibility(tr, ftr, s_tr, cap)
            res["nb_factor_from_2023"] = info
            P["si_fix_nb"] = norm(np.where(nb, base * f, base))
        del prior; gc.collect()
    if line == "motor":
        for k in ["old", "si_fix", "si_fix_nb"]:
            P[k + "+freqsev@0.25"] = 0.75 * P[k] + 0.25 * fs
    res["scores"] = {k: {m: round(v, 5) for m, v in scores(y, p, nb, e).items()} for k, p in P.items()}
    # paired bootstrap by policy: deviance gain of si_fix over emerging-only
    t = y.sum(); gain = dev15(y, P["emerging_only_freqsev"] * t) - dev15(y, P["si_fix"] * t)
    ids = pd.factorize(te.policy_id)[0]; g = np.bincount(ids, weights=gain)
    denom = dev15(y, np.full(len(y), y.mean())).sum(); rng = np.random.default_rng(918)
    boots = [np.dot(rng.poisson(1, len(g)), g) / denom for _ in range(300)]
    res["si_fix_minus_emerging_only_d2_95pct"] = [round(float(q), 4) for q in np.quantile(boots, [.025, .975])]
    res["si_repaired_rows"] = {"dev": int(dev.get("si_unit_repaired", pd.Series([0])).sum()), "em_2023": int(tr.get("si_unit_repaired", pd.Series([0])).sum()), "em_2024": int(te.get("si_unit_repaired", pd.Series([0])).sum())}
    out[line] = res
    print(line, json.dumps(res), flush=True)
    del dev, em, tr, te; gc.collect()

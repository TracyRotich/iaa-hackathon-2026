"""Holdout tests of further modelling ideas, against the submission-3 set-up.
Usage: deep_tests.py <line> <test>
tests:
  base        submission-3 configuration (reference)
  cap         large-loss cap on the training target for the relativity models
  excess      total excess as an explicit feature (claims are net of excess)
  season      share of the row's days in winter / summer (seasonality of weather)
  factors     drop each 'review' factor in turn
  tweedie     Tweedie GLM benchmark (power 1.5) in place of Poisson x Gamma
  knots6      more flexible splines in the GLM (vary df)
Scores: D² after rescaling to the actual total (level-neutral), 2023 and 2024 holdouts.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import TweedieRegressor
from iaa import features, metrics, gbm, glm as glm_mod
from iaa.config import REVIEW
from iaa.glm import FreqSevGLM, Design, CredibilityEncoder

line, test = sys.argv[1], sys.argv[2]
W = {"motor": 0.3, "buildings": 0.7, "contents": 0.5}[line]
CAPS = {"motor": [100_000, 250_000], "buildings": [100_000, 250_000], "contents": [25_000, 50_000]}
df = pd.read_parquet(f"data/work/{line}_train.parquet")
drop = ["occupation"] if line == "buildings" else []
FEATS = [f for f in features.feature_list(df) if f not in drop]


def si_growth(tr):
    p = tr.pivot_table(index="policy_id", columns="policy_year", values="sum_insured", aggfunc="first")
    yrs = sorted(p.columns)
    return float(np.nanmedian(np.concatenate([(p[y] / p[x]).dropna().to_numpy() for x, y in zip(yrs, yrs[1:])])))


def season_share(f):
    s = f.period_start.to_numpy("datetime64[D]"); e = f.period_end.to_numpy("datetime64[D]")
    y = f.policy_year.to_numpy()
    def overlap(a, b):
        lo = np.maximum(s, a); hi = np.minimum(e, b)
        return np.clip((hi - lo).astype(int) + 1, 0, None)
    yy = y.astype(str)
    D = lambda m, d: np.array([np.datetime64(f"{v}-{m}-{d}") for v in yy])
    days = (e - s).astype(int) + 1
    winter = overlap(D("01", "01"), D("02", "28")) + overlap(D("12", "01"), D("12", "31"))
    summer = overlap(D("06", "01"), D("08", "31"))
    return winter / days, summer / days


def prep(tr, te):
    if "sum_insured" in tr.columns:
        g = si_growth(tr)
        tr = tr.assign(sum_insured=tr.sum_insured / g ** (tr.policy_year - 2021))
        te = te.assign(sum_insured=te.sum_insured / g ** (te.policy_year - 2021))
    return tr, te


def tweedie_glm_predict(tr, te, feats):
    d = Design().fit(tr[feats])
    y = tr.claim_cost.to_numpy(); e = tr.exposure.to_numpy()
    tv_tr, tv_te = {}, {}
    for c in d.te:
        enc = CredibilityEncoder().fit(tr[c], y, e, process_var=np.var(y / e, ddof=1) * np.mean(e))
        tv_tr[c] = np.clip(enc.transform_train(tr[c], y, e), 1e-6, None)
        tv_te[c] = np.clip(enc.transform(te[c]), 1e-6, None)
    m = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=500).fit(d.transform(tr[feats], tv_tr), y / e, sample_weight=e)
    return m.predict(d.transform(te[feats], tv_te)) * te.exposure.to_numpy()


def run(tr, te, feats, train_cost=None, glm_kind="freqsev"):
    """Fit both relativity models and the blend; returns rescaled D² for gbm, glm, blend."""
    tr_fit = tr if train_cost is None else tr.assign(claim_cost=train_cost[0], reported_cost=train_cost[1])
    pg = gbm.fit_predict(line, tr_fit, te, feats, te_make_model=(line == "motor"))
    if glm_kind == "tweedie":
        pl = tweedie_glm_predict(tr_fit, te, feats)
    else:
        pl = FreqSevGLM(backfit=True).fit(tr_fit[feats], tr_fit.n_claims / tr_fit.completeness,
                                          tr_fit.reported_cost, tr_fit.exposure).predict(te[feats], te.exposure)
    y, e = te.claim_cost.to_numpy(), te.exposure.to_numpy()
    bl = (W * pg / pg.sum() + (1 - W) * pl / pl.sum()) * y.sum()
    return {k: round(metrics.scorecard(y, p, e)["d2_rescaled"], 4) for k, p in (("gbm", pg), ("glm", pl), ("blend", bl))}


out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)].copy(); te = df[df.policy_year == test_year].copy()
    tr, te = prep(tr, te)
    r = {"base": run(tr, te, FEATS)} if test != "factors" else {}
    if test == "cap":
        for cap in CAPS[line]:
            rep = tr.reported_cost.clip(upper=cap)
            r[f"cap_{cap // 1000}k"] = run(tr, te, FEATS, train_cost=(rep / tr.completeness, rep))
            r[f"cap_{cap // 1000}k_rows_affected"] = int((tr.reported_cost > cap).sum())
    elif test == "excess":
        for f in (tr, te):
            f["total_excess"] = f.voluntary_excess + f.compulsory_excess
        r["total_excess"] = run(tr, te, FEATS + ["total_excess"])
    elif test == "season":
        for f in (tr, te):
            f["winter_share"], f["summer_share"] = season_share(f)
        r["season"] = run(tr, te, FEATS + ["winter_share", "summer_share"])
    elif test == "factors":
        r["base"] = run(tr, te, FEATS)
        for fac in [f for f in REVIEW if f in FEATS]:
            r[f"drop_{fac}"] = run(tr, te, [f for f in FEATS if f != fac])
    elif test == "tweedie":
        r["tweedie_glm"] = run(tr, te, FEATS, glm_kind="tweedie")
    elif test == "knots6":
        glm_mod.N_KNOTS = 6
        r["knots6"] = run(tr, te, FEATS)
        glm_mod.N_KNOTS = 4
    out[test_year] = r
print(line, test, json.dumps(out))

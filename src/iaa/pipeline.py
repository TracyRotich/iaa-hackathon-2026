"""Relativity pipeline for submissions 4 and 5, shared by validation and the final run.

v4 (Claude): per line, blend of LightGBM and the backfitted Poisson x Gamma GLM; sums insured
on-levelled; motor fitted on row cost capped at 25k.

v5 (fixes from the Codex review, 23 Oct, ported here so they live in Tracy Rotich's own pipeline):
  R2  weather averages use only years before the scored year; prior-year weather dropped
  R3  sqrt(exposure) weights for Tweedie rate models
  R4  GLM gets cost and counts on the same (ultimate, IBNR-adjusted) basis
plus one challenger per line, blended with v4. Choices fixed by the review on the 2023
holdout and confirmed on 2024 (see docs/decision_log.md, decisions 34-36).

All outputs are relativities normalised to sum to 1 within the scored rows; the level
(total cost) comes from level.py.
"""
import numpy as np

from . import features, gbm
from .glm import FreqSevGLM

MOTOR_CAP = 25_000
V4 = {  # weight on GBM; rest on GLM
    "motor": {"w_gbm": 0.3, "te_make_model": True, "drop": []},
    "buildings": {"w_gbm": 0.7, "te_make_model": False, "drop": ["occupation"]},
    "contents": {"w_gbm": 0.5, "te_make_model": False, "drop": []},
}
V5 = {"motor": ("clean_glm", 0.75), "buildings": ("shallow", 0.5), "contents": ("freqsev", 0.25)}


def norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def onlevel_sum_insured(tr, te):
    """Deflate sums insured to first-year money at the observed same-policy growth."""
    if "sum_insured" not in tr:
        return tr, te, None
    p = tr.pivot_table(index="policy_id", columns="policy_year", values="sum_insured", aggfunc="first")
    yrs = sorted(p.columns)
    if len(yrs) < 2:  # one training year (emerging holdout): nothing to on-level
        return tr, te, None
    g = float(np.nanmedian(np.concatenate([(p[b] / p[a]).dropna().to_numpy() for a, b in zip(yrs, yrs[1:])])))
    y0 = min(yrs)
    tr = tr.assign(sum_insured=tr.sum_insured / g ** (tr.policy_year - y0))
    te = te.assign(sum_insured=te.sum_insured / g ** (te.policy_year - y0))
    return tr, te, g


def feature_names(tr, line):
    return [f for f in features.feature_list(tr) if f not in V4[line]["drop"]]


def capped(tr, line):
    """Training frame for relativities: motor row cost capped at 25k (basic limit), on both bases."""
    if line != "motor":
        return tr
    rep = tr.reported_cost.clip(upper=MOTOR_CAP)
    return tr.assign(reported_cost=rep, claim_cost=rep / tr.completeness)


def v4_relativities(line, tr, te):
    """Submission 4 as built (old weather features, GLM on reported cost)."""
    tr, te, _ = onlevel_sum_insured(tr, te)
    f = feature_names(tr, line)
    fit = capped(tr, line)
    pg = gbm.fit_predict(line, fit, te, f, V4[line]["te_make_model"])
    g = FreqSevGLM(backfit=True).fit(fit[f], fit.n_claims / fit.completeness, fit.reported_cost, fit.exposure)
    pl = g.predict(te[f], te.exposure)
    w = V4[line]["w_gbm"]
    return {"base": w * norm(pg) + (1 - w) * norm(pl), "gbm": norm(pg), "glm": norm(pl)}


def challenger(line, kind, tr, te, weather, cutoff):
    """One v5 challenger with fixes R2-R4 applied."""
    tr, te, _ = onlevel_sum_insured(tr, te)
    tr, te, keep = features.time_safe_weather(tr, te, weather, cutoff)
    f = keep(feature_names(tr, line))
    fit = capped(tr, line)
    if kind == "clean_glm":  # R4: ultimate cost with ultimate counts
        g = FreqSevGLM(backfit=True).fit(fit[f], fit.n_claims / fit.completeness, fit.claim_cost, fit.exposure)
        return norm(g.predict(te[f], te.exposure))
    return norm(gbm.fit_predict_variant(fit, te, f, kind, fit.claim_cost))


def v5_relativities(line, tr, te, weather, cutoff, base=None):
    """Returns dict with v4 base, the line's challenger and the v5 blend."""
    out = dict(base) if base is not None else v4_relativities(line, tr, te)
    kind, w = V5[line]
    out[kind] = challenger(line, kind, tr, te, weather, cutoff)
    out["v5"] = (1 - w) * out["base"] + w * out[kind]
    return out

"""Part 2 submission (emerging market).

Per line:
  relativity q(x) = developed prior (freq x sev LightGBM on all developed years, harmonised
                    features; motor severity on row cost capped at 25k), re-levelled to emerging;
                    motor blended 25% with an emerging-only freq x sev GBM (2024 holdout)
  [--si-fix]       sums insured repaired (1,000x unit error) and put in 2023 money in both markets,
                    so 2025 indexation enters once, through the severity trend in the level
  [--nb-cred]      prior multiplied on new-business rows by a credibility-weighted emerging
                    new-business factor (classical credibility, 1,082 claims for full credibility)
  level           = 2025 cost rate at the 2023-24 emerging mix (scripts/level_emerging.py)
  prediction_i    = p_i x level / mu_train  (transfer.calibrate_to_level; mix enters once)
Usage: run_part2.py <out_csv> [--si-fix] [--nb-cred]"""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import transfer, gbm, guards, submission, features
from iaa.pipeline import onlevel_sum_insured

W = "data/emerging/work"
SI_FIX, NB_CRED = "--si-fix" in sys.argv, "--nb-cred" in sys.argv
CHOICE = {"motor": {"cap": 25_000, "w_emerging": 0.25},
          "buildings": {"cap": None, "w_emerging": 0.0},
          "contents": {"cap": None, "w_emerging": 0.0}}
level = pd.read_csv(f"{W}/level_lines.csv").set_index("line")["cost_rate_2025"]
parts, report = {}, {}
for line, cfg in CHOICE.items():
    dev = transfer.harmonise(pd.read_parquet(f"data/work/{line}_train.parquet"), line)
    tr = transfer.harmonise(pd.read_parquet(f"{W}/{line}_train.parquet"), line)
    te = transfer.harmonise(pd.read_parquet(f"{W}/{line}_test.parquet"), line)
    rep = {}
    if not SI_FIX and "sum_insured_raw" in dev:   # reproduce submissions 1-2 exactly: raw values
        dev, tr, te = (d.assign(sum_insured=d.sum_insured_raw) for d in (dev, tr, te))
    if SI_FIX and "sum_insured" in dev:
        g_dev, g_em = transfer.si_growth(dev), transfer.si_growth(tr)
        dev, tr, te = transfer.onlevel_si(dev, g_dev), transfer.onlevel_si(tr, g_em), transfer.onlevel_si(te, g_em)
        rep.update(si_growth_dev=round(g_dev, 4), si_growth_em=round(g_em, 4),
                   si_repaired_train=int(tr.si_unit_repaired.sum()), si_repaired_test=int(te.si_unit_repaired.sum()))
    feats = transfer.transfer_features(dev, tr)
    prior = transfer.DevPrior(cap=cfg["cap"]).fit(dev, feats)
    del dev; gc.collect()
    pr_tr, pr_te = prior.predict(tr), prior.predict(te)
    p_tr = transfer.boost_from_prior(tr, tr, feats, pr_tr, pr_tr, 0, 0, cap=cfg["cap"])
    p_te = transfer.boost_from_prior(tr, te, feats, pr_tr, pr_te, 0, 0, cap=cfg["cap"])
    f, info = transfer.nb_credibility(tr, pr_tr[0], pr_tr[1], cfg["cap"])
    rep["nb_credibility" + ("" if NB_CRED else "_reported_not_applied")] = info
    if NB_CRED:
        p_tr = np.where(tr.new_business.astype(bool), p_tr * f, p_tr)
        p_te = np.where(te.new_business.astype(bool), p_te * f, p_te)
    m_tr, m_te = p_tr / p_tr.sum(), p_te / p_tr.sum()   # same scale: train total = 1
    if cfg["w_emerging"]:
        a, b, _ = onlevel_sum_insured(tr, te)
        fe = [c for c in features.feature_list(a) if not c.startswith("wx_prev_")]
        e_tr = gbm.fit_predict_variant(a, a, fe, "freqsev", a.claim_cost)
        e_te = gbm.fit_predict_variant(a, b, fe, "freqsev", a.claim_cost)
        w = cfg["w_emerging"]
        m_tr, m_te = (1 - w) * m_tr + w * e_tr / e_tr.sum(), (1 - w) * m_te + w * e_te / e_tr.sum()
    pred, mix = transfer.calibrate_to_level(m_tr, tr.exposure, m_te, te.exposure, level[line])
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te.row_id.astype(str), "prediction": pred})
    nb = te.new_business.astype(bool).to_numpy()
    report[line] = {"features": len(feats), "level_2025_at_train_mix": round(float(level[line]), 2), "mix_factor": round(float(mix), 4),
                    "rate_2025": round(float(pred.sum() / te.exposure.sum()), 2), "total": round(float(pred.sum())),
                    "nb_vs_renewal_rate": round(float((pred[nb].sum() / te.exposure[nb].sum()) / (pred[~nb].sum() / te.exposure[~nb].sum())), 3), **rep}
    print(line, json.dumps(report[line]), flush=True)
    del tr, te, prior; gc.collect()
sub = submission.build(parts)
ids = pd.read_parquet("data/emerging/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub.set_index("row_id").loc[ids].reset_index().to_csv(sys.argv[1], index=False)
print(json.dumps(report, indent=1))

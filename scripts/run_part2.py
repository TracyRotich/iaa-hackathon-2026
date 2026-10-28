"""Part 2 submission (emerging market).

Per line:
  relativity m(x) = developed prior (freq x sev LightGBM on all developed years, harmonised
                    features; motor severity on row cost capped at 25k), optionally blended with
                    an emerging-only freq x sev GBM (motor 25%: chosen on the 2024 holdout)
  level           = 2025 cost rate at the 2023-24 emerging mix (scripts/level_emerging.py)
  mix factor      = mean m per exposure on the 2025 rows / mean m per exposure on 2023-24 rows,
                    which carries the shift to 76% new business and younger drivers into the total
  prediction      = m(x) * level * mix factor * sum(exposure) / sum(m), i.e. rescaled to the total
Usage: run_part2.py <out_csv>"""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import transfer, gbm, guards, submission, features
from iaa.pipeline import onlevel_sum_insured

W = "data/emerging/work"
CHOICE = {"motor": {"cap": 25_000, "w_emerging": 0.25},
          "buildings": {"cap": None, "w_emerging": 0.0},
          "contents": {"cap": None, "w_emerging": 0.0}}
level = pd.read_csv(f"{W}/level_lines.csv").set_index("line")["cost_rate_2025"]
parts, report = {}, {}
for line, cfg in CHOICE.items():
    dev = transfer.harmonise(pd.read_parquet(f"data/work/{line}_train.parquet"), line)
    tr = transfer.harmonise(pd.read_parquet(f"{W}/{line}_train.parquet"), line)
    te = transfer.harmonise(pd.read_parquet(f"{W}/{line}_test.parquet"), line)
    feats = transfer.transfer_features(dev, tr)
    prior = transfer.DevPrior(cap=cfg["cap"]).fit(dev, feats)
    del dev; gc.collect()
    p_tr = transfer.boost_from_prior(tr, tr, feats, prior.predict(tr), prior.predict(tr), 0, 0, cap=cfg["cap"])
    p_te = transfer.boost_from_prior(tr, te, feats, prior.predict(tr), prior.predict(te), 0, 0, cap=cfg["cap"])
    m_tr, m_te = p_tr / p_tr.sum(), p_te / p_tr.sum()   # same scale: train total = 1
    if cfg["w_emerging"]:
        a, b, _ = onlevel_sum_insured(tr, te)
        f = [c for c in features.feature_list(a) if not c.startswith("wx_prev_")]
        e_tr = gbm.fit_predict_variant(a, a, f, "freqsev", a.claim_cost)
        e_te = gbm.fit_predict_variant(a, b, f, "freqsev", a.claim_cost)
        w = cfg["w_emerging"]
        m_tr, m_te = (1 - w) * m_tr + w * e_tr / e_tr.sum(), (1 - w) * m_te + w * e_te / e_tr.sum()
    rate_tr = m_tr.sum() / tr.exposure.sum(); rate_te = m_te.sum() / te.exposure.sum()
    mix = rate_te / rate_tr
    total = level[line] * mix * te.exposure.sum()
    pred = m_te / m_te.sum() * total
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te.row_id.astype(str), "prediction": pred})
    nb = te.new_business.astype(bool).to_numpy()
    report[line] = {"features": len(feats), "level_2025_at_train_mix": round(float(level[line]), 2), "mix_factor": round(float(mix), 4),
                    "rate_2025": round(float(total / te.exposure.sum()), 2), "total": round(float(total)),
                    "nb_vs_renewal_rate": round(float((pred[nb].sum() / te.exposure[nb].sum()) / (pred[~nb].sum() / te.exposure[~nb].sum())), 3)}
    print(line, json.dumps(report[line]), flush=True)
    del tr, te, prior; gc.collect()
sub = submission.build(parts)
ids = pd.read_parquet("data/emerging/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub.set_index("row_id").loc[ids].reset_index().to_csv(sys.argv[1], index=False)
print(json.dumps(report, indent=1))

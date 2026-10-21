"""Submission 2: per line, blend of LightGBM and a Poisson x Gamma GLM, rescaled to the
2025 level. Blend weights chosen on the 2023 holdout and confirmed on 2024.
Usage: run_part1_v2.py <out_csv> [--gi-methods]
--gi-methods (submission 3) adds two methods that improved both
holdouts: sum insured on-levelled to 2021 money (s.10.3.2) and backfitted credibility smoothing of
high-cardinality factors in the GLM (s.28.1.2)."""
import sys, json
import numpy as np, pandas as pd
from iaa import features, submission, guards, gbm
from iaa.glm import FreqSevGLM

GI_METHODS = "--gi-methods" in sys.argv


def si_growth(tr):
    p = tr.pivot_table(index="policy_id", columns="policy_year", values="sum_insured", aggfunc="first")
    yrs = sorted(p.columns)
    return float(np.nanmedian(np.concatenate([(p[y] / p[x]).dropna().to_numpy() for x, y in zip(yrs, yrs[1:])])))

CONFIG = {  # weight on GBM; rest on GLM
    "motor": {"w_gbm": 0.3, "te_make_model": True, "drop": []},
    "buildings": {"w_gbm": 0.7, "te_make_model": False, "drop": ["occupation"]},
    "contents": {"w_gbm": 0.5, "te_make_model": False, "drop": []},
}
level = pd.read_csv("data/work/level_lines.csv").set_index("line")["cost_rate_2025"]
parts, report = {}, {}
for line, cfg in CONFIG.items():
    tr = pd.read_parquet(f"data/work/{line}_train.parquet"); te = pd.read_parquet(f"data/work/{line}_test.parquet")
    feats = [f for f in features.feature_list(tr) if f not in cfg["drop"]]
    assert not guards.leakage_suspects(feats)
    if GI_METHODS and "sum_insured" in feats:
        g = si_growth(tr)
        tr = tr.assign(sum_insured=tr.sum_insured / g ** (tr.policy_year - 2021))
        te = te.assign(sum_insured=te.sum_insured / g ** (te.policy_year - 2021))
        report_si = round(g, 4)
    else:
        report_si = None
    p_gbm = gbm.fit_predict(line, tr, te, feats, cfg["te_make_model"])
    g = FreqSevGLM(backfit=GI_METHODS).fit(tr[feats], tr.n_claims / tr.completeness, tr.reported_cost, tr.exposure)
    p_glm = g.predict(te[feats], te.exposure)
    mix = cfg["w_gbm"] * p_gbm / p_gbm.sum() + (1 - cfg["w_gbm"]) * p_glm / p_glm.sum()
    pred = mix * level[line] * te.exposure.sum()
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te.row_id, "prediction": pred})
    report[line] = {"level_rate": round(level[line], 2), "pred_total": round(pred.sum()),
                    "corr_gbm_glm": round(float(np.corrcoef(np.log(p_gbm), np.log(p_glm))[0, 1]), 3),
                    "kappas": g.kappas(), "si_growth": report_si}
sub = submission.build(parts)
ids = pd.read_parquet("data/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub.set_index("row_id").loc[ids].reset_index().to_csv(sys.argv[1], index=False)
print(json.dumps(report, indent=1))

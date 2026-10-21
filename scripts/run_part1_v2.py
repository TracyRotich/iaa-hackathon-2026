"""Submission 2: per line, blend of LightGBM and a Poisson x Gamma GLM, rescaled to the
2025 level. Blend weights chosen on the 2023 holdout and confirmed on 2024. Usage: run_part1_v2.py <out_csv>"""
import sys, json
import numpy as np, pandas as pd
from iaa import features, submission, guards, gbm
from iaa.glm import FreqSevGLM

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
    p_gbm = gbm.fit_predict(line, tr, te, feats, cfg["te_make_model"])
    g = FreqSevGLM().fit(tr[feats], tr.n_claims / tr.completeness, tr.reported_cost, tr.exposure)
    p_glm = g.predict(te[feats], te.exposure)
    mix = cfg["w_gbm"] * p_gbm / p_gbm.sum() + (1 - cfg["w_gbm"]) * p_glm / p_glm.sum()
    pred = mix * level[line] * te.exposure.sum()
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te.row_id, "prediction": pred})
    report[line] = {"level_rate": round(level[line], 2), "pred_total": round(pred.sum()),
                    "corr_gbm_glm": round(float(np.corrcoef(np.log(p_gbm), np.log(p_glm))[0, 1]), 3),
                    "kappas": g.kappas()}
sub = submission.build(parts)
ids = pd.read_parquet("data/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub.set_index("row_id").loc[ids].reset_index().to_csv(sys.argv[1], index=False)
print(json.dumps(report, indent=1))

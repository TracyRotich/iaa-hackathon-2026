"""Submission 5 from Tracy Rotich's own pipeline: v4 relativities blended with the review challengers
(fixes R2-R4), rescaled to the same 2025 level as v1-v4.
Usage: run_part1_v5.py <out_csv> [--compare path/to/codex_challenger.csv]"""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import load, pipeline, submission, guards
from iaa import config
WORK = config.WORK_DIR

out_csv = sys.argv[1]
level = pd.read_csv(f"{WORK}/level_lines.csv").set_index("line")["cost_rate_2025"]
weather = load.read("weather")
parts, report = {}, {}
for line in pipeline.V5:
    tr = pd.read_parquet(f"{WORK}/{line}_train.parquet"); te = pd.read_parquet(f"{WORK}/{line}_test.parquet")
    r = pipeline.v5_relativities(line, tr, te, weather, cutoff=2025)
    total = level[line] * te.exposure.sum()
    pred = r["v5"] * total
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te.row_id, "prediction": pred})
    kind, w = pipeline.V5[line]
    report[line] = {"challenger": kind, "weight": w, "total": round(float(pred.sum())),
                    "corr_log_base_challenger": round(float(np.corrcoef(np.log(r["base"]), np.log(r[kind]))[0, 1]), 4)}
    del tr, te, r; gc.collect()
sub = submission.build(parts)
ids = pd.read_parquet(f"{config.DATA_DIR}/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub = sub.set_index("row_id").loc[ids].reset_index()
sub.to_csv(out_csv, index=False)
if "--compare" in sys.argv:
    ref = pd.read_csv(sys.argv[sys.argv.index("--compare") + 1], dtype={"row_id": str}).set_index("row_id").prediction.loc[ids]
    rel = sub.prediction.to_numpy() / ref.to_numpy() - 1
    report["vs_reference"] = {"max_abs_rel_diff": float(np.abs(rel).max()), "median_abs_rel_diff": float(np.median(np.abs(rel))),
                              "corr": float(np.corrcoef(sub.prediction, ref)[0, 1]), "total_diff": float(sub.prediction.sum() - ref.sum())}
print(json.dumps(report, indent=1))

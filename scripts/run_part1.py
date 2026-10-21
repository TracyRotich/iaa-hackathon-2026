"""Final Part 1 model: LightGBM Tweedie per line on 2021-2024, rescaled to the 2025 level.
Usage: python scripts/run_part1.py <out_csv>"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from iaa import features, submission, guards
from iaa.config import SEED, TWEEDIE_POWER

TREES = {"motor": 130, "buildings": 160, "contents": 140}  # ~1.15 x mean early-stopping rounds on 2023/2024 holdouts
PARAMS = dict(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, learning_rate=0.03,
              num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, cat_smooth=50, min_data_per_group=200,
              max_cat_to_onehot=8, verbose=-1, random_state=SEED)
level = pd.read_csv("data/work/level_lines.csv").set_index("line")["cost_rate_2025"]
parts, report = {}, {}
for line in ("motor", "buildings", "contents"):
    tr = pd.read_parquet(f"data/work/{line}_train.parquet")
    te = pd.read_parquet(f"data/work/{line}_test.parquet")
    feats = features.feature_list(tr)
    assert not guards.leakage_suspects(feats), guards.leakage_suspects(feats)
    cats = {c: sorted(tr[c].dropna().astype(str).unique()) for c in feats if not pd.api.types.is_numeric_dtype(tr[c])}
    def prep(X):
        X = X[feats].copy()
        for c, lv in cats.items():
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=lv)
        return X
    m = lgb.LGBMRegressor(n_estimators=TREES[line], **PARAMS).fit(
        prep(tr), tr["claim_cost"] / tr["exposure"], sample_weight=tr["exposure"])
    raw = m.predict(prep(te)) * te["exposure"].to_numpy()
    target_total = level[line] * te["exposure"].sum()
    pred = raw * target_total / raw.sum()
    guards.check_predictions(pred)
    parts[line] = pd.DataFrame({"row_id": te["row_id"], "prediction": pred})
    report[line] = {"rows": len(te), "features": len(feats), "test_exposure": round(te["exposure"].sum(), 1),
                    "level_rate": round(level[line], 2), "scale_factor": round(target_total / raw.sum(), 4),
                    "pred_total": round(pred.sum()), "pred_mean": round(pred.mean(), 2),
                    "pred_p99": round(np.quantile(pred, .99), 1)}
sub = submission.build(parts)
ids = pd.read_parquet("data/sample_submission.parquet")["row_id"].astype(str)
submission.validate(sub, ids)
sub = sub.set_index("row_id").loc[ids].reset_index()  # same order as sample_submission
sub.to_csv(sys.argv[1], index=False)
print(json.dumps(report, indent=1)); print("written", sys.argv[1], sub.shape)

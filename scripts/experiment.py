"""Holdout experiments per line. Usage: python scripts/experiment.py <line> [variant]"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from iaa import features, metrics
from iaa.config import SEED, TWEEDIE_POWER

line = sys.argv[1]; variant = sys.argv[2] if len(sys.argv) > 2 else "base"
df = pd.read_parquet(f"data/work/{line}_train.parquet")
extra = {"base": [], "same_year_wx": [], "with_premium": [], "init": []}[variant]
feats = features.feature_list(df)
if variant == "same_year_wx":
    feats += [c for c in df.columns if c.startswith("wx_same_")]
if variant == "with_premium":
    feats += ["written_premium", "discount_applied"]

def prep(X, cats=None):
    X = X.copy()
    if cats is None:
        cats = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])}
    for c, lv in cats.items():
        X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=lv)
    return X, cats

PARAMS = dict(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, learning_rate=0.03,
              num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, cat_smooth=50, min_data_per_group=200,
              max_cat_to_onehot=8, verbose=-1, random_state=SEED, n_estimators=3000)

res = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    tr = df[df.policy_year.isin(train_years)]; te = df[df.policy_year == test_year]
    # early-stopping set: 10% of policies inside the training years
    pol = tr.policy_id.unique(); rng = np.random.default_rng(SEED)
    es_ids = set(rng.choice(pol, size=len(pol) // 10, replace=False))
    es = tr.policy_id.isin(es_ids)
    Xtr, cats = prep(tr.loc[~es, feats]); Xes, _ = prep(tr.loc[es, feats], cats); Xte, _ = prep(te[feats], cats)
    ytr = tr.loc[~es, "claim_cost"] / tr.loc[~es, "exposure"]; wtr = tr.loc[~es, "exposure"]
    yes = tr.loc[es, "claim_cost"] / tr.loc[es, "exposure"]; wes = tr.loc[es, "exposure"]
    t0 = time.time()
    init = np.log(np.average(ytr, weights=wtr)) if variant == "init" else None
    kw = dict(init_score=np.full(len(ytr), init), eval_init_score=[np.full(len(yes), init)]) if init is not None else {}
    m = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr, sample_weight=wtr, eval_X=(Xes,), eval_y=(yes,),
        eval_sample_weight=[wes], callbacks=[lgb.early_stopping(100, verbose=False)], **kw)
    p = np.exp(m.predict(Xte, raw_score=True) + (init or 0)) * te.exposure.values
    y = te.claim_cost.values
    scaled = p * y.sum() / p.sum()
    res[test_year] = {"trees": int(m.best_iteration_), "secs": round(time.time() - t0),
                      "d2_raw": round(metrics.d2(y, p), 5), "d2_rescaled": round(metrics.d2(y, scaled), 5),
                      "pred/actual": round(p.sum() / y.sum(), 4)}
    if test_year == 2023:
        imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
        res["top_features"] = (imp / imp.sum()).head(15).round(4).to_dict()
        lt = metrics.lift_table(y, scaled); res["lift_2023_top_vs_bottom"] = round(lt.actual.iloc[-1] / max(lt.actual.iloc[0], 1e-9), 1)
print(line, variant, json.dumps(res, indent=1))

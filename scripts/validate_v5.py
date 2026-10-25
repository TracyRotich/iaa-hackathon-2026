"""Holdout check of the v5 pipeline: train on years before Y, score year Y (2023 = selection,
2024 = confirmation). Holdout scores rescale predictions to the actual holdout total, so they
measure allocation (relativities) only, not the level.
Usage: validate_v5.py 2023 2024 [--decompose]
--decompose also scores motor's GLM with only one fix at a time (R2 weather, R4 IBNR), to show
where the v5 gain comes from."""
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd
from iaa import load, metrics, pipeline, features
from iaa.glm import FreqSevGLM
from iaa import config
WORK = config.WORK_DIR

OUT = Path(f"{WORK}/results")
weather = load.read("weather")
years = [int(a) for a in sys.argv[1:] if a.isdigit()]
report = {}
for year in years:
    for line in pipeline.V5:
        df = pd.read_parquet(f"{WORK}/{line}_train.parquet")
        tr, te = df[df.policy_year < year], df[df.policy_year == year]
        del df; gc.collect()
        r = pipeline.v5_relativities(line, tr, te, weather, cutoff=year)
        y = te.claim_cost.to_numpy()
        if "--decompose" in sys.argv and line == "motor":
            a, b, _ = pipeline.onlevel_sum_insured(tr, te)
            f = pipeline.feature_names(a, line); fit = pipeline.capped(a, line)
            g = FreqSevGLM(backfit=True).fit(fit[f], fit.n_claims / fit.completeness, fit.claim_cost, fit.exposure)
            r["glm_R4_only"] = pipeline.norm(g.predict(b[f], b.exposure))
            a2, b2, keep = features.time_safe_weather(a, b, weather, year); f2 = keep(f); fit2 = pipeline.capped(a2, line)
            g = FreqSevGLM(backfit=True).fit(fit2[f2], fit2.n_claims / fit2.completeness, fit2.reported_cost, fit2.exposure)
            r["glm_R2_only"] = pipeline.norm(g.predict(b2[f2], b2.exposure))
        scores = {k: metrics.d2(y, v * y.sum()) for k, v in r.items()}
        report[f"{line}_{year}"] = scores
        np.savez_compressed(OUT / f"v5_{line}_{year}.npz", y=y, exposure=te.exposure.to_numpy(),
                            policy_id=te.policy_id.to_numpy().astype(str), **r)
        print(line, year, json.dumps({k: round(v, 5) for k, v in scores.items()}), flush=True)
        del tr, te, r; gc.collect()
(OUT / f"v5_scores_{'_'.join(map(str, years))}.json").write_text(json.dumps(report, indent=1))

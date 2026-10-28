"""Transfer models on the emerging 2024 holdout (train emerging 2023; developed prior fitted on all
developed years). Saves predictions next to validate_part2.py's for blending.
Usage: validate_transfer.py [line ...]"""
import sys, json, gc
import numpy as np, pandas as pd
from iaa import metrics, transfer
from iaa.pipeline import norm

W = "data/emerging/work"
for line in sys.argv[1:] or ["motor", "buildings", "contents"]:
    dev = transfer.harmonise(pd.read_parquet(f"data/work/{line}_train.parquet"), line)
    em = transfer.harmonise(pd.read_parquet(f"{W}/{line}_train.parquet"), line)
    feats = transfer.transfer_features(dev, em)
    caps = [None, 25_000] if line == "motor" else [None]
    old = dict(np.load(f"{W}/val_{line}_2024.npz"))
    tr, te = em[em.policy_year == 2023], em[em.policy_year == 2024]
    y = te.claim_cost.to_numpy(); P = {}
    for cap in caps:
        prior = transfer.DevPrior(cap=cap).fit(dev, feats)
        ptr, pte = prior.predict(tr), prior.predict(te)
        tag = "" if cap is None else "_cap25k"
        P["prior" + tag] = norm(transfer.boost_from_prior(tr, te, feats, ptr, pte, 0, 0, cap=cap))
        for nf, ns in [(100, 40), (250, 100), (500, 200)]:
            P[f"boost{nf}{tag}"] = norm(transfer.boost_from_prior(tr, te, feats, ptr, pte, nf, ns, cap=cap))
        del prior; gc.collect()
    for k in list(P):
        for w in (0.25, 0.5):
            P[f"{k}+freqsev@{w}"] = (1 - w) * P[k] + w * old["freqsev"]
    s = {k: round(metrics.d2(y, v * y.sum()), 5) for k, v in P.items()}
    s["freqsev(emerging only)"] = round(metrics.d2(y, old["freqsev"] * y.sum()), 5)
    np.savez_compressed(f"{W}/valT_{line}_2024.npz", **P)
    print(line, len(feats), "features", json.dumps(dict(sorted(s.items(), key=lambda kv: -kv[1]))), flush=True)
    del dev, em, tr, te; gc.collect()

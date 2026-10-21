"""Build cleaned feature frames per line (train with target, test) into data/work/."""
import json
from pathlib import Path

import pandas as pd

from iaa import features, load, target
from iaa.level import LARGE_CAP

out = Path("data/work"); out.mkdir(parents=True, exist_ok=True)
claims = load.read("claims")
customers = load.read("customers")
addresses = load.read("addresses")
weather = load.read("weather")
wx_hist = features.weather_features(weather)
wx_prev = features.prior_year_weather(weather)
diag_all = {"claims_checks": {k: int(v) for k, v in target.claims_data_checks(claims).items()}}

for line in ("motor", "buildings", "contents"):
    pol = load.read(f"policies_{line}")
    tst = load.read(f"test_{line}")
    rows, diag = target.build_target(pol, claims[claims["line"] == line], cap=LARGE_CAP)
    diag["exposure_rows_repaired"] = pol.attrs["exposure_rows_repaired"]
    diag_all[line] = diag
    tr = features.build(pol, customers, addresses, wx_hist, wx_prev, same_year_weather=weather)
    tr = tr.merge(rows[["row_id", "reported_cost", "n_claims", "completeness", "claim_cost"]], on="row_id")
    te = features.build(tst, customers, addresses, wx_hist, wx_prev)
    assert len(tr) == len(pol) and len(te) == len(tst), "join changed row counts"
    tr.to_parquet(out / f"{line}_train.parquet"); te.to_parquet(out / f"{line}_test.parquet")
    print(line, tr.shape, te.shape, diag)

Path("data/work/diagnostics.json").write_text(json.dumps(diag_all, indent=2, default=float))

"""Feature frames for the emerging market into data/emerging/work/.
Differences from prepare.py (developed): reporting-delay pattern borrowed from the developed market
by line; weather averages from years before 2025 only; text categories normalised."""
import json
from pathlib import Path
import pandas as pd
from iaa import features, load, target

DEV, EM = "data", "data/emerging"
out = Path(EM, "work"); out.mkdir(parents=True, exist_ok=True)
dev_claims, _ = target.drop_duplicate_claims(load.read("claims", DEV))
claims = load.read("claims", EM); customers = load.read("customers", EM)
addresses = load.read("addresses", EM); weather = load.read("weather", EM)
wx_hist = features.weather_features(weather, hist_years=tuple(y for y in weather.year.unique() if y < 2025))
wx_prev = features.prior_year_weather(weather)
diag_all = {"claims_checks": {k: int(v) for k, v in target.claims_data_checks(claims).items()}}
for line in ("motor", "buildings", "contents"):
    pol = load.read(f"policies_{line}", EM); tst = load.read(f"test_{line}", EM)
    delay = target.reporting_delay_cdf(dev_claims[dev_claims.line == line])
    rows, diag = target.build_target(pol, claims[claims.line == line], delay=delay)
    diag["exposure_rows_repaired"] = pol.attrs["exposure_rows_repaired"]; diag_all[line] = diag
    tr = features.build(pol, customers, addresses, wx_hist, wx_prev)
    tr = tr.merge(rows[["row_id", "reported_cost", "n_claims", "completeness", "claim_cost"]], on="row_id")
    te = features.build(tst, customers, addresses, wx_hist, wx_prev)
    assert len(tr) == len(pol) and len(te) == len(tst), "join changed row counts"
    tr.to_parquet(out / f"{line}_train.parquet"); te.to_parquet(out / f"{line}_test.parquet")
    print(line, tr.shape, te.shape, json.dumps(diag, default=float))
Path(out, "diagnostics.json").write_text(json.dumps(diag_all, indent=2, default=float))

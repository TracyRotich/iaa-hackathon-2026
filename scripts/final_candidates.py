"""Confirm the two candidate changes from deep_tests on both holdouts:
 motor: training-target cap at 50k / 75k / 100k / 150k for the relativity models;
 buildings: drop credit_band, marital_status, missed_payments_12m, household_size together."""
import sys, json, importlib.util
import numpy as np, pandas as pd
line = sys.argv[1]
CAPS_TO_TEST = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [50_000, 75_000, 100_000, 150_000]
sys.argv = ["deep_tests.py", line, "none"]
spec = importlib.util.spec_from_file_location("dt", "scripts/deep_tests.py")
src = open("scripts/deep_tests.py").read().split("\nout = {}")[0]   # reuse helpers, skip the main loop
dt = {}; exec(compile(src, "deep_tests_helpers", "exec"), dt)
out = {}
for train_years, test_year in [((2021, 2022), 2023), ((2021, 2022, 2023), 2024)]:
    df = dt["df"]; tr = df[df.policy_year.isin(train_years)].copy(); te = df[df.policy_year == test_year].copy()
    tr, te = dt["prep"](tr, te)
    r = {"base": dt["run"](tr, te, dt["FEATS"])}
    if line == "motor":
        for cap in CAPS_TO_TEST:
            rep = tr.reported_cost.clip(upper=cap)
            r[f"cap_{cap//1000}k"] = dt["run"](tr, te, dt["FEATS"], train_cost=(rep / tr.completeness, rep))
    if line == "buildings":
        drop = {"credit_band", "marital_status", "missed_payments_12m", "household_size"}
        r["drop_4"] = dt["run"](tr, te, [f for f in dt["FEATS"] if f not in drop])
    out[test_year] = r
print(line, json.dumps(out))

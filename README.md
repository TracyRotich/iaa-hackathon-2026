# IAA AI-Assisted Hackathon 2026: Actuarial Pricing Across Markets

Predict expected ultimate claim cost for 2025 (net of excess and recoveries, excluding declined claims) for private motor, home buildings and home contents. Metric: Tweedie D² with power 1.5.

## Layout
```
src/iaa/      config (feature policy), target (claims → ultimate cost), metric, guards, models, validation, submission
tests/        automated checks (metric, calibration, reproducibility, leakage, exposure, submission format)
docs/         decision_log.md (governance), ai_log.md (AI use and verification)
data/         Kaggle files, not committed
submissions/  CSVs for upload
notebooks/    EDA
```

## Run
```
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

## Status
Week 1. Target construction (claims → row-level ultimate cost) and model pipeline built and tested on hand-built and synthetic data (12 tests). Next: join the competition, download the parquet files, run `target.claims_data_checks` and the trap checks in `docs/decision_log.md`.

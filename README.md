# IAA AI-Assisted Hackathon 2026: Actuarial Pricing Across Markets

Predict expected ultimate claim cost for 2025 (net of excess and recoveries, excluding declined claims) for private motor, home buildings and home contents. Metric: Tweedie D² with power 1.5.

## Layout
```
src/iaa/      config, metric, guards, models, validation, submission, synthetic test data
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
Week 1. Pipeline and tests run on synthetic data. Next: load the Kaggle data, set real column names in `src/iaa/config.py`, answer the target questions in `docs/decision_log.md`.

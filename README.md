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
Part 1. Real data loaded 21 Oct. Data repairs (exposure basis, duplicate claims, one suspect 11.9m claim), target by row, 2025 level by peril, LightGBM relativities per line, first submission `submissions/sub01_gbm_level.csv`. Submission 2 adds a Poisson x Gamma GLM (splines, credibility target encoding, ridge) blended with LightGBM. Submission 3 adds on-levelled sums insured and backfitted credibility smoothing. 18 tests pass.

## Reproduce
```
PYTHONPATH=src python scripts/prepare.py      # needs the Kaggle parquet files in data/
PYTHONPATH=src python scripts/experiment.py motor   # holdout checks per line
PYTHONPATH=src python scripts/level.py        # 2025 level by line and peril
PYTHONPATH=src python scripts/run_part1.py submissions/sub01_gbm_level.csv
PYTHONPATH=src python scripts/compare_models.py motor    # LightGBM vs GLM vs blends
PYTHONPATH=src python scripts/run_part1_v2.py submissions/sub02_blend_glm_gbm.csv
PYTHONPATH=src python scripts/gi_pricing_experiments.py motor   # on-levelled sums insured and credibility smoothing on holdouts
PYTHONPATH=src python scripts/gi_pricing_level_checks.py        # trend methods, IBNER evidence
PYTHONPATH=src python scripts/uncertainty.py             # parameter / assumption / process uncertainty
PYTHONPATH=src python scripts/run_part1_v2.py submissions/sub03_blend_gi_methods.csv --gi-methods
```

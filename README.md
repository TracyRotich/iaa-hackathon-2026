# IAA AI-Assisted Hackathon 2026: Actuarial Pricing Across Markets

Predict expected ultimate claim cost for 2025 (net of excess and recoveries, excluding declined claims) for private motor, home buildings and home contents. Metric: Tweedie D² with power 1.5.

## Layout
```
src/iaa/      config (feature policy, market folders), target (claims → ultimate cost), level, glm, gbm,
              pipeline (v4/v5 relativities), metric, guards, submission
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
Part 1. Real data loaded 21 Oct. Data repairs (exposure basis, duplicate claims, one suspect 11.9m claim), target by row, 2025 level by peril, LightGBM relativities per line, first submission `submissions/sub01_gbm_level.csv`. Submission 2 adds a Poisson x Gamma GLM (splines, credibility target encoding, ridge) blended with LightGBM. Submission 3 adds on-levelled sums insured and backfitted credibility smoothing. Submission 4 fits motor relativities on row cost capped at 25k. Submission 5 (final, private 0.13293) adds the fixes from an independent Codex review, now ported into `src/iaa/pipeline.py`: time-safe weather (R2), sqrt-exposure Tweedie weights (R3), IBNR-consistent GLM inputs (R4), and one challenger per line blended with v4.

Part 2 (emerging market): relativities from a developed-market frequency x severity model re-levelled to the emerging market (`src/iaa/transfer.py`), emerging 2025 level with borrowed reporting patterns and trends, and a mix adjustment for the 76% new-business 2025 portfolio. Run `prepare_emerging.py`, `level_emerging.py`, `run_part2.py <out.csv>`; holdout checks in `validate_part2.py`, `validate_transfer.py`, `validate_correction.py`; level sensitivity in `uncertainty_part2.py`. For other scripts, set `IAA_DATA=data/emerging` and the same scripts read and write that market's folder; the level's year settings are derived from the data (`level.history_settings`) and must be reviewed for a two-year history.

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
PYTHONPATH=src python scripts/run_part1_v2.py submissions/submission04_motor_basic_limit.csv --gi-methods --motor-cap
PYTHONPATH=src python scripts/validate_v5.py 2023 2024 --decompose   # v5 holdout checks, gain by fix
PYTHONPATH=src python scripts/run_part1_v5.py submissions/submission05_v5_own_pipeline.csv
PYTHONPATH=src python scripts/check_level_reconciliation.py           # review findings R6, R7
```

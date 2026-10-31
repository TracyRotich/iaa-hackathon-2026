# IAA AI-Assisted Hackathon 2026: Pricing Across Markets

Tracy Rotich (Kaggle team "Tracy Rotich").

The task: predict the expected ultimate claim cost of each 2025 policy row (net of excess and recoveries, declined claims count as zero) for private motor, home buildings and home contents, first in a developed market (Part 1) and then in an emerging market with only two years of history (Part 2). The metric is Tweedie D² with power 1.5.

## Results

| Part | Final submission | Score |
|---|---|---|
| Part 1, developed market | v5 (`submission05_review_fixes_blend.csv`) | private D² 0.13293 (public 0.12552) |
| Part 2, emerging market | submission 3 (`part2_submission03_si_fix.csv`) | public D² 0.15361, the best of three |

## The approach in five steps

1. **Target.** Each claim is matched to the policy row in force on its loss date. Declined claims are dropped, value is incurred less recoveries, and recent rows are grossed up for claims not yet reported (IBNR) using the reporting-delay pattern (`src/iaa/target.py`).
2. **Level.** The 2025 cost per exposure year is built peril by peril as frequency × severity, with trend (`src/iaa/level.py`). Part 1: 662.8 motor, 528.1 buildings, 123.9 contents. Part 2: 586 motor, 480 buildings, 95.4 contents.
3. **Relativities (who is riskier).** Part 1 blends a Poisson × Gamma GLM with LightGBM per line (`src/iaa/glm.py`, `gbm.py`, `pipeline.py`). Part 2 carries over a developed-market frequency × severity model, re-levelled to emerging frequency and severity (`src/iaa/transfer.py`).
4. **Prediction** = relativity × level × exposure. In Part 2 the shift to a 77% new-business book enters once, through the model (the mix identity, tested in `tests/test_transfer.py`).
5. **Validation.** Every choice was made on out-of-time holdouts: train on earlier years, score a later year (`src/iaa/validation.py`). Holdout D² is computed after rescaling to the actual total, so it measures how cost is allocated between policies; the level rests on the stated assumptions and the sensitivity tests.

## Who did what

- **Tracy Rotich** directed the project: she supplied the data and brief, developed the prompts, guided the modelling approach, commissioned the independent reviews, made both final selections, and completed the final review and approval.
- **Claude** proposed most technical approaches, wrote most of the code, ran the initial checks and drafted the AI log.
- **Codex** (Part 1) and **ChatGPT** (Part 2 and the final review) challenged the work independently. Every finding was checked in the code or data before anything changed.

Details: `docs/ai_log.md` (each AI suggestion, the decision, the check and the outcome).

## Repository map

| Path | What it holds |
|---|---|
| `src/iaa/` | The pricing library: `load.py` (read and repair the data), `target.py`, `level.py`, `features.py`, `glm.py`, `gbm.py`, `pipeline.py` (Part 1 v4/v5), `transfer.py` (Part 2), `validation.py`, `metrics.py` (local D²), `guards.py` and `submission.py` (checks before any file is trusted), `config.py` (feature policy) |
| `scripts/` | Runnable steps: data preparation, level, each submission, holdout validation, sensitivity and diagnostics (see Reproduce) |
| `tests/` | 26 automated tests: metric, target, leakage guard, exposure repair, weighting and mix identities, submission format |
| `docs/` | Decision log, AI log, review files, the brief given to Codex, checksums of the submitted files, and a readable version history |
| `data/`, `submissions/`, `notebooks/` | Empty placeholders. Competition data and prediction files are never committed (`.gitignore`) |

## Reproduce

```
pip install -r requirements.txt
PYTHONPATH=src pytest -q                                  # 26 tests; no competition data needed
```

Part 1 (Kaggle files in `data/`):
```
PYTHONPATH=src python scripts/prepare.py                  # cleaned feature frames per line
PYTHONPATH=src python scripts/level.py                    # 2025 level by line and peril
PYTHONPATH=src python scripts/validate_v5.py 2023 2024    # holdout checks of the final pipeline
PYTHONPATH=src python scripts/run_part1_v5.py submissions/submission05_v5_own_pipeline.csv
```
The Part 1 file scored by Kaggle was built by Codex; this pipeline reproduces its holdout scores within 0.00003 and its 2025 predictions exactly for property and within a median 0.07% for motor (GLM solver differences across machines).

Part 2 (developed files in `data/`, emerging files in `data/emerging/`):
```
PYTHONPATH=src python scripts/prepare.py                  # developed frames, used for the transfer prior
PYTHONPATH=src python scripts/prepare_emerging.py         # emerging frames
PYTHONPATH=src python scripts/level_emerging.py           # 2025 level by peril with severity credibility
PYTHONPATH=src python scripts/run_part2.py submissions/part2_sub03_si_fix.csv --si-fix
PYTHONPATH=src python scripts/validate_review.py          # holdout evidence for the final choices
PYTHONPATH=src python scripts/uncertainty_part2.py        # level sensitivity
```

Earlier submissions and the other analyses: `run_part1.py` (v1), `run_part1_v2.py` (v2; `--gi-methods` for v3, plus `--motor-cap` for v4), `compare_models.py`, `deep_tests.py`, `gi_pricing_experiments.py`, `check_*.py`, `validate_part2.py`, `validate_transfer.py`, `validate_correction.py`, `uncertainty.py`, `diagnostics_writeup.py`, `reference_premiums.py`.

## Governance and review evidence

| File | Content |
|---|---|
| `docs/decision_log.md` | 58 decisions, each with its evidence and residual risk, plus the review tables |
| `docs/ai_log.md` | 26 AI suggestions with the decision, the check and the outcome |
| `docs/neutral_task_brief.md` | The brief given to Codex for its independent review |
| `docs/part2_review_brief_for_chatgpt.md`, `docs/part2_response_to_chatgpt_review.md` | ChatGPT's review of Part 2 and the response |
| `docs/final_review_*.md` | The final review: Tracy Rotich with ChatGPT, ChatGPT's follow-up, and Claude's checks of each point |
| `docs/final_submission_sha256.txt` | Checksums identifying the two submitted prediction files |
| `docs/version_history.html` | The commit history in readable form |

## Main limitations

- Claims are valued at a single date (31 December 2024), so case-estimate development cannot be measured and the level has no forward test.
- The emerging market has one holdout year, and its 2025 book is 77% new business against 9% in the holdout; new business is the largest open uncertainty.
- Late-reported bodily-injury claims cost more than average, so the count-based IBNR gross-up slightly understates recent bodily-injury-heavy rows.

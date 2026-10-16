# Decision log

One entry per material choice. Each states what was decided, why, what evidence supported it, and what would change it. This is the governance record for the 20% validation and governance criterion.

| # | Date | Decision | Why | Evidence / check | Revisit if |
|---|---|---|---|---|---|
| 1 | 16 Oct | Local metric = `sklearn.metrics.d2_tweedie_score(power=1.5)` | Matches the competition metric as described on the Overview tab | Unit test: a constant mean prediction scores 0 | Kaggle's evaluation page defines it differently |
| 2 | 16 Oct | Validate by time: fit 2021–2023, score 2024, then refit on all years | The task is to predict a future year; random splits would overstate performance | — | 2024 targets turn out to be too immature to score fairly |
| 3 | 16 Oct | Model the rate y/exposure with weight = exposure | Equivalent to a log offset under a compound Poisson-gamma model with frequency proportional to exposure | Synthetic test: calibrated totals | The data has no exposure field, or rows are not on a comparable exposure basis |
| 4 | 16 Oct | Separate model per line (motor, buildings, contents) | Different perils, features and severity shapes | — | A line is too thin to fit alone; then pool with a line factor |
| 5 | 16 Oct | Tweedie GLM as benchmark, LightGBM Tweedie as challenger, blend weight set on the 2024 holdout only | Explainability plus accuracy; the blend is chosen without touching test | — | GBM shows no lift over GLM on holdout |

## Findings so far

- **16 Oct, synthetic data:** with a 5% a year trend built in and no trend term in the models, the GLM predicted 83% and LightGBM 78% of the held-out 2024 total. Calibration to the projection year needs an explicit trend step (or year as a feature, extrapolated with care), checked on the holdout before use on 2025. The default LightGBM settings also overfit 6,000 rows (D² 0.002 vs GLM 0.072): tune with early stopping on the time split.

## Open questions on the target (answer from the Data tab before modelling)

1. Is the training target already **ultimate**, or incurred to date? If to date, 2024 is understated and needs development (link to reserving: a chain-ladder-style uplift by line).
2. Are excess and recoveries already netted off, and declined claims already removed, in the training target?
3. Is there an exposure field? Are rows full years or part periods?
4. How should 2025 trend be handled: claims inflation, frequency trend, mix change?
5. Any fields recorded after a loss (claim dates, statuses, payments)? These are excluded as leakage.

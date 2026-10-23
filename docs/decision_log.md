# Decision log

One entry per material choice: what was decided, why, the evidence, and what would change it. This is the governance record for the 20% validation and governance criterion.

| # | Date | Decision | Why | Evidence / check | Revisit if |
|---|---|---|---|---|---|
| 1 | 16 Oct | Local metric = `sklearn.metrics.d2_tweedie_score(power=1.5)` | Matches the competition metric | Unit test: constant mean prediction scores 0 | Kaggle's evaluation page defines it differently |
| 2 | 16 Oct | Validate by time: fit 2021–2023, score 2024, then refit on all years | The task predicts a future year; random splits overstate performance | — | 2024 targets too immature to score fairly even after IBNR |
| 3 | 16 Oct | Model the cost rate y/exposure with weight = exposure | Equivalent to a log offset under a compound Poisson-gamma model with frequency proportional to exposure. Rows are part-years (mid-term changes, cancellations), so this matters | Synthetic test: calibrated totals | — |
| 4 | 16 Oct | Separate model per line | Different perils, features and severity shapes | — | A line is too thin; pool with a line factor |
| 5 | 16 Oct | Tweedie GLM benchmark, LightGBM Tweedie challenger, blend weight set on the 2024 holdout only | Explainability plus accuracy, without touching test | — | GBM shows no lift over GLM on holdout |
| 6 | 16 Oct | **Build the target from the claims table** (`src/iaa/target.py`): match each claim to the row in force on its loss_date; drop repudiated; value = incurred_to_date − recoveries (incurred is already net of excess); sum by row | The Data tab gives no target column. Policies have several rows per year, so matching by policy_id alone would misallocate claims | Unit tests on mid-term change, cancellation, repudiation and recoveries | Recoveries turn out to be already netted in incurred (check: compare paid, incurred and recoveries on closed claims) |
| 7 | 16 Oct | Claims with loss dates outside every row of their policy are **reported and excluded**, not forced onto a row | They are either data errors or post-cancellation losses; neither belongs to the row's risk | Count reported in diagnostics | Count is material (then investigate before excluding) |
| 8 | 16 Oct | **IBNR**: gross up each row's reported cost by 1 / completeness, where completeness = the line's empirical reporting-delay CDF averaged over the row's period | Claims are valued at 31 Dec 2024 and some 2024 claims are unreported. Without this, 2024 frequency is understated and the model learns a false downward trend | Unit test: a Q4 2024 row is materially incomplete, a 2022 row is complete | Delay differs strongly by peril (then do it by peril); seasonality in losses |
| 9 | 16 Oct | **IBNER** factor on open claims defaults to 1.0, with sensitivity runs at e.g. 1.05 and 1.10 reported | A single snapshot shows no case-estimate development, so it cannot be estimated from history. Stated as a limitation, not hidden | Diagnostics report the share of value in open claims | Open claims are a large share of 2024 value |
| 10 | 16 Oct | Feature policy in `config.EXCLUDE` / `config.REVIEW`: exclude leakage, fairness proxies, marketing artefacts and pricing-circular fields; review permitted-but-sensitive factors | See the trap list below | Test that excluded fields stay excluded | — |
| 11 | 21 Oct | **Exposure recomputed from the row dates** (days / 365) for every row | 5,357 motor, 3,895 buildings and 4,032 contents rows had exposure in days (e.g. 327.0) or on a 360-day year (1.0139 for a full year). Raw totals overstated exposure by about 35% and understated every rate | Test files are all on days / 365; unit test | — |
| 12 | 21 Oct | **Drop exact duplicate claims** (identical except claim_id; 604 pairs, 0.4% of incurred) | Double-counting; one id of each pair is usually 9 digits, the rest 8 | Unit test | — |
| 13 | 21 Oct | **Cap claims at 1.2m** in the target and the level | One open motor bodily-injury case estimate is 11.9m, ten times the next largest (1.12m closed). It looks like a keying error (paid 0.74m) | Listed top 8 incurred | Organisers confirm it is genuine |
| 14 | 21 Oct | Recoveries are **not** inside incurred: value = incurred − recoveries | On closed claims incurred = paid exactly, including the 9.8% with recoveries | Data check | — |
| 15 | 21 Oct | **Level and relativities separated.** Level = 2025 frequency × severity by peril (`level.py`); relativities = LightGBM per line; predictions rescaled so each line's total = level × test exposure | Motor bodily injury is immature (80% of 2024 BI claims open, only 56% reported), so a model trained on rows would understate 2025. A peril-level projection handles IBNR, IBNER and trend explicitly | Holdout D² for relativities; level checked against 2021–2024 rates | — |
| 16 | 21 Oct | **IBNR by peril** for the level (delay pattern from 2021 losses) | Bodily injury and subsidence report far later than other perils (2024 completeness 56% and 69%) | — | — |
| 17 | 21 Oct | **Weather: 2021–2024 average and prior year only; 2025 weather not used** | The file contains 2025 weather. It is not known when the 2025 price is set, so using it is leakage, even if it would raise the leaderboard score | — | Organisers state that 2025 weather is meant as a forecast input |
| 18 | 21 Oct | Drop `previous_insurer` | 37 levels with near-identical frequency; it was the top contents feature by gain, i.e. the GBM was fitting noise. Contents holdout D² rose from 0.069 to 0.078 when removed | Holdout | — |

## Holdout results (21 Oct, LightGBM Tweedie per line, D² after rescaling to actual total)

| Line | Train 2021–22 → test 2023 | Train 2021–23 → test 2024 | Top features |
|---|---|---|---|
| Motor | 0.074 | 0.068 | make_model, ncb_years, years_licensed, driver_age, occupation, annual_mileage |
| Buildings | 0.105 | 0.104 | occupation, flood_zone, elevation_m, sum_insured, region, distance_to_river_km |
| Contents | 0.086 | 0.089 | flood_zone, sum_insured, occupation, elevation_m, region |

2025 level (cost per exposure year): motor 662.8, buildings 528.1, contents 123.9. History after repairs: motor 564/576/537/486 (2021–24, later years immature), buildings 384/403/433/473, contents 106/106/117/120.

**Submission 1 (21 Oct):** `sub01_gbm_level.csv`.

## To review next
- Occupation is the top buildings feature (severity 7.9k–10.3k by occupation). Check whether it is genuine or a proxy, and test dropping it.
- Measure what same-year weather would add on the holdout, to quantify the trap for the write-up.
- GLM benchmark per line; blend.
- Level backtest and sensitivity (severity trend ±2%, BI IBNER 1.0/1.1).


## GLM benchmark (21 Oct)

GLM methods used and how each was adapted:

| Method | Used here | Adaptation |
|---|---|---|
| Poisson GLM (frequency) × Gamma GLM (severity), log link | `glm.FreqSevGLM` | Exposure as weight on the rate (offset-equivalent); IBNR-adjusted counts |
| Natural splines with knots learned on training data only | Spline basis per numeric feature, knots at training quantiles, frozen for test | Applied to every numeric feature with 12+ values, after 1–99% clipping |
| Actuarial target encoding (credibility, leave-one-out) | `glm.CredibilityEncoder` for make_model and occupation | κ estimated by Bühlmann–Straub (the fully data-driven option) instead of κ = 20; unit test reproduces the slide example (2.513 / 2.536) |
| Ridge (L2) shrinkage GLM | `alpha` in PoissonRegressor / GammaRegressor | Stein-type savvyGLM is R-only; ridge is the available Python analogue |
| Metric suite: Gini (normalised, weighted), A/E, winsorised RMSE, deviance | `metrics.scorecard`, `weighted_gini` ported from R | D² (competition metric) added; Gini on cost rate weighted by exposure |
| No leakage from test into training | All encoders, splines, medians fitted on training years only | Validation by **time** (2021–22 → 2023, 2021–23 → 2024), not random 5-fold CV, because the same policy appears in several years and the task is a future year |
| Bias–variance trade-off; shrinkage towards a target | Blend of GLM and GBM; credibility | Planned for Part 2: shrink new-market estimates towards the Part 1 model |

### Results on the holdouts (D² after rescaling; 2023 / 2024)

| Line | LightGBM | GLM | Best blend (weight on GBM) |
|---|---|---|---|
| Motor | 0.075 / 0.070 | **0.090 / 0.082** | 0.3 → 0.090 / 0.082 |
| Buildings | **0.106 / 0.110** | 0.097 / 0.102 | 0.7 → **0.108 / 0.111** |
| Contents | 0.088 / 0.089 | 0.089 / 0.091 | 0.5 → **0.093 / 0.094** |

The blend weight was chosen on 2023 and the same weight was best on 2024 for every line.

### What credibility said about the features
- **Buildings occupation:** Bühlmann κ = ∞ for both frequency and severity, i.e. no between-occupation signal beyond noise. It had been the top buildings GBM feature by gain. Dropping it changed holdout D² by −0.0002 (2023) and −0.0004 (2024), so it is **removed** (decision 19): no accuracy cost, and one fewer sensitive factor to justify.
- **Contents occupation:** frequency κ ≈ 2,700 exposure-years (some signal), severity κ = ∞.
- **Motor occupation:** frequency κ ≈ 130–190 (strong signal: students), severity none.
- **Motor make_model** (300 models): κ ≈ 3,000 exposure-years for frequency, so each model gets about 45% credibility. Credibility-encoding it for LightGBM lifted motor GBM D² from 0.075 to 0.081 (2023) and 0.070 to 0.074 (2024); the raw categorical was overfitting (48% of GBM gain).

| 19 | 21 Oct | Drop occupation from buildings | Credibility κ = ∞; no holdout loss when removed; fairness | Holdout | — |
| 20 | 21 Oct | **Submission 2** = per-line blend of LightGBM and GLM (motor 0.3 / buildings 0.7 / contents 0.5 on GBM), rescaled to the same 2025 level | Best holdout D² on every line; weights stable across both holdouts | Holdout tables above | — |

## Sums insured, credibility smoothing and trend (21 Oct)

| 21 | 21 Oct | On-level sums insured to 2021 money (index from same-policy growth: 3% buildings, 2% contents) | 2025 values otherwise sit above the training range for tree splits | GBM holdout D² up on both years for both lines | — |
| 22 | 21 Oct | Backfitted credibility smoothing for make_model and occupation in the GLM | one-way encoding double-counts effects shared with age and other factors | Motor GLM D² +0.0004 / +0.0002; motor occupation κ ~150 → ~600; contents occupation κ = ∞ | — |
| 23 | 21 Oct | Keep the all-in severity trend; report median and mix-adjusted trends as the assumption range |  | Level moves −5% to +4% across methods | Organisers' view on trend |
| 24 | 21 Oct | **Submission 3** = submission 2 + decisions 21–22 | Small, consistent holdout gains; principled | `run_part1_v2.py --gi-methods`; sub02 still reproduces byte-identically | — |

## Findings so far

- **16 Oct, synthetic data:** with a 5% a year trend and no trend term, the GLM predicted 83% and LightGBM 78% of the held-out total. The 2025 projection needs an explicit trend step, checked on the holdout before use. Default LightGBM overfits small data (D² 0.002 vs GLM 0.072): tune with early stopping.

## Traps spotted in the data description (check each once the data is loaded)

| Field or issue | Risk | Handling |
|---|---|---|
| `renewal_decision` | Leakage: an end-of-period outcome, present in training only | Excluded |
| `claims` table fields (status, paid, incurred, report_date, repairer_id, third_party_id) | Leakage if joined as features | Used only to build the target |
| `weather` by address **and year** | If 2025 weather is supplied for test rows, it is observed after the price is set. Using it would be leakage | Use the prior year or a multi-year average per address; test whether same-year weather is available |
| `customer_service_contacts` | May include contacts caused by a claim in the same period | Excluded until shown to be independent of claims |
| `surname`, `email_domain` | Fairness: proxies for ethnicity, age, income. No risk rationale | Excluded |
| `marketing_segment`, `campaign_code`, `branch_id`, `document_preference`, `marketing_consent` | No risk rationale; proxies or noise | Excluded |
| `written_premium`, `discount_applied` | Circular: they encode current pricing; the target is risk premium. Week 2 market may price differently | Excluded from the main model; tested as a challenger only |
| `vehicle_colour` | Folklore rating factor | Kept in REVIEW; drop unless it earns its place on holdout |
| `credit_band`, `occupation`, `marital_status`, `previous_insurer` | Permitted in UK pricing but sensitive under Consumer Duty | Kept with a stated rationale and a proxy check |
| `telematics_score` blank when not opted in | Missing not at random | Keep opt-in flag; treat blank as its own level |
| Part-year rows (mid-term changes, cancellations, lapses) | Exposure and claim matching | Rate model with exposure weight; claims matched by loss_date |
| Claims whose loss_date falls outside any row | Post-cancellation or data error | Counted and excluded (decision 7) |
| Repudiated claims with payments; recoveries above incurred; report before loss | Data quality | `target.claims_data_checks` counts each |
| Sum insured | Caps the claim size | Use as exposure-of-value feature; check claims never exceed it |

## Still open

1. Trend to 2025: frequency and severity by line and peril from the 2021–2024 data (after IBNR).
2. Whether 2025 weather appears in the weather table.
3. Competition-specific rules not yet seen (team size, external data, AI tools, submissions per day): visible only after joining.

## Open items (logged 23 Oct)
1. **2025 level:** largest scoring risk; ±1 pp severity trend ≈ ±3% on predictions. Locked for Part 1 at close; report as uncertainty in the write-up and revisit for the Week 2 market.
2. **Checks not yet run (write-up evidence, do not change Part 1 predictions):** calibration by price band (decile A/E), double lift chart sub02 vs sub03, proxy-fairness test (occupation, credit band, marital status), explanation charts (SHAP / partial dependence). Planned for Week 2.

## Kaggle public leaderboard (23 Oct, Part 1)

| Submission | Public D² | Holdout ranking agreed? |
|---|---|---|
| submission01_gbm_level (v1) | 0.11933 | — |
| submission02_blend_glm_gbm (v2) | 0.12489 | yes (v2 > v1) |
| submission03_blend_pricing_concepts_applied (v3) | **0.12545** | yes (v3 > v2) |

The public ranking matches the holdout ranking: the GLM blend added +0.0056 and on-levelled sums insured and credibility smoothing a further +0.0006. Public scores sit above our per-line holdout scores (0.08–0.11) because Kaggle scores all three lines together: the constant baseline cannot separate motor (≈663) from contents (≈124), so part of the score comes from getting the line levels right. **Final Submission selected: v3.**

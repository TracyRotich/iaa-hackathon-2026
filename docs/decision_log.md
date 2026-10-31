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
| 23 | 21 Oct | Keep the all-in severity trend; report median and mix-adjusted trends as the assumption range | Trend method is uncertain | Level moves −5% to +4% across methods | Organisers' view on trend |
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

| 25 | 23 Oct | **Keep written_premium excluded** (tested after seeing the leaderboard gap) | Adding log(written premium rate) to the v3 blend did not help consistently on the holdouts: blend D² change motor −0.0002 / +0.0005, buildings −0.0017 / −0.0011, contents −0.0001 / +0.0007 (2023 / 2024). The current tariff adds no reliable information beyond our rating factors, and using it would make the model depend on the old prices | `scripts/check_written_premium.py` | — |

## Further method tests (23 Oct)

| 26 | 23 Oct | **Motor basic limit:** relativity models trained on claims capped at 25k; level unchanged | Increased-limits practice: large BI claims are largely random with respect to rating factors | Holdout blend D² +0.0011 / +0.0021 (10k–150k all positive; 25k best overall). Home lines tested and rejected | Part 2 data shows predictable large losses |
| 27 | 23 Oct | Keep frequency × severity rather than Tweedie GLM | Motor GLM 0.0906 vs 0.0868 (2023), 0.0818 vs 0.0804 (2024); home lines about equal | `scripts/deep_tests.py tweedie` | — |
| 28 | 23 Oct | Reject total-excess feature, seasonality shares, 6-knot splines | No consistent holdout gain; Ockham | `scripts/deep_tests.py` | — |
| 29 | 23 Oct | Keep buildings credit band, marital status, missed payments, household size | Individually neutral, but dropping all four costs −0.0011 in both years | `scripts/final_candidates.py buildings` | — |
| 30 | 23 Oct | Log but do not adjust: 10 claims above sum insured (open case estimates above limit), 0.3% impossible licence years | Immaterial (≤0.03% of incurred; no frequency difference); flagged as data-quality findings | Data checks | — |
| 31 | 23 Oct | **Submission 4** = submission 3 + decision 26 (motor only) | Buildings and contents identical to v3; motor total unchanged | Validation script; `run_part1_v2.py --gi-methods --motor-cap` | — |

## Review of submission 4 (23 Oct)

Concern raised in review: capping motor claims at 25k and rescaling proportionally spreads large-loss cost in proportion to *attritional* cost. On the 2025 test rows this lowered third-party-only by 9.7%, third-party fire & theft by 4.8% and drivers aged 17–21 by 3.4% compared with v3. Every motor policy carries third-party liability, which is where large bodily-injury claims arise.

Test (`scripts/check_large_loss_allocation.py`): three ways to allocate the cost above 25k, compared with the uncapped v3 method on both holdouts (actual / expected; 1.00 = right):

| | U: uncapped (v3) | A: proportional (v4) | B: uniform load per exposure | C: load ∝ uncapped model |
|---|---|---|---|---|
| D² 2023 / 2024 | 0.0905 / 0.0819 | **0.0916 / 0.0840** | 0.0889 / 0.0818 | **0.0916 / 0.0838** |
| Third-party only | 1.06 / 0.84 | 1.15 / 0.94 | 1.12 / 0.91 | 1.14 / 0.92 |
| Third-party fire & theft | 0.90 / 0.78 | 0.95 / 0.83 | 0.94 / 0.81 | 0.94 / 0.82 |
| Age 17–21 | 1.00 / 1.00 | 1.04 / 1.04 | 1.14 / 1.13 | 1.04 / 1.03 |
| Age 21–25 | 0.91 / 0.87 | 0.93 / 0.89 | 0.99 / 0.94 | 0.93 / 0.89 |

Findings: (1) the D² gain of v4 is real and holds whichever allocation is used, except a uniform load (B), which is worse; (2) the concern is partly confirmed: v4 under-prices drivers aged 17–21 by about 4% on both holdouts, where v3 was right; (3) but it improves third-party fire & theft and ages 21–25 in both years, and third-party only is noisy (under in 2023, over in 2024 for every method); (4) C is the most defensible allocation in principle but performs the same as A.

| 32 | 23 Oct | **Keep submission 4 as built (allocation A)**; record the young-driver effect as a known limitation | D² better on both holdouts; segment A/E mixed rather than worse overall; C no better in practice | Allocation test above; v4 and v3 both reproduce byte-identically | Part 2: if young-driver A/E stays above 1.03, allocate the large-loss load by bodily-injury frequency instead |

## Independent review by a second AI model (Codex, 23 Oct)

Tracy Rotich gave Codex the task brief (`docs/neutral_task_brief.md`), the data, and this repo. Its findings, each checked here:

| # | Codex finding | Checked | Verdict |
|---|---|---|---|
| R1 | Holdout scores rescaled to the actual total hide level errors | Yes: by design; it added a forecast-level diagnostic | Valid. Report both in the write-up |
| R2 | **Weather averages cross the time boundary in validation** (2023 holdout rows use a 2021–24 average); prior-year weather is missing only for 2021, so it acts as a year flag | Confirmed in `features.weather_features` (default 2021–24) | **Valid. Validation leakage only**; 2025 test features use past years, so the submissions are not leaked. Fix for Part 2 |
| R3 | At Tweedie p = 1.5, row deviance = √exposure × rate deviance, so √exposure weights match the metric | Confirmed numerically | Valid nuance. Exposure weights remain the actuarial choice for the Poisson part |
| R4 | **IBNR cancels in the GLM**: counts are divided by completeness but costs are not, so severity is understated and the product reverts to reported cost | Confirmed in `run_part1_v2.py` (GLM receives `reported_cost`) | **Valid bug.** The level is unaffected; relativities of recent rows are slightly distorted. Fix: pass `claim_cost` |
| R5 | The motor 25k cap is applied to row cost, not individual claims; the 11.9m claim is not *proven* an error | Confirmed | Valid. Wording corrected below |
| R6 | Row targets use line-level delay; the level uses peril-level delay | Confirmed | Valid. Reconcile in Part 2 |
| R7 | The level includes claims unmatched to a row (296 motor, 67 buildings, 50 contents), while targets exclude them | Confirmed; most motor ones have 2020 loss dates, which the level drops anyway | Minor. Reconcile |
| R8 | Splines are cubic B-splines (SplineTransformer), not natural splines; 2025 weather exclusion is an assumption, not a rule | Confirmed | Valid. Documentation corrected |

Codex challenger (`codex_challenger.csv`): same line totals as v4. It uses the corrected GLM (motor), shallow √exposure-weighted boosting (buildings) and a frequency/severity GBM (contents), blended with v4. The local combined gain is +0.00077 (2023) and +0.00075 (2024), and the 95% bootstrap intervals per line include zero. File checked here: 626,639 rows, sample order, non-negative, totals equal to v4.

| 33 | 23 Oct | Accept review findings R2, R4, R6–R8 as fixes for the Part 2 pipeline; R1 and R3 into write-up methodology | Independent review; each finding verified in code or numerically | Table above | — |
| 34 | 23 Oct | Treat `codex_challenger.csv` as a credible Part 1 candidate: small, consistent local gains; it corrects R2 and R4 | Selection fixed in advance by Codex (choose on 2023, confirm on 2024) | Its validation_results.csv; our file checks | Public score lower than v3/v4 |

Correction to the wording of decision 26: the motor "25k cap" caps each **row's** total reported cost at 25k, not each claim.

## Part 1 result (23 Oct, after close)

| Submission | Private (~67%) | Public (~33%) |
|---|---|---|
| v5 `submission05_review_fixes_blend` (**selected**) | **0.13293** | 0.12552 |
| v4 `submission04_motor_basic_limit` | 0.13231 | **0.12578** |
| v3 `submission03_blend_pricing_concepts_applied` | 0.13227 | 0.12545 |

Final private rank 34 (public rank at close 29).

| 35 | 23 Oct | **Selected v5 over v4** despite v4's higher public score | Public gap (0.00026) within noise on a 33% slice; v5 gained ~0.00075 on both full holdout years and fixes R2/R4 | Private result: v5 beat v4 by +0.00062, close to the holdout estimate; the public board ranked them the wrong way round | Lesson for Part 2: select on out-of-time holdouts, not the public leaderboard |

## v5 ported into our own pipeline (25 Oct)

The review fixes now live in `src/iaa` (`pipeline.py`, `features.time_safe_weather`, `gbm.fit_predict_variant`), run by `scripts/validate_v5.py` and `scripts/run_part1_v5.py`.

**Reproduction check.** Holdout D² (allocation only, rescaled to the actual total):

| Line | Year | v4 base (ours / Codex) | Challenger (ours / Codex) | v5 (ours / Codex) |
|---|---|---|---|---|
| Motor | 2023 | 0.09156 / 0.09158 | clean GLM 0.09208 / 0.09208 | 0.09220 / 0.09221 |
| Motor | 2024 | 0.08402 / 0.08402 | 0.08458 / 0.08458 | 0.08464 / 0.08464 |
| Buildings | 2023 | 0.10732 / 0.10732 | shallow GBM 0.10666 / 0.10666 | 0.10849 / 0.10849 |
| Buildings | 2024 | 0.11080 / 0.11080 | 0.11142 / 0.11142 | 0.11217 / 0.11216 |
| Contents | 2023 | 0.09331 / 0.09331 | freq-sev GBM 0.09179 / 0.09179 | 0.09384 / 0.09384 |
| Contents | 2024 | 0.09377 / 0.09378 | 0.09091 / 0.09091 | 0.09380 / 0.09380 |

Combined: v5 gains +0.00077 (2023) and +0.00075 (2024) over v4, as Codex reported.

2025 file (`submission05_v5_own_pipeline.csv`) against `codex_challenger.csv`: same rows and order; line totals equal; buildings and contents identical; motor differs by a median 0.07% (max 0.5%). Cause traced: our v4 refit equals the v4 CSV to 1e-15, and every LightGBM model matches exactly, so the difference is the motor GLM, whose lbfgs solution differs slightly between Codex's Windows environment and ours (numpy/pandas versions, BLAS). Immaterial for pricing; the file from our pipeline is the reproducible version of v5.

**Where the v5 gain comes from (decomposition, motor GLM, D² 2023 / 2024):**

| GLM variant | 2023 | 2024 |
|---|---|---|
| Original (reported cost, leaky weather) | 0.09219 | 0.08454 |
| R4 only (ultimate cost) | 0.09220 | 0.08453 |
| R2 only (time-safe weather) | 0.09209 | 0.08459 |
| Both (clean GLM) | 0.09208 | 0.08458 |

R4 changes the motor score by ±0.00001: an IBNR correction matters little because few rows are immature. R2 lowers the 2023 score by 0.0001, which is the leak being removed (the old validation was slightly flattering), not a loss of real accuracy. **So the motor gain comes from moving weight to the GLM** (v5 motor is about 7.5% GBM / 92.5% GLM, against 30/70 in v4), not from the fixes. The buildings and contents gains come from the new challengers (shallow sqrt-weighted Tweedie GBM; frequency × severity GBM). R2-R4 are correctness fixes: they make the validation honest and the model internally consistent, which matters more in Part 2, where 2024 rows are a larger share of a two-year history.

**R6 and R7 measured** (`scripts/check_level_reconciliation.py`): excluding unmatched claims from the level changes the 2025 rate by −0.16% motor, −0.17% buildings, −0.10% contents; line-level vs peril-level reporting patterns differ by at most 0.15 percentage points in completeness for any loss year. Both immaterial; level unchanged for Part 1. Part 2: one shared delay function for target and level.

| 36 | 25 Oct | Port the v5 fixes and model choices into our own pipeline; keep v5 settings unchanged | Our code must reproduce the final submission; Part 2 builds on it | Holdout scores match Codex within 0.00003; 2025 file matches except a 0.07% median GLM numerical difference on motor | Pin scipy and threadpoolctl; consider a tighter GLM tolerance in Part 2 (changes outputs, so not applied to Part 1) |
| 37 | 25 Oct | Describe v5's gain honestly in the write-up: fixes R2-R4 are correctness fixes; the score gain is from GLM weight (motor) and new challengers (buildings, contents) | Decomposition above | Single-fix runs on both holdouts | — |
| 38 | 25 Oct | R6, R7: no change to the Part 1 level; R8: wording corrected in `glm.py` and `features.py` | Measured effects ≤ 0.17% | Reconciliation script | Part 2: shared delay function |
| 39 | 25 Oct | Market folders (`IAA_DATA`, `IAA_WORK`) and data-derived level years (`level.history_settings`) for Part 2 | Emerging market has 2023-24 only | Developed level reproduces 662.78 / 528.10 / 123.88; new tests | Review every year setting against the emerging data before use; two years cannot support a fitted trend |

# Part 2: emerging market (from 28 Oct)

## What changed (EDA, 28 Oct)

| Area | Developed | Emerging | Consequence |
|---|---|---|---|
| History | 2021-24, ~2.4m rows | 2023-24, ~109k rows (1/20) | Models on emerging alone overfit; borrow from developed |
| 2025 portfolio | 15% new business | **76% new business** (broker portfolio), younger drivers; test exposure 2x training | Level needs a mix adjustment |
| Motor | freq 0.141, sev 3,883 | freq 0.182, sev 2,250 | Different level; shape tested for transfer |
| Climate | | hotter, more hail and wildfire; glass/hail freq 0.035 vs 0.020 | Weather features kept; level from emerging data |
| Features | driver/policyholder age, credit band, telematics, mileage | age **bands**, gender; no credit, telematics, mileage | Harmonise ages to bands; gender excluded |
| Data quality | exposure on wrong basis | same trap (565 rows); **occupation 80 spellings of 40 jobs**; ~12% missing on several fields (2% developed) | Exposure repaired; text normalised; GBM handles missing natively |
| Part 1 miss | `construction_type` had 11 spellings of 5 types | | Found while checking emerging; noted, Part 1 not rerun |
| Reporting delays | | same distribution as developed at observable ages | Developed delay pattern borrowed |
| New files | | `expenses` (loadings by market), `reference_policies` (20 policies) | Awaiting Data-tab instructions |

| 40 | 28 Oct | Borrow developed reporting-delay patterns (row targets by line; level by peril) | Emerging losses too recent to observe a 2-year tail; observable delay quantiles match (e.g. motor median 9 days, 90th pct 46 vs 48) | Delay quantile comparison | IBNR in emerging is a borrowed assumption |
| 41 | 28 Oct | Exclude `gender` from pricing; measure only | UK/EU standard (Test-Achats); emerging regulation unknown; holdout gain negligible (motor GLM 0.051 -> 0.054, buildings worse) | Holdout test | Revisit if told gender pricing is allowed and material |
| 42 | 28 Oct | Level: emerging peril level with developed severity trends (6.6/7.4/3.2%), no frequency trend, BI and subsidence severity from 2023 only | Two years cannot support fitted trends; emerging 2023->24 short-tail severity change (motor 8.3%, buildings 7.9%, contents -2.0%) is consistent for motor/buildings, noisy for contents | `level_emerging.py` | 2025 rates at 2023-24 mix: motor 505.6, buildings 476.1, contents 94.9; buildings carries fire volatility (70 claims, 44% of cost) |
| 43 | 28 Oct | **Relativities from the developed market** (freq x sev GBM on harmonised features, re-levelled to emerging); motor blended 25% with an emerging-only GBM | 2024 holdout D²: prior 0.110/0.097/0.083 (buildings/contents/motor capped) vs best emerging-only 0.064/0.054/0.072; emerging GLMs below zero; boosting emerging trees from the prior made it worse | `validate_part2.py`, `validate_transfer.py` | Single holdout year; ~500 property claims a line |
| 44 | 28 Oct | 2025 level adjusted for mix: mean model rate on 2025 rows / on 2023-24 rows | 76% new business and younger drivers in 2025; new business frequency is +39% (developed) and +30% (emerging) for motor only; buildings 1.05 / 0.94 and contents 1.16 / 1.37 (corrected 29 Oct, decision 52) | Mix factors: motor 1.131, buildings 1.076, contents 1.043 | NB effect taken from the model; A/E by NB on holdout checked (decision 45) |
| 45 | 28 Oct | **Reject** a credibility-style frequency correction (region, payment method, new business, urban density, channel, age band) on top of the developed prior | Full correction worse on every line at every penalty (e.g. buildings 0.110 -> 0.104-0.107); single-factor gains ≤ 0.001 are within one-year noise and choosing them would fit the holdout | `validate_correction.py` | Holdout A/E: monthly payment ~1.09-1.10 in all lines and region differences are emerging-specific signals the prior lacks; revisit with more emerging data |
| 46 | 28 Oct | Reject make_model in the motor prior | Holdout 0.0830 -> 0.0806 | scratch test | — |
| 47 | 28 Oct | Borrowed bodily-injury reporting pattern confirmed | Share of H1-2023 BI losses reported within 365 days: emerging 0.887, developed 0.892 | Delay comparison | — |
| 48 | 28 Oct | Submit `part2_submission01_developed_prior.csv`: rates 572 / 512 / 99 (motor / buildings / contents) | Decisions 40-47 | 209,063 rows, sample order, non-negative | Level uncertainty below |

**Level sensitivity** (`scripts/uncertainty_part2.py`, % change in 2025 rate): severity trend ±2pp → ±3% all lines; BI open claims, outstanding +20% (IBNER) → motor +3.0% (**corrected 30 Oct, decision 57**: the first run scaled the whole incurred amount, including paid, and gave +3.8%); BI/subsidence severity from both years → buildings −1.5%; emerging (truncated) completeness → motor −1.3%; **removing the five fires above 250k net (four from 2023, one from 2024; the script's rule is net value > 250k) → buildings −15.9% (wording corrected 29 Oct after the ChatGPT review; the script always removed five)**. Buildings is the least certain line: fire is 44% of its cost and rests on 70 claims. The large fires are kept: they are genuine closed claims, and fire severity is plausibly higher in a hotter, higher-wildfire climate.
| 49 | 28 Oct | **Severity credibility in the emerging level**: benchmark = emerging median × developed mean/median; Z = s_h²/(s_h² + se²) | Fire rests on 70 claims with a heavier tail than developed (mean/median 2.10 vs 1.77; medians equal); data-driven Z rather than a judgement cap | Z: fire 0.29 (−11%), subsidence 0.38 (+11%), BI 0.70 (+7%), high-volume perils ≥ 0.93. Level: motor 505.6 → 518.0, buildings 476.1 → 455.7, contents 94.9 → 93.5 | s_h from 15 perils; reported as an assumption |
| 50 | 28 Oct | Review finding R6 is **material in the emerging market**: the line-level reporting pattern in row targets understates motor 2024 by about 14% (row rate 410 vs peril-based 474), because bodily injury's slow reporting is diluted. The level uses peril-level patterns (correct). Relativities are unaffected in practice: motor relativities are fitted on row cost capped at 25k | BI ultimate frequency is stable by peril (0.0150 in 2023, 0.0162 in 2024) | Year-by-year reconciliation of level vs row targets | Part 2 code: a peril-level row IBNR would make 2024 validation targets consistent; not done before the deadline |
| 51 | 28 Oct | Submit `part2_submission02_credibility_level.csv`: same relativities as submission 1, level with severity credibility; rates 586 / 490 / 97.5 (motor / buildings / contents) | Decisions 49-50 | Ratio to submission 1 is exactly 1.0245 / 0.9572 / 0.9847 per line (level only); 209,063 rows | Preferred over submission 1 on method; the leaderboard cannot separate them reliably |

## Independent review of Part 2 by a second AI model (ChatGPT, 28-29 Oct)

ChatGPT checked the data, both CSVs and the brief (it had no Part 2 code). Each finding was checked here:

| # | Finding | Checked | Verdict and action |
|---|---|---|---|
| G1 | Property **sums insured carry a 1,000x unit error** in both markets | Confirmed: same-policy ratio median exactly 1,000; clean gap (buildings ordinary max 1.06m vs flagged min 52m; contents 0.35m vs 7.6m). Developed 1,395 + 1,402 training rows; emerging 53 + 57 training and 104 + 135 test rows | **Valid; missed in Part 1 and Part 2.** Repaired (÷1,000, raw kept, indicator added) in both markets |
| G2 | Holdout scores rescale to actual totals, so they cannot validate v2's level change over v1 | Correct by construction (v2/v1 is a constant per line) | Valid. v2 is preferred on method, not on evidence; stated as such |
| G3 | New-business shift is ~8% → 77% of exposure; the holdout is renewal-dominated; the "+30-40% in both markets" claim fails for buildings | Confirmed: crude frequency ratios developed / emerging are motor 1.39 / 1.30, buildings 1.05 / 0.94, contents 1.16 / 1.37 | Valid; wording corrected (decision 44). Tested a credibility new-business factor (decision 53) |
| G4 | Row targets (pooled) vs level (peril): motor 2024 410 vs 474 | Same as decision 50 | Valid; already logged. Not refactored before the deadline |
| G5 | Five fires above 250k, not four | Confirmed; the script removed five, the text said four | Wording corrected |
| G6 | Mix logic is sound only if exposure-weighted and applied once | Code implements p_i x level / mu_train (exposure-weighted) | Refactored into `transfer.calibrate_to_level`, with tests: the two formulas agree, model scale cancels, identical portfolios give mix = 1 |
| G7 | Sum-insured indexation may enter twice (severity trend in the level + higher 2025 SI in the mix) | Plausible: the prior used raw SI | **Valid.** SI now in 2023 money in both markets (same-policy growth), so indexation enters once |
| G8 | Developed prior uses developed 2024, so it is not a strict prospective test | Correct | Tested: prior from developed years before 2024 still far ahead of emerging-only (decision 54) |
| G9 | Credibility benchmark uses the emerging median, so it is correlated with the emerging mean; s_h from 15 perils is fragile | Correct | Accepted as a stated limitation; no further tuning (to avoid fitting the leaderboard) |
| G10 | `expenses` / `reference_policies`: no mandatory deliverable established; comparison-site commission missing for emerging | Correct | Not used in the CSV; held until instructions are confirmed |

| 52 | 29 Oct | **Repair sums insured (÷1,000) and put them in 2023 money** in both markets (G1, G7) | Holdout D² (all / new-business-weighted): buildings 0.1102 → 0.1121 / 0.0914 → 0.0919; contents 0.0965 → 0.0970 / 0.1249 → 0.1250; no line worse | `validate_review.py` | Part 1 had the same error; not rerun |
| 53 | 29 Oct | **Reject** a credibility new-business factor (classical credibility, 1,082 claims) | Estimated on 2023: motor 1.13 (Z_freq 0.54), buildings 1.01, contents 1.00. On 2024 it worsened motor (all 0.0830 → 0.0827; new-business-weighted 0.0873 → 0.0857); property unchanged. Conditional on the prior's factors, emerging new-business A/E is within noise and flips sign between years for motor | `validate_review.py`; the full-data factor is reported in the run, not applied | The 77% new-business 2025 book remains the largest uncertainty |
| 54 | 29 Oct | Transfer conclusion survives a strict time test and a paired bootstrap (G8, brief weakness 1) | Prior from developed < 2024: 0.105 / 0.097 / 0.081 (buildings / contents / motor) vs emerging-only 0.064 / 0.054 / 0.072. Bootstrap by policy, 95% interval of gain over emerging-only: buildings +0.013 to +0.097, contents +0.016 to +0.075, motor −0.0002 to +0.021 | `validate_review.py` | Motor's margin is marginal, consistent with keeping 25% emerging |
| 55 | 29 Oct | Submit `part2_submission03_si_fix.csv` (**preferred**): v2 + sums insured repaired and in 2023 money | Decision 52; mix factors fall from 1.076 to 1.053 (buildings) and 1.043 to 1.021 (contents), confirming ~2% of indexation was counted twice (G7); motor identical to v2, which also confirms the refactored calibration | Rates 586 / 480 / 95.4 (motor / buildings / contents); 209,063 rows; within-line correlation with v2 0.995 (buildings), 0.997 (contents) | Full-data new-business factors (reported, not applied): motor 1.079, buildings 0.962, contents 1.070. Applying them would move totals by about +6% / −3% / +5%; quoted as the new-business uncertainty band |

## Final selection and final review (30 Oct)

| # | Date | Decision | Why | Evidence / check | Revisit if |
|---|---|---|---|---|---|
| 56 | 30 Oct | **Part 2 final selection: submission 3** (`part2_submission03_si_fix.csv`), selected on Kaggle by Tracy Rotich | Best-evidenced file (decision 55); also the best of three on the public board | Selection confirmed on Kaggle. Public D²: submission 1 0.15217, submission 2 0.15310, submission 3 0.15361 | — |
| 57 | 30 Oct | **Final review: disclose, do not rebuild.** Tracy Rotich reviewed the work with ChatGPT (`docs/final_review_tracy_rotich_with_chatgpt.md`, `docs/final_review_chatgpt_followup.md`); Claude checked each point in the code and data (`docs/final_review_claude_checks.md`) | Points were about validation scope, IBNR cost completeness, the IBNER stress, fairness wording and attribution. None justified changing predictions hours before the deadline without a holdout to test the change | Late-reported 2021 motor BI claims cost about 26% more than those reported within 30 days (n = 304), so count-based IBNR understates recent BI-heavy rows: disclosed. IBNER stress scaled full incurred: fixed in `target.claim_value` and `uncertainty_part2.py` to outstanding only (motor +3.8% → +3.0%); predictions unaffected because the factor is 1; test updated. Part 1 rates equal the level to the cent (no Part 1 mix factor): disclosed. Fairness and attribution wording corrected in the write-up | Post-hackathon: severity-aware reporting pattern, hierarchical transfer, segment fairness tests |
| 58 | 31 Oct | **Final approval of both submissions** by Tracy Rotich | Her review before submission: the code, technical decisions and supporting evidence, her own coding tests, and the final results | Both final selections confirmed: Part 1 v5, Part 2 submission 3 | — |

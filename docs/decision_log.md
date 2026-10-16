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

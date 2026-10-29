# Response to the ChatGPT review of Part 2 (29 Oct 2026)

The code is now attached as `iaa-hackathon-part2-code.zip` (code and documentation only; no competition data or predictions). Each finding was checked against the data or code before acting. The full table is in `docs/decision_log.md` (G1-G10, decisions 52-55).

## Accepted and fixed
- **G1, sum-insured unit error (1,000x).** Confirmed in both markets: same-policy ratio median exactly 1,000; clean gaps (buildings 1.06m → 52m, contents 0.35m → 7.6m). Repaired in `transfer.repair_sum_insured` (÷1,000; raw value and indicator kept). Missed in Part 1 too.
- **G7, indexation counted twice.** Sums insured are now in 2023 money in both markets (same-policy growth: 3% buildings, 2% contents in both), so 2025 indexation enters once, through the severity trend. Mix factors fell from 1.076 to 1.053 (buildings) and 1.043 to 1.021 (contents).
- **G6, mix logic.** Moved to `transfer.calibrate_to_level`: prediction = p × level / mu_train, with mu_train exposure-weighted. Tests check that the two formulas agree, that model scale cancels, and that identical portfolios give mix = 1. Motor (no sum insured) reproduces submission 2 exactly through the refactored path.
- **G5, fire count.** There are five fires above 250k net; the text is corrected (the script always removed five).
- **G3, wording.** "+30-40% in both markets" applied to motor only. Corrected with line-level ratios.

## Tested
Holdout: train emerging 2023, score 2024. D² is rescaled to the actual total.

| Variant | Buildings: all / NB-weighted | Contents: all / NB-weighted | Motor (with 25% emerging GBM): all / NB-weighted |
|---|---|---|---|
| As submitted (v1/v2 relativities) | 0.1102 / 0.0914 | 0.0965 / 0.1249 | 0.0847 / 0.0893 |
| Sum-insured fix | **0.1121 / 0.0919** | **0.0970 / 0.1250** | 0.0847 / 0.0893 |
| Sum-insured fix + credibility new-business factor | 0.1120 / 0.0915 | 0.0970 / 0.1251 | 0.0844 / 0.0876 |
| Prior from developed years before 2024 only | 0.1052 / 0.0871 | 0.0971 / 0.1264 | 0.0813 / 0.0823 (prior alone) |
| Emerging-only GBM | 0.0644 / 0.0424 | 0.0535 / 0.0889 | 0.0722 / 0.0796 |

"NB-weighted" means rows reweighted so new business is 77% of exposure, as in 2025.

- **New-business factor rejected.** A classical-credibility factor estimated on 2023 (motor 1.13, Z_freq 0.54) worsened motor in 2024. Conditional on the prior, emerging new-business A/E is within noise and changes sign between years for motor. Full-data factors are reported, not applied: motor 1.079, buildings 0.962, contents 1.070. Applying them would move totals by about +6% / −3% / +5%; this is quoted as the new-business uncertainty band.
- **G8, strict time test.** With the prior fitted only on developed years before 2024, it still beats emerging-only models clearly.
- **Paired bootstrap by policy** (300 draws), 95% interval for the D² gain of the prior over emerging-only: buildings +0.013 to +0.097, contents +0.016 to +0.075, motor −0.0002 to +0.021. Motor's margin is marginal, so 25% of the emerging model is kept.

## Kept as stated limitations (not changed before the deadline)
- **G2:** v2's credibility level cannot be validated by rescaled scores. It is preferred on method, not evidence.
- **G4:** pooled row targets vs peril-level level (motor 2024: 410 vs 474). The level is peril-based; the relativity targets are not.
- **G9:** the credibility benchmark is correlated with the emerging sample; s_h comes from 15 perils.
- **G10:** `expenses` / `reference_policies` are unused until instructions are confirmed; the comparison-site commission is missing for emerging.

## Submission 3 (preferred)
`part2_submission03_si_fix.csv`: rates per exposure year motor 586.0, buildings 480.0, contents 95.4. 209,063 rows, sample order, all predictions positive.

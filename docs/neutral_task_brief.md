# IAA AI-Assisted Hackathon 2026, Part 1: neutral task brief

Prepared 23 Oct 2026 so the same task could be given to another AI model (Codex) and the results compared. It deliberately did not describe Claude's approach or results.

## 1. The task (as given by the organisers)

- **Competition:** IAA AI Assisted Hackathon 2026 – Part 1, Kaggle (`kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1`). Sponsor: International Actuarial Association.
- **Role:** pricing team of an established insurer in a developed market. Three lines: private motor, home buildings, home contents. Synthetic data, policy years 2021–2024.
- **Predict:** for each 2025 test row, the **expected ultimate claim cost**: the final cost of all claims with a loss date in the row's period, net of the policy excess and net of recoveries, excluding declined (repudiated) claims. This is the risk premium, not the premium charged. Predictions must be ≥ 0.
- **Submission file:** one CSV covering test_motor, test_buildings and test_contents, header `row_id,prediction`, 626,639 rows (format as in `sample_submission`).
- **Metric:** Tweedie D² (power 1.5) against a constant-rate baseline; higher is better. The public leaderboard uses about 33% of the test data and the final standing uses the other 67%.
- **Week 2 (Part 2):** a new dataset from an expansion (emerging) market; show what changed, how the approach adapts, and the remaining uncertainty.
- **Judging after both parts:**

| Weight | Criterion |
|---|---|
| 40% | Pricing performance and adaptation (leaderboards for both parts, modelling choices, data quality and uncertainty) |
| 25% | Effective use of AI across the workflow, with rigorous verification of AI outputs |
| 20% | Validation, automated tests, governance, reproducibility, human review |
| 15% | Video (first 5 minutes graded), write-up (first 5 pages graded), version-controlled repository |

- **Rules of note:**
  - Teams of up to 3, the same team for both parts.
  - 10 submissions a day; one Final Submission is selected.
  - Competition data may be processed with AI tools, but must not be shared with non-participants.
  - No private code sharing outside the team.
  - External data and tools are allowed if public or low-cost.
  - Winners must provide reproducible training and inference code.

### Data description (from the Kaggle Data tab)

| File | One row per | Content |
|---|---|---|
| customers | customer | Demographics, credit band, prior claims, licence points, marketing fields |
| addresses | address | Region, urban density, flood zone, river distance, elevation, soil, property characteristics |
| weather | address and year | Hail days, wind exposure, freezing days, dry summer index, wildfire index, rainfall, temperature |
| policies_motor / policies_buildings / policies_contents | policy period | Training data 2021–2024, with line-specific rating fields, excesses, channel, premium, renewal_decision (training only) |
| claims | claim | All reported claims on training policies, valued 31 Dec 2024: peril, loss and report dates, status (closed / open / repudiated), paid_to_date (net of excess), incurred_to_date, recoveries |
| test_motor / test_buildings / test_contents | policy period | 2025 rows to predict |
| sample_submission | test row | Format |

Linking: `customer_id` links policies to customers; `address_id` links policies to addresses and weather; `policy_id` links claims to policies (constant across years, so match each claim to the row in force on its loss date); `row_id` identifies test rows. A policy can have several rows in a year (mid-term changes), and rows end early on cancellation or lapse. `exposure` is the row length in years. Some 2024 claims were not yet reported at valuation, and open claims may still develop. Identifiers are text.

---


## 4. Suggested prompt for the other model

> You are assisting an actuary in the IAA AI-Assisted Hackathon 2026, Part 1. Using the task in section 1 and the competition data files, build a reproducible pipeline that predicts expected ultimate claim cost per 2025 row. Explain how you construct the target from the claims table, how you validate, which data-quality issues you find, and which rating factors you exclude and why. Report holdout performance, uncertainty, and every point where AI output needed checking.

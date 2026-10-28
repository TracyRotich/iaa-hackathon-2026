# Independent review brief: IAA AI-Assisted Hackathon 2026, Part 2 (emerging market)

Prepared 28 Oct 2026 for review by a second AI model (ChatGPT). Please review critically and assume nothing here is correct until checked. Tracy Rotich owns all decisions. Claude built the pipeline. Codex reviewed Part 1, and that review caught two real bugs.

## 1. The task
- Predict expected ultimate claim cost for every 2025 policy row in an **emerging market**: private motor, home buildings, home contents. The target is net of excess and recoveries and excludes declined claims; it is the risk premium, not the office premium.
- **Metric:** Tweedie D² with power 1.5. It is **not** rescaled to actual totals, so the level matters as much as the ranking.
- **Submission:** one CSV with `row_id,prediction`, 209,063 rows. Part 1 and Part 2 private scores count equally inside the 40% performance criterion.
- **Brief:** "There is less data, and the customers, climate, regulation and costs are different. We want to see how your team identifies what has changed, decides what to carry over from Part 1, and communicates any remaining uncertainty."
- **Part 1 result:** private 0.13293, rank 34. Part 2 public leaderboard leader is about 0.173 (33% of rows).

## 2. Data facts (from our EDA; please challenge)

| Item | Developed (Part 1) | Emerging (Part 2) |
|---|---|---|
| History | 2021-24, ~2.4m rows | 2023-24 only, ~109k rows |
| 2025 test | 15% new business | **76% new business** (broker portfolio taken on); test exposure is 2x training; younger drivers |
| Motor frequency / severity | 0.141 / 3,883 | 0.182 / 2,250 |
| Property claims (2 years) | ~25k a line | ~1,000 a line; buildings fire = 70 claims, 44% of buildings cost |
| Features | driver/policyholder age, credit band, telematics, mileage | age **bands**, gender; no credit, telematics or mileage |
| Data traps found | exposure on days or 360-day basis; 604 duplicate claims; one 11.9m claim | same exposure trap (565 rows); occupation has 80 spellings of 40 jobs; ~12% missing on several fields |
| New files | | `expenses.parquet` (loadings by market: fixed 85/policy, variable 15%, broker commission 20%, tax 8%, cat reinsurance 12% of property risk premium, profit 8%) and `reference_policies.parquet` (20 sample 2025 policies). **We have not yet seen instructions for these two files.** |

Reporting delays at observable ages match the developed market. Bodily injury: 88.7% of H1-2023 losses reported within 365 days, against 89.2% developed.

## 3. Method (submission 2, preferred)
**Prediction = relativity × 2025 level × mix factor**, rescaled per line.

1. **Target per row:**
   - claims matched to the row in force on the loss date;
   - repudiated claims dropped; incurred − recoveries;
   - IBNR gross-up using the **developed** line-level reporting pattern;
   - IBNER = 1.
2. **Level per line, built peril by peril** (`scripts/level_emerging.py`):
   - frequency = exposure-weighted 2023-24 mean, grossed up with developed peril completeness, no frequency trend;
   - severity = 2023-24 mean (bodily injury and subsidence: 2023 only), trended at the **developed** 2021-24 severity trends (motor 6.56%, buildings 7.39%, contents 3.24%);
   - then **severity credibility**: benchmark = emerging median × developed mean/median; Z = s_h² / (s_h² + se²), with se = CV/√n and s_h = 0.105 (spread across perils of the emerging/developed tail ratio). Fire Z = 0.29 (−11%), bodily injury Z = 0.70 (+7%), high-volume perils Z ≥ 0.93;
   - resulting 2025 rate at the 2023-24 mix: motor 518.0, buildings 455.7, contents 93.45.
3. **Relativities** (`src/iaa/transfer.py`):
   - a frequency × severity LightGBM trained on **all developed years**, using harmonised features. Ages are banded; region, construction type, occupation and make/model are excluded because they are coded differently or don't help;
   - re-levelled to emerging frequency and severity;
   - motor severity fitted on row cost capped at 25k, then blended 75/25 with an emerging-only frequency × severity GBM.
4. **Mix factor** = mean model rate on 2025 rows / mean model rate on 2023-24 rows: motor 1.131, buildings 1.076, contents 1.043. This mainly reflects new business: frequency is +30-40% in both markets.
5. **Final 2025 rates per exposure year:** motor 586, buildings 490, contents 97.5.

## 4. Evidence for the key choices (holdout: train emerging 2023, score emerging 2024; D² after rescaling to the actual 2024 total)

| Model | Motor | Buildings | Contents |
|---|---|---|---|
| Emerging-only GLM (freq × sev, backfit) | 0.051 | **−0.151** | −0.003 |
| Emerging-only freq × sev GBM | 0.072 | 0.064 | 0.054 |
| Developed prior, re-levelled | 0.083 (25k cap) | **0.110** | **0.097** |
| Prior + 25% emerging GBM | **0.085** | 0.108 | 0.095 |
| Boosting emerging trees from the prior (100-500 trees) | ≤ 0.084 | ≤ 0.110 | ≤ 0.095 |
| Prior × credibility frequency correction (region, payment, NB, density, channel, age band) | ≤ 0.084 | ≤ 0.107 | ≤ 0.096 |

Also rejected:
- make/model in the motor prior: 0.083 → 0.081;
- gender, which is excluded on fairness grounds (ECJ Test-Achats standard) and gave negligible gain.

**Level sensitivity** (% change in rate):
- severity trend ±2pp → ±3%;
- bodily-injury open claims +20% → motor +3.8%;
- removing the four fires above 250k → buildings −15.9% (before credibility).

## 5. Known weaknesses (please probe)
1. **Single holdout year** with ~500 property claims a line. Model choices may be noisy. Is the prior's margin over emerging-only models (0.04 D² on property) too large to be noise?
2. **The level is mostly borrowed assumptions** (developed trends and completeness, two years of experience). The metric is not rescaled, so a level error costs directly.
3. **Review finding R6 is material here:** row targets use a line-level reporting pattern and understate motor 2024 by about 14% against the peril-based level (410 vs 474 per exposure year). The level is peril-based; the relativity targets are not.
4. The **mix factor** trusts the developed model's new-business effect. Emerging holdout A/E by new business (cost-based, noisy) was 0.90 motor, 0.76 buildings and 1.40 contents.
5. **IBNER = 1** while 58% of emerging bodily-injury claims are open.
6. **Credibility s_h** is estimated from 15 perils across lines; Z is sensitive to it.
7. **`expenses` / `reference_policies` are unused.** Is a deliverable expected from them (for example, office premiums per )?

## 6. What we would like from the review
- Check the logic of section 3 for errors or double counting, especially level × mix and the re-levelling of the prior.
- Is the developed prior plausible as the best relativity source, or is the holdout flattering it?
- Is any part of the level biased? Consider trend, completeness, credibility, and the 2023-only bodily-injury base.
- Is the mix adjustment sound? Would you use it, cap it, or estimate the new-business effect differently?
- Suggest at most three changes worth making before the deadline (about 2 days), with how to validate each.
- Say what should be stated as a limitation in the write-up (5 pages) and video (5 minutes).

## 7. Where things are (code repository `iaa-hackathon`, private; data not shared)
- `scripts/prepare_emerging.py`, `scripts/level_emerging.py`, `scripts/run_part2.py`: build the submission
- `scripts/validate_part2.py`, `scripts/validate_transfer.py`, `scripts/validate_correction.py`: holdout evidence
- `scripts/uncertainty_part2.py`: level sensitivity
- `src/iaa/transfer.py` (developed prior), `src/iaa/level.py`, `src/iaa/target.py`, `src/iaa/features.py`
- `docs/decision_log.md` (decisions 40-51 for Part 2), `docs/ai_log.md`
- Submissions: `part2_submission01_developed_prior.csv` (v1, level without credibility) and `part2_submission02_credibility_level.csv` (v2, preferred; ratio to v1 is exactly 1.0245 / 0.9572 / 0.9847 by line)

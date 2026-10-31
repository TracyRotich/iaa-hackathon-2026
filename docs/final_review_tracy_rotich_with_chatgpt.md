> Prepared by Tracy Rotich in discussion with ChatGPT, 30 Oct 2026; included as evidence of human review. ChatGPT did not check its figures against the files (its sections 1 and 8 say so). Claude checked each point in the code and data: `final_review_claude_checks.md`.

# IAA AI-Assisted Actuarial Hackathon 2026
## Independent actuarial review and recommendations

Prepared: 30 October 2026  
Status: Review recovered from the prior discussion; direct source verification pending.

## 1. Scope and evidence standard

This document consolidates the technical recommendations discussed in [Review Project Slides](chatgpt-conversation://6ae4d568-6750-83ed-8cac-b961c4e3c14e). The recovered conversation includes the emerging-market review, governance review and overall methodological assessment. It does **not** expose the original uploaded `ALL_IN_ONE.md` or `00_START_HERE.md`; neither file was available in the current workspace. Consequently this is not an independently reproduced audit or a directly verified review of those two files. Reattach them before treating this document as fully source-grounded.

Evidence labels used throughout:

- **Recorded finding:** the recovered discussion explicitly attributes the statement to project documentation. This confirms what was reported, not the underlying code or numerical result.
- **Unverified assumption:** a modelling assumption or implementation detail requiring source inspection or execution.
- **Proposed alternative:** an independent recommendation whose superiority has not been demonstrated.

The earlier assistant's confidence, qualitative scores and assertions of actuarial sufficiency are opinions, not validation evidence. No new model runs, submission checks or tests have been performed for this document. No competition rules, deadline, leaderboard or production-compliance claims have been independently verified.

### Recoverable source map

| Reference | Recovered discussion | Project-file locator quoted in that discussion |
|---|---|---|
| R1 | Overall methodological assessment, assistant message `9d2039de-413a-4ab7-bcde-d9f704a78e93` | `turn0file1`, lines 175–208: trend sensitivity; 567–585: audit/target discrepancy |
| R2 | Emerging-market review, assistant message `9282f654-af6b-4f49-94bf-b4e430afca2c` | `turn0file1`, lines 326–344: transfer; 355–380: corrections; 418–455: mix and levels |
| R3 | Governance review, assistant message `de89b373-595c-4b06-8998-1b1498873c7f` | `turn0file1`, lines 487–505: controls; 525–545: fairness; 545–566: attribution; 567–584: A1–A6 |

The cached file identifier cannot currently be mapped reliably to either named Markdown file. These are retrieval leads, not fresh file citations. `00_START_HERE.md` has not been inspected; no claim about its contents is made here.

## 2. Overall assessment

Retain the overall architecture unless a material error is demonstrated: estimate the portfolio cost level actuarially, then allocate it using statistical risk relativities. The discussion records Poisson/Gamma GLMs, LightGBM, chronological holdouts, developed-market transfer and independent reviews. These are sensible components of a hackathon prototype; their presence does not establish correct implementation or production readiness.

The highest-value improvements concern claims targets, prospective portfolio calibration and reproducibility. Additional model complexity should follow those checks. Reported evidence supports investigating alternatives, not declaring that hierarchical models, a different reserve method or another blend must perform better.

For line-level projected total T and non-negative policy weights w_i, allocation can be expressed as prediction_i = T × w_i / sum(w). Exposure must be included exactly once in the weights or prediction construction. This guarantees reconciliation to T but does not validate T. A 15% error in T remains a 15% error in aggregate predictions, irrespective of ranking quality.

## 3. Audit findings: status and required action

All statuses below are **recorded findings from R3**, not independently confirmed current statuses.

| ID | Recorded issue/status | Pre-submission action | Verification evidence required |
|---|---|---|---|
| A1 | Part 1 omitted an explicit 2025 mix factor; Part 2 included one. Open/disclose. | Trace how mix enters Part 1. Do not add a second factor automatically. Disclose any remaining omission. | Reconcile historical standardised rate, prospective exposure and model-weighted mix; compare totals with and without adjustment. |
| A2 | Improvement was over-attributed to bug fixes; recorded as corrected. | Check every performance claim against version-specific evidence. | Submission hashes, change log and controlled ablation separating fixes, challengers and blend weights. |
| A3 | Part 2 motor blended capped developed-model and uncapped emerging-model components. Open. | State the different bases and inspect calibration/segment effects. Retain only with defensible evidence. | Component target definitions, normalisation order, tail allocation and matched holdout comparisons. |
| A4 | Part 1 retained sums-insured scaling errors and inconsistent construction labels; Part 1 recorded as closed. | Disclose residual effects. Preserve the historical submitted artefact; version any corrected reproduction separately. | Affected counts, correction rule, before/after diagnostics and confirmation of what is still editable. |
| A5 | Historical row targets used line-level reporting curves; portfolio projections used peril-level curves. Open. | Reconcile both bases and quantify materiality before interpreting validation. | Year × market × line × peril bridge, including exposure, counts, reported costs, adjusted targets and projected totals. |
| A6 | Both final selections were attributed to the user, but Part 2 selection was not evidenced. Open. | Confirm using an actual selection record or reword as “preferred candidate”. | Dated decision, exact filename/hash and selector; do not infer selection from preference. |

R1/R3 report that emerging motor 2024 row targets were approximately 14% below the corresponding peril-based estimate. This is a historical target discrepancy, **not evidence that final 2025 predictions were 14% too low**, because final levels were separately projected. Confirm the denominator and target basis before quoting the percentage externally.

## 4. Essential checks before submission

These are proposed release gates, not assertions that the project has failed them. If the competition artefact is already locked, record the limitation and apply corrections only to a clearly versioned research copy.

### P0 — Identify and reproduce the actual artefacts

The discussion identifies Part 1 `Week1_submission05_review_fixes_blend.csv` as selected and Part 2 `Week2_part2_submission03_si_fix.csv` as preferred. Neither file nor its upload status was inspected here.

1. Freeze the exact code revision, input hashes, configurations and selected output hashes.
2. Run the documented build from a clean environment, starting with the permitted raw inputs.
3. Check exact required columns, policy IDs, row count/order where relevant, duplicate IDs, missing rows, finite predictions and permitted prediction domain.
4. Reconcile each line's exported sum to its selected projected total using documented rounding tolerances.
5. Compare reproduced predictions with the selected artefact; investigate every unexplained difference. Distinguish byte equality from numerical agreement if file ordering or metadata differ.

Acceptance evidence: command/run record, environment lock, test results, reconciliation table and signed artefact manifest. A record that tests exist is insufficient evidence that the selected pipeline passes them.

### P0 — Correct claims and attribution

Resolve A6, verify A2 wording, and disclose A1/A3/A4/A5 if unresolved. The discussion says Claude wrote most code and selected most methods; the user provided inputs/references, directed work, commissioned independent reviews and selected Part 1 following a recommendation. Codex reportedly designed v5 challengers/weights; ChatGPT reportedly identified Part 2 errors G1/G7. Verify these details against logs before publication.

Do not describe AI-authored first-person logs as proof the user personally executed every check. Distinguish suggested, implemented, executed, independently checked and accepted actions.

### P0 — Separate level validation from allocation validation

For each holdout and insurance line, publish two evaluations:

- **Prospective, unscaled:** build predictions with only information available at the validation origin. Report total predicted/observed cost, the exact official metric and meaningful segment calibration.
- **Oracle-normalised diagnostic:** optionally scale to the holdout actual total to isolate relative allocation. Label it explicitly as using the realised total; it cannot support a prospective level-calibration claim.

Reimplement the official metric exactly, including objective/deviance, sample weights, null model and zero handling. Do not assume per-line D² scores can be averaged to reproduce the combined score. Show a line-only baseline alongside within-line models to explain how between-line differences affect the combined metric.

If ultimate holdout losses remain immature, distinguish evaluation against reported costs from evaluation against estimated ultimate costs. Record the valuation date and avoid constructing “actual” ultimate outcomes using information that was unavailable at the validation origin.

### P1 — Reconcile A5 and financial bases

Use one explicit target specification recording: reporting adjustment, open/closed claim treatment, cap, inflation basis, exposure units and treatment of zero-claim rows. Generate row targets and aggregate projections from shared definitions where appropriate, with explicit exceptions when their purposes differ.

Test toy claims with known peril-specific completeness, confirm aggregate/row reconciliation on a common basis, and inspect small or zero completeness denominators. Quantify the impact of line-versus-peril adjustment on training, validation, segment relativities and final totals separately. Correcting training targets can change allocation even if total predictions are re-levelled.

### P1 — Run a compact materiality dashboard

Report base/scenario totals and segment changes for severity trend, reporting completeness, open-claim adequacy, credibility and prospective mix. R1 reports approximately 3% movement in Part 1 predictions for a one-percentage-point severity-trend change; R2/R3 report roughly −3% to +6% line-total uncertainty from new-business assumptions. These are recorded sensitivities, not confidence intervals. Reproduce them before reuse and do not add them mechanically as independent risks.

## 5. Actuarial improvements and concrete tests

### 5.1 IBNR: reporting delay is not the whole cost-development problem

**Recorded approach:** claim-count reporting completeness was applied to cost, with different curve granularity in historical targets and portfolio projections (R1/R3).

**Concern:** using count completeness for cost implicitly assumes the unreported claims' expected severity is adequately represented by the reported claims after conditioning and adjustments. Delay/severity dependence can invalidate that assumption. It is a concern to test, not a proven bias direction.

**Implementation:** investigate peril-specific delay distributions and conditional severity by reporting delay. Handle right truncation: a recent accident cohort cannot yet contain long-delay reports. Use mature cohorts or a correctly specified truncation-adjusted likelihood; do not fit an ordinary delay distribution blindly to observed recent claims. Pool sparse perils and disclose tail extrapolation. Compare count-based and cost-sensitive corrections on a common monetary basis.

**Tests:** withhold later reports from mature cohorts at simulated valuation dates; predict subsequent reported count and cost separately. Check calibration by development age, peril and severity band. Such tests validate delayed reporting only to the extent the data contain the necessary dates; they do not recreate historical case estimates from a single snapshot.

### 5.2 IBNER: make open-claim adequacy visible

**Recorded assumption:** case estimates were treated as adequate, effectively IBNER = 1.0 (R1/R2).

**Implementation:** separate paid, outstanding case reserves and unreported claims. If claim status and case balances exist, apply explicit adequacy scenarios to outstanding case reserves, by material peril where justified. For example, ultimate reported cost = paid + a × case outstanding; keep the IBNR component separate to avoid double counting. If those fields are absent, state that an aggregate stress is a proxy.

**Tests:** verify that a = 1 reproduces the base, closed claims receive no unsupported uplift and no adjustment is counted twice. Choose scenario ranges using available evidence and judgement; arbitrary scenarios are not estimated reserve distributions. Later compare realised development of open claims with initial estimates when successive valuations become available.

**Method choice:** the discussion describes a single valuation snapshot. Conventional paid/incurred Chain Ladder or Bornhuetter–Ferguson benchmarks require suitable development data and, for BF, a defensible prior expectation. Do not fabricate triangles from one snapshot or present their absence as a confirmed methodological error. Investigate these benchmarks if reliable valuation histories become available.

### 5.3 Portfolio level, trends and mix

**Implementation:** construct a frequency–severity bridge from historical exposure/counts/costs to prospective totals, by peril where credible. Isolate reporting, case adequacy, severity trend, frequency trend, mix and credibility. Fit trend alternatives only on admissible history; inspect exposure and benefit/sum-insured changes before interpreting calendar-year effects as inflation.

R2 records borrowed developed-market severity trends of 6.56% motor, 7.39% buildings and 3.24% contents, and no emerging-market frequency trend. These are reported selections, not verified appropriate assumptions for local conditions.

**Tests:** rolling-origin backtests where data permit; matched-horizon predictions using only origin-available information; stability across mature accident periods; low/base/high trend comparisons; reconciliation of sum(predictions) to exposure × selected rate. Sparse history limits trend identifiability and should be disclosed.

For mix adjustment, compare exposure-weighted model rates on historical and prospective books under the same feature and monetary basis. Audit normalisation order to prevent omitting or double counting mix. R2 records that 77% of emerging-market 2025 exposure was new business, making extrapolation diagnostics particularly relevant.

### 5.4 GLM, LightGBM and blending

**Recorded strengths:** Poisson/Gamma GLMs provide an interpretable benchmark; LightGBM supplies nonlinear alternatives; chronological validation reportedly informed selection (R1).

**Implementation suggestions:** verify a Poisson frequency offset/weight structure correctly represents exposure; verify Gamma severity uses strictly positive claim-severity observations and suitable claim-count weighting for aggregated means. Inspect overdispersion, sparse categories, missing/unseen factor levels and prediction stability. Keep preprocessing, target adjustment and tuning within training folds. Document whether model outputs are rates or amounts.

Compare GLM, boosting and a simple baseline on identical holdouts, target definitions and financial bases. Select blend weights using training/validation data only; report sensitivity to weight changes and avoid leaderboard-driven overfitting. If feasible, use time-separated tuning and evaluation.

**Tests:** proportionality to exposure holding risk constant; finite predictions for unseen categories; no holdout-dependent encoders or weather summaries; exact cap boundaries; component and blend totals. For A3, compare a harmonised target basis and a separate attritional/excess-cost model as challengers. Restoring the aggregate level after capping does not guarantee correct large-loss allocation by segment.

R3 reports roughly 4% underpricing for ages 17–21 on tested holdouts associated with the £25,000 motor cap. Reproduce the segment definition, denominator and uncertainty before treating that number as established. Examine large sums insured, new business and other material segments with exposure/claim counts and uncertainty, avoiding conclusions from very small cells.

### 5.5 Emerging-market transfer and credibility

**Recorded approach:** separate transferable developed-market frequency/severity LightGBM models using shared features, followed by local re-levelling, severity credibility and prospective mix adjustment (R2). It was not simply the entire Part 1 v5 blend copied across.

R2 records 2024 oracle-relevelled D² comparisons: transferred versus local-only models of 0.083 versus 0.0722 motor, 0.1102 versus 0.0644 buildings, and 0.0965 versus 0.0535 contents. These pre-correction, single-holdout results support the tested transferred allocation, not independently validated 2025 calibration or universal transfer superiority.

**Concern:** the credibility benchmark reportedly combines the emerging-market median with a developed-market mean/median ratio, with a common hyperparameter estimated from 15 perils. The benchmark partly reuses the experience being adjusted. Its dependence and parameter uncertainty should be acknowledged; this is not automatically invalid credibility.

**Implementation:** document the exact weight formula, effective sample size, benchmark construction and monetary conversion. Compare no shrinkage, complete pooling, current credibility and a hierarchical market/peril model. Estimate all hyperparameters and benchmark quantities within each training split. A hierarchical model could allow market effects and partial pooling, but requires justified structure and convergence diagnostics.

**Tests:** resample at the relevant dependency unit rather than treating correlated claims as independent; re-estimate benchmark and weights within each replicate. Compare unscaled calibration and allocation on common holdouts. Inspect feature support, exposure-weighted mix, missingness and response shifts between markets; distinguish covariate shift from changed risk relationships. Test sensitivity to borrowed delay curves, inflation, exchange/monetary basis and new-business assumptions separately and jointly. One small local holdout cannot establish general robustness.

## 6. Reproducibility and AI-governance checklist

The discussion records 26 tests, 55 decisions and 24 AI-log entries (R3). Treat these as inventory claims until checked against the selected revision.

- [ ] Raw/derived data inventory, provenance, permissions, schema, valuation date and file hashes captured.
- [ ] Monetary units, sums-insured correction rules, construction-label mapping and inflation base documented.
- [ ] Training/validation periods, available-at-origin rules and prohibited features specified.
- [ ] Dependencies locked; runtime/platform, random seeds and any nondeterminism recorded.
- [ ] A clean run builds selected outputs without undocumented manual edits or hidden local files.
- [ ] Target-basis specification and row/peril/line reconciliation saved.
- [ ] Tests cover the previously reported weather leakage, GLM IBNR inconsistency, sums-insured scaling and double indexation corrections.
- [ ] G1 scaling correction is evidence-based, versioned and idempotent; rerunning cannot divide already-corrected values again.
- [ ] G7 indexation correction uses a consistent monetary basis; genuine differences in insured coverage are not erased as inflation.
- [ ] Official metric matches a trusted reference calculation on hand-checkable fixtures.
- [ ] Unscaled and oracle-normalised holdout results are clearly separated.
- [ ] Final output schema, IDs, numerical domain and line-total reconciliations pass.
- [ ] Selected artefact hashes map to code revision, configuration, test run and any recorded leaderboard score.
- [ ] Each AI-assisted change names the tool/model where known, suggestion, human decision, implementation, verification and residual risk.
- [ ] Rejected review suggestions and rationale retained; review counts are not presented as proof of correctness.
- [ ] Actual human actions distinguished from AI-drafted narratives and AI-executed checks.
- [ ] Sensitive inputs and credentials excluded from public code/log packages; competition data-sharing rules checked directly.
- [ ] Feature exclusions and proxy diagnostics documented. Aggregate female/male prediction ratios near one do not establish fairness; inspect relevant segments, uncertainty and applicable requirements before making such claims.
- [ ] Final selection, report, code package and presentation describe the same version and disclose unresolved findings.

## 7. Post-hackathon research priorities

| Priority | Research question | Candidate work | Evidence needed before replacement |
|---|---|---|---|
| 1 | Can ultimate targets be estimated more reliably? | Delay/severity dependence, open-claim development, later valuation snapshots | Improved subsequent-development calibration; quantified uncertainty and no leakage |
| 2 | Can prospective levels be better calibrated? | Rolling-origin level models, trend/mix decomposition, local economic scenarios | Better unscaled totals and stable segment calibration across periods |
| 3 | Does partial pooling improve transfer? | Hierarchical frequency/severity market-peril models | Common-basis out-of-time improvement, uncertainty calibration and stable parameters |
| 4 | Can large-loss allocation improve? | Attritional/excess decomposition or alternative severity models | Tail and segment improvements without material deterioration elsewhere |
| 5 | Is greater complexity worth maintaining? | Alternative objectives, interactions and blends | Reproducible incremental benefit against simple baselines, with runtime/explainability tradeoffs |

Before experiments, define primary metrics and materiality thresholds in cost/decision terms. Preserve a time-separated evaluation period where feasible. Report failed challengers as well as successful ones. Quantify sampling, parameter, process and assumption uncertainty separately where identifiable; do not label a scenario range as a probabilistic interval.

## 8. Evidence required to finalise this review

Supply `ALL_IN_ONE.md` and `00_START_HERE.md` to replace the indirect source map with verified headings/line references and check all quoted numbers. Code, raw data, configuration, test logs and final artefacts would additionally be required for an execution-based audit. For now, all technical defects and fixes described as recorded remain reported evidence; the proposed implementation steps and tests remain recommendations.

The practical recommendation is to retain the documented architecture, prioritise truthful attribution, exact artefact reproduction, consistent claims bases and prospective level diagnostics, and defer major redesign until a fair comparison demonstrates a benefit.

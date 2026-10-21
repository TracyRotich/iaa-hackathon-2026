# AI use log

Every AI suggestion that affected the work, with the human decision and the check that verified it. This is the evidence for the 25% "effective use of AI, including verification of AI outputs" criterion, and for catching any deliberate traps in the data or brief.

| # | Date | Stage | AI tool | What the AI suggested | Decision (accept / modify / reject) | How I verified it | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | 16 Oct | Setup | Claude | Repo structure, local D² metric, guards as tests, rate-with-exposure-weight approach | Accept | Ran the test suite; checked the exposure-weight equivalence against the compound Poisson-gamma variance | Tests pass on synthetic data |
| 2 | 16 Oct | Target | Claude | Build the target from claims: match on loss_date within row period, drop repudiated, incurred − recoveries, IBNR gross-up from reporting-delay CDF, IBNER = 1.0 with sensitivity | Accept, pending data checks | Unit tests on hand-built cases (mid-term change, cancellation, repudiation, recoveries, IBNR by row age). Still to verify on real data: whether recoveries are already netted in incurred | Tests pass |
| 3 | 16 Oct | Features | Claude | Name-based leakage check flagged `prior_claims_5y` | Modify: allow-listed, because it is declared before inception | Read the column definition on the Data tab | Caught a false positive from the AI's own rule |
| 4 | 21 Oct | Data checks | Claude | Recompute exposure from dates; drop duplicate claims; cap the 11.9m BI claim | Accept | Inspected the rows: days-basis and 360-day patterns; duplicate pairs identical on every field; 11.9m vs paid 0.74m and next largest 1.12m | Exposure was overstated by ~35% |
| 5 | 21 Oct | Features | Claude (via GBM gain) | `previous_insurer` was the top contents feature | Reject the feature | Tabulated frequency by insurer: flat across 37 levels → noise | Holdout D² 0.069 → 0.078 |
| 6 | 21 Oct | Guardrails | Claude | Leakage name check flagged `marital_status` (contains "status") | Modify: allow-listed; added a test | Read the column definition | Second false positive from the AI's own rule |
| 7 | 21 Oct | Level | Claude | Severity trend first fitted on 2021–23; contents level came out below the 2024 rate | Modify: refit on 2021–24 short-tail perils | Compared projected level with history by year | Contents 117 → 124 |
| 8 | | | | | | | |

**Rule for myself:** no AI-written number, feature or claim goes into the write-up until I have reproduced it or checked it against the data.

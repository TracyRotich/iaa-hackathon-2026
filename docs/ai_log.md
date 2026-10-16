# AI use log

Every AI suggestion that affected the work, with the human decision and the check that verified it. This is the evidence for the 25% "effective use of AI, including verification of AI outputs" criterion, and for catching any deliberate traps in the data or brief.

| # | Date | Stage | AI tool | What the AI suggested | Decision (accept / modify / reject) | How I verified it | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | 16 Oct | Setup | Claude | Repo structure, local D² metric, guards as tests, rate-with-exposure-weight approach | Accept | Ran the test suite; checked the exposure-weight equivalence against the compound Poisson-gamma variance | Tests pass on synthetic data |
| 2 | 16 Oct | Target | Claude | Build the target from claims: match on loss_date within row period, drop repudiated, incurred − recoveries, IBNR gross-up from reporting-delay CDF, IBNER = 1.0 with sensitivity | Accept, pending data checks | Unit tests on hand-built cases (mid-term change, cancellation, repudiation, recoveries, IBNR by row age). Still to verify on real data: whether recoveries are already netted in incurred | Tests pass |
| 3 | 16 Oct | Features | Claude | Name-based leakage check flagged `prior_claims_5y` | Modify: allow-listed, because it is declared before inception | Read the column definition on the Data tab | Caught a false positive from the AI's own rule |
| 4 | | | | | | | |

**Rule for myself:** no AI-written number, feature or claim goes into the write-up until I have reproduced it or checked it against the data.

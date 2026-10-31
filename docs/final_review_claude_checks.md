# Final review: Claude's checks of the review points (30 Oct 2026)

Tracy Rotich reviewed the work with ChatGPT (`final_review_tracy_rotich_with_chatgpt.md`). Claude checked each point against the code and data before anything changed. The rule was: change wording and disclosures tonight; change no prediction without a holdout test.

| Review point | Check | Result | Action |
|---|---|---|---|
| Submitted files must be exactly reproducible and valid | Row counts, IDs against the sample, order, finite and non-negative values | Both files pass: 626,639 and 209,063 rows, IDs unique and in sample order. Part 1 file (built by Codex) is reproduced by our pipeline within a median 0.07% on motor (property identical); a clean-environment rebuild was not repeated | Disclosed in the write-up; SHA-256 in `final_submission_sha256.txt` |
| A1: Part 1 has no explicit 2025 mix factor | Rate per exposure year in the Part 1 file | Equals the level to the cent: 662.78 / 528.10 / 123.88. Part 2: 586.01 / 479.97 / 95.41 | Disclosed as a Part 1 limitation |
| Holdout D² is rescaled to the actual total | Validation code | Correct: holdouts measure allocation between policies, not the level. A forward test of the level is not possible from one claims snapshot (claims as at end-2023 are not observable) | Labelled in the write-up; stated as a limitation |
| Count-based IBNR may understate cost if late claims are larger | Mean cost by report delay, 2021 losses | Motor bodily injury reported after more than a year costs about 26% more than claims reported within 30 days (304 late claims). No consistent pattern in other perils, which have few late claims | Disclosed. Mainly affects recent row targets; the level uses the most mature BI severity, which reduces but does not remove the effect |
| IBNER should apply to outstanding, not incurred | `target.claim_value`, `scripts/uncertainty_part2.py` | **Error confirmed**: both scaled the whole incurred amount, including paid. Predictions unaffected (factor 1) | Fixed to outstanding only; test updated; motor stress +3.8% → +3.0% |
| Sum-insured repair must not apply twice | Repair rule | Idempotent: only values above 1e7 (buildings) / 1e6 (contents) are divided, and repaired values fall below the threshold | None |
| Aggregate gender ratios near 1 do not prove fairness | Write-up and video wording | Agreed: the check shows the model does not rebuild gender in aggregate; segments were not tested | Wording softened in the write-up and video script |
| Attribution: separate suggested, implemented, checked, accepted | AI log voice | Entries were drafted by Claude in the first person | Attribution note added to `ai_log.md`; "Who did what" in the write-up |

ChatGPT's follow-up (`final_review_chatgpt_followup.md`) accepted these checks and made two further corrections, both adopted: the level is not "largely protected" from the late-claim effect, only less exposed; and the write-up should say Tracy Rotich led the final review rather than independently verified all code.

Not done tonight, by design: hierarchical transfer models, severity-aware reporting patterns, segment fairness tests, and a clean-environment rebuild. These are post-hackathon work.

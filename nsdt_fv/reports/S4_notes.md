## What the tables say (hand-written; every number it relies on is in the generated sections above)

- **Answer to the S4 question: yes on ranking, by a small margin; mixed on calibration.** On the primary split the
  Appendix E fragment ranks hidden true pairs better than every baseline on every ranking metric, and the paired
  intervals against the best baseline (shape frequency, on every metric) exclude zero: AUC 0.887 vs 0.847 (+0.040
  [0.032, 0.048]), P@5 0.880 vs 0.852, P@10 0.823 vs 0.798, P@20 0.732 vs 0.716 (+0.016 to +0.028). The two
  replicate splits give the same ordering and nearly the same numbers. Nearest neighbour is below frequency on every
  metric; random order sits at AUC 0.51 as it should.
- **The margin is small.** The relative gain in precision at k over shape frequency is 2-3% (gate section), an order of
  magnitude below the 20% that G4 asks of C6 on the development repositories. Most of PLN's ranking power on the toy is
  the shape prior sT it shares with the frequency baseline; transfer and deduction add a consistent but modest
  increment. This is a statement about this toy table, not a prediction for C6.
- **Calibration is mixed.** PLN has the lower Brier score (0.124 vs 0.139, interval excluding zero) but the higher
  expected calibration error (0.077 vs 0.054, PLN worse, interval excluding zero). The reliability table shows where:
  PLN overestimates in the 0.1-0.5 range (e.g. predicted 0.25, accepted 0.10; predicted 0.45, accepted 0.32), is close
  to the diagonal between 0.5 and 0.8, and is underconfident at the top (predicted 0.84, accepted 0.94). It separates
  the candidates better (lower Brier) while its probabilities in the middle range are too high. Part of the low-range
  overestimate is structural (ISSUES 45): the candidates are what remains after 80% of the true cells were made
  visible, so their base rate (2,860 of 9,681, 0.30) is lower than the table's (0.68), and both PLN and frequency
  estimate from the visible table.
- **The gain is not an artefact of cross-project twins.** Functions with the same template definition in several
  projects have near-identical rows (ISSUES 43), which could favour the rankers that read other rows. The breakdown
  shows the PLN-minus-frequency AUC gain is of similar size on shared template functions (0.842 vs 0.805), on template
  functions present in one project (0.917 vs 0.888) and on project-specific wrappers (0.908 vs 0.862).
- **No leakage, labels from the checker only.** The leak checks pass on all three splits (no premise PLN read is
  hidden or outside the visible set; no hidden statement has a visible cell). Every positive label is a proof the
  checker accepted; 0 prover proofs were rejected by the checker, and 0 accepted statements have a counterexample
  (no checker soundness problem observed). 2,254 statements have neither a proof within budget nor a
  counterexample; they count as negatives (ISSUES 44).

## Anomalies and open issues

- The first full run aborted: labelling workers died with a segmentation fault (C-stack overflow in the ground
  evaluator on deep unary recursion, e.g. `tri(tri(tri(n)))`) and the process pool hung. Fixed by running each job in a
  1 GiB-stack thread (commit 9207192); aborted logs in runs/S4/logs/aborted/. The aborted run (about 8 minutes on the
  same 2 cores) is not in the cost section. 3 statements hit the evaluator limit (descriptive column only).
- Re-running the ranking stage (`--stage rank`, commit 9012614) reproduced the hidden sets and every score line of the
  three splits exactly (decompressed sha256 in runs/S4/logs/rank_rerun_check.txt); leak.json differs only in its timing
  field. reports/S4.md regenerates byte-identically from runs/S4 (checked twice, same sha256).
- Readings the plan leaves open are in CHANGES.md (S4) and ISSUES 42-48: shapes from channel C's output, the statement
  group as the unit of hiding, the positive-unlabelled candidate set, Implies by direct evaluation over the table, the
  overlap-discount reading and the fixed revision order. The nearest-neighbour baseline is a 1-NN vote; its scores are
  almost all 0 or 1, which explains its poor Brier and ECE.
- One stream seed (the S1-S3 toy); three split seeds. No tuning: configs/S4.yaml and the code were fixed before the
  first ranking result.

<!-- next -->
## Recommended next step

- S4's acceptance items are met (T11 passes; G0's S4 part is T11). Proceed to S5 as the ladder orders. For C6, carry
  the fragment unchanged and evaluate it with the store's refutations and test evidence available to every ranker (the
  toy run withholds them by design), and report calibration on the full candidate population as well as on the hidden
  set; do not tune the fragment on the toy, where its margin over frequency is 2-3% relative.

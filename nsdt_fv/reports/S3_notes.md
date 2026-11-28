## Gate evaluation

S3 carries no gate of its own (plan 8.1, A.7); its acceptance conditions are those of the order, all met in the checks above:
five channels run with per-channel yield tables, T10 passed, the guard raised 0 times in training and every canary
(including one per channel on the channel's own extraction path) raised, T4, T8 and T9 passed, and D1 re-run through
the S3 driver reproduces S2's D1 line for line. For orientation only, C5's retention rule (a gain of at least 2 points
on FWD or ACC with a 90% interval excluding zero, or +25% non-trivial dream yield at no loss) is met by no channel here:
no N arm gains on any metric and every N arm's non-trivial dream yield is 0, as D1's.

## What the tables say (hand-written; every number it relies on is in the generated sections above)

- **The answer to the S3 question is a null effect, and it comes from the learner, not from the channels.** On ToyEq
  with DT-ref as the learner, no channel changed what D1 learns in a way the evaluation can see. N-B, N-C and N-D
  equal D1 on LA, ACC and F (paired intervals [0.0, 0.0]); N-E differs by a tenth of a point; N-A is slightly worse
  (dLA -1.6, dACC -0.8, intervals reaching 0). No arm solves a non-trivial test obligation, as D1 does not.
- **The channels themselves produce plenty of valid, non-trivial material.** B emitted 1,200 statements, every one with
  a machine proof the checker accepts (T10) and none closed by the baseline tactics. C emitted 1,200 conjectures that
  survived 400 random tests; the reference prover proves 869 of them, all non-trivial. A emitted 712 statements (690
  accepted, 582 non-trivial) and about 2,700 checker-confirmed auxiliary examples; D produced 546 accepted repair
  triples; E induced methods in every stage, 596 plan examples and 83 accepted method instantiations.
- **Why the material has no effect: the learner cannot prove any of it.** Conjecture sleep attempted 480 channel
  statements in each of N-A, N-B and N-C (its full budget of 40 per stage). The model proved none of B's or C's
  (0 of 1,920 candidates each) and, of A's, only trivial ones (91 accepted candidates, none for a non-trivial
  statement). With the kernel filter nothing unproved reaches the learner, so N-B and N-C train exactly as D1 and
  their evaluation logs coincide with D1's. This is S2's finding seen from the other side: the DT-ref cap on the frozen
  toy base proves only what the baseline tactics close, so material that is valid and new but not trivial is unusable
  to it.
- **Auxiliary wake tasks change training but not competence.** A, D and E add 2,686, 546 and 596 wake examples (more
  optimizer steps, Cost table) without a measurable gain; A's NEXTSTATE/PREMISE/UNFOLD examples coincide with a small
  loss in LA, which is consistent with interference from a different output format in a single shared epoch, but the
  interval reaches 0 and there is one seed.
- **Re-derivation priorities (D, E)** redirected 316 and 66 draws to the blast radius and to method-matching statements
  without effect, because those statements are as unprovable to the cap as the rest of the non-trivial pool.
- **Guard.** It refused channel items that would have leaked: A goal states and B compositions and C conjectures equal
  to a test statement (L3), and B and C items whose text contains a distinctive test proof or test proof state (L1).
  These are real near-leaks that the channels generate on their own (composition and enumeration reach test
  statements), which is the case the order's "useless if" names; every one was refused before use.
- **Counterexamples.** Testing on 200 inputs matters: classes that agreed with another on the first 5 inputs and were
  separated later number 3,320, on the first 20 inputs 238. The second, independent 200-input test refuted 9
  candidates that had passed the first (stages 4 and 10); they are recorded and never emitted. No emitted seed has a
  recorded counterexample (T10).

## Anomalies and open issues

- The first N-A run crashed at stage 4: a channel-A goal-state statement was longer than the model's 192-token context.
  All N arms were re-run on commit 6ea08b7, which skips channel statements whose prompt leaves less than conj_max_new
  tokens (the rule D1 already applies to its own conjecture prompts; 'too long' column). The aborted logs are kept in
  runs/S3/logs/aborted/. D1 (S3) ran on f3e4c66; the fix does not touch its code path (it has no channel), and its
  logs equal S2's.
- C's enumeration budget (4,000 terms per stage) stops at size 6 in eight of the twelve stages (C.5 says up to size 7).
- One seed; ISSUES.md items 33-41 list the S3 readings and limitations (D's machine-proof repair targets, unused
  annotations, budget accounting, common random numbers, arm-dependent duplicate counts) and the two items carried
  over from S2's review (the D1x/B2c confound and the 20,160 vs 17,280 slip).

<!-- next -->
## Recommended next step

- Do not tune the channels on the toy: their output is valid and plentiful, and the bottleneck is the learner. Carry
  channels A-E unchanged into Stage C (experiment C5), where the base is a pretrained prover and the question whether
  channel material changes what the learner learns can have a non-null answer.
- S4 (PLN) can take its property tables from channel C: its surviving equations per function and shape are exactly
  the function/property pairs S4 ranks.
- If a toy answer is still wanted before Stage C, the cheapest informative probe is supervised, not A.4: train the cap
  directly on B's statements with their machine proofs (an X5-style ablation) and see whether non-trivial test
  obligations move. That is a new task for the chief of staff, not part of S3.

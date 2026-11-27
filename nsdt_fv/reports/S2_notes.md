## What the tables say (hand-written; every number it relies on is in the generated sections above)

- **The answer to the S2 question is negative on this toy, and it is not informative about DT.** DT-ref as A.4
  states it (one wake epoch, re-derivation and conjecture sleep on a frozen base) does not learn the new windows:
  its accuracy matrix is the frozen base's on the trivial obligations, it solves no non-trivial test obligation at
  the end of the stream, and its LA is below B2's (the D1 - B2 dLA interval excludes 0 at seed 0). Its forgetting is
  0 and its ACC is slightly above B2's and B0's, but a learner that learns nothing cannot forget; F = 0 here is not
  evidence that sleep protects anything. D1 is "no better than B2" on learning new windows and "better than B2" on
  forgetting only in this vacuous sense.
- **The like-for-like baseline B2c learns no more than D1, but it forgets.** Sequential fine-tuning of the same
  cap with S1's 30-epoch recipe has the same LA as D1 within noise and solves no non-trivial obligation at the end,
  yet its F is far above D1's and its ACC below the frozen base's: it loses obligations the base could close. So
  at equal (near-zero) learning, D1 keeps what the base could do and B2c does not. This is the one contrast in S2
  that is not vacuous, and it is narrow: it concerns retention of trivial competence, not of learned windows.
  S1's phenomena (B2 learning windows and forgetting them) came from training all weights; on a frozen base
  pretrained only on the prelude, neither DT-ref nor ordinary cap training reaches them within the S2 budget
  (CHANGES.md probes: even 30 epochs over the whole stream fit 44% of the training proofs).
- **What re-derivation sleep did.** It re-proved only statements the baseline tactics already close, and learning
  those proofs pushed the trivial stratum to 100% (B0 misses a few trivial obligations at temperature 1.0). That
  is the whole of D1's ACC advantage over B0.
- **Conjecture sleep was inert.** No generated statement parsed in any stage: one epoch of the auxiliary
  statement task did not teach the cap to write statements (ISSUES.md item 27). D1 and D3 therefore differ only
  through sampling noise, and nothing is learned about conjecture sleep.
- **Negative phase.** Pairs existed (an accepted and a rejected proof of the same statement) and were used, but
  with nothing non-trivial being learned D1 and D4 coincide; nothing is learned about the negative phase.
- **Kernel filter (D1 vs D2).** D2's dreams are unfiltered: most of what D2 trained on were wrong proofs (counts
  above), against none for D1. The measured cost is small (a few trivial obligations lost; dACC interval touches
  0) because neither arm learns anything non-trivial. The filter's value cannot be assessed when the learner does
  not learn.
- **Diagnostic D1x** (not A.4, added after D1-D4 were seen) separates the two explanations of the D1-B2c
  contrast: D1's wake is a much smaller update (one epoch against thirty), and D1 also sleeps. D1x has B2c's wake
  plus D1's sleep. It learns no more than B2c (dLA interval includes 0) but keeps the trivial obligations as D1
  does (dF and dACC against B2c exclude 0, D1x - D1 is 0 on ACC and F). With the same wake update as B2c, adding
  D1's sleep removes B2c's forgetting, so on this toy the retention goes with the sleep phases (in practice
  re-derivation, which re-proves and relearns trivial statements; conjecture sleep produced nothing), not with the
  smaller wake. Because no cap arm learned a non-trivial window, this shows only that re-derivation sleep protects
  competence the base already had; whether it protects newly learned proofs is untested. One seed.

## Anomalies and open issues

- The cap is not A.4's: A.4's adapters on the top third underfit completely, so the cap was sized up to
  attention-block adapters after every layer (CHANGES.md "S2 calibration", ISSUES.md item 26). D1-D4 are
  therefore DT-ref with a larger cap; the conclusion above holds for it, and a fortiori for A.4's smaller cap.
- Conjecture sleep never produced a statement (item 27); "dream yield" is re-derivation yield only.
- FWD is measured without snapshot context, as in S1 (item 19). For the D arms FWD equals LA within noise
  because the model does not change on non-trivial obligations.
- GC uses S1's full-parameter B2 and B6 (item 29); a GC near 0.15 for the D arms reflects only the trivial stratum.
- Seed-1 rows use S1's seed-1 base; the D-arm matrices are almost identical across seeds because both bases solve
  exactly the trivial obligations and nothing else (item 32).
- Runs executed in parallel on disjoint core pairs on a shared host (other load present); wall times are therefore
  not comparable with S1's single-run times.
- D1x was run from a later commit than D1-D4 and B2c; the later commits add only the D1x arm and documents.
- The verdict cache is in memory per run (R5 holds within a run); no verdict was recomputed with another timeout.

## How the numbers were produced (for checking)

- `./run_s2.sh`: pytest, then `python3 -m nsdt_fv.eval.s2 --config nsdt_fv/configs/S2.yaml --arm <arm> --seed <s>`
  per run on two pinned cores, then `python3 -m nsdt_fv.eval.report_s2` twice with a byte comparison. Raw logs per run in
  `nsdt_fv/runs/S2/<arm>/` and `nsdt_fv/runs/S2_seed1/<arm>/`: `verdicts.jsonl.gz` (evaluation, as S1),
  `dreams.jsonl.gz` (one line per dream candidate: verdict id, statement, candidate, guard result, kernel verdict,
  trivial flag), `conjectures.jsonl.gz` (every statement generation and what became of it),
  `train_examples.jsonl.gz` (one line per training example: phase, kind, source, source ids, verdict id, negative's
  verdict id), `train.jsonl` (per stage: wake, sleep statistics, base hash, cap hash, guard counts), `events.jsonl`,
  `evalsets.json`, `canary/`, `meta.json`. S1 rows are recomputed from `nsdt_fv/runs/S1*/`. Cap checkpoints
  (per stage) and stream data files are written but not committed (.gitignore); the base weights are S1's (hash checked).

<!-- next -->
## Recommended next step

S3 (channels on top of D1) would sit on a learner that learns nothing here, so it should wait for a human decision
on the toy's learner (no change was made, R2):

1. Give the toy a base that is a reasonable frozen substrate, as Stage C's pretrained prover will be: for example,
   pretrain the base on held-out projects from all six templates (not the stream's projects), then rerun B0, B2c and
   D1-D4. This changes the S1 toy and needs approval.
2. Or accept that on the toy DT-ref is only a harness check: S2 as run validates the harness (T4, T8, T9, guard,
   regeneration from logs) and move the DT questions to Stage C.
3. Separately decide whether conjecture sleep should take statements from the channels only (S3) rather than from
   the model, since a prover never trained to write statements produces none.

## How the numbers were produced (for checking)

- Run: `./run_s1.sh` (pytest, then `taskset -c 0,1 python3 -m nsdt_fv.eval.s1 --config nsdt_fv/configs/S1.yaml`,
  then `python3 -m nsdt_fv.eval.report_s1`). Raw logs: `nsdt_fv/runs/S1/`: `<arm>/verdicts.jsonl.gz` (one line per
  sample: run, stage, eval set, obligation, sample index, candidate text and sha, verdict), `<arm>/train.jsonl`,
  `<arm>/events.jsonl` (train/eval order, L4), `<arm>/evalsets.json` (T4 hashes), `canary/`, `test_sets.json`,
  `test_obligations.jsonl`, `meta.json`. Model weights and per-stage checkpoints are written but not committed (hashes in
  `train.jsonl` and `meta.json`).
- Order of work per arm (L4): evaluate at stage 0; for t = 1..12, train on window t, then evaluate E_j for j <= min(t+1, 12).
  The structural check after each stage refuses the stage unless every obligation of every evaluated E_j has exactly
  8 samples with valid verdicts; for B0 the base hash is checked at every stage boundary, and for every arm the stored
  base state is checked to be unmodified.
- Evaluation on training items is excluded three ways: wake and test identifiers are disjoint (checked, 0 overlap);
  every training example passes `HoldoutGuard.check`, which raises on a test identifier, on a wake example whose
  statement equals a test statement (alpha- and orientation-insensitive) and on distinctive test proof text or proof
  states; and two injected leaks (a test obligation offered as a wake example) are confirmed to raise before training.

## Anomalies and open issues

- The recorded code commit of the main run ends in `+dirty`: the flag was taken when the run finished, after files were
  edited while it ran (the `--seed` option of `eval/s1.py` and documentation; diff `git diff 493cc69 f959832`). The run
  imported the modules of 493cc69 at start. From f959832 on the commit is recorded at start (`code_commit`) and at end.
- Forgetting is above the target but not by a wide margin: F(B2) = 11.9 with a 95% interval whose lower end is below 10,
  and the frozen B0, which cannot forget, shows F = 1.0 from sampling noise alone (F takes a maximum over stages).
  The seed-1 replicate below, if present, is the check that the effect is not a single-seed accident.
- B6 beats B0 on ACC by 28 points but on LA by only 12: re-trained from the base on the union, B6 sees the newest window
  for 30 epochs among all earlier ones, and its accuracy on the newest window (LA) is below B5-10's. Its strength is at the
  end of the stream (ACC, BWT = +18), which is what the ceiling is for.

- Arms B2, B5-10 and B6 train all weights of the tiny model, not adapters (ISSUES.md item 18, CHANGES.md "S1 calibration").
- The toy's "curated context" is the statement alone, so FWD here is not the section 7 FWD with snapshot context
  (ISSUES.md item 19). B0's accuracy comes almost entirely from obligations that `simp refl` closes (stratum table).
- B1 is not an S1 arm; the gap is taken against B0 (item 20). The gap is evaluated on ACC, LA is shown for reference (item 21).
- One prelude lemma was excluded from base pretraining by the guard's free-text rule (item 22).
- One training seed; small per-stage test sets (item 24). Intervals cover obligation sampling only.

## Recommended next step

S2 (DT arms D1 to D4 with DT-ref) on this toy and model, after a human decision on ISSUES.md items 18 (which B2
variant GC is measured against when DT trains only a cap) and 19 (toy context for FWD).

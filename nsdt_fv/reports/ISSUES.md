# Issues and interpretations (rule R2)

No protocol element of plan v0.1 (splits, metrics, thresholds, budgets, prompts) was changed in S0.
The points below are places where the plan leaves room, or where a choice made for the toy will
matter for Lean. Each says what was done and what a human may want to decide.

## ToyEq (Appendix C)

1. **IH generalisation (C.2).** "IH equal to the original goal with x replaced by y" does not say whether
   the other variables are fixed or universally quantified. Implemented: the IH is universally quantified
   over the goal's other variables that occur free in no hypothesis, and `induct x` is refused when x
   occurs free in a hypothesis (that IH would need the hypothesis as a premise, which the grammar cannot
   express). With fixed variables, accumulator lemmas such as `qrev(xs, ys) = app(rev(xs), ys)` have no
   proof in the grammar. Soundness of this choice is what T1 tests.
2. **`rw` rewrites one occurrence**, the leftmost-outermost match, as C.2 says (Lean's `rw` rewrites
   every instance of the instantiated pattern). `rw` fails if the right side would keep a pattern
   variable that the match did not fix, and pattern variables match only subterms of their own type.
3. **Definitions (C.1)** are restricted to matching on one argument with flat constructor patterns,
   one equation per constructor in declaration order (or a single all-variable equation for wrappers),
   and calls only to earlier functions or structural self-calls. This is a sufficient condition for
   "exhaustive, non-overlapping, structurally recursive"; two-argument matching needs a helper function.
4. **Fresh names** after `cases`/`induct`: the recursive child keeps the split variable's name
   (two recursive children of Node: `<x>l`, `<x>r`); other arguments are `h` (Cons) or `v` (Node),
   with a numeric suffix on clashes. Models will have to learn this convention.
5. **simp fuel**: running out of the 10,000 rewrites is a step error (rejection).
6. **check_statement context** for ToyEq: `""` means every lemma of the snapshot is available;
   `"before NAME"` means only lemmas that precede NAME. The Lean backend will take a file prefix instead.
7. **Baseline closing tactics (section 7)** for the toy are `refl` and `simp refl`; they set
   `closed_by_baseline_tactics`.
8. **One template seed was not proved**: `sumt_mirror` in toy-trees needs more than six non-simp steps
   for the reference prover, so it is not in the library (C.3: only lemmas with a found proof enter).
9. **The toy libraries are easy**: about a fifth of the lemmas close by `simp refl`, and most proofs follow
   a few shapes. That is S1's calibration question, not S0's.

## Acceptance test T1

10. **Mutating only proofs cannot reveal unsoundness.** A mutated proof of an unchanged, true statement
    can never be "accepted with a statement that fails on an instance". So that T1 has something to
    catch, 500 of the 1,000 mutants also mutate the statement (argument swap, function replacement,
    base constructor, variable swap, wrapping, side swap), 300 further mutants pair a mutated statement
    with the ground-truth proof (reported separately), and a planted checker bug (refl accepting two
    sides with the same head symbol) is run on the same mutants as a canary. The report gives the
    rejection reasons. A proposal to word T1 this way is in CHANGES.md.

## Hold-out guard (R4, L1-L7)

11. **"Text overlap with a test proof".** Toy proofs are short and generic: most test proofs are
    identical to proofs of non-test lemmas (for example `induct xs { simp refl } { simp rw IH refl }`),
    so raising on any textual overlap would reject almost all legitimate training data. Implemented:
    free-text overlap raises only for *distinctive* test proofs (at least 8 tokens and not the proof of
    any non-test lemma; 5 of 240 test proofs in this stream), and distinctive intermediate proof states of
    test proofs raise anywhere. Every proof-derived example (wake, premise, nextstate, diag, repair,
    plan, aux) and every dream must carry the statement it is about; the guard raises when that
    statement is a test statement (alpha- and orientation-insensitive). That rule, not text overlap,
    catches the five leaks of T3. A non-distinctive test proof placed under a different statement is not
    detectable by text and is not a leak of information specific to the test obligation. Lean proofs are
    longer and more specific, so the distinctiveness threshold should be revisited in C1.
12. **"Raises on any test identifier"** is read as obligation identifiers (`<repo>@<commit>:<decl>`) in
    identifier fields or text. Declaration names of test lemmas are *not* treated as leaks: later proofs
    legitimately cite earlier lemmas by name (`rw <lemma>`), as Lean proofs do.
13. **File prefixes (Stage C, needs a decision).** In ToyEq `src_up_to_decl` holds definitions and earlier
    lemma *statements* only. In a Lean repository the file prefix of a later obligation contains the
    proofs of earlier test obligations, so prompts built from `src_up_to_decl` for wake training would
    carry test proofs and violate L1. Stage C needs a rule (for example: strip test proofs from prefixes
    used in training) before C1.
14. **L4-L7 in the toy.** L4 is a check on the order of train and eval events (`check_eval_order`); L5 is
    `split_by_release` (no base release date exists for the toy); L6 treats all toy projects as
    development projects unless a list is given; L7 rejects examples whose source is a frontier model.
    None of these is exercised by data in S0 beyond unit tests.

## Harness

15. **R5 timeouts.** "A cached verdict is never recomputed with a different timeout" is implemented as:
    a lookup with another timeout returns the cached verdict and increments a mismatch counter; writing
    a second verdict for a cached key raises.
16. **Benchmark field names (5.3).** They must be read from `data/verisoftbench.jsonl`, which is S5's job.
    `data/schemas.py` uses the plan's illustrative names, and `BENCHMARK_FIELD_MAP` is empty until S5.
17. **G0's interface condition** is shown with `backends/mock.py` and the evaluator (`eval/runner.py`,
    `eval/metrics.py`). The trainer (`dt/`, `train/`) does not exist before S1; the same import rule
    (only `backends/base.py`) should be tested for it then.

## S1 (baselines on the toy stream)

No protocol element was changed in S1. The calibration steps are in CHANGES.md ("S1 calibration").

18. **Full-parameter training instead of adapters.** Table 3 gives B2, B5 and B6 low-rank adapters on the
    base. With the toy's 0.83M-parameter model on two cores, LoRA r=16 underfits (it cannot even reproduce
    its own training proofs; CHANGES.md step 1), so S1 trains all weights, which the S1 order allows for
    the toy. The trainer keeps both modes (`adapt.mode`). For S2 the cap of A.4 is again a set of trainable
    blocks on a frozen base, so the B2 reference in S2 may need to be the cap-only arm B2c, or a larger
    adapter, for GC to compare like with like. Human decision wanted before S2.
19. **B0's "curated context" in the toy is the obligation's statement only.** The model's window
    (192 tokens) cannot hold a snapshot's file prefix, and no lemma statements are added to the prompt.
    Consequence: FWD in reports/S1.md is measured with no snapshot context, not with "the snapshot of
    stage j as context" as section 7 defines it; a model learns a project's lemma names only by training
    on that project. The S1 targets (gap on ACC, forgetting F) do not involve FWD. If FWD is to be a
    primary S2 metric, a toy context (for example the statements of the k lemmas whose left-hand side
    matches a subterm of the goal, which needs no proof information) should be decided first.
20. **B1 is not an S1 arm.** The order's acceptance says "B6 minus max(B0, B1-equivalent)"; plan 8.1
    lists B0, B2, B5-10 and B6 for S1, and a retrieval-context B1 has no meaning for a base trained
    without context (item 19). The gap is therefore B6 minus B0.
21. **Gate metric for the gap.** 8.1 says "joint retraining should beat the frozen base by at least 20
    points" without naming the metric. The report evaluates it on ACC (end-of-stream accuracy over all
    in-window test sets, the quantity GC uses) and also shows LA.
22. **Base material excluded by the guard.** One prelude lemma (`pre_d174`, proof
    `simp induct xs { simp refl } { simp rw IH refl }`) is rejected by HoldoutGuard because the same
    generic proof is the "distinctive" proof of a test obligation (`fm_d278`): the free-text rule of
    item 11 has false positives on generic proofs. The lemma is left out of base pretraining
    (conservative) and listed in meta.json. Wake examples of the stream are never filtered; a guard raise
    during an arm aborts the run.
23. **The prelude shares the lists template with the stream project toy-lists** (S0 design, `PRELUDE_TEMPLATE`).
    Part of B0's accuracy on toy-lists can come from the analogy `pre_*` to `al_*` names; this
    favours B0 and so makes the gap conservative.
24. **One training seed.** Stage S uses one seed; the intervals are bootstrap intervals over obligations
    (section 7) and do not include training-seed variance. Test sets are small (15 to 25 obligations per
    stage), so single cells a[t, j] are noisy.
25. **Sampling seeds and batches.** R6 is implemented with a per-sample Gumbel-noise generator seeded by
    sha256(run|stage|obligation|sample), so a sample depends only on the model, the prompt and its seed,
    not on batch composition (tests/test_G0_trainer.py). Bit-identical reruns additionally need the same
    torch build and thread count (logged in meta.json).

## S2 (DT-ref on the toy stream)

No protocol element was changed in S2. DT-ref's free parameters and the cap size are in CHANGES.md ("S2").

26. **A.4's cap underfits on the toy.** With S1's base (pretrained on 159 prelude lemmas) frozen, residual MLP
    adapters on the top third plus a head cannot fit even the training proofs (CHANGES.md probes: 0
    non-trivial test proofs after 30 epochs on the whole stream). The cap was sized up as the order allows;
    see CHANGES.md for the rule and the probe that decided it. Consequence: S2 measures DT-ref with a
    larger cap than A.4 states, and still a much weaker learner than S1's full-parameter B2. Whether this is a
    property of A.4's design or of a base pretrained on so little is not separable here; on Colab the base is
    a real pretrained prover, so this toy result says little about Stage C.
27. **Conjecture sleep needs a statement-writing skill.** A.4 prompts the model with three store statements;
    a model trained only to map statements to proofs answers with a proof. DT-ref therefore trains an
    auxiliary wake task on the window's own statements (no proofs). The S1 model's 192-token context forces
    the three prompt statements to be short. A human may prefer conjectures from the channels only (S3).
28. **D2 definition.** "Without the kernel filter" is implemented as: every generated candidate (deduplicated,
    guard-passed) is learned, and the learner never sees a verdict (no fading term, no negatives, no dream
    library). Verdicts are still computed and logged so that the share of wrong proofs D2 trains on is
    measured. Another reading (verdicts used for fading or the library, only the training filter removed)
    would make D2 closer to D1.
29. **GC reference.** GC uses S1's B2 and B6, which train all weights (item 18), while every D arm trains a
    cap on a frozen base. The supplementary B2c (cap-only sequential fine-tuning, S1's recipe) is reported
    alongside so that D1 can be compared with a sequential learner of the same capacity.
30. **D3 budget.** D3 removes conjecture sleep without giving its 160 attempts to re-derivation; it has 240
    attempts per stage against D1's 400. An NSDT-style equal-budget D3 would reallocate them.
31. **Guard on dreams.** HoldoutGuard.check runs on every dream candidate before checking; refused candidates
    count as failed attempts and are never learned. The L1 free-text rule has false positives on generic
    proofs (item 22): a model-written generic proof that happens to equal a "distinctive" test proof is
    refused. The counts are in reports/S2.md.
32. **Seeds.** Seed 1 of a D arm uses S1's seed-1 base (as S1's replicate did), so seed-1 rows compare with
    S1's seed-1 B0 and B2; S1 has no seed-1 B6, so GC of seed-1 rows uses the seed-0 B2 and B6.

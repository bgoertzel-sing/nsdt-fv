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

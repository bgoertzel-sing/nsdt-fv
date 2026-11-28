# C1prep notes (hand-written; the numbers are in the generated reports/C1prep.md)

Order t-2026-11-28-nsdt-fv-c1-prep. Base: S5 head 791589b (S5 is not merged into main 917d326; this branch builds on
791589b). No Lean, elan or Mathlib installed, no repository built, no GPU; network github.com only.

## What was done

1. **History (step 1).** `nsdt_fv/runs/C1prep/clone.sh <scratch>` makes bare clones with `--filter=blob:limit=2m` of the
   23 repos.json URLs into a scratch directory outside this repository (all 23 cloned; every pinned commit is present
   and is an ancestor of HEAD), then runs `python3 -m nsdt_fv.bench.vsb_history`. The scanner walks the first-parent
   history up to c*_r, then c*_r..HEAD (forward), diffs .lean files commit to commit (vendored .lake/lake-packages
   excluded) and counts theorem/lemma keys (qualified name + statement hash) entering and leaving at each commit.
   Output per repository: `runs/C1prep/history/<repo>.json.gz`, with per-commit rows (sha, committer date, files
   changed, new, gone, live) and per-key presence intervals. Names, statements and paths are stored only as sha256
   prefixes; no source text is committed. Scan time: 51 s for all 23 on one core.
2. **Windows and stream (step 2).** `data/stream.py` (Lean section): window_cuts / window_members / split_window /
   forward_members / repo_windows / merge_stages implement 5.2 on the presence intervals (definitions in the module
   comment and tested on a hand-made history with known answers). W = 2.
3. **Shortlist (step 3).** `eval/c1prep.shortlist`, criteria and sources in the report's section 3.
4. **Code (step 4).** `LeanStream` and `build_lean_stream` (same interface as the toy Stream: obligations, edges,
   stages, backend(), by_id(), test_sets(), data_version()); `LeanHoldoutGuard` (same interface as HoldoutGuard:
   check, check_all, filter_dreams, filter_edges, check_eval_order, split_by_release, is_test_statement);
   `backends/lean_vsb.py`, `backends/lean_repl.py` (CheckerBackend skeletons, NotImplementedError where Lean is
   needed; `check_input` and `agreement` work without Lean). Stored Lean obligations use the benchmark's names for
   the five same-meaning fields (lean_root, thm_name, thm_stmt, transitive_dep_count, ground_truth_proof;
   `schemas.to_benchmark_names` / `to_internal_names` / `validate_lean_obligation`). The toy code in stream.py,
   holdout.py and schemas.py is unchanged (each file at 791589b is an exact prefix of the new file:
   runs/C1prep/toy_unchanged.txt).
5. **Report (step 5).** `python3 -m nsdt_fv.eval.c1prep t3 --real <clones>` then `... c1prep tables` regenerate
   runs/C1prep/*.json and reports/C1prep.md; tests/test_C1prep.py checks that the committed files regenerate
   byte-identically from the committed metadata.

## Decisions taken inside the order

- **First-parent history.** Mainline order is the order a team sees; side-branch commits appear at their merge commit.
  Consequence: coarse histories where projects squash (ISSUES 59).
- **Identity = qualified name + statement hash** (5.2). Windows: W = 2 cut that minimises the difference between the
  two windows' new-key counts (ties: earliest). Test = last 20% by entry commit date (committer date, UTC).
- **"At least 300 obligations" is checked twice**: keys new across the two windows net of benchmark-named keys, and keys
  live at c*. Both must reach 300 (the stricter reading; ISSUES 57 for the one borderline case).
- **Held out = test, benchmark and forward** (5.4's three kinds). Benchmark-named history keys are held out in every
  window (ISSUES 63). The benchmark records themselves (from verisoftbench.jsonl, committed in S5) are obligations of
  split benchmark at the last window's snapshot, with thm_stmt cut before the proof (ISSUES 60).
- **Collisions** (same normalised statement trained and held out): first appearance decides (ISSUES 61, 62).
- **Shortlist selection rule** (5.1 leaves the choice open when a category has several eligible repositories): plan-5.1
  provisional first, then not large, then more benchmark tasks, then more estimated obligations. Alternates: the next
  eligible repositories, not large first.
- **Guard on Lean text, without Lean**: statement key = Lean tokens without header, binder names alpha-renamed, single
  top-level =/↔ oriented; proof text = distinctive 12-token shingles; annotation = the obligation's whole proof as
  written. T3's five leaks were adapted one to one (proof text as a wake example; the premise list as a premise example;
  a dream that is the held-out statement renamed, re-spaced, with binders renamed and the equation flipped; a repair
  pair whose target is the proof; an annotation containing the proof), each injected with and without the identifier
  for every held-out obligation, plus decoys (a distinctive held-out proof under another statement) and controls.
- **Two T3 runs.** (a) Placeholder texts generated from the committed hashes (what pytest runs, on the metadata of
  capless-lean, lean-formal-reasoning-program and clean; the script runs all seven shortlisted repositories),
  together with the real benchmark records. (b) Real source texts of the same seven repositories from the local
  clones (statement, proof and redacted file prefix cut by the scanner); only counts are committed.

## Problems found and fixed during the run (before the final numbers)

- Namespace stack: 'namespace A.B' opens two scopes and 'end B' closes one; 'mutual ... end'. Canary rose from 470 to
  488 of 500 benchmark names.
- Guard tokenizer: unicode identifiers split per character, and '==>' read as '=' '=>'; binder detection picked up
  symbols. Fixed; every one of the 500 benchmark statements now has the same key as its alpha-variant.
- 36 benchmark thm_stmt values contain the proof (ISSUES 60): cut.
- Duplicate obligation ids (one name, two statements in one snapshot; three benchmark names shared by two tasks):
  disambiguated, counted in the stream config.
- A 1-token proof (':= rfl') escaped the annotation check: the check now uses the proof as written.
- Real-source run: proof spans in generated lean-mlir files (proof on the next line at column 0) were empty; the end
  of a declaration is now the next line that starts with a command keyword. Statements were cut at a ':=' inside
  '{ s with f := x }' / '(h := x)'; the cut is now at bracket depth 0 (history rescanned, counts unchanged).
  Type ascriptions in the conclusion were read as binders, and lambda/quantifier binders of a flipped equation were
  numbered differently; the statement key now renames signature binders by order and conclusion binders to one
  placeholder. Wake file prefixes count as legitimate text for the shingle test. Residuals: ISSUES 66.
- Speed: quadratic span ends and needle search replaced (bisect; one dictionary per needle length).

## Acceptance

1. Per-repository history counts and windows for all 23: report sections 1 and 2, labelled as estimates; per-commit
   rows in runs/C1prep/history. No repository failed to clone.
2. Shortlist of 5 + 2 alternates meeting 5.1 on everything checkable without Lean, with each criterion's source:
   section 3. Build time is open (C0). Risks: ISSUES 56-58.
3. Stream builder and guard tests: tests/test_C1prep.py (Lean obligations from the step 1/2 metadata, the five
   leak injections, controls, L2/L3/L4/L6/L7); toy S-stage tests unchanged and passing (pytest log in the result).
4. No repository source committed (hashes and counts only; real-source T3 commits counts only); no Lean installed;
   tables regenerate (test_tables_regenerate_byte_identically).

## Cost

GPU-hours 0. CPU: clones ~1 min wall, history scan 51 s, T3 runs a few minutes, pytest a few minutes; < 0.2 core-hours.

## Next step

C0 on the five chosen repositories, starting with lean-mlir's build time (it decides whether five categories are
possible, ISSUES 56), then C1 with the real extractor plugged into build_lean_stream as a materializer (texts(repo,
commit, decl_rows, held_keys) -> {key: obligation texts}); the guard then also gets proof states.

# S5 notes (hand-written; every number is in the generated reports/S5.md or runs/S5/*.json)

## What the files say

- The task file has 19 fields per task, all present on all 500 lines. Five have the same meaning as an Appendix B
  field and are now recorded in `data/schemas.BENCHMARK_FIELD_MAP` (`lean_root`, `thm_name`, `thm_stmt`,
  `transitive_dep_count`, `ground_truth_proof`). Six more are related but not equal (`rel_path`, `imports`,
  `local_ctx`, the four project dependency lists, the two library lists). Eight Appendix B fields have no benchmark
  source; four of them (`first_seen`, `window`, `snapshot_id`, `closed_by_baseline_tactics`) need history or Lean
  and are C1 work.
- Our records keep the Appendix B names for now. Plan 5.3 says to adopt the benchmark's names "wherever they exist";
  doing so renames fields of the toy's obligation files, which this order forbids (no change to the toy). The
  rename, or an adapter at the prompt builder, is a C1 decision (ISSUES 49).
- `target_theorem` is not a safe statement source: on 38 tasks it stops at the first `:=` inside the statement
  (a `let` or named argument). `thm_stmt` is complete on all 500.
- Regular/large is not a field. The split follows from the benchmark's own pre-generated capped prompts
  (`prompts/full_limited/`), which exist for ArkLib and lean-mlir only, one per task; with those two as large the
  counts are 381/119 over 21/2 repositories, as the plan says. The 200k-token threshold itself could not be checked
  from the files.
- Dependency depth: `nesting_depth` (median 5, max 27) and `transitive_dep_count` (median 49, max 480). The paper's
  finding that success falls with the transitive closure can be stratified on `transitive_dep_count` directly.
- Two plan names differ from the files: LeroyCompilerVerif is `LeroyCompilerVerificationCourse`, lean-formal-reasoning
  is `lean-formal-reasoning-program`.

## The shortlist

- The provisional shortlist fails the 5.1 rule on what can be checked without Lean:
  - veil has 6 benchmark tasks (rule: at least 10);
  - iris-lean, loom and veil share the benchmark category Framework, so the five cover 3 of the 5 categories.
- Reading "topic categories" as the benchmark's `category` field is an interpretation (the plan does not define the
  term). Under a looser reading of topic (separation logic, program verification, distributed protocols), iris-lean,
  loom and veil could count as distinct. The file gives only five categories, with one category per repository.
- Pool that passes the checkable criteria (at least 10 tasks, regular), by category: Applied verif. (clean, VCV-io,
  FVIntmax, wadray_verification), Compiler (LeroyCompilerVerificationCourse only), Framework (iris-lean, loom,
  splean), Semantics (juvix-lean, lean-formal-reasoning-program), Type systems (capless-lean, TTBFL). A shortlist
  with one repository per category needs LeroyCompilerVerificationCourse (LGPL-2.1, no mathlib) for Compiler.
  The non-large Compiler alternatives have 6 tasks (verified-compiler); lean-mlir is large.
- I do not choose here. The rule is applied in C1 with measured build times. My suggestion for C0's timing list is the
  provisional five minus veil, plus LeroyCompilerVerificationCourse and one Semantics repository (juvix-lean or
  lean-formal-reasoning-program); then C1 keeps one Framework repository out of iris-lean and loom. This needs the
  owner's agreement because it changes the plan's provisional list (ISSUES 50).
- Licences in the pool include GPL-3.0 (wadray_verification), LGPL-2.1 (LeroyCompilerVerificationCourse) and zlib
  (TTBFL); IntroEffects is "Written-Consent". This matters only for redistribution of extracted data (section 12).
- `has_mathlib` and the toolchain are hints for C0's build-time measurement and do not replace it. iris-lean and
  LeroyCompilerVerificationCourse build without mathlib. The other candidates pin mathlib revisions across toolchains
  v4.9 to v4.26, so the prebuilt cache has to be fetched separately for each repository.

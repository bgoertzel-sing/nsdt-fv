"""Data schemas of plan Appendix B (all files JSON Lines).

Benchmark field-name mapping: section 5.3 says to adopt VeriSoftBench's own field names, read from
data/verisoftbench.jsonl. S5 read that file and filled BENCHMARK_FIELD_MAP below; the plan's
illustrative names are still used in our records (see the comment at the map). Recorded in reports/ISSUES.md.
"""
from __future__ import annotations

OBLIGATION_FIELDS = {
    "id": str, "repo": str, "commit": str, "snapshot_id": str, "module": str, "decl": str,
    "kind": str, "statement": str, "src_up_to_decl": str, "scope": str,
    "deps_project": list, "deps_library": list, "n_deps_direct": int, "n_deps_transitive": int,
    "proof": str, "first_seen": str, "window": int, "split": str, "closed_by_baseline_tactics": bool,
}
SPLITS = ("wake", "test", "benchmark", "forward")

VERDICT_FIELDS = {
    "run": str, "stage": int, "phase": str, "target_id": str, "sample": int,
    "candidate_sha": str, "accepted": bool, "category": str, "elapsed_s": float, "repair_round": int,
}
PHASES = ("eval", "nrem", "rem", "repair")

DREAM_FIELDS = {
    "id": str, "stage": int, "kind": str, "channel": str, "statement": str, "parents": list,
    "tests_passed": int, "counterexample": object, "pln": object, "attempts": int,
    "accepted_attempt": object, "trivial": bool,
}
DREAM_KINDS = ("rederive", "compose", "explore", "repair", "plan", "model")

EDGE_FIELDS = {"snapshot_id": str, "src": str, "dst": str, "type": str}
EDGE_TYPES = ("statement_uses", "proof_uses", "unfolds_to", "similar_to", "has_property",
              "instance_of_method", "changed_in")

# benchmark name -> our name, read from VeriSoftBench's data/verisoftbench.jsonl in S5 (commit 8f5e8b36,
# sha256 ea584250...; nsdt_fv/reports/S5.md). Only fields with the same meaning are listed; related fields
# (local_ctx, imports, used_*/lemma lists, rel_path) are mapped in nsdt_fv/bench/vsb_inspect.APPENDIX_B.
# Our obligation records keep the Appendix B names for now: renaming them would change the toy's data
# files, which S5 may not do; the rename (or an adapter) is a C1 decision (ISSUES.md).
BENCHMARK_FIELD_MAP: dict = {
    "lean_root": "repo",
    "thm_name": "decl",
    "thm_stmt": "statement",
    "transitive_dep_count": "n_deps_transitive",
    "ground_truth_proof": "proof",
}


class SchemaError(ValueError):
    pass


def _check(rec: dict, fields: dict, what: str, optional=()):
    for k, ty in fields.items():
        if k not in rec:
            if k in optional:
                continue
            raise SchemaError(f"{what}: missing field {k!r}")
        if ty is object:
            continue
        v = rec[k]
        if ty is float and isinstance(v, int) and not isinstance(v, bool):
            continue
        if ty is int and isinstance(v, bool):
            raise SchemaError(f"{what}: field {k!r} must be int")
        if not isinstance(v, ty):
            raise SchemaError(f"{what}: field {k!r} has type {type(v).__name__}, expected {ty.__name__}")


def validate_obligation(rec: dict, test_view: bool = False):
    _check(rec, OBLIGATION_FIELDS, "obligation", optional=("proof",))
    if rec["split"] not in SPLITS:
        raise SchemaError(f"obligation: bad split {rec['split']!r}")
    if test_view and "proof" in rec:
        raise SchemaError("obligation: a test view must not contain the proof")
    if not test_view and rec["split"] != "test" and "proof" not in rec:
        raise SchemaError("obligation: missing proof")
    if not rec["id"].startswith(f"{rec['repo']}@{rec['commit']}:"):
        raise SchemaError("obligation: id must be <repo>@<commit>:<decl>")


def validate_verdict_line(rec: dict):
    _check(rec, VERDICT_FIELDS, "verdict line")
    if rec["phase"] not in PHASES:
        raise SchemaError(f"verdict line: bad phase {rec['phase']!r}")
    if rec["accepted"] != (rec["category"] == "ok"):
        raise SchemaError("verdict line: accepted must equal (category == 'ok')")


def validate_dream(rec: dict):
    _check(rec, DREAM_FIELDS, "dream")
    if rec["kind"] not in DREAM_KINDS:
        raise SchemaError(f"dream: bad kind {rec['kind']!r}")


def validate_edge(rec: dict):
    _check(rec, EDGE_FIELDS, "edge")
    if rec["type"] not in EDGE_TYPES:
        raise SchemaError(f"edge: bad type {rec['type']!r}")



# C1-prep: Lean obligation records are stored with the benchmark's names (5.3) for the fields that have the
# same meaning, and read back into the Appendix B names that the trainer, the guard and the evaluator use.
# The toy's records are unchanged (they never pass through these two functions).
def to_benchmark_names(rec: dict) -> dict:
    inv = {v: k for k, v in BENCHMARK_FIELD_MAP.items()}
    return {inv.get(k, k): v for k, v in rec.items()}


def to_internal_names(rec: dict) -> dict:
    return {BENCHMARK_FIELD_MAP.get(k, k): v for k, v in rec.items()}


def validate_lean_obligation(rec: dict, test_view: bool = False):
    """Validates a stored (benchmark-named) Lean obligation."""
    for k in BENCHMARK_FIELD_MAP.values():
        if k in rec:
            raise SchemaError(f"lean obligation: stored record uses Appendix B name {k!r}; use the benchmark's")
    validate_obligation(to_internal_names(rec), test_view=test_view)

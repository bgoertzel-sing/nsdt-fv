"""Data schemas of plan Appendix B (all files JSON Lines).

Benchmark field-name mapping: section 5.3 says to adopt VeriSoftBench's own field names, read from
data/verisoftbench.jsonl. That file is read in experiment S5, not S0, so BENCHMARK_FIELD_MAP is
empty for now and the plan's illustrative names are used. Recorded in reports/ISSUES.md.
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

# benchmark name -> our name; filled in S5 once data/verisoftbench.jsonl has been read
BENCHMARK_FIELD_MAP: dict = {}


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

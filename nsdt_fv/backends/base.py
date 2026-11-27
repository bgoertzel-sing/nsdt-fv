"""Interfaces of plan Appendix A.3: CheckerBackend, Verdict, EvalResult.

The trainer and the evaluator import only this module, never a concrete backend,
so that ToyEq can be replaced by a Lean backend (gate G0, last condition).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Protocol, runtime_checkable

# Verdict categories of A.3.
CATEGORIES = ("ok", "error", "timeout", "sorry", "axiom", "changed_statement")


@dataclass
class Verdict:
    accepted: bool
    category: str          # ok | error | timeout | sorry | axiom | changed_statement
    messages: list = field(default_factory=list)   # dicts: severity, text, line, column, goal_state
    elapsed_s: float = 0.0

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown verdict category {self.category!r}")
        if self.accepted != (self.category == "ok"):
            raise ValueError("accepted must be True exactly when category == 'ok'")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Verdict":
        return Verdict(bool(d["accepted"]), d["category"], list(d.get("messages", [])),
                       float(d.get("elapsed_s", 0.0)))


@dataclass
class EvalResult:
    ok: bool
    value: str | None
    category: str = "ok"   # ok | error | timeout
    message: str = ""
    elapsed_s: float = 0.0


@runtime_checkable
class CheckerBackend(Protocol):
    def snapshots(self) -> list[str]: ...

    def check(self, snapshot_id: str, obligation_id: str,
              candidate: str, timeout_s: int) -> Verdict: ...

    def check_statement(self, snapshot_id: str, context: str, statement: str,
                        candidate: str, timeout_s: int) -> Verdict: ...

    def evaluate_term(self, snapshot_id: str, context: str,
                      term: str, timeout_s: int) -> EvalResult: ...

"""StubLeanBackend: a CheckerBackend (plan A.3) that stands where the Lean backend of Stage C will
stand (lean_vsb.py / lean_repl.py), for gate G0's last condition. No Lean is installed or called.

Verdicts are canned:
  1. an explicit table of canned verdicts keyed by (snapshot, obligation, candidate) wins;
  2. otherwise Lean's acceptance rules of R8 are imitated on the text: sorry/admit -> 'sorry', a new
     axiom -> 'axiom', a candidate that restates the declaration with another statement -> 'changed_statement';
  3. otherwise a candidate whose Lean tokens equal those of the ground-truth proof is 'ok' (a leading ':='
     is ignored, as in VeriSoftBench's ground_truth_proof field), anything else 'error' with a Lean-style
     message ('unsolved goals').

Snapshots are '<repo>@<window or commit>' and obligations are Appendix B records, as in Stage C. This module
imports nothing from ToyEq (checked by tests/test_G0_stub_lean.py).
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import Counter

from .base import EvalResult, Verdict

_TOK = re.compile(r"[A-Za-z0-9_.'!?]+|\S")
_DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(theorem|lemma)\s+(\S+)(.*)$", re.S)


def lean_tokens(text: str) -> list:
    t = text.strip()
    if t.startswith(":="):
        t = t[2:]
    return _TOK.findall(t)


def _key(snapshot_id, obligation_id, candidate):
    return hashlib.sha256(f"{snapshot_id}\x00{obligation_id}\x00{candidate}".encode()).hexdigest()


def _msg(text, goal=""):
    return [{"severity": "error", "text": text, "line": 1, "column": 0, "goal_state": goal}]


class StubLeanBackend:
    def __init__(self, obligations, canned=None, evals=None):
        """obligations: Appendix B records (id, snapshot_id, decl, statement, proof when known).
        canned: {(snapshot_id, obligation_id, candidate): category} explicit verdicts.
        evals: {(snapshot_id, term): value} for evaluate_term."""
        self.truth = {}
        self.statement = {}
        self.decl = {}
        for o in obligations:
            k = (o["snapshot_id"], o["id"])
            self.statement[k] = o["statement"]
            self.decl[k] = o["decl"]
            if o.get("proof") is not None:
                self.truth[k] = lean_tokens(o["proof"])
        self.canned = {_key(s, i, c): cat for (s, i, c), cat in (canned or {}).items()}
        self.evals = dict(evals or {})
        self.calls = Counter()

    def snapshots(self):
        return sorted({s for s, _ in self.statement})

    def _rules(self, k, candidate):
        if re.search(r"\b(sorry|admit)\b", candidate):
            return "sorry", "declaration uses 'sorry'"
        if re.search(r"^\s*axiom\b", candidate, re.M):
            return "axiom", "new axiom declared"
        m = _DECL.match(candidate)
        if m and k in self.statement:
            stated = lean_tokens(m.group(0).split(":=", 1)[0])
            if stated != lean_tokens(self.statement[k]):
                return "changed_statement", "the candidate changes the statement of the obligation"
        return None, None

    def _verdict(self, cat, msg, t0):
        self.calls[cat] += 1
        if cat == "ok":
            return Verdict(True, "ok", [], time.monotonic() - t0)
        if cat == "timeout":
            return Verdict(False, "timeout", _msg("(deterministic) timeout"), time.monotonic() - t0)
        return Verdict(False, cat, _msg(msg or cat, "⊢ ?"), time.monotonic() - t0)

    def check(self, snapshot_id, obligation_id, candidate, timeout_s):
        t0 = time.monotonic()
        k = (snapshot_id, obligation_id)
        if k not in self.statement:
            raise KeyError(f"unknown obligation {obligation_id} in {snapshot_id}")
        canned = self.canned.get(_key(snapshot_id, obligation_id, candidate))
        if canned is not None:
            return self._verdict(canned, f"canned verdict {canned}", t0)
        cat, msg = self._rules(k, candidate)
        if cat:
            return self._verdict(cat, msg, t0)
        body = candidate
        m = _DECL.match(candidate)
        if m and ":=" in candidate:
            body = candidate.split(":=", 1)[1]
        if k in self.truth and lean_tokens(body) == self.truth[k]:
            return self._verdict("ok", None, t0)
        return self._verdict("error", "unsolved goals", t0)

    def check_statement(self, snapshot_id, context, statement, candidate, timeout_s):
        t0 = time.monotonic()
        canned = self.canned.get(_key(snapshot_id, "stmt:" + statement, candidate))
        if canned is not None:
            return self._verdict(canned, f"canned verdict {canned}", t0)
        if re.search(r"\b(sorry|admit)\b", candidate):
            return self._verdict("sorry", "declaration uses 'sorry'", t0)
        return self._verdict("error", "unsolved goals", t0)

    def evaluate_term(self, snapshot_id, context, term, timeout_s):
        key = (snapshot_id, term)
        if key in self.evals:
            return EvalResult(True, self.evals[key], "ok", "", 0.0)
        return EvalResult(False, None, "error", "failed to compile definition, consider marking it as 'noncomputable'", 0.0)

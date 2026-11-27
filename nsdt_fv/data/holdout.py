"""HoldoutGuard: rules L1-L7 of plan section 5.4, enforced in code (rule R4).

All training and mining data pass through HoldoutGuard.check(example), which raises LeakError.
An example is a dict:
  kind        wake | premise | nextstate | unfold | diag | repair | plan | aux | dream | annotation | statement
  statement   the statement the example is about (required for proof-derived kinds and dreams)
  input, target, and any other text fields
  source_ids / parents / target_id / obligation_id   identifiers it was derived from
  source      human | model | channel | frontier
  obligation_id  for an annotation: the obligation whose prompt it annotates (may be a test one)

What is checked (toy reading of L1-L7; the choices are listed in reports/ISSUES.md):
  L1  a test obligation identifier among the example's source identifiers or in its text;
      a proof-derived example (wake, premise, nextstate, diag, repair, plan, aux) whose statement is
      a test statement (this covers proof text, premise lists and repair pairs of test obligations);
      a distinctive test proof (at least min_tokens tokens and not the proof of any non-test lemma)
      anywhere in the text; a distinctive intermediate goal state of a test proof anywhere in the text;
      an annotation that contains the proof of the obligation it annotates.
  L2  filter_edges drops the outgoing proof_uses edges of test statements.
  L3  a dream whose normalised statement (alpha- and orientation-insensitive) equals a test
      statement: check() raises; filter_dreams() drops and counts.
  L4  check_eval_order raises if a score a_{t,j} is recorded after a later window was trained.
  L5  split_by_release returns the obligations first committed after a base model's release date.
  L6  before the freeze, an identifier from a non-development project raises.
  L7  an example whose source is a frontier model raises.
"""
from __future__ import annotations

from ..backends.toyeq import (ToyError, World, check_proof, parse_statement, proof_tokens, show,
                              statement_key, canonical, tokenize)

PROOF_DERIVED = {"wake", "premise", "nextstate", "diag", "repair", "plan", "aux"}
KINDS = PROOF_DERIVED | {"unfold", "dream", "annotation", "statement"}
ID_FIELDS = ("source_ids", "parents", "target_id", "obligation_id", "id")


class LeakError(Exception):
    def __init__(self, rule, detail):
        super().__init__(f"[{rule}] {detail}")
        self.rule = rule
        self.detail = detail


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def _ids(example):
    out = []
    for f in ID_FIELDS:
        v = example.get(f)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(str(i) for i in v)
    return out


_OPEN = {"(", ","}
_CLOSE = {"(", ",", ")"}


class _Needles:
    """Token-sequence search with term boundaries."""

    def __init__(self):
        self.by_first = {}

    def add(self, toks, tag):
        if toks:
            self.by_first.setdefault(toks[0], []).append((tuple(toks), tag))

    def find(self, toks):
        n = len(toks)
        for i, t in enumerate(toks):
            for needle, tag in self.by_first.get(t, ()):
                m = len(needle)
                if i + m <= n and tuple(toks[i:i + m]) == needle:
                    if i > 0 and toks[i - 1] in _OPEN:
                        continue
                    if i + m < n and toks[i + m] in _CLOSE:
                        continue
                    return tag
        return None


def _body_tokens(lhs, rhs):
    return [t.text for t in tokenize(f"{show(lhs)} = {show(rhs)}")]


class HoldoutGuard:
    def __init__(self, obligations, worlds: dict, min_tokens: int = 8, dev_projects=None,
                 frozen: bool = False):
        self.min_tokens = min_tokens
        self.frozen = frozen
        self.dev_projects = set(dev_projects) if dev_projects is not None else None
        self.funcs = {}
        for w in worlds.values():
            self.funcs.update(w.funcs)
        self.by_id = {o["id"]: o for o in obligations}
        test = [o for o in obligations if o["split"] == "test"]
        self.test_ids = {o["id"] for o in test}
        self.test_keys = {}
        self.test_proof = {}
        self.test_premises = {}
        self.proof_needles = _Needles()
        self.state_needles = _Needles()
        self.id_needles = _Needles()
        generic_proofs = {tuple(proof_tokens(o["proof"])) for o in obligations
                          if o["split"] != "test" and "proof" in o}
        statement_bodies = set()
        for o in obligations:
            st = parse_statement(o["statement"], self.funcs)
            statement_bodies.add(tuple(_body_tokens(st.lhs, st.rhs)))
        nontest_states = set()
        for o in obligations:
            if o["split"] != "test" and "proof" in o:
                nontest_states |= self._states(o, worlds)
        self.n_distinctive_proofs = 0
        self.n_states = 0
        for o in test:
            st = parse_statement(o["statement"], self.funcs)
            self.test_keys[statement_key(st)] = o["id"]
            ptoks = proof_tokens(o["proof"])
            self.test_proof[o["id"]] = ptoks
            if len(ptoks) >= min_tokens and tuple(ptoks) not in generic_proofs:
                self.proof_needles.add(ptoks, o["id"])
                self.n_distinctive_proofs += 1
            for s in self._states(o, worlds) - nontest_states - statement_bodies:
                self.state_needles.add(list(s), o["id"])
                self.n_states += 1
            self.id_needles.add([t.text for t in tokenize(o["id"])], o["id"])
        self.dropped_dreams = 0

    def _states(self, o, worlds):
        """Intermediate goal states (bodies) of the ground-truth proof, root excluded."""
        w: World = worlds[o["snapshot_id"]]
        pos = w.lemma_pos[o["decl"]]
        st = w.lemmas[pos][1]
        trace = []
        check_proof(w, st, o["proof"], pos, trace=trace)
        root = tuple(_body_tokens(st.lhs, st.rhs))
        out = set()
        for _, _, g in trace:
            b = tuple(_body_tokens(g.lhs, g.rhs))
            if b != root and len(b) >= 5:
                out.add(b)
        return out

    # ------------------------------------------------------------------ L1, L3, L6, L7
    def _key(self, text):
        try:
            return statement_key(parse_statement(text, self.funcs))
        except ToyError as e:
            raise LeakError("schema", f"statement does not parse: {e}")

    def is_test_statement(self, text) -> bool:
        return self._key(text) in self.test_keys

    def check(self, ex: dict) -> None:
        kind = ex.get("kind")
        if kind not in KINDS:
            raise LeakError("schema", f"unknown example kind {kind!r}")
        if ex.get("source") == "frontier":
            raise LeakError("L7", "frontier-model output may not enter training")
        annotated = ex.get("obligation_id") if kind == "annotation" else None
        for i in _ids(ex):
            if i in self.test_ids and i != annotated:
                raise LeakError("L1", f"test identifier {i}")
            if self.dev_projects is not None and not self.frozen:
                repo = i.split("@", 1)[0]
                if i in self.by_id and repo not in self.dev_projects:
                    raise LeakError("L6", f"{i} is not from a development project")
        if (kind in PROOF_DERIVED or kind == "dream") and not ex.get("statement"):
            raise LeakError("schema", f"a {kind} example must carry its statement")
        if ex.get("statement"):
            key = self._key(ex["statement"])
            tid = self.test_keys.get(key)
            if tid is not None:
                if kind == "dream":
                    raise LeakError("L3", f"dream equals test statement of {tid}")
                if kind in PROOF_DERIVED:
                    raise LeakError("L1", f"{kind} example about test statement {tid}")
        toks = []
        for s in _strings({k: v for k, v in ex.items() if k not in ID_FIELDS}):
            toks.extend(t.text for t in tokenize(s))
            toks.append("\u241e")         # field separator: no match across fields
        hit = self.id_needles.find(toks)
        if hit is not None and hit != annotated:
            raise LeakError("L1", f"text contains test identifier {hit}")
        hit = self.proof_needles.find(toks)
        if hit is not None:
            raise LeakError("L1", f"text overlaps the proof of test obligation {hit}")
        hit = self.state_needles.find(toks)
        if hit is not None:
            raise LeakError("L1", f"text contains a proof state of test obligation {hit}")
        if annotated in self.test_proof:
            ann = _Needles()
            ann.add(self.test_proof[annotated], annotated)
            if ann.find(toks) is not None:
                raise LeakError("L1", f"annotation contains the proof of {annotated}")

    def check_all(self, examples):
        for ex in examples:
            self.check(ex)
            yield ex

    def filter_dreams(self, dreams):
        """L3: drop dreams whose normalised statement equals a test statement; count them."""
        kept = []
        for d in dreams:
            if self._key(d["statement"]) in self.test_keys:
                self.dropped_dreams += 1
            else:
                kept.append(d)
        return kept

    # ------------------------------------------------------------------ L2, L4, L5
    def filter_edges(self, edges):
        test_decls = {(self.by_id[i]["snapshot_id"].split("@")[0], self.by_id[i]["decl"]) for i in self.test_ids}
        out = []
        for e in edges:
            if e["type"] == "proof_uses" and (e["snapshot_id"].split("@")[0], e["src"]) in test_decls:
                continue
            out.append(e)
        return out

    @staticmethod
    def check_eval_order(events):
        """events in time order: {'type': 'train', 'stage': t} or {'type': 'eval', 'model_stage': t, ...}."""
        trained = 0
        for e in events:
            if e["type"] == "train":
                trained = max(trained, e["stage"])
            elif e["type"] == "eval" and e["model_stage"] < trained:
                raise LeakError("L4", f"score for model after stage {e['model_stage']} recorded after stage {trained} was trained")

    @staticmethod
    def split_by_release(obligations, release_date: str):
        return list(obligations), [o for o in obligations if o["first_seen"] > release_date]

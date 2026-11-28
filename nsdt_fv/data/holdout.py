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



# =============================================================================================== Lean
# C1-prep: the same guard for Lean obligations. Same interface as HoldoutGuard (check, check_all,
# filter_dreams, filter_edges, check_eval_order, split_by_release, is_test_statement); nothing above changed.
# Without Lean there is no parser, so the toy's structural checks become text checks:
#   held-out sets   5.4's three kinds: split 'test' (in-window), 'benchmark' and 'forward'.
#   statements      lean_statement_key: Lean tokens, declaration keyword/name/attributes dropped, binder names
#                   alpha-renamed in order of binding, the two sides of a single top-level '=' or '↔' in the
#                   conclusion put in a fixed order. Not handled: binders moved between the signature and a
#                   '∀' in the type, implicit/explicit changes, notation unfolding (needs Lean; C1).
#   proof text      every held-out proof contributes its distinctive K-token shingles (K = shingle, default 12):
#                   shingles that occur in no non-held-out proof and in no statement. A held-out proof shorter
#                   than K but at least min_tokens long is one needle, if no non-held-out proof equals it.
#                   Any example text containing one raises L1.
#   proof states    if obligations carry 'proof_states' (the extractor's goal states, C1), their distinctive
#                   states are needles too.
#   annotations     an annotation for obligation X raises if it contains X's whole proof as written, including a
#                   leading ':=' (at least 2 tokens, so ':= rfl' counts).
import hashlib  # noqa: E402
import re  # noqa: E402

# Lean tokens: comments dropped; unicode identifiers (Γ, ρ₁, ℕ) are single tokens; a run of ASCII operator
# characters (':=', '==>', '<=', '-∗' gives '-' '∗') is one token; every other non-space character is one token.
_LTOK = re.compile(r"[\w.'!?]+|[:=<>~+*/\\|&^%$#-]+|\S")
_COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)


def _lean_tokens(text: str) -> list:
    return _LTOK.findall(_COMMENT.sub(" ", text))

LEAN_HELD_OUT = ("test", "benchmark", "forward")
_BINDER_OPEN = {"(", "{", "[", "⦃", "{{"}
_QUANT = {"∀", "∃", "∃!", "fun", "λ", "Π", "Σ", "forall", "exists"}
_IDENT_RE = re.compile(r"^[^\W\d][\w'!?]*$")
_KEYWORDS = {"fun", "λ", "forall", "exists", "let", "have", "show", "from", "by", "at", "with", "if", "then", "else",
             "match", "do", "Type", "Prop", "Sort", "in", "where"}


class _Ident:
    @staticmethod
    def match(t):
        return t not in _KEYWORDS and _IDENT_RE.match(t) is not None


_IDENT = _Ident()
_MODS = {"private", "protected", "noncomputable", "nonrec", "unsafe", "partial", "scoped", "local"}


def _strip_header(toks):
    """Drop '@[...]' attributes, modifiers, 'theorem|lemma NAME' and a trailing ':=' / 'by'."""
    i = 0
    while i < len(toks):
        if toks[i] == "@" and i + 1 < len(toks) and toks[i + 1] == "[":
            depth, i = 0, i + 1
            while i < len(toks):
                depth += toks[i] == "["
                depth -= toks[i] == "]"
                i += 1
                if depth == 0:
                    break
            continue
        if toks[i] in _MODS:
            i += 1
            continue
        break
    if i < len(toks) and toks[i] in ("theorem", "lemma"):
        i += 2
    toks = toks[i:]
    while toks and toks[-1] in (":=", "by", ":", "="):
        if toks[-1] == "=" and len(toks) >= 2 and toks[-2] == ":":
            toks = toks[:-2]
            continue
        if toks[-1] in (":=", "by"):
            toks = toks[:-1]
            continue
        break
    return toks


def _join_assign(toks):
    out = []
    for t in toks:
        if t == "=" and out and out[-1] == ":":
            out[-1] = ":="
        elif t == ">" and out and out[-1] == "=":
            out[-1] = "=>"
        else:
            out.append(t)
    return out


def lean_statement_tokens(text: str) -> list:
    """Lean tokens of a statement without attributes, modifiers, keyword and name."""
    return _strip_header(_join_assign(_lean_tokens(text)))


def lean_binder_names_statement(toks) -> list:
    """Binder names of a statement: bracket groups and quantifiers in the signature (before the conclusion's ':'),
    quantifiers only in the conclusion (a '(x : T)' there is a type ascription, not a binder)."""
    head, concl = lean_conclusion_split(toks)
    return lean_binder_names(head) + lean_binder_names(concl, brackets=False)


def lean_binder_names(toks, brackets: bool = True) -> list:
    """Binder names in order: the identifiers of a bracket group '(x y : T)', '{x : T}', '[inst : C]', '⦃x : T⦄'
    (a group with any non-identifier before its ':' is not a binder group), and the identifiers after a
    quantifier up to ',' ':' '=>' or a bracket."""
    names = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t in _BINDER_OPEN and brackets:
            j = i + 1
            cand = []
            while j < n and toks[j] != ":" and _IDENT.match(toks[j]) and len(cand) < 12:
                cand.append(toks[j])
                j += 1
            if cand and j < n and toks[j] == ":":
                names.extend(c for c in cand if c != "_")
        elif t in _QUANT:
            j = i + 1
            while j < n and _IDENT.match(toks[j]) and j - i < 12:
                if toks[j] != "_":
                    names.append(toks[j])
                j += 1
        i += 1
    return names


def lean_conclusion_split(toks):
    """(head, conclusion): the conclusion is what follows the last ':' at bracket depth 0."""
    depth, colon = 0, -1
    for i, t in enumerate(toks):
        if t in ("(", "{", "[", "⦃"):
            depth += 1
        elif t in (")", "}", "]", "⦄"):
            depth -= 1
        elif t == ":" and depth == 0:
            colon = i
    return toks[: colon + 1], toks[colon + 1:]


def lean_top_eq(concl):
    """Index of the single top-level '=' or '↔' of a conclusion, or None."""
    depth, ops = 0, []
    for i, t in enumerate(concl):
        if t in ("(", "{", "[", "⦃"):
            depth += 1
        elif t in (")", "}", "]", "⦄"):
            depth -= 1
        elif t in ("=", "↔") and depth == 0:
            ops.append(i)
        elif t in ("→", "∧", "∨", ",") and depth == 0:
            return None
    return ops[0] if len(ops) == 1 else None


def lean_statement_key(text: str) -> str:
    """Hash of a statement up to binder names, whitespace, keyword/name and the orientation of a single top-level
    '=' or '↔' in the conclusion."""
    toks = lean_statement_tokens(text)
    head, concl = lean_conclusion_split(toks)
    names = lean_binder_names(head)
    ren = {}
    for nm in names:
        if nm not in ren and nm not in _QUANT and nm not in _BINDER_OPEN:
            ren[nm] = f"_v{len(ren)}"
    # signature binders by order of binding; binders of the conclusion ('∀ x,', 'fun x =>') all become '_q', so the
    # key does not depend on the order in which the two sides of a flipped equation bind them
    inner = {n for n in lean_binder_names(concl, brackets=False) if n not in ren}
    head = [ren.get(t, t) for t in head]
    concl = [ren.get(t, "_q" if t in inner else t) for t in concl]
    op = lean_top_eq(concl)
    if op is not None:
        a, b = concl[:op], concl[op + 1:]
        concl = min(a, b) + [concl[op]] + max(a, b)
    return hashlib.sha256("\x1f".join(head + concl).encode()).hexdigest()


def _toks(text: str) -> list:
    return _join_assign(_lean_tokens(text))


def lean_proof_tokens(proof: str) -> list:
    t = _toks(proof)
    if t and t[0] == ":=":
        t = t[1:]
    return t


class _Shingles:
    def __init__(self, k):
        self.k = k
        self.tag = {}

    @staticmethod
    def _h(seq):
        return hashlib.blake2b("\x1f".join(seq).encode(), digest_size=10).digest()

    def grams(self, toks):
        k = self.k
        return [self._h(toks[i:i + k]) for i in range(0, max(0, len(toks) - k + 1))]

    def find(self, toks):
        for g in self.grams(toks):
            if g in self.tag:
                return self.tag[g]
        return None


class _Exact:
    """Whole-sequence needles (proofs shorter than the shingle size, proof states), matched anywhere in the text:
    one dictionary per needle length, so a search costs (text length) x (number of distinct needle lengths)."""
    def __init__(self):
        self.by_first = {}          # first token -> [(needle, tag)] (kept for callers that list the needles)
        self.by_len = {}

    def add(self, toks, tag):
        if toks:
            t = tuple(toks)
            self.by_first.setdefault(t[0], []).append((t, tag))
            self.by_len.setdefault(len(t), {}).setdefault(t, tag)

    def find(self, toks):
        n = len(toks)
        for m, table in self.by_len.items():
            for i in range(0, n - m + 1):
                tag = table.get(tuple(toks[i:i + m]))
                if tag is not None:
                    return tag
        return None


class LeanHoldoutGuard:
    """HoldoutGuard for Lean obligations (internal Appendix B field names; see schemas.to_internal_names)."""

    def __init__(self, obligations, worlds=None, min_tokens: int = 8, shingle: int = 12, dev_projects=None,
                 frozen: bool = False, held_out=LEAN_HELD_OUT):
        self.min_tokens = min_tokens
        self.frozen = frozen
        self.dev_projects = set(dev_projects) if dev_projects is not None else None
        self.by_id = {o["id"]: o for o in obligations}
        held = [o for o in obligations if o["split"] in held_out]
        self.test_ids = {o["id"] for o in held}
        self.test_keys = {}
        for o in held:
            self.test_keys.setdefault(lean_statement_key(o["statement"]), o["id"])
        self.test_proof = {o["id"]: lean_proof_tokens(o["proof"]) for o in held if o.get("proof")}
        self.test_proof_full = {o["id"]: _toks(o["proof"]) for o in held if o.get("proof")}   # with ':=' if written
        k = shingle
        self.shingles = _Shingles(k)
        self.exact = _Exact()
        generic = set()
        generic_whole = set()
        for o in obligations:
            if o["split"] not in held_out and o.get("proof"):
                pt = lean_proof_tokens(o["proof"])
                generic.update(self.shingles.grams(pt))
                generic_whole.add(tuple(pt))
            generic.update(self.shingles.grams(_toks(o["statement"])))
            if o["split"] not in held_out and o.get("src_up_to_decl"):
                # a wake obligation's file prefix is legitimate training text (held-out proofs in it are redacted
                # by the materializer), so its shingles are not evidence of a leak
                generic.update(self.shingles.grams(_toks(o["src_up_to_decl"])))
        self.n_distinctive_proofs = 0
        for o in held:
            pt = self.test_proof.get(o["id"])
            if not pt:
                continue
            added = False
            if len(pt) >= k:
                for g in self.shingles.grams(pt):
                    if g not in generic:
                        self.shingles.tag.setdefault(g, o["id"])
                        added = True
            elif len(pt) >= min_tokens and tuple(pt) not in generic_whole:
                self.exact.add(pt, o["id"])
                added = True
            self.n_distinctive_proofs += added
        nontest_states = set()
        for o in obligations:
            if o["split"] not in held_out:
                nontest_states |= {tuple(_toks(s)) for s in o.get("proof_states", ())}
        self.n_states = 0
        for o in held:
            for s in o.get("proof_states", ()):
                st = tuple(_toks(s))
                if len(st) >= 5 and st not in nontest_states:
                    self.exact.add(list(st), o["id"])
                    self.n_states += 1
        self.dropped_dreams = 0

    def is_test_statement(self, text) -> bool:
        return lean_statement_key(text) in self.test_keys

    def check(self, ex: dict) -> None:
        kind = ex.get("kind")
        if kind not in KINDS:
            raise LeakError("schema", f"unknown example kind {kind!r}")
        if ex.get("source") == "frontier":
            raise LeakError("L7", "frontier-model output may not enter training")
        annotated = ex.get("obligation_id") if kind == "annotation" else None
        for i in _ids(ex):
            if i in self.test_ids and i != annotated:
                raise LeakError("L1", f"held-out identifier {i}")
            if self.dev_projects is not None and not self.frozen:
                repo = i.split("@", 1)[0]
                if i in self.by_id and repo not in self.dev_projects:
                    raise LeakError("L6", f"{i} is not from a development repository")
        if (kind in PROOF_DERIVED or kind == "dream") and not ex.get("statement"):
            raise LeakError("schema", f"a {kind} example must carry its statement")
        if ex.get("statement"):
            tid = self.test_keys.get(lean_statement_key(ex["statement"]))
            if tid is not None:
                if kind == "dream":
                    raise LeakError("L3", f"dream equals held-out statement of {tid}")
                if kind in PROOF_DERIVED:
                    raise LeakError("L1", f"{kind} example about held-out statement {tid}")
        toks = []
        for s in _strings({k: v for k, v in ex.items() if k not in ID_FIELDS}):
            toks.extend(_toks(s))
            toks.append("\u241e")
        text = " ".join(_strings({k: v for k, v in ex.items() if k not in ID_FIELDS}))
        for i in self._ids_in_text(text):
            if i != annotated:
                raise LeakError("L1", f"text contains held-out identifier {i}")
        hit = self.shingles.find(toks)
        if hit is not None:
            raise LeakError("L1", f"text overlaps the proof of held-out obligation {hit}")
        hit = self.exact.find(toks)
        if hit is not None:
            raise LeakError("L1", f"text contains the proof or a proof state of held-out obligation {hit}")
        if annotated in self.test_proof_full and len(self.test_proof_full[annotated]) >= 2:
            ann = _Exact()
            ann.add(self.test_proof_full[annotated], annotated)
            if ann.find(toks) is not None:
                raise LeakError("L1", f"annotation contains the proof of {annotated}")

    _ID_RE = re.compile(r"[^\s\"'`,()\[\]{}]+@[0-9a-fA-F]{7,40}:[^\s\"'`,()\[\]{}]+")

    def _ids_in_text(self, text):
        if "@" not in text:
            return []
        out = []
        for m in self._ID_RE.findall(text):
            for cand in (m, m.rstrip(".;:")):
                if cand in self.test_ids:
                    out.append(cand)
                    break
        return out

    def check_all(self, examples):
        for ex in examples:
            self.check(ex)
            yield ex

    def filter_dreams(self, dreams):
        kept = []
        for d in dreams:
            if lean_statement_key(d["statement"]) in self.test_keys:
                self.dropped_dreams += 1
            else:
                kept.append(d)
        return kept

    def filter_edges(self, edges):
        test_decls = {(self.by_id[i]["snapshot_id"].split("@")[0], self.by_id[i]["decl"]) for i in self.test_ids}
        return [e for e in edges
                if not (e["type"] == "proof_uses" and (e["snapshot_id"].split("@")[0], e["src"]) in test_decls)]

    check_eval_order = staticmethod(HoldoutGuard.check_eval_order)
    split_by_release = staticmethod(HoldoutGuard.split_by_release)

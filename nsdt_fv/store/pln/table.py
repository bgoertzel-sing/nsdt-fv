"""S4 property tables HasProperty(f, T) on the six ToyEq projects (plan 8.1 S4, 9.2).

Shapes come from channel C's output (S3, arm N-C, runs/S3/N-C/channels.jsonl.gz): every guard-passed equation
C proposed is abstracted once per function symbol f occurring in it, by replacing f with the hole ?F. The other
symbols are written project-independently: a template function by its template name (the project prefix
stripped; its definition is the same in every project that has it, checked by base_defs), a project wrapper by
'wrap__<prefixed name>' (so a shape mentioning a wrapper applies inside its project only), constructors as they are.
Variables are alpha-normalised and the equation orientation-normalised (the smaller of the two canonical texts),
so a shape is a template with one function hole.

Vocabulary: shapes that C's output instantiates for at least min_support (2) distinct functions.

Cells: every (project function f, shape T) whose instance T[f] parses and type-checks in the project's last
snapshot (window 2, all functions), minus instances equal to a test statement of the stream
(HoldoutGuard.is_test_statement). Cells with the same instance statement (one equation abstracted at two of its
functions) form one statement group; hiding works on groups.

Labels come from the checker only: label 1 iff a proof of T[f] is found within budget (the baseline tactics,
then the reference prover with prover_nodes on the project's test-free library, as ChannelBase.prove) and
ToyEqBackend.check_statement accepts it on the snapshot. The random-test evaluator is run as well (200 inputs);
its counterexamples are a descriptive column only and never set a label.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re

from ...backends.toyeq import CTORS, Statement, ToyError, World, canonical, parse_term, show, symbols

HOLE = "?F"
WRAP = "wrap__"
_WRAP = re.compile(r"^w\d+$")
_POOLS = {"Nat": ["n", "m", "k", "i", "j", "p", "q"], "List": ["xs", "ys", "zs", "ws", "us", "vs"],
          "Tree": ["t", "u", "s", "r"], "Bool": ["b", "c", "d", "e"]}
_VTYPE = {v: ty for ty, pool in _POOLS.items() for v in pool}


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def project_of_sid(sid):
    return sid.split("@")[0]


class Naming:
    """Project-independent names for the functions of the stream's projects."""

    def __init__(self, stream):
        self.prefix = {p.name: p.prefix for p in stream.projects}

    def base(self, proj, fname):
        pre = self.prefix[proj]
        b = fname[len(pre):] if fname.startswith(pre) else fname
        if _WRAP.match(b):
            return WRAP + fname           # the prefixed name is unique to its project
        return b

    def concrete(self, proj, base, funcs):
        """Inverse of base() in project proj, or None if the project lacks the symbol."""
        if base.startswith(WRAP):
            f = base[len(WRAP):]
            return f if f in funcs else None
        f = self.prefix[proj] + base
        return f if f in funcs else None


def _rename(t, m):
    if isinstance(t, str):
        return t
    return (m.get(t[0], t[0]),) + tuple(_rename(a, m) for a in t[1:])


def _fd_text(fd, proj, naming):
    eqs = []
    for l, r in fd.eqs:
        m = {s: naming.base(proj, s) for s in symbols(l) | symbols(r) if s not in CTORS}
        eqs.append(show(_rename(l, m)) + " = " + show(_rename(r, m)))
    return f"{naming.base(proj, fd.name)} : {' '.join(fd.argtypes)} -> {fd.rettype} | " + " ; ".join(eqs)


def base_defs(stream, naming):
    """{template name: definition text}; raises if a template name has two different definitions."""
    out = {}
    for p in stream.projects:
        for fd in p.world.funcs.values():
            b = naming.base(p.name, fd.name)
            if b.startswith(WRAP):
                continue
            txt = _fd_text(fd, p.name, naming)
            if b in out and out[b] != txt:
                raise ValueError(f"template function {b} defined differently: {out[b]!r} vs {txt!r}")
            out[b] = txt
    return out


def shape_of(st, focus, proj, naming):
    """The shape of statement st at function symbol focus (a function of project proj)."""
    m = {s: (HOLE if s == focus else naming.base(proj, s)) for s in symbols(st.lhs) | symbols(st.rhs) if s not in CTORS}
    st2 = Statement(st.vars, _rename(st.lhs, m), _rename(st.rhs, m))
    return min(canonical(st2).show(), canonical(st2, flip=True).show())


def _parse_shape(text):
    if text.startswith("forall"):
        head, body = text.split(".", 1)
        names = head.split()[1:]
    else:
        names, body = [], text
    l, r = body.split("=", 1)
    return names, parse_term(l.strip()), parse_term(r.strip())


_SHAPE_CACHE = {}


def instantiate(shape, f, proj, naming, world):
    """Statement text of shape[f] in proj, or None if a symbol is missing or the instance is ill typed."""
    if shape not in _SHAPE_CACHE:
        _SHAPE_CACHE[shape] = _parse_shape(shape.replace(HOLE, "QQHOLE"))
    names, l, r = _SHAPE_CACHE[shape]
    m = {}
    for s in symbols(l) | symbols(r):
        if s in CTORS:
            continue
        if s == "QQHOLE":
            m[s] = f
            continue
        c = naming.concrete(proj, s, world.funcs)
        if c is None or c == f:          # missing symbol, or the hole would coincide with a fixed symbol
            return None
        m[s] = c
    vs = " ".join(names)
    text = (f"forall {vs}. " if vs else "") + f"{show(_rename(l, m))} = {show(_rename(r, m))}"
    try:
        st = world.statement(text)
    except ToyError:
        return None
    env = dict(st.vars)
    for v in names:                        # canonical variable names carry the shape's variable types
        if env.get(v) != _VTYPE.get(v.rstrip("0123456789")):
            return None
    return text


def private_world(stream, proj, test_decls):
    """Last snapshot of proj with every test lemma of either window removed from the library."""
    w = stream.worlds[f"{proj}@w2"]
    pw = World(f"{proj}@w2/private")
    for fd in w.funcs.values():
        pw.add_def(fd)
    for name, st in w.lemmas:
        if (proj, name) not in test_decls:
            pw.add_lemma(name, st)
    return pw


def channel_c_shapes(stream, naming, log_path):
    """({shape: set of (proj, f)}, statements read) from the guard-passed statements of channel C's log."""
    stage_sid = {t: sid for t, (sid, _) in enumerate(stream.stages, 1)}
    occ = {}
    n_stmt = 0
    with gzip.open(log_path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("what") != "statement" or r.get("guard") != "pass" or not r.get("statement"):
                continue
            sid = stage_sid[int(r["stage"])]
            proj = project_of_sid(sid)
            w = stream.worlds[sid]
            try:
                st = w.statement(r["statement"])
            except ToyError:
                continue
            n_stmt += 1
            for f in sorted((symbols(st.lhs) | symbols(st.rhs)) & set(w.funcs)):
                occ.setdefault(shape_of(st, f, proj, naming), set()).add((proj, f))
    return occ, n_stmt

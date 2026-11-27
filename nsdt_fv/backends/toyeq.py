"""ToyEq: the toy domain of plan Appendix C, with its exact checker.

Terms are immutable Python values:
  * a variable is a str (lower-case identifier);
  * an application is a tuple (head, arg1, ..., argn); nullary constructors are ('Z',).

Types: Nat (Z, S(n)), List of Nat (Nil, Cons(n, l)), Tree (Leaf, Node(t, n, t)), Bool (T, F).

A function is defined by oriented equations. It either has a single equation whose argument
patterns are all variables (a wrapper / alias), or it matches on exactly one argument k with one
equation per constructor of that argument's type, in declaration order; every other pattern is a
variable. Recursive calls may only call the function itself with, at position k, a recursive
constructor argument of the pattern (structural recursion); other calls go to earlier functions.
So rewriting with definitions terminates (C.1).

Proof language (C.2):
  proof ::= step* ; step ::= simp | refl | rw NAME | rw<- NAME | cases VAR block+ | induct VAR block+
  block ::= "{" proof "}"

Design decisions where C.2 leaves room (recorded in reports/ISSUES.md):
  * IH generalisation: the IH of 'induct x' is the current goal with x replaced by the recursive
    argument, universally quantified over the goal's other variables that occur free in no
    hypothesis. 'induct x' is refused if x occurs free in a hypothesis (otherwise the IH would
    have to carry that hypothesis as a premise).
  * 'rw' rewrites one occurrence (the leftmost-outermost match) and fails if the instantiated
    right side would contain an unbound pattern variable. Pattern variables only match subterms
    of their own type.
  * The recursive argument of a constructor keeps the name of the variable that was split; other
    arguments get fresh names (Cons: h; Node: <x>l, v, <x>r; S keeps x).
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from dataclasses import dataclass, field

from .base import Verdict, EvalResult

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

# ----------------------------------------------------------------------------- types
CTORS = {
    "Z": ((), "Nat"), "S": (("Nat",), "Nat"),
    "Nil": ((), "List"), "Cons": (("Nat", "List"), "List"),
    "Leaf": ((), "Tree"), "Node": (("Tree", "Nat", "Tree"), "Tree"),
    "T": ((), "Bool"), "F": ((), "Bool"),
}
TYPES = {"Nat": ("Z", "S"), "List": ("Nil", "Cons"), "Tree": ("Leaf", "Node"), "Bool": ("T", "F")}
RESERVED = {"forall", "simp", "refl", "rw", "cases", "induct", "def", "lemma", "sorry", "admit",
            "axiom", "theorem", "by", "before"} | set(CTORS) | set(TYPES)
SIMP_FUEL = 10_000
MAX_STEPS = 40


class ToyError(Exception):
    """Ill-formed input (parse or type error)."""


class StepError(Exception):
    def __init__(self, text, tok=None, goal=None):
        super().__init__(text)
        self.text = text
        self.tok = tok
        self.goal = goal


class TimeoutHit(Exception):
    pass


class FuelOut(Exception):
    pass


# ----------------------------------------------------------------------------- printing
def show(t) -> str:
    if isinstance(t, str):
        return t
    if len(t) == 1:
        return t[0]
    return t[0] + "(" + ", ".join(show(a) for a in t[1:]) + ")"


def term_size(t) -> int:
    if isinstance(t, str):
        return 1
    return 1 + sum(term_size(a) for a in t[1:])


def term_vars(t, acc=None):
    if acc is None:
        acc = []
    if isinstance(t, str):
        if t not in acc:
            acc.append(t)
    else:
        for a in t[1:]:
            term_vars(a, acc)
    return acc


def symbols(t, acc=None):
    """Head symbols (functions and constructors) occurring in t."""
    if acc is None:
        acc = set()
    if isinstance(t, tuple):
        acc.add(t[0])
        for a in t[1:]:
            symbols(a, acc)
    return acc


def subst(t, s):
    if isinstance(t, str):
        return s.get(t, t)
    if len(t) == 1:
        return t
    return (t[0],) + tuple(subst(a, s) for a in t[1:])


# ----------------------------------------------------------------------------- tokens
_TOK = re.compile(r"(rw<-)|([A-Za-z_][A-Za-z0-9_']*)|(\S)")


@dataclass(frozen=True)
class Tok:
    text: str
    line: int
    col: int


def tokenize(text: str) -> list:
    out = []
    for ln, line in enumerate(text.split("\n"), 1):
        code = line.split("--", 1)[0]          # '--' starts a comment
        for m in _TOK.finditer(code):
            out.append(Tok(m.group(0), ln, m.start() + 1))
    return out


def _is_ident(s):
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", s))


class _P:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j].text if j < len(self.toks) else None

    def tok(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        if self.i >= len(self.toks):
            raise ToyError("unexpected end of input")
        t = self.toks[self.i]
        self.i += 1
        return t.text

    def expect(self, s):
        got = self.next()
        if got != s:
            raise ToyError(f"expected {s!r}, got {got!r}")

    def done(self):
        return self.i >= len(self.toks)


def _parse_term(p: _P):
    name = p.next()
    if not _is_ident(name):
        raise ToyError(f"expected a term, got {name!r}")
    if p.peek() == "(":
        p.next()
        args = [_parse_term(p)]
        while p.peek() == ",":
            p.next()
            args.append(_parse_term(p))
        p.expect(")")
        if name in CTORS and len(CTORS[name][0]) != len(args):
            raise ToyError(f"constructor {name} takes {len(CTORS[name][0])} arguments")
        return (name,) + tuple(args)
    if name in CTORS:
        if CTORS[name][0]:
            raise ToyError(f"constructor {name} needs arguments")
        return (name,)
    if name[0].isupper() or name in RESERVED:
        raise ToyError(f"bad variable name {name!r}")
    return name


def parse_term(text: str):
    p = _P(tokenize(text))
    t = _parse_term(p)
    if not p.done():
        raise ToyError(f"trailing input after term: {p.peek()!r}")
    return t


# ----------------------------------------------------------------------------- definitions
@dataclass
class FunDef:
    name: str
    argtypes: tuple
    rettype: str
    k: int | None                     # index of the matched argument, or None for a wrapper
    params: tuple = ()                # wrapper: parameter variables
    rhs: object = None                # wrapper: right side
    cases: dict = field(default_factory=dict)   # ctor -> (param vars with None at k, ctor arg vars, rhs)
    eqs: list = field(default_factory=list)     # [(lhs_term, rhs_term)] in declaration order

    def reduce(self, args):
        """One definitional rewrite of name(args) at the root, or None if stuck."""
        if self.k is None:
            return subst(self.rhs, dict(zip(self.params, args)))
        a = args[self.k]
        if not isinstance(a, tuple):
            return None
        case = self.cases.get(a[0])
        if case is None:
            return None
        pvars, cvars, rhs = case
        s = {}
        for v, x in zip(pvars, args):
            if v is not None:
                s[v] = x
        for v, x in zip(cvars, a[1:]):
            s[v] = x
        return subst(rhs, s)

    def source(self) -> str:
        lines = [f"def {self.name} : {' '.join(self.argtypes)} -> {self.rettype}"]
        for l, r in self.eqs:
            lines.append(f"  {show(l)} = {show(r)}")
        return "\n".join(lines)


def _typeof(t, env, funcs):
    if isinstance(t, str):
        if t not in env:
            raise ToyError(f"unknown variable {t}")
        return env[t]
    h = t[0]
    if h in CTORS:
        ats, rt = CTORS[h]
    elif h in funcs:
        ats, rt = funcs[h].argtypes, funcs[h].rettype
    else:
        raise ToyError(f"unknown function {h}")
    if len(ats) != len(t) - 1:
        raise ToyError(f"{h} expects {len(ats)} arguments, got {len(t) - 1}")
    return rt


def typecheck(t, env, funcs):
    """Return the type of t, checking every argument; raise ToyError when ill typed."""
    rt = _typeof(t, env, funcs)
    if isinstance(t, tuple):
        ats = CTORS[t[0]][0] if t[0] in CTORS else funcs[t[0]].argtypes
        for a, at in zip(t[1:], ats):
            got = typecheck(a, env, funcs)
            if got != at:
                raise ToyError(f"argument {show(a)} of {t[0]} has type {got}, expected {at}")
    return rt


def infer_var_types(terms, funcs, known=None):
    """Infer variable types from argument positions. Raises on conflict."""
    env = dict(known or {})

    def walk(t):
        if isinstance(t, str):
            return
        h = t[0]
        if h in CTORS:
            ats = CTORS[h][0]
        elif h in funcs:
            ats = funcs[h].argtypes
        else:
            raise ToyError(f"unknown function {h}")
        if len(ats) != len(t) - 1:
            raise ToyError(f"{h} expects {len(ats)} arguments, got {len(t) - 1}")
        for a, at in zip(t[1:], ats):
            if isinstance(a, str):
                if a in env and env[a] != at:
                    raise ToyError(f"variable {a} used at types {env[a]} and {at}")
                env[a] = at
            else:
                walk(a)

    for t in terms:
        walk(t)
    if len(terms) == 2:            # a bare-variable side takes the type of the other side
        l, r = terms
        for x, y in ((l, r), (r, l)):
            if isinstance(x, str) and x not in env:
                try:
                    env[x] = _typeof(y, env, funcs)
                except ToyError:
                    pass
    return env


def make_fundef(name, argtypes, rettype, eqs, funcs) -> FunDef:
    """Validate and build a definition. 'funcs' holds the earlier functions only."""
    if not re.fullmatch(r"[a-z][A-Za-z0-9_]*", name) or name in RESERVED:
        raise ToyError(f"bad function name {name!r}")
    if name in funcs:
        raise ToyError(f"duplicate function {name}")
    if not argtypes:
        raise ToyError("functions need at least one argument")
    for ty in list(argtypes) + [rettype]:
        if ty not in TYPES:
            raise ToyError(f"unknown type {ty}")
    argtypes = tuple(argtypes)
    me = FunDef(name, argtypes, rettype, None)
    scope = dict(funcs)
    scope[name] = me
    if not eqs:
        raise ToyError(f"{name}: no equations")
    ks = set()
    for lhs, rhs in eqs:
        if not isinstance(lhs, tuple) or lhs[0] != name or len(lhs) - 1 != len(argtypes):
            raise ToyError(f"{name}: bad left side {show(lhs)}")
        for i, pat in enumerate(lhs[1:]):
            if isinstance(pat, tuple):
                ks.add(i)
    if len(ks) > 1:
        raise ToyError(f"{name}: patterns may match on one argument only")
    k = ks.pop() if ks else None
    me.k = k
    me.eqs = list(eqs)
    if k is None:
        if len(eqs) != 1:
            raise ToyError(f"{name}: overlapping equations")
        lhs, rhs = eqs[0]
        params = lhs[1:]
        if len(set(params)) != len(params):
            raise ToyError(f"{name}: non-linear pattern")
        env = dict(zip(params, argtypes))
        _check_rhs(name, rhs, env, scope, None, ())
        me.params, me.rhs = tuple(params), rhs
        return me
    mty = argtypes[k]
    ctors = TYPES[mty]
    if len(eqs) != len(ctors):
        raise ToyError(f"{name}: need one equation per constructor of {mty} (exhaustive, non-overlapping)")
    for (lhs, rhs), c in zip(eqs, ctors):
        pat = lhs[1 + k]
        if not isinstance(pat, tuple) or pat[0] != c:
            raise ToyError(f"{name}: equations must cover {', '.join(ctors)} in declaration order")
        cvars = pat[1:]
        if any(not isinstance(v, str) for v in cvars):
            raise ToyError(f"{name}: nested patterns are not allowed")
        pvars = [None if i == k else v for i, v in enumerate(lhs[1:])]
        if any(not isinstance(v, str) for v in pvars if v is not None):
            raise ToyError(f"{name}: only argument {k} may be a constructor pattern")
        allv = [v for v in pvars if v is not None] + list(cvars)
        if len(set(allv)) != len(allv):
            raise ToyError(f"{name}: non-linear pattern")
        env = {v: argtypes[i] for i, v in enumerate(pvars) if v is not None}
        for v, ct in zip(cvars, CTORS[c][0]):
            env[v] = ct
        rec = tuple(v for v, ct in zip(cvars, CTORS[c][0]) if ct == mty)
        _check_rhs(name, rhs, env, scope, k, rec)
        me.cases[c] = (tuple(pvars), tuple(cvars), rhs)
    return me


def _check_rhs(name, rhs, env, scope, k, rec):
    for v in term_vars(rhs):
        if v not in env:
            raise ToyError(f"{name}: right side uses unbound variable {v}")
    ty = typecheck(rhs, env, scope)
    if ty != scope[name].rettype:
        raise ToyError(f"{name}: right side has type {ty}, expected {scope[name].rettype}")

    def walk(t):
        if isinstance(t, str):
            return
        if t[0] == name:
            if k is None:
                raise ToyError(f"{name}: a wrapper may not call itself")
            if t[1 + k] not in rec:
                raise ToyError(f"{name}: recursion must be structural in argument {k}")
        for a in t[1:]:
            walk(a)
    walk(rhs)


# ----------------------------------------------------------------------------- statements
@dataclass(frozen=True)
class Statement:
    vars: tuple          # ((name, type), ...) in binder order
    lhs: object
    rhs: object

    def show(self) -> str:
        vs = " ".join(v for v, _ in self.vars)
        body = f"{show(self.lhs)} = {show(self.rhs)}"
        return f"forall {vs}. {body}" if vs else body

    def body(self) -> str:
        return f"{show(self.lhs)} = {show(self.rhs)}"

    def env(self):
        return dict(self.vars)


def parse_statement(text: str, funcs) -> Statement:
    p = _P(tokenize(text))
    st = _parse_statement(p, funcs)
    if not p.done():
        raise ToyError(f"trailing input after statement: {p.peek()!r}")
    return st


def _parse_statement(p, funcs):
    names = []
    if p.peek() == "forall":
        p.next()
        while p.peek() != ".":
            v = p.next()
            if not _is_ident(v) or v[0].isupper() or v in RESERVED or v in funcs:
                raise ToyError(f"bad bound variable {v!r}")
            if v in names:
                raise ToyError(f"duplicate bound variable {v}")
            names.append(v)
        p.expect(".")
    lhs = _parse_term(p)
    p.expect("=")
    rhs = _parse_term(p)
    free = term_vars(lhs) + [v for v in term_vars(rhs) if v not in term_vars(lhs)]
    for v in free:
        if v in funcs:
            raise ToyError(f"{v} is a function, not a variable")
        if v not in names:
            raise ToyError(f"variable {v} is not bound by forall")
    env = infer_var_types([lhs, rhs], funcs)
    for v in names:
        if v not in env:
            raise ToyError(f"cannot determine the type of {v}")
    tl = typecheck(lhs, env, funcs)
    tr = typecheck(rhs, env, funcs)
    if tl != tr:
        raise ToyError(f"sides have different types {tl} and {tr}")
    return Statement(tuple((v, env[v]) for v in names), lhs, rhs)


_NAME_POOLS = {"Nat": ["n", "m", "k", "i", "j", "p", "q"], "List": ["xs", "ys", "zs", "ws", "us", "vs"],
               "Tree": ["t", "u", "s", "r"], "Bool": ["b", "c", "d", "e"]}


def canonical(st: Statement, flip=False) -> Statement:
    """Alpha-normalise: variables renamed by type and order of first occurrence."""
    lhs, rhs = (st.rhs, st.lhs) if flip else (st.lhs, st.rhs)
    order = term_vars(lhs)
    for v in term_vars(rhs):
        if v not in order:
            order.append(v)
    env = st.env()
    for v, _ in st.vars:        # unused binders last
        if v not in order:
            order.append(v)
    used = {ty: 0 for ty in TYPES}
    ren = {}
    for v in order:
        ty = env[v]
        i = used[ty]
        used[ty] += 1
        pool = _NAME_POOLS[ty]
        ren[v] = pool[i] if i < len(pool) else f"{pool[0]}{i}"
    return Statement(tuple((ren[v], env[v]) for v in order), subst(lhs, ren), subst(rhs, ren))


def statement_key(st: Statement) -> str:
    """Orientation- and alpha-insensitive key (used for dedup and for rule L3)."""
    a = canonical(st).show()
    b = canonical(st, flip=True).show()
    return min(a, b)


# ----------------------------------------------------------------------------- proofs
def parse_proof(text: str):
    """Parse proof text into steps. Each step: (kind, arg, blocks, tok)."""
    toks = tokenize(text)
    p = _P(toks)
    steps = _parse_steps(p, top=True)
    if not p.done():
        raise ToyError(f"unexpected {p.peek()!r}")
    return steps


def _parse_steps(p, top):
    steps = []
    while not p.done() and p.peek() != "}":
        tk = p.tok()
        w = p.next()
        if w in ("simp", "refl"):
            steps.append((w, None, None, tk))
        elif w in ("rw", "rw<-"):
            name = p.next()
            if not _is_ident(name):
                raise ToyError(f"rw needs a name, got {name!r}")
            steps.append((w, name, None, tk))
        elif w in ("cases", "induct"):
            var = p.next()
            if not _is_ident(var):
                raise ToyError(f"{w} needs a variable, got {var!r}")
            blocks = []
            while p.peek() == "{":
                p.next()
                blocks.append(_parse_steps(p, top=False))
                p.expect("}")
            if not blocks:
                raise ToyError(f"{w} {var} needs at least one block")
            steps.append((w, var, blocks, tk))
        else:
            raise ToyError(f"unknown step {w!r} at line {tk.line}, column {tk.col}")
    if top and not p.done():
        raise ToyError(f"unbalanced '}}' at line {p.tok().line}")
    return steps


def count_steps(steps) -> int:
    n = 0
    for kind, _, blocks, _ in steps:
        n += 1
        if blocks:
            n += sum(count_steps(b) for b in blocks)
    return n


def show_proof(steps) -> str:
    out = []
    for kind, arg, blocks, _ in steps:
        if kind in ("simp", "refl"):
            out.append(kind)
        elif kind in ("rw", "rw<-"):
            out.append(f"{kind} {arg}")
        else:
            out.append(f"{kind} {arg} " + " ".join("{ " + show_proof(b) + " }" if b else "{ }" for b in blocks))
    return " ".join(out)


def proof_tokens(text: str) -> list:
    return [t.text for t in tokenize(text)]


@dataclass(frozen=True)
class Hyp:
    name: str
    bound: tuple          # ((var, type), ...) universally quantified (pattern) variables
    lhs: object
    rhs: object

    def free_vars(self):
        b = {v for v, _ in self.bound}
        return [v for v in term_vars(self.lhs) + term_vars(self.rhs) if v not in b]

    def show(self):
        body = f"{show(self.lhs)} = {show(self.rhs)}"
        if self.bound:
            return f"{self.name} : forall {' '.join(v for v, _ in self.bound)}. {body}"
        return f"{self.name} : {body}"


@dataclass(frozen=True)
class Goal:
    lhs: object
    rhs: object
    env: tuple            # ((var, type), ...) free variables in scope
    hyps: tuple = ()      # Hyp, most recent last

    def show(self):
        hs = "; ".join(h.show() for h in self.hyps)
        g = f"{show(self.lhs)} = {show(self.rhs)}"
        return (hs + " |- " + g) if hs else g

    def envd(self):
        return dict(self.env)


class Fuel:
    __slots__ = ("left", "deadline")

    def __init__(self, n, deadline):
        self.left = n
        self.deadline = deadline

    def tick(self):
        self.left -= 1
        if self.left < 0:
            raise FuelOut()
        if (self.left & 255) == 0 and time.monotonic() > self.deadline:
            raise TimeoutHit()


class World:
    """One snapshot: definitions in order, and a library of lemmas in order."""

    def __init__(self, snapshot_id: str = "anon"):
        self.snapshot_id = snapshot_id
        self.funcs: dict = {}
        self.lemmas: list = []           # (name, Statement)
        self.lemma_pos: dict = {}
        self.proofs: dict = {}           # name -> proof text (ground truth; may be absent)

    def add_def(self, fd: FunDef):
        if fd.name in self.funcs:
            raise ToyError(f"duplicate function {fd.name}")
        self.funcs[fd.name] = fd

    def define(self, name, argtypes, rettype, eqs_text):
        eqs = []
        for line in eqs_text:
            l, r = line.split("=", 1)
            eqs.append((parse_term(l), parse_term(r)))
        fd = make_fundef(name, argtypes, rettype, eqs, self.funcs)
        self.add_def(fd)
        return fd

    def add_lemma(self, name, st: Statement, proof: str | None = None):
        if name in self.lemma_pos or name in self.funcs or re.fullmatch(r"IH\d*", name):
            raise ToyError(f"bad or duplicate lemma name {name}")
        self.lemma_pos[name] = len(self.lemmas)
        self.lemmas.append((name, st))
        if proof is not None:
            self.proofs[name] = proof

    def statement(self, text: str) -> Statement:
        return parse_statement(text, self.funcs)

    def source(self, upto: int | None = None, with_proofs=False, hide=()):
        out = [fd.source() for fd in self.funcs.values()]
        for name, st in self.lemmas[: upto if upto is not None else len(self.lemmas)]:
            out.append(f"lemma {name} : {st.show()}")
            if with_proofs and name in self.proofs and name not in hide:
                out.append(f"  by {self.proofs[name]}")
        return "\n".join(out)

    def normalize(self, t, fuel, memo):
        if isinstance(t, str):
            return t
        r = memo.get(t)
        if r is not None:
            return r
        if len(t) > 1:
            args = tuple(self.normalize(a, fuel, memo) for a in t[1:])
            t2 = (t[0],) + args
        else:
            args, t2 = (), t
        fd = self.funcs.get(t[0])
        res = t2
        if fd is not None:
            red = fd.reduce(args)
            if red is not None:
                fuel.tick()
                res = self.normalize(red, fuel, memo)
        memo[t] = res
        return res


# ----------------------------------------------------------------------------- the checker core
def _match(pat, t, bound, s, env, funcs):
    if isinstance(pat, str):
        if pat in bound:
            if pat in s:
                return s[pat] == t
            if _typeof(t, env, funcs) != bound[pat]:
                return False
            s[pat] = t
            return True
        return pat == t
    if not isinstance(t, tuple) or t[0] != pat[0] or len(t) != len(pat):
        return False
    for a, b in zip(pat[1:], t[1:]):
        if not _match(a, b, bound, s, env, funcs):
            return False
    return True


def _replace_at(t, path, new):
    if not path:
        return new
    i = path[0]
    return t[:i] + (_replace_at(t[i], path[1:], new),) + t[i + 1:]


def _positions(t, path=()):
    yield path, t
    if isinstance(t, tuple):
        for i in range(1, len(t)):
            yield from _positions(t[i], path + (i,))


def rewrite_once(goal: Goal, bound: dict, l, r, funcs):
    """Rewrite the leftmost-outermost subterm (left side of the goal first) matching l with r.
    Returns the new goal, or None if nothing matches. Raises StepError on unbound variables."""
    env = goal.envd()
    for side in (0, 1):
        t = goal.lhs if side == 0 else goal.rhs
        for path, sub in _positions(t):
            s = {}
            if _match(l, sub, bound, s, env, funcs):
                extra = [v for v in term_vars(r) if v in bound and v not in s]
                if extra:
                    raise StepError(f"rw: right side has variables not fixed by the match: {', '.join(extra)}")
                new = _replace_at(t, path, subst(r, s))
                if side == 0:
                    return Goal(new, goal.rhs, goal.env, goal.hyps)
                return Goal(goal.lhs, new, goal.env, goal.hyps)
    return None


def _fresh(base, taken):
    if base not in taken:
        return base
    i = 1
    while f"{base}{i}" in taken:
        i += 1
    return f"{base}{i}"


def _hyp_names(hyps):
    out = set()
    for h in hyps:
        out.update(v for v, _ in h.bound)
        out.update(h.free_vars())
    return out


def _subst_hyp(h: Hyp, s):
    b = {v for v, _ in h.bound}
    s2 = {k: v for k, v in s.items() if k not in b}
    return Hyp(h.name, h.bound, subst(h.lhs, s2), subst(h.rhs, s2))


def split_goal(goal: Goal, x: str, induct: bool, funcs):
    """Subgoals of cases/induct on x, one per constructor in declaration order."""
    env = goal.envd()
    if x not in env:
        raise StepError(f"unknown variable {x}")
    ty = env[x]
    if induct:
        for h in goal.hyps:
            if x in h.free_vars():
                raise StepError(f"induct {x}: {x} occurs in hypothesis {h.name}")
    taken = set(env) | _hyp_names(goal.hyps) | set(funcs)
    taken.discard(x)
    lv = term_vars(goal.lhs)
    goal_vars = lv + [v for v in term_vars(goal.rhs) if v not in lv]
    hyp_free = set()
    for h in goal.hyps:
        hyp_free.update(h.free_vars())
    gen = tuple((v, env[v]) for v in goal_vars if v != x and v not in hyp_free)
    subgoals = []
    for c in TYPES[ty]:
        cts = CTORS[c][0]
        rec_idx = [i for i, ct in enumerate(cts) if ct == ty]
        names = []
        local = set(taken)
        for i, ct in enumerate(cts):
            if ct == ty and len(rec_idx) == 1:
                base = x
            elif ct == ty:
                base = x + ("l" if i == rec_idx[0] else "r")
            elif c == "Cons":
                base = "h"
            elif c == "Node":
                base = "v"
            else:
                base = "a"
            nm = _fresh(base, local)
            local.add(nm)
            names.append(nm)
        rep = (c,) + tuple(names)
        s = {x: rep}
        new_env = tuple((v, t) for v, t in goal.env if v != x) + tuple(zip(names, cts))
        hyps = tuple(_subst_hyp(h, s) for h in goal.hyps)
        if induct and rec_idx:
            ihs = []
            for j, i in enumerate(rec_idx):
                nm = "IH" if len(rec_idx) == 1 else f"IH{j + 1}"
                ss = {x: names[i]}
                ihs.append(Hyp(nm, gen, subst(goal.lhs, ss), subst(goal.rhs, ss)))
            hyps = hyps + tuple(ihs)
        subgoals.append(Goal(subst(goal.lhs, s), subst(goal.rhs, s), new_env, hyps))
    return subgoals


def find_equation(name, goal: Goal, world: World, avail: int):
    """Return (bound dict, lhs, rhs) for a hypothesis (most recent first) or an available lemma."""
    for h in reversed(goal.hyps):
        if h.name == name:
            return dict(h.bound), h.lhs, h.rhs
    pos = world.lemma_pos.get(name)
    if pos is None:
        raise StepError(f"unknown identifier {name}")
    if pos >= avail:
        raise StepError(f"{name} is not available here (it does not precede the current lemma)")
    st = world.lemmas[pos][1]
    return dict(st.vars), st.lhs, st.rhs


def simp_goal(goal: Goal, world: World, deadline):
    fuel = Fuel(SIMP_FUEL, deadline)
    memo = {}
    try:
        l = world.normalize(goal.lhs, fuel, memo)
        r = world.normalize(goal.rhs, fuel, memo)
    except FuelOut:
        raise StepError(f"simp: fuel limit of {SIMP_FUEL} rewrites exceeded")
    return Goal(l, r, goal.env, goal.hyps)


def apply_rw(goal, name, backwards, world, avail):
    bound, l, r = find_equation(name, goal, world, avail)
    if backwards:
        l, r = r, l
    new = rewrite_once(goal, bound, l, r, world.funcs)
    if new is None:
        raise StepError(f"rw{'<-' if backwards else ''} {name}: no subterm matches {show(l)}")
    return new


def run_steps(steps, goal, world, avail, deadline, trace=None):
    """Run steps on goal. Raises StepError unless the goal ends closed.
    If trace is a list, (step kind, argument, goal before the step) is appended for every step."""
    closed = False
    for kind, arg, blocks, tk in steps:
        if time.monotonic() > deadline:
            raise TimeoutHit()
        if trace is not None:
            trace.append((kind, arg, goal))
        if closed:
            raise StepError("step after the goal was closed", tk, None)
        try:
            if kind == "simp":
                goal = simp_goal(goal, world, deadline)
            elif kind == "refl":
                if goal.lhs != goal.rhs:
                    raise StepError("refl: the two sides differ")
                closed = True
            elif kind in ("rw", "rw<-"):
                goal = apply_rw(goal, arg, kind == "rw<-", world, avail)
            else:
                subs = split_goal(goal, arg, kind == "induct", world.funcs)
                if len(blocks) != len(subs):
                    raise StepError(f"{kind} {arg}: expected {len(subs)} blocks, got {len(blocks)}")
                for b, sg in zip(blocks, subs):
                    run_steps(b, sg, world, avail, deadline, trace)
                closed = True
        except StepError as e:
            if e.tok is None:
                e.tok = tk
            if e.goal is None:
                e.goal = goal
            raise
    if not closed:
        raise StepError("unsolved goal", steps[-1][3] if steps else None, goal)


def check_proof(world: World, st: Statement, proof_text: str, avail: int, timeout_s: float = 120,
                trace=None) -> Verdict:
    """The exact checker of C.2. Deterministic; acceptance only through run_steps."""
    t0 = time.monotonic()
    deadline = t0 + timeout_s

    def v(cat, text="", tok=None, goal=None):
        msgs = []
        if text:
            msgs.append({"severity": "error", "text": text,
                         "line": tok.line if tok else 0, "column": tok.col if tok else 0,
                         "goal_state": goal.show() if goal is not None else ""})
        return Verdict(cat == "ok", cat, msgs, time.monotonic() - t0)

    toks = proof_tokens(proof_text)
    if any(w in ("sorry", "admit") for w in toks):
        return v("sorry", "proof contains sorry/admit")
    if "axiom" in toks:
        return v("axiom", "proof declares an axiom")
    try:
        steps = parse_proof(proof_text)
    except ToyError as e:
        return v("error", f"parse error: {e}")
    n = count_steps(steps)
    if n > MAX_STEPS:
        return v("error", f"proof has {n} steps; the limit is {MAX_STEPS}")
    goal = Goal(st.lhs, st.rhs, st.vars, ())
    try:
        run_steps(steps, goal, world, avail, deadline, trace)
    except StepError as e:
        return v("error", e.text, e.tok, e.goal)
    except TimeoutHit:
        return v("timeout", f"timeout after {timeout_s}s")
    return v("ok")


def split_header(candidate: str):
    """A candidate may start with 'theorem <statement> :=' ; returns (statement_text|None, proof_text)."""
    s = candidate.strip()
    if s.startswith("theorem") and ":=" in s:
        head, proof = s.split(":=", 1)
        return head[len("theorem"):].strip(), proof
    return None, candidate


def sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


# ----------------------------------------------------------------------------- backend
class ToyEqBackend:
    """CheckerBackend over ToyEq snapshots (A.3).

    worlds: {snapshot_id: World}; obligations: {obligation_id: (snapshot_id, lemma name)}.
    The obligation's statement is the lemma's statement; available lemmas are those that precede it
    in the snapshot's library order.
    """

    def __init__(self, worlds: dict, obligations: dict):
        self.worlds = dict(worlds)
        self.obligations = dict(obligations)

    def snapshots(self) -> list:
        return sorted(self.worlds)

    def _world(self, snapshot_id) -> World:
        if snapshot_id not in self.worlds:
            raise KeyError(f"unknown snapshot {snapshot_id}")
        return self.worlds[snapshot_id]

    @staticmethod
    def _err(cat, text, t0=None):
        el = time.monotonic() - t0 if t0 else 0.0
        return Verdict(False, cat, [{"severity": "error", "text": text, "line": 0, "column": 0,
                                     "goal_state": ""}], el)

    def check(self, snapshot_id, obligation_id, candidate, timeout_s):
        w = self._world(snapshot_id)
        _, name = self.obligations[obligation_id]
        if name not in w.lemma_pos:
            raise KeyError(f"obligation {obligation_id} is not in snapshot {snapshot_id}")
        pos = w.lemma_pos[name]
        st = w.lemmas[pos][1]
        head, proof = split_header(candidate)
        if head is not None:
            try:
                st2 = parse_statement(head, w.funcs)
            except ToyError as e:
                return self._err("changed_statement", f"statement: {e}")
            if canonical(st2).show() != canonical(st).show():
                return self._err("changed_statement", "the candidate changes the statement")
        return check_proof(w, st, proof, pos, timeout_s)

    def _avail(self, w: World, context: str) -> int:
        ctx = (context or "").strip()
        if not ctx:
            return len(w.lemmas)
        m = re.fullmatch(r"before\s+(\S+)", ctx)
        if m:
            if m.group(1) not in w.lemma_pos:
                raise ToyError(f"context: unknown lemma {m.group(1)}")
            return w.lemma_pos[m.group(1)]
        raise ToyError(f"context not understood: {ctx!r} (use '' or 'before NAME')")

    def check_statement(self, snapshot_id, context, statement, candidate, timeout_s):
        w = self._world(snapshot_id)
        t0 = time.monotonic()
        try:
            avail = self._avail(w, context)
            st = parse_statement(statement, w.funcs)
        except ToyError as e:
            return self._err("error", f"statement: {e}", t0)
        head, proof = split_header(candidate)
        if head is not None:
            try:
                same = canonical(parse_statement(head, w.funcs)).show() == canonical(st).show()
            except ToyError:
                same = False
            if not same:
                return self._err("changed_statement", "the candidate changes the statement", t0)
        return check_proof(w, st, proof, avail, timeout_s)

    def evaluate_term(self, snapshot_id, context, term, timeout_s):
        w = self._world(snapshot_id)
        t0 = time.monotonic()
        try:
            t = parse_term(term)
            if term_vars(t):
                return EvalResult(False, None, "error", "term is not ground", time.monotonic() - t0)
            typecheck(t, {}, w.funcs)
            fuel = Fuel(10 ** 6, t0 + timeout_s)
            val = w.normalize(t, fuel, {})
            return EvalResult(True, show(val), "ok", "", time.monotonic() - t0)
        except ToyError as e:
            return EvalResult(False, None, "error", str(e), time.monotonic() - t0)
        except (FuelOut, TimeoutHit):
            return EvalResult(False, None, "timeout", "evaluation limit", time.monotonic() - t0)

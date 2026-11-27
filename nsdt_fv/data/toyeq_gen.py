"""Project and library generator for ToyEq (plan C.3).

A project is generated from a template: every function is renamed with a project prefix, wrapper
definitions that compose or alias earlier functions are added, the template's seed lemmas come
first, and further lemmas are derived by specialization, composition (rewriting one library
equation with another) and wrapper folding. A candidate enters the library only if it is true on
random ground instances (independent oracle) and the reference prover finds a proof, which the
checker then accepts. A lemma may use only earlier lemmas. The first window holds the first half of
the library; the wrappers of the second phase are new definitions of the second window.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from ..backends.toyeq import (CTORS, TYPES, Statement, ToyError, World, canonical, make_fundef,
                              parse_statement, parse_term, show, statement_key, subst, symbols,
                              term_size, term_vars, typecheck, _match, _positions, _replace_at)
from ..backends.toyeq_oracle import counterexample
from .toyeq_prover import RefProver
from .toyeq_templates import TEMPLATES


class GenerationError(Exception):
    pass


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


@dataclass
class LemmaRec:
    name: str
    statement: Statement
    proof: str
    origin: str          # seed | specialize | compose | fold
    parents: list
    phase: int           # 1 or 2


@dataclass
class Project:
    name: str
    template: str
    prefix: str
    seed: int
    world: World                          # full world (all definitions, all lemmas, proofs)
    defs_phase: dict                      # function name -> phase in which it was defined
    lemmas: list = field(default_factory=list)   # LemmaRec in library order
    cut: int = 0                          # lemmas[:cut] form window 1
    stats: dict = field(default_factory=dict)


def parse_defs(text: str):
    """Parse 'def name : A B -> R' blocks with indented equations into [(name, argtypes, rettype, eq_lines)]."""
    out = []
    cur = None
    for line in text.split("\n"):
        if not line.strip():
            continue
        m = re.match(r"def\s+(\w+)\s*:\s*(.*?)\s*->\s*(\w+)\s*$", line.strip())
        if m:
            cur = (m.group(1), m.group(2).split(), m.group(3), [])
            out.append(cur)
        else:
            cur[3].append(line.strip())
    return out


def _rename_tokens(text, mapping):
    return re.sub(r"[A-Za-z_][A-Za-z0-9_']*", lambda m: mapping.get(m.group(0), m.group(0)), text)


class Generator:
    def __init__(self, template: str, prefix: str, seed: int, target: int,
                 n_wrappers: int = 8, node_limit_seed: int = 60000, node_limit: int = 2500,
                 truth_instances: int = 60):
        self.tpl = TEMPLATES[template]
        self.template = template
        self.prefix = prefix
        self.rng = random.Random(seed)
        self.seed = seed
        self.target = target
        self.n_wrappers = n_wrappers
        self.node_limit_seed = node_limit_seed
        self.node_limit = node_limit
        self.truth_instances = truth_instances
        self.world = World(f"{prefix}full")
        self.defs_phase = {}
        self.lemmas = []
        self.keys = set()
        self.tried = set()
        self.stats = {"candidates": 0, "duplicates": 0, "false_by_test": 0, "unproved": 0,
                      "accepted": 0, "seed_failed": [], "trivial": 0, "false_attempted": 0,
                      "false_proved": 0, "ill_formed": 0}
        self.n_derived = 0

    # ------------------------------------------------------------------ definitions
    def add_template_defs(self):
        defs = parse_defs(self.tpl["defs"])
        self.base_names = [d[0] for d in defs]
        self.ren = {n: self.prefix + n for n in self.base_names}
        for name, ats, rt, eqs in defs:
            eqs2 = []
            for e in eqs:
                l, r = e.split("=", 1)
                eqs2.append((parse_term(_rename_tokens(l, self.ren)), parse_term(_rename_tokens(r, self.ren))))
            fd = make_fundef(self.ren[name], ats, rt, eqs2, self.world.funcs)
            self.world.add_def(fd)
            self.defs_phase[fd.name] = 1

    def add_wrappers(self, count, phase):
        funcs = list(self.world.funcs.values())
        made = 0
        attempts = 0
        while made < count and attempts < 200:
            attempts += 1
            f = self.rng.choice(funcs)
            params = []
            args = []
            nonvar = 0
            for at in f.argtypes:
                gs = [g for g in self.world.funcs.values() if g.rettype == at and g.name != f.name]
                if gs and self.rng.random() < 0.6:
                    g = self.rng.choice(gs)
                    sub = []
                    for gt in g.argtypes:
                        v = f"x{len(params)}"
                        params.append((v, gt))
                        sub.append(v)
                    args.append((g.name,) + tuple(sub))
                    nonvar += 1
                else:
                    v = f"x{len(params)}"
                    params.append((v, at))
                    args.append(v)
            body = (f.name,) + tuple(args)
            if len(params) > 3:
                continue
            name = f"{self.prefix}w{sum(1 for n in self.world.funcs if n.startswith(self.prefix + 'w'))}"
            lhs = (name,) + tuple(v for v, _ in params)
            try:
                fd = make_fundef(name, [t for _, t in params], f.rettype, [(lhs, body)], self.world.funcs)
            except ToyError:
                continue
            self.world.add_def(fd)
            self.defs_phase[name] = phase
            funcs.append(fd)
            made += 1

    # ------------------------------------------------------------------ lemmas
    def _name(self):
        self.n_derived += 1
        return f"{self.prefix}d{self.n_derived:03d}"

    def try_add(self, name, st: Statement, origin, parents, phase, node_limit, force_trivial_ok=True):
        self.stats["candidates"] += 1
        try:
            typecheck(st.lhs, st.env(), self.world.funcs)
        except ToyError:
            self.stats["ill_formed"] += 1
            return False
        if st.lhs == st.rhs:
            self.stats["duplicates"] += 1
            return False
        key = statement_key(st)
        if key in self.keys or key in self.tried:
            self.stats["duplicates"] += 1
            return False
        self.tried.add(key)
        cex = counterexample(self.world, st, self.truth_instances, seed=_stable_seed(key))
        if cex is not None:
            self.stats["false_by_test"] += 1
            # every 10th false candidate is still given to the prover: it must never succeed
            if self.stats["false_by_test"] % 10 == 1:
                self.stats["false_attempted"] += 1
                pr = RefProver(self.world, len(self.world.lemmas), node_limit=min(node_limit, 1500))
                if pr.prove(st) is not None:
                    self.stats["false_proved"] += 1
                    raise GenerationError(f"SOUNDNESS: reference prover proved a false statement {st.show()}")
            return False
        st = canonical(st)
        pr = RefProver(self.world, len(self.world.lemmas), node_limit=node_limit)
        proof = pr.prove(st)
        if proof is None:
            self.stats["unproved"] += 1
            return False
        if proof == "simp refl":
            if not force_trivial_ok and self.stats["trivial"] >= 0.25 * self.target:
                return False
            self.stats["trivial"] += 1
        self.world.add_lemma(name, st, proof)
        self.keys.add(key)
        self.lemmas.append(LemmaRec(name, st, proof, origin, parents, phase))
        self.stats["accepted"] += 1
        return True

    def add_seeds(self):
        for nm, text in self.tpl["seeds"]:
            st = parse_statement(_rename_tokens(text, self.ren), self.world.funcs)
            ok = self.try_add(self.prefix + nm, st, "seed", [], 1, self.node_limit_seed)
            if not ok:
                self.stats["seed_failed"].append(nm)

    # candidate generators -------------------------------------------------
    def _fresh_var(self, ty, taken):
        base = {"Nat": "n", "List": "xs", "Tree": "t", "Bool": "b"}[ty]
        i = 0
        while f"{base}{i}" in taken:
            i += 1
        taken.add(f"{base}{i}")
        return f"{base}{i}"

    def specialize(self, rec: LemmaRec):
        st = rec.statement
        if not st.vars:
            return None
        v, ty = self.rng.choice(st.vars)
        taken = {x for x, _ in st.vars}
        newvars = []
        r = self.rng.random()
        if r < 0.25:
            rep = (TYPES[ty][0],)
        elif r < 0.55:
            c = TYPES[ty][1]
            args = []
            for at in CTORS[c][0]:
                nv = self._fresh_var(at, taken)
                newvars.append((nv, at))
                args.append(nv)
            rep = (c,) + tuple(args)
        else:
            gs = [g for g in self.world.funcs.values() if g.rettype == ty]
            if not gs:
                return None
            g = self.rng.choice(gs)
            args = []
            for at in g.argtypes:
                nv = self._fresh_var(at, taken)
                newvars.append((nv, at))
                args.append(nv)
            rep = (g.name,) + tuple(args)
        s = {v: rep}
        vars_ = tuple(x for x in st.vars if x[0] != v) + tuple(newvars)
        return Statement(vars_, subst(st.lhs, s), subst(st.rhs, s))

    def compose(self, a: LemmaRec, b: LemmaRec):
        sa, sb = a.statement, b.statement
        # rename b's variables apart from a's
        taken = {x for x, _ in sa.vars}
        ren = {}
        for x, ty in sb.vars:
            ren[x] = self._fresh_var(ty, taken)
        bl, br = subst(sb.lhs, ren), subst(sb.rhs, ren)
        bound = {ren[x]: ty for x, ty in sb.vars}
        if self.rng.random() < 0.5:
            bl, br = br, bl
        if isinstance(bl, str):
            return None
        side = self.rng.randint(0, 1)
        t = sa.rhs if side == 0 else sa.lhs
        other = sa.lhs if side == 0 else sa.rhs
        env = sa.env()
        hits = []
        for path, sub in _positions(t):
            s = {}
            try:
                if _match(bl, sub, bound, s, env, self.world.funcs):
                    if all(x in s for x in term_vars(br) if x in bound):
                        hits.append((path, s))
            except ToyError:
                continue
        if not hits:
            return None
        path, s = self.rng.choice(hits)
        new = _replace_at(t, path, subst(br, s))
        used = set(term_vars(other)) | set(term_vars(new))
        vars_ = tuple(x for x in sa.vars if x[0] in used)
        if side == 0:
            return Statement(vars_, other, new)
        return Statement(vars_, new, other)

    def fold(self, rec: LemmaRec, wrappers):
        st = rec.statement
        env = st.env()
        self.rng.shuffle(wrappers)
        for w in wrappers:
            fd = self.world.funcs[w]
            bound = dict(zip(fd.params, fd.argtypes))
            for side in (0, 1):
                t = st.lhs if side == 0 else st.rhs
                for path, sub in _positions(t):
                    s = {}
                    if isinstance(sub, tuple) and _match(fd.rhs, sub, bound, s, env, self.world.funcs):
                        if not all(p in s for p in fd.params):
                            continue
                        new = _replace_at(t, path, (w,) + tuple(s[p] for p in fd.params))
                        if side == 0:
                            return Statement(st.vars, new, st.rhs)
                        return Statement(st.vars, st.lhs, new)
        return None

    def wrapper_intro(self, w):
        """A statement unfolding a wrapper applied to variables or constructor terms one step."""
        fd = self.world.funcs[w]
        taken = set()
        args = []
        vars_ = []
        for at in fd.argtypes:
            nv = self._fresh_var(at, taken)
            vars_.append((nv, at))
            args.append(nv)
        lhs = (w,) + tuple(args)
        rhs = subst(fd.rhs, dict(zip(fd.params, args)))
        return Statement(tuple(vars_), lhs, rhs)

    def grow(self, upto, phase):
        rounds = 0
        while len(self.lemmas) < upto and rounds < 60:
            rounds += 1
            wrappers = [n for n, p in self.defs_phase.items() if n.startswith(self.prefix + "w") and p <= phase]
            cands = []
            pool = self.lemmas[:]
            for _ in range(60):
                a = self.rng.choice(pool)
                r = self.rng.random()
                if r < 0.35:
                    st, origin, parents = self.specialize(a), "specialize", [a.name]
                elif r < 0.75:
                    b = self.rng.choice(pool)
                    st, origin, parents = self.compose(a, b), "compose", [a.name, b.name]
                elif wrappers and r < 0.95:
                    st, origin, parents = self.fold(a, list(wrappers)), "fold", [a.name]
                elif wrappers:
                    w = self.rng.choice(wrappers)
                    st, origin, parents = self.wrapper_intro(w), "fold", []
                else:
                    st = None
                if st is None:
                    continue
                if term_size(st.lhs) + term_size(st.rhs) > 22 or len(st.vars) > 4:
                    continue
                cands.append((st, origin, parents))
            for st, origin, parents in cands:
                if len(self.lemmas) >= upto:
                    break
                self.try_add(self._name(), st, origin, parents, phase, self.node_limit, force_trivial_ok=False)
        return len(self.lemmas)

    def build(self, name) -> Project:
        self.add_template_defs()
        self.add_wrappers(self.n_wrappers // 2, 1)
        self.add_seeds()
        half = self.target // 2
        self.grow(half, 1)
        cut = len(self.lemmas)
        self.add_wrappers(self.n_wrappers - self.n_wrappers // 2, 2)
        # second-phase wrappers get their unfolding lemmas first so they enter the library
        for w in [n for n, p in self.defs_phase.items() if p == 2]:
            self.try_add(self._name(), self.wrapper_intro(w), "fold", [], 2, self.node_limit)
        self.grow(self.target, 2)
        p = Project(name, self.template, self.prefix, self.seed, self.world, self.defs_phase,
                    self.lemmas, cut, self.stats)
        if not (150 <= len(self.lemmas) <= 250):
            raise GenerationError(f"{name}: library has {len(self.lemmas)} lemmas, outside 150..250 ({self.stats})")
        return p


def build_project(name, template, prefix, seed, target) -> Project:
    return Generator(template, prefix, seed, target).build(name)

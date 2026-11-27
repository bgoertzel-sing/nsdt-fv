"""Independent ground evaluator for ToyEq (used by test T1 and by the generators' truth filter).

It shares nothing with the checker's rewriting code: values are Python values
(Nat -> int, List -> tuple of int, Tree -> None | (left, int, right), Bool -> bool) and function
equations are interpreted by a big-step evaluator written separately from World.normalize. Only
the parsed definitions (patterns and right sides) are shared with the checker.
"""
from __future__ import annotations

import random

from .toyeq import Statement, World, term_vars


class EvalLimit(Exception):
    pass


def _ctor_of(ty, val):
    """(constructor name, child values) of a Python value of type ty."""
    if ty == "Nat":
        return ("Z", ()) if val == 0 else ("S", (val - 1,))
    if ty == "List":
        return ("Nil", ()) if len(val) == 0 else ("Cons", (val[0], val[1:]))
    if ty == "Tree":
        return ("Leaf", ()) if val is None else ("Node", val)
    if ty == "Bool":
        return ("T", ()) if val else ("F", ())
    raise ValueError(ty)


def _build(c, vals):
    if c == "Z":
        return 0
    if c == "S":
        return vals[0] + 1
    if c == "Nil":
        return ()
    if c == "Cons":
        return (vals[0],) + vals[1]
    if c == "Leaf":
        return None
    if c == "Node":
        return (vals[0], vals[1], vals[2])
    if c == "T":
        return True
    if c == "F":
        return False
    raise ValueError(c)


class Oracle:
    def __init__(self, world: World, budget: int = 200_000):
        self.w = world
        self.budget = budget

    def ev(self, t, env):
        self.steps += 1
        if self.steps > self.budget:
            raise EvalLimit()
        if isinstance(t, str):
            return env[t]
        h = t[0]
        vals = [self.ev(a, env) for a in t[1:]]
        fd = self.w.funcs.get(h)
        if fd is None:
            return _build(h, vals)
        if fd.k is None:
            return self.ev(fd.rhs, dict(zip(fd.params, vals)))
        c, kids = _ctor_of(fd.argtypes[fd.k], vals[fd.k])
        pvars, cvars, rhs = fd.cases[c]
        e = {}
        for v, x in zip(pvars, vals):
            if v is not None:
                e[v] = x
        for v, x in zip(cvars, kids):
            e[v] = x
        return self.ev(rhs, e)

    def holds(self, st: Statement, inst: dict) -> bool:
        self.steps = 0
        return self.ev(st.lhs, inst) == self.ev(st.rhs, inst)

    def eval_ground(self, t):
        self.steps = 0
        return self.ev(t, {})


def random_value(ty, rng: random.Random, size=3):
    if ty == "Nat":
        return rng.randint(0, size + 1)
    if ty == "Bool":
        return rng.random() < 0.5
    if ty == "List":
        return tuple(rng.randint(0, size) for _ in range(rng.randint(0, size + 1)))
    if ty == "Tree":
        if size <= 0 or rng.random() < 0.25:
            return None
        return (random_value("Tree", rng, size - 1), rng.randint(0, 3), random_value("Tree", rng, size - 1))
    raise ValueError(ty)


def random_instances(st: Statement, n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        size = 1 + (i % 4)
        out.append({v: random_value(ty, rng, size) for v, ty in st.vars})
    return out


def counterexample(world: World, st: Statement, n: int = 200, seed: int = 0):
    """First instance (of n random ground instances) on which st fails, or None.
    An instance whose evaluation exceeds the budget is reported as {'_limit': True}."""
    o = Oracle(world)
    for inst in random_instances(st, n, seed):
        try:
            if not o.holds(st, inst):
                return inst
        except EvalLimit:
            return {"_limit": True}
    return None


def value_to_term_text(ty, val):
    if ty == "Nat":
        s = "Z"
        for _ in range(val):
            s = f"S({s})"
        return s
    if ty == "Bool":
        return "T" if val else "F"
    if ty == "List":
        s = "Nil"
        for x in reversed(val):
            s = f"Cons({value_to_term_text('Nat', x)}, {s})"
        return s
    if ty == "Tree":
        if val is None:
            return "Leaf"
        l, x, r = val
        return f"Node({value_to_term_text('Tree', l)}, {value_to_term_text('Nat', x)}, {value_to_term_text('Tree', r)})"
    raise ValueError(ty)


__all__ = ["Oracle", "EvalLimit", "random_value", "random_instances", "counterexample",
           "value_to_term_text", "term_vars"]

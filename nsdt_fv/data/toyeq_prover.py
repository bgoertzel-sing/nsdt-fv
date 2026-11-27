"""Reference prover for ToyEq (plan C.3): iterative deepening over the step grammar, simp tried
first at every node, at most six other steps (rw, rw<-, cases, induct). Deterministic: limits are
node counts, never wall-clock time. Every proof it returns is re-checked by the checker."""
from __future__ import annotations

from ..backends.toyeq import (Goal, StepError, TimeoutHit, World, Statement, apply_rw, check_proof,
                              simp_goal, split_goal, symbols, term_vars)

_FAR = float("inf")


class _Out(Exception):
    pass


def _cost(steps):
    c = 0
    for s in steps:
        if isinstance(s, tuple):
            c += 1 + sum(_cost(b) for b in s[2])
        elif s not in ("simp", "refl"):
            c += 1
    return c


def render(steps) -> str:
    out = []
    for s in steps:
        if isinstance(s, tuple):
            kind, x, blocks = s
            out.append(f"{kind} {x} " + " ".join("{ " + render(b) + " }" for b in blocks))
        else:
            out.append(s)
    return " ".join(out)


class RefProver:
    def __init__(self, world: World, avail: int, max_other: int = 6, node_limit: int = 4000,
                 max_rw: int = 24, max_vars: int = 3):
        self.w = world
        self.avail = avail
        self.max_other = max_other
        self.node_limit = node_limit
        self.max_rw = max_rw
        self.max_vars = max_vars
        self.nodes = 0
        self.memo = {}
        # index available lemma directions by the head symbol of their (directed) left side
        self.index = {}
        for pos in range(avail):
            name, st = world.lemmas[pos]
            bound = st.env()
            for back, (l, r) in ((False, (st.lhs, st.rhs)), (True, (st.rhs, st.lhs))):
                if isinstance(l, str):
                    continue                      # a bare pattern variable matches everything
                if any(v not in term_vars(l) for v in term_vars(r)):
                    continue                      # would leave unbound variables
                syms = symbols(l) | symbols(r)
                self.index.setdefault(l[0], []).append((pos, name, back, syms))
        self._deadline = float("inf")

    def prove(self, st: Statement):
        goal = Goal(st.lhs, st.rhs, st.vars, ())
        self.nodes = 0
        self.memo = {}
        try:
            for b in range(self.max_other + 1):
                r = self._solve(goal, b, frozenset())
                if r is not None:
                    text = render(r)
                    v = check_proof(self.w, st, text, self.avail)
                    if not v.accepted:          # the checker is the only authority
                        raise AssertionError(f"reference prover produced a rejected proof: {text} {v}")
                    return text
        except _Out:
            return None
        return None

    def _rw_moves(self, goal: Goal):
        moves = []
        for h in reversed(goal.hyps):
            for back in (False, True):
                moves.append((h.name, back))
        gs = symbols(goal.lhs) | symbols(goal.rhs)
        cands = []
        for head in gs:
            for pos, name, back, syms in self.index.get(head, ()):
                overlap = len(syms & gs) - len(syms - gs)
                cands.append((-overlap, -pos, back, name))
        cands.sort()
        for _, _, back, name in cands[: self.max_rw]:
            moves.append((name, back))
        return moves

    def _split_vars(self, goal: Goal):
        env = goal.envd()
        hyp_free = set()
        for h in goal.hyps:
            hyp_free.update(h.free_vars())
        # prefer variables in matched (recursive) argument positions
        scored = []
        seen = set()

        def walk(t, depth):
            if isinstance(t, str):
                return
            fd = self.w.funcs.get(t[0])
            for i, a in enumerate(t[1:]):
                if isinstance(a, str) and a not in seen and fd is not None and fd.k == i:
                    seen.add(a)
                    scored.append(a)
                walk(a, depth + 1)
        walk(goal.lhs, 0)
        walk(goal.rhs, 0)
        for v in term_vars(goal.lhs) + term_vars(goal.rhs):
            if v not in seen:
                seen.add(v)
                scored.append(v)
        out = []
        for v in scored[: self.max_vars]:
            ty = env[v]
            if ty == "Bool" or v in hyp_free:
                out.append(("cases", v))
            else:
                out.append(("induct", v))
        return out

    def _solve(self, goal: Goal, budget: int, path):
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise _Out()
        key = (goal, budget)
        if key in self.memo:
            return self.memo[key]
        res = self._solve2(goal, budget, path)
        self.memo[key] = res
        return res

    def _solve2(self, goal, budget, path):
        if goal.lhs == goal.rhs:
            return ["refl"]
        pre = []
        try:
            g = simp_goal(goal, self.w, self._deadline)
        except (StepError, TimeoutHit):
            g = goal
        if g != goal:
            pre = ["simp"]
            goal = g
            if goal.lhs == goal.rhs:
                return ["simp", "refl"]
        if budget <= 0:
            return None
        path = path | {goal}
        for name, back in self._rw_moves(goal):
            try:
                g2 = apply_rw(goal, name, back, self.w, self.avail)
            except StepError:
                continue
            if g2 in path:
                continue
            r = self._solve(g2, budget - 1, path)
            if r is not None:
                return pre + [("rw<- " if back else "rw ") + name] + r
        for kind, x in self._split_vars(goal):
            try:
                subs = split_goal(goal, x, kind == "induct", self.w.funcs)
            except StepError:
                continue
            remaining = budget - 1
            blocks = []
            for sg in subs:
                found = None
                for b in range(0, remaining + 1):
                    found = self._solve(sg, b, path)
                    if found is not None:
                        break
                if found is None:
                    blocks = None
                    break
                remaining -= _cost(found)
                blocks.append(found)
            if blocks is not None:
                return pre + [(kind, x, blocks)]
        return None

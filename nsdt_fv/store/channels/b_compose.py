"""Channel B, dependency composition (plan section 4 B, C.5): "rewrites one library equation with another
and states k-step unfoldings of wrappers".

Input: the snapshot's library statements WITHOUT test lemmas (ChannelBase.private_world), its wrapper
definitions, and the window's non-test statements (for the relevance ranking only).
  * rewrite:  for library lemmas a (L = R) and b (l = r, either direction), and an occurrence of an
    instance s(l) in R (or in L), the statement L = R[s(r)] (or L[s(r)] = R);
  * unfold:   for a wrapper w(x1..xn) = body, the statement w(x1..xn) = body_k where body_k unfolds the
    wrapper calls of the right side k times (k = 1..k_max, while it changes).
Every candidate is proved at emission: constructed proofs (rw a / rw b orders, simp refl) are tried
first, then the reference prover on the test-free library; the proof is checked by
backend.check_statement on the real snapshot. Only statements with an accepted machine proof are
emitted (T10); the proof is logged and never trained on (section 4: "only the statement is kept").
Ranking: symbol overlap with the window's new statements (the toy reading of "overlap with the
dependency cones of the day's new statements"), non-trivial first. Output (A.3): rem_seeds.
"""
from __future__ import annotations

from ...backends.toyeq import (Statement, ToyError, _match, _positions, _replace_at, subst, symbols, term_vars)
from . import ChannelBase


def _show_stmt(lhs, rhs, env):
    vs = term_vars(lhs)
    for v in term_vars(rhs):
        if v not in vs:
            vs.append(v)
    return Statement(tuple((v, env[v]) for v in vs), lhs, rhs).show()


class ChannelB(ChannelBase):
    id = "B"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = set()

    def _fresh(self, st_vars, taken):
        ren, out = {}, {}
        for x, ty in st_vars:
            n, i = x + "_b", 0
            while n in taken:
                i += 1
                n = f"{x}_b{i}"
            taken.add(n)
            ren[x] = n
            out[n] = ty
        return ren, out

    def produce(self, stage, sid):
        c = self.cfg
        pw = self.private_world(sid)
        win = self.window(sid)
        wsyms = set()
        for o in win:
            st = pw.statement(o["statement"])
            wsyms |= (symbols(st.lhs) | symbols(st.rhs)) & set(pw.funcs)
        lem = list(pw.lemmas)
        # relevance: overlap with the window's symbols; a's then b's in that order
        def rel(st):
            return len((symbols(st.lhs) | symbols(st.rhs)) & wsyms)
        order = sorted(range(len(lem)), key=lambda i: (-rel(lem[i][1]), i))
        cands = {}                                     # key -> (text, origin, hints, rank, parents)
        max_c = int(c.get("max_candidates", 400))
        for ia in order:
            if len(cands) >= max_c:
                break
            na, sa = lem[ia]
            env = sa.env()
            for ib in order:
                if len(cands) >= max_c:
                    break
                nb, sb = lem[ib]
                if ib == ia:
                    continue
                taken = set(env)
                ren, bound = self._fresh(sb.vars, taken)
                bl0, br0 = subst(sb.lhs, ren), subst(sb.rhs, ren)
                for back, (bl, br) in ((False, (bl0, br0)), (True, (br0, bl0))):
                    if isinstance(bl, str):
                        continue
                    hit = None
                    for side in (1, 0):
                        t = sa.rhs if side == 1 else sa.lhs
                        for path, sub in _positions(t):
                            s = {}
                            try:
                                if _match(bl, sub, bound, s, env, pw.funcs) and all(x in s for x in term_vars(br) if x in bound):
                                    hit = (side, path, s)
                                    break
                            except ToyError:
                                continue
                        if hit:
                            break
                    if not hit:
                        continue
                    side, path, s = hit
                    t = sa.rhs if side == 1 else sa.lhs
                    new = _replace_at(t, path, subst(br, s))
                    lhs, rhs = (sa.lhs, new) if side == 1 else (new, sa.rhs)
                    if lhs == rhs:
                        continue
                    try:
                        text = _show_stmt(lhs, rhs, env)
                        k = self.key(sid, text)
                    except Exception:
                        continue
                    if k in cands:
                        continue
                    bdir = f"rw<- {nb}" if back else f"rw {nb}"
                    hints = [f"rw {na} {bdir} refl", f"{bdir} rw {na} refl", f"rw {na} simp {bdir} refl",
                             f"simp rw {na} {bdir} refl"]
                    cands[k] = (text, "rewrite", hints, rel(sa) + rel(sb), [na, nb])
        # k-step unfoldings of wrappers
        kmax = int(c.get("k_max", 3))
        wrappers = [fd for fd in pw.funcs.values() if fd.k is None]
        for fd in wrappers:
            args = tuple(fd.params)
            env = dict(zip(fd.params, fd.argtypes))
            lhs = (fd.name,) + args
            body = subst(fd.rhs, dict(zip(fd.params, args)))
            cur = body
            for kk in range(1, kmax + 1):
                if kk > 1:
                    nxt = self._unfold_wrappers(cur, pw)
                    if nxt == cur:
                        break
                    cur = nxt
                try:
                    text = _show_stmt(lhs, cur, env)
                    key = self.key(sid, text)
                except Exception:
                    break
                if key not in cands:
                    cands[key] = (text, f"unfold_k{kk}", ["simp refl"], len({fd.name} & wsyms) + 1, [fd.name])
        # prove (T10), guard, rank
        for key, (text, origin, hints, rank, parents) in cands.items():
            proof = self.prove(sid, text, hints)
            if proof is None:
                rec = {"stage": stage, "channel": "B", "what": "unproved", "origin": origin, "guard": "-",
                       "statement": text, "accepted": False, "proof": None, "rank": rank, "source_ids": [],
                       "emitted": False, "item": f"B|{stage}|unproved|{len(self.records)}", "_ex": {}}
                self.records.append(rec)
                continue
            triv = self.trivial(sid, text)
            ex = {"kind": "dream", "statement": text, "proof": proof, "source": "channel", "source_ids": [],
                  "parents": list(parents)}
            rec = self.offer(stage, sid, "statement", origin, ex, statement=text, accepted=True, proof=proof,
                             rank=rank + (100.0 if not triv else 0.0), extra={"trivial": triv, "repeat": key in self.seen,
                                                                              "parents": list(parents)})
            rec["eligible"] = not rec.get("dup_store") and not rec["repeat"]
        self.finish(stage, "statement", int(c.get("cap_statements", 100)), "seed")
        for r in self.records:
            if r["what"] == "statement" and r["emitted"]:
                self.seen.add(self.key(sid, r["statement"]))

    @staticmethod
    def _unfold_wrappers(t, w):
        if isinstance(t, str):
            return t
        args = tuple(ChannelB._unfold_wrappers(a, w) for a in t[1:])
        fd = w.funcs.get(t[0])
        if fd is not None and fd.k is None:
            return subst(fd.rhs, dict(zip(fd.params, args)))
        return (t[0],) + args


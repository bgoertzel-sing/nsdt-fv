"""Channel A, artifact mining (plan section 4 A, C.5): "records the goal before and after each step of a
ground-truth proof and the lemma each rw used".

Input: the window's non-test obligations and their human proofs, replayed through the checker with a
trace (check_proof(..., trace=...)). Outputs (A.3):
  wake_examples   NEXTSTATE  "next: <goal> ; <step>"  -> goal after a simp / rw / rw<- step
                  PREMISE    "premise: <goal>"         -> the project lemma the next rw uses
                  UNFOLD     "unfold: <term>"          -> the term after one definitional rewrite at its root
                  (auxiliary tasks of the wake pass of the window they come from, section 4 A)
  rem_seeds       statements: the equation lemmas of the snapshot's definitions (C.5 / section 4:
                  "Lean also generates equation lemmas for each definition") and the intermediate
                  goal states of the window's proofs, stated as equations (each is true: it arises
                  from an instance of a true statement by rewriting with true equations)
Checker acceptance of an item: NEXTSTATE - re-applying the step with the checker's own step function
reproduces the recorded state; PREMISE - the checker's rw with that lemma succeeds on the state;
UNFOLD - check_statement accepts 'simp refl' for term = unfolded; statements - a checker-accepted
proof exists (ChannelBase.prove: baseline tactics, then the reference prover on the test-free library).
"""
from __future__ import annotations

from ...backends.toyeq import (Statement, StepError, apply_rw, check_proof, show, simp_goal, term_vars)
from . import ChannelBase


def _stmt(lhs, rhs, env):
    vs = term_vars(lhs)
    for v in term_vars(rhs):
        if v not in vs:
            vs.append(v)
    return Statement(tuple((v, env[v]) for v in vs), lhs, rhs).show()


def _apps(t, acc):
    if isinstance(t, tuple):
        acc.append(t)
        for a in t[1:]:
            _apps(a, acc)
    return acc


class ChannelA(ChannelBase):
    id = "A"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = set()                      # statement keys this channel has emitted (no re-emission)

    def produce(self, stage, sid):
        w = self.stream.worlds[sid]
        stmts = {}                             # key -> (text, origin, source id)
        for o in self.window(sid):
            pos = w.lemma_pos[o["decl"]]
            st = w.lemmas[pos][1]
            tr = []
            v = check_proof(w, st, o["proof"], pos, trace=tr)
            if not v.accepted:
                continue
            base = {"statement": o["statement"], "source": "channel", "source_ids": [o["id"]]}
            for i, (kind, arg, goal) in enumerate(tr):
                if kind in ("simp", "rw", "rw<-") and i + 1 < len(tr):
                    after = tr[i + 1][2]
                    step = kind if kind == "simp" else f"{kind} {arg}"
                    try:
                        re = simp_goal(goal, w, float("inf")) if kind == "simp" else apply_rw(goal, arg, kind == "rw<-", w, pos)
                        ok = re.show() == after.show()
                    except StepError:
                        ok = False
                    p, t = f"next: {goal.show()} ; {step}", after.show()
                    if self.fits(p, t):
                        self.offer(stage, sid, "nextstate", "trace", dict(base, kind="nextstate", prompt=p, target=t),
                                   accepted=ok, rank=1.0 if kind != "simp" else 0.5)
                if kind in ("rw", "rw<-") and not arg.startswith("IH"):
                    try:
                        apply_rw(goal, arg, kind == "rw<-", w, pos)
                        ok = True
                    except StepError:
                        ok = False
                    p = f"premise: {goal.show()}"
                    if self.fits(p, arg):
                        self.offer(stage, sid, "premise", "trace", dict(base, kind="premise", prompt=p, target=arg),
                                   accepted=ok, rank=1.0)
                if i > 0 and not goal.hyps and (goal.lhs != goal.rhs):
                    try:
                        text = _stmt(goal.lhs, goal.rhs, goal.envd())
                        k = self.key(sid, text)
                    except Exception:
                        continue
                    stmts.setdefault(k, (text, "goal_state", o["id"]))
                elif i > 0 and goal.hyps and goal.lhs != goal.rhs:
                    try:
                        text = _stmt(goal.lhs, goal.rhs, goal.envd())
                        k = self.key(sid, text)
                    except Exception:
                        continue
                    stmts.setdefault(k, (text, "goal_state_ih", o["id"]))
            # UNFOLD: one definitional rewrite at the root of each application in the statement
            env = st.env()
            for side in (st.lhs, st.rhs):
                for sub in _apps(side, []):
                    fd = w.funcs.get(sub[0])
                    if fd is None:
                        continue
                    red = fd.reduce(sub[1:])
                    if red is None:
                        continue
                    p, t = f"unfold: {show(sub)}", show(red)
                    try:
                        ok = self.check_stmt(sid, _stmt(sub, red, env), "simp refl").accepted
                    except Exception:
                        ok = False
                    if self.fits(p, t):
                        self.offer(stage, sid, "unfold", "statement", dict(base, kind="unfold", prompt=p, target=t),
                                   accepted=ok, rank=0.5)
        # equation lemmas of the snapshot's definitions
        for fd in w.funcs.values():
            env = {}
            for (l, r) in fd.eqs:
                try:
                    from ...backends.toyeq import infer_var_types
                    env = infer_var_types([l, r], w.funcs)
                    text = _stmt(l, r, env)
                    k = self.key(sid, text)
                except Exception:
                    continue
                stmts.setdefault(k, (text, "equation_lemma", None))
        for k, (text, origin, src) in stmts.items():
            triv = self.trivial(sid, text)
            ex = {"kind": "dream", "statement": text, "source": "channel", "source_ids": [src] if src else []}
            rec = self.offer(stage, sid, "statement", origin, ex, statement=text,
                             rank=(2.0 if not triv else 0.0) + (0.5 if origin == "goal_state" else 0.0),
                             extra={"trivial": triv, "repeat": k in self.seen})
            rec["eligible"] = not rec.get("dup_store") and not rec["repeat"]
        c = self.cfg
        for r in self.records:                 # an auxiliary item the checker does not confirm is never emitted
            if r["what"] != "statement":
                r["eligible"] = bool(r["accepted"])
        for what in ("nextstate", "premise", "unfold"):
            self.finish(stage, what, int(c.get("cap_" + what, 100)), "wake")
        self.finish(stage, "statement", int(c.get("cap_statements", 60)), "seed")
        for r in self.records:
            if r["what"] == "statement" and r["emitted"]:
                self.seen.add(self.key(sid, r["statement"]))
                p = self.prove(sid, r["statement"])
                r["accepted"], r["proof"] = p is not None, p


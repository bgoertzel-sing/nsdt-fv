"""Channel C, exploration by execution (plan section 4 C, C.5; QuickSpec-style): "enumerates terms up to
size 7 over a project's functions, evaluates them on 200 random inputs, and proposes equations between
terms that agree".

Signature: the snapshot's functions, the nullary constructors Z, Nil, Leaf, and typed variables
(Nat n m, List xs ys, Tree t u, Bool b). Terms are enumerated bottom-up by size (number of nodes),
building only on class representatives (the first term with a given type, variable set and value
vector), up to size max_size (7) or max_terms terms. Each term is evaluated on the same 200 random
input environments by an evaluator independent of the checker's rewriting (backends/toyeq_oracle).
A term whose value vector equals a representative's gives a candidate equation term = representative.

Counterexample statistics (logged per stage, T10):
  * pairs that agree on the first 20 inputs but are separated by a counterexample within the 200
    (false candidates removed by testing before any proof search);
  * candidates re-tested on 200 fresh inputs (an independent seed): a counterexample there
    discards the candidate and is recorded with it;
  * evaluations stopped by the evaluation budget (terms dropped).
Only candidates with no recorded counterexample are offered (T10: no seed has a recorded counterexample).
Checker acceptance of an emitted conjecture: a checker-accepted proof found by ChannelBase.prove
(baseline tactics, then the reference prover on the test-free library); conjectures without one are
still emitted as seeds (that is the channel's purpose) and count as not accepted.
Ranking: not closed by the baseline tactics first, then involving a function of the window's new
statements, then smaller. Output (A.3): rem_seeds.
"""
from __future__ import annotations

import random
from itertools import product

from ...backends.toyeq import Statement, show, symbols, term_vars
from ...backends.toyeq_oracle import EvalLimit, Oracle, counterexample, random_value
from . import ChannelBase, sha12

VARS = (("n", "Nat"), ("m", "Nat"), ("xs", "List"), ("ys", "List"), ("t", "Tree"), ("u", "Tree"), ("b", "Bool"))
CONSTS = ((("Z",), "Nat", 0), (("Nil",), "List", ()), (("Leaf",), "Tree", None))


def _size(t):
    return 1 if isinstance(t, str) else 1 + sum(_size(a) for a in t[1:])


class ChannelC(ChannelBase):
    id = "C"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = set()

    def produce(self, stage, sid):
        c = self.cfg
        w = self.stream.worlds[sid]
        n_in = int(c.get("inputs", 200))
        max_size = int(c.get("max_size", 7))
        max_terms = int(c.get("max_terms", 4000))
        rng = random.Random(f"C-inputs|{self.seed}|{stage}")
        envs = []
        for i in range(n_in):
            size = 1 + (i % 4)
            envs.append({v: random_value(ty, rng, size) for v, ty in VARS})
        orc = Oracle(w, budget=int(c.get("eval_budget", 20000)))
        memo = {}

        def apply(fname, vals):
            key = (fname, vals)
            if key in memo:
                return memo[key]
            orc.steps = 0
            names = [f"a{i}" for i in range(len(vals))]
            r = orc.ev((fname,) + tuple(names), dict(zip(names, vals)))
            memo[key] = r
            return r

        funcs = list(w.funcs.values())
        # representatives: by (type, value vector) -> term ; terms by size and type
        reps = {}
        by_size = {}
        stats = {"terms": 0, "eval_limit": 0, "classes": 0, "candidates": 0, "sep_within_200": 0, "sep_after_5": 0,
                 "dup_or_unparsed": 0,
                 "fresh_counterexample": 0, "fresh_limit": 0, "survivors": 0, "max_size_reached": 0}
        fp20, fp5 = {}, {}
        cands = []

        def add(term, ty, vec):
            stats["terms"] += 1
            k = (ty, vec)
            if k in reps:
                cands.append((reps[k], term, ty))
                return
            k20, k5 = (ty, vec[:20]), (ty, vec[:5])
            if k20 in fp20:
                stats["sep_within_200"] += 1       # agreed on 20 inputs with a class, refuted within 200
            else:
                fp20[k20] = term
            if k5 in fp5:
                stats["sep_after_5"] += 1          # agreed on 5 inputs with a class, refuted within 200
            else:
                fp5[k5] = term
            reps[k] = term
            by_size.setdefault(_size(term), {}).setdefault(ty, []).append((term, vec))

        for v, ty in VARS:
            add(v, ty, tuple(e[v] for e in envs))
        for t, ty, val in CONSTS:
            add(t, ty, tuple(val for _ in envs))
        done = False
        for s in range(2, max_size + 1):
            if done:
                break
            for fd in funcs:
                if done:
                    break
                ar = len(fd.argtypes)
                if ar == 0 or s - 1 < ar:
                    continue
                for parts in _compositions(s - 1, ar):
                    pools = [by_size.get(p, {}).get(at, []) for p, at in zip(parts, fd.argtypes)]
                    if any(not p for p in pools):
                        continue
                    for combo in product(*pools):
                        if stats["terms"] >= max_terms:
                            done = True
                            break
                        term = (fd.name,) + tuple(x[0] for x in combo)
                        try:
                            vec = tuple(apply(fd.name, tuple(x[1][i] for x in combo)) for i in range(n_in))
                        except EvalLimit:
                            stats["eval_limit"] += 1
                            continue
                        add(term, fd.rettype, vec)
                        stats["max_size_reached"] = max(stats["max_size_reached"], s)
                    if done:
                        break
        stats["classes"] = len(reps)
        stats["candidates"] = len(cands)
        tyenv = dict(VARS)
        wsyms = set()
        for o in self.window(sid):
            st = w.statement(o["statement"])
            wsyms |= (symbols(st.lhs) | symbols(st.rhs)) & set(w.funcs)
        seen_keys = set()
        fresh_seed = int(c.get("fresh_seed", 1))
        for rep, term, ty in cands:
            if rep == term:
                continue
            vs = term_vars(term)
            for v in term_vars(rep):
                if v not in vs:
                    vs.append(v)
            st = Statement(tuple((v, tyenv[v]) for v in vs), term, rep)
            text = st.show()
            try:
                st = w.statement(text)
                key = self.key(sid, text)
            except Exception:
                stats["dup_or_unparsed"] += 1
                continue
            if key in seen_keys:
                stats["dup_or_unparsed"] += 1
                continue
            seen_keys.add(key)
            cex = counterexample(w, st, n=n_in, seed=int(sha12(f"{fresh_seed}|{stage}|{key}"), 16))
            if cex is not None:
                if cex.get("_limit"):
                    stats["fresh_limit"] += 1
                else:
                    stats["fresh_counterexample"] += 1
                self.records.append({"stage": stage, "channel": "C", "what": "refuted", "origin": "fresh_test",
                                     "guard": "-", "statement": text, "accepted": False, "proof": None, "rank": 0.0,
                                     "source_ids": [], "emitted": False, "item": f"C|{stage}|refuted|{len(self.records)}",
                                     "counterexample": {k: repr(v) for k, v in cex.items()}, "_ex": {}})
                continue
            stats["survivors"] += 1
            triv = self.trivial(sid, text)
            rel = len((symbols(st.lhs) | symbols(st.rhs)) & wsyms)
            ex = {"kind": "dream", "statement": text, "source": "channel", "source_ids": []}
            rec = self.offer(stage, sid, "statement", "quickspec", ex, statement=text,
                             rank=(100.0 if not triv else 0.0) + 10.0 * min(rel, 3) - _size(term) - _size(rep),
                             extra={"trivial": triv, "repeat": key in self.seen, "counterexample": None})
            rec["eligible"] = not rec.get("dup_store") and not rec["repeat"]
        self.stats = stats
        self.finish(stage, "statement", int(c.get("cap_statements", 100)), "seed")
        for r in self.records:
            if r["what"] == "statement" and r["emitted"]:
                self.seen.add(self.key(sid, r["statement"]))
                p = self.prove(sid, r["statement"])
                r["accepted"], r["proof"] = p is not None, p
        self.records.append({"stage": stage, "channel": "C", "what": "stats", "origin": "quickspec", "guard": "-",
                             "statement": None, "accepted": None, "proof": None, "rank": 0.0, "source_ids": [],
                             "emitted": False, "item": f"C|{stage}|stats", "stats": dict(stats), "_ex": {}})


def _compositions(n, k):
    """Ordered k-tuples of positive integers summing to n."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


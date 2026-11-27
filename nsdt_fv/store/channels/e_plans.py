"""Channel E, proof-plan induction (plan section 4 E, C.5): "abstracts proofs to skeletons (the sequence of
step kinds with lemma names replaced by the head symbol of the lemma) and anti-unifies skeletons that
occur at least five times".

Skeleton of a proof: the step sequence with blocks; 'rw NAME' becomes 'rw[h]' where h is the head
symbol of the side of NAME's statement that the step rewrites from ('rw IH' stays); 'induct x' and
'cases x' become 'induct <type of x>' (types read from the checker's trace). Skeletons of the window's
human proofs (non-test) are counted; those occurring at least min_count (5) times are grouped by shape
(the skeleton with heads erased) and each group is anti-unified into a method: positions where all
members agree keep the head, the others become typed slots ?k (the slot's type is the set of heads seen).
Outputs (A.3):
  wake_examples   PLAN "plan: <statement>" -> "<method> ; <proof>" for window proofs a method covers
                  (the plan-then-proof examples of section 4 E; auxiliary, wake of the same window).
  nrem_targets    "conjecture sleep can instantiate a method on statements whose type signature
                  matches": store statements of the project (not closed by the baseline tactics) on
                  which an instantiation of a method (induction variable and slot lemmas filled from
                  the test-free library) is accepted by backend.check; these get priority in the
                  following re-derivation sleep. The instantiated proof is logged, never trained on.
Checker acceptance: PLAN items - the human proof is accepted and its skeleton matches the method;
priority items - an instantiation was accepted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product

from ...backends.toyeq import check_proof, parse_proof
from . import ChannelBase


def _walk(steps):
    for s in steps:
        yield s
        for b in s[2] or []:
            yield from _walk(b)


class ChannelE(ChannelBase):
    id = "E"
    statement_kinds = ("priority",)

    def skeleton(self, w, st, proof, pos):
        tr = []
        if not check_proof(w, st, proof, pos, trace=tr).accepted:
            return None
        steps = parse_proof(proof)
        flat = list(_walk(steps))
        if len(flat) != len(tr):
            return None
        info = {}
        for s, (kind, arg, goal) in zip(flat, tr):
            if kind in ("induct", "cases"):
                info[id(s)] = goal.envd().get(arg, "?")
            elif kind in ("rw", "rw<-") and not arg.startswith("IH"):
                if arg in w.lemma_pos:
                    lst = w.lemmas[w.lemma_pos[arg]][1]
                    side = lst.rhs if kind == "rw<-" else lst.lhs
                    info[id(s)] = side[0] if isinstance(side, tuple) else "_"
                else:
                    info[id(s)] = "_"

        def render(steps):
            out = []
            for s in steps:
                kind, arg, blocks, _ = s
                if kind in ("simp", "refl"):
                    out.append(kind)
                elif kind in ("rw", "rw<-"):
                    out.append(f"{kind} IH" if arg.startswith("IH") else f"{kind}[{info[id(s)]}]")
                else:
                    out.append(f"{kind} {info[id(s)]} " + " ".join("{ " + " ".join(render(b)) + " }" for b in blocks))
            return out
        return " ".join(render(steps))

    @staticmethod
    def tokens(sk):
        return sk.split(" ")

    @staticmethod
    def shape(sk):
        return " ".join("rw[?]" if t.startswith("rw[") else ("rw<-[?]" if t.startswith("rw<-[") else t)
                        for t in sk.split(" "))

    def produce(self, stage, sid):
        c = self.cfg
        w = self.stream.worlds[sid]
        pw = self.private_world(sid)
        sks = {}
        for o in self.window(sid):
            pos = w.lemma_pos[o["decl"]]
            sks[o["id"]] = self.skeleton(w, w.lemmas[pos][1], o["proof"], pos)
        cnt = Counter(s for s in sks.values() if s)
        frequent = [s for s, n in cnt.items() if n >= int(c.get("min_count", 5))]
        groups = defaultdict(list)
        for s in frequent:
            groups[self.shape(s)].append(s)
        methods = []
        for shp, members in sorted(groups.items()):
            toks = [m.split(" ") for m in members]
            out, slots = [], []
            for i, col in enumerate(zip(*toks)):
                if len(set(col)) == 1:
                    out.append(col[0])
                else:
                    pre = "rw<-" if col[0].startswith("rw<-") else "rw"
                    heads = sorted({x[len(pre) + 1:-1] for x in col})
                    slots.append(heads)
                    out.append(f"{pre}[?{len(slots)}]")
            methods.append({"text": " ".join(out), "shape": shp, "slots": slots,
                            "support": sum(cnt[m] for m in members), "members": members})
        methods.sort(key=lambda m: -m["support"])
        self.methods = methods

        def covers(m, sk):
            if self.shape(sk) != m["shape"]:
                return False
            for a, b in zip(m["text"].split(" "), sk.split(" ")):
                if "[?" in a:
                    continue
                if a != b:
                    return False
            return True
        # PLAN examples
        for o in self.window(sid):
            sk = sks.get(o["id"])
            if not sk:
                continue
            m = next((m for m in methods if covers(m, sk)), None)
            if m is None:
                continue
            ok = self.backend.check(sid, o["id"], o["proof"], 120).accepted
            p, t = f"plan: {o['statement']}", f"{m['text']} ; {o['proof']}"
            ex = {"kind": "plan", "statement": o["statement"], "prompt": p, "target": t, "source": "channel",
                  "source_ids": [o["id"]]}
            rec = self.offer(stage, sid, "plan", "method", ex, accepted=ok, rank=float(m["support"]),
                             extra={"method": m["text"]})
            rec["eligible"] = ok and self.fits(p, t)
        self.finish(stage, "plan", int(c.get("cap_plans", 100)), "wake")
        # method instantiation on store statements (priorities for re-derivation)
        p = sid.split("@")[0]
        cands = []
        if self.store is not None:
            for e in self.store.statements(project=p, kind="repo"):
                if not e["trivial"]:
                    cands.append(e)
        cands.sort(key=lambda e: (0 if e["snapshot_id"] == sid else 1))
        tries = int(c.get("tries", 8))
        for e in cands[: int(c.get("max_targets", 60))]:
            ew = self.stream.worlds[e["snapshot_id"]]
            epw = self.private_world(e["snapshot_id"])
            st = ew.statement(e["statement"])
            pos_priv = self.priv_pos(e["snapshot_id"], e["decl"])
            avail = epw.lemmas[:pos_priv]
            byhead = defaultdict(list)
            for n, ls in avail:
                for back, side in ((False, ls.lhs), (True, ls.rhs)):
                    if isinstance(side, tuple):
                        byhead[(back, side[0])].append(n)
            found = None
            n_try = 0
            for m in methods:
                if found or n_try >= tries:
                    break
                toks = m["text"].split(" ")
                ind_types = [toks[i + 1] for i, t in enumerate(toks) if t in ("induct", "cases")]
                vars_by_type = defaultdict(list)
                for v, ty in st.vars:
                    vars_by_type[ty].append(v)
                if ind_types and not vars_by_type.get(ind_types[0]):
                    continue
                # fixed heads and slots -> candidate lemma names
                choices = []
                for t in toks:
                    if t.startswith("rw[") or t.startswith("rw<-["):
                        back = t.startswith("rw<-")
                        h = t[t.index("[") + 1:-1]
                        heads = m["slots"][int(h[1:]) - 1] if h.startswith("?") else [h]
                        names = [n for hh in heads for n in byhead.get((back, hh), [])]
                        choices.append(names[-3:] if names else [])
                if any(not ch for ch in choices):
                    continue
                xs = vars_by_type.get(ind_types[0], [None])[:2] if ind_types else [None]
                for x in xs:
                    for fill in product(*choices):
                        if n_try >= tries:
                            break
                        n_try += 1
                        it = iter(fill)
                        out, k = [], 0
                        for t in toks:
                            if t.startswith("rw[") or t.startswith("rw<-["):
                                out.append(("rw<- " if t.startswith("rw<-") else "rw ") + next(it))
                            elif t in ("induct", "cases"):
                                out.append(t)
                            elif k == 0 and out and out[-1] in ("induct", "cases"):
                                out.append(x or "?")
                                k = 1
                            elif out and out[-1] in ("induct", "cases"):
                                out.append(x or "?")
                            else:
                                out.append(t)
                        proof = " ".join(out)
                        if "?" in proof:
                            continue
                        if self.backend.check(e["snapshot_id"], e["obligation_id"], proof, 120).accepted:
                            found = (m, proof)
                            break
                    if found:
                        break
            if found is None:
                continue
            ex = {"kind": "dream", "statement": e["statement"], "proof": found[1], "source": "channel",
                  "source_ids": [e["key"]]}
            self.offer(stage, sid, "priority", "instantiation", ex, statement=e["statement"], accepted=True,
                       proof=found[1], rank=1.0 if e["snapshot_id"] == sid else 0.5,
                       extra={"target_key": e["key"], "trivial": e["trivial"], "method": found[0]["text"]})
        self.finish(stage, "priority", int(c.get("cap_priorities", 60)), "prio")
        self.records.append({"stage": stage, "channel": "E", "what": "methods", "origin": "anti_unify", "guard": "-",
                             "statement": None, "accepted": None, "proof": None, "rank": 0.0, "source_ids": [],
                             "emitted": False, "item": f"E|{stage}|methods",
                             "methods": [{k: m[k] for k in ("text", "support", "slots")} for m in methods],
                             "skeletons": len([s for s in sks.values() if s]), "distinct_skeletons": len(cnt), "_ex": {}})


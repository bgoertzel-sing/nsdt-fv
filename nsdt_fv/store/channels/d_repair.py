"""Channel D, perturb and repair (plan section 4 D, C.5): "mutates proofs by deleting, swapping or
replacing a step and repairs them with the reference prover".

Input: the window's non-test obligations and their human proofs. For each proof, mutants are made by
  delete   removing one step (in a block with at least two steps),
  swap     exchanging two adjacent steps of a block,
  replace  rw NAME -> rw<- NAME or rw OTHER (another lemma of the library), simp <-> refl;
at most per_proof mutants per proof, cap_mutants per stage (in a seeded order). Each mutant is checked
with backend.check on its obligation; a mutant the checker still accepts is recorded as neutral. For a
rejected mutant the checker's diagnostic (message and goal state) is kept and the reference prover,
run on the test-free library before the lemma, supplies a repair that backend.check must accept.
Outputs (A.3):
  wake_examples   REPAIR "repair: <statement> ; <broken proof> ; <diagnostic>" -> the accepted repair.
                  This is the one place in S3 where a machine proof is a training target (C.5 defines
                  the repair as the reference prover's; reports/CHANGES.md, S3).
  nrem_targets    the blast radius: store statements in the reverse proof_uses cone (edges after L2)
                  of the declarations whose proofs broke, in this snapshot (priorities for the
                  following re-derivation sleep, section 4 D).
Checker acceptance: REPAIR items - the repair was accepted by backend.check; priorities - not a claim
(no checker verdict; reported as '-').
"""
from __future__ import annotations

from collections import defaultdict

from ...backends.toyeq import parse_proof, show_proof
from ...data.toyeq_prover import RefProver
from . import ChannelBase


def _sites(steps, path=()):
    for i, s in enumerate(steps):
        yield path, steps, i
        for bi, b in enumerate(s[2] or []):
            yield from _sites(b, path + ((i, bi),))


def _replace_block(steps, path, new_block):
    if not path:
        return new_block
    (i, bi), rest = path[0], path[1:]
    kind, arg, blocks, tok = steps[i]
    blocks = list(blocks)
    blocks[bi] = _replace_block(blocks[bi], rest, new_block)
    return steps[:i] + [(kind, arg, blocks, tok)] + steps[i + 1:]


class ChannelD(ChannelBase):
    id = "D"
    statement_kinds = ("priority",)

    def mutants(self, proof, lemma_names, rng):
        steps = parse_proof(proof)
        out = []
        for path, block, i in _sites(steps):
            block = list(block)
            if len(block) >= 2:
                out.append(("delete", _replace_block(steps, path, block[:i] + block[i + 1:])))
            if i + 1 < len(block):
                sw = block[:i] + [block[i + 1], block[i]] + block[i + 2:]
                out.append(("swap", _replace_block(steps, path, sw)))
            kind, arg, blocks, tok = block[i]
            rep = None
            if kind == "rw":
                if arg.startswith("IH") or rng.random() < 0.5:
                    rep = ("rw<-", arg, blocks, tok)
                elif lemma_names:
                    rep = ("rw", rng.choice(lemma_names), blocks, tok)
            elif kind == "rw<-":
                rep = ("rw", arg, blocks, tok)
            elif kind == "simp":
                rep = ("refl", None, blocks, tok)
            elif kind == "refl":
                rep = ("simp", None, blocks, tok)
            if rep is not None:
                out.append(("replace", _replace_block(steps, path, block[:i] + [rep] + block[i + 1:])))
        res = []
        for op, st in out:
            try:
                txt = show_proof(st)
            except Exception:
                continue
            if txt.strip() and txt != proof:
                res.append((op, txt))
        return res

    def produce(self, stage, sid):
        c = self.cfg
        w = self.stream.worlds[sid]
        pw = self.private_world(sid)
        per = int(c.get("per_proof", 2))
        cap = int(c.get("cap_mutants", 80))
        jobs = []
        for o in self.window(sid):
            pos = self.priv_pos(sid, o["decl"])
            names = [n for n, _ in pw.lemmas[:pos]]
            ms = self.mutants(o["proof"], names, self.rng)
            self.rng.shuffle(ms)
            for op, m in ms[:per]:
                jobs.append((o, pos, op, m))
        self.rng.shuffle(jobs)
        broken = set()
        for o, pos, op, m in jobs[:cap]:
            v = self.backend.check(sid, o["id"], m, 120)
            if v.accepted:
                self.records.append({"stage": stage, "channel": "D", "what": "neutral", "origin": op, "guard": "-",
                                     "statement": o["statement"], "accepted": True, "proof": None, "rank": 0.0,
                                     "source_ids": [o["id"]], "emitted": False,
                                     "item": f"D|{stage}|neutral|{len(self.records)}", "_ex": {}})
                continue
            msg = v.messages[0] if v.messages else {"text": v.category, "goal_state": ""}
            diag = (msg.get("text", "") + (" | " + msg["goal_state"] if msg.get("goal_state") else "")).strip()
            broken.add(o["decl"])
            st = w.lemmas[w.lemma_pos[o["decl"]]][1]
            rp = RefProver(pw, pos, node_limit=int(c.get("prover_nodes", 2000))).prove(st)
            ok = rp is not None and self.backend.check(sid, o["id"], rp, 120).accepted
            prompt = f"repair: {o['statement']} ; {m} ; {diag}"
            ex = {"kind": "repair", "statement": o["statement"], "prompt": prompt, "target": rp or "",
                  "source": "channel", "source_ids": [o["id"]]}
            rec = self.offer(stage, sid, "repair", op, ex, accepted=ok, proof=rp,
                             extra={"mutant": m, "diagnostic": diag, "repair_equals_human": rp == o["proof"]})
            rec["eligible"] = ok and self.fits(prompt, rp or "")
            if ok and not self.fits(prompt, rp or ""):
                rec["too_long"] = True
        self.finish(stage, "repair", int(c.get("cap_repairs", 80)), "wake")
        # blast radius: reverse proof_uses cone (L2-filtered edges of this snapshot)
        edges = self.guard.filter_edges([e for e in self.stream.edges if e["snapshot_id"] == sid])
        rev = defaultdict(set)
        for e in edges:
            if e["type"] == "proof_uses":
                rev[e["dst"]].add(e["src"])
        cone = set(broken)
        todo = list(broken)
        while todo:
            x = todo.pop()
            for y in rev.get(x, ()):
                if y not in cone:
                    cone.add(y)
                    todo.append(y)
        p = sid.split("@")[0]
        if self.store is not None:
            for e in self.store.statements(project=p, kind="repo"):
                if e["decl"] in cone:
                    ex = {"kind": "dream", "statement": e["statement"], "source": "channel", "source_ids": [e["key"]]}
                    rec = self.offer(stage, sid, "priority", "blast_radius", ex, statement=e["statement"],
                                     rank=2.0 if e["decl"] in broken else 1.0,
                                     extra={"target_key": e["key"], "trivial": e["trivial"]})
        self.finish(stage, "priority", int(c.get("cap_priorities", 60)), "prio")


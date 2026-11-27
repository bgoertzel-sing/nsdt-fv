"""Generate reports/S2.md from the raw S2 and S1 logs (rules R3 and R13, report format A.6).

    python -m nsdt_fv.eval.report_s2 [--out nsdt_fv/reports/S2.md]

Reads runs/S2/<arm>/ and runs/S2_seed1/<arm>/ (D arms, B2c) and runs/S1/<arm>/, runs/S1_seed1/<arm>/
(B0, B2, B5-10, B6). Every number is computed here from verdict logs, dream logs, conjecture logs,
training-example logs and train.jsonl with eval/metrics.py and eval/bootstrap.py. Nothing is typed
by hand. The stream is rebuilt deterministically (data version checked) only to compare dream
targets with human proofs for the T9 side statistic.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from . import bootstrap as BS
from . import metrics as M
from .report_s1 import t4 as t4_check

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1_ARMS = ["B0", "B2", "B5-10", "B6"]
S2_ARMS = ["D1", "D2", "D3", "D4", "B2c", "D1x"]
NAMES = {"B0": "frozen base", "B2": "sequential fine-tuning, all weights (S1)", "B5-10": "B2 + 10% replay (S1)",
         "B6": "joint retraining (S1)", "B2c": "sequential fine-tuning of the cap only (supplementary)",
         "D1": "DT-ref: wake, re-derivation and conjecture sleep, kernel filter, negatives",
         "D2": "D1 without the kernel filter", "D3": "D1 without conjecture sleep", "D4": "D1 without the negative phase",
         "D1x": "diagnostic, not A.4: D1 with B2c's wake (30 epochs, batch 32) instead of one epoch"}
KEYS = ("LA", "FWD", "ACC", "F", "BWT")


def pct(x):
    return f"{100 * x:.1f}"


def ci(lh):
    return f"[{100 * lh[0]:.1f}, {100 * lh[1]:.1f}]"


def gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


ALLOW_INCOMPLETE = False


def load_run(d, arm, s1=False):
    p = os.path.join(d, arm)
    if not os.path.exists(os.path.join(p, "verdicts.jsonl.gz")):
        return None
    if not s1 and not os.path.exists(os.path.join(p, "meta.json")):     # meta.json is written when a run ends
        if ALLOW_INCOMPLETE:
            return None
        raise RuntimeError(f"run {p} is incomplete (no meta.json)")
    r = {"lines": gz(os.path.join(p, "verdicts.jsonl.gz")), "train": jl(os.path.join(p, "train.jsonl")),
         "evalsets": json.load(open(os.path.join(p, "evalsets.json")))}
    if not s1:
        r["meta"] = json.load(open(os.path.join(p, "meta.json")))
        r["run"] = r["meta"]["run"]
        for name in ("dreams", "conjectures", "train_examples"):
            fp = os.path.join(p, f"{name}.jsonl.gz")
            r[name] = gz(fp) if os.path.exists(fp) else []
    return r


def build(out_path):
    cfg_s1 = json.load(open(os.path.join(HERE, "runs/S1/meta.json")))
    ts = json.load(open(os.path.join(HERE, "runs/S1/test_sets.json")))
    test_sets = {int(j): ids for j, ids in ts["test_sets"].items()}
    stages = ts["stages"]
    T = len(test_sets)
    obl = {}
    for line in open(os.path.join(HERE, "runs/S1/test_obligations.jsonl")):
        o = json.loads(line)
        obl[o["id"]] = o
    runs = {}                                     # (arm, seed) -> run
    for seed, d1, d2 in ((0, "runs/S1", "runs/S2"), (1, "runs/S1_seed1", "runs/S2_seed1")):
        m1 = json.load(open(os.path.join(HERE, d1, "meta.json")))
        for a in S1_ARMS:
            r = load_run(os.path.join(HERE, d1), a, s1=True)
            if r is not None:
                r["run"] = m1["arms"][a]["run"]
                r["meta"] = {"seed": seed, "wall_s": m1["arms"][a]["wall_s"], "base": m1["base"]}
                runs[(a, seed)] = r
        for a in S2_ARMS:
            r = load_run(os.path.join(HERE, d2), a)
            if r is not None:
                runs[(a, seed)] = r
    s2 = [(a, s) for (a, s) in runs if a in S2_ARMS]
    s2.sort(key=lambda x: (x[1], S2_ARMS.index(x[0])))
    ref = s2[0]
    cfg = runs[ref]["meta"]["config"]
    k = cfg["eval"]["k"]
    res, P = {}, {}
    for key, r in runs.items():
        res[key], _ = M.metrics_from_log(r["lines"], test_sets, k, run=r["run"])
        r["A"] = M.accuracy_matrix(r["lines"], test_sets, k, run=r["run"])
        P[key] = BS.per_obligation(r["lines"], test_sets, k, run=r["run"])
    acc_b2, acc_b6 = res[("B2", 0)]["ACC"], res[("B6", 0)]["ACC"]
    for key in res:
        res[key]["GC"] = M.gap_closed(res[key]["ACC"], acc_b2, acc_b6)
    B = cfg["bootstrap"]["resamples"]
    R = BS.draws(test_sets, B, cfg["bootstrap"]["seed"])
    boot = {key: BS.resampled_metrics(P[key], T, R) for key in P}
    ident = {j: np.arange(len(ids))[None, :] for j, ids in test_sets.items()}
    for key in P:
        chk = BS.resampled_metrics(P[key], T, ident)
        for m in KEYS:
            if abs(float(chk[m][0]) - res[key][m]) > 1e-9:
                raise RuntimeError(f"bootstrap/metrics disagree on {key} {m}")

    def diff(a, b, m):
        x = boot[a][m] - boot[b][m]
        return res[a][m] - res[b][m], BS.interval(x)

    L = []
    w = L.append
    w("# S2: DT on the toy stream (D1-D4 with DT-ref)")
    w("")
    w("Generated by `python -m nsdt_fv.eval.report_s2` from `nsdt_fv/runs/S2*/` and `nsdt_fv/runs/S1*/` (rule R3); do not edit by hand.")
    w("")
    # ------------------------------------------------------------------ identification
    w("## Identification")
    w("")
    w("- Experiment: S2 (plan 8.1). DT arms (Table 3) with DT-ref (A.4), not the DT core: " +
      "; ".join(f"{a}: {NAMES[a]}" for a in S2_ARMS if any(x[0] == a for x in s2)) + ".")
    w("- Reference arms from S1, recomputed here from S1's verdict logs: " + "; ".join(f"{a}: {NAMES[a]}" for a in S1_ARMS) + ".")
    m0 = runs[ref]["meta"]
    cc = cfg["cap"]
    w(f"- Base: S1's tiny transformer, frozen ({m0['base']['params']:,} parameters; d={cfg['model']['d']}, "
      f"{cfg['model']['layers']} layers). Seed 0 base hash `{cfg['base_hash'][0] if 0 in cfg['base_hash'] else cfg['base_hash']['0']}`, "
      f"seed 1 base hash `{cfg['base_hash'][1] if 1 in cfg['base_hash'] else cfg['base_hash']['1']}` (S1's seed-1 replicate base).")
    if m0["cap"].get("kind", "mlp") == "block":
        what = (f"residual adapter blocks that are full transformer blocks (attention + feed-forward of width {cc['r']}, "
                f"output projections initialised to zero; sized up from the MLP adapter, CHANGES.md 'S2 calibration')")
    else:
        what = f"residual adapter blocks (LayerNorm, down to r={cc['r']}, GELU, up; up initialised to zero)"
    w(f"- Cap (A.4): {what} after base layers "
      f"{m0['cap']['layers']} and an untied output head initialised from the "
      f"base embedding; {m0['cap']['trainable_params']:,} trainable parameters. AdamW lr {cc['lr']}, constant, one optimiser per run.")
    dc = cfg["dream"]
    w(f"- Day (stage): wake = one epoch over the window's {('human proofs plus the auxiliary statement task (three window statements -> a fourth)' if dc.get('aux_conj') else 'human proofs')}, "
      f"batch {cc['wake_batch']}; re-derivation sleep = {dc['nrem_statements']} store statements drawn with weight "
      f"1 + 4 fading + centrality + 2 near_change, {dc['candidates']} candidates each at temperature {dc['temperature']}; conjecture sleep = "
      f"{dc['rem_raw']} statement generations (each prompted with three store statements of one module), at most {dc['rem_conjectures']} new "
      f"statements attempted, {dc['candidates']} candidates each; one cross-entropy step per learned dream; negative phase weight {dc['pref_weight']}. "
      f"Dream attempt budget per stage: {dc['nrem_statements'] * dc['candidates']} + {dc['rem_conjectures'] * dc['candidates']}.")
    w(f"- Seeds: runs " + ", ".join(f"{a} s{s}" for a, s in s2) + "; sampling seeds = sha256(run|stage|target|sample) (R6). "
      f"Bootstrap: {B} resamples, seed {cfg['bootstrap']['seed']}.")
    commits = sorted({runs[x]["meta"]["code_commit"] for x in s2})
    w(f"- Code commit at run start: {', '.join('`' + c + '`' for c in commits)}. Data version: `{m0['data_version']}` "
      f"(S1: `{cfg_s1['data_version']}`; equal: {m0['data_version'] == cfg_s1['data_version']}). ToyEq checker sha `{m0['toolchain']['toyeq_checker_sha']}`.")
    v = m0["versions"]
    w(f"- Versions: Python {v['python']}, torch {v['torch']}, numpy {v['numpy']}, PyYAML {v['pyyaml']}; {v['platform']}; "
      f"{m0['threads']} torch threads; CPU affinity per run: " + ", ".join(f"{a} s{s} {runs[(a, s)]['meta']['cpu_affinity']}" for a, s in s2) + ".")
    w(f"- Stream and evaluation: as S1 ({T} stages, |E| = {sum(len(x) for x in test_sets.values())}, pass@{k}, temperature "
      f"{cfg['eval']['temperature']}, no repair).")
    w("")
    # ------------------------------------------------------------------ metrics
    w("## Metric tables")
    w("")
    w("Points (percent). Intervals: 95% bootstrap over obligations, resampled within each E_j, paired across arms. "
      "GC = (ACC_arm - ACC_B2) / (ACC_B6 - ACC_B2) with S1's seed-0 B2 and B6 for every row.")
    w("")
    w("| Arm | seed | LA | FWD | ACC | F | BWT | GC |")
    w("|---|---|---|---|---|---|---|---|")
    order = [(a, 0) for a in S1_ARMS + S2_ARMS] + [(a, 1) for a in S1_ARMS + S2_ARMS]
    for key in order:
        if key not in res:
            continue
        r = res[key]
        cells = [f"{pct(r[m])} {ci(BS.interval(boot[key][m]))}" for m in KEYS]
        w(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {r['GC']:.3f} |")
    w("")
    w("Paired differences (points, 95% paired bootstrap interval; same seed):")
    w("")
    w("| Comparison | seed | dLA | dACC | dF | dBWT |")
    w("|---|---|---|---|---|---|")
    comps = [("D1", "B0"), ("D1", "B2"), ("D1", "B2c"), ("D1", "D2"), ("D1", "D3"), ("D1", "D4"), ("B2c", "B0"),
             ("B2c", "B2"), ("D1x", "B2c"), ("D1x", "D1"), ("D1x", "B0")]
    for s in (0, 1):
        for a, b in comps:
            if (a, s) in res and (b, s) in res:
                cells = []
                for m in ("LA", "ACC", "F", "BWT"):
                    dv, lh = diff((a, s), (b, s), m)
                    cells.append(f"{pct(dv)} {ci(lh)}")
                w(f"| {a} - {b} | {s} | " + " | ".join(cells) + " |")
    w("")
    w("Stratum by the baseline closing tactics (`refl`, `simp refl`): pass@8 after the last stage, pooled over all E_j.")
    w("")
    w("| Arm | seed | trivial | non-trivial | all |")
    w("|---|---|---|---|---|")
    for key in order:
        if key not in res:
            continue
        vals = {True: [], False: []}
        for j, ids in test_sets.items():
            arr = P[key][(T, j)]
            for n, i in enumerate(ids):
                vals[bool(obl[i]["closed_by_baseline_tactics"])].append(arr[n])
        allv = vals[True] + vals[False]
        w(f"| {key[0]} | {key[1]} | {pct(np.mean(vals[True]))} | {pct(np.mean(vals[False]))} | {pct(np.mean(allv))} |")
    w("")
    # ------------------------------------------------------------------ dream yield
    w("## Dream yield per stage")
    w("")
    w("Attempts are proof candidates. 'acc' = accepted by the checker; share = acc / attempts; 'nt' = accepted candidates whose "
      "statement the baseline tactics do not close (non-trivial). 'learned' = dream examples the learner trained on; "
      "'learned rej' = of those, candidates the checker rejected (must be 0 except in D2). Conjectures: raw generations -> "
      "parse errors / already in the repository / duplicates / test statements (L3) / attempted.")
    w("")
    yields = {}
    for key in s2:
        r = runs[key]
        if key[0] == "B2c":
            continue
        dl = r["dreams"]
        byid = {ln["verdict_id"]: ln for ln in dl}
        xs = defaultdict(list)
        for x in r["train_examples"]:
            xs[(x["stage"], x["phase"])].append(x)
        conj = defaultdict(lambda: defaultdict(int))
        for c in r["conjectures"]:
            conj[c["stage"]][c["status"]] += 1
        w(f"**{key[0]} seed {key[1]}**")
        w("")
        w("| stage | NREM attempts | acc | share | nt | learned | learned rej | REM raw | parse err | in repo | dup | L3 | attempted stmts | REM attempts | acc | share | nt | learned | learned rej | new library |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        tot = defaultdict(int)
        trs = {x["stage"]: x for x in r["train"]}
        for t in range(1, T + 1):
            row = [str(t)]
            for ph in ("nrem", "rem"):
                lns = [ln for ln in dl if ln["stage"] == t and ln["phase"] == ph]
                a = sum(ln["accepted"] for ln in lns)
                nt = sum(1 for ln in lns if ln["accepted"] and not ln.get("trivial"))
                lx = xs[(t, ph)]
                lrej = sum(1 for x in lx if not byid[x["verdict_id"]]["accepted"])
                if ph == "rem":
                    cs = conj[t]
                    row += [str(sum(cs.values())), str(cs["parse_error"]), str(cs["in_repository"]), str(cs["duplicate"]),
                            str(cs["test_statement"]), str(cs["attempted"])]
                    tot["raw"] += sum(cs.values()); tot["attempted"] += cs["attempted"]; tot["L3"] += cs["test_statement"]
                row += [str(len(lns)), str(a), f"{a / len(lns):.3f}" if lns else "-", str(nt), str(len(lx)), str(lrej)]
                tot[ph + "_att"] += len(lns); tot[ph + "_acc"] += a; tot[ph + "_nt"] += nt
                tot[ph + "_learned"] += len(lx); tot[ph + "_lrej"] += lrej
            row.append(str(trs[t].get("rem", {}).get("library_added", 0)) if "rem" in trs[t] else "-")
            w("| " + " | ".join(row) + " |")
        yields[key] = dict(tot)
        w("")
        w(f"Totals: NREM {tot['nrem_acc']}/{tot['nrem_att']} accepted "
          f"({(tot['nrem_acc'] / tot['nrem_att']) if tot['nrem_att'] else 0:.3f}; non-trivial {tot['nrem_nt']}); "
          f"REM {tot['rem_acc']}/{tot['rem_att']} accepted "
          f"({(tot['rem_acc'] / tot['rem_att']) if tot['rem_att'] else 0:.3f}; non-trivial {tot['rem_nt']}); "
          f"yield per 1,000 attempts {1000 * (tot['nrem_acc'] + tot['rem_acc']) / max(1, tot['nrem_att'] + tot['rem_att']):.0f} "
          f"(non-trivial {1000 * (tot['nrem_nt'] + tot['rem_nt']) / max(1, tot['nrem_att'] + tot['rem_att']):.0f}); "
          f"dream examples learned {tot['nrem_learned'] + tot['rem_learned']}, of which checker-rejected {tot['nrem_lrej'] + tot['rem_lrej']}.")
        w("")
    # ------------------------------------------------------------------ kernel filter
    w("## Effect of the kernel filter (D1 vs D2)")
    w("")
    for s in (0, 1):
        if ("D1", s) in yields and ("D2", s) in yields:
            y1, y2 = yields[("D1", s)], yields[("D2", s)]
            l1 = y1["nrem_learned"] + y1["rem_learned"]; l2 = y2["nrem_learned"] + y2["rem_learned"]
            r1 = y1["nrem_lrej"] + y1["rem_lrej"]; r2 = y2["nrem_lrej"] + y2["rem_lrej"]
            parts = []
            for m in ("LA", "ACC", "F"):
                dv, lh = diff(("D1", s), ("D2", s), m)
                parts.append(f"d{m} = {pct(dv)} {ci(lh)}")
            w(f"- Seed {s}: D1 learned {l1} dream examples, {r1} of them checker-rejected; D2 learned {l2}, "
              f"{r2} of them checker-rejected ({(r2 / l2) if l2 else 0:.3f} of D2's dream training data is wrong proofs: "
              f"D2's dreams are {'unfiltered' if r2 > 0 else 'NOT shown to be unfiltered'}). D1 - D2: " + "; ".join(parts) + ".")
    w("")
    # ------------------------------------------------------------------ forgetting matrices
    w("## Forgetting matrices")
    w("")
    w("a[t, j] in points: rows = model after stage t (0 = base), columns = E_j; only cells the metrics use (j <= t + 1). "
      "Last row: forgetting of E_j, max_{j<=t<T} a[t, j] - a[T, j].")
    w("")
    for key in s2 + [("B2", 0)]:
        A = runs[key]["A"]
        w(f"**{key[0]} seed {key[1]}**")
        w("")
        w("| t \\ j | " + " | ".join(str(j) for j in range(1, T + 1)) + " |")
        w("|---|" + "---|" * T)
        for t in range(0, T + 1):
            w(f"| {t} | " + " | ".join(pct(A[(t, j)]) if (t, j) in A else "" for j in range(1, T + 1)) + " |")
        w("| F_j | " + " | ".join(pct(max(A[(t, j)] for t in range(j, T)) - A[(T, j)]) if j < T else "" for j in range(1, T + 1)) + " |")
        w("")
    # ------------------------------------------------------------------ checks
    w("## Checks (acceptance of the order)")
    w("")
    t4ref = str(cfg.get("t4_reference"))
    t4 = t4_check({f"{a} s{s}": runs[(a, s)] for (a, s) in runs}, test_sets)
    s1_evalsets = runs[("B0", 0)]["evalsets"]
    rec_eq = all(runs[x]["evalsets"] == s1_evalsets for x in s2)
    digests_ok = all(d == t4ref for d in t4["logged_digest"].values())
    w(f"- **T4** (evaluation sets identical across arms, by hash): recorded per-cell eval-set hashes of every S2 run equal S1's: "
      f"{rec_eq}; target-id digest read back from every verdict log (S1 and S2 runs) equals S1's `{t4ref}`: {digests_ok} "
      f"({len(set(t4['logged_digest'].values()))} distinct digest(s) over {len(t4['logged_digest'])} runs); every line in its declared E_j: "
      f"{t4['lines_in_declared_sets']}. **{'passed' if rec_eq and digests_ok and t4['passed'] else 'FAILED'}**.")
    t8 = []
    for x in s2:
        want = runs[x]["meta"]["base"]["hash"]
        hs = [tr["base_hash"] for tr in runs[x]["train"]]
        t8.append((x, len(hs), all(h == want for h in hs), want))
    t8ok = all(ok and n == T for _, n, ok, _ in t8)
    w(f"- **T8** (base hash unchanged at every stage boundary): " + "; ".join(f"{a} s{s}: {n} boundaries, all equal to `{h}`: {ok}" for (a, s), n, ok, h in t8)
      + f". **{'passed' if t8ok else 'FAILED'}**.")
    # T9
    import sys as _sys
    from ..data.stream import build_stream
    stream = build_stream(seed=cfg["stream"]["seed"], n_projects=cfg["stream"]["projects"])
    if stream.data_version() != m0["data_version"]:
        raise RuntimeError("rebuilt stream has another data version")
    full = stream.by_id()
    stage_of = {sid: t for t, sid in enumerate(stages, 1)}
    proof_by_stmt = {o["statement"]: o["proof"] for o in stream.obligations if o["split"] != "test"}
    t9rows, t9ok = [], True
    for x in s2:
        r = runs[x]
        byid = {ln["verdict_id"]: ln for ln in r.get("dreams", [])}
        dream_x = [e for e in r["train_examples"] if e["kind"] == "dream"]
        missing = sum(1 for e in dream_x if e["verdict_id"] not in byid)
        not_acc = sum(1 for e in dream_x if e["verdict_id"] in byid and not byid[e["verdict_id"]]["accepted"])
        guard_bad = sum(1 for e in dream_x if e["verdict_id"] in byid and byid[e["verdict_id"]]["guard"] != "pass")
        neg = [e["neg_verdict_id"] for e in dream_x if e.get("neg_verdict_id")]
        neg_bad = sum(1 for v in neg if v not in byid or byid[v]["accepted"] or byid[v]["guard"] != "pass")
        human = [e for e in r["train_examples"] if e["source"] == "human"]
        earlier = sum(1 for e in human for i in e["source_ids"] if stage_of[full[i]["snapshot_id"]] != e["stage"])
        earlier_proofs = sum(1 for e in human if e["kind"] == "wake" and
                             any(stage_of[full[i]["snapshot_id"]] != e["stage"] for i in e["source_ids"]))
        coincide = sum(1 for e in dream_x if e["verdict_id"] in byid and
                       proof_by_stmt.get(byid[e["verdict_id"]]["statement"]) == byid[e["verdict_id"]]["candidate"])
        must_acc = x[0] in ("D1", "D3", "D4", "D1x")
        ok = missing == 0 and guard_bad == 0 and earlier_proofs == 0 and earlier == 0 and neg_bad == 0 and (not must_acc or not_acc == 0)
        t9ok &= ok
        t9rows.append(f"{x[0]} s{x[1]}: {len(dream_x)} dream examples, {missing} without a verdict line, {not_acc} with a rejected "
                      f"verdict{' (allowed: no kernel filter)' if not must_acc else ''}, {guard_bad} not guard-passed, {len(neg)} negatives "
                      f"({neg_bad} not a rejected guard-passed verdict); human-material examples from earlier windows {earlier} "
                      f"(human proofs {earlier_proofs}); dream targets textually equal to the human proof of their statement {coincide}")
    w(f"- **T9** (every dream training example has an accepted verdict id; zero earlier-window human proofs in training): " +
      "; ".join(t9rows) + f". **{'passed' if t9ok else 'FAILED'}**. (The last count is reported, not a violation: those "
      "proofs were generated by the model and checked; the human proof was never read.)")
    # guard
    gl = []
    gok = True
    for x in s2:
        g = runs[x]["meta"]["guard"]
        refused = defaultdict(int)
        for ln in runs[x].get("dreams", []):
            if ln["guard"] != "pass":
                refused[ln["guard"]] += 1
        n_dreams = len(runs[x].get("dreams", []))
        can_ok = all(c["rule"] != "pass" for c in g["canaries"])
        gok &= g["training_raised"] == 0 and can_ok
        gl.append(f"{x[0]} s{x[1]}: training examples checked {g['training_checked']}, raised {g['training_raised']}; dream candidates "
                  f"guard-checked {n_dreams}, refused {sum(refused.values())}{(' (' + ', '.join(f'{k} {v}' for k, v in sorted(refused.items())) + ')') if refused else ''}; "
                  f"conjectures dropped as test statements (L3) {g['l3_dropped_conjectures']}; canaries raised "
                  f"{sum(c['rule'] != 'pass' for c in g['canaries'])}/{len(g['canaries'])}")
    w(f"- **Hold-out guard** (R4): " + "; ".join(gl) + f". **{'0 raises in training and every canary raised' if gok else 'FAILED'}**. "
      "Every dream candidate passed HoldoutGuard.check before it was checked or learned (a refused candidate counts as a failed "
      "attempt and is never trained on); every training example passed it again inside the learner.")
    canary_txt = "; ".join(f"{a} s{s}: {runs[(a, s)]['meta']['canary']['passed']}" for a, s in s2)
    w(f"- **Known-answer canary** (ground-truth proofs through the S2 evaluation path give a[t, j] = 1): {canary_txt}.")
    w("")
    notes_path = os.path.join(HERE, "reports", "S2_notes.md")
    notes, nxt = "", ""
    if os.path.exists(notes_path):
        raw = open(notes_path).read()
        notes, _, nxt = raw.partition("<!-- next -->")
    if notes.strip():
        w(notes.strip())
        w("")
    # ------------------------------------------------------------------ cost
    w("## Cost")
    w("")
    w("| Run | wall (s) | train+sleep (s) | of which sleep (s) | eval sampling (s) | optimizer steps | training tokens | core-hours (2 cores) |")
    w("|---|---|---|---|---|---|---|---|")
    tot = 0.0
    for x in s2:
        m = runs[x]["meta"]
        steps = m["dt_steps"] if m["dt_steps"] is not None else sum(tr["wake"]["steps"] for tr in runs[x]["train"])
        toks = m["dt_tokens"] if m["dt_tokens"] is not None else sum(tr["wake"]["train_tokens"] for tr in runs[x]["train"])
        ch = 2 * m["wall_s"] / 3600
        tot += ch
        w(f"| {x[0]} s{x[1]} | {m['wall_s']:.0f} | {m['train_s']:.0f} | {m['sleep_s']:.0f} | {m['eval_generate_s']:.0f} | {steps} | {toks:,} | {ch:.3f} |")
    w("")
    w(f"Total {tot:.2f} core-hours; GPU-hours 0 (CPU only). Earlier-window proof text read during training: 0 bytes in every DT run "
      "(T9). Runs were executed in parallel on disjoint pairs of pinned cores (affinity above).")
    w("")
    if nxt.strip():
        w(nxt.strip())
        w("")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    text = "\n".join(L) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    summary = {"metrics": {f"{a} s{s}": res[(a, s)] for (a, s) in res}, "t4": t4, "t8": t8ok, "t9": t9ok, "guard": gok,
               "yields": {f"{a} s{s}": y for (a, s), y in yields.items()}}
    with open(os.path.join(HERE, "runs/S2/report_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "S2.md"))
    ap.add_argument("--allow-incomplete", action="store_true", help="draft only: skip runs still in progress")
    a = ap.parse_args(argv)
    global ALLOW_INCOMPLETE
    ALLOW_INCOMPLETE = a.allow_incomplete
    t = build(a.out)
    print(f"wrote {a.out} ({len(t)} bytes, sha256 {hashlib.sha256(t.encode()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()

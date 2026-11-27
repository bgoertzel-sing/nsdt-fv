"""Generate reports/S4.md from runs/S4 (rule R3: every number below is recomputed from the logs; the hand-written
part is reports/S4_notes.md, appended verbatim, with the recommended next step after '<!-- next -->').

Metrics (plan section 7, PLN metrics; computed on the candidates of a split, i.e. every cell that is not a
visible true cell; positive = the checker accepted a proof found within budget):
  * AUC: pooled over all candidates, ties counted one half (Mann-Whitney).
  * precision at k: per function, the expected share of positives among its k best-scored candidates under a
    random order of tied scores (k capped at the function's number of candidates), averaged over the functions
    that have at least one hidden positive.
  * calibration: reliability table (predicted value in ten bins of width 0.1 against the share the checker
    accepts), Brier score and expected calibration error (bin-size-weighted |mean predicted - share accepted|).
Intervals: 95% paired bootstrap over functions (10,000 resamples, seed 0): a resample draws functions with
replacement and weights every candidate of a drawn function by its multiplicity; the same draws serve every
method, so differences are paired.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(HERE, "runs", "S4")
METHODS = [("pln_p", "PLN (Appendix E)"), ("frequency", "shape frequency"), ("nn", "nearest neighbour"),
           ("random", "random order")]
BASELINES = ["frequency", "nn", "random"]


def rjsonl(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


# ------------------------------------------------------------------------------------------------- metrics
class Split:
    def __init__(self, rows, ks):
        self.rows = rows
        self.funcs = sorted({r["function"] for r in rows})
        fidx = {f: i for i, f in enumerate(self.funcs)}
        self.f = np.array([fidx[r["function"]] for r in rows])
        self.y = np.array([r["label"] for r in rows], dtype=float)
        self.ks = ks
        self.score = {m: np.array([r[m] for r in rows], dtype=float) for m, _ in METHODS}
        # ties for AUC
        self.groups = {}
        for m in self.score:
            u, inv = np.unique(self.score[m], return_inverse=True)
            self.groups[m] = (len(u), inv)
        # per-function precision at k (fixed; the bootstrap reweights functions)
        self.eligible = np.zeros(len(self.funcs), dtype=bool)
        for i in range(len(self.funcs)):
            self.eligible[i] = self.y[self.f == i].sum() > 0
        self.patk = {m: {k: np.array([self._patk(self.score[m][self.f == i], self.y[self.f == i], k)
                                      for i in range(len(self.funcs))]) for k in ks} for m in self.score}

    @staticmethod
    def _patk(s, y, k):
        n = len(s)
        if n == 0:
            return 0.0
        k = min(k, n)
        order = np.argsort(-s, kind="stable")
        s, y = s[order], y[order]
        taken, tot, i = 0, 0.0, 0
        while taken < k:
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            size = j - i
            pos = y[i:j].sum()
            room = k - taken
            if size <= room:
                tot += pos
                taken += size
            else:
                tot += pos * room / size
                taken += room
            i = j
        return tot / k

    def auc(self, m, w=None):
        ng, inv = self.groups[m]
        w = np.ones(len(self.y)) if w is None else w
        pos = np.bincount(inv, weights=w * self.y, minlength=ng)
        neg = np.bincount(inv, weights=w * (1 - self.y), minlength=ng)
        below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
        P, N = pos.sum(), neg.sum()
        return float((pos * (below + 0.5 * neg)).sum() / (P * N))

    def prec(self, m, k, fw=None):
        fw = np.ones(len(self.funcs)) if fw is None else fw
        e = fw * self.eligible
        return float((self.patk[m][k] * e).sum() / e.sum())

    def prob(self, m):
        return np.clip(self.score[m], 0.0, 1.0)

    def brier(self, m, w=None):
        w = np.ones(len(self.y)) if w is None else w
        return float((w * (self.prob(m) - self.y) ** 2).sum() / w.sum())

    def bins(self, m, nb):
        return np.minimum((self.prob(m) * nb).astype(int), nb - 1)

    def ece(self, m, nb, w=None):
        w = np.ones(len(self.y)) if w is None else w
        b = self.bins(m, nb)
        W = np.bincount(b, weights=w, minlength=nb)
        P = np.bincount(b, weights=w * self.prob(m), minlength=nb)
        Y = np.bincount(b, weights=w * self.y, minlength=nb)
        ok = W > 0
        return float((np.abs(P[ok] - Y[ok])).sum() / W.sum())

    def reliability(self, m, nb):
        b = self.bins(m, nb)
        out = []
        for i in range(nb):
            sel = b == i
            n = int(sel.sum())
            out.append((i, n, float(self.prob(m)[sel].mean()) if n else None, float(self.y[sel].mean()) if n else None))
        return out

    def all_metrics(self, nb, fw=None):
        w = None if fw is None else fw[self.f]
        res = {}
        for m, _ in METHODS:
            r = {"AUC": self.auc(m, w)}
            for k in self.ks:
                r[f"P@{k}"] = self.prec(m, k, fw)
            r["Brier"] = self.brier(m, w)
            r["ECE"] = self.ece(m, nb, w)
            res[m] = r
        return res


def bootstrap(sp, nb, B, seed):
    rng = np.random.default_rng(seed)
    nf = len(sp.funcs)
    names = list(sp.all_metrics(nb).keys())
    keys = list(sp.all_metrics(nb)[names[0]].keys())
    out = {m: {k: np.empty(B) for k in keys} for m in names}
    for b in range(B):
        fw = np.bincount(rng.integers(0, nf, size=nf), minlength=nf).astype(float)
        if (fw * sp.eligible).sum() == 0:
            fw = np.ones(nf)
        r = sp.all_metrics(nb, fw)
        for m in names:
            for k in keys:
                out[m][k][b] = r[m][k]
    return out


def ci(x):
    lo, hi = np.quantile(x, [0.025, 0.975])
    return float(lo), float(hi)


# ------------------------------------------------------------------------------------------------- report
def f3(x):
    return f"{x:.3f}"


def build(out_path):
    meta = json.load(open(os.path.join(RUN, "meta.json")))
    cfg = meta["config"]
    shapes = json.load(open(os.path.join(RUN, "shapes.json")))
    cells = rjsonl(os.path.join(RUN, "cells.jsonl.gz"))
    labels = rjsonl(os.path.join(RUN, "labels.jsonl.gz"))
    ks = [int(k) for k in cfg["metrics"]["ks"]]
    nb = int(cfg["metrics"]["bins"])
    B = int(cfg["bootstrap"]["resamples"])
    seeds = [int(s) for s in cfg["split"]["seeds"]]
    L = []
    w = L.append

    w("# S4: PLN reference fragment on the toy property tables")
    w("")
    w("Generated by \`python -m nsdt_fv.eval.report_s4\` from \`nsdt_fv/runs/S4/\` (rule R3); do not edit by hand. "
      "The hand-written part (reports/S4_notes.md) follows the generated sections.")
    w("")
    # ---------------------------------------------------------------- identification
    t = meta["table"]
    runs = meta["runs"]
    commits = sorted({r["commit"] for r in runs})
    last = runs[-1]
    w("## Identification")
    w("")
    w("- Experiment: S4 (plan 8.1, 9.4 P0 on the toy, Appendix E, A.5 T11). Question: on the toy projects' property "
      "tables, does the Appendix E fragment rank hidden true (function, shape) pairs better than shape frequency, "
      "nearest neighbour by shared properties and random order, and are its truth values calibrated against the checker?")
    w("- Arms (rankers): PLN (Appendix E: transfer through Similar(f, g), deduction through Implies(T1, T) induced over the "
      "visible table, revision with the premise-overlap discount, P(q true) = c s + (1 - c) sT); shape frequency (sT, the "
      "share of functions where T applies that have T visibly); nearest neighbour (share of f's nearest neighbours by "
      "number of shared visible properties that have T); random order (uniform scores). No learner, no model (S4 does "
      "not depend on DT-ref).")
    w(f"- Data: the S1-S3 toy stream (seed {cfg['stream']['seed']}, {cfg['stream']['projects']} projects, data version "
      f"\`{t['stream_data_version']}\`), last snapshot of each project (window 2). Shapes from channel C's logged output "
      f"\`{cfg['shapes']['source']}\` (S3 head 12828b0): {t['statements_read']:,} guard-passed equations, abstracted at each of "
      f"their functions into {t['shapes_seen']:,} one-hole shapes; vocabulary = the {t['vocabulary']:,} shapes C instantiated for "
      f">= {cfg['shapes']['min_support']} functions.")
    w(f"- Property table: {t['cells']:,} cells (function, shape) whose instance type-checks, over {t['functions']} of the "
      f"{t['functions_in_projects']} project functions, in {t['groups']:,} distinct instance statements (statement groups); "
      f"{t['cells_dropped_test_statement']} cells whose instance is a test statement of the stream were dropped.")
    w(f"- Labels (checker only): a cell is true iff a proof found within budget (baseline tactics, then the reference prover "
      f"with {cfg['label']['prover_nodes']} nodes on the project's test-free library) is accepted by the ToyEq checker "
      f"(ToyEqBackend.check_statement). The random-test evaluator ({cfg['label']['cex_inputs']} inputs, seed "
      f"{cfg['label']['cex_seed']}) is a descriptive column only.")
    w(f"- Split: {cfg['split']['hide_fraction']:.0%} of the true statement groups hidden (all cells of a hidden statement "
      f"hidden together), random.Random(seed) over the sorted group keys; seed {seeds[0]} is the primary analysis, seeds "
      f"{', '.join(str(s) for s in seeds[1:])} are replicate splits. Candidates = every cell that is not a visible true cell.")
    w(f"- Seeds: stream {cfg['stream']['seed']}; split {seeds}; random baseline {cfg['random_baseline_seed']} + split seed; "
      f"bootstrap {B} resamples, seed {cfg['bootstrap']['seed']}, paired over functions.")
    w(f"- Code commit(s) of the logged runs: {', '.join('\`' + c[:12] + '\`' for c in commits)}; uncommitted changes under "
      f"nsdt_fv at the last run: {len(last['dirty'])}.")
    w(f"- Versions: Python {last['python']}, numpy {last['numpy']}, PyYAML {last['pyyaml']}; {last['platform']}; CPU affinity "
      f"{last['cpu_affinity']}.")
    w("")
    # ---------------------------------------------------------------- tests and checks
    w("## Checks")
    w("")
    t11 = open(os.path.join(RUN, "logs", "T11.log")).read()
    m = re.search(r"(\d+) passed", t11)
    failed = re.search(r"(\d+) failed", t11)
    w(f"- **T11** (\`nsdt_fv/tests/test_T11.py\`, log \`runs/S4/logs/T11.log\`): {m.group(1) if m else 0} passed, "
      f"{failed.group(1) if failed else 0} failed: revision commutative and associative on counts (random vectors, "
      f"disjoint premises), revision, overlap discount, deduction, inversion, induction, abduction and transfer against "
      f"hand-computed vectors, confidence/count conversion, truth values in [0, 1] on 20,000 random premises and the "
      f"edge grid. **{'passed' if m and not failed else 'FAILED'}**.")
    cn = meta["canary"]
    w(f"- **Known-answer canary** (production ranking path, synthetic table where f0 shares all visible properties with "
      f"a holder of T* and f2 none): P(f0 has T*) = {cn['p_f0_Tstar']}, P(f2 has T*) = {cn['p_f2_Tstar']}, prior "
      f"{cn['prior_Tstar']}; hidden premise read: {cn['used_hidden']}. **{'passed' if cn['passed'] else 'FAILED'}**.")
    for s in seeds:
        lk = json.load(open(os.path.join(RUN, f"split_s{s}", "leak.json")))
        w(f"- **Hidden pairs never reach the PLN premises** (split seed {s}, \`runs/S4/split_s{s}/leak.json\`): the rankers "
          f"receive only the cell list without labels and the ids of the {lk['visible_true_cells']:,} visible true cells; "
          f"PLN read {lk['premise_cells_used_by_pln']:,} distinct premise cells, of which outside the visible set "
          f"{lk['premise_cells_outside_visible']} and hidden {lk['premise_cells_hidden']}; hidden cells in the visible table "
          f"{lk['table_cells_hidden']}; hidden statements with a visible cell {lk['hidden_groups_with_a_visible_cell']}; "
          f"hidden cells missing from the candidates {lk['hidden_cells_not_candidates']}; positive candidates that are not "
          f"hidden {lk['candidate_positives_not_hidden']}. **{'passed' if lk['passed'] else 'FAILED'}**.")
    acc = [r for r in labels if r["accepted"]]
    cex = [r for r in labels if r["counterexample"] not in (None, "limit")]
    unsound = [r for r in acc if r["counterexample"] not in (None, "limit")]
    rej = [r for r in labels if r["how"] == "prover_rejected"]
    w(f"- **Labels and the checker**: every label-1 statement has a proof the checker accepted when it was labelled "
      f"({len(acc):,}); prover proofs the checker rejected (labelled 0): {len(rej)}; accepted statements with a recorded "
      f"counterexample (would be a checker soundness bug): {len(unsound)}.")
    w("")
    # ---------------------------------------------------------------- table description
    w("## The property table")
    w("")
    n_g = len(labels)
    base_acc = sum(1 for r in labels if r["how"] == "baseline")
    prov_acc = sum(1 for r in labels if r["how"] == "prover")
    lim = sum(1 for r in labels if r["counterexample"] == "limit")
    unk = sum(1 for r in labels if not r["accepted"] and r["counterexample"] is None)
    lab = {r["group"]: r["accepted"] for r in labels}
    cpos = sum(1 for c in cells if lab[c["group"]])
    w("| statement groups | accepted (baseline tactics) | accepted (reference prover) | not accepted: counterexample | "
      "not accepted: no counterexample, no proof in budget | evaluation limit | cells | true cells |")
    w("|---|---|---|---|---|---|---|---|")
    w(f"| {n_g:,} | {base_acc:,} | {prov_acc:,} | {len(cex):,} | {unk:,} | {lim} | {len(cells):,} | {cpos:,} |")
    w("")
    w("Per project (cells, true cells, functions):")
    w("")
    w("| project | functions | cells | true cells | share true |")
    w("|---|---|---|---|---|")
    for proj in sorted({c["project"] for c in cells}):
        cc = [c for c in cells if c["project"] == proj]
        pp = sum(1 for c in cc if lab[c["group"]])
        w(f"| {proj} | {len({c['function'] for c in cc})} | {len(cc):,} | {pp:,} | {pp / len(cc):.3f} |")
    w("")
    sup = [v["support"] for v in shapes["vocabulary"]]
    w(f"Shape vocabulary: {len(sup):,} shapes, support (functions C instantiated the shape for) median "
      f"{int(np.median(sup))}, max {max(sup)}. Most frequent shapes in C's output: " +
      "; ".join(f"\`{v['shape']}\` ({v['support']})" for v in sorted(shapes["vocabulary"], key=lambda v: (-v['support'], v['shape']))[:6]) + ".")
    w("")
    # ---------------------------------------------------------------- metric tables
    summary = {}
    for si, s in enumerate(seeds):
        rows = rjsonl(os.path.join(RUN, f"split_s{s}", "scores.jsonl.gz"))
        sp = Split(rows, ks)
        point = sp.all_metrics(nb)
        summary[s] = {"point": point}
        if si > 0:
            continue
        bs = bootstrap(sp, nb, B, int(cfg["bootstrap"]["seed"]))
        npos = int(sp.y.sum())
        w(f"## Metric tables (primary split, seed {s})")
        w("")
        w(f"Candidates {len(rows):,} (hidden true cells = positives {npos:,}, negatives {len(rows) - npos:,}) over "
          f"{len(sp.funcs)} functions; precision at k averaged over the {int(sp.eligible.sum())} functions with at least one "
          f"hidden positive. Intervals: 95% paired bootstrap over functions.")
        w("")
        keys = ["AUC"] + [f"P@{k}" for k in ks] + ["Brier", "ECE"]
        w("| ranker | " + " | ".join(keys) + " |")
        w("|---|" + "---|" * len(keys))
        for m, name in METHODS:
            cellsx = []
            for k in keys:
                lo, hi = ci(bs[m][k])
                cellsx.append(f"{f3(point[m][k])} [{f3(lo)}, {f3(hi)}]")
            w(f"| {name} | " + " | ".join(cellsx) + " |")
        w("")
        w("Brier and ECE: lower is better. The nearest-neighbour score is a 1-NN vote share and the random score a uniform "
          "draw; they are scored as probabilities as they stand.")
        w("")
        # paired differences
        w("Paired differences, PLN minus each baseline and minus the best baseline (best = highest point estimate for AUC "
          "and P@k, lowest for Brier and ECE; chosen per metric):")
        w("")
        w("| comparison | " + " | ".join(keys) + " |")
        w("|---|" + "---|" * len(keys))
        best = {}
        for k in keys:
            lower = k in ("Brier", "ECE")
            best[k] = (min if lower else max)(BASELINES, key=lambda b: point[b][k])
        diffs = {}
        for b in BASELINES + ["best"]:
            cellsx = []
            for k in keys:
                bb = best[k] if b == "best" else b
                d = point["pln_p"][k] - point[bb][k]
                lo, hi = ci(bs["pln_p"][k] - bs[bb][k])
                diffs[(b, k)] = (d, lo, hi, bb)
                cellsx.append(f"{'+' if d >= 0 else ''}{f3(d)} [{f3(lo)}, {f3(hi)}]" + (f" ({dict(METHODS)[bb]})" if b == "best" else ""))
            w(f"| PLN - {'best baseline' if b == 'best' else dict(METHODS)[b]} | " + " | ".join(cellsx) + " |")
        w("")
        summary[s]["diff_best"] = {k: diffs[("best", k)] for k in keys}
        summary[s]["ci"] = {m: {k: ci(bs[m][k]) for k in keys} for m, _ in METHODS}
        # reliability tables
        w(f"### Calibration against the checker (split seed {s})")
        w("")
        w("Reliability table: candidates by predicted value (bin), number of candidates, mean predicted value, share the "
          "checker accepts.")
        w("")
        rel = {m: sp.reliability(m, nb) for m, _ in METHODS}
        w("| bin | " + " | ".join(f"{name}: n / predicted / accepted" for _, name in METHODS) + " |")
        w("|---|" + "---|" * len(METHODS))
        for i in range(nb):
            row = []
            for m, _ in METHODS:
                _, n, pm, ya = rel[m][i]
                row.append(f"{n:,} / -- / --" if n == 0 else f"{n:,} / {pm:.3f} / {ya:.3f}")
            w(f"| [{i / nb:.1f}, {(i + 1) / nb:.1f}{']' if i == nb - 1 else ')'} | " + " | ".join(row) + " |")
        w("")
        # PLN evidence use
        nconc = np.array([r["pln_conclusions"] for r in rows])
        w(f"PLN evidence per candidate: conclusions revised median {int(np.median(nconc))}, candidates with no conclusion "
          f"(P = prior sT) {int((nconc == 0).sum()):,}.")
        w("")
    # ---------------------------------------------------------------- replicate splits
    if len(seeds) > 1:
        w("## Replicate splits (point estimates)")
        w("")
        keys = ["AUC"] + [f"P@{k}" for k in ks] + ["Brier", "ECE"]
        w("| split seed | ranker | " + " | ".join(keys) + " |")
        w("|---|---|" + "---|" * len(keys))
        for s in seeds:
            for m, name in METHODS:
                w(f"| {s} | {name} | " + " | ".join(f3(summary[s]['point'][m][k]) for k in keys) + " |")
        w("")
    # ---------------------------------------------------------------- gate
    s0 = seeds[0]
    w("## Gate evaluation")
    w("")
    db = summary[s0]["diff_best"]
    pts = []
    for k in ks:
        d, lo, hi, bb = db[f"P@{k}"]
        base = summary[s0]["point"][bb][f"P@{k}"]
        rel_gain = d / base if base > 0 else float("nan")
        pts.append(f"P@{k}: PLN {f3(summary[s0]['point']['pln_p'][f'P@{k}'])} vs {dict(METHODS)[bb]} {f3(base)}, relative "
                   f"{rel_gain:+.1%} (difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}])")
    w("S4 has no gate of its own; it feeds G0 through T11 (plan Table 4: G0 needs the acceptance tests of S1-S4). T11: "
      f"{'passed' if m and not failed else 'failed'}. For orientation only, G4 (experiment C6, development repositories) "
      "asks for a 20% relative improvement in precision at k over the best other ranking; on the toy: " + "; ".join(pts) + ".")
    w("")
    # ---------------------------------------------------------------- cost
    lb = meta["label"]
    rank_s = sum(meta["splits"][str(s)]["rank_s"] for s in seeds)
    w("## Cost")
    w("")
    w(f"GPU-hours 0 (CPU only). Labelling: {lb['groups_labelled']:,} statements, {lb['label_core_s'] / 3600:.2f} core-hours "
      f"of prover and evaluator time ({lb['label_wall_s'] / 60:.1f} min wall on {cfg['label']['workers']} cores); ranking "
      f"{rank_s / 60:.1f} min for {len(seeds)} splits on one core; total wall of the logged runs "
      f"{sum(r['wall_s'] for r in runs) / 60:.1f} min.")
    w("")
    notes_path = os.path.join(HERE, "reports", "S4_notes.md")
    notes, nxt = "", ""
    if os.path.exists(notes_path):
        notes, _, nxt = open(notes_path).read().partition("<!-- next -->")
    if notes.strip():
        w(notes.strip())
        w("")
    if nxt.strip():
        w(nxt.strip())
        w("")
    text = "\n".join(L) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    js = {str(s): {"point": summary[s]["point"]} for s in seeds}
    js[str(s0)]["diff_best"] = {k: list(v) for k, v in summary[s0]["diff_best"].items()}
    js[str(s0)]["ci"] = summary[s0]["ci"]
    with open(os.path.join(RUN, "report_summary.json"), "w") as f:
        json.dump(js, f, indent=1, sort_keys=True)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "S4.md"))
    a = ap.parse_args(argv)
    t = build(a.out)
    print(f"wrote {a.out} ({len(t)} bytes, sha256 {hashlib.sha256(t.encode()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()

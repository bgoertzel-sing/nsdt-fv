"""C1-prep (order t-2026-11-28-nsdt-fv-c1-prep): windows, stream, shortlist and the Lean hold-out guard check,
from the source-level history estimates of bench/vsb_history.py. No Lean.

  python -m nsdt_fv.eval.c1prep tables               # runs/C1prep/*.json + reports/C1prep.md
  python -m nsdt_fv.eval.c1prep t3 [--real CLONES]   # runs/C1prep/t3_placeholder.jsonl.gz, t3_*_summary.json

Every number in reports/C1prep.md is generated here from runs/C1prep/history/*.json.gz, bench/vsb/*,
runs/S5/per_repo.json and the T3 summaries. Nothing from the repositories' source is written.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import time

from ..backends.lean_stub import lean_tokens  # noqa: F401
from ..data.holdout import (LEAN_HELD_OUT, LeakError, LeanHoldoutGuard, lean_binder_names, lean_conclusion_split,
                            lean_statement_tokens, lean_top_eq)
from ..data.stream import (LEAN_W_STAGE_C, _live_at, build_lean_stream, load_history, merge_stages, name_hash,
                           repo_windows)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(PKG, "runs", "C1prep")
HIST = os.path.join(RUN, "history")
VSB = os.path.join(PKG, "bench", "vsb")
THRESH_OBLIGATIONS = 300
THRESH_TASKS = 10
PROVISIONAL = ["iris-lean", "loom", "veil", "clean", "capless-lean"]       # plan 5.1
LEAK_TYPES = ["proof_text", "premise_list", "dream_equals_test", "repair_pair", "annotation_with_proof"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tasks():
    with open(os.path.join(VSB, "verisoftbench.jsonl")) as f:
        return [json.loads(l) for l in f]


def load_histories(repos=None):
    out = []
    for p in sorted(glob.glob(os.path.join(HIST, "*.json.gz"))):
        h = load_history(p)
        if repos is None or h["repo"] in repos:
            out.append(h)
    return out


# ----------------------------------------------------------------------------- source-level materializer
class SourceScanMaterializer:
    """Real Lean texts from a local bare clone at the snapshot commit, by the same regex scan as vsb_history.
    Held-out proofs in file prefixes are replaced by ':= by sorry'. Used only for the local real-source guard
    check; nothing it produces is written to the repository. prefix_chars caps src_up_to_decl."""
    kind = "source_scan"

    def __init__(self, clones_dir, prefix_chars=4000):
        self.clones = clones_dir
        self.prefix_chars = prefix_chars

    def texts(self, repo, commit, decl_rows, held_keys):
        from ..bench.vsb_history import Git, h16, scan_lean_spans
        g = Git(os.path.join(self.clones, repo + ".git"))
        want = {(d[0], d[1]) for d in decl_rows}
        out = {}
        for path, sha in sorted(g.lean_tree(commit).items()):
            text, spans = scan_lean_spans(g.text(sha))
            keyed = [((h16(s["qname"]), h16(s["stmt_norm"])), s) for s in spans]
            if not any(k in want for k, _ in keyed):
                continue
            pieces, starts, cut, acc = [], {}, 0, 0
            for k, s in keyed:
                starts.setdefault(k, acc + (s["start"] - cut))
                if k in held_keys and s["start"] >= cut:
                    piece = text[cut:s["stop"]] + " := by sorry\n"
                    pieces.append(piece)
                    acc += len(piece)
                    cut = s["end"]
            pieces.append(text[cut:])
            rtext = "".join(pieces)
            module = path[:-5].replace("/", ".")
            for k, s in keyed:
                if k not in want or k in out:
                    continue
                out[k] = {"decl": s["qname"], "statement": text[s["start"]:s["stop"]].strip(),
                          "proof": text[s["stop"]:s["end"]].strip(), "module": module,
                          "src_up_to_decl": rtext[:starts[k]][-self.prefix_chars:], "scope": "",
                          "deps_project": [], "deps_library": []}
        return out


# ----------------------------------------------------------------------------- T3 adapted to Lean obligations
def alpha_variant(statement: str, tag: str = "dream") -> str:
    """The statement with keyword 'lemma', a new name, binder names renamed, a single top-level '='/'↔' flipped,
    and token-spaced whitespace: what a conjecture generator might emit for a held-out statement."""
    head, concl = lean_conclusion_split(lean_statement_tokens(statement))
    ren = {n: n + "_z" for n in lean_binder_names(head)}                # signature binders: whole statement
    inner = {n: n + "_z" for n in lean_binder_names(concl, brackets=False) if n not in ren}   # conclusion only
    head = [ren.get(t, t) for t in head]
    concl = [ren.get(t, inner.get(t, t)) for t in concl]
    op = lean_top_eq(concl)
    if op is not None:
        concl = concl[op + 1:] + [concl[op]] + concl[:op]
    return "lemma " + tag + " " + " ".join(head + concl)


def broken_proof(proof: str, rng) -> str:
    lines = [l for l in proof.splitlines() if l.strip()]
    if len(lines) > 1:
        del lines[rng.randrange(1, len(lines))]
        return "\n".join(lines)
    return proof.replace("simp", "rfl") if "simp" in proof else proof + " <;> rfl"


def leak_examples(o, rng) -> dict:
    _head, concl = lean_conclusion_split(lean_statement_tokens(o["statement"]))
    deps = o["deps_project"] + o["deps_library"]
    return {
        "proof_text": {"kind": "wake", "statement": o["statement"],
                       "input": o["src_up_to_decl"] + "\n" + o["statement"], "target": o["proof"]},
        "premise_list": {"kind": "premise", "statement": o["statement"], "input": "goal: " + " ".join(concl),
                         "target": ", ".join(deps) if deps else "(none)"},
        "dream_equals_test": {"kind": "dream", "source": "model",
                              "statement": alpha_variant(o["statement"], "dream_" + str(rng.randrange(10 ** 6)))},
        "repair_pair": {"kind": "repair", "statement": o["statement"],
                        "input": "attempt:\n" + broken_proof(o["proof"], rng) + "\nerror: unsolved goals",
                        "target": o["proof"]},
        "annotation_with_proof": {"kind": "annotation", "obligation_id": o["id"],
                                  "input": "similar proof:\n" + o["proof"]},
    }


def control_examples(stream):
    """Legitimate training/mining data that must pass the guard."""
    out = []
    for o in stream.obligations:
        if o["split"] in LEAN_HELD_OUT:
            out.append(("annotation_statement", {"kind": "annotation", "obligation_id": o["id"],
                        "input": "statement:\n" + o["statement"] + "\nmodule: " + o["module"]}))
            out.append(("store_statement", {"kind": "statement", "statement": o["statement"]}))
            continue
        src = {"source_ids": [o["id"]]}
        out.append(("wake", dict(src, kind="wake", statement=o["statement"],
                                 input=o["src_up_to_decl"] + "\n" + o["statement"], target=o["proof"])))
        for dep in o["deps_project"][:3]:
            out.append(("premise", dict(src, kind="premise", statement=o["statement"], input="goal: ...", target=dep)))
        out.append(("dream_rederive", {"kind": "dream", "statement": o["statement"], "source": "model"}))
        out.append(("repair", dict(src, kind="repair", statement=o["statement"],
                                   input="attempt:\nby rfl\nerror: unsolved goals", target=o["proof"])))
    return out


def _try(guard, e):
    try:
        guard.check(e)
        return False, None, None
    except LeakError as err:
        return True, err.rule, err.detail


def run_t3(stream, guard=None, seed=0):
    guard = guard or LeanHoldoutGuard(stream.obligations)
    rng = random.Random(seed)
    lines = []
    held = [o for o in stream.obligations if o["split"] in LEAN_HELD_OUT and o.get("proof")]
    distinctive = set(guard.shingles.tag.values()) | {t for lst in guard.exact.by_first.values() for _, t in lst}
    wake = next((o for o in stream.obligations if o["split"] == "wake"), None)
    for o in held:
        for lt, ex in leak_examples(o, rng).items():
            for variant, e in (("stealth", ex), ("with_id", dict(ex, source_ids=[o["id"]]))):
                raised, rule, _ = _try(guard, e)
                lines.append({"part": "leak", "leak": lt, "variant": variant, "split": o["split"],
                              "target_id": o["id"], "raised": raised, "rule": rule})
        if wake is not None and o["id"] in distinctive:
            raised, rule, _ = _try(guard, {"kind": "wake", "statement": wake["statement"], "input": "Prove:",
                                           "target": o["proof"]})
            lines.append({"part": "decoy", "leak": "proof_text_under_other_statement", "variant": "stealth",
                          "split": o["split"], "target_id": o["id"], "raised": raised, "rule": rule})
    for kind, e in control_examples(stream):
        raised, rule, detail = _try(guard, e)
        lines.append({"part": "control", "leak": kind, "variant": "legit", "split": "", "target_id": "",
                      "raised": raised, "rule": rule, "detail": detail})
    kept = guard.filter_edges(stream.edges)
    kept_ids = {id(e) for e in kept}
    removed = [e for e in stream.edges if id(e) not in kept_ids]
    held_decls = {(o["repo"], o["decl"]) for o in stream.obligations if o["split"] in LEAN_HELD_OUT}

    def is_held(e):
        return e["type"] == "proof_uses" and (e["snapshot_id"].split("@")[0], e["src"]) in held_decls
    lines.append({"part": "L2", "leak": "proof_uses_edges_of_held_out", "variant": "filter", "split": "",
                  "target_id": "", "raised": False, "rule": "L2", "edges_in": len(stream.edges),
                  "edges_removed": len(removed),
                  "removed_exactly_held_out": all(is_held(e) for e in removed) and not any(is_held(e) for e in kept)})
    return lines, guard


def t3_summary(lines):
    s = {"leaks": {}, "controls": {"n": 0, "raised": 0}, "decoys": {"n": 0, "raised": 0}}
    for l in lines:
        if l["part"] == "leak":
            k = s["leaks"].setdefault(l["leak"], {"n": 0, "raised": 0, "by_split": {}})
            k["n"] += 1
            k["raised"] += l["raised"]
            b = k["by_split"].setdefault(l["split"], {"n": 0, "raised": 0})
            b["n"] += 1
            b["raised"] += l["raised"]
        elif l["part"] in ("control", "decoy"):
            key = l["part"] + "s"
            s[key]["n"] += 1
            s[key]["raised"] += l["raised"]
        elif l["part"] == "L2":
            s["L2"] = {k: l[k] for k in ("edges_in", "edges_removed", "removed_exactly_held_out")}
    return s


# ----------------------------------------------------------------------------- tables
def canary(hists, tasks):
    """Recall of the scanner on the benchmark: are the benchmark tasks' qualified names live at c* in the scan?
    A structural check of the scanner, not of the windows."""
    rows, miss_all = [], {}
    for h in hists:
        ts = [t for t in tasks if t["lean_root"] == h["repo"]]
        last = h["n_history_commits"] - 1
        live = {d[0] for d in h["decls"] if _live_at(d[4], last) >= 0}
        miss = sorted(t["thm_name"] for t in ts if name_hash(t["thm_name"]) not in live)
        rows.append({"repo": h["repo"], "tasks": len(ts), "found": len(ts) - len(miss)})
        if miss:
            miss_all[h["repo"]] = miss
    return {"rows": rows, "tasks": sum(r["tasks"] for r in rows), "found": sum(r["found"] for r in rows),
            "missing": miss_all}


def history_rows(hists):
    rows = []
    for h in hists:
        c = h["commits"]
        nh = h["n_history_commits"]
        hist_new = [x[3] for x in c[:nh]]
        imax = max(range(nh), key=lambda i: hist_new[i])
        rows.append({
            "repo": h["repo"], "pinned_commit": h["pinned_commit"], "pinned_date": c[nh - 1][1],
            "first_date": c[0][1], "n_commits": nh, "live_at_pinned": h["totals"]["live_at_pinned"],
            "live_at_pinned_names": h["totals"]["live_at_pinned_names"],
            "live_at_pinned_testlike": h["totals"]["live_at_pinned_testlike"],
            "entries_to_pinned": sum(hist_new), "commits_with_new": sum(1 for x in hist_new if x),
            "max_single_commit_new": hist_new[imax], "max_single_commit_sha": c[imax][0],
            "forward_commits": h["n_forward_commits"], "head_at_scan": h["head_at_scan"],
            "scan_version": h["scan_version"],
        })
    return rows


def shortlist(per_repo, hist_rows, windows):
    s5 = {r["repo"]: r for r in per_repo}
    hr = {r["repo"]: r for r in hist_rows}
    wn = {w["repo"]: w for w in windows}
    rows = []
    for repo in sorted(s5, key=lambda r: (-s5[r]["tasks"], r)):
        r = s5[repo]
        w = wn.get(repo)
        n_stream = sum(x["n_new"] for x in w["windows"]) if w else 0
        n_bench = sum(x["n_benchmark"] for x in w["windows"]) if w else 0
        live = hr[repo]["live_at_pinned"] if repo in hr else 0
        rows.append({
            "repo": repo, "category": r["category"][0], "tasks": r["tasks"], "large": r["large"],
            "license": r["license"], "has_mathlib": r["has_mathlib"], "lean_toolchain": r["lean_toolchain"],
            "est_window_obligations": n_stream, "est_window_obligations_ex_benchmark": n_stream - n_bench,
            "est_live_at_pinned": live, "ge_10_tasks": r["tasks"] >= THRESH_TASKS,
            "ge_300_est": n_stream - n_bench >= THRESH_OBLIGATIONS and live >= THRESH_OBLIGATIONS,
            "windowable": w is not None, "build_under_30_min": "C0", "provisional_5_1": repo in PROVISIONAL,
        })
    for x in rows:
        x["eligible_without_lean"] = x["ge_10_tasks"] and x["ge_300_est"] and x["windowable"]

    def rank(x):
        return (not x["provisional_5_1"], x["large"], -x["tasks"], -x["est_window_obligations_ex_benchmark"], x["repo"])
    cats = sorted({x["category"] for x in rows})
    by_cat = {c: sorted([x for x in rows if x["category"] == c and x["eligible_without_lean"]], key=rank) for c in cats}
    chosen = [by_cat[c][0]["repo"] for c in cats if by_cat[c]]
    rest = sorted([x for c in cats for x in by_cat[c][1:]], key=lambda x: (x["large"], -x["tasks"], x["repo"]))
    return {"rows": rows, "categories": cats, "eligible_by_category": {c: [x["repo"] for x in by_cat[c]] for c in cats},
            "chosen": chosen, "alternates": [x["repo"] for x in rest[:2]], "five_categories": len(chosen) == 5,
            "rule": "5.1: build < 30 min on Colab (C0), >= 300 extractable obligations, >= 10 benchmark tasks, "
                    "five different categories",
            "selection": "per category: plan-5.1 provisional first, then not large, then more tasks, then more "
                         "estimated obligations; alternates: the next eligible repositories not chosen, not large "
                         "first, then more tasks"}


def fmt_date(s):
    return s[:10] if s else "-"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return "\n".join(out)


def compute_tables():
    """{relative path under nsdt_fv/: text} for every generated table file and the report."""
    tasks = load_tasks()
    hists = load_histories()
    per_repo = json.load(open(os.path.join(PKG, "runs", "S5", "per_repo.json")))
    can = canary(hists, tasks)
    hrows = history_rows(hists)
    windows, not_windowable = [], []
    for h in hists:
        bench = {name_hash(t["thm_name"]) for t in tasks if t["lean_root"] == h["repo"]}
        try:
            windows.append(repo_windows(h, LEAN_W_STAGE_C, exclude_names=bench))
        except ValueError as e:
            not_windowable.append(str(e))
    stages = merge_stages(windows)
    sl = shortlist(per_repo, hrows, windows)
    dev_stages = merge_stages([w for w in windows if w["repo"] in sl["chosen"]])
    inputs = {"repos.json": sha256_file(os.path.join(VSB, "repos.json")),
              "verisoftbench.jsonl": sha256_file(os.path.join(VSB, "verisoftbench.jsonl")),
              "s5_per_repo.json": sha256_file(os.path.join(PKG, "runs", "S5", "per_repo.json")),
              "history": {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(HIST, "*.json.gz")))},
              "not_windowable": not_windowable}
    files = {}
    for name, obj in (("canary.json", can), ("history_summary.json", hrows), ("windows.json", windows),
                      ("stream_all.json", {"stages": stages}),
                      ("stream_dev.json", {"repos": sl["chosen"], "stages": dev_stages}),
                      ("shortlist.json", sl), ("inputs.json", inputs)):
        files[f"runs/C1prep/{name}"] = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    t3 = {}
    for tag in ("placeholder", "real"):
        p = os.path.join(RUN, f"t3_{tag}_summary.json")
        t3[tag] = json.load(open(p)) if os.path.exists(p) else None
    files["reports/C1prep.md"] = render(can, hrows, windows, stages, sl, dev_stages, inputs, t3)
    return files, sl, can


def cmd_tables(args):
    files, sl, can = compute_tables()
    os.makedirs(RUN, exist_ok=True)
    for rel, text in files.items():
        with open(os.path.join(PKG, rel), "w") as f:
            f.write(text)
    print(f"shortlist {sl['chosen']} alternates {sl['alternates']}; canary {can['found']}/{can['tasks']}")


def render(can, hrows, windows, stages, sl, dev_stages, inputs, t3):
    wn = {w["repo"]: w for w in windows}
    L = ["# C1prep: candidate repositories' history, provisional windows, shortlist, Lean-side stream and guard\n",
         "Generated by ¤python -m nsdt_fv.eval.c1prep tables¤; hand-written notes are in reports/C1prep_notes.md.\n",
         "**All declaration counts here are source-level estimates** (a regular-expression scan of the .lean files "
         "at every commit, bench/vsb_history.py). The Lean extractor decides at C1.\n",
         "## Identification\n",
         "- Experiment: C1prep, the parts of C0/C1 that need neither Lean nor a GPU (order t-2026-11-28-nsdt-fv-c1-prep).",
         "- Arms and bases: none (data preparation). Seed: 0 (only the T3 proof mutations).",
         "- Code commit: the commit that contains this file (see the result post); base S5 head 791589b.",
         f"- Data version: repos.json sha256 {inputs['repos.json'][:16]}, verisoftbench.jsonl sha256 "
         f"{inputs['verisoftbench.jsonl'][:16]}, S5 per_repo.json sha256 {inputs['s5_per_repo.json'][:16]}; "
         "history files by sha256 in runs/C1prep/inputs.json.",
         f"- Scanner {sorted({r['scan_version'] for r in hrows})}; clones ¤git clone --bare --filter=blob:limit=2m¤; "
         "history = first-parent commits up to the pinned commit c*; forward = first-parent commits c*..HEAD at scan "
         "time (HEAD per repository in runs/C1prep/history_summary.json).\n",
         "## 1. History per repository (estimates)\n",
         "live@c*: theorem/lemma keys (qualified name + statement hash) present at c*. names@c*: distinct qualified "
         "names. test-like: of live@c*, those in a path with a test/bench/example directory. entries: key entries "
         "summed over the commits to c*. max/commit: the most keys entering in one first-parent commit (a squash or "
         "a merge makes the history coarse). fwd: first-parent commits after c*, and keys present at HEAD but not "
         "at c*. Per-commit counts (sha, date, files changed, new, gone, live) are in runs/C1prep/history/<repo>.json.gz.\n"]
    rows = []
    for r in sorted(hrows, key=lambda r: r["repo"].lower()):
        w = wn.get(r["repo"])
        rows.append([r["repo"], r["pinned_commit"][:10], fmt_date(r["first_date"]), fmt_date(r["pinned_date"]),
                     r["n_commits"], r["live_at_pinned"], r["live_at_pinned_names"], r["live_at_pinned_testlike"],
                     r["entries_to_pinned"], f"{r['max_single_commit_new']} ({r['max_single_commit_sha'][:8]})",
                     r["forward_commits"], w["forward"]["n_new"] if w else "-"])
    L.append(md_table(["repo", "c*", "first commit", "c* date", "commits to c*", "live@c*", "names@c*", "test-like",
                       "entries", "max/commit", "fwd commits", "fwd new keys"], rows))
    L.append(f"\n{len(hrows)} of 23 repositories cloned and scanned; "
             f"{sum(r['live_at_pinned'] for r in hrows)} keys live at the pinned commits in total.\n")
    L.append("### Scanner canary: benchmark task names\n")
    L.append(f"Of the {can['tasks']} benchmark tasks, {can['found']} ({100.0 * can['found'] / can['tasks']:.1f}%) have "
             "their qualified thm_name among the keys live at c* in the scan. Missed: "
             + ("; ".join(f"{k} {len(v)}" for k, v in sorted(can["missing"].items())) or "none")
             + " (names in runs/C1prep/canary.json).\n")
    L.append("## 2. Provisional windows (W = 2, section 5.2)\n")
    L.append("Cut so that the two windows have about equal numbers of new keys (new = present in the window's "
             "snapshot and absent from the previous one, by name and statement hash). bench: new keys named like a "
             "benchmark task, held out (split benchmark) before the 80/20 wake/test split by entry commit date. "
             "Dates are committer dates (UTC).\n")
    rows = []
    for w in sorted(windows, key=lambda w: w["repo"].lower()):
        for x in w["windows"]:
            rows.append([w["repo"], x["window"], x["first_commit"][:8], x["end_commit"][:8], fmt_date(x["start_date"]),
                         fmt_date(x["end_date"]), x["n_commits"], x["n_new"], x["n_benchmark"], x["n_wake"], x["n_test"]])
    L.append(md_table(["repo", "w", "first", "end (cut)", "from", "to", "commits", "new", "bench", "wake", "test"], rows))
    if inputs["not_windowable"]:
        L.append("\nNot windowable: " + "; ".join(inputs["not_windowable"]) + "\n")
    L.append(f"\n### Stream of all {len(wn)} repositories ({len(stages)} stages), by window end date, ties by repository name\n")
    L.append(", ".join(f"{i}. {sid} ({fmt_date(d)})" for i, (sid, d) in enumerate(stages, 1)) + "\n")
    L.append(f"### Development stream of the chosen repositories ({len(dev_stages)} stages)\n")
    L.append(md_table(["stage", "snapshot", "end date"], [[i, sid, fmt_date(d)] for i, (sid, d) in enumerate(dev_stages, 1)]))
    L.append("\n## 3. Shortlist against 5.1\n")
    L.append("Sources: tasks, category, large, license from S5 (runs/S5/per_repo.json, i.e. verisoftbench.jsonl and "
             "repos.json at 8f5e8b36); estimated obligations from section 2 (window total minus benchmark-named keys) "
             "and section 1 (live@c*), both must reach 300; build time is measured at C0.\n")
    rows = []
    for x in sl["rows"]:
        rows.append([x["repo"], x["category"], x["tasks"], "yes" if x["large"] else "",
                     x["est_window_obligations_ex_benchmark"], x["est_live_at_pinned"],
                     "met" if x["ge_10_tasks"] else "failed", "met (est.)" if x["ge_300_est"] else "failed (est.)",
                     x["build_under_30_min"], "yes" if x["provisional_5_1"] else "", x["license"],
                     "**chosen**" if x["repo"] in sl["chosen"] else ("alternate" if x["repo"] in sl["alternates"] else "")])
    L.append(md_table(["repo", "category", "tasks", "large", "est. oblig. (windows)", "est. live@c*", ">=10 tasks",
                       ">=300 oblig.", "build <30 min", "plan 5.1", "license", "status"], rows))
    L.append("\nEligible on everything checkable without Lean, by category: " + "; ".join(
        f"{c}: {', '.join(v) if v else 'none'}" for c, v in sl["eligible_by_category"].items()) + ".\n")
    L.append(f"Chosen ({'five categories' if sl['five_categories'] else 'FEWER THAN FIVE categories'}): "
             + ", ".join(sl["chosen"]) + ". Alternates: " + ", ".join(sl["alternates"]) + ". Selection rule: "
             + sl["selection"] + ".\n")
    L.append("## 4. Lean-side stream builder and hold-out guard (T3 adapted to Lean obligations)\n")
    for tag, name in (("placeholder", "Placeholder texts from the committed metadata (what the tests run)"),
                      ("real", "Real source texts from the local clones (counts only; no text committed)")):
        t = t3.get(tag)
        if not t:
            L.append(f"{name}: not run.\n")
            continue
        L.append(f"**{name}.** Repositories: {', '.join(t['repos'])}. Obligations {t['obligations']} ("
                 + ", ".join(f"{k} {v}" for k, v in t["held_by_split"].items())
                 + f"); held-out proofs with distinctive text: {t['distinctive']}.\n")
        rows = [[lt, v["n"], v["raised"], "; ".join(f"{s}: {b['raised']}/{b['n']}" for s, b in sorted(v["by_split"].items()))]
                for lt, v in sorted(t["summary"]["leaks"].items())]
        L.append(md_table(["leak (stealth and with id)", "injected", "raised", "raised/injected by split"], rows))
        s = t["summary"]
        L.append(f"\nControls (legitimate examples): {s['controls']['raised']} of {s['controls']['n']} raised. "
                 f"Decoys (a distinctive held-out proof under another statement): {s['decoys']['raised']} of "
                 f"{s['decoys']['n']} raised. L2: {s['L2']['edges_removed']} of {s['L2']['edges_in']} edges removed; "
                 f"exactly the held-out proof_uses edges: {s['L2']['removed_exactly_held_out']}.\n")
    L.append("## 5. Gate evaluation\n")
    L.append("No gate belongs to this order. The acceptance items are checked in reports/C1prep_notes.md.\n")
    L.append("## 6. Cost\n")
    L.append("GPU-hours: 0. Core-hours: below 0.2 (clones a few seconds each; history scan under a minute; T3 "
             "about 3 minutes; tables under a second; see the notes).\n")
    return ("\n".join(L) + "\n").replace("¤", "\u0060")


def stream_for(repos, materializer=None):
    return build_lean_stream(load_histories(set(repos)), LEAN_W_STAGE_C, materializer=materializer,
                             benchmark_tasks=load_tasks())


def cmd_t3(args):
    sl = json.load(open(os.path.join(RUN, "shortlist.json")))
    repos = sl["chosen"] + sl["alternates"]
    runs = [("placeholder", None)] + ([("real", SourceScanMaterializer(args.real))] if args.real else [])
    for tag, mat in runs:
        t0 = time.time()
        s = stream_for(repos, mat)
        lines, g = run_t3(s, LeanHoldoutGuard(s.obligations))
        res = {"repos": repos, "obligations": len(s.obligations),
               "held_by_split": {k: sum(o["split"] == k for o in s.obligations) for k in ("wake",) + LEAN_HELD_OUT},
               "distinctive": g.n_distinctive_proofs, "summary": t3_summary(lines),
               "data_version": s.data_version(), "seconds": round(time.time() - t0, 1),
               "control_raises": [{k: l[k] for k in ("leak", "rule")} for l in lines if l["part"] == "control" and l["raised"]][:50],
               "missed_leaks": [{k: l[k] for k in ("leak", "variant", "split")} for l in lines
                                if l["part"] == "leak" and not l["raised"]][:50]}
        if tag == "placeholder":
            import gzip
            with gzip.GzipFile(os.path.join(RUN, "t3_placeholder.jsonl.gz"), "wb", mtime=0) as f:
                for l in lines:
                    f.write((json.dumps(l, sort_keys=True) + "\n").encode())
        with open(os.path.join(RUN, f"t3_{tag}_summary.json"), "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
            f.write("\n")
        print(tag, json.dumps(res["summary"]), res["seconds"], "s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tables")
    t3 = sub.add_parser("t3")
    t3.add_argument("--real", default=None, help="directory of bare clones for the real-source run")
    a = ap.parse_args(argv)
    {"tables": cmd_tables, "t3": cmd_t3}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Experiment S5: read VeriSoftBench's data/verisoftbench.jsonl and repos.json (pinned commit, sha256
checked), record every field with its type and an example, map the fields onto Appendix B, count
tasks per repository (regular/large split against 381/119), give the dependency-depth distributions,
and check the provisional development shortlist against the rule of plan section 5.1.

    python3 -m nsdt_fv.bench.vsb_inspect            # writes nsdt_fv/runs/S5/*.json and nsdt_fv/reports/S5.md

No network, no Lean: the inputs are the committed files under nsdt_fv/bench/vsb/ (refetch with
nsdt_fv/bench/vsb_fetch.sh). Output is deterministic (no timestamps).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics

from ..data.schemas import BENCHMARK_FIELD_MAP, OBLIGATION_FIELDS

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSB_DIR = os.path.join(PKG, "bench", "vsb")
COMMIT = "8f5e8b36ab5fbdec6fd8d8f4ee0580d29647097a"
SHA256 = {
    "verisoftbench.jsonl": "ea584250021fbfaddc060834be2e05b9b2fae4f720c770a9932c71d039a50961",
    "repos.json": "c9cc7cb9650c482d81ee78d7f704c61d0d6073418064d11c55674938f35760de",
}
SHORTLIST = ["iris-lean", "loom", "veil", "clean", "capless-lean"]     # plan 5.1, provisional
# plan 5.1 names that differ from the benchmark's lean_root / repos.json keys
PLAN_NAMES = {"LeroyCompilerVerif": "LeroyCompilerVerificationCourse",
              "lean-formal-reasoning": "lean-formal-reasoning-program"}
PLAN_LIST = ["ArkLib", "capless-lean", "clean", "lean-mlir", "VCV-io", "iris-lean", "loom", "splean", "TTBFL",
             "FVIntmax", "juvix-lean", "lean-formal-reasoning", "wadray_verification", "LeroyCompilerVerif",
             "CvxLean", "pcf-lean", "LeanExprEvaluator", "veil", "verified-compiler", "lean-hoare",
             "IntroEffects", "formal-snarks-project", "Lentil"]
MIN_TASKS = 10

# Appendix B obligation field -> (benchmark source, relation). "same" = same meaning, adopted in
# data/schemas.BENCHMARK_FIELD_MAP; "partial" = related but not the same thing; None = no benchmark source.
APPENDIX_B = {
    "id": (None, "no source: the benchmark's `id` is an integer task index 1..500, not <repo>@<commit>:<decl>; "
                 "it can be built from lean_root, the repos.json commit and thm_name"),
    "repo": ("lean_root", "same"),
    "commit": ("repos.json repos.<lean_root>.commit", "partial: per repository, not per task (one pinned commit)"),
    "snapshot_id": (None, "no source: windows/snapshots are ours (section 5.2)"),
    "module": ("rel_path", "partial: file path (e.g. ArkLib/Data/Nat/Bitwise.lean); module name = path with / -> . "
                           "and no .lean"),
    "decl": ("thm_name", "same"),
    "kind": (None, "no source field; thm_stmt's keyword (theorem/lemma, after attributes) gives it"),
    "statement": ("thm_stmt", "same (target_theorem is thm_stmt with the trailing ':=' ; see the field notes)"),
    "src_up_to_decl": ("local_ctx", "partial: the file context before the declaration as the benchmark builds it "
                                    "(proofs may be elided), not checked to be the raw file prefix"),
    "scope": ("imports", "partial: imports only; namespaces and variables are inside local_ctx"),
    "deps_project": ("used_repo_defs, repo_lemmas, used_local_defs, used_local_lemmas",
                     "partial: lists of {name, content}; definitions and lemmas used, from the repo and the file"),
    "deps_library": ("used_lib_defs, lib_lemmas", "partial: lists of {name, module}"),
    "n_deps_direct": (None, "no source field; len() of the six used_*/lemma lists is a proxy"),
    "n_deps_transitive": ("transitive_dep_count", "same"),
    "proof": ("ground_truth_proof", "same (text starts at ':=' on most tasks; count in the field notes)"),
    "first_seen": (None, "no source: needs git history (C1)"),
    "window": (None, "no source: ours (section 5.2)"),
    "split": (None, "no source: every benchmark task is split 'benchmark'"),
    "closed_by_baseline_tactics": (None, "no source: needs Lean (C1)"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(vsb_dir=VSB_DIR):
    for fn, want in SHA256.items():
        got = sha256(os.path.join(vsb_dir, fn))
        if got != want:
            raise SystemExit(f"{fn}: sha256 {got} != pinned {want}; refetch with vsb_fetch.sh")
    with open(os.path.join(vsb_dir, "verisoftbench.jsonl")) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    with open(os.path.join(vsb_dir, "repos.json")) as f:
        repos_doc = json.load(f)
    with open(os.path.join(vsb_dir, "tree.txt")) as f:
        tree = [ln.rstrip("\n") for ln in f if ln.strip()]
    return rows, repos_doc, tree


def _type(v):
    if isinstance(v, list):
        inner = sorted({_type(x) for x in v})
        return f"list[{'|'.join(inner)}]" if inner else "list"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_type(x)}" for k, x in sorted(v.items())) + "}"
    return type(v).__name__


def _example(v, n=110):
    if isinstance(v, list):
        if not v:
            return "[]"
        x = v[0]
        if isinstance(x, dict):
            x = {k: (s[:60] + "..." if isinstance(s, str) and len(s) > 60 else s) for k, s in x.items()}
        s = json.dumps(x, ensure_ascii=False)
        return f"[{s[:n]}{'...' if len(s) > n else ''}, ...] ({len(v)} items)"
    s = json.dumps(v, ensure_ascii=False)
    return s[:n] + ("..." if len(s) > n else "")


def field_table(rows):
    keys = list(rows[0].keys())
    allkeys = collections.Counter(k for r in rows for k in r)
    out = []
    ex = rows[0]
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if all(isinstance(v, list) for v in vals):
            el = sorted({_type(x) for v in vals for x in v})
            ty = f"list[{'|'.join(el)}]"
        else:
            ty = "|".join(sorted({_type(v) for v in vals}))
        out.append({"field": k, "type": ty, "present_in": allkeys[k], "example_task_id": ex["id"],
                    "example": _example(ex[k])})
    extra = sorted(k for k in allkeys if k not in keys)
    return out, extra


def _dist(xs):
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "median": statistics.median(xs), "max": xs[-1],
            "mean": round(statistics.fmean(xs), 2)}


def analyse(rows, repos_doc, tree):
    repos = repos_doc["repos"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["lean_root"]].append(r)
    # large repositories: the benchmark's pre-generated full-context prompts (prompts/full_limited/<repo>/),
    # read from the pinned commit's file listing (the task file has no regular/large field)
    large_prompts = collections.Counter(p.split("/")[2] for p in tree if p.startswith("prompts/full_limited/"))
    large = sorted(large_prompts)
    per_repo = []
    for name in sorted(by, key=lambda n: (-len(by[n]), n)):
        rs = by[name]
        meta = repos.get(name, {})
        cats = collections.Counter(r["category"] for r in rs)
        per_repo.append({
            "repo": name, "tasks": len(rs), "large": name in large,
            "large_prompt_files": large_prompts.get(name, 0),
            "category": sorted(cats), "subset_aristotle": sum(bool(r["subset_aristotle"]) for r in rs),
            "nesting_depth": _dist([r["nesting_depth"] for r in rs]),
            "transitive_dep_count": _dist([r["transitive_dep_count"] for r in rs]),
            "commit": meta.get("commit"), "commit_date": meta.get("commit_date"),
            "license": meta.get("license"), "lean_toolchain": meta.get("lean_toolchain"),
            "has_mathlib": meta.get("has_mathlib"), "in_repos_json": name in repos,
        })
    n_large = sum(p["tasks"] for p in per_repo if p["large"])
    stripped = [r["target_theorem"].rstrip() for r in rows]
    totals = {
        "tasks": len(rows), "distinct_ids": len({r["id"] for r in rows}),
        "repos_in_task_file": len(by), "repos_in_repos_json": len(repos),
        "repos_json_not_in_tasks": sorted(set(repos) - set(by)),
        "tasks_repo_not_in_repos_json": sorted(set(by) - set(repos)),
        "large_repos": large, "large_tasks": n_large, "regular_tasks": len(rows) - n_large,
        "regular_repos": len(by) - len(large),
        "large_prompt_files_equal_task_counts": all(large_prompts[n] == len(by[n]) for n in large),
        "reconciles_381_119": (len(rows) - n_large, n_large) == (381, 119) and len(by) - len(large) == 21,
        "category": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
        "one_category_per_repo": all(len({r["category"] for r in rs}) == 1 for rs in by.values()),
        "nesting_depth_hist": dict(sorted(collections.Counter(r["nesting_depth"] for r in rows).items())),
        "transitive_dep_count": _dist([r["transitive_dep_count"] for r in rows]),
        "nesting_depth": _dist([r["nesting_depth"] for r in rows]),
        "subset_aristotle": sum(bool(r["subset_aristotle"]) for r in rows),
        "thm_stmt_eq_target_theorem": sum(r["thm_stmt"] == r["target_theorem"] for r in rows),
        "target_theorem_eq_thm_stmt_plus_assign": sum(s.endswith(":=") and s[:-2].rstrip() == r["thm_stmt"].rstrip()
                                                      for s, r in zip(stripped, rows)),
        "target_theorem_prefix_of_thm_stmt_cut_at_inner_assign": sorted(
            r["id"] for s, r in zip(stripped, rows)
            if s.endswith(":=") and s[:-2].rstrip() != r["thm_stmt"].rstrip()
            and r["thm_stmt"].startswith(s[:-2].rstrip())),
        "proof_starts_with_assign": sum(r["ground_truth_proof"].lstrip().startswith(":=") for r in rows),
        "proof_not_assign": {str(r["id"]): r["ground_truth_proof"].lstrip().split()[0] for r in rows
                             if not r["ground_truth_proof"].lstrip().startswith(":=")},
        "repos_json_fields": dict(sorted(collections.Counter(k for v in repos.values() for k in v).items())),
        "repos_json_top_level": sorted(repos_doc),
    }
    plan_map = {p: PLAN_NAMES.get(p, p) for p in PLAN_LIST}
    totals["plan_names_found"] = {p: (m in by) for p, m in plan_map.items()}
    totals["plan_name_mismatches"] = {p: m for p, m in plan_map.items() if p != m}
    return per_repo, totals


def shortlist_check(per_repo):
    rp = {p["repo"]: p for p in per_repo}
    rows = []
    cat_count = collections.Counter(rp[n]["category"][0] for n in SHORTLIST)
    for n in SHORTLIST:
        p = rp[n]
        rows.append({
            "repo": n, "tasks": p["tasks"], "category": p["category"][0],
            "ge_10_tasks": "met" if p["tasks"] >= MIN_TASKS else "failed",
            "distinct_category": "met" if cat_count[p["category"][0]] == 1 else "failed",
            "build_under_30_min": "C0/C1", "ge_300_obligations": "C0/C1",
            "large": p["large"], "license": p["license"], "has_mathlib": p["has_mathlib"],
            "lean_toolchain": p["lean_toolchain"], "commit": p["commit"],
        })
    pool = collections.defaultdict(list)
    for p in per_repo:
        if p["tasks"] >= MIN_TASKS and not p["large"]:
            pool[p["category"][0]].append({"repo": p["repo"], "tasks": p["tasks"], "license": p["license"],
                                            "has_mathlib": p["has_mathlib"], "lean_toolchain": p["lean_toolchain"]})
    all_cats = sorted({p["category"][0] for p in per_repo})
    return {"shortlist": rows, "categories_covered": dict(cat_count),
            "five_distinct_categories": len(cat_count) == len(SHORTLIST),
            "benchmark_categories": all_cats, "pool_by_category": {c: pool.get(c, []) for c in all_cats}}


def appendix_b_map(fields):
    names = {f["field"] for f in fields}
    out = [{"appendix_b": o, "benchmark": s, "relation": r} for o, (s, r) in APPENDIX_B.items()]
    used = set()
    for s, _ in APPENDIX_B.values():
        if s and not s.startswith("repos.json"):
            used |= {x.strip() for x in s.split(",")}
    unmapped = sorted(names - used)
    same = {s: o for o, (s, r) in APPENDIX_B.items() if r.startswith("same")}
    consistent = (all(k in names for k in BENCHMARK_FIELD_MAP) and
                  all(v in OBLIGATION_FIELDS for v in BENCHMARK_FIELD_MAP.values()) and
                  BENCHMARK_FIELD_MAP == same and set(APPENDIX_B) == set(OBLIGATION_FIELDS))
    return out, unmapped, consistent


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return "\n".join(out)


def _d(d):
    return f"{d['min']} / {d['median']:g} / {d['max']}"


def report(fields, extra, per_repo, totals, sl, bmap, unmapped, consistent, sums):
    L = []
    A = L.append
    A("# S5: VeriSoftBench files and the development shortlist")
    A("")
    A("Generated by `python3 -m nsdt_fv.bench.vsb_inspect` from the committed files in `nsdt_fv/bench/vsb/`"
      " (raw numbers in `nsdt_fv/runs/S5/`). Hand-written commentary: `nsdt_fv/reports/S5_notes.md`.")
    A("")
    A("## Identification")
    A("")
    A("- Experiment S5 (plan 8.1, A.7). No arms, bases or seeds (no training). CPU only, no Lean, no build.")
    A(f"- Source: github.com/utopia-group/VeriSoftBench at commit `{COMMIT}` (main on 2026-11-28 00:41 UTC),"
      " fetched from raw.githubusercontent.com by `nsdt_fv/bench/vsb_fetch.sh` (file listing via a blob-less"
      " git clone from github.com); benchmark licence MIT (copied as `nsdt_fv/bench/vsb/LICENSE`).")
    A("- Files and sha256:")
    for fn, h in sums.items():
        A(f"  - `{fn}`: `{h}`")
    A("")
    A("## Fields of data/verisoftbench.jsonl")
    A("")
    A(f"{totals['tasks']} lines, {totals['distinct_ids']} distinct `id` values; every line has the same "
      f"{len(fields)} fields (fields present only on some lines: {extra or 'none'}). Types are over all lines;"
      f" the example is task id {fields[0]['example_task_id']} (public benchmark text, truncated).")
    A("")
    A(md_table(["field", "type", "lines", "example"],
               [(f"`{f['field']}`", f"`{f['type']}`", f["present_in"], f["example"]) for f in fields]))
    A("")
    A(f"- `thm_stmt` == `target_theorem` on {totals['thm_stmt_eq_target_theorem']} lines;"
      f" `target_theorem` is `thm_stmt` followed by `:=` on"
      f" {totals['target_theorem_eq_thm_stmt_plus_assign']} lines.")
    cut = totals["target_theorem_prefix_of_thm_stmt_cut_at_inner_assign"]
    A(f"- On the other {totals['tasks'] - totals['target_theorem_eq_thm_stmt_plus_assign']} lines `target_theorem`"
      f" stops at an earlier `:=` inside the statement (a `let` or named argument) and is a strict prefix of"
      f" `thm_stmt` on {len(cut)} of them (task ids {', '.join(map(str, cut[:12]))}{', ...' if len(cut) > 12 else ''})."
      " The statement is therefore taken from `thm_stmt`, not `target_theorem`.")
    A(f"- `ground_truth_proof` starts with `:=` on {totals['proof_starts_with_assign']} lines; the others"
      f" (task id: first token) are {', '.join(f'{k}: `{v}`' for k, v in totals['proof_not_assign'].items())}"
      " (equation-compiler and `where` forms).")
    A(f"- repos.json: top-level keys {totals['repos_json_top_level']}; per-repository fields (count over the"
      f" {totals['repos_in_repos_json']}): {', '.join(f'`{k}` {v}' for k, v in totals['repos_json_fields'].items())}.")
    A("")
    A("## Mapping onto Appendix B (obligation record)")
    A("")
    A(md_table(["Appendix B field", "benchmark source", "relation"],
               [(f"`{m['appendix_b']}`", m["benchmark"] or "-", m["relation"]) for m in bmap]))
    A("")
    A(f"- Adopted (same meaning; `data/schemas.BENCHMARK_FIELD_MAP`, benchmark name -> ours): "
      f"{', '.join(f'`{k}` -> `{v}`' for k, v in sorted(BENCHMARK_FIELD_MAP.items()))}. "
      f"Map consistent with this table and with both field lists: {consistent}.")
    A(f"- Benchmark fields with no Appendix B counterpart: {', '.join(f'`{u}`' for u in unmapped)}.")
    A("- Appendix B fields with no benchmark source: " +
      ", ".join(f"`{m['appendix_b']}`" for m in bmap if m["benchmark"] is None) + ".")
    A("")
    A("## Tasks per repository")
    A("")
    A(f"- {totals['repos_in_task_file']} repositories in the task file (`lean_root`), {totals['repos_in_repos_json']}"
      f" in repos.json; in repos.json without tasks: {totals['repos_json_not_in_tasks'] or 'none'}; with tasks"
      f" but not in repos.json: {totals['tasks_repo_not_in_repos_json'] or 'none'}.")
    A(f"- Regular/large: neither file has such a field. The pinned commit's `prompts/full_limited/` holds"
      f" pre-generated capped full-context prompts for {', '.join(totals['large_repos'])} only (the README calls them"
      f" 'the two largest repos'); prompt files per repository equal its task count:"
      f" {totals['large_prompt_files_equal_task_counts']}. With those two as large: regular"
      f" {totals['regular_repos']} repositories / {totals['regular_tasks']} tasks, large"
      f" {len(totals['large_repos'])} / {totals['large_tasks']}. Reconciles with 21/381 and 2/119:"
      f" **{totals['reconciles_381_119']}**. (The 200k-token threshold is in neither file.)")
    A(f"- Plan 5.1's 23 names found in the files: {all(totals['plan_names_found'].values())}; names spelled differently"
      f" in the files: {', '.join(f'{a} = `{b}`' for a, b in totals['plan_name_mismatches'].items())}.")
    A(f"- `category`: {', '.join(f'{k} {v}' for k, v in totals['category'].items())}; one category per"
      f" repository: {totals['one_category_per_repo']}.")
    A(f"- `subset_aristotle` true on {totals['subset_aristotle']} tasks.")
    A("")
    A("Dependency depth: the task file gives two per-task numbers, `nesting_depth` and"
      " `transitive_dep_count` (shown as min / median / max). Pinned commit, licence, toolchain and"
      " mathlib flag from repos.json.")
    A("")
    A(md_table(["repo", "tasks", "large", "category", "nesting_depth", "transitive_dep_count", "aristotle",
                "pinned commit", "commit date", "licence", "toolchain", "mathlib"],
               [(p["repo"], p["tasks"], "yes" if p["large"] else "", p["category"][0], _d(p["nesting_depth"]),
                 _d(p["transitive_dep_count"]), p["subset_aristotle"], f"`{p['commit'][:12]}`",
                 (p["commit_date"] or "")[:10], p["license"],
                 (p["lean_toolchain"] or "").replace("leanprover/lean4:", ""),
                 "yes" if p["has_mathlib"] else "no") for p in per_repo]))
    A("")
    A(f"All tasks: nesting_depth {_d(totals['nesting_depth'])} (mean {totals['nesting_depth']['mean']});"
      f" transitive_dep_count {_d(totals['transitive_dep_count'])} (mean {totals['transitive_dep_count']['mean']}).")
    A("")
    A("Histogram of `nesting_depth` over all tasks:")
    A("")
    A(md_table(["nesting_depth"] + [str(k) for k in totals["nesting_depth_hist"]],
               [["tasks"] + list(totals["nesting_depth_hist"].values())]))
    A("")
    A("## Shortlist check (plan 5.1)")
    A("")
    A("Rule: the build finishes in under 30 minutes on Colab; at least 300 extractable obligations; at least 10"
      " benchmark tasks; the five cover different topic categories. 'Topic category' is read as the task file's"
      " `category` field, the only category either file gives. Build time and obligation counts need Lean:"
      " marked C0/C1.")
    A("")
    A(md_table(["repo", "tasks", ">= 10 tasks", "category", "distinct category", "build < 30 min",
                ">= 300 obligations", "large", "licence", "mathlib", "toolchain"],
               [(r["repo"], r["tasks"], r["ge_10_tasks"], r["category"], r["distinct_category"],
                 r["build_under_30_min"], r["ge_300_obligations"], "yes" if r["large"] else "no", r["license"],
                 "yes" if r["has_mathlib"] else "no", r["lean_toolchain"].replace("leanprover/lean4:", ""))
                for r in sl["shortlist"]]))
    A("")
    A(f"Source of every met/failed: the task file (`lean_root` counts, `category`). Categories covered by"
      f" the shortlist: {sl['categories_covered']}; five distinct: **{sl['five_distinct_categories']}**. The benchmark"
      f" has {len(sl['benchmark_categories'])} categories: {', '.join(sl['benchmark_categories'])}.")
    A("")
    A("Repositories meeting both file-checkable conditions (>= 10 tasks; regular, not large), by category"
      " (the pool C0 would time; not a selection):")
    A("")
    A(md_table(["category", "repositories (tasks, licence, mathlib)"],
               [(c, "; ".join(f"{x['repo']} ({x['tasks']}, {x['license']}, "
                              f"{'mathlib' if x['has_mathlib'] else 'no mathlib'})" for x in v) or "none")
                for c, v in sl["pool_by_category"].items()]))
    A("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsb", default=VSB_DIR)
    ap.add_argument("--out", default=os.path.join(PKG, "runs", "S5"))
    ap.add_argument("--report", default=os.path.join(PKG, "reports", "S5.md"))
    a = ap.parse_args(argv)
    rows, repos_doc, tree = load(a.vsb)
    sums = {fn: sha256(os.path.join(a.vsb, fn)) for fn in ("verisoftbench.jsonl", "repos.json", "LICENSE", "tree.txt")}
    fields, extra = field_table(rows)
    per_repo, totals = analyse(rows, repos_doc, tree)
    sl = shortlist_check(per_repo)
    bmap, unmapped, consistent = appendix_b_map(fields)
    os.makedirs(a.out, exist_ok=True)
    for name, obj in (("fields.json", {"commit": COMMIT, "sha256": sums, "fields": fields, "extra": extra}),
                      ("per_repo.json", per_repo), ("totals.json", totals), ("shortlist.json", sl),
                      ("appendix_b_map.json", {"map": bmap, "unmapped_benchmark_fields": unmapped,
                                               "benchmark_field_map": BENCHMARK_FIELD_MAP,
                                               "benchmark_field_map_consistent": consistent})):
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)
            f.write("\n")
    with open(a.report, "w") as f:
        f.write(report(fields, extra, per_repo, totals, sl, bmap, unmapped, consistent, sums))
    print(f"S5: {totals['tasks']} tasks, {totals['repos_in_task_file']} repos; regular/large "
          f"{totals['regular_tasks']}/{totals['large_tasks']} (reconciles: {totals['reconciles_381_119']}); "
          f"shortlist five distinct categories: {sl['five_distinct_categories']}; map consistent: {consistent}")


if __name__ == "__main__":
    main()

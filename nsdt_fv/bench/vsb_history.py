"""C1-prep step 1: source-level history scan of the VeriSoftBench repositories (no Lean).

For each repository r of bench/vsb/repos.json this walks the first-parent history up to the pinned
commit c*_r, then the first-parent commits after c*_r (the forward set F_r), and counts theorem/lemma
declarations with a regular-expression scan of the .lean files at every commit.

THIS IS AN ESTIMATE. A declaration here is a 'theorem' or 'lemma' keyword at the start of a line (after
attributes and modifiers), outside comments; its qualified name is the enclosing 'namespace' stack plus the
written name ('_root_.' honoured); its statement is the text from the name up to the first ':=' (or a
pattern-match bar / 'where'), whitespace-normalised. Identity is (qualified name, statement hash), as in
plan 5.2. Declarations made by macros, 'instance's, Prop-valued 'def's, generated names, and 'open ... in'
prefixes on the same line are missed or seen imperfectly; the Lean extractor (ntp-toolkit or
fallback_extract.lean) decides at C1.

Nothing from the repositories' source is written out: names, statements and paths are stored only as
sha256 prefixes. The output is per-commit counts (traceable to commit hashes) and per-declaration presence
intervals over the commit sequence, from which data/stream.py computes windows exactly as 5.2 defines them.

Usage: python -m nsdt_fv.bench.vsb_history --clones DIR --out nsdt_fv/runs/C1prep/history [repo ...]
The clones are 'git clone --bare --filter=blob:limit=2m <url>' (runs/C1prep/clone.sh); blobs above the
limit are fetched lazily when read.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS_JSON = os.path.join(HERE, "vsb", "repos.json")
SCAN_VERSION = "vsb_history/3"

VENDORED = re.compile(r"(^|/)(\.lake|lake-packages|lean_packages|_target|\.git)/")
TESTLIKE = {"test", "tests", "testing", "bench", "benchmark", "benchmarks", "example", "examples", "demo",
            "demos", "scratch", "playground", "archive"}

_DECL = re.compile(
    r"^[ \t]*(?:@\[[^\]\n]*\][ \t\n]*)*"
    r"(?:(?:private|protected|noncomputable|nonrec|unsafe|partial|scoped|local)[ \t]+)*"
    r"(theorem|lemma)[ \t]+([^\s:({\[⦃]+)", re.M)
_NS = re.compile(r"^[ \t]*(namespace|section|end|mutual|(?:noncomputable|public)[ \t]+section)\b[ \t]*([^\s\-]*)", re.M)
_STOP = re.compile(r":=|\n[ \t]*\||\bwhere\b|\n[ \t]*(?:theorem|lemma|def|instance|example|end|namespace|section)\b")


_OPEN_B = "([{⟨⦃"
_CLOSE_B = ")]}⟩⦄"


def _stop_at_depth0(rest: str):
    """Offset of the first statement terminator (':=', a line starting with '|', 'where', or the next command)
    outside brackets, so that ':=' inside '{ s with f := x }' or '(h := x)' does not cut the statement."""
    depth, pos = 0, 0
    for m in _STOP.finditer(rest):
        seg = rest[pos:m.start()]
        depth += sum(seg.count(c) for c in _OPEN_B) - sum(seg.count(c) for c in _CLOSE_B)
        pos = m.start()
        g = m.group().strip()
        is_command = not (g.startswith(":=") or g.startswith("|") or g == "where")
        if depth <= 0 or is_command:
            return m.start()
    return None


def h16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogateescape")).hexdigest()[:16]


_COMMENT_START = re.compile(r"/-|--")
_BLOCK_EDGE = re.compile(r"/-|-/")


def strip_comments(src: str) -> str:
    """Remove nested /- -/ block comments and -- line comments; keep newlines so line starts survive.
    String literals are not tracked (a '--' inside a string truncates that line; acceptable for an estimate)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        m = _COMMENT_START.search(src, i)
        if m is None:
            out.append(src[i:])
            break
        out.append(src[i:m.start()])
        if m.group() == "--":
            j = src.find("\n", m.end())
            if j < 0:
                break
            i = j
            continue
        depth, j = 1, m.end()
        while depth and j < n:
            mm = _BLOCK_EDGE.search(src, j)
            if mm is None:
                j = n
                break
            depth += 1 if mm.group() == "/-" else -1
            j = mm.end()
        out.append("\n" * src.count("\n", m.start(), j))
        i = j
    return "".join(out)


_CMD_LINE = re.compile(
    r"\n(?=@\[|#|(?:theorem|lemma|def|instance|example|abbrev|structure|inductive|class|end|namespace|section|"
    r"open|variable|universe|set_option|attribute|macro|macro_rules|syntax|notation|infix|infixl|infixr|prefix|"
    r"postfix|elab|import|mutual|private|protected|noncomputable|partial|unsafe|opaque|axiom|initialize|deriving|"
    r"local|scoped|omit|include|export|register_simp_attr|add_decl_doc)\b)")


def scan_lean(src: str) -> list:
    """[(qualified_name, statement_text)] for the theorem/lemma declarations of one .lean file."""
    return [(d["qname"], d["stmt_norm"]) for d in scan_lean_spans(src)[1]]


def scan_lean_spans(src: str):
    """(comment-stripped text, [declaration spans]). A span: qname, stmt_norm, start (line start of the
    declaration incl. attributes), stop (end of the statement: ':=', '|' or 'where'), end (start of the next
    command: next namespace/section/end/declaration or the next line starting in column 0 that is not a
    continuation bar or bracket). Source-level estimate, as everything in this module."""
    text = strip_comments(src)
    events = []
    for m in _NS.finditer(text):
        kw = m.group(1).split()[-1]
        events.append((m.start(), 0, kw, m.group(2)))
    for m in _DECL.finditer(text):
        events.append((m.start(), 1, "decl", m))
    events.sort(key=lambda e: (e[0], e[1]))
    starts = [e[0] for e in events]
    stack = []                      # one entry per scope: [component] for a namespace, [] for section/mutual
    out = []
    for _pos, _o, kw, arg in events:
        comps = [c for c in arg.split(".") if c] if kw in ("namespace", "section", "end") else []
        if kw == "namespace":
            stack.extend([c] for c in comps)            # 'namespace A.B' opens two scopes
        elif kw == "section":
            stack.extend([] for _ in (comps or [None]))
        elif kw == "mutual":
            stack.append([])
        elif kw == "end":
            for _ in range(max(1, len(comps))):        # 'end A.B' closes two
                if stack:
                    stack.pop()
        else:
            m = arg
            name = m.group(2)
            if name.startswith("_root_."):
                q = name[len("_root_."):]
            else:
                q = ".".join([c for ent in stack for c in ent] + [name])
            rest = text[m.end(): m.end() + 20000]
            s = _stop_at_depth0(rest)
            stmt = rest[: s] if s is not None else rest[:2000]
            stop = m.end() + (s if s is not None else len(stmt))
            b = bisect.bisect_right(starts, stop)
            nxt = starts[b:b + 1]
            c = _CMD_LINE.search(text, stop)
            end = min([len(text)] + nxt + ([c.start() + 1] if c else []))
            out.append({"qname": q, "stmt_norm": " ".join(stmt.split()), "start": m.start(), "stop": stop,
                        "end": end})
    return text, out


def scan_keys(src: str) -> list:
    return [(q, h16(st)) for q, st in scan_lean(src)]


class Git:
    def __init__(self, repo_dir):
        self.dir = repo_dir
        self.cat = subprocess.Popen(["git", "-C", repo_dir, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
        self.cache = {}

    def run(self, *args) -> str:
        return subprocess.run(["git", "-C", self.dir, *args], check=True, capture_output=True).stdout.decode()

    def blob(self, sha) -> bytes:
        self.cat.stdin.write((sha + "\n").encode())
        self.cat.stdin.flush()
        header = self.cat.stdout.readline().decode().split()
        if len(header) < 3 or header[1] == "missing":
            raise RuntimeError(f"blob {sha} missing")
        size = int(header[2])
        data = self.cat.stdout.read(size)
        self.cat.stdout.read(1)
        return data

    def text(self, sha) -> str:
        return self.blob(sha).decode("utf-8", "replace")

    def decls(self, sha) -> list:
        if sha not in self.cache:
            self.cache[sha] = scan_keys(self.text(sha))
        return self.cache[sha]

    def lean_tree(self, commit) -> dict:
        out = {}
        for line in self.run("ls-tree", "-r", "-z", commit).split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            _mode, typ, sha = meta.split()
            if typ == "blob" and path.endswith(".lean") and not VENDORED.search(path):
                out[path] = sha
        return out

    def diff(self, a, b) -> list:
        """[(path, old_sha|None, new_sha|None)] for .lean files changed between commits a and b."""
        parts = self.run("diff-tree", "-r", "-z", "--no-renames", "--no-commit-id", a, b).split("\0")
        out = []
        i = 0
        z = "0" * 40
        while i + 1 < len(parts):
            meta, path = parts[i], parts[i + 1]
            i += 2
            if not meta.startswith(":"):
                i -= 1
                continue
            _m1, _m2, old, new, _status = meta[1:].split()
            if not path.endswith(".lean") or VENDORED.search(path):
                continue
            out.append((path, None if old == z else old, None if new == z else new))
        return out


def utc(iso: str) -> str:
    return dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def testlike(path: str) -> bool:
    return any(c.lower() in TESTLIKE for c in path.split("/")[:-1])


def scan_repo(name: str, info: dict, clone_dir: str) -> dict:
    g = Git(clone_dir)
    cstar = info["commit"]
    head = g.run("rev-parse", "HEAD").strip()
    hist = g.run("rev-list", "--first-parent", "--reverse", cstar).split()
    fwd = g.run("rev-list", "--first-parent", "--reverse", f"{cstar}..HEAD").split()
    seq = hist + fwd
    dates = {}
    for line in g.run("log", "--no-walk=unsorted", "--format=%H %cI", *seq).splitlines():
        sha, d = line.split()
        dates[sha] = utc(d)
    live = Counter()            # (qname, stmt_hash) -> multiplicity over files
    live_names = Counter()
    where = {}                  # key -> path it was last seen in (path hash, test-like flag)
    intervals = {}              # key -> [[enter_idx, exit_idx|None], ...]
    n_open = 0
    commits = []
    prev = None
    for idx, c in enumerate(seq):
        if prev is None:
            changes = [(p, None, s) for p, s in sorted(g.lean_tree(c).items())]
        else:
            changes = g.diff(prev, c)
        touched = set()
        for path, old, new in changes:
            if old is not None:
                for k in g.decls(old):
                    live[k] -= 1
                    live_names[k[0]] -= 1
                    touched.add(k)
            if new is not None:
                for k in g.decls(new):
                    live[k] += 1
                    live_names[k[0]] += 1
                    touched.add(k)
                    where[k] = path
        n_new = n_gone = 0
        for k in touched:
            iv = intervals.get(k)
            is_live = live[k] > 0
            was_live = bool(iv) and iv[-1][1] is None
            if is_live and not was_live:
                intervals.setdefault(k, []).append([idx, None])
                n_new += 1
            elif was_live and not is_live:
                iv[-1][1] = idx
                n_gone += 1
            if live[k] <= 0:
                del live[k]
        for k in [k for k in touched if live_names.get(k[0], 1) <= 0]:
            live_names.pop(k[0], None)
        n_open += n_new - n_gone
        commits.append([c, dates[c], len(changes), n_new, n_gone, n_open, len(live_names)])
        prev = c
    n_hist = len(hist)
    decls = []
    for k, iv in intervals.items():
        p = where.get(k, "")
        decls.append([h16(k[0]), k[1], h16(p), int(testlike(p)), iv])
    decls.sort(key=lambda d: (d[4][0][0], d[0], d[1]))
    last = n_hist - 1
    pinned = [d for d in decls if any(a <= last and (b is None or b > last) for a, b in d[4])]
    return {
        "repo": name, "url": info["url"], "pinned_commit": cstar, "pinned_commit_date": info.get("commit_date"),
        "head_at_scan": head, "scan_version": SCAN_VERSION,
        "estimate": "source-level regex scan of .lean files; NOT the Lean extractor (which decides at C1)",
        "n_history_commits": n_hist, "n_forward_commits": len(fwd),
        "commit_fields": ["sha", "committer_date_utc", "lean_files_changed", "n_new", "n_gone", "n_live",
                          "n_live_names"],
        "commits": commits,
        "decl_fields": ["name_sha16", "stmt_sha16", "path_sha16", "testlike_path", "intervals"],
        "decls": decls,
        "totals": {"live_at_pinned": len(pinned),
                   "live_at_pinned_testlike": sum(d[3] for d in pinned),
                   "live_at_pinned_names": commits[last][6] if n_hist else 0,
                   "keys_ever": len(decls),
                   "live_at_head": commits[-1][5] if commits else 0},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clones", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("repos", nargs="*")
    a = ap.parse_args(argv)
    repos = json.load(open(REPOS_JSON))["repos"]
    os.makedirs(a.out, exist_ok=True)
    for name in sorted(a.repos or repos):
        d = os.path.join(a.clones, name + ".git")
        if not os.path.isdir(d):
            print(f"{name}: no clone at {d}", file=sys.stderr)
            continue
        rec = scan_repo(name, repos[name], d)
        with gzip.GzipFile(os.path.join(a.out, name + ".json.gz"), "wb", mtime=0) as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode())
        t = rec["totals"]
        print(f"{name}: {rec['n_history_commits']} commits to c*, live at c* {t['live_at_pinned']} "
              f"(test-like paths {t['live_at_pinned_testlike']}), forward {rec['n_forward_commits']} commits",
              flush=True)


if __name__ == "__main__":
    main()

"""Evaluation runner: checks model samples through any CheckerBackend, writes verdict logs
(Appendix B) and caches verdicts (rule R5). Imports only the backend protocol, never a concrete
backend, so a Lean backend can replace ToyEq without changes here (gate G0).

    python -m nsdt_fv.eval.runner --config nsdt_fv/configs/S0.yaml      # runs experiment S0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from ..backends.base import CheckerBackend, Verdict
from ..data.schemas import validate_verdict_line


def candidate_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sample_seed(run: str, stage: int, obligation_id: str, sample: int) -> int:
    """R6: sampling seeds are a function of run, stage, obligation and sample index."""
    h = hashlib.sha256(f"{run}|{stage}|{obligation_id}|{sample}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class VerdictCache:
    """R5: verdicts cached under the hash of snapshot, obligation and candidate text.

    A cached verdict is never recomputed with a different timeout: a lookup with another timeout
    returns the cached verdict and counts a timeout mismatch."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.entries = {}
        self.hits = 0
        self.misses = 0
        self.timeout_mismatches = 0
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    e = json.loads(line)
                    self.entries[e["key"]] = e

    @staticmethod
    def key(snapshot_id, obligation_id, candidate):
        return hashlib.sha256(f"{snapshot_id}\x00{obligation_id}\x00{candidate}".encode()).hexdigest()

    def get(self, key, timeout_s):
        e = self.entries.get(key)
        if e is None:
            return None
        self.hits += 1
        if e["timeout_s"] != timeout_s:
            self.timeout_mismatches += 1
        return Verdict.from_dict(e["verdict"])

    def put(self, key, verdict: Verdict, timeout_s):
        if key in self.entries:
            raise RuntimeError("R5: a cached verdict is never recomputed")
        e = {"key": key, "timeout_s": timeout_s, "verdict": verdict.to_dict()}
        self.entries[key] = e
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(e) + "\n")


class CachedChecker:
    def __init__(self, backend: CheckerBackend, cache: VerdictCache):
        if not isinstance(backend, CheckerBackend):
            raise TypeError("backend does not implement CheckerBackend")
        self.backend = backend
        self.cache = cache

    def check(self, snapshot_id, obligation_id, candidate, timeout_s):
        k = VerdictCache.key(snapshot_id, obligation_id, candidate)
        v = self.cache.get(k, timeout_s)
        if v is not None:
            return v, True
        self.cache.misses += 1
        v = self.backend.check(snapshot_id, obligation_id, candidate, timeout_s)
        self.cache.put(k, v, timeout_s)
        return v, False

    def check_statement(self, snapshot_id, context, statement, candidate, timeout_s):
        k = VerdictCache.key(snapshot_id, "stmt:" + context + "\x00" + statement, candidate)
        v = self.cache.get(k, timeout_s)
        if v is not None:
            return v, True
        self.cache.misses += 1
        v = self.backend.check_statement(snapshot_id, context, statement, candidate, timeout_s)
        self.cache.put(k, v, timeout_s)
        return v, False


def verdict_line(run, stage, phase, target_id, sample, candidate, v: Verdict, repair_round=0, **extra):
    ln = {"run": run, "stage": int(stage), "phase": phase, "target_id": target_id, "sample": int(sample),
          "candidate_sha": candidate_sha(candidate), "accepted": bool(v.accepted), "category": v.category,
          "elapsed_s": float(round(v.elapsed_s, 6)), "repair_round": int(repair_round)}
    validate_verdict_line(ln)
    ln.update(extra)
    return ln


def evaluate(checker: CachedChecker, run: str, stage: int, obligations, sampler, k: int,
             timeout_s: int = 120, log=None):
    """Score obligations (dicts with id and snapshot_id, test views) at one stage: k samples each.
    sampler(obligation, seeds) -> list of candidate strings. Returns verdict lines."""
    out = []
    for o in obligations:
        seeds = [sample_seed(run, stage, o["id"], s) for s in range(k)]
        cands = sampler(o, seeds)
        if len(cands) != k:
            raise ValueError("sampler returned the wrong number of candidates")
        for s, c in enumerate(cands):
            v, _ = checker.check(o["snapshot_id"], o["id"], c, timeout_s)
            ln = verdict_line(run, stage, "eval", o["id"], s, c, v)
            out.append(ln)
            if log is not None:
                log.write(json.dumps(ln) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from . import s0
    import yaml
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("experiment") != "S0":
        raise SystemExit("only experiment S0 exists so far")
    t0 = time.time()
    s0.run_all(cfg, out_dir=a.out)
    print(f"S0 finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

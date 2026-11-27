"""Synthetic verdict logs with hand-computed answers for test T2.

The expected values below were derived by hand (fractions in the comments), not by metrics.py.
The logs are built from per-cell accepted counts, so metrics.py must go all the way from raw
verdict lines to LA, FWD, ACC, F, BWT and GC."""
from __future__ import annotations

# Case A: T = 3, k = 1, one sample per obligation, |E_j| = 4, so a[t, j] = (#accepted)/4.
# accepted counts acc[t][j-1] for t = 0..3:
CASE_A = {
    "T": 3, "k": 1, "n": 1, "size": 4,
    "acc": [[1, 0, 1],      # a[0,.] = .25 .00 .25
            [3, 1, 1],      # a[1,.] = .75 .25 .25
            [2, 4, 2],      # a[2,.] = .50 1.0 .50
            [2, 3, 3]],     # a[3,.] = .50 .75 .75
    # LA  = (.75 + 1.0 + .75)/3          = 5/6
    # FWD = (a01 + a12 + a23)/3 = (.25 + .25 + .50)/3 = 1/3
    # ACC = (.50 + .75 + .75)/3          = 2/3
    # F   = ((max(.75,.50) - .50) + (1.0 - .75))/2 = 1/4
    # BWT = ((.50 - .75) + (.75 - 1.0))/2 = -1/4
    "expected": {"LA": 0.8333333333333334, "FWD": 0.3333333333333333, "ACC": 0.6666666666666666,
                 "F": 0.25, "BWT": -0.25},
}

# Case B: T = 2, pass@2 from n = 4 samples, |E_j| = 2. Per-obligation accepted counts c.
# pass@2 with n = 4: c=0 -> 0; c=1 -> 1 - C(3,2)/C(4,2) = 1/2; c=2 -> 1 - 1/6 = 5/6; c>=3 -> 1.
CASE_B = {
    "T": 2, "k": 2, "n": 4, "size": 2,
    "c": {(0, 1): (0, 1), (0, 2): (0, 0),         # a01 = 1/4,  a02 = 0
          (1, 1): (2, 1), (1, 2): (1, 0),         # a11 = (5/6 + 1/2)/2 = 2/3,  a12 = 1/4
          (2, 1): (1, 1), (2, 2): (4, 2)},        # a21 = 1/2,  a22 = (1 + 5/6)/2 = 11/12
    # LA  = (2/3 + 11/12)/2 = 19/24 ; FWD = (1/4 + 1/4)/2 = 1/4 ; ACC = (1/2 + 11/12)/2 = 17/24
    # F   = 2/3 - 1/2 = 1/6 ; BWT = 1/2 - 2/3 = -1/6
    "expected": {"LA": 0.7916666666666666, "FWD": 0.25, "ACC": 0.7083333333333334,
                 "F": 0.16666666666666666, "BWT": -0.16666666666666666},
}

# Case C: gap closed. The arm is case A (ACC = 2/3). B2 and B6 are T = 3 logs like case A.
CASE_C = {
    "B2": [[0, 0, 0], [2, 0, 0], [1, 3, 0], [1, 1, 2]],      # ACC_B2 = (1/4 + 1/4 + 1/2)/3 = 1/3
    "B6": [[0, 0, 0], [2, 0, 0], [3, 3, 0], [4, 4, 3]],      # ACC_B6 = (1 + 1 + 3/4)/3 = 11/12
    # GC = (2/3 - 1/3)/(11/12 - 1/3) = (1/3)/(7/12) = 4/7
    "expected": {"ACC_B2": 0.3333333333333333, "ACC_B6": 0.9166666666666666, "GC": 0.5714285714285714},
}


def _line(run, t, oid, s, ok):
    return {"run": run, "stage": t, "phase": "eval", "target_id": oid, "sample": s,
            "candidate_sha": f"{run}|{t}|{oid}|{s}", "accepted": ok, "category": "ok" if ok else "error",
            "elapsed_s": 0.0, "repair_round": 0}


def build_quarter_log(run, acc, size=4, prefix="o"):
    """Log for a matrix of accepted counts with one sample per obligation."""
    T = len(acc) - 1
    test_sets = {j: [f"{prefix}{j}_{i}" for i in range(size)] for j in range(1, T + 1)}
    lines = []
    for t, row in enumerate(acc):
        for j, c in enumerate(row, 1):
            for i, oid in enumerate(test_sets[j]):
                lines.append(_line(run, t, oid, 0, i < c))
    return lines, test_sets


def build_case_a(run="T2/A"):
    lines, ts = build_quarter_log(run, CASE_A["acc"])
    # noise that must be ignored: other phases, other runs, non-test targets
    noise = []
    for ln in lines[:12]:
        x = dict(ln)
        x["phase"] = "nrem"
        x["accepted"] = not x["accepted"]
        x["category"] = "ok" if x["accepted"] else "error"
        noise.append(x)
        y = dict(ln)
        y["run"] = "T2/other"
        y["accepted"] = True
        y["category"] = "ok"
        noise.append(y)
    noise.append(_line(run, 1, "wake_obligation_7", 0, True))
    mixed = []
    for i, ln in enumerate(lines):          # deterministic interleave, not sorted by stage
        mixed.append(ln)
        if i < len(noise):
            mixed.append(noise[i])
    mixed.extend(noise[len(lines):])
    mixed.reverse()
    return mixed, ts


def build_case_b(run="T2/B"):
    ts = {1: ["b1_0", "b1_1"], 2: ["b2_0", "b2_1"]}
    lines = []
    for (t, j), cs in CASE_B["c"].items():
        for oid, c in zip(ts[j], cs):
            for s in range(CASE_B["n"]):
                lines.append(_line(run, t, oid, s, s < c))
    return lines, ts

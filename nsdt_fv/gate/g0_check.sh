#!/usr/bin/env bash
# Gate G0 evidence (S5). Run from the repository root after committing:
#   nsdt_fv/gate/g0_check.sh [base]      (base = S4 head e876afe, the commit S5 is built on)
# Writes nsdt_fv/runs/G0/:
#   diff_trainer_evaluator.txt  git diff base..HEAD and uncommitted changes for nsdt_fv/train and nsdt_fv/eval
#   diff_all_stat.txt           what S5 did change (git diff --stat base..HEAD)
#   pytest.log                  the whole suite (T1-T4, T8/T9, T10 via S3 tests, T11, G0 tests) with -rA
#   stub_run/                   the G0 stub-backend run (result.json, verdict logs)
set -euo pipefail
BASE=${1:-e876afe}
OUT=nsdt_fv/runs/G0
mkdir -p "$OUT" "$OUT/stub_run"
{
  echo "base $(git rev-parse "$BASE")  head $(git rev-parse HEAD)"
  echo "--- git diff --stat $BASE HEAD -- nsdt_fv/train nsdt_fv/eval"
  git diff --stat "$BASE" HEAD -- nsdt_fv/train nsdt_fv/eval
  echo "--- git status --porcelain -- nsdt_fv/train nsdt_fv/eval (uncommitted)"
  git status --porcelain -- nsdt_fv/train nsdt_fv/eval
  n=$( (git diff "$BASE" HEAD -- nsdt_fv/train nsdt_fv/eval; git status --porcelain -- nsdt_fv/train nsdt_fv/eval) | wc -c)
  echo "--- bytes of diff + status: $n"
  if [ "$n" -eq 0 ]; then echo "TRAINER_EVALUATOR_DIFF_EMPTY=yes"; else echo "TRAINER_EVALUATOR_DIFF_EMPTY=no"; fi
} > "$OUT/diff_trainer_evaluator.txt"
git diff --stat "$BASE" HEAD > "$OUT/diff_all_stat.txt"
python3 -m pytest -p no:cacheprovider -rA 2>&1 | grep -E "^(PASSED|FAILED|ERROR|SKIPPED)|passed|failed" > "$OUT/pytest.log" || true
python3 -m nsdt_fv.gate.g0_stub_run --out "$OUT/stub_run" > "$OUT/stub_run/stdout.txt" 2>&1 || { echo "stub run failed"; exit 1; }
cat "$OUT/diff_trainer_evaluator.txt" | tail -2
tail -1 "$OUT/pytest.log"

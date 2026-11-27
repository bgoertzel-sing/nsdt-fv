#!/usr/bin/env bash
# S1 from a clean checkout (plan 8.1): tests (T1-T4, G0 interface), then the S1 run on two pinned cores,
# which writes raw logs to nsdt_fv/runs/S1/, then reports/S1.md regenerated from those logs.
#   pip install -r requirements.txt   # pyyaml, pytest, numpy, torch (CPU)
#   ./run_s1.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest
taskset -c 0,1 python3 -m nsdt_fv.eval.s1 --config nsdt_fv/configs/S1.yaml
python3 -m nsdt_fv.eval.report_s1

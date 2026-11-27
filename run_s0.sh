#!/usr/bin/env bash
# S0 from a clean checkout: acceptance tests T1-T3 (and the G0 interface test), then the S0 run that
# writes the raw logs to nsdt_fv/runs/S0/ and regenerates nsdt_fv/reports/S0.md from them.
#   pip install -r requirements.txt   # pyyaml, pytest
#   ./run_s0.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest
python3 -m nsdt_fv.eval.runner --config nsdt_fv/configs/S0.yaml

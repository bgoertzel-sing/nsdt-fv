#!/usr/bin/env bash
# S4: PLN reference fragment on the toy property tables (CPU only, 2 cores, about 20 minutes).
# Needs the S3 logs (nsdt_fv/runs/S3/N-C/channels.jsonl.gz, committed). Labels are checkpointed per project.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p nsdt_fv/runs/S4/logs
python3 -m pytest -q nsdt_fv/tests/test_T11.py > nsdt_fv/runs/S4/logs/T11.log 2>&1 || { cat nsdt_fv/runs/S4/logs/T11.log; exit 1; }
taskset -c 0,1 python3 -m nsdt_fv.eval.s4 --stage all 2>&1 | tee nsdt_fv/runs/S4/logs/s4.log
python3 -m nsdt_fv.eval.report_s4

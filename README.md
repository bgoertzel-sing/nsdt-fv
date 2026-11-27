# nsdt-fv

Continual learning for Lean 4 provers with Dreaming Transformers (DT/NSDT): harness, symbolic channels,
PLN branch and experiment reports, following Ben Goertzel's plan "Continual Learning for Provers on
Living Verification Codebases" v0.1 (Appendices A-F are the contract for this code).

## Status

- **S0** (Stage S, CPU only): ToyEq world and exact checker, project and stream generators, the harness
  with its checker interface, the metrics module and the hold-out guard. Acceptance tests T1-T3.
  Report: `nsdt_fv/reports/S0.md` (generated from `nsdt_fv/runs/S0/`).

## Run S0 from a clean checkout

```
pip install -r requirements.txt      # pyyaml, pytest (PyPI only); Python >= 3.10
./run_s0.sh                          # pytest (T1, T2, T3, G0 interface), then the S0 run and report
```

Tests only: `python3 -m pytest`. Report only, from existing logs: `python3 -m nsdt_fv.eval.report_s0`.
No GPU, no pretrained weights, no network access.

## Layout (plan A.2)

```
nsdt_fv/
  backends/  base.py (A.3 interfaces) toyeq.py (ToyEq + checker) toyeq_oracle.py (independent evaluator) mock.py
  data/      stream.py schemas.py holdout.py toyeq_gen.py toyeq_prover.py toyeq_templates.py
  eval/      runner.py metrics.py s0.py report_s0.py t2_cases.py
  configs/   S0.yaml
  tests/     test_T01.py test_T02.py test_T03.py test_G0_interface.py
  reports/   S0.md ISSUES.md CHANGES.md
  runs/S0/   raw logs of the S0 run (verdict lines, guard results, data, meta)
```

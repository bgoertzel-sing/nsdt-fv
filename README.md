# nsdt-fv

Continual learning for Lean 4 provers with Dreaming Transformers (DT/NSDT): harness, symbolic channels,
PLN branch and experiment reports, following Ben Goertzel's plan "Continual Learning for Provers on
Living Verification Codebases" v0.1 (Appendices A-F are the contract for this code).

## Status

- **S0** (Stage S, CPU only): ToyEq world and exact checker, project and stream generators, the harness
  with its checker interface, the metrics module and the hold-out guard. Acceptance tests T1-T3.
  Report: `nsdt_fv/reports/S0.md` (generated from `nsdt_fv/runs/S0/`).
- **S1** (Stage S, CPU only): tiny transformer trained from scratch, arms B0, B2, B5-10, B6 on the twelve-stage toy
  stream; LA, FWD, ACC, F, BWT, GC with bootstrap intervals; acceptance test T4. Report: `nsdt_fv/reports/S1.md`
  (generated from `nsdt_fv/runs/S1/` by `python3 -m nsdt_fv.eval.report_s1`); calibration log in `reports/CHANGES.md`.

- **S2** (Stage S, CPU only): DT-ref (A.4) arms D1-D4 on the toy stream. Report: `nsdt_fv/reports/S2.md`.
- **S3** (Stage S, CPU only): symbolic channels A-E (`nsdt_fv/store/channels/`, A.3 Channel interface) and arms
  N-A to N-E (D1 plus one channel); per-channel yield tables, T10. Report: `nsdt_fv/reports/S3.md` (generated from
  `nsdt_fv/runs/S3/` by `python3 -m nsdt_fv.eval.report_s3`); run with `./run_s3.sh` (needs S1's seed-0 base at
  `nsdt_fv/runs/S3/base/base.pt`).

- **S4** (Stage S, CPU only): Python reference of the Appendix E PLN fragment (`nsdt_fv/store/pln/ref.py`, test T11)
  and the toy property tables HasProperty(f, T) built from channel C's S3 output (`nsdt_fv/store/pln/table.py`);
  checker labels, 20% of the true pairs hidden, PLN against shape frequency, nearest neighbour and random order (AUC,
  precision at k, calibration). Report: `nsdt_fv/reports/S4.md` (generated from `nsdt_fv/runs/S4/` by
  `python3 -m nsdt_fv.eval.report_s4`); run with `./run_s4.sh`.

## Run S0 from a clean checkout

```
pip install -r requirements.txt      # pyyaml, pytest (PyPI only); Python >= 3.10
./run_s0.sh                          # pytest (T1, T2, T3, G0 interface), then the S0 run and report
```

Tests only: `python3 -m pytest`. Report only, from existing logs: `python3 -m nsdt_fv.eval.report_s0`.
No GPU, no pretrained weights, no network access.

## Run S1

```
pip install -r requirements.txt      # adds numpy and torch (CPU)
./run_s1.sh                          # pytest, S1 run on cores 0-1 (about an hour), report
```

Report only, from existing logs: `python3 -m nsdt_fv.eval.report_s1`.

## Layout (plan A.2)

```
nsdt_fv/
  backends/  base.py (A.3 interfaces) toyeq.py (ToyEq + checker) toyeq_oracle.py (independent evaluator) mock.py
  data/      stream.py schemas.py holdout.py toyeq_gen.py toyeq_prover.py toyeq_templates.py
  train/     model.py (tokenizer, tiny transformer, LoRA, seeded sampler) common.py lora.py (B2) replay.py (B5-p) joint.py (B6)
  eval/      runner.py metrics.py bootstrap.py s0.py report_s0.py s1.py report_s1.py t2_cases.py
  configs/   S0.yaml S1.yaml
  tests/     test_T01.py test_T02.py test_T03.py test_T04.py test_G0_interface.py test_G0_trainer.py
  reports/   S0.md S1.md S1_notes.md ISSUES.md CHANGES.md
  runs/S0/   raw logs of the S0 run (verdict lines, guard results, data, meta)
  runs/S1/   raw logs of the S1 run; runs/S1_calibration/ outputs of the calibration steps
```

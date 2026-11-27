# Proposed changes (rule R2: proposals only, nothing applied)

1. **T1 wording (A.5).** Proposed: "For 1,000 randomly mutated proofs, half of them checked against a
   mutated statement, any accepted proof whose statement fails on one of 200 random ground instances is a
   soundness bug." Reason: with unchanged statements every statement is true, so the test as worded cannot
   fail (ISSUES.md, item 10). S0 already does this.
2. **Lean file prefixes and L1.** Add to 5.4: "Training prompts built from a file prefix have the proofs
   of test obligations removed." (ISSUES.md, item 13.)

## S1 calibration (plan 8.1: "adjusting the toy to get there is calibration and is not a scientific gate")

No protocol element was changed: splits (last 20% of each window), metrics (section 7, eval/metrics.py
from S0, unchanged), pass@8 without repair, the S1 targets (gap >= 20, forgetting >= 10), the stream
(S0 generator, stream seed 0, 150-250 lemmas per project, six projects, twelve stages) and the hold-out
guard are as in S0. Only the model and trainer settings were calibrated. Raw outputs of each step are in
`nsdt_fv/runs/S1_calibration/` (the probe script there ran against the step-0 base, which the final run
reproduces with the same seed). Times are host clock (MST), 2026-11-27, on two pinned cores.

| Step | Setting | Observation | Decision |
|---|---|---|---|
| 0 (pilot c1, 07:56) | d=128, 4 layers, 4 heads, ff=512 (834,304 parameters); base: 60 epochs on the 159 guard-passed prelude lemmas (lr 1e-3); arms with LoRA r=16, alpha=32 on every linear layer, 20 epochs per stage, lr 2e-3 | B0: a[t,j] between 0 and 44%, about 18% on average, nearly all on obligations closed by `simp refl`. B2 after stages 1 and 2: a[1,1] = a[0,1] = 16.7%, training loss 0.62 and 0.90 (`c1_B2_train.jsonl`): the adapters had not learnt the window | stopped the pilot after B2 stage 2; probe the ceiling first |
| 1 (probe 1) | one LoRA r=16 fit on all 962 wake examples of the stream, 60 epochs (1,804 steps), lr 3e-3 | 604 s; loss 0.42; test pass@8 27.5% (38 of 47 trivial, 28 of 193 non-trivial); pass@8 on 120 of its own training items only 33% | LoRA underfits within the CPU budget: too weak for a ceiling |
| 2 (probe 2) | the same fit with all weights trained (full-parameter), 30 epochs (902 steps), lr 1e-3 | 304 s; loss 0.22; test pass@8 42.9% (43 trivial, 60 non-trivial); training items 65% | the ceiling is about 25 points above B0: use full-parameter training |
| 3 (final run) | adapt.mode = full, 30 epochs per stage for B2 and B5-10, 30 epochs over the union for B6 (re-started from the base at every stage), lr 1e-3, batch 32; model, base and toy unchanged | see reports/S1.md | adopted |

The order allows "train/lora.py or a full-parameter equivalent for the toy". Table 3 specifies adapters
for B2, B5 and B6, so the S1 numbers are for the full-parameter equivalent; the LoRA path stays in the
code (adapt.mode = lora) and its base-frozen check (base hash at every stage boundary) is active in that
mode. Whether adapter arms show the same two phenomena is not tested here (ISSUES.md, S1 items).

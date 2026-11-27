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

## S2 calibration and choices (DT-ref on the S1 toy)

No protocol element was changed: stream, splits, evaluation sets (T4), pass@8 at temperature 1.0 without
repair, metrics (eval/metrics.py), the S1 base weights (hash checked) and the hold-out guard are those of S1.
What S2 had to fix is DT-ref itself (A.4 leaves sizes, rates and the conjecture mechanism open) and the size
of the cap, which the S2 order allows to be increased if it underfits ("log in CHANGES.md").

**Cap capacity probes** (runs/S2_calibration/, probe_cap.py; same protocol as S1 probe 2: one fit on all 962
wake examples of the stream, 30 epochs, batch 32, lr 1e-3, then pass@8 on the 240 test obligations and on
120 sampled training items; S1 seed-0 base frozen, base hash checked unchanged after every probe). Host
clock MST, 2026-11-27; the probes ran in parallel, each on two pinned cores.

| Probe | Cap | Trainable params | Final loss | Test pass@8 (trivial / non-trivial) | Training items pass@8 |
|---|---|---|---|---|---|
| S1 probe 1 | LoRA r=16 on every linear layer (S1) | - | 0.42 | 27.5% (38 / 28) | 33% |
| S1 probe 2 | all weights (S1) | 834,304 | 0.22 | 42.9% (43 / 60) | 65% |
| 1 | A.4 MLP adapters r=64 after layers 2,3 + head | 50,048 | 0.94 | 16.7% (40 / 0) | 16.7% |
| 3 | A.4 MLP adapters r=256 after layers 2,3 + head | 148,736 | 0.81 | 16.7% (40 / 0) | 18.3% |
| 2 | MLP adapters r=256 after all 4 layers + head | 281,088 | 0.56 | 22.1% (40 / 13) | 25.0% |
| 4 | attention-block adapters (ff 512) after layers 2,3 + head | 412,928 | 0.54 | 18.3% (42 / 2) | 25.8% |
| 5 | attention-block adapters (ff 512) after all 4 layers + head | 809,472 | 0.38 | 31.3% (41 / 34) | 44.2% |

Observation: with A.4's cap as first sized (probe 1), the capped model learns no non-trivial proof even
after 30 epochs on the whole stream; a frozen base pretrained on 159 prelude lemmas is a weak substrate.
Decision rule, fixed before probe 4 and 5 results were read and before any D-arm run with a sized-up cap:
take the cap with the highest training-item pass@8 (a capacity measure, not a DT outcome); between caps
within 5 points, prefer A.4's placement (top third), then the smaller cap. Decision: probe 5 (44.2% against 25.8% for the best top-third cap, probe 4; outside the
5-point margin). The S2 cap is therefore an attention-block adapter (a full transformer block, feed-forward
width 512, output projections initialised to zero) after each of the four base layers, plus an untied
output head initialised from the base embedding: 809,472 trainable parameters on the frozen 834,304-parameter
base. It departs from A.4 in placement (all layers, not the top third) and in adapter form (attention plus
feed-forward, not a feed-forward bottleneck); the base stays frozen and all learning is in the cap (T8).
Even so the cap fits its training data less well than S1's full-parameter training (44% against 65%).

**DT-ref choices** (configs/S2.yaml):

- Wake: one epoch over the window (A.4), one optimizer step per example (batch 1), the same unit as a
  sleep step ("one cross-entropy step per accepted proof"); AdamW lr 1e-3, constant, one optimizer per run
  (its moment estimates are cap state, not stored examples).
- Conjecture sleep needs a model that writes statements. The base never learned that, so the wake pass
  also trains an auxiliary task built from the window's own (non-test) statements only: prompt "conj:" +
  three window statements, target a fourth (no proof text). REM prompts use three store statements of one
  module (A.4). The 192-token context of the S1 model forces the three prompt statements to be short
  (rejection sampling for triples that fit); this biases conjectures toward short statements.
- Dream budget per stage: 400 proof attempts, 60/40 as in section 3.2: 60 store statements x 4 candidates
  (re-derivation) and at most 40 new statements x 4 candidates (conjecture), from 120 statement generations.
  Temperature 0.8, four candidates (A.4). D3 drops conjecture sleep without reallocating its attempts.
- Re-derivation weights 1 + 4 fading + centrality + 2 near_change: fading from the store's verdict history of
  sleep attempts; centrality = in-degree in the proof_uses graph (after L2 filtering), scaled by the pool
  maximum; near_change = 1 for the current window, 0.5 for the project's earlier window, else 0. The pool
  is the repository's non-test statements of all windows so far plus the dream library.
- Negative phase: for every learned accepted dream whose statement also has a rejected candidate in the
  same sleep, the step adds 0.1 * softplus(-(log p(accepted) - log p(rejected))) (summed token
  log-probabilities). Rejected candidates without an accepted sibling are not used (A.4 defines only the
  preference form).
- D2 (no kernel filter): every distinct guard-passing candidate is learned; checker verdicts are logged
  for measurement only and never reach the learner (no fading term, no negatives, no dream library).
  Statement parsing (needed for the guard's L3 check) is still required of conjectures.
- Guard: every dream candidate passes HoldoutGuard.check before it is checked or learned; a refused
  candidate is a failed attempt and is never learned (counts in reports/S2.md). Every training example is
  checked again inside the learner; any raise there aborts the run.
- Supplementary arm B2c (Table 3; ISSUES.md item 18): S1's B2 recipe (30 epochs per stage, batch 32) on the
  same cap with the same frozen base, for a like-for-like sequential baseline. It is not one of D1-D4.
- Diagnostic arm D1x (added after the seed-0 D1-D4 results were seen, so it is labelled as a
  diagnostic and does not replace D1): D1 whose wake is B2c's supervised recipe (30 epochs per stage,
  batch 32, cosine schedule) instead of A.4's single epoch; sleep phases, budgets and guard as D1. Reason:
  with the one-epoch wake the cap does not learn the windows at all (D1 equals the frozen base within
  noise), so D1-D4 cannot say what the sleep phases do to a cap that is learning. D1x minus B2c isolates
  the sleep phases on top of the same supervised training. It is not A.4 and not a protocol change.

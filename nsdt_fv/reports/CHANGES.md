# Proposed changes (rule R2: proposals only, nothing applied)

1. **T1 wording (A.5).** Proposed: "For 1,000 randomly mutated proofs, half of them checked against a
   mutated statement, any accepted proof whose statement fails on one of 200 random ground instances is a
   soundness bug." Reason: with unchanged statements every statement is true, so the test as worded cannot
   fail (ISSUES.md, item 10). S0 already does this.
2. **Lean file prefixes and L1.** Add to 5.4: "Training prompts built from a file prefix have the proofs
   of test obligations removed." (ISSUES.md, item 13.)

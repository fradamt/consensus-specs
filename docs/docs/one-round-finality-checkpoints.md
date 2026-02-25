# One-Round Finality Implementation Checkpoints

This file tracks concrete rollback points during one-round-finality development.

## Rollback Point: `n=5f+1` Threshold Baseline

Use the commit named:

`one_round_finality: switch finality gadget to 5f+1 thresholds`

What this checkpoint contains:

1. Justification/progress threshold moved to `> 2/5`.
2. Timeout/skip threshold moved to `> 2/5`.
3. Finalization threshold moved to `> 4/5`.
4. `justified_checkpoint` update gated separately at `> 1/2`.
5. No second-vote/progress-attestation feature yet.

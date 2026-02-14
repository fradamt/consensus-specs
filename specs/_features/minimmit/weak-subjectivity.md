# Minimmit -- Weak Subjectivity Guide

## Introduction

This document is an extension of the
[Electra -- Weak Subjectivity Guide](../../electra/weak-subjectivity.md). All
behaviors and definitions defined in this document, and documents it extends,
carry over unless explicitly noted or overridden.

This document adapts the Weak Subjectivity Period (WSP) calculations for
Minimmit's one-round finality protocol, where accountable safety is 1/6 of total
stake (not 1/3 as in FFG).

### Why the baseline differs

In FFG, accountable safety is 1/3: two conflicting finalized checkpoints require
at least 1/3 equivocation (from 2/3 + 2/3 - 1 = 1/3 quorum intersection). In
Minimmit, the binding constraint is **finalization vs timeout**:

- Branch A finalizes block B at height h: requires 5/6 votes for B.
- Branch C times out at height h: requires 5/6 total votes, with no single
  target reaching 1/2 (justification threshold). So at most 1/2 - epsilon voted
  for B on branch C, meaning at least 5/6 - 1/2 = 1/3 voted for not-B.
- Quorum intersection: 5/6 + 1/3 - 1 = **1/6** of total stake.

This is strictly worse than finalization-vs-finalization (which gives 2/3
overlap). For any positive active stake, finalization-vs-timeout is the binding
constraint.

### Safety degradation from churn

The safety degradation rate from validator set churn is **identical** to FFG:

**D(n) = 2\*n\*d / S**

where n is the number of epochs since the common ancestor, d is the churn limit
in stake per epoch, and S is the total stake at the common ancestor.

The individual churn coefficients differ from FFG:

| Churn type           | Safety loss (Minimmit) | Safety loss (FFG) |
| -------------------- | ---------------------- | ----------------- |
| Exits E              | (7/6)E                 | (4/3)E            |
| Activations A        | (5/6)A                 | (2/3)A            |
| Consolidations C     | 2C                     | 2C                |
| Symmetric (E=A=n\*d) | 2\*n\*d                | 2\*n\*d           |

The symmetric combination always produces the same aggregate decay rate.

### Accountable liveness

The inactivity leak provides an economic backstop beyond the WSP. Any period
without finalization incurs inactivity penalties on non-participants, so a
delayed safety violation (double finalization after the WSP) always has economic
cost — either through slashing (within WSP) or inactivity penalties (beyond
WSP).

## Weak Subjectivity Period

### Calculating the Weak Subjectivity Period

*Note*: `SAFETY_DECAY` may need to be reduced from 10 to a lower value (e.g., 5)
for Minimmit. With `SAFETY_DECAY = 10`, the residual accountable safety is only
1/6 - 10/100 = 6.7% of total stake — thinner than FFG's 23.3%. With
`SAFETY_DECAY = 5`, the residual is 1/6 - 5/100 = 11.7%.

| SAFETY_DECAY | Total Active Balance (ETH) | WSP (epochs) | Residual safety |
| -----------: | -------------------------: | -----------: | --------------: |
|           10 |                  1,048,576 |          665 |            6.7% |
|           10 |                  2,097,152 |        1,075 |            6.7% |
|           10 |                  4,194,304 |        1,894 |            6.7% |
|           10 |                  8,388,608 |        3,532 |            6.7% |
|           10 |                 16,777,216 |        3,532 |            6.7% |
|           10 |                 33,554,432 |        3,532 |            6.7% |
|            5 |                  1,048,576 |          460 |           11.7% |
|            5 |                  2,097,152 |          665 |           11.7% |
|            5 |                  4,194,304 |        1,075 |           11.7% |
|            5 |                  8,388,608 |        1,894 |           11.7% |
|            5 |                 16,777,216 |        1,894 |           11.7% |
|            5 |                 33,554,432 |        1,894 |           11.7% |

#### `compute_weak_subjectivity_period`

```python
def compute_weak_subjectivity_period(state: BeaconState) -> uint64:
    """
    Returns the weak subjectivity period for the current ``state``.
    This computation takes into account the effect of:
        - validator set churn (bounded by ``get_balance_churn_limit()`` per epoch)

    The binding safety constraint is finalization-vs-timeout, with baseline
    accountable safety of 1/6 of total stake (vs 1/3 in FFG).
    Safety degradation from churn: D(n) = 2*n*delta/t (same rate as FFG).
    """
    t = get_total_active_balance(state)
    delta = get_balance_churn_limit(state)
    epochs_for_validator_set_churn = SAFETY_DECAY * t // (2 * delta * 100)
    return MIN_VALIDATOR_WITHDRAWABILITY_DELAY + epochs_for_validator_set_churn
```

*Note*: The formula is structurally identical to Electra's. The difference is
that the `SAFETY_DECAY` parameter should be interpreted against a 1/6 baseline
rather than 1/3. A `SAFETY_DECAY` of 10 consumes 60% of Minimmit's baseline
safety margin but only 30% of FFG's.

#### `is_within_weak_subjectivity_period`

```python
def is_within_weak_subjectivity_period(
    store: Store, ws_state: BeaconState, ws_checkpoint: Checkpoint
) -> bool:
    # Clients may choose to validate the input state against the input Weak Subjectivity Checkpoint
    assert ws_state.latest_block_header.state_root == ws_checkpoint.root
    assert compute_epoch_at_slot(ws_state.slot) == ws_checkpoint.epoch

    ws_period = compute_weak_subjectivity_period(ws_state)
    ws_state_epoch = compute_epoch_at_slot(ws_state.slot)
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    return current_epoch <= ws_state_epoch + ws_period
```

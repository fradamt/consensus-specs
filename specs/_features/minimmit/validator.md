# Minimmit -- Honest Validator

## Introduction

This document describes validator behavior for one-round finality.

*Note*: This specification is built upon [Fulu](../../fulu/validator.md).

## Attesting

### Attestation data

#### Finality vote

The `attestation_data.target` field is optional. Validators vote for finality
**once per height**, but attest for LMD-GHOST **every epoch**.

The validator client must track locally which height it has already voted for.
Finality votes can be cast for either the current height or the previous height
(to allow late votes to still count). Use `attestation_data.target = None` for
LMD-only attestations.

Each height has a **canonical target** — the epoch boundary block at the epoch
when the height started, stored in `state.current_height_target_epoch`. All
validators should vote for this canonical target regardless of which epoch they
attest in. Voting for the canonical target is the only way to avoid inactivity
leak penalties during non-finality; votes for other targets still count toward
justification and timeout but do not exempt the validator from the leak.

To construct a finality vote target:

```python
def get_finality_target(state: BeaconState, height: Height) -> Checkpoint:
    """
    Construct the canonical finality vote target for the given height.
    Uses the height's target epoch so all validators produce the same
    target regardless of which epoch they attest in.
    """
    if height == state.current_height:
        epoch = state.current_height_target_epoch
    else:
        epoch = state.previous_height_target_epoch
    root = get_block_root(state, epoch)
    return Checkpoint(epoch=epoch, root=root, height=height)
```

- If not yet voted at current height: Set `attestation_data.target` to
  `get_finality_target(head_state, head_state.current_height)`
- If already voted at current height but not at previous height (e.g., missed
  attestation duty before height transition): Set `attestation_data.target` to
  `get_finality_target(head_state, Height(head_state.current_height - 1))`
- If already voted at both heights: Set `attestation_data.target = None`
  (LMD-only attestation)

## How to avoid slashing

### Attester slashing

To avoid "attester slashings", a validator must not:

1. Sign two different attestations in the same epoch
2. Sign two conflicting finality votes at the same height

Non-finality attestations (where `target = None`) are still subject to (1) but
not (2).

*With one-round finality, a validator is safe as long as they cast only one
attestation per epoch and one finality vote per height.*

Specifically, when signing an `Attestation`:

1. Save a record to hard disk that an attestation has been signed for this
   epoch.
2. If the attestation includes a finality vote (`target != None`), also save a
   record for the height (i.e. `attestation_data.target.height`).
3. Generate and broadcast attestation.

If the software crashes at some point within this routine, then when the
validator comes back online, the hard disk has the record of the *potentially*
signed/broadcast attestation and can effectively avoid slashing.

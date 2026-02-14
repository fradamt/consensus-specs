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

To construct a finality vote target:

```python
def get_finality_target(state: BeaconState, height: Height) -> Checkpoint:
    epoch = get_current_epoch(state)
    # At epoch boundary, use previous epoch (get_block_root requires start_slot < state.slot)
    if state.slot == compute_start_slot_at_epoch(epoch):
        epoch = Epoch(epoch - 1) if epoch > GENESIS_EPOCH else GENESIS_EPOCH
    root = get_block_root(state, epoch)
    return Checkpoint(epoch=epoch, root=root, height=height)
```

- If not yet voted at current height: Set `attestation_data.target` to
  `get_finality_target(head_state, head_state.current_height)`
- If already voted at current height but not at previous height (e.g., missed
  attestation duty before height transition): Set `attestation_data.target` to
  `get_finality_target(head_state, head_state.current_height - 1)`
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

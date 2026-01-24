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
When the state's `current_height` advances, include a finality vote in the next
attestation. After that, use `attestation_data.target = None` until the height
advances again.

- If this is the first attestation at the current height:
  Set `attestation_data.target = head_state.current_target`
- If already voted at this height:
  Set `attestation_data.target = None` (LMD-only attestation)

## How to avoid slashing

### Attester slashing

To avoid "attester slashings", a validator must not sign two conflicting
finality votes at the same height. Non-finality attestations (where
`target = None`) cannot be slashed.

*With one-round finality, a validator is safe as long as they only cast one
finality vote per height.*

Specifically, when signing an `Attestation` with a finality vote:

1. Save a record to hard disk that a finality vote has been signed for the
   height (i.e. `attestation_data.target.height`).
2. Generate and broadcast attestation.

If the software crashes at some point within this routine, then when the
validator comes back online, the hard disk has the record of the *potentially*
signed/broadcast finality vote and can effectively avoid slashing.

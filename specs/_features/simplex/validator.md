# Simplex -- Honest Validator

This is an accompanying document to
[Simplex -- The Beacon Chain](./beacon-chain.md) and
[Simplex -- Fork Choice](./fork-choice.md), describing the expected behavior
of an honest validator in the simplex finality gadget.

## Table of contents

<!-- TOC -->

- [Overview](#overview)
- [Local state](#local-state)
- [Finality attestation](#finality-attestation)
  - [When to attest](#when-to-attest)
  - [Constructing `AttestationData`](#constructing-attestationdata)
    - [LMD head vote](#lmd-head-vote)
    - [Target vote](#target-vote)
    - [Height](#height)
    - [Finalize piggyback](#finalize-piggyback)
    - [Payload present](#payload-present)
  - [Broadcast](#broadcast)
- [Available attestation](#available-attestation)
- [How to avoid slashing](#how-to-avoid-slashing)
  - [E2: finalize-target conflict](#e2-finalize-target-conflict)
  - [Round double-vote](#round-double-vote)

<!-- /TOC -->

## Overview

Simplex splits validator attestation duties into two types:

1. **Finality attestation** (`Attestation`): assigned via beacon committees
   spread across `SLOTS_PER_ROUND` slots per round. Carries the finality
   target vote, LMD head vote, and optional finalize piggyback. One per round.

2. **Available attestation** (`AvailableAttestation`): assigned via a 512-member
   available committee per slot. Carries the LMD head vote and payload
   availability signal. One per slot.

The key differences from the base spec:

- **No source checkpoint.** The `source` field is removed from `AttestationData`.
- **Target = a real block**, not an epoch-boundary block. The target is
  identified by `Checkpoint(slot, root)` where `slot` is the block's actual
  proposal slot.
- **Finalize piggyback**: an optional second vote confirming a previously
  justified checkpoint, carried on the same attestation.
- **Multiple targets per height are allowed.** A validator can update their
  target in later rounds at the same height. The overwrite rule keeps the
  highest-slot target.

## Local state

An honest validator maintains a small amount of local state for anti-slashing:

```python
voted_target_at: Dict[Height, Checkpoint]
```

On signing any `AttestationData` with `height = H` and `target = T`, record
`voted_target_at[H] = T`. In practice, honest validators vote for one target
per height (the canonical target from the fork-choice), so this is O(1) per
height. Old entries can be pruned once the height is finalized.

## Finality attestation

### When to attest

A validator assigned to a beacon committee at slot `S` attests once per round.
The timing is the same as the base spec: attest when a valid block for slot `S`
is received from the expected proposer, or when `1/INTERVALS_PER_SLOT` of the
slot has elapsed, whichever comes first.

A validator signs at most **one `AttestationData` per round**. Signing two
different `AttestationData` in the same round triggers the round double-vote
penalty (forced exit, not full slashing — but still undesirable).

### Constructing `AttestationData`

#### LMD head vote

Set `beacon_block_root` to the root of the head block returned by `get_head(store)`.

```python
head = get_head(store)
attestation_data.beacon_block_root = head.root
```

#### Target vote

The target identifies the block being proposed for justification at the current
height. Set it to the **canonical target** from the head state:

```python
head_state = store.block_states[head.root]
if head_state.current_height == store_current_height:
    attestation_data.target = head_state.current_height_canonical_target
else:
    attestation_data.target = head_state.previous_height_canonical_target
```

The canonical target is set by `advance_height` to the latest block at the
start of each height. Under synchrony with an honest proposer, this is a
recent block at a slot above `justified_checkpoint.slot`.

*Note*: The IC model permits voting for non-canonical targets (any on-chain
block). An honest validator votes for the canonical target. The overwrite rule
means that if the canonical target changes (e.g., a new block arrives), the
validator can re-attest in a later round with a higher-slot target — the
on-chain record keeps the highest slot.

#### Height

Set `height` to the current height of the head state:

```python
attestation_data.height = head_state.current_height
```

An attestation at the previous height is also accepted (for late votes), but
under normal conditions validators attest at the current height.

#### Finalize piggyback

The finalize piggyback confirms a previously justified checkpoint. Setting it
is **optional but important for finality progress**. The rule:

1. Look up `voted_target_at[head_state.justified_height]`.
2. If it exists **and equals `head_state.justified_checkpoint`**: set the
   finalize fields.
3. Otherwise: abstain (set sentinel values).

```python
justified_height = head_state.justified_height
my_target_at_justified = voted_target_at.get(justified_height)

if (
    my_target_at_justified is not None
    and my_target_at_justified == head_state.justified_checkpoint
    and head_state.finalized_checkpoint != head_state.justified_checkpoint
):
    attestation_data.finalize_target = my_target_at_justified
    attestation_data.finalize_height = justified_height
else:
    attestation_data.finalize_target = Checkpoint()
    attestation_data.finalize_height = FAR_FUTURE_HEIGHT
```

**Why "only if your target was itself justified"**: this prevents locking
yourself on a side branch. If you voted for a descendant of the justified
checkpoint (at a higher slot, on a different branch), finalizing with that
target would lock you (via E2) on a target that might not be on the canonical
chain. By only finalizing when your target IS the justified checkpoint, you
ensure your locked target is always on every chain that justified at this
height.

**Why check `finalized != justified`**: no point in finalizing something that
is already finalized. The on-chain acceptance also checks this.

#### Payload present

Set `payload_present` based on whether the head block's execution payload has
been observed as available:

```python
attestation_data.payload_present = has_available_payload(head.root)
```

For same-slot attestations (`beacon_block_root.slot == data.slot`), set
`payload_present = False` (the PTC handles first-slot payload determination).

### Broadcast

Broadcast the signed attestation on the appropriate subnet. Aggregation follows
the same pattern as the base spec (aggregation selection via
`is_aggregator`, aggregate construction, timed broadcast at
`2/INTERVALS_PER_SLOT` of the slot).

## Available attestation

Validators assigned to the available committee for slot `S` produce an
`AvailableAttestation` carrying the LMD head vote and payload availability
signal. One per slot. Timing: attest before the available-confirmation
deadline (`AVAILABLE_CONFIRMATION_DUE_BPS` of the slot). This feeds the
Goldfish fork-choice layer (Layer 3).

## How to avoid slashing

### E2: finalize-target conflict

The only slashing condition is E2: if you sign `finalize_target = T` at
`finalize_height = H`, then any attestation you signed at `height = H` with
`target != T` is slashable evidence.

**How to stay safe**: maintain `voted_target_at[H]` and only set
`finalize_target` when your recorded target at the justified height matches
the justified checkpoint (the rule in [Finalize piggyback](#finalize-piggyback)
above). Since honest validators vote for one target per height (the canonical
target), and only finalize when that target was itself justified, no
conflicting pair can exist.

*Note*: There is no E1 (height double-target) condition. Voting for multiple
targets at the same height (e.g., in different rounds, as the canonical target
changes) is safe **as long as you do not carry a finalize piggyback for that
height**. If you voted for targets A and B at height H, do NOT set
`finalize_height = H` — you cannot safely commit to either without creating
an E2 pair with the other.

### Round double-vote

Signing two different `AttestationData` in the same round triggers
`RoundDoubleVoteEvidence`: forced exit plus a fixed deduction (not full
slashing). **Sign at most one `AttestationData` per round.**

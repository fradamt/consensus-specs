# Simplex -- Honest Validator

This is an accompanying document to
[Simplex -- The Beacon Chain](./beacon-chain.md) and
[Simplex -- Fork Choice](./fork-choice.md), describing the expected behavior of
an honest validator in the fresh-simplex-with-height-filter-and-timeouts
finality gadget.

## Table of contents

<!-- TOC -->

- [Overview](#overview)
- [Local state](#local-state)
- [Finality attestation](#finality-attestation)
  - [When to attest](#when-to-attest)
  - [R1 (justify) vs R2 (timeout)](#r1-justify-vs-r2-timeout)
  - [Constructing `AttestationData`](#constructing-attestationdata)
    - [LMD head vote](#lmd-head-vote)
    - [Target vote (justify or timeout)](#target-vote-justify-or-timeout)
    - [Height](#height)
    - [Finality piggyback (R1 only)](#finality-piggyback-r1-only)
    - [Payload present](#payload-present)
  - [Broadcast](#broadcast)
- [Available attestation](#available-attestation)
- [How to avoid slashing](#how-to-avoid-slashing)
  - [E1 avoidance](#e1-avoidance)
  - [Round double-vote](#round-double-vote)

<!-- /TOC -->

## Overview

Simplex splits validator attestation duties into two types:

1. **Finality attestation** (`Attestation`): assigned via beacon committees
   spread across `SLOTS_PER_ROUND` slots per round. Carries the finality target
   vote (or `Checkpoint()` for a timeout), LMD head vote, and optional finality
   piggyback. One per round.

2. **Available attestation** (`AvailableAttestation`): assigned via a 512-member
   available committee per slot. Carries the LMD head vote and payload
   availability signal. One per slot.

Key differences from the base spec:

- **No source checkpoint.** The `source` field is removed from
  `AttestationData`.
- **Target = a real block** (or `Checkpoint()` for a timeout vote), not an
  epoch-boundary block. The target is identified by `Checkpoint(slot, root)`
  where `slot` is the block's actual proposal slot.
- **Two vote kinds, encoded via `target`**: a justification vote (R1) has
  `target != Checkpoint()` and commits to a specific block at the current
  state-height; a timeout vote (R2) has `target == Checkpoint()` and signals
  inability to justify. Each validator casts at most one R1 vote per
  state-height. R2 votes drive the timeout-cert branch of `processHeight`.
- **Viability gate** (`is_viable_attestation_target`): the state-machine records
  `justification_targets[i]` only for justification attestations whose `height`
  equals the current state-height and whose `target.slot` lies in the
  current-height interval on the current chain. Timeout votes bypass this gate;
  they set `timeouts[i]` directly when their height matches the current
  state-height. (`finality_participation` updates are independent of viability.)
- **Finality piggyback** is only valid on R1 (justification) votes.
- **Timeout votes are slashable** when they conflict with a finality commitment
  at the same height: a vote with `target = Checkpoint()` at height `H`
  conflicts with any commitment `finality_target = T ≠ Checkpoint()` at
  `finality_height = H` (paper def:slashing).

## Local state

An honest validator maintains a small amount of local state for anti-slashing
and for the R1/R2 decision:

- `voted_target_at: Dict[Height, Checkpoint]` — first target signed at each
  height (used for discriminating R1 vs R2 and for the piggyback alignment
  check).
- `voted_finality_at: Dict[Height, Checkpoint]` — finality commitment made at
  each height, if any. Key is `finality_height`, value is `finality_target`.
  Used for the lock rule.

On signing any `AttestationData` with `height = H`, `target = T`, and a finality
piggyback `(finality_height, finality_target)`:

- If `H not in voted_target_at`, set `voted_target_at[H] = T`.
- If `finality_target != Checkpoint()`, set
  `voted_finality_at[finality_height] = finality_target`. (A validator MUST NOT
  sign two different finality commitments at the same `finality_height` — this
  is the E1 lock.)

Old entries can be pruned once the height is finalized.

## Finality attestation

### When to attest

A validator assigned to a beacon committee at slot `S` attests once per round.
The timing is the same as the base spec: attest when a valid block for slot `S`
is received from the expected proposer, or when `1/INTERVALS_PER_SLOT` of the
slot has elapsed, whichever comes first.

A validator signs at most **one `AttestationData` per round**. Signing two
different `AttestationData` in the same round triggers the round double-vote
penalty (forced exit, not full slashing — but still undesirable).

### R1 (justify) vs R2 (timeout)

An honest validator casts at most one R1 vote per state-height, committing to a
target on the current chain. R1 has `target != Checkpoint()` and may carry a
finality piggyback.

If a later voting opportunity at the same state-height arises (e.g., a later
round that has not yet advanced height), the honest validator casts an R2
(timeout) vote with `target = Checkpoint()`. R2 votes do NOT carry a finality
piggyback. R2 votes drive the timeout-cert branch of `processHeight` on
`state.timeouts`.

In practice, an honest validator's attestation sequence within a single
state-height is either:

- `(R1 at round r)` — a single justify vote, possibly with a finality piggyback,
  after which the height advances; or
- `(R1 at round r) → (R2 at round r+1, r+2, …)` — when the initial R1 quorum
  fragments and the timeout-cert branch is needed.

**R2 self-slash guard**: an R2 vote at height `H` with `target = Checkpoint()`
conflicts with any prior finality commitment at `finality_height = H` (whose
`finality_target` is non-empty). Therefore R2 is unsafe at `H` whenever the
validator has cast a finality piggyback at `finality_height = H`. In that case,
the validator MUST fall back to casting another R1 with the locked target (no
piggyback). See [E1 avoidance](#e1-avoidance).

### Constructing `AttestationData`

#### LMD head vote

Set `beacon_block_root` to the root of the head block returned by
`get_head(store)`.

```
head = get_head(store)
attestation_data.beacon_block_root = head.root
```

#### Target vote (justify or timeout)

The target identifies the block being voted for at the current state-height, or
is set to `Checkpoint()` to encode a timeout vote. The choice depends on whether
the validator has already voted at this height and whether a retroactive
finality lock applies:

```
head_state = store.block_states[head.root]
current_height = head_state.current_height

if current_height in voted_target_at:
    # R2 path: validator already voted at current_height. Either cast a
    # timeout (target = Checkpoint()) OR — if a finality commitment at
    # current_height locks the validator to a target T — re-cast another R1
    # with target = T (no piggyback). Casting target = Checkpoint() at H
    # while a finality commitment at H exists self-slashes via E1.
    if current_height in voted_finality_at:
        # Locked: re-submit lock target as another R1 (no piggyback).
        attestation_data.target = voted_finality_at[current_height]
    else:
        # Safe to cast R2 timeout.
        attestation_data.target = Checkpoint()
else:
    # First vote at current_height (R1 justify). The retroactive finality
    # lock from a prior higher-height R1 vote, if any, takes precedence.
    lock = voted_finality_at.get(current_height)
    if lock is not None:
        attestation_data.target = lock
    else:
        # Pick the latest on-chain block at state-height current_height.
        attestation_data.target = Checkpoint(
            slot=head_state.latest_block_header.slot,
            root=hash_tree_root(head_state.latest_block_header),
        )
```

*Note*: If the validator cast an R1 at `current_height` and subsequently
observes a newer block on the same chain, it is NOT safe to retarget — any later
vote at the same height must reuse the locked target (or be a timeout when no
finality lock applies).

*Note*: The retroactive finality lock matters when the view reverts to
`current_height` after the validator has already attached a finality commitment
at `finality_height = current_height` from a higher-height R1 vote. Under E1,
the validator is bound to that finality target at `current_height`; voting a
different target (or a timeout, since `Checkpoint() ≠ T`) at `current_height`
would self-evidence E1.

#### Height

Set `height` to the current state-height of the head state:

```
attestation_data.height = head_state.current_height
```

Votes with a stale height are still accepted by `process_attestation` (the
`finality_participation` update may still be useful), but they do not update
target tracking and earn no TIMELY_TARGET reward.

#### Finality piggyback (R1 only)

The finality piggyback confirms a previously justified checkpoint. It is **only
valid on R1 (justification) votes** — the on-chain `process_attestation` rejects
timeout votes (`target == Checkpoint()`) carrying a non-empty finality target.

The rule:

1. Only consider setting the piggyback when
   `attestation_data.target != Checkpoint()` (i.e., the vote is a justification,
   not a timeout).
2. Let `justified_height = head_state.justified_height` and
   `J = head_state.justified_checkpoint`.
3. Attach the piggyback only if the validator's prior commitments at
   `justified_height` are compatible with `J`:
   `voted_target_at.get(justified_height) in (None, J)` AND
   `voted_finality_at.get(justified_height) in (None, J)`. Also require
   finalization still pending
   (`head_state.finalized_checkpoint != head_state.justified_checkpoint`). On a
   match: set `finality_height = justified_height`, `finality_target = J`.
4. Otherwise: abstain (sentinel values).

```
if attestation_data.target != Checkpoint():
    justified_height = head_state.justified_height
    J = head_state.justified_checkpoint
    prior_target_at_justified_height = voted_target_at.get(justified_height)
    prior_finality_at_justified_height = voted_finality_at.get(justified_height)
    target_compatible = prior_target_at_justified_height in (None, J)
    finality_compatible = prior_finality_at_justified_height in (None, J)
    if (
        target_compatible
        and finality_compatible
        and head_state.finalized_checkpoint != head_state.justified_checkpoint
    ):
        attestation_data.finality_target = J
        attestation_data.finality_height = justified_height
    else:
        attestation_data.finality_target = Checkpoint()
        attestation_data.finality_height = FAR_FUTURE_HEIGHT
else:
    attestation_data.finality_target = Checkpoint()
    attestation_data.finality_height = FAR_FUTURE_HEIGHT
```

**Why condition on "your prior commitments at `justified_height` are compatible
with `J`"**: this is the lem:honest-e1-safety invariant. The finality commitment
locks the voter to `J` at height `justified_height` via E1. If your recorded
vote at `justified_height` differs from `J`, or if you earlier attached a
finality commitment `(finality_height = justified_height, finality_target = J')`
with `J' != J`, attaching the new piggyback would self-evidence an E1 violation.
Abstain unless both prior commitments at `justified_height` (if any) agree with
`J`.

#### Payload present

Set `payload_present` based on whether the head block's execution payload has
been observed as available:

```
attestation_data.payload_present = has_available_payload(head.root)
```

For same-slot attestations (`beacon_block_root.slot == data.slot`), set
`payload_present = False` (the PTC handles first-slot payload determination).

### Broadcast

Broadcast the signed attestation on the appropriate subnet. Aggregation follows
the same pattern as the base spec (aggregation selection via `is_aggregator`,
aggregate construction, timed broadcast at `2/INTERVALS_PER_SLOT` of the slot).

## Available attestation

Validators assigned to the available committee for slot `S` produce an
`AvailableAttestation` carrying the LMD head vote and payload availability
signal. One per slot. Timing: attest before the available-confirmation deadline
(`AVAILABLE_CONFIRMATION_DUE_BPS` of the slot). This feeds the Goldfish
fork-choice layer (Layer 3).

## How to avoid slashing

### E1 avoidance

The only slashing condition is E1: if you sign `finality_target = T` at
`finality_height = H`, then any attestation you signed at `height = H` with
`target != T` is slashable evidence. **Timeout votes are slashable too**:
`target = Checkpoint()` at `height = H` conflicts with
`finality_target = T ≠ Checkpoint()` at `finality_height = H` (paper
def:slashing).

**How to stay safe**: maintain `voted_target_at[H]` and `voted_finality_at[H]`.
Use these (plus the retroactive `voted_finality_at[H]` lock) to drive the target
choice at height `H`. Only set `finality_target` when your prior commitments at
the justified height match the justified checkpoint (the rule in
[Finality piggyback](#finality-piggyback-r1-only) above). The
[Target vote](#target-vote-justify-or-timeout) construction above bakes in the
R2 self-slash guard: if `voted_finality_at[current_height]` is set, the
validator re-submits another R1 with the locked target rather than casting a
timeout.

*Note*: There is no E2 (height double-target) condition. Signing an R2 (timeout)
vote when no finality lock at `current_height` exists is safe even though it
differs from a prior R1 at the same height — the two votes alone are not
directly slashable. The slashable case is exactly when an R2 timeout collides
with a same-height finality commitment.

### Round double-vote

Signing two different `AttestationData` in the same round triggers
`RoundDoubleVoteEvidence`: forced exit plus a fixed deduction (not full
slashing). **Sign at most one `AttestationData` per round.**

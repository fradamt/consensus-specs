# Simplex -- Honest Validator

This is an accompanying document to
[Simplex -- The Beacon Chain](./beacon-chain.md) and
[Simplex -- Fork Choice](./fork-choice.md), describing the expected behavior of
an honest validator in the fresh-simplex-with-notarizations finality gadget.

## Table of contents

<!-- TOC -->

- [Overview](#overview)
- [Local state](#local-state)
- [Finality attestation](#finality-attestation)
  - [When to attest](#when-to-attest)
  - [R1 (justify) vs R2 (notarize)](#r1-justify-vs-r2-notarize)
  - [Constructing `AttestationData`](#constructing-attestationdata)
    - [LMD head vote](#lmd-head-vote)
    - [Target vote](#target-vote)
    - [Height](#height)
    - [Kind](#kind)
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
   vote, LMD head vote, attestation kind (justify or notarize), and optional
   finality piggyback. One per round.

2. **Available attestation** (`AvailableAttestation`): assigned via a 512-member
   available committee per slot. Carries the LMD head vote and payload
   availability signal. One per slot.

Key differences from the base spec:

- **No source checkpoint.** The `source` field is removed from
  `AttestationData`.
- **Target = a real block**, not an epoch-boundary block. The target is
  identified by `Checkpoint(slot, root)` where `slot` is the block's actual
  proposal slot.
- **Two attestation kinds**: `ATTESTATION_KIND_JUSTIFY` (R1) and
  `ATTESTATION_KIND_NOTARIZE` (R2). Each validator casts at most one R1 vote per
  state-height. R2 votes serve as prefix-notarization fallback when R1 quorums
  fragment across targets.
- **Fresh-vote gate** (`is_vote_fresh`): the state-machine records
  `justification_targets[i]` / `notarization_targets[i]` only for votes whose
  `height` equals the current state-height and whose `target.slot` lies in the
  current-height interval on the current chain.
- **Finality piggyback** is only valid on R1 (justify) votes.

## Local state

An honest validator maintains a small amount of local state for anti-slashing
and for the R1/R2 decision:

- `voted_target_at: Dict[Height, Checkpoint]` — first target signed at each
  height (used for discriminating R1 vs R2 and for the piggyback alignment
  check).
- `voted_finality_at: Dict[Height, Checkpoint]` — finality commitment made at
  each height, if any. Key is `finality_height`, value is `finality_target`.
  Used for the lock rule.

On signing any `AttestationData` with `height = H`, `target = T`, and a
finality piggyback `(finality_height, finality_target)`:

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

### R1 (justify) vs R2 (notarize)

An honest validator casts at most one R1 vote per state-height, committing to a
target on the current chain. The R1 target is locked once signed: every
subsequent vote at the same height must use the same target.

If a later voting opportunity at the same state-height arises (e.g., a later
round that has not yet advanced height), the validator casts an R2 (notarize)
vote. R2 votes also carry a target, but they do NOT carry a finality piggyback.
R2 votes contribute only to the prefix-notarize quorum on
`notarization_targets`.

In practice, an honest validator's attestation sequence within a single
state-height is either:

- `(R1 at round r)` — a single justify vote, possibly with a finality piggyback,
  after which the height advances; or
- `(R1 at round r) → (R2 at round r+1, r+2, …)` — when the initial R1 quorum
  fragments and the prefix-notarize branch is needed.

### Constructing `AttestationData`

#### LMD head vote

Set `beacon_block_root` to the root of the head block returned by
`get_head(store)`.

```
head = get_head(store)
attestation_data.beacon_block_root = head.root
```

#### Target vote

The target identifies the block being voted for at the current state-height.
Under the lock rules, the choice is constrained:

```
head_state = store.block_states[head.root]
current_height = head_state.current_height

# Lock: the strongest commitment to a target at current_height is the *retroactive*
# finality lock from a prior higher-height R1 vote. Fall back to the R1 lock
# at current_height itself (equal to ``voted_target_at[current_height]`` under the honest rule,
# since any R1 recorded at current_height writes ``voted_target_at[current_height]``).
lock = voted_finality_at.get(current_height)
if lock is None:
    lock = voted_target_at.get(current_height)
if lock is not None:
    # Once locked by a prior commitment at current_height, reuse the same target.
    attestation_data.target = lock
else:
    # First vote at current_height: pick the latest on-chain block at state-height current_height.
    # Typically this is head_state's latest block header.
    attestation_data.target = Checkpoint(
        slot=head_state.latest_block_header.slot,
        root=hash_tree_root(head_state.latest_block_header),
    )
```

*Note*: If the validator cast an R1 at `current_height` and subsequently
observes a newer block on the same chain, it is NOT safe to retarget — the lock
prevents E1-style evidence.

*Note*: The retroactive finality lock matters when the view reverts to
`current_height` after the validator has already attached a finality commitment
at `finality_height = current_height` from a higher-height R1 vote. Under E1,
the validator is bound to that finality target at `current_height`; voting a
different target at `current_height` would self-evidence E1.

#### Height

Set `height` to the current state-height of the head state:

```
attestation_data.height = head_state.current_height
```

Votes with a stale height are rejected at inclusion time by
`process_attestation`.

#### Kind

Set `kind` based on whether the validator has already voted at the current
state-height:

```
if current_height in voted_target_at:
    attestation_data.kind = ATTESTATION_KIND_NOTARIZE  # R2
else:
    attestation_data.kind = ATTESTATION_KIND_JUSTIFY   # R1
```

#### Finality piggyback (R1 only)

The finality piggyback confirms a previously justified checkpoint. It is **only
valid on R1 (justify) votes** — the on-chain process_attestation rejects R2
votes carrying a non-empty finality target.

The rule:

1. Only consider setting the piggyback when
   `attestation_data.kind == ATTESTATION_KIND_JUSTIFY`.
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
if attestation_data.kind == ATTESTATION_KIND_JUSTIFY:
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
`target != T` is slashable evidence.

**How to stay safe**: maintain `voted_target_at[H]` and `voted_finality_at[H]`.
Use `voted_target_at[H]` (plus the retroactive `voted_finality_at[H]` lock) to
drive the target choice at height `H`. Only set `finality_target` when your
prior commitments at the justified height match the justified checkpoint (the
rule in [Finality piggyback](#finality-piggyback-r1-only) above). Since an
honest validator's lock means all votes at height H carry the same target `T`,
and only the R1 vote carries a piggyback, no conflicting
`(finality_target, target)` pair can exist.

*Note*: There is no E2 (height double-target) condition. Signing an R2
(notarize) vote with a target different from your R1 is not permitted by the
honest rule (the R1 lock), but it is not directly slashable either. The
protection is that R2 honest votes replay the R1 target.

### Round double-vote

Signing two different `AttestationData` in the same round triggers
`RoundDoubleVoteEvidence`: forced exit plus a fixed deduction (not full
slashing). **Sign at most one `AttestationData` per round.**

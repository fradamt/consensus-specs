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
  - [Vote kinds: justify (R1), timeout (R2), empty](#vote-kinds-justify-r1-timeout-r2-empty)
  - [Constructing `AttestationData`](#constructing-attestationdata)
    - [Head field (LMD head vote)](#head-field-lmd-head-vote)
    - [Interval-first target](#interval-first-target)
    - [Safe-confirmed head](#safe-confirmed-head)
    - [The uniform gate](#the-uniform-gate)
    - [Finality piggyback](#finality-piggyback)
  - [Broadcast](#broadcast)
- [Block proposal: pointing to a fresh quorum](#block-proposal-pointing-to-a-fresh-quorum)
- [Available attestation](#available-attestation)
- [How to avoid slashing](#how-to-avoid-slashing)
  - [E1 avoidance](#e1-avoidance)
  - [Round double-vote](#round-double-vote)

<!-- /TOC -->

## Overview

Simplex splits validator attestation duties into two types:

1. **Finality attestation** (`Attestation`): assigned via beacon committees
   spread across the slots of a round (length per `ROUND_SCHEDULE`). Carries the
   finality target vote (or `Checkpoint()` for a timeout or empty vote), the
   head field (LMD head vote), and optional finality piggyback. One per round.

2. **Available attestation** (`AvailableAttestation`): assigned via a 512-member
   available committee per slot. Carries the LMD head vote and payload
   availability signal. One per slot.

Key differences from the base spec:

- **No source checkpoint.** The `source` field is removed from
  `AttestationData`.
- **Target = a real block** (or `Checkpoint()` for a timeout or empty vote), not
  an epoch-boundary block. The target is identified by `Checkpoint(slot, root)`
  where `slot` is the block's actual proposal slot.
- **Three vote kinds, encoded via `target` and `height`**: a justification vote
  (R1) has `target != Checkpoint()` and commits to a specific block at the
  current state-height; a timeout vote (R2) has `target == Checkpoint()` at a
  real height and signals inability to justify; an **empty vote** has an empty
  voted checkpoint — `target == Checkpoint()` *and* `height == Height(0)` — and
  makes no claim about any height. The empty vote acts only through its head
  field (the on-chain record layer) and its finality piggyback: it contributes
  to no justification and sets no timeout marker. It replaces whole-vote
  abstention everywhere, so head fields — the fresh-quorum material — keep
  flowing at all heights.
- **Uniform confirmation gate.** At every height, the choice among the three
  vote kinds is driven by the validator's *safe-confirmed head* (fork-choice
  `get_safe_confirmed_head`): a target vote requires the safe-confirmed head to
  have reached the target; a timeout vote requires it to have reached the voted
  interval; otherwise the empty vote. The gate is the same at every height; a
  *nonjustifiable* height (`is_nonjustifiable_height`) only differs in that it
  never admits a target vote.
- **Protected repeat.** A validator that has already cast a target vote at a
  height never casts a timeout vote at that height: it re-emits the recorded
  target (when it is still the interval-first target and the target gate holds),
  or casts the empty vote. This keeps its finalize gate at that height open.
- **Viability gate** (`is_viable_attestation_target`): the state-machine records
  `justification_targets[i]` only for justification attestations whose `height`
  equals the current state-height and whose `target.slot` lies in the
  current-height interval on the current chain. Timeout votes bypass this gate;
  they set `timeouts[i]` directly when their height matches the current
  state-height. Empty votes never match (their height is `Height(0)`), so they
  set neither. (`finality_participation` updates are independent of viability.)
- **Finality piggyback** confirms a lower-height justified checkpoint when the
  validator's vote record shows this is E1-safe. It is independent of the
  current vote's kind, so a timeout or empty vote may carry a lower-height
  piggyback.
- **Timeout votes are slashable** when they conflict with a finality commitment
  at the same height: a vote with `target = Checkpoint()` at height `H`
  conflicts with any commitment `finality_target = T ≠ Checkpoint()` at
  `finality_height = H` (paper def:slashing). The empty vote makes no height
  claim; since no honest finality commitment at `finality_height = Height(0)`
  exists, an honest validator's empty votes never pair into slashable evidence.

## Local state

An honest validator maintains a vote record for anti-slashing and for the vote
construction:

- `voted_target_at: Dict[Height, Checkpoint]` - the first non-empty target
  signed at each height. This is paper `Delta.T`.
- `voted_timeout_at: Set[Height]` - heights at which the validator has signed a
  timeout vote (`target == Checkpoint()` at a real height). This is paper
  `Delta.tau`.
- `voted_finality_at: Dict[Height, Checkpoint]` - heights for which the
  validator has signed a finality commitment, keyed by `finality_height` and
  storing `finality_target`. This is paper `Delta.lambda` plus the locked
  target.

On signing any `AttestationData` with `height = H`, `target = T`, and a finality
piggyback `(finality_height, finality_target)`:

- If `T != Checkpoint()` and `H not in voted_target_at`, set
  `voted_target_at[H] = T`.
- If `T == Checkpoint()` and `H != Height(0)` (a timeout vote), add `H` to
  `voted_timeout_at`. An empty vote (`H == Height(0)`) records nothing here: it
  is not a vote at any height.
- If `finality_target != Checkpoint()`, set
  `voted_finality_at[finality_height] = finality_target`. (A validator MUST NOT
  sign two different finality commitments at the same `finality_height` — this
  is the E1 lock.)

Old entries below the store's viable height frontier can be pruned
conservatively; missing `voted_target_at` entries simply prevent future
piggybacks for that height.

## Finality attestation

### When to attest

A validator assigned to a beacon committee at slot `S` attests once per round.
The timing is the same as the base spec: attest when a valid block for slot `S`
is received from the expected proposer, or by the attestation deadline
(`ATTESTATION_DUE_BPS_GLOAS` of the slot), whichever comes first.

A validator signs at most **one `AttestationData` per round**. Signing two
different `AttestationData` in the same round triggers the round double-vote
penalty (forced exit, not full slashing — but still undesirable).

*Note*: The paper's round-atomic model fixes all vote content at the round's
first slot; spreading emission across the round's slots is the deployment
schedule adopted here. Vote content is derived from the validator's store at its
own attestation slot; the round's *pointed anchor* is a per-round object
(fork-choice `update_pointed_anchor`), so every committee member of the round
walks from the same anchor once the round-start proposal has been processed.

### Vote kinds: justify (R1), timeout (R2), empty

An honest validator casts at most one R1 vote per state-height, committing to a
target on its head chain. R1 has `target != Checkpoint()` and may carry a
finality piggyback. R2 votes (`target == Checkpoint()` at a real height) drive
the timeout-cert branch of `processHeight` on `state.timeouts`. The empty vote
(`target == Checkpoint()` at `height == Height(0)`) drives nothing: it exists so
that the validator's head field and finality piggyback keep flowing while the
uniform gate blocks both a target and a timeout.

Which kind to cast is decided by [the uniform gate](#the-uniform-gate) below,
subject to two overriding disciplines:

- **E1 lock**: a validator with a finality commitment at the current height
  (`voted_finality_at[current_height]` set) never casts a timeout there — it
  re-submits the locked target as another R1 (see
  [E1 avoidance](#e1-avoidance)). At a *nonjustifiable* height, where no target
  vote is admissible, the lock resolves to the empty vote instead: never a
  timeout, never a target.
- **Protected repeat**: a validator with a recorded target at the current height
  (`voted_target_at[current_height]` set, no finality lock) never casts a
  timeout there — it re-emits the *same* recorded target (never a fresh
  retarget, hence E1-safe) when the target gate still admits it, and otherwise
  casts the empty vote. Not setting `voted_timeout_at[current_height]` keeps the
  finalize gate at that height open for a later piggyback.

In practice, an honest validator's attestation sequence within a single
state-height is either `(R1 at round r)` followed by re-emissions or empty votes
at later rounds of the same height, or `(R2 at round r)` repeated while the gate
admits it — never an R1 followed by an R2 at the same height.

### Constructing `AttestationData`

#### Head field (LMD head vote)

Set `beacon_block_root` to the walk output — the head returned by fork-choice
`get_head`, which walks from the round's anchor (the pointed fresh quorum's
anchor if the round-start proposal carries a valid one, else the record anchor),
then follows the Goldfish descent and the viability descent. The head field is
the validator's SG record vote; it is populated on **every** vote kind,
including the empty vote.

```
head_root = get_head(store).root
head_state = store.block_states[head_root]
current_height = head_state.current_height
attestation_data.beacon_block_root = head_root
attestation_data.height = current_height  # Height(0) instead for an empty vote
```

By the walk's viability descent, `current_height` is at least `h_max - 1`: the
vote height always sits at the height frontier. The base construction's
whole-vote abstention (and its separate height-filter check on the confirmed
head) is gone; what used to be abstention is now the empty vote, decided by the
uniform gate below.

#### Interval-first target

The candidate target is the *interval-first block*: the earliest block on the
head's chain whose slot is in the current height's interval — the block that
opened height `current_height` on that chain. The up-walk is total: it stops at
the finalized root or where the parent is unknown (the store root, under
checkpoint sync — also the stop in the genesis era, where
`current_height_start_slot == GENESIS_SLOT` puts every parent in the interval),
and the reached block is the target.

```
target_root = head_root
while target_root != store.finalized_checkpoint.root:
    parent_root = store.blocks[target_root].parent_root
    if parent_root not in store.blocks:
        break
    if store.blocks[parent_root].slot < head_state.current_height_start_slot:
        break
    target_root = parent_root
base_target = Checkpoint(slot=store.blocks[target_root].slot, root=target_root)
```

#### Safe-confirmed head

The gate reads the validator's *safe-confirmed head* `C` (fork-choice
`get_safe_confirmed_head`): the deepest availability-confirmed block that is
G0-clear. Two conditions on `C` are used: *caught up* — `C` has reached the
height frontier — and *into the interval* — `C` has reached the very height
being voted.

```
safe_confirmed_root = get_safe_confirmed_head(store)
safe_confirmed_height = store.block_states[safe_confirmed_root].current_height
caught_up = safe_confirmed_height >= get_viability_height_threshold(store)
into_interval = safe_confirmed_height >= current_height
```

#### The uniform gate

At every height, in order: (a) a **target vote** `(base_target, current_height)`
iff the safe-confirmed head is the target or a descendant of it; (b) else a
**timeout vote** iff the safe-confirmed head is into the interval; (c) else the
**empty vote**. A *nonjustifiable* height admits no target vote: there, a
timeout iff caught up, else the empty vote. Both the E1 lock and the protected
repeat override a timeout as described above — at a nonjustifiable height each
resolves to the empty vote, since no target vote is admissible there.

```
locked = current_height in voted_finality_at
repeat_at_height = not locked and current_height in voted_target_at

if is_nonjustifiable_height(current_height, head_state.finalized_height):
    # Never a target vote at a nonjustifiable height.
    if caught_up:
        if locked or repeat_at_height:
            # E1 lock / protected repeat: a timeout at a locked height would
            # self-slash against the finality commitment, and an
            # already-targeted height never gets a timeout — but no target
            # vote is admissible at a nonjustifiable height either, so
            # re-submitting is not an option. The empty vote is the E1-safe
            # resolution: it makes no height claim.
            attestation_data.target = Checkpoint()  # empty vote
            attestation_data.height = Height(0)
        else:
            attestation_data.target = Checkpoint()  # timeout vote
    else:
        attestation_data.target = Checkpoint()  # empty vote
        attestation_data.height = Height(0)
else:
    # (a) Target case: C >= T. The target's G0-clearance is implied: C is
    # G0-clear (safe confirmation) and any block conflicting with T conflicts
    # with its descendant C, so no explicit G0 check on T is needed.
    target_case = is_ancestor(
        store,
        ForkChoiceNode(root=safe_confirmed_root, payload_status=PAYLOAD_STATUS_PENDING),
        ForkChoiceNode(root=target_root, payload_status=PAYLOAD_STATUS_PENDING),
    )
    if repeat_at_height:
        # Protected repeat: re-emit the same target, or the empty vote;
        # never a timeout at an already-targeted height.
        if target_case and base_target == voted_target_at[current_height]:
            attestation_data.target = voted_target_at[current_height]
        else:
            attestation_data.target = Checkpoint()  # empty vote
            attestation_data.height = Height(0)
    elif target_case:
        if locked:
            # E1 lock: re-submit the locked target as another R1.
            attestation_data.target = voted_finality_at[current_height]
        else:
            # First target vote at this height (R1 justify).
            attestation_data.target = base_target
    elif into_interval:
        # (b) Confirmed into the interval, but the interval-first target is
        # not a target case (C does not extend it, or it is G0-vetoed): time
        # the height out. Timeouts are confirmation-gated but not G0-gated.
        if locked:
            attestation_data.target = voted_finality_at[current_height]
        else:
            attestation_data.target = Checkpoint()  # timeout vote
    else:
        # (c) Not confirmed into the interval: the empty vote.
        attestation_data.target = Checkpoint()
        attestation_data.height = Height(0)
```

*Note*: If the validator cast an R1 at `current_height` and subsequently
observes a newer block on the same chain, it is NOT safe to retarget — any later
vote at the same height must re-emit the recorded target (or be an empty vote; a
timeout only where no target was recorded and no finality lock applies).

*Note*: The retroactive finality lock matters when the view reverts to
`current_height` after the validator has already attached a finality commitment
at `finality_height = current_height` from a higher-height vote. Under E1, the
validator is bound to that finality target at `current_height`; voting a
different target (or a timeout, since `Checkpoint() ≠ T`) at `current_height`
would self-evidence E1. The empty vote is always safe: it makes no height claim.

Votes with a stale height are still accepted by `process_attestation` (the
`finality_participation` update may still be useful), but they do not update
target tracking and earn no TIMELY_TARGET reward.

#### Finality piggyback

The finality piggyback confirms a previously justified checkpoint. It is valid
when it points to a lower height than the current state-height and the vote
record shows that the validator previously voted for the same target at that
lower height and has not timed out there. It attaches independently of the
current vote's kind: target, timeout, and empty votes all carry it when the gate
passes.

The rule:

1. Let `justified_height = head_state.justified_height` and
   `J = head_state.justified_checkpoint`.
2. Attach the piggyback only if `justified_height < current_height`,
   `voted_target_at.get(justified_height) == J`,
   `justified_height not in voted_timeout_at`, and any prior finality commitment
   at `justified_height` is also to `J`.
3. Otherwise: leave the piggyback empty (`finality_target = Checkpoint()`,
   `finality_height = FAR_FUTURE_HEIGHT`).

```
justified_height = head_state.justified_height
J = head_state.justified_checkpoint
prior_finality_at_justified_height = voted_finality_at.get(justified_height)

if (
    justified_height < current_height
    and voted_target_at.get(justified_height) == J
    and justified_height not in voted_timeout_at
    and prior_finality_at_justified_height in (None, J)
):
    attestation_data.finality_target = J
    attestation_data.finality_height = justified_height
else:
    attestation_data.finality_target = Checkpoint()
    attestation_data.finality_height = FAR_FUTURE_HEIGHT
```

**Why require `voted_target_at[justified_height] == J` and no timeout there**:
this is the paper's `Delta.T(h_f) = T_f` and `Delta.tau(h_f) = false` gate. The
finality commitment locks the voter to `J` at height `justified_height` via E1.
If the validator never voted for `J` at that height, voted for another target,
or has timed out at that height, attaching the piggyback would either be unsafe
or would create a slashable trace once paired with the timeout. The protected
repeat exists precisely to keep `Delta.tau` false at already-targeted heights,
so this gate can still pass later.

*Note*: A finality attestation carries no payload-availability signal. It is an
LMD vote for a beacon block at `PAYLOAD_STATUS_PENDING`; the payload decision is
made by the available-attestation / Goldfish layer, not the finality vote.

### Broadcast

Broadcast the signed attestation on the appropriate subnet. Aggregation follows
the same pattern as the base spec (aggregation selection via `is_aggregator`,
aggregate construction, timed broadcast at the aggregate deadline
(`AGGREGATE_DUE_BPS_GLOAS` of the slot)).

## Block proposal: pointing to a fresh quorum

Block proposal duties follow the base spec, with one addition for the proposer
of a round's **first slot**. Such a proposer MAY *point to a fresh quorum* of
the previous round: it selects, from the aggregates it has collected for
finality-attestation processing, a set of previous-round finality attestations
whose distinct signers' effective balances sum to at least two-thirds of the
**total** active balance (signers are union-counted, so duplicate signers across
or within aggregates are tolerated and add no weight), and places them in
`body.anchor_quorum`. No new signatures are involved — the reference reuses
aggregates the proposer already has.

If the reference verifies (fork-choice `get_pointed_anchor`: aggregate
signatures over distinct previous-round signers, the absolute two-thirds weight,
known head fields), its anchor — the *highest common ancestor* of the head
fields, i.e. the deepest block whose subtree contains all of them — is adopted
by every validator for the whole round as the walk's starting point, even if
other or deeper fresh quorums exist or later become visible. An invalid
reference is simply ignored and never invalidates the block.

When more than one such two-thirds-weight subset is available, an honest
proposer SHOULD prefer the subset whose anchor is *deepest*: including head
fields from a conflicting minority (e.g. validators still voting an old head on
a diverged fork) only pulls the highest common ancestor shallower, and an anchor
that no longer descends from the round's cascade root (`get_cascade_root`) is
verified but never adopted by the walk (`get_walk_anchor`) — dropping those
minority head fields keeps the anchor as deep as the honest majority supports.
The proposer SHOULD also verify its own reference locally before including it —
apply the pointed-anchor verification procedure (`get_pointed_anchor`) to the
candidate reference and confirm the resulting anchor descends from its cascade
root — so it does not ship a reference that other validators would ignore. This
is proposer behavior, not consensus: every validator re-verifies the reference
independently, so a shallow or dropped selection only forgoes the one-round
synchronization benefit, never safety.

An honest round-start proposer SHOULD point whenever it can assemble such a
quorum. Under at least two-thirds honest-and-online stake the previous round's
honest attestations always form one: gated-out honest validators cast empty
votes rather than abstaining, so their head fields keep flowing. Pointing is the
one cross-view record-synchronization object of the protocol — it is what gives
all honest validators a common anchor within one honest-proposer round during
healing.

## Available attestation

Validators assigned to the available committee for slot `S` produce an
`AvailableAttestation` carrying the LMD head vote and payload availability
signal. One per slot. Timing: attest before the available-confirmation deadline
(`AVAILABLE_CONFIRMATION_DUE_BPS` of the slot). This feeds the Goldfish
fork-choice layer. This duty and the user-facing available confirmation are
never gated by finality-gadget or record state.

## How to avoid slashing

### E1 avoidance

The only slashing condition is E1: if you sign `finality_target = T` at
`finality_height = H`, then any attestation you signed at `height = H` with
`target != T` is slashable evidence. **Timeout votes are slashable too**:
`target = Checkpoint()` at `height = H` conflicts with
`finality_target = T ≠ Checkpoint()` at `finality_height = H` (paper
def:slashing). **Empty votes are safe for honest validators**: with
`height = Height(0)` they conflict only with a finality commitment at
`finality_height = Height(0)`, which no honest validator ever signs — heights
start at `GENESIS_HEIGHT == Height(1)`, so no honest finality commitment at
height `0` exists.

**How to stay safe**: maintain `voted_target_at[H]`, `voted_timeout_at`, and
`voted_finality_at[H]`. Use these (plus the retroactive `voted_finality_at[H]`
lock) to drive the vote choice at height `H`. Only set `finality_target` when
your prior target at the justified height matches the justified checkpoint and
no timeout was signed at that height (the rule in
[Finality piggyback](#finality-piggyback) above). The
[uniform gate](#the-uniform-gate) construction bakes in both disciplines: if
`voted_finality_at[current_height]` is set, the validator re-submits another R1
with the locked target rather than casting a timeout (at a nonjustifiable
height, where no target is admissible, it casts the empty vote); if only
`voted_target_at[current_height]` is set, the protected repeat re-emits the
recorded target or casts the empty vote, never a timeout.

*Note*: There is no E2 (height double-target) condition. Signing an R2 (timeout)
vote when no finality lock at `current_height` exists is safe even though it
differs from a prior R1 at the same height — the two votes alone are not
directly slashable. The honest construction nevertheless avoids it via the
protected repeat, to keep the finalize gate at that height open; the slashable
case is exactly when an R2 timeout collides with a same-height finality
commitment.

### Round double-vote

Signing two different `AttestationData` in the same round triggers
`RoundDoubleVoteEvidence`: forced exit plus a fixed deduction (not full
slashing). **Sign at most one `AttestationData` per round.**

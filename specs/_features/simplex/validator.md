# Simplex -- Honest Validator

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Overview](#overview)
- [Local state](#local-state)
  - [New `RoundSelectionEvent` and `FrozenRoundVote`](#new-roundselectionevent-and-frozenroundvote)
- [Validator assignments](#validator-assignments)
  - [Modified `get_committee_assignment`](#modified-get_committee_assignment)
- [Finality attestation](#finality-attestation)
  - [When to attest](#when-to-attest)
    - [New `freeze_round_vote`](#new-freeze_round_vote)
  - [Vote kinds: justify (R1), timeout (R2), empty](#vote-kinds-justify-r1-timeout-r2-empty)
  - [Constructing `AttestationData`](#constructing-attestationdata)
    - [Processing the selected states](#processing-the-selected-states)
    - [Head field (LMD head vote)](#head-field-lmd-head-vote)
    - [Current-height target](#current-height-target)
    - [Safe-confirmed head](#safe-confirmed-head)
    - [The uniform gate](#the-uniform-gate)
      - [New `is_authenticated_checkpoint_on_chain`](#new-is_authenticated_checkpoint_on_chain)
      - [New `is_attestation_target_on_chain`](#new-is_attestation_target_on_chain)
      - [New `get_attestation_target`](#new-get_attestation_target)
    - [Finality piggyback](#finality-piggyback)
    - [Complete construction](#complete-construction)
  - [Signing](#signing)
  - [Broadcast](#broadcast)
- [Block proposal](#block-proposal)
  - [Fixed-quorum TSQ root syncing](#fixed-quorum-tsq-root-syncing)
  - [Finality attestations and historical proofs](#finality-attestations-and-historical-proofs)
  - [Available-attestation aggregates](#available-attestation-aggregates)
  - [Round-double-vote evidence](#round-double-vote-evidence)
- [Available attestation](#available-attestation)
  - [Constructing the data](#constructing-the-data)
  - [Committee positions and signing](#committee-positions-and-signing)
  - [Publication and aggregation](#publication-and-aggregation)
- [How to avoid slashing](#how-to-avoid-slashing)
  - [E1 and E2 avoidance](#e1-and-e2-avoidance)
  - [Round double-vote](#round-double-vote)

<!-- mdformat-toc end -->

This is an accompanying document to
[Simplex -- The Beacon Chain](./beacon-chain.md) and
[Simplex -- Fork Choice](./fork-choice.md), describing the expected behavior of
an honest validator in the fresh-simplex-with-height-filter-and-timeouts
finality gadget.

This document specifies honest behavior for the executable Simplex research
profile. Dynamic-validator-set safety follows the inherited Ethereum weak
subjectivity model. This document does not discharge the committee-convergence,
TSQ selection-state convergence, timing, or healing obligations recorded in the
beacon-chain and fork-choice specifications. In particular, the fail-closed
committee checks below make those assumptions explicit; they do not turn them
into proven production guarantees.

## Overview

Simplex splits validator attestation duties into two types:

1. **Finality attestation** (`Attestation`): assigned via beacon committees
   spread across the slots of a round (length per `ROUND_SCHEDULE`). Carries the
   finality target vote (or `Checkpoint()` for a timeout or empty vote), the
   head field (LMD head vote), and optional finality piggyback. The vote kind,
   height, target, and finality piggyback are frozen together at the round-start
   selection event; the head field is read live when the assigned duty is
   emitted. One per round.

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
  field (the latest-head-vote grade input) and its finality piggyback: it
  contributes to no justification and sets no timeout marker. Within the uniform
  gate it replaces whole-vote abstention, so head fields — the SG and TSQ inputs
  — keep flowing at all heights. A separate duty-time ancestry failure still
  suppresses the whole attestation.
- **Uniform confirmation gate.** At every height, the choice among the three
  vote kinds is driven by the validator's *safe-confirmed head* (fork-choice
  `get_safe_confirmed_head`): a target vote requires the safe-confirmed head to
  have reached the target; a timeout vote requires it to have reached the voted
  interval; otherwise the empty vote. The gate is the same at every height; a
  *nonjustifiable* height (`is_nonjustifiable_height`) only differs in that it
  never admits a *fresh* target vote. A compatible saved-target repeat is
  marker-only because the state suppresses justification there.
- **Durable history and protected repeat.** Timeout-first validators repeat only
  timeout (or empty). Target-first validators repeat the same compatible target,
  cast empty before confirmation reaches the interval, or cross once to timeout
  when an unlocked saved target is incompatible with the selected chain. That
  bridge deliberately closes the validator's later finality gate at the height.
  E1-locked validators never take it.
- **Current-height tracking**: each branch stores its first block checkpoint as
  `current_height_target`. Every valid current-height attestation sets
  `timeouts[i]`; validation already requires a nonempty target to be on the
  including chain. It sets `target_participation[i]` only when that target is
  nonempty and exactly `current_height_target`. Empty votes never match (their
  height is `Height(0)`), so they set neither. (`finality_participation` updates
  are independent of current-height tracking.)
- **Finality piggyback** confirms a lower-height justified checkpoint when the
  validator's local signing history shows this is E1-safe. It is independent of
  the current vote's kind, so a timeout or empty vote may carry a lower-height
  piggyback.
- **Timeout votes are slashable** when they conflict with a finality commitment
  at the same height: a vote with `target = Checkpoint()` at height `H`
  conflicts with any commitment `finality_target = T ≠ Checkpoint()` at
  `finality_height = H` (paper def:slashing). The empty vote makes no height
  claim; since no honest finality commitment at `finality_height = Height(0)`
  exists, an honest validator's empty votes never pair into slashable evidence.

## Local state

An honest validator maintains local signing history for anti-slashing and vote
construction:

### New `RoundSelectionEvent` and `FrozenRoundVote`

```python
@dataclass(eq=True, frozen=True)
class RoundSelectionEvent:
    """Deadline capability carrying the selection view's frozen roots."""

    slot: Slot
    head_root: Root
    safe_confirmed_root: Root
    finalized_root: Root


@dataclass(eq=True, frozen=True)
class FrozenRoundVote:
    """The FG fields selected once at the start of a round."""

    target: Checkpoint
    height: Height
    finality_target: Checkpoint
    finality_height: Height
```

- `voted_target_at: Dict[Height, Checkpoint]` - the first non-empty target
  signed at each height. This is paper `Delta.T`.
- `voted_timeout_at: Set[Height]` - heights at which the validator has signed a
  timeout vote (`target == Checkpoint()` at a real height). This is paper
  `Delta.tau`.
- `voted_finality_at: Dict[Height, Checkpoint]` - heights for which the
  validator has signed a finality commitment, keyed by `finality_height` and
  storing `finality_target`. This is paper `Delta.lambda` plus the locked
  target.
- `signed_attestation_at_round: Dict[Round, AttestationData]` - the one distinct
  finality-attestation message signed in each round. This is the restart-safe
  guard against `RoundDoubleVoteEvidence`.
- `frozen_round_vote_at: Dict[Round, FrozenRoundVote]` - the vote kind, height,
  target, and finality piggyback selected at that round's common selection
  event. A later duty reads this entry and never recomputes those fields from a
  newer view.
- `skipped_attestation_at_round: Set[Round]` - rounds whose scheduled finality
  duty failed the live-head ancestry check. The skip is persisted before the
  helper returns and prevents reconstruction or retry after a restart.
- `signed_available_attestation_at: Dict[Tuple[ValidatorIndex, Slot], AvailableAttestationData]`
  - the one distinct available-attestation datum signed by each locally managed
    validator in each slot. Available equivocation is not an E1 slashing
    offense, but it is excluded by Goldfish and must not be created by an honest
    signer.

On signing any `AttestationData` with `height = H`, `target = T`, and a finality
piggyback `(finality_height, finality_target)`:

- If `T != Checkpoint()` and `H not in voted_target_at`, set
  `voted_target_at[H] = T`.
- If `T == Checkpoint()` and `H != Height(0)` (a timeout vote), add `H` to
  `voted_timeout_at`. An empty vote (`H == Height(0)`) stores nothing here: it
  is not a vote at any height.
- If `finality_target != Checkpoint()`, set
  `voted_finality_at[finality_height] = finality_target`. (A validator MUST NOT
  sign two different finality commitments at the same `finality_height` — this
  is the E1 lock.)
- Let `round = compute_round_at_slot(attestation_data.slot)`. If
  `signed_attestation_at_round[round]` already exists and differs from the
  complete `AttestationData`, refuse to sign; otherwise store the complete data
  at that key.

The three height-keyed maps, `signed_attestation_at_round`,
`skipped_attestation_at_round`, and the current entry of `frozen_round_vote_at`
are durable slashing/penalty-protection data. The frozen entry MUST be committed
before the round's first signature is released; the signing-history updates and
signed message MUST then be committed atomically before releasing that
signature. The viable height frontier is **not** a safe pruning boundary: an
older target may still need a finality piggyback, and forgetting a timeout,
finality lock, or round message can permit slashable or penalizable signing
after a view rollback. A signer fails closed if any required history or the
current round's frozen entry is unavailable; it never reconstructs that entry
from a later view. The height histories and signed messages may be archived only
under the operator's normal slashing-protection retention policy after the key
is permanently disabled (for example, after withdrawal). A frozen entry may be
pruned after its round has ended and either the validator's signed message or
its skipped-duty marker for that round is durably stored.

`signed_available_attestation_at` has a separate bounded lifetime because an
available equivocation is not slashable. Its entry is committed under the same
crash-consistency rule before signing, but MUST be pruned once its slot's round
is older than the previous round. At that point the vote is outside both its
one-slot fork-choice use and the current/previous-round block-inclusion window,
so retaining the entry provides no protocol protection.

## Validator assignments

Simplex repeats the beacon-committee shuffle in every round of an epoch. A
validator therefore requests one assignment per round, rather than one
assignment per epoch. To enumerate an epoch's duties, start at the round
containing `compute_start_slot_at_epoch(epoch)` and call the helper once for
each round returned by `get_rounds_per_epoch_at_slot`.

### Modified `get_committee_assignment`

```python
def get_committee_assignment(
    state: BeaconState, round: Round, validator_index: ValidatorIndex
) -> Optional[Tuple[Sequence[ValidatorIndex], CommitteeIndex, Slot]]:
    """
    [Modified in Simplex] Return ``validator_index``'s committee assignment in
    ``round``. The returned tuple contains the committee, committee index, and
    assigned slot. Return ``None`` if the validator has no assignment.
    """
    assignment_epoch = compute_epoch_at_round(round)
    next_epoch = Epoch(get_current_epoch(state) + 1)
    assert assignment_epoch <= next_epoch

    start_slot = compute_start_slot_at_round(round)
    slots_per_round = get_slots_per_round_at_slot(start_slot)
    committee_count_per_slot = get_committee_count_per_slot(state, assignment_epoch)
    for slot in range(start_slot, start_slot + slots_per_round):
        for index in range(committee_count_per_slot):
            committee = get_beacon_committee(state, Slot(slot), CommitteeIndex(index))
            if validator_index in committee:
                return committee, CommitteeIndex(index), Slot(slot)
    return None
```

## Finality attestation

### When to attest

A validator assigned to a beacon committee at slot `S` attests once per round.
Its FG fields are selected once in the round's first slot.

The executable profile designates every round as a TSQ synchronization round. At
`VIEW_FREEZE_DUE_BPS` in the preceding round's last slot, after processing
messages delivered exactly at that boundary, the high-resolution scheduler calls
`on_tick_per_high_resolution`, which freezes the TSQ view. At the next round
boundary, before admitting any proposal for the new round, the ordinary
`on_tick` path calls `freeze_tsq_selection`. This pins the proposal-independent
candidate tree, electorate, and `TSQSelection.candidate_root`. A missed view
freeze or selection event is never reconstructed from a later view.

At the common proposal/attestation deadline (`ATTESTATION_DUE_BPS_GLOAS`), after
processing all valid round-start proposal copies admitted to fork choice for
that slot and all re-gossiped support-round messages, conflicting copies, and
ancestry delivered by the boundary, the scheduler calls
`on_tick_per_high_resolution`, which freezes the stable root. This computes the
receiver's fixed-quorum TSQ lock from the same signed data present in both its
frozen and current views, excluding every known round equivocator. It uses that
lock only if the pinned selection exists and the lock remains viable and
descends from the action state's current Simplex root; otherwise it freezes the
ordinary grade-1 fallback. The scheduler then computes the head from that stable
root, snapshots the head, safe-confirmed, and finalized roots into the local
`RoundSelectionEvent`, and calls `freeze_round_vote` for every locally managed
validator. In the first slot this walk starts at the unique proposal
distinguished by the action, if any. Proposals do not trigger an earlier
selection. At equality, inbound proposals and support timestamped at the
deadline are processed before this sequence. Later votes, proposals, or
equivocation evidence cannot change the completed stable-root decision or the
frozen FG fields.

At its assigned slot `S`, the validator constructs and signs its attestation
when a valid block for `S` is received from the expected proposer, or by that
slot's attestation deadline, whichever comes first. It uses the frozen FG fields
and a live duty-time head field. A first-slot assignment is the exception to
early construction: it waits for the common selection deadline, freezes first,
and only then constructs its attestation. If the current round's frozen entry is
unavailable after a restart or checkpoint sync, the validator does not sign a
finality attestation for the remainder of that round; it never reconstructs the
entry from a later view.

A validator signs at most **one `AttestationData` per round**. Signing two
different `AttestationData` in the same round triggers at least the round
double-vote penalty; if the pair also satisfies E1 or E2, it instead supplies
full `AttesterSlashing` evidence.

*Note*: This is the distributed-duty refinement of the paper's round-atomic
target selection. In a counted TSQ honest-proposer round, time-shifted delivery
ensures that the pinned proposer candidate extends every honest receiver's
fixed-quorum lock. The honest proposal descends from that candidate and the
proposer's live available-confirmed head. A counted round additionally requires
that it descend every receiver's action-time live available-confirmed head (the
common confirmed prefix in the paper premise). With the required common Goldfish
input, it therefore puts every honest validator on the same canonical chain and
yields the same current-height target. These are explicit liveness premises; a
missing or incompatible TSQ selection safely falls back to grade 1 and supplies
no same-round synchronization guarantee.

The executable profile does not yet implement the paper's staged confirmation,
standalone SG pre-vote, and FG-vote schedule. It instead distributes one
combined finality attestation across the round: the FG fields are frozen at the
common first-slot action, while the SG head field is read at the assigned duty.
TSQ therefore consumes the publicly assigned preceding round's
combined-attestation head fields. The paper's confirm-then-SG-then-FG same-round
and numerical liveness bounds do not directly apply to this executable schedule.

If the proposal itself opened the current height but is not yet safe-confirmed
at the selection event, the uniform gate freezes an empty or timeout vote rather
than speculating on that block. The target is considered in the next round; it
is never adopted asynchronously by later duties in the current round.

#### New `freeze_round_vote`

The helper is idempotent: the first selection for a round wins. The caller MUST
invoke it at the exact common deadline and persist the returned entry before
releasing any signature in the round. A missed deadline does not authorize a
late reconstruction; without an entry, the validator skips that round's duty.
`RoundSelectionEvent` is a local capability issued only by the high-resolution
duty scheduler after it has processed messages delivered at or before the
deadline. Its head, safe-confirmed, and finalized roots snapshot that
deadline-filtered view, so retaining the event cannot admit later messages. It
is needed because `Store.time` has whole-second resolution while some presets
have subsecond deadlines; it is not a wire object and cannot be synthesized for
a missed event.

```python
def freeze_round_vote(
    store: Store,
    selection_event: RoundSelectionEvent,
    voted_target_at: Dict[Height, Checkpoint],
    voted_timeout_at: Set[Height],
    voted_finality_at: Dict[Height, Checkpoint],
    frozen_round_vote_at: Dict[Round, FrozenRoundVote],
) -> FrozenRoundVote:
    """Select and persist this validator's FG fields for the current round."""
    slot = get_current_slot(store)
    round = compute_round_at_slot(slot)
    if round in frozen_round_vote_at:
        return frozen_round_vote_at[round]
    assert slot == compute_start_slot_at_round(round)
    assert selection_event.slot == slot

    head_root = selection_event.head_root
    safe_confirmed_root = selection_event.safe_confirmed_root
    assert head_root in store.blocks
    finalized_root = selection_event.finalized_root
    assert safe_confirmed_root in store.blocks
    assert finalized_root in store.blocks
    head_state = get_current_slot_state(store, head_root)
    safe_confirmed_state = copy(store.block_states[safe_confirmed_root])
    base_target = head_state.current_height_target
    # The first block of a height cannot contain its own root in its post-state.
    # At same-slot selection, derive that target from the selected head. A head
    # below the processed height start means empty slots advanced the height and
    # there is no target yet.
    if (
        base_target == Checkpoint()
        and store.blocks[head_root].slot >= head_state.current_height_start_slot
    ):
        base_target = Checkpoint(slot=store.blocks[head_root].slot, root=head_root)
    target, height = get_attestation_target(
        store,
        head_root,
        head_state,
        safe_confirmed_root,
        safe_confirmed_state,
        base_target,
        voted_target_at,
        voted_timeout_at,
        voted_finality_at,
    )

    justified_height = head_state.justified_height
    justified_checkpoint = head_state.justified_checkpoint
    prior_finality = voted_finality_at.get(justified_height)
    if (
        justified_height < head_state.current_height
        and voted_target_at.get(justified_height) == justified_checkpoint
        and justified_height not in voted_timeout_at
        and prior_finality in (None, justified_checkpoint)
    ):
        finality_target = justified_checkpoint
        finality_height = justified_height
    else:
        finality_target = Checkpoint()
        finality_height = FAR_FUTURE_HEIGHT

    frozen_vote = FrozenRoundVote(
        target=target,
        height=height,
        finality_target=finality_target,
        finality_height=finality_height,
    )
    frozen_round_vote_at[round] = frozen_vote
    return frozen_vote
```

### Vote kinds: justify (R1), timeout (R2), empty

An honest validator casts at most one R1 vote per state-height, committing to a
target on its head chain. R1 has `target != Checkpoint()` and may carry a
finality piggyback. R2 votes (`target == Checkpoint()` at a real height) drive
the timeout-cert branch of `processHeight` on `state.timeouts`. The empty vote
(`target == Checkpoint()` at `height == Height(0)`) drives nothing: it exists so
that the validator's head field and finality piggyback keep flowing while the
uniform gate blocks both a target and a timeout.

Which kind to cast is decided by [the uniform gate](#the-uniform-gate) below,
subject to three overriding disciplines:

- **E1 lock**: a validator with a finality commitment at the current height
  (`voted_finality_at[current_height]` set) never casts a timeout there — it
  re-submits the locked target as another R1 only while that exact checkpoint is
  authenticated on both its current voted-head chain and its safe-confirmed
  chain (see [E1 avoidance](#e1-avoidance)). It need not remain the branch's
  `current_height_target`: in that case it contributes only to `timeouts`.
  Otherwise the lock resolves to the empty vote. At a *nonjustifiable* height a
  compatible repeat remains marker-only and is allowed; an incompatible lock
  resolves to empty: never a timeout, never a different target.
- **Timeout-first discipline**: a validator with
  `voted_timeout_at[current_height]` set never casts a target there. It repeats
  the timeout only when the safe-confirmed head has reached the interval;
  otherwise it casts empty.
- **Protected repeat**: a validator with a previously signed target at the
  current height (`voted_target_at[current_height]` set, no finality lock)
  re-emits the *same* target (never a fresh retarget, hence E1-safe) when the
  target gate still admits it. If that target is incompatible with the selected
  chain, the validator may instead bridge once to a timeout after safe
  confirmation has reached the interval, at either ordinary or nonjustifiable
  height. Until then it casts empty. Not taking that bridge keeps the finalize
  gate at the height open; taking it closes that gate durably.

These disciplines do not remove a contribution already made at the current
height. Processing any valid current-height vote sets `timeouts[i]`, and that
bit persists until the height advances even if the validator later emits an
empty vote. In particular, votes for a lock target continue to count toward the
height-advance quorum on any canonical chain containing that target, even when
it is not that chain's `current_height_target`. The marker-only repeat and
unlocked bridge prevent an old branch-relative target history from silently
stranding an honest marker. A justification completed from old, pre-convergence
histories need not itself have two-thirds finality-eligible honest weight. It
remains safe and may be skipped: conditional on a clean common first-selection
opportunity at a fresh height, the honest target votes alone justify it and a
later honest-only quorum finalizes it.

In practice, an honest validator's attestation sequence at a numeric height is
either `(R1 at round r)` followed by compatible re-emissions, empty votes, or
one confirmation-gated transition to R2; or `(R2 at round r)` followed by
confirmation-gated R2 repeats or empty votes. The reverse R2-to-R1 sequence
never occurs.

### Constructing `AttestationData`

#### Processing the selected states

Fork choice stores the post-state of the last block on each branch. When one or
more slots are empty, that state may be older than the round-start selection
slot. At the selection event, the selected head state is advanced on a copy to
determine the chain height for the entire round; this MUST NOT mutate
`store.block_states`. Later duties use the resulting `FrozenRoundVote` even if
the live state has advanced to another height.

The safe-confirmation gate is intentionally different: it reads the stored
post-block state of the safe-confirmed block itself. The paper defines a block's
state-height as its height immediately after block processing. Advancing that
state through later empty slots would manufacture a higher
`safe_confirmed_height` without any safe-confirmed block in the new interval,
allowing a timeout marker that violates the safe-confirmation certificate. Thus
empty-slot processing may advance the frozen vote height and make the interval
target empty, but only an actual safe-confirmed block can open the
target/timeout gate at that height.

```python
def get_current_slot_state(store: Store, root: Root) -> BeaconState:
    """Return a copy of ``root``'s state processed to the store's current slot."""
    state = copy(store.block_states[root])
    current_slot = get_current_slot(store)
    assert state.slot <= current_slot
    if state.slot < current_slot:
        process_slots(state, current_slot)
    return state
```

#### Head field (LMD head vote)

Set `beacon_block_root` to the walk output — the head returned by fork-choice
`get_head`, which walks from the round's stable root (the fixed-quorum
frozen/current TSQ lock when it is available and passes the action-state checks,
else the grade-1 fallback), then follows the Goldfish descent and the viability
descent. In the round-start slot, a unique proposal distinguished by the common
action starts that walk when it descends both the stable root and the
action-time live available-confirmed head. It may replace an unconfirmed
ordinary head; from the next slot onward it has no special status. The head
field is the validator's SG latest head vote; it is populated on **every** vote
kind, including the empty vote, and contributes to grades as soon as a valid
attestation is received over the network.

Unlike the FG fields, this head is deliberately read at the assigned duty. The
common frozen target is an ancestor of every honest duty-time head under the
post-convergence common-prefix premise. If either frozen non-empty checkpoint is
not on the live head chain, construction fails closed and the validator skips
that scheduled duty; it never changes the frozen FG fields or retries from a
later view.

```
head = get_head(store)
head_root = head.root
attestation_data.beacon_block_root = head_root
attestation_data.target = frozen_vote.target
attestation_data.height = frozen_vote.height
```

At selection, the walk's viability descent puts the frozen height at least at
`h_max - 1`. The base construction's gate-level abstention (and its separate
height-filter check on the confirmed head) is gone; what used to be that
abstention is now the empty vote, decided once by the uniform gate below. The
duty-time ancestry invariant remains a whole-attestation fail-closed check.

#### Current-height target

The round's candidate target is the first block checkpoint in the processed
selection state:

```
base_target = head_state.current_height_target
```

The field is `Checkpoint()` until the current height has a block whose root can
be computed without circularity. At the following slot, `process_slot` fills the
latest header's state root and sets the target to the first block of the height.
For that block's own slot, `freeze_round_vote` derives the same checkpoint from
the selected head when its slot is at or above `current_height_start_slot`. If
processed empty slots advanced the height after that head, the head slot is
below the start and the candidate remains empty. The first block cannot include
a vote for itself. Before any descendant includes such a vote, `process_slot`
has set the same root.

This state field avoids reconstructing the first block with a local ancestry
walk and makes every accepted target on one branch identical. Under checkpoint
sync, the target block may be below the locally stored anchor. The gate accepts
state equality on both selected chains, or a target strictly below the trusted
finalized anchor when both selected roots descend from that anchor. The exact
anchor itself remains directly known. The gate rejects an unknown post-anchor
target that the safe-confirmed state does not contain. Different branches may
have different targets; signing both remains E2-slashable.

#### Safe-confirmed head

The gate reads the validator's *safe-confirmed head* `C` (fork-choice
`get_safe_confirmed_head`): the deepest availability-confirmed block that is
G0-clear. A timeout marker requires `C` to be *into the interval*: its stored
post-block state has reached the very height being voted.

```
safe_confirmed_root = selection_event.safe_confirmed_root
safe_confirmed_state = copy(store.block_states[safe_confirmed_root])
safe_confirmed_height = safe_confirmed_state.current_height
into_interval = safe_confirmed_height >= current_height
```

#### The uniform gate

At every height, in order: (a) a **target vote** `(base_target, current_height)`
iff the safe-confirmed head is the target or a descendant of it; (b) else a
**timeout vote** iff the safe-confirmed head is into the interval; (c) else the
**empty vote**. A durable E1 lock repeats its exact locked target while that
checkpoint is authenticated on both the current voted-head chain and the
safe-confirmed chain; it need not equal `base_target`. An incompatible or
unavailable E1-locked target resolves to the empty vote, never a timeout or a
different target. An unlocked protected repeat follows the same rule for its
exact saved target, or bridges to timeout once safe confirmation is into the
interval; it never selects a different target. A *nonjustifiable* height admits
no **fresh** target vote. A compatible saved target may be re-emitted there; the
state's latched class suppresses justification, so this is marker-only and can
help close the height without changing `J`. Otherwise there is a timeout iff `C`
is into the interval and the signer has no E1 finality lock, else the empty
vote. Vote kind is otherwise durable at a numeric height: timeout-first
validators may only repeat the timeout (subject to the same confirmation gate).
Target-first validators may make one one-way transition to timeout only when
their saved target is incompatible and no E1 finality lock exists. Once used,
the timeout history prevents either a later target or a finality piggyback at
that height.

*Why the unlocked bridge is required*: before stabilization, honest validators
may first target the same numeric height on mutually incompatible histories.
After they converge to a common chain, protected repeat without an escape would
strand their height-progress markers even if every branch classifies the height
as ordinary: no old target need retain two-thirds support, while no target-first
validator could time out. Both target and timeout votes set the on-chain marker,
and an R1-to-R2 transition is not E1 evidence before a finality lock. The bridge
therefore permits that transition only after safe confirmation reaches the
selected interval and only before a lock. Its durable timeout history prevents
an unsafe later finality piggyback; the reverse timeout-to-target transition is
never permitted.

*Pre-convergence histories*: protected repeat does not imply that every
justification assembled from old votes has a two-thirds finality-eligible
electorate. Byzantine weight can complete a target quorum whose incompatible
honest complement cannot finalize that particular target. E1 correctly keeps the
result safe. Conditional on a clean common first-selection opportunity at a
fresh height, the honest-only target quorum justifies and a later honest-only
quorum finalizes it. An earlier mixed-history justification may simply be
skipped. No additional precommit phase is required for this conditional restart
guarantee.

##### New `is_authenticated_checkpoint_on_chain`

```python
def is_authenticated_checkpoint_on_chain(
    store: Store,
    chain_root: Root,
    checkpoint: Checkpoint,
) -> bool:
    """
    Return whether ``chain_root`` contains a checkpoint already authenticated by
    a trusted state. A checkpoint strictly below the finalized/checkpoint-sync
    anchor need not have its own block retained locally; an exact-slot checkpoint
    must name the anchor root itself and therefore takes the known-root path.
    """
    if checkpoint == Checkpoint():
        return False
    chain_node = ForkChoiceNode(root=chain_root, payload_status=PAYLOAD_STATUS_PENDING)
    if checkpoint.root in store.blocks:
        return store.blocks[checkpoint.root].slot == checkpoint.slot and is_ancestor(
            store,
            chain_node,
            ForkChoiceNode(root=checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING),
        )

    anchor = store.finalized_checkpoint
    if checkpoint.slot >= anchor.slot or anchor.root not in store.blocks:
        return False
    return is_ancestor(
        store,
        chain_node,
        ForkChoiceNode(root=anchor.root, payload_status=PAYLOAD_STATUS_PENDING),
    )
```

##### New `is_attestation_target_on_chain`

```python
def is_attestation_target_on_chain(
    store: Store,
    chain_root: Root,
    chain_state: BeaconState,
    target: Checkpoint,
    height: Height,
) -> bool:
    """
    Return whether ``chain_root`` is known to contain ``target``.

    Prefer ordinary ancestry when the target block is retained. If it is pruned
    below a trusted checkpoint-sync/finalized anchor, ancestry to that anchor is
    sufficient only when ``chain_state`` authenticates the target as its exact
    current-height target. An arbitrary unknown checkpoint below the anchor is
    not authenticated merely by its slot.
    """
    if target == Checkpoint():
        return False
    if target.root in store.blocks:
        return is_authenticated_checkpoint_on_chain(store, chain_root, target)
    if chain_state.current_height == height and chain_state.current_height_target == target:
        anchor = store.finalized_checkpoint
        return anchor.root in store.blocks and is_ancestor(
            store,
            ForkChoiceNode(root=chain_root, payload_status=PAYLOAD_STATUS_PENDING),
            ForkChoiceNode(root=anchor.root, payload_status=PAYLOAD_STATUS_PENDING),
        )
    return False
```

##### New `get_attestation_target`

```python
def get_attestation_target(
    store: Store,
    head_root: Root,
    head_state: BeaconState,
    safe_confirmed_root: Root,
    safe_confirmed_state: BeaconState,
    base_target: Checkpoint,
    voted_target_at: Dict[Height, Checkpoint],
    voted_timeout_at: Set[Height],
    voted_finality_at: Dict[Height, Checkpoint],
) -> Tuple[Checkpoint, Height]:
    """
    Return the target and height selected by the uniform gate.

    ``head_state`` is a branch-relative copy processed to the store's current
    slot. ``safe_confirmed_state`` is the stored post-block state of
    ``safe_confirmed_root``: its unprocessed state-height is the certificate
    input. A durable target is repeatable only when that exact authenticated
    checkpoint remains on both the current voted-head chain and the
    safe-confirmed chain; it need not equal this branch's current-height
    target. An incompatible E1 lock returns the empty vote. An unlocked
    incompatible target may bridge once to a confirmation-gated timeout, after
    which timeout-first discipline is durable.
    """
    current_height = head_state.current_height
    empty_vote = (Checkpoint(), Height(0))
    assert head_state.slot == get_current_slot(store)
    assert safe_confirmed_state.slot <= get_current_slot(store)
    safe_confirmed_height = safe_confirmed_state.current_height
    into_interval = safe_confirmed_height >= current_height

    target_case = is_attestation_target_on_chain(
        store,
        head_root,
        head_state,
        base_target,
        current_height,
    ) and is_attestation_target_on_chain(
        store,
        safe_confirmed_root,
        safe_confirmed_state,
        base_target,
        current_height,
    )

    locked = current_height in voted_finality_at
    timed_out = current_height in voted_timeout_at
    repeat_at_height = not (locked or timed_out) and current_height in voted_target_at

    # Timeout-first is durable: after signing a timeout at this height, never
    # cross back to a target even if another branch classifies the same numeric
    # height as ordinary. Repeating the timeout remains confirmation-gated.
    if timed_out:
        if into_interval and not locked:
            return Checkpoint(), current_height
        return empty_vote

    repeated_target = None
    if locked:
        repeated_target = voted_finality_at[current_height]
    elif repeat_at_height:
        repeated_target = voted_target_at[current_height]

    repeated_target_case = (
        repeated_target is not None
        and is_attestation_target_on_chain(
            store,
            head_root,
            head_state,
            repeated_target,
            current_height,
        )
        and is_attestation_target_on_chain(
            store,
            safe_confirmed_root,
            safe_confirmed_state,
            repeated_target,
            current_height,
        )
    )

    if head_state.current_height_nonjustifiable:
        # A compatible saved target remains safe to repeat. Because the state
        # class is latched timeout-only, it sets only the height-progress marker
        # and can never become this branch's justification.
        if repeated_target is not None and repeated_target_case:
            return repeated_target, current_height
        # An unlocked signer whose saved target is incompatible with the
        # selected chain may safely bridge to the timeout marker. Recording the
        # timeout permanently disables a later finality piggyback at this height.
        if into_interval and not locked:
            return Checkpoint(), current_height
        return empty_vote

    if repeated_target is not None:
        if repeated_target_case:
            return repeated_target, current_height
        # Old target histories on incompatible branches must not strand the
        # height. R1-to-R2 is not E1-slashable before a finality lock, and the
        # durable timeout record closes the later finality gate.
        if into_interval and not locked:
            return Checkpoint(), current_height
        return empty_vote
    if target_case:
        return base_target, current_height
    if into_interval:
        return Checkpoint(), current_height  # timeout vote
    return empty_vote
```

Apply the result to the data under construction:

```
attestation_data.target, attestation_data.height = get_attestation_target(
    store,
    head_root,
    head_state,
    safe_confirmed_root,
    safe_confirmed_state,
    base_target,
    voted_target_at,
    voted_timeout_at,
    voted_finality_at,
)
```

*Note*: If the validator cast an R1 at `current_height` and subsequently
observes a newer block, it is NOT safe to retarget — any later non-empty target
vote at the same height must re-emit the previously signed target. An unlocked
incompatible history may instead cross once to a confirmation-gated timeout; no
later target or finality piggyback is then allowed at that height.

*Note*: The retroactive finality lock matters when the view reverts to
`current_height` after the validator has already attached a finality commitment
at `finality_height = current_height` from a higher-height vote. Under E1, the
validator is bound to that finality target at `current_height`; voting a
different target (or a timeout, since `Checkpoint() ≠ T`) at `current_height`
would self-evidence E1. Re-emitting `T` is useful whenever that exact checkpoint
is authenticated on both the current voted-head chain and the safe-confirmed
chain. If either chain does not authenticate it, the empty vote is the only
admissible E1-safe fallback: it makes no height claim.

Votes with a stale height are still accepted by `process_attestation` (the
`finality_participation` update may still be useful), but they do not update
target tracking and earn no TIMELY_TARGET reward. A matching finality piggyback
may still earn TIMELY_FINALITY_TARGET.

#### Finality piggyback

The finality piggyback confirms a previously justified checkpoint. It is
selected together with the vote kind and target at the round-start event and is
valid when it points to a lower height than that selection state-height and the
local signing history shows that the validator previously voted for the same
target at that lower height and has not timed out there. It attaches
independently of the vote kind: target, timeout, and empty votes all carry it
when the gate passes. A later justification in the same round does not
retroactively modify the frozen piggyback. A piggyback matching the state
transition's current justified checkpoint and height earns the 14/64
TIMELY_FINALITY_TARGET participation weight. This reuses Altair's source-weight
position for the source-like act of identifying the checkpoint to finalize; the
vote's own valid current-height contribution separately earns the 26/64
TIMELY_TARGET weight.

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
or would create a slashable trace once paired with the timeout. A compatible
protected repeat keeps `Delta.tau` false so this gate can still pass later;
using the unlocked history bridge deliberately sets it and closes this gate.

#### Complete construction

The following helper combines the live head field with the round-start frozen FG
fields. It returns unsigned data, or `None` after durably recording a consumed
duty whose live head is incompatible with a frozen non-empty checkpoint. The
caller MUST persist that skip before returning and never retries it from a later
view. The remaining durable checks in [Signing](#signing) run before a signature
is released.

```python
def get_attestation_data(
    store: Store,
    frozen_round_vote_at: Dict[Round, FrozenRoundVote],
    skipped_attestation_at_round: Set[Round],
) -> Optional[AttestationData]:
    """Construct an attestation from the current round's frozen FG fields."""
    slot = get_current_slot(store)
    round = compute_round_at_slot(slot)
    if round in skipped_attestation_at_round:
        return None
    # Never reconstruct a missed round-start selection from a later view.
    assert round in frozen_round_vote_at
    frozen_vote = frozen_round_vote_at[round]

    head = get_head(store)
    # Under the common-prefix premise the live head remains above both frozen
    # checkpoints. Outside that premise, fail closed rather than sign a target
    # on another branch or alter any round-frozen FG field.
    head_state = get_current_slot_state(store, head.root)
    if frozen_vote.target != Checkpoint():
        target_on_head = is_attestation_target_on_chain(
            store,
            head.root,
            head_state,
            frozen_vote.target,
            frozen_vote.height,
        )
        if not target_on_head:
            skipped_attestation_at_round.add(round)
            return None
    if frozen_vote.finality_target != Checkpoint():
        finality_on_head = (
            is_authenticated_checkpoint_on_chain(
                store,
                head.root,
                frozen_vote.finality_target,
            )
            or (
                head_state.justified_height == frozen_vote.finality_height
                and head_state.justified_checkpoint == frozen_vote.finality_target
            )
            or (
                head_state.finalized_height == frozen_vote.finality_height
                and head_state.finalized_checkpoint == frozen_vote.finality_target
            )
        )
        if not finality_on_head:
            skipped_attestation_at_round.add(round)
            return None

    return AttestationData(
        slot=slot,
        beacon_block_root=head.root,
        target=frozen_vote.target,
        height=frozen_vote.height,
        finality_target=frozen_vote.finality_target,
        finality_height=frozen_vote.finality_height,
    )
```

*Note*: A finality attestation carries no payload-availability signal. It is an
LMD vote for a beacon block at `PAYLOAD_STATUS_PENDING`; the payload decision is
made by the available-attestation / Goldfish layer, not the finality vote.

### Signing

The attester domain is selected from the attestation slot. A Simplex target is
an exact block checkpoint and no longer carries an epoch.

Before calling the signing helper, the validator checks the complete data
against `signed_attestation_at_round` and atomically persists that entry plus
all three height-keyed history updates described in [Local state](#local-state).
If the round already contains different data, or persistence fails, it MUST NOT
sign. Re-requesting the identical message may return the previously stored
signature.

The signing boundary also enforces E2 independently of vote construction. For a
nonempty target, `voted_target_at[height]` must be absent or equal to that exact
checkpoint; a different prior nonempty target is a hard refusal. The history is
write-once and is persisted before releasing the signature:

```text
if attestation_data.target != Checkpoint():
    prior_target = voted_target_at.get(attestation_data.height)
    assert prior_target in (None, attestation_data.target)
    voted_target_at.setdefault(attestation_data.height, attestation_data.target)
```

```python
def get_attestation_signature(
    state: BeaconState, attestation_data: AttestationData, privkey: int
) -> BLSSignature:
    epoch = compute_epoch_at_slot(attestation_data.slot)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, epoch)
    signing_root = compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)
```

### Broadcast

Broadcast the signed attestation on the appropriate subnet. Aggregation follows
the same pattern as the base spec (aggregation selection via `is_aggregator`,
aggregate construction, timed broadcast at the aggregate deadline
(`AGGREGATE_DUE_BPS_GLOAS` of the slot)).

## Block proposal

The inherited block-proposal duties remain in force. The rules below populate
the Simplex operation lists in `BeaconBlockBody` and attach proposer-supplied
proofs to finality attestations. An honest proposer simulates
`process_operations` on a copy of the proposal state in body order and omits any
candidate that would fail; operation-pool validation does not replace this
sequential check.

### Fixed-quorum TSQ root syncing

Only the designated proposer for a round's first slot uses the round's
`candidate_root` when constructing a block. At the proposal-independent
round-boundary cutoff, before admitting any proposal for the round, fork choice
persists a `TSQSelection`. It pins the support round, the selection-state
Simplex root and viable candidate tree, the fixed validator weights and total
active balance, and `candidate_root`.

For total active balance `W`, let `q = ceil(2 * W / 3)`. Using the current
support-round view at that cutoff, `candidate_root` is the deepest candidate
whose subtree has support of at least `q`, or the pinned Simplex root when no
strict descendant qualifies. Each signer supplies at most one usable signed
head. A signer for which a distinct same-round message is known is excluded from
TSQ support; equivocation never supplies generic credit. TSQ has no lower
positive root-selection threshold.

The proposer uses the persisted `TSQSelection.candidate_root`; it does not
recompute the candidate after admitting the proposal or later messages. It
re-gossips every valid support-round head and every conflicting copy it knows,
including messages excluded from support, together with each named block and the
ancestry needed to validate and project the messages. The proposal carries no
TSQ root, quorum, signer list, vote bundle, or certificate.

The ordinary attestation pool retains those concrete signed messages through the
following action; `Store.round_attestations` alone is not a relay cache. This is
a required client obligation outside the executable fork-choice Store. Receivers
keep a message whose named block or ancestry is still unknown in the bounded
dependency queue specified by the P2P document and retry it when the dependency
arrives. Loss of this local material makes the round uncounted; a client never
reconstructs a signature from `AttestationData`.

The proposer also obtains its current `live_confirmed_head`. `get_proposer_head`
first finds the deeper of that root and `candidate_root` when they lie on one
chain. If the ordinary head descends from this TSQ base, it is preserved;
otherwise the TSQ base replaces the conflicting unconfirmed head. If the
required roots are incompatible, unknown, or too new, the helper returns the
ordinary head and the round is not counted for TSQ liveness. It does not
substitute a shallower TSQ candidate or install a synchronization root through
the proposal.

At the common action, each receiver independently computes the deepest fixed-`q`
lock from the same signed support-round data present in both its immutable
frozen view and its current view. Any signer known to have sent distinct round
messages in either view is excluded. The receiver uses the lock only if the
pinned selection is present and the lock remains viable and descends from the
action state's current Simplex root. If no strict descendant has enough shared
support, the lock is the pinned Simplex root. A missing selection or failed
action-state check instead uses the ordinary grade-1 fallback.

The TSQ lock, not the proposal, is the round's stable root. Exactly one timely
round-start proposal may receive proposal-specific treatment at the first-slot
available-vote action when it is viable and descends from both that stable root
and the receiver's action-time live available-confirmed head. It may replace an
ordinary unconfirmed head, which is necessary to align split Goldfish walks, but
cannot replace that action-time live prefix. This guard deliberately does not
use the historical non-retracting confirmation record, which could conflict
before healing. From the next slot onward it is an ordinary prior block and must
win through actual available votes. If the proposer publishes two distinct
round-start blocks, neither receives that treatment; the independently computed
TSQ lock is unchanged.

Messages learned after the action may update live fork-choice data or supply
equivocation evidence for other purposes, but they do not change the completed
TSQ view, selection, stable-root decision, or frozen round vote. The current
combined finality-attestation schedule and its relationship to the paper's
staged confirmation, SG, and FG actions are described in
[When to attest](#when-to-attest).

### Finality attestations and historical proofs

Finality-attestation selection and aggregation follow the inherited beacon
attestation rules, except that aggregates MUST have identical complete
`AttestationData` and Simplex proofs are attached after aggregation. Because
`historical_block_proofs` is not signed, a proposer MUST discard any supplied
list and construct the minimal list required by the proposal state:

1. Consider each distinct non-empty checkpoint in `data.target` and
   `data.finality_target`.
2. If `checkpoint.slot + SLOTS_PER_HISTORICAL_ROOT > state.slot`, attach no
   proof for that checkpoint; the state transition reads the live `block_roots`
   ring.
3. Otherwise, construct one `HistoricalBlockProof` containing the checkpoint
   slot/root branch and, except at `GENESIS_SLOT`, the branch for slot
   `checkpoint.slot - 1`. One proof serves both fields when their checkpoints
   are equal.
4. Call `verify_historical_block_proof(state, checkpoint, proof)` locally. If a
   required proof cannot be constructed or does not verify, omit the attestation
   rather than proposing an invalid block.

`historical_summaries` stores only vector commitments and cannot reconstruct a
Merkle branch. A proposer or its proof service therefore MUST retain, or be able
to reconstruct from canonical history, the full block-root vector and branch
material for every historical period containing such a target or its preceding
slot. This retention can end only after the checkpoint is no longer a repeatable
target or usable finality commitment and every already-signed current/previous-
epoch attestation naming it has left the inclusion window.

After attaching proofs, the proposer checks each candidate with
`validate_attestation` against the evolving proposal-state copy. If more valid
aggregates exist than `MAX_ATTESTATIONS_ELECTRA`, it SHOULD select
previous-epoch aggregates first because they expire first, then aggregates
covering the greatest not-yet-included effective balance; ties are broken by
`hash_tree_root(attestation)`.

### Available-attestation aggregates

The proposer collects valid `available_attestation` messages and retains them
through their current/previous-round inclusion window. As a one-time fork
transition rule, it MUST retain activation-slot contributions until the end of
the Simplex activation epoch: a matching aggregate is also the builder-payment
receipt for residual claims from the final Gloas epoch. It groups contributions
only when their complete `AvailableAttestationData` is equal. For each group it:

1. resolves committee positions using
   `get_available_committee(state, data.slot)`;
2. sets every position held by each included validator identity, including all
   duplicate seats;
3. aggregates exactly one signature per distinct validator identity; and
4. checks the result with `process_available_attestation` on the evolving
   proposal-state copy.

Bitwise-unioning overlapping aggregates and aggregating both signatures can add
the same validator signature twice while the verifier uses one public key. A
proposer MUST therefore merge only disjoint-identity aggregates, or rebuild the
aggregate from retained singleton contributions so each identity occurs once.
Different data roots are never merged. If more than `MAX_AVAILABLE_ATTESTATIONS`
valid aggregates remain, the proposer SHOULD select previous-round aggregates
first, then the aggregates adding the greatest number of not-yet-recorded
committee seats; ties are broken by `hash_tree_root(available_attestation)`.
Replays may be valid but SHOULD be omitted when they add no state participation.
During the activation epoch, a matching activation-slot aggregate that adds
transition-payment seats takes priority over aggregates with no financial
effect. The transition receipt is root-only: separately valid groups with the
same activation head but different `payload_present` values may both be
included, and their disjoint validator seats union for the financial quorum.
They are never BLS-aggregated across distinct complete data.

### Round-double-vote evidence

A node that receives two valid finality attestations from the same round with
different `AttestationData` retains the signed attestations, their resolved
indices, and the duty-state information needed to construct two valid
`IndexedAttestation`s. If the pair satisfies E1 or E2, it forms an
`AttesterSlashing`; otherwise it forms `RoundDoubleVoteEvidence` only when the
index sets overlap. Evidence has no age limit, so this material MUST be retained
until every overlapping validator is marked in `round_double_vote_penalized` in
a finalized state (or the evidence is otherwise durably available from finalized
history).

Before inclusion, the proposer calls `process_round_double_vote_evidence` on the
evolving proposal-state copy. This rejects malformed pairs and evidence that
would newly penalize nobody. At most `MAX_ROUND_DOUBLE_VOTE_EVIDENCE` items are
included. If several candidates compete for the limit, the proposer SHOULD
maximize the number of not-yet-penalized overlapping validators, then prefer the
earliest evidence round, with `hash_tree_root(evidence)` as the final
tiebreaker.

## Available attestation

Validators assigned to the available committee for slot `S` produce an
`AvailableAttestation` carrying the LMD head vote and payload availability
signal. This duty and the user-facing available confirmation are never gated by
finality-gadget or grade state.

### Constructing the data

At the duty action, set `head = get_head(store)` and preserve the complete
`ForkChoiceNode`, not only its root: the payload status is part of the vote when
the selected block predates the duty slot. In the first slot of a round this
call follows the proposal distinguished by the completed action, if any; in
later slots the proposal is selected only through actual previous-slot available
votes. Set `beacon_block_root = head.root` and `slot = S`. A same-slot block is
always `PAYLOAD_STATUS_PENDING` for this vote and MUST set
`payload_present = False`. For an older block, set `payload_present = True`
exactly when the selected node is `PAYLOAD_STATUS_FULL`; `EMPTY` and `PENDING`
map to `False`.

```python
def get_available_attestation_data(store: Store, head: ForkChoiceNode) -> AvailableAttestationData:
    """Construct the current slot's available-attestation data for ``head``."""
    slot = get_current_slot(store)
    block_slot = store.blocks[head.root].slot
    assert block_slot <= slot
    payload_present = block_slot < slot and head.payload_status == PAYLOAD_STATUS_FULL
    return AvailableAttestationData(
        slot=slot,
        payload_present=payload_present,
        beacon_block_root=head.root,
    )
```

The available committee is pinned at the slot boundary in
`store.available_committees[S]`. Before signing, process a copy of the selected
head state to `S` with `get_current_slot_state` and derive
`get_available_committee(head_state, S)`. The two committees MUST be identical,
the pinned entry MUST exist, and the validator MUST occur in them; otherwise it
fails closed and does not sign. Equality follows only under the research
profile's committee-convergence assumption. This check exposes a failed
assumption instead of producing a vote whose signer positions are interpreted
differently by honest nodes.

```
head = get_head(store)
head_state = get_current_slot_state(store, head.root)
committee = get_available_committee(head_state, S)
if S not in store.available_committees or committee != store.available_committees[S]:
    do_not_sign()
data = get_available_attestation_data(store, head)
```

### Committee positions and signing

A balance-weighted committee can contain the same validator more than once. A
validator signs `data` once and sets **every** position it occupies. Verifiers
deduplicate the resulting validator identities before fast aggregate signature
verification, while reward and Goldfish accounting credit every seat.

```python
def get_available_attestation_aggregation_bits(
    state: BeaconState, slot: Slot, validator_index: ValidatorIndex
) -> Bitvector[AVAILABLE_COMMITTEE_SIZE]:
    """Return all available-committee positions held by ``validator_index``."""
    committee = get_available_committee(state, slot)
    aggregation_bits = Bitvector[AVAILABLE_COMMITTEE_SIZE]()
    for position, index in enumerate(committee):
        if index == validator_index:
            aggregation_bits[position] = True
    assert any(aggregation_bits)
    return aggregation_bits
```

Before signing, look up `(validator_index, S)` in
`signed_available_attestation_at`. If it contains different data, or if the
durable write fails, the validator MUST NOT sign. Otherwise atomically persist
the data and release the signature; a retry of identical data returns the stored
signature. Sign only `AvailableAttestationData`, once, using the duty-slot epoch
and `DOMAIN_AVAILABLE_ATTESTER`:

```python
def get_available_attestation_signature(
    state: BeaconState, data: AvailableAttestationData, privkey: int
) -> BLSSignature:
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, compute_epoch_at_slot(data.slot))
    signing_root = compute_signing_root(data, domain)
    return bls.Sign(privkey, signing_root)
```

Construct
`AvailableAttestation(aggregation_bits=aggregation_bits, data=data, signature=signature)`
using the processed head state for both committee lookup and domain selection.

### Publication and aggregation

The available-confirmation deadline is **not** the signing deadline: construct
and publish the validator's own contribution on the global
`available_attestation` topic no later than `AVAILABLE_ATTESTATION_DUE_BPS` (25%
of the slot). `AVAILABLE_CONFIRMATION_DUE_BPS` (50%) is the receiver-side
inclusive cutoff. In the first slot of a round, the validator waits for the
common 25% action to call `freeze_stable_root` before constructing its available
vote; it does not sign from a pre-action walk. This boundary ordering matches
only the paper's block-to-available-vote-to-confirmation segment: a block
published at slot start is available for the vote action after one 25%-of-slot
delivery phase, and that vote is available for confirmation after a second such
phase. It does not supply the paper's later standalone SG and FG actions. At
exactly 50%, clients MUST process inbound messages timestamped at the cutoff
before running the confirmation freeze/tick; strictly later messages are not
timely. Aggregation MUST NOT delay this initial publication.

There is no selection-proof wrapper for this topic. Any node MAY combine
contributions with identical complete data by OR-ing their normalized committee
bits and BLS-aggregating exactly one signature per distinct validator identity,
then publish the resulting `AvailableAttestation`. Contributions with different
data are never combined. When source aggregates overlap in validator identity,
the aggregator MUST reconstruct from deduplicated singleton signatures or leave
them separate; blindly aggregating both signatures produces an invalid fast
aggregate. Gossip superset rules suppress redundant aggregates, while proposers
may continue collecting valid contributions for block inclusion as described
above.

## How to avoid slashing

### E1 and E2 avoidance

E1 says that if you sign `finality_target = T` at `finality_height = H`, then
any attestation you signed at `height = H` with `target != T` is slashable
evidence. **Timeout votes are slashable too**: `target = Checkpoint()` at
`height = H` conflicts with `finality_target = T ≠ Checkpoint()` at
`finality_height = H` (paper def:slashing). **Empty votes are safe for honest
validators**: with `height = Height(0)` they conflict only with a finality
commitment at `finality_height = Height(0)`, which no honest validator ever
signs — heights start at `GENESIS_HEIGHT == Height(1)`, so no honest finality
commitment at height `0` exists.

E2 says that signing two distinct nonempty `target` checkpoints at the same
`height` is slashable. A timeout is empty and therefore does not form E2
evidence with an earlier target; it can still form E1 evidence after a
same-height finality commitment.

**How to stay safe**: maintain write-once `voted_target_at[H]`,
`voted_timeout_at`, and `voted_finality_at[H]`. Use these (plus the retroactive
`voted_finality_at[H]` lock) to drive the vote choice at height `H`. Only set
`finality_target` when your prior target at the justified height matches the
justified checkpoint and no timeout was signed at that height (the rule in
[Finality piggyback](#finality-piggyback) above). The
[uniform gate](#the-uniform-gate) construction bakes in both disciplines: if
`voted_finality_at[current_height]` is set, the validator re-submits another R1
with the locked target only when that exact checkpoint is authenticated on both
the voted-head and safe-confirmed chains; otherwise it casts the empty vote. At
a nonjustifiable height, a compatible locked target may still be re-emitted as a
marker-only repeat; an incompatible lock casts the empty vote. If
`voted_timeout_at` is already set, it may only repeat the confirmation-gated
timeout or cast empty. If only `voted_target_at[current_height]` is set, the
protected repeat uses the same compatibility check and re-emits the previously
signed target. If it is incompatible, the validator may cross once to a
confirmation-gated timeout at either height class only while no finality lock
exists; before confirmation reaches the interval it casts empty. The atomic
signing-history update records that timeout before the signature is released.

*Note*: Signing an R2 (timeout) vote when no finality lock at `current_height`
exists is safe even after a prior R1 at the same height: E2 requires two
nonempty targets. The honest construction normally avoids this transition via
the protected repeat, to keep the finalize gate at that height open. Its one-way
liveness bridge applies when the saved target is incompatible with the selected
chain; using it deliberately closes the local finalize gate by recording the
timeout. The slashable case is when that timeout collides with a same-height
finality commitment, which the bridge forbids.

### Round double-vote

Signing two different `AttestationData` in the same round triggers
`RoundDoubleVoteEvidence`: forced exit plus a fixed deduction (not full
slashing) only when the pair satisfies neither E1 nor E2. An E1/E2 pair instead
uses `AttesterSlashing`. **Sign at most one distinct `AttestationData` per
round**, enforced across restarts by `signed_attestation_at_round`.

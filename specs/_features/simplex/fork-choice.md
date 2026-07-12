# Simplex Finality -- Fork Choice

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Configuration](#configuration)
- [Containers](#containers)
  - [New `RecordVote`](#new-recordvote)
  - [New `FrozenAvailableVotes`](#new-frozenavailablevotes)
  - [Modified `Store`](#modified-store)
  - [Modified `LatestMessage`](#modified-latestmessage)
- [Helper functions](#helper-functions)
  - [Modified `get_forkchoice_store`](#modified-get_forkchoice_store)
  - [New `update_justified`](#new-update_justified)
  - [New `get_viability_height_threshold`](#new-get_viability_height_threshold)
  - [New `is_viable_leaf`](#new-is_viable_leaf)
  - [New `is_viable`](#new-is_viable)
  - [Modified `filter_block_tree`](#modified-filter_block_tree)
  - [Modified `get_filtered_block_tree`](#modified-get_filtered_block_tree)
  - [New `is_in_filtered_block_tree`](#new-is_in_filtered_block_tree)
  - [New `update_finalized`](#new-update_finalized)
  - [New `has_unexpired_latest_message`](#new-has_unexpired_latest_message)
  - [New `get_view_freeze_due_ms`](#new-get_view_freeze_due_ms)
  - [New `is_before_view_freeze_deadline`](#new-is_before_view_freeze_deadline)
  - [New `get_available_confirmation_due_ms`](#new-get_available_confirmation_due_ms)
  - [New `is_before_available_confirmation_deadline`](#new-is_before_available_confirmation_deadline)
  - [New `is_before_attestation_deadline`](#new-is_before_attestation_deadline)
  - [New `is_ptc_decision_node`](#new-is_ptc_decision_node)
  - [Modified `get_supported_node`](#modified-get_supported_node)
  - [New `is_supporting_vote`](#new-is_supporting_vote)
  - [New `get_available_majority_threshold`](#new-get_available_majority_threshold)
  - [New `get_available_vote_payload_status`](#new-get_available_vote_payload_status)
  - [New `get_available_attestation_score`](#new-get_available_attestation_score)
  - [New `is_available_attestation_viable`](#new-is_available_attestation_viable)
  - [New `freeze_available_votes`](#new-freeze_available_votes)
  - [New `get_available_confirmation_score`](#new-get_available_confirmation_score)
  - [New `get_available_confirmation_majority_threshold`](#new-get_available_confirmation_majority_threshold)
  - [New `is_available_confirmation_viable`](#new-is_available_confirmation_viable)
  - [New `get_best_available_confirmation_child`](#new-get_best_available_confirmation_child)
  - [New `get_fast_confirmation_score`](#new-get_fast_confirmation_score)
  - [New `is_fast_confirmation_viable`](#new-is_fast_confirmation_viable)
  - [New `get_fast_confirmation_head`](#new-get_fast_confirmation_head)
  - [New `is_record_in_window`](#new-is_record_in_window)
  - [New `update_records`](#new-update_records)
  - [New `prune_records`](#new-prune_records)
  - [New `buffer_unknown_head_attestation`](#new-buffer_unknown_head_attestation)
  - [New `replay_unknown_head_attestations`](#new-replay_unknown_head_attestations)
  - [New `prune_unknown_head_attestations`](#new-prune_unknown_head_attestations)
  - [New `is_record_equivocator`](#new-is_record_equivocator)
  - [New `get_live_record_vote`](#new-get_live_record_vote)
  - [New `is_live_record_validator`](#new-is_live_record_validator)
  - [New `get_record_weight`](#new-get_record_weight)
  - [New `get_record_support`](#new-get_record_support)
  - [New `is_g0_clear`](#new-is_g0_clear)
  - [New `get_attestation_checkpoint_state`](#new-get_attestation_checkpoint_state)
  - [New `get_quorum_anchor`](#new-get_quorum_anchor)
  - [New `get_pointed_anchor`](#new-get_pointed_anchor)
  - [New `update_pointed_anchor`](#new-update_pointed_anchor)
  - [New `get_cascade_root`](#new-get_cascade_root)
  - [New `get_record_anchor`](#new-get_record_anchor)
  - [New `get_walk_anchor`](#new-get_walk_anchor)
  - [New `get_safe_confirmed_head`](#new-get_safe_confirmed_head)
  - [New `get_available_confirmation_head`](#new-get_available_confirmation_head)
  - [New `get_payload_participant_count`](#new-get_payload_participant_count)
  - [New `get_payload_full_support`](#new-get_payload_full_support)
  - [New `get_payload_data_available_support`](#new-get_payload_data_available_support)
  - [New `is_payload_verified`](#new-is_payload_verified)
  - [Modified `is_payload_timely`](#modified-is_payload_timely)
  - [Modified `is_payload_data_available`](#modified-is_payload_data_available)
  - [Modified `should_extend_payload`](#modified-should_extend_payload)
  - [Modified `should_build_on_full`](#modified-should_build_on_full)
  - [Modified `update_latest_messages`](#modified-update_latest_messages)
  - [Modified `get_attestation_score`](#modified-get_attestation_score)
  - [Modified `get_weight`](#modified-get_weight)
  - [Modified `get_head`](#modified-get_head)
  - [New `is_valid_from_block_attestation`](#new-is_valid_from_block_attestation)
  - [Modified `validate_on_attestation`](#modified-validate_on_attestation)
  - [New `validate_on_available_attestation`](#new-validate_on_available_attestation)
- [Handlers](#handlers)
  - [Modified `on_tick_per_slot`](#modified-on_tick_per_slot)
  - [Modified `on_block`](#modified-on_block)
  - [Modified `on_payload_attestation_message`](#modified-on_payload_attestation_message)
  - [Modified `on_attestation`](#modified-on_attestation)
  - [New `on_available_attestation`](#new-on_available_attestation)
- [Deprecated overrides](#deprecated-overrides)
  - [Modified `get_voting_source`](#modified-get_voting_source)
  - [Modified `update_unrealized_checkpoints`](#modified-update_unrealized_checkpoints)
  - [Modified `compute_pulled_up_tip`](#modified-compute_pulled_up_tip)
  - [Modified `record_block_timeliness`](#modified-record_block_timeliness)
  - [Modified `update_proposer_boost_root`](#modified-update_proposer_boost_root)
  - [Modified `is_head_late`](#modified-is_head_late)
  - [Modified `is_ffg_competitive`](#modified-is_ffg_competitive)
  - [Modified `is_head_weak`](#modified-is_head_weak)
  - [Modified `is_parent_strong`](#modified-is_parent_strong)
  - [Modified `should_apply_proposer_boost`](#modified-should_apply_proposer_boost)
  - [Modified `should_override_forkchoice_update`](#modified-should_override_forkchoice_update)
  - [Modified `get_proposer_head`](#modified-get_proposer_head)
  - [Modified `is_finalization_ok`](#modified-is_finalization_ok)
  - [Modified `validate_target_epoch_against_current_time`](#modified-validate_target_epoch_against_current_time)

<!-- mdformat-toc end -->

## Introduction

This is the fork choice specification for simplex-based finality. It modifies
the fork choice to use the justified and finalized checkpoints from the
height-filter-and-timeouts simplex finality gadget instead of Casper FFG, and
removes the unrealized justification/finalization machinery.

The fork choice operates in three layers. Layer 1 is the finality gadget: it
maintains `store.justified_checkpoint` (paper's `Σ.J`) as the lex-max over
justification cert events (paper's `updateJustified`), and advances
`store.finalized_checkpoint` via `update_finalized` when the incoming checkpoint
strictly extends the current finalized, descends from
`store.justified_checkpoint`, and is in the **viable subtree**. The store also
tracks `h_max` (the highest `state.current_height` ever observed) which drives
the **height filter**: only blocks whose state-height is at least `h_max - 1`
(or whose descendants reach that bound) are viable. Layer 2 is the record/anchor
layer: on-chain **records** — the head fields of finality attestations included
in blocks, latest-per-validator, in the record window, with same-round
equivocators excluded — and the **fresh quorums** built from previous-round
finality attestations, whose **anchor** (the highest common ancestor of the
quorum's head fields) a round-start proposal may point to. Layer 3 is the
Goldfish available-chain layer: per-slot available-committee attestations and
the availability confirmations derived from them.

The fork-choice head is computed by the **walk** (`get_head`), in three phases:
an **anchor** fixed round-atomically from finality attestations — the pointed
fresh quorum's anchor when the round-start proposal carries a valid one, else
the two-thirds record descent from the cascade root — then the **Goldfish**
descent from the anchor within the viable subtree, then the **viability
descent** down to the height frontier. Confirmation is likewise split in two:
the user-facing **available confirmation** (`store.latest_confirmed_head`),
never gated by finality-gadget or record state, and the internal **safe
confirmation** (`get_safe_confirmed_head`) — the deepest availability-confirmed
block that is G0-clear — which is what the finality-vote gates read.

The record thresholds are uniform at every height: the two-thirds record anchor
and the one-third conflict veto (`is_g0_clear`) are the only record-layer
fork-choice objects. Two framings coexist deliberately (paper Section: healing):
reaching a height `h + 2` from a justifiable height `h + 1` **unconditionally**
certifies — assuming only that less than a third of the stake is slashable —
that some honest validator safe-confirmed into `h + 1`'s interval, because
empty votes set no timeout marker and every other vote at a justifiable height
is confirmation-gated into the voted interval; **healing**
(canonical convergence and finality resumption) is conditional on at least
two-thirds of the stake being honest and online, and closes at one
honest-proposer inclusion of a pointed fresh quorum.

*Note*: This specification is built upon Gloas (EIP-7732 ePBS fork choice).

## Configuration

| Name                                      | Value                  | Description                                                                                                                                                                                                                                                                                            |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LATEST_MESSAGE_EXPIRY_SLOTS`             | `uint64(2**7)` (= 128) | Staleness bound on latest-message weight (`get_attestation_score`): a validator's `latest_message` is ignored once its slot is more than this many slots in the past. The walk itself reads records and available attestations, not latest messages.                                                   |
| `RECORD_WINDOW_SLOTS`                     | `uint64(2**7)` (= 128) | On-chain record window `W_R`: an included finality-attestation head record is in window while `record.slot + RECORD_WINDOW_SLOTS >= current_slot` (paper `v.slot >= s - W_R`); out-of-window records are ignored in both numerator and denominator of the record-support arithmetic.                   |
| `AVAILABLE_CONFIRMATION_DUE_BPS`          | `uint64(5000)`         | basis points; 50% of `SLOT_DURATION_MS`. Dual role: in-slot cutoff for an available vote to count as *timely*, and the time at which the previous slot's available-confirmation rule is run. Sits between the attestation deadline and the view-freeze deadline (propose / attest / confirm / freeze). |
| `FAST_CONFIRMATION_COMMITTEE_NUMERATOR`   | `uint64(3)`            | Numerator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                          |
| `FAST_CONFIRMATION_COMMITTEE_DENOMINATOR` | `uint64(4)`            | Denominator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                        |
| `VIEW_FREEZE_DUE_BPS`                     | `uint64(7500)`         | basis points; 75% of `SLOT_DURATION_MS`. In-slot vote-freeze boundary for view-merge: wire votes after this time are deferred to the next proposer's view.                                                                                                                                             |

## Containers

### New `RecordVote`

*Note*: An on-chain SG record: the head field (`head`) of a finality attestation
included in a block, tagged with the attestation's slot (`slot`). Records are
kept per validator per round within the record window; a validator's *live
record vote* is its latest in-window record across rounds. They drive the
record-support arithmetic (`get_record_support`, `is_g0_clear`) that the
record-anchor descent (`get_record_anchor`) and safe confirmation
(`get_safe_confirmed_head`) consume.

```python
@dataclass(eq=True, frozen=True)
class RecordVote:
    # [New in Simplex]
    slot: Slot
    head: Root
```

### New `FrozenAvailableVotes`

*Note*: The per-slot freeze of the time-shifted quorum (paper Definition:
Goldfish available chain and confirmation): the slot's available committee,
pinned to the justified-checkpoint state at freeze time, and the timely,
non-equivocating first votes as of the slot's `AVAILABLE_CONFIRMATION_DUE_BPS`
deadline. Captured once per slot by `freeze_available_votes`; the confirmation
rules read only this snapshot, so messages arriving after the freeze cannot
change a frozen slot's confirmation numerator or denominator. The freeze pins
exactly those two quantities: the confirmation-head walk that consumes them
(`get_available_confirmation_head`) reads the live block tree from the live
finalized root, so a mid-slot finality advance can move the walk's start.
Non-retraction claims scope to the frozen evaluation, not to the walk.

```python
@dataclass
class FrozenAvailableVotes:
    # [New in Simplex]
    committee: Sequence[ValidatorIndex]
    votes: Dict[ValidatorIndex, AvailableAttestationData]
```

### Modified `Store`

*Note*: `justified_checkpoint` (paper's `Σ.J`) and `justified_height` (paper's
`Σ.h_j`) jointly track the lex-max justification cert event ever observed, under
the lex key `(h_j, hash(J))`. `h_max` (paper's `Σ.h_max`) tracks the maximum
`state.current_height` over all known block states; it drives the height filter
/ viable subtree (paper Definition: viable subtree). `finalized_checkpoint` is
the paper's `Σ.F`. Downstream consumers (e.g., `get_record_weight`,
`get_attestation_score`) read
`store.block_states[store.justified_checkpoint.root]` as a weight-accounting
base state.

```python
@dataclass
class Store:
    time: uint64
    genesis_time: uint64
    # [Modified in Simplex]
    justified_checkpoint: Checkpoint  # paper's Σ.J
    # [New in Simplex]
    justified_height: Height  # paper's Σ.h_j
    finalized_checkpoint: Checkpoint
    # [New in Simplex]
    h_max: Height  # paper's Σ.h_max
    equivocating_indices: Set[ValidatorIndex]
    # [New in Simplex]
    # On-chain SG record layer: per validator, the latest included
    # finality-attestation head record of each round within the record window
    # (paper record vote), plus per-round equivocation marks (the second-latest
    # distinct record slot of a round holding two distinct records). Entries
    # and marks expire with the window, bounding both maps to the window's
    # rounds. The record state is a function of the accepted block set and the
    # current slot, independent of processing order. Seeded in
    # ``get_forkchoice_store``; fed by ``update_records`` (directly, or after
    # an unknown-head buffer-and-replay); expired entries dropped by
    # ``prune_records``.
    record_votes: Dict[ValidatorIndex, Dict[Round, RecordVote]]
    record_equivocations: Dict[ValidatorIndex, Dict[Round, Slot]]
    # [New in Simplex]
    # From-block finality attestations whose head field names a block not yet
    # in the store, buffered under that head root and replayed through
    # ``on_attestation`` when the head block arrives (``on_block``). The
    # buffer is what makes the record state order-independent across block
    # arrivals: an attestation-bearing block processed before its named head
    # contributes the same records as one processed after. Bounded by the
    # record window (``prune_unknown_head_attestations``).
    unknown_head_attestations: Dict[Root, Sequence[Attestation]]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    # [New in Simplex]
    # Last confirmed head (root, slot-confirmed-at) from the available-confirmation rule,
    # maintained by ``on_tick_per_slot`` at ``AVAILABLE_CONFIRMATION_DUE_BPS``.
    latest_confirmed_head: Tuple[Root, Slot] = (Root(), Slot(0))
    # [New in Simplex]
    # Immediate confirmed head (root, slot-confirmed-at) from the 75%-absolute
    # fast-confirmation rule, maintained by ``on_tick_per_slot`` at
    # ``AVAILABLE_CONFIRMATION_DUE_BPS``.
    fast_confirmed_head: Tuple[Root, Slot] = (Root(), Slot(0))
    # [New in Simplex]
    # The pointed anchor: the anchor of the fresh quorum the current round's
    # round-start proposal points to, set by ``update_pointed_anchor`` (first
    # verified reference of the round wins). Keyed by round: a stored anchor is
    # read only while ``pointed_anchor_round`` equals the current round, so it
    # expires at the round boundary — anchors carry no cross-round state.
    # ``Root()`` means no anchor has been adopted for ``pointed_anchor_round``.
    pointed_anchor_root: Root = Root()
    pointed_anchor_round: Round = Round(0)
    # Verified execution payload envelopes (gloas model); membership is the
    # local-availability signal consulted by the payload gates.
    payloads: Dict[Root, ExecutionPayloadEnvelope] = field(default_factory=dict)
    # [Modified in Simplex]
    # Per-slot PTC first-seen votes keyed by validator identity. Duplicate PTC
    # seats are counted at read time against the slot's PTC committee.
    payload_votes: Dict[Slot, Dict[ValidatorIndex, PayloadAttestationData]] = field(
        default_factory=dict
    )
    payload_vote_equivocations: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # Per-slot available-attestation tracking for Goldfish, keyed by validator
    # identity. Duplicate committee seats are counted at read time against the
    # slot's available committee.
    available_votes: Dict[Slot, Dict[ValidatorIndex, AvailableAttestationData]] = field(
        default_factory=dict
    )
    available_vote_equivocations: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    available_timely_attesters: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # Per-slot time-shifted-quorum freezes (committee and timely,
    # non-equivocating votes as of the slot's confirmation deadline), captured
    # by ``freeze_available_votes`` and read by the confirmation rules.
    frozen_available_votes: Dict[Slot, FrozenAvailableVotes] = field(default_factory=dict)
```

### Modified `LatestMessage`

*Note*: `payload_present` is removed. A stored latest message is a finality LMD
vote for a beacon block, which makes no payload decision; the payload status is
supplied explicitly at the support check (see `get_supported_node`). The
available / Goldfish layer builds its own transient `LatestMessage` and passes
the status it decides.

```python
@dataclass(eq=True, frozen=True)
class LatestMessage:
    # [Modified in Simplex]
    # Removed `payload_present`
    slot: Slot
    root: Root
```

## Helper functions

### Modified `get_forkchoice_store`

*Note*: The anchor is treated as a pre-justified, pre-finalized block at height
`0`, while its state-height is `GENESIS_HEIGHT == Height(1)`. Thus
`justified_height` is initialized to `Height(0)` and `h_max` to
`GENESIS_HEIGHT`, matching the paper convention that genesis is pre-justified at
height `0` and starts the finality gadget at state-height `1`. The empty
initialization of `payload_votes` means no initial strict-majority payload
support; the first post-anchor payload decision resolves through the tiebreak
path until PTC votes are recorded. `payloads` starts empty (matching gloas): the
anchor block's payload is implicitly treated as available because the payload
gates are only consulted post-anchor — they short-circuit to `False` for
non-anchor roots until an envelope arrives via `on_execution_payload_envelope`,
and the pre-finalized anchor never needs the timely/DA gate.
`latest_confirmed_head` and `fast_confirmed_head` are seeded to the anchor
`(root, slot)`.

```python
def get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock) -> Store:
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    justified_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    finalized_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    anchor_slot = anchor_state.slot
    return Store(
        time=uint64(anchor_state.genesis_time + SLOT_DURATION_MS * anchor_slot // 1000),
        genesis_time=anchor_state.genesis_time,
        justified_checkpoint=justified_checkpoint,
        # [New in Simplex]
        # Genesis is pre-justified at height 0
        justified_height=Height(0),
        finalized_checkpoint=finalized_checkpoint,
        # [New in Simplex]
        # Genesis state-height; bumped by on_block thereafter
        h_max=GENESIS_HEIGHT,
        equivocating_indices=set(),
        # [New in Simplex]
        record_votes={},
        record_equivocations={},
        unknown_head_attestations={},
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={},
        # [New in Simplex]
        latest_confirmed_head=(anchor_root, anchor_slot),
        # [New in Simplex]
        fast_confirmed_head=(anchor_root, anchor_slot),
        # [New in Simplex]
        # No pointed anchor yet (Root() marks the round anchorless).
        pointed_anchor_root=Root(),
        pointed_anchor_round=GENESIS_ROUND,
        # [Modified in Simplex]
        # gloas payloads model: starts empty; populated by on_execution_payload_envelope.
        payloads={},
        payload_votes={anchor_slot: {}},
        payload_vote_equivocations={anchor_slot: set()},
        available_votes={anchor_slot: {}},
        available_vote_equivocations={anchor_slot: set()},
        available_timely_attesters={anchor_slot: set()},
        # [New in Simplex]
        # No time-shifted-quorum freeze captured yet.
        frozen_available_votes={},
    )
```

### New `update_justified`

*Note*: Paper's `updateJustified`. For a justification cert event `(J', h')` —
accept the event only if the candidate descends from the current finalized
checkpoint (`F-filter`), then update
`(store.justified_checkpoint, store.justified_height)` iff the candidate's lex
key `(h', hash_tree_root(J'))` strictly exceeds the current store key
`(store.justified_height, hash_tree_root(store.justified_checkpoint))`. The
tiebreaker is `hash_tree_root(Checkpoint)` — deterministic and uniform across
clients — where the paper keys on the block hash. A letter deviation,
intended: the running max only needs some fixed injective key on candidates,
and the checkpoint's hash tree root is the canonical one here.

```python
def update_justified(
    store: Store, justified_checkpoint: Checkpoint, justified_height: Height
) -> None:
    """
    [New in Simplex] Paper's updateJustified. Filter candidates by F ⪯ J',
    then lex running-max on ``(h_j, hash_tree_root(J))``.
    """
    if justified_checkpoint == Checkpoint():
        return  # No justification yet (empty checkpoint)
    if justified_checkpoint.root not in store.blocks:
        return
    # F-filter: candidate must descend from (or equal) the current finalized checkpoint.
    if justified_checkpoint.root != store.finalized_checkpoint.root:
        if (
            get_ancestor(
                store,
                ForkChoiceNode(
                    root=justified_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING
                ),
                store.finalized_checkpoint.slot,
            ).root
            != store.finalized_checkpoint.root
        ):
            return
    new_key = (justified_height, hash_tree_root(justified_checkpoint))
    current_key = (store.justified_height, hash_tree_root(store.justified_checkpoint))
    if new_key > current_key:
        store.justified_checkpoint = justified_checkpoint
        store.justified_height = justified_height
```

### New `get_viability_height_threshold`

```python
def get_viability_height_threshold(store: Store) -> Height:
    """
    [New in Simplex] Return the minimum state-height required by the viable-tree
    height filter.
    """
    return Height(store.h_max - 1) if store.h_max > 0 else GENESIS_HEIGHT
```

### New `is_viable_leaf`

```python
def is_viable_leaf(store: Store, block_root: Root) -> bool:
    """
    [New in Simplex] A leaf of ``store.blocks`` is viable iff its state-height
    is at least ``store.h_max - 1`` (paper Definition: viable subtree).
    """
    block_state = store.block_states[block_root]
    return block_state.current_height >= get_viability_height_threshold(store)
```

### New `is_viable`

```python
def is_viable(store: Store, block_root: Root) -> bool:
    """
    [New in Simplex] A block is viable iff some leaf descendant of it in
    ``store.blocks`` is a viable leaf (paper Definition: viable subtree).
    """
    children = [r for r in store.blocks if store.blocks[r].parent_root == block_root]
    if not children:
        return is_viable_leaf(store, block_root)
    return any(is_viable(store, child) for child in children)
```

### Modified `filter_block_tree`

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    """
    [Modified in Simplex] Add ``block_root`` and its viable descendants to
    ``blocks`` iff the branch contains a leaf whose state-height satisfies the
    viable-tree height filter.
    """
    block = store.blocks[block_root]
    child_roots = [root for root in store.blocks if store.blocks[root].parent_root == block_root]
    child_results = [filter_block_tree(store, child, blocks) for child in child_roots]
    if any(child_results) or (len(child_roots) == 0 and is_viable_leaf(store, block_root)):
        blocks[block_root] = block
        return True
    return False
```

### Modified `get_filtered_block_tree`

```python
def get_filtered_block_tree(store: Store) -> Dict[Root, BeaconBlock]:
    """
    [Modified in Simplex] Retrieve the viable subtree rooted at the finalized
    checkpoint. The cascade may walk from either ``store.finalized_checkpoint``
    or ``store.justified_checkpoint``, so filtering from finalized keeps both
    possible roots available.
    """
    blocks: Dict[Root, BeaconBlock] = {}
    filter_block_tree(store, store.finalized_checkpoint.root, blocks)
    return blocks
```

### New `is_in_filtered_block_tree`

```python
def is_in_filtered_block_tree(
    store: Store, blocks: Dict[Root, BeaconBlock], node: ForkChoiceNode
) -> bool:
    """
    [New in Simplex] Return whether ``node`` is in the filtered viable subtree.
    Payload-decision nodes share their root with the parent block, so they are in
    the filtered tree only if the block itself satisfies the height bound or if
    that payload-status branch has a filtered child block.
    """
    if node.root not in blocks:
        return False
    if not is_ptc_decision_node(store, node):
        return True
    if store.block_states[node.root].current_height >= get_viability_height_threshold(store):
        return True
    return len(get_node_children(store, blocks, node)) > 0
```

### New `update_finalized`

*Note*: Paper's `updateFinalized`. Advance `store.finalized_checkpoint` only if
the candidate strictly extends the current finalized checkpoint, descends from
`store.justified_checkpoint` (paper thm:fleqr: `Σ.F ⪯ Σ.J`), AND is in the
viable subtree (paper viability guard, lem:viable-finalized).

```python
def update_finalized(store: Store, finalized_checkpoint: Checkpoint) -> None:
    """
    [New in Simplex] Advance Σ.F if candidate strictly extends Σ.F, descends
    from Σ.J, and lies in the viable subtree.
    """
    if finalized_checkpoint.slot <= store.finalized_checkpoint.slot:
        return
    if finalized_checkpoint.root not in store.blocks:
        return
    # F' must descend from the current finalized checkpoint.
    if (
        get_ancestor(
            store,
            ForkChoiceNode(root=finalized_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING),
            store.finalized_checkpoint.slot,
        ).root
        != store.finalized_checkpoint.root
    ):
        return
    # F' must be ancestor-or-self of Σ.J.
    if store.justified_checkpoint.root != finalized_checkpoint.root:
        if store.justified_checkpoint.root not in store.blocks:
            return
        if (
            get_ancestor(
                store,
                ForkChoiceNode(
                    root=store.justified_checkpoint.root,
                    payload_status=PAYLOAD_STATUS_PENDING,
                ),
                finalized_checkpoint.slot,
            ).root
            != finalized_checkpoint.root
        ):
            return
    # Viability guard: F' must be in the viable subtree.
    if not is_viable(store, finalized_checkpoint.root):
        return
    store.finalized_checkpoint = finalized_checkpoint
```

### New `has_unexpired_latest_message`

*Note*: Latest-message expiry. A validator's `latest_message` is counted by
`get_attestation_score` only while its slot is within
`LATEST_MESSAGE_EXPIRY_SLOTS` of the current slot, i.e. while
`message.slot > current_slot - LATEST_MESSAGE_EXPIRY_SLOTS`. The walk itself
reads on-chain records and available attestations, not latest messages; the
latest-message weight helpers are retained for auxiliary consumers only.

```python
def has_unexpired_latest_message(store: Store, index: ValidatorIndex) -> bool:
    """
    [New in Simplex] Whether ``index`` has a non-equivocating ``latest_message``
    whose slot is within ``LATEST_MESSAGE_EXPIRY_SLOTS`` of the current slot.
    """
    if index not in store.latest_messages:
        return False
    if index in store.equivocating_indices:
        return False
    # Unexpired iff message.slot > current_slot - LATEST_MESSAGE_EXPIRY_SLOTS,
    # written additively to avoid underflow at low slots.
    return store.latest_messages[index].slot + LATEST_MESSAGE_EXPIRY_SLOTS > get_current_slot(store)
```

### New `get_view_freeze_due_ms`

```python
def get_view_freeze_due_ms() -> uint64:
    """[New in Simplex] Return the in-slot vote-freeze boundary for view-merge."""
    return get_slot_component_duration_ms(VIEW_FREEZE_DUE_BPS)
```

### New `is_before_view_freeze_deadline`

```python
def is_before_view_freeze_deadline(store: Store) -> bool:
    """
    [New in Simplex] Return whether current local time is before the
    view-merge vote-freeze boundary.
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_view_freeze_due_ms()
```

### New `get_available_confirmation_due_ms`

```python
def get_available_confirmation_due_ms() -> uint64:
    """[New in Simplex] Return the in-slot timely cutoff for available-confirmation votes."""
    return get_slot_component_duration_ms(AVAILABLE_CONFIRMATION_DUE_BPS)
```

### New `is_before_available_confirmation_deadline`

```python
def is_before_available_confirmation_deadline(store: Store) -> bool:
    """
    [New in Simplex] Return whether current local time is before the
    available-confirmation timely cutoff.
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_available_confirmation_due_ms()
```

### New `is_before_attestation_deadline`

```python
def is_before_attestation_deadline(store: Store) -> bool:
    """[New in Simplex] Return whether current local time is before the attestation deadline."""
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_attestation_due_ms()
```

### New `is_ptc_decision_node`

```python
def is_ptc_decision_node(store: Store, node: ForkChoiceNode) -> bool:
    """Return whether ``node`` is a previous-slot payload decision (EMPTY/FULL)."""
    return node.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        node.root
    ].slot + 1 == get_current_slot(store)
```

### Modified `get_supported_node`

*Note*: `LatestMessage` no longer carries `payload_present`, so the supported
node's payload status is passed in explicitly. A finality vote makes no payload
decision — its caller passes `PAYLOAD_STATUS_PENDING` (it stabilizes the beacon
block and the payloads already in its chain, leaving the tip payload to the
available / Goldfish layer, which passes the status its vote decides).

```python
def get_supported_node(message: LatestMessage, payload_status: PayloadStatus) -> ForkChoiceNode:
    return ForkChoiceNode(root=message.root, payload_status=payload_status)
```

### New `is_supporting_vote`

*Note*: Gloas removed `is_supporting_vote`; simplex reintroduces it. The payload
status the vote supports is supplied by the caller (see `get_supported_node`).

```python
def is_supporting_vote(
    store: Store,
    node: ForkChoiceNode,
    message: LatestMessage,
    payload_status: PayloadStatus,
) -> bool:
    return is_ancestor(store, get_supported_node(message, payload_status), node)
```

### New `get_available_majority_threshold`

```python
def get_available_majority_threshold(store: Store) -> uint64:
    """
    Return the majority threshold for previous-slot available-attestation gating.
    A child's score must exceed this value to be viable.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT:
        return uint64(0)
    previous_slot = Slot(current_slot - 1)
    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot entry.
    if previous_slot not in store.available_votes:
        return uint64(0)
    previous_votes = store.available_votes[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity at read
    # time against the previous slot's available committee so the denominator is
    # seat-counted to match the seat-counted child score. The committee is
    # sampled from the justified-checkpoint state (the shared fork-choice weight
    # base). TODO(healing): the recording state and the evaluation state are
    # inconsistent — votes are recorded under the committee resolved on each
    # attestation's own head chain (on_available_attestation), but evaluated
    # here against a committee sampled from the justified-checkpoint state, and
    # on diverged branches the two (RANDAO-dependent) samplings can differ. The
    # justified-checkpoint state is the simplest faithful common base for
    # evaluation; maps to the paper's open items.
    base_state = store.block_states[store.justified_checkpoint.root]
    previous_committee = get_available_committee(base_state, previous_slot)
    participant_count = uint64(
        len([index for index in previous_committee if index in previous_votes])
    )
    return participant_count // 2
```

### New `get_available_vote_payload_status`

*Note*: The payload status an available vote supports, mirroring the base
`get_supported_node` derivation: a vote whose block precedes its slot decides
`FULL`/`EMPTY` by `payload_present`; a same-slot vote is `PENDING`.

```python
def get_available_vote_payload_status(
    store: Store, data: AvailableAttestationData
) -> PayloadStatus:
    block = store.blocks[data.beacon_block_root]
    if block.slot < data.slot:
        if data.payload_present:
            return PAYLOAD_STATUS_FULL
        return PAYLOAD_STATUS_EMPTY
    return PAYLOAD_STATUS_PENDING
```

### New `get_available_attestation_score`

```python
def get_available_attestation_score(store: Store, child: ForkChoiceNode) -> uint64:
    """
    Return the available-attestation score for ``child``: non-equivocating
    supporting votes plus total equivocations from the previous slot's
    available committee.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, child):
        return uint64(0)

    previous_slot = Slot(current_slot - 1)
    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot entry.
    if previous_slot not in store.available_votes:
        return uint64(0)
    previous_votes = store.available_votes[previous_slot]
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; iterate the previous slot's
    # available committee to resolve seat multiplicity (a validator holding k
    # seats contributes k, and all k seats are excluded when it equivocates).
    base_state = store.block_states[store.justified_checkpoint.root]
    previous_committee = get_available_committee(base_state, previous_slot)
    score = uint64(0)
    for member_index in previous_committee:
        if member_index in previous_equivocations:
            score += 1  # Equivocator counted for viability
            continue
        if member_index not in previous_votes:
            continue
        vote = previous_votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, child, message, payload_status):
            score += 1
    return score
```

### New `is_available_attestation_viable`

```python
def is_available_attestation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in Layer 3 Goldfish walk: PTC decision
    nodes and current-slot proposals always pass through; other children require
    available-attestation score exceeding the majority threshold.
    """
    if is_ptc_decision_node(store, child):
        return True
    if store.blocks[child.root].slot == get_current_slot(store):
        return True
    return get_available_attestation_score(store, child) > get_available_majority_threshold(store)
```

### New `freeze_available_votes`

*Note*: The per-slot freeze of the time-shifted quorum. Everything the
available-confirmation evaluation reads is captured here at the slot's
`AVAILABLE_CONFIRMATION_DUE_BPS` deadline: the timely attester set, the
equivocation exclusions, the first votes, and the committee derivation (pinned
to the justified-checkpoint state at freeze time). A message arriving after the
freeze — a straggler vote, an equivocation report (wire or from-block), or a
justification changing the committee base state — cannot change an
already-frozen slot's confirmation numerator or denominator; it can of course
affect later slots. A slot's confirmation *evaluation* — its numerator and
denominator — is therefore non-retracting and independent of post-deadline
message timing (paper Definition: Goldfish available chain and confirmation,
structural confirmation consistency). The confirmation-head walk that consumes
the frozen scores starts from the live finalized root over the live block tree
(`get_available_confirmation_head`), so it is not covered by the freeze.

```python
def freeze_available_votes(store: Store, slot: Slot) -> None:
    """
    [New in Simplex] Capture the time-shifted-quorum freeze for ``slot``: the
    slot's available committee and its timely, non-equivocating first votes as
    of the freeze. Idempotent: the first capture wins.
    """
    if slot in store.frozen_available_votes:
        return
    base_state = store.block_states[store.justified_checkpoint.root]
    committee = get_available_committee(base_state, slot)
    votes = store.available_votes.get(slot, {})
    equivocations = store.available_vote_equivocations.get(slot, set())
    timely_attesters = store.available_timely_attesters.get(slot, set())
    frozen_votes = {
        index: votes[index]
        for index in votes
        if index in timely_attesters and index not in equivocations
    }
    store.frozen_available_votes[slot] = FrozenAvailableVotes(
        committee=committee, votes=frozen_votes
    )
```

### New `get_available_confirmation_score`

*Note*: `store.frozen_available_votes[slot]` is the per-slot time-shifted-quorum
freeze for available-confirmation votes, capturing the timely, non-equivocating
votes and the committee as of the slot's `AVAILABLE_CONFIRMATION_DUE_BPS`
deadline (see `freeze_available_votes`). The available-confirmation rule reads
the previous slot's freeze, while fast confirmation reads the current slot's
freeze immediately. Stragglers arriving after the deadline are not in the freeze
and are never counted — by design, not by omission: the freeze is what makes all
honest validators evaluate the same quorum. The freeze covers this numerator
and its matching denominator (`get_available_confirmation_majority_threshold`);
the walk consuming the scores reads the live tree from the live finalized root
(`get_available_confirmation_head`).

```python
def get_available_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    Return delayed available-confirmation support for ``node`` from the
    previous slot's freeze: timely, non-equivocating available attesters as of
    that slot's confirmation deadline.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    previous_slot = Slot(current_slot - 1)
    # The freeze is captured per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot freeze.
    if previous_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; iterate the frozen committee to
    # resolve seat multiplicity. Timeliness and equivocation exclusions are
    # baked into the frozen votes, so a post-deadline equivocation report
    # cannot retro-mutate the score.
    count = uint64(0)
    for member_index in freeze.committee:
        if member_index not in freeze.votes:
            continue
        vote = freeze.votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, node, message, payload_status):
            count += 1
    return count
```

### New `get_available_confirmation_majority_threshold`

*Note*: The available-confirmation relative quorum must freeze BOTH the
numerator and the denominator over the same time-shifted-quorum set, or the
confirmation outcome would depend on straggler timing and from-block inclusions
and could differ across honest views. This denominator therefore counts the
previous slot's *frozen* electorate — the timely, non-equivocating available
attesters captured by `freeze_available_votes` — exactly matching the
numerator's electorate in `get_available_confirmation_score`. It is distinct
from `get_available_majority_threshold` (the all-votes threshold gating the
Goldfish head), whose base-branch semantics is unchanged.

```python
def get_available_confirmation_majority_threshold(store: Store) -> uint64:
    """
    [New in Simplex] Return the relative-majority threshold for delayed
    available confirmation over the frozen electorate: the previous slot's
    timely, non-equivocating available attesters (seat-counted). Numerator and
    this denominator read the same freeze, so confirmation is
    straggler-independent.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT:
        return uint64(0)
    previous_slot = Slot(current_slot - 1)
    # The freeze is captured per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot freeze.
    if previous_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[previous_slot]
    participant_count = uint64(0)
    for member_index in freeze.committee:
        if member_index in freeze.votes:
            participant_count += 1
    return participant_count // 2
```

### New `is_available_confirmation_viable`

```python
def is_available_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in delayed available confirmation:
    PTC decision nodes always pass through; other children require
    available-confirmation score exceeding the frozen-electorate majority
    threshold.
    """
    if is_ptc_decision_node(store, child):
        return True
    # [Modified in Simplex]
    # Numerator and denominator both read the same per-slot freeze (timely,
    # non-equivocating, committee pinned at the freeze), so a slot's
    # confirmation does not depend on straggler timing.
    return get_available_confirmation_score(
        store, child
    ) > get_available_confirmation_majority_threshold(store)
```

### New `get_best_available_confirmation_child`

```python
def get_best_available_confirmation_child(
    store: Store,
    blocks: Dict[Root, BeaconBlock],
    head: ForkChoiceNode,
) -> Optional[ForkChoiceNode]:
    """
    [New in Simplex] Return the best child for delayed available confirmation.
    ``blocks`` is the unfiltered accepted block tree: user-facing confirmation
    applies no viability filter.
    """
    children = [
        child
        for child in get_node_children(store, blocks, head)
        if is_available_confirmation_viable(store, child)
    ]
    if len(children) == 0:
        return None
    return max(
        children,
        key=lambda child: (
            get_available_confirmation_score(store, child),
            child.root,
            get_payload_status_tiebreaker(store, child),
        ),
    )
```

### New `get_fast_confirmation_score`

```python
def get_fast_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    [New in Simplex] Return immediate fast-confirmation support for ``node``
    from the current slot's freeze: timely, non-equivocating available
    attesters as of the slot's confirmation deadline.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    # The freeze is captured per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no current-slot freeze.
    if current_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[current_slot]
    # [New in Simplex]
    # Votes are keyed by validator identity; iterate the frozen committee to
    # resolve seat multiplicity. Timeliness and equivocation exclusions are
    # baked into the frozen votes.
    count = uint64(0)
    for member_index in freeze.committee:
        if member_index not in freeze.votes:
            continue
        vote = freeze.votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, node, message, payload_status):
            count += 1
    return count
```

### New `is_fast_confirmation_viable`

```python
def is_fast_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    [New in Simplex] Return whether ``child`` is viable in immediate fast
    confirmation: PTC decision nodes always pass through; other children require
    an absolute 75% of ``AVAILABLE_COMMITTEE_SIZE`` seats.
    """
    if is_ptc_decision_node(store, child):
        return True
    return (
        get_fast_confirmation_score(store, child) * FAST_CONFIRMATION_COMMITTEE_DENOMINATOR
        >= AVAILABLE_COMMITTEE_SIZE * FAST_CONFIRMATION_COMMITTEE_NUMERATOR
    )
```

### New `get_fast_confirmation_head`

```python
def get_fast_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    [New in Simplex] Return the immediate fast-confirmation head for the
    current slot, from the current slot's frozen available votes and an
    absolute 75% committee-seat threshold, over all accepted descendants of
    the finalized root.
    """
    # User-facing confirmation is floorless and never gated by finality-gadget
    # or record state: it walks the unfiltered accepted block tree from the
    # finalized root, with no height-filter viability bound.
    blocks = store.blocks
    head = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )

    # Fast confirmation. Among fast-viable children pick by confirmation
    # score, then root, then payload-status tiebreaker. At the 75%-absolute
    # threshold, at most one block child can cross.
    while True:
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_fast_confirmation_viable(store, child)
        ]
        if len(children) == 0:
            return head
        head = max(
            children,
            key=lambda child: (
                get_fast_confirmation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
```

### New `is_record_in_window`

```python
def is_record_in_window(store: Store, slot: Slot) -> bool:
    """
    [New in Simplex] Return whether an attestation slot is inside the record
    window: ``slot + RECORD_WINDOW_SLOTS >= current_slot`` (paper
    ``v.slot >= s - W_R``), written additively to avoid underflow at low slots.
    """
    return slot + RECORD_WINDOW_SLOTS >= get_current_slot(store)
```

### New `update_records`

*Note*: The on-chain SG record layer is fed only from finality attestations
included in blocks (`on_attestation` with `is_from_block=True`); the record head
is the attestation's `beacon_block_root`, so the head field of timeout votes and
empty votes is recorded too. Records are stored per validator per round: an
incoming record compares against its own round's stored entry, and any two
same-round records with distinct `(slot, head)` content are a record
equivocation. Equivocation is thus *content-based*: a same-round pair with the
same slot and the same head is one record, not an equivocation — deliberately
narrower than the paper's rule (Definition: records), under which any two
same-round attestations equivocate. A letter deviation, intended: records are
content-counted, so a pair identical in `(slot, head)` cannot distort record
support, and content-counting is also what makes re-inclusion of the same
attestation on another branch harmless. The equivocation mark is per round and
holds the second-latest distinct record slot of that round, so the validator is
excluded (from both the numerator and the denominator of the record-support
arithmetic) exactly while two of the round's records are jointly in window —
never permanently. Acceptance is gated
by the record window only, never by wall-clock epochs, and late-beyond-window
acceptance is a no-op, not a divergence. An included attestation whose head
field names a block not yet in the store is buffered and replayed through the
same attribution path when the head arrives
(`buffer_unknown_head_attestation` / `replay_unknown_head_attestations`), so
the property holds across head-arrival orderings too. The record state is
therefore a function of the accepted block set and the current slot,
independent of processing order: a syncing node reconstructs exactly the
record state of a node that processed the same blocks live (paper Definition:
records — "records are a deterministic function of `Σ.T`"). The records feed the record-anchor descent
(`get_record_anchor`) and G0-clearance (`is_g0_clear`, read by safe
confirmation).

```python
def update_records(
    store: Store, attestation: Attestation, attesting_indices: Sequence[ValidatorIndex]
) -> None:
    """
    [New in Simplex] Feed on-chain SG records from finality attestations included
    in blocks. The record head is ``attestation.data.beacon_block_root``.
    """
    data = attestation.data
    head = data.beacon_block_root
    assert head in store.blocks
    slot = data.slot
    # Window-gated acceptance: a record beyond the window is a no-op (its
    # window-gated reads would ignore it regardless), so nodes processing the
    # same blocks at different times hold the same observable record state.
    if not is_record_in_window(store, slot):
        return
    record = RecordVote(slot=slot, head=head)
    record_round = compute_round_at_slot(slot)

    for index in attesting_indices:
        if index not in store.record_votes:
            store.record_votes[index] = {}
        round_votes = store.record_votes[index]
        if record_round not in round_votes:
            round_votes[record_round] = record
            continue
        existing = round_votes[record_round]
        # Records are content-counted: a re-inclusion of the same record (on
        # any branch) is a no-op, not an equivocation.
        if existing == record:
            continue
        # Two distinct same-round records are an equivocation regardless of
        # their heads. Keep the round's latest record and mark the round with
        # the second-latest distinct slot. The same-slot tiebreak (by head
        # root) is unobservable: a same-slot pair keeps the round marked for
        # as long as either entry is in window.
        latest = max(existing, record, key=lambda record_vote: (record_vote.slot, record_vote.head))
        earlier = min(
            existing, record, key=lambda record_vote: (record_vote.slot, record_vote.head)
        )
        round_votes[record_round] = latest
        if index not in store.record_equivocations:
            store.record_equivocations[index] = {}
        marks = store.record_equivocations[index]
        if record_round in marks:
            marks[record_round] = max(marks[record_round], earlier.slot)
        else:
            marks[record_round] = earlier.slot
```

### New `prune_records`

```python
def prune_records(store: Store) -> None:
    """
    [New in Simplex] Drop record entries and equivocation marks whose slot has
    left the record window, bounding the record maps to the window's rounds.
    Pruning is transparent: every read is already window-gated, so the
    observable record state is unchanged.
    """
    for index in list(store.record_votes.keys()):
        round_votes = store.record_votes[index]
        for record_round in list(round_votes.keys()):
            if not is_record_in_window(store, round_votes[record_round].slot):
                del round_votes[record_round]
        if len(round_votes) == 0:
            del store.record_votes[index]
    for index in list(store.record_equivocations.keys()):
        marks = store.record_equivocations[index]
        for record_round in list(marks.keys()):
            if not is_record_in_window(store, marks[record_round]):
                del marks[record_round]
        if len(marks) == 0:
            del store.record_equivocations[index]
```

### New `buffer_unknown_head_attestation`

*Note*: Record attribution needs the attestation's head-chain checkpoint state
(`get_attestation_checkpoint_state`), which does not exist while the named
head block is unknown — so an unknown-head from-block attestation cannot be
recorded on the spot, and dropping it would make the record state depend on
whether the attestation-bearing block arrived before or after its named head.
Instead it is buffered here, keyed by the head root, and replayed by
`replay_unknown_head_attestations` when the head block arrives. Buffering is
window-gated like every record write (a replay of an out-of-window attestation
would be a no-op), duplicates are stored once, and expired entries are dropped
by `prune_unknown_head_attestations`, so the buffer is bounded by the record
window.

```python
def buffer_unknown_head_attestation(store: Store, attestation: Attestation) -> None:
    """
    [New in Simplex] Buffer a from-block finality attestation whose head field
    names a block not yet in the store, for replay when the head block arrives.
    """
    data = attestation.data
    if not is_record_in_window(store, data.slot):
        return
    buffered = store.unknown_head_attestations.get(data.beacon_block_root, [])
    if attestation in buffered:
        return
    store.unknown_head_attestations[data.beacon_block_root] = list(buffered) + [attestation]
```

### New `replay_unknown_head_attestations`

*Note*: Called by `on_block` after the block has been added to the store, so
the head-chain checkpoint state that record attribution needs is available.
Replay goes through `on_attestation(..., is_from_block=True)` — the same
attribution path as a from-block attestation whose head was already known —
so buffered-then-replayed records land in the same per-round buckets, with the
same content-counting and equivocation semantics, as live-processed ones. With
the buffer, the record state is a function of the accepted block set and the
current slot, independent of the order in which the blocks arrived: a node
that processes an attestation-bearing block before its named head reconstructs
exactly the record state of a node that processed them in the other order.

```python
def replay_unknown_head_attestations(store: Store, block_root: Root) -> None:
    """
    [New in Simplex] Replay the from-block finality attestations buffered
    against ``block_root``, now that the block is in the store.
    """
    if block_root not in store.unknown_head_attestations:
        return
    buffered = store.unknown_head_attestations.pop(block_root)
    for attestation in buffered:
        on_attestation(store, attestation, is_from_block=True)
```

### New `prune_unknown_head_attestations`

```python
def prune_unknown_head_attestations(store: Store) -> None:
    """
    [New in Simplex] Drop buffered unknown-head attestations whose slot has
    left the record window, bounding the buffer to the window. Pruning is
    transparent: a replay of an out-of-window attestation would be a no-op.
    """
    for head_root in list(store.unknown_head_attestations.keys()):
        buffered = [
            attestation
            for attestation in store.unknown_head_attestations[head_root]
            if is_record_in_window(store, attestation.data.slot)
        ]
        if len(buffered) == 0:
            del store.unknown_head_attestations[head_root]
        else:
            store.unknown_head_attestations[head_root] = buffered
```

### New `is_record_equivocator`

```python
def is_record_equivocator(store: Store, index: ValidatorIndex) -> bool:
    """
    [New in Simplex] Return whether ``index`` is an on-chain record equivocator:
    some round holds two distinct records that are both still in the record
    window. A round's mark stores the second-latest distinct record slot of
    that round, so the mark is live exactly while such a pair exists (the
    round's latest record has a slot at least the mark's). Equivocator status
    is window-scoped, not permanent: newer rounds count again once the
    offending round's pair leaves the window.
    """
    marks = store.record_equivocations.get(index, {})
    return any(is_record_in_window(store, mark_slot) for mark_slot in marks.values())
```

### New `get_live_record_vote`

```python
def get_live_record_vote(store: Store, index: ValidatorIndex) -> Optional[RecordVote]:
    """
    [New in Simplex] Return ``index``'s live record vote — the latest in-window
    record across rounds — or ``None`` if the validator has no in-window record
    or is a record equivocator (equivocators hold no live record vote).
    """
    if is_record_equivocator(store, index):
        return None
    round_votes = store.record_votes.get(index, {})
    in_window_records = [
        record
        for record in round_votes.values()
        if is_record_in_window(store, record.slot) and record.head in store.blocks
    ]
    if len(in_window_records) == 0:
        return None
    return max(in_window_records, key=lambda record: (record.slot, record.head))
```

### New `is_live_record_validator`

```python
def is_live_record_validator(store: Store, index: ValidatorIndex) -> bool:
    """
    [New in Simplex] Return whether ``index`` has a non-equivocating, in-window
    on-chain record vote for a known block.
    """
    return get_live_record_vote(store, index) is not None
```

### New `get_record_weight`

*Note*: The record electorate — here (the denominator `D`) and in
`get_record_support` (the numerator `R`) — excludes slashed and inactive
validators, matching the spec-wide tally convention (e.g.
`compute_best_justification_target`, `get_pointed_anchor`). This is a letter
deviation from the paper's Definition: record weight, whose electorate is
purely possession-based (every non-equivocating holder of a live record vote);
it shifts `D` and `R` by the slashed/inactive holders. Intended: a slashed
validator must not keep steering the record thresholds, and both sides of
every record inequality read the same electorate, so the thresholds stay
consistent.

```python
def get_record_weight(store: Store) -> Gwei:
    """
    [New in Simplex] Return ``D``: total effective balance of active, unslashed
    validators with a live on-chain record vote, weighed from the justified
    checkpoint base state.
    """
    state = store.block_states[store.justified_checkpoint.root]
    live_indices = {
        index
        for index in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[index].slashed and is_live_record_validator(store, index)
    }
    return get_total_balance(state, live_indices)
```

### New `get_record_support`

```python
def get_record_support(store: Store, node: ForkChoiceNode) -> Gwei:
    """
    [New in Simplex] Return ``R(node)``: total effective balance of live record
    voters whose recorded head descends from ``node``.
    """
    state = store.block_states[store.justified_checkpoint.root]
    unslashed_and_active_indices = [
        index
        for index in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[index].slashed
    ]
    support = Gwei(0)
    for index in unslashed_and_active_indices:
        record = get_live_record_vote(store, index)
        if record is None:
            continue
        message = LatestMessage(slot=record.slot, root=record.head)
        if is_supporting_vote(store, node, message, PAYLOAD_STATUS_PENDING):
            support += state.validators[index].effective_balance
    return support
```

### New `is_g0_clear`

```python
def is_g0_clear(store: Store, target_root: Root) -> bool:
    """
    [New in Simplex] Return whether no block conflicting with ``target_root``
    has at least one-third record support (paper grade ``G0``).
    """
    target = ForkChoiceNode(root=target_root, payload_status=PAYLOAD_STATUS_PENDING)
    record_weight = get_record_weight(store)
    # With zero live record weight the bound degenerates to >= 0, so any
    # conflicting block (support 0) trips it and only conflict-free blocks are
    # G0-clear. This is the paper inequality read conservatively; live records
    # reappear with the first in-window inclusion.
    for root in store.blocks:
        node = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
        conflicts = not is_ancestor(store, node, target) and not is_ancestor(store, target, node)
        if conflicts and get_record_support(store, node) * 3 >= record_weight:
            return False
    return True
```

### New `get_attestation_checkpoint_state`

*Note*: Committee membership and signing domains are epoch-based, so an
attestation is verified against the epoch-boundary state on its own head chain.
This helper caches that state; it is shared by `on_attestation` (wire votes) and
`get_pointed_anchor` (fresh-quorum references).

```python
def get_attestation_checkpoint_state(store: Store, data: AttestationData) -> BeaconState:
    """
    [New in Simplex] Return (and cache) the state used to resolve committees and
    verify signatures for an attestation: the state at the epoch boundary of
    ``data.slot`` on the chain of ``data.beacon_block_root``.
    """
    attestation_epoch = compute_epoch_at_slot(data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    epoch_root = get_ancestor(
        store,
        ForkChoiceNode(root=data.beacon_block_root, payload_status=PAYLOAD_STATUS_PENDING),
        epoch_boundary_slot,
    ).root
    checkpoint = Checkpoint(slot=epoch_boundary_slot, root=epoch_root)
    if checkpoint not in store.checkpoint_states:
        base_state = copy(store.block_states[epoch_root])
        if base_state.slot < epoch_boundary_slot:
            process_slots(base_state, epoch_boundary_slot)
        store.checkpoint_states[checkpoint] = base_state
    return store.checkpoint_states[checkpoint]
```

### New `get_quorum_anchor`

*Note*: Paper Definition: round-r quorum, fresh quorum, and anchor. The anchor
of a quorum is the *highest common ancestor* of the head fields of its
attestations: the deepest block whose subtree contains every one of them. It is
a pure function of the referenced head fields, not a free choice; every head
root must be a known block.

```python
def get_quorum_anchor(store: Store, head_roots: Sequence[Root]) -> Root:
    """
    [New in Simplex] Return the highest common ancestor of ``head_roots``: the
    deepest block whose subtree contains every one of them.
    """
    anchor = head_roots[0]
    for head_root in head_roots[1:]:
        head_node = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
        # Walk the anchor candidate up until it is an ancestor of this head too.
        # The walk terminates: the finalized root is an ancestor of every known
        # block. Pairwise folding yields the common anchor of the whole set.
        while not is_ancestor(
            store, head_node, ForkChoiceNode(root=anchor, payload_status=PAYLOAD_STATUS_PENDING)
        ):
            anchor = store.blocks[anchor].parent_root
    return anchor
```

### New `get_pointed_anchor`

*Note*: Verification of a fresh-quorum reference (paper "Pointing", Definition:
round-r quorum, fresh quorum, and anchor). The reference is `body.anchor_quorum`
of a round-start proposal: previous-round finality attestations, as standard
aggregates. Verification reads only the aggregates and objective chain data —
aggregate signatures over distinct previous-round signers, effective balances
summing to at least two-thirds of the **total** active balance (an absolute
threshold, not a fraction of the live record weight), and known head fields
whose highest common ancestor is the anchor. It does **not** require the
attestations to be included as records: the reference is a threshold certificate
that creates no records (paper lem:anchor-support). Distinct signers are counted
once across aggregates (union weight), so duplicates never double-count; a
signer contributing two different head fields within the reference only makes
the anchor shallower, and its same-round record equivocation is the record
layer's separate concern. Accepting such in-reference duplicate signers is a
letter deviation from the paper's Definition: round-r quorum, fresh quorum, and
anchor (which draws the quorum from distinct validators), and it is
safety-preserving: duplicates add no weight and can only make the anchor
shallower. The threshold's denominator is `get_total_active_balance` of the
proposal's own state — a single reference every verifier of this proposal
shares; the paper reads it as that round's total stake. A letter deviation,
intended: verification needs a denominator common across views, and the
proposal state supplies one. Any failure returns `None`: the reference is
ignored and the block remains valid.

```python
def get_pointed_anchor(store: Store, block_root: Root) -> Optional[Root]:
    """
    [New in Simplex] Verify the fresh-quorum reference carried by the proposal
    ``block_root`` and return the anchor it designates (the highest common
    ancestor of the quorum's head fields), or ``None`` if the reference does
    not verify.
    """
    block = store.blocks[block_root]
    block_round = compute_round_at_slot(block.slot)
    # Only a round-start (first-slot) proposal may point, and only to a fresh
    # quorum: one drawn from the immediately preceding round.
    if block.slot != compute_start_slot_at_round(block_round):
        return None
    if block_round == GENESIS_ROUND:
        return None
    previous_round = Round(block_round - 1)

    quorum = block.body.anchor_quorum
    if len(quorum) == 0:
        return None

    quorum_indices: Set[ValidatorIndex] = set()
    head_roots = []
    for attestation in quorum:
        data = attestation.data
        # Attestations must be from the immediately preceding round and their
        # head fields must be known blocks (else no anchor is computable).
        if compute_round_at_slot(data.slot) != previous_round:
            return None
        if data.beacon_block_root not in store.blocks:
            return None
        checkpoint_state = get_attestation_checkpoint_state(store, data)
        # Committee structure (Electra pattern), checked rather than asserted:
        # a malformed aggregate invalidates the reference, never the block.
        data_epoch = compute_epoch_at_slot(data.slot)
        committee_offset = 0
        for committee_index in get_committee_indices(attestation.committee_bits):
            if committee_index >= get_committee_count_per_slot(checkpoint_state, data_epoch):
                return None
            committee_offset += len(
                get_beacon_committee(checkpoint_state, data.slot, committee_index)
            )
        if len(attestation.aggregation_bits) != committee_offset:
            return None
        if not is_valid_indexed_attestation(
            checkpoint_state, get_indexed_attestation(checkpoint_state, attestation)
        ):
            return None
        quorum_indices |= get_attesting_indices(checkpoint_state, attestation)
        head_roots.append(data.beacon_block_root)

    # Absolute two-thirds threshold: distinct signers' effective balances,
    # weighed on the proposal's own chain state, against the total active
    # balance. Slashed validators are excluded from the numerator only,
    # matching the spec-wide quorum tallies (compute_best_justification_target).
    state = store.block_states[block_root]
    # The indices were derived on each attestation's own chain; bound-check
    # them against the registry they are weighed in. An out-of-range index
    # makes the reference invalid — never an exception, never the block.
    if any(index >= len(state.validators) for index in quorum_indices):
        return None
    quorum_weight = Gwei(
        sum(
            state.validators[index].effective_balance
            for index in quorum_indices
            if is_active_validator(state.validators[index], get_current_epoch(state))
            and not state.validators[index].slashed
        )
    )
    total_active_balance = get_total_active_balance(state)
    if (
        quorum_weight * FINALITY_QUORUM_DENOMINATOR
        < total_active_balance * FINALITY_QUORUM_NUMERATOR
    ):
        return None

    return get_quorum_anchor(store, head_roots)
```

### New `update_pointed_anchor`

*Note*: Called by `on_block`. If the reference verifies, its anchor is adopted
for the whole round (paper "Pointing"): the walk reads it while
`pointed_anchor_round` equals the current round, so it expires at the round
boundary and no cross-round anchor state exists. The first verified reference of
a round wins; a round-start proposer equivocating with two different references
is proposal equivocation, confined to the one round by the same round expiry. A
round-start proposal that arrives after its round has ended is not adopted (its
round is already over).

```python
def update_pointed_anchor(store: Store, block_root: Root) -> None:
    """
    [New in Simplex] Adopt the anchor designated by a round-start proposal's
    fresh-quorum reference, for the proposal's round only.
    """
    block = store.blocks[block_root]
    block_round = compute_round_at_slot(block.slot)
    # Adopt only during the reference's own round.
    if compute_round_at_slot(get_current_slot(store)) != block_round:
        return
    # First verified reference of the round wins.
    if store.pointed_anchor_round == block_round and store.pointed_anchor_root != Root():
        return
    anchor_root = get_pointed_anchor(store, block_root)
    if anchor_root is None:
        return
    store.pointed_anchor_root = anchor_root
    store.pointed_anchor_round = block_round
```

### New `get_cascade_root`

*Note*: Paper Definition: cascade root — the walk-from block of the abstract
`getConfirmed`: `store.justified_checkpoint` when it sits at the height
frontier, else the always-viable `store.finalized_checkpoint` (paper
lem:F-viable).

```python
def get_cascade_root(store: Store) -> Root:
    """
    [New in Simplex] Return the cascade root: ``store.justified_checkpoint``
    when ``store.h_max == store.justified_height + 1``, else
    ``store.finalized_checkpoint``.
    """
    if store.h_max == store.justified_height + 1:
        return store.justified_checkpoint.root
    return store.finalized_checkpoint.root
```

### New `get_record_anchor`

*Note*: Paper Definition: record anchor `G1` — the walk's *fallback* anchor,
used when the round's round-start proposal does not point to a valid fresh
quorum. From the cascade root, descend while a viable child holds at least
two-thirds of the live record weight (a *relative* threshold, over
`get_record_weight`'s live-record denominator — unlike the fresh quorum's
absolute one). At most one child of any block can hold two-thirds: a validator
contributes to at most one child subtree, so children's record supports sum to
at most the live record weight and the descent is unique whenever it steps.

```python
def get_record_anchor(store: Store, blocks: Dict[Root, BeaconBlock]) -> ForkChoiceNode:
    """
    [New in Simplex] Return the record anchor (paper ``G1``): the record-descent
    output under the two-thirds threshold, from the cascade root, restricted to
    the viable subtree.
    """
    root = get_cascade_root(store)
    head = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    record_weight = get_record_weight(store)
    # With no live records there is no supported child; stay at the cascade root.
    if record_weight == Gwei(0):
        return head
    while True:
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_in_filtered_block_tree(store, blocks, child)
            and get_record_support(store, child) * 3 >= record_weight * 2
        ]
        if len(children) == 0:
            return head
        # Unique: at most one child can hold >= 2/3 of the live record weight.
        head = children[0]
```

### New `get_walk_anchor`

*Note*: Phase 1 of the walk (paper Definition: the walk). The anchor is the
pointed anchor — the fresh quorum's highest common ancestor adopted by
`update_pointed_anchor` — when the current round's round-start proposal points
to a valid fresh quorum whose anchor descends from the cascade root and is in
the viable subtree; otherwise the record anchor. The cascade root and a valid
pointed anchor are comparable (both lie on the chain of every head of a fresh
quorum), so taking the pointed anchor when it descends from the cascade root is
taking the deeper of the two; a pointed anchor that is shallower, on a
conflicting branch, or behind the viability frontier is ignored and the record
descent applies. Either way the anchor is in the viable subtree.

```python
def get_walk_anchor(store: Store, blocks: Dict[Root, BeaconBlock]) -> ForkChoiceNode:
    """
    [New in Simplex] Return the walk's anchor: the current round's pointed
    anchor if valid, else the record anchor (the two-thirds record-descent
    fallback).
    """
    cascade_node = ForkChoiceNode(
        root=get_cascade_root(store), payload_status=PAYLOAD_STATUS_PENDING
    )
    current_round = compute_round_at_slot(get_current_slot(store))
    if store.pointed_anchor_root != Root() and store.pointed_anchor_round == current_round:
        anchor = ForkChoiceNode(
            root=store.pointed_anchor_root, payload_status=PAYLOAD_STATUS_PENDING
        )
        # Adopt the pointed anchor only if it descends from the cascade root
        # (equivalently, is the deeper of the two) and is in the viable
        # subtree; otherwise ignore the pointed quorum.
        if is_in_filtered_block_tree(store, blocks, anchor) and is_ancestor(
            store, anchor, cascade_node
        ):
            return anchor
    return get_record_anchor(store, blocks)
```

### New `get_safe_confirmed_head`

*Note*: Paper Definition: safe confirmation. A block is *safe-confirmed* iff it
is availability-confirmed (an ancestor of `store.latest_confirmed_head`) and
G0-clear — no conflicting block holds at least one-third record support. The
safe-confirmed head is the deepest such block, well defined because G0-clearance
is monotone along a chain: a block conflicting with an ancestor also conflicts
with the block, so every ancestor of a G0-clear block is G0-clear. Safe
confirmation is an *internal* notion, read by the finality-vote gates (validator
spec); the user-facing available confirmation (`store.latest_confirmed_head`) is
untouched by it and is never gated by record or finality-gadget state.

```python
def get_safe_confirmed_head(store: Store) -> Root:
    """
    [New in Simplex] Return the safe-confirmed head: the deepest ancestor of
    the available-confirmed head that is G0-clear.
    """
    head = store.latest_confirmed_head[0]
    if head not in store.blocks:
        return store.finalized_checkpoint.root
    # By monotonicity, the first G0-clear block on the walk up from the
    # available-confirmed head is the deepest G0-clear ancestor. The walk
    # terminates at the store anchor: every accepted block descends from it
    # (on_block requires a known parent), so the anchor conflicts with nothing
    # and is G0-clear. Blocks conflicting with the current finalized root may
    # legitimately be in the store and hold live records — which is why the
    # walk can pass below the finalized root before clearing.
    while not is_g0_clear(store, head):
        head = store.blocks[head].parent_root
    return head
```

### New `get_available_confirmation_head`

*Note*: Called by `on_tick_per_slot` at `AVAILABLE_CONFIRMATION_DUE_BPS` to
maintain `store.latest_confirmed_head`. The head is computed over **all**
accepted descendants of the finalized root, not the viability-filtered subtree:
the user-facing available confirmation is floorless and never gated by
finality-gadget or record state (paper Definition: Goldfish available chain and
confirmation). The internal, record-gated notion the finality-vote gates read is
`get_safe_confirmed_head`.

```python
def get_available_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    [New in Simplex] Return the delayed available-confirmation head for slot
    ``n`` when called in slot ``n+1``, from the previous slot's frozen
    available votes, over all accepted descendants of the finalized root.
    """
    # User-facing confirmation is floorless and never gated by finality-gadget
    # or record state: it walks the unfiltered accepted block tree from the
    # finalized root, with no height-filter viability bound.
    blocks = store.blocks
    head = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )

    # Delayed available confirmation. Among viable children pick by
    # confirmation score, then root, then payload-status tiebreaker -- matching
    # get_head's disambiguation so that at a payload-decision (EMPTY/FULL) node
    # the better-supported payload wins rather than the inherited EMPTY-first
    # child order.
    while True:
        child = get_best_available_confirmation_child(store, blocks, head)
        if child is None:
            return head
        head = child
```

### New `get_payload_participant_count`

```python
def get_payload_participant_count(store: Store, root: Root) -> uint64:
    """Return the number of PTC seats with a vote in the block's slot."""
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    return uint64(len([index for index in ptc if index in payload_votes]))
```

### New `get_payload_full_support`

```python
def get_payload_full_support(store: Store, root: Root) -> uint64:
    """
    Return payload FULL support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``payload_present == True`` count.
    Equivocating participants in the slot are included for viability.
    """
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    equivocations = store.payload_vote_equivocations.get(vote_slot, set())
    full_support_count = uint64(0)
    for ptc_member_index in ptc:
        if ptc_member_index not in payload_votes:
            continue
        vote = payload_votes[ptc_member_index]
        if ptc_member_index in equivocations or (
            vote.beacon_block_root == root and vote.payload_present
        ):
            full_support_count += 1
    return full_support_count
```

### New `get_payload_data_available_support`

```python
def get_payload_data_available_support(store: Store, root: Root) -> uint64:
    """
    Return payload data-availability support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``blob_data_available == True``
    count. Equivocating participants in the slot are included for viability.
    """
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    equivocations = store.payload_vote_equivocations.get(vote_slot, set())
    data_available_support_count = uint64(0)
    for ptc_member_index in ptc:
        if ptc_member_index not in payload_votes:
            continue
        vote = payload_votes[ptc_member_index]
        if ptc_member_index in equivocations or (
            vote.beacon_block_root == root and vote.blob_data_available
        ):
            data_available_support_count += 1
    return data_available_support_count
```

### New `is_payload_verified`

*Note*: Adopted verbatim from gloas. Membership in `store.payloads` is the
local-availability gate consulted by the payload-decision helpers below.

```python
def is_payload_verified(store: Store, root: Root) -> bool:
    """
    Return whether the execution payload envelope for the beacon block with
    root ``root`` has been locally delivered and verified via
    ``on_execution_payload_envelope``.
    """
    return root in store.payloads
```

### Modified `is_payload_timely`

```python
def is_payload_timely(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload FULL support.
    """
    # [Modified in Simplex]
    # Local-availability gate now reads ``store.payloads`` via ``is_payload_verified``.
    if not is_payload_verified(store, root):
        return False

    participant_count = get_payload_participant_count(store, root)
    full_support_count = get_payload_full_support(store, root)
    return full_support_count > participant_count // 2
```

### Modified `is_payload_data_available`

```python
def is_payload_data_available(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload data-availability support.
    """
    # [Modified in Simplex]
    # Local-availability gate now reads ``store.payloads`` via ``is_payload_verified``.
    if not is_payload_verified(store, root):
        return False

    participant_count = get_payload_participant_count(store, root)
    data_available_support_count = get_payload_data_available_support(store, root)
    return data_available_support_count > participant_count // 2
```

### Modified `should_extend_payload`

```python
def should_extend_payload(store: Store, root: Root) -> bool:
    # [Modified in Simplex]
    # Strict majority required for both payload presence and data availability.
    return is_payload_timely(store, root) and is_payload_data_available(store, root)
```

### Modified `should_build_on_full`

*Note*: gloas's version calls `payload_timeliness` /
`payload_data_availability`, which read the removed
`store.payload_timeliness_vote` / `payload_data_availability_vote`. Simplex
decides build-on-full from its own strict-majority `should_extend_payload`,
keeping the proposer's choice consistent with the fork-choice payload decision.

```python
def should_build_on_full(store: Store, head: ForkChoiceNode) -> bool:
    assert head.payload_status != PAYLOAD_STATUS_PENDING
    if store.blocks[head.root].slot + 1 != get_current_slot(store):
        return head.payload_status == PAYLOAD_STATUS_FULL
    if head.payload_status == PAYLOAD_STATUS_EMPTY:
        return False
    # [Modified in Simplex]
    # Strict-majority decision via simplex's own payload counters.
    return should_extend_payload(store, head.root)
```

### Modified `update_latest_messages`

```python
def update_latest_messages(
    store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation
) -> None:
    # [Modified in Simplex]
    # A finality vote is an LMD vote for the beacon block; it carries no payload
    # decision (the supported node is PAYLOAD_STATUS_PENDING -- see
    # get_attestation_score). LatestMessage no longer carries payload_present.
    slot = attestation.data.slot
    beacon_block_root = attestation.data.beacon_block_root
    non_equivocating_attesting_indices = [
        i for i in attesting_indices if i not in store.equivocating_indices
    ]
    for i in non_equivocating_attesting_indices:
        if i not in store.latest_messages or slot > store.latest_messages[i].slot:
            store.latest_messages[i] = LatestMessage(
                slot=slot,
                root=beacon_block_root,
            )
```

### Modified `get_attestation_score`

*Note*: The base fork choice provides `get_attestation_score`; simplex overrides
it to add the latest-message expiry filter (`has_unexpired_latest_message`).
Only unexpired, non-equivocating supporting latest messages contribute. The walk
does not read this score (records and available attestations drive it); the
override is kept so that the base fork choice's latest-message semantics never
leak into this fork.

```python
def get_attestation_score(
    store: Store,
    node: ForkChoiceNode,
    state: BeaconState,
    window_slots: uint64 = LATEST_MESSAGE_EXPIRY_SLOTS,
) -> Gwei:
    """
    [New in Simplex] Effective-balance weight of unexpired, non-equivocating
    latest messages supporting ``node`` cast within the last ``window_slots``
    slots (default: the whole unexpired set).
    """
    current_slot = get_current_slot(store)
    unslashed_and_active_indices = [
        i
        for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    return Gwei(
        sum(
            state.validators[i].effective_balance
            for i in unslashed_and_active_indices
            if (
                has_unexpired_latest_message(store, i)
                and store.latest_messages[i].slot + window_slots >= current_slot
                # Finality votes are beacon-block votes at PAYLOAD_STATUS_PENDING.
                and is_supporting_vote(
                    store, node, store.latest_messages[i], PAYLOAD_STATUS_PENDING
                )
            )
        )
    )
```

### Modified `get_weight`

```python
def get_weight(
    store: Store, node: ForkChoiceNode, window_slots: uint64 = LATEST_MESSAGE_EXPIRY_SLOTS
) -> Gwei:
    # [Modified in Simplex]
    # Returns 0 for payload-decision nodes; no proposer boost. Counts only votes
    # cast within the last ``window_slots`` slots (default: whole unexpired set).
    if is_ptc_decision_node(store, node):
        return Gwei(0)
    state = store.block_states[store.justified_checkpoint.root]
    return get_attestation_score(store, node, state, window_slots)
```

### Modified `get_head`

*Note*: The walk (paper Definition: the walk), in three phases. (1) *Anchor*:
the pointed anchor — the highest common ancestor of the fresh quorum the current
round's round-start proposal points to — when it verifies, descends from the
cascade root, and is viable; otherwise the record anchor, the two-thirds record
descent from the cascade root (`get_walk_anchor`). (2) *Goldfish*: from the
anchor, follow the available chain within the viable subtree, descending by
previous-slot participant majority. (3) *Viability descent*: the phase-2 descent
continued without its majority gate — stepping into the viable child with the
greatest previous-slot participating vote weight — until the head's state-height
reaches the height-filter bound `h_max - 1`, so every walk output sits at the
height frontier. All consumers (proposals, available votes, finality-vote
construction, confirmations) read the same walk.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # [Modified in Simplex]
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)

    # Phase 1 -- anchor: the pointed anchor if the current round's round-start
    # proposal points to a valid fresh quorum, else the record anchor.
    head = get_walk_anchor(store, blocks)

    # Phase 2 -- Goldfish descent from the anchor, within the viable subtree,
    # using previous-slot available attestations.
    while True:
        children = get_node_children(store, blocks, head)
        viable_children = [
            child
            for child in children
            if is_in_filtered_block_tree(store, blocks, child)
            and is_available_attestation_viable(store, child)
        ]
        if len(viable_children) == 0:
            break
        head = max(
            viable_children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )

    # Phase 3 -- viability descent: the phase-2 descent continued without its
    # majority gate (paper Definition: viability descent), until the head's
    # state-height reaches the height-filter bound. A viable child exists
    # until the bound is met; the emptiness guard only keeps the walk total.
    while store.block_states[head.root].current_height < get_viability_height_threshold(store):
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_in_filtered_block_tree(store, blocks, child)
        ]
        if len(children) == 0:
            break
        head = max(
            children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
    return head
```

### New `is_valid_from_block_attestation`

*Note*: The skip-only validator for block-included finality attestations,
called by `on_attestation` on the from-block path. It never asserts: any
failure skips the attestation's effects and leaves the block accepted, so a
block's fork-choice acceptance can never depend on the local view of the
attestations it carries. (An assert here would split block acceptance across
views — e.g. a target-slot mismatch is only detectable by nodes that know the
target root.)

The signature check is what makes record attribution correct: the state
transition verified the aggregate under the *including* chain, while record
attribution resolves the committee on the attestation's own head chain — on
diverged forks the two samplings can disagree, and unverified attribution
would let one included aggregate mark non-signers (two such same-round
inclusions could manufacture equivocation marks against honest validators).
Re-verifying under the head-chain-resolved committee makes the attributed
indices actual signers, at the cost of one extra aggregate-signature
verification per block-included finality attestation. The committee-structure
walk also length-guards the bits/committee mapping, which on mismatched
committee sizes would otherwise raise.

```python
def is_valid_from_block_attestation(store: Store, attestation: Attestation) -> bool:
    """
    [New in Simplex] Return whether a block-included finality attestation may
    feed record attribution: well-formed data and a valid aggregate signature
    under the committee resolved on the attestation's own head chain. Checked,
    never asserted: a failure skips the attestation, never rejects the block.
    """
    data = attestation.data
    # The named head block must not be later than the attestation.
    if store.blocks[data.beacon_block_root].slot > data.slot:
        return False
    if data.target != Checkpoint():
        # A known justification target must be a real block at its actual
        # proposal slot; an unknown target (on another fork) is tolerated so
        # the head record still flows.
        if data.target.root in store.blocks:
            if store.blocks[data.target.root].slot != data.target.slot:
                return False
        # Target slot may precede attestation slot (height-based finality).
        if data.target.slot > data.slot:
            return False
    # Attestations can only affect fork choice of subsequent slots.
    if get_current_slot(store) < data.slot + 1:
        return False
    # Committee structure (Electra pattern) under the head-chain checkpoint
    # state: on a diverged fork the committee sizes can differ from the
    # including chain's, so the bits/committee mapping is length-guarded.
    checkpoint_state = get_attestation_checkpoint_state(store, data)
    data_epoch = compute_epoch_at_slot(data.slot)
    committee_offset = 0
    for committee_index in get_committee_indices(attestation.committee_bits):
        if committee_index >= get_committee_count_per_slot(checkpoint_state, data_epoch):
            return False
        committee_offset += len(
            get_beacon_committee(checkpoint_state, data.slot, committee_index)
        )
    if len(attestation.aggregation_bits) != committee_offset:
        return False
    # Attribution is signature-verified under the resolving state: the state
    # transition verified this aggregate under the including chain, whose
    # committee sampling can disagree with the head chain's on diverged forks.
    return is_valid_indexed_attestation(
        checkpoint_state, get_indexed_attestation(checkpoint_state, attestation)
    )
```

### Modified `validate_on_attestation`

*Note*: Wire-only. Wire attestations must name known blocks (the head, and the
justification target when one is set) and are epoch-bounded (current or
previous epoch), asserting on violation. From-block attestations never reach
this function: they take the skip-only `is_valid_from_block_attestation` path
in `on_attestation`, bounded by the record window instead of wall-clock
epochs. Timeout votes and empty votes use `Checkpoint()` as the target and
still carry a head vote.

```python
def validate_on_attestation(store: Store, attestation: Attestation) -> None:
    data = attestation.data
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    if data.target != Checkpoint():
        # [Modified in Simplex]
        # A wire justification target must name a known real block at its
        # actual proposal slot.
        assert data.target.root in store.blocks
        assert store.blocks[data.target.root].slot == data.target.slot
        # Target slot may precede attestation slot (height-based finality)
        assert data.target.slot <= data.slot

    # Attestations can only affect fork choice of subsequent slots.
    # Delay consideration in the fork-choice until their slot is in the past.
    assert get_current_slot(store) >= data.slot + 1

    # [Modified in Simplex]
    # Epoch-bounded: attestation slot must be in current or previous epoch.
    current_epoch = get_current_store_epoch(store)
    previous_epoch = GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)
    attestation_epoch = compute_epoch_at_slot(data.slot)
    assert attestation_epoch in (current_epoch, previous_epoch)
```

### New `validate_on_available_attestation`

```python
def validate_on_available_attestation(
    store: Store, attestation: AvailableAttestation, is_from_block: bool
) -> None:
    if not is_from_block:
        # Wire votes are only accepted for the current slot
        # (view-merge synchronization window).
        assert attestation.data.slot == get_current_slot(store)

    # Attestations must be for a known block.
    assert attestation.data.beacon_block_root in store.blocks
    # Attestations must not be for blocks in the future.
    block_slot = store.blocks[attestation.data.beacon_block_root].slot
    assert block_slot <= attestation.data.slot
    # Available attestation bits must match the fixed committee size.
    assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE

    # Same-slot attestation cannot signal payload availability
    if block_slot == attestation.data.slot:
        assert not attestation.data.payload_present
```

## Handlers

### Modified `on_tick_per_slot`

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    # [Modified in Simplex]
    # No epoch boundary pull-up; initializes per-slot vote tracking.
    previous_slot = get_current_slot(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        # [New in Simplex]
        # A slot crossed without a post-deadline tick still gets its freeze,
        # from the best data available at the boundary. A multi-slot tick jump
        # leaves the skipped slots unfrozen: the next confirmation evaluation
        # conservatively regresses for one slot and self-heals.
        freeze_available_votes(store, previous_slot)
        store.payload_votes[current_slot] = {}
        store.payload_vote_equivocations[current_slot] = set()
        store.available_votes[current_slot] = {}
        store.available_vote_equivocations[current_slot] = set()
        store.available_timely_attesters[current_slot] = set()
        # [New in Simplex]
        # Drop record entries and equivocation marks that left the record
        # window (reads are window-gated; this only bounds the maps), and
        # buffered unknown-head attestations along with them (a replay would
        # be a no-op for them regardless).
        prune_records(store)
        prune_unknown_head_attestations(store)

    # [New in Simplex]
    # Confirmation rules: once local time is past the confirmation deadline,
    # capture the current slot's time-shifted-quorum freeze, then record the
    # available-confirmed head from the previous slot's freeze and the
    # fast-confirmed head from the current slot's freeze. The freeze is
    # captured once per slot, so repeat ticks within the slot re-read the same
    # frozen numerator and denominator; the confirmation-head walk itself
    # reads the live tree from the live finalized root, so a mid-slot finality
    # advance can move the walk's start (the frozen evaluation never retracts).
    if not is_before_available_confirmation_deadline(store):
        freeze_available_votes(store, get_current_slot(store))
        store.latest_confirmed_head = (
            get_available_confirmation_head(store).root,
            get_current_slot(store),
        )
        store.fast_confirmed_head = (
            get_fast_confirmation_head(store).root,
            get_current_slot(store),
        )
```

### Modified `on_block`

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    assert block.parent_root in store.block_states

    # [Modified in Simplex]
    # gloas payloads model: never cache a post-payload state. If the parent is
    # full, only assert the parent envelope was locally verified; the parent's
    # executed payload is folded into this child during its own state transition
    # (process_parent_execution_payload). Always copy the parent's block state.
    if is_parent_node_full(store, block):
        assert is_payload_verified(store, block.parent_root)
    state = copy(store.block_states[block.parent_root])

    current_slot = get_current_slot(store)
    assert current_slot >= block.slot

    # (a) Assert block descends from finalized: slot guard rejects equivocating forks,
    #     ancestry check enforces linear descent from the current finalized root.
    assert block.slot > store.finalized_checkpoint.slot
    assert (
        store.finalized_checkpoint.root
        == get_ancestor(
            store,
            ForkChoiceNode(root=block.parent_root, payload_status=PAYLOAD_STATUS_PENDING),
            store.finalized_checkpoint.slot,
        ).root
    )

    # (b) State transition
    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)  # noqa: FBT003

    store.blocks[block_root] = block
    store.block_states[block_root] = state

    notify_ptc_messages(store, state, block.body.payload_attestations)

    # [Modified in Simplex]
    # Process finality attestations (update records and latest_messages)
    for attestation in block.body.attestations:
        on_attestation(store, attestation, is_from_block=True)

    # [New in Simplex]
    # Replay buffered from-block finality attestations whose head field named
    # this block before it arrived: with the block (and hence its head-chain
    # checkpoint state) now in the store, they take the same attribution path
    # as live-processed ones.
    replay_unknown_head_attestations(store, block_root)

    # [Modified in Simplex]
    # Process available attestations (per-slot Goldfish tracking)
    for available_attestation in block.body.available_attestations:
        on_available_attestation(store, available_attestation, is_from_block=True)

    # [New in Simplex]
    # Bump h_max so the viability guard sees the new maximum
    # before update_finalized evaluates it.
    store.h_max = max(store.h_max, state.current_height)

    # [New in Simplex]
    # Single justification cert event: lex-max on
    # ``(h_j, hash_tree_root(J))`` with the F-filter.
    update_justified(store, state.justified_checkpoint, state.justified_height)

    # [New in Simplex]
    # Advance Σ.F if the block's finalized checkpoint improves
    # on the stored one, descends from Σ.J, and lies in the viable subtree.
    update_finalized(store, state.finalized_checkpoint)

    # [New in Simplex]
    # Adopt the round's pointed anchor from a round-start proposal carrying a
    # valid fresh-quorum reference. An invalid reference is ignored.
    update_pointed_anchor(store, block_root)
```

### Modified `on_payload_attestation_message`

*Note*: Payload votes use first-vote + equivocation tracking with view-merge
freeze handling. Non-proposer wire votes after freeze are ignored; the next
proposer may continue collecting via `is_next_proposer=True`.

```python
def on_payload_attestation_message(
    store: Store,
    ptc_message: PayloadAttestationMessage,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    data = ptc_message.data
    assert data.beacon_block_root in store.block_states

    state = store.block_states[data.beacon_block_root]
    ptc = get_ptc(state, data.slot)

    # PTC votes can only affect the slot they are assigned to.
    if data.slot != state.slot:
        return
    assert ptc_message.validator_index in ptc

    if not is_from_block:
        # [Modified in Simplex]
        # Wire votes accepted only for the current slot,
        # with view-freeze gating (proposer may override via ``is_next_proposer``).
        assert data.slot == get_current_slot(store)
        if not is_next_proposer and not is_before_view_freeze_deadline(store):
            return
        assert is_valid_indexed_payload_attestation(
            state,
            IndexedPayloadAttestation(
                attesting_indices=[ptc_message.validator_index],
                data=data,
                signature=ptc_message.signature,
            ),
        )

    vote_slot = data.slot
    if vote_slot not in store.payload_votes:
        return

    payload_votes = store.payload_votes[vote_slot]
    equivocations = store.payload_vote_equivocations[vote_slot]
    validator_index = ptc_message.validator_index
    # [Modified in Simplex]
    # Votes are keyed by validator identity: equivocation is same validator +
    # same slot + different data, regardless of committee position or branch
    # family. Seat multiplicity (a validator may hold multiple PTC seats under
    # balance-weighted selection) is resolved at read time against the slot's
    # PTC committee.
    if validator_index in equivocations:
        return
    if validator_index not in payload_votes:
        payload_votes[validator_index] = data
    elif payload_votes[validator_index] != data:
        equivocations.add(validator_index)
```

### Modified `on_attestation`

*Note*: Finality attestations included in blocks feed on-chain `record_votes`
via `update_records`, on a skip-only path: every from-block validation failure
skips the attestation's effects and never raises, so a block's acceptance
never depends on the local view of the attestations it carries. Record
attribution is correct by signature under the resolving state, independent of
the including chain: `is_valid_from_block_attestation` re-verifies the
aggregate under the committee resolved on the attestation's own head chain,
so the indices fed to `update_records` are actual signers even when the
including chain's committee sampling diverges. The `latest_messages` helper
state is still updated when the finality target path is known; it is inert,
retained base-fork machinery (the walk never consumes it — records and
available attestations drive the walk; see `get_attestation_score`). A
from-block attestation whose finality target is unknown (the voter may have
voted for a block on a different fork) still feeds its head field into the
record layer, but does not update `latest_messages`.

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """[Modified in Simplex]"""
    data = attestation.data
    if is_from_block:
        # [New in Simplex]
        # Skip-only path: no failure below raises, so a block's acceptance
        # never depends on the local view of the attestations it carries.
        # An attestation whose head vote references an unknown block (the
        # voter may have voted for a block on a different fork) is buffered
        # for replay through this same path when the head block arrives.
        if data.beacon_block_root not in store.blocks:
            buffer_unknown_head_attestation(store, attestation)
            return
        # From-block acceptance is bounded by the record window, not by
        # wall-clock epochs: a node processing an old block late must
        # reconstruct the same record set as a node that processed it live. A
        # beyond-window attestation is a no-op (and stale for
        # ``latest_messages``, whose expiry equals the record window).
        if not is_record_in_window(store, data.slot):
            return
        if not is_valid_from_block_attestation(store, attestation):
            return
        # Attribution: the committee is resolved (and the aggregate was just
        # verified) on the attestation's own head chain, so the recorded
        # indices are actual signers regardless of the including chain.
        target_state = get_attestation_checkpoint_state(store, data)
        attesting_indices = get_attesting_indices(target_state, attestation)
        update_records(store, attestation, sorted(attesting_indices))
        # A justification vote whose finality target is unknown still feeds
        # its head field into the record layer, but does not update
        # latest_messages.
        if data.target != Checkpoint() and data.target.root not in store.blocks:
            return
        update_latest_messages(store, sorted(attesting_indices), attestation)
        return

    validate_on_attestation(store, attestation)

    # Derive the checkpoint state for signature verification and attesting
    # indices (epoch boundary on the attestation's own head chain).
    target_state = get_attestation_checkpoint_state(store, data)

    # Verify signature against beacon committee
    assert is_valid_indexed_attestation(
        target_state, get_indexed_attestation(target_state, attestation)
    )

    attesting_indices = get_attesting_indices(target_state, attestation)
    update_latest_messages(store, sorted(attesting_indices), attestation)
```

### New `on_available_attestation`

*Note*: Available attestations track per-slot per-committee-member votes for the
Goldfish fork choice layer. Non-proposers ignore wire votes after freeze;
proposers may continue collecting via `is_next_proposer=True`.

```python
def on_available_attestation(
    store: Store,
    attestation: AvailableAttestation,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    """[New in Simplex]"""
    # Skip from-block attestations whose head vote references an unknown block
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        return

    if not is_from_block and not is_next_proposer and not is_before_view_freeze_deadline(store):
        # Late wire vote: ignored for view-merge.
        return

    validate_on_available_attestation(store, attestation, is_from_block)

    # Derive checkpoint state for signature verification and committee positions.
    # Committee membership is epoch-based; use epoch boundary slot for Checkpoint.
    attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    epoch_root = get_ancestor(
        store,
        ForkChoiceNode(
            root=attestation.data.beacon_block_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        ),
        epoch_boundary_slot,
    ).root
    checkpoint = Checkpoint(slot=epoch_boundary_slot, root=epoch_root)

    if checkpoint not in store.checkpoint_states:
        base_state = copy(store.block_states[epoch_root])
        epoch_start_slot = compute_start_slot_at_epoch(attestation_epoch)
        if base_state.slot < epoch_start_slot:
            process_slots(base_state, epoch_start_slot)
        store.checkpoint_states[checkpoint] = base_state

    target_state = store.checkpoint_states[checkpoint]

    if not is_from_block:
        # Verify signature against available committee
        attesting_indices = get_available_attesting_indices(target_state, attestation)
        pubkeys = [target_state.validators[i].pubkey for i in sorted(attesting_indices)]
        domain = get_domain(target_state, DOMAIN_AVAILABLE_ATTESTER, attestation_epoch)
        signing_root = compute_signing_root(attestation.data, domain)
        assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Store individual votes for Goldfish tracking.
    vote_slot = attestation.data.slot
    if vote_slot not in store.available_votes:
        return
    available_votes = store.available_votes[vote_slot]
    available_vote_equivocations = store.available_vote_equivocations[vote_slot]
    available_timely_attesters = store.available_timely_attesters[vote_slot]
    current_slot = get_current_slot(store)

    # [Modified in Simplex]
    # Votes are keyed by validator identity: equivocation is same validator +
    # same slot + different data, regardless of committee position or branch
    # family. Seat multiplicity is resolved at read time against the slot's
    # available committee.
    for member_index in get_available_attesting_indices(target_state, attestation):
        # Ignore further votes once the member has equivocated.
        if member_index in available_vote_equivocations:
            continue
        if member_index not in available_votes:
            # First vote from this committee member for this slot
            available_votes[member_index] = attestation.data
            if (
                vote_slot == current_slot
                and not is_from_block
                and is_before_available_confirmation_deadline(store)
            ):
                available_timely_attesters.add(member_index)
        elif available_votes[member_index] != attestation.data:
            # Second (different) vote — record as equivocation
            available_vote_equivocations.add(member_index)
```

## Deprecated overrides

*Note*: The following functions shadow inherited implementations whose
underlying mechanisms (unrealized justifications, proposer boost, block
timeliness) are removed in simplex.

### Modified `get_voting_source`

```python
def get_voting_source(store: Store, block_root: Root) -> Checkpoint:
    # [Modified in Simplex]
    # No unrealized justification pull-up.
    head_state = store.block_states[block_root]
    return head_state.justified_checkpoint
```

### Modified `update_unrealized_checkpoints`

```python
def update_unrealized_checkpoints(
    store: Store,
    unrealized_justified_checkpoint: Checkpoint,
    unrealized_finalized_checkpoint: Checkpoint,
) -> None:
    # [Modified in Simplex]
    # No unrealized checkpoints.
    pass
```

### Modified `compute_pulled_up_tip`

```python
def compute_pulled_up_tip(store: Store, block_root: Root) -> None:
    # [Modified in Simplex]
    # No pull-up needed.
    pass
```

### Modified `record_block_timeliness`

```python
def record_block_timeliness(store: Store, root: Root) -> None:
    # [Modified in Simplex]
    # Block timeliness tracking removed.
    pass
```

### Modified `update_proposer_boost_root`

```python
def update_proposer_boost_root(store: Store, head: Root, root: Root) -> None:
    # [Modified in Simplex]
    # Proposer boost removed.
    pass
```

### Modified `is_head_late`

```python
def is_head_late(store: Store, head_root: Root) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Block timeliness tracking removed.
    return False
```

### Modified `is_ffg_competitive`

```python
def is_ffg_competitive(store: Store, head_root: Root, parent_root: Root) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Unrealized justifications removed.
    return True
```

### Modified `is_head_weak`

```python
def is_head_weak(store: Store, head_root: Root) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Proposer-boost path removed.
    return False
```

### Modified `is_parent_strong`

```python
def is_parent_strong(store: Store, root: Root) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Proposer-boost path removed.
    return True
```

### Modified `should_apply_proposer_boost`

```python
def should_apply_proposer_boost(store: Store) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Proposer boost disabled.
    return False
```

### Modified `should_override_forkchoice_update`

```python
def should_override_forkchoice_update(store: Store, head_root: Root) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Override path removed.
    return False
```

### Modified `get_proposer_head`

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:  # noqa: ARG001
    # [Modified in Simplex]
    # Proposer override removed.
    return head_root
```

### Modified `is_finalization_ok`

```python
def is_finalization_ok(store: Store, slot: Slot) -> bool:  # noqa: ARG001
    # [Modified in Simplex]
    # Not used — proposer reorg path removed.
    return True
```

### Modified `validate_target_epoch_against_current_time`

```python
def validate_target_epoch_against_current_time(store: Store, attestation: Attestation) -> None:
    # [Modified in Simplex]
    # Not used — validate_on_attestation uses slot-based check.
    pass
```

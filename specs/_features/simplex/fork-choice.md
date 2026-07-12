# Simplex Finality -- Fork Choice

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Configuration](#configuration)
- [Containers](#containers)
  - [New `RecordVote`](#new-recordvote)
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
  - [New `get_total_active_voting_weight`](#new-get_total_active_voting_weight)
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
  - [New `get_available_confirmation_score`](#new-get_available_confirmation_score)
  - [New `get_available_confirmation_majority_threshold`](#new-get_available_confirmation_majority_threshold)
  - [New `is_available_confirmation_viable`](#new-is_available_confirmation_viable)
  - [New `get_best_available_confirmation_child`](#new-get_best_available_confirmation_child)
  - [New `get_fast_confirmation_score`](#new-get_fast_confirmation_score)
  - [New `is_fast_confirmation_viable`](#new-is_fast_confirmation_viable)
  - [New `get_fast_confirmation_head`](#new-get_fast_confirmation_head)
  - [New `update_records`](#new-update_records)
  - [New `is_live_record_validator`](#new-is_live_record_validator)
  - [New `get_record_weight`](#new-get_record_weight)
  - [New `get_record_support`](#new-get_record_support)
  - [New `is_g0_clear`](#new-is_g0_clear)
  - [New `get_attestation_checkpoint_state`](#new-get_attestation_checkpoint_state)
  - [New `get_quorum_anchor`](#new-get_quorum_anchor)
  - [New `get_pointed_anchor`](#new-get_pointed_anchor)
  - [New `update_pointed_anchor`](#new-update_pointed_anchor)
  - [New `get_iterated_majority_head`](#new-get_iterated_majority_head)
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
(or whose descendants reach that bound) are viable. Layer 2 is the
iterated-majority stabilization gadget (`get_iterated_majority_head`), which
walks from a cascade root chosen between `store.justified_checkpoint` and
`store.finalized_checkpoint` and refines the head over shrinking windows of the
most recent committees — each window's majority walk anchored at the previous
output and restricted to the viable subtree. Layer 3 is the Goldfish
available-chain walk (`get_head`), which extends the Layer 2 head using
previous-slot available attestations.

*Note*: This specification is built upon Gloas (EIP-7732 ePBS fork choice).

## Configuration

| Name                                      | Value                  | Description                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LATEST_MESSAGE_EXPIRY_SLOTS`             | `uint64(2**7)` (= 128) | Outer staleness bound on Layer 2 weight: a validator's `latest_message` is ignored (in both numerator and denominator) once its slot is more than this many slots in the past. The iterated-majority head's round-based committee windows are normally tighter, so this only binds for pathologically long rounds. |
| `RECORD_WINDOW_SLOTS`                     | `uint64(2**7)` (= 128) | On-chain record window `W_R`: an included finality-attestation head record older than this many slots is ignored (in both numerator and denominator of the record-support arithmetic).                                                                                                                             |
| `AVAILABLE_CONFIRMATION_DUE_BPS`          | `uint64(5000)`         | basis points; 50% of `SLOT_DURATION_MS`. Dual role: in-slot cutoff for an available vote to count as *timely*, and the time at which the previous slot's available-confirmation rule is run. Sits between the attestation deadline and the view-freeze deadline (propose / attest / confirm / freeze).             |
| `FAST_CONFIRMATION_COMMITTEE_NUMERATOR`   | `uint64(3)`            | Numerator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                                      |
| `FAST_CONFIRMATION_COMMITTEE_DENOMINATOR` | `uint64(4)`            | Denominator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                                    |
| `VIEW_FREEZE_DUE_BPS`                     | `uint64(7500)`         | basis points; 75% of `SLOT_DURATION_MS`. In-slot vote-freeze boundary for view-merge: wire votes after this time are deferred to the next proposer's view.                                                                                                                                                         |

## Containers

### New `RecordVote`

*Note*: An on-chain SG record: the head field (`head`) of a finality attestation
included in a block, tagged with the attestation's slot (`slot`). Records are
kept latest-per-validator and drive the record-support arithmetic
(`get_record_support`, `is_g0_clear`) that the Layer 2 walk and the safe-
confirmation gates consume.

```python
@dataclass(eq=True, frozen=True)
class RecordVote:
    # [New in Simplex]
    slot: Slot
    head: Root
```

### Modified `Store`

*Note*: `justified_checkpoint` (paper's `Σ.J`) and `justified_height` (paper's
`Σ.h_j`) jointly track the lex-max justification cert event ever observed, under
the lex key `(h_j, hash(J))`. `h_max` (paper's `Σ.h_max`) tracks the maximum
`state.current_height` over all known block states; it drives the height filter
/ viable subtree (paper Definition: viable subtree). `finalized_checkpoint` is
the paper's `Σ.F`. Downstream consumers (e.g., `get_total_active_voting_weight`,
`get_weight`, `get_record_weight`) read
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
    # On-chain SG record layer: the latest included finality-attestation head
    # record per validator (paper record vote) and the on-chain-provable record
    # equivocators. Seeded in ``get_forkchoice_store``; fed by ``update_records``.
    record_votes: Dict[ValidatorIndex, RecordVote]
    record_equivocators: Set[ValidatorIndex]
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
height `0` and starts the finality gadget at state-height `1`. Sentinel
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
        record_equivocators=set(),
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
    )
```

### New `update_justified`

*Note*: Paper's `updateJustified`. For a justification cert event `(J', h')` —
accept the event only if the candidate descends from the current finalized
checkpoint (`F-filter`), then update
`(store.justified_checkpoint, store.justified_height)` iff the candidate's lex
key `(h', hash_tree_root(J'))` strictly exceeds the current store key
`(store.justified_height, hash_tree_root(store.justified_checkpoint))`.

```python
def update_justified(
    store: Store, justified_checkpoint: Checkpoint, justified_height: Height
) -> None:
    """
    [New in Simplex] Paper's updateJustified. Filter candidates by F ⪯ J',
    then lex running-max on ``(h_j, hash_tree_root(J))``.
    """
    if justified_checkpoint == Checkpoint():
        return  # No justification yet (sentinel)
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

*Note*: Latest-message expiry for Layer 2 weight. A validator's `latest_message`
is counted in `get_iterated_majority_head` (both the numerator
`get_attestation_score` and the majority denominator
`get_total_active_voting_weight`) only while its slot is within
`LATEST_MESSAGE_EXPIRY_SLOTS` of the current slot, i.e. while
`message.slot > current_slot - LATEST_MESSAGE_EXPIRY_SLOTS`. This keeps the
majority threshold tied to recently-active weight rather than to every validator
that has ever voted.

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

### New `get_total_active_voting_weight`

```python
def get_total_active_voting_weight(
    store: Store, window_slots: uint64 = LATEST_MESSAGE_EXPIRY_SLOTS
) -> Gwei:
    """
    Return the total effective balance of unslashed active validators that have
    an unexpired, non-equivocating ``latest_message`` cast within the last
    ``window_slots`` slots. The default window is the whole unexpired set; the
    iterated-majority head (Layer 2) calls it with shrinking windows to weigh the
    most recent committees. A validator's committee slot is ``latest_message.slot``.
    """
    state = store.block_states[store.justified_checkpoint.root]
    current_slot = get_current_slot(store)
    participating_indices: Set[ValidatorIndex] = set()
    for index in get_active_validator_indices(state, get_current_epoch(state)):
        if state.validators[index].slashed:
            continue
        # [New in Simplex]
        # Only validators with an unexpired latest message within the window count.
        if not has_unexpired_latest_message(store, index):
            continue
        if store.latest_messages[index].slot + window_slots < current_slot:
            continue
        participating_indices.add(index)

    return get_total_balance(state, participating_indices)
```

### New `get_view_freeze_due_ms`

```python
def get_view_freeze_due_ms() -> uint64:
    """Return the in-slot vote-freeze boundary for view-merge."""
    return get_slot_component_duration_ms(VIEW_FREEZE_DUE_BPS)
```

### New `is_before_view_freeze_deadline`

```python
def is_before_view_freeze_deadline(store: Store) -> bool:
    """Return whether current local time is before the view-merge vote-freeze boundary."""
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_view_freeze_due_ms()
```

### New `get_available_confirmation_due_ms`

```python
def get_available_confirmation_due_ms() -> uint64:
    """Return the in-slot timely cutoff for available-confirmation votes."""
    return get_slot_component_duration_ms(AVAILABLE_CONFIRMATION_DUE_BPS)
```

### New `is_before_available_confirmation_deadline`

```python
def is_before_available_confirmation_deadline(store: Store) -> bool:
    """Return whether current local time is before the available-confirmation timely cutoff."""
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_available_confirmation_due_ms()
```

### New `is_before_attestation_deadline`

```python
def is_before_attestation_deadline(store: Store) -> bool:
    """Return whether current local time is before the attestation deadline."""
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
    # base). TODO(healing): committee sampling is branch-relative (RANDAO); using
    # the justified-checkpoint state is the simplest faithful common base.
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

### New `get_available_confirmation_score`

*Note*: `store.available_timely_attesters[slot]` is the per-slot
time-shifted-quorum freeze for available-confirmation votes. The set is filled
only by current-slot wire votes received before
`AVAILABLE_CONFIRMATION_DUE_BPS`; the available-confirmation rule reads the
previous slot's frozen set, while fast confirmation reads the current slot's
frozen set immediately.

```python
def get_available_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    Return delayed available-confirmation support for ``node`` from the previous
    slot, counting only timely, non-equivocating available attesters.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    previous_slot = Slot(current_slot - 1)
    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot entry.
    if previous_slot not in store.available_votes:
        return uint64(0)
    previous_votes = store.available_votes[previous_slot]
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    # TODO(healing): Stragglers arriving after AVAILABLE_CONFIRMATION_DUE_BPS
    # are not in the frozen timely set and are not counted by the next slot's
    # available confirmation.
    previous_timely = store.available_timely_attesters[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; iterate the previous slot's
    # available committee to resolve seat multiplicity. Equivocators are excluded
    # here (unlike the attestation score, which counts them for viability).
    base_state = store.block_states[store.justified_checkpoint.root]
    previous_committee = get_available_committee(base_state, previous_slot)
    count = uint64(0)
    for member_index in previous_committee:
        if member_index not in previous_timely:
            continue
        if member_index in previous_equivocations:
            continue
        if member_index not in previous_votes:
            continue
        vote = previous_votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, node, message, payload_status):
            count += 1
    return count
```

### New `get_available_confirmation_majority_threshold`

*Note*: The available-confirmation relative quorum must freeze BOTH the
numerator and the denominator over the same time-shifted-quorum (TSQ) set, or
the confirmation outcome would depend on straggler timing and from-block
inclusions and could differ across honest views. This denominator therefore
counts the previous slot's *frozen* electorate — timely, non-equivocating
available attesters — exactly matching the numerator's electorate in
`get_available_confirmation_score`. It is distinct from
`get_available_majority_threshold` (the all-votes threshold gating the Goldfish
head), whose base-branch semantics is unchanged.

```python
def get_available_confirmation_majority_threshold(store: Store) -> uint64:
    """
    [New in Simplex] Return the relative-majority threshold for delayed available
    confirmation over the frozen TSQ electorate: the previous slot's timely,
    non-equivocating available attesters (seat-counted). Numerator and this
    denominator read the same frozen set, so confirmation is straggler-independent.
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
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    # TODO(healing): as in get_available_confirmation_score, only pre-deadline
    # (timely) votes are in the frozen set; stragglers are excluded.
    previous_timely = store.available_timely_attesters[previous_slot]
    base_state = store.block_states[store.justified_checkpoint.root]
    previous_committee = get_available_committee(base_state, previous_slot)
    participant_count = uint64(0)
    for member_index in previous_committee:
        if member_index not in previous_timely:
            continue
        if member_index in previous_equivocations:
            continue
        if member_index not in previous_votes:
            continue
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
    # Numerator and denominator both read the same frozen TSQ electorate (timely,
    # non-equivocating), so confirmation does not depend on straggler timing
    # (see get_available_confirmation_score TODO on stragglers).
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
    [New in Simplex] Return the best filtered child for delayed available
    confirmation.
    """
    children = [
        child
        for child in get_node_children(store, blocks, head)
        if is_in_filtered_block_tree(store, blocks, child)
        and is_available_confirmation_viable(store, child)
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
    from the current slot, counting only timely, non-equivocating available
    attesters.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no current-slot entry.
    if current_slot not in store.available_votes:
        return uint64(0)
    current_votes = store.available_votes[current_slot]
    current_equivocations = store.available_vote_equivocations[current_slot]
    current_timely = store.available_timely_attesters[current_slot]
    # [New in Simplex]
    # Votes are keyed by validator identity; iterate the current slot's
    # available committee to resolve seat multiplicity. Equivocators are
    # excluded from the fast-confirmation score.
    base_state = store.block_states[store.justified_checkpoint.root]
    current_committee = get_available_committee(base_state, current_slot)
    count = uint64(0)
    for member_index in current_committee:
        if member_index not in current_timely:
            continue
        if member_index in current_equivocations:
            continue
        if member_index not in current_votes:
            continue
        vote = current_votes[member_index]
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
    [New in Simplex] Return the immediate fast-confirmation head for the current
    slot, using the current slot's timely available attesters and an absolute
    75% committee-seat threshold.
    """
    blocks = get_filtered_block_tree(store)
    head = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )

    # Fast confirmation. Among filtered fast-viable children pick by
    # confirmation score, then root, then payload-status tiebreaker. At the
    # 75%-absolute threshold, at most one block child can cross.
    while True:
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_in_filtered_block_tree(store, blocks, child)
            and is_fast_confirmation_viable(store, child)
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

### New `update_records`

*Note*: The on-chain SG record layer is fed only from finality attestations
included in blocks (`on_attestation` with `is_from_block=True`); the record head
is the attestation's `beacon_block_root`, so the head field of timeout votes and
empty votes is recorded too. Records are latest-per-validator by attestation
slot. A validator that has two included finality attestations in the same round
with different heads is a same-round record equivocator and is excluded from
both the numerator and the denominator of the record-support arithmetic. In
stage 1 these records are populated but not yet consumed by `get_head`; the
Layer 2 walk that reads them is added in a later stage.

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
    record_round = compute_round_at_slot(slot)

    # TODO(healing): Same-round equivocation is detected only against the
    # currently stored latest record; a full implementation would scan all
    # in-window included attestations.
    for index in attesting_indices:
        if index in store.record_equivocators:
            continue
        previous_record = store.record_votes.get(index)
        if previous_record is not None:
            previous_round = compute_round_at_slot(previous_record.slot)
            if previous_round == record_round and previous_record.head != head:
                store.record_equivocators.add(index)
                del store.record_votes[index]
                continue
        if previous_record is None or slot > previous_record.slot:
            store.record_votes[index] = RecordVote(slot=slot, head=head)
```

### New `is_live_record_validator`

```python
def is_live_record_validator(store: Store, index: ValidatorIndex) -> bool:
    """
    [New in Simplex] Return whether ``index`` has a non-equivocating, in-window
    on-chain record vote for a known block.
    """
    if index not in store.record_votes:
        return False
    if index in store.record_equivocators:
        return False
    record = store.record_votes[index]
    if record.slot + RECORD_WINDOW_SLOTS <= get_current_slot(store):
        return False
    return record.head in store.blocks
```

### New `get_record_weight`

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
    return Gwei(
        sum(
            state.validators[index].effective_balance
            for index in unslashed_and_active_indices
            if (
                is_live_record_validator(store, index)
                and is_supporting_vote(
                    store,
                    node,
                    LatestMessage(
                        slot=store.record_votes[index].slot,
                        root=store.record_votes[index].head,
                    ),
                    PAYLOAD_STATUS_PENDING,
                )
            )
        )
    )
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
layer's separate concern. Any failure returns `None`: the reference is ignored
and the block remains valid.

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

### New `get_iterated_majority_head`

*Note*: Iterated-majority stabilization (Layer 2), replacing a single full-set
majority walk. Each iteration is anchored at the previous iteration's output and
only ever extends it, so a recent committee can push the head deeper but can
never override a block already backed by a wider committee union. The committee
of slot `s` is identified by `latest_message.slot == s` (the slot a validator
voted in). The widest window starts at the first slot of the previous round --
the earliest point at which every validator is guaranteed to have voted, since
the previous round is the last one to have completed -- and each later window
drops the oldest slot's committee, narrowing to the most recent committee.

```python
def get_iterated_majority_head(
    store: Store, blocks: Optional[Dict[Root, BeaconBlock]] = None
) -> ForkChoiceNode:
    """
    [New in Simplex] Iterated-majority stabilization head. Start at the cascade
    root (paper getConfirmed: ``store.justified_checkpoint`` when
    ``store.h_max == store.justified_height + 1``, else
    ``store.finalized_checkpoint``, always viable per paper lem:F-viable), then
    refine the head over successively smaller windows of the most recent
    committees, each walk anchored at (and only extending) the previous output and
    restricted to the viable subtree.
    """
    if blocks is None:
        blocks = get_filtered_block_tree(store)

    if store.h_max == store.justified_height + 1:
        root = store.justified_checkpoint.root
    else:
        root = store.finalized_checkpoint.root
    head = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)

    # The widest window spans the last full round (plus the current partial
    # round): from the first slot of the previous round to now, the earliest
    # point at which every validator is guaranteed to have voted. Each narrower
    # window drops the oldest slot's committee, down to the most recent one.
    current_slot = get_current_slot(store)
    current_round = compute_round_at_slot(current_slot)
    previous_round = GENESIS_ROUND if current_round == GENESIS_ROUND else Round(current_round - 1)
    window = uint64(current_slot - compute_start_slot_at_round(previous_round))

    while window >= 1:
        majority_threshold = get_total_active_voting_weight(store, window) // 2
        while True:
            viable_children = [
                child
                for child in get_node_children(store, blocks, head)
                if is_in_filtered_block_tree(store, blocks, child)
                and get_weight(store, child, window) > majority_threshold
            ]
            if len(viable_children) == 0:
                break
            head = viable_children[0]
        window = uint64(window - 1)
    return head
```

### New `get_available_confirmation_head`

*Note*: Called by `on_tick_per_slot` at `AVAILABLE_CONFIRMATION_DUE_BPS` to
maintain `store.latest_confirmed_head`.

```python
def get_available_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    [New in Simplex] Return the delayed available-confirmation head for slot ``n``
    when called in slot ``n+1``, using timely previous-slot available attesters.
    This tracks the availability-confirmed chain; validators separately apply
    the height-filter bound before using the result for finality/stabilization
    voting.
    """
    blocks = get_filtered_block_tree(store)
    # [Modified in Simplex]
    # Decoupled from SG/record state (E5): user-facing confirmation is floorless
    # and never gated by FG or record state, so it starts at finalized.
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
Only unexpired, non-equivocating supporting latest messages contribute, so the
numerator here is always a subset of the `get_total_active_voting_weight`
denominator.

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

*Note*: Staged fork choice: (1) majority-gated LMD-GHOST with viable-subtree
filter and two-way cascade between `store.justified_checkpoint` and
`store.finalized_checkpoint` (paper getConfirmed), then (2) Goldfish walk using
previous-slot available attestations.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # [Modified in Simplex]
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)

    # Layer 2: majority-gated LMD-GHOST
    # TODO(stage2): replace the iterated-majority stabilization with the record
    # walk from the anchor (pointed fresh quorum if valid, else the 2/3 record
    # descent get_record_support/get_record_weight added in the record layer).
    head = get_iterated_majority_head(store, blocks)

    # Layer 3: Goldfish fork-choice using available attestations
    while True:
        children = get_node_children(store, blocks, head)
        viable_children = [
            child
            for child in children
            if is_in_filtered_block_tree(store, blocks, child)
            and is_available_attestation_viable(store, child)
        ]
        if len(viable_children) == 0:
            return head
        head = max(
            viable_children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
```

### Modified `validate_on_attestation`

*Note*: Attestations are epoch-bounded (current or previous epoch). Wire
attestations assert; from-block attestations skip silently in `on_attestation`.
Wire justification votes must name a known real target. A from-block attestation
whose target is unknown may still feed on-chain records if its head is known,
but it does not update `latest_messages`. Timeout votes and empty votes use
`Checkpoint()` as the target and still carry a head vote.

```python
def validate_on_attestation(store: Store, attestation: Attestation, is_from_block: bool) -> None:
    data = attestation.data
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    if data.target != Checkpoint():
        # [Modified in Simplex]
        # A wire justification target (and a from-block one whose block is known)
        # must name a known real block at its actual proposal slot; a from-block
        # target on another fork is tolerated so its head record still flows.
        if not is_from_block or data.target.root in store.blocks:
            assert data.target.root in store.blocks
            assert store.blocks[data.target.root].slot == data.target.slot
        # Target slot may precede attestation slot (height-based finality)
        assert data.target.slot <= data.slot

    # Attestations can only affect fork choice of subsequent slots.
    # Delay consideration in the fork-choice until their slot is in the past.
    assert get_current_slot(store) >= data.slot + 1

    # [Modified in Simplex]
    # Epoch-bounded: attestation slot must be in current or previous epoch.
    if not is_from_block:
        current_epoch = get_current_store_epoch(store)
        previous_epoch = (
            GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)
        )
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
        store.payload_votes[current_slot] = {}
        store.payload_vote_equivocations[current_slot] = set()
        store.available_votes[current_slot] = {}
        store.available_vote_equivocations[current_slot] = set()
        store.available_timely_attesters[current_slot] = set()

    # [New in Simplex]
    # Confirmation rules: once local time is past the confirmation deadline,
    # record the available-confirmed head from the previous slot's frozen
    # available votes and the fast-confirmed head from the current slot's frozen
    # available votes. Idempotent across ticks within the same slot.
    if not is_before_available_confirmation_deadline(store):
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
via `update_records`. The `latest_messages` helper state is still updated when
the finality target path is known. A from-block attestation whose finality
target is unknown (the voter may have voted for a block on a different fork) may
still feed its head field into the record layer, but it does not update
`latest_messages`.

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """[Modified in Simplex]"""
    data = attestation.data
    # Skip from-block attestations whose head vote references an unknown block
    # (voter may have voted for a block on a different fork)
    if is_from_block and data.beacon_block_root not in store.blocks:
        return
    # [New in Simplex]
    # Skip from-block attestations from old epochs.
    if is_from_block:
        current_epoch = get_current_store_epoch(store)
        previous_epoch = (
            GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)
        )
        attestation_epoch = compute_epoch_at_slot(data.slot)
        if attestation_epoch not in (current_epoch, previous_epoch):
            return
    # [New in Simplex]
    # A from-block justification vote whose finality target is unknown still feeds
    # its head field into the record layer, but does not update latest_messages.
    target_unknown = (
        is_from_block and data.target != Checkpoint() and data.target.root not in store.blocks
    )
    validate_on_attestation(store, attestation, is_from_block)

    # Derive the checkpoint state for signature verification and attesting
    # indices (epoch boundary on the attestation's own head chain).
    target_state = get_attestation_checkpoint_state(store, data)

    if not is_from_block:
        # Verify signature against beacon committee
        assert is_valid_indexed_attestation(
            target_state, get_indexed_attestation(target_state, attestation)
        )

    attesting_indices = get_attesting_indices(target_state, attestation)
    # [New in Simplex]
    # Feed on-chain records from block-included finality attestations.
    if is_from_block:
        update_records(store, attestation, sorted(attesting_indices))
    if target_unknown:
        return
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

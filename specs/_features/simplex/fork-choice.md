# Simplex Finality -- Fork Choice

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Containers](#containers)
  - [Modified `Store`](#modified-store)
- [Helper functions](#helper-functions)
  - [Modified `get_forkchoice_store`](#modified-get_forkchoice_store)
  - [New `are_non_conflicting`](#new-are_non_conflicting)
  - [New `should_update_justified`](#new-should_update_justified)
  - [New `update_checkpoints`](#new-update_checkpoints)
  - [New `get_total_active_voting_weight`](#new-get_total_active_voting_weight)
  - [New `get_view_freeze_due_ms`](#new-get_view_freeze_due_ms)
  - [New `is_before_view_freeze_deadline`](#new-is_before_view_freeze_deadline)
  - [New `get_available_confirmation_due_ms`](#new-get_available_confirmation_due_ms)
  - [New `is_before_available_confirmation_deadline`](#new-is_before_available_confirmation_deadline)
  - [New `is_before_attestation_deadline`](#new-is_before_attestation_deadline)
  - [New `is_ptc_decision_node`](#new-is_ptc_decision_node)
  - [New `get_available_majority_threshold`](#new-get_available_majority_threshold)
  - [New `get_available_attestation_score`](#new-get_available_attestation_score)
  - [New `is_available_attestation_viable`](#new-is_available_attestation_viable)
  - [New `get_available_confirmation_score`](#new-get_available_confirmation_score)
  - [New `is_available_confirmation_viable`](#new-is_available_confirmation_viable)
  - [New `get_lmd_ghost_head`](#new-get_lmd_ghost_head)
  - [New `get_available_confirmation_head`](#new-get_available_confirmation_head)
  - [New `get_payload_participant_count`](#new-get_payload_participant_count)
  - [New `get_payload_full_support`](#new-get_payload_full_support)
  - [New `get_payload_data_available_support`](#new-get_payload_data_available_support)
  - [Modified `is_payload_timely`](#modified-is_payload_timely)
  - [Modified `is_payload_data_available`](#modified-is_payload_data_available)
  - [Modified `should_extend_payload`](#modified-should_extend_payload)
  - [Modified `update_latest_messages`](#modified-update_latest_messages)
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
  - [Modified `filter_block_tree`](#modified-filter_block_tree)
  - [Modified `get_filtered_block_tree`](#modified-get_filtered_block_tree)
  - [Modified `is_finalization_ok`](#modified-is_finalization_ok)
  - [Modified `validate_target_epoch_against_current_time`](#modified-validate_target_epoch_against_current_time)

<!-- mdformat-toc end -->

## Introduction

This is the fork choice specification for simplex-based finality. It modifies
the fork choice to use the justified and finalized checkpoints from the
two-round simplex finality gadget instead of Casper FFG, and removes the
unrealized justification/finalization machinery.

*Note*: This specification is built upon [Gloas](../../gloas/fork-choice.md).

## Containers

### Modified `Store`

*Note*: The fields `unrealized_justified_checkpoint`,
`unrealized_finalized_checkpoint`, and `unrealized_justifications` are removed.
The fields `proposer_boost_root`, `payload_timeliness_vote`, and
`payload_data_availability_vote` are also removed. The `justified_height` field
is added for height-based tie-breaking. Goldfish vote synchronization uses
first-vote + equivocation-vote tracking for both available attestations and
payload votes.

*Note*: Only votes from the previous and current slot are ever needed by fork
choice. Implementations may prune entries for older slots.

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: (
        Checkpoint  # [Modified in Simplex] simplex finality justified
    )
    finalized_checkpoint: Checkpoint
    justified_height: Height  # [New in Simplex]
    has_conflicting_justification: bool  # [New in Simplex] Conditional fork-choice filter
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    execution_payload_states: Dict[Root, BeaconState] = field(default_factory=dict)
    # [Modified in Simplex] PTC first-seen vote per member; ``PayloadAttestationData()`` = missing.
    # Keyed by slot (not by block root as in Gloas) for consistency with the per-slot Goldfish
    # tracking pattern. Implementations may prune entries older than the previous slot.
    payload_votes: Dict[Slot, Vector[PayloadAttestationData, PTC_SIZE]] = field(
        default_factory=dict
    )
    payload_vote_equivocations: Dict[Slot, Vector[boolean, PTC_SIZE]] = field(default_factory=dict)
    # [New in Simplex] Goldfish: per-slot available-attestation tracking.
    # Keyed by slot; ``AvailableAttestationData()`` = missing vote.
    available_votes: Dict[Slot, Vector[AvailableAttestationData, AVAILABLE_COMMITTEE_SIZE]] = field(
        default_factory=dict
    )
    available_vote_equivocations: Dict[Slot, Vector[boolean, AVAILABLE_COMMITTEE_SIZE]] = field(
        default_factory=dict
    )
    available_timely_attesters: Dict[Slot, Vector[boolean, AVAILABLE_COMMITTEE_SIZE]] = field(
        default_factory=dict
    )
```

## Helper functions

### Modified `get_forkchoice_store`

*Note*: Sentinel payload-vote initialization (`PayloadAttestationData()`) means
that at anchor there is no initial strict-majority payload support; thus the
first post-anchor payload decision resolves through the EMPTY/FULL tiebreak path
until PTC votes are recorded.

```python
def get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock) -> Store:
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    justified_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    finalized_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    anchor_slot = anchor_state.slot
    return Store(
        time=uint64(anchor_state.genesis_time + SECONDS_PER_SLOT * anchor_slot),
        genesis_time=anchor_state.genesis_time,
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=finalized_checkpoint,
        justified_height=anchor_state.justified_height,  # [New in Simplex]
        has_conflicting_justification=False,
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={},
        execution_payload_states={anchor_root: copy(anchor_state)},
        payload_votes={anchor_slot: [PayloadAttestationData() for _ in range(PTC_SIZE)]},
        payload_vote_equivocations={anchor_slot: [False] * PTC_SIZE},
        available_votes={
            anchor_slot: [AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)]
        },
        available_vote_equivocations={anchor_slot: [False] * AVAILABLE_COMMITTEE_SIZE},
        available_timely_attesters={anchor_slot: [False] * AVAILABLE_COMMITTEE_SIZE},
    )
```

### New `are_non_conflicting`

```python
def are_non_conflicting(store: Store, a: Checkpoint, b: Checkpoint) -> bool:
    """
    [New in Simplex] Return whether checkpoints ``a`` and ``b`` are on the same
    chain (one is an ancestor of the other in the block tree). Used by
    ``update_checkpoints`` to apply the store-level max: when two justified
    checkpoints are non-conflicting, the higher-slot one is kept.
    """
    if a.slot <= b.slot:
        return get_ancestor(store, b.root, a.slot).root == a.root
    else:
        return get_ancestor(store, a.root, b.slot).root == b.root
```

### New `should_update_justified`

```python
def should_update_justified(
    current: Checkpoint,
    current_height: Height,
    candidate: Checkpoint,
    candidate_height: Height,
) -> bool:
    """
    Determine if candidate should replace current justified checkpoint.
    Higher height wins, then higher slot, then root as deterministic tiebreaker.
    """
    candidate_with_height = (candidate_height, candidate.slot, candidate.root)
    current_with_height = (current_height, current.slot, current.root)
    return candidate_with_height > current_with_height
```

### New `update_checkpoints`

```python
def update_checkpoints(
    store: Store,
    justified_checkpoint: Checkpoint,
    justified_height: Height,
    finalized_checkpoint: Checkpoint,
) -> None:
    old_justified_height = store.justified_height

    # [Modified in Simplex] Store-level max: when the candidate and current justified
    # are on the same chain (non-conflicting), keep the higher-slot checkpoint. This
    # prevents cross-chain slot regression when different chains justify at different
    # heights with ancestrally-related checkpoints (descendant-based justification can
    # justify ancestors). When conflicting (different forks), use should_update_justified.
    if are_non_conflicting(store, store.justified_checkpoint, justified_checkpoint):
        if justified_checkpoint.slot > store.justified_checkpoint.slot:
            store.justified_checkpoint = justified_checkpoint
        store.justified_height = max(store.justified_height, justified_height)
    else:
        # [New in Simplex] Conflicting checkpoint at the same justified height:
        # set the filter flag regardless of tiebreaker outcome, so all nodes
        # converge on the same flag state.
        if justified_height == store.justified_height:
            store.has_conflicting_justification = True
        if should_update_justified(
            store.justified_checkpoint,
            store.justified_height,
            justified_checkpoint,
            justified_height,
        ):
            store.justified_checkpoint = justified_checkpoint
            store.justified_height = justified_height

    # [New in Simplex] Clear the conflict flag when justified_height advances
    # past the conflicting height. Normal unfiltered operation resumes.
    if store.justified_height > old_justified_height:
        store.has_conflicting_justification = False

    # [New in Simplex] Only advance finalized if justified descends from it,
    # as a defense-in-depth guard. Under f < n/3 this always passes (quorum
    # intersection: 2/3 + 2/3 - 1 = 1/3 > f ensures the F <= J invariant
    # holds at the state level); the guard prevents deadlock under >= n/3
    # equivocators.
    if finalized_checkpoint.slot > store.finalized_checkpoint.slot:
        if (
            get_ancestor(store, store.justified_checkpoint.root, finalized_checkpoint.slot).root
            == finalized_checkpoint.root
        ):
            store.finalized_checkpoint = finalized_checkpoint
```

### New `get_total_active_voting_weight`

```python
def get_total_active_voting_weight(store: Store) -> Gwei:
    """
    Return the total effective balance of unslashed active validators that have
    non-equivocating ``latest_messages``.
    Used as the relative-majority denominator for Layer 2.
    """
    state = store.block_states[store.justified_checkpoint.root]
    participating_indices: Set[ValidatorIndex] = set()
    for index in get_active_validator_indices(state, get_current_epoch(state)):
        if state.validators[index].slashed:
            continue
        if index not in store.latest_messages:
            continue
        if index in store.equivocating_indices:
            continue
        participating_indices.add(index)

    return get_total_balance(state, participating_indices)
```

### New `get_view_freeze_due_ms`

```python
def get_view_freeze_due_ms(epoch: Epoch) -> uint64:
    """
    Return the in-slot vote-freeze boundary for view-merge.
    ``epoch`` is unused but kept for consistency with the Gloas slot-component pattern.
    """
    return get_slot_component_duration_ms(VIEW_FREEZE_DUE_BPS)
```

### New `is_before_view_freeze_deadline`

```python
def is_before_view_freeze_deadline(store: Store) -> bool:
    """
    Return whether current local time is before the view-merge vote-freeze boundary.
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_view_freeze_due_ms(get_current_store_epoch(store))
```

### New `get_available_confirmation_due_ms`

```python
def get_available_confirmation_due_ms(epoch: Epoch) -> uint64:
    """
    Return the in-slot timely cutoff for available-confirmation votes.
    ``epoch`` is unused but kept for consistency with the Gloas slot-component pattern.
    """
    return get_slot_component_duration_ms(AVAILABLE_CONFIRMATION_DUE_BPS)
```

### New `is_before_available_confirmation_deadline`

```python
def is_before_available_confirmation_deadline(store: Store) -> bool:
    """
    Return whether current local time is before the available-confirmation
    timely cutoff in the slot.
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_available_confirmation_due_ms(get_current_store_epoch(store))
```

### New `is_before_attestation_deadline`

```python
def is_before_attestation_deadline(store: Store) -> bool:
    """
    Return whether current local time is before the attestation deadline in the
    slot.
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_attestation_due_ms(get_current_store_epoch(store))
```

### New `is_ptc_decision_node`

```python
def is_ptc_decision_node(store: Store, node: ForkChoiceNode) -> bool:
    """
    Return whether ``node`` is a previous-slot payload decision (EMPTY/FULL),
    which defers to ``get_payload_status_tiebreaker`` instead of score-based ranking.
    """
    return node.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        node.root
    ].slot + 1 == get_current_slot(store)
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
    previous_votes = store.available_votes[previous_slot]
    missing = AvailableAttestationData()
    participant_count = uint64(len([v for v in previous_votes if v != missing]))
    return participant_count // 2
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
    previous_votes = store.available_votes[previous_slot]
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    missing_available_vote = AvailableAttestationData()
    score = uint64(0)
    for member_index in range(len(previous_votes)):
        if previous_equivocations[member_index]:
            score += 1  # Equivocator counted for viability
            continue
        vote = previous_votes[member_index]
        if vote != missing_available_vote:
            message = LatestMessage(
                slot=vote.slot,
                root=vote.beacon_block_root,
                payload_present=vote.payload_present,
            )
            if is_supporting_vote(store, child, message):
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
    previous_votes = store.available_votes[previous_slot]
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    previous_timely = store.available_timely_attesters[previous_slot]
    missing_available_vote = AvailableAttestationData()
    count = uint64(0)
    for member_index in range(len(previous_votes)):
        if not previous_timely[member_index]:
            continue
        vote = previous_votes[member_index]
        if vote == missing_available_vote:
            continue
        if previous_equivocations[member_index]:
            continue
        message = LatestMessage(
            slot=vote.slot,
            root=vote.beacon_block_root,
            payload_present=vote.payload_present,
        )
        if is_supporting_vote(store, node, message):
            count += 1
    return count
```

### New `is_available_confirmation_viable`

```python
def is_available_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in delayed available confirmation:
    PTC decision nodes always pass through; other children require
    available-confirmation score exceeding the majority threshold.
    """
    if is_ptc_decision_node(store, child):
        return True
    return get_available_confirmation_score(store, child) > get_available_majority_threshold(store)
```

### New `get_lmd_ghost_head`

```python
def get_lmd_ghost_head(store: Store) -> ForkChoiceNode:
    """
    Return the majority-gated LMD-GHOST head (Layer 2). Walks store.blocks from
    the justified checkpoint, advancing through the unique child with
    strict-majority weight at each depth.
    """
    # [New in Simplex] Walk store directly; no permanent filter.
    # Conditional filter: when conflicting justified checkpoints are detected
    # at the same height, prefer children whose state has advanced past the
    # justified height. This restricts the fork-choice to chains that have
    # demonstrated progress, preventing locked validators from being leaked
    # on a stuck chain. Falls back to all children if none have advanced.
    blocks = store.blocks
    head = ForkChoiceNode(
        root=store.justified_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    majority_threshold = get_total_active_voting_weight(store) // 2

    while True:
        children = get_node_children(store, blocks, head)

        # [New in Simplex] Conditional filter on conflicting justifications
        if store.has_conflicting_justification:
            advanced_children = [
                child for child in children
                if store.block_states[child.root].current_height > store.justified_height
            ]
            if len(advanced_children) > 0:
                children = advanced_children

        viable_children = [
            child
            for child in children
            if get_weight(store, child) > majority_threshold
        ]
        if len(viable_children) == 0:
            return head
        head = viable_children[0]
```

### New `get_available_confirmation_head`

*Note*: This helper is currently an external/research signal (no consensus
state-transition caller in simplex at this stage).

```python
def get_available_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    Return the delayed available-confirmation head for slot ``n`` when called in
    slot ``n+1``, using timely previous-slot available attesters.
    """
    head = get_lmd_ghost_head(store)

    # Delayed available confirmation (TSQ: at most one viable child per depth)
    while True:
        children = get_node_children(store, store.blocks, head)
        viable_children = [
            child for child in children if is_available_confirmation_viable(store, child)
        ]
        if len(viable_children) == 0:
            return head
        head = viable_children[0]
```

### New `get_payload_participant_count`

```python
def get_payload_participant_count(store: Store, root: Root) -> uint64:
    """
    Return the participant count for payload-vote majority gating.
    """
    vote_slot = store.blocks[root].slot
    payload_votes = store.payload_votes.get(vote_slot, [])
    return uint64(len([vote for vote in payload_votes if vote != PayloadAttestationData()]))
```

### New `get_payload_full_support`

```python
def get_payload_full_support(store: Store, root: Root) -> uint64:
    """
    Return payload FULL support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``payload_present == True`` count.
    Equivocating participants in the slot are included for viability.
    """
    vote_slot = store.blocks[root].slot
    payload_votes = store.payload_votes.get(vote_slot, [])
    equivocations = store.payload_vote_equivocations.get(vote_slot, [])
    missing_payload_vote = PayloadAttestationData()
    full_support_count = uint64(0)
    for ptc_member_index in range(len(payload_votes)):
        vote = payload_votes[ptc_member_index]
        if vote == missing_payload_vote:
            continue
        if equivocations[ptc_member_index] or (
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
    vote_slot = store.blocks[root].slot
    payload_votes = store.payload_votes.get(vote_slot, [])
    equivocations = store.payload_vote_equivocations.get(vote_slot, [])
    missing_payload_vote = PayloadAttestationData()
    data_available_support_count = uint64(0)
    for ptc_member_index in range(len(payload_votes)):
        vote = payload_votes[ptc_member_index]
        if vote == missing_payload_vote:
            continue
        if equivocations[ptc_member_index] or (
            vote.beacon_block_root == root and vote.blob_data_available
        ):
            data_available_support_count += 1
    return data_available_support_count
```

### Modified `is_payload_timely`

```python
def is_payload_timely(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload FULL support.
    """
    # [Modified in Simplex] strict relative-majority over the
    # active payload vote set (with equivocation inclusion in support helper).
    if root not in store.execution_payload_states:
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
    # [Modified in Simplex] strict relative-majority over the
    # active payload vote set (with equivocation inclusion in support helper).
    if root not in store.execution_payload_states:
        return False

    participant_count = get_payload_participant_count(store, root)
    data_available_support_count = get_payload_data_available_support(store, root)
    return data_available_support_count > participant_count // 2
```

### Modified `should_extend_payload`

*Note*: Payload extension is strict and proposer-independent:

1. local payload availability is required, and
2. the payload FULL vote must pass strict relative majority after including
   equivocation count for viability, and
3. the payload data-availability vote must pass the same strict majority check.

```python
def should_extend_payload(store: Store, root: Root) -> bool:
    # [Modified in Simplex] The PTC majority requirement is now
    #  strict, removing the exception if a new block proposal extended the payload.
    return is_payload_timely(store, root) and is_payload_data_available(store, root)
```

### Modified `update_latest_messages`

*Note*: Updated to accept `Attestation` (finality attestations carry
`beacon_block_root` and `payload_present` for LMD-GHOST fork choice).

```python
def update_latest_messages(
    store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation
) -> None:
    # [Modified in Simplex] Uses Attestation with beacon_block_root and payload_present
    slot = attestation.data.slot
    beacon_block_root = attestation.data.beacon_block_root
    payload_present = attestation.data.payload_present
    non_equivocating_attesting_indices = [
        i for i in attesting_indices if i not in store.equivocating_indices
    ]
    for i in non_equivocating_attesting_indices:
        if i not in store.latest_messages or slot > store.latest_messages[i].slot:
            store.latest_messages[i] = LatestMessage(
                slot=slot,
                root=beacon_block_root,
                payload_present=payload_present,
            )
```

### Modified `get_weight`

*Note*: Returns 0 for previous-slot payload-decision nodes (EMPTY/FULL),
deferring the payload decision to `get_payload_status_tiebreaker`. Proposer
boost is not used in simplex.

```python
def get_weight(store: Store, node: ForkChoiceNode) -> Gwei:
    # [Modified in Simplex] Defer previous-slot payload decisions
    # to ``get_payload_status_tiebreaker``.
    if is_ptc_decision_node(store, node):
        return Gwei(0)
    state = store.block_states[store.justified_checkpoint.root]
    return get_attestation_score(store, node, state)
```

### Modified `get_head`

*Note*: `get_head` implements a staged fork choice:

1. **Majority stage**: Start from the justified checkpoint, run LMD-GHOST over
   `store.blocks` using `latest_messages` (from finality attestations),
   requiring
   > 50% of participating voting weight to proceed. Stop when no child has
   > majority support.
2. **Goldfish stage**: From where the majority stage stopped, run previous-slot
   available-attestation voting with:
   - viability gate (score > majority threshold, with PTC decision-node and
     current-slot proposal pass-throughs), and
   - plurality winner by available-attestation score among the resulting
     children.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # [Modified in Simplex] Layer 2: majority-gated LMD-GHOST
    head = get_lmd_ghost_head(store)

    # [New in Simplex] Layer 3: Goldfish fork-choice
    # using available attestations
    while True:
        children = get_node_children(store, store.blocks, head)
        viable_children = [
            child for child in children if is_available_attestation_viable(store, child)
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

*Note*: All attestations (wire and from-block) are epoch-bounded by slot epoch
(current/previous epoch). Wire attestations assert; from-block attestations
skip silently (in `on_attestation`). Under the IC model with descendant-based
justification, honest validators alone (locked + non-locked > 2n/3) always
suffice for justification via the suffix-sum — transferring old adversary
votes across chains is unnecessary for liveness.

```python
def validate_on_attestation(store: Store, attestation: Attestation, is_from_block: bool) -> None:
    data = attestation.data
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    # [Modified in Simplex] Target must be a real checkpoint (IC consensus: no timeout votes)
    assert data.target != Checkpoint()
    # Target must be for a known block at its actual proposal slot
    assert data.target.root in store.blocks
    assert store.blocks[data.target.root].slot == data.target.slot
    # [Modified in Simplex] target slot may be older than attestation
    # slot (height-based finality), but it cannot be from the future.
    assert data.target.slot <= data.slot
    # Same-slot attestation cannot signal payload availability
    # (PTC does the first payload availability determination)
    if block_slot == data.slot:
        assert not data.payload_present

    # Attestations can only affect fork choice of subsequent slots.
    # Delay consideration in the fork-choice until their slot is in the past.
    assert get_current_slot(store) >= data.slot + 1

    # [Modified in Simplex] Epoch-bounded: attestation slot must be in
    # current or previous epoch. Under IC consensus with descendant-based
    # justification, honest validators alone provide >= 2/3 for the
    # suffix-sum — old adversary votes from other chains are not needed
    # for liveness. For from-block attestations this is enforced in
    # on_attestation (skip, not assert).
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

*Note*: The epoch boundary unrealized checkpoint pull-up is removed.

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    # [Modified in Simplex] Removed epoch boundary unrealized checkpoint pull-up
    previous_slot = get_current_slot(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        # [New in Simplex] Initialize per-slot vote tracking for the new slot.
        store.payload_votes[current_slot] = [PayloadAttestationData() for _ in range(PTC_SIZE)]
        store.payload_vote_equivocations[current_slot] = [False] * PTC_SIZE
        store.available_votes[current_slot] = [
            AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ]
        store.available_vote_equivocations[current_slot] = [False] * AVAILABLE_COMMITTEE_SIZE
        store.available_timely_attesters[current_slot] = [False] * AVAILABLE_COMMITTEE_SIZE
```

### Modified `on_block`

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    assert block.parent_root in store.block_states

    # Check if this block builds on empty or full parent block
    parent_block = store.blocks[block.parent_root]
    bid = block.body.signed_execution_payload_bid.message
    parent_bid = parent_block.body.signed_execution_payload_bid.message
    if is_parent_node_full(store, block):
        assert block.parent_root in store.execution_payload_states
        state = copy(store.execution_payload_states[block.parent_root])
    else:
        assert bid.parent_block_hash == parent_bid.parent_block_hash
        state = copy(store.block_states[block.parent_root])

    current_slot = get_current_slot(store)
    assert current_slot >= block.slot

    assert block.slot > store.finalized_checkpoint.slot
    assert (
        store.finalized_checkpoint.root
        == get_ancestor(store, block.parent_root, store.finalized_checkpoint.slot).root
    )

    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)

    store.blocks[block_root] = block
    store.block_states[block_root] = state

    notify_ptc_messages(store, state, block.body.payload_attestations)

    # [Modified in Simplex] Process finality attestations for fork choice (update latest_messages)
    for attestation in block.body.attestations:
        on_attestation(store, attestation, is_from_block=True)

    # [Modified in Simplex] Process available attestations for Goldfish (per-slot vote tracking)
    for available_attestation in block.body.available_attestations:
        on_available_attestation(store, available_attestation, is_from_block=True)

    # [Modified in Simplex] Update checkpoints with height, no unrealized pull-up
    update_checkpoints(
        store, state.justified_checkpoint, state.justified_height, state.finalized_checkpoint
    )
```

### Modified `on_payload_attestation_message`

*Note*: Payload votes use first-vote + equivocation-vote tracking with
view-merge freeze handling:

1. non-proposer wire votes after freeze are ignored;
2. block-carried votes are always processed;
3. the next proposer may continue collecting wire votes after freeze by calling
   with `is_next_proposer=True`;
4. only the active payload-vote window is considered:
   - before attestation deadline: previous-slot PTC votes,
   - after attestation deadline: current-slot PTC votes;
5. a second distinct vote marks equivocation and removes non-equivocating
   support.

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
        # [Modified in Simplex] view-freeze gating with
        # proposer-only late-wire collection for view-merge.
        # Wire votes are accepted only for the current slot.
        # Non-proposers enforce freeze; the next proposer may override via
        # ``is_next_proposer``.
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
    ptc_member_index = ptc.index(ptc_message.validator_index)

    # Ignore additional votes after the first equivocation.
    if equivocations[ptc_member_index]:
        return

    missing_payload_vote = PayloadAttestationData()
    first_vote = payload_votes[ptc_member_index]
    if first_vote == missing_payload_vote:
        payload_votes[ptc_member_index] = data
        return

    if first_vote != data:
        equivocations[ptc_member_index] = True
```

### Modified `on_attestation`

*Note*: Finality attestations now update `latest_messages` for the majority fork
choice layer. `AttestationData` carries `beacon_block_root` (LMD head vote) and
`payload_present` (payload availability signal).

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """
    [Modified in Simplex] Finality attestations update latest_messages
    for the majority fork choice layer.
    """
    # Skip from-block attestations whose head vote references an unknown block
    # (voter may have voted for a block on a different fork)
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        return
    # Skip from-block attestations whose finality target references an unknown block
    # (voter may have voted for a block on a different fork).
    if is_from_block and attestation.data.target.root not in store.blocks:
        return
    # [New in Simplex] Skip from-block attestations from old epochs.
    # Under IC consensus with descendant-based justification, honest
    # validators provide >= 2/3 via the suffix-sum — transferring old
    # adversary votes across chains is unnecessary for liveness.
    if is_from_block:
        current_epoch = get_current_store_epoch(store)
        previous_epoch = (
            GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)
        )
        attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
        if attestation_epoch not in (current_epoch, previous_epoch):
            return
    validate_on_attestation(store, attestation, is_from_block)

    # Derive checkpoint state for signature verification and attesting indices.
    # Committee membership is epoch-based; use epoch boundary slot for Checkpoint.
    attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    epoch_root = get_ancestor(store, attestation.data.beacon_block_root, epoch_boundary_slot).root
    checkpoint = Checkpoint(slot=epoch_boundary_slot, root=epoch_root)

    if checkpoint not in store.checkpoint_states:
        base_state = copy(store.block_states[epoch_root])
        epoch_start_slot = compute_start_slot_at_epoch(attestation_epoch)
        if base_state.slot < epoch_start_slot:
            process_slots(base_state, epoch_start_slot)
        store.checkpoint_states[checkpoint] = base_state

    target_state = store.checkpoint_states[checkpoint]

    if not is_from_block:
        # Verify signature against beacon committee
        assert is_valid_indexed_attestation(
            target_state, get_indexed_attestation(target_state, attestation)
        )

    attesting_indices = get_attesting_indices(target_state, attestation)
    update_latest_messages(store, sorted(attesting_indices), attestation)
```

### New `on_available_attestation`

*Note*: Available attestations track per-slot per-committee-member votes for the
Goldfish fork choice layer, instead of updating `latest_messages`. Non-proposers
ignore wire votes after freeze; proposers may continue collecting wire votes
after freeze with `is_next_proposer=True`. Equivocation tracking uses the first
vote / second vote pattern (like PTC).

```python
def on_available_attestation(
    store: Store,
    attestation: AvailableAttestation,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    """
    [New in Simplex] Available attestations store per-slot
    per-committee-member votes for the Goldfish fork choice layer.
    """
    # Skip from-block attestations whose head vote references an unknown block
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        return

    if not is_from_block and not is_next_proposer and not is_before_view_freeze_deadline(store):
        # Late wire vote: ignored for view-merge.
        return

    validate_on_available_attestation(store, attestation, is_from_block)

    if not is_from_block:
        # Derive checkpoint state for signature verification.
        # Committee membership is epoch-based; use epoch boundary slot for Checkpoint.
        attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
        epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
        epoch_root = get_ancestor(
            store, attestation.data.beacon_block_root, epoch_boundary_slot
        ).root
        checkpoint = Checkpoint(slot=epoch_boundary_slot, root=epoch_root)

        if checkpoint not in store.checkpoint_states:
            base_state = copy(store.block_states[epoch_root])
            epoch_start_slot = compute_start_slot_at_epoch(attestation_epoch)
            if base_state.slot < epoch_start_slot:
                process_slots(base_state, epoch_start_slot)
            store.checkpoint_states[checkpoint] = base_state

        target_state = store.checkpoint_states[checkpoint]

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

    missing_available_vote = AvailableAttestationData()
    for member_index in range(len(attestation.aggregation_bits)):
        if not attestation.aggregation_bits[member_index]:
            continue
        first_vote = available_votes[member_index]
        is_equivocating = available_vote_equivocations[member_index]
        # Ignore further equivocations
        if is_equivocating:
            continue
        if first_vote == missing_available_vote:
            # First vote from this committee member for this slot
            available_votes[member_index] = attestation.data
            if (
                vote_slot == current_slot
                and not is_from_block
                and is_before_available_confirmation_deadline(store)
            ):
                available_timely_attesters[member_index] = True
        elif first_vote != attestation.data:
            # Second (different) vote — record as equivocation
            available_vote_equivocations[member_index] = True
```

## Deprecated overrides

*Note*: The following functions are overridden as no-ops or trivial returns
because their underlying mechanisms (Casper FFG unrealized justifications,
proposer boost, block timeliness) are removed in simplex. They exist
only to shadow inherited implementations that reference removed Store fields.

### Modified `get_voting_source`

```python
def get_voting_source(store: Store, block_root: Root) -> Checkpoint:
    # [Modified in Simplex] No unrealized justification pull-up.
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
    # [Modified in Simplex] No unrealized checkpoints.
    pass
```

### Modified `compute_pulled_up_tip`

```python
def compute_pulled_up_tip(store: Store, block_root: Root) -> None:
    # [Modified in Simplex] No pull-up needed.
    pass
```

### Modified `record_block_timeliness`

```python
def record_block_timeliness(store: Store, root: Root) -> None:
    # [Modified in Simplex] Block timeliness tracking removed.
    pass
```

### Modified `update_proposer_boost_root`

```python
def update_proposer_boost_root(store: Store, root: Root) -> None:
    # [Modified in Simplex] Proposer boost removed.
    pass
```

### Modified `is_head_late`

```python
def is_head_late(store: Store, head_root: Root) -> bool:
    # [Modified in Simplex] Block timeliness tracking removed.
    return False
```

### Modified `is_ffg_competitive`

```python
def is_ffg_competitive(store: Store, head_root: Root, parent_root: Root) -> bool:
    # [Modified in Simplex] Unrealized justifications removed.
    return True
```

### Modified `is_head_weak`

```python
def is_head_weak(store: Store, head_root: Root) -> bool:
    # [Modified in Simplex] Proposer-boost path removed.
    return False
```

### Modified `is_parent_strong`

```python
def is_parent_strong(store: Store, root: Root) -> bool:
    # [Modified in Simplex] Proposer-boost path removed.
    return True
```

### Modified `should_apply_proposer_boost`

```python
def should_apply_proposer_boost(store: Store) -> bool:
    # [Modified in Simplex] Proposer boost disabled.
    return False
```

### Modified `should_override_forkchoice_update`

```python
def should_override_forkchoice_update(store: Store, head_root: Root) -> bool:
    # [Modified in Simplex] Override path removed.
    return False
```

### Modified `get_proposer_head`

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    # [Modified in Simplex] Proposer override removed.
    return head_root
```

### Modified `filter_block_tree`

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    # [Modified in Simplex] Not used — block-based checkpoints eliminate
    # the round-boundary ambiguity that required filtering.
    return True
```

### Modified `get_filtered_block_tree`

```python
def get_filtered_block_tree(store: Store) -> Dict[Root, BeaconBlock]:
    # [Modified in Simplex] Not used — block-based checkpoints eliminate
    # the round-boundary ambiguity that required filtering.
    return store.blocks
```

### Modified `is_finalization_ok`

```python
def is_finalization_ok(store: Store, slot: Slot) -> bool:
    # [Modified in Simplex] Not used — proposer reorg path removed.
    return True
```

### Modified `validate_target_epoch_against_current_time`

```python
def validate_target_epoch_against_current_time(store: Store, attestation: Attestation) -> None:
    # [Modified in Simplex] Not used — validate_on_attestation uses slot-based check.
    pass
```

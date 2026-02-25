# One-Round Finality -- Fork Choice

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Containers](#containers)
  - [Modified `Store`](#modified-store)
- [Helper functions](#helper-functions)
  - [Modified `get_forkchoice_store`](#modified-get_forkchoice_store)
  - [Modified `filter_block_tree`](#modified-filter_block_tree)
  - [New `should_update_justified`](#new-should_update_justified)
  - [New `update_checkpoints`](#new-update_checkpoints)
  - [New `get_total_active_voting_weight`](#new-get_total_active_voting_weight)
  - [New `get_view_freeze_due_ms`](#new-get_view_freeze_due_ms)
  - [New `is_before_view_freeze_deadline`](#new-is_before_view_freeze_deadline)
  - [New `is_before_available_attestation_deadline`](#new-is_before_available_attestation_deadline)
  - [New `get_available_attestation_score`](#new-get_available_attestation_score)
  - [New `is_available_attestation_viable`](#new-is_available_attestation_viable)
  - [New `get_available_confirmation_score`](#new-get_available_confirmation_score)
  - [New `is_available_confirmation_viable`](#new-is_available_confirmation_viable)
  - [New `get_available_confirmation_head`](#new-get_available_confirmation_head)
  - [New `get_payload_participant_count`](#new-get_payload_participant_count)
  - [New `get_payload_full_support`](#new-get_payload_full_support)
  - [New `get_payload_data_available_support`](#new-get_payload_data_available_support)
  - [Modified `is_payload_timely`](#modified-is_payload_timely)
  - [Modified `is_payload_data_available`](#modified-is_payload_data_available)
  - [Modified `should_extend_payload`](#modified-should_extend_payload)
  - [Modified `is_head_weak`](#modified-is_head_weak)
  - [Modified `is_parent_strong`](#modified-is_parent_strong)
  - [Modified `should_apply_proposer_boost`](#modified-should_apply_proposer_boost)
  - [Modified `should_override_forkchoice_update`](#modified-should_override_forkchoice_update)
  - [Modified `get_proposer_head`](#modified-get_proposer_head)
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

<!-- mdformat-toc end -->

## Introduction

This is the fork choice specification for one-round finality. It modifies the
fork choice to use the justified checkpoint from one-round finality instead of
Casper FFG's justified checkpoint, and removes the unrealized
justification/finalization machinery.

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

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: (
        Checkpoint  # [Modified in One-Round Finality] one-round finality justified
    )
    finalized_checkpoint: Checkpoint
    justified_height: Height  # [New in One-Round Finality]
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    execution_payload_states: Dict[Root, BeaconState] = field(default_factory=dict)
    # [Modified in One-Round Finality] PTC first vote per member.
    # ``PayloadAttestationData()`` denotes missing vote.
    payload_vote: Vector[PayloadAttestationData, PTC_SIZE] = field(
        default_factory=lambda: [PayloadAttestationData() for _ in range(PTC_SIZE)]
    )
    # [New in One-Round Finality] payload-vote equivocation flag per PTC member
    payload_vote_equivocations: Vector[boolean, PTC_SIZE] = field(
        default_factory=lambda: [False] * PTC_SIZE
    )
    # [New in One-Round Finality] Goldfish: first-seen available attestation per committee member
    # for current slot. ``AvailableAttestationData()`` denotes missing vote.
    current_available_votes: Vector[
        AvailableAttestationData, AVAILABLE_COMMITTEE_SIZE
    ] = field(default_factory=lambda: [AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)])
    # [New in One-Round Finality] Goldfish: available-attestation equivocation flag
    # per committee member for current slot
    current_available_vote_equivocations: Vector[boolean, AVAILABLE_COMMITTEE_SIZE] = field(
        default_factory=lambda: [False] * AVAILABLE_COMMITTEE_SIZE
    )
    # [New in One-Round Finality] Goldfish: tracks which current-slot available
    # committee members were seen by the delayed-confirm timely cutoff
    # (payload-vote/confirm deadline).
    current_available_timely_attesters: Vector[boolean, AVAILABLE_COMMITTEE_SIZE] = field(
        default_factory=lambda: [False] * AVAILABLE_COMMITTEE_SIZE
    )
    # [New in One-Round Finality] Goldfish: first-seen available attestation per committee member
    # for previous slot (used for available confirmation).
    # ``AvailableAttestationData()`` denotes missing vote.
    previous_available_votes: Vector[
        AvailableAttestationData, AVAILABLE_COMMITTEE_SIZE
    ] = field(default_factory=lambda: [AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)])
    # [New in One-Round Finality] Goldfish: available-attestation equivocation flag
    # per committee member for previous slot
    previous_available_vote_equivocations: Vector[boolean, AVAILABLE_COMMITTEE_SIZE] = field(
        default_factory=lambda: [False] * AVAILABLE_COMMITTEE_SIZE
    )
    # [New in One-Round Finality] Goldfish: previous-slot timely attesters (for
    # delayed available confirmation in slot n+1).
    previous_available_timely_attesters: Vector[boolean, AVAILABLE_COMMITTEE_SIZE] = field(
        default_factory=lambda: [False] * AVAILABLE_COMMITTEE_SIZE
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
    anchor_epoch = get_current_epoch(anchor_state)
    justified_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    finalized_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    return Store(
        time=uint64(anchor_state.genesis_time + SECONDS_PER_SLOT * anchor_state.slot),
        genesis_time=anchor_state.genesis_time,
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=finalized_checkpoint,
        justified_height=anchor_state.justified_height,  # [New in One-Round Finality]
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={justified_checkpoint: copy(anchor_state)},
        execution_payload_states={anchor_root: copy(anchor_state)},
        payload_vote=Vector[PayloadAttestationData, PTC_SIZE](
            PayloadAttestationData() for _ in range(PTC_SIZE)
        ),
        payload_vote_equivocations=Vector[boolean, PTC_SIZE](False for _ in range(PTC_SIZE)),
        current_available_votes=Vector[AvailableAttestationData, AVAILABLE_COMMITTEE_SIZE](
            AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
        current_available_vote_equivocations=Vector[boolean, AVAILABLE_COMMITTEE_SIZE](
            False for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
        current_available_timely_attesters=Vector[boolean, AVAILABLE_COMMITTEE_SIZE](
            False for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
        previous_available_votes=Vector[AvailableAttestationData, AVAILABLE_COMMITTEE_SIZE](
            AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
        previous_available_vote_equivocations=Vector[boolean, AVAILABLE_COMMITTEE_SIZE](
            False for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
        previous_available_timely_attesters=Vector[boolean, AVAILABLE_COMMITTEE_SIZE](
            False for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ),
    )
```

### Modified `filter_block_tree`

*Note*: Simplified to check descent from justified and finalized checkpoints
only. The unrealized justification/voting source check is removed.

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    block = store.blocks[block_root]
    children = [
        root for root in store.blocks.keys() if store.blocks[root].parent_root == block_root
    ]

    if any(children):
        filter_block_tree_result = [filter_block_tree(store, child, blocks) for child in children]
        if any(filter_block_tree_result):
            blocks[block_root] = block
            return True
        return False

    # [Modified in One-Round Finality] Leaf node: simplified to justified + finalized descent only
    # (removed unrealized justification/voting source check)

    # Check justified: block must descend from justified checkpoint
    justified_checkpoint_block = get_checkpoint_block(
        store, block_root, store.justified_checkpoint.epoch
    )
    correct_justified = (
        store.justified_checkpoint.epoch == GENESIS_EPOCH
        or store.justified_checkpoint.root == justified_checkpoint_block
    )

    # Check finalized: block must descend from finalized checkpoint
    finalized_checkpoint_block = get_checkpoint_block(
        store, block_root, store.finalized_checkpoint.epoch
    )
    correct_finalized = (
        store.finalized_checkpoint.epoch == GENESIS_EPOCH
        or store.finalized_checkpoint.root == finalized_checkpoint_block
    )

    if correct_justified and correct_finalized:
        blocks[block_root] = block
        return True

    return False
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
    Higher height wins, then higher epoch, then root as deterministic tiebreaker.
    """
    return (candidate_height, candidate.epoch, candidate.root) > (
        current_height,
        current.epoch,
        current.root,
    )
```

### New `update_checkpoints`

```python
def update_checkpoints(
    store: Store,
    justified_checkpoint: Checkpoint,
    justified_height: Height,
    finalized_checkpoint: Checkpoint,
) -> None:
    # Use height-based tie-breaker for justified checkpoint.
    # Note: LJ monotonicity (epoch check) is applied at the state level only,
    # not here. The Store spans multiple chains, so epoch comparison is not a
    # valid descendant check across forks -- it could block convergence when
    # the winning chain has a higher-height but lower-epoch justified checkpoint.
    #
    # Concrete failure mode if epoch gating were added here:
    # - Chain 1 justifies C=(epoch 100) at height H -> store.justified = C
    # - Chain 2 times out at H, then justifies/finalizes D=(epoch 99) at H+1
    # - Epoch gate would block justified update (99 < 100), but finalized update
    #   still succeeds (99 > previous finalized epoch)
    # - Store ends with justified on chain 1 and finalized on chain 2
    # - filter_block_tree then has no leaf descending from both checkpoints.
    if should_update_justified(
        store.justified_checkpoint,
        store.justified_height,
        justified_checkpoint,
        justified_height,
    ):
        store.justified_checkpoint = justified_checkpoint
        store.justified_height = justified_height
    if finalized_checkpoint.epoch > store.finalized_checkpoint.epoch:
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
    state = store.checkpoint_states[store.justified_checkpoint]
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
    return time_into_slot_ms <= get_view_freeze_due_ms(get_current_store_epoch(store))
```

### New `is_before_available_attestation_deadline`

```python
def is_before_available_confirmation_timely_deadline(store: Store) -> bool:
    """
    Return whether current local time is before the delayed available-confirmation
    timely cutoff in the slot (payload-vote/confirm deadline).
    """
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms <= get_payload_attestation_due_ms(get_current_store_epoch(store))


def is_before_available_attestation_deadline(store: Store) -> bool:
    """
    Compatibility alias for ``is_before_available_confirmation_timely_deadline``.
    """
    return is_before_available_confirmation_timely_deadline(store)
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
    return time_into_slot_ms <= get_attestation_due_ms(get_current_store_epoch(store))
```

### New `get_available_attestation_score`

```python
def get_available_attestation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    Return the number of available attestation votes supporting the given ``node``
    from the previous slot's available committee. Each vote counts as 1
    (not weighted by balance). Only counts votes from committee members who
    sent exactly one vote (no equivocations).
    """
    if node.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        node.root
    ].slot + 1 == get_current_slot(store):
        # For previous-slot payload decision (EMPTY/FULL), defer to
        # ``get_payload_status_tiebreaker``.
        return uint64(0)

    missing_available_vote = AvailableAttestationData()
    count = uint64(0)
    for member_index in range(AVAILABLE_COMMITTEE_SIZE):
        vote = store.previous_available_votes[member_index]
        # Only count if: there's a vote and no equivocation
        if vote != missing_available_vote and not store.previous_available_vote_equivocations[
            member_index
        ]:
            # Convert AvailableAttestationData to LatestMessage for is_supporting_vote
            message = LatestMessage(
                slot=vote.slot,
                root=vote.beacon_block_root,
                payload_present=vote.payload_available,
            )
            if is_supporting_vote(store, node, message):
                count += 1
    return count
```


### New `is_available_attestation_viable`

```python
def is_available_attestation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable under strict relative majority with
    equivocation inclusion.
    """
    if child.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        child.root
    ].slot + 1 == get_current_slot(store):
        # For previous-slot payload decision (EMPTY/FULL), viability filtering
        # is not applied; decision is delegated to ``get_payload_status_tiebreaker``.
        return True

    missing_available_vote = AvailableAttestationData()
    participant_count = uint64(
        len([vote for vote in store.previous_available_votes if vote != missing_available_vote])
    )
    available_attestation_score = get_available_attestation_score(store, child)
    equivocation_count = uint64(
        len(
            [
                is_equivocating
                for is_equivocating in store.previous_available_vote_equivocations
                if is_equivocating
            ]
        )
    )
    return available_attestation_score + equivocation_count > participant_count // 2
```

### New `get_available_confirmation_score`

```python
def get_available_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    Return delayed available-confirmation support for ``node`` from the previous
    slot, counting only timely, non-equivocating available attesters.
    """
    if node.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        node.root
    ].slot + 1 == get_current_slot(store):
        # For previous-slot payload decision (EMPTY/FULL), defer to
        # ``get_payload_status_tiebreaker``.
        return uint64(0)

    missing_available_vote = AvailableAttestationData()
    count = uint64(0)
    for member_index in range(AVAILABLE_COMMITTEE_SIZE):
        if not store.previous_available_timely_attesters[member_index]:
            continue
        vote = store.previous_available_votes[member_index]
        if vote == missing_available_vote:
            continue
        if store.previous_available_vote_equivocations[member_index]:
            continue
        message = LatestMessage(
            slot=vote.slot,
            root=vote.beacon_block_root,
            payload_present=vote.payload_available,
        )
        if is_supporting_vote(store, node, message):
            count += 1
    return count
```

### New `is_available_confirmation_viable`

```python
def is_available_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in delayed available confirmation for the
    previous slot, using timely attesters only.
    """
    if child.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        child.root
    ].slot + 1 == get_current_slot(store):
        # For previous-slot payload decision (EMPTY/FULL), viability filtering
        # is not applied; decision is delegated to ``get_payload_status_tiebreaker``.
        return True

    missing_available_vote = AvailableAttestationData()
    participant_count = len(
        [
            vote
            for member_index, vote in enumerate(store.previous_available_votes)
            if (
                store.previous_available_timely_attesters[member_index]
                and vote != missing_available_vote
            )
        ]
    )
    timely_equivocation_count = len(
        [
            is_equivocating
            for member_index, is_equivocating in enumerate(
                store.previous_available_vote_equivocations
            )
            if store.previous_available_timely_attesters[member_index] and is_equivocating
        ]
    )
    confirmation_score = get_available_confirmation_score(store, child)
    return confirmation_score + timely_equivocation_count > participant_count // 2
```

### New `get_available_confirmation_head`

*Note*: This helper is currently an external/research signal (no consensus
state-transition caller in one-round-finality at this stage).

```python
def get_available_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    Return the delayed available-confirmation head for slot ``n`` when called in
    slot ``n+1``, using timely previous-slot available attesters.
    """
    blocks = get_filtered_block_tree(store)

    # Layer 2: Majority LMD-GHOST base head
    head = ForkChoiceNode(
        root=store.justified_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    total_voting_weight = get_total_active_voting_weight(store)
    majority_threshold = total_voting_weight // 2

    while True:
        children = get_node_children(store, blocks, head)
        if len(children) == 0:
            break
        best_child = max(
            children,
            key=lambda child: (
                get_weight(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
        if get_weight(store, best_child) <= majority_threshold:
            break
        head = best_child

    # Delayed available confirmation (timely previous-slot votes only)
    while True:
        children = get_node_children(store, blocks, head)
        viable_children = [
            child for child in children if is_available_confirmation_viable(store, child)
        ]
        if len(viable_children) == 0:
            return head

        head = max(
            viable_children,
            key=lambda child: (
                get_available_confirmation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
```

### New `get_payload_participant_count`

```python
def get_payload_participant_count(store: Store) -> uint64:
    """
    Return the participant count for payload-vote majority gating.
    """
    return uint64(
        len(
            [
                payload_vote
                for payload_vote in store.payload_vote
                if payload_vote != PayloadAttestationData()
            ]
        )
    )
```

### New `get_payload_full_support`

```python
def get_payload_full_support(store: Store, root: Root) -> uint64:
    """
    Return payload FULL support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``payload_present == True`` count.
    Equivocating participants in the slot are included for viability.
    """
    missing_payload_vote = PayloadAttestationData()
    full_support_count = uint64(0)
    for ptc_member_index in range(PTC_SIZE):
        payload_vote = store.payload_vote[ptc_member_index]
        if payload_vote == missing_payload_vote:
            continue
        if store.payload_vote_equivocations[ptc_member_index]:
            full_support_count += 1
            continue
        if payload_vote.beacon_block_root == root and payload_vote.payload_present:
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
    missing_payload_vote = PayloadAttestationData()
    data_available_support_count = uint64(0)
    for ptc_member_index in range(PTC_SIZE):
        payload_vote = store.payload_vote[ptc_member_index]
        if payload_vote == missing_payload_vote:
            continue
        if store.payload_vote_equivocations[ptc_member_index]:
            data_available_support_count += 1
            continue
        if payload_vote.beacon_block_root == root and payload_vote.blob_data_available:
            data_available_support_count += 1
    return data_available_support_count
```

### Modified `is_payload_timely`

```python
def is_payload_timely(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload FULL support.
    """
    # [Modified in One-Round Finality] strict relative-majority over the
    # active payload vote set (with equivocation inclusion in support helper).
    if root not in store.execution_payload_states:
        return False

    participant_count = get_payload_participant_count(store)
    full_support_count = get_payload_full_support(store, root)
    return full_support_count > participant_count // 2
```

### Modified `is_payload_data_available`

```python
def is_payload_data_available(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload data-availability support.
    """
    # [Modified in One-Round Finality] strict relative-majority over the
    # active payload vote set (with equivocation inclusion in support helper).
    if root not in store.execution_payload_states:
        return False

    participant_count = get_payload_participant_count(store)
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
    # [Modified in One-Round Finality] no proposer-boost fallback path.
    return is_payload_timely(store, root) and is_payload_data_available(store, root)
```

### Modified `is_head_weak`

*Note*: Proposer-boost reorg logic is disabled in one-round finality.

```python
def is_head_weak(store: Store, head_root: Root) -> bool:
    # [Modified in One-Round Finality] proposer-boost path removed.
    return False
```

### Modified `is_parent_strong`

*Note*: Proposer-boost reorg logic is disabled in one-round finality.

```python
def is_parent_strong(store: Store, root: Root) -> bool:
    # [Modified in One-Round Finality] proposer-boost path removed.
    return True
```

### Modified `should_apply_proposer_boost`

*Note*: Proposer boost is disabled in one-round finality.

```python
def should_apply_proposer_boost(store: Store) -> bool:
    # [Modified in One-Round Finality] proposer boost disabled.
    return False
```

### Modified `should_override_forkchoice_update`

*Note*: Proposer-override reorg path is disabled in one-round finality.

```python
def should_override_forkchoice_update(store: Store, head_root: Root) -> bool:
    # [Modified in One-Round Finality] override path removed.
    return False
```

### Modified `get_proposer_head`

*Note*: Proposer head override is disabled in one-round finality.

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    # [Modified in One-Round Finality] proposer override removed.
    return head_root
```

### Modified `update_latest_messages`

*Note*: Updated to accept `Attestation` (finality attestations carry
`beacon_block_root` and `payload_available` for LMD-GHOST fork choice).

```python
def update_latest_messages(
    store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation
) -> None:
    # [Modified in One-Round Finality] Uses Attestation with beacon_block_root and payload_available
    slot = attestation.data.slot
    beacon_block_root = attestation.data.beacon_block_root
    payload_present = attestation.data.payload_available
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
boost is not used in one-round finality.

```python
def get_weight(store: Store, node: ForkChoiceNode) -> Gwei:
    # [Modified in One-Round Finality] Defer previous-slot payload decisions
    # to ``get_payload_status_tiebreaker``.
    if node.payload_status == PAYLOAD_STATUS_PENDING or store.blocks[
        node.root
    ].slot + 1 != get_current_slot(store):
        state = store.checkpoint_states[store.justified_checkpoint]
        return get_attestation_score(store, node, state)
    else:
        # Previous-slot payload decision: defer to ``get_payload_status_tiebreaker``.
        return Gwei(0)
```

### Modified `get_head`

*Note*: `get_head` implements a staged fork choice:

1. **Filter stage**: Start from the justified checkpoint, filter the block tree
   (existing `get_filtered_block_tree`).
2. **Majority stage**: Run LMD-GHOST using `latest_messages` (from finality
   attestations), requiring >50% of participating voting weight to proceed.
   Stop when no child has majority support.
3. **Goldfish refinement stage**: From where the majority stage stopped, run
   previous-slot available-attestation voting with:
   - viability gate based on strict majority after adding equivocation count,
   - current-slot proposal pass-through (pending children only), and
   - plurality winner by non-equivocating support among the resulting children.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)

    # [Modified in One-Round Finality] Majority stage: majority-gated LMD-GHOST.
    # Execute LMD-GHOST requiring majority support to proceed
    head = ForkChoiceNode(
        root=store.justified_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    total_voting_weight = get_total_active_voting_weight(store)
    majority_threshold = total_voting_weight // 2

    while True:
        children = get_node_children(store, blocks, head)
        if len(children) == 0:
            break
        # Find the child with the most weight
        best_child = max(
            children,
            key=lambda child: (
                get_weight(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
        # Stop if the best child doesn't have majority support
        if get_weight(store, best_child) <= majority_threshold:
            break
        head = best_child

    # [Modified in One-Round Finality] Goldfish refinement over previous-slot
    # available votes with strict viability, plus current-slot pass-through.
    current_slot = get_current_slot(store)
    while True:
        children = get_node_children(store, blocks, head)
        # Current-slot pass-through is for proposal children only.
        # FULL/EMPTY nodes are already auto-viable for previous-slot payload
        # decisions in ``is_available_attestation_viable``. Keeping this
        # ``PAYLOAD_STATUS_PENDING`` guard avoids introducing same-slot
        # payload-status resolution via pass-through.
        viable_children = [
            child
            for child in children
            if (
                is_available_attestation_viable(store, child)
                or (
                    child.payload_status == PAYLOAD_STATUS_PENDING
                    and store.blocks[child.root].slot == current_slot
                )
            )
        ]
        if len(viable_children) == 0:
            return head

        # Choose plurality winner among viable children, deterministically.
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

*Note*: Wire attestations are epoch-bounded by slot epoch (current/previous
epoch), and finality vote persistence across longer non-finality windows is
handled by re-submitting the same `(height, target)` vote in new slots.

```python
def validate_on_attestation(store: Store, attestation: Attestation, is_from_block: bool) -> None:
    data = attestation.data
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    # Target must be for a known block
    assert data.target.root in store.blocks
    # [Modified in One-Round Finality] target epoch may be older than attestation
    # epoch (height-based finality), but it cannot be from a future epoch.
    assert data.target.epoch <= compute_epoch_at_slot(data.slot)
    # Same-slot attestation cannot signal payload availability
    # (PTC does the first payload availability determination)
    if block_slot == data.slot:
        assert not data.payload_available

    # Attestations can only affect fork choice of subsequent slots.
    # Delay consideration in the fork-choice until their slot is in the past.
    assert get_current_slot(store) >= data.slot + 1

    if not is_from_block:
        # [Modified in One-Round Finality] keep wire attestations epoch-bounded
        # by attestation slot epoch (current/previous), while allowing
        # cross-epoch finality targets by height.
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
        assert not attestation.data.payload_available
```

## Handlers

### Modified `on_tick_per_slot`

*Note*: The epoch boundary unrealized checkpoint pull-up is removed.

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    # [Modified in One-Round Finality] Removed epoch boundary unrealized checkpoint pull-up
    previous_time = store.time
    previous_slot = get_current_slot(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        # [New in One-Round Finality] Keep exactly two available-attestation vote sets:
        # current slot and previous slot.
        store.previous_available_votes = store.current_available_votes
        store.previous_available_vote_equivocations = (
            store.current_available_vote_equivocations
        )
        store.previous_available_timely_attesters = (
            store.current_available_timely_attesters
        )
        store.current_available_votes = [
            AvailableAttestationData() for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ]
        store.current_available_vote_equivocations = [False] * AVAILABLE_COMMITTEE_SIZE
        store.current_available_timely_attesters = [False] * AVAILABLE_COMMITTEE_SIZE

    # [New in One-Round Finality] Reset payload votes when crossing the
    # attestation deadline within the slot. Before this deadline, previous-slot
    # payload votes remain relevant; after it, they are no longer needed.
    previous_time_into_slot_ms = seconds_to_milliseconds(
        previous_time - store.genesis_time
    ) % SLOT_DURATION_MS
    current_time_into_slot_ms = seconds_to_milliseconds(
        store.time - store.genesis_time
    ) % SLOT_DURATION_MS
    attestation_due_ms = get_attestation_due_ms(get_current_store_epoch(store))
    if previous_time_into_slot_ms < attestation_due_ms <= current_time_into_slot_ms:
        store.payload_vote = [PayloadAttestationData() for _ in range(PTC_SIZE)]
        store.payload_vote_equivocations = [False] * PTC_SIZE
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

    finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)
    assert block.slot > finalized_slot
    finalized_checkpoint_block = get_checkpoint_block(
        store, block.parent_root, store.finalized_checkpoint.epoch
    )
    assert store.finalized_checkpoint.root == finalized_checkpoint_block

    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)

    store.blocks[block_root] = block
    store.block_states[block_root] = state

    notify_ptc_messages(store, state, block.body.payload_attestations)

    # [Modified in One-Round Finality] Process finality attestations for fork choice (update latest_messages)
    for attestation in block.body.attestations:
        on_attestation(store, attestation, is_from_block=True)

    # [Modified in One-Round Finality] Process available attestations for Goldfish (per-slot vote tracking)
    for available_attestation in block.body.available_attestations:
        on_available_attestation(store, available_attestation, is_from_block=True)

    # [Modified in One-Round Finality] Update checkpoints with height, no unrealized pull-up
    update_checkpoints(
        store, state.justified_checkpoint, state.justified_height, state.finalized_checkpoint
    )

    # Populate checkpoint_states cache for the new justified checkpoint
    if store.justified_checkpoint not in store.checkpoint_states:
        jcp_state = copy(store.block_states[store.justified_checkpoint.root])
        jcp_epoch_slot = compute_start_slot_at_epoch(store.justified_checkpoint.epoch)
        if jcp_state.slot < jcp_epoch_slot:
            process_slots(jcp_state, jcp_epoch_slot)
        store.checkpoint_states[store.justified_checkpoint] = jcp_state
```

### Modified `on_payload_attestation_message`

*Note*: Payload votes use first-vote + equivocation-vote tracking with
view-merge freeze handling:

1. non-proposer wire votes after freeze are ignored;
2. block-carried votes are always processed;
3. the next proposer may continue collecting wire votes after freeze by calling
   with ``is_next_proposer=True``;
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
        # [Modified in One-Round Finality] view-freeze gating with
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
    current_slot = get_current_slot(store)
    is_pre_attestation_deadline = is_before_attestation_deadline(store)
    if is_pre_attestation_deadline and current_slot > 0:
        active_vote_slot = Slot(current_slot - 1)
    else:
        active_vote_slot = current_slot

    # Only process votes from the active payload-vote slot.
    if vote_slot != active_vote_slot:
        return

    missing_payload_vote = PayloadAttestationData()
    # Keep a single active payload vote set aligned to the active slot window.
    tracked_vote_slot = next(
        (
            payload_vote.slot
            for payload_vote in store.payload_vote
            if payload_vote != missing_payload_vote
        ),
        None,
    )
    if tracked_vote_slot is not None and tracked_vote_slot != active_vote_slot:
        store.payload_vote = [PayloadAttestationData() for _ in range(PTC_SIZE)]
        store.payload_vote_equivocations = [False] * PTC_SIZE

    ptc_member_index = ptc.index(ptc_message.validator_index)

    # Ignore additional votes after the first equivocation.
    if store.payload_vote_equivocations[ptc_member_index]:
        return

    first_vote = store.payload_vote[ptc_member_index]
    if first_vote == missing_payload_vote:
        store.payload_vote[ptc_member_index] = data
        return

    if first_vote != data:
        store.payload_vote_equivocations[ptc_member_index] = True
```

### Modified `on_attestation`

*Note*: Finality attestations now update `latest_messages` for the majority fork
choice layer. `AttestationData` carries `beacon_block_root` (LMD head vote) and
`payload_available` (payload availability signal).

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """
    [Modified in One-Round Finality] Finality attestations update latest_messages
    for the majority fork choice layer.
    """
    # Skip from-block attestations whose head vote references an unknown block
    # (voter may have voted for a block on a different fork)
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        return
    # Skip from-block attestations whose finality target references an unknown block
    # (voter may have voted for a block on a different fork)
    if is_from_block and attestation.data.target.root not in store.blocks:
        return
    validate_on_attestation(store, attestation, is_from_block)

    # Derive checkpoint state for signature verification and attesting indices
    attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
    epoch_root = get_checkpoint_block(store, attestation.data.beacon_block_root, attestation_epoch)
    checkpoint = Checkpoint(epoch=attestation_epoch, root=epoch_root)

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

*Note*: Available attestations track per-slot per-committee-member votes for
the Goldfish fork choice layer, instead of updating `latest_messages`.
Non-proposers ignore wire votes after freeze; proposers may continue collecting
wire votes after freeze with ``is_next_proposer=True``.
Equivocation tracking uses the first vote / second vote pattern (like PTC).

```python
def on_available_attestation(
    store: Store,
    attestation: AvailableAttestation,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    """
    [New in One-Round Finality] Available attestations store per-slot
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
        # Derive checkpoint state for signature verification
        attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
        epoch_root = get_checkpoint_block(
            store, attestation.data.beacon_block_root, attestation_epoch
        )
        checkpoint = Checkpoint(epoch=attestation_epoch, root=epoch_root)

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
    current_slot = get_current_slot(store)
    vote_slot = attestation.data.slot
    if vote_slot == current_slot:
        available_votes = store.current_available_votes
        available_vote_equivocations = store.current_available_vote_equivocations
        available_timely_attesters = store.current_available_timely_attesters
    elif current_slot > 0 and vote_slot + 1 == current_slot:
        available_votes = store.previous_available_votes
        available_vote_equivocations = store.previous_available_vote_equivocations
        available_timely_attesters = store.previous_available_timely_attesters
    else:
        # Only current and previous slot are retained.
        return

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
                and is_before_available_confirmation_timely_deadline(store)
            ):
                available_timely_attesters[member_index] = True
        elif first_vote != attestation.data:
            # Second (different) vote — record as equivocation
            available_vote_equivocations[member_index] = True
```

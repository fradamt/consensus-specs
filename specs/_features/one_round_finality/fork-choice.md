# One-Round Finality -- Fork Choice

<!-- mdformat-toc start --no-anchors -->

---

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
The `justified_height` field is added for height-based tie-breaking.
The `available_votes` and `available_vote_equivocations` fields track per-slot
per-committee-member available attestation votes for the Goldfish fork choice layer.

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: Checkpoint  # [Modified in One-Round Finality] one-round finality justified
    finalized_checkpoint: Checkpoint
    justified_height: Height  # [New in One-Round Finality]
    proposer_boost_root: Root
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    block_timeliness: Dict[Root, Vector[boolean, NUM_BLOCK_TIMELINESS_DEADLINES]] = field(
        default_factory=dict
    )
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    execution_payload_states: Dict[Root, BeaconState] = field(default_factory=dict)
    payload_timeliness_vote: Dict[Root, Vector[boolean, PTC_SIZE]] = field(default_factory=dict)
    payload_data_availability_vote: Dict[Root, Vector[boolean, PTC_SIZE]] = field(
        default_factory=dict
    )
    # [New in One-Round Finality] Goldfish: first-seen available attestation per committee member per slot
    available_votes: Dict[Slot, Vector[Optional[AvailableAttestationData], AVAILABLE_COMMITTEE_SIZE]] = field(
        default_factory=dict
    )
    # [New in One-Round Finality] Goldfish: equivocating available attestation per committee member per slot
    available_vote_equivocations: Dict[Slot, Vector[Optional[AvailableAttestationData], AVAILABLE_COMMITTEE_SIZE]] = field(
        default_factory=dict
    )
```

## Helper functions

### Modified `get_forkchoice_store`

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
        proposer_boost_root=Root(),
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        block_timeliness={anchor_root: [True, True]},
        checkpoint_states={justified_checkpoint: copy(anchor_state)},
        execution_payload_states={anchor_root: copy(anchor_state)},
        payload_timeliness_vote={
            anchor_root: Vector[boolean, PTC_SIZE](True for _ in range(PTC_SIZE))
        },
        payload_data_availability_vote={
            anchor_root: Vector[boolean, PTC_SIZE](True for _ in range(PTC_SIZE))
        },
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
    return (candidate_height, candidate.epoch, candidate.root) > (current_height, current.epoch, current.root)
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
    Return the total effective balance of all unslashed active validators.
    Used as the majority threshold denominator for the 3-layer fork choice.
    """
    state = store.checkpoint_states[store.justified_checkpoint]
    return Gwei(
        sum(
            state.validators[i].effective_balance
            for i in get_active_validator_indices(state, get_current_epoch(state))
            if not state.validators[i].slashed
        )
    )
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
    current_slot = get_current_slot(store)
    if current_slot == 0:
        return uint64(0)
    previous_slot = Slot(current_slot - 1)

    if previous_slot not in store.available_votes:
        return uint64(0)

    votes = store.available_votes[previous_slot]
    equivocations = store.available_vote_equivocations[previous_slot]

    count = uint64(0)
    for i in range(AVAILABLE_COMMITTEE_SIZE):
        vote = votes[i]
        # Only count if: there's a vote and no equivocation
        if vote is not None and equivocations[i] is None:
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

### Modified `is_head_weak`

*Note*: Reverted to standard Gloas behavior. Reorg thresholds scale from
`total_active_balance / SLOTS_PER_EPOCH`. Latest messages now come from
finality attestations (all validators).

```python
def is_head_weak(store: Store, head_root: Root) -> bool:
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    reorg_threshold = calculate_committee_fraction(justified_state, REORG_HEAD_WEIGHT_THRESHOLD)

    # Compute head weight including equivocations
    head_state = store.block_states[head_root]
    head_block = store.blocks[head_root]
    epoch = compute_epoch_at_slot(head_block.slot)
    head_node = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
    head_weight = get_attestation_score(store, head_node, justified_state)
    for index in range(get_committee_count_per_slot(head_state, epoch)):
        committee = get_beacon_committee(head_state, head_block.slot, CommitteeIndex(index))
        head_weight += Gwei(
            sum(
                justified_state.validators[i].effective_balance
                for i in committee
                if i in store.equivocating_indices
            )
        )

    return head_weight < reorg_threshold
```

### Modified `is_parent_strong`

```python
def is_parent_strong(store: Store, root: Root) -> bool:
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    parent_threshold = calculate_committee_fraction(justified_state, REORG_PARENT_WEIGHT_THRESHOLD)
    block = store.blocks[root]
    parent_payload_status = get_parent_payload_status(store, block)
    parent_node = ForkChoiceNode(root=block.parent_root, payload_status=parent_payload_status)
    parent_weight = get_attestation_score(store, parent_node, justified_state)
    return parent_weight > parent_threshold
```

### Modified `should_apply_proposer_boost`

```python
def should_apply_proposer_boost(store: Store) -> bool:
    if store.proposer_boost_root == Root():
        return False

    block = store.blocks[store.proposer_boost_root]
    parent_root = block.parent_root
    parent = store.blocks[parent_root]
    slot = block.slot

    # Apply proposer boost if `parent` is not from the previous slot
    if parent.slot + 1 < slot:
        return True

    # Apply proposer boost if `parent` is not weak
    if not is_head_weak(store, parent_root):
        return True

    # If `parent` is weak and from the previous slot, apply
    # proposer boost if there are no early equivocations
    equivocations = [
        root
        for root, block in store.blocks.items()
        if (
            store.block_timeliness[root][PTC_TIMELINESS_INDEX]
            and block.proposer_index == parent.proposer_index
            and block.slot + 1 == slot
            and root != parent_root
        )
    ]

    return len(equivocations) == 0
```

### Modified `should_override_forkchoice_update`

*Note*: Removed `is_ffg_competitive` (no unrealized justifications in one-round finality).

```python
def should_override_forkchoice_update(store: Store, head_root: Root) -> bool:
    head_block = store.blocks[head_root]
    parent_root = head_block.parent_root
    parent_block = store.blocks[parent_root]
    current_slot = get_current_slot(store)
    proposal_slot = head_block.slot + Slot(1)

    head_late = is_head_late(store, head_root)
    shuffling_stable = is_shuffling_stable(proposal_slot)
    # [Modified in One-Round Finality] is_ffg_competitive removed (no unrealized justifications)
    finalization_ok = is_finalization_ok(store, proposal_slot)

    parent_state_advanced = store.block_states[parent_root].copy()
    process_slots(parent_state_advanced, proposal_slot)
    proposer_index = get_beacon_proposer_index(parent_state_advanced)
    proposing_reorg_slot = validator_is_connected(proposer_index)

    parent_slot_ok = parent_block.slot + 1 == head_block.slot
    proposing_on_time = is_proposing_on_time(store)

    current_time_ok = head_block.slot == current_slot or (
        proposal_slot == current_slot and proposing_on_time
    )
    single_slot_reorg = parent_slot_ok and current_time_ok

    if current_slot > head_block.slot:
        head_weak = is_head_weak(store, head_root)
        parent_strong = is_parent_strong(store, head_root)
    else:
        head_weak = True
        parent_strong = True

    return all(
        [
            head_late,
            shuffling_stable,
            finalization_ok,
            proposing_reorg_slot,
            single_slot_reorg,
            head_weak,
            parent_strong,
        ]
    )
```

### Modified `get_proposer_head`

*Note*: `is_ffg_competitive` removed (no unrealized justifications).

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    head_block = store.blocks[head_root]
    parent_root = head_block.parent_root
    parent_block = store.blocks[parent_root]

    head_late = is_head_late(store, head_root)
    shuffling_stable = is_shuffling_stable(slot)
    # [Modified in One-Round Finality] is_ffg_competitive removed (no unrealized justifications)
    finalization_ok = is_finalization_ok(store, slot)
    proposing_on_time = is_proposing_on_time(store)

    parent_slot_ok = parent_block.slot + 1 == head_block.slot
    current_time_ok = head_block.slot + 1 == slot
    single_slot_reorg = parent_slot_ok and current_time_ok

    assert store.proposer_boost_root != head_root
    head_weak = is_head_weak(store, head_root)
    parent_strong = is_parent_strong(store, head_root)

    proposer_equivocation = is_proposer_equivocation(store, head_root)

    if all(
        [
            head_late,
            shuffling_stable,
            finalization_ok,
            proposing_on_time,
            single_slot_reorg,
            head_weak,
            parent_strong,
        ]
    ):
        return parent_root
    elif all([head_weak, current_time_ok, proposer_equivocation]):
        return parent_root
    else:
        return head_root
```

### Modified `update_latest_messages`

*Note*: Updated to accept `Attestation` (finality attestations carry `beacon_block_root`
and `payload_available` for LMD-GHOST fork choice).

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

*Note*: Returns 0 for current-slot payload-decision nodes (EMPTY/FULL),
deferring the payload decision to the Goldfish layer.

```python
def get_weight(store: Store, node: ForkChoiceNode) -> Gwei:
    # [Modified in One-Round Finality] Defer current-slot payload decisions to Goldfish
    if node.payload_status == PAYLOAD_STATUS_PENDING or store.blocks[
        node.root
    ].slot + 1 != get_current_slot(store):
        state = store.checkpoint_states[store.justified_checkpoint]
        attestation_score = get_attestation_score(store, node, state)
        if not should_apply_proposer_boost(store):
            return attestation_score

        proposer_score = Gwei(0)
        message = LatestMessage(
            slot=get_current_slot(store),
            root=store.proposer_boost_root,
            payload_present=False,
        )
        if is_supporting_vote(store, node, message):
            proposer_score = get_proposer_score(store)

        return attestation_score + proposer_score
    else:
        # Current-slot payload decision: defer to Goldfish layer
        return Gwei(0)
```

### Modified `get_head`

*Note*: `get_head` implements a 3-layer fork choice:

1. **Layer 1 (Filter)**: Start from the justified checkpoint, filter the block tree (existing `get_filtered_block_tree`).
2. **Layer 2 (Majority)**: Run LMD-GHOST using `latest_messages` (from finality attestations), requiring >50% of total active voting weight to proceed. Stop when no child has majority support.
3. **Layer 3 (Goldfish)**: From where Layer 2 stopped, run GHOST using available attestation votes from the previous slot's available committee. Each vote counts as 1 (not weighted by balance), and no majority threshold is required.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)

    # Layer 2: Majority LMD-GHOST
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

    # Layer 3: Goldfish GHOST
    # Continue with available attestation voting (each vote counts as 1)
    while True:
        children = get_node_children(store, blocks, head)
        if len(children) == 0:
            return head
        # Sort by available attestation score with ties broken lexicographically
        head = max(
            children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
```

### New `validate_on_attestation`

*Note*: No epoch restriction. Finality attestations carry height-based votes
that need to be processable for previous/current height even if from older
epochs. Height validation is handled by `process_attestation` (state transition).

```python
def validate_on_attestation(
    store: Store, attestation: Attestation, is_from_block: bool
) -> None:
    data = attestation.data
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    # Target must be for a known block
    assert data.target.root in store.blocks
    # Same-slot attestation cannot signal payload availability
    # (PTC does the first payload availability determination)
    if block_slot == data.slot:
        assert not data.payload_available
    if not is_from_block:
        # Attestation can only affect fork choice of subsequent slots
        assert get_current_slot(store) >= data.slot + 1
```

### New `validate_on_available_attestation`

```python
def validate_on_available_attestation(
    store: Store, attestation: AvailableAttestation, is_from_block: bool
) -> None:
    # If the given attestation is not from a beacon block message,
    # check the attestation epoch scope.
    if not is_from_block:
        current_epoch = get_current_store_epoch(store)
        previous_epoch = current_epoch - 1 if current_epoch > GENESIS_EPOCH else GENESIS_EPOCH
        attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
        assert attestation_epoch in (current_epoch, previous_epoch)

    # Attestations must be for a known block.
    assert attestation.data.beacon_block_root in store.blocks
    # Attestations must not be for blocks in the future.
    block_slot = store.blocks[attestation.data.beacon_block_root].slot
    assert block_slot <= attestation.data.slot

    # Same-slot attestation cannot signal payload availability
    if block_slot == attestation.data.slot:
        assert not attestation.data.payload_available

    if not is_from_block:
        # Attestations can only affect the fork-choice of subsequent slots.
        assert get_current_slot(store) >= attestation.data.slot + 1
```

## Handlers

### Modified `on_tick_per_slot`

*Note*: The epoch boundary unrealized checkpoint pull-up is removed.

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    # [Modified in One-Round Finality] Removed epoch boundary unrealized checkpoint pull-up
    previous_slot = get_current_slot(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        store.proposer_boost_root = Root()
        # [New in One-Round Finality] Prune old Goldfish votes (only previous slot needed)
        for slot in list(store.available_votes.keys()):
            if slot < current_slot - 1:
                del store.available_votes[slot]
                del store.available_vote_equivocations[slot]
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

    store.payload_timeliness_vote[block_root] = [False] * PTC_SIZE
    store.payload_data_availability_vote[block_root] = [False] * PTC_SIZE
    notify_ptc_messages(store, state, block.body.payload_attestations)

    # [Modified in One-Round Finality] Process finality attestations for fork choice (update latest_messages)
    for attestation in block.body.attestations:
        on_attestation(store, attestation, is_from_block=True)

    # [Modified in One-Round Finality] Process available attestations for Goldfish (per-slot vote tracking)
    for available_attestation in block.body.available_attestations:
        on_available_attestation(store, available_attestation, is_from_block=True)

    record_block_timeliness(store, block_root)
    update_proposer_boost_root(store, block_root)

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

### Modified `on_attester_slashing`

```python
def on_attester_slashing(store: Store, attester_slashing: AttesterSlashing) -> None:
    """
    Run ``on_attester_slashing`` immediately upon receiving a new ``AttesterSlashing``
    from either within a block or directly on the wire.
    [Modified in One-Round Finality] Uses epoch double-vote and height target conflict via is_slashable_attestation_data.
    """
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    state = store.block_states[store.justified_checkpoint.root]
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)

    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    for index in indices:
        store.equivocating_indices.add(index)
```

### Modified `on_attestation`

*Note*: Finality attestations now update `latest_messages` for the majority
fork choice layer. `AttestationData` carries `beacon_block_root` (LMD head vote)
and `payload_available` (payload availability signal).

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """
    [Modified in One-Round Finality] Finality attestations update latest_messages
    for the majority fork choice layer. No epoch restriction — finality attestations
    need to be processable for previous/current height even if from older epochs.
    """
    # Skip from-block attestations whose head vote references an unknown block
    # (voter may have voted for a block on a different fork)
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
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
        assert is_valid_indexed_attestation(target_state, get_indexed_attestation(target_state, attestation))

    attesting_indices = get_attesting_indices(target_state, attestation)
    update_latest_messages(store, sorted(attesting_indices), attestation)
```

### Modified `on_available_attestation`

*Note*: Available attestations now track per-slot per-committee-member votes
for the Goldfish fork choice layer, instead of updating `latest_messages`.
Equivocation tracking uses the first vote / second vote pattern (like PTC).

```python
def on_available_attestation(
    store: Store, attestation: AvailableAttestation, is_from_block: bool = False
) -> None:
    """
    [Modified in One-Round Finality] Available attestations store per-slot
    per-committee-member votes for the Goldfish fork choice layer.
    """
    # Skip from-block attestations whose head vote references an unknown block
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        return
    validate_on_available_attestation(store, attestation, is_from_block)

    if not is_from_block:
        # Derive checkpoint state for signature verification
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

        # Verify signature against available committee
        attesting_indices = get_available_attesting_indices(target_state, attestation)
        pubkeys = [target_state.validators[i].pubkey for i in sorted(attesting_indices)]
        domain = get_domain(target_state, DOMAIN_AVAILABLE_ATTESTER, attestation_epoch)
        signing_root = compute_signing_root(attestation.data, domain)
        assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Store individual votes for Goldfish tracking
    slot = attestation.data.slot
    if slot not in store.available_votes:
        store.available_votes[slot] = [None] * AVAILABLE_COMMITTEE_SIZE
        store.available_vote_equivocations[slot] = [None] * AVAILABLE_COMMITTEE_SIZE

    for i in range(len(attestation.aggregation_bits)):
        if not attestation.aggregation_bits[i]:
            continue
        first_vote = store.available_votes[slot][i]
        second_vote = store.available_vote_equivocations[slot][i]
        # Ignore further equivocations
        if second_vote is not None:
            continue
        if first_vote is None:
            # First vote from this committee member for this slot
            store.available_votes[slot][i] = attestation.data
        elif first_vote != attestation.data:
            # Second (different) vote — record as equivocation
            store.available_vote_equivocations[slot][i] = attestation.data
```

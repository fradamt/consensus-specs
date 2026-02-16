# Minimmit -- Fork Choice

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

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: Checkpoint  # [Modified in Minimmit] one-round finality justified
    finalized_checkpoint: Checkpoint
    justified_height: Height  # [New in Minimmit]
    proposer_boost_root: Root
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    block_timeliness: Dict[Root, Vector[boolean, NUM_BLOCK_TIMELINESS_DEADLINES]] = field(
        default_factory=dict
    )
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    # [New in Gloas:EIP7732]
    execution_payload_states: Dict[Root, BeaconState] = field(default_factory=dict)
    # [New in Gloas:EIP7732]
    payload_timeliness_vote: Dict[Root, Vector[boolean, PTC_SIZE]] = field(default_factory=dict)
    # [New in Gloas:EIP7732]
    payload_data_availability_vote: Dict[Root, Vector[boolean, PTC_SIZE]] = field(
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
        justified_height=anchor_state.justified_height,  # [New in Minimmit]
        proposer_boost_root=Root(),
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        block_timeliness={anchor_root: [True, True]},
        checkpoint_states={justified_checkpoint: copy(anchor_state)},
        # [New in Gloas:EIP7732]
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

    # Leaf node: check justified and finalized consistency

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

### Modified `is_head_weak`

*Note*: Modified to use the available committee instead of iterating over
multiple beacon committees. Only available committee members' equivocations
affect the LMD fork choice weight.

```python
def is_head_weak(store: Store, head_root: Root) -> bool:
    # Calculate weight threshold for weak head
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    reorg_threshold = calculate_committee_fraction(justified_state, REORG_HEAD_WEIGHT_THRESHOLD)

    # Compute head weight including equivocations
    head_state = store.block_states[head_root]
    head_block = store.blocks[head_root]
    head_node = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
    head_weight = get_attestation_score(store, head_node, justified_state)
    # [Modified in Minimmit] Only available committee members for equivocations
    committee = get_available_committee(head_state, head_block.slot)
    head_weight += Gwei(
        sum(
            justified_state.validators[i].effective_balance
            for i in committee
            if i in store.equivocating_indices
        )
    )

    return head_weight < reorg_threshold
```

### Modified `get_proposer_head`

*Note*: The `is_ffg_competitive` check is removed since there are no
unrealized justifications.

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    head_block = store.blocks[head_root]
    parent_root = head_block.parent_root
    parent_block = store.blocks[parent_root]

    head_late = is_head_late(store, head_root)
    shuffling_stable = is_shuffling_stable(slot)
    # [Modified in Minimmit] is_ffg_competitive removed (no unrealized justifications)
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

### Modified `validate_on_attestation`

*Note*: Target validation is removed since `AttestationData` no longer contains
a target field. Only beacon block root, slot, and index checks remain.

```python
def validate_on_attestation(store: Store, attestation: Attestation, is_from_block: bool) -> None:
    # [Modified in Minimmit] No target field in AttestationData.

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

    # [Gloas:EIP7732] Validate index field for payload signaling
    assert attestation.data.index in [0, 1]
    if block_slot == attestation.data.slot:
        assert attestation.data.index == 0

    # Attestations can only affect the fork-choice of subsequent slots.
    assert get_current_slot(store) >= attestation.data.slot + 1
```

## Handlers

### Modified `on_tick_per_slot`

*Note*: The epoch boundary unrealized checkpoint pull-up is removed.

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    previous_slot = get_current_slot(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        store.proposer_boost_root = Root()
```

### Modified `on_block`

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    assert block.parent_root in store.block_states

    # [Gloas:EIP7732] Check if this block builds on empty or full parent block
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

    # [Gloas:EIP7732]
    store.payload_timeliness_vote[block_root] = [False] * PTC_SIZE
    store.payload_data_availability_vote[block_root] = [False] * PTC_SIZE
    notify_ptc_messages(store, state, block.body.payload_attestations)

    record_block_timeliness(store, block_root)
    update_proposer_boost_root(store, block_root)

    # [Modified in Minimmit] Update checkpoints with height, no unrealized pull-up
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

### New `on_finality_slashing`

```python
def on_finality_slashing(store: Store, finality_slashing: FinalitySlashing) -> None:
    """
    Run ``on_finality_slashing`` immediately upon receiving a new ``FinalitySlashing``
    from either within a block or directly on the wire.
    """
    attestation_1 = finality_slashing.attestation_1
    attestation_2 = finality_slashing.attestation_2
    assert is_slashable_finality_attestation_data(attestation_1.data, attestation_2.data)
    state = store.block_states[store.justified_checkpoint.root]
    assert is_valid_indexed_finality_attestation(state, attestation_1)
    assert is_valid_indexed_finality_attestation(state, attestation_2)

    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    for index in indices:
        store.equivocating_indices.add(index)
```

### Modified `on_attestation`

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    validate_on_attestation(store, attestation, is_from_block)

    # [Modified in Minimmit] Derive checkpoint from slot epoch (no target)
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
    indexed_attestation = get_indexed_attestation(target_state, attestation)
    assert is_valid_indexed_attestation(target_state, indexed_attestation)

    update_latest_messages(store, indexed_attestation.attesting_indices, attestation)
```

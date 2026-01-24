# Minimmit -- Fork Choice

## Introduction

This is the fork choice specification for one-round finality. It modifies
the fork choice to use the latest notarized checkpoint instead of the
justified checkpoint.

*Note*: This specification is built upon [Fulu](../../fulu/fork-choice.md).

## Containers

### Modified `Store`

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    latest_notarized: Checkpoint  # [Modified in Minimmit] replaces justified_checkpoint
    finalized_checkpoint: Checkpoint
    current_height: Height  # [New in Minimmit]
    current_target: Checkpoint  # [New in Minimmit]
    height_votes: Gwei  # [New in Minimmit]
    height_total_votes: Gwei  # [New in Minimmit]
    proposer_boost_root: Root
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    block_timeliness: Dict[Root, boolean] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
```

*Note*: The fields `justified_checkpoint`, `unrealized_justified_checkpoint`,
`unrealized_finalized_checkpoint`, and `unrealized_justifications` are removed.

## Helper functions

### `get_forkchoice_store`

```python
def get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock) -> Store:
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    anchor_checkpoint = Checkpoint(
        epoch=get_current_epoch(anchor_state),
        root=anchor_root,
        height=anchor_state.current_height,
    )
    return Store(
        time=uint64(anchor_state.genesis_time + SECONDS_PER_SLOT * anchor_state.slot),
        genesis_time=anchor_state.genesis_time,
        latest_notarized=anchor_checkpoint,
        finalized_checkpoint=anchor_checkpoint,
        current_height=anchor_state.current_height,
        current_target=anchor_state.current_target,
        height_votes=anchor_state.height_votes,
        height_total_votes=anchor_state.height_total_votes,
        proposer_boost_root=Root(),
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={anchor_checkpoint: copy(anchor_state)},
    )
```

### `get_fork_choice_root`

```python
def get_fork_choice_root(store: Store) -> Root:
    """
    Return the root from which LMD-GHOST fork choice runs.
    """
    return store.latest_notarized.root
```

### Modified `get_proposer_score`

```python
def get_proposer_score(store: Store) -> Gwei:
    notarized_state = store.checkpoint_states[store.latest_notarized]
    committee_weight = get_total_active_balance(notarized_state) // SLOTS_PER_EPOCH
    return (committee_weight * PROPOSER_SCORE_BOOST) // 100
```

### Modified `get_weight`

```python
def get_weight(store: Store, root: Root) -> Gwei:
    state = store.checkpoint_states[store.latest_notarized]
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    attestation_score = Gwei(sum(
        state.validators[i].effective_balance
        for i in unslashed_and_active_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and get_ancestor(store, store.latest_messages[i].root, store.blocks[root].slot) == root)
    ))
    if store.proposer_boost_root == Root():
        return attestation_score

    proposer_score = Gwei(0)
    if get_ancestor(store, store.proposer_boost_root, store.blocks[root].slot) == root:
        proposer_score = get_proposer_score(store)
    return attestation_score + proposer_score
```

### Modified `filter_block_tree`

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    block = store.blocks[block_root]
    children = [
        root for root in store.blocks.keys()
        if store.blocks[root].parent_root == block_root
    ]

    if any(children):
        filter_block_tree_result = [filter_block_tree(store, child, blocks) for child in children]
        if any(filter_block_tree_result):
            blocks[block_root] = block
            return True
        return False

    finalized_checkpoint_block = get_checkpoint_block(
        store, block_root, store.finalized_checkpoint.epoch
    )
    correct_finalized = (
        store.finalized_checkpoint.epoch == GENESIS_EPOCH
        or store.finalized_checkpoint.root == finalized_checkpoint_block
    )

    if correct_finalized:
        blocks[block_root] = block
        return True

    return False
```

### Modified `get_filtered_block_tree`

```python
def get_filtered_block_tree(store: Store) -> Dict[Root, BeaconBlock]:
    base = get_fork_choice_root(store)
    blocks: Dict[Root, BeaconBlock] = {}
    filter_block_tree(store, base, blocks)
    return blocks
```

### Modified `get_head`

```python
def get_head(store: Store) -> Root:
    blocks = get_filtered_block_tree(store)
    head = get_fork_choice_root(store)
    while True:
        children = [root for root in blocks.keys() if blocks[root].parent_root == head]
        if len(children) == 0:
            return head
        head = max(children, key=lambda root: (get_weight(store, root), root))
```

### `update_finality`

```python
def update_finality(store: Store, latest_notarized: Checkpoint, finalized_checkpoint: Checkpoint) -> None:
    if latest_notarized.epoch > store.latest_notarized.epoch:
        store.latest_notarized = latest_notarized
    if finalized_checkpoint.epoch > store.finalized_checkpoint.epoch:
        store.finalized_checkpoint = finalized_checkpoint
```

### `sync_finality_from_state`

```python
def sync_finality_from_state(store: Store, state: BeaconState) -> None:
    store.current_height = state.current_height
    store.current_target = state.current_target
    store.height_votes = state.height_votes
    store.height_total_votes = state.height_total_votes
    update_finality(store, state.latest_notarized, state.finalized_checkpoint)
```

### Modified proposer reorg helpers

#### Modified `is_head_weak`

```python
def is_head_weak(store: Store, head_root: Root) -> bool:
    notarized_state = store.checkpoint_states[store.latest_notarized]
    reorg_threshold = calculate_committee_fraction(notarized_state, REORG_HEAD_WEIGHT_THRESHOLD)
    head_weight = get_weight(store, head_root)
    return head_weight < reorg_threshold
```

#### Modified `is_parent_strong`

```python
def is_parent_strong(store: Store, parent_root: Root) -> bool:
    notarized_state = store.checkpoint_states[store.latest_notarized]
    parent_threshold = calculate_committee_fraction(notarized_state, REORG_PARENT_WEIGHT_THRESHOLD)
    parent_weight = get_weight(store, parent_root)
    return parent_weight > parent_threshold
```

#### Modified `get_proposer_head`

*Note*: The `is_ffg_competitive` check is removed.

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    head_block = store.blocks[head_root]
    parent_root = head_block.parent_root
    parent_block = store.blocks[parent_root]

    head_late = is_head_late(store, head_root)
    shuffling_stable = is_shuffling_stable(slot)
    finalization_ok = is_finalization_ok(store, slot)
    proposing_on_time = is_proposing_on_time(store)

    parent_slot_ok = parent_block.slot + 1 == head_block.slot
    current_time_ok = head_block.slot + 1 == slot
    single_slot_reorg = parent_slot_ok and current_time_ok

    assert store.proposer_boost_root != head_root
    head_weak = is_head_weak(store, head_root)
    parent_strong = is_parent_strong(store, parent_root)

    if all([head_late, shuffling_stable, finalization_ok, proposing_on_time,
            single_slot_reorg, head_weak, parent_strong]):
        return parent_root
    else:
        return head_root
```

## Handlers

### Modified `on_tick_per_slot`

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
    pre_state = copy(store.block_states[block.parent_root])
    assert get_current_slot(store) >= block.slot

    finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)
    assert block.slot > finalized_slot
    finalized_checkpoint_block = get_checkpoint_block(
        store, block.parent_root, store.finalized_checkpoint.epoch
    )
    assert store.finalized_checkpoint.root == finalized_checkpoint_block

    state = pre_state.copy()
    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)

    store.blocks[block_root] = block
    store.block_states[block_root] = state

    time_into_slot = (store.time - store.genesis_time) % SECONDS_PER_SLOT
    is_before_attesting_interval = time_into_slot < SECONDS_PER_SLOT // INTERVALS_PER_SLOT
    is_timely = get_current_slot(store) == block.slot and is_before_attesting_interval
    store.block_timeliness[block_root] = is_timely

    if is_timely and store.proposer_boost_root == Root():
        store.proposer_boost_root = block_root

    sync_finality_from_state(store, state)
```

### Modified `on_attestation`

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    validate_on_attestation(store, attestation, is_from_block)

    attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
    epoch_root = get_checkpoint_block(store, attestation.data.beacon_block_root, attestation_epoch)
    checkpoint = Checkpoint(epoch=attestation_epoch, root=epoch_root, height=Height(0))

    if checkpoint not in store.checkpoint_states:
        if epoch_root in store.block_states:
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

### Modified `on_attester_slashing`

```python
def on_attester_slashing(store: Store, attester_slashing: AttesterSlashing) -> None:
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    state = store.block_states[store.latest_notarized.root]
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)

    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    for index in indices:
        store.equivocating_indices.add(index)
```

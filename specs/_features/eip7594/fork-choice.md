# EIP-7594 -- Fork Choice

## Table of contents
<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
  - [Helpers](#helpers)
    - [`get_custody_parameters`](#get_custody_parameters)
    - [`get_sampling_columns`](#get_sampling_columns)
    - [`retrieve_column_sidecars`](#retrieve_column_sidecars)
    - [`is_data_available`](#is_data_available)
    - [`is_chain_available`](#is_chain_available)
    - [`get_head`](#get_head)
    - [`is_peer_sampling_required`](#is_peer_sampling_required)
- [Updated fork-choice handlers](#updated-fork-choice-handlers)
  - [`on_block`](#on_block)
    - [Pull-up tip helpers](#pull-up-tip-helpers)
      - [`compute_pulled_up_tip`](#compute_pulled_up_tip)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->

## Introduction

This is the modification of the fork choice accompanying EIP-7594

### Helpers

#### `retrieve_column_sidecars`

`def retrieve_column_sidecars(beacon_block_root: Root, columns_to_retrieve: Sequence[ColumnIndex]) -> Sequence[DataColumnSidecar]`

#### `is_data_available`

```python
def is_data_available(beacon_block_root: Root, require_peer_sampling: bool=False) -> bool:
    column_sidecars = retrieve_column_sidecars(beacon_block_root, require_peer_sampling)
    return all(
        verify_data_column_sidecar_kzg_proofs(column_sidecar)
        for column_sidecar in column_sidecars
    )
```

#### `is_chain_available`

```python
def is_chain_available(store: Store, beacon_block_root: Root) -> bool: 
    if beacon_block_root not in store.blocks:
        # Deal with Genesis edge cases, where current_justified_checkpoint and parent_root are not set
        return True
    block = store.blocks[beacon_block_root]
    block_epoch = compute_epoch_at_slot(block.slot)
    current_epoch = get_current_store_epoch(store)
    if block_epoch + MIN_EPOCHS_FOR_DATA_COLUMN_SIDECARS_REQUESTS <= current_epoch:
        return True
    parent_root = block.parent_root
    return (
        is_data_available(beacon_block_root, require_peer_sampling=True) 
        and is_chain_available(store, parent_root)
    )
    
```

#### `get_head`

```python
def get_head(store: Store) -> Root:
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)
    # Execute the LMD-GHOST fork choice
    head = store.justified_checkpoint.root
    while True:
        # Get available children for the current slot
        children = [
            root for (root, block) in blocks.items()
            if (
                block.parent_root == head
                and is_data_available(
                    root, 
                    require_peer_sampling=is_peer_sampling_required(store, block.slot)
                )
            )
        ]
        if len(children) == 0:
            return head
        # Sort by latest attesting balance with ties broken lexicographically
        # Ties broken by favoring block with lexicographically higher root
        head = max(children, key=lambda root: (get_weight(store, root), root))
```

#### `is_peer_sampling_required`

```python
def is_peer_sampling_required(store: Store, slot: Slot):
    return compute_epoch_at_slot(slot) + 2 <= get_current_store_epoch(store)
```

## Updated fork-choice handlers

### `on_block`

*Note*: The blob data availability check is removed and replaced with an availability
check on the on the justified checkpoint in the "pulled up state" of the block, after
applying `process_justification_and_finalization`.

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    """
    Run ``on_block`` upon receiving a new block.
    """
    block = signed_block.message
    # Parent block must be known
    assert block.parent_root in store.block_states
    # Make a copy of the state to avoid mutability issues
    state = copy(store.block_states[block.parent_root])
    # Blocks cannot be in the future. If they are, their consideration must be delayed until they are in the past.
    assert get_current_slot(store) >= block.slot

    # Check that block is later than the finalized epoch slot (optimization to reduce calls to get_ancestor)
    finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)
    assert block.slot > finalized_slot
    # Check block is a descendant of the finalized block at the checkpoint finalized slot
    finalized_checkpoint_block = get_checkpoint_block(
        store,
        block.parent_root,
        store.finalized_checkpoint.epoch,
    )
    assert store.finalized_checkpoint.root == finalized_checkpoint_block

    # Check the block is valid and compute the post-state
    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)

    # [New in EIP7594] Do not import the block if its unrealized justified checkpoint is not available
    pulled_up_state = state.copy()
    process_justification_and_finalization(pulled_up_state)
    assert is_chain_available(store, pulled_up_state.current_justified_checkpoint.root)

    # Add new block to the store
    store.blocks[block_root] = block
    # Add new state for this block to the store
    store.block_states[block_root] = state

    # Add block timeliness to the store
    time_into_slot = (store.time - store.genesis_time) % SECONDS_PER_SLOT
    is_before_attesting_interval = time_into_slot < SECONDS_PER_SLOT // INTERVALS_PER_SLOT
    is_timely = get_current_slot(store) == block.slot and is_before_attesting_interval
    store.block_timeliness[hash_tree_root(block)] = is_timely

    # Add proposer score boost if the block is timely and not conflicting with an existing block
    is_first_block = store.proposer_boost_root == Root()
    if is_timely and is_first_block:
        store.proposer_boost_root = hash_tree_root(block)

    # Update checkpoints in store if necessary
    update_checkpoints(store, state.current_justified_checkpoint, state.finalized_checkpoint)

    # Eagerly compute unrealized justification and finality.
    compute_pulled_up_tip(store, pulled_up_state, block_root)
```

#### Pull-up tip helpers

##### `compute_pulled_up_tip`

Modified to take `pulled_up_state`, the block's state after applying `processing_justification_and_finalization`.
The application of `processing_justification_and_finalization` now happens in `on_block`.

```python
def compute_pulled_up_tip(store: Store, pulled_up_state: BeaconState, block_root: Root) -> None:
    store.unrealized_justifications[block_root] = pulled_up_state.current_justified_checkpoint
    unrealized_justified = pulled_up_state.current_justified_checkpoint
    unrealized_finalized = pulled_up_state.finalized_checkpoint
    update_unrealized_checkpoints(store, unrealized_justified, unrealized_finalized)

    # If the block is from a prior epoch, apply the realized values
    block_epoch = compute_epoch_at_slot(store.blocks[block_root].slot)
    current_epoch = get_current_store_epoch(store)
    if block_epoch < current_epoch:
        update_checkpoints(store, unrealized_justified, unrealized_finalized)
```
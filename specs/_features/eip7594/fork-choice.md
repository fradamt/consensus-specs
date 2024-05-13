# EIP-7594 -- Fork Choice

## Table of contents
<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Containers](#containers)
- [Helpers](#helpers)
  - [Extended `PayloadAttributes`](#extended-payloadattributes)
  - [`is_data_available`](#is_data_available)
- [Updated fork-choice handlers](#updated-fork-choice-handlers)
  - [`on_block`](#on_block)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->

## Introduction

This is the modification of the fork choice accompanying EIP-7594

### Helpers

#### `LatestMessage`

*Note*: The only modification is the replacement of `epoch` with `slot`

```python
@dataclass(eq=True, frozen=True)
class LatestMessage(object):
    slot: Slot
    root: Root
```

#### `BackoffStatus`

```python
@dataclass
class BackoffStatus(object):
    branch_emptiness_score: uint64
    is_backoff_active: boolean
    slots_since_backoff_status_change: Slot
    min_slots_to_backoff: Slot
```

### `is_data_available`

```python
def is_data_available(beacon_block_root: Root, require_peer_sampling: bool) -> bool:
    # Unimplemented function which returns the node_id and custody_subnet_count
    # These are required to compute the columns to custody
    node_id, custody_subnet_count = get_custody_parameters()
    custody_columns = get_custody_columns(node_id, custody_subnet_count)
    column_sidecars = retrieve_column_sidecars(beacon_block_root, custody_columns)
    if require_peer_sampling:	
        peer_sampling_columns = get_sampling_columns()	
        column_sidecars.append(retrieve_column_sidecars(beacon_block_root, peer_sampling_columns))
    return all(
        verify_data_column_sidecar_kzg_proofs(column_sidecar)
        for column_sidecar in column_sidecars
        )
```

### `get_head`

```python
def get_head(store: Store) -> Root:
    # Initialize backoff status
    backoff_status = BackoffStatus(
        branch_emptiness_score=0,
        is_backoff_active=False,
        slots_since_backoff_status_change=Slot(0),
        min_slots_to_backoff=Slot(1)
    )
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)
    # Execute the LMD-GHOST fork choice
    head = store.justified_checkpoint.root
    slot = Slot(blocks[head].slot + 1)
    while slot <= get_current_slot(store):
        current_head = head
        # [Modified in EIP7594] Filter children with is_data_available
        children = [
            root for (root, block) in blocks.items()
            if (
                block.parent_root == head
                and block.slot == slot
                and is_data_available(root, require_peer_sampling=False)
            )
        ]
        if len(children) > 0:
            # Sort by latest attesting balance with ties broken lexicographically
            # Ties broken by favoring block with lexicographically higher root
            best_child = max(children, key=lambda root: (get_weight(store, root), root))
            best_child_weight = get_weight(store, best_child)
            empty_slot_weight = get_empty_slot_weight(store, best_child, backoff_status.is_backoff_active)
            if best_child_weight > empty_slot_weight:
                head = best_child
        update_backoff_status(backoff_status, is_empty_slot=(current_head == head))
        slot = Slot(slot + 1)
    return head
```

#### `update_backoff_status`

```python
def update_backoff_status(backoff_status: BackoffStatus, is_empty_slot: boolean) -> None:
    if is_empty_slot:
        backoff_status.branch_emptiness_score = min(
            backoff_status.branch_emptiness_score + BRANCH_EMPTINESS_SCORE_UP,
            MAX_BRANCH_EMPTINESS_SCORE
        )
    elif backoff_status.branch_emptiness_score > 0:
        backoff_status.branch_emptiness_score -= 1

    if not backoff_status.is_backoff_active:
        if backoff_status.branch_emptiness_score >= BACKOFF_ACTIVATION_THRESHOLD:
            backoff_status.is_backoff_active = True
            backoff_status.slots_since_backoff_status_change = Slot(0)
            backoff_status.min_slots_to_backoff = Slot(min(
                2 * int(backoff_status.min_slots_to_backoff),
                int(MAX_SLOTS_TO_BACKOFF)
            ))
        elif (backoff_status.branch_emptiness_score <= BACKOFF_DEACTIVATION_THRESHOLD
                and backoff_status.slots_since_backoff_status_change % SLOTS_TO_REDUCE_BACKOFF_TIME == 0):
            backoff_status.min_slots_to_backoff = Slot(max(int(backoff_status.min_slots_to_backoff) // 2, 1))
    elif (backoff_status.is_backoff_active 
            and backoff_status.branch_emptiness_score <= BACKOFF_DEACTIVATION_THRESHOLD
            and backoff_status.slots_since_backoff_status_change >= backoff_status.min_slots_to_backoff):
        backoff_status.is_backoff_active = False
        backoff_status.slots_since_backoff_status_change = Slot(0)
    backoff_status.slots_since_backoff_status_change = Slot(backoff_status.slots_since_backoff_status_change + 1)
```


#### `get_empty_slot_weight`


```python
def get_empty_slot_weight(store: Store,
                          best_child_root: Root,
                          is_backoff_active: boolean) -> Gwei:
    state = store.checkpoint_states[store.justified_checkpoint]
    slot = store.blocks[best_child_root].slot
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    attestation_score = Gwei(sum(
        state.validators[i].effective_balance for i in unslashed_and_active_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and store.latest_messages[i].slot >= slot + 1 if is_backoff_active else 0
            and not get_ancestor(store, store.latest_messages[i].root, slot) != best_child_root)
    ))
    if store.proposer_boost_root == Root():
        # Return only attestation score if ``proposer_boost_root`` is not set
        return attestation_score

    # Calculate proposer score if ``proposer_boost_root`` is set
    proposer_score = Gwei(0)
    # Boost is applied if the parent of ``best_child_root`` is an ancestor of ``proposer_boost_root`` at ``slot``
    if get_ancestor(store, store.proposer_boost_root, slot) == store.blocks[best_child_root].parent_root:
        proposer_score = get_proposer_score(store)
    return attestation_score + proposer_score
```


## Updated fork-choice handlers

### `on_block`

*Note*: The only modification is the removal of the blob data availability check.

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    """
    Run ``on_block`` upon receiving a new block.
    """
    block = signed_block.message
    # Parent block must be known
    assert block.parent_root in store.block_states
    # Make a copy of the state to avoid mutability issues
    pre_state = copy(store.block_states[block.parent_root])
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
    state = pre_state.copy()
    block_root = hash_tree_root(block)
    state_transition(state, signed_block, True)

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
    compute_pulled_up_tip(store, block_root)
```

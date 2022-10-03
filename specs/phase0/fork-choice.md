# Phase 0 -- Beacon Chain Fork Choice

## Table of contents
<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Fork choice](#fork-choice)
  - [Constant](#constant)
  - [Preset](#preset)
  - [Helpers](#helpers)
    - [`LatestMessage`](#latestmessage)
    - [`Store`](#store)
    - [`AggregateAndProof`](#aggregateandproof)
    - [`SignedAggregateAndProof`](#signedaggregateandproof)
    - [`AggregatorEquivocation`](#aggregatorequivocation)
    - [`get_forkchoice_store`](#get_forkchoice_store)
    - [`get_slots_since_genesis`](#get_slots_since_genesis)
    - [`get_current_slot`](#get_current_slot)
    - [`compute_slots_since_epoch_start`](#compute_slots_since_epoch_start)
    - [`get_ancestor`](#get_ancestor)
    - [`get_latest_attesting_balance`](#get_latest_attesting_balance)
    - [`filter_block_tree`](#filter_block_tree)
    - [`get_filtered_block_tree`](#get_filtered_block_tree)
    - [`get_head`](#get_head)
    - [`should_update_justified_checkpoint`](#should_update_justified_checkpoint)
    - [`on_aggregate and on_aggregator_equivocation` helpers](#on_aggregate`-and-`on_aggregator_equivocation`-helpers)
        - [`is_valid_signed_aggregate_and_proof`](#is_valid_signed_aggregate_and_proof)
        - [`verify_aggregator_signature`](#verify_aggregator_signature)
        - [`verify_aggregate_selection_proof`](#verify_aggregate_selection_proof)
        - [`is_slashable_aggregate_and_proof`](#is_slashable_aggregate_and_proof)
    - [`on_attestation` helpers](#on_attestation-helpers)
      - [`validate_target_epoch_against_current_time`](#validate_target_epoch_against_current_time)
      - [`validate_on_attestation`](#validate_on_attestation)
      - [`store_target_checkpoint_state`](#store_target_checkpoint_state)
      - [`update_latest_messages`](#update_latest_messages)
  - [Handlers](#handlers)
    - [`on_tick`](#on_tick)
    - [`on_block`](#on_block)
    - [`on_attestation`](#on_attestation)
    - [`on_aggregate`](#on_aggregate)
    - [`on_attester_slashing`](#on_attester_slashing)
    - [`on_aggregator_equivocation`](#on_aggregator_equivocation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->

## Introduction

This document is the beacon chain fork choice spec, part of Phase 0. It assumes the [beacon chain state transition function spec](./beacon-chain.md).

## Fork choice

The head block root associated with a `store` is defined as `get_head(store)`. At genesis, let `store = get_forkchoice_store(genesis_state)` and update `store` by running:

- `on_tick(store, time)` whenever `time > store.time` where `time` is the current Unix time
- `on_block(store, block)` whenever a block `block: SignedBeaconBlock` is received
- `on_attestation(store, attestation)` whenever an attestation `attestation` is received from within a block
- `on_aggregate(store, signed_aggregate)` whenever a signed aggregate `signed_aggregate_and_proof` is received

Any of the above handlers that trigger an unhandled exception (e.g. a failed assert or an out-of-range list access) are considered invalid. Invalid calls to handlers must not modify `store`.

*Notes*:

1) **Leap seconds**: Slots will last `SECONDS_PER_SLOT + 1` or `SECONDS_PER_SLOT - 1` seconds around leap seconds. This is automatically handled by [UNIX time](https://en.wikipedia.org/wiki/Unix_time).
2) **Honest clocks**: Honest nodes are assumed to have clocks synchronized within `SECONDS_PER_SLOT` seconds of each other.
3) **Eth1 data**: The large `ETH1_FOLLOW_DISTANCE` specified in the [honest validator document](./validator.md) should ensure that `state.latest_eth1_data` of the canonical beacon chain remains consistent with the canonical Ethereum proof-of-work chain. If not, emergency manual intervention will be required.
4) **Manual forks**: Manual forks may arbitrarily change the fork choice rule but are expected to be enacted at epoch transitions, with the fork details reflected in `state.fork`.
5) **Implementation**: The implementation found in this specification is constructed for ease of understanding rather than for optimization in computation, space, or any other resource. A number of optimized alternatives can be found [here](https://github.com/protolambda/lmd-ghost).


### Constant

| Name                 | Value       |
| -------------------- | ----------- |
| `INTERVALS_PER_SLOT` | `uint64(4)` |

### Preset

| Name                             | Value        | Unit  |  Duration  |
| -------------------------------- | ------------ | :---: | :--------: |
| `SAFE_SLOTS_TO_UPDATE_JUSTIFIED` | `2**3` (= 8) | slots | 96 seconds |

### Helpers

#### `LatestMessage`

```python
@dataclass(eq=True, frozen=True)
class LatestMessage(object):
    epoch: Epoch
    root: Root
    reference_count: uint64
```

#### `Store`

```python
@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    best_justified_checkpoint: Checkpoint
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    previous_latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    current_latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    previous_epoch_aggregates: Dict[ValidatorIndex, Set[Attestation]] = field(default_factory=dict)
    current_epoch_aggregates: Dict[ValidatorIndex, Set[Attestation]] = field(default_factory=dict)
```

#### `AggregateAndProof`

```python
class AggregateAndProof(Container):
    aggregator_index: ValidatorIndex
    aggregate: Attestation
    selection_proof: BLSSignature
```

####  `SignedAggregateAndProof`

```python
class SignedAggregateAndProof(Container):
    message: AggregateAndProof
    signature: BLSSignature
```

#### `AggregatorEquivocation`

```python
class AggregatorEquivocation(Container):
    signed_aggregate_1: SignedAggregateAndProof
    signed_aggregate_2: SignedAggregateAndProof
```

#### `get_forkchoice_store`

The provided anchor-state will be regarded as a trusted state, to not roll back beyond.
This should be the genesis state for a full client.

*Note* With regards to fork choice, block headers are interchangeable with blocks. The spec is likely to move to headers for reduced overhead in test vectors and better encapsulation. Full implementations store blocks as part of their database and will often use full blocks when dealing with production fork choice.

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
        best_justified_checkpoint=justified_checkpoint,
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={justified_checkpoint: copy(anchor_state)},
    )
```


#### `get_slots_since_genesis`

```python
def get_slots_since_genesis(store: Store) -> int:
    return (store.time - store.genesis_time) // SECONDS_PER_SLOT
```

#### `get_current_slot`

```python
def get_current_slot(store: Store) -> Slot:
    return Slot(GENESIS_SLOT + get_slots_since_genesis(store))
```

#### `compute_slots_since_epoch_start`

```python
def compute_slots_since_epoch_start(slot: Slot) -> int:
    return slot - compute_start_slot_at_epoch(compute_epoch_at_slot(slot))
```

#### `get_ancestor`

```python
def get_ancestor(store: Store, root: Root, slot: Slot) -> Root:
    block = store.blocks[root]
    if block.slot > slot:
        return get_ancestor(store, block.parent_root, slot)
    elif block.slot == slot:
        return root
    else:
        # root is older than queried slot, thus a skip slot. Return most recent root prior to slot
        return root
```

#### `get_latest_attesting_balance`

```python
def get_latest_attesting_balance(store: Store, root: Root) -> Gwei:
    state = store.checkpoint_states[store.justified_checkpoint]
    active_indices = get_active_validator_indices(state, get_current_epoch(state))
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    attestation_score = Gwei(sum(
        state.validators[i].effective_balance for i in active_indices
        if (
            i not in store.equivocating_indices
            and 
            (i in store.current_latest_messages
            and store.current_latest_messages[i].epoch in [current_epoch, current_epoch-1] # Attestation expiry
            and store.current_latest_messages[i].reference_count > 0 # reference_count == 0 means all aggregates which included this message were from equivocating aggregators
            and get_ancestor(store, store.current_latest_messages[i].root, store.blocks[root].slot) == root)
            or
            (i in store.previous_latest_messages
            and store.previous_latest_messages[i].epoch in [current_epoch, current_epoch-1] 
            and store.previous_latest_messages[i].reference_count > 0
            and get_ancestor(store, store.previous_latest_messages[i].root, store.blocks[root].slot) == root)
        )))
    return attestation_score

```

#### `filter_block_tree`

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    block = store.blocks[block_root]
    children = [
        root for root in store.blocks.keys()
        if store.blocks[root].parent_root == block_root
    ]

    # If any children branches contain expected finalized/justified checkpoints,
    # add to filtered block-tree and signal viability to parent.
    if any(children):
        filter_block_tree_result = [filter_block_tree(store, child, blocks) for child in children]
        if any(filter_block_tree_result):
            blocks[block_root] = block
            return True
        return False

    # If leaf block, check finalized/justified checkpoints as matching latest.
    head_state = store.block_states[block_root]

    correct_justified = (
        store.justified_checkpoint.epoch == GENESIS_EPOCH
        or head_state.current_justified_checkpoint == store.justified_checkpoint
    )
    correct_finalized = (
        store.finalized_checkpoint.epoch == GENESIS_EPOCH
        or head_state.finalized_checkpoint == store.finalized_checkpoint
    )
    # If expected finalized/justified, add to viable block-tree and signal viability to parent.
    if correct_justified and correct_finalized:
        blocks[block_root] = block
        return True

    # Otherwise, branch not viable
    return False
```

#### `get_filtered_block_tree`

```python
def get_filtered_block_tree(store: Store) -> Dict[Root, BeaconBlock]:
    """
    Retrieve a filtered block tree from ``store``, only returning branches
    whose leaf state's justified/finalized info agrees with that in ``store``.
    """
    base = store.justified_checkpoint.root
    blocks: Dict[Root, BeaconBlock] = {}
    filter_block_tree(store, base, blocks)
    return blocks
```

#### `get_head`

```python
def get_head(store: Store) -> Root:
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)
    # Execute the LMD-GHOST fork choice
    head = store.justified_checkpoint.root
    while True:
        children = [
            root for root in blocks.keys()
            if blocks[root].parent_root == head
        ]
        if len(children) == 0:
            return head
        # Sort by latest attesting balance with ties broken lexicographically
        # Ties broken by favoring block with lexicographically higher root
        head = max(children, key=lambda root: (get_latest_attesting_balance(store, root), root))
```

#### `should_update_justified_checkpoint`

```python
def should_update_justified_checkpoint(store: Store, new_justified_checkpoint: Checkpoint) -> bool:
    """
    To address the bouncing attack, only update conflicting justified
    checkpoints in the fork choice if in the early slots of the epoch.
    Otherwise, delay incorporation of new justified checkpoint until next epoch boundary.

    See https://ethresear.ch/t/prevention-of-bouncing-attack-on-ffg/6114 for more detailed analysis and discussion.
    """
    if compute_slots_since_epoch_start(get_current_slot(store)) < SAFE_SLOTS_TO_UPDATE_JUSTIFIED:
        return True

    justified_slot = compute_start_slot_at_epoch(store.justified_checkpoint.epoch)
    if not get_ancestor(store, new_justified_checkpoint.root, justified_slot) == store.justified_checkpoint.root:
        return False

    return True
```

#### `on_aggregate` and `on_aggregator_equivocation` helpers


##### `is_valid_signed_aggregate_and_proof`

```python
def is_valid_signed_aggregate_and_proof(state: BeaconState, signed_aggregate_and_proof: SignedAggregateAndProof) -> bool:
    return (
        verify_aggregator_signature(state, signed_aggregate_and_proof) and
        verify_aggregate_selection_proof(state, signed_aggregate_and_proof.message)
    )
```

##### `verify_aggregator_signature`

```python
def verify_aggregator_signature(state: BeaconState, signed_aggregate_and_proof: SignedAggregateAndProof) -> bool:
    aggregator = state.validators[signed_aggregate_and_proof.message.aggregator_index]
    domain = get_domain(state, DOMAIN_AGGREGATE_AND_PROOF, compute_epoch_at_slot(signed_aggregate_and_proof.message.aggregate.data.slot))
    signing_root = compute_signing_root(signed_aggregate_and_proof.message, domain)
    return bls.Verify(aggregator.pubkey, signing_root, signed_aggregate_and_proof.signature)
```

##### `verify_aggregate_selection_proof`

```python
def verify_aggregate_selection_proof(state: BeaconState, aggregate_and_proof: AggregateAndProof) -> bool:
    aggregate = aggregate_and_proof_aggregate
    aggregator = state.validators[aggregate_and_proof.aggregator_index]
    domain = get_domain(state, DOMAIN_SELECTION_PROOF, compute_epoch_at_slot(aggregate_and_proof.aggregate.data.slot))
    signing_root = compute_signing_root(aggregate_and_proof.aggregate.data.slot, domain)
    return (
        aggregate_and_proof.aggregator_index in get_beacon_committee(state, aggregate.data.slot, aggregate.data.index) and
        is_aggregator(state, aggregate.data.slot, aggregate.data.index, aggregate_and_proof.selection_proof) and
        bls.Verify(aggregator.pubkey, signing_root, aggregate_and_proof.selection_proof)
        )
```

##### `is_slashable_aggregate_and_proof`

```python
def is_slashable_aggregate_and_proof(aggregate_and_proof_1: AggregateAndProof, aggregate_and_proof_2: AggregateAndProof) -> bool:
    return (
        aggregate_and_proof_1.aggregator_index == aggregate_and_proof_2.aggregator_index and
        aggregate_and_proof_1.aggregate.slot == aggregate_and_proof_2.aggregate.slot and
        aggregate_and_proof_1 != aggregate_and_proof_2
        )
```


#### `on_attestation` helpers


##### `validate_target_epoch_against_current_time`

```python
def validate_target_epoch_against_current_time(store: Store, attestation: Attestation) -> None:
    target = attestation.data.target

    # Attestations must be from the current or previous epoch
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    # Use GENESIS_EPOCH for previous when genesis to avoid underflow
    previous_epoch = current_epoch - 1 if current_epoch > GENESIS_EPOCH else GENESIS_EPOCH
    # If attestation target is from a future epoch, delay consideration until the epoch arrives
    assert target.epoch in [current_epoch, previous_epoch]
```

##### `validate_on_attestation`

```python
def validate_on_attestation(store: Store, attestation: Attestation, is_from_block: bool) -> None:
    target = attestation.data.target

    # If the given attestation is not from a beacon block message, we have to check the target epoch scope.
    if not is_from_block:
        validate_target_epoch_against_current_time(store, attestation)

    # Check that the epoch number and slot number are matching
    assert target.epoch == compute_epoch_at_slot(attestation.data.slot)

    # Attestations target be for a known block. If target block is unknown, delay consideration until the block is found
    assert target.root in store.blocks

    # Attestations must be for a known block. If block is unknown, delay consideration until the block is found
    assert attestation.data.beacon_block_root in store.blocks
    # Attestations must not be for blocks in the future. If not, the attestation should not be considered
    assert store.blocks[attestation.data.beacon_block_root].slot <= attestation.data.slot

    # LMD vote must be consistent with FFG vote target
    target_slot = compute_start_slot_at_epoch(target.epoch)
    assert target.root == get_ancestor(store, attestation.data.beacon_block_root, target_slot)

    # Attestations can only affect the fork choice of subsequent slots.
    # Delay consideration in the fork choice until their slot is in the past.
    assert get_current_slot(store) >= attestation.data.slot + 1

    # Get state at the `target` to fully validate attestation
    target_state = store.checkpoint_states[attestation.data.target]
    indexed_attestation = get_indexed_attestation(target_state, attestation)
    assert is_valid_indexed_attestation(target_state, indexed_attestation)
```

##### `store_target_checkpoint_state`

```python
def store_target_checkpoint_state(store: Store, target: Checkpoint) -> None:
    # Store target checkpoint state if not yet seen
    if target not in store.checkpoint_states:
        base_state = copy(store.block_states[target.root])
        if base_state.slot < compute_start_slot_at_epoch(target.epoch):
            process_slots(base_state, compute_start_slot_at_epoch(target.epoch))
        store.checkpoint_states[target] = base_state
```

##### `update_latest_messages`

```python
def update_latest_messages(store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation) -> None:
    target = attestation.data.target
    beacon_block_root = attestation.data.beacon_block_root
    non_equivocating_attesting_indices = [i for i in attesting_indices if i not in store.equivocating_indices]
    for i in non_equivocating_attesting_indices:
        if i not in store.current_latest_messages: # no latest messages, add this one
            store.current_latest_messages[i] = LatestMessage(epoch=target.epoch, root=beacon_block_root, reference_count=1)
        elif target.epoch > store.current_latest_messages[i].epoch: # this one is newer, make it the current latest messages and move the existing one to previous
            store.previous_latest_messages[i] = store.current_latest_messages[i]
            store.current_latest_messages[i] = LatestMessage(epoch=target.epoch, root=beacon_block_root, reference_count=1)
        elif target.epoch == store.current_latest_messages[i].epoch: # same epoch as the current latest message, up the reference count
            # the root might not match the current latest message, but in that case the validator is equivocating
            # and they'll anyway be eliminated from the fork-choice once that is discovered, so we don't check this
            store.current_latest_messages[i].reference_count += 1
        elif i not in store.previous_latest_messages or target.epoch > store.previous_latest_messages[i].epoch: # a higher epoch message than the previous latest message, update it
            store.previous_latest_messages[i] = LatestMessage(epoch=target.epoch, root=beacon_block_root, reference_count=1)
        elif target.epoch == store.previous_latest_messages[i].epoch: # same epoch as previous latest message, up the reference count
            store.previous_latest_messages[i].reference_count += 1
            
```



### Handlers

#### `on_tick`

```python
def on_tick(store: Store, time: uint64) -> None:
    previous_slot = get_current_slot(store)

    # update store time
    store.time = time

    current_slot = get_current_slot(store)

    # Not a new epoch, return
    if not (current_slot > previous_slot and compute_slots_since_epoch_start(current_slot) == 0):
        return

    store.current_epoch_aggregates = store.previous_epoch_aggregates
    store.current_epoch_aggregates: Dict[ValidatorIndex, Set(Attestation)] = {}

    # Update store.justified_checkpoint if a better checkpoint on the store.finalized_checkpoint chain
    if store.best_justified_checkpoint.epoch > store.justified_checkpoint.epoch:
        finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)    
        ancestor_at_finalized_slot = get_ancestor(store, store.best_justified_checkpoint.root, finalized_slot)
        if ancestor_at_finalized_slot == store.finalized_checkpoint.root:
            store.justified_checkpoint = store.best_justified_checkpoint
```

#### `on_block`

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    # Parent block must be known
    assert block.parent_root in store.block_states
    # Make a copy of the state to avoid mutability issues
    pre_state = copy(store.block_states[block.parent_root])
    # Blocks cannot be in the future. If they are, their consideration must be delayed until the are in the past.
    assert get_current_slot(store) >= block.slot

    # Check that block is later than the finalized epoch slot (optimization to reduce calls to get_ancestor)
    finalized_slot = compute_start_slot_at_epoch(store.finalized_checkpoint.epoch)
    assert block.slot > finalized_slot
    # Check block is a descendant of the finalized block at the checkpoint finalized slot
    assert get_ancestor(store, block.parent_root, finalized_slot) == store.finalized_checkpoint.root

    # Check the block is valid and compute the post-state
    state = pre_state.copy()
    state_transition(state, signed_block, True)
    # Add new block to the store
    store.blocks[hash_tree_root(block)] = block
    # Add new state for this block to the store
    store.block_states[hash_tree_root(block)] = state

    # Update justified checkpoint
    if state.current_justified_checkpoint.epoch > store.justified_checkpoint.epoch:
        if state.current_justified_checkpoint.epoch > store.best_justified_checkpoint.epoch:
            store.best_justified_checkpoint = state.current_justified_checkpoint
        if should_update_justified_checkpoint(store, state.current_justified_checkpoint):
            store.justified_checkpoint = state.current_justified_checkpoint

    # Update finalized checkpoint
    if state.finalized_checkpoint.epoch > store.finalized_checkpoint.epoch:
        store.finalized_checkpoint = state.finalized_checkpoint
        store.justified_checkpoint = state.current_justified_checkpoint
```

#### `on_attestation`

```python
def on_attestation(store: Store, attestation: Attestation) -> None:
    """
    Run ``on_attestation`` upon receiving a new ``attestation`` from within a block.
    Don't run upon receiving a new ``attestation`` from within an aggregate.

    An ``attestation`` that is asserted as invalid may be valid at a later time,
    consider scheduling it for later processing in such case.
    """
    
    store_target_checkpoint_state(store, attestation.data.target)
    validate_on_attestation(store, attestation, is_from_block=True)

    # Update latest messages for attesting indices
    update_latest_messages(store, indexed_attestation.attesting_indices, attestation)
```

#### `on_aggregate`

```python
def on_aggregate(store: Store, signed_aggregate_and_proof: SignedAggregateAndProof) -> None:

    aggregate_and_proof = signed_aggregate_and_proof.message

    assert aggregate_and_proof.aggregator_index not in store.equivocating indices

    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    aggregate_epoch = compute_epoch_at_slot(aggregate_and_proof.aggregate.slot)
    assert aggregate_epoch in [current_epoch, current_epoch-1]

    assert aggregate_and_proof.aggregate not in previous_epoch_aggregates[aggregate_and_proof.aggregator_index]
    assert aggregate_and_proof.aggregate not in current_epoch_aggregates[aggregate_and_proof.aggregator_index]

    store_target_checkpoint_state(store, attestation.data.target)
    target_state = store.checkpoint_states[aggregate_and_proof.aggregate.data.target]
    assert is_valid_signed_aggregate_and_proof(target_state, signed_aggregate_and_proof)

    validate_on_attestation(store, attestation, is_from_block=False)

    if aggregate_epoch == current_epoch:
        store.current_epoch_aggregates[aggregate_and_proof.aggregator_index].add(aggregate_and_proof.aggregate)
    elif aggregate_epoch == previous_epoch:
        store.previous_epoch_aggregates[aggregate_and_proof.aggregator_index].add(aggregate_and_proof.aggregate)
```

#### `on_attester_slashing`

*Note*: `on_attester_slashing` should be called while syncing and a client MUST maintain the equivocation set of `AttesterSlashing`s from at least the latest finalized checkpoint.

```python
def on_attester_slashing(store: Store, attester_slashing: AttesterSlashing) -> None:
    """
    Run ``on_attester_slashing`` immediately upon receiving a new ``AttesterSlashing``
    from either within a block or directly on the wire.
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


#### `on_aggregator_equivocation`

```python
def on_aggregator_equivocation(store: Store, aggregator_equivocation: AggregatorEquivocation) -> None:
    """
    Run ``on_aggregator_equivocation`` immediately upon 
    receiving a new ``AggregatorEquivocation`` on the wire.
    """

    aggregate_and_proof_1 = aggregator_equivocation.signed_aggregate_1.message
    aggregate_and_proof_2 = aggregator_equivocation.signed_aggregate_2.message
    assert is_slashable_aggregate_and_proof(aggregate_and_proof_1, aggregate_and_proof_2)
    state = store.block_states[store.justified_checkpoint.root]
    assert is_valid_signed_aggregate_and_proof(state, aggregator_equivocation.signed_aggregate_1)
    assert is_valid_signed_aggregate_and_proof(state, aggregator_equivocation.signed_aggregate_2)
    aggregator_index = aggregate_and_proof_1.aggregator_index
    store.equivocating_indices.add(aggregator_index)
    unexpired_aggregates_from_equivocator = current_epoch_aggregates[aggregator_index].union(previous_epoch_aggregates[aggregator_index])
    for attestation in unexpired_aggregates_from_equivocator:
        target_state = store.checkpoint_states[attestation.data.target]
        attesting_indices = get_attesting_indices(target_state, attestation.data, attestation.aggregation_bits)
        for i in attesting_indices:
            if (i in store.current_latest_messages
                and store.current_latest_messages[i].root == attestation.data.beacon_block_root
                and store.current_latest_messages[i].epoch == compute_epoch_at_slot(attestation.data.slot)):
                store.current_latest_messages[i].reference_count -= 1
            elif (i in store.previous_latest_messages
                and store.previous_latest_messages[i].root == attestation.data.beacon_block_root
                and store.previous_latest_messages[i].epoch == compute_epoch_at_slot(attestation.data.slot)):
                store.previous_latest_messages[i].reference_count -= 1 
```
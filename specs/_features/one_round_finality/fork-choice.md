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
        justified_height=anchor_state.justified_height,  # [New in One-Round Finality]
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
    Tie-breaking: higher height wins, then higher epoch, then lower root.
    """
    if candidate_height > current_height:
        return True
    if candidate_height < current_height:
        return False
    # Same height: higher epoch wins (more recent)
    if candidate.epoch > current.epoch:
        return True
    if candidate.epoch < current.epoch:
        return False
    # Same height and epoch: lower root wins (deterministic)
    return candidate.root < current.root
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

### New `get_available_committee_weight`

```python
def get_available_committee_weight(state: BeaconState, slot: Slot) -> Gwei:
    """
    Return the total effective balance represented by the available
    committee for ``slot``.
    """
    committee = set(get_available_committee(state, slot))
    return Gwei(sum(state.validators[index].effective_balance for index in committee))
```

### Modified `calculate_committee_fraction`

*Note*: One-round finality's LMD fork choice scales reorg thresholds from the exact
available committee weight at a slot, not from
`total_active_balance / SLOTS_PER_EPOCH`.

```python
def calculate_committee_fraction(
    state: BeaconState, slot: Slot, committee_percent: uint64
) -> Gwei:
    committee_weight = get_available_committee_weight(state, slot)
    return Gwei((committee_weight * committee_percent) // 100)
```

### Modified `compute_proposer_score`

*Note*: Proposer boost scales from the exact available committee weight at the
boosted block's slot.

```python
def compute_proposer_score(state: BeaconState, slot: Slot) -> Gwei:
    committee_weight = get_available_committee_weight(state, slot)
    return Gwei((committee_weight * PROPOSER_SCORE_BOOST) // 100)
```

### Modified `get_proposer_score`

```python
def get_proposer_score(store: Store) -> Gwei:
    if store.proposer_boost_root == Root():
        return Gwei(0)

    current_slot = get_current_slot(store)
    previous_slot = GENESIS_SLOT if current_slot == GENESIS_SLOT else Slot(current_slot - 1)
    proposer_boost_root = store.proposer_boost_root
    proposer_state = store.block_states[proposer_boost_root]
    return compute_proposer_score(proposer_state, previous_slot)
```

### Modified `is_head_weak`

*Note*: Modified to use the available committee instead of iterating over
multiple beacon committees. Thresholds and equivocation committee membership
are evaluated at the **previous slot** (`slot - 1`) from proposer context,
not `head_block.slot`, to avoid missed-slot drift.

```python
def is_head_weak(store: Store, head_root: Root, slot: Slot) -> bool:
    # Calculate weight threshold for weak head
    previous_slot = GENESIS_SLOT if slot == GENESIS_SLOT else Slot(slot - 1)
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    head_state = store.block_states[head_root]
    reorg_threshold = calculate_committee_fraction(
        head_state, previous_slot, REORG_HEAD_WEIGHT_THRESHOLD
    )

    # Compute head weight including equivocations
    head_node = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
    head_weight = get_attestation_score(store, head_node, justified_state)
    # [Modified in One-Round Finality] Only available committee members for equivocations
    committee = get_available_committee(head_state, previous_slot)
    head_weight += Gwei(
        sum(
            justified_state.validators[i].effective_balance
            for i in set(committee)
            if i in store.equivocating_indices
        )
    )

    return head_weight < reorg_threshold
```

### Modified `is_parent_strong`

```python
def is_parent_strong(store: Store, root: Root, slot: Slot) -> bool:
    previous_slot = GENESIS_SLOT if slot == GENESIS_SLOT else Slot(slot - 1)
    justified_state = store.checkpoint_states[store.justified_checkpoint]
    head_state = store.block_states[root]
    parent_threshold = calculate_committee_fraction(
        head_state, previous_slot, REORG_PARENT_WEIGHT_THRESHOLD
    )
    block = store.blocks[root]
    parent_payload_status = get_parent_payload_status(store, block)
    parent_node = ForkChoiceNode(root=block.parent_root, payload_status=parent_payload_status)
    parent_weight = get_attestation_score(store, parent_node, justified_state)
    return parent_weight > parent_threshold
```

### Modified `should_apply_proposer_boost`

*Note*: Updated to pass `slot` to the modified `is_head_weak`.

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
    # [Modified in One-Round Finality] Pass slot to is_head_weak
    if not is_head_weak(store, parent_root, slot):
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

*Note*: Updated to pass `slot` to `is_head_weak` and `is_parent_strong`, and
removed `is_ffg_competitive` (no unrealized justifications in one-round finality).

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
        # [Modified in One-Round Finality] Pass proposal_slot to is_head_weak and is_parent_strong
        head_weak = is_head_weak(store, head_root, proposal_slot)
        parent_strong = is_parent_strong(store, head_root, proposal_slot)
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

*Note*: The `is_ffg_competitive` check is removed since there are no
unrealized justifications.

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
    head_weak = is_head_weak(store, head_root, slot)
    parent_strong = is_parent_strong(store, head_root, slot)

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

*Note*: Updated to accept `AvailableAttestation` instead of `Attestation`.

```python
def update_latest_messages(
    store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: AvailableAttestation
) -> None:
    slot = attestation.data.slot
    beacon_block_root = attestation.data.beacon_block_root
    payload_present = attestation.data.index == 1
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

*Note*: Finality attestations do not update fork choice weights. LMD-GHOST
fork choice is updated by `on_available_attestation` below.

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    pass
```

### New `on_available_attestation`

```python
def on_available_attestation(
    store: Store, attestation: AvailableAttestation, is_from_block: bool = False
) -> None:
    validate_on_available_attestation(store, attestation, is_from_block)

    # Derive checkpoint from slot epoch for checkpoint_states cache
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

    update_latest_messages(store, sorted(attesting_indices), attestation)
```

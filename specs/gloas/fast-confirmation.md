# Gloas -- Fast Confirmation

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Fast Confirmation Rule](#fast-confirmation-rule)
  - [Helpers](#helpers)
    - [Modified `get_node_for_root`](#modified-get_node_for_root)
    - [New `get_parent_payload_support_between_slots`](#new-get_parent_payload_support_between_slots)
    - [Modified `has_broadcast_certificate`](#modified-has_broadcast_certificate)
    - [Modified `compute_empty_slot_support_discount`](#modified-compute_empty_slot_support_discount)
- [Safe execution block](#safe-execution-block)
  - [Modified `get_safe_execution_block_hash`](#modified-get_safe_execution_block_hash)

<!-- mdformat-toc end -->

## Introduction

This is the modification of the fast confirmation rule specification
accompanying Gloas.

## Fast Confirmation Rule

### Helpers

#### Modified `get_node_for_root`

```python
def get_node_for_root(block_root: Root) -> ForkChoiceNode:
    # [Modified in Gloas:EIP7732]
    return ForkChoiceNode(root=block_root, payload_status=PAYLOAD_STATUS_PENDING)
```

#### New `get_parent_payload_support_between_slots`

Count parent votes only when they support the payload branch required by the
child. A parent vote for the other payload status supports the competing branch.

```python
def get_parent_payload_support_between_slots(
    store: Store,
    balance_source: BeaconState,
    block_root: Root,
    payload_status: PayloadStatus,
    start_slot: Slot,
    end_slot: Slot,
) -> Gwei:
    participants: Set[ValidatorIndex] = set()
    for slot in range(start_slot, end_slot + 1):
        participants.update(get_slot_committee(store, Slot(slot)))

    unslashed_and_active_indices = [
        i
        for i in participants
        if (
            not balance_source.validators[i].slashed
            and is_active_validator(balance_source.validators[i], get_current_epoch(balance_source))
        )
    ]

    return Gwei(
        sum(
            balance_source.validators[i].effective_balance
            for i in unslashed_and_active_indices
            if (
                i in store.latest_messages
                and store.latest_messages[i].root == block_root
                and is_duty_fresh_message(store, i, store.latest_messages[i])
                and i not in store.equivocating_indices
                and get_supported_node(store, store.latest_messages[i]).payload_status == payload_status
            )
        )
    )
```

#### Modified `has_broadcast_certificate`

At genesis, no slot is complete. Return `False` before subtracting one from the
typed current slot. Later slots use the weak rule's certificate check.

```python
def has_broadcast_certificate(store: Store, balance_source: BeaconState, block_root: Root) -> bool:
    """Return whether completed-slot votes certify broadcast of ``block_root``."""
    if get_current_slot(store) == 0:
        return False

    start_slot = get_block_slot(store, block_root)
    end_slot = get_current_slot(store) - 1
    participants: Set[ValidatorIndex] = set()
    for slot in range(start_slot, end_slot + 1):
        participants.update(
            index
            for index in get_slot_committee(store, Slot(slot))
            if index in store.latest_messages
            and get_latest_message_epoch(store.latest_messages[index])
            == compute_epoch_at_slot(Slot(slot))
        )

    support = Gwei(
        sum(
            balance_source.validators[index].effective_balance
            for index in participants
            if (
                is_active_validator(
                    balance_source.validators[index], get_current_epoch(balance_source)
                )
                and not balance_source.validators[index].slashed
                and index not in store.equivocating_indices
                and is_ancestor(
                    store,
                    get_supported_node(store, store.latest_messages[index]),
                    get_node_for_root(block_root),
                )
            )
        )
    )
    return support > compute_adversarial_weight(store, balance_source, start_slot, end_slot)
```

#### Modified `compute_empty_slot_support_discount`

*Note*: The empty-slot range and adversarial weight are unchanged from Phase 0.

```python
def compute_empty_slot_support_discount(
    store: Store, balance_source: BeaconState, block_root: Root
) -> Gwei:
    block = store.blocks[block_root]
    parent_block = store.blocks[block.parent_root]
    if parent_block.slot + 1 == block.slot:
        return Gwei(0)

    parent_support_in_empty_slots = get_parent_payload_support_between_slots(
        store,
        balance_source,
        block.parent_root,
        get_parent_payload_status(store, block),
        parent_block.slot + 1,
        block.slot - 1,
    )
    adversarial_weight = compute_adversarial_weight(
        store, balance_source, parent_block.slot + 1, block.slot - 1
    )
    if parent_support_in_empty_slots > adversarial_weight:
        return parent_support_in_empty_slots - adversarial_weight
    else:
        return Gwei(0)
```

## Safe execution block

### Modified `get_safe_execution_block_hash`

*Note*: In Gloas, only the parent payload of a confirmed beacon block is safe.

```python
def get_safe_execution_block_hash(fcr_store: FastConfirmationStore) -> Hash32:
    safe_block = fcr_store.store.blocks[fcr_store.confirmed_root]
    # [Modified in Gloas:EIP7732]
    return safe_block.body.signed_execution_payload_bid.message.parent_block_hash
```

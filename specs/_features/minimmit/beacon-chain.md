# Minimmit -- The Beacon Chain

## Introduction

This is the beacon chain specification for one-round finality based on the
Minimmit protocol. It replaces Casper FFG with a simplified finality gadget
where n >= 6f+1.

*Note*: This specification is built upon [Fulu](../../fulu/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Only advance when notarization or timeout occurs

Height advances when EITHER:
1. **Block notarization**: 3f+1 votes (~50%) for the target at height h
2. **Timeout notarization**: 5f+1 total votes (~83%) at height h without block notarization

### Thresholds (n >= 6f+1)

| Threshold | Stake | Purpose |
|-----------|-------|---------|
| Notarization | 3f+1 (~50%) | Block notarized, height advances |
| Finalization | 5f+1 (~83%) | Block finalized |
| Timeout | 5f+1 (~83%) | Height advances without notarization |

## Custom types

| Name | SSZ equivalent | Description |
| ---- | -------------- | ----------- |
| `Height` | `uint64` | A finality height |

## Constants

| Name | Value |
| ---- | ----- |
| `GENESIS_HEIGHT` | `Height(0)` |

## Containers

### Modified containers

#### `Checkpoint`

```python
class Checkpoint(Container):
    epoch: Epoch
    root: Root
    height: Height  # [New in Minimmit]
```

#### `AttestationData`

The `target` field is now optional. When `target` is `None`, the attestation
only counts for LMD-GHOST. When present, it also counts toward one-round
finality. Validators vote for finality **once per height**, but attest for
LMD **every epoch**.

```python
class AttestationData(Container):
    slot: Slot
    index: CommitteeIndex
    beacon_block_root: Root
    target: Optional[Checkpoint]  # [Modified in Minimmit] Optional finality vote
```

#### `BeaconState`

```python
class BeaconState(Container):
    # ... Fulu fields unchanged ...
    # Finality [Modified in Minimmit]
    current_height: Height  # [New in Minimmit]
    current_target: Checkpoint  # [New in Minimmit]
    latest_notarized: Checkpoint  # [New in Minimmit]
    finalized_checkpoint: Checkpoint
    # Height vote tracking [New in Minimmit]
    height_votes: Gwei  # [New in Minimmit]
    height_total_votes: Gwei  # [New in Minimmit]
    height_participation: Bitvector[VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` are removed.

## Helper functions

### Predicates

#### `is_slashable_attestation_data`

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    Check if ``data_1`` and ``data_2`` are slashable.
    Slashable if same validator cast two different finality votes at the same height.
    Non-finality attestations (target=None) are not slashable.
    """
    if data_1.target is None or data_2.target is None:
        return False
    return data_1.target.height == data_2.target.height and data_1.target != data_2.target
```

### Beacon state accessors

#### `get_notarization_threshold`

```python
def get_notarization_threshold(state: BeaconState) -> Gwei:
    """
    Return the notarization threshold (3f+1 where n >= 6f+1, ~50%).
    """
    total = get_total_active_balance(state)
    return total // 2 + EFFECTIVE_BALANCE_INCREMENT
```

#### `get_finalization_threshold`

```python
def get_finalization_threshold(state: BeaconState) -> Gwei:
    """
    Return the finalization/timeout threshold (5f+1 where n >= 6f+1, ~83%).
    """
    total = get_total_active_balance(state)
    return (total * 5) // 6 + EFFECTIVE_BALANCE_INCREMENT
```

## Beacon chain state transition function

### Epoch processing

#### Height transition

```python
def advance_height(state: BeaconState) -> None:
    """
    Advance to the next height, set new target, and reset vote tracking.
    """
    state.current_height = Height(state.current_height + 1)

    current_epoch = get_current_epoch(state)
    state.current_target = Checkpoint(
        epoch=current_epoch,
        root=get_block_root(state, current_epoch),
        height=state.current_height,
    )

    state.height_votes = Gwei(0)
    state.height_total_votes = Gwei(0)
    state.height_participation = Bitvector[VALIDATOR_REGISTRY_LIMIT]()
```

```python
def check_height_transition(state: BeaconState) -> None:
    """
    Check if current height should advance due to notarization or timeout.
    """
    notarization_threshold = get_notarization_threshold(state)
    finalization_threshold = get_finalization_threshold(state)

    if state.height_votes >= notarization_threshold:
        state.latest_notarized = state.current_target
        if state.height_votes >= finalization_threshold:
            state.finalized_checkpoint = state.current_target
        advance_height(state)
        return

    if state.height_total_votes >= finalization_threshold:
        advance_height(state)
        return
```

#### Modified `process_epoch`

*Note*: `process_justification_and_finalization` is replaced with `process_finality`.

```python
def process_finality(state: BeaconState) -> None:
    """
    Process one-round finality height transition.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH:
        return
    check_height_transition(state)
```

### Block processing

#### Modified `process_attestation`

```python
def process_finality_vote(state: BeaconState, attestation: Attestation) -> None:
    """
    Process finality component of an attestation.
    """
    data = attestation.data
    if data.target is None:
        return

    if data.target.height != state.current_height:
        return
    if data.target != state.current_target:
        return

    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    for committee_index in committee_indices:
        committee = get_beacon_committee(state, data.slot, committee_index)
        for i, validator_index in enumerate(committee):
            if not attestation.aggregation_bits[committee_offset + i]:
                continue
            if state.height_participation[validator_index]:
                continue
            validator = state.validators[validator_index]
            if not is_active_validator(validator, get_current_epoch(state)):
                continue
            weight = validator.effective_balance
            state.height_participation[validator_index] = True
            state.height_votes += weight
            state.height_total_votes += weight
        committee_offset += len(committee)
```

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    # ... existing Electra/Fulu validation ...

    # Process LMD participation flags (unchanged from Fulu)
    participation_flag_indices = get_attestation_participation_flag_indices(state, data, state.slot - data.slot)
    epoch_participation = state.current_epoch_participation
    if compute_epoch_at_slot(data.slot) == get_previous_epoch(state):
        epoch_participation = state.previous_epoch_participation

    for index in get_attesting_indices(state, attestation):
        for flag_index in participation_flag_indices:
            if not has_flag(epoch_participation[index], flag_index):
                epoch_participation[index] = add_flag(epoch_participation[index], flag_index)

    # Process finality vote [New in Minimmit]
    process_finality_vote(state, attestation)
```

## Testing

*Note*: The function `initialize_beacon_state_from_eth1` is modified for testing
purposes to initialize the new finality fields.

```python
def initialize_beacon_state_from_eth1(...) -> BeaconState:
    # ... existing initialization ...

    genesis_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root(), height=GENESIS_HEIGHT)
    state.current_height = GENESIS_HEIGHT
    state.current_target = genesis_checkpoint
    state.latest_notarized = genesis_checkpoint
    state.finalized_checkpoint = genesis_checkpoint
    state.height_votes = Gwei(0)
    state.height_total_votes = Gwei(0)
    state.height_participation = Bitvector[VALIDATOR_REGISTRY_LIMIT]()

    return state
```

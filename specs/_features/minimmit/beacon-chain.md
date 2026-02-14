# Minimmit -- The Beacon Chain

## Introduction

This is the beacon chain specification for one-round finality based on the
Minimmit protocol. It replaces Casper FFG with a simplified finality gadget
where n >= 6f+1.

*Note*: This specification is built upon [Fulu](../../fulu/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Only advance when justification or timeout occurs

Height advances when EITHER:

1. **Block justification**: 3f+1 votes (~50%) for the SAME target at height h
2. **Timeout**: 5f+1 total votes (~83%) at height h without block justification

### Thresholds (n >= 6f+1)

| Threshold     | Stake       | Purpose                               |
| ------------- | ----------- | ------------------------------------- |
| Justification | 3f+1 (~50%) | Block justified, height advances      |
| Finalization  | 5f+1 (~83%) | Block finalized                       |
| Timeout       | 5f+1 (~83%) | Height advances without justification |

### Vote Tracking

Votes are tracked per validator for the current and previous height. Each
validator can vote once per height. At justification time, votes are counted per
distinct target. Only votes for blocks that can be verified as on the current
chain (within the `block_roots` history window) count toward justification.
Votes for unverifiable targets count toward the total vote weight for timeout
purposes only.

## Custom types

| Name     | SSZ equivalent | Description       |
| -------- | -------------- | ----------------- |
| `Height` | `uint64`       | A finality height |

## Constants

### Finality constants

| Name                | Value               |
| ------------------- | ------------------- |
| `GENESIS_HEIGHT`    | `Height(0)`         |
| `FAR_FUTURE_HEIGHT` | `Height(2**64 - 1)` |

### Participation flag indices

*Note*: The source flag is removed in Minimmit since there is no source
checkpoint to vote on.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

| Name                         | Value                                        |
| ---------------------------- | -------------------------------------------- |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

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
finality. Validators vote for finality **once per height**, but attest for LMD
**every epoch**.

```python
class AttestationData(Container):
    slot: Slot
    index: CommitteeIndex
    beacon_block_root: Root
    target: Optional[Checkpoint]  # [Modified in Minimmit] Optional finality vote
```

*Note*: The `source` field from Altair is removed. With one-round finality,
there is no need to vote for a source checkpoint - validators only vote for the
target at the current height. The `get_attestation_participation_flag_indices`
function is modified accordingly.

#### `BeaconState`

```python
class BeaconState(Container):
    # Genesis
    genesis_time: uint64
    genesis_validators_root: Root
    # State
    slot: Slot
    fork: Fork
    latest_block_header: BeaconBlockHeader
    block_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT]
    state_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT]
    historical_roots: List[Root, HISTORICAL_ROOTS_LIMIT]
    # Eth1
    eth1_data: Eth1Data
    eth1_data_votes: List[Eth1Data, EPOCHS_PER_ETH1_VOTING_PERIOD * SLOTS_PER_EPOCH]
    eth1_deposit_index: uint64
    # Registry
    validators: List[Validator, VALIDATOR_REGISTRY_LIMIT]
    balances: List[Gwei, VALIDATOR_REGISTRY_LIMIT]
    # Randomness
    randao_mixes: Vector[Bytes32, EPOCHS_PER_HISTORICAL_VECTOR]
    # Slashings
    slashings: Vector[Gwei, EPOCHS_PER_SLASHINGS_VECTOR]
    # Participation
    previous_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    current_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    # Finality [Modified in Minimmit]
    justified_checkpoint: Checkpoint  # [Modified in Minimmit] replaces previous/current_justified
    finalized_checkpoint: Checkpoint
    # Inactivity
    inactivity_scores: List[uint64, VALIDATOR_REGISTRY_LIMIT]
    # Sync committees
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
    # Execution
    latest_execution_payload_header: ExecutionPayloadHeader
    # Withdrawals
    next_withdrawal_index: WithdrawalIndex
    next_withdrawal_validator_index: ValidatorIndex
    # History
    historical_summaries: List[HistoricalSummary, HISTORICAL_ROOTS_LIMIT]
    # Electra
    deposit_requests_start_index: uint64
    deposit_balance_to_consume: Gwei
    exit_balance_to_consume: Gwei
    earliest_exit_epoch: Epoch
    consolidation_balance_to_consume: Gwei
    earliest_consolidation_epoch: Epoch
    pending_deposits: List[PendingDeposit, PENDING_DEPOSITS_LIMIT]
    pending_partial_withdrawals: List[PendingPartialWithdrawal, PENDING_PARTIAL_WITHDRAWALS_LIMIT]
    pending_consolidations: List[PendingConsolidation, PENDING_CONSOLIDATIONS_LIMIT]
    # Fulu
    proposer_lookahead: Vector[ValidatorIndex, (MIN_SEED_LOOKAHEAD + 1) * SLOTS_PER_EPOCH]
    # Minimmit
    current_height: Height  # [New in Minimmit]
    current_height_participation: Bitvector[VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    current_height_vote_epochs: Vector[Epoch, VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    current_height_accumulated_weight: Gwei  # [New in Minimmit] Monotonic vote weight counter
    previous_height_participation: Bitvector[VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    previous_height_vote_epochs: Vector[Epoch, VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Fulu are removed.

*Note*: In the `*_vote_epochs` vectors, `FAR_FUTURE_EPOCH` indicates the
validator voted for a target that was not on the current chain (unverifiable).
The root for verifiable votes can be reconstructed using
`get_block_root_at_slot`.

## Helper functions

### Predicates

#### `is_slashable_attestation_data`

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    Check if ``data_1`` and ``data_2`` are slashable.
    Slashable if same validator cast two different finality votes at the same height.
    Non-finality attestations (target=None) are not slashable for height rules.
    """
    epoch_1 = compute_epoch_at_slot(data_1.slot)
    epoch_2 = compute_epoch_at_slot(data_2.slot)

    # Double vote at epoch
    if data_1 != data_2 and epoch_1 == epoch_2:
        return True

    if data_1.target is None or data_2.target is None:
        return False

    # Double vote at height
    return data_1 != data_2 and data_1.target.height == data_2.target.height
```

#### `is_height_participant`

```python
def is_height_participant(state: BeaconState, index: ValidatorIndex) -> bool:
    """
    Check if validator has voted at the current or previous height.
    Used for inactivity scoring: validators who have voted at either height
    are considered participating and avoid inactivity penalties.
    """
    return state.current_height_participation[index] or state.previous_height_participation[index]
```

### Beacon state accessors

#### `get_justification_threshold`

```python
def get_justification_threshold(state: BeaconState) -> Gwei:
    """
    Return the justification threshold (3f+1 where n >= 6f+1, ~50%).
    """
    total = get_total_active_balance(state)
    return total // 2
```

#### `get_finalization_threshold`

```python
def get_finalization_threshold(state: BeaconState) -> Gwei:
    """
    Return the finalization/timeout threshold (5f+1 where n >= 6f+1, ~83%).
    """
    total = get_total_active_balance(state)
    return (total * 5) // 6
```

#### `should_update_justified`

```python
def should_update_justified(current: Checkpoint, candidate: Checkpoint) -> bool:
    """
    Determine if candidate should replace current justified checkpoint.
    Tie-breaking: higher height wins, then higher epoch, then lower root.
    """
    if candidate.height > current.height:
        return True
    if candidate.height < current.height:
        return False
    # Same height: higher epoch wins (more recent)
    if candidate.epoch > current.epoch:
        return True
    if candidate.epoch < current.epoch:
        return False
    # Same height and epoch: lower root wins (deterministic)
    return candidate.root < current.root
```

## Beacon chain state transition function

### Epoch processing

#### Height transition

```python
def advance_height(state: BeaconState) -> None:
    """
    Advance to the next height and rotate vote tracking.
    """
    # Rotate current to previous
    state.previous_height_participation = state.current_height_participation
    state.previous_height_vote_epochs = state.current_height_vote_epochs

    # Advance height
    state.current_height = Height(state.current_height + 1)

    # Reset current height vote tracking
    state.current_height_participation = Bitvector[VALIDATOR_REGISTRY_LIMIT]()
    state.current_height_vote_epochs = Vector[Epoch, VALIDATOR_REGISTRY_LIMIT](FAR_FUTURE_EPOCH)
    state.current_height_accumulated_weight = Gwei(0)
```

```python
def count_height_votes(
    state: BeaconState,
    participation: Bitvector[VALIDATOR_REGISTRY_LIMIT],
    vote_epochs: Vector[Epoch, VALIDATOR_REGISTRY_LIMIT],
    height: Height,
) -> Tuple[Dict[Checkpoint, Gwei], Gwei]:
    """
    Count votes per distinct target and total votes for a height.
    Votes with FAR_FUTURE_EPOCH are unverifiable and count toward timeout only.
    Verifiable votes (valid epoch) count toward both justification and timeout.

    Returns (target_weights, total_weight).
    """
    target_weights: Dict[Checkpoint, Gwei] = {}
    total_weight = Gwei(0)
    current_epoch = get_current_epoch(state)

    for validator_index in range(len(state.validators)):
        if not participation[validator_index]:
            continue
        validator = state.validators[validator_index]
        if not is_active_validator(validator, current_epoch):
            continue

        weight = validator.effective_balance
        total_weight += weight

        epoch = vote_epochs[validator_index]
        if epoch != FAR_FUTURE_EPOCH:
            # Reconstruct checkpoint from epoch
            epoch_start_slot = compute_start_slot_at_epoch(epoch)
            # Verify epoch is still within history window
            if (
                epoch_start_slot < state.slot
                and epoch_start_slot + SLOTS_PER_HISTORICAL_ROOT >= state.slot
            ):
                root = get_block_root_at_slot(state, epoch_start_slot)
                target = Checkpoint(epoch=epoch, root=root, height=height)
                if target in target_weights:
                    target_weights[target] += weight
                else:
                    target_weights[target] = weight

    return (target_weights, total_weight)
```

```python
def update_height_justification_and_finalization(
    state: BeaconState,
    participation: Bitvector[VALIDATOR_REGISTRY_LIMIT],
    vote_epochs: Vector[Epoch, VALIDATOR_REGISTRY_LIMIT],
    height: Height,
) -> bool:
    """
    Process justification and finalization for a given height.
    Returns True if height should advance due to justification.

    Justification: 3f+1 votes for the SAME target.
    Finalization: 5f+1 votes for the SAME target.
    Timeout is handled separately via the monotonic weight counter.
    """
    justification_threshold = get_justification_threshold(state)
    finalization_threshold = get_finalization_threshold(state)

    target_weights, _ = count_height_votes(state, participation, vote_epochs, height)

    # Check for justification (3f+1 for same target)
    justified_target = None
    justified_weight = Gwei(0)
    for target, weight in target_weights.items():
        if weight >= justification_threshold:
            if justified_target is None or should_update_justified(justified_target, target):
                justified_target = target
                justified_weight = weight

    if justified_target is not None:
        # Update justified checkpoint
        if should_update_justified(state.justified_checkpoint, justified_target):
            state.justified_checkpoint = justified_target

        # Check for finalization (5f+1 for same target)
        if justified_weight >= finalization_threshold:
            state.finalized_checkpoint = justified_target

        return True  # Height advances on justification

    return False
```

```python
def process_height_progress(state: BeaconState) -> None:
    """
    Process one-round finality height transitions.
    Called after attestation processing in blocks and at epoch boundaries.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH:
        return

    # Process previous height (late-arriving votes may still justify or finalize)
    previous_height = (
        Height(state.current_height - 1)
        if state.current_height > GENESIS_HEIGHT
        else GENESIS_HEIGHT
    )
    if state.current_height > GENESIS_HEIGHT + 1:
        update_height_justification_and_finalization(
            state,
            state.previous_height_participation,
            state.previous_height_vote_epochs,
            previous_height,
        )

    # Process current height: justification check
    should_advance = update_height_justification_and_finalization(
        state,
        state.current_height_participation,
        state.current_height_vote_epochs,
        state.current_height,
    )

    # Timeout via monotonic counter: 5f+1 accumulated weight
    if not should_advance:
        should_advance = state.current_height_accumulated_weight >= get_finalization_threshold(
            state
        )

    if should_advance:
        advance_height(state)
```

#### Modified `process_inactivity_updates`

*Note*: Inactivity scoring is based on **height participation** rather than
epoch-based `TIMELY_TARGET_FLAG_INDEX`. A validator is considered participating
if it has voted at the current or previous height. The leak trigger uses
`finality_delay` (epochs since last finalization), providing **accountable
liveness**: any period without finalization incurs an economic cost on
non-participants. This means a delayed safety violation (double finalization
after the weak subjectivity period) always requires either slashable
equivocation or having endured inactivity leak penalties — the adversary cannot
cheaply wait out the quorum intersection and then double-finalize for free.

```python
def process_inactivity_updates(state: BeaconState) -> None:
    # Skip the genesis epoch as score updates are based on the previous epoch participation
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    for index in get_eligible_validator_indices(state):
        # Increase the inactivity score of height-inactive validators
        if is_height_participant(state, ValidatorIndex(index)):
            state.inactivity_scores[index] -= min(1, state.inactivity_scores[index])
        else:
            state.inactivity_scores[index] += INACTIVITY_SCORE_BIAS
        # Decrease the inactivity score of all eligible validators during a leak-free epoch
        if not is_in_inactivity_leak(state):
            state.inactivity_scores[index] -= min(
                INACTIVITY_SCORE_RECOVERY_RATE, state.inactivity_scores[index]
            )
```

#### Modified `get_inactivity_penalty_deltas`

```python
def get_inactivity_penalty_deltas(state: BeaconState) -> Tuple[Sequence[Gwei], Sequence[Gwei]]:
    """
    Return the inactivity penalty deltas by considering height participation and inactivity scores.
    [Modified in Minimmit] Uses height participation instead of epoch-based target flag.
    """
    rewards = [Gwei(0) for _ in range(len(state.validators))]
    penalties = [Gwei(0) for _ in range(len(state.validators))]
    for index in get_eligible_validator_indices(state):
        if not is_height_participant(state, ValidatorIndex(index)):
            penalty_numerator = (
                state.validators[index].effective_balance * state.inactivity_scores[index]
            )
            penalty_denominator = INACTIVITY_SCORE_BIAS * INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
            penalties[index] += Gwei(penalty_numerator // penalty_denominator)
    return rewards, penalties
```

#### Modified `process_epoch`

```python
def process_epoch(state: BeaconState) -> None:
    # [Modified in Minimmit] process_justification_and_finalization removed
    # (height transitions happen per-block in process_operations)
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_registry_updates(state)
    process_slashings(state)
    process_eth1_data_reset(state)
    process_pending_deposits(state)
    process_pending_consolidations(state)
    process_effective_balance_updates(state)
    process_slashings_reset(state)
    process_randao_mixes_reset(state)
    process_historical_summaries_update(state)
    process_participation_flag_updates(state)
    process_sync_committee_updates(state)
    process_proposer_lookahead(state)
```

### Block processing

#### Modified `get_attestation_participation_flag_indices`

```python
def get_attestation_participation_flag_indices(
    state: BeaconState, data: AttestationData, inclusion_delay: uint64, is_matching_target: bool
) -> Sequence[int]:
    """
    Return the flag indices that are satisfied by an attestation.
    [Modified in Minimmit] source field removed, target validity passed as parameter.
    """
    is_matching_head = data.beacon_block_root == get_block_root_at_slot(state, data.slot)

    participation_flag_indices = []
    if is_matching_target and inclusion_delay <= SLOTS_PER_EPOCH:
        participation_flag_indices.append(TIMELY_TARGET_FLAG_INDEX)
    if (
        is_matching_target
        and is_matching_head
        and inclusion_delay == MIN_ATTESTATION_INCLUSION_DELAY
    ):
        participation_flag_indices.append(TIMELY_HEAD_FLAG_INDEX)

    return participation_flag_indices
```

#### Modified `process_attestation`

```python
def process_finality_vote(
    state: BeaconState, data: AttestationData, attesting_indices: Sequence[ValidatorIndex]
) -> bool:
    """
    Process finality component of an attestation.
    Records each validator's vote epoch for the current or previous height.
    Returns True if the target is valid (correct height and on current chain).
    """
    if data.target is None:
        return False

    # Determine which height this vote is for
    previous_height = (
        Height(state.current_height - 1)
        if state.current_height > GENESIS_HEIGHT
        else GENESIS_HEIGHT
    )

    if data.target.height == state.current_height:
        participation = state.current_height_participation
        vote_epochs = state.current_height_vote_epochs
    elif data.target.height == previous_height:
        participation = state.previous_height_participation
        vote_epochs = state.previous_height_vote_epochs
    else:
        return False  # Wrong height

    # Check if target is on the current chain
    epoch_start_slot = compute_start_slot_at_epoch(data.target.epoch)
    target_on_chain = False
    if epoch_start_slot + SLOTS_PER_HISTORICAL_ROOT > state.slot:
        if epoch_start_slot < state.slot:
            if get_block_root_at_slot(state, epoch_start_slot) == data.target.root:
                target_on_chain = True

    current_epoch = get_current_epoch(state)
    for validator_index in attesting_indices:
        if participation[validator_index]:
            continue
        validator = state.validators[validator_index]
        if not is_active_validator(validator, current_epoch):
            continue

        # Record the vote
        participation[validator_index] = True
        if target_on_chain:
            vote_epochs[validator_index] = data.target.epoch
        else:
            vote_epochs[validator_index] = FAR_FUTURE_EPOCH

        # Accumulate weight into monotonic counter (current height only)
        if data.target.height == state.current_height:
            state.current_height_accumulated_weight += validator.effective_balance

    return target_on_chain
```

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    # [Modified in Minimmit] target is optional, validate epoch from slot instead
    assert compute_epoch_at_slot(data.slot) in (get_previous_epoch(state), get_current_epoch(state))
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot

    assert data.index == 0
    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    participants_epoch = compute_epoch_at_slot(data.slot)
    for committee_index in committee_indices:
        assert committee_index < get_committee_count_per_slot(state, participants_epoch)
        committee = get_beacon_committee(state, data.slot, committee_index)
        committee_attesters = set(
            attester_index
            for i, attester_index in enumerate(committee)
            if attestation.aggregation_bits[committee_offset + i]
        )
        assert len(committee_attesters) > 0
        committee_offset += len(committee)

    assert len(attestation.aggregation_bits) == committee_offset
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))

    # Get attesting indices once [Modified in Minimmit]
    attesting_indices = get_attesting_indices(state, attestation)

    # Process finality vote and get target validity [New in Minimmit]
    is_matching_target = process_finality_vote(state, data, attesting_indices)

    participation_flag_indices = get_attestation_participation_flag_indices(
        state, data, state.slot - data.slot, is_matching_target
    )

    if participants_epoch == get_current_epoch(state):
        epoch_participation = state.current_epoch_participation
    else:
        epoch_participation = state.previous_epoch_participation

    proposer_reward_numerator = 0
    for index in attesting_indices:
        for flag_index, weight in enumerate(PARTICIPATION_FLAG_WEIGHTS):
            if flag_index in participation_flag_indices and not has_flag(
                epoch_participation[index], flag_index
            ):
                epoch_participation[index] = add_flag(epoch_participation[index], flag_index)
                proposer_reward_numerator += get_base_reward(state, index) * weight

    proposer_reward_denominator = (
        (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
    )
    proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
    increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### Modified `process_operations`

*Note*: After processing all attestations, call `process_height_progress` to
check if justification or timeout thresholds have been reached.

```python
def process_operations(state: BeaconState, body: BeaconBlockBody) -> None:
    eth1_deposit_index_limit = min(
        state.eth1_data.deposit_count, state.deposit_requests_start_index
    )
    if state.eth1_deposit_index < eth1_deposit_index_limit:
        assert len(body.deposits) == min(
            MAX_DEPOSITS, eth1_deposit_index_limit - state.eth1_deposit_index
        )
    else:
        assert len(body.deposits) == 0

    def for_ops(operations: Sequence[Any], fn: Callable[[BeaconState, Any], None]) -> None:
        for operation in operations:
            fn(state, operation)

    for_ops(body.proposer_slashings, process_proposer_slashing)
    for_ops(body.attester_slashings, process_attester_slashing)
    for_ops(body.attestations, process_attestation)

    # Process height transition after all attestations [New in Minimmit]
    process_height_progress(state)

    for_ops(body.deposits, process_deposit)
    for_ops(body.voluntary_exits, process_voluntary_exit)
    for_ops(body.bls_to_execution_changes, process_bls_to_execution_change)
    for_ops(body.execution_requests.deposits, process_deposit_request)
    for_ops(body.execution_requests.withdrawals, process_withdrawal_request)
    for_ops(body.execution_requests.consolidations, process_consolidation_request)
```

## Testing

*Note*: For testing purposes, `initialize_beacon_state_from_eth1` must be
modified to initialize the new finality fields:

- `current_height = GENESIS_HEIGHT`
- `justified_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root(), height=GENESIS_HEIGHT)`
- `finalized_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root(), height=GENESIS_HEIGHT)`
- `current_height_participation = Bitvector[VALIDATOR_REGISTRY_LIMIT]()`
- `current_height_vote_epochs = Vector[Epoch, VALIDATOR_REGISTRY_LIMIT](FAR_FUTURE_EPOCH)`
- `current_height_accumulated_weight = Gwei(0)`
- `previous_height_participation = Bitvector[VALIDATOR_REGISTRY_LIMIT]()`
- `previous_height_vote_epochs = Vector[Epoch, VALIDATOR_REGISTRY_LIMIT](FAR_FUTURE_EPOCH)`

## Design Notes

### Inactivity Leak and Accountable Liveness

The inactivity leak in Minimmit serves three purposes:

1. **Eventual height progress.** During non-finality, the inactivity leak
   reduces non-participants' balances, shrinking the denominator
   (`total_active_balance`) while the numerator
   (`current_height_accumulated_weight`) is monotonically non-decreasing. The
   ratio eventually exceeds 5/6, triggering a timeout and advancing the height.

2. **Incentivizing participation.** Validators who do not vote at the current
   height lose balance, creating an economic incentive to participate in
   finality.

3. **Accountable liveness.** The leak trigger is `finality_delay` — epochs since
   the last finalization. Any period without finalization incurs an economic
   cost on non-participants, regardless of whether heights are advancing via
   timeout. This provides a crucial safety complement: an adversary cannot
   simply wait for the weak subjectivity period to expire (allowing the quorum
   intersection to degrade via validator set churn) and then double-finalize at
   zero cost. During the waiting period, non-finalizing validators accumulate
   inactivity penalties. A delayed safety violation therefore always has an
   economic cost — either direct slashing (if within the weak subjectivity
   period) or inactivity penalties (if waiting beyond it). The two mechanisms
   are complementary: slashing handles violations with detectable equivocation,
   while the leak handles the case where the adversary avoids equivocation by
   waiting.

### Monotonic Weight Counter

The `current_height_accumulated_weight` counter snapshots each validator's
`effective_balance` at the time their vote is included. The balance is never
retroactively adjusted, even if the validator is later penalized. This
monotonicity property is essential for the eventual height progress guarantee:

- **Numerator** (`C(h)`): non-decreasing (new votes add; nothing subtracts)
- **Denominator** (`total_active_balance`): strictly decreasing during
  non-finality (no activations; non-participants leak; ejections)
- **Convergence**: `C(h) / total_active_balance` eventually exceeds 5/6

The counter may overcount the *current* weight of participants if a validator's
balance decreases after voting (e.g., due to slashing). This overcounting is
bounded by the slashed amount and is within the adversarial model for
accountable safety. Whether recounting from current balances would be preferable
is an open design question — the monotonic approach provides a cleaner
convergence argument at the cost of potential overcounting.

### Dual Participation Tracking

Minimmit tracks participation along two independent dimensions:

| Dimension    | Tracked via                         | Used for                                      |
| ------------ | ----------------------------------- | --------------------------------------------- |
| Epoch-based  | `*_epoch_participation` flags       | LMD-GHOST rewards, available chain incentives |
| Height-based | `*_height_participation` bitvectors | Inactivity scoring, finality progress         |

The epoch-based flags (target, head) keep the available chain functioning even
during non-finality. The height-based bitvectors drive the inactivity leak and
the monotonic counter. A validator can be an active epoch participant (attesting
for LMD-GHOST every epoch) while being a height non-participant (not voting for
finality), and will be penalized by the inactivity leak for the latter.

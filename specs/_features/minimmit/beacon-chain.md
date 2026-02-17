# Minimmit -- The Beacon Chain

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for one-round finality based on the
Minimmit protocol. It replaces Casper FFG with a simplified finality gadget
where n >= 6f+1, and separates finality votes from LMD-GHOST attestations.

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at epoch boundaries

Within an epoch, a height becomes **advance-eligible** when EITHER:

1. **Block justification**: 3f+1 votes (~50%) for the SAME target at height h
1. **Timeout**: allVotes - maxVotes > n/3 at height h (genuine vote disagreement)

At the next epoch transition, the height advances by one if it was marked
advance-eligible.

### Thresholds (n >= 6f+1)

| Threshold     | Stake                   | Purpose                                                      |
| ------------- | ----------------------- | ------------------------------------------------------------ |
| Justification | 3f+1 (~50%)             | Block justified, marks height to advance at epoch transition |
| Finalization  | 5f+1 (~83%)             | Block finalized                                              |
| Timeout       | allVotes-maxVotes > n/3 | Marks height to advance without justification                |

### Decoupled Consensus

Finality votes and LMD-GHOST attestations are fully separated:

- **Finality attestations**: All active validators vote once per height via
  `FinalityAttestation`. Carried in blocks with a bitfield over all validators.
- **LMD attestations**: A small committee (~512 validators) attests per slot for
  fork choice. Uses the same selection logic as the PTC but with different
  randomness.

### Vote Tracking

Finality votes are tracked per validator for the current and previous height.
Each validator can vote once per height. The actual `Checkpoint` voted for is
stored per validator. At justification time, votes are counted per distinct
target. Targets are considered on-chain if they are either:

- Verifiable in the current `block_roots` history window, or
- Proven via a historical Merkle proof against `historical_summaries`.

Timeout uses vote distribution: it fires when `allVotes - maxVotes > 1/3` of
total active balance, ensuring a branch where one target dominates cannot time
out (see Notarization Path Safety).

## Configuration

Warning: this configuration is not definitive.

| Name                     | Value                                 |
| ------------------------ | ------------------------------------- |
| `MINIMMIT_FORK_VERSION`  | `Version('0x10000000')`               |
| `MINIMMIT_FORK_EPOCH`    | `Epoch(18446744073709551615)` **TBD** |

## Custom types

| Name     | SSZ equivalent | Description       |
| -------- | -------------- | ----------------- |
| `Height` | `uint64`       | A finality height |

## Constants

### Finality constants

| Name             | Value       |
| ---------------- | ----------- |
| `GENESIS_HEIGHT` | `Height(0)` |

### Participation flag indices

*Note*: The source flag is removed in Minimmit since there is no source
checkpoint to vote on.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

*Note*: The source weight (14/64) is redistributed to target in Minimmit since
the source flag is removed. The sum of participation weights remains 54/64
(same as Altair: 14 + 26 + 14 = 54, now 40 + 14 = 54).

| Name                         | Value                                        |
| ---------------------------- | -------------------------------------------- |
| `TIMELY_TARGET_WEIGHT`       | `uint64(40)`                                 |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

### Domain types

| Name                       | Value                      |
| -------------------------- | -------------------------- |
| `DOMAIN_FINALITY_ATTESTER` | `DomainType('0x0E000000')` |
| `DOMAIN_AVAILABLE_ATTESTER` | `DomainType('0x0F000000')` |

### Misc

| Name                            | Value                                          |
| ------------------------------- | ---------------------------------------------- |
| `AVAILABLE_COMMITTEE_SIZE`      | `uint64(2**9)` (= 512)                         |
| `HISTORICAL_TARGET_PROOF_DEPTH` | `uint64(floorlog2(SLOTS_PER_HISTORICAL_ROOT))` |

## Preset

### Max operations per block

| Name                           | Value       |
| ------------------------------ | ----------- |
| `MAX_FINALITY_ATTESTATIONS`    | `uint64(4)` |
| `MAX_FINALITY_SLASHINGS`       | `uint64(1)` |
| `MAX_HISTORICAL_TARGET_PROOFS` | `uint64(1)` |

## Containers

### New containers

#### `FinalityAttestationData`

```python
class FinalityAttestationData(Container):
    target: Checkpoint  # Standard (epoch, root) — the finality vote target
    height: Height  # The finality height being voted on
```

#### `FinalityAttestation`

```python
class FinalityAttestation(Container):
    data: FinalityAttestationData
    aggregation_bits: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # Bit i = validator index i
    signature: BLSSignature  # Aggregate signature over all attesting validators
```

*Note*: The `aggregation_bits` has actual length equal to `len(state.validators)`.
Bit `i` corresponds to validator index `i`. Bits for non-active validators must
not be set. This prevents bloating aggregates with inactive validators.

#### `IndexedFinalityAttestation`

```python
class IndexedFinalityAttestation(Container):
    attesting_indices: List[ValidatorIndex, VALIDATOR_REGISTRY_LIMIT]
    data: FinalityAttestationData
    signature: BLSSignature
```

#### `FinalitySlashing`

```python
class FinalitySlashing(Container):
    attestation_1: IndexedFinalityAttestation
    attestation_2: IndexedFinalityAttestation
```

#### `HistoricalTargetProof`

```python
class HistoricalTargetProof(Container):
    target: Checkpoint
    block_root_proof: Vector[Bytes32, HISTORICAL_TARGET_PROOF_DEPTH]
```

### Modified containers

#### `AttestationData`

*Note*: Both `source` and `target` are removed. Attestations are now pure
LMD-GHOST votes. Finality is handled by `FinalityAttestation`.

```python
class AttestationData(Container):
    slot: Slot
    index: CommitteeIndex
    beacon_block_root: Root  # LMD vote
```

#### `Attestation`

*Note*: Minimmit uses a single available committee per slot, so `committee_bits`
is removed.

```python
class Attestation(Container):
    aggregation_bits: Bitlist[AVAILABLE_COMMITTEE_SIZE]
    data: AttestationData
    signature: BLSSignature
```

#### `BeaconBlockBody`

*Note*: `attester_slashings` is removed in Minimmit. LMD attestation data is
non-slashable; slashing for voting applies only to `FinalityAttestationData`
double-votes at the same height.

```python
class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    # Gloas:EIP7732
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: List[PayloadAttestation, MAX_PAYLOAD_ATTESTATIONS]
    # Minimmit
    finality_attestations: List[FinalityAttestation, MAX_FINALITY_ATTESTATIONS]  # [New in Minimmit]
    finality_slashings: List[FinalitySlashing, MAX_FINALITY_SLASHINGS]  # [New in Minimmit]
    historical_target_proofs: List[
        HistoricalTargetProof, MAX_HISTORICAL_TARGET_PROOFS
    ]  # [New in Minimmit]
```

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
    justified_checkpoint: Checkpoint  # [Modified in Minimmit] replaces justification_bits + previous/current_justified
    finalized_checkpoint: Checkpoint
    justified_height: (
        Height  # [New in Minimmit] Height at which current justified checkpoint was justified
    )
    # Inactivity
    inactivity_scores: List[uint64, VALIDATOR_REGISTRY_LIMIT]
    # Sync committees
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
    # Gloas:EIP7732
    latest_execution_payload_bid: ExecutionPayloadBid
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
    # Gloas:EIP7732
    builders: List[Builder, BUILDER_REGISTRY_LIMIT]
    next_withdrawal_builder_index: BuilderIndex
    execution_payload_availability: Bitvector[SLOTS_PER_HISTORICAL_ROOT]
    builder_pending_payments: Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH]
    builder_pending_withdrawals: List[BuilderPendingWithdrawal, BUILDER_PENDING_WITHDRAWALS_LIMIT]
    latest_block_hash: Hash32
    payload_expected_withdrawals: List[Withdrawal, MAX_WITHDRAWALS_PER_PAYLOAD]
    # Minimmit
    current_height: Height  # [New in Minimmit]
    current_height_participation: Bitvector[VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    current_height_vote_targets: Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    current_height_canonical_target: (
        Checkpoint  # [New in Minimmit] Canonical target for incentives/leak
    )
    previous_height_participation: Bitvector[VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    previous_height_vote_targets: Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT]  # [New in Minimmit]
    previous_height_canonical_target: (
        Checkpoint  # [New in Minimmit] Canonical target for previous height
    )
    height_advance_pending: (
        boolean  # [New in Minimmit] Cached per-epoch height advancement decision
    )
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: The `*_vote_targets` vectors store the actual `Checkpoint` each
validator voted for. The participation bitvectors track whether a validator has
voted. The default zero value `Checkpoint()` in unvoted entries is distinguished
from an actual vote by the participation bit.

*Note*: The fields `current_height_canonical_target` and
`previous_height_canonical_target` store the full canonical `Checkpoint` for each
tracked height. Only votes matching the canonical target exempt a validator from
the inactivity leak (see `is_height_participant`). Votes for other on-chain
targets still count toward justification and timeout but do not protect against
leaking.

*Note*: `height_advance_pending` is set during block processing when the current
height becomes advance-eligible (via justification or timeout), and consumed at
epoch processing to perform the actual height transition.

## Helper functions

### Predicates

#### New `is_slashable_finality_attestation_data`

```python
def is_slashable_finality_attestation_data(
    data_1: FinalityAttestationData, data_2: FinalityAttestationData
) -> bool:
    """
    Check if two finality attestations are slashable.
    Slashable if different votes at the same height.
    """
    return data_1 != data_2 and data_1.height == data_2.height
```

#### New `is_height_participant`

```python
def is_height_participant(state: BeaconState, index: ValidatorIndex) -> bool:
    """
    Check if validator voted for the canonical target at the current height.
    Only votes matching the height's canonical target checkpoint count as participation
    for inactivity scoring. Votes for other targets still contribute to justification
    and timeout but do not exempt the validator from the inactivity leak.

    Unlike epoch participation (which checks current and previous epoch), this only
    checks the current height: once a height advances, its finality is settled and
    participation at previous heights is irrelevant for leak purposes.
    """
    return (
        not state.validators[index].slashed
        and state.current_height_participation[index]
        and state.current_height_vote_targets[index] == state.current_height_canonical_target
    )
```

### Beacon state accessors

#### New `get_previous_height`

```python
def get_previous_height(state: BeaconState) -> Height:
    if state.current_height > GENESIS_HEIGHT:
        return Height(state.current_height - 1)
    return GENESIS_HEIGHT
```

#### New `get_justification_threshold`

```python
def get_justification_threshold(state: BeaconState) -> Gwei:
    """
    Return the justification threshold (3f+1 where n >= 6f+1, ~50%).
    """
    total = get_total_active_balance(state)
    return total // 2
```

#### New `get_finalization_threshold`

```python
def get_finalization_threshold(state: BeaconState) -> Gwei:
    """
    Return the finalization threshold (5f+1 where n >= 6f+1, ~83%).
    """
    total = get_total_active_balance(state)
    return (total * 5) // 6
```

#### New `should_update_justified`

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

#### New `get_available_committee`

```python
def get_available_committee(
    state: BeaconState, slot: Slot
) -> Sequence[ValidatorIndex]:
    """
    [New in Minimmit] Return the 512-member available committee for the given slot.
    This committee votes for LMD-GHOST fork choice via on-chain attestations.
    """
    epoch = compute_epoch_at_slot(slot)
    seed = hash(get_seed(state, epoch, DOMAIN_AVAILABLE_ATTESTER) + uint_to_bytes(slot))
    active_indices = get_active_validator_indices(state, epoch)
    return compute_balance_weighted_selection(
        state, active_indices, seed, size=AVAILABLE_COMMITTEE_SIZE, shuffle_indices=True
    )
```

*Note*: Both the available committee and PTC use `compute_balance_weighted_selection`
from the full active validator set. They differ only in the seed (different
domain types: `DOMAIN_AVAILABLE_ATTESTER` vs `DOMAIN_PTC_ATTESTER`).

#### Modified `get_beacon_committee`

*Note*: The standard beacon committees are restored to their Gloas behavior for
network-level subnet assignment and finality vote duty triggers. On-chain LMD
attestations use the available committee above.

```python
def get_beacon_committee(
    state: BeaconState, slot: Slot, index: CommitteeIndex
) -> Sequence[ValidatorIndex]:
    """
    [Modified in Minimmit] Beacon committees are used for network-level subnet
    assignment, aggregation, and finality vote duty triggers. On-chain LMD
    attestations use the available committee.
    """
    epoch = compute_epoch_at_slot(slot)
    committees_per_slot = get_committee_count_per_slot(state, epoch)
    return compute_committee(
        indices=get_active_validator_indices(state, epoch),
        seed=get_seed(state, epoch, DOMAIN_BEACON_ATTESTER),
        index=(slot % SLOTS_PER_EPOCH) * committees_per_slot + index,
        count=committees_per_slot * SLOTS_PER_EPOCH,
    )
```

#### Modified `get_ptc`

```python
def get_ptc(state: BeaconState, slot: Slot) -> Vector[ValidatorIndex, PTC_SIZE]:
    """
    [Modified in Minimmit] Select PTC from the entire active validator set,
    not from beacon committee members.
    """
    epoch = compute_epoch_at_slot(slot)
    seed = hash(get_seed(state, epoch, DOMAIN_PTC_ATTESTER) + uint_to_bytes(slot))
    active_indices = get_active_validator_indices(state, epoch)
    return compute_balance_weighted_selection(
        state, active_indices, seed, size=PTC_SIZE, shuffle_indices=True
    )
```

#### Modified `get_attesting_indices`

```python
def get_attesting_indices(state: BeaconState, attestation: Attestation) -> Set[ValidatorIndex]:
    """
    Return the set of attesting indices for a single-committee attestation.
    [Modified in Minimmit] Uses available committee (no committee_bits).
    """
    committee = get_available_committee(state, attestation.data.slot)
    assert len(attestation.aggregation_bits) == len(committee)
    return set(
        attester_index
        for i, attester_index in enumerate(committee)
        if attestation.aggregation_bits[i]
    )
```

### Finality attestation helpers

#### New `get_finality_attesting_indices`

```python
def get_finality_attesting_indices(
    state: BeaconState, finality_attestation: FinalityAttestation
) -> Set[ValidatorIndex]:
    """
    Return the set of attesting validator indices from a finality attestation.
    """
    current_epoch = get_current_epoch(state)
    indices = set()
    for i, bit in enumerate(finality_attestation.aggregation_bits):
        if bit:
            assert is_active_validator(state.validators[i], current_epoch)
            indices.add(ValidatorIndex(i))
    return indices
```

#### New `get_indexed_finality_attestation`

```python
def get_indexed_finality_attestation(
    state: BeaconState, finality_attestation: FinalityAttestation
) -> IndexedFinalityAttestation:
    """
    Return the indexed finality attestation corresponding to ``finality_attestation``.
    """
    attesting_indices = get_finality_attesting_indices(state, finality_attestation)
    return IndexedFinalityAttestation(
        attesting_indices=sorted(attesting_indices),
        data=finality_attestation.data,
        signature=finality_attestation.signature,
    )
```

#### New `is_valid_indexed_finality_attestation`

```python
def is_valid_indexed_finality_attestation(
    state: BeaconState, indexed_attestation: IndexedFinalityAttestation
) -> bool:
    """
    Check if ``indexed_attestation`` is not empty, has sorted and unique indices
    and has a valid aggregate signature.
    """
    indices = indexed_attestation.attesting_indices
    if len(indices) == 0 or not indices == sorted(set(indices)):
        return False
    pubkeys = [state.validators[i].pubkey for i in indices]
    domain = get_domain(state, DOMAIN_FINALITY_ATTESTER, indexed_attestation.data.target.epoch)
    signing_root = compute_signing_root(indexed_attestation.data, domain)
    return bls.FastAggregateVerify(pubkeys, signing_root, indexed_attestation.signature)
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(state: BeaconState) -> None:
    """
    Advance to the next height and rotate vote tracking.
    """
    # Reuse prior-height buffers to avoid re-allocating large fixed-size vectors.
    next_current_participation = state.previous_height_participation
    next_current_vote_targets = state.previous_height_vote_targets

    # Rotate current to previous
    state.previous_height_participation = state.current_height_participation
    state.previous_height_vote_targets = state.current_height_vote_targets
    state.previous_height_canonical_target = state.current_height_canonical_target

    # Advance height
    state.current_height = Height(state.current_height + 1)

    # Set canonical target for the new height
    epoch = get_current_epoch(state)
    state.current_height_canonical_target = Checkpoint(
        epoch=epoch,
        root=get_block_root_at_slot(state, compute_start_slot_at_epoch(epoch)),
    )

    # Reset current height vote tracking
    state.current_height_participation = next_current_participation
    state.current_height_vote_targets = next_current_vote_targets
    for validator_index in range(len(state.validators)):
        state.current_height_participation[validator_index] = False
        state.current_height_vote_targets[validator_index] = Checkpoint()
```

#### New `count_height_votes`

```python
def count_height_votes(
    state: BeaconState,
    participation: Bitvector[VALIDATOR_REGISTRY_LIMIT],
    vote_targets: Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT],
) -> Tuple[Dict[Checkpoint, Gwei], Gwei]:
    """
    Count votes per distinct target and total votes for a height.
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

        target = vote_targets[validator_index]
        if target not in target_weights:
            target_weights[target] = Gwei(0)
        target_weights[target] += weight

    return (target_weights, total_weight)
```

#### New `is_target_on_chain`

```python
def is_target_on_chain(
    state: BeaconState,
    target: Checkpoint,
    height: Height,
    has_historical_target: bool,
    historical_target: Checkpoint,
) -> bool:
    """
    Check if a target checkpoint is verifiably on the current chain.
    Three paths: (1) canonical target for this height (stored in state),
    (2) in-window ``block_roots`` check, (3) block-provided historical proof.
    """
    # Canonical target is always on-chain (recorded in state at height start)
    if height == state.current_height:
        canonical = state.current_height_canonical_target
    else:
        canonical = state.previous_height_canonical_target
    if target == canonical:
        return True

    # In-window check via block_roots
    if is_target_in_block_roots_window(state, target):
        epoch_start_slot = compute_start_slot_at_epoch(target.epoch)
        return get_block_root_at_slot(state, epoch_start_slot) == target.root

    # Historical proof fallback
    return has_historical_target and target == historical_target
```

#### New `update_height_justification_and_finalization`

```python
def update_height_justification_and_finalization(
    state: BeaconState,
    participation: Bitvector[VALIDATOR_REGISTRY_LIMIT],
    vote_targets: Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT],
    height: Height,
    historical_target_proofs: Sequence[HistoricalTargetProof],
) -> bool:
    """
    Process justification, finalization, and timeout for a given height.
    Returns True if this height is advance-eligible.

    Justification: > 1/2 of total active balance votes for the same on-chain target.
    Finalization: > 5/6 of total active balance votes for the same on-chain target.
    Timeout: total voting weight - max single-target weight > 1/3 of total active balance.
    The timeout rule uses ALL votes (including off-chain targets), preventing
    timeout when a conflicting branch has finalization.
    """
    assert len(historical_target_proofs) <= 1
    has_historical_target = len(historical_target_proofs) == 1
    consumed_historical_target = False
    should_advance = False
    historical_target = Checkpoint()
    if has_historical_target:
        proof = historical_target_proofs[0]
        target = proof.target
        # Special fallback path only for targets outside the block_roots window.
        assert not is_target_in_block_roots_window(state, target)
        assert is_target_in_historical_summaries(state, historical_target_proofs[0])
        historical_target = target

    justification_threshold = get_justification_threshold(state)
    finalization_threshold = get_finalization_threshold(state)

    target_weights, total_weight = count_height_votes(state, participation, vote_targets)

    # Find the unique target exceeding justification threshold.
    # Strict > guarantees at most one target can qualify.
    has_justified_target = False
    justified_target = Checkpoint()
    justified_weight = Gwei(0)
    for target, weight in target_weights.items():
        if weight > justification_threshold:
            has_justified_target = True
            justified_target = target
            justified_weight = weight
            break

    if has_justified_target and is_target_on_chain(
        state, justified_target, height, has_historical_target, historical_target
    ):
        if has_historical_target:
            assert justified_target == historical_target

        # LJ monotonicity: only update justified checkpoint if epoch >= current
        if justified_target.epoch >= state.justified_checkpoint.epoch:
            state.justified_checkpoint = justified_target
            state.justified_height = height

        # Check for finalization (5/6 for same target)
        if justified_weight > finalization_threshold:
            if justified_target.epoch > state.finalized_checkpoint.epoch:
                state.finalized_checkpoint = justified_target

        state.height_advance_pending = True  # Height is advance-eligible on justification
        should_advance = True
        consumed_historical_target = has_historical_target

    # Timeout: allVotes - maxVotes > 1/3 of total active balance
    # This counts ALL votes (including off-chain targets), so a branch
    # where 5/6 voted for the same (off-chain) target cannot timeout
    max_target_weight = max(target_weights.values()) if target_weights else Gwei(0)
    timeout_threshold = get_total_active_balance(state) // 3
    if total_weight - max_target_weight > timeout_threshold:
        assert not has_historical_target
        state.height_advance_pending = True
        should_advance = True

    assert not has_historical_target or consumed_historical_target
    return should_advance
```

#### New `process_height_progress`

```python
def process_height_progress(
    state: BeaconState, historical_target_proofs: Sequence[HistoricalTargetProof]
) -> None:
    """
    Process one-round finality checkpoint updates during block processing.
    Called after finality attestation processing in blocks.
    Height advancement itself occurs in epoch processing.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    # Process previous height (late-arriving votes may still justify or finalize,
    # but must not set height_advance_pending -- that height already advanced)
    if state.current_height > GENESIS_HEIGHT + 1:
        saved_pending = state.height_advance_pending
        update_height_justification_and_finalization(
            state,
            state.previous_height_participation,
            state.previous_height_vote_targets,
            get_previous_height(state),
            [],
        )
        state.height_advance_pending = saved_pending

    # Process current height
    update_height_justification_and_finalization(
        state,
        state.current_height_participation,
        state.current_height_vote_targets,
        state.current_height,
        historical_target_proofs,
    )
```

#### New `process_height_epoch_transition`

```python
def process_height_epoch_transition(state: BeaconState) -> None:
    """
    Advance height at epoch transition if block processing marked it eligible.
    """
    if state.height_advance_pending:
        advance_height(state)
        state.height_advance_pending = False
```

#### New `is_target_in_block_roots_window`

```python
def is_target_in_block_roots_window(state: BeaconState, target: Checkpoint) -> bool:
    """
    Return True if ``target`` can be checked directly in ``state.block_roots``.
    """
    epoch_start_slot = compute_start_slot_at_epoch(target.epoch)
    return (
        epoch_start_slot < state.slot and epoch_start_slot + SLOTS_PER_HISTORICAL_ROOT >= state.slot
    )
```

#### New `is_target_in_historical_summaries`

```python
def is_target_in_historical_summaries(
    state: BeaconState, historical_target_proof: HistoricalTargetProof
) -> bool:
    """
    Verify a target root against ``historical_summaries`` for out-of-window epochs.
    """
    target = historical_target_proof.target
    epoch_start_slot = compute_start_slot_at_epoch(target.epoch)
    if epoch_start_slot >= state.slot:
        return False

    historical_summary_index = uint64(epoch_start_slot // SLOTS_PER_HISTORICAL_ROOT)
    if historical_summary_index >= len(state.historical_summaries):
        return False

    return is_valid_merkle_branch(
        leaf=target.root,
        branch=historical_target_proof.block_root_proof,
        depth=HISTORICAL_TARGET_PROOF_DEPTH,
        index=uint64(epoch_start_slot % SLOTS_PER_HISTORICAL_ROOT),
        root=state.historical_summaries[historical_summary_index].block_summary_root,
    )
```

#### Modified `process_inactivity_updates`

*Note*: Inactivity scoring is based on the **canonical target** rather than
epoch-based `TIMELY_TARGET_FLAG_INDEX`. Each height has a fixed canonical target
epoch, set when the height starts. A validator is considered participating only
if it voted for the canonical target at the current height. This gives a tight
property analogous to FFG: **either finalization occurs, or at least 1/6 of
total stake is being leaked**. The only way to avoid the leak is for finality to
happen — there is no middle ground where validators participate but finality
stalls without penalty. The leak trigger uses `finality_delay` (epochs since
last finalization), providing **accountable liveness**: any period without
finalization incurs an economic cost on non-participants regardless of whether
heights are advancing via timeout.

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
    # (checkpoint updates happen per-block; height transitions happen at epoch transition)
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_registry_updates(state)
    process_slashings(state)
    process_eth1_data_reset(state)
    process_pending_deposits(state)
    process_pending_consolidations(state)
    process_builder_pending_payments(state)  # [New in Gloas:EIP7732]
    process_effective_balance_updates(state)
    process_slashings_reset(state)
    process_randao_mixes_reset(state)
    process_historical_summaries_update(state)
    process_participation_flag_updates(state)
    process_sync_committee_updates(state)
    process_proposer_lookahead(state)
    process_height_epoch_transition(state)
```

### Block processing

#### Modified `is_valid_indexed_attestation`

```python
def is_valid_indexed_attestation(
    state: BeaconState, indexed_attestation: IndexedAttestation
) -> bool:
    """
    Check if ``indexed_attestation`` is not empty, has sorted and unique indices and has a valid aggregate signature.
    [Modified in Minimmit] Always uses slot epoch for the signing domain (no target).
    """
    indices = indexed_attestation.attesting_indices
    if len(indices) == 0 or not indices == sorted(set(indices)):
        return False
    pubkeys = [state.validators[i].pubkey for i in indices]
    epoch = compute_epoch_at_slot(indexed_attestation.data.slot)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, epoch)
    signing_root = compute_signing_root(indexed_attestation.data, domain)
    return bls.FastAggregateVerify(pubkeys, signing_root, indexed_attestation.signature)
```

#### Modified `get_attestation_participation_flag_indices`

```python
def get_attestation_participation_flag_indices(
    state: BeaconState, data: AttestationData, inclusion_delay: uint64
) -> Sequence[int]:
    """
    Return the flag indices that are satisfied by an attestation.
    [Modified in Minimmit] Only head flag from LMD attestations. Target flag
    is set by finality attestation processing.
    """
    is_matching_head = data.beacon_block_root == get_block_root_at_slot(state, data.slot)

    participation_flag_indices = []
    if is_matching_head and inclusion_delay == MIN_ATTESTATION_INCLUSION_DELAY:
        participation_flag_indices.append(TIMELY_HEAD_FLAG_INDEX)

    return participation_flag_indices
```

#### Modified `process_attestation`

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in Minimmit] LMD-only attestation processing. No finality component.
    Strict current/previous epoch only. Uses available committee.
    """
    data = attestation.data
    attestation_epoch = compute_epoch_at_slot(data.slot)
    assert attestation_epoch in (get_previous_epoch(state), get_current_epoch(state))
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    assert data.index < 2  # [Gloas:EIP7732] Payload availability signal

    committee = get_available_committee(state, data.slot)
    assert len(attestation.aggregation_bits) == len(committee)
    assert any(attestation.aggregation_bits)
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))

    attesting_indices = get_attesting_indices(state, attestation)

    participation_flag_indices = get_attestation_participation_flag_indices(
        state, data, state.slot - data.slot
    )

    # [Modified in Gloas:EIP7732]
    if attestation_epoch == get_current_epoch(state):
        epoch_participation = state.current_epoch_participation
        payment = state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH]
    else:
        epoch_participation = state.previous_epoch_participation
        payment = state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH]

    proposer_reward_numerator = 0
    for index in attesting_indices:
        # [New in Gloas:EIP7732]
        will_set_new_flag = False

        for flag_index, weight in enumerate(PARTICIPATION_FLAG_WEIGHTS):
            if flag_index in participation_flag_indices and not has_flag(
                epoch_participation[index], flag_index
            ):
                epoch_participation[index] = add_flag(epoch_participation[index], flag_index)
                proposer_reward_numerator += get_base_reward(state, index) * weight
                # [New in Gloas:EIP7732]
                will_set_new_flag = True

        # [New in Gloas:EIP7732]
        if (
            will_set_new_flag
            and is_attestation_same_slot(state, data)
            and payment.withdrawal.amount > 0
        ):
            payment.weight += state.validators[index].effective_balance

    proposer_reward_denominator = (
        (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
    )
    proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
    increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### New `process_finality_attestation`

```python
def process_finality_attestation(
    state: BeaconState, finality_attestation: FinalityAttestation
) -> None:
    """
    Process a finality attestation: validate, record votes, and set target flag.
    """
    data = finality_attestation.data

    # Validate bitfield length
    assert len(finality_attestation.aggregation_bits) == len(state.validators)
    assert any(finality_attestation.aggregation_bits)

    # Validate height
    assert data.height in (state.current_height, get_previous_height(state))

    # Validate and get attesting indices (also asserts active validators only)
    indexed = get_indexed_finality_attestation(state, finality_attestation)
    assert is_valid_indexed_finality_attestation(state, indexed)

    # Determine which height this vote is for
    if data.height == state.current_height:
        participation = state.current_height_participation
        vote_targets = state.current_height_vote_targets
    else:
        participation = state.previous_height_participation
        vote_targets = state.previous_height_vote_targets

    # Check if this vote matches the canonical target (for incentives)
    is_matching_canonical = False
    if data.height == state.current_height:
        is_matching_canonical = data.target == state.current_height_canonical_target
    elif data.height == get_previous_height(state):
        is_matching_canonical = data.target == state.previous_height_canonical_target

    current_epoch = get_current_epoch(state)
    proposer_reward_numerator = 0

    for validator_index in indexed.attesting_indices:
        if participation[validator_index]:
            continue
        validator = state.validators[validator_index]
        if not is_active_validator(validator, current_epoch):
            continue

        # Record the vote
        participation[validator_index] = True
        vote_targets[validator_index] = data.target

        # Set TIMELY_TARGET flag if matching canonical target
        if is_matching_canonical:
            epoch_participation = state.current_epoch_participation
            if not has_flag(epoch_participation[validator_index], TIMELY_TARGET_FLAG_INDEX):
                epoch_participation[validator_index] = add_flag(
                    epoch_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
                )
                proposer_reward_numerator += (
                    get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT
                )

    # Proposer reward for included finality votes
    if proposer_reward_numerator > 0:
        proposer_reward_denominator = (
            (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
        )
        proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
        increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### New `process_finality_slashing`

```python
def process_finality_slashing(state: BeaconState, finality_slashing: FinalitySlashing) -> None:
    """
    Process a finality slashing: two conflicting finality attestations at the same height.
    """
    attestation_1 = finality_slashing.attestation_1
    attestation_2 = finality_slashing.attestation_2

    assert is_slashable_finality_attestation_data(attestation_1.data, attestation_2.data)
    assert is_valid_indexed_finality_attestation(state, attestation_1)
    assert is_valid_indexed_finality_attestation(state, attestation_2)

    slashable_indices = set(attestation_1.attesting_indices).intersection(
        attestation_2.attesting_indices
    )

    for index in sorted(slashable_indices):
        if is_slashable_validator(state.validators[index], get_current_epoch(state)):
            slash_validator(state, index)
```

#### Modified `process_operations`

*Note*: Historical target proof is an optional fallback for targets outside the
`block_roots` window. At most one proof may be included per block, and it must
be consumed by an actual justification in that block. This strict-use rule is
enforced inside `update_height_justification_and_finalization`.

```python
def process_operations(state: BeaconState, body: BeaconBlockBody) -> None:
    # Disable former deposit mechanism once all prior deposits are processed
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
    for_ops(body.attestations, process_attestation)
    for_ops(body.deposits, process_deposit)
    for_ops(body.voluntary_exits, process_voluntary_exit)
    for_ops(body.bls_to_execution_changes, process_bls_to_execution_change)
    # [New in Gloas:EIP7732]
    for_ops(body.payload_attestations, process_payload_attestation)
    # [New in Minimmit]
    for_ops(body.finality_slashings, process_finality_slashing)
    for_ops(body.finality_attestations, process_finality_attestation)

    # Update height justification/finalization after all finality attestations [New in Minimmit]
    process_height_progress(state, body.historical_target_proofs)
```

## Fork transition

### New `upgrade_to_minimmit`

```python
def upgrade_to_minimmit(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)

    post = BeaconState(
        # Genesis
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        # State
        slot=pre.slot,
        fork=Fork(
            previous_version=pre.fork.current_version,
            current_version=MINIMMIT_FORK_VERSION,  # [Modified in Minimmit]
            epoch=epoch,
        ),
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=pre.historical_roots,
        # Eth1
        eth1_data=pre.eth1_data,
        eth1_data_votes=pre.eth1_data_votes,
        eth1_deposit_index=pre.eth1_deposit_index,
        # Registry
        validators=pre.validators,
        balances=pre.balances,
        # Randomness
        randao_mixes=pre.randao_mixes,
        # Slashings
        slashings=pre.slashings,
        # Participation
        previous_epoch_participation=pre.previous_epoch_participation,
        current_epoch_participation=pre.current_epoch_participation,
        # Finality [Modified in Minimmit]
        # Removed: justification_bits, previous_justified_checkpoint, current_justified_checkpoint
        justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        justified_height=GENESIS_HEIGHT,  # [New in Minimmit]
        # Inactivity
        inactivity_scores=pre.inactivity_scores,
        # Sync committees
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
        # Gloas:EIP7732
        latest_execution_payload_bid=pre.latest_execution_payload_bid,
        # Withdrawals
        next_withdrawal_index=pre.next_withdrawal_index,
        next_withdrawal_validator_index=pre.next_withdrawal_validator_index,
        # History
        historical_summaries=pre.historical_summaries,
        # Electra
        deposit_requests_start_index=pre.deposit_requests_start_index,
        deposit_balance_to_consume=pre.deposit_balance_to_consume,
        exit_balance_to_consume=pre.exit_balance_to_consume,
        earliest_exit_epoch=pre.earliest_exit_epoch,
        consolidation_balance_to_consume=pre.consolidation_balance_to_consume,
        earliest_consolidation_epoch=pre.earliest_consolidation_epoch,
        pending_deposits=pre.pending_deposits,
        pending_partial_withdrawals=pre.pending_partial_withdrawals,
        pending_consolidations=pre.pending_consolidations,
        # Fulu
        proposer_lookahead=pre.proposer_lookahead,
        # Gloas:EIP7732
        builders=pre.builders,
        next_withdrawal_builder_index=pre.next_withdrawal_builder_index,
        execution_payload_availability=pre.execution_payload_availability,
        builder_pending_payments=pre.builder_pending_payments,
        builder_pending_withdrawals=pre.builder_pending_withdrawals,
        latest_block_hash=pre.latest_block_hash,
        payload_expected_withdrawals=pre.payload_expected_withdrawals,
        # Minimmit [New in Minimmit]
        current_height=GENESIS_HEIGHT,
        current_height_participation=Bitvector[VALIDATOR_REGISTRY_LIMIT](),
        current_height_vote_targets=Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT](),
        # [Modified in Minimmit] Direct block_roots access because get_block_root
        # asserts slot < state.slot, which fails at the fork epoch boundary where
        # state.slot == compute_start_slot_at_epoch(epoch).
        current_height_canonical_target=Checkpoint(
            epoch=epoch,
            root=pre.block_roots[compute_start_slot_at_epoch(epoch) % SLOTS_PER_HISTORICAL_ROOT],
        ),
        previous_height_participation=Bitvector[VALIDATOR_REGISTRY_LIMIT](),
        previous_height_vote_targets=Vector[Checkpoint, VALIDATOR_REGISTRY_LIMIT](),
        previous_height_canonical_target=Checkpoint(
            epoch=epoch,
            root=pre.block_roots[compute_start_slot_at_epoch(epoch) % SLOTS_PER_HISTORICAL_ROOT],
        ),
        height_advance_pending=False,
    )

    return post
```

## Genesis

#### Modified `initialize_beacon_state_from_eth1`

*Note*: The `current_height_canonical_target` and
`previous_height_canonical_target` use a zero root at genesis since no block
exists yet. The `epoch <= GENESIS_EPOCH + 1` guard in `process_height_progress`
prevents finality processing in the first two epochs, so this zero root is
never used for on-chain verification.

```python
def initialize_beacon_state_from_eth1(
    eth1_block_hash: Hash32, eth1_timestamp: uint64, deposits: Sequence[Deposit]
) -> BeaconState:
    fork = Fork(
        previous_version=GENESIS_FORK_VERSION,
        current_version=GENESIS_FORK_VERSION,
        epoch=GENESIS_EPOCH,
    )
    state = BeaconState(
        genesis_time=eth1_timestamp + GENESIS_DELAY,
        fork=fork,
        eth1_data=Eth1Data(deposit_count=uint64(len(deposits)), block_hash=eth1_block_hash),
        latest_block_header=BeaconBlockHeader(body_root=hash_tree_root(BeaconBlockBody())),
        randao_mixes=[eth1_block_hash]
        * EPOCHS_PER_HISTORICAL_VECTOR,  # Seed RANDAO with Eth1 entropy
    )

    # Process deposits
    leaves = list(map(lambda deposit: deposit.data, deposits))
    for index, deposit in enumerate(deposits):
        deposit_data_list = List[DepositData, 2**DEPOSIT_CONTRACT_TREE_DEPTH](*leaves[: index + 1])
        state.eth1_data.deposit_root = hash_tree_root(deposit_data_list)
        process_deposit(state, deposit)

    # Process activations
    for index, validator in enumerate(state.validators):
        balance = state.balances[index]
        validator.effective_balance = min(
            balance - balance % EFFECTIVE_BALANCE_INCREMENT, MAX_EFFECTIVE_BALANCE
        )
        if validator.effective_balance == MAX_EFFECTIVE_BALANCE:
            validator.activation_eligibility_epoch = GENESIS_EPOCH
            validator.activation_epoch = GENESIS_EPOCH

    # Set genesis validators root for domain separation and chain versioning
    state.genesis_validators_root = hash_tree_root(state.validators)

    # [New in Minimmit] Initialize finality fields
    state.current_height = GENESIS_HEIGHT
    state.justified_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.finalized_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.justified_height = GENESIS_HEIGHT
    state.current_height_canonical_target = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.previous_height_canonical_target = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.height_advance_pending = False

    return state
```

## Design Notes

### Decoupled Finality and LMD-GHOST

In Minimmit, finality and fork choice are fully separated:

- **Finality attestations** (`FinalityAttestation`): All active validators vote
  once per height. Carried in blocks via a bitfield over all validators indexed
  by validator index. These determine justification, finalization, and timeout.
- **LMD attestations** (`Attestation`): A small 512-member available committee
  votes per slot for fork choice. This committee is selected from the full
  active set using `compute_balance_weighted_selection` (same mechanism as PTC).

This separation means finality votes can be included in blocks regardless of
attestation age constraints. Heights may span many epochs, and finality votes
from earlier epochs are always relevant. The bitfield indexed by validator index
makes this possible without committee derivation.

### Inactivity Leak and Accountable Liveness

The inactivity leak in Minimmit serves three purposes:

1. **Eventual height progress.** During non-finality, the inactivity leak
   reduces non-participants' balances, shrinking `total_active_balance` while
   the recomputed total voted weight stays relatively stable (voters for the
   canonical target are not leaked). As non-voters' balances shrink, vote
   dispersion (`allVotes - maxVotes`) grows relative to the shrinking total
   active balance, eventually exceeding 1/3 and triggering a timeout.

1. **Incentivizing participation.** Validators who do not vote at the current
   height lose balance, creating an economic incentive to participate in
   finality.

1. **Accountable liveness.** The leak trigger is `finality_delay` — epochs since
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

### Timeout via Vote Distribution

The timeout rule fires when `allVotes - maxVotes > 1/3` of total active
balance, where `allVotes` is the total voted weight and `maxVotes` is the weight
of the most-voted-for target. Both values are recomputed from current
`effective_balance` values by `count_height_votes`, reflecting actual current
weight rather than snapshotting balances at vote time.

Convergence during non-finality: the inactivity leak reduces non-participants'
balances (shrinking `total_active_balance`), while canonical-target voters are
not leaked and retain their weight. As non-voters' balances shrink, the voted
fraction increases and vote dispersion (allVotes - maxVotes) grows relative to
total active balance, eventually exceeding 1/3 and triggering timeout.

### On-Chain Target Verification

Justification requires that the winning target is **verifiably on the current
chain** (accountability: without this, an adversary could double-finalize
without slashing). The `is_target_on_chain` function checks three paths in
order:

1. **Canonical target** (stored in state at height start): always on-chain by
   construction, even if the epoch boundary slot has since left the
   `block_roots` window. This is the fast path for honest validators.
1. **In-window `block_roots`** check: for non-canonical targets whose epoch
   boundary slot is still within the 8192-slot circular buffer.
1. **Historical proof fallback**: a block-provided `HistoricalTargetProof`
   proving the target root against `historical_summaries` for out-of-window
   epochs. The proof is accepted only for out-of-window targets and must be
   consumed by a successful justification in that block (enforced by
   `update_height_justification_and_finalization`), otherwise the block is
   invalid.

The strict `>` justification threshold guarantees at most one target can
qualify, eliminating the need for tie-breaking in candidate selection.

### Slashed Validators and Finality Progress

Slashed-but-active validators' votes count toward justification and timeout.
This is a deliberate deviation from FFG, where `get_unslashed_participating_indices`
excludes slashed validators from the finality numerator. The rationale: once a
validator is slashed, the equivocating votes already exist and can be replayed on
another chain history. Excluding slashed weight from the canonical chain only delays
recovery (especially timeout) without improving safety. The slashed weight is bounded
by the adversarial model.

For inactivity scoring, `is_height_participant` checks `not slashed`, matching
Altair's `get_unslashed_participating_indices` behavior: slashed validators always
accumulate inactivity scores regardless of their votes.

### Canonical Target and the Tight Leak Property

Each height has a **canonical target**: a fixed checkpoint `(epoch, root)`
determined when the height starts (in `advance_height`). The canonical target
epoch is the current epoch at the time of height advancement, and the root is
the epoch boundary block root. This is stored in
`current_height_canonical_target`.

All honest validators vote for the canonical target regardless of which epoch
they attest in. Since the target epoch and root are explicit in the
`FinalityAttestationData`, a validator voting in a later epoch still votes for
the same canonical target.

This design gives a **tight unconditional property** analogous to FFG:

> **Either finalization occurs, or a minimum fraction of stake is being
> leaked.**

Specifically, during non-finality:

- If no justification at the current height: the canonical target has weight \<
  total/2, so >50% of stake is being leaked (did not vote for the canonical
  target).
- If justification but no finalization: the canonical target has weight >
  total/2 but \<= 5\*total/6, so at least total/6 of stake is being leaked.
- If finalization occurs: no leak. This is the **only** escape from being
  leaked.

The minimum leaked fraction (total/6) matches Minimmit's accountable safety
baseline. This is structural — it does not depend on honest behavior.

### Incentive Design

Target rewards (`TIMELY_TARGET` flag) are earned through finality attestation
processing: validators whose finality vote matches the canonical target get the
flag set in `process_finality_attestation`. Head rewards (`TIMELY_HEAD` flag)
are earned through regular LMD attestation processing.

Head rewards are independent of target matching, unlike FFG where head requires
target. This ensures validators always have incentive to participate in
LMD-GHOST regardless of finality state.

### Notarization Path Safety

The timeout rule (`allVotes - maxVotes > 1/3`) ensures that a height can only
time out if there is genuine disagreement among voters. If any single target has
`>= 2/3` of the vote weight, the timeout cannot fire (`allVotes - maxVotes <= 1/3`). This prevents a critical attack vector:

If block B is finalized at height H (5/6 votes for B), a conflicting branch
cannot time out at height H, because `maxVotes >= 5/6` implies `allVotes - maxVotes <= 1/6 < 1/3`. The conflicting branch is stuck at height H and cannot
make progress -- exactly the safety property required by the Minimmit protocol.

This is achieved by tracking the actual `Checkpoint` each validator voted for
(not just the epoch), so that `maxVotes` can be computed across ALL votes,
including those for targets not on the current chain. Without this, a vote for
an off-chain target was only counted toward total weight (as an "unverifiable"
vote), which allowed the same signed vote to contribute to finalization on one
branch and timeout on another -- a zero-slashable conflicting finalization
vulnerability.

### Justified Checkpoint Monotonicity

In `BeaconState`, the justified checkpoint only updates if the new target's
epoch is >= the current justified checkpoint's epoch. On the same chain, this
is equivalent to a descendant check. Height advancement is independent of the
LJ update — it always occurs on justification regardless.

This check is state-level only (not in the Store's `update_checkpoints`), since
the Store spans multiple chains where epoch comparison is not a valid descendant
check.

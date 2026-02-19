# One-Round Finality -- The Beacon Chain

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for one-round finality based on the
one-round finality protocol. It replaces Casper FFG with a simplified finality gadget
where n >= 6f+1, and separates finality attestations from LMD-GHOST attestations.

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at epoch boundaries

At each epoch transition, the height advances if EITHER condition holds:

1. **Justification**: 3f+1 attestations (~50%) for the SAME target at height h
1. **Skip**: allVotes - maxVotes > n/3 at height h (genuine attestation disagreement)

### Thresholds (n >= 6f+1)

| Threshold     | Stake                   | Purpose                                                      |
| ------------- | ----------------------- | ------------------------------------------------------------ |
| Justification | 3f+1 (~50%)             | Block justified, height advances at epoch transition         |
| Finalization  | 5f+1 (~83%)             | Block finalized                                              |
| Skip          | allVotes-maxVotes > n/3 | Marks height to advance without justification                |

### Decoupled Consensus

Finality and LMD-GHOST use different attestation types:

- **Attestations**: All active validators attest once per height via standard
  beacon committee attestations (Electra format). `AttestationData` carries a
  finality target and height. These determine justification, finalization, and
  skip. Attester slashings enforce the height double-vote condition.
- **Available attestations**: A small 512-member available committee attests per
  slot for fork choice via `AvailableAttestation`. This committee is selected
  from the full active set using `compute_balance_weighted_selection` (same
  mechanism as PTC).

### Attestation Tracking

Finality attestations are tracked per validator for the current and previous
height. Each validator can attest once per height. The actual `Checkpoint`
attested to is stored per validator. At justification time, attestations are
counted per distinct target. Targets are considered on-chain if they are either:

- Verifiable in the current `block_roots` history window, or
- Proven via a historical Merkle proof against `historical_summaries`.

Skip uses attestation distribution: it fires when `allVotes - maxVotes >
1/3` of total active balance, ensuring a branch where one target dominates
cannot skip (see Notarization Path Safety).

## Configuration

Warning: this configuration is not definitive.

| Name                     | Value                                 |
| ------------------------ | ------------------------------------- |
| `ONE_ROUND_FINALITY_FORK_VERSION`  | `Version('0x10000000')`               |
| `ONE_ROUND_FINALITY_FORK_EPOCH`    | `Epoch(18446744073709551615)` **TBD** |

## Custom types

| Name     | SSZ equivalent | Description       |
| -------- | -------------- | ----------------- |
| `Height` | `uint64`       | A finality height |

## Constants

### Finality constants

| Name                                   | Value       |
| -------------------------------------- | ----------- |
| `GENESIS_HEIGHT`                       | `Height(0)` |
| `JUSTIFICATION_THRESHOLD_NUMERATOR`    | `uint64(1)` |
| `JUSTIFICATION_THRESHOLD_DENOMINATOR`  | `uint64(2)` |
| `FINALIZATION_THRESHOLD_NUMERATOR`     | `uint64(5)` |
| `FINALIZATION_THRESHOLD_DENOMINATOR`   | `uint64(6)` |
| `SKIP_THRESHOLD_NUMERATOR`             | `uint64(1)` |
| `SKIP_THRESHOLD_DENOMINATOR`           | `uint64(3)` |

### Participation flag indices

*Note*: The source flag is removed in one-round finality since there is no source
checkpoint to attest to.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

*Note*: The source weight (14/64) is redistributed to target in one-round finality since
the source flag is removed. The sum of participation weights remains 54/64
(same as Altair: 14 + 26 + 14 = 54, now 40 + 14 = 54).

| Name                         | Value                                        |
| ---------------------------- | -------------------------------------------- |
| `TIMELY_TARGET_WEIGHT`       | `uint64(40)`                                 |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

### Domain types

| Name                       | Value                      |
| -------------------------- | -------------------------- |
| `DOMAIN_AVAILABLE_ATTESTER` | `DomainType('0x0F000000')` |

### Misc

| Name                            | Value                                          |
| ------------------------------- | ---------------------------------------------- |
| `AVAILABLE_COMMITTEE_SIZE`      | `uint64(2**9)` (= 512)                         |
| `HISTORICAL_TARGET_PROOF_DEPTH` | `uint64(floorlog2(SLOTS_PER_HISTORICAL_ROOT))` |

## Preset

### Max operations per block

| Name                            | Value       |
| ------------------------------- | ----------- |
| `MAX_AVAILABLE_ATTESTATIONS`    | `uint64(8)` |
| `MAX_HISTORICAL_TARGET_PROOFS`  | `uint64(1)` |

## Containers

### New containers

#### `AvailableAttestationData`

```python
class AvailableAttestationData(Container):
    slot: Slot
    payload_available: boolean  # Payload availability signal
    beacon_block_root: Root  # LMD attestation for fork choice
```

#### `AvailableAttestation`

```python
class AvailableAttestation(Container):
    aggregation_bits: Bitlist[AVAILABLE_COMMITTEE_SIZE]
    data: AvailableAttestationData
    signature: BLSSignature
```

#### `HistoricalTargetProof`

```python
class HistoricalTargetProof(Container):
    target: Checkpoint
    block_root_proof: Vector[Bytes32, HISTORICAL_TARGET_PROOF_DEPTH]
```

### Modified containers

#### `AttestationData`

*Note*: The `source`, `index`, and `beacon_block_root` fields are removed.
`target` is repurposed as a one-round finality target (not FFG), and `height`
is added. LMD-GHOST head attestations use `AvailableAttestationData` instead.

```python
class AttestationData(Container):
    slot: Slot
    target: Checkpoint  # [Modified in One-Round Finality] Finality target (one-round, not FFG)
    height: Height  # [New in One-Round Finality] Finality height being attested to
```

#### `Attestation`

*Note*: `AttestationData` is modified (see above), but `Attestation` retains
the standard Electra committee-based format.

```python
class Attestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE * MAX_COMMITTEES_PER_SLOT]
    data: AttestationData
    signature: BLSSignature
    committee_bits: Bitvector[MAX_COMMITTEES_PER_SLOT]
```

#### `BeaconBlockBody`

```python
class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS_ELECTRA]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA]  # [Modified in One-Round Finality]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: List[PayloadAttestation, MAX_PAYLOAD_ATTESTATIONS]
    # One-Round Finality
    available_attestations: List[
        AvailableAttestation, MAX_AVAILABLE_ATTESTATIONS
    ]  # [New in One-Round Finality]
    historical_target_proofs: List[
        HistoricalTargetProof, MAX_HISTORICAL_TARGET_PROOFS
    ]  # [New in One-Round Finality]
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
    # Finality [Modified in One-Round Finality]
    justified_checkpoint: Checkpoint  # [Modified in One-Round Finality] replaces justification_bits + previous/current_justified
    finalized_checkpoint: Checkpoint
    # Inactivity
    inactivity_scores: List[uint64, VALIDATOR_REGISTRY_LIMIT]
    # Sync committees
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
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
    builders: List[Builder, BUILDER_REGISTRY_LIMIT]
    next_withdrawal_builder_index: BuilderIndex
    execution_payload_availability: Bitvector[SLOTS_PER_HISTORICAL_ROOT]
    builder_pending_payments: Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH]
    builder_pending_withdrawals: List[BuilderPendingWithdrawal, BUILDER_PENDING_WITHDRAWALS_LIMIT]
    latest_block_hash: Hash32
    payload_expected_withdrawals: List[Withdrawal, MAX_WITHDRAWALS_PER_PAYLOAD]
    # One-Round Finality
    justified_height: Height  # [New in One-Round Finality] latest justification height
    current_height: Height  # [New in One-Round Finality]
    current_height_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    current_height_attestation_targets: List[Checkpoint, VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    current_height_canonical_target: (
        Checkpoint  # [New in One-Round Finality] Canonical target for incentives/leak
    )
    previous_height_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    previous_height_attestation_targets: List[Checkpoint, VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    previous_height_canonical_target: (
        Checkpoint  # [New in One-Round Finality] Canonical target for previous height
    )
    proven_historical_target: (
        Checkpoint  # [New in One-Round Finality] Cached historical target proof for epoch-boundary use
    )
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: The `*_attestation_targets` lists store the actual `Checkpoint` each
validator attested to. The participation bitlists track whether a validator has
attested. Both have actual length equal to `len(state.validators)`.
The default zero value `Checkpoint()` in entries without an attestation is
distinguished from an actual attestation by the participation bit. Implementations may represent
these fields more compactly under the hood — e.g. a target lookup table with
a per-validator index (2–4 bytes per validator instead of 40) — as long as the
logical content and SSZ serialization remain equivalent.

*Note*: The fields `current_height_canonical_target` and
`previous_height_canonical_target` store the full canonical `Checkpoint` for each
tracked height. Only attestations matching the canonical target exempt a validator from
the inactivity leak (see `is_height_participant`). Attestations for other on-chain
targets still count toward justification and skip but do not protect against
leaking.

*Note*: `proven_historical_target` caches a historical target proof validated
during block processing. At epoch boundary, `is_target_on_chain` uses it as a
fallback for out-of-window non-canonical targets. Reset after each epoch's
finality check. The zero value `Checkpoint()` means no proof is cached.

## Helper functions

### Predicates

#### New `is_height_participant`

```python
def is_height_participant(state: BeaconState, index: ValidatorIndex) -> bool:
    """
    Check if validator attested to the canonical target at the current height.
    Only attestations matching the height's canonical target checkpoint count as participation
    for inactivity scoring. Attestations for other targets still contribute to justification
    and skip but do not exempt the validator from the inactivity leak.

    Unlike epoch participation (which checks current and previous epoch), this only
    checks the current height: once a height advances, its finality is settled and
    participation at previous heights is irrelevant for leak purposes.
    """
    return (
        not state.validators[index].slashed
        and state.current_height_participation[index]
        and state.current_height_attestation_targets[index] == state.current_height_canonical_target
    )
```

### Beacon state accessors

#### Modified `is_slashable_attestation_data`

*Note*: One-round finality replaces the FFG double-vote and surround-vote conditions with
a height-based double-vote condition: two different `AttestationData` at the
same finality height.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    [Modified in One-Round Finality] Height-based double vote.
    Slashable if different attestation data at the same height.
    """
    return data_1 != data_2 and data_1.height == data_2.height
```

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
    return (total * JUSTIFICATION_THRESHOLD_NUMERATOR) // JUSTIFICATION_THRESHOLD_DENOMINATOR
```

#### New `get_finalization_threshold`

```python
def get_finalization_threshold(state: BeaconState) -> Gwei:
    """
    Return the finalization threshold (5f+1 where n >= 6f+1, ~83%).
    """
    total = get_total_active_balance(state)
    return (total * FINALIZATION_THRESHOLD_NUMERATOR) // FINALIZATION_THRESHOLD_DENOMINATOR
```

#### New `get_skip_threshold`

```python
def get_skip_threshold(state: BeaconState) -> Gwei:
    """
    Return the skip threshold (~33%). If attestation dispersion exceeds this,
    the height advances without justification.
    """
    total = get_total_active_balance(state)
    return (total * SKIP_THRESHOLD_NUMERATOR) // SKIP_THRESHOLD_DENOMINATOR
```

#### New `get_available_committee`

```python
def get_available_committee(
    state: BeaconState, slot: Slot
) -> Sequence[ValidatorIndex]:
    """
    [New in One-Round Finality] Return the 512-member available committee for the given slot.
    This committee attests for LMD-GHOST fork choice via on-chain attestations.
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

### Available attestation helpers

#### New `get_available_attesting_indices`

```python
def get_available_attesting_indices(
    state: BeaconState, attestation: AvailableAttestation
) -> Set[ValidatorIndex]:
    """
    Return the set of attesting indices from an available committee attestation.
    """
    committee = get_available_committee(state, attestation.data.slot)
    assert len(attestation.aggregation_bits) == len(committee)
    return set(
        attester_index
        for i, attester_index in enumerate(committee)
        if attestation.aggregation_bits[i]
    )
```

### Modified helpers

#### Modified `add_validator_to_registry`

```python
def add_validator_to_registry(
    state: BeaconState, pubkey: BLSPubkey, withdrawal_credentials: Bytes32, amount: uint64
) -> None:
    index = get_index_for_new_validator(state)
    validator = get_validator_from_deposit(pubkey, withdrawal_credentials, amount)
    set_or_append_list(state.validators, index, validator)
    set_or_append_list(state.balances, index, amount)
    set_or_append_list(state.previous_epoch_participation, index, ParticipationFlags(0b0000_0000))
    set_or_append_list(state.current_epoch_participation, index, ParticipationFlags(0b0000_0000))
    set_or_append_list(state.inactivity_scores, index, uint64(0))
    # [New in One-Round Finality]
    set_or_append_list(state.current_height_participation, index, False)
    set_or_append_list(state.current_height_attestation_targets, index, Checkpoint())
    set_or_append_list(state.previous_height_participation, index, False)
    set_or_append_list(state.previous_height_attestation_targets, index, Checkpoint())
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(state: BeaconState) -> None:
    """
    Advance to the next height and rotate attestation tracking.
    """
    # Rotate current to previous
    state.previous_height_participation = state.current_height_participation
    state.previous_height_attestation_targets = state.current_height_attestation_targets
    state.previous_height_canonical_target = state.current_height_canonical_target

    # Advance height
    state.current_height = Height(state.current_height + 1)

    # Set canonical target for the new height
    epoch = get_current_epoch(state)
    state.current_height_canonical_target = Checkpoint(
        epoch=epoch,
        root=get_block_root_at_slot(state, compute_start_slot_at_epoch(epoch)),
    )

    # Reset current height attestation tracking
    state.current_height_participation = [False for _ in range(len(state.validators))]
    state.current_height_attestation_targets = [Checkpoint() for _ in range(len(state.validators))]
```

#### New `count_height_attestations`

```python
def count_height_attestations(
    state: BeaconState,
    participation: Bitlist[VALIDATOR_REGISTRY_LIMIT],
    attestation_targets: List[Checkpoint, VALIDATOR_REGISTRY_LIMIT],
) -> Tuple[Dict[Checkpoint, Gwei], Gwei]:
    """
    Count attestations per distinct target and total attesting weight for a height.
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

        target = attestation_targets[validator_index]
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
) -> bool:
    """
    Check if a target checkpoint is verifiably on the current chain.
    Three paths: (1) canonical target for this height (stored in state),
    (2) in-window ``block_roots`` check, (3) historical proof cached in state.
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

    # Historical proof fallback (cached from block processing)
    return target == state.proven_historical_target
```

#### New `update_height_justification_and_finalization`

```python
def update_height_justification_and_finalization(
    state: BeaconState,
    participation: Bitlist[VALIDATOR_REGISTRY_LIMIT],
    attestation_targets: List[Checkpoint, VALIDATOR_REGISTRY_LIMIT],
    height: Height,
) -> bool:
    """
    Process justification, finalization, and skip for a given height.
    Returns True if this height should advance.

    Justification: > 1/2 of total active balance attests for the same on-chain target.
    Finalization: > 5/6 of total active balance attests for the same on-chain target.
    Skip: total attesting weight - max single-target weight > 1/3 of total active balance.
    The skip rule uses ALL attestations (including off-chain targets), preventing
    skip when a conflicting branch has finalization.
    """
    justification_threshold = get_justification_threshold(state)
    finalization_threshold = get_finalization_threshold(state)

    target_weights, total_weight = count_height_attestations(state, participation, attestation_targets)

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

    if has_justified_target and is_target_on_chain(state, justified_target, height):
        # Always update justified height when justification occurs at this height.
        # The height tracks the chain's progress, not the checkpoint's origin.
        state.justified_height = height  # [Modified in One-Round Finality]

        # LJ monotonicity: only update justified checkpoint if epoch >= current
        if justified_target.epoch >= state.justified_checkpoint.epoch:
            state.justified_checkpoint = justified_target

        # Check for finalization (5/6 for same target)
        if justified_weight > finalization_threshold:
            if justified_target.epoch > state.finalized_checkpoint.epoch:
                state.finalized_checkpoint = justified_target

        return True  # Advance-eligible on justification

    # Skip: allVotes - maxVotes > 1/3 of total active balance
    # This counts ALL attestations (including off-chain targets), so a branch
    # where 5/6 attested to the same (off-chain) target cannot skip
    max_target_weight = max(target_weights.values()) if target_weights else Gwei(0)
    skip_threshold = get_skip_threshold(state)
    if total_weight - max_target_weight > skip_threshold:
        return True  # Advance-eligible on skip

    return False
```

#### New `process_historical_target_proof`

```python
def process_historical_target_proof(
    state: BeaconState, proof: HistoricalTargetProof
) -> None:
    """
    Validate a historical target proof and cache it for epoch-boundary use.
    """
    assert not is_target_in_block_roots_window(state, proof.target)
    assert is_target_in_historical_summaries(state, proof)
    state.proven_historical_target = proof.target
```

#### New `process_height_progress`

```python
def process_height_progress(state: BeaconState) -> None:
    """
    Check justification, finalization, and skip at epoch boundary,
    then advance height if eligible.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    # Process previous height (late-arriving attestations may still justify or finalize)
    if state.current_height > GENESIS_HEIGHT + 1:
        update_height_justification_and_finalization(
            state,
            state.previous_height_participation,
            state.previous_height_attestation_targets,
            get_previous_height(state),
        )

    # Process current height and advance if eligible
    if update_height_justification_and_finalization(
        state,
        state.current_height_participation,
        state.current_height_attestation_targets,
        state.current_height,
    ):
        advance_height(state)

    # Reset proven historical target (consumed or unused)
    state.proven_historical_target = Checkpoint()
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
if it attested to the canonical target at the current height. This gives a tight
property analogous to FFG: **either finalization occurs, or at least 1/6 of
total stake is being leaked**. The only way to avoid the leak is for finality to
happen — there is no middle ground where validators participate but finality
stalls without penalty. The leak trigger uses `finality_delay` (epochs since
last finalization), providing **accountable liveness**: any period without
finalization incurs an economic cost on non-participants regardless of whether
heights are advancing via skip.

```python
def process_inactivity_updates(state: BeaconState) -> None:
    # Skip the genesis epoch as score updates are based on the previous epoch participation
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    for index in get_eligible_validator_indices(state):
        # [Modified in One-Round Finality] Uses is_height_participant instead of epoch-based target flag
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
    [Modified in One-Round Finality] Uses height participation instead of epoch-based target flag.
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
    # [Modified in One-Round Finality] process_justification_and_finalization removed
    # (height justification, finalization, and advancement happen at epoch transition)
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_registry_updates(state)
    process_slashings(state)
    process_eth1_data_reset(state)
    process_pending_deposits(state)
    process_pending_consolidations(state)
    process_builder_pending_payments(state)
    process_effective_balance_updates(state)
    process_slashings_reset(state)
    process_randao_mixes_reset(state)
    process_historical_summaries_update(state)
    process_participation_flag_updates(state)
    process_sync_committee_updates(state)
    process_proposer_lookahead(state)
    process_height_progress(state)
```

### Block processing

#### Modified `is_valid_indexed_attestation`

```python
def is_valid_indexed_attestation(
    state: BeaconState, indexed_attestation: IndexedAttestation
) -> bool:
    """
    Check if ``indexed_attestation`` is not empty, has sorted and unique indices and has a valid aggregate signature.
    [Modified in One-Round Finality] Uses slot epoch for signing domain (target epoch may differ).
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

#### Modified `process_attestation`

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in One-Round Finality] Records finality attestations and sets TIMELY_TARGET flag
    for canonical target matches.
    """
    data = attestation.data

    # Validate slot and height
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    assert data.height in (state.current_height, get_previous_height(state))

    attestation_epoch = compute_epoch_at_slot(data.slot)

    # Validate committee structure (Electra pattern)
    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    for committee_index in committee_indices:
        assert committee_index < get_committee_count_per_slot(state, attestation_epoch)
        committee = get_beacon_committee(state, data.slot, committee_index)
        committee_attesters = set(
            attester_index
            for i, attester_index in enumerate(committee)
            if attestation.aggregation_bits[committee_offset + i]
        )
        assert len(committee_attesters) > 0
        committee_offset += len(committee)
    assert len(attestation.aggregation_bits) == committee_offset

    # Validate signature
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))

    # Determine which height this attestation is for
    if data.height == state.current_height:
        participation = state.current_height_participation
        attestation_targets = state.current_height_attestation_targets
    else:
        participation = state.previous_height_participation
        attestation_targets = state.previous_height_attestation_targets

    # Check if this attestation matches the canonical target (for incentives)
    if data.height == state.current_height:
        is_matching_target = data.target == state.current_height_canonical_target
    else:
        is_matching_target = data.target == state.previous_height_canonical_target

    # Determine epoch participation list for TIMELY_TARGET rewards
    current_epoch = get_current_epoch(state)
    is_within_reward_window = attestation_epoch in (current_epoch, get_previous_epoch(state))
    if attestation_epoch == current_epoch:
        epoch_participation = state.current_epoch_participation
    else:
        epoch_participation = state.previous_epoch_participation

    proposer_reward_numerator = 0

    attesting_indices = get_attesting_indices(state, attestation)
    for validator_index in attesting_indices:
        if participation[validator_index]:
            continue
        validator = state.validators[validator_index]
        if not is_active_validator(validator, current_epoch):
            continue

        # Record the attestation (for finality counting regardless of epoch)
        participation[validator_index] = True
        attestation_targets[validator_index] = data.target

        # Set TIMELY_TARGET flag if matching canonical target and within reward window
        if is_matching_target and is_within_reward_window:
            if not has_flag(
                epoch_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
            ):
                epoch_participation[validator_index] = add_flag(
                    epoch_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
                )
                proposer_reward_numerator += (
                    get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT
                )

    # Proposer reward for included finality attestations
    if proposer_reward_numerator > 0:
        proposer_reward_denominator = (
            (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
        )
        proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
        increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### New `process_available_attestation`

```python
def process_available_attestation(
    state: BeaconState, attestation: AvailableAttestation
) -> None:
    """
    [New in One-Round Finality] Process an available committee attestation for LMD-GHOST.
    Sets TIMELY_HEAD flag and handles builder payment weight.
    """
    data = attestation.data
    attestation_epoch = compute_epoch_at_slot(data.slot)
    assert attestation_epoch in (get_previous_epoch(state), get_current_epoch(state))
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    committee = get_available_committee(state, data.slot)
    assert len(attestation.aggregation_bits) == len(committee)
    assert any(attestation.aggregation_bits)

    # Signature verification
    attesting_indices = get_available_attesting_indices(state, attestation)
    pubkeys = [state.validators[i].pubkey for i in sorted(attesting_indices)]
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, attestation_epoch)
    signing_root = compute_signing_root(data, domain)
    assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Head matching
    is_matching_head = data.beacon_block_root == get_block_root_at_slot(state, data.slot)

    # Epoch participation and builder payment weight
    if attestation_epoch == get_current_epoch(state):
        epoch_participation = state.current_epoch_participation
        payment = state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH]
    else:
        epoch_participation = state.previous_epoch_participation
        payment = state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH]

    proposer_reward_numerator = 0
    for index in attesting_indices:
        if (
            is_matching_head
            and (state.slot - data.slot) == MIN_ATTESTATION_INCLUSION_DELAY
            and not has_flag(epoch_participation[index], TIMELY_HEAD_FLAG_INDEX)
        ):
            epoch_participation[index] = add_flag(epoch_participation[index], TIMELY_HEAD_FLAG_INDEX)
            proposer_reward_numerator += get_base_reward(state, index) * TIMELY_HEAD_WEIGHT
            # Same-slot check: real block was proposed at attestation slot
            if (
                (data.slot == 0 or data.beacon_block_root != get_block_root_at_slot(state, Slot(data.slot - 1)))
                and payment.withdrawal.amount > 0
            ):
                payment.weight += state.validators[index].effective_balance

    proposer_reward_denominator = (
        (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
    )
    proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
    increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### Modified `process_operations`

*Note*: Historical target proofs are validated during block processing and cached
in `state.proven_historical_target` for use at the next epoch boundary. At most
one proof may be included per block. If multiple blocks in the same epoch include
proofs, only the last one is retained.

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
    for_ops(body.attester_slashings, process_attester_slashing)
    for_ops(body.attestations, process_attestation)
    for_ops(body.deposits, process_deposit)
    for_ops(body.voluntary_exits, process_voluntary_exit)
    for_ops(body.bls_to_execution_changes, process_bls_to_execution_change)
    for_ops(body.payload_attestations, process_payload_attestation)
    # [New in One-Round Finality]
    for_ops(body.available_attestations, process_available_attestation)
    # [New in One-Round Finality] Validate and cache historical target proofs for epoch-boundary use
    for_ops(body.historical_target_proofs, process_historical_target_proof)
```

## Fork transition

### New `upgrade_to_one_round_finality`

*Note*: At the fork-epoch boundary, the current epoch start-slot root is not
yet guaranteed to be available in `block_roots`. Initialize canonical targets
from the previous epoch boundary checkpoint (or zero at genesis) to avoid stale
ring-buffer reads.

```python
def upgrade_to_one_round_finality(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)
    if epoch > GENESIS_EPOCH:
        canonical_target_epoch = Epoch(epoch - 1)
        canonical_target_root = gloas.get_block_root(pre, canonical_target_epoch)
    else:
        canonical_target_epoch = GENESIS_EPOCH
        canonical_target_root = Root()

    post = BeaconState(
        # Genesis
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        # State
        slot=pre.slot,
        fork=Fork(
            previous_version=pre.fork.current_version,
            current_version=ONE_ROUND_FINALITY_FORK_VERSION,  # [Modified in One-Round Finality]
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
        # Finality [Modified in One-Round Finality]
        # Removed: justification_bits, previous_justified_checkpoint, current_justified_checkpoint
        justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        # Inactivity
        inactivity_scores=pre.inactivity_scores,
        # Sync committees
        current_sync_committee=pre.current_sync_committee,
        next_sync_committee=pre.next_sync_committee,
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
        builders=pre.builders,
        next_withdrawal_builder_index=pre.next_withdrawal_builder_index,
        execution_payload_availability=pre.execution_payload_availability,
        builder_pending_payments=pre.builder_pending_payments,
        builder_pending_withdrawals=pre.builder_pending_withdrawals,
        latest_block_hash=pre.latest_block_hash,
        payload_expected_withdrawals=pre.payload_expected_withdrawals,
        # One-Round Finality [New in One-Round Finality]
        justified_height=GENESIS_HEIGHT,
        current_height=GENESIS_HEIGHT,
        current_height_participation=[False for _ in range(len(pre.validators))],
        current_height_attestation_targets=[Checkpoint() for _ in range(len(pre.validators))],
        current_height_canonical_target=Checkpoint(
            epoch=canonical_target_epoch,
            root=canonical_target_root,
        ),
        previous_height_participation=[False for _ in range(len(pre.validators))],
        previous_height_attestation_targets=[Checkpoint() for _ in range(len(pre.validators))],
        previous_height_canonical_target=Checkpoint(
            epoch=canonical_target_epoch,
            root=canonical_target_root,
        ),
        proven_historical_target=Checkpoint(),
    )

    return post
```

## Genesis

#### Modified `initialize_beacon_state_from_eth1`

*Note*: The `current_height_canonical_target` and
`previous_height_canonical_target` use a zero root at genesis since no block
exists yet. The `epoch <= GENESIS_EPOCH + 1` guard in
`process_height_progress` prevents finality processing in the first two
epochs, so this zero root is never used for on-chain verification.

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

    # [New in One-Round Finality] Initialize finality fields
    state.current_height = GENESIS_HEIGHT
    state.justified_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.finalized_checkpoint = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.justified_height = GENESIS_HEIGHT
    state.current_height_canonical_target = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.previous_height_canonical_target = Checkpoint(epoch=GENESIS_EPOCH, root=Root())
    state.proven_historical_target = Checkpoint()

    return state
```

# One-Round Finality -- The Beacon Chain

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
  - [Core Concept: Height vs Epoch](#core-concept-height-vs-epoch)
  - [Thresholds (n >= 5f+1)](#thresholds-n--5f1)
  - [Decoupled Consensus](#decoupled-consensus)
  - [Attestation Tracking](#attestation-tracking)
- [Configuration](#configuration)
  - [Round schedule](#round-schedule)
- [Custom types](#custom-types)
- [Constants](#constants)
  - [Finality constants](#finality-constants)
  - [Slashing constants](#slashing-constants)
  - [Participation flag indices](#participation-flag-indices)
  - [Incentivization weights](#incentivization-weights)
  - [Domain types](#domain-types)
  - [Misc](#misc)
- [Preset](#preset)
  - [Round parameters](#round-parameters)
  - [Max operations per block](#max-operations-per-block)
- [Containers](#containers)
  - [New containers](#new-containers)
    - [`AvailableAttestationData`](#availableattestationdata)
    - [`AvailableAttestation`](#availableattestation)
    - [`HistoricalTargetProof`](#historicaltargetproof)
    - [`RoundDoubleVoteEvidence`](#rounddoublevoteevidence)
  - [Modified containers](#modified-containers)
    - [`Checkpoint`](#checkpoint)
    - [`AttestationData`](#attestationdata)
    - [`Attestation`](#attestation)
    - [`BeaconBlockBody`](#beaconblockbody)
    - [`BeaconState`](#beaconstate)
- [Helper functions](#helper-functions)
  - [Round helpers](#round-helpers)
    - [New `compute_round_at_slot`](#new-compute_round_at_slot)
    - [New `compute_start_slot_at_round`](#new-compute_start_slot_at_round)
    - [New `compute_epoch_at_round`](#new-compute_epoch_at_round)
  - [Predicates](#predicates)
    - [New `is_leak_exempt`](#new-is_leak_exempt)
    - [Modified `is_eligible_for_activation`](#modified-is_eligible_for_activation)
    - [Modified `is_active_builder`](#modified-is_active_builder)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_current_round`](#new-get_current_round)
    - [New `get_previous_round`](#new-get_previous_round)
    - [Modified `get_finality_delay`](#modified-get_finality_delay)
    - [Modified `get_unslashed_participating_indices`](#modified-get_unslashed_participating_indices)
    - [Modified `is_slashable_attestation_data`](#modified-is_slashable_attestation_data)
    - [New `get_previous_height`](#new-get_previous_height)
    - [New `get_height_progress_threshold`](#new-get_height_progress_threshold)
    - [New `get_justification_threshold`](#new-get_justification_threshold)
    - [New `get_finalization_threshold`](#new-get_finalization_threshold)
    - [New `get_available_committee`](#new-get_available_committee)
    - [Modified `get_committee_count_per_slot`](#modified-get_committee_count_per_slot)
    - [Modified `get_beacon_committee`](#modified-get_beacon_committee)
  - [Available attestation helpers](#available-attestation-helpers)
    - [New `get_available_attesting_indices`](#new-get_available_attesting_indices)
  - [Modified helpers](#modified-helpers)
    - [Modified `add_validator_to_registry`](#modified-add_validator_to_registry)
- [Beacon chain state transition function](#beacon-chain-state-transition-function)
  - [Epoch processing](#epoch-processing)
    - [New `advance_height`](#new-advance_height)
    - [New `compute_target_weights`](#new-compute_target_weights)
    - [New `is_target_on_chain`](#new-is_target_on_chain)
    - [New `update_height_justification_and_finalization`](#new-update_height_justification_and_finalization)
    - [New `process_historical_target_proof`](#new-process_historical_target_proof)
    - [Modified `process_justification_and_finalization`](#modified-process_justification_and_finalization)
    - [New `is_target_in_block_roots_window`](#new-is_target_in_block_roots_window)
    - [New `is_target_in_historical_summaries`](#new-is_target_in_historical_summaries)
    - [Modified `process_inactivity_updates`](#modified-process_inactivity_updates)
    - [Modified `get_flag_index_deltas`](#modified-get_flag_index_deltas)
    - [Modified `get_inactivity_penalty_deltas`](#modified-get_inactivity_penalty_deltas)
    - [Modified `process_slashings`](#modified-process_slashings)
    - [Modified `process_pending_deposits`](#modified-process_pending_deposits)
    - [Modified `process_participation_flag_updates`](#modified-process_participation_flag_updates)
    - [New `process_round`](#new-process_round)
    - [Modified `process_epoch`](#modified-process_epoch)
    - [Modified `process_slots`](#modified-process_slots)
  - [Block processing](#block-processing)
    - [Modified `is_valid_indexed_attestation`](#modified-is_valid_indexed_attestation)
    - [Modified `process_attestation`](#modified-process_attestation)
    - [New `process_available_attestation`](#new-process_available_attestation)
    - [New `process_round_double_vote_evidence`](#new-process_round_double_vote_evidence)
    - [Modified `process_operations`](#modified-process_operations)
- [Fork transition](#fork-transition)
  - [New `upgrade_to_one_round_finality`](#new-upgrade_to_one_round_finality)
- [Genesis](#genesis)
  - [Modified `initialize_beacon_state_from_eth1`](#modified-initialize_beacon_state_from_eth1)

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for one-round finality based on the
one-round finality protocol. It replaces Casper FFG with a simplified finality
gadget where n >= 5f+1, and separates finality attestations from LMD-GHOST
attestations.

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at epoch boundaries

At each epoch transition, the height advances if EITHER condition holds:

1. **Height progress by target support**: 2f+1 attestations (~40%) for the SAME
   target at height h
2. **Skip**: allVotes - maxVotes > 2n/5 at height h (genuine attestation
   disagreement)

### Thresholds (n >= 5f+1)

| Threshold                         | Stake                           | Purpose                                                |
| --------------------------------- | ------------------------------- | ------------------------------------------------------ |
| Height progress by target support | 2f+1 (~40%)                     | Height advances at epoch transition                    |
| Justification                     | >1/2                            | Update `justified_checkpoint` only on majority support |
| Finalization                      | 4f+1 (~80%)                     | Block finalized                                        |
| Skip                              | allVotes-maxVotes > 2f+1 (~40%) | Marks height to advance without justification          |

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

Skip uses attestation distribution: it fires when `allVotes - maxVotes > 2/5` of
total active balance, ensuring a branch where one target dominates cannot skip
(see Notarization Path Safety).

## Configuration

Warning: this configuration is not definitive.

| Name                              | Value                                 |
| --------------------------------- | ------------------------------------- |
| `ONE_ROUND_FINALITY_FORK_VERSION` | `Version('0x10000000')`               |
| `ONE_ROUND_FINALITY_FORK_EPOCH`   | `Epoch(18446744073709551615)` **TBD** |

### Round schedule

*[New in One-Round Finality]* This schedule defines `SLOTS_PER_ROUND` for each
era, starting from the era's activation slot. For slots before the first entry,
`SLOTS_PER_EPOCH` is used (i.e., one round per epoch).

There MUST NOT exist multiple round schedule entries with the same slot value.
The `SLOTS_PER_ROUND` in each entry MUST divide `SLOTS_PER_EPOCH`. The round
schedule entries SHOULD be sorted by slot in ascending order.

<!-- list-of-records:round_schedule -->

| Slot | Slots Per Round |     Description |
| ---: | --------------: | --------------: |
|    0 |              32 | Pre-fork (Fulu) |

## Custom types

| Name     | SSZ equivalent | Description                      |
| -------- | -------------- | -------------------------------- |
| `Height` | `uint64`       | A finality height                |
| `Round`  | `uint64`       | A global attestation round index |

## Constants

### Finality constants

| Name                                    | Value       |
| --------------------------------------- | ----------- |
| `GENESIS_HEIGHT`                        | `Height(0)` |
| `GENESIS_ROUND`                         | `Round(0)`  |
| `HEIGHT_PROGRESS_THRESHOLD_NUMERATOR`   | `uint64(2)` |
| `HEIGHT_PROGRESS_THRESHOLD_DENOMINATOR` | `uint64(5)` |
| `JUSTIFICATION_THRESHOLD_NUMERATOR`     | `uint64(1)` |
| `JUSTIFICATION_THRESHOLD_DENOMINATOR`   | `uint64(2)` |
| `FINALIZATION_THRESHOLD_NUMERATOR`      | `uint64(4)` |
| `FINALIZATION_THRESHOLD_DENOMINATOR`    | `uint64(5)` |

### Slashing constants

| Name                                                  | Value       |
| ----------------------------------------------------- | ----------- |
| `PROPORTIONAL_SLASHING_MULTIPLIER_ONE_ROUND_FINALITY` | `uint64(5)` |

### Participation flag indices

*Note*: The source flag is removed in one-round finality since there is no
source checkpoint to attest to.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

*Note*: The source weight (14/64) is redistributed to target in one-round
finality since the source flag is removed. The sum of participation weights
remains 54/64 (same as Altair: 14 + 26 + 14 = 54, now 40 + 14 = 54).

| Name                         | Value                                        |
| ---------------------------- | -------------------------------------------- |
| `TIMELY_TARGET_WEIGHT`       | `uint64(40)`                                 |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

### Domain types

| Name                        | Value                      |
| --------------------------- | -------------------------- |
| `DOMAIN_AVAILABLE_ATTESTER` | `DomainType('0x0F000000')` |

### Misc

| Name                            | Value                                          |
| ------------------------------- | ---------------------------------------------- |
| `AVAILABLE_COMMITTEE_SIZE`      | `uint64(2**9)` (= 512)                         |
| `HISTORICAL_TARGET_PROOF_DEPTH` | `uint64(floorlog2(SLOTS_PER_HISTORICAL_ROOT))` |

## Preset

### Round parameters

| Name               | Value                                        |
| ------------------ | -------------------------------------------- |
| `SLOTS_PER_ROUND`  | `uint64(4)` (must divide `SLOTS_PER_EPOCH`)  |
| `ROUNDS_PER_EPOCH` | `uint64(SLOTS_PER_EPOCH // SLOTS_PER_ROUND)` |

### Max operations per block

| Name                             | Value       |
| -------------------------------- | ----------- |
| `MAX_AVAILABLE_ATTESTATIONS`     | `uint64(8)` |
| `MAX_HISTORICAL_TARGET_PROOFS`   | `uint64(1)` |
| `MAX_ROUND_DOUBLE_VOTE_EVIDENCE` | `uint64(1)` |

## Containers

### New containers

#### `AvailableAttestationData`

```python
class AvailableAttestationData(Container):
    slot: Slot
    payload_present: boolean  # Payload availability signal
    beacon_block_root: Root  # LMD attestation for fork choice
```

#### `AvailableAttestation`

```python
class AvailableAttestation(Container):
    aggregation_bits: Bitvector[AVAILABLE_COMMITTEE_SIZE]
    data: AvailableAttestationData
    signature: BLSSignature
```

#### `HistoricalTargetProof`

```python
class HistoricalTargetProof(Container):
    target: Checkpoint
    block_root_proof: Vector[Bytes32, HISTORICAL_TARGET_PROOF_DEPTH]
```

#### `RoundDoubleVoteEvidence`

```python
class RoundDoubleVoteEvidence(Container):
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation
```

### Modified containers

#### `Checkpoint`

```python
class Checkpoint(Container):
    round: Round  # [Modified in One-Round Finality] was epoch: Epoch
    root: Root
```

#### `AttestationData`

*Note*: The `source` and `index` fields are removed. `beacon_block_root` is
repurposed as an LMD head vote for fork choice (set to the voter's head).
`target` is repurposed as a one-round finality target (not FFG), `height` is
added, and `payload_present` signals payload availability for the voted block.
The `beacon_block_root` and `payload_present` fields are used by the fork choice
only — `process_attestation` uses `target` and `height`.

```python
class AttestationData(Container):
    slot: Slot
    beacon_block_root: Root  # [Modified in One-Round Finality] LMD head vote for fork choice
    target: Checkpoint  # [Modified in One-Round Finality] Finality target (one-round, not FFG)
    height: Height  # [New in One-Round Finality] Finality height being attested to
    payload_present: boolean  # [New in One-Round Finality] Payload availability signal
```

#### `Attestation`

*Note*: `AttestationData` is modified (see above), but `Attestation` retains the
standard Electra committee-based format.

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
    round_double_vote_evidence: List[
        RoundDoubleVoteEvidence, MAX_ROUND_DOUBLE_VOTE_EVIDENCE
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
    previous_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    current_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
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
    justified_height: Height  # [New in One-Round Finality] height of ``justified_checkpoint``
    current_height: Height  # [New in One-Round Finality]
    current_height_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    current_height_attestation_targets: List[
        Checkpoint, VALIDATOR_REGISTRY_LIMIT
    ]  # [New in One-Round Finality]
    current_height_canonical_target: (
        Checkpoint  # [New in One-Round Finality] Canonical target for incentives/leak
    )
    previous_height_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in One-Round Finality]
    previous_height_attestation_targets: List[
        Checkpoint, VALIDATOR_REGISTRY_LIMIT
    ]  # [New in One-Round Finality]
    previous_height_canonical_target: (
        Checkpoint  # [New in One-Round Finality] Canonical target for previous height
    )
    proven_historical_target: Checkpoint  # [New in One-Round Finality] Cached historical target proof for epoch-boundary use
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: The `*_attestation_targets` lists store the actual `Checkpoint` each
validator attested to. The participation bitlists track whether a validator has
attested. Both have actual length equal to `len(state.validators)`. The default
zero value `Checkpoint()` in entries without an attestation is distinguished
from an actual attestation by the participation bit. Implementations may
represent these fields more compactly under the hood — e.g. a target lookup
table with a per-validator index (2–4 bytes per validator instead of 40) — as
long as the logical content and SSZ serialization remain equivalent.

*Note*: The fields `current_height_canonical_target` and
`previous_height_canonical_target` store the full canonical `Checkpoint` for
each tracked height. Inactivity leak exemption uses a two-layer design (see
`is_leak_exempt`): when the height advances, only canonical-target attesters at
the completed height are exempt; when stalled, any voter is exempt. Attestations
for non-canonical on-chain targets still count toward justification and skip.

*Note*: `proven_historical_target` caches a historical target proof validated
during block processing. At epoch boundary, `is_target_on_chain` uses it as a
fallback for out-of-window non-canonical targets. Reset after each epoch's
finality check. The zero value `Checkpoint()` means no proof is cached.

## Helper functions

### Round helpers

#### New `compute_round_at_slot`

```python
def compute_round_at_slot(slot: Slot) -> Round:
    """
    Return the round number at ``slot``.
    Walks ``ROUND_SCHEDULE`` to handle forks that change ``SLOTS_PER_ROUND``.
    For slots before the first schedule entry, ``SLOTS_PER_EPOCH`` is used.
    """
    total_rounds = Round(0)
    prev_start = Slot(0)
    prev_spr = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda e: e["SLOT"]):
        era_start = entry["SLOT"]
        if slot < era_start:
            return total_rounds + Round((slot - prev_start) // prev_spr)
        total_rounds += Round((era_start - prev_start) // prev_spr)
        prev_start = era_start
        prev_spr = entry["SLOTS_PER_ROUND"]
    return total_rounds + Round((slot - prev_start) // prev_spr)
```

#### New `compute_start_slot_at_round`

```python
def compute_start_slot_at_round(round: Round) -> Slot:
    """
    Return the start slot of ``round``.
    Inverse of ``compute_round_at_slot``; walks ``ROUND_SCHEDULE``.
    """
    remaining = round
    prev_start = Slot(0)
    prev_spr = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda e: e["SLOT"]):
        era_start = entry["SLOT"]
        era_rounds = Round((era_start - prev_start) // prev_spr)
        if remaining < era_rounds:
            return Slot(prev_start + remaining * prev_spr)
        remaining -= era_rounds
        prev_start = era_start
        prev_spr = entry["SLOTS_PER_ROUND"]
    return Slot(prev_start + remaining * prev_spr)
```

#### New `compute_epoch_at_round`

```python
def compute_epoch_at_round(round: Round) -> Epoch:
    """
    Return the epoch number at the start of ``round``.
    """
    return compute_epoch_at_slot(compute_start_slot_at_round(round))
```

### Predicates

#### New `is_leak_exempt`

```python
def is_leak_exempt(state: BeaconState, index: ValidatorIndex) -> bool:
    """
    Check if a validator is exempt from the inactivity leak for this epoch.
    Two-layer design:

    - **Height advanced this epoch**: The height's finality is resolved.
      Penalize validators that did NOT vote for the canonical target at the
      *previous* height (now rotated into ``previous_height_*``). This drives
      justification and finalization.
    - **Height stalled**: Only penalize validators that did not vote
      at all at the current height. Validators locked into a wrong vote
      (not canonical or not even onchain target) are not penalized.
    """
    if state.validators[index].slashed:
        return False

    height_advanced = state.current_height_canonical_target.round == get_current_round(
        state
    )  # [Modified in One-Round Finality]

    if height_advanced:
        # canonical-target check on the completed height (now previous_height_*)
        return (
            state.previous_height_participation[index]
            and state.previous_height_attestation_targets[index]
            == state.previous_height_canonical_target
        )
    else:
        # participation-only check on the stalled current height
        return state.current_height_participation[index]
```

#### Modified `is_eligible_for_activation`

```python
def is_eligible_for_activation(state: BeaconState, validator: Validator) -> bool:
    """
    [Modified in One-Round Finality] Uses compute_epoch_at_round for finalized checkpoint.
    """
    return (
        # Placement in queue is finalized
        validator.activation_eligibility_epoch
        <= compute_epoch_at_round(state.finalized_checkpoint.round)
        # Has not yet been activated
        and validator.activation_epoch == FAR_FUTURE_EPOCH
    )
```

#### Modified `is_active_builder`

```python
def is_active_builder(state: BeaconState, builder_index: BuilderIndex) -> bool:
    """
    [Modified in One-Round Finality] Uses compute_epoch_at_round for finalized checkpoint.
    """
    builder = state.builders[builder_index]
    return (
        # Placement in builder list is finalized
        builder.deposit_epoch < compute_epoch_at_round(state.finalized_checkpoint.round)
        # Has not initiated exit
        and builder.withdrawable_epoch == FAR_FUTURE_EPOCH
    )
```

### Beacon state accessors

#### New `get_current_round`

```python
def get_current_round(state: BeaconState) -> Round:
    """
    Return the current round.
    """
    return compute_round_at_slot(state.slot)
```

#### New `get_previous_round`

```python
def get_previous_round(state: BeaconState) -> Round:
    """
    Return the previous round (unless the current round is ``GENESIS_ROUND``).
    """
    current_round = get_current_round(state)
    return GENESIS_ROUND if current_round == GENESIS_ROUND else Round(current_round - 1)
```

#### Modified `get_finality_delay`

```python
def get_finality_delay(state: BeaconState) -> uint64:
    # [Modified in One-Round Finality] Uses compute_epoch_at_round for finalized checkpoint
    return get_previous_epoch(state) - compute_epoch_at_round(state.finalized_checkpoint.round)
```

#### Modified `get_unslashed_participating_indices`

```python
def get_unslashed_participating_indices(
    state: BeaconState, flag_index: int, round: Round
) -> Set[ValidatorIndex]:
    """
    Return the set of validator indices that are both active and unslashed for the given
    ``flag_index`` and ``round``.
    [Modified in One-Round Finality] Takes a round instead of an epoch. Selects current or
    previous round participation based on ``round``, and derives the epoch for active-set
    lookup from the round.
    """
    assert round in (get_current_round(state), get_previous_round(state))
    if round == get_current_round(state):
        round_participation = state.current_round_participation
    else:
        round_participation = state.previous_round_participation
    active_validator_indices = get_active_validator_indices(state, compute_epoch_at_round(round))
    participating_indices = [
        i for i in active_validator_indices if has_flag(round_participation[i], flag_index)
    ]
    return set(filter(lambda index: not state.validators[index].slashed, participating_indices))
```

#### Modified `is_slashable_attestation_data`

*Note*: One-round finality replaces the FFG double-vote and surround-vote
conditions with a single slashing condition: height target conflict — different
targets at the same height. Round double-vote (same round, different data) uses
a lighter penalty via `RoundDoubleVoteEvidence`.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    [Modified in One-Round Finality] Height target conflict is the only slashing condition.
    Round double-vote uses a lighter penalty via RoundDoubleVoteEvidence.
    """
    return data_1.height == data_2.height and data_1.target != data_2.target
```

#### New `get_previous_height`

```python
def get_previous_height(state: BeaconState) -> Height:
    if state.current_height > GENESIS_HEIGHT:
        return Height(state.current_height - 1)
    return GENESIS_HEIGHT
```

#### New `get_height_progress_threshold`

```python
def get_height_progress_threshold(state: BeaconState) -> Gwei:
    """
    Return the height-progress threshold (2f+1 where n >= 5f+1, ~40%).
    """
    total = get_total_active_balance(state)
    return (total * HEIGHT_PROGRESS_THRESHOLD_NUMERATOR) // HEIGHT_PROGRESS_THRESHOLD_DENOMINATOR
```

#### New `get_justification_threshold`

```python
def get_justification_threshold(state: BeaconState) -> Gwei:
    """
    Return the justification threshold (> 1/2) for updating
    ``state.justified_checkpoint``.
    """
    total = get_total_active_balance(state)
    return (total * JUSTIFICATION_THRESHOLD_NUMERATOR) // JUSTIFICATION_THRESHOLD_DENOMINATOR
```

#### New `get_finalization_threshold`

```python
def get_finalization_threshold(state: BeaconState) -> Gwei:
    """
    Return the finalization threshold (4f+1 where n >= 5f+1, ~80%).
    """
    total = get_total_active_balance(state)
    return (total * FINALIZATION_THRESHOLD_NUMERATOR) // FINALIZATION_THRESHOLD_DENOMINATOR
```

#### New `get_available_committee`

```python
def get_available_committee(state: BeaconState, slot: Slot) -> Sequence[ValidatorIndex]:
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

*Note*: Both the available committee and PTC use
`compute_balance_weighted_selection` from the full active validator set. They
differ only in the seed (different domain types: `DOMAIN_AVAILABLE_ATTESTER` vs
`DOMAIN_PTC_ATTESTER`).

#### Modified `get_committee_count_per_slot`

```python
def get_committee_count_per_slot(state: BeaconState, epoch: Epoch) -> uint64:
    """
    Return the number of committees in each slot for the given ``epoch``.
    """
    return max(
        uint64(1),
        min(
            MAX_COMMITTEES_PER_SLOT,
            uint64(
                len(get_active_validator_indices(state, epoch))
                // SLOTS_PER_ROUND  # [Modified in One-Round Finality]
                // TARGET_COMMITTEE_SIZE
            ),
        ),
    )
```

#### Modified `get_beacon_committee`

```python
def get_beacon_committee(
    state: BeaconState, slot: Slot, index: CommitteeIndex
) -> Sequence[ValidatorIndex]:
    """
    Return the beacon committee at ``slot`` for ``index``.
    """
    epoch = compute_epoch_at_slot(slot)
    committees_per_slot = get_committee_count_per_slot(state, epoch)
    # [Modified in One-Round Finality] Slot-within-round via round helpers (schedule-safe)
    slot_in_round = slot - compute_start_slot_at_round(compute_round_at_slot(slot))
    return compute_committee(
        indices=get_active_validator_indices(state, epoch),
        seed=get_seed(state, epoch, DOMAIN_BEACON_ATTESTER),
        index=slot_in_round * committees_per_slot + index,
        count=committees_per_slot * SLOTS_PER_ROUND,
    )
```

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
    assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
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
    set_or_append_list(state.previous_round_participation, index, ParticipationFlags(0b0000_0000))
    set_or_append_list(state.current_round_participation, index, ParticipationFlags(0b0000_0000))
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

    # Set canonical target for the new height (round-boundary block)
    state.current_height_canonical_target = Checkpoint(
        round=get_current_round(state),
        root=get_block_root_at_slot(state, compute_start_slot_at_round(get_current_round(state))),
    )

    # Reset current height attestation tracking
    state.current_height_participation = [False for _ in range(len(state.validators))]
    state.current_height_attestation_targets = [Checkpoint() for _ in range(len(state.validators))]
```

#### New `compute_target_weights`

```python
def compute_target_weights(
    state: BeaconState,
    participation: Bitlist[VALIDATOR_REGISTRY_LIMIT],
    attestation_targets: List[Checkpoint, VALIDATOR_REGISTRY_LIMIT],
) -> Dict[Checkpoint, Gwei]:
    """
    Compute the attesting weight per distinct target for a height.
    """
    target_weights: Dict[Checkpoint, Gwei] = {}
    for validator_index in get_active_validator_indices(state, get_current_epoch(state)):
        if not participation[validator_index]:
            continue
        weight = state.validators[validator_index].effective_balance
        target = attestation_targets[validator_index]
        if target not in target_weights:
            target_weights[target] = Gwei(0)
        target_weights[target] += weight
    return target_weights
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
    current_height = state.current_height
    assert height in (current_height, current_height - 1 if current_height > 0 else 0)
    # Canonical target is always on-chain (recorded in state at height start)
    if height == current_height:
        canonical = state.current_height_canonical_target
    else:
        canonical = state.previous_height_canonical_target
    if target == canonical:
        return True

    # In-window check via block_roots
    if is_target_in_block_roots_window(state, target):
        round_start_slot = compute_start_slot_at_round(
            target.round
        )  # [Modified in One-Round Finality]
        return get_block_root_at_slot(state, round_start_slot) == target.root

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
) -> None:
    """
    Process justification, finalization, and skip for a given height.
    Advances the height if eligible and ``height == state.current_height``.

    Height progress by target support: > 2/5 of total active balance attests
    for the same on-chain target.
    Justification: > 1/2 support for the same on-chain target, updating
    ``state.justified_checkpoint``.
    Finalization: > 4/5 of total active balance attests for the same on-chain target.
    Skip: total attesting weight - max single-target weight > 2/5 of total active balance.
    The skip rule uses ALL attestations (including off-chain targets), preventing
    skip when a conflicting branch has finalization.
    """
    height_progress_threshold = get_height_progress_threshold(state)
    justification_threshold = get_justification_threshold(state)
    finalization_threshold = get_finalization_threshold(state)

    target_weights = compute_target_weights(state, participation, attestation_targets)

    # Select the heaviest on-chain target above the height-progress threshold.
    # Ties are broken deterministically by (round, root).
    progress_candidates = {
        target: weight
        for target, weight in target_weights.items()
        if weight > height_progress_threshold and is_target_on_chain(state, target, height)
    }

    should_advance_height = False
    if len(progress_candidates) > 0:
        should_advance_height = True
        target = max(
            progress_candidates,
            key=lambda target: (progress_candidates[target], target.round, target.root),
        )
        weight = progress_candidates[target]
        # Update justified checkpoint only at strict majority support.
        if weight > justification_threshold and target.round > state.justified_checkpoint.round:
            state.justified_checkpoint = target
            state.justified_height = height

        # Check for finalization (4/5 for same target)
        if weight > finalization_threshold:
            if target.round > state.finalized_checkpoint.round:
                state.finalized_checkpoint = target
    else:
        # Skip: allVotes - maxVotes > 2/5 of total active balance
        # This counts ALL attestations (including off-chain targets), so a branch
        # where 4/5 attested to the same (off-chain) target cannot skip
        max_target_weight = max(target_weights.values()) if target_weights else Gwei(0)
        if sum(target_weights.values()) - max_target_weight > height_progress_threshold:
            should_advance_height = True

    if should_advance_height and height == state.current_height:
        advance_height(state)
```

#### New `process_historical_target_proof`

```python
def process_historical_target_proof(state: BeaconState, proof: HistoricalTargetProof) -> None:
    """
    Validate a historical target proof and cache it for epoch-boundary use.
    """
    assert not is_target_in_block_roots_window(state, proof.target)
    assert is_target_in_historical_summaries(state, proof)
    state.proven_historical_target = proof.target
```

#### Modified `process_justification_and_finalization`

```python
def process_justification_and_finalization(state: BeaconState) -> None:
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

    # Process current height (advances height internally if eligible)
    update_height_justification_and_finalization(
        state,
        state.current_height_participation,
        state.current_height_attestation_targets,
        state.current_height,
    )

    # Reset proven historical target (consumed or unused)
    state.proven_historical_target = Checkpoint()
```

#### New `is_target_in_block_roots_window`

```python
def is_target_in_block_roots_window(state: BeaconState, target: Checkpoint) -> bool:
    """
    Return True if ``target`` can be checked directly in ``state.block_roots``.
    """
    round_start_slot = compute_start_slot_at_round(target.round)  # [Modified in One-Round Finality]
    return (
        round_start_slot < state.slot and round_start_slot + SLOTS_PER_HISTORICAL_ROOT >= state.slot
    )
```

#### New `is_target_in_historical_summaries`

```python
def is_target_in_historical_summaries(
    state: BeaconState, historical_target_proof: HistoricalTargetProof
) -> bool:
    """
    Verify a target root against ``historical_summaries`` for out-of-window rounds.
    """
    target = historical_target_proof.target
    round_start_slot = compute_start_slot_at_round(target.round)  # [Modified in One-Round Finality]
    if round_start_slot >= state.slot:
        return False

    historical_summary_index = uint64(round_start_slot // SLOTS_PER_HISTORICAL_ROOT)
    if historical_summary_index >= len(state.historical_summaries):
        return False

    return is_valid_merkle_branch(
        leaf=target.root,
        branch=historical_target_proof.block_root_proof,
        depth=HISTORICAL_TARGET_PROOF_DEPTH,
        index=uint64(round_start_slot % SLOTS_PER_HISTORICAL_ROOT),
        root=state.historical_summaries[historical_summary_index].block_summary_root,
    )
```

#### Modified `process_inactivity_updates`

*Note*: Inactivity scoring uses a **two-layer design** conditioned on whether
the height advanced this epoch (see `is_leak_exempt`):

- **Height advanced**: We just advanced to a new height. Validators that did not
  attest to the canonical target are penalized. This gives a tight property:
  **either finalization occurs, or at least 1/5 of total stake is being
  leaked**.
- **Height stalled**: Only non-voters at the current height are penalized.
  Validators locked into a wrong vote are not penalized. Three possibilities:
  1. A justification happens *on some branch* (requiring 1/2)
  2. A skip happens (requiring 2/5)
  3. At least 1/10 of the stake leaks (1 - 1/2 - 2/5) In other words, either we
     have height progress or at least 1/10 of the stake leaks.

The leak trigger uses `finality_delay` (epochs since last finalization),
providing **accountable liveness**: any period without finalization incurs an
economic cost on non-participants regardless of whether heights are advancing
via skip.

```python
def process_inactivity_updates(state: BeaconState) -> None:
    # Skip the genesis epoch as score updates are based on the previous epoch participation
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    for index in get_eligible_validator_indices(state):
        # [Modified in One-Round Finality] Two-layer leak: canonical-target when height advanced, participation when stalled
        if is_leak_exempt(state, ValidatorIndex(index)):
            state.inactivity_scores[index] -= min(1, state.inactivity_scores[index])
        else:
            state.inactivity_scores[index] += INACTIVITY_SCORE_BIAS
        # Decrease the inactivity score of all eligible validators during a leak-free epoch
        if not is_in_inactivity_leak(state):
            state.inactivity_scores[index] -= min(
                INACTIVITY_SCORE_RECOVERY_RATE, state.inactivity_scores[index]
            )
```

#### Modified `get_flag_index_deltas`

```python
def get_flag_index_deltas(
    state: BeaconState, flag_index: int
) -> Tuple[Sequence[Gwei], Sequence[Gwei]]:
    """
    [Modified in One-Round Finality] Rewards and penalties are scaled by 1/ROUNDS_PER_EPOCH
    to keep per-epoch totals constant when running per-round.
    """
    rewards = [Gwei(0)] * len(state.validators)
    penalties = [Gwei(0)] * len(state.validators)
    # [Modified in One-Round Finality] Pass previous round instead of previous epoch
    unslashed_participating_indices = get_unslashed_participating_indices(
        state, flag_index, get_previous_round(state)
    )
    weight = PARTICIPATION_FLAG_WEIGHTS[flag_index]
    unslashed_participating_balance = get_total_balance(state, unslashed_participating_indices)
    unslashed_participating_increments = (
        unslashed_participating_balance // EFFECTIVE_BALANCE_INCREMENT
    )
    active_increments = get_total_active_balance(state) // EFFECTIVE_BALANCE_INCREMENT
    for index in get_eligible_validator_indices(state):
        base_reward = get_base_reward(state, index)
        if index in unslashed_participating_indices:
            if not is_in_inactivity_leak(state):
                reward_numerator = base_reward * weight * unslashed_participating_increments
                # [Modified in One-Round Finality] Scale by 1/ROUNDS_PER_EPOCH
                rewards[index] += Gwei(
                    reward_numerator // (active_increments * WEIGHT_DENOMINATOR * ROUNDS_PER_EPOCH)
                )
        elif flag_index != TIMELY_HEAD_FLAG_INDEX:
            # [Modified in One-Round Finality] Scale by 1/ROUNDS_PER_EPOCH
            penalties[index] += Gwei(
                base_reward * weight // (WEIGHT_DENOMINATOR * ROUNDS_PER_EPOCH)
            )
    return rewards, penalties
```

#### Modified `get_inactivity_penalty_deltas`

```python
def get_inactivity_penalty_deltas(state: BeaconState) -> Tuple[Sequence[Gwei], Sequence[Gwei]]:
    """
    Return the inactivity penalty deltas by considering height participation and inactivity scores.
    [Modified in One-Round Finality] Two-layer leak: canonical-target when height advanced,
    participation when stalled. Scaled by 1/ROUNDS_PER_EPOCH**2 per round (one factor for per-round
    application, one for scores accumulating ROUNDS_PER_EPOCH times faster).
    """
    rewards = [Gwei(0) for _ in range(len(state.validators))]
    penalties = [Gwei(0) for _ in range(len(state.validators))]
    for index in get_eligible_validator_indices(state):
        if not is_leak_exempt(state, ValidatorIndex(index)):
            penalty_numerator = (
                state.validators[index].effective_balance * state.inactivity_scores[index]
            )
            # [Modified in One-Round Finality] ROUNDS_PER_EPOCH ** 2: one factor for
            # per-round penalty application, one for scores accumulating RPE times faster
            penalty_denominator = (
                INACTIVITY_SCORE_BIAS
                * INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
                * (ROUNDS_PER_EPOCH**2)
            )
            penalties[index] += Gwei(penalty_numerator // penalty_denominator)
    return rewards, penalties
```

#### Modified `process_slashings`

```python
def process_slashings(state: BeaconState) -> None:
    epoch = get_current_epoch(state)
    total_balance = get_total_active_balance(state)
    # [Modified in One-Round Finality] Increased from 3 to compensate for 1/5 accountable safety (vs FFG's 1/3)
    adjusted_total_slashing_balance = min(
        sum(state.slashings) * PROPORTIONAL_SLASHING_MULTIPLIER_ONE_ROUND_FINALITY, total_balance
    )
    increment = (
        EFFECTIVE_BALANCE_INCREMENT  # Factored out from total balance to avoid uint64 overflow
    )
    penalty_per_effective_balance_increment = adjusted_total_slashing_balance // (
        total_balance // increment
    )
    for index, validator in enumerate(state.validators):
        if (
            validator.slashed
            and epoch + EPOCHS_PER_SLASHINGS_VECTOR // 2 == validator.withdrawable_epoch
        ):
            effective_balance_increments = validator.effective_balance // increment
            # [Modified in Electra:EIP7251]
            penalty = penalty_per_effective_balance_increment * effective_balance_increments
            decrease_balance(state, ValidatorIndex(index), penalty)
```

#### Modified `process_pending_deposits`

```python
def process_pending_deposits(state: BeaconState) -> None:
    next_epoch = Epoch(get_current_epoch(state) + 1)
    available_for_processing = state.deposit_balance_to_consume + get_activation_exit_churn_limit(
        state
    )
    processed_amount = 0
    next_deposit_index = 0
    deposits_to_postpone = []
    is_churn_limit_reached = False
    # [Modified in One-Round Finality] Uses round-based finalized checkpoint
    finalized_slot = compute_start_slot_at_round(state.finalized_checkpoint.round)

    for deposit in state.pending_deposits:
        # Do not process deposit requests if Eth1 bridge deposits are not yet applied.
        if (
            # Is deposit request
            deposit.slot > GENESIS_SLOT
            and
            # There are pending Eth1 bridge deposits
            state.eth1_deposit_index < state.deposit_requests_start_index
        ):
            break

        # Check if deposit has been finalized, otherwise, stop processing.
        if deposit.slot > finalized_slot:
            break

        # Check if number of processed deposits has not reached the limit, otherwise, stop processing.
        if next_deposit_index >= MAX_PENDING_DEPOSITS_PER_EPOCH:
            break

        # Read validator state
        is_validator_exited = False
        is_validator_withdrawn = False
        validator_pubkeys = [v.pubkey for v in state.validators]
        if deposit.pubkey in validator_pubkeys:
            validator = state.validators[ValidatorIndex(validator_pubkeys.index(deposit.pubkey))]
            is_validator_exited = validator.exit_epoch < FAR_FUTURE_EPOCH
            is_validator_withdrawn = validator.withdrawable_epoch < next_epoch

        if is_validator_withdrawn:
            # Deposited balance will never become active. Increase balance but do not consume churn
            apply_pending_deposit(state, deposit)
        elif is_validator_exited:
            # Validator is exiting, postpone the deposit until after withdrawable epoch
            deposits_to_postpone.append(deposit)
        else:
            # Check if deposit fits in the churn, otherwise, do no more deposit processing in this epoch.
            is_churn_limit_reached = processed_amount + deposit.amount > available_for_processing
            if is_churn_limit_reached:
                break

            # Consume churn and apply deposit.
            processed_amount += deposit.amount
            apply_pending_deposit(state, deposit)

        # Regardless of how the deposit was handled, we move on in the queue.
        next_deposit_index += 1

    state.pending_deposits = state.pending_deposits[next_deposit_index:] + deposits_to_postpone

    # Accumulate churn only if the churn limit has been hit.
    if is_churn_limit_reached:
        state.deposit_balance_to_consume = available_for_processing - processed_amount
    else:
        state.deposit_balance_to_consume = Gwei(0)
```

#### Modified `process_participation_flag_updates`

```python
def process_participation_flag_updates(state: BeaconState) -> None:
    # [Modified in One-Round Finality] Uses round-based participation arrays
    state.previous_round_participation = state.current_round_participation
    state.current_round_participation = [
        ParticipationFlags(0b0000_0000) for _ in range(len(state.validators))
    ]
```

#### New `process_round`

```python
def process_round(state: BeaconState) -> None:
    """
    [New in One-Round Finality] Per-round processing: finality voting cycle functions
    that run every SLOTS_PER_ROUND slots. Epoch boundaries are always round boundaries,
    so process_round runs before process_epoch at epoch transitions.
    """
    process_justification_and_finalization(state)
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_participation_flag_updates(state)
```

#### Modified `process_epoch`

```python
def process_epoch(state: BeaconState) -> None:
    # [Modified in One-Round Finality] Finality-cycle functions moved to process_round.
    # process_epoch retains administrative functions only.
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
    process_sync_committee_updates(state)
    process_proposer_lookahead(state)
```

#### Modified `process_slots`

```python
def process_slots(state: BeaconState, slot: Slot) -> None:
    """
    [Modified in One-Round Finality] Adds round processing at round boundaries.
    Round processing runs before epoch processing. Since epoch boundaries are
    always round boundaries, the order at epoch transition is:
    process_round (last round of epoch) → process_epoch (administrative).
    """
    assert state.slot < slot
    while state.slot < slot:
        process_slot(state)
        # [New in One-Round Finality] Round processing at round boundaries
        if (state.slot + 1) % SLOTS_PER_ROUND == 0:
            process_round(state)
        if (state.slot + 1) % SLOTS_PER_EPOCH == 0:
            process_epoch(state)
        state.slot = Slot(state.slot + 1)
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
    for canonical target matches. Attestations are accepted only from the
    current/previous slot epoch.
    """
    data = attestation.data

    # Validate slot and height
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    assert data.height in (state.current_height, get_previous_height(state))

    # [Modified in One-Round Finality] Round-based acceptance window
    attestation_round = compute_round_at_slot(data.slot)
    assert attestation_round in (get_previous_round(state), get_current_round(state))

    # Validate committee structure (Electra pattern)
    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    for committee_index in committee_indices:
        assert committee_index < get_committee_count_per_slot(
            state, compute_epoch_at_slot(data.slot)
        )
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

    # Determine round participation list for TIMELY_TARGET rewards
    if attestation_round == get_current_round(state):
        round_participation = state.current_round_participation
    else:
        round_participation = state.previous_round_participation

    proposer_reward_numerator = 0
    current_epoch = get_current_epoch(state)

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

        # Set TIMELY_TARGET flag if matching canonical target
        if is_matching_target:
            if not has_flag(round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX):
                round_participation[validator_index] = add_flag(
                    round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
                )
                proposer_reward_numerator += (
                    get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT
                )

    # *Note*: Proposer rewards are only earned for canonical-target attestations. Attestations for
    # non-canonical targets contribute to justification/skip but earn no proposer reward.
    if proposer_reward_numerator > 0:
        proposer_reward_denominator = (
            (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
        )
        proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
        increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### New `process_available_attestation`

```python
def process_available_attestation(state: BeaconState, attestation: AvailableAttestation) -> None:
    """
    [New in One-Round Finality] Process an available committee attestation for LMD-GHOST.
    Sets TIMELY_HEAD flag and handles builder payment weight.
    """
    data = attestation.data
    # [Modified in One-Round Finality] Round-based acceptance window
    attestation_round = compute_round_at_slot(data.slot)
    assert attestation_round in (get_previous_round(state), get_current_round(state))
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    committee = get_available_committee(state, data.slot)
    assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
    assert len(attestation.aggregation_bits) == len(committee)
    assert any(attestation.aggregation_bits)

    # Signature verification
    attesting_indices = get_available_attesting_indices(state, attestation)
    pubkeys = [state.validators[i].pubkey for i in sorted(attesting_indices)]
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, compute_epoch_at_slot(data.slot))
    signing_root = compute_signing_root(data, domain)
    assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Head matching
    is_matching_head = data.beacon_block_root == get_block_root_at_slot(state, data.slot)

    # Round participation and builder payment weight
    if attestation_round == get_current_round(state):
        round_participation = state.current_round_participation
        payment = state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH]
    else:
        round_participation = state.previous_round_participation
        payment = state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH]

    proposer_reward_numerator = 0
    for index in attesting_indices:
        if (
            is_matching_head
            and (state.slot - data.slot) == MIN_ATTESTATION_INCLUSION_DELAY
            and not has_flag(round_participation[index], TIMELY_HEAD_FLAG_INDEX)
        ):
            round_participation[index] = add_flag(
                round_participation[index], TIMELY_HEAD_FLAG_INDEX
            )
            proposer_reward_numerator += get_base_reward(state, index) * TIMELY_HEAD_WEIGHT
            # Same-slot check: real block was proposed at attestation slot
            if (
                data.slot == 0
                or data.beacon_block_root != get_block_root_at_slot(state, Slot(data.slot - 1))
            ) and payment.withdrawal.amount > 0:
                payment.weight += state.validators[index].effective_balance

    proposer_reward_denominator = (
        (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
    )
    proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
    increase_balance(state, get_beacon_proposer_index(state), proposer_reward)

    # [Modified in One-Round Finality] Write back updated builder payment weight
    if attestation_round == get_current_round(state):
        state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH] = payment
    else:
        state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH] = payment
```

#### New `process_round_double_vote_evidence`

```python
def process_round_double_vote_evidence(
    state: BeaconState, evidence: RoundDoubleVoteEvidence
) -> None:
    """
    [New in One-Round Finality] Process round double-vote evidence.
    Lighter penalty than slashing: forced exit + fixed penalty, NOT marked slashed.
    """
    attestation_1 = evidence.attestation_1
    attestation_2 = evidence.attestation_2
    # Verify same round, different data
    assert compute_round_at_slot(attestation_1.data.slot) == compute_round_at_slot(
        attestation_2.data.slot
    )
    assert attestation_1.data != attestation_2.data
    # Verify signatures
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)

    for index in sorted(
        set(attestation_1.attesting_indices) & set(attestation_2.attesting_indices)
    ):
        validator = state.validators[index]
        if validator.exit_epoch == FAR_FUTURE_EPOCH:
            # Initiate exit (NOT slashed)
            initiate_validator_exit(state, ValidatorIndex(index))
            # Fixed penalty: one epoch's worth of base reward
            penalty = get_base_reward(state, ValidatorIndex(index))
            decrease_balance(state, ValidatorIndex(index), penalty)
            # Proposer reward
            proposer_reward = Gwei(penalty // PROPOSER_REWARD_QUOTIENT)
            increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### Modified `process_operations`

*Note*: Historical target proofs are validated during block processing and
cached in `state.proven_historical_target` for use at the next epoch boundary.
At most one proof may be included per block. If multiple blocks in the same
epoch include proofs, only the last one is retained.

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
    # [New in One-Round Finality] Round double-vote evidence (lighter penalty than attester slashing)
    for_ops(body.round_double_vote_evidence, process_round_double_vote_evidence)
```

## Fork transition

### New `upgrade_to_one_round_finality`

*Note*: At the fork-epoch boundary, the current epoch start-slot root is not yet
guaranteed to be available in `block_roots`. Initialize canonical targets from
the previous epoch boundary checkpoint (or zero at genesis) to avoid stale
ring-buffer reads.

```python
def upgrade_to_one_round_finality(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)
    if epoch > GENESIS_EPOCH:
        canonical_target_round = compute_round_at_slot(  # [Modified in One-Round Finality]
            compute_start_slot_at_epoch(Epoch(epoch - 1))
        )
        canonical_target_root = gloas.get_block_root(pre, Epoch(epoch - 1))
    else:
        canonical_target_round = GENESIS_ROUND
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
        previous_round_participation=pre.previous_epoch_participation,
        current_round_participation=pre.current_epoch_participation,
        # Finality [Modified in One-Round Finality]
        # Removed: justification_bits, previous_justified_checkpoint, current_justified_checkpoint
        # Convert epoch-based Checkpoints to round-based
        justified_checkpoint=Checkpoint(
            round=compute_round_at_slot(
                compute_start_slot_at_epoch(pre.current_justified_checkpoint.epoch)
            ),
            root=pre.current_justified_checkpoint.root,
        ),
        finalized_checkpoint=Checkpoint(
            round=compute_round_at_slot(
                compute_start_slot_at_epoch(pre.finalized_checkpoint.epoch)
            ),
            root=pre.finalized_checkpoint.root,
        ),
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
            round=canonical_target_round,
            root=canonical_target_root,
        ),
        previous_height_participation=[False for _ in range(len(pre.validators))],
        previous_height_attestation_targets=[Checkpoint() for _ in range(len(pre.validators))],
        previous_height_canonical_target=Checkpoint(
            round=canonical_target_round,
            root=canonical_target_root,
        ),
        proven_historical_target=Checkpoint(),
    )

    return post
```

## Genesis

### Modified `initialize_beacon_state_from_eth1`

*Note*: The `current_height_canonical_target` and
`previous_height_canonical_target` use a zero root at genesis since no block
exists yet. The `epoch <= GENESIS_EPOCH + 1` guard in
`process_justification_and_finalization` prevents finality processing in the
first two epochs, so this zero root is never used for on-chain verification.

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
    state.justified_checkpoint = Checkpoint(round=GENESIS_ROUND, root=Root())
    state.finalized_checkpoint = Checkpoint(round=GENESIS_ROUND, root=Root())
    state.justified_height = GENESIS_HEIGHT
    state.current_height_canonical_target = Checkpoint(round=GENESIS_ROUND, root=Root())
    state.previous_height_canonical_target = Checkpoint(round=GENESIS_ROUND, root=Root())
    state.proven_historical_target = Checkpoint()

    return state
```

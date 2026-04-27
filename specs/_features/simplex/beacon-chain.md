# Simplex Finality -- The Beacon Chain

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
  - [Core Concept: Height vs Epoch](#core-concept-height-vs-epoch)
  - [Thresholds (n >= 3f+1)](#thresholds-n--3f1)
  - [Decoupled Consensus](#decoupled-consensus)
  - [Attestation Tracking](#attestation-tracking)
- [Configuration](#configuration)
  - [Round schedule](#round-schedule)
- [Custom types](#custom-types)
- [Constants](#constants)
  - [Finality constants](#finality-constants)
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
    - [`RoundDoubleVoteEvidence`](#rounddoublevoteevidence)
    - [`HistoricalBlockProof`](#historicalblockproof)
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
    - [New `compute_leak_penalty_units`](#new-compute_leak_penalty_units)
    - [Modified `is_slashable_attestation_data`](#modified-is_slashable_attestation_data)
    - [Modified `is_eligible_for_activation`](#modified-is_eligible_for_activation)
    - [New `is_active_builder`](#new-is_active_builder)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_current_round`](#new-get_current_round)
    - [New `get_previous_round`](#new-get_previous_round)
    - [New `get_target_slot_weights`](#new-get_target_slot_weights)
    - [Modified `get_finality_delay`](#modified-get_finality_delay)
    - [Modified `get_unslashed_participating_indices`](#modified-get_unslashed_participating_indices)
    - [New `is_target_on_chain`](#new-is_target_on_chain)
    - [New `verify_historical_block_proof`](#new-verify_historical_block_proof)
    - [New `is_viable_attestation_target`](#new-is_viable_attestation_target)
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
    - [New `compute_justified_checkpoint`](#new-compute_justified_checkpoint)
    - [New `compute_notarized_checkpoint`](#new-compute_notarized_checkpoint)
    - [New `has_new_finalization`](#new-has_new_finalization)
    - [New `compute_best_justification_target`](#new-compute_best_justification_target)
    - [Modified `process_justification_and_finalization`](#modified-process_justification_and_finalization)
    - [Modified `process_inactivity_updates`](#modified-process_inactivity_updates)
    - [Modified `get_flag_index_deltas`](#modified-get_flag_index_deltas)
    - [Modified `get_inactivity_penalty_deltas`](#modified-get_inactivity_penalty_deltas)
    - [Modified `process_pending_deposits`](#modified-process_pending_deposits)
    - [Modified `process_participation_flag_updates`](#modified-process_participation_flag_updates)
    - [New `process_round`](#new-process_round)
    - [Modified `process_epoch`](#modified-process_epoch)
    - [Modified `process_slots`](#modified-process_slots)
  - [Block processing](#block-processing)
    - [Modified `is_valid_indexed_attestation`](#modified-is_valid_indexed_attestation)
    - [New `validate_attestation`](#new-validate_attestation)
    - [New `update_justification_and_notarization_targets`](#new-update_justification_and_notarization_targets)
    - [New `update_finality_participation`](#new-update_finality_participation)
    - [New `record_timely_target`](#new-record_timely_target)
    - [Modified `process_attestation`](#modified-process_attestation)
    - [New `process_available_attestation`](#new-process_available_attestation)
    - [New `process_round_double_vote_evidence`](#new-process_round_double_vote_evidence)
    - [Modified `process_operations`](#modified-process_operations)
- [Fork transition](#fork-transition)
  - [New `upgrade_to_simplex`](#new-upgrade_to_simplex)
- [Genesis](#genesis)
  - [Modified `initialize_beacon_state_from_eth1`](#modified-initialize_beacon_state_from_eth1)

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for simplex-based finality. It replaces
Casper FFG with a fresh-simplex-with-notarizations finality gadget. The model is
n >= 3f+1, with 2/3 quorums for justification, prefix-notarization, and
finalization. Each validator casts at most one **justify** (R1) and one
**notarize** (R2) attestation per state-height; both are subject to a
**fresh-vote** gate that keys a vote to the current height's interval on the
current chain. Finalization takes two steps: justify at height H, then confirm
via piggybacked finality votes at any subsequent height (extended finalization
window). The fork-choice root is maintained as the running maximum of
justification and notarization cert events (no viability filter).

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at round boundaries

At each round boundary, the height may advance via one of two mechanisms:
**justify** (some target T reaches a 2/3 quorum on `justification_targets`) or
**prefix-notarize** (the prefix walk from the head reaches a 2/3 cumulative
quorum on `notarization_targets` at some on-chain ancestor). Finality is
separate: `F ← J` fires whenever the finality participation bitlist reaches 2/3;
this does NOT advance height.

### Thresholds (n >= 3f+1)

| Threshold           | Stake  | Purpose                                                           |
| ------------------- | ------ | ----------------------------------------------------------------- |
| Justification       | >= 2/3 | Per-target quorum on `justification_targets` (R1 votes)           |
| Prefix-notarization | >= 2/3 | Cumulative prefix walk on `notarization_targets` (R1 or R2 votes) |
| Finalization        | >= 2/3 | Piggybacked confirm of justified checkpoint                       |
| Accountable safety  | 1/3    | Standard BFT (single slashing condition E1)                       |

### Decoupled Consensus

Finality and LMD-GHOST use different attestation types:

- **Attestations**: All active validators attest once per round via standard
  beacon committee attestations (Electra format). `AttestationData` carries a
  finality target, height, kind (justify/notarize), finality target, and
  finality height. These determine justification, notarization, and
  finalization. Attester slashings enforce the finality-target conflict
  condition (E1 only).
- **Available attestations**: A small 512-member available committee attests per
  slot for fork choice via `AvailableAttestation`. This committee is selected
  from the full active set using `compute_balance_weighted_selection` (same
  mechanism as PTC).

### Attestation Tracking

Finality attestations are tracked per validator using two **target slot lists**:

- `justification_targets[i]`: the slot of validator `i`'s last (fresh)
  **justify** vote this height, or `FAR_FUTURE_SLOT` if none.
- `notarization_targets[i]`: the slot of validator `i`'s highest (fresh) target
  across both justify and notarize votes this height, or `FAR_FUTURE_SLOT` if
  none.

Both are reset on height advance. Since only on-chain targets (verified by
`is_target_on_chain`) can update these arrays, the slot uniquely identifies the
target block — the root is recoverable via `get_block_root_at_slot` when needed.
The justification branch uses per-target counting on `justification_targets`
(highest slot where a 2/3 quorum exists); the prefix-notarization branch walks
on-chain slots from `latest_block_header.slot` down to
`current_height_start_slot`, summing per-slot `notarization_targets` weights
until the running total reaches 2/3.

A separate **finality participation** bitlist tracks finalization confirmations
across the extended window. It persists until the justified checkpoint changes,
at which point it resets.

No previous-height data is retained: stale votes (height below
`state.current_height`) are rejected at inclusion time by the freshness gate.

## Configuration

Warning: this configuration is not definitive.

| Name                   | Value                                 |
| ---------------------- | ------------------------------------- |
| `SIMPLEX_FORK_VERSION` | `Version('0x10000000')`               |
| `SIMPLEX_FORK_EPOCH`   | `Epoch(18446744073709551615)` **TBD** |

### Round schedule

*[New in Simplex]* This schedule defines `SLOTS_PER_ROUND` for each era,
starting from the era's activation slot. For slots before the first entry,
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

| Name                          | Value               |
| ----------------------------- | ------------------- |
| `GENESIS_HEIGHT`              | `Height(0)`         |
| `FAR_FUTURE_HEIGHT`           | `Height(2**64 - 1)` |
| `GENESIS_ROUND`               | `Round(0)`          |
| `FINALITY_QUORUM_NUMERATOR`   | `uint64(2)`         |
| `FINALITY_QUORUM_DENOMINATOR` | `uint64(3)`         |
| `ATTESTATION_KIND_NOTARIZE`   | `uint8(0)`          |
| `ATTESTATION_KIND_JUSTIFY`    | `uint8(1)`          |

### Participation flag indices

*Note*: The source flag is removed in simplex finality since there is no source
checkpoint to attest to.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

*Note*: The source weight (14/64) is redistributed to target in simplex finality
since the source flag is removed. The sum of participation weights remains 54/64
(same as Altair: 14 + 26 + 14 = 54, now 40 + 14 = 54).

| Name                         | Value                                        |
| ---------------------------- | -------------------------------------------- |
| `TIMELY_TARGET_WEIGHT`       | `uint64(40)`                                 |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

### Domain types

| Name                        | Value                      |
| --------------------------- | -------------------------- |
| `DOMAIN_AVAILABLE_ATTESTER` | `DomainType('0x0F000000')` |

### Misc

| Name                       | Value                       |
| -------------------------- | --------------------------- |
| `AVAILABLE_COMMITTEE_SIZE` | `uint64(2**9)` (= 512)      |
| `BLOCK_ROOTS_PROOF_DEPTH`  | `uint64(13)` (= log2(8192)) |
| `FAR_FUTURE_SLOT`          | `Slot(2**64 - 1)`           |

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

#### `RoundDoubleVoteEvidence`

```python
class RoundDoubleVoteEvidence(Container):
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation
```

#### `HistoricalBlockProof`

*Note*: Self-verifiable proof that a block was genuinely proposed at a given
slot on this chain, for targets outside the `block_roots` window. Both `slot`
and `block_root` are redundant with the attestation's `target` but included for
self-verifiability.

```python
class HistoricalBlockProof(Container):
    slot: Slot
    block_root: Root
    block_proof: Vector[Bytes32, BLOCK_ROOTS_PROOF_DEPTH]
    prev_slot_root: Root  # Root at slot - 1; must differ from block_root
    prev_slot_proof: Vector[Bytes32, BLOCK_ROOTS_PROOF_DEPTH]
```

### Modified containers

#### `Checkpoint`

```python
class Checkpoint(Container):
    slot: Slot  # [Modified in Simplex] was epoch: Epoch
    root: Root
```

#### `AttestationData`

*Note*: The `source` and `index` fields are removed. `beacon_block_root` is
repurposed as an LMD head vote for fork choice (set to the voter's head).
`target` is repurposed as a simplex finality target (every attestation must have
a non-empty target). `height` carries the state-height at which the vote is
cast. `kind` discriminates the two attestation kinds: `ATTESTATION_KIND_JUSTIFY`
(R1) and `ATTESTATION_KIND_NOTARIZE` (R2). `finality_target` is a piggyback vote
specifying which justified checkpoint to confirm (`Checkpoint()` means no
finality vote); `finality_height` is the height at which `finality_target` was
justified (`FAR_FUTURE_HEIGHT` when no finality vote). `payload_present` signals
payload availability for the voted block. The `beacon_block_root` and
`payload_present` fields are used by the fork choice only —
`process_attestation` uses `target`, `height`, `kind`, `finality_target`, and
`finality_height`.

```python
class AttestationData(Container):
    slot: Slot
    beacon_block_root: Root  # [Modified in Simplex] LMD head vote for fork choice
    target: Checkpoint  # [Modified in Simplex] Finality target (must be non-empty)
    height: Height  # [New in Simplex] Finality height being attested to
    kind: uint8  # [New in Simplex] ATTESTATION_KIND_JUSTIFY or ATTESTATION_KIND_NOTARIZE
    # [New in Simplex] Finalize commitment target, or Checkpoint() for none
    finality_target: Checkpoint
    # [New in Simplex] Height at which finality_target was justified, or FAR_FUTURE_HEIGHT
    finality_height: Height
    payload_present: boolean  # [New in Simplex] Payload availability signal
```

#### `Attestation`

*Note*: `AttestationData` is modified (see above). `Attestation` extends the
Electra committee-based format with an optional `HistoricalBlockProof` for
non-canonical target votes when the target block is outside the `block_roots`
window. The proof is unsigned (not part of `AttestationData`) — the proposer
attaches it when including the attestation in a block.

```python
class Attestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE * MAX_COMMITTEES_PER_SLOT]
    data: AttestationData
    signature: BLSSignature
    committee_bits: Bitvector[MAX_COMMITTEES_PER_SLOT]
    historical_block_proof: Optional[HistoricalBlockProof]  # [New in Simplex]
```

#### `BeaconBlockBody`

```python
class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS_ELECTRA]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA]  # [Modified in Simplex]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: List[PayloadAttestation, MAX_PAYLOAD_ATTESTATIONS]
    # Simplex
    available_attestations: List[
        AvailableAttestation, MAX_AVAILABLE_ATTESTATIONS
    ]  # [New in Simplex]
    round_double_vote_evidence: List[
        RoundDoubleVoteEvidence, MAX_ROUND_DOUBLE_VOTE_EVIDENCE
    ]  # [New in Simplex]
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
    previous_round_participation: List[
        ParticipationFlags, VALIDATOR_REGISTRY_LIMIT
    ]  # [Modified in Simplex]
    current_round_participation: List[
        ParticipationFlags, VALIDATOR_REGISTRY_LIMIT
    ]  # [Modified in Simplex]
    # Finality [Modified in Simplex]
    # [Modified in Simplex] replaces justification_bits + previous/current_justified
    justified_checkpoint: Checkpoint
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
    # Simplex finality gadget
    justified_height: Height  # [New in Simplex] height of ``justified_checkpoint``
    current_height: Height  # [New in Simplex]
    # [New in Simplex] slot at which the current height began (paper's s_h)
    current_height_start_slot: Slot
    justification_targets: List[
        Slot, VALIDATOR_REGISTRY_LIMIT
    ]  # [New in Simplex] per-validator justify target slot
    notarization_targets: List[
        Slot, VALIDATOR_REGISTRY_LIMIT
    ]  # [New in Simplex] per-validator highest (fresh) target slot
    notarized_checkpoint: Checkpoint  # [New in Simplex] paper's N
    finality_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in Simplex] extended window
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: See [Attestation Tracking](#attestation-tracking) for field roles. Key
invariants: fresh votes update `justification_targets` / `notarization_targets`
only; `finality_participation` persists across height advances and is reset only
when `advance_height` is called with `also_justify=True`.

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
    prev_slots_per_round = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda e: e["SLOT"]):
        era_start = entry["SLOT"]
        if slot < era_start:
            return total_rounds + Round((slot - prev_start) // prev_slots_per_round)
        total_rounds += Round((era_start - prev_start) // prev_slots_per_round)
        prev_start = era_start
        prev_slots_per_round = entry["SLOTS_PER_ROUND"]
    return total_rounds + Round((slot - prev_start) // prev_slots_per_round)
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
    prev_slots_per_round = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda e: e["SLOT"]):
        era_start = entry["SLOT"]
        era_rounds = Round((era_start - prev_start) // prev_slots_per_round)
        if remaining < era_rounds:
            return Slot(prev_start + remaining * prev_slots_per_round)
        remaining -= era_rounds
        prev_start = era_start
        prev_slots_per_round = entry["SLOTS_PER_ROUND"]
    return Slot(prev_start + remaining * prev_slots_per_round)
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

#### New `compute_leak_penalty_units`

```python
def compute_leak_penalty_units(
    state: BeaconState,
    index: ValidatorIndex,
    new_notarization: bool,
    new_justification: bool,
    new_finalization: bool,
    best_justification_slot: Slot,
) -> int:
    """
    [New in Simplex] Return penalty units in [0, 3] per paper Fig. leak-processslot.
    Three independent guards fire when the corresponding step does not happen
    this round. Slashed validators always accrue the maximum.
    ``best_justification_slot`` is consulted only when ``new_justification`` is
    ``False``; callers may pass ``FAR_FUTURE_SLOT`` otherwise.
    """
    if state.validators[index].slashed:
        return 3

    penalty = 0
    if not new_notarization and state.notarization_targets[index] == FAR_FUTURE_SLOT:
        penalty += 1
    if not new_justification:
        justification_slot = state.justification_targets[index]
        if justification_slot == FAR_FUTURE_SLOT or justification_slot != best_justification_slot:
            penalty += 1
    if (
        state.finalized_checkpoint != state.justified_checkpoint
        and not new_finalization
        and not state.finality_participation[index]
    ):
        penalty += 1
    return penalty
```

#### Modified `is_slashable_attestation_data`

*Note*: Fresh-simplex uses a **single slashing condition** (E1: finality-target
conflict). Validators may cast at most one justify (R1) and one notarize (R2)
vote per state-height; neither kind carries a self-slashing penalty on its own.
The only slashing condition is: if a validator commits to finality target T at
`finality_height = H` (via `finality_target = T`), they must not have voted for
any target other than T at `height = H`. Conflicting finalizations at the same
height require quorum intersection, and E1 ensures at least 1/3 of validators
are slashable. Round double-vote (same round, different data) uses a lighter
penalty via `RoundDoubleVoteEvidence`.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    # [Modified in Simplex] Single slashing condition (E1):
    # One vote commits to finality target T at height H; the other voted something != T at H.
    return (
        data_2.finality_target != Checkpoint()
        and data_1.height == data_2.finality_height
        and data_1.target != data_2.finality_target
    ) or (
        data_1.finality_target != Checkpoint()
        and data_2.height == data_1.finality_height
        and data_2.target != data_1.finality_target
    )
```

#### Modified `is_eligible_for_activation`

```python
def is_eligible_for_activation(state: BeaconState, validator: Validator) -> bool:
    """
    [Modified in Simplex] Uses compute_epoch_at_slot for finalized checkpoint.
    """
    return (
        # Placement in queue is finalized
        validator.activation_eligibility_epoch
        <= compute_epoch_at_slot(state.finalized_checkpoint.slot)
        # Has not yet been activated
        and validator.activation_epoch == FAR_FUTURE_EPOCH
    )
```

#### New `is_active_builder`

```python
def is_active_builder(state: BeaconState, builder_index: BuilderIndex) -> bool:
    """
    [New in Simplex] Uses compute_epoch_at_slot for finalized checkpoint.
    """
    builder = state.builders[builder_index]
    return (
        # Placement in builder list is finalized
        builder.deposit_epoch < compute_epoch_at_slot(state.finalized_checkpoint.slot)
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

#### New `get_target_slot_weights`

```python
def get_target_slot_weights(state: BeaconState, targets: Sequence[Slot]) -> Dict[Slot, Gwei]:
    """
    [New in Simplex] Sum active-validator effective balance per target slot.
    Excludes ``FAR_FUTURE_SLOT`` entries and slashed validators.
    """
    weights: Dict[Slot, Gwei] = {}
    active_indices = get_active_validator_indices(state, get_current_epoch(state))
    for index in active_indices:
        target_slot = targets[index]
        if target_slot == FAR_FUTURE_SLOT or state.validators[index].slashed:
            continue
        weights[target_slot] = Gwei(
            weights.get(target_slot, Gwei(0)) + state.validators[index].effective_balance
        )
    return weights
```

#### Modified `get_finality_delay`

```python
def get_finality_delay(state: BeaconState) -> uint64:
    # [Modified in Simplex] Uses compute_epoch_at_slot for finalized checkpoint.
    # Guard against underflow: in simplex, J&F runs at every round boundary,
    # so mid-epoch finalization can place finalized_epoch > previous_epoch.
    previous_epoch = get_previous_epoch(state)
    finalized_epoch = compute_epoch_at_slot(state.finalized_checkpoint.slot)
    return uint64(0) if finalized_epoch > previous_epoch else previous_epoch - finalized_epoch
```

#### Modified `get_unslashed_participating_indices`

```python
def get_unslashed_participating_indices(
    state: BeaconState, flag_index: int, round: Round
) -> Set[ValidatorIndex]:
    """
    Return the set of validator indices that are both active and unslashed for the given
    ``flag_index`` and ``round``.
    [Modified in Simplex] Takes a round instead of an epoch. Selects current or
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

#### New `is_target_on_chain`

```python
def is_target_on_chain(
    state: BeaconState, target: Checkpoint, historical_proof: Optional[HistoricalBlockProof] = None
) -> bool:
    """
    Check if ``target`` references an actual block that exists on this chain.
    Returns ``True`` if the block root at ``target.slot`` matches ``target.root``
    and a block was genuinely proposed at that slot (not a carried-forward root
    from an earlier slot). For targets outside the ``block_roots`` window, a
    ``HistoricalBlockProof`` against ``historical_summaries`` is required.
    """
    # Target slot must be in the past
    if target.slot >= state.slot:
        return False
    # In-window: use block_roots directly
    if target.slot + SLOTS_PER_HISTORICAL_ROOT > state.slot:
        # Block root must match
        if get_block_root_at_slot(state, target.slot) != target.root:
            return False
        # Verify an actual block was proposed at target.slot (not carried forward)
        if target.slot > 0 and get_block_root_at_slot(state, Slot(target.slot - 1)) == target.root:
            return False
        return True
    # Out-of-window: require valid historical proof
    if historical_proof is None:
        return False
    # *Note*: assert failure = block rejection. This is intentional: the proof is
    # proposer-supplied data, so an invalid proof is a proposer error, not a
    # graceful-degradation case.
    verify_historical_block_proof(state, target, historical_proof)
    return True
```

#### New `verify_historical_block_proof`

```python
def verify_historical_block_proof(
    state: BeaconState, target: Checkpoint, proof: HistoricalBlockProof
) -> None:
    """
    Verify that ``target`` references an actual block on this chain using a Merkle
    proof against ``historical_summaries``.
    """
    # Proof must be consistent with target
    assert proof.slot == target.slot
    assert proof.block_root == target.root
    assert target.slot > 0
    # Verify block_root at target.slot
    summary_index = target.slot // SLOTS_PER_HISTORICAL_ROOT
    assert summary_index < len(state.historical_summaries)
    block_summary_root = state.historical_summaries[summary_index].block_summary_root
    assert is_valid_merkle_branch(
        leaf=proof.block_root,
        branch=proof.block_proof,
        depth=BLOCK_ROOTS_PROOF_DEPTH,
        index=target.slot % SLOTS_PER_HISTORICAL_ROOT,
        root=block_summary_root,
    )
    # Verify prev_slot_root at target.slot - 1 (may be in a different summary)
    prev_slot = Slot(target.slot - 1)
    prev_summary_index = prev_slot // SLOTS_PER_HISTORICAL_ROOT
    assert prev_summary_index < len(state.historical_summaries)
    prev_block_summary_root = state.historical_summaries[prev_summary_index].block_summary_root
    assert is_valid_merkle_branch(
        leaf=proof.prev_slot_root,
        branch=proof.prev_slot_proof,
        depth=BLOCK_ROOTS_PROOF_DEPTH,
        index=prev_slot % SLOTS_PER_HISTORICAL_ROOT,
        root=prev_block_summary_root,
    )
    # Verify actual block was proposed (not carried forward)
    assert proof.prev_slot_root != proof.block_root
```

#### New `is_viable_attestation_target`

*Note*: Paper Definition: fresh-vote. Only votes passing this gate update
`justification_targets` / `notarization_targets`; by the gate, the recorded
slots lie in `[current_height_start_slot, latest_block_header.slot]`.

```python
def is_viable_attestation_target(state: BeaconState, attestation: Attestation) -> bool:
    """
    [New in Simplex] Viability gate for target-tracking: the attestation must
    carry the current state-height and target the current chain within the
    current-height interval. Non-viable attestations may still affect
    freshness-independent state (e.g. ``finality_participation``).
    """
    data = attestation.data
    return (
        data.height == state.current_height
        and data.target.slot >= state.current_height_start_slot
        and is_target_on_chain(state, data.target, attestation.historical_block_proof)
    )
```

#### New `get_available_committee`

```python
def get_available_committee(state: BeaconState, slot: Slot) -> Sequence[ValidatorIndex]:
    """
    [New in Simplex] Return the 512-member available committee for the given slot.
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
                // SLOTS_PER_ROUND  # [Modified in Simplex]
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
    # [Modified in Simplex] Slot-within-round via round helpers (schedule-safe)
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
    # [New in Simplex]
    set_or_append_list(state.justification_targets, index, FAR_FUTURE_SLOT)
    set_or_append_list(state.notarization_targets, index, FAR_FUTURE_SLOT)
    set_or_append_list(state.finality_participation, index, False)
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(state: BeaconState, new_notarized: Checkpoint, also_justify: bool) -> None:
    """
    [New in Simplex] Advance ``current_height``: record ``new_notarized`` as
    ``notarized_checkpoint``; if ``also_justify`` additionally set
    ``justified_checkpoint`` and reset ``finality_participation``; bump
    ``current_height``, update ``current_height_start_slot``, and reset the
    per-validator target arrays.
    """
    if also_justify:
        state.justified_checkpoint = new_notarized
        state.justified_height = state.current_height
        state.finality_participation = Bitlist[VALIDATOR_REGISTRY_LIMIT](
            [False] * len(state.validators)
        )
    state.notarized_checkpoint = new_notarized
    state.current_height = Height(state.current_height + 1)
    state.current_height_start_slot = state.latest_block_header.slot
    num_validators = len(state.validators)
    state.justification_targets = [FAR_FUTURE_SLOT for _ in range(num_validators)]
    state.notarization_targets = [FAR_FUTURE_SLOT for _ in range(num_validators)]
```

#### New `compute_justified_checkpoint`

```python
def compute_justified_checkpoint(state: BeaconState) -> Checkpoint:
    """
    [New in Simplex] Return the checkpoint the justify branch would fire on:
    the plurality slot on ``justification_targets``, if its total weight
    reaches a 2/3 quorum. Returns ``Checkpoint()`` otherwise. Under the
    paper's honest rule and f < n/3, ``lem:just-unique-height`` guarantees
    that at most one slot can reach quorum, so checking only the plurality
    is sufficient.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return Checkpoint()

    best_slot, weight = compute_best_justification_target(state)
    if best_slot == FAR_FUTURE_SLOT:
        return Checkpoint()
    total_active_balance = get_total_active_balance(state)
    if weight * FINALITY_QUORUM_DENOMINATOR < total_active_balance * FINALITY_QUORUM_NUMERATOR:
        return Checkpoint()
    return Checkpoint(slot=best_slot, root=get_block_root_at_slot(state, best_slot))
```

#### New `compute_notarized_checkpoint`

```python
def compute_notarized_checkpoint(state: BeaconState) -> Checkpoint:
    """
    [New in Simplex] Prefix-notarization walk (Definition: prefix-notarization).
    Walks on-chain slots from ``latest_block_header.slot`` down to
    ``current_height_start_slot``, cumulating
    ``|{i : notarization_targets[i] == slot, not slashed}|`` at each step;
    returns the checkpoint at the first (highest) slot where the cumulative
    sum reaches 2/3, or ``Checkpoint()`` if no slot qualifies.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return Checkpoint()

    total_active_balance = get_total_active_balance(state)
    slot_weights = get_target_slot_weights(state, state.notarization_targets)

    head_slot = state.latest_block_header.slot
    cumulative_weight = Gwei(0)
    for slot in sorted(slot_weights.keys(), reverse=True):
        if slot < state.current_height_start_slot or slot > head_slot:
            continue
        cumulative_weight += slot_weights[slot]
        if (
            cumulative_weight * FINALITY_QUORUM_DENOMINATOR
            >= total_active_balance * FINALITY_QUORUM_NUMERATOR
        ):
            return Checkpoint(slot=slot, root=get_block_root_at_slot(state, slot))
    return Checkpoint()
```

#### New `has_new_finalization`

```python
def has_new_finalization(state: BeaconState) -> bool:
    """
    [New in Simplex] Return ``True`` iff a 2/3 quorum holds on
    ``finality_participation`` (non-slashed only) and finality is still
    pending (``finalized_checkpoint != justified_checkpoint``).
    """
    if state.finalized_checkpoint == state.justified_checkpoint:
        return False
    total_active_balance = get_total_active_balance(state)
    active_indices = get_active_validator_indices(state, get_current_epoch(state))
    participation_weight = Gwei(
        sum(
            state.validators[index].effective_balance
            for index in active_indices
            if state.finality_participation[index] and not state.validators[index].slashed
        )
    )
    return (
        participation_weight * FINALITY_QUORUM_DENOMINATOR
        >= total_active_balance * FINALITY_QUORUM_NUMERATOR
    )
```

#### New `compute_best_justification_target`

```python
def compute_best_justification_target(state: BeaconState) -> Tuple[Slot, Gwei]:
    """
    [New in Simplex] Return the plurality slot on ``justification_targets`` and
    its total effective-balance weight (excluding slashed validators). Returns
    ``(FAR_FUTURE_SLOT, Gwei(0))`` if no validator has a recorded justify
    target at the current height. Tiebreak: highest weight first, then highest
    slot.

    *Note*: The paper keeps slashed validators in the quorum; the spec
    excludes them uniformly across all ``justification_targets`` /
    ``notarization_targets`` tallies (adaptation, not a paper match).
    """
    slot_weights = get_target_slot_weights(state, state.justification_targets)
    if not slot_weights:
        return FAR_FUTURE_SLOT, Gwei(0)
    best_slot = max(slot_weights.keys(), key=lambda slot: (slot_weights[slot], slot))
    return best_slot, slot_weights[best_slot]
```

#### Modified `process_justification_and_finalization`

*Note*: Paper's `processHeight` (alg:state-machine). Runs AFTER
`process_inactivity_updates` / `process_rewards_and_penalties` so they see
pre-advance state. At most one of the justify/prefix-notarize branches advances
height per invocation; the finality branch is independent.

```python
def process_justification_and_finalization(state: BeaconState) -> None:
    """
    [Modified in Simplex] Three-branch state machine per paper processHeight.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    # (1) Finality: F ← J (does not advance height)
    if has_new_finalization(state):
        state.finalized_checkpoint = state.justified_checkpoint

    # (2) Justify branch
    justified = compute_justified_checkpoint(state)
    if justified != Checkpoint():
        advance_height(state, new_notarized=justified, also_justify=True)
        return

    # (3) Prefix-notarize branch
    notarized = compute_notarized_checkpoint(state)
    if notarized != Checkpoint():
        advance_height(state, new_notarized=notarized, also_justify=False)
        return
```

#### Modified `process_inactivity_updates`

*Note*: Three-guard design (paper alg:leak-processslot). Guards are computed
against the pre-advance state; `process_justification_and_finalization` runs
later in `process_round`.

```python
def process_inactivity_updates(state: BeaconState) -> None:
    # Skip early epochs — aligned with round-outcome primitives' guard
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    # [Modified in Simplex] Pre-advance signals from paper's three branches.
    # A justify quorum implies a prefix-notarize quorum on the same chain, so
    # ``new_justification ⇒ new_notarization``.
    new_justification = compute_justified_checkpoint(state) != Checkpoint()
    new_notarization = compute_notarized_checkpoint(state) != Checkpoint()
    new_finalization = has_new_finalization(state)
    if not new_justification:
        best_justification_slot, _ = compute_best_justification_target(state)
    else:
        best_justification_slot = FAR_FUTURE_SLOT

    for index in get_eligible_validator_indices(state):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_notarization,
            new_justification,
            new_finalization,
            best_justification_slot,
        )
        if penalty_units == 0:
            state.inactivity_scores[index] -= min(1, state.inactivity_scores[index])
        else:
            state.inactivity_scores[index] += INACTIVITY_SCORE_BIAS * penalty_units
        # Decrease the inactivity score of all eligible validators during a leak-free round
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
    [Modified in Simplex] Rewards and penalties are scaled by 1/ROUNDS_PER_EPOCH
    to keep per-epoch totals constant when running per-round.
    """
    rewards = [Gwei(0)] * len(state.validators)
    penalties = [Gwei(0)] * len(state.validators)
    # [Modified in Simplex] Pass previous round instead of previous epoch
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
                # [Modified in Simplex] Scale by 1/ROUNDS_PER_EPOCH
                rewards[index] += Gwei(
                    reward_numerator // (active_increments * WEIGHT_DENOMINATOR * ROUNDS_PER_EPOCH)
                )
        elif flag_index != TIMELY_HEAD_FLAG_INDEX:
            # [Modified in Simplex] Scale by 1/ROUNDS_PER_EPOCH
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
    [Modified in Simplex] Three-guard leak: a penalty unit accrues for each
    of notarization / justification / finalization that did not happen this
    round. Scaled by 1/ROUNDS_PER_EPOCH**2 per round (one factor for
    per-round application, one for scores accumulating ROUNDS_PER_EPOCH
    times faster). Up to 3 penalty units per round.
    """
    rewards = [Gwei(0) for _ in range(len(state.validators))]
    penalties = [Gwei(0) for _ in range(len(state.validators))]

    # [Modified in Simplex] Pre-advance signals from paper's three branches.
    # A justify quorum implies a prefix-notarize quorum on the same chain, so
    # ``new_justification ⇒ new_notarization``.
    new_justification = compute_justified_checkpoint(state) != Checkpoint()
    new_notarization = compute_notarized_checkpoint(state) != Checkpoint()
    new_finalization = has_new_finalization(state)
    if not new_justification:
        best_justification_slot, _ = compute_best_justification_target(state)
    else:
        best_justification_slot = FAR_FUTURE_SLOT

    for index in get_eligible_validator_indices(state):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_notarization,
            new_justification,
            new_finalization,
            best_justification_slot,
        )
        if penalty_units > 0:
            penalty_numerator = (
                state.validators[index].effective_balance * state.inactivity_scores[index]
            )
            # [Modified in Simplex] ROUNDS_PER_EPOCH ** 2: one factor for
            # per-round penalty application, one for scores accumulating RPE times faster
            penalty_denominator = (
                INACTIVITY_SCORE_BIAS
                * INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
                * (ROUNDS_PER_EPOCH**2)
            )
            penalties[index] += Gwei(penalty_numerator // penalty_denominator) * penalty_units
    return rewards, penalties
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
    # [Modified in Simplex] Uses slot-based finalized checkpoint
    finalized_slot = state.finalized_checkpoint.slot

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
    # [Modified in Simplex] Uses round-based participation arrays
    state.previous_round_participation = state.current_round_participation
    state.current_round_participation = [
        ParticipationFlags(0b0000_0000) for _ in range(len(state.validators))
    ]
```

#### New `process_round`

```python
def process_round(state: BeaconState) -> None:
    """
    [New in Simplex] Per-round processing: finality voting cycle functions
    that run every SLOTS_PER_ROUND slots. Epoch boundaries are always round
    boundaries, so process_round runs before process_epoch at epoch
    transitions. Inactivity updates run before justification and finalization
    so that they see the pre-advance state (the three leak guards reference
    ``justification_targets``, ``notarization_targets``, and ``finality_participation`` before the
    state machine resets them).
    """
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_justification_and_finalization(state)
    process_participation_flag_updates(state)
```

#### Modified `process_epoch`

```python
def process_epoch(state: BeaconState) -> None:
    # [Modified in Simplex] Finality-cycle functions moved to process_round.
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
    [Modified in Simplex] Adds round processing at round boundaries.
    Round processing runs before epoch processing. Since epoch boundaries are
    always round boundaries, the order at epoch transition is:
    process_round (last round of epoch) → process_epoch (administrative).
    """
    assert state.slot < slot
    while state.slot < slot:
        process_slot(state)
        # [New in Simplex] Round processing at round boundaries (schedule-aware)
        if compute_round_at_slot(Slot(state.slot + 1)) > compute_round_at_slot(state.slot):
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
    [Modified in Simplex] Uses slot epoch for signing domain (target epoch may differ).
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

#### New `validate_attestation`

```python
def validate_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [New in Simplex] Assert attestation data well-formedness, inclusion
    window (current or previous epoch), committee structure (Electra
    pattern), and signature validity. Does NOT gate on
    ``data.height == state.current_height``: older-height votes may still
    carry useful ``finality_participation`` updates (and future extensions
    may reward them). Viability for target tracking is enforced separately
    via ``is_viable_attestation_target``.
    """
    data = attestation.data

    # Inclusion delay
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot

    # Finality fields
    assert data.target != Checkpoint()
    assert data.kind in (ATTESTATION_KIND_JUSTIFY, ATTESTATION_KIND_NOTARIZE)
    # Only R1 (justify) votes may carry a finality piggyback.
    if data.finality_target == Checkpoint():
        assert data.finality_height == FAR_FUTURE_HEIGHT
    else:
        assert data.kind == ATTESTATION_KIND_JUSTIFY
        assert data.finality_height < data.height

    # Bounded inclusion window: current or previous epoch. Mirrors the
    # wire-side bound in ``validate_on_attestation``. Older attestations are
    # never needed because honest validators re-submit via R2 (notarize).
    data_epoch = compute_epoch_at_slot(data.slot)
    assert data_epoch in (get_current_epoch(state), get_previous_epoch(state))

    # Committee structure (Electra pattern)
    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    for committee_index in committee_indices:
        assert committee_index < get_committee_count_per_slot(state, data_epoch)
        committee = get_beacon_committee(state, data.slot, committee_index)
        committee_attesters = set(
            attester_index
            for i, attester_index in enumerate(committee)
            if attestation.aggregation_bits[committee_offset + i]
        )
        assert len(committee_attesters) > 0
        committee_offset += len(committee)
    assert len(attestation.aggregation_bits) == committee_offset

    # Signature
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))
```

#### New `update_justification_and_notarization_targets`

```python
def update_justification_and_notarization_targets(
    state: BeaconState,
    validator_index: ValidatorIndex,
    data: AttestationData,
) -> None:
    """
    [New in Simplex] Record an attestation's contribution to the per-validator
    target trackers: update ``notarization_targets[i]`` to the max seen target
    slot (all kinds) and assign ``justification_targets[i]`` on the justify
    kind. Caller must gate on ``is_viable_attestation_target``.
    """
    current_notarization_slot = state.notarization_targets[validator_index]
    if current_notarization_slot == FAR_FUTURE_SLOT or data.target.slot > current_notarization_slot:
        state.notarization_targets[validator_index] = data.target.slot
    if data.kind == ATTESTATION_KIND_JUSTIFY:
        state.justification_targets[validator_index] = data.target.slot
```

#### New `update_finality_participation`

```python
def update_finality_participation(
    state: BeaconState,
    validator_index: ValidatorIndex,
    data: AttestationData,
) -> None:
    """
    [New in Simplex] Set the voter's bit in ``finality_participation`` when
    the attestation's finality piggyback matches the current justified
    checkpoint and finalization is still pending. Independent of viability
    (per paper ``processVote``): a piggyback matching ``(justified_height,
    justified_checkpoint)`` records progress toward finalizing the current
    justified checkpoint regardless of the attestation's own target.
    """
    if (
        data.finality_target != Checkpoint()
        and data.finality_height == state.justified_height
        and data.finality_target == state.justified_checkpoint
        and state.finalized_checkpoint != state.justified_checkpoint
        and not state.finality_participation[validator_index]
    ):
        state.finality_participation[validator_index] = True
```

#### New `record_timely_target`

```python
def record_timely_target(
    state: BeaconState,
    validator_index: ValidatorIndex,
    round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT],
) -> Gwei:
    """
    [New in Simplex] Set the TIMELY_TARGET flag on ``round_participation``
    for this validator (if not already set) and return the proposer-reward
    numerator contribution.
    """
    if has_flag(round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX):
        return Gwei(0)
    round_participation[validator_index] = add_flag(
        round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
    )
    return Gwei(get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT)
```

#### Modified `process_attestation`

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in Simplex] Delegate to ``validate_attestation`` for
    assertions. Per-validator: ``update_finality_participation`` always runs
    (so older-height votes can still carry valid finality piggybacks);
    ``update_justification_and_notarization_targets`` and the TIMELY_TARGET
    reward fire only for viable attestations (current height, on-chain
    target, ``target.slot >= current_height_start_slot``).

    *Note*: Any viable vote earns the TIMELY_TARGET reward. This does not
    specifically incentivize justification over notarization; inactivity
    penalties handle that asymmetry in the negative direction
    (justification-missed guard).
    """
    data = attestation.data
    validate_attestation(state, attestation)
    viable_target = is_viable_attestation_target(state, attestation)

    # Reward-eligible round-participation list, or None if the attestation's
    # round is outside the current/previous-round reward window.
    attestation_round = compute_round_at_slot(data.slot)
    if attestation_round == get_current_round(state):
        round_participation = state.current_round_participation
    elif attestation_round == get_previous_round(state):
        round_participation = state.previous_round_participation
    else:
        round_participation = None

    proposer_reward_numerator = Gwei(0)
    current_epoch = get_current_epoch(state)
    for validator_index in get_attesting_indices(state, attestation):
        if not is_active_validator(state.validators[validator_index], current_epoch):
            continue
        update_finality_participation(state, validator_index, data)
        if viable_target:
            update_justification_and_notarization_targets(state, validator_index, data)
            if round_participation is not None:
                proposer_reward_numerator += record_timely_target(
                    state, validator_index, round_participation
                )

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
    [New in Simplex] Process an available committee attestation for LMD-GHOST.
    Sets TIMELY_HEAD flag and handles builder payment weight.
    """
    data = attestation.data
    # [Modified in Simplex] Round-based acceptance window
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

    # [Modified in Simplex] Write back updated builder payment weight
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
    [New in Simplex] Process round double-vote evidence.
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
    # [New in Simplex]
    for_ops(body.available_attestations, process_available_attestation)
    # [New in Simplex] Round double-vote evidence (lighter penalty than attester slashing)
    for_ops(body.round_double_vote_evidence, process_round_double_vote_evidence)
```

## Fork transition

### New `upgrade_to_simplex`

*Note*: The notarized checkpoint is seeded from the pre-state's justified
checkpoint (after slot conversion). The current height's start slot is set to
the latest block header slot so that the first fresh-vote gate references the
pre-fork tip.

```python
def upgrade_to_simplex(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)
    justified_checkpoint = Checkpoint(
        slot=compute_start_slot_at_epoch(pre.current_justified_checkpoint.epoch),
        root=pre.current_justified_checkpoint.root,
    )

    post = BeaconState(
        # Genesis
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        # State
        slot=pre.slot,
        fork=Fork(
            previous_version=pre.fork.current_version,
            current_version=SIMPLEX_FORK_VERSION,  # [Modified in Simplex]
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
        # Finality [Modified in Simplex]
        # Removed: justification_bits, previous_justified_checkpoint, current_justified_checkpoint
        # Convert epoch-based Checkpoints to slot-based
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=Checkpoint(
            slot=compute_start_slot_at_epoch(pre.finalized_checkpoint.epoch),
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
        # Simplex [New in Simplex]
        justified_height=GENESIS_HEIGHT,
        current_height=GENESIS_HEIGHT,
        current_height_start_slot=pre.latest_block_header.slot,
        justification_targets=[FAR_FUTURE_SLOT for _ in range(len(pre.validators))],
        notarization_targets=[FAR_FUTURE_SLOT for _ in range(len(pre.validators))],
        notarized_checkpoint=justified_checkpoint,
        finality_participation=[False for _ in range(len(pre.validators))],
    )

    return post
```

## Genesis

### Modified `initialize_beacon_state_from_eth1`

*Note*: `notarized_checkpoint` and `justified_checkpoint` are initialized to
zero roots at genesis; the `epoch <= GENESIS_EPOCH + 1` guard in
`process_justification_and_finalization` ensures they are never used on-chain.

```python
def initialize_beacon_state_from_eth1(
    eth1_block_hash: Hash32, eth1_timestamp: uint64, deposits: Sequence[Deposit]
) -> BeaconState:
    fork = Fork(
        previous_version=GENESIS_FORK_VERSION,
        current_version=SIMPLEX_FORK_VERSION,
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

    # [New in Simplex] Initialize finality fields
    state.current_height = GENESIS_HEIGHT
    state.justified_checkpoint = Checkpoint(slot=GENESIS_SLOT, root=Root())
    state.finalized_checkpoint = Checkpoint(slot=GENESIS_SLOT, root=Root())
    state.justified_height = GENESIS_HEIGHT
    state.current_height_start_slot = GENESIS_SLOT
    state.notarized_checkpoint = Checkpoint(slot=GENESIS_SLOT, root=Root())

    return state
```

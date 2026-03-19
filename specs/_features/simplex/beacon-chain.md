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
    - [Modified `is_active_builder`](#modified-is_active_builder)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_current_round`](#new-get_current_round)
    - [New `get_previous_round`](#new-get_previous_round)
    - [Modified `get_finality_delay`](#modified-get_finality_delay)
    - [Modified `get_unslashed_participating_indices`](#modified-get_unslashed_participating_indices)
    - [New `get_previous_height`](#new-get_previous_height)
    - [New `is_target_on_chain`](#new-is_target_on_chain)
    - [New `verify_historical_block_proof`](#new-verify_historical_block_proof)
    - [New `get_confirmed_target`](#new-get_confirmed_target)
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
    - [New `compute_round_outcome`](#new-compute_round_outcome)
    - [Modified `process_justification_and_finalization`](#modified-process_justification_and_finalization)
    - [Modified `process_inactivity_updates`](#modified-process_inactivity_updates)
    - [Modified `get_flag_index_deltas`](#modified-get_flag_index_deltas)
    - [Modified `get_inactivity_penalty_deltas`](#modified-get_inactivity_penalty_deltas)
    - [Modified `process_rewards_and_penalties`](#modified-process_rewards_and_penalties)
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
  - [New `upgrade_to_simplex`](#new-upgrade_to_simplex)
- [Genesis](#genesis)
  - [Modified `initialize_beacon_state_from_eth1`](#modified-initialize_beacon_state_from_eth1)

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for simplex-based finality. It replaces
Casper FFG with a two-round simplex-based finality gadget adapted from the
Simplex consensus protocol (Chan & Pass, 2023). The model is n >= 3f+1, with
2/3 quorums for justification, height advance, and finalization. Finalization takes two steps: justify at height H, then confirm via
piggybacked finalize votes at any subsequent height (extended finalization
window).

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at round boundaries

At each round transition, the height may advance in three ways: (1)
**justification** — target T alone reaches >= 2/3, (2) **timeout-assisted
advance** — T-voters + timeout >= 2/3 with T >= 1/3 minimum target support,
or (3) **pure timeout** — timeout alone >= 2/3.

### Thresholds (n >= 3f+1)

| Threshold                | Stake  | Purpose                                              |
| ------------------------ | ------ | ---------------------------------------------------- |
| Timeout-assisted advance | >= 2/3 | T-voters ∪ timeout-voters reach quorum → advance     |
| Minimum target support   | >= 1/3 | T alone must meet this for timeout-assisted advance   |
| Justification            | >= 2/3 | Target alone reaches quorum (subset of advance)      |
| Finalization             | >= 2/3 | Piggybacked confirm of justified checkpoint          |
| Accountable safety       | 1/3    | Standard BFT                                         |

### Decoupled Consensus

Finality and LMD-GHOST use different attestation types:

- **Attestations**: All active validators attest once per height via standard
  beacon committee attestations (Electra format). `AttestationData` carries a
  finality target, height, finalize target, and finalize height. These determine
  justification, timeout, and finalization. Attester slashings enforce height
  double-target and finalize-target conflict conditions.
- **Available attestations**: A small 512-member available committee attests per
  slot for fork choice via `AvailableAttestation`. This committee is selected
  from the full active set using `compute_balance_weighted_selection` (same
  mechanism as PTC).

### Attestation Tracking

Finality attestations are tracked per validator using two independent
structures:

1. **Target slot list** (`current_height_target_slots`): each validator's
   voted target, encoded as `FAR_FUTURE_SLOT` (hasn't voted) or a valid on-chain
   slot. Since only on-chain targets are tracked (verified by
   `is_target_on_chain`), the slot uniquely identifies the target block — the
   root is recoverable via `get_block_root_at_slot`. More compact than a full
   checkpoint per validator (8 bytes vs 40 bytes) while preserving enough
   information for per-target justification counting.
2. **Timeout bitlist** (`current_height_timeout_participation`): whether the
   validator voted explicit timeout (`target = Checkpoint()`). Independent of
   the target slot — a validator can have both a target vote AND a timeout
   vote at the same height (voted target in round R, timeout in round R+k).

First on-chain target vote wins in `current_height_target_slots` — no
overwrites. Timeout is tracked independently (set once, never cleared within a
height).

A separate **finalize participation** bitlist tracks finalization confirmations
across the extended window. It is NOT tied to a single height — it persists
until the justified checkpoint changes (new justification), at which point
it resets. This ensures finalize votes accumulate across heights until
finalization occurs, preventing the adversary from stranding justified
checkpoints by alternating justification and timeout.

On height advance, a target participation bitlist is derived from
`current_height_target_slots` (validators whose slot equals the canonical
target's slot) and rotated to `previous_height_target_participation` for late
processing, rewards, and the inactivity leak's Layer 2. Previous height data is
kept for one height (same as Gasper keeping `previous_epoch_participation`).
The canonical target retains special status for **TIMELY_TARGET rewards** only
(outside the leak). During the inactivity leak, both Layer 1 and Layer 2 use
the **majority target** (highest non-slashed weight in `slot_weights`) — this
penalizes lack of coordination directly, without requiring a k-deep confirmation
mechanism as Schelling point.

## Configuration

Warning: this configuration is not definitive.

| Name                        | Value                                 |
| --------------------------- | ------------------------------------- |
| `SIMPLEX_FORK_VERSION`      | `Version('0x10000000')`               |
| `SIMPLEX_FORK_EPOCH`        | `Epoch(18446744073709551615)` **TBD** |

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

| Name                          | Value       |
| ----------------------------- | ----------- |
| `GENESIS_HEIGHT`              | `Height(0)` |
| `FAR_FUTURE_HEIGHT`           | `Height(2**64 - 1)` |
| `GENESIS_ROUND`               | `Round(0)`  |
| `FINALITY_QUORUM_NUMERATOR`   | `uint64(2)` |
| `FINALITY_QUORUM_DENOMINATOR` | `uint64(3)` |

### Slashing constants

| Name                                       | Value       |
| ------------------------------------------ | ----------- |
| `PROPORTIONAL_SLASHING_MULTIPLIER_SIMPLEX` | `uint64(3)` |

### Participation flag indices

*Note*: The source flag is removed in simplex finality since there is no source
checkpoint to attest to.

| Name                       | Value |
| -------------------------- | ----- |
| `TIMELY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_HEAD_FLAG_INDEX`   | `1`   |

### Incentivization weights

*Note*: The source weight (14/64) is redistributed to target in simplex
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

| Name                       | Value                        |
| -------------------------- | ---------------------------- |
| `AVAILABLE_COMMITTEE_SIZE` | `uint64(2**9)` (= 512)       |
| `BLOCK_ROOTS_PROOF_DEPTH`  | `uint64(13)` (= log2(8192))  |
| `FAR_FUTURE_SLOT`          | `Slot(2**64 - 1)`             |

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

*Note*: Self-verifiable proof that a block was genuinely proposed at a given slot
on this chain, for targets outside the `block_roots` window. Both `slot` and
`block_root` are redundant with the attestation's `target` but included for
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
`target` is repurposed as a simplex finality target, `height` is added,
`finalize_target` is a piggyback vote specifying which justified checkpoint to
confirm (`Checkpoint()` means no finalize vote), `finalize_height` is the
height at which `finalize_target` was justified (`FAR_FUTURE_HEIGHT` when no
finalize vote), and `payload_present` signals payload availability for the
voted block. The `beacon_block_root` and `payload_present` fields are used by
the fork choice only — `process_attestation` uses `target`, `height`,
`finalize_target`, and `finalize_height`.

```python
class AttestationData(Container):
    slot: Slot
    beacon_block_root: Root  # [Modified in Simplex] LMD head vote for fork choice
    target: Checkpoint  # [Modified in Simplex] Finality target or Checkpoint() for timeout
    height: Height  # [New in Simplex] Finality height being attested to
    finalize_target: Checkpoint  # [New in Simplex] Justified checkpoint to confirm, or Checkpoint() for none
    finalize_height: Height  # [New in Simplex] Height at which finalize_target was justified, or FAR_FUTURE_HEIGHT
    payload_present: boolean  # [New in Simplex] Payload availability signal
```

#### `Attestation`

*Note*: `AttestationData` is modified (see above). `Attestation` extends the
Electra committee-based format with an optional `HistoricalBlockProof` for
non-canonical target votes when the target block is outside the `block_roots` window.
The proof is unsigned (not part of `AttestationData`) — the proposer attaches it
when including the attestation in a block.

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
    previous_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]  # [Modified in Simplex]
    current_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]  # [Modified in Simplex]
    # Finality [Modified in Simplex]
    justified_checkpoint: Checkpoint  # [Modified in Simplex] replaces justification_bits + previous/current_justified
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
    justification_floor_slot: Slot  # [New in Simplex] fork-choice monotonicity floor
    current_height_canonical_target: Checkpoint  # [New in Simplex] set by advance_height
    current_height_target_slots: List[Slot, VALIDATOR_REGISTRY_LIMIT]  # [New in Simplex] per-validator vote slot
    current_height_timeout_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in Simplex] independent timeout tracking
    finalize_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in Simplex] extended window
    # Previous height
    previous_height_canonical_target: Checkpoint  # [New in Simplex]
    previous_height_target_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # [New in Simplex]
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: Target votes and timeout votes are tracked **independently**.
`current_height_target_slots` records each validator's target (or `FAR_FUTURE_SLOT`
if not yet voted). `current_height_timeout_participation` records whether the
validator voted explicit timeout. A validator can have both set — they voted
target in one round and timeout in a later round at the same stalled height.
First on-chain target vote wins in `target_slots` — no overwrites. Timeout is
set once, independently. Any on-chain target that independently reaches 2/3 is
justified. Height also advances via **timeout-assisted advance**: for some target T
with >= 1/3 non-slashed weight, the union of non-slashed T-voters and timeout
voters reaches 2/3. Pure timeout (timeout alone >= 2/3) also advances without
any target threshold. The `finalize_participation`
bitlist persists across
heights (extended finalization window) until the justified checkpoint changes.
On `advance_height`, a target participation bitlist is derived from
`current_height_target_slots` and rotated to `previous_height_target_participation`
for late processing and the inactivity leak's Layer 2.

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

#### New `compute_leak_penalty_units`

```python
def compute_leak_penalty_units(
    state: BeaconState, index: ValidatorIndex, has_height_progress: bool,
    has_pending_finalization: bool, majority_target_slot: Slot,
    majority_has_threshold: bool,
) -> int:
    """
    [New in Simplex] Returns the number of inactivity penalty units for this
    validator (0, 1, or 2). Two-layer design with **stacking penalties**:

    - **Height completing this round (Layer 2)**: Two independent checks:
      1. **Target check**: did the validator vote for the **majority target**?
         Uses the ex-post majority target (highest non-slashed weight in
         ``slot_weights``) rather than the ex-ante canonical target. This
         penalizes lack of coordination directly, removing the need for a k-deep
         confirmation mechanism as Schelling point. The exclusive check (must NOT
         also have voted timeout) aligns penalty-sense weight with advance-sense
         ``slot_weights``.
      2. **Finalize check** (at ``justified_height + 1`` with pending only):
         did the validator confirm finalization?
      Each failed check adds one penalty unit. A validator failing both gets
      2 × ``INACTIVITY_SCORE_BIAS``. The checks are independent — the target
      penalty and finalize penalty do not subsume each other. This ensures
      the amortized leak rate is at least N/3 penalty units per round: the
      advance round's double penalty exactly compensates for zero-penalty
      justification rounds past ``justified_height + 1``.
    - **Height stalled (Layer 1)**: Single check (0 or 1 unit). Exempt if
      voted explicit timeout, OR voted for the **majority target** when the
      majority target has >= 1/3 non-slashed weight (``majority_has_threshold``).
      The conditional exemption mirrors the minimum target support threshold:
      if the majority target has < 1/3 support, only timeout voters are exempt.
      This preserves the tight 1/3 leak bound: during a stall, the exempt set
      is always < 2/3 (either the advance condition would have fired, or the
      exempt set is bounded).
    """
    if state.validators[index].slashed:
        return 2  # [Modified in Simplex] max penalty for slashed validators

    if has_height_progress:
        # (advance_height has not yet run, so the completing height's data is in current_height_*)
        count = 0
        # Check 1: majority target (exclusive — must NOT also have voted timeout)  # [Modified in Simplex]
        # A validator who hedged (voted majority AND timeout) contributed to the
        # advance via timeout weight, not justification weight.
        if (state.current_height_target_slots[index] != majority_target_slot
                or state.current_height_timeout_participation[index]):
            count += 1
        # Check 2: finalize (independent, once-per-checkpoint at J+1 with pending)
        if has_pending_finalization and state.current_height == state.justified_height + 1:
            if not state.finalize_participation[index]:
                count += 1
        return count
    else:
        # [Modified in Simplex] Stall: exempt if timeout, or majority target with >= 1/3 support
        if state.current_height_timeout_participation[index]:
            return 0
        voted_majority = state.current_height_target_slots[index] == majority_target_slot
        if voted_majority and majority_has_threshold:
            return 0
        return 1
```

#### Modified `is_slashable_attestation_data`

*Note*: Simplex finality replaces the FFG double-vote and surround-vote
conditions with two slashing conditions: (1) height double-target — different
non-empty targets at the same height, and (2) finalize-target conflict —
voting to finalize target T (justified at height H) while also having voted
for something other than T at height H (either timeout or a different target).
The finalize vote can be carried at any subsequent height (extended window),
so the slashing condition checks `finalize_target` and `finalize_height`
regardless of which height the finalize vote was cast at. The extended
Condition 2 is essential for safety under the timeout-assisted advance rule: it
catches validators who voted for a different target E at the justified height
and then committed to finalize T (see the safety proof, sub-case c-ii).
Round double-vote (same round, different data) uses a lighter penalty via
`RoundDoubleVoteEvidence`.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    [Modified in Simplex] Two slashing conditions:
    1. Height double-target: same height, both non-empty targets, different targets.
    2. Finalize-target conflict: finalize_target = T at height H + any vote != T at H.
       Covers both timeout at H (target = Checkpoint()) and wrong target at H (target = D != T).
    Assumes data_1.height <= data_2.height (caller must order by height;
    the proposer is responsible for ordering evidence correctly).
    """
    assert data_1.height <= data_2.height
    # Condition 1: Two different non-empty targets at the same height
    height_double_target = (
        data_1.height == data_2.height
        and data_1.target != Checkpoint()
        and data_2.target != Checkpoint()
        and data_1.target != data_2.target
    )
    # Condition 2: Finalize-target conflict (extended)
    # data_2 commits to finalize target T (justified at some height H).
    # data_1 voted for something other than T at height H.
    # Catches timeout (Checkpoint() != T) AND wrong target (D != T).
    finalize_target_conflict = (
        data_2.finalize_target != Checkpoint()
        and data_1.height == data_2.finalize_height
        and data_1.target != data_2.finalize_target
    )
    return height_double_target or finalize_target_conflict
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

#### Modified `is_active_builder`

```python
def is_active_builder(state: BeaconState, builder_index: BuilderIndex) -> bool:
    """
    [Modified in Simplex] Uses compute_epoch_at_slot for finalized checkpoint.
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

#### New `get_previous_height`

```python
def get_previous_height(state: BeaconState) -> Height:
    if state.current_height > GENESIS_HEIGHT:
        return Height(state.current_height - 1)
    return GENESIS_HEIGHT
```

#### New `is_target_on_chain`

```python
def is_target_on_chain(state: BeaconState, target: Checkpoint,
                       historical_proof: Optional[HistoricalBlockProof] = None) -> bool:
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
def verify_historical_block_proof(state: BeaconState, target: Checkpoint,
                                  proof: HistoricalBlockProof) -> None:
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

#### New `get_confirmed_target`

*Note*: This function is no longer used by `advance_height` (which always sets
the canonical target to the latest block). It may still be useful for the
validator specification as a suggestion for what target to vote for during the
inactivity leak.

```python
def get_confirmed_target(state: BeaconState) -> Checkpoint:
    """
    Return an objectively confirmed canonical target for use during an inactivity leak.
    Uses a k-deep block (one round old) as an undeniable Schelling point — all validators
    on this chain should agree on it. The justified checkpoint is the lower bound since it
    is confirmed by definition (received 2/3 attestation weight).
    """
    current_round = get_current_round(state)
    if current_round > GENESIS_ROUND:
        slot = compute_start_slot_at_round(Round(current_round - 1))
    else:
        slot = GENESIS_SLOT
    # Walk back past empty slots to find the actual proposal slot,
    # but no further than the justified checkpoint (confirmed by definition).
    # If justified checkpoint is outside the block_roots window, return it directly.
    justified_slot = state.justified_checkpoint.slot
    if justified_slot < state.slot - SLOTS_PER_HISTORICAL_ROOT + 1:
        return state.justified_checkpoint
    root = get_block_root_at_slot(state, slot)
    while slot > justified_slot and get_block_root_at_slot(state, Slot(slot - 1)) == root:
        slot = Slot(slot - 1)
    if slot <= justified_slot:
        return state.justified_checkpoint
    return Checkpoint(slot=slot, root=root)
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
    set_or_append_list(state.current_height_target_slots, index, FAR_FUTURE_SLOT)
    set_or_append_list(state.current_height_timeout_participation, index, False)
    set_or_append_list(state.finalize_participation, index, False)
    set_or_append_list(state.previous_height_target_participation, index, False)
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(state: BeaconState) -> None:
    """
    Advance to the next height and rotate attestation tracking.
    Current-height target fields rotate to previous-height for late processing
    and leak scoring. Previous-height data is kept for one height (same as
    Gasper keeping previous_epoch_participation). The ``finalize_participation``
    bitlist is NOT rotated — it spans the extended
    finalization window and persists until the justified checkpoint changes.
    """
    # Rotate current → previous
    state.previous_height_canonical_target = state.current_height_canonical_target
    # [Modified in Simplex] Derive target participation bitlist from target_slots
    canonical_slot = state.current_height_canonical_target.slot
    state.previous_height_target_participation = [
        state.current_height_target_slots[i] == canonical_slot
        for i in range(len(state.validators))
    ]

    # Advance height
    state.current_height = Height(state.current_height + 1)

    # [Modified in Simplex] Set canonical target for the new height.
    # Always use the latest block — the leak uses majority target (ex post) for
    # penalty purposes, not the canonical target. No k-deep confirmation needed.
    state.current_height_canonical_target = Checkpoint(
        slot=state.latest_block_header.slot,
        root=hash_tree_root(state.latest_block_header),
    )

    # Reset current height attestation tracking (extended-window bitlists are NOT reset here)
    num_validators = len(state.validators)
    state.current_height_target_slots = [FAR_FUTURE_SLOT for _ in range(num_validators)]  # [Modified in Simplex]
    state.current_height_timeout_participation = [False for _ in range(num_validators)]
```

#### New `compute_round_outcome`

```python
def compute_round_outcome(
    state: BeaconState,
) -> Tuple[bool, bool, bool, Slot, bool]:
    """
    [New in Simplex] Pre-compute the round outcome from current state without mutations.
    Returns ``(has_height_progress, has_pending_finalization, has_justification,
    majority_target_slot, majority_has_threshold)``.
    Called independently by each round-processing function so they can operate
    without parameter coupling.

    **Height advance rule**: three cases:

    1. **Justification**: target T alone reaches >= 2/3 (non-slashed T-voters).
    2. **Timeout-assisted advance**: T-voters + timeout >= 2/3, AND T has >= 1/3
       non-slashed weight (minimum target support). The 1/3 threshold prevents a
       < n/3 adversary from triggering advance with an arbitrary target.
    3. **Pure timeout**: timeout weight alone >= 2/3. No target needed.

    Slashed validators' target votes are excluded from ``slot_weights`` and their
    weight counts as timeout instead (slashed-to-timeout conversion). This ensures
    slashed validators cannot artificially inflate a target's justification weight
    while contributing only to the weaker advance condition.

    ``majority_target_slot`` is the slot of the target with the highest non-slashed
    weight in ``slot_weights``, used by ``compute_leak_penalty_units`` for both
    Layer 2 (target check) and Layer 1 (conditional exemption). Falls back to
    ``canonical_target.slot`` if no target votes exist.

    ``majority_has_threshold`` indicates whether the majority target has >= 1/3
    non-slashed weight, used by Layer 1 for conditional exemption.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return False, False, False, FAR_FUTURE_SLOT, False

    total = get_total_active_balance(state)
    active = get_active_validator_indices(state, get_current_epoch(state))

    # Finalization pending (resolved if finalize quorum reached this round)
    has_pending_finalization = state.finalized_checkpoint != state.justified_checkpoint
    if state.current_height > GENESIS_HEIGHT and has_pending_finalization:
        finalize_weight = Gwei(sum(
            state.validators[i].effective_balance
            for i in active if state.finalize_participation[i]
        ))
        if finalize_weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR:
            has_pending_finalization = False

    # [Modified in Simplex] Height advance: slashed-to-timeout conversion.
    # Slashed validators count as timeout regardless of their target vote.
    # This does NOT modify target_slots — only the weight computation.
    timeout_weight = Gwei(0)
    slot_weights: Dict[Slot, Gwei] = {}
    for i in active:
        ts = state.current_height_target_slots[i]
        if state.validators[i].slashed:
            # Slashed validators count as timeout regardless of their target vote
            timeout_weight += state.validators[i].effective_balance
        elif state.current_height_timeout_participation[i]:
            timeout_weight += state.validators[i].effective_balance
        elif ts != FAR_FUTURE_SLOT:
            # Non-slashed, non-timeout, has a target vote
            slot_weights[ts] = Gwei(slot_weights.get(ts, Gwei(0)) + state.validators[i].effective_balance)

    # Justification: any single on-chain target with >= 2/3 (non-slashed T-voters alone)
    has_justification = any(
        w * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR
        for w in slot_weights.values()
    )

    # Timeout-assisted advance: T-voters + timeout >= 2/3, with T >= 1/3 (minimum target support)
    # [Modified in Simplex] The 1/3 threshold prevents a < n/3 adversary from triggering
    # advance with an arbitrary target (their epsilon vote + honest timeout).
    has_height_progress = False
    for w in slot_weights.values():
        if w * 3 >= total:  # minimum target support: w >= total/3
            if (w + timeout_weight) * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR:
                has_height_progress = True
                break
    # Pure timeout also advances (no target needed, no threshold)
    if timeout_weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR:
        has_height_progress = True

    # Majority target: slot with the highest non-slashed weight (for leak Layer 1 and Layer 2).
    # When no target has any non-slashed votes, fall back to the canonical target.
    majority_target_slot = (
        max(slot_weights, key=lambda s: slot_weights[s]) if slot_weights
        else state.current_height_canonical_target.slot
    )

    # Whether majority target meets the 1/3 minimum support threshold (for Layer 1 conditional exemption)
    majority_has_threshold = (
        slot_weights.get(majority_target_slot, Gwei(0)) * 3 >= total
    ) if slot_weights else False

    return has_height_progress, has_pending_finalization, has_justification, majority_target_slot, majority_has_threshold
```

#### Modified `process_justification_and_finalization`

*Note*: Simplex finality uses 2/3 quorums. Heights advance in three ways:
(1) **justification** — target T alone reaches >= 2/3, (2) **timeout-assisted
advance** — T + timeout >= 2/3 with T >= 1/3, or (3) **pure timeout** — timeout
alone >= 2/3. Justification is the strongest outcome (updates
`justified_checkpoint`). Timeout-assisted advance updates `justification_floor_slot`
to prevent fork-choice regression past the advance target.

Finalization uses an **extended window**: justify at height H, then confirm via
piggybacked finalize votes at any subsequent height. The `finalize_participation`
bitlist accumulates across heights until finalization occurs or a new checkpoint
is justified (which resets it). This prevents the adversary from stranding
justified checkpoints by alternating justification and timeout rounds. On
justification resets `finalize_participation`.

Justification takes priority over timeout-assisted advance. If a target reaches
2/3 alone (justification), the height advances with a `justified_checkpoint`
update. If no single target reaches 2/3 but the timeout-assisted condition
(T >= 1/3, T + timeout >= 2/3) is met, the height advances without justifying
and `justification_floor_slot` is updated.

Both previous height (late justification from late-arriving votes) and current
height are processed. Finalization is pending when `finalized_checkpoint !=
justified_checkpoint` — no separate `pending_finalization_target` field is
needed since there can only be one justified target per height under f < n/3.

```python
def process_justification_and_finalization(state: BeaconState) -> None:
    """
    [Modified in Simplex] Apply finalization, justification, and height advance at round boundary.
    Uses ``compute_round_outcome`` to determine which actions to take, then mutates state.
    Must run AFTER ``process_inactivity_updates`` and ``process_rewards_and_penalties``
    so that those functions see ``finalize_participation`` before this function
    resets it on new justification.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    has_height_progress, _, has_justification, _, _ = compute_round_outcome(state)
    total = get_total_active_balance(state)
    active = get_active_validator_indices(state, get_current_epoch(state))

    # --- Finalization (extended window): justified checkpoint not yet finalized ---
    if state.current_height > GENESIS_HEIGHT and state.finalized_checkpoint != state.justified_checkpoint:
        finalize_weight = Gwei(sum(
            state.validators[i].effective_balance
            for i in active if state.finalize_participation[i]
        ))
        if finalize_weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR:
            state.finalized_checkpoint = state.justified_checkpoint

    # --- Late justification of previous height ---
    # (late-arriving votes may push the previous height's target past 2/3)
    if state.current_height > GENESIS_HEIGHT and state.justified_height < get_previous_height(state):
        previous_target_weight = Gwei(sum(
            state.validators[i].effective_balance
            for i in active if state.previous_height_target_participation[i]
        ))
        # [Modified in Simplex] Slot monotonicity: only accept if it advances justified_checkpoint.slot
        if (
            previous_target_weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR
            and state.previous_height_canonical_target.slot > state.justified_checkpoint.slot
        ):
            state.justified_checkpoint = state.previous_height_canonical_target
            state.justified_height = get_previous_height(state)
            # Reset extended-window tracking for new justified checkpoint
            state.finalize_participation = [False for _ in range(len(state.validators))]

    # --- Current height: justification or timeout-assisted/pure-timeout advance ---
    # Justification takes priority (updates justified_checkpoint).
    # Timeout-assisted advance updates justification_floor_slot but does not justify.
    if has_justification:
        # [Modified in Simplex] Find the justified target: non-slashed T-voters alone >= 2/3
        slot_weights: Dict[Slot, Gwei] = {}
        for i in active:
            ts = state.current_height_target_slots[i]
            if ts != FAR_FUTURE_SLOT and not state.validators[i].slashed:
                slot_weights[ts] = Gwei(slot_weights.get(ts, Gwei(0)) + state.validators[i].effective_balance)
        for justified_slot, weight in slot_weights.items():
            if weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR:
                # [Modified in Simplex] Slot monotonicity: only update checkpoint if slot advances
                if justified_slot > state.justified_checkpoint.slot:
                    state.justified_checkpoint = Checkpoint(
                        slot=justified_slot,
                        root=get_block_root_at_slot(state, justified_slot),
                    )
                    state.justified_height = state.current_height
                    # Reset extended-window tracking for new justified checkpoint
                    state.finalize_participation = [False for _ in range(len(state.validators))]
                break
        advance_height(state)
    elif has_height_progress:
        # [Modified in Simplex] Update justification_floor_slot for timeout-assisted advance.
        # Find the advance target (highest-weight target meeting the minimum support threshold).
        # This prevents the fork choice from regressing past this slot.
        # Recompute slot_weights and timeout_weight matching compute_round_outcome's semantics.
        slot_weights: Dict[Slot, Gwei] = {}
        timeout_weight = Gwei(0)
        for i in active:
            ts = state.current_height_target_slots[i]
            if state.validators[i].slashed:
                timeout_weight += state.validators[i].effective_balance
            elif state.current_height_timeout_participation[i]:
                timeout_weight += state.validators[i].effective_balance
            elif ts != FAR_FUTURE_SLOT:
                slot_weights[ts] = Gwei(slot_weights.get(ts, Gwei(0)) + state.validators[i].effective_balance)
        for advance_slot in sorted(slot_weights, key=lambda s: slot_weights[s], reverse=True):
            w = slot_weights[advance_slot]
            if (w * 3 >= total  # minimum target support: >= 1/3
                    and (w + timeout_weight) * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR):
                state.justification_floor_slot = max(state.justification_floor_slot, advance_slot)
                break
        advance_height(state)
```

#### Modified `process_inactivity_updates`

*Note*: Inactivity scoring uses a **two-layer design** conditioned on whether
the height is completing this round (see `compute_leak_penalty_units`):

- **Height completing (Layer 2)**: Penalize validators that did NOT vote for the
  **majority target** (the on-chain target with the highest non-slashed weight).
  Uses the ex-post majority target rather than the ex-ante canonical target,
  directly penalizing lack of coordination without requiring a k-deep
  confirmation mechanism. When finalization is pending
  (`finalized_checkpoint != justified_checkpoint`), also require finalize
  confirmation from the extended-window `finalize_participation` bitlist.
  The finalize penalty is applied **once per justified checkpoint**: it only
  fires at the first height after justification (`current_height ==
  justified_height + 1`), so validators are penalized at most once. This
  bounds the unfairness for validators who genuinely cannot finalize (e.g.,
  voted for a different target at the justified height — slashing condition 2
  prevents them from finalizing). The target check independently maintains the
  tight property: **either justification occurs, or at least 1/3 of total
  stake is being leaked**.
- **Height stalled (Layer 1)**: Exempt if voted explicit timeout, OR voted for
  the **majority target** when the majority target has >= 1/3 non-slashed
  weight. The conditional exemption mirrors the minimum target support threshold
  on the advance condition: if the majority target has < 1/3 support, only
  timeout voters are exempt. This preserves the tight 1/3 leak bound: during
  a stall, the exempt set is always < 2/3 (either the advance condition would
  have fired, or the exempt set is bounded).

Both layers have a tight bound of 1/3 with no dead zone (compared to 1/5 and
1/10 in one-round finality). The liveness argument is purely local: on any
given chain, the leak drives the advance condition (majority target ∪ timeout)
to 2/3, giving progress on that chain — no global fork-choice reasoning about
justification on other chains is needed.

```python
def process_inactivity_updates(state: BeaconState) -> None:
    # Skip early epochs — aligned with compute_round_outcome's guard
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    has_height_progress, has_pending_finalization, _, majority_target_slot, majority_has_threshold = compute_round_outcome(state)
    for index in get_eligible_validator_indices(state):
        # [Modified in Simplex] Stacking penalties: target and finalize checks are independent
        penalty_units = compute_leak_penalty_units(
            state, ValidatorIndex(index), has_height_progress, has_pending_finalization,
            majority_target_slot, majority_has_threshold,
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
    [Modified in Simplex] Two-layer leak: majority-target when height completing,
    majority-target/timeout when stalled. Scaled by 1/ROUNDS_PER_EPOCH**2 per round (one factor
    for per-round application, one for scores accumulating ROUNDS_PER_EPOCH times faster).
    """
    has_height_progress, has_pending_finalization, _, majority_target_slot, majority_has_threshold = compute_round_outcome(state)
    rewards = [Gwei(0) for _ in range(len(state.validators))]
    penalties = [Gwei(0) for _ in range(len(state.validators))]
    for index in get_eligible_validator_indices(state):
        # [Modified in Simplex] Stacking penalties: penalty scaled by number of failed checks
        penalty_units = compute_leak_penalty_units(
            state, ValidatorIndex(index), has_height_progress, has_pending_finalization,
            majority_target_slot, majority_has_threshold,
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

#### Modified `process_rewards_and_penalties`

```python
def process_rewards_and_penalties(state: BeaconState) -> None:
    """
    [Modified in Simplex] Uses ``compute_round_outcome`` via ``get_inactivity_penalty_deltas``
    for the two-layer leak design.
    """
    # No rewards are applied at the end of `GENESIS_EPOCH` because rewards are for work done in the previous round
    if get_current_epoch(state) == GENESIS_EPOCH:
        return

    flag_deltas = [
        get_flag_index_deltas(state, flag_index)
        for flag_index in range(len(PARTICIPATION_FLAG_WEIGHTS))
    ]
    deltas = flag_deltas + [get_inactivity_penalty_deltas(state)]
    for rewards, penalties in deltas:
        for index in range(len(state.validators)):
            increase_balance(state, ValidatorIndex(index), rewards[index])
            decrease_balance(state, ValidatorIndex(index), penalties[index])
```

#### Modified `process_slashings`

```python
def process_slashings(state: BeaconState) -> None:
    epoch = get_current_epoch(state)
    total_balance = get_total_active_balance(state)
    # [Modified in Simplex] Standard multiplier 3 (1/3 accountable safety, same as FFG)
    adjusted_total_slashing_balance = min(
        sum(state.slashings) * PROPORTIONAL_SLASHING_MULTIPLIER_SIMPLEX, total_balance
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
    that run every SLOTS_PER_ROUND slots. Epoch boundaries are always round boundaries,
    so process_round runs before process_epoch at epoch transitions.
    Inactivity updates run before justification and finalization so that they see
    ``finalize_participation`` before J&F resets it on new justification.
    Each function calls ``compute_round_outcome`` independently.
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

#### Modified `process_attestation`

*Note*: Votes are recorded in `current_height_target_slots` (per-validator
slot). Since only on-chain targets are tracked, the slot uniquely identifies the
target block on this chain. Any on-chain target with >= 2/3 is justified —
canonical and non-canonical targets are treated uniformly for justification.
The canonical target retains special status only for rewards (TIMELY_TARGET)
and leak scoring (Layer 2 exemption). Non-canonical target votes solve the
catch-up problem: if a target is justified on another chain, any chain
containing that target block can advance past its height by reprocessing the
justification votes as non-canonical target votes — and with per-validator
slot tracking, the state can now justify the specific target directly.

The round window is relaxed: the height check (`data.height`) is the binding
constraint. Old-round attestations are accepted for finality vote recording
(target slot, timeout, and finalize participation) but do not earn
TIMELY_TARGET rewards (only current-round and previous-round attestations
earn rewards via `round_participation`).

The finalize vote uses `finalize_target` (a specific `Checkpoint`) paired with
`finalize_height` (the height at which the target was justified). This binds the
finalization commitment to a specific justified checkpoint rather than just a
height. The extended Condition 2 slashing condition uses both fields: signing
`finalize_target = T` while having voted for something other than T at
`finalize_height` is slashable.

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in Simplex] Records finality attestations using ``current_height_target_slots``.
    First on-chain target vote wins — no overwrites. Timeout tracked independently.
    TIMELY_TARGET reward only for canonical target at the current round.
    """
    data = attestation.data

    # Validate slot and height
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    assert data.height in (state.current_height, get_previous_height(state))
    # [New in Simplex] Finalize piggyback validation:
    # Either no finalize vote (sentinel) or well-formed (target + height pair)
    if data.finalize_target == Checkpoint():
        assert data.finalize_height == FAR_FUTURE_HEIGHT
    else:
        assert data.finalize_height < data.height

    # [Modified in Simplex] Height-based acceptance: the height check is the binding
    # constraint. No round window check — old-round attestations are accepted for
    # finality vote recording. TIMELY_TARGET rewards are only earned for current-round
    # attestations (checked below via round_participation).
    attestation_round = compute_round_at_slot(data.slot)

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

    # Determine round participation list for TIMELY_TARGET rewards
    # [Modified in Simplex] Only current-round attestations earn TIMELY_TARGET
    is_current_round = attestation_round == get_current_round(state)
    if is_current_round:
        round_participation = state.current_round_participation
    elif attestation_round == get_previous_round(state):
        round_participation = state.previous_round_participation
    else:
        round_participation = None  # Old-round: no reward participation

    # Determine which height this attestation is for
    is_current_height = data.height == state.current_height
    if is_current_height:
        target_slots = state.current_height_target_slots
        canonical_target = state.current_height_canonical_target
    else:
        canonical_target = state.previous_height_canonical_target

    proposer_reward_numerator = 0
    current_epoch = get_current_epoch(state)
    is_canonical_target = data.target == canonical_target
    is_timeout = data.target == Checkpoint()
    # [Modified in Simplex] Non-canonical target: a non-empty target whose block exists on
    # this chain but differs from the canonical target.
    is_non_canonical_target = (
        is_current_height
        and not is_canonical_target
        and not is_timeout
        and is_target_on_chain(state, data.target, attestation.historical_block_proof)
    )

    attesting_indices = get_attesting_indices(state, attestation)
    for validator_index in attesting_indices:
        validator = state.validators[validator_index]
        if not is_active_validator(validator, current_epoch):
            continue

        if is_current_height:
            current_slot = target_slots[validator_index]
            # [Modified in Simplex] First on-chain target vote wins. Target and timeout are independent.
            if (is_canonical_target or is_non_canonical_target) and current_slot == FAR_FUTURE_SLOT:
                target_slots[validator_index] = data.target.slot
                # TIMELY_TARGET reward for canonical target only (current/previous round)
                if is_canonical_target and round_participation is not None:
                    if not has_flag(round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX):
                        round_participation[validator_index] = add_flag(
                            round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
                        )
                        proposer_reward_numerator += (
                            get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT
                        )

            # Timeout is tracked independently (set once, never cleared within a height)
            if is_timeout and not state.current_height_timeout_participation[validator_index]:
                state.current_height_timeout_participation[validator_index] = True
        else:
            # Previous height: only canonical target votes update previous_height_target_participation
            if is_canonical_target and not state.previous_height_target_participation[validator_index]:
                state.previous_height_target_participation[validator_index] = True
                # TIMELY_TARGET reward for canonical target (current/previous round)
                if round_participation is not None:
                    if not has_flag(round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX):
                        round_participation[validator_index] = add_flag(
                            round_participation[validator_index], TIMELY_TARGET_FLAG_INDEX
                        )
                        proposer_reward_numerator += (
                            get_base_reward(state, validator_index) * TIMELY_TARGET_WEIGHT
                        )

        # Process finalize vote (extended window — accepted at any height)
        # [Modified in Simplex] Checks finalize_target matches justified_checkpoint
        # AND finalize_height matches justified_height.
        if (
            data.finalize_target == state.justified_checkpoint
            and data.finalize_height == state.justified_height
            and state.finalized_checkpoint != state.justified_checkpoint
            and not state.finalize_participation[validator_index]
        ):
            state.finalize_participation[validator_index] = True

    # *Note*: Proposer rewards are only earned for canonical-target attestations.
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

*Note*: The canonical target uses the actual latest block (not an epoch-boundary
block) to ensure the target slot is the block's actual proposal slot, consistent
with `advance_height`.

```python
def upgrade_to_simplex(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)
    if epoch > GENESIS_EPOCH:
        canonical_target_slot = pre.latest_block_header.slot
        canonical_target_root = hash_tree_root(pre.latest_block_header)
    else:
        canonical_target_slot = GENESIS_SLOT
        canonical_target_root = Root()

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
        justified_checkpoint=Checkpoint(
            slot=compute_start_slot_at_epoch(pre.current_justified_checkpoint.epoch),
            root=pre.current_justified_checkpoint.root,
        ),
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
        justification_floor_slot=GENESIS_SLOT,
        current_height_canonical_target=Checkpoint(
            slot=canonical_target_slot,
            root=canonical_target_root,
        ),
        current_height_target_slots=[FAR_FUTURE_SLOT for _ in range(len(pre.validators))],
        current_height_timeout_participation=[False for _ in range(len(pre.validators))],
        finalize_participation=[False for _ in range(len(pre.validators))],
        previous_height_canonical_target=Checkpoint(
            slot=canonical_target_slot,
            root=canonical_target_root,
        ),
        previous_height_target_participation=[False for _ in range(len(pre.validators))],
    )

    return post
```

## Genesis

### Modified `initialize_beacon_state_from_eth1`

*Note*: The `current_height_canonical_target` uses a zero root at genesis since
no block exists yet. The `epoch <= GENESIS_EPOCH + 1` guard in
`process_justification_and_finalization` prevents finality processing in the
first two epochs, so this zero root is never used for on-chain verification.

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
    state.justification_floor_slot = GENESIS_SLOT
    state.current_height_canonical_target = Checkpoint(slot=GENESIS_SLOT, root=Root())
    state.previous_height_canonical_target = Checkpoint(slot=GENESIS_SLOT, root=Root())

    return state
```

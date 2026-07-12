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
    - [New `get_slots_per_round_at_slot`](#new-get_slots_per_round_at_slot)
    - [New `get_rounds_per_epoch_at_slot`](#new-get_rounds_per_epoch_at_slot)
  - [Predicates](#predicates)
    - [New `compute_leak_penalty_units`](#new-compute_leak_penalty_units)
    - [Modified `is_slashable_attestation_data`](#modified-is_slashable_attestation_data)
    - [Modified `is_eligible_for_activation`](#modified-is_eligible_for_activation)
    - [Modified `is_active_builder`](#modified-is_active_builder)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_current_round`](#new-get_current_round)
    - [New `get_previous_round`](#new-get_previous_round)
    - [New `get_target_slot_weights`](#new-get_target_slot_weights)
    - [Modified `get_finality_delay`](#modified-get_finality_delay)
    - [Modified `get_unslashed_participating_indices`](#modified-get_unslashed_participating_indices)
    - [New `is_target_on_chain`](#new-is_target_on_chain)
    - [New `verify_historical_block_proof`](#new-verify_historical_block_proof)
    - [New `is_timeout_vote`](#new-is_timeout_vote)
    - [New `is_empty_vote`](#new-is_empty_vote)
    - [New `is_nonjustifiable_height`](#new-is_nonjustifiable_height)
    - [New `is_viable_attestation_target`](#new-is_viable_attestation_target)
    - [New `get_available_committee`](#new-get_available_committee)
    - [Modified `get_committee_count_per_slot`](#modified-get_committee_count_per_slot)
    - [Modified `get_beacon_committee`](#modified-get_beacon_committee)
  - [Available attestation helpers](#available-attestation-helpers)
    - [New `get_available_attesting_positions`](#new-get_available_attesting_positions)
    - [New `get_available_attesting_indices`](#new-get_available_attesting_indices)
  - [Modified helpers](#modified-helpers)
    - [Modified `add_validator_to_registry`](#modified-add_validator_to_registry)
- [Beacon chain state transition function](#beacon-chain-state-transition-function)
  - [Epoch processing](#epoch-processing)
    - [New `advance_height`](#new-advance_height)
    - [New `compute_justified_checkpoint`](#new-compute_justified_checkpoint)
    - [New `has_timeout_quorum`](#new-has_timeout_quorum)
    - [New `has_new_finalization`](#new-has_new_finalization)
    - [New `compute_best_justification_target`](#new-compute_best_justification_target)
    - [Modified `process_justification_and_finalization`](#modified-process_justification_and_finalization)
    - [Modified `process_inactivity_updates`](#modified-process_inactivity_updates)
    - [Modified `get_flag_index_deltas`](#modified-get_flag_index_deltas)
    - [Modified `get_inactivity_penalty_deltas`](#modified-get_inactivity_penalty_deltas)
    - [Modified `process_pending_deposits`](#modified-process_pending_deposits)
    - [Modified `process_participation_flag_updates`](#modified-process_participation_flag_updates)
    - [Modified `process_rewards_and_penalties`](#modified-process_rewards_and_penalties)
    - [New `process_inactivity_penalties`](#new-process_inactivity_penalties)
    - [New `process_round`](#new-process_round)
    - [Modified `process_epoch`](#modified-process_epoch)
    - [Modified `process_slots`](#modified-process_slots)
  - [Block processing](#block-processing)
    - [Modified `is_valid_indexed_attestation`](#modified-is_valid_indexed_attestation)
    - [New `validate_attestation`](#new-validate_attestation)
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
Casper FFG with a fresh-simplex-with-height-filter-and-timeouts finality gadget.
The model is n >= 3f+1, with 2/3 quorums for justification, timeout cert, and
finalization. Each validator casts at most one **justify** (R1) and one
**timeout** (R2) attestation per state-height; the justify vote is subject to a
**fresh-vote** gate that keys it to the current height's interval on the current
chain. A timeout vote is encoded as `target == Checkpoint()` at a real height
(`target == Checkpoint()` at `height == Height(0)` is instead the empty vote,
introduced below). Finalization takes two steps: justify at height H, then
confirm via piggybacked finality votes at any subsequent height (extended
finalization window). The fork-choice store maintains a single justification
root and a height-filter (viable subtree).

Three vote kinds share the attestation format: a justification vote (a real
`target` at the current state-height), a timeout vote (`target == Checkpoint()`
at a real height), and the **empty vote** (`target == Checkpoint()` at
`height == Height(0)`), which makes no height claim and acts only through its
head field — the on-chain record layer of the fork choice — and its finality
piggyback. Under sustained non-finality (*finality debt*), every
`K_NONJUSTIFIABLE`-th height is **nonjustifiable**: the justification branch is
disabled there and the height advances only by timeout cert. A round-start
proposal may carry `body.anchor_quorum`, a fresh-quorum reference consumed by
the fork choice that designates the round's common walk anchor. The gates that
choose among the vote kinds are specified in the
[validator document](./validator.md); the record layer, the walk, and safe
confirmation in the [fork-choice document](./fork-choice.md).

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Advance only at round boundaries

At each round boundary, the height may advance via one of two mechanisms:
**justify** (some target T reaches a 2/3 quorum on `justification_targets`) or
**timeout cert** (a 2/3 quorum on the per-validator `timeouts` bitlist).
Finality is separate: `F ← J` fires whenever the finality participation bitlist
reaches 2/3; this does NOT advance height.

### Thresholds (n >= 3f+1)

| Threshold          | Stake  | Purpose                                                            |
| ------------------ | ------ | ------------------------------------------------------------------ |
| Justification      | >= 2/3 | Per-target quorum on `justification_targets` (R1 votes)            |
| Timeout cert       | >= 2/3 | Quorum on `timeouts` bitlist (timeout votes or fresh justify R1's) |
| Finalization       | >= 2/3 | Piggybacked confirm of justified checkpoint                        |
| Accountable safety | 1/3    | Standard BFT (single slashing condition E1)                        |

### Decoupled Consensus

Finality and LMD-GHOST use different attestation types:

- **Attestations**: All active validators attest once per round via standard
  beacon committee attestations (Electra format). `AttestationData` carries a
  finality target (`Checkpoint()` for timeout votes), height, finality piggyback
  target, and finality height. These determine justification, timeout cert, and
  finalization. Attester slashings enforce the finality-target conflict
  condition (E1 only).
- **Available attestations**: A small 512-member available committee attests per
  slot for fork choice via `AvailableAttestation`. This committee is selected
  from the full active set using `compute_balance_weighted_selection` (same
  mechanism as PTC).

### Attestation Tracking

Finality attestations are tracked per validator with one slot list and one
bitlist:

- `justification_targets[i]`: the slot of validator `i`'s last (fresh)
  **justify** vote this height, or `FAR_FUTURE_SLOT` if none.
- `timeouts[i]`: a bit set when validator `i` cast either a timeout vote
  (`target == Checkpoint()`) at this height or a height-fresh justification vote
  on this chain.

Both are reset on height advance. Since only on-chain targets (verified by
`is_target_on_chain`) that already existed before the including block can update
`justification_targets`, the slot uniquely identifies the target block - the
root is recoverable via `get_block_root_at_slot` when needed. The justification
branch uses per-target counting on `justification_targets` (highest slot where a
2/3 quorum exists); the timeout branch checks whether the `timeouts` bitlist
holds a 2/3 quorum.

A separate **finality participation** bitlist tracks finalization confirmations
across the extended window. It persists until the justified checkpoint changes,
at which point it resets.

No previous-height data is retained: stale votes (height below
`state.current_height`) do not update target tracking but may still carry valid
finality piggybacks.

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
The `SLOTS_PER_ROUND` in each entry MUST divide `SLOTS_PER_EPOCH`, and each
entry's activation slot MUST be a multiple of `SLOTS_PER_EPOCH` (epoch-aligned).
Together these ensure the round length is constant within any epoch and every
epoch boundary is a round boundary -- relied on by `get_beacon_committee`
(`slot_in_round` vs the epoch-keyed committee `count`) and the height/round
bookkeeping. The round schedule entries SHOULD be sorted by slot in ascending
order.

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
| `GENESIS_HEIGHT`              | `Height(1)`         |
| `FAR_FUTURE_HEIGHT`           | `Height(2**64 - 1)` |
| `GENESIS_ROUND`               | `Round(0)`          |
| `FINALITY_QUORUM_NUMERATOR`   | `uint64(2)`         |
| `FINALITY_QUORUM_DENOMINATOR` | `uint64(3)`         |
| `K_NONJUSTIFIABLE`            | `uint64(8)`         |
| `FINALITY_DEBT_THRESHOLD`     | `uint64(2)`         |

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

### Max operations per block

*Note*: `MAX_ANCHOR_QUORUM_ATTESTATIONS` bounds the fresh-quorum reference a
round-start proposal may carry (see [`BeaconBlockBody`](#beaconblockbody)). It
accommodates a full round of maximally packed on-chain aggregates
(`MAX_ATTESTATIONS_ELECTRA` per slot times the 32-slot mainnet round); when head
fields are fragmented across many distinct votes, more aggregates are needed to
reach the two-thirds weight, so this bound is flagged for review.

| Name                             | Value                  |
| -------------------------------- | ---------------------- |
| `MAX_AVAILABLE_ATTESTATIONS`     | `uint64(8)`            |
| `MAX_ROUND_DOUBLE_VOTE_EVIDENCE` | `uint64(1)`            |
| `MAX_ANCHOR_QUORUM_ATTESTATIONS` | `uint64(2**8)` (= 256) |

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
    # [Modified in Simplex]
    slot: Slot  # was epoch: Epoch
    root: Root
```

#### `AttestationData`

*Note*: The `source` and `index` fields are removed. `beacon_block_root` is
repurposed as an LMD head vote for fork choice (set to the voter's head).
`target` is repurposed as a simplex finality target. A vote with
`target == Checkpoint()` at a real height is a **timeout vote** (R2); with
`target == Checkpoint()` at `height == Height(0)` it is the **empty vote**,
which makes no height claim; a vote with non-empty target is a **justification
vote** (R1). `height` carries the state-height at which the vote is cast.
`finality_target` is a piggyback vote specifying which justified checkpoint to
confirm (`Checkpoint()` means no finality vote); `finality_height` is the height
at which `finality_target` was justified (`FAR_FUTURE_HEIGHT` when no finality
vote). The `beacon_block_root` field is used by the fork choice only —
`process_attestation` uses `target`, `height`, `finality_target`, and
`finality_height`. A finality vote is an LMD vote for a beacon block at
`PAYLOAD_STATUS_PENDING`: it stabilizes the voted block and the payloads already
in its chain, but makes no decision on the payload at the tip — that is left to
the available-attestation / Goldfish layer. Hence there is no `payload_present`
field.

```python
class AttestationData(Container):
    slot: Slot
    # [Modified in Simplex]
    beacon_block_root: Root  # LMD head vote for fork choice
    # [Modified in Simplex]
    # Justification target, or Checkpoint() for timeout vote
    target: Checkpoint
    # [New in Simplex]
    height: Height  # Finality height being attested to
    # [New in Simplex]
    # Finalize commitment target, or Checkpoint() for none
    finality_target: Checkpoint
    # [New in Simplex]
    # Height at which finality_target was justified, or FAR_FUTURE_HEIGHT
    finality_height: Height
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
    # [New in Simplex]
    # Empty = no proof; one element = proof for an out-of-window target.
    # (SSZ has no Optional; a length-1 List encodes the optional proof.)
    historical_block_proof: List[HistoricalBlockProof, 1]
```

#### `BeaconBlockBody`

*Note*: `anchor_quorum` is the fresh-quorum reference of the record/anchor layer
(paper Definition: round-r quorum, fresh quorum, and anchor; the "Pointing"
rule). A round-start (first-slot-of-round) proposal MAY use it to point to one
fresh quorum of the previous round: a set of previous-round finality
attestations, as standard aggregates, from distinct validators whose effective
balances sum to at least two-thirds of the **total** active balance (an absolute
threshold), all of whose head fields lie in one subtree. The deepest block whose
subtree contains every head field — their highest common ancestor — is the
quorum's *anchor*, adopted by every validator for the whole round as the
starting point of the fork-choice walk (fork-choice `update_pointed_anchor`).

The field is a threshold certificate, not an operation: `process_operations`
does not process it, it creates no on-chain records, and its verification (in
the fork choice) does not require its attestations to be included as records —
the aggregates riding the proposal make them available on their own (paper
lem:anchor-support). An invalid or misplaced reference never invalidates the
block; it is simply ignored, and the round proceeds without a pointed anchor.

```python
class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data
    graffiti: Bytes32
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS_ELECTRA]
    # [Modified in Simplex]
    attestations: List[Attestation, MAX_ATTESTATIONS_ELECTRA]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    signed_execution_payload_bid: SignedExecutionPayloadBid
    payload_attestations: List[PayloadAttestation, MAX_PAYLOAD_ATTESTATIONS]
    parent_execution_requests: ExecutionRequests
    # Simplex
    # [New in Simplex]
    available_attestations: List[AvailableAttestation, MAX_AVAILABLE_ATTESTATIONS]
    # [New in Simplex]
    round_double_vote_evidence: List[RoundDoubleVoteEvidence, MAX_ROUND_DOUBLE_VOTE_EVIDENCE]
    # [New in Simplex]
    # Fresh-quorum reference (threshold certificate): previous-round finality
    # attestations designating the round's anchor. Consumed by the fork choice
    # only; creates no records and is not processed by process_operations.
    anchor_quorum: List[Attestation, MAX_ANCHOR_QUORUM_ATTESTATIONS]
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
    # [Modified in Simplex]
    previous_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    # [Modified in Simplex]
    current_round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    # [Modified in Simplex]
    # replaces justification_bits + previous/current_justified
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
    ptc_window: Vector[Vector[ValidatorIndex, PTC_SIZE], (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH]
    # Simplex finality gadget
    # [New in Simplex]
    justified_height: Height  # height of ``justified_checkpoint``
    # [New in Simplex]
    finalized_height: Height  # height of ``finalized_checkpoint``
    # [New in Simplex]
    current_height: Height  # paper's h
    # [New in Simplex]
    # slot at which the current height began (paper's s_h)
    current_height_start_slot: Slot
    # [New in Simplex]
    justification_targets: List[Slot, VALIDATOR_REGISTRY_LIMIT]  # per-validator justify target slot
    # [New in Simplex]
    timeouts: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # paper's timeouts[]
    # [New in Simplex]
    finality_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # extended window
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: See [Attestation Tracking](#attestation-tracking) for field roles. Key
invariants: fresh justification votes update `justification_targets[i]` and set
`timeouts[i]`; timeout votes (target = `Checkpoint()`) at the current height set
`timeouts[i]` only. `finality_participation` persists across height advances and
is reset only when a new justification fires (the `justify_target` branch of
`advance_height`).

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

#### New `get_slots_per_round_at_slot`

```python
def get_slots_per_round_at_slot(slot: Slot) -> uint64:
    """
    Return the number of slots per round in effect at ``slot``, per
    ``ROUND_SCHEDULE``. Slots before the first schedule entry use
    ``SLOTS_PER_EPOCH``.
    """
    slots_per_round = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda entry: entry["SLOT"]):
        if slot < entry["SLOT"]:
            break
        slots_per_round = entry["SLOTS_PER_ROUND"]
    return slots_per_round
```

#### New `get_rounds_per_epoch_at_slot`

```python
def get_rounds_per_epoch_at_slot(slot: Slot) -> uint64:
    """
    Return the number of rounds per epoch in effect at ``slot``.
    """
    return SLOTS_PER_EPOCH // get_slots_per_round_at_slot(slot)
```

### Predicates

#### New `compute_leak_penalty_units`

```python
def compute_leak_penalty_units(
    state: BeaconState,
    index: ValidatorIndex,
    new_height_advance: bool,
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
    # Layer 1 (stall): no height advance and validator did not set the timeout marker.
    if not new_height_advance and not state.timeouts[index]:
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

*Note*: Simplex uses a **single slashing condition** (E1: finality-target
conflict). Validators may cast at most one justify (R1) and one timeout (R2)
vote per state-height; neither kind carries a self-slashing penalty on its own.
The only slashing condition is: if a validator commits to finality target T at
`finality_height = H` (via `finality_target = T`), they must not have voted for
any target other than T at `height = H`. Note that timeout votes
(`target = Checkpoint()`) are themselves in conflict with any commitment
`finality_target = T ≠ Checkpoint()` at the same height, since
`Checkpoint() ≠ T` (paper def:slashing). Conflicting finalizations at the same
height require quorum intersection, and E1 ensures at least 1/3 of validators
are slashable. An empty vote (`height == Height(0)`) pairs into E1 evidence only
against a finality commitment at `finality_height == Height(0)`; no honest
finality commitment at height `0` exists (heights start at
`GENESIS_HEIGHT == Height(1)`), so an honest validator's empty votes never
become slashable evidence — the predicate is deliberately left uniform. Round
double-vote (same round, different data) uses a lighter penalty via
`RoundDoubleVoteEvidence`.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    # [Modified in Simplex]
    # Single slashing condition (E1):
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

#### Modified `is_active_builder`

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
    # [Modified in Simplex]
    # Uses compute_epoch_at_slot for finalized checkpoint.
    # Guard against underflow: outside the inactivity leak, J&F runs at every
    # round boundary, so mid-epoch finalization can place
    # finalized_epoch > previous_epoch.
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

#### New `is_timeout_vote`

```python
def is_timeout_vote(data: AttestationData) -> bool:
    """
    [New in Simplex] A timeout vote has an empty target at a real height (paper
    Definition: vote, ``target = ⊥``). The empty vote (``is_empty_vote``: empty
    target at ``height == 0``) is explicitly NOT a timeout vote and must never
    enter the timeout certificate; excluding it here makes the classification
    robust independent of the downstream height guard.
    """
    return data.target == Checkpoint() and not is_empty_vote(data)
```

#### New `is_empty_vote`

*Note*: The empty vote (paper Definition: empty vote) carries an *empty voted
checkpoint* — both `target == Checkpoint()` and `height == Height(0)` — while
its head field remains populated. It makes no claim about any height, so it sets
no timeout marker and contributes to no justification or timeout certificate;
only its head field (records) and finalize piggyback have effect. Height `0` is
the empty marker: no honest vote is ever cast at height `0`, since the first
real state-height is `GENESIS_HEIGHT == Height(1)`.

```python
def is_empty_vote(data: AttestationData) -> bool:
    """
    [New in Simplex] Return whether ``data`` carries an empty voted checkpoint
    (``target == Checkpoint()`` and ``height == Height(0)``) as an empty vote.
    """
    return data.target == Checkpoint() and data.height == Height(0)
```

#### New `is_nonjustifiable_height`

*Note*: A *nonjustifiable height* (paper Definition: nonjustifiable height) is a
timeout-only height. Under finality debt — the finalized height lagging the
current height by more than `FINALITY_DEBT_THRESHOLD` — every
`K_NONJUSTIFIABLE`-th height is nonjustifiable: `compute_justified_checkpoint`
returns `Checkpoint()` there, so the height can only advance via the timeout
cert. The predicate is a deterministic function of `(height, finalized_height)`,
so honest validators never cast a target vote at such a height (see the
vote-construction gates).

```python
def is_nonjustifiable_height(height: Height, finalized_height: Height) -> bool:
    """
    [New in Simplex] Return whether ``height`` is a nonjustifiable (timeout-only)
    height under finality debt: every ``K_NONJUSTIFIABLE``-th height once the
    finalized height lags by more than ``FINALITY_DEBT_THRESHOLD``.
    """
    return (height > finalized_height + FINALITY_DEBT_THRESHOLD) and (
        height % K_NONJUSTIFIABLE == 0
    )
```

#### New `is_viable_attestation_target`

*Note*: Paper Definition: height freshness. Timeout votes pass the gate on the
height match alone and set `timeouts[i]`; they do not update
`justification_targets[i]`. Only justification votes passing the full gate
update `justification_targets[i]`, and by the gate the recorded slots lie in the
current-height interval and strictly before the including block.

```python
def is_viable_attestation_target(state: BeaconState, attestation: Attestation) -> bool:
    """
    [New in Simplex] Viability gate for current-height tracking. The vote
    must carry the current state-height. Timeout votes are viable on the
    height match alone (they set ``timeouts[i]``); justification votes
    additionally require the target to lie in the current-height interval
    on this chain and name a block that already exists before the including
    block. Non-viable attestations may still affect
    freshness-independent state (e.g. ``finality_participation``).
    """
    data = attestation.data
    if data.height != state.current_height:
        return False
    if is_timeout_vote(data):
        return True
    # The historical proof is a length-1 List (empty = absent).
    historical_proof = (
        attestation.historical_block_proof[0]
        if len(attestation.historical_block_proof) > 0
        else None
    )
    return data.target.slot >= state.current_height_start_slot and is_target_on_chain(
        state, data.target, historical_proof
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
                # [Modified in Simplex]
                # Spread the validator set across the round's slots (per ROUND_SCHEDULE)
                // get_slots_per_round_at_slot(compute_start_slot_at_epoch(epoch))
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
    # [Modified in Simplex]
    # Slot-within-round via round helpers (schedule-safe)
    slot_in_round = slot - compute_start_slot_at_round(compute_round_at_slot(slot))
    return compute_committee(
        indices=get_active_validator_indices(state, epoch),
        seed=get_seed(state, epoch, DOMAIN_BEACON_ATTESTER),
        index=slot_in_round * committees_per_slot + index,
        # [Modified in Simplex]
        # Round length per ROUND_SCHEDULE, keyed off the epoch start like
        # get_committee_count_per_slot. Era starts are epoch-aligned, so the
        # round length is constant within the epoch and equals the length of the
        # round containing ``slot`` (used by ``slot_in_round`` above).
        count=committees_per_slot * get_slots_per_round_at_slot(compute_start_slot_at_epoch(epoch)),
    )
```

### Available attestation helpers

#### New `get_available_attesting_positions`

```python
def get_available_attesting_positions(
    state: BeaconState, attestation: AvailableAttestation
) -> Set[uint64]:
    """
    Return the set of attesting committee positions from an available committee
    attestation. If a validator has duplicate committee seats, any signed vote
    from that validator counts for all of its seats.
    """
    committee = get_available_committee(state, attestation.data.slot)
    assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
    assert len(attestation.aggregation_bits) == len(committee)
    attesting_indices = {
        attester_index
        for i, attester_index in enumerate(committee)
        if attestation.aggregation_bits[i]
    }
    return {
        uint64(i)
        for i, attester_index in enumerate(committee)
        if attester_index in attesting_indices
    }
```

#### New `get_available_attesting_indices`

```python
def get_available_attesting_indices(
    state: BeaconState, attestation: AvailableAttestation
) -> Set[ValidatorIndex]:
    """
    Return the set of attesting indices from an available committee attestation.
    """
    committee = get_available_committee(state, attestation.data.slot)
    return {
        committee[position] for position in get_available_attesting_positions(state, attestation)
    }
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
    set_or_append_list(state.timeouts, index, False)  # noqa: FBT003
    set_or_append_list(state.finality_participation, index, False)  # noqa: FBT003
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(state: BeaconState, justify_target: Optional[Checkpoint] = None) -> None:
    """
    [New in Simplex] Advance ``current_height`` by 1 (paper processHeight
    advance branches). If ``justify_target is not None`` the justify branch
    fired: set ``justified_checkpoint`` / ``justified_height`` and reset
    ``finality_participation``. The timeout branch (``justify_target is None``)
    skips those updates. Either way: bump ``current_height``, set
    ``current_height_start_slot`` to the current slot boundary, reset
    ``justification_targets`` and ``timeouts``.
    """
    if justify_target is not None:
        state.justified_checkpoint = justify_target
        state.justified_height = state.current_height
        state.finality_participation = Bitlist[VALIDATOR_REGISTRY_LIMIT](
            [False] * len(state.validators)
        )
    state.current_height = Height(state.current_height + 1)
    # [New in Simplex]
    # process_round runs at the round's last slot (before process_slots
    # increments state.slot), so the new height begins at the next slot -- the
    # first slot of the new round.
    state.current_height_start_slot = Slot(state.slot + 1)
    num_validators = len(state.validators)
    state.justification_targets = [FAR_FUTURE_SLOT for _ in range(num_validators)]
    state.timeouts = Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * num_validators)
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

    # [New in Simplex]
    # Nonjustifiable heights are timeout-only: the state never produces a
    # justification at a nonjustifiable height, so the height can advance only via
    # the timeout-cert branch.
    if is_nonjustifiable_height(state.current_height, state.finalized_height):
        return Checkpoint()

    best_slot, weight = compute_best_justification_target(state)
    if best_slot == FAR_FUTURE_SLOT:
        return Checkpoint()
    total_active_balance = get_total_active_balance(state)
    if weight * FINALITY_QUORUM_DENOMINATOR < total_active_balance * FINALITY_QUORUM_NUMERATOR:
        return Checkpoint()
    # [New in Simplex]
    # Only justify a target whose root is reconstructible from ``block_roots``.
    # An out-of-window plurality target is not justified; height still advances
    # via the timeout-cert branch, because a viable justification vote also sets
    # ``timeouts[i]`` -- so a justification quorum always implies a timeout
    # quorum. We let the natural flow time out rather than triggering it here.
    in_window = best_slot < state.slot <= best_slot + SLOTS_PER_HISTORICAL_ROOT
    if not in_window:
        return Checkpoint()
    return Checkpoint(slot=best_slot, root=get_block_root_at_slot(state, best_slot))
```

#### New `has_timeout_quorum`

```python
def has_timeout_quorum(state: BeaconState) -> bool:
    """
    [New in Simplex] Return ``True`` iff a 2/3 quorum of non-slashed active
    validators have ``state.timeouts[i] = True`` (paper Definition: Timeout).
    A timeout cert advances height without updating the justified checkpoint.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return False
    total_active_balance = get_total_active_balance(state)
    active_indices = get_active_validator_indices(state, get_current_epoch(state))
    timeout_weight = Gwei(
        sum(
            state.validators[index].effective_balance
            for index in active_indices
            if state.timeouts[index] and not state.validators[index].slashed
        )
    )
    return (
        timeout_weight * FINALITY_QUORUM_DENOMINATOR
        >= total_active_balance * FINALITY_QUORUM_NUMERATOR
    )
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
    excludes them uniformly across all ``justification_targets`` and
    ``timeouts`` tallies (adaptation, not a paper match).
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
pre-advance state. At most one of the justify/timeout branches advances height
per invocation; the finality branch is independent.

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
        # [New in Simplex]
        # finalized_height is updated before the justify branch's
        # nonjustifiable-height check reads it, matching the paper's
        # processHeightEvents branch order (finality first, then
        # justification). The within-round debt reduction is benign: honest
        # validators already withheld targets at a nonjustifiable height, so
        # no justification quorum exists to fire there regardless.
        state.finalized_height = state.justified_height

    # (2) Justify branch
    justified = compute_justified_checkpoint(state)
    if justified != Checkpoint():
        advance_height(state, justify_target=justified)
        return

    # (3) Timeout cert branch
    if has_timeout_quorum(state):
        advance_height(state)
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

    # [Modified in Simplex]
    # Pre-advance signals from paper's three branches.
    # A fresh justification vote also sets ``timeouts[i]``, so a justify quorum
    # implies a timeout quorum on the same chain: ``new_justification ⇒
    # new_height_advance``.
    new_justification = compute_justified_checkpoint(state) != Checkpoint()
    new_height_advance = new_justification or has_timeout_quorum(state)
    new_finalization = has_new_finalization(state)
    if not new_justification:
        best_justification_slot, _ = compute_best_justification_target(state)
    else:
        best_justification_slot = FAR_FUTURE_SLOT

    for index in get_eligible_validator_indices(state):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_height_advance,
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
    [Modified in Simplex] Rewards and penalties are scaled by 1/rounds-per-epoch
    to keep per-epoch totals constant when running per-round.
    """
    rewards = [Gwei(0)] * len(state.validators)
    penalties = [Gwei(0)] * len(state.validators)
    # [Modified in Simplex]
    # Pass previous round instead of previous epoch
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
                # [Modified in Simplex]
                # Scale by 1/ROUNDS_PER_EPOCH (round length per ROUND_SCHEDULE)
                rounds_per_epoch = get_rounds_per_epoch_at_slot(state.slot)
                rewards[index] += Gwei(
                    reward_numerator // (active_increments * WEIGHT_DENOMINATOR * rounds_per_epoch)
                )
        elif flag_index != TIMELY_HEAD_FLAG_INDEX:
            # [Modified in Simplex]
            # Scale by 1/ROUNDS_PER_EPOCH (round length per ROUND_SCHEDULE)
            rounds_per_epoch = get_rounds_per_epoch_at_slot(state.slot)
            penalties[index] += Gwei(
                base_reward * weight // (WEIGHT_DENOMINATOR * rounds_per_epoch)
            )
    return rewards, penalties
```

#### Modified `get_inactivity_penalty_deltas`

```python
def get_inactivity_penalty_deltas(state: BeaconState) -> Tuple[Sequence[Gwei], Sequence[Gwei]]:
    """
    Return the inactivity penalty deltas by considering height participation and inactivity scores.
    [Modified in Simplex] Three-guard leak: a penalty unit accrues for each of
    stall / justification / finalization that did not happen. Applied at the
    leak settlement cadence (once per round normally, once per epoch while the
    inactivity leak is active -- see ``process_round``). The per-epoch base
    magnitude is used directly, with no rounds-per-epoch rescaling: while the
    leak is active the score increments once per epoch and this penalty applies
    once per epoch, matching the base per-epoch leak. (Outside the leak the
    score is ~0 and ``penalty_units`` is ~0, so the per-round application is
    negligible regardless of magnitude.) Up to 3 penalty units.
    """
    rewards = [Gwei(0) for _ in range(len(state.validators))]
    penalties = [Gwei(0) for _ in range(len(state.validators))]

    # [Modified in Simplex]
    # Pre-advance signals from paper's three branches.
    # A fresh justification vote also sets ``timeouts[i]``, so a justify quorum
    # implies a timeout quorum on the same chain: ``new_justification ⇒
    # new_height_advance``.
    new_justification = compute_justified_checkpoint(state) != Checkpoint()
    new_height_advance = new_justification or has_timeout_quorum(state)
    new_finalization = has_new_finalization(state)
    if not new_justification:
        best_justification_slot, _ = compute_best_justification_target(state)
    else:
        best_justification_slot = FAR_FUTURE_SLOT

    for index in get_eligible_validator_indices(state):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_height_advance,
            new_justification,
            new_finalization,
            best_justification_slot,
        )
        if penalty_units > 0:
            penalty_numerator = (
                state.validators[index].effective_balance * state.inactivity_scores[index]
            )
            # [Modified in Simplex]
            # Per-epoch base magnitude (no rounds-per-epoch
            # rescaling): while the leak is active the score increments once per
            # epoch and this penalty applies once per epoch (see process_round).
            penalty_denominator = INACTIVITY_SCORE_BIAS * INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
            penalties[index] += Gwei(penalty_numerator // penalty_denominator * penalty_units)
    return rewards, penalties
```

#### Modified `process_pending_deposits`

```python
def process_pending_deposits(state: BeaconState) -> None:
    next_epoch = Epoch(get_current_epoch(state) + 1)
    # [Modified in Gloas:EIP8061]
    # Deposits still consume the activation-only churn budget in Gloas.
    available_for_processing = state.deposit_balance_to_consume + get_activation_churn_limit(state)
    processed_amount = 0
    next_deposit_index = 0
    deposits_to_postpone = []
    is_churn_limit_reached = False
    # [Modified in Simplex]
    # Uses slot-based finalized checkpoint
    finalized_slot = state.finalized_checkpoint.slot

    for deposit in state.pending_deposits:
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
    # [Modified in Simplex]
    # Uses round-based participation arrays
    state.previous_round_participation = state.current_round_participation
    state.current_round_participation = [
        ParticipationFlags(0b0000_0000) for _ in range(len(state.validators))
    ]
```

#### Modified `process_rewards_and_penalties`

```python
def process_rewards_and_penalties(state: BeaconState) -> None:
    """
    [Modified in Simplex] Apply per-round attestation flag rewards/penalties
    only. The inactivity-leak penalty is split out into
    ``process_inactivity_penalties`` so it can settle at the leak cadence (once
    per round normally, once per epoch while the inactivity leak is active),
    whereas flag rewards always settle every round.
    """
    if get_current_epoch(state) == GENESIS_EPOCH:
        return
    flag_deltas = [
        get_flag_index_deltas(state, flag_index)
        for flag_index in range(len(PARTICIPATION_FLAG_WEIGHTS))
    ]
    for rewards, penalties in flag_deltas:
        for index in range(len(state.validators)):
            increase_balance(state, ValidatorIndex(index), rewards[index])
            decrease_balance(state, ValidatorIndex(index), penalties[index])
```

#### New `process_inactivity_penalties`

```python
def process_inactivity_penalties(state: BeaconState) -> None:
    """
    [New in Simplex] Apply the inactivity-leak penalty deltas. Split out of
    ``process_rewards_and_penalties`` so it can settle at the leak cadence set
    by ``process_round``: once per round normally, once per epoch while the
    inactivity leak is active.
    """
    # Skip early epochs, matching ``process_inactivity_updates``.
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return
    rewards, penalties = get_inactivity_penalty_deltas(state)
    for index in range(len(state.validators)):
        increase_balance(state, ValidatorIndex(index), rewards[index])
        decrease_balance(state, ValidatorIndex(index), penalties[index])
```

#### New `process_round`

```python
def process_round(state: BeaconState) -> None:
    """
    [New in Simplex] Per-round processing run at every round boundary (round
    length per ``ROUND_SCHEDULE``). Epoch boundaries are always round
    boundaries, so process_round runs before process_epoch at epoch transitions.

    Attestation flag rewards/penalties and participation rotation settle every
    round. Height advance (processHeight) and the inactivity leak normally also
    settle every round, but while the inactivity leak is active they back off
    to once per epoch (the epoch's final round only). The per-validator
    ``justification_targets`` / ``timeouts`` / ``finality_participation`` arrays
    reset only on height advance, so deferring the advance lets them accumulate
    over the whole epoch: a vote delayed -- or censored by a proposer -- within
    the epoch then still counts before the leak attributes a penalty, and the
    Layer 2 (justification-missed) attribution stays well defined because height
    advances at most once per epoch. Inactivity updates run before justification
    and finalization so the three leak guards see the pre-advance state.
    """
    process_rewards_and_penalties(state)
    is_epoch_final_round = (state.slot + 1) % SLOTS_PER_EPOCH == 0
    if not is_in_inactivity_leak(state) or is_epoch_final_round:
        process_inactivity_updates(state)
        process_inactivity_penalties(state)
        process_justification_and_finalization(state)
    process_participation_flag_updates(state)
```

#### Modified `process_epoch`

```python
def process_epoch(state: BeaconState) -> None:
    # [Modified in Simplex]
    # Finality-cycle functions moved to process_round.
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
    process_ptc_window(state)
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
        # [New in Simplex]
        # Round processing at round boundaries (schedule-aware)
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
    if len(indices) == 0 or indices != sorted(set(indices)):
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

    # Finality piggyback well-formedness: either both fields are empty or
    # both are set, and the piggyback height precedes the vote's own height.
    # Timeout votes may carry such a lower-height finality piggyback.
    if data.finality_target == Checkpoint():
        assert data.finality_height == FAR_FUTURE_HEIGHT
    elif is_empty_vote(data):
        # [New in Simplex]
        # Empty vote (empty voted checkpoint): makes no height claim, so a
        # lower-height finalize piggyback carries no height-ordering assert
        # (the empty vote's own ``height`` is the empty marker ``0``) — but the
        # pair must still be a real commitment (paper Definition: vote).
        assert data.finality_height != FAR_FUTURE_HEIGHT
    else:
        assert data.finality_height < data.height

    # Bounded inclusion window: current or previous epoch. Mirrors the
    # wire-side bound in ``validate_on_attestation``. Older attestations are
    # never needed because honest validators re-submit via timeout votes.
    data_epoch = compute_epoch_at_slot(data.slot)
    assert data_epoch in (get_current_epoch(state), get_previous_epoch(state))

    # Committee structure (Electra pattern)
    committee_indices = get_committee_indices(attestation.committee_bits)
    committee_offset = 0
    for committee_index in committee_indices:
        assert committee_index < get_committee_count_per_slot(state, data_epoch)
        committee = get_beacon_committee(state, data.slot, committee_index)
        committee_attesters = {
            attester_index
            for i, attester_index in enumerate(committee)
            if attestation.aggregation_bits[committee_offset + i]
        }
        assert len(committee_attesters) > 0
        committee_offset += len(committee)
    assert len(attestation.aggregation_bits) == committee_offset

    # Signature
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))
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

*Note*: The empty vote needs no special handling here and is excluded from the
timeout certificate by construction. `is_viable_attestation_target` returns
`False` for an empty vote because its `height == Height(0)` never equals
`state.current_height >= GENESIS_HEIGHT`, so the `viable_target` branch below
sets neither `timeouts[i]` nor `justification_targets[i]`. Hence
`has_timeout_quorum` — which counts only `state.timeouts[i]` — never counts an
empty vote toward a timeout certificate. `update_finality_participation` still
runs independently of viability, so the empty vote's finalize piggyback is
processed normally, and its head field enters the fork-choice record layer
(fork-choice.md).

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in Simplex] Delegate to ``validate_attestation`` for
    assertions. Per-validator: ``update_finality_participation`` always runs
    (so older-height votes can still carry valid finality piggybacks).
    A viable timeout vote sets the timeout bit; a viable justification vote
    additionally sets ``justification_targets[i]`` (a fresh justification
    subsumes a timeout per paper processVote) and earns a TIMELY_TARGET
    reward. ``update_finality_participation`` always runs so older-height
    votes can still carry valid finality piggybacks.

    *Note*: Only viable justification votes earn the TIMELY_TARGET reward.
    Inactivity penalties handle the justification-vs-timeout asymmetry in
    the negative direction (justification-missed guard).
    """
    data = attestation.data
    validate_attestation(state, attestation)

    timeout_vote = is_timeout_vote(data)
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
            state.timeouts[validator_index] = True
            if not timeout_vote:
                state.justification_targets[validator_index] = data.target.slot
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
    # [Modified in Simplex]
    # Round-based acceptance window
    attestation_round = compute_round_at_slot(data.slot)
    assert attestation_round in (get_previous_round(state), get_current_round(state))
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    committee = get_available_committee(state, data.slot)
    assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
    assert len(attestation.aggregation_bits) == len(committee)
    assert any(attestation.aggregation_bits)

    is_same_slot_block = data.beacon_block_root == get_block_root_at_slot(state, data.slot) and (
        data.slot == GENESIS_SLOT
        or data.beacon_block_root != get_block_root_at_slot(state, Slot(data.slot - 1))
    )
    if is_same_slot_block:
        assert not data.payload_present

    # Signature verification
    attesting_indices = get_available_attesting_indices(state, attestation)
    pubkeys = [state.validators[i].pubkey for i in sorted(attesting_indices)]
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, compute_epoch_at_slot(data.slot))
    signing_root = compute_signing_root(data, domain)
    assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Head matching
    is_matching_head = data.beacon_block_root == get_block_root_at_slot(state, data.slot)

    # Round participation (round-rotated)
    if attestation_round == get_current_round(state):
        round_participation = state.current_round_participation
    else:
        round_participation = state.previous_round_participation
    # [Modified in Simplex]
    # builder_pending_payments is epoch-structured (rotated once per epoch), so
    # select its half by epoch, not round -- these coincide only when a round
    # equals an epoch.
    if compute_epoch_at_slot(data.slot) == get_current_epoch(state):
        payment = state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH]
    else:
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

    # [Modified in Simplex]
    # Write back updated builder payment weight (epoch-structured buffer)
    if compute_epoch_at_slot(data.slot) == get_current_epoch(state):
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
    assert len(body.deposits) == 0

    def for_ops(operations: Sequence[Any], fn: Callable[[BeaconState, Any], None]) -> None:
        for operation in operations:
            fn(state, operation)

    for_ops(body.proposer_slashings, process_proposer_slashing)
    for_ops(body.attester_slashings, process_attester_slashing)
    for_ops(body.attestations, process_attestation)
    for_ops(body.voluntary_exits, process_voluntary_exit)
    for_ops(body.bls_to_execution_changes, process_bls_to_execution_change)
    for_ops(body.payload_attestations, process_payload_attestation)
    # [New in Simplex]
    for_ops(body.available_attestations, process_available_attestation)
    # [New in Simplex]
    # Round double-vote evidence (lighter penalty than attester slashing)
    for_ops(body.round_double_vote_evidence, process_round_double_vote_evidence)
    # [New in Simplex]
    # body.anchor_quorum is deliberately NOT processed here: it is a threshold
    # certificate consumed by the fork choice (update_pointed_anchor), has no
    # state effect, and creates no records. An invalid reference is ignored by
    # the fork choice and never invalidates the block.
```

## Fork transition

### New `upgrade_to_simplex`

*Note*: The current height's start slot is set to the latest block header slot
so that the first fresh-vote gate references the pre-fork tip.

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
            # [Modified in Simplex]
            current_version=SIMPLEX_FORK_VERSION,
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
        ptc_window=pre.ptc_window,
        # Simplex [New in Simplex]
        justified_height=Height(0),
        # The pre-fork finalized checkpoint has no height in the new counter;
        # it is seeded at 0, alongside current_height at GENESIS_HEIGHT.
        # Finality debt is the gap between the two, so debt materializes only
        # if finality genuinely stalls after the fork — any post-fork
        # finalization resets it.
        finalized_height=Height(0),
        current_height=GENESIS_HEIGHT,
        current_height_start_slot=pre.latest_block_header.slot,
        justification_targets=[FAR_FUTURE_SLOT for _ in range(len(pre.validators))],
        timeouts=Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * len(pre.validators)),
        finality_participation=Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * len(pre.validators)),
    )

    return post
```

## Genesis

### Modified `initialize_beacon_state_from_eth1`

*Note*: `justified_checkpoint` is initialized to a zero-root checkpoint at
genesis; the `epoch <= GENESIS_EPOCH + 1` guard in
`process_justification_and_finalization` ensures it is never used on-chain
before validator deposits are processed.

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
    leaves = [deposit.data for deposit in deposits]
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

    # [New in Simplex]
    # Initialize finality fields
    state.current_height = GENESIS_HEIGHT
    state.justified_checkpoint = Checkpoint(slot=GENESIS_SLOT, root=Root())
    state.finalized_checkpoint = Checkpoint(slot=GENESIS_SLOT, root=Root())
    state.justified_height = Height(0)
    state.finalized_height = Height(0)
    state.current_height_start_slot = GENESIS_SLOT

    return state
```

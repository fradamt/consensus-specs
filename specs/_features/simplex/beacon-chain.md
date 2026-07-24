# Simplex Finality -- The Beacon Chain

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
  - [Proof and deployment status](#proof-and-deployment-status)
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
    - [`BuilderPendingPayment`](#builderpendingpayment)
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
    - [New `get_attestation_proposer_reward_denominator`](#new-get_attestation_proposer_reward_denominator)
  - [Predicates](#predicates)
    - [New `is_attestation_from_active_simplex_fork`](#new-is_attestation_from_active_simplex_fork)
    - [New `is_round_from_active_simplex_fork`](#new-is_round_from_active_simplex_fork)
    - [New `get_round_eligible_validator_indices`](#new-get_round_eligible_validator_indices)
    - [New `get_base_reward_at_epoch`](#new-get_base_reward_at_epoch)
    - [New `compute_leak_penalty_units`](#new-compute_leak_penalty_units)
    - [Modified `is_slashable_attestation_data`](#modified-is_slashable_attestation_data)
    - [Modified `is_eligible_for_activation`](#modified-is_eligible_for_activation)
    - [Modified `is_active_builder`](#modified-is_active_builder)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_current_round`](#new-get_current_round)
    - [New `get_previous_round`](#new-get_previous_round)
    - [Modified `get_finality_delay`](#modified-get_finality_delay)
    - [Modified `get_unslashed_participating_indices`](#modified-get_unslashed_participating_indices)
    - [New `is_target_on_chain`](#new-is_target_on_chain)
    - [New `get_historical_block_proof`](#new-get_historical_block_proof)
    - [New `verify_historical_block_proof`](#new-verify_historical_block_proof)
    - [New `is_timeout_vote`](#new-is_timeout_vote)
    - [New `is_empty_vote`](#new-is_empty_vote)
    - [New `is_nonjustifiable_height`](#new-is_nonjustifiable_height)
    - [New `compute_available_committee`](#new-compute_available_committee)
    - [New `get_available_committee`](#new-get_available_committee)
    - [New `initialize_available_committee_window`](#new-initialize_available_committee_window)
    - [Modified `get_committee_count_per_slot`](#modified-get_committee_count_per_slot)
    - [Modified `get_beacon_committee`](#modified-get_beacon_committee)
  - [Available attestation helpers](#available-attestation-helpers)
    - [New `get_available_attesting_positions`](#new-get_available_attesting_positions)
    - [New `get_available_attesting_indices`](#new-get_available_attesting_indices)
    - [New `get_available_head_reward_per_seat`](#new-get_available_head_reward_per_seat)
  - [Modified helpers](#modified-helpers)
    - [Modified `add_validator_to_registry`](#modified-add_validator_to_registry)
- [Beacon chain state transition function](#beacon-chain-state-transition-function)
  - [Epoch processing](#epoch-processing)
    - [New `advance_height`](#new-advance_height)
    - [New `compute_justified_checkpoint`](#new-compute_justified_checkpoint)
    - [New `has_timeout_quorum`](#new-has_timeout_quorum)
    - [New `has_new_finalization`](#new-has_new_finalization)
    - [Modified `process_justification_and_finalization`](#modified-process_justification_and_finalization)
    - [Modified `process_inactivity_updates`](#modified-process_inactivity_updates)
    - [Modified `get_flag_index_deltas`](#modified-get_flag_index_deltas)
    - [Modified `get_inactivity_penalty_deltas`](#modified-get_inactivity_penalty_deltas)
    - [Modified `process_builder_pending_payments`](#modified-process_builder_pending_payments)
    - [Modified `process_pending_deposits`](#modified-process_pending_deposits)
    - [Modified `process_participation_flag_updates`](#modified-process_participation_flag_updates)
    - [Modified `process_rewards_and_penalties`](#modified-process_rewards_and_penalties)
    - [New `process_inactivity_penalties`](#new-process_inactivity_penalties)
    - [New `process_height_outcome`](#new-process_height_outcome)
    - [New `process_overdue_height_outcomes`](#new-process_overdue_height_outcomes)
    - [New `process_pending_height_outcome`](#new-process_pending_height_outcome)
    - [Modified `process_slot`](#modified-process_slot)
    - [New `process_round`](#new-process_round)
    - [New `process_available_committee_window`](#new-process_available_committee_window)
    - [Modified `process_epoch`](#modified-process_epoch)
    - [Modified `process_slots`](#modified-process_slots)
  - [Block processing](#block-processing)
    - [Modified `process_block`](#modified-process_block)
    - [Modified `is_valid_indexed_attestation`](#modified-is_valid_indexed_attestation)
    - [New `validate_attestation`](#new-validate_attestation)
    - [New `update_finality_participation`](#new-update_finality_participation)
    - [New `record_timely_finality_target`](#new-record_timely_finality_target)
    - [New `record_timely_target`](#new-record_timely_target)
    - [Modified `process_attestation`](#modified-process_attestation)
    - [Modified `process_attester_slashing`](#modified-process_attester_slashing)
    - [Modified `process_proposer_slashing`](#modified-process_proposer_slashing)
    - [New `process_available_attestation`](#new-process_available_attestation)
    - [New `process_round_double_vote_evidence`](#new-process_round_double_vote_evidence)
    - [Modified `process_operations`](#modified-process_operations)
- [Fork transition](#fork-transition)
  - [New `upgrade_to_simplex`](#new-upgrade_to_simplex)

<!-- mdformat-toc end -->

## Introduction

This is the beacon chain specification for simplex-based finality. It replaces
Casper FFG with a fresh-simplex-with-height-filter-and-timeouts finality gadget.
The paper's model is n >= 3f+1. This executable profile interprets `n`, `f`, and
the 2/3 quorums as effective-balance weight rather than validator count. The
denominator is total active effective balance, including validators that are
active but slashed; a slashed validator contributes no support and therefore
remains denominator-only until exit. That weighted-adversary translation is an
additional protocol assumption, not a consequence of the paper's count-model
proof. The quorums govern justification, timeout cert, and finalization. Each
validator signs at most one distinct attestation per round and never signs a
different non-empty target within a state-height. It may re-emit the same
**justify** (R1) target, repeat a **timeout** (R2), or make the validator rule's
one-way unlocked transition from an incompatible R1 history to R2 while that
height stays open. The justify vote is subject to a **fresh-vote** gate that
keys it to the current height's interval on the current chain. A timeout vote is
encoded as `target == Checkpoint()` at a real height (`target == Checkpoint()`
at `height == Height(0)` is instead the empty vote, introduced below).
Finalization takes two steps: justify at height H, then confirm via piggybacked
finality votes at any subsequent height (extended finalization window). The
fork-choice store maintains a single justification root and a height-filter
(viable subtree).

Three vote kinds share the attestation format: a justification vote (a real
`target` at the current state-height), a timeout vote (`target == Checkpoint()`
at a real height), and the **empty vote** (`target == Checkpoint()` at
`height == Height(0)`), which makes no height claim and acts only through its
head field — a latest head vote used by the fork-choice grades — and its
finality piggyback. Under sustained non-finality (*finality debt*), every
`K_NONJUSTIFIABLE`-th height is **nonjustifiable**: the justification branch is
disabled there and the height advances only by timeout cert. A round-start
proposal may carry `body.anchor_root`. Let `q = ceil(2 * W / 3)` for total
active balance `W`, and let `g = 2 * q - W`. Each validator accepts the pointed
root as the stable root only if its local previous-round view credits at least
`g` to it and credits strictly less than `q` to every immediate child in the
pre-existing candidate tree. A signer counts once for either a supporting head
vote or any detected round equivocation. The gates that choose among the vote
kinds are specified in the [validator document](./validator.md); the
latest-head-vote grades, the walk, and safe confirmation in the
[fork-choice document](./fork-choice.md).

*Note*: This specification is built upon [Gloas](../../gloas/beacon-chain.md).

### Proof and deployment status

This feature is an executable research profile, not a theorem-complete mapping
of the accompanying paper. The paper uses a fixed validator universe, while the
Ethereum profile inherits Gloas activation, exit, consolidation, balance, and
withdrawal processing. The resulting gradual accountable-safety degradation is
the standard weak-subjectivity model specified in
[Simplex weak subjectivity](./weak-subjectivity.md), not a separate electorate
snapshot requirement. The distributed round duty schedule and leak batching
below still require a timing/incentive proof distinct from the paper's abstract
machine. The fork-choice document records the corresponding committee and
healing assumptions. The paper also leaves base frontier liveness, clean common
first-selection liveness, residual adversarial-straggler alignment at the
available-confirmation freeze, proposer fairness/exogenous proposer selection,
and the RANDAO/proposer-grinding analysis as open obligations. Nothing in this
executable profile closes those items.

### Core Concept: Height vs Epoch

- **Epochs**: Progress automatically with time (every 32 slots)
- **Heights**: Become due at round boundaries and advance when the due outcome
  is consumed after its one-slot operation-inclusion opportunity

At each round boundary, a height outcome becomes due. It is evaluated after
operations in the next available block, so the round's final-slot votes have
their required one-slot inclusion opportunity. The height may then advance via
**justify** (the current-height target reaches a 2/3 quorum on
`target_participation`) or **timeout cert** (a 2/3 quorum on the per-validator
`timeouts` bitlist). Finality is separate: `F ← J` fires whenever the finality
participation bitlist reaches 2/3; this does NOT advance height.

### Thresholds (n >= 3f+1)

| Threshold          | Stake  | Purpose                                                      |
| ------------------ | ------ | ------------------------------------------------------------ |
| Justification      | >= 2/3 | Quorum for `current_height_target` on `target_participation` |
| Timeout cert       | >= 2/3 | Quorum on `timeouts` bitlist (valid current-height votes)    |
| Finalization       | >= 2/3 | Piggybacked confirm of justified checkpoint                  |
| Accountable safety | 1/3    | Standard BFT (slashing conditions E1 and E2)                 |

Every threshold above uses inherited `get_total_active_balance` as its
denominator. Thus active slashed validators remain in the denominator until
exit, while every support numerator excludes them. Grade-gap pointer support in
fork choice uses the same electorate rule.

### Decoupled Consensus

Finality and LMD-GHOST use different attestation types:

- **Attestations**: All active validators attest once per round via standard
  beacon committee attestations (Electra format). `AttestationData` carries a
  finality target (`Checkpoint()` for timeout votes), height, finality piggyback
  target, and finality height. These determine justification, timeout cert, and
  finalization. Attester slashings enforce the finality-target conflict
  conditions (E1 and E2).
- **Available attestations**: A small 512-member available committee attests per
  slot for fork choice via `AvailableAttestation`. This committee is selected
  from the full active set using `compute_balance_weighted_selection` (same
  mechanism as PTC).

### Attestation Tracking

The state stores one target per height and tracks two per-validator bits:

- `current_height_target`: the first block checkpoint in the current-height
  interval on this chain, or `Checkpoint()` until that block's root is
  available.
- `target_participation[i]`: a bit set when validator `i` cast a fresh
  justification vote for exactly `current_height_target` on this chain.
- `timeouts[i]`: a bit set when validator `i` cast any valid vote at this
  height. A nonempty target need only be on this chain; it need not equal
  `current_height_target`.

The target and both bitlists are reset on height advance. The target is set to
the first block of the height once its state root, and therefore its block root,
is available. Validation requires every nonempty target to name an actual block
on the including chain. A valid current-height vote always sets `timeouts[i]`,
while it sets `target_participation[i]` only when its nonempty target is exactly
`current_height_target`. Thus there is no per-validator target root in
`BeaconState`: full checkpoints remain in signed attestations and in E2 slashing
evidence. The justification branch checks `target_participation` directly; the
timeout branch checks `timeouts`.

This broader timeout marker does not broaden justification. It also preserves
accountable safety: if checkpoint `C` is finalized at height `H` while a chain
conflicting with `C` advances from `H` using a timeout quorum, every signer in
the intersection voted at `H` for either an empty target or a target on the
conflicting chain, hence for a target different from `C`; together with its
finality commitment to `C`, that pair is E1 evidence. E2 remains unchanged. For
leak attribution, a wrong on-chain target avoids only the stall layer: it does
not set `target_participation`, so the ordinary-height target layer still
applies when justification is missed. The target layer is intentionally absent
at a nonjustifiable height.

A separate **finality participation** bitlist tracks finalization confirmations
across the extended window. It persists until a new justification fires and
changes the justified `(height, checkpoint)` pair, at which point it resets. The
reset still occurs when the new height reuses the same checkpoint root (the
bootstrap sentinel case).

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
starting from the era's activation slot. Each entry also records `START_ROUND`,
the round index at which the era begins, so that converting between slots and
rounds is a single-era lookup rather than a walk that accumulates rounds across
eras. For slots before the first entry, `SLOTS_PER_EPOCH` is used (i.e., one
round per epoch) starting from round `0`.

There MUST NOT exist multiple round schedule entries with the same slot value.
The `SLOTS_PER_ROUND` in each entry MUST divide `SLOTS_PER_EPOCH`, and each
entry's activation slot MUST be a multiple of `SLOTS_PER_EPOCH` (epoch-aligned).
Together these ensure the round length is constant within any epoch and every
epoch boundary is a round boundary -- relied on by `get_beacon_committee`
(`slot_in_round` vs the epoch-keyed committee `count`) and the height/round
bookkeeping. Each entry's `START_ROUND` MUST equal the round index of its
activation slot under the preceding eras: for entries sorted by slot, the first
entry's `START_ROUND` is `SLOT // SLOTS_PER_EPOCH`, and every later entry's
`START_ROUND` is
`prev_START_ROUND + (SLOT - prev_SLOT) // prev_SLOTS_PER_ROUND`, where `prev_*`
denotes the previous entry's fields. The round schedule entries SHOULD be sorted
by slot in ascending order.

<!-- list-of-records:round_schedule -->

| Slot | Slots Per Round | Start Round |      Description |
| ---: | --------------: | ----------: | ---------------: |
|    0 |              32 |           0 | Pre-fork (Gloas) |

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

`K_NONJUSTIFIABLE` MUST be at least 2. The healing argument requires the height
immediately following a nonjustifiable height to be an ordinary, justifiable
height; a value of 1 would make every height timeout-only.

### Participation flag indices

*Note*: Simplex repurposes Altair's source-flag position as
`TIMELY_FINALITY_TARGET_FLAG_INDEX`. The finality piggyback's
`(finality_height, finality_target)` identifies the justified checkpoint being
confirmed, which is the source-like function that remains after removing the FFG
source field. The inherited `TIMELY_TARGET_FLAG_INDEX` is interpreted as timely
height-progress participation: any valid current-height vote sets it, whether it
carries the exact justification target, another on-chain target, or the empty
target of a timeout vote. Empty votes set neither marker and do not earn it.

| Name                                | Value |
| ----------------------------------- | ----- |
| `TIMELY_FINALITY_TARGET_FLAG_INDEX` | `0`   |
| `TIMELY_TARGET_FLAG_INDEX`          | `1`   |
| `TIMELY_HEAD_FLAG_INDEX`            | `2`   |

### Incentivization weights

*Note*: The Altair source weight rewards a matching finality piggyback. The
14/26/14 split and total participation weight of 54/64 are unchanged.

| Name                         | Value                                                              |
| ---------------------------- | ------------------------------------------------------------------ |
| `TIMELY_TARGET_WEIGHT`       | `uint64(26)`                                                       |
| `PARTICIPATION_FLAG_WEIGHTS` | `[TIMELY_SOURCE_WEIGHT, TIMELY_TARGET_WEIGHT, TIMELY_HEAD_WEIGHT]` |

TIMELY_FINALITY_TARGET and TIMELY_TARGET are accounted per round. Their attester
deltas and proposer inclusion rewards are divided by the number of rounds per
epoch, so changing the round length does not change maximum per-epoch issuance.
TIMELY_HEAD remains a round-level observability flag, but its rewards are
accounted directly per newly included available-committee seat. The fixed seat
reward divides the inherited epoch-wide 14/64 head budget among all
`SLOTS_PER_EPOCH * AVAILABLE_COMMITTEE_SIZE` seats. This avoids sparse committee
participation suppressing rewards twice and avoids multiplying a
balance-weighted selection by a second validator-balance factor.

*Design consideration*: An alternative is to keep head participation per round
but retain target and finality-target participation across rounds, rewarding
each validator only once per height. This would tie FG issuance to height
progress and avoid repeated rewards while a height stalls. It is not adopted
here: positive attester flag rewards already stop during the inactivity leak,
mixed flag lifetimes require separate settlement and proposer-accounting paths,
and full-validator finality attestations remain useful every round for grade-gap
root syncing. Proposer inclusion rewards retain Altair's behavior during the
leak but remain round-scaled. If revisited, target participation naturally keys
to the current height, while finality-target participation may more naturally
key to the justified pair `(justified_height, justified_checkpoint)` that it
confirms.

### Domain types

| Name                        | Value                      |
| --------------------------- | -------------------------- |
| `DOMAIN_AVAILABLE_ATTESTER` | `DomainType('0x0F000000')` |

### Misc

| Name                       | Value                  |
| -------------------------- | ---------------------- |
| `AVAILABLE_COMMITTEE_SIZE` | `uint64(2**9)` (= 512) |
| `FAR_FUTURE_SLOT`          | `Slot(2**64 - 1)`      |

## Preset

| Name                      | Value        | Description                                  |
| ------------------------- | ------------ | -------------------------------------------- |
| `BLOCK_ROOTS_PROOF_DEPTH` | `uint64(13)` | `floorlog2(SLOTS_PER_HISTORICAL_ROOT)` depth |

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
    beacon_block_root: Root  # Goldfish attestation for fork choice
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
    block_proof: Vector[Bytes32, int(BLOCK_ROOTS_PROOF_DEPTH)]
    # Unused for the genesis slot; otherwise the root at slot - 1, which must
    # differ from block_root.
    prev_slot_root: Root
    prev_slot_proof: Vector[Bytes32, int(BLOCK_ROOTS_PROOF_DEPTH)]
```

### Modified containers

#### `BuilderPendingPayment`

*Note*: Gloas accumulated full-validator beacon-committee balance in `weight`.
Simplex keeps that field solely for pending pre-fork payments and adds exact
available-committee seat bitmaps for post-fork payment support and TIMELY_HEAD
rewards. The bitmaps prevent duplicate aggregate inclusions from increasing
support or rewards. Seat accounting also preserves each duplicate selection; the
reward schedule pays a fixed amount per selected seat rather than applying a
validator-balance reward on top of balance-weighted selection. Because the fork
changes `AttestationData`, a Gloas attestation cannot be included in a Simplex
block and legacy `weight` stops changing at activation. The transition behavior
is specified explicitly below.

```python
class BuilderPendingPayment(Container):
    weight: Gwei  # Legacy Gloas weight for payments created before the fork
    available_participation: Bitvector[AVAILABLE_COMMITTEE_SIZE]
    timely_head_participation: Bitvector[AVAILABLE_COMMITTEE_SIZE]
    withdrawal: BuilderPendingWithdrawal
```

#### `Checkpoint`

*Note*: Post-Simplex checkpoints pair an actual proposal slot with its block
root. The fork transition has one explicit exception: checkpoints whose
associated Simplex height is `Height(0)` preserve the inherited FFG
epoch-boundary slot and root. Such a root may have been carried through an empty
boundary slot and is therefore a **legacy boundary sentinel**, not a valid new
vote target unless it also happens to name a block at that slot.

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
    # Justification target, or Checkpoint() for a timeout or empty vote
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
Electra committee-based format with up to two `HistoricalBlockProof`s for
commitments outside the `block_roots` window: one for `target` and one for
`finality_target` when both need them. Proofs are unsigned (not part of
`AttestationData`) — the proposer attaches them when including the attestation
in a block. A single proof may serve both fields when the checkpoints are equal.

```python
class Attestation(Container):
    aggregation_bits: Bitlist[MAX_VALIDATORS_PER_COMMITTEE * MAX_COMMITTEES_PER_SLOT]
    data: AttestationData
    signature: BLSSignature
    committee_bits: Bitvector[MAX_COMMITTEES_PER_SLOT]
    # [New in Simplex]
    historical_block_proofs: List[HistoricalBlockProof, 2]
```

#### `BeaconBlockBody`

*Note*: `anchor_root` is the grade-gap synchronization pointer. A round-start
(first-slot-of-round) proposal MAY point to one pre-existing block root. The
proposal carries no supporting votes or aggregate. At voting time, each
validator evaluates the root against valid finality attestations it received
from the immediately preceding round. A validator's balance counts once if the
receiver saw either a head descending from `anchor_root` or two distinct
attestations by that validator in the round, even when neither locally seen head
supports the root. The root needs credited support of at least
`g = 2 * ceil(2 * W / 3) - W`, and every immediate candidate-tree child needs
credited support strictly below `q = ceil(2 * W / 3)`.

`Root()` means no pointer. The field is not an operation and has no state
effect. An invalid, misplaced, unknown, or locally under-supported pointer never
invalidates the block; fork choice ignores it and uses the grade-1 fallback.

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
    # Grade-gap synchronization pointer. Fork choice evaluates it from the
    # receiver's live previous-round view; no votes ride the proposal.
    anchor_root: Root
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
    # Cached exactly like ``ptc_window``: previous, current, and lookahead
    # epochs. Old-slot verification and later builder-payment settlement must
    # use the same balance-weighted committee that validators signed for.
    available_committee_window: Vector[
        Vector[ValidatorIndex, AVAILABLE_COMMITTEE_SIZE],
        (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH,
    ]
    # [New in Simplex]
    justified_height: Height  # height of ``justified_checkpoint``
    # [New in Simplex]
    finalized_height: Height  # height of ``finalized_checkpoint``
    # [New in Simplex]
    current_height: Height  # paper's h
    # [New in Simplex]
    # Branch-relative height class, fixed when ``current_height`` is entered.
    # Finalization during the height must not reopen its justification branch.
    current_height_nonjustifiable: boolean
    # [New in Simplex]
    # slot at which the current height began (paper's s_h)
    current_height_start_slot: Slot
    # [New in Simplex]
    # First block checkpoint in the current-height interval on this chain. It is
    # empty until the first such block root can be computed without circularity.
    current_height_target: Checkpoint
    # [New in Simplex]
    # Incremented at each due round boundary. On the next block, older missed
    # outcomes settle before operations and the newest settles afterward,
    # giving the immediately preceding final-slot votes their one-slot
    # inclusion opportunity without collapsing multiple empty-round events.
    pending_height_outcomes: uint64
    # [New in Simplex]
    target_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]
    # [New in Simplex]
    timeouts: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # paper's timeouts[]
    # [New in Simplex]
    finality_participation: Bitlist[VALIDATOR_REGISTRY_LIMIT]  # extended window
    # [New in Simplex]
    # One-time marker for the lighter round-double-vote penalty. Separate from
    # ``Validator.slashed`` because this offense forces exit without slashing.
    round_double_vote_penalized: Bitlist[VALIDATOR_REGISTRY_LIMIT]
```

*Note*: The fields `justification_bits`, `previous_justified_checkpoint`, and
`current_justified_checkpoint` from Gloas are removed.

*Note*: See [Attestation Tracking](#attestation-tracking) for field roles. Key
invariants: every valid current-height vote sets `timeouts[i]`; an exact,
nonempty vote for `current_height_target` additionally sets
`target_participation[i]`. `finality_participation` persists across height
advances and is reset only when a new justification fires (the `justify_target`
branch of `advance_height`).

## Helper functions

### Round helpers

#### New `compute_round_at_slot`

```python
def compute_round_at_slot(slot: Slot) -> Round:
    """
    Return the round number at ``slot``.
    Looks up the era containing ``slot`` in ``ROUND_SCHEDULE`` and offsets from
    the era's ``START_ROUND``. For slots before the first schedule entry,
    ``SLOTS_PER_EPOCH`` is used starting from round ``0``.
    """
    era_start = Slot(0)
    start_round = Round(0)
    slots_per_round = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda entry: entry["SLOT"]):
        if slot < entry["SLOT"]:
            break
        era_start = entry["SLOT"]
        start_round = entry["START_ROUND"]
        slots_per_round = entry["SLOTS_PER_ROUND"]
    return Round(start_round + (slot - era_start) // slots_per_round)
```

#### New `compute_start_slot_at_round`

```python
def compute_start_slot_at_round(round: Round) -> Slot:
    """
    Return the start slot of ``round``.
    Inverse of ``compute_round_at_slot``; looks up the era containing ``round``
    in ``ROUND_SCHEDULE`` and offsets from the era's ``SLOT``.
    """
    era_start = Slot(0)
    start_round = Round(0)
    slots_per_round = SLOTS_PER_EPOCH
    for entry in sorted(ROUND_SCHEDULE, key=lambda entry: entry["START_ROUND"]):
        if round < entry["START_ROUND"]:
            break
        era_start = entry["SLOT"]
        start_round = entry["START_ROUND"]
        slots_per_round = entry["SLOTS_PER_ROUND"]
    return Slot(era_start + (round - start_round) * slots_per_round)
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

#### New `get_attestation_proposer_reward_denominator`

```python
def get_attestation_proposer_reward_denominator(slot: Slot) -> uint64:
    """
    [New in Simplex] Return the Altair proposer-reward denominator scaled by
    the number of rounds per epoch at the attestation slot. Altair lets a
    validator set each participation flag once per epoch; Simplex can set each
    flag once per round, so proposer inclusion rewards use the same 1/R scaling
    as attester flag deltas.
    """
    base_denominator = (
        (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
    )
    return uint64(base_denominator * get_rounds_per_epoch_at_slot(slot))
```

### Predicates

#### New `is_attestation_from_active_simplex_fork`

```python
def is_attestation_from_active_simplex_fork(state: BeaconState, data: AttestationData) -> bool:
    """Return whether ``data`` is from a slot with Simplex attestation duties."""
    # While Simplex is the current fork, ``state.fork.epoch`` is authoritative
    # even in synthetic transition tests. At a later fork, recover the original
    # Simplex activation from configuration so indefinitely valid evidence still
    # cannot reach back into the pre-Simplex era.
    simplex_fork_epoch = (
        state.fork.epoch
        if state.fork.current_version == SIMPLEX_FORK_VERSION
        else SIMPLEX_FORK_EPOCH
    )
    return compute_epoch_at_slot(data.slot) >= simplex_fork_epoch
```

#### New `is_round_from_active_simplex_fork`

```python
def is_round_from_active_simplex_fork(state: BeaconState, round: Round) -> bool:
    """Return whether ``round`` had Simplex participation duties."""
    simplex_fork_epoch = (
        state.fork.epoch
        if state.fork.current_version == SIMPLEX_FORK_VERSION
        else SIMPLEX_FORK_EPOCH
    )
    return compute_epoch_at_round(round) >= simplex_fork_epoch
```

#### New `get_round_eligible_validator_indices`

```python
def get_round_eligible_validator_indices(
    state: BeaconState, round: Round
) -> Sequence[ValidatorIndex]:
    """
    Return validators eligible for accounting in ``round``. Unlike the
    inherited epoch processor, round settlement can occur within an epoch and
    therefore must key eligibility to the duty round rather than to
    ``get_previous_epoch(state)``.
    """
    round_epoch = compute_epoch_at_round(round)
    return [
        ValidatorIndex(index)
        for index, validator in enumerate(state.validators)
        if is_active_validator(validator, round_epoch)
        or (validator.slashed and round_epoch + 1 < validator.withdrawable_epoch)
    ]
```

#### New `get_base_reward_at_epoch`

```python
def get_base_reward_at_epoch(state: BeaconState, index: ValidatorIndex, epoch: Epoch) -> Gwei:
    """Return ``index``'s base reward using the active balance at ``epoch``."""
    active_balance = get_total_balance(state, set(get_active_validator_indices(state, epoch)))
    base_reward_per_increment = Gwei(
        EFFECTIVE_BALANCE_INCREMENT * BASE_REWARD_FACTOR // integer_squareroot(active_balance)
    )
    increments = state.validators[index].effective_balance // EFFECTIVE_BALANCE_INCREMENT
    return Gwei(increments * base_reward_per_increment)
```

#### New `compute_leak_penalty_units`

```python
def compute_leak_penalty_units(
    state: BeaconState,
    index: ValidatorIndex,
    new_height_advance: bool,
    new_justification: bool,
    new_finalization: bool,
) -> int:
    """
    [New in Simplex] Return penalty units in [0, 3] per paper Fig. leak-processslot.
    Three independent guards fire when the corresponding step does not happen
    this round. Slashed validators always accrue the maximum.
    """
    nonjustifiable = state.current_height_nonjustifiable
    if state.validators[index].slashed:
        # Every applicable layer fires. The target layer is inapplicable at a
        # protocol-mandated timeout-only height.
        return 2 if nonjustifiable else 3

    penalty = 0
    # Layer 1 (stall): no height advance and validator did not set the timeout marker.
    # By design this also fires on an empty voter that was confirmation-gated: an empty
    # vote provides no height-progress evidence. Exempting it would let every validator
    # avoid the stall penalty by voting empty while preventing progress. Under stabilization
    # the gate eventually opens; without stabilization, reward loss is unavoidable because
    # the validator cannot safely cast a height-progress vote.
    if not new_height_advance and not state.timeouts[index]:
        penalty += 1
    # A nonjustifiable height forbids fresh target decisions. A saved target may
    # be re-emitted only as a marker, but the latched class still makes the
    # target-participation layer inapplicable. Applying that layer would penalize
    # honest timeout/marker voters for following the timeout-only state rule.
    if not new_justification and not nonjustifiable and not state.target_participation[index]:
        penalty += 1
    finality_pending = (
        state.finalized_height != state.justified_height
        or state.finalized_checkpoint != state.justified_checkpoint
    )
    if finality_pending and not new_finalization and not state.finality_participation[index]:
        penalty += 1
    return penalty
```

#### Modified `is_slashable_attestation_data`

*Note*: Simplex uses two accountable-safety slashing conditions. **E1** says
that if a validator commits to finality target T at `finality_height = H` (via
`finality_target = T`), they must not have voted for any target other than T at
`height = H`. Note that timeout votes (`target = Checkpoint()`) are themselves
in conflict with any commitment `finality_target = T ≠ Checkpoint()` at the same
height, since `Checkpoint() ≠ T`. **E2** forbids two distinct nonempty targets
at the same height. Different chains can have different current-height targets.
E2 is checked from the pair of signed messages rather than represented by a set
in `BeaconState`. Conflicting finalizations at the same height require quorum
intersection, and E1 ensures at least 1/3 of validators are slashable. An empty
vote (`height == Height(0)`) pairs into E1 evidence only against a finality
commitment at `finality_height == Height(0)`; no honest finality commitment at
height `0` exists (heights start at `GENESIS_HEIGHT == Height(1)`), so an honest
validator's empty votes never become slashable evidence — the predicate is
deliberately left uniform. A same-round different-data pair satisfying neither
E1 nor E2 uses the lighter `RoundDoubleVoteEvidence` penalty.

```python
def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    # [Modified in Simplex]
    # E2: two distinct nonempty targets at the same height.
    height_double_target = (
        data_1.height == data_2.height
        and data_1.target != Checkpoint()
        and data_2.target != Checkpoint()
        and data_1.target != data_2.target
    )
    # E1: one vote commits to finality target T at height H; the other voted
    # something other than T at H.
    finality_target_conflict = (
        data_2.finality_target != Checkpoint()
        and data_1.height == data_2.finality_height
        and data_1.target != data_2.finality_target
    ) or (
        data_1.finality_target != Checkpoint()
        and data_2.height == data_1.finality_height
        and data_2.target != data_1.finality_target
    )
    return height_double_target or finality_target_conflict
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

#### New `get_historical_block_proof`

```python
def get_historical_block_proof(
    attestation: Attestation, target: Checkpoint
) -> Optional[HistoricalBlockProof]:
    """Return the proposer-supplied historical proof for ``target``, if any."""
    for proof in attestation.historical_block_proofs:
        if proof.slot == target.slot and proof.block_root == target.root:
            return proof
    return None
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
    # The proof container must match this preset's block-roots vector. Both
    # supported presets use a power-of-two SLOTS_PER_HISTORICAL_ROOT.
    assert 2**BLOCK_ROOTS_PROOF_DEPTH == SLOTS_PER_HISTORICAL_ROOT
    # Proof must be consistent with target
    assert proof.slot == target.slot
    assert proof.block_root == target.root
    # ``historical_summaries[0]`` is the first period summarized after Capella,
    # not global historical period zero. Entries are contiguous through the
    # last completed period, so derive the global-period origin from the
    # current period and list length (also robust in synthetic configurations).
    current_period = state.slot // SLOTS_PER_HISTORICAL_ROOT
    assert len(state.historical_summaries) <= current_period
    summary_origin = current_period - len(state.historical_summaries)

    # Verify block_root at target.slot.
    target_period = target.slot // SLOTS_PER_HISTORICAL_ROOT
    assert target_period >= summary_origin
    summary_index = target_period - summary_origin
    assert summary_index < len(state.historical_summaries)
    block_summary_root = state.historical_summaries[summary_index].block_summary_root
    assert is_valid_merkle_branch(
        leaf=proof.block_root,
        branch=proof.block_proof,
        depth=BLOCK_ROOTS_PROOF_DEPTH,
        index=target.slot % SLOTS_PER_HISTORICAL_ROOT,
        root=block_summary_root,
    )
    # Slot zero is the genesis block and therefore needs no carried-forward
    # exclusion proof. This special case keeps the genesis interval target
    # verifiable even if height 1 remains open beyond the block-roots window.
    if target.slot == GENESIS_SLOT:
        return
    # Verify prev_slot_root at target.slot - 1 (may be in a different summary)
    prev_slot = Slot(target.slot - 1)
    prev_period = prev_slot // SLOTS_PER_HISTORICAL_ROOT
    assert prev_period >= summary_origin
    prev_summary_index = prev_period - summary_origin
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
only its latest head vote and finalize piggyback have effect. Height `0` is the
empty marker: no honest vote is ever cast at height `0`, since the first real
state-height is `GENESIS_HEIGHT == Height(1)`.

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
newly entered height by more than `FINALITY_DEBT_THRESHOLD` — every
`K_NONJUSTIFIABLE`-th height is nonjustifiable. The predicate below classifies a
height **once, when `advance_height` enters it**; the result is latched in
`current_height_nonjustifiable` for that branch's entire stay at the height.
Finalization can update `finalized_height` while the height remains open, but it
must not retroactively enable a justification quorum that honest validators were
forbidden to cast. Branches may still enter the same numeric height with
different classes because their finalized histories differ; validator signing
history handles that case with the one-way marker bridge specified in
`validator.md`.

```python
def is_nonjustifiable_height(height: Height, finalized_height: Height) -> bool:
    """
    [New in Simplex] Classify a newly entered ``height`` as nonjustifiable
    (timeout-only): every ``K_NONJUSTIFIABLE``-th height once the finalized
    height lags by more than ``FINALITY_DEBT_THRESHOLD``.
    """
    assert K_NONJUSTIFIABLE >= 2
    return (height > finalized_height + FINALITY_DEBT_THRESHOLD) and (
        height % K_NONJUSTIFIABLE == 0
    )
```

#### New `compute_available_committee`

```python
def compute_available_committee(
    state: BeaconState, slot: Slot
) -> Vector[ValidatorIndex, AVAILABLE_COMMITTEE_SIZE]:
    """
    [New in Simplex] Compute the 512-seat balance-weighted available committee
    for ``slot``. Callers normally use the cached ``get_available_committee``;
    this helper fills and advances that cache.
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

#### New `get_available_committee`

```python
def get_available_committee(
    state: BeaconState, slot: Slot
) -> Vector[ValidatorIndex, AVAILABLE_COMMITTEE_SIZE]:
    """
    [New in Simplex] Return the cached available committee for ``slot``.
    The window preserves the exact balance-weighted selection across epoch
    boundaries, when active balances may already have changed in ``state``.
    """
    epoch = compute_epoch_at_slot(slot)
    state_epoch = get_current_epoch(state)
    if epoch < state_epoch:
        assert epoch + 1 == state_epoch
        return state.available_committee_window[slot % SLOTS_PER_EPOCH]
    assert epoch <= state_epoch + MIN_SEED_LOOKAHEAD
    offset = (epoch - state_epoch + 1) * SLOTS_PER_EPOCH
    return state.available_committee_window[offset + slot % SLOTS_PER_EPOCH]
```

#### New `initialize_available_committee_window`

```python
def initialize_available_committee_window(
    state: BeaconState,
) -> Vector[
    Vector[ValidatorIndex, AVAILABLE_COMMITTEE_SIZE],
    (2 + MIN_SEED_LOOKAHEAD) * SLOTS_PER_EPOCH,
]:
    """
    [New in Simplex] Initialize cached committees for the current epoch through
    the configured seed-lookahead horizon. The pre-Simplex previous epoch has
    no available committees and receives an all-zero placeholder.
    """
    empty_previous_epoch = [
        Vector[ValidatorIndex, AVAILABLE_COMMITTEE_SIZE]([
            ValidatorIndex(0) for _ in range(AVAILABLE_COMMITTEE_SIZE)
        ])
        for _ in range(SLOTS_PER_EPOCH)
    ]
    committees = []
    current_epoch = get_current_epoch(state)
    for epoch_offset in range(1 + MIN_SEED_LOOKAHEAD):
        start_slot = compute_start_slot_at_epoch(Epoch(current_epoch + epoch_offset))
        committees += [
            compute_available_committee(state, Slot(start_slot + slot_offset))
            for slot_offset in range(SLOTS_PER_EPOCH)
        ]
    return empty_previous_epoch + committees
```

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

#### New `get_available_head_reward_per_seat`

```python
def get_available_head_reward_per_seat(state: BeaconState, slot: Slot) -> Gwei:
    """
    [New in Simplex] Return the fixed TIMELY_HEAD reward for one available-
    committee seat. At full participation, all
    ``SLOTS_PER_EPOCH * AVAILABLE_COMMITTEE_SIZE`` seats collectively receive
    the inherited epoch-wide TIMELY_HEAD budget. Because committee selection is
    already balance weighted, a fixed seat reward makes expected reward linear
    in validator stake rather than weighting stake twice.
    """
    duty_epoch = compute_epoch_at_slot(slot)
    active_balance = get_total_balance(state, set(get_active_validator_indices(state, duty_epoch)))
    active_increments = active_balance // EFFECTIVE_BALANCE_INCREMENT
    base_reward_per_increment = Gwei(
        EFFECTIVE_BALANCE_INCREMENT * BASE_REWARD_FACTOR // integer_squareroot(active_balance)
    )
    total_base_rewards = active_increments * base_reward_per_increment
    return Gwei(
        total_base_rewards
        * TIMELY_HEAD_WEIGHT
        // (WEIGHT_DENOMINATOR * SLOTS_PER_EPOCH * AVAILABLE_COMMITTEE_SIZE)
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
    set_or_append_list(state.target_participation, index, False)  # noqa: FBT003
    set_or_append_list(state.timeouts, index, False)  # noqa: FBT003
    set_or_append_list(state.finality_participation, index, False)  # noqa: FBT003
    set_or_append_list(state.round_double_vote_penalized, index, False)  # noqa: FBT003
```

## Beacon chain state transition function

### Epoch processing

#### New `advance_height`

```python
def advance_height(
    state: BeaconState,
    justify_target: Optional[Checkpoint] = None,
    height_start_slot: Optional[Slot] = None,
) -> None:
    """
    [New in Simplex] Advance ``current_height`` by 1 (paper processHeight
    advance branches). If ``justify_target is not None`` the justify branch
    fired: set ``justified_checkpoint`` / ``justified_height`` and reset
    ``finality_participation``. The timeout branch (``justify_target is None``)
    skips those updates. Either way: bump ``current_height``, set
    ``current_height_start_slot`` to ``height_start_slot`` (the current slot by
    default), reset ``current_height_target``, ``target_participation``, and
    ``timeouts``.
    """
    if justify_target is not None:
        state.justified_checkpoint = justify_target
        state.justified_height = state.current_height
        state.finality_participation = Bitlist[VALIDATOR_REGISTRY_LIMIT](
            [False] * len(state.validators)
        )
    state.current_height = Height(state.current_height + 1)
    # [New in Simplex]
    # Freeze the new height's class before any later finality event can change
    # ``finalized_height``. All consumers read this latch, never recompute it.
    state.current_height_nonjustifiable = is_nonjustifiable_height(
        state.current_height,
        state.finalized_height,
    )
    # [New in Simplex]
    # A block-delivered outcome starts the new height in the deciding block's
    # slot, allowing validators later in that slot to use it. An outcome
    # consumed after an empty slot explicitly passes the next slot instead, so
    # duties already performed in the empty slot remain in the old interval.
    if height_start_slot is None:
        height_start_slot = state.slot
    state.current_height_start_slot = height_start_slot
    state.current_height_target = Checkpoint()
    num_validators = len(state.validators)
    state.target_participation = Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * num_validators)
    state.timeouts = Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * num_validators)
```

#### New `compute_justified_checkpoint`

```python
def compute_justified_checkpoint(state: BeaconState) -> Checkpoint:
    """
    [New in Simplex] Return ``current_height_target`` when its exact
    ``target_participation`` weight reaches a 2/3 quorum. Returns
    ``Checkpoint()`` otherwise.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return Checkpoint()

    # [New in Simplex]
    # Nonjustifiable heights are timeout-only: the state never produces a
    # justification at a nonjustifiable height, so the height can advance only via
    # the timeout-cert branch.
    if state.current_height_nonjustifiable:
        return Checkpoint()

    if state.current_height_target == Checkpoint():
        return Checkpoint()
    total_active_balance = get_total_active_balance(state)
    active_indices = get_active_validator_indices(state, get_current_epoch(state))
    target_weight = Gwei(
        sum(
            state.validators[index].effective_balance
            for index in active_indices
            if state.target_participation[index] and not state.validators[index].slashed
        )
    )
    if (
        target_weight * FINALITY_QUORUM_DENOMINATOR
        < total_active_balance * FINALITY_QUORUM_NUMERATOR
    ):
        return Checkpoint()
    return state.current_height_target
```

#### New `has_timeout_quorum`

```python
def has_timeout_quorum(state: BeaconState) -> bool:
    """
    [New in Simplex] Return ``True`` iff timeout support from non-slashed active
    validators reaches 2/3 of total active effective balance. Active slashed
    balance remains in that denominator but supplies no support. A timeout cert
    advances height without updating the justified checkpoint.
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
    [New in Simplex] Return ``True`` iff non-slashed active support on
    ``finality_participation`` reaches 2/3 of total active effective balance
    and the justified ``(height, checkpoint)`` pair is still pending. Active
    slashed balance remains in the denominator but supplies no support.
    """
    if (
        state.finalized_height == state.justified_height
        and state.finalized_checkpoint == state.justified_checkpoint
    ):
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

#### Modified `process_justification_and_finalization`

*Note*: Paper's `processHeight` (alg:state-machine). Runs AFTER
`process_inactivity_updates` / `process_rewards_and_penalties` so they see
pre-advance state. At most one of the justify/timeout branches advances height
per invocation; the finality branch is independent.

```python
def process_justification_and_finalization(
    state: BeaconState, height_start_slot: Optional[Slot] = None
) -> None:
    """
    [Modified in Simplex] Three-branch state machine per paper processHeight.
    """
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return

    # (1) Finality: F ← J (does not advance height)
    if has_new_finalization(state):
        state.finalized_checkpoint = state.justified_checkpoint
        # [New in Simplex]
        # Finality is processed first, but the current height's class was
        # latched when that height began. Reducing finality debt here therefore
        # cannot reopen a timeout-only height after honest validators already
        # withheld its targets. ``advance_height`` classifies the next height
        # from the updated finalized height.
        state.finalized_height = state.justified_height

    # (2) Justify branch
    justified = compute_justified_checkpoint(state)
    if justified != Checkpoint():
        advance_height(
            state,
            justify_target=justified,
            height_start_slot=height_start_slot,
        )
        return

    # (3) Timeout cert branch
    if has_timeout_quorum(state):
        advance_height(state, height_start_slot=height_start_slot)
        return
```

#### Modified `process_inactivity_updates`

*Note*: Three-guard design (paper alg:leak-processslot). Guards are computed
against the pre-advance state; `process_justification_and_finalization` runs
later in the same `process_pending_height_outcome` call.

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
    settlement_round = get_previous_round(state)
    for index in get_round_eligible_validator_indices(state, settlement_round):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_height_advance,
            new_justification,
            new_finalization,
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
    [Modified in Simplex] Finality-target and target rewards/penalties are
    scaled by 1/rounds-per-epoch to keep per-epoch totals constant when running
    per-round. TIMELY_HEAD is paid per newly included available-committee seat
    by ``process_available_attestation`` and has no flag delta here.
    """
    rewards = [Gwei(0)] * len(state.validators)
    penalties = [Gwei(0)] * len(state.validators)
    if flag_index == TIMELY_HEAD_FLAG_INDEX:
        return rewards, penalties
    # [Modified in Simplex]
    # Pass previous round instead of previous epoch
    settlement_round = get_previous_round(state)
    settlement_epoch = compute_epoch_at_round(settlement_round)
    unslashed_participating_indices = get_unslashed_participating_indices(
        state, flag_index, settlement_round
    )
    weight = PARTICIPATION_FLAG_WEIGHTS[flag_index]
    unslashed_participating_balance = get_total_balance(state, unslashed_participating_indices)
    unslashed_participating_increments = (
        unslashed_participating_balance // EFFECTIVE_BALANCE_INCREMENT
    )
    active_balance = get_total_balance(
        state, set(get_active_validator_indices(state, settlement_epoch))
    )
    active_increments = active_balance // EFFECTIVE_BALANCE_INCREMENT
    settlement_round_start = compute_start_slot_at_round(settlement_round)
    rounds_per_epoch = get_rounds_per_epoch_at_slot(settlement_round_start)
    for index in get_round_eligible_validator_indices(state, settlement_round):
        base_reward = get_base_reward_at_epoch(state, index, settlement_epoch)
        if index in unslashed_participating_indices:
            if not is_in_inactivity_leak(state):
                reward_numerator = base_reward * weight * unslashed_participating_increments
                # [Modified in Simplex]
                # Scale for the round being settled. At a schedule boundary,
                # ``state.slot`` is in the new era while the participation is
                # from the preceding round in the old era.
                rewards[index] += Gwei(
                    reward_numerator // (active_increments * WEIGHT_DENOMINATOR * rounds_per_epoch)
                )
        elif flag_index != TIMELY_HEAD_FLAG_INDEX:
            # [Modified in Simplex]
            # Scale for the preceding round, including across schedule changes.
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
    settlement_round = get_previous_round(state)
    for index in get_round_eligible_validator_indices(state, settlement_round):
        penalty_units = compute_leak_penalty_units(
            state,
            ValidatorIndex(index),
            new_height_advance,
            new_justification,
            new_finalization,
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

#### Modified `process_builder_pending_payments`

*Note*: A post-fork builder payment is released by at least the configured 60%
threshold of the slot's available-committee **seats**. Committee selection is
already balance weighted, so weighting each selected seat by balance again would
square validator influence. The stored bitmap is expanded over duplicate seats
and read against the cached committee chosen for that slot. This replaces
Gloas's full-validator per-slot committee denominator, which a fixed 512-seat
committee cannot reach at mainnet validator counts. Pending pre-fork payments
retain their already accumulated Gloas `weight` and use the legacy threshold.
During the activation epoch, the activation-slot available committee also acts
as a one-time transition receipt for residual payments from the final Gloas
epoch. Either an already-earned Gloas weight quorum or an available-seat quorum
is sufficient; partial transition receipts therefore cannot suppress legacy
support. This is deliberately a **new transition eligibility rule**, not an
equivalent reconstruction of each legacy slot's Gloas attester weight. A
residual claim with zero Gloas weight can be paid when the later activation
committee certifies the activation head containing it, even if the parent was
extended EMPTY. Protocol deployment must accept that one-off retroactive
canonical-ancestry receipt. In either case, a zero withdrawal amount is the
cancellation marker set by `process_proposer_slashing`: retained participation
bits protect reward replay but cannot resurrect the canceled financial claim.
The receipt counts the activation attestation's **head-root seats only**:
`payload_present` deliberately does not split the financial quorum. Aggregates
with the same root and different payload signals remain separately signed and
validated, but their disjoint seats union toward this ancestry receipt.

```python
def process_builder_pending_payments(state: BeaconState) -> None:
    legacy_quorum = get_builder_payment_quorum_threshold(state)

    for payment in state.builder_pending_payments[:SLOTS_PER_EPOCH]:
        has_legacy_quorum = payment.weight > 0 and payment.weight >= legacy_quorum
        has_available_quorum = (
            sum(payment.available_participation) * BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
            >= AVAILABLE_COMMITTEE_SIZE * BUILDER_PAYMENT_THRESHOLD_NUMERATOR
        )
        should_pay = has_legacy_quorum or has_available_quorum
        if should_pay and payment.withdrawal.amount > 0:
            state.builder_pending_withdrawals.append(payment.withdrawal)

    old_payments = state.builder_pending_payments[SLOTS_PER_EPOCH:]
    new_payments = [BuilderPendingPayment() for _ in range(SLOTS_PER_EPOCH)]
    state.builder_pending_payments = old_payments + new_payments
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
    # There is no preceding participation interval to settle in the genesis
    # round. Later rounds in genesis epoch are real intervals and must not be
    # rotated away unpaid when the schedule has multiple rounds per epoch.
    if get_current_round(state) == GENESIS_ROUND:
        return
    # The upgrade settles the final Gloas interval and starts empty Simplex
    # arrays. At the first Simplex boundary, ``previous_round_participation``
    # therefore names a pre-fork duty and must not be charged a second time
    # under Simplex rules.
    if not is_round_from_active_simplex_fork(state, get_previous_round(state)):
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

#### New `process_height_outcome`

```python
def process_height_outcome(state: BeaconState, height_start_slot: Optional[Slot] = None) -> None:
    """
    [New in Simplex] Settle one due height/leak outcome against the current
    accumulated vote state. ``height_start_slot`` distinguishes an outcome
    decided by a block in the current slot from one decided after that slot was
    empty.
    """
    process_inactivity_updates(state)
    process_inactivity_penalties(state)
    process_justification_and_finalization(state, height_start_slot)
```

#### New `process_overdue_height_outcomes`

```python
def process_overdue_height_outcomes(state: BeaconState) -> None:
    """
    [New in Simplex] Defensively consume every due outcome except the newest
    before current block operations. In an ordinary transition, an empty
    inclusion-opportunity slot settles its outcome in ``process_slot``, so at
    most one outcome remains. Retaining the counter and this guard makes the
    ordering total for imported or test-constructed states as well.
    """
    while state.pending_height_outcomes > 1:
        process_height_outcome(state)
        state.pending_height_outcomes -= 1
```

#### New `process_pending_height_outcome`

```python
def process_pending_height_outcome(
    state: BeaconState, height_start_slot: Optional[Slot] = None
) -> None:
    """
    [New in Simplex] After current block operations, consume the newest due
    outcome, including attestations from the preceding round's final slot.
    """
    if state.pending_height_outcomes == 0:
        return
    process_height_outcome(state, height_start_slot)
    state.pending_height_outcomes -= 1
```

#### Modified `process_slot`

```python
def process_slot(state: BeaconState) -> None:
    # Cache state root
    previous_state_root = hash_tree_root(state)
    state.state_roots[state.slot % SLOTS_PER_HISTORICAL_ROOT] = previous_state_root
    # Cache latest block header state root
    if state.latest_block_header.state_root == Bytes32():
        state.latest_block_header.state_root = previous_state_root
    # Cache block root
    previous_block_root = hash_tree_root(state.latest_block_header)
    state.block_roots[state.slot % SLOTS_PER_HISTORICAL_ROOT] = previous_block_root
    # [New in Simplex]
    # The first block of a height cannot contain its own root in its state. At
    # the following slot its header state root has been filled, so set the target
    # if it is the first block in the current-height interval.
    if (
        state.current_height_target == Checkpoint()
        and state.latest_block_header.slot >= state.current_height_start_slot
    ):
        state.current_height_target = Checkpoint(
            slot=state.latest_block_header.slot,
            root=previous_block_root,
        )
    # [New in Gloas:EIP7732]
    # Unset the next payload availability
    state.execution_payload_availability[(state.slot + 1) % SLOTS_PER_HISTORICAL_ROOT] = 0b0
    # [New in Simplex]
    # Reaching process_slot for an inclusion-opportunity slot proves that this
    # chain has no block at that slot. Settle the preceding boundary's outcome
    # now, after its empty operation phase and before any later round/epoch
    # boundary is evaluated. A block at this slot instead settles the same
    # outcome after process_operations in process_block.
    process_pending_height_outcome(state, Slot(state.slot + 1))
```

#### New `process_round`

```python
def process_round(state: BeaconState) -> None:
    """
    [New in Simplex] Per-round processing run at every round boundary (round
    length per ``ROUND_SCHEDULE``). Epoch boundaries are always round
    boundaries, so process_round runs before process_epoch at epoch transitions.

    Attestation flag rewards/penalties and participation rotation settle every
    round. Height advance (processHeight) and the inactivity leak normally
    become due every round, but while the inactivity leak is active they back
    off to once per epoch (the epoch's final round only). A due outcome is not
    evaluated here. If the next slot has a block,
    ``process_pending_height_outcome`` runs after that block's operations; if
    it is empty, ``process_slot`` settles the outcome after the empty operation
    phase. Thus final-slot attestations have their minimum one-slot inclusion
    opportunity, while later cadence decisions always observe the preceding
    outcome. The per-validator
    ``target_participation`` / ``timeouts`` / ``finality_participation`` arrays
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
        state.pending_height_outcomes += 1
    process_participation_flag_updates(state)
```

#### New `process_available_committee_window`

```python
def process_available_committee_window(state: BeaconState) -> None:
    """
    [New in Simplex] Shift the cached available committees by one epoch and
    compute the newly reachable lookahead epoch. This runs after effective
    balance and RANDAO updates, matching ``process_ptc_window``.
    """
    window = state.available_committee_window
    window[: len(window) - SLOTS_PER_EPOCH] = window[SLOTS_PER_EPOCH:]
    next_epoch = Epoch(get_current_epoch(state) + MIN_SEED_LOOKAHEAD + 1)
    start_slot = compute_start_slot_at_epoch(next_epoch)
    window[len(window) - SLOTS_PER_EPOCH :] = [
        compute_available_committee(state, Slot(start_slot + slot_offset))
        for slot_offset in range(SLOTS_PER_EPOCH)
    ]
```

#### Modified `process_epoch`

```python
def process_epoch(state: BeaconState) -> None:
    # [Modified in Simplex]
    # Finality-cycle functions are marked due by process_round and consumed
    # after operations (or the empty operation phase) in the next slot.
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
    # [New in Simplex]
    process_available_committee_window(state)
```

#### Modified `process_slots`

```python
def process_slots(state: BeaconState, slot: Slot) -> None:
    """
    [Modified in Simplex] Adds round processing at round boundaries.
    Round processing runs before epoch processing. Since epoch boundaries are
    always round boundaries, the order at epoch transition is: process_slot
    (including any prior outcome) → process_round (last round of epoch) →
    process_epoch (administrative).
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

#### Modified `process_block`

```python
def process_block(state: BeaconState, block: BeaconBlock) -> None:
    # [Modified in Simplex]
    # Gloas block processing. A defensive backlog settles before this block;
    # the newest outcome settles only after operations. This ordering lets
    # final-slot attestations included at
    # MIN_ATTESTATION_INCLUSION_DELAY affect the immediately preceding round
    # without collapsing earlier leak events.
    process_overdue_height_outcomes(state)
    process_parent_execution_payload(state, block)
    process_block_header(state, block)
    process_withdrawals(state)
    process_execution_payload_bid(state, block)
    process_randao(state, block.body)
    process_eth1_data(state, block.body)
    process_operations(state, block.body)
    process_pending_height_outcome(state)
    process_sync_aggregate(state, block.body.sync_aggregate)
```

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
    # [Modified in Simplex]
    # Evidence is not age-bounded. Resolve the duty epoch through the complete
    # configured fork schedule instead of ``state.fork``, which retains only
    # the current and immediately previous versions. Future forks that inherit
    # this attestation format extend ``compute_fork_version`` in the usual way.
    if state.fork.current_version == SIMPLEX_FORK_VERSION:
        fork_version = (
            state.fork.previous_version if epoch < state.fork.epoch else state.fork.current_version
        )
    else:
        fork_version = compute_fork_version(epoch)
    domain = compute_domain(
        DOMAIN_BEACON_ATTESTER,
        fork_version,
        state.genesis_validators_root,
    )
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
    may reward them). Current-height participation is classified separately
    in ``process_attestation`` after this validation succeeds.
    """
    data = attestation.data

    # A Simplex-shaped attestation cannot claim a duty from before the fork,
    # even though its slot epoch would select the preceding fork's BLS domain.
    assert is_attestation_from_active_simplex_fork(state, data)

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

    # Paper vote validity: every non-empty target commitment must name a real
    # block that already exists on the including chain. The proposer supplies
    # a historical proof when either commitment is outside ``block_roots``.
    for target in (data.target, data.finality_target):
        if target == Checkpoint():
            continue
        # A signer cannot commit at ``data.slot`` to a block proposed later.
        # Inclusion may occur much later, so checking only against
        # ``state.slot`` would admit a backdated vote for a future target.
        assert target.slot <= data.slot
        historical_proof = get_historical_block_proof(attestation, target)
        assert is_target_on_chain(state, target, historical_proof)

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
        and (
            state.finalized_height != state.justified_height
            or state.finalized_checkpoint != state.justified_checkpoint
        )
        and not state.finality_participation[validator_index]
    ):
        state.finality_participation[validator_index] = True
```

#### New `record_timely_finality_target`

```python
def record_timely_finality_target(
    state: BeaconState,
    validator_index: ValidatorIndex,
    data: AttestationData,
    round_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT],
) -> Gwei:
    """
    [New in Simplex] Set the TIMELY_FINALITY_TARGET flag when the piggyback
    names the state's current justified checkpoint at its justified height.
    Return the proposer-reward numerator contribution if newly set.

    Unlike ``update_finality_participation``, reward matching does not require
    finalization to still be pending. This mirrors Altair source matching: the
    flag rewards correctly identifying the justified checkpoint, while the
    persistent finality tally mutates only when that checkpoint still needs to
    be finalized.
    """
    is_matching_finality_target = (
        data.finality_target != Checkpoint()
        and data.finality_height == state.justified_height
        and data.finality_target == state.justified_checkpoint
    )
    if not is_matching_finality_target or has_flag(
        round_participation[validator_index], TIMELY_FINALITY_TARGET_FLAG_INDEX
    ):
        return Gwei(0)
    round_participation[validator_index] = add_flag(
        round_participation[validator_index], TIMELY_FINALITY_TARGET_FLAG_INDEX
    )
    duty_epoch = compute_epoch_at_slot(data.slot)
    return Gwei(get_base_reward_at_epoch(state, validator_index, duty_epoch) * TIMELY_SOURCE_WEIGHT)
```

#### New `record_timely_target`

```python
def record_timely_target(
    state: BeaconState,
    validator_index: ValidatorIndex,
    data: AttestationData,
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
    duty_epoch = compute_epoch_at_slot(data.slot)
    return Gwei(get_base_reward_at_epoch(state, validator_index, duty_epoch) * TIMELY_TARGET_WEIGHT)
```

#### Modified `process_attestation`

*Note*: The empty vote needs no special handling here and is excluded from the
timeout certificate by construction. Its `height == Height(0)` never equals
`state.current_height >= GENESIS_HEIGHT`, so it sets neither `timeouts[i]` nor
`target_participation[i]`. Hence `has_timeout_quorum` — which counts only
`state.timeouts[i]` — never counts an empty vote toward a timeout certificate.
`update_finality_participation` still runs independently of viability, so the
empty vote's finalize piggyback is processed normally, and its head field enters
fork choice as a latest message when the valid attestation is delivered
(fork-choice.md).

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    """
    [Modified in Simplex] Delegate to ``validate_attestation`` for
    assertions. Per-validator: ``update_finality_participation`` always runs
    (so older-height votes can still carry valid finality piggybacks).
    A piggyback matching the current justified checkpoint earns the
    TIMELY_FINALITY_TARGET flag independently of the attestation's own target
    viability.
    Every valid current-height vote sets the timeout bit. An exact nonempty
    vote for ``current_height_target`` independently sets
    ``target_participation[i]``. Thus another on-chain target can contribute to
    height progress without contributing to justification. Every valid
    current-height vote earns TIMELY_TARGET: in Simplex this flag rewards a
    timely contribution to height progress, so protocol-required timeout
    voters are not penalized at nonjustifiable heights.

    *Note*: Current-height votes earn TIMELY_TARGET; empty and stale-height
    votes do not. Any vote kind, including timeout and empty votes, can earn
    TIMELY_FINALITY_TARGET through a matching piggyback. The inactivity target
    layer retains the exact-target distinction at ordinary heights, and is
    disabled at a protocol-mandated nonjustifiable height.
    """
    data = attestation.data
    validate_attestation(state, attestation)

    counts_for_timeout = data.height == state.current_height
    target_vote = (
        counts_for_timeout
        and data.target != Checkpoint()
        and data.target == state.current_height_target
    )

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
    attestation_epoch = compute_epoch_at_slot(data.slot)
    for validator_index in get_attesting_indices(state, attestation):
        # A previous-round vote may be included just across an epoch boundary.
        # Judge the signer against the duty epoch; using the including state's
        # current epoch would discard the final vote of a validator whose exit
        # became effective at that boundary and then penalize it at settlement.
        if not is_active_validator(state.validators[validator_index], attestation_epoch):
            continue
        update_finality_participation(state, validator_index, data)
        if round_participation is not None:
            proposer_reward_numerator += record_timely_finality_target(
                state, validator_index, data, round_participation
            )
        if counts_for_timeout:
            state.timeouts[validator_index] = True
            if round_participation is not None:
                proposer_reward_numerator += record_timely_target(
                    state, validator_index, data, round_participation
                )
        if target_vote:
            state.target_participation[validator_index] = True

    if proposer_reward_numerator > 0:
        proposer_reward_denominator = get_attestation_proposer_reward_denominator(data.slot)
        proposer_reward = Gwei(proposer_reward_numerator // proposer_reward_denominator)
        increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
```

#### Modified `process_attester_slashing`

*Note*: E1 and E2 are Simplex slashing rules over the Simplex `AttestationData`
meaning. Both signed duties therefore have to originate at or after Simplex
activation. Without this guard, a post-fork block could reinterpret arbitrary
pre-fork slots under these rules and manufacture a slashable duty that did not
exist.

```python
def process_attester_slashing(state: BeaconState, attester_slashing: AttesterSlashing) -> None:
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2
    assert is_attestation_from_active_simplex_fork(state, attestation_1.data)
    assert is_attestation_from_active_simplex_fork(state, attestation_2.data)
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)

    slashed_any = False
    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    for index in sorted(indices):
        if is_slashable_validator(state.validators[index], get_current_epoch(state)):
            slash_validator(state, index)
            slashed_any = True
    assert slashed_any
```

#### Modified `process_proposer_slashing`

*Note*: Gloas deletes the complete `BuilderPendingPayment` for a slashed
proposal. In Simplex that container also holds the per-seat
`timely_head_participation` replay guard. Deleting the guard would let the same
valid available aggregate earn attester and inclusion-proposer rewards again
while it remains in the current/previous-round inclusion window. Slashing
therefore clears the financial claim but retains both slot-local participation
bitmaps until their normal epoch-buffer rotation.

```python
def process_proposer_slashing(state: BeaconState, proposer_slashing: ProposerSlashing) -> None:
    header_1 = proposer_slashing.signed_header_1.message
    header_2 = proposer_slashing.signed_header_2.message

    assert header_1.slot == header_2.slot
    assert header_1.proposer_index == header_2.proposer_index
    assert header_1 != header_2
    proposer = state.validators[header_1.proposer_index]
    assert is_slashable_validator(proposer, get_current_epoch(state))
    for signed_header in (proposer_slashing.signed_header_1, proposer_slashing.signed_header_2):
        domain = get_domain(
            state, DOMAIN_BEACON_PROPOSER, compute_epoch_at_slot(signed_header.message.slot)
        )
        signing_root = compute_signing_root(signed_header.message, domain)
        assert bls.Verify(proposer.pubkey, signing_root, signed_header.signature)

    # [Modified in Simplex]
    # Cancel the builder withdrawal but retain the available-attestation
    # receipt/reward bitmaps as replay-protection state.
    slot = header_1.slot
    proposal_epoch = compute_epoch_at_slot(slot)
    payment_index = None
    if proposal_epoch == get_current_epoch(state):
        payment_index = SLOTS_PER_EPOCH + slot % SLOTS_PER_EPOCH
    elif proposal_epoch == get_previous_epoch(state):
        payment_index = slot % SLOTS_PER_EPOCH
    if payment_index is not None:
        payment = state.builder_pending_payments[payment_index]
        payment.weight = Gwei(0)
        payment.withdrawal = BuilderPendingWithdrawal()
        state.builder_pending_payments[payment_index] = payment

    slash_validator(state, header_1.proposer_index)
```

#### New `process_available_attestation`

```python
def process_available_attestation(state: BeaconState, attestation: AvailableAttestation) -> None:
    """
    [New in Simplex] Process an available committee attestation for LMD-GHOST.
    Sets TIMELY_HEAD, pays each newly included timely head seat once, and
    records builder-payment committee seats. During the Simplex activation
    epoch, a matching activation-slot attestation also records the same seats
    on every residual payment from the final Gloas epoch. This native Simplex
    receipt replaces Gloas attestations that cannot be included after their SSZ
    type changes at the fork.
    """
    data = attestation.data
    # [Modified in Simplex]
    # Round-based acceptance window
    attestation_round = compute_round_at_slot(data.slot)
    # During the Simplex activation fork, the cached previous-epoch window is
    # an all-zero transition placeholder, not a real committee. At later forks
    # the previous epoch already had Simplex duties, so do not reapply this
    # activation-only guard using the later fork's epoch.
    if state.fork.current_version == SIMPLEX_FORK_VERSION:
        assert data.slot >= compute_start_slot_at_epoch(state.fork.epoch)
    activation_slot = compute_start_slot_at_epoch(state.fork.epoch)
    is_transition_receipt = (
        state.fork.current_version == SIMPLEX_FORK_VERSION
        and state.fork.previous_version == GLOAS_FORK_VERSION
        and get_current_epoch(state) == state.fork.epoch
        and data.slot == activation_slot
    )
    # The transition receipt remains includable until the activation epoch's
    # payment rotation. Do not make its financial effect depend on a future
    # schedule choosing a shorter first Simplex round.
    assert is_transition_receipt or attestation_round in (
        get_previous_round(state),
        get_current_round(state),
    )
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
    elif attestation_round == get_previous_round(state):
        round_participation = state.previous_round_participation
    else:
        # Only a transition receipt can be older than the previous round, and
        # such a late inclusion is not eligible for round rewards.
        assert is_transition_receipt
        round_participation = None
    # [Modified in Simplex]
    # builder_pending_payments is epoch-structured (rotated once per epoch), so
    # select its half by epoch, not round -- these coincide only when a round
    # equals an epoch.
    if compute_epoch_at_slot(data.slot) == get_current_epoch(state):
        payment = state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH]
    else:
        payment = state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH]

    attesting_positions = get_available_attesting_positions(state, attestation)

    # Builder-payment support is slot-local. Record exact committee positions
    # independently of head rewards so late support remains useful, a validator
    # selected in more than one slot can support each payment, and repeated
    # aggregates cannot double-count a position.
    if is_same_slot_block and payment.withdrawal.amount > 0:
        for position in attesting_positions:
            payment.available_participation[position] = True

    proposer_reward_basis = Gwei(0)
    if is_matching_head and (state.slot - data.slot) == MIN_ATTESTATION_INCLUSION_DELAY:
        assert round_participation is not None
        seat_reward = get_available_head_reward_per_seat(state, data.slot)
        for position in attesting_positions:
            if payment.timely_head_participation[position]:
                continue
            payment.timely_head_participation[position] = True
            index = committee[position]
            proposer_reward_basis += seat_reward
            if not state.validators[index].slashed and not is_in_inactivity_leak(state):
                increase_balance(state, index, seat_reward)

        # Retain the round flag for participation observability; rewards are
        # seat-accounted above and do not read this flag.
        for index in attesting_indices:
            if not has_flag(round_participation[index], TIMELY_HEAD_FLAG_INDEX):
                round_participation[index] = add_flag(
                    round_participation[index], TIMELY_HEAD_FLAG_INDEX
                )

    proposer_reward_denominator = (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) // PROPOSER_WEIGHT
    proposer_reward = Gwei(proposer_reward_basis // proposer_reward_denominator)
    increase_balance(state, get_beacon_proposer_index(state), proposer_reward)

    # [Modified in Simplex]
    # Write back updated builder-payment participation (epoch-structured buffer)
    if compute_epoch_at_slot(data.slot) == get_current_epoch(state):
        state.builder_pending_payments[SLOTS_PER_EPOCH + data.slot % SLOTS_PER_EPOCH] = payment
    else:
        state.builder_pending_payments[data.slot % SLOTS_PER_EPOCH] = payment

    # The Gloas epoch transition rotates all residual payments created in the
    # final Gloas epoch into the first half of the ring before
    # ``upgrade_to_simplex`` runs. A matching activation-slot head vote is a
    # native Simplex certificate that the entire inherited chain was observed.
    # Apply its exact normalized seat positions to every remaining claim in
    # that transition-only half. ``get_current_epoch(state) == state.fork.epoch``
    # above prevents a late replay from touching the next epoch's rotated
    # post-fork payments. This runs after the slot-local writeback so an SSZ
    # child view retained in ``payment`` cannot overwrite sibling ring edits.
    if is_transition_receipt and is_matching_head:
        for payment_index in range(SLOTS_PER_EPOCH):
            legacy_payment = state.builder_pending_payments[payment_index]
            if legacy_payment.withdrawal.amount == 0:
                continue
            for position in attesting_positions:
                legacy_payment.available_participation[position] = True
            state.builder_pending_payments[payment_index] = legacy_payment
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
    # Evidence has no age limit, but the signed messages must still correspond
    # to slots in which the Simplex attestation type and duty were active.
    assert is_attestation_from_active_simplex_fork(state, attestation_1.data)
    assert is_attestation_from_active_simplex_fork(state, attestation_2.data)
    # Verify same round, different data
    assert compute_round_at_slot(attestation_1.data.slot) == compute_round_at_slot(
        attestation_2.data.slot
    )
    assert attestation_1.data != attestation_2.data
    # E1/E2 offenses use AttesterSlashing and its full slashing penalty. This
    # operation is only the lighter fallback for a same-round pair that is not
    # slashable under either accountable-safety condition.
    assert not is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    # Evidence cannot prove a duty that has not yet had a block-inclusion
    # opportunity. Past evidence remains valid indefinitely, like slashings.
    assert attestation_1.data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    assert attestation_2.data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot
    # Verify signatures
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)

    offenders = set(attestation_1.attesting_indices) & set(attestation_2.attesting_indices)
    assert len(offenders) > 0
    penalized_any = False
    for index in sorted(offenders):
        validator = state.validators[index]
        if not state.round_double_vote_penalized[index]:
            state.round_double_vote_penalized[index] = True
            penalized_any = True
            # Initiate exit if needed (NOT slashed). A validator that already
            # exited voluntarily is still charged once for this offense.
            if validator.exit_epoch == FAR_FUTURE_EPOCH:
                initiate_validator_exit(state, ValidatorIndex(index))
            # Fixed penalty: one epoch's worth of base reward
            penalty = get_base_reward(state, ValidatorIndex(index))
            decrease_balance(state, ValidatorIndex(index), penalty)
            # Proposer reward
            proposer_reward = Gwei(penalty // PROPOSER_REWARD_QUOTIENT)
            increase_balance(state, get_beacon_proposer_index(state), proposer_reward)
    # Reject disjoint evidence and replays/no-ops, matching the base slashing
    # operation convention that an included proof must newly affect someone.
    assert penalized_any
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
    # body.anchor_root is deliberately NOT processed here. It has no state
    # effect; fork choice evaluates it from locally received previous-round
    # attestations, and an unaccepted pointer never invalidates the block.
```

## Fork transition

### New `upgrade_to_simplex`

*Note*: The current height's start slot is set to the latest block header slot
so that the first fresh-vote gate references the pre-fork tip.

Gloas epoch processing runs before this upgrade at the activation boundary. It
first resolves the older payment half, then rotates every residual payment from
the final Gloas epoch into `builder_pending_payments[:SLOTS_PER_EPOCH]`.
`upgrade_to_simplex` preserves each legacy weight and initializes its available
bitmap empty. The activation-slot available committee can fill that first half
through the transition-receipt rule in `process_available_attestation`; the
second half is reserved for payments created during the activation epoch. This
native Simplex receipt substitutes later canonical-ancestry evidence for the
legacy slot-local electorate; it intentionally broadens eligibility as described
above rather than pretending to reconstruct an incompatible Gloas attestation.
The inherited `apply_parent_execution_payload` path remains complementary: a
Simplex descendant importing the FULL parent settles its payment immediately,
independently of either attestation quorum.

The same boundary leaves one Gloas participation interval financially open.
Gloas epoch processing pays `previous_epoch_participation` and only afterward
rotates the just-ended epoch's `current_epoch_participation` into that field.
Consequently, at activation `pre.previous_epoch_participation` is the final
Gloas epoch and has not yet received its flag rewards/penalties or inactivity
update. The upgrade settles that interval once, on a copy, with the Gloas
accounting functions before starting empty Simplex round arrays. This is an
explicit transition settlement: the old epoch-FFG certificate state is not
imported into the new finality gadget, but earned monetary accounting is not
silently canceled or reinterpreted under per-round Simplex rules.

```python
def upgrade_to_simplex(pre: gloas.BeaconState) -> BeaconState:
    epoch = gloas.get_current_epoch(pre)
    # Settle the just-ended, newly rotated Gloas participation interval without
    # mutating the caller's pre-state. Gloas order is inactivity update first,
    # then flag rewards/penalties (including its inactivity penalty).
    accounting_pre = copy(pre)
    gloas.process_inactivity_updates(accounting_pre)
    gloas.process_rewards_and_penalties(accounting_pre)
    # The inherited root commits to an FFG epoch boundary, which may be an
    # empty slot. BeaconState cannot recover the original proposal slot for an
    # arbitrarily old carried root, so height-0 checkpoints deliberately retain
    # legacy boundary semantics. The first post-fork justification replaces J
    # with an exact proposal-slot checkpoint; a legacy sentinel that is not an
    # actual boundary block cannot be used as a new vote/finality target.
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
        balances=accounting_pre.balances,
        # Randomness
        randao_mixes=pre.randao_mixes,
        # Slashings
        slashings=pre.slashings,
        # Participation
        # The final Gloas interval was settled explicitly above. Simplex starts
        # a new round-accounting domain; copying either legacy array would pay
        # or penalize it under incompatible rules.
        previous_round_participation=[
            ParticipationFlags(0b0000_0000) for _ in range(len(pre.validators))
        ],
        current_round_participation=[
            ParticipationFlags(0b0000_0000) for _ in range(len(pre.validators))
        ],
        # Finality [Modified in Simplex]
        # Removed: justification_bits, previous_justified_checkpoint, current_justified_checkpoint
        # Convert epoch-based Checkpoints to slot-based
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=Checkpoint(
            slot=compute_start_slot_at_epoch(pre.finalized_checkpoint.epoch),
            root=pre.finalized_checkpoint.root,
        ),
        # Inactivity
        inactivity_scores=accounting_pre.inactivity_scores,
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
        builder_pending_payments=[
            BuilderPendingPayment(
                weight=payment.weight,
                available_participation=Bitvector[AVAILABLE_COMMITTEE_SIZE](),
                timely_head_participation=Bitvector[AVAILABLE_COMMITTEE_SIZE](),
                withdrawal=payment.withdrawal,
            )
            for payment in pre.builder_pending_payments
        ],
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
        current_height_nonjustifiable=is_nonjustifiable_height(
            GENESIS_HEIGHT,
            Height(0),
        ),
        # The fork-slot block, if any, is the first eligible target. Starting at
        # ``pre.slot`` prevents use of the latest pre-fork block as the target.
        current_height_start_slot=pre.slot,
        current_height_target=Checkpoint(),
        pending_height_outcomes=uint64(0),
        target_participation=Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * len(pre.validators)),
        timeouts=Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * len(pre.validators)),
        finality_participation=Bitlist[VALIDATOR_REGISTRY_LIMIT]([False] * len(pre.validators)),
        round_double_vote_penalized=Bitlist[VALIDATOR_REGISTRY_LIMIT](
            [False] * len(pre.validators)
        ),
    )

    post.available_committee_window = initialize_available_committee_window(post)
    return post
```

*Note*: Simplex activates only through `upgrade_to_simplex`; it does not
redefine the genesis-state initialization helper that Altair removed from the
executable specification. Test infrastructure may construct synthetic Simplex
states directly, but that is not a protocol genesis path.

# One-Round Finality -- Honest Validator

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Configuration](#configuration)
  - [Time parameters](#time-parameters)
- [Beacon chain responsibilities](#beacon-chain-responsibilities)
  - [Goldfish vote handling](#goldfish-vote-handling)
  - [Attestation (finality attestation)](#attestation-finality-attestation)
    - [Modified attestation data](#modified-attestation-data)
    - [When to attest](#when-to-attest)
    - [Constructing a finality attestation target](#constructing-a-finality-attestation-target)
    - [Publishing](#publishing)
  - [Available attestation (LMD-GHOST attestation)](#available-attestation-lmd-ghost-attestation)
    - [Constructing available attestation data](#constructing-available-attestation-data)
    - [Signing](#signing)
    - [Publishing](#publishing-1)
  - [Available confirmation (delayed)](#available-confirmation-delayed)
  - [Block proposal](#block-proposal)
    - [Constructing `attestations`](#constructing-attestations)
    - [Constructing `available_attestations`](#constructing-available_attestations)
    - [Constructing `historical_target_proofs` (optional fallback)](#constructing-historical_target_proofs-optional-fallback)
    - [Constructing `attester_slashings`](#constructing-attester_slashings)
- [How to avoid slashing](#how-to-avoid-slashing)

<!-- mdformat-toc end -->

## Introduction

This document describes validator behavior for one-round finality. Validators
have two distinct attestation duties: finality attestations (all active
validators per height via standard `Attestation`) and LMD-GHOST attestations
(available committee per slot via `AvailableAttestation`).

*Note*: This specification is built upon [Gloas](../../gloas/validator.md).

## Configuration

### Time parameters

In addition to Gloas timing parameters, one-round finality defines:

| Name                             | Value          |     Unit     |         Duration          |
| -------------------------------- | -------------- | :----------: | :-----------------------: |
| `AVAILABLE_CONFIRMATION_DUE_BPS` | `uint64(5000)` | basis points | 50% of `SLOT_DURATION_MS` |
| `VIEW_FREEZE_DUE_BPS`            | `uint64(7500)` | basis points | 75% of `SLOT_DURATION_MS` |

## Beacon chain responsibilities

All validator responsibilities remain unchanged from Gloas other than the
following:

- **Attestations** carry one-round finality data (`target` and `height`) instead
  of FFG source/target. All active validators attest once per height via
  standard beacon committee attestations (Electra format). Attester slashings
  enforce the height double-attestation condition.
- **Available attestations** are a new duty. The 512-member available committee
  attests per slot for LMD-GHOST fork choice via `AvailableAttestation`.
- **Block proposals** must additionally include `available_attestations` and may
  include one `historical_target_proof` when needed.

### Goldfish vote handling

For one-round-finality Goldfish vote synchronization:

1. Available-attestation votes and payload votes use first-vote + equivocation
   tracking per committee member per slot.
2. Non-proposers consider wire votes only within the local view-merge window.
   Proposers may continue collecting late wire votes until proposal time for
   view-merge inclusion.
3. Block-carried votes are always processed through the same first/equivocation
   transition rule.
4. Payload extension is proposer-independent and threshold-driven: extend `FULL`
   only with local payload availability and strict relative-majority payload
   vote support.

### Attestation (finality attestation)

#### Modified attestation data

[Modified in One-Round Finality] `AttestationData` carries finality attestation
data. The `source` and `index` fields are removed; `beacon_block_root` is
repurposed as an LMD head vote for fork choice, `target` is repurposed as a
one-round finality target, `height` is added, and `payload_present` signals
payload availability for the voted block.

To construct `attestation_data`:

- Set `attestation_data.slot` to the assigned slot.
- Set `attestation_data.beacon_block_root` to the validator's current head block
  root from fork choice.
- Set `attestation_data.target` to the canonical target for the height being
  attested to (see `get_finality_target` below).
- Set `attestation_data.height` to the finality height being attested to.
- Set `attestation_data.payload_present` to `False` if attesting to the current
  slot's block (PTC does the first payload availability determination), or
  `True`/`False` to signal payload availability of a previous block.

The signing domain is `DOMAIN_BEACON_ATTESTER` at the epoch of the attestation
slot:

```python
def get_attestation_signature(
    state: BeaconState, attestation_data: AttestationData, privkey: int
) -> BLSSignature:
    # [Modified in One-Round Finality] Uses slot epoch (target epoch may differ)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, compute_epoch_at_slot(attestation_data.slot))
    signing_root = compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)
```

*Note*: Only validators with a beacon committee duty at a given slot create
attestations. The beacon committee structure (Electra format with
`committee_bits` + `aggregation_bits`) is used for aggregation and on-chain
inclusion.

#### When to attest

All active validators must cast exactly one finality attestation per height. The
duty to attest is triggered by **beacon committee membership**: at each slot
where the validator has a beacon committee duty (network-level), it checks
whether to cast a finality attestation.

- If not yet attested at the current height: create a finality attestation for
  the current height.
- If already attested at the current height but not at the previous height
  (e.g., missed an attestation before a height transition): create a finality
  attestation for the previous height.
- If already attested at both heights: skip (no finality attestation this slot).

*Note*: Beacon committees are the standard multi-committee-per-slot assignment
used for subnet routing (same as Gloas). They are distinct from the available
committee, which is the small 512-member committee used for on-chain LMD
attestations.

#### Constructing a finality attestation target

Each height has a **canonical target** -- the epoch boundary block at the epoch
when the height started, stored as a full checkpoint in
`state.current_height_canonical_target`. All validators should attest to this
canonical target regardless of which epoch they attest in. Attesting to the
canonical target is the only way to avoid inactivity leak penalties during
non-finality.

```python
def get_finality_target(
    state: BeaconState, height: Height, slot: Slot, head_root: Root, payload_present: boolean
) -> AttestationData:
    """
    Construct the canonical finality attestation for the given height at the given slot.
    ``head_root`` is the validator's current head block root from fork choice.
    ``payload_present`` signals payload availability for the head block:
    False for same-slot blocks (PTC handles first determination), True/False for older blocks.
    """
    if height == state.current_height:
        target = state.current_height_canonical_target
    else:
        target = state.previous_height_canonical_target
    return AttestationData(
        slot=slot,
        beacon_block_root=head_root,
        target=target,
        height=height,
        payload_present=payload_present,
    )
```

#### Publishing

The validator signs the attestation and publishes it for block inclusion.
Attestations use the standard Electra committee-based encoding:

1. Determine the validator's beacon committee index and position within that
   committee at `data.slot`.
2. Set `committee_bits` to flag the validator's committee index.
3. Set `aggregation_bits` to flag the validator's position within the committee.

When aggregating, multiple attestations for the same `AttestationData` (same
slot, target, height) are merged: OR the `aggregation_bits` within the same
committee, and set `committee_bits` to flag all included committees. BLS
signatures are aggregated accordingly.

### Available attestation (LMD-GHOST attestation)

Validators in the available committee for a given slot create LMD-GHOST
attestations via `AvailableAttestation`. The available committee is a 512-member
committee selected per slot via `get_available_committee(state, slot)`.

#### Constructing available attestation data

To construct `available_attestation_data`:

- Set `available_attestation_data.slot` to the assigned slot.
- Compute `vote_head = get_head(store)`.
- Set `available_attestation_data.beacon_block_root = vote_head.root`.
- Set `available_attestation_data.payload_present` to `False` if
  `store.blocks[vote_head.root].slot == slot`; otherwise signal local payload
  availability for `vote_head.root` (`True` or `False`).

`get_head(store)` includes a current-slot proposal pass-through in Goldfish
refinement: when a pending child is from the current slot, it may be selected
even before it has majority-qualified previous-slot support. This is required
beyond genesis/anchor bootstrap; otherwise, freshly proposed children cannot
accumulate the available-attestation support needed to become canonical.

#### Signing

```python
def get_available_attestation_signature(
    state: BeaconState, data: AvailableAttestationData, privkey: int
) -> BLSSignature:
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, compute_epoch_at_slot(data.slot))
    signing_root = compute_signing_root(data, domain)
    return bls.Sign(privkey, signing_root)
```

#### Publishing

The validator constructs an `AvailableAttestation`:

1. Determine the validator's position within the available committee at
   `data.slot`.
2. Set `aggregation_bits` to flag the validator's position. Bit `i` corresponds
   to index `i` in `get_available_committee(state, data.slot)`.

When aggregating, multiple attestations for the same `AvailableAttestationData`
are merged by OR-ing `aggregation_bits` and BLS-aggregating signatures.

### Available confirmation (delayed)

One-round finality supports a delayed available-confirmation step for slot `n`
in slot `n+1`.

- The check is run in the payload-vote/confirm phase of slot `n+1` (the `t_ptc`
  phase in the slot timing model).
- Membership in the timely set is fixed by the delayed-confirm timely cutoff: a
  committee member is timely if its first vote was seen before
  `get_payload_attestation_due_ms(epoch)`. This gives propagation slack after
  the attestation-send deadline, so votes sent at the deadline can still be
  counted timely.
- At slot transition, this timely set is carried into
  `previous_available_timely_attesters`.
- Confirmation then runs over previous-slot votes using
  `get_available_confirmation_head(store)`.

`get_available_confirmation_head(store)` applies strict relative-majority
viability (`> n/2`) using only timely previous-slot participants and
deletion-only equivocation handling.

At this stage, this delayed available-confirmation output is external (research
signal). It can later be used as a precondition for finality/stabilization
voting without changing the underlying vote-tracking rule.

### Block proposal

#### Constructing `attestations`

Up to `MAX_ATTESTATIONS_ELECTRA` aggregate finality attestations can be included
in the block. The block proposer should:

1. Collect attestation aggregates available to the proposer.
2. Include only attestations that satisfy `process_attestation` validation.
3. Aggregate all attestations with the same `AttestationData` into a single
   `Attestation` using the Electra committee-based pattern: sort by committee
   index, concatenate `aggregation_bits` from each committee (OR-ing within the
   same committee), set `committee_bits` to flag all included committees, and
   BLS-aggregate signatures.

*Note*: Each `Attestation` covers one slot's committees. Attestations from
different slots (even with the same target and height) cannot be merged into a
single `Attestation` because different slots have different committee
structures. Over an epoch, the proposer spreads finality attestation inclusion
across blocks.

#### Constructing `available_attestations`

Up to `MAX_AVAILABLE_ATTESTATIONS` aggregate available attestations can be
included in the block. Following the same pattern as Gloas payload-attestation
aggregation, the proposer should:

1. Collect available attestation aggregates available to the proposer.
2. Include only attestations that satisfy `process_available_attestation`
   validation.
3. Aggregate all attestations with the same `AvailableAttestationData` into a
   single `AvailableAttestation` by OR-ing `aggregation_bits`.

One-round finality uses a single available committee per slot for on-chain LMD
attestations, so no `committee_bits` are needed. Bit `i` in `aggregation_bits`
corresponds to the validator at index `i` in
`get_available_committee(state, data.slot)`:

```python
def compute_on_chain_aggregate(
    network_aggregates: Sequence[AvailableAttestation],
) -> AvailableAttestation:
    data = network_aggregates[0].data
    aggregation_bits = Bitvector[AVAILABLE_COMMITTEE_SIZE](
        False for _ in range(AVAILABLE_COMMITTEE_SIZE)
    )

    for a in network_aggregates:
        assert a.data == data
        assert len(a.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
        for i, bit in enumerate(a.aggregation_bits):
            aggregation_bits[i] = aggregation_bits[i] or bit

    signature = bls.Aggregate([a.signature for a in network_aggregates])

    return AvailableAttestation(
        aggregation_bits=aggregation_bits,
        data=data,
        signature=signature,
    )
```

#### Constructing `historical_target_proofs` (optional fallback)

When a target that may justify in this block is outside the `block_roots`
window, the proposer may include a `HistoricalTargetProof`:

1. Select an out-of-window target checkpoint `(epoch, root)` that is expected to
   justify in this block.
2. Build a Merkle branch from `root` at
   `compute_start_slot_at_epoch(epoch) % SLOTS_PER_HISTORICAL_ROOT` to the
   matching
   `historical_summaries[slot // SLOTS_PER_HISTORICAL_ROOT].block_summary_root`.
3. Include `HistoricalTargetProof(target, block_root_proof)` in
   `body.historical_target_proofs` (at most one per block).

The proof path is valid only for out-of-window targets and is strict-use: if a
proof is included, it must be consumed by an actual justification in that block;
otherwise the block is invalid.

#### Constructing `attester_slashings`

If the proposer detects two attestations from the same validator that satisfy
`is_slashable_attestation_data` (epoch double-vote or height target conflict),
they should:

1. Construct two `IndexedAttestation` objects from the conflicting attestations.
2. Verify that `is_slashable_attestation_data` returns `True`.
3. Include the `AttesterSlashing` in the block (up to
   `MAX_ATTESTER_SLASHINGS_ELECTRA`).

*Note*: Processing remains inherited from `on_attester_slashing` in Gloas; in
one-round finality, the changed behavior comes from
`is_slashable_attestation_data` (height-aware slashing conditions).

## How to avoid slashing

`AvailableAttestationData` is not slashable. The slashing conditions for
attesting are:

1. **Epoch double-vote**: Sign two different `AttestationData` messages in the
   same epoch.
2. **Height target conflict**: Sign two `AttestationData` messages at the same
   `height` with different `target` (even across epochs).

*With one-round finality, a validator signs exactly one `AttestationData` per
epoch and never changes their target for a given height. Across epochs, the same
finality vote `(height, target)` may be repeated with different fork-choice
fields (`beacon_block_root`, `payload_present`).*

Specifically:

- When signing an `AttestationData`:

  1. Save a record to hard disk of the full `AttestationData` that has been
     signed, keyed by `(epoch, height)`.
  2. Do not sign if any different `AttestationData` was already signed in this
     epoch.
  3. Do not sign if the same `height` was already signed with a different
     `target` (in any epoch).
  4. Generate and broadcast the attestation.

- When signing an `AvailableAttestationData`:

  1. Save a record to hard disk that an available attestation has been signed
     for this slot.
  2. Generate and broadcast the available attestation.

If the software crashes at some point within this routine, then when the
validator comes back online, the hard disk has the record of the *potentially*
signed/broadcast message and can effectively avoid slashing.

*Note*: Surround voting is no longer possible since FFG source/target are
removed. Available attestation data is non-slashable. The epoch double-vote and
height target conflict are the slashing conditions handled by
`AttesterSlashing`.

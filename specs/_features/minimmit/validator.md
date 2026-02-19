# Minimmit -- Honest Validator

<!-- mdformat-toc start --no-anchors -->

---

<!-- mdformat-toc end -->

## Introduction

This document describes validator behavior for one-round finality. Validators
have two distinct voting duties: finality attestations (all active validators
per height via standard `Attestation`) and LMD-GHOST attestations (available
committee per slot via `AvailableAttestation`).

*Note*: This specification is built upon [Gloas](../../gloas/validator.md).

## Beacon chain responsibilities

All validator responsibilities remain unchanged from Gloas other than the
following:

- **Attestations** carry one-round finality vote data (`target` and `height`)
  instead of FFG source/target. All active validators vote once per height via
  standard beacon committee attestations (Electra format). Attester slashings
  enforce the height double-vote condition.
- **Available attestations** are a new duty. The 512-member available committee
  votes per slot for LMD-GHOST fork choice via `AvailableAttestation`.
- **Block proposals** must additionally include `available_attestations` and may
  include one `historical_target_proof` when needed.

### Attestation (finality vote)

#### Modified attestation data

[Modified in Minimmit] `AttestationData` carries finality vote data. The
`source`, `index`, and `beacon_block_root` fields are removed; `target` is
repurposed as a one-round finality target, and `height` is added.

To construct `attestation_data`:

- Set `attestation_data.slot` to the assigned slot.
- Set `attestation_data.target` to the canonical target for the height being
  voted on (see `get_finality_target` below).
- Set `attestation_data.height` to the finality height being voted on.

The signing domain is `DOMAIN_BEACON_ATTESTER` at the epoch of the attestation
slot:

```python
def get_attestation_signature(
    state: BeaconState, attestation_data: AttestationData, privkey: int
) -> BLSSignature:
    # [Modified in Minimmit] Uses slot epoch (target epoch may differ)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, compute_epoch_at_slot(attestation_data.slot))
    signing_root = compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)
```

*Note*: Only validators with a beacon committee duty at a given slot create
attestations. The beacon committee structure (Electra format with
`committee_bits` + `aggregation_bits`) is used for aggregation and on-chain
inclusion.

#### When to vote

All active validators must cast exactly one finality vote per height. The duty
to vote is triggered by **beacon committee membership**: at each slot where the
validator has a beacon committee duty (network-level), it checks whether to cast
a finality vote.

- If not yet voted at the current height: create a finality vote for the current
  height.
- If already voted at the current height but not at the previous height (e.g.,
  missed a vote before a height transition): create a finality vote for the
  previous height.
- If already voted at both heights: skip (no finality vote this slot).

*Note*: Beacon committees are the standard multi-committee-per-slot assignment
used for subnet routing (same as Gloas). They are distinct from the available
committee, which is the small 512-member committee used for on-chain LMD
attestations.

#### Constructing a finality vote target

Each height has a **canonical target** -- the epoch boundary block at the epoch
when the height started, stored as a full checkpoint in
`state.current_height_canonical_target`. All validators should vote for this
canonical target regardless of which epoch they attest in. Voting for the
canonical target is the only way to avoid inactivity leak penalties during
non-finality.

```python
def get_finality_target(state: BeaconState, height: Height, slot: Slot) -> AttestationData:
    """
    Construct the canonical finality vote for the given height at the given slot.
    """
    if height == state.current_height:
        target = state.current_height_canonical_target
    else:
        target = state.previous_height_canonical_target
    return AttestationData(
        slot=slot,
        target=target,
        height=height,
    )
```

#### Publishing

The validator signs the attestation and publishes it for block inclusion.
Attestations use the standard Electra committee-based encoding:

1. Determine the validator's beacon committee index and position within that
   committee at `data.slot`.
2. Set `committee_bits` to flag the validator's committee index.
3. Set `aggregation_bits` to flag the validator's position within the committee.

When aggregating, multiple votes for the same `AttestationData` (same slot,
target, height) are merged: OR the `aggregation_bits` within the same committee,
and set `committee_bits` to flag all included committees. BLS signatures are
aggregated accordingly.

### Available attestation (LMD-GHOST vote)

Validators in the available committee for a given slot create LMD-GHOST votes
via `AvailableAttestation`. The available committee is a 512-member committee
selected per slot via `get_available_committee(state, slot)`.

#### Constructing available attestation data

To construct `available_attestation_data`:

- Set `available_attestation_data.slot` to the assigned slot.
- Set `available_attestation_data.index` as in Gloas: `0` if attesting to the
  current slot's block, or `0`/`1` to signal payload status of a previous block.
- Set `available_attestation_data.beacon_block_root` to the head block root from
  the validator's fork choice.

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

When aggregating, multiple votes for the same `AvailableAttestationData` are
merged by OR-ing `aggregation_bits` and BLS-aggregating signatures.

### Block proposal

#### Constructing `attestations`

Up to `MAX_ATTESTATIONS_ELECTRA` aggregate finality attestations can be included
in the block. The block proposer should:

1. Collect attestation aggregates available to the proposer.
1. Include only attestations that satisfy `process_attestation` validation.
1. Aggregate all attestations with the same `AttestationData` into a single
   `Attestation` using the Electra committee-based pattern: sort by committee
   index, concatenate `aggregation_bits` from each committee (OR-ing within the
   same committee), set `committee_bits` to flag all included committees, and
   BLS-aggregate signatures.

*Note*: Each `Attestation` covers one slot's committees. Votes from different
slots (even with the same target and height) cannot be merged into a single
`Attestation` because different slots have different committee structures. Over
an epoch, the proposer spreads finality vote inclusion across blocks.

#### Constructing `available_attestations`

Up to `MAX_AVAILABLE_ATTESTATIONS` aggregate available attestations can be
included in the block. Following the same pattern as Gloas payload-attestation
aggregation, the proposer should:

1. Collect available attestation aggregates available to the proposer.
1. Include only attestations that satisfy `process_available_attestation`
   validation.
1. Aggregate all attestations with the same `AvailableAttestationData` into a
   single `AvailableAttestation` by OR-ing `aggregation_bits`.

Minimmit uses a single available committee per slot for on-chain LMD
attestations, so no `committee_bits` are needed. Bit `i` in `aggregation_bits`
corresponds to the validator at index `i` in
`get_available_committee(state, data.slot)`:

```python
def compute_on_chain_aggregate(network_aggregates: Sequence[AvailableAttestation]) -> AvailableAttestation:
    data = network_aggregates[0].data
    bit_count = len(network_aggregates[0].aggregation_bits)
    aggregation_bits = Bitlist[AVAILABLE_COMMITTEE_SIZE](False for _ in range(bit_count))

    for a in network_aggregates:
        assert a.data == data
        assert len(a.aggregation_bits) == bit_count
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
1. Build a Merkle branch from `root` at
   `compute_start_slot_at_epoch(epoch) % SLOTS_PER_HISTORICAL_ROOT` to the
   matching `historical_summaries[slot // SLOTS_PER_HISTORICAL_ROOT].block_summary_root`.
1. Include `HistoricalTargetProof(target, block_root_proof)` in
   `body.historical_target_proofs` (at most one per block).

The proof path is valid only for out-of-window targets and is strict-use: if a
proof is included, it must be consumed by an actual justification in that
block; otherwise the block is invalid.

#### Constructing `attester_slashings`

If the proposer detects two conflicting attestations at the same height from
the same validator, they should:

1. Construct two `IndexedAttestation` objects from the conflicting votes.
1. Verify that `is_slashable_attestation_data` returns `True`.
1. Include the `AttesterSlashing` in the block (up to
   `MAX_ATTESTER_SLASHINGS_ELECTRA`).

## How to avoid slashing

`AvailableAttestationData` is not slashable. The only slashing condition for
voting is the attestation height double vote:

1. **Height double vote**: Sign two different `AttestationData` messages at the
   same height (different slot or different target).

*With one-round finality, a validator is safe as long as they cast only one
attestation per height.*

Specifically:

- When signing an `AttestationData`:

  1. Save a record to hard disk that an attestation has been signed for this
     height (i.e., `data.height`).
  1. Generate and broadcast the attestation.

- When signing an `AvailableAttestationData`:

  1. Save a record to hard disk that an available attestation has been signed for
     this slot.
  1. Generate and broadcast the available attestation.

If the software crashes at some point within this routine, then when the
validator comes back online, the hard disk has the record of the *potentially*
signed/broadcast message and can effectively avoid slashing.

*Note*: Surround voting is no longer possible since FFG source/target are
removed. Available attestation data is non-slashable. The height double vote is
the slashing condition handled by `AttesterSlashing`.

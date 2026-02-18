# Minimmit -- Honest Validator

<!-- mdformat-toc start --no-anchors -->

---

<!-- mdformat-toc end -->

## Introduction

This document describes validator behavior for one-round finality. Validators
have two distinct voting duties: LMD-GHOST attestations (available committee per
slot) and finality attestations (all active validators per height).

*Note*: This specification is built upon [Gloas](../../gloas/validator.md).

## Beacon chain responsibilities

All validator responsibilities remain unchanged from Gloas other than the
following:

- **LMD attestations** no longer include source or target fields. The
  `AttestationData` contains only `slot`, `index`, and `beacon_block_root`.
- **LMD attester slashings** are removed in Minimmit.
- **Finality attestations** are a new duty. All active validators vote once per
  height via `FinalityAttestation`.
- **Block proposals** must additionally include `finality_attestations` and
  `finality_slashings`, and may include one `historical_target_proof` when
  needed.

### Attestation

#### Modified attestation data

[Modified in Minimmit] The `source` and `target` fields are removed from
`AttestationData`. Attestations are now pure LMD-GHOST votes.

To construct `attestation_data`:

- Set `attestation_data.slot` to the assigned slot.
- Set `attestation_data.index` as in Gloas: `0` if attesting to the current
  slot's block, or `0`/`1` to signal payload status of a previous block.
- Set `attestation_data.beacon_block_root` to the head block root from the
  validator's fork choice.

The signing domain is `DOMAIN_BEACON_ATTESTER` at the epoch of the attestation
slot:

```python
def get_attestation_signature(
    state: BeaconState, attestation_data: AttestationData, privkey: int
) -> BLSSignature:
    # [Modified in Minimmit] Uses slot epoch (no target epoch)
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, compute_epoch_at_slot(attestation_data.slot))
    signing_root = compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)
```

*Note*: Only validators in the available committee for a given slot create
LMD attestations. The available committee is a 512-member committee selected
per slot via `get_available_committee(state, slot)`.

### Finality attestation

All active validators participate in finality attestation by voting once per
height. The validator client must track locally which height it has already
voted for. Finality votes can be cast for either the current height or the
previous height (to allow late votes to still count).

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
def get_finality_target(state: BeaconState, height: Height) -> FinalityAttestationData:
    """
    Construct the canonical finality vote for the given height.
    """
    if height == state.current_height:
        target = state.current_height_canonical_target
    else:
        target = state.previous_height_canonical_target
    return FinalityAttestationData(
        target=target,
        height=height,
    )
```

#### Signing

```python
def get_finality_attestation_signature(
    state: BeaconState, data: FinalityAttestationData, privkey: int
) -> BLSSignature:
    domain = get_domain(state, DOMAIN_FINALITY_ATTESTER, data.target.epoch)
    signing_root = compute_signing_root(data, domain)
    return bls.Sign(privkey, signing_root)
```

#### Publishing

The validator signs the finality vote and makes it available for block
inclusion. Transport is currently out of scope in this branch: finality votes
are included via block `finality_attestations`, and no dedicated
`finality_attestation` gossip topic is specified yet. When aggregating, bit `i`
in `aggregation_bits` corresponds to the `i`-th active validator in
`get_active_validator_indices(state, current_epoch)`.

### Block proposal

#### Constructing `attestations`

Up to `MAX_ATTESTATIONS_ELECTRA` aggregate LMD attestations can be included in
the block. Following the same pattern as Gloas payload-attestation aggregation,
the proposer should:

1. Collect attestation aggregates available to the proposer.
1. Include only attestations that satisfy `process_attestation` validation.
1. Aggregate all attestations with the same `AttestationData` into a single
   `Attestation` by OR-ing `aggregation_bits`.

Minimmit uses a single available committee per slot for on-chain LMD
attestations, so no `committee_bits` are needed. Bit `i` in `aggregation_bits`
corresponds to the validator at index `i` in
`get_available_committee(state, data.slot)`:

```python
def compute_on_chain_aggregate(network_aggregates: Sequence[Attestation]) -> Attestation:
    data = network_aggregates[0].data
    bit_count = len(network_aggregates[0].aggregation_bits)
    aggregation_bits = Bitlist[AVAILABLE_COMMITTEE_SIZE](False for _ in range(bit_count))

    for a in network_aggregates:
        assert a.data == data
        assert len(a.aggregation_bits) == bit_count
        for i, bit in enumerate(a.aggregation_bits):
            aggregation_bits[i] = aggregation_bits[i] or bit

    signature = bls.Aggregate([a.signature for a in network_aggregates])

    return Attestation(
        aggregation_bits=aggregation_bits,
        data=data,
        signature=signature,
    )
```

#### Constructing `finality_attestations`

Up to `MAX_FINALITY_ATTESTATIONS` aggregate finality attestations can be
included in the block. The block proposer should:

1. Collect individual finality votes available to the proposer (transport is
   implementation-specific and out of scope here).
1. Aggregate votes with the same `FinalityAttestationData` into a single
   `FinalityAttestation` by setting the corresponding bits in
   `aggregation_bits` (bit `i` = `i`-th active validator by
   `get_active_validator_indices` ordering).
1. Include aggregated finality attestations that satisfy the validation
   conditions in `process_finality_attestation`:
   - The height must be the current or previous height.
   - The bitfield length must equal the number of active validators.
   - The aggregate signature must verify.

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

#### Constructing `finality_slashings`

If the proposer detects two conflicting finality votes at the same height from
the same validator, they should:

1. Construct two `IndexedFinalityAttestation` objects from the conflicting
   votes.
1. Verify that `is_slashable_finality_attestation_data` returns `True`.
1. Include the `FinalitySlashing` in the block (up to
   `MAX_FINALITY_SLASHINGS`).

## How to avoid slashing

LMD `AttestationData` is not slashable in Minimmit. The only slashing condition
for voting is finality double vote:

1. **Height double vote**: Sign two different `FinalityAttestationData` messages
   at the same height.

*With one-round finality, a validator is safe as long as they cast only one
finality vote per height.*

Specifically:

- When signing an `AttestationData`:

  1. Save a record to hard disk that an attestation has been signed for this slot.
  1. Generate and broadcast the attestation.

- When signing a `FinalityAttestationData`:

  1. Save a record to hard disk that a finality vote has been signed for this
     height (i.e., `data.height`).
  1. Generate and broadcast the finality attestation.

If the software crashes at some point within this routine, then when the
validator comes back online, the hard disk has the record of the *potentially*
signed/broadcast message and can effectively avoid slashing.

*Note*: With the removal of source/target from `AttestationData`, surround
voting is no longer possible. LMD attestation data itself is non-slashable in
Minimmit. The finality height double vote is the slashing condition handled by
`FinalitySlashing`.

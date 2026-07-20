# Simplex -- P2P Interface

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Helpers](#helpers)
  - [Modified `Seen`](#modified-seen)
  - [Modified `compute_fork_version`](#modified-compute_fork_version)
  - [New `is_checkpoint_on_store_chain`](#new-is_checkpoint_on_store_chain)
  - [New `is_status_finalized_checkpoint_compatible`](#new-is_status_finalized_checkpoint_compatible)
  - [New `validate_attestation_data_gossip`](#new-validate_attestation_data_gossip)
  - [New `get_available_attestation_checkpoint_state`](#new-get_available_attestation_checkpoint_state)
  - [New `has_new_attestation_evidence`](#new-has_new_attestation_evidence)
  - [New `record_attestation_evidence`](#new-record_attestation_evidence)
  - [New `on_gossip_single_attestation`](#new-on_gossip_single_attestation)
- [The gossip domain: gossipsub](#the-gossip-domain-gossipsub)
  - [Topics and messages](#topics-and-messages)
    - [Global topics](#global-topics)
      - [Modified `beacon_block`](#modified-beacon_block)
      - [Modified `beacon_aggregate_and_proof`](#modified-beacon_aggregate_and_proof)
      - [Modified `execution_payload`](#modified-execution_payload)
      - [New `available_attestation`](#new-available_attestation)
    - [Attestation subnets](#attestation-subnets)
      - [Modified `beacon_attestation_{subnet_id}`](#modified-beacon_attestation_subnet_id)
- [The Req/Resp domain](#the-reqresp-domain)
  - [Messages](#messages)
    - [New Status v3](#new-status-v3)
    - [Modified block response mappings](#modified-block-response-mappings)
    - [Modified execution-payload-envelope response mappings](#modified-execution-payload-envelope-response-mappings)
    - [Modified data-column response mappings](#modified-data-column-response-mappings)

<!-- mdformat-toc end -->

## Introduction

This document extends the
[Gloas -- P2P Interface](../../gloas/p2p-interface.md). All inherited networking
behavior remains unchanged unless explicitly overridden here.

Simplex finality attestations are one-per-round duties, use slot checkpoints,
and immediately update the per-validator latest-head view as soon as a valid
wire message is received. Committee and signature validation is resolved on the
attestation's own head chain. Available attestations use a new global topic
because their fixed-size committee is independent of the beacon-attestation
subnet committees.

In the first slot of a round, networking and validator duties use a strict local
event order at `ATTESTATION_DUE_BPS_GLOAS`: process all valid round-start
proposal copies admitted to fork choice and all proposal support delivered by
the boundary; call `freeze_stable_root(store)` to fix either the accepted
pointer or the then-current G1 fallback; snapshot that stable-root head, the
safe-confirmed root, and the finalized root into a local `RoundSelectionEvent`;
freeze each managed validator's round vote from that event; then construct any
first-slot finality attestations. A proposal does not trigger an earlier freeze.
A proposal or support message delivered exactly at the deadline is processed
first; a strictly later message may affect other live state but cannot change
that round's stable root or frozen FG fields. `Store.time` is not used as proof of
this boundary because it has only whole-second resolution and the minimal preset
has a subsecond-aligned deadline.

## Helpers

### Modified `Seen`

The epoch-keyed attestation caches are replaced by round-keyed caches. The
single-attestation cache retains the first data root and permits one distinct
second message so that receivers can learn and forward round-equivocation
evidence. Further messages from a known round equivocator are ignored. Clients
MUST prune round entries after the corresponding attestations leave the
`LATEST_MESSAGE_EXPIRY_SLOTS` window, and slot entries after their Goldfish use
has passed.

```python
@dataclass
class Seen:
    proposer_slots: Set[Tuple[ValidatorIndex, Slot]]
    # [Modified in Simplex]
    aggregator_rounds: Set[Tuple[ValidatorIndex, Round]]
    aggregate_data_roots: Dict[Tuple[Root, CommitteeIndex], Set[Tuple[boolean, ...]]]
    voluntary_exit_indices: Set[ValidatorIndex]
    proposer_slashing_indices: Set[ValidatorIndex]
    attester_slashing_indices: Set[ValidatorIndex]
    # [Modified in Simplex]
    attestation_validator_round_data_roots: Dict[Tuple[ValidatorIndex, Round], Root]
    attestation_validator_round_equivocations: Set[Tuple[ValidatorIndex, Round]]
    sync_contribution_aggregator_slots: Set[Tuple[ValidatorIndex, Slot, uint64]]
    sync_contribution_data: Dict[Tuple[Slot, Root, uint64], Set[Tuple[boolean, ...]]]
    sync_message_validator_slots: Set[Tuple[Slot, ValidatorIndex, uint64]]
    bls_to_execution_change_indices: Set[ValidatorIndex]
    # [Modified in Simplex]
    data_column_sidecar_tuples: Set[Tuple[Root, ColumnIndex]]
    # [New in Simplex]
    available_attestation_data_roots: Dict[Root, Set[Tuple[boolean, ...]]]
    available_attestation_validator_slot_data_roots: Dict[Tuple[ValidatorIndex, Slot], Root]
    available_attestation_validator_slot_equivocations: Set[Tuple[ValidatorIndex, Slot]]
```

### Modified `compute_fork_version`

```python
def compute_fork_version(epoch: Epoch) -> Version:
    """Return the fork version at ``epoch``."""
    if epoch >= SIMPLEX_FORK_EPOCH:
        return SIMPLEX_FORK_VERSION
    if epoch >= GLOAS_FORK_EPOCH:
        return GLOAS_FORK_VERSION
    if epoch >= FULU_FORK_EPOCH:
        return FULU_FORK_VERSION
    if epoch >= ELECTRA_FORK_EPOCH:
        return ELECTRA_FORK_VERSION
    if epoch >= DENEB_FORK_EPOCH:
        return DENEB_FORK_VERSION
    if epoch >= CAPELLA_FORK_EPOCH:
        return CAPELLA_FORK_VERSION
    if epoch >= BELLATRIX_FORK_EPOCH:
        return BELLATRIX_FORK_VERSION
    if epoch >= ALTAIR_FORK_EPOCH:
        return ALTAIR_FORK_VERSION
    return GENESIS_FORK_VERSION
```

### New `is_checkpoint_on_store_chain`

```python
def is_checkpoint_on_store_chain(store: Store, head_root: Root, checkpoint: Checkpoint) -> bool:
    """Return whether ``checkpoint`` is on ``head_root``'s chain."""
    if checkpoint == Checkpoint():
        return True
    if checkpoint.root not in store.blocks:
        return False

    root_slot = store.blocks[checkpoint.root].slot
    # At activation, the finalized height-0 checkpoint carries an inherited
    # FFG boundary slot even when its root was proposed in an earlier slot.
    # Only this store-owned sentinel has legacy boundary semantics. New
    # attestation checkpoints remain exact-slot references (and are checked as
    # such by ``validate_attestation_data_gossip`` below).
    is_legacy_finalized = checkpoint == store.finalized_checkpoint and root_slot < checkpoint.slot
    if root_slot != checkpoint.slot and not is_legacy_finalized:
        return False
    head = ForkChoiceNode(root=head_root, payload_status=PAYLOAD_STATUS_PENDING)
    ancestor = get_ancestor(store, head, root_slot)
    return ancestor.root == checkpoint.root
```

### New `is_status_finalized_checkpoint_compatible`

Return `False` only when the local store contains enough information to prove
that the peer's finalized checkpoint conflicts with the local finalized chain.
An exact match with `store.finalized_checkpoint` is always compatible. This
includes the activation height-0 legacy boundary sentinel, whose root may have
been proposed before its inherited FFG boundary slot.

When the peer is behind, a known root proposed before an epoch-boundary
`finalized_slot` is treated as a possible legacy sentinel. This exception is
only for Status compatibility: new attestation checkpoints remain exact-slot
references under `validate_attestation_data_gossip`.

```python
def is_status_finalized_checkpoint_compatible(
    store: Store, finalized_root: Root, finalized_slot: Slot
) -> bool:
    """Return whether a peer's finalized checkpoint may share the local chain."""
    peer_finalized = Checkpoint(slot=finalized_slot, root=finalized_root)
    local_finalized = store.finalized_checkpoint

    # This equality covers both ordinary exact-slot checkpoints and the
    # activation height-0 legacy boundary sentinel.
    if peer_finalized == local_finalized:
        return True
    if peer_finalized.slot == local_finalized.slot:
        return False

    if peer_finalized.slot > local_finalized.slot:
        # An unknown checkpoint ahead of local finality is not evidence of a
        # conflict. If known, require the local finalized checkpoint on its
        # chain; ``is_checkpoint_on_store_chain`` recognizes a local sentinel.
        if peer_finalized.root not in store.blocks:
            return True
        if store.blocks[peer_finalized.root].slot != peer_finalized.slot:
            return False
        return is_checkpoint_on_store_chain(store, peer_finalized.root, local_finalized)

    # A missing historical block may have been pruned and does not prove a
    # conflict. For a known older checkpoint, compare ancestry at the root's
    # actual proposal slot. ``root_slot < finalized_slot`` is the inherited
    # height-0 sentinel representation.
    if peer_finalized.root not in store.blocks:
        return True
    peer_root_slot = store.blocks[peer_finalized.root].slot
    local_root_slot = store.blocks[local_finalized.root].slot
    if peer_root_slot > peer_finalized.slot or peer_root_slot > local_root_slot:
        return False
    peer_epoch_boundary = compute_start_slot_at_epoch(compute_epoch_at_slot(peer_finalized.slot))
    if peer_root_slot < peer_finalized.slot and peer_finalized.slot != peer_epoch_boundary:
        return False
    head = ForkChoiceNode(root=local_finalized.root, payload_status=PAYLOAD_STATUS_PENDING)
    ancestor = get_ancestor(store, head, peer_root_slot)
    return ancestor.root == peer_finalized.root
```

### New `validate_attestation_data_gossip`

Unknown referenced blocks are ignored and may be queued for later processing;
known references with an incorrect slot or ancestry are rejected. This wire
policy is intentionally stricter than the fork-choice handler's minimum
prechecks: gossip requires each non-empty target and finality target to be on
the voted-head chain, while non-gossip delivery paths retain their own
validation rules.

```python
def validate_attestation_data_gossip(
    store: Store, state: BeaconState, data: AttestationData
) -> None:
    """Validate the branch references common to Simplex attestation topics."""
    if not is_attestation_from_active_simplex_fork(state, data):
        raise GossipReject("attestation predates the active Simplex fork")

    head_root = data.beacon_block_root
    if head_root not in store.blocks:
        raise GossipIgnore("block being voted for has not been seen")
    if head_root not in store.block_states:
        raise GossipReject("block being voted for failed validation")
    if store.blocks[head_root].slot > data.slot:
        raise GossipReject("block being voted for is later than the attestation")

    # The finality piggyback is either fully empty or fully populated.
    if data.finality_target == Checkpoint():
        if data.finality_height != FAR_FUTURE_HEIGHT:
            raise GossipReject("empty finality target has a non-empty height")
    elif is_empty_vote(data):
        if data.finality_height == FAR_FUTURE_HEIGHT:
            raise GossipReject("non-empty finality target has an empty height")
    elif data.finality_height >= data.height:
        raise GossipReject("finality target height is not below vote height")

    for checkpoint in (data.target, data.finality_target):
        if checkpoint == Checkpoint():
            continue
        if checkpoint.root not in store.blocks:
            raise GossipIgnore("attestation checkpoint has not been seen")
        if store.blocks[checkpoint.root].slot != checkpoint.slot:
            raise GossipReject("attestation checkpoint has an incorrect slot")
        if checkpoint.slot > data.slot:
            raise GossipReject("attestation checkpoint is later than the vote")
        if not is_checkpoint_on_store_chain(store, head_root, checkpoint):
            raise GossipReject("attestation checkpoint is not on the voted chain")

    if not is_checkpoint_on_store_chain(store, head_root, store.finalized_checkpoint):
        raise GossipIgnore("finalized checkpoint is not an ancestor of voted block")
```

### New `get_available_attestation_checkpoint_state`

```python
def get_available_attestation_checkpoint_state(
    store: Store, data: AvailableAttestationData
) -> BeaconState:
    """Return the duty-epoch state on an available vote's own head chain."""
    attestation_epoch = compute_epoch_at_slot(data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    checkpoint_key = (data.beacon_block_root, attestation_epoch)
    if checkpoint_key not in store.checkpoint_states:
        base_state = copy(store.block_states[data.beacon_block_root])
        if base_state.slot < epoch_boundary_slot:
            process_slots(base_state, epoch_boundary_slot)
        store.checkpoint_states[checkpoint_key] = base_state
    return store.checkpoint_states[checkpoint_key]
```

### New `has_new_attestation_evidence`

```python
def has_new_attestation_evidence(
    seen: Seen,
    attesting_indices: Sequence[ValidatorIndex],
    round: Round,
    data_root: Root,
) -> bool:
    """Return whether the message adds a first or second datum for a signer."""
    for index in attesting_indices:
        key = (index, round)
        if key in seen.attestation_validator_round_equivocations:
            continue
        previous_root = seen.attestation_validator_round_data_roots.get(key)
        if previous_root is None or previous_root != data_root:
            return True
    return False
```

### New `record_attestation_evidence`

```python
def record_attestation_evidence(
    seen: Seen,
    attesting_indices: Sequence[ValidatorIndex],
    round: Round,
    data_root: Root,
) -> None:
    """Record at most two distinct data roots for each signer and round."""
    for index in attesting_indices:
        key = (index, round)
        if key in seen.attestation_validator_round_equivocations:
            continue
        previous_root = seen.attestation_validator_round_data_roots.get(key)
        if previous_root is None:
            seen.attestation_validator_round_data_roots[key] = data_root
        elif previous_root != data_root:
            seen.attestation_validator_round_equivocations.add(key)
```

### New `on_gossip_single_attestation`

After `validate_beacon_attestation_gossip` succeeds, clients MUST call this
helper before forwarding the message. This gives the single validator's vote the
same fork-choice delivery path as a validated aggregate without waiting for
aggregation or block inclusion. The helper immediately updates the validator's
latest-head message, including for a same-slot vote.

```python
def on_gossip_single_attestation(store: Store, attestation: SingleAttestation) -> None:
    aggregate = Attestation(data=attestation.data)
    apply_attestation_latest_messages(store, [attestation.attester_index], aggregate)
```

## The gossip domain: gossipsub

### Topics and messages

#### Global topics

##### Modified `beacon_block`

Payload-parent validation uses the locally verified envelopes in
`store.payloads`. If the child selects its parent's bid block hash, the parent
is FULL and its envelope must be verified. Otherwise the parent is EMPTY and the
child must preserve the latest previously applied execution block hash.

```python
def validate_beacon_block_gossip(
    seen: Seen,
    store: Store,
    state: BeaconState,
    signed_beacon_block: SignedBeaconBlock,
    current_time_ms: uint64,
) -> None:
    """Validate a Simplex ``SignedBeaconBlock`` for gossip propagation."""
    block = signed_beacon_block.message
    bid = block.body.signed_execution_payload_bid.message

    if not is_not_from_future_slot(state, block.slot, current_time_ms):
        raise GossipIgnore("block is from a future slot")
    if block.slot <= store.finalized_checkpoint.slot:
        raise GossipIgnore("block is not later than the finalized checkpoint")
    if (block.proposer_index, block.slot) in seen.proposer_slots:
        raise GossipIgnore("block is not the first valid block for proposer and slot")
    if block.parent_root not in store.blocks:
        raise GossipIgnore("block's parent has not been seen")
    if block.parent_root not in store.block_states:
        raise GossipReject("block's parent failed validation")
    if block.slot <= store.blocks[block.parent_root].slot:
        raise GossipReject("block is not from a higher slot than its parent")
    if not is_checkpoint_on_store_chain(store, block.parent_root, store.finalized_checkpoint):
        raise GossipReject("finalized checkpoint is not an ancestor of block")
    if bid.parent_block_root != block.parent_root:
        raise GossipReject("bid parent root does not match block parent root")
    if bid.slot != block.slot:
        raise GossipReject("bid slot does not match block slot")

    max_blobs = get_blob_parameters(compute_epoch_at_slot(block.slot)).max_blobs_per_block
    if len(bid.blob_kzg_commitments) > max_blobs:
        raise GossipReject("too many blob KZG commitments")

    parent_state = store.block_states[block.parent_root].copy()
    parent_bid = parent_state.latest_execution_payload_bid
    if bid.parent_block_hash == parent_bid.block_hash:
        if not is_payload_verified(store, block.parent_root):
            raise GossipIgnore("parent execution payload has not been verified")
    elif bid.parent_block_hash != parent_state.latest_block_hash:
        raise GossipReject("empty parent does not preserve latest execution block hash")

    # [Modified in Simplex]
    # Live Store migration advances every retained branch state through the
    # empty-slot processing at the activation slot. The first Simplex block may
    # therefore have an already slot-processed parent state, matching the
    # boundary path in ``on_block``.
    if parent_state.slot == block.slot:
        assert block.slot == compute_start_slot_at_epoch(SIMPLEX_FORK_EPOCH)
        assert parent_state.fork.current_version == SIMPLEX_FORK_VERSION
        assert parent_state.fork.epoch == SIMPLEX_FORK_EPOCH
        assert parent_state.latest_block_header.slot < block.slot
    else:
        process_slots(parent_state, block.slot)
    if block.proposer_index >= len(parent_state.validators):
        raise GossipReject("proposer index out of range")
    expected_proposer = get_beacon_proposer_index(parent_state)
    if block.proposer_index != expected_proposer:
        raise GossipReject("block proposer index does not match expected proposer")

    proposer = parent_state.validators[block.proposer_index]
    domain = get_domain(parent_state, DOMAIN_BEACON_PROPOSER, compute_epoch_at_slot(block.slot))
    signing_root = compute_signing_root(block, domain)
    if not bls.Verify(proposer.pubkey, signing_root, signed_beacon_block.signature):
        raise GossipReject("invalid proposer signature")

    seen.proposer_slots.add((block.proposer_index, block.slot))
```

##### Modified `beacon_aggregate_and_proof`

After this validator succeeds, clients MUST call
`on_attestation(store, signed_aggregate_and_proof.message.aggregate)` before
forwarding the message.

```python
def validate_beacon_aggregate_and_proof_gossip(
    seen: Seen,
    store: Store,
    state: BeaconState,
    signed_aggregate_and_proof: SignedAggregateAndProof,
    current_time_ms: uint64,
) -> None:
    """Validate a Simplex aggregate for gossip propagation."""
    aggregate_and_proof = signed_aggregate_and_proof.message
    aggregate = aggregate_and_proof.aggregate
    data = aggregate.data

    if not is_not_from_future_slot(state, data.slot, current_time_ms):
        raise GossipIgnore("aggregate slot is from the future")
    current_slot = get_current_slot(store)
    if data.slot > current_slot:
        raise GossipIgnore("aggregate slot is from the future")
    if data.slot + LATEST_MESSAGE_EXPIRY_SLOTS <= current_slot:
        raise GossipIgnore("aggregate is outside the latest-message window")

    validate_attestation_data_gossip(store, state, data)
    target_state = get_attestation_checkpoint_state(store, data)
    attestation_epoch = compute_epoch_at_slot(data.slot)

    committee_indices = get_committee_indices(aggregate.committee_bits)
    if len(committee_indices) != 1:
        raise GossipReject("aggregate must specify exactly one committee")
    committee_index = committee_indices[0]
    committee_count = get_committee_count_per_slot(target_state, attestation_epoch)
    if committee_index >= committee_count:
        raise GossipReject("committee index out of range")
    committee = get_beacon_committee(target_state, data.slot, committee_index)
    if len(aggregate.aggregation_bits) != len(committee):
        raise GossipReject("aggregation bits length does not match committee size")

    attesting_indices = get_attesting_indices(target_state, aggregate)
    if len(attesting_indices) == 0:
        raise GossipReject("aggregate has no participants")

    aggregate_data_root = hash_tree_root(data)
    aggregate_cache_key = (aggregate_data_root, committee_index)
    aggregate_bits = tuple(bool(bit) for bit in aggregate.aggregation_bits)
    seen_bits = seen.aggregate_data_roots.get(aggregate_cache_key, set())
    if is_non_strict_superset(seen_bits, aggregate_bits):
        raise GossipIgnore("already seen an aggregate superset for this data")

    aggregator_index = aggregate_and_proof.aggregator_index
    aggregate_round = compute_round_at_slot(data.slot)
    if not is_aggregator(
        target_state,
        data.slot,
        committee_index,
        aggregate_and_proof.selection_proof,
    ):
        raise GossipReject("validator is not selected as aggregator")
    if aggregator_index not in committee:
        raise GossipReject("aggregator is not in the committee")

    aggregator = target_state.validators[aggregator_index]
    domain = get_domain(target_state, DOMAIN_SELECTION_PROOF, attestation_epoch)
    signing_root = compute_signing_root(data.slot, domain)
    if not bls.Verify(aggregator.pubkey, signing_root, aggregate_and_proof.selection_proof):
        raise GossipReject("invalid selection proof signature")
    domain = get_domain(target_state, DOMAIN_AGGREGATE_AND_PROOF, attestation_epoch)
    signing_root = compute_signing_root(aggregate_and_proof, domain)
    if not bls.Verify(aggregator.pubkey, signing_root, signed_aggregate_and_proof.signature):
        raise GossipReject("invalid aggregator signature")
    if not is_valid_indexed_attestation(
        target_state, get_indexed_attestation(target_state, aggregate)
    ):
        raise GossipReject("invalid aggregate signature")

    sorted_attesting_indices = sorted(attesting_indices)
    aggregator_key = (aggregator_index, aggregate_round)
    has_new_evidence = has_new_attestation_evidence(
        seen, sorted_attesting_indices, aggregate_round, aggregate_data_root
    )
    # Aggregator identity cannot make a third datum from an already-known
    # equivocator useful. Accept the one distinct second datum because it adds
    # signer evidence; ignore every aggregate that adds none, even when it uses
    # a previously unseen aggregator.
    if not has_new_evidence:
        raise GossipIgnore("aggregate adds no new signer evidence")

    record_attestation_evidence(
        seen, sorted_attesting_indices, aggregate_round, aggregate_data_root
    )
    seen.aggregator_rounds.add(aggregator_key)
    seen.aggregate_data_roots.setdefault(aggregate_cache_key, set()).add(aggregate_bits)
```

##### Modified `execution_payload`

The inherited execution-payload-envelope validations continue to apply, except
that the finalized lower bound uses the exact Simplex checkpoint slot:

- _[IGNORE]_ The envelope is from a slot greater than or equal to the latest
  finalized slot -- i.e.
  `envelope.payload.slot_number >= store.finalized_checkpoint.slot`.

For an unresolved Gloas root retained across live Simplex activation, the
inherited signature and consistency checks use the historical post-block state
in `store.legacy_payload_verification_states`, as specified by the modified
`on_execution_payload_envelope` handler. The activation-slot Simplex branch
state cannot reconstruct the pre-fork block root, slot, domain, or committed
bid.

##### New `available_attestation`

| Name                    | Message Type           |
| ----------------------- | ---------------------- |
| `available_attestation` | `AvailableAttestation` |

The topic carries individual or aggregated available-committee votes for the
current slot. After this validator succeeds, clients MUST call
`on_available_attestation(store, attestation)` before forwarding the message.

```python
def validate_available_attestation_gossip(
    seen: Seen,
    store: Store,
    state: BeaconState,
    attestation: AvailableAttestation,
    current_time_ms: uint64,
) -> None:
    """Validate an available attestation for gossip propagation."""
    data = attestation.data
    if not is_not_from_future_slot(state, data.slot, current_time_ms):
        raise GossipIgnore("available attestation is from a future slot")
    if data.slot != get_current_slot(store):
        raise GossipIgnore("available attestation is not for the current slot")
    if data.beacon_block_root not in store.blocks:
        raise GossipIgnore("block being voted for has not been seen")
    if data.beacon_block_root not in store.block_states:
        raise GossipReject("block being voted for failed validation")

    head_state = store.block_states[data.beacon_block_root]
    if (
        head_state.fork.current_version == SIMPLEX_FORK_VERSION
        and data.slot < compute_start_slot_at_epoch(head_state.fork.epoch)
    ):
        raise GossipReject("available attestation predates the active fork")

    block_slot = store.blocks[data.beacon_block_root].slot
    if block_slot > data.slot:
        raise GossipReject("block being voted for is later than the attestation")
    if block_slot == data.slot and data.payload_present:
        raise GossipReject("same-slot available attestation signals payload presence")
    if (
        block_slot < data.slot
        and data.payload_present
        and not is_payload_verified(store, data.beacon_block_root)
    ):
        raise GossipIgnore("voted payload has not been verified")
    if not is_checkpoint_on_store_chain(store, data.beacon_block_root, store.finalized_checkpoint):
        raise GossipIgnore("finalized checkpoint is not an ancestor of voted block")

    target_state = get_available_attestation_checkpoint_state(store, data)
    if len(attestation.aggregation_bits) != AVAILABLE_COMMITTEE_SIZE:
        raise GossipReject("available aggregation bits have an incorrect length")
    attesting_indices = get_available_attesting_indices(target_state, attestation)
    if len(attesting_indices) == 0:
        raise GossipReject("available attestation has no participants")

    attestation_epoch = compute_epoch_at_slot(data.slot)
    pubkeys = [target_state.validators[i].pubkey for i in sorted(attesting_indices)]
    domain = get_domain(target_state, DOMAIN_AVAILABLE_ATTESTER, attestation_epoch)
    signing_root = compute_signing_root(data, domain)
    if not bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature):
        raise GossipReject("invalid available attestation signature")

    data_root = hash_tree_root(data)
    aggregate_bits = tuple(bool(bit) for bit in attestation.aggregation_bits)
    seen_bits = seen.available_attestation_data_roots.get(data_root, set())
    if is_non_strict_superset(seen_bits, aggregate_bits):
        raise GossipIgnore("already seen an available aggregate superset for this data")
    extends_seen = any(
        prior_bits != aggregate_bits
        and all(
            not prior_bit or new_bit
            for prior_bit, new_bit in zip(prior_bits, aggregate_bits, strict=True)
        )
        for prior_bits in seen_bits
    )

    has_new_evidence = False
    for index in attesting_indices:
        key = (index, data.slot)
        if key in seen.available_attestation_validator_slot_equivocations:
            continue
        previous_root = seen.available_attestation_validator_slot_data_roots.get(key)
        if previous_root is None or previous_root != data_root:
            has_new_evidence = True
            break
    if not has_new_evidence and not extends_seen:
        raise GossipIgnore("available aggregate adds no new signer evidence")

    for index in attesting_indices:
        key = (index, data.slot)
        if key in seen.available_attestation_validator_slot_equivocations:
            continue
        previous_root = seen.available_attestation_validator_slot_data_roots.get(key)
        if previous_root is None:
            seen.available_attestation_validator_slot_data_roots[key] = data_root
        elif previous_root != data_root:
            seen.available_attestation_validator_slot_equivocations.add(key)
    seen.available_attestation_data_roots.setdefault(data_root, set()).add(aggregate_bits)
```

#### Attestation subnets

##### Modified `beacon_attestation_{subnet_id}`

The first distinct valid message for a validator in a round is forwarded. One
second distinct message is also forwarded and marks the validator as a known
round equivocator; this is necessary for the fresh-root credit rule. After this
validator succeeds, clients MUST call `on_gossip_single_attestation` before
forwarding the message so that its latest-head effect is applied immediately.

```python
def validate_beacon_attestation_gossip(
    seen: Seen,
    store: Store,
    state: BeaconState,
    attestation: SingleAttestation,
    current_time_ms: uint64,
    subnet_id: SubnetID,
) -> None:
    """Validate a Simplex single attestation for gossip propagation."""
    data = attestation.data
    if not is_not_from_future_slot(state, data.slot, current_time_ms):
        raise GossipIgnore("attestation slot is from the future")
    current_slot = get_current_slot(store)
    if data.slot > current_slot:
        raise GossipIgnore("attestation slot is from the future")
    if data.slot + LATEST_MESSAGE_EXPIRY_SLOTS <= current_slot:
        raise GossipIgnore("attestation is outside the latest-message window")

    validate_attestation_data_gossip(store, state, data)
    target_state = get_attestation_checkpoint_state(store, data)
    attestation_epoch = compute_epoch_at_slot(data.slot)
    committee_index = attestation.committee_index
    committees_per_slot = get_committee_count_per_slot(target_state, attestation_epoch)
    if committee_index >= committees_per_slot:
        raise GossipReject("committee index out of range")
    expected_subnet = compute_subnet_for_attestation(
        committees_per_slot, data.slot, committee_index
    )
    if expected_subnet != subnet_id:
        raise GossipReject("attestation is for the wrong subnet")

    committee = get_beacon_committee(target_state, data.slot, committee_index)
    attester_index = attestation.attester_index
    if attester_index not in committee:
        raise GossipReject("attester is not a member of the committee")

    attestation_round = compute_round_at_slot(data.slot)
    data_root = hash_tree_root(data)
    if not has_new_attestation_evidence(seen, [attester_index], attestation_round, data_root):
        raise GossipIgnore("attestation adds no new signer evidence")

    attester = target_state.validators[attester_index]
    domain = get_domain(target_state, DOMAIN_BEACON_ATTESTER, attestation_epoch)
    signing_root = compute_signing_root(data, domain)
    if not bls.Verify(attester.pubkey, signing_root, attestation.signature):
        raise GossipReject("invalid attestation signature")

    record_attestation_evidence(seen, [attester_index], attestation_round, data_root)
```

## The Req/Resp domain

### Messages

#### New Status v3

**Protocol ID:** `/eth2/beacon_chain/req/status/3/`

Request, Response Content:

```
(
  fork_digest: ForkDigest
  finalized_root: Root
  finalized_slot: Slot
  head_root: Root
  head_slot: Slot
  earliest_available_slot: Slot
)
```

Status v3 replaces the epoch field with the Simplex finalized-checkpoint slot.
The remaining fields and encoding follow Status v2. Clients SHOULD immediately
disconnect if
`is_status_finalized_checkpoint_compatible(store, finalized_root, finalized_slot)`
returns `False`. In particular, the activation height-0 legacy boundary sentinel
is compatible even when its root was proposed before its advertised FFG boundary
slot. Once the handshake completes, the client with the lower `finalized_slot`,
or lower `head_slot` when the finalized slots are equal, SHOULD request blocks
from its counterparty.

#### Modified block response mappings

The response mappings for `BeaconBlocksByRange v2`, `BeaconBlocksByRoot v2`, and
`BeaconBlocksByHead v1` are each extended with:

<!-- eth_consensus_specs: skip -->

| `fork_version`         | Chunk SSZ type              |
| ---------------------- | --------------------------- |
| `SIMPLEX_FORK_VERSION` | `simplex.SignedBeaconBlock` |

#### Modified execution-payload-envelope response mappings

The response mappings for `ExecutionPayloadEnvelopesByRange v1` and
`ExecutionPayloadEnvelopesByRoot v1` are each extended with:

<!-- eth_consensus_specs: skip -->

| `fork_version`         | Chunk SSZ type                           |
| ---------------------- | ---------------------------------------- |
| `SIMPLEX_FORK_VERSION` | `simplex.SignedExecutionPayloadEnvelope` |

#### Modified data-column response mappings

For both `DataColumnSidecarsByRange v1` and `DataColumnSidecarsByRoot v1`, a
response whose context resolves to `SIMPLEX_FORK_VERSION` uses the modified
Simplex sidecar type. In both protocols, derive the response context epoch as
`compute_epoch_at_slot(data_column_sidecar.slot)`; the inherited Fulu expression
uses a signed block header that Gloas removed from the sidecar.

<!-- eth_consensus_specs: skip -->

| `fork_version`         | Chunk SSZ type              |
| ---------------------- | --------------------------- |
| `SIMPLEX_FORK_VERSION` | `simplex.DataColumnSidecar` |

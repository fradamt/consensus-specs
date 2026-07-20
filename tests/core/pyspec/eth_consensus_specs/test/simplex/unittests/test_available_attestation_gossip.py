from types import SimpleNamespace

from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store,
)
from eth_consensus_specs.test.helpers.gossip import get_seen
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _expect_gossip_ignore(spec, fn):
    try:
        fn()
    except spec.GossipIgnore:
        return
    raise AssertionError("gossip input was not ignored")


def _expect_gossip_reject(spec, fn):
    try:
        fn()
    except spec.GossipReject:
        return
    raise AssertionError("gossip input was not rejected")


def _time_at_slot(spec, state, slot):
    return spec.compute_time_at_slot_ms(state, slot)


def _advance_store_to_slot(spec, store, slot):
    time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)
    spec.on_tick(store, time)


def _add_child(spec, store, state, parent_root, slot, marker, with_state=True):
    block = spec.BeaconBlock(
        slot=slot,
        parent_root=parent_root,
        body=spec.BeaconBlockBody(graffiti=bytes([marker]) * 32),
    )
    root = block.hash_tree_root()
    store.blocks[root] = block
    if with_state:
        store.block_states[root] = state.copy()
    return root


def _distinct_committee_members(spec, state, slot, count):
    members = []
    for validator_index in spec.get_available_committee(state, slot):
        if validator_index not in members:
            members.append(validator_index)
            if len(members) == count:
                return members
    raise AssertionError(f"available committee has fewer than {count} distinct members")


def _signed_available_attestation(
    spec,
    state,
    slot,
    beacon_block_root,
    validator_indices,
    payload_present=False,
):
    data = spec.AvailableAttestationData(
        slot=slot,
        beacon_block_root=beacon_block_root,
        payload_present=payload_present,
    )
    attestation = spec.AvailableAttestation(data=data)
    requested_indices = set(validator_indices)
    selected_indices = set()
    for position, validator_index in enumerate(spec.get_available_committee(state, slot)):
        if validator_index in requested_indices and validator_index not in selected_indices:
            attestation.aggregation_bits[position] = True
            selected_indices.add(validator_index)
    assert selected_indices == requested_indices

    epoch = spec.compute_epoch_at_slot(slot)
    domain = spec.get_domain(state, spec.DOMAIN_AVAILABLE_ATTESTER, epoch)
    signing_root = spec.compute_signing_root(data, domain)
    signatures = [bls.Sign(privkeys[index], signing_root) for index in sorted(selected_indices)]
    attestation.signature = bls.Aggregate(signatures)
    return attestation


def _validate(spec, seen, store, state, attestation, current_time_ms):
    spec.validate_available_attestation_gossip(
        seen,
        store,
        state,
        attestation,
        current_time_ms,
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_timing_and_head_reference_classification(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    seen = get_seen(spec)

    future_slot = spec.Slot(spec.GENESIS_SLOT + 1)
    future = spec.AvailableAttestation(
        data=spec.AvailableAttestationData(
            slot=future_slot,
            beacon_block_root=anchor_root,
            payload_present=False,
        )
    )
    _expect_gossip_ignore(
        spec,
        lambda: _validate(
            spec,
            seen,
            store,
            state,
            future,
            _time_at_slot(spec, state, spec.GENESIS_SLOT),
        ),
    )

    _advance_store_to_slot(spec, store, future_slot)
    current_time_ms = _time_at_slot(spec, state, future_slot)

    wrong_slot = future.copy()
    wrong_slot.data.slot = spec.GENESIS_SLOT
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, wrong_slot, current_time_ms),
    )

    unknown_head = future.copy()
    unknown_head.data.beacon_block_root = spec.Root(b"\x91" * 32)
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, unknown_head, current_time_ms),
    )

    failed_validation_root = _add_child(
        spec,
        store,
        state,
        anchor_root,
        future_slot,
        marker=0xA1,
        with_state=False,
    )
    known_but_malformed = future.copy()
    known_but_malformed.data.beacon_block_root = failed_validation_root
    _expect_gossip_reject(
        spec,
        lambda: _validate(
            spec,
            seen,
            store,
            state,
            known_but_malformed,
            current_time_ms,
        ),
    )

    later_root = _add_child(
        spec,
        store,
        state,
        anchor_root,
        spec.Slot(future_slot + 1),
        marker=0xA2,
    )
    later_block = future.copy()
    later_block.data.beacon_block_root = later_root
    _expect_gossip_reject(
        spec,
        lambda: _validate(spec, seen, store, state, later_block, current_time_ms),
    )

    same_slot_root = _add_child(
        spec,
        store,
        state,
        anchor_root,
        future_slot,
        marker=0xA3,
    )
    same_slot_payload = future.copy()
    same_slot_payload.data.beacon_block_root = same_slot_root
    same_slot_payload.data.payload_present = True
    _expect_gossip_reject(
        spec,
        lambda: _validate(spec, seen, store, state, same_slot_payload, current_time_ms),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_payload_verification_and_finalized_ancestry(spec, state):
    slot = spec.Slot(spec.GENESIS_SLOT + 1)
    validator_index = _distinct_committee_members(spec, state, slot, count=1)[0]

    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    _advance_store_to_slot(spec, store, slot)
    current_time_ms = _time_at_slot(spec, state, slot)
    payload_vote = _signed_available_attestation(
        spec,
        state,
        slot,
        anchor_root,
        [validator_index],
        payload_present=True,
    )
    seen = get_seen(spec)
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, payload_vote, current_time_ms),
    )

    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    _validate(spec, seen, store, state, payload_vote, current_time_ms)

    forked_store = get_genesis_forkchoice_store(spec, state)
    forked_anchor = forked_store.finalized_checkpoint.root
    _advance_store_to_slot(spec, forked_store, slot)
    finalized_root = _add_child(
        spec,
        forked_store,
        state,
        forked_anchor,
        slot,
        marker=0xB1,
    )
    sibling_root = _add_child(
        spec,
        forked_store,
        state,
        forked_anchor,
        slot,
        marker=0xB2,
    )
    forked_store.finalized_checkpoint = spec.Checkpoint(slot=slot, root=finalized_root)
    sibling_vote = _signed_available_attestation(
        spec,
        state,
        slot,
        sibling_root,
        [validator_index],
    )
    _expect_gossip_ignore(
        spec,
        lambda: _validate(
            spec,
            get_seen(spec),
            forked_store,
            state,
            sibling_vote,
            current_time_ms,
        ),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_bit_structure_and_signature(spec, state):
    slot = spec.Slot(spec.GENESIS_SLOT + 1)
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    _advance_store_to_slot(spec, store, slot)
    head_root = _add_child(spec, store, state, anchor_root, slot, marker=0xC1)
    current_time_ms = _time_at_slot(spec, state, slot)
    validator_index = _distinct_committee_members(spec, state, slot, count=1)[0]
    valid = _signed_available_attestation(
        spec,
        state,
        slot,
        head_root,
        [validator_index],
    )

    wrong_length = SimpleNamespace(
        aggregation_bits=[False] * (spec.AVAILABLE_COMMITTEE_SIZE - 1),
        data=valid.data,
        signature=valid.signature,
    )
    _expect_gossip_reject(
        spec,
        lambda: _validate(
            spec,
            get_seen(spec),
            store,
            state,
            wrong_length,
            current_time_ms,
        ),
    )

    empty = spec.AvailableAttestation(data=valid.data)
    _expect_gossip_reject(
        spec,
        lambda: _validate(
            spec,
            get_seen(spec),
            store,
            state,
            empty,
            current_time_ms,
        ),
    )

    invalid_signature = valid.copy()
    invalid_signature.signature = bls.Sign(
        privkeys[validator_index],
        spec.Root(b"\x55" * 32),
    )
    _expect_gossip_reject(
        spec,
        lambda: _validate(
            spec,
            get_seen(spec),
            store,
            state,
            invalid_signature,
            current_time_ms,
        ),
    )

    _validate(spec, get_seen(spec), store, state, valid, current_time_ms)


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_duplicate_and_strict_superset_handling(spec, state):
    slot = spec.Slot(spec.GENESIS_SLOT + 1)
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    _advance_store_to_slot(spec, store, slot)
    head_root = _add_child(spec, store, state, anchor_root, slot, marker=0xD1)
    current_time_ms = _time_at_slot(spec, state, slot)
    validator_1, validator_2 = _distinct_committee_members(spec, state, slot, count=2)
    single = _signed_available_attestation(
        spec,
        state,
        slot,
        head_root,
        [validator_1],
    )
    superset = _signed_available_attestation(
        spec,
        state,
        slot,
        head_root,
        [validator_1, validator_2],
    )
    unseen_subset = _signed_available_attestation(
        spec,
        state,
        slot,
        head_root,
        [validator_2],
    )
    seen = get_seen(spec)

    _validate(spec, seen, store, state, single, current_time_ms)
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, single, current_time_ms),
    )

    # A new strict superset is useful and must be forwarded.
    _validate(spec, seen, store, state, superset, current_time_ms)

    # Once the superset is known, both its duplicate and a previously unseen
    # subset are stale.
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, superset, current_time_ms),
    )
    _expect_gossip_ignore(
        spec,
        lambda: _validate(spec, seen, store, state, unseen_subset, current_time_ms),
    )

    data_root = single.data.hash_tree_root()
    assert len(seen.available_attestation_data_roots[data_root]) == 2
    assert seen.available_attestation_validator_slot_data_roots[(validator_1, slot)] == data_root
    assert seen.available_attestation_validator_slot_data_roots[(validator_2, slot)] == data_root


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_forwards_one_conflict_then_suppresses_third_copy(spec, state):
    slot = spec.Slot(spec.GENESIS_SLOT + 1)
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    _advance_store_to_slot(spec, store, slot)
    head_roots = [
        _add_child(spec, store, state, anchor_root, slot, marker=marker)
        for marker in (0xE1, 0xE2, 0xE3)
    ]
    current_time_ms = _time_at_slot(spec, state, slot)
    validator_index = _distinct_committee_members(spec, state, slot, count=1)[0]
    attestations = [
        _signed_available_attestation(
            spec,
            state,
            slot,
            head_root,
            [validator_index],
        )
        for head_root in head_roots
    ]
    seen = get_seen(spec)

    _validate(spec, seen, store, state, attestations[0], current_time_ms)
    spec.on_available_attestation(store, attestations[0])
    _validate(spec, seen, store, state, attestations[1], current_time_ms)
    spec.on_available_attestation(store, attestations[1])

    evidence_key = (validator_index, slot)
    assert evidence_key in seen.available_attestation_validator_slot_equivocations
    assert validator_index in store.available_vote_equivocations[slot]

    _expect_gossip_ignore(
        spec,
        lambda: _validate(
            spec,
            seen,
            store,
            state,
            attestations[2],
            current_time_ms,
        ),
    )
    assert attestations[2].data.hash_tree_root() not in seen.available_attestation_data_roots


def _signed_indexed_attestation(spec, state, validator_index, data):
    domain = spec.get_domain(
        state,
        spec.DOMAIN_BEACON_ATTESTER,
        spec.compute_epoch_at_slot(data.slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    return spec.IndexedAttestation(
        attesting_indices=[validator_index],
        data=data,
        signature=bls.Sign(privkeys[validator_index], signing_root),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_on_round_double_vote_evidence_records_round_and_global_equivocation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    validator_index = spec.ValidatorIndex(0)
    data_1 = spec.AttestationData(
        slot=spec.GENESIS_SLOT,
        beacon_block_root=anchor_root,
        target=spec.Checkpoint(),
        height=spec.Height(1),
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    data_2 = data_1.copy()
    data_2.height = spec.Height(2)
    evidence = spec.RoundDoubleVoteEvidence(
        attestation_1=_signed_indexed_attestation(spec, state, validator_index, data_1),
        attestation_2=_signed_indexed_attestation(spec, state, validator_index, data_2),
    )

    spec.on_round_double_vote_evidence(store, evidence)

    round_ = spec.compute_round_at_slot(data_1.slot)
    assert store.round_equivocating_indices[round_] == {validator_index}
    assert store.equivocating_indices == {validator_index}

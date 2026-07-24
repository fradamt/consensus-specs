from eth_consensus_specs.test.context import always_bls, spec_state_test, with_simplex_and_later
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    sign_attestation,
    to_single_attestation,
)
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store,
    get_genesis_forkchoice_store_and_block,
)
from eth_consensus_specs.test.helpers.gossip import get_seen
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _add_child(spec, store, state, parent_root, slot, marker):
    block = spec.BeaconBlock(
        slot=slot,
        parent_root=parent_root,
        body=spec.BeaconBlockBody(graffiti=bytes([marker]) * 32),
    )
    root = block.hash_tree_root()
    store.blocks[root] = block
    store.block_states[root] = state
    return root


def _expect_gossip_reject(spec, fn):
    try:
        fn()
    except spec.GossipReject:
        return
    raise AssertionError("gossip input was not rejected")


def _expect_gossip_ignore(spec, fn):
    try:
        fn()
    except spec.GossipIgnore:
        return
    raise AssertionError("gossip input was not ignored")


@with_simplex_and_later
@spec_state_test
def test_round_evidence_bookkeeping_retains_exactly_two_data_roots(spec, state):
    validator_index = spec.ValidatorIndex(0)
    round_ = spec.Round(3)
    first_root = spec.Root(b"\x11" * 32)
    second_root = spec.Root(b"\x22" * 32)
    third_root = spec.Root(b"\x33" * 32)

    single_seen = get_seen(spec)
    aggregate_seen = get_seen(spec)
    for seen, indices in (
        (single_seen, [validator_index]),
        (aggregate_seen, [validator_index, spec.ValidatorIndex(1)]),
    ):
        assert spec.has_new_attestation_evidence(seen, indices, round_, first_root)
        spec.record_attestation_evidence(seen, indices, round_, first_root)
        assert not spec.has_new_attestation_evidence(seen, [validator_index], round_, first_root)

        # The one distinct second datum must be forwarded so the signer can be
        # credited as a round equivocator by grade-gap root syncing.
        assert spec.has_new_attestation_evidence(seen, [validator_index], round_, second_root)
        spec.record_attestation_evidence(seen, [validator_index], round_, second_root)
        assert (validator_index, round_) in seen.attestation_validator_round_equivocations

        # Once the evidence exists, no third copy from that signer is useful.
        assert not spec.has_new_attestation_evidence(seen, [validator_index], round_, third_root)

    assert (
        single_seen.attestation_validator_round_data_roots[(validator_index, round_)]
        == aggregate_seen.attestation_validator_round_data_roots[(validator_index, round_)]
    )
    assert single_seen.attestation_validator_round_equivocations == {(validator_index, round_)}
    assert (validator_index, round_) in aggregate_seen.attestation_validator_round_equivocations


def _selected_aggregators(spec, state, slot, committee_index, count):
    committee = spec.get_beacon_committee(state, slot, committee_index)
    selected = []
    for index in committee:
        selection_proof = spec.get_slot_signature(state, slot, privkeys[index])
        if spec.is_aggregator(state, slot, committee_index, selection_proof):
            selected.append(index)
            if len(selected) == count:
                break
    assert len(selected) == count
    return selected


def _signed_aggregate_and_proof(spec, state, attestation, aggregator_index):
    privkey = privkeys[aggregator_index]
    aggregate_and_proof = spec.get_aggregate_and_proof(
        state,
        aggregator_index,
        attestation,
        privkey,
    )
    return spec.SignedAggregateAndProof(
        message=aggregate_and_proof,
        signature=spec.get_aggregate_and_proof_signature(
            state,
            aggregate_and_proof,
            privkey,
        ),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_single_and_aggregate_forward_exactly_one_distinct_second_vote(spec, state):
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    slot = spec.GENESIS_SLOT
    committee_index = spec.CommitteeIndex(0)
    committee = spec.get_beacon_committee(state, slot, committee_index)
    attester_index = committee[0]
    aggregators = _selected_aggregators(spec, state, slot, committee_index, count=3)

    aggregates = []
    singles = []
    for height in (spec.Height(1), spec.Height(2), spec.Height(3)):
        attestation = get_valid_attestation(
            spec,
            state,
            slot=slot,
            index=committee_index,
            beacon_block_root=anchor_root,
            filter_participant_set=lambda _participants, index=attester_index: {index},
        )
        attestation.data.target = spec.Checkpoint()
        attestation.data.height = height
        attestation.data.finality_target = spec.Checkpoint()
        attestation.data.finality_height = spec.FAR_FUTURE_HEIGHT
        sign_attestation(spec, state, attestation)
        aggregates.append(attestation)
        singles.append(
            to_single_attestation(
                spec,
                state,
                attestation,
                attester_index=attester_index,
            )
        )

    current_time_ms = spec.compute_time_at_slot_ms(state, slot)
    committees_per_slot = spec.get_committee_count_per_slot(state, spec.compute_epoch_at_slot(slot))
    subnet_id = spec.compute_subnet_for_attestation(committees_per_slot, slot, committee_index)

    single_seen = get_seen(spec)
    spec.validate_beacon_attestation_gossip(
        single_seen,
        store,
        state,
        singles[0],
        current_time_ms,
        subnet_id,
    )
    spec.on_gossip_single_attestation(store, singles[0])
    assert store.latest_messages[attester_index] == spec.LatestMessage(slot=slot, root=anchor_root)
    spec.validate_beacon_attestation_gossip(
        single_seen,
        store,
        state,
        singles[1],
        current_time_ms,
        subnet_id,
    )
    spec.on_gossip_single_attestation(store, singles[1])
    attestation_round = spec.compute_round_at_slot(slot)
    assert attester_index in store.round_equivocating_indices[attestation_round]
    assert attester_index in store.equivocating_indices
    _expect_gossip_ignore(
        spec,
        lambda: spec.validate_beacon_attestation_gossip(
            single_seen,
            store,
            state,
            singles[2],
            current_time_ms,
            subnet_id,
        ),
    )

    aggregate_seen = get_seen(spec)
    signed_aggregates = [
        _signed_aggregate_and_proof(spec, state, aggregate, aggregator)
        for aggregate, aggregator in zip(aggregates, aggregators, strict=True)
    ]
    aggregate_store = get_genesis_forkchoice_store(spec, state)
    spec.validate_beacon_aggregate_and_proof_gossip(
        aggregate_seen,
        aggregate_store,
        state,
        signed_aggregates[0],
        current_time_ms,
    )
    spec.on_attestation(aggregate_store, aggregates[0], is_from_block=False)
    assert aggregate_store.latest_messages[attester_index] == spec.LatestMessage(
        slot=slot, root=anchor_root
    )
    spec.validate_beacon_aggregate_and_proof_gossip(
        aggregate_seen,
        aggregate_store,
        state,
        signed_aggregates[1],
        current_time_ms,
    )
    spec.on_attestation(aggregate_store, aggregates[1], is_from_block=False)
    assert attester_index in aggregate_store.round_equivocating_indices[attestation_round]
    assert attester_index in aggregate_store.equivocating_indices
    # A new aggregator does not make a third datum from an already-known
    # equivocator useful.
    assert len(set(aggregators)) == 3
    _expect_gossip_ignore(
        spec,
        lambda: spec.validate_beacon_aggregate_and_proof_gossip(
            aggregate_seen,
            aggregate_store,
            state,
            signed_aggregates[2],
            current_time_ms,
        ),
    )


@with_simplex_and_later
@spec_state_test
def test_attestation_gossip_rejects_wrong_slot_ancestry_and_future_checkpoint(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    head_a = _add_child(spec, store, state, anchor_root, spec.Slot(2), 0xA1)
    head_b = _add_child(spec, store, state, anchor_root, spec.Slot(2), 0xB2)
    future_root = _add_child(spec, store, state, head_a, spec.Slot(3), 0xA3)
    base = spec.AttestationData(
        slot=spec.Slot(2),
        beacon_block_root=head_a,
        target=spec.Checkpoint(slot=spec.Slot(2), root=head_a),
        height=spec.GENESIS_HEIGHT,
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    spec.validate_attestation_data_gossip(store, state, base)

    unknown_head = base.copy()
    unknown_head.beacon_block_root = spec.Root(b"\x91" * 32)
    _expect_gossip_ignore(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, unknown_head),
    )

    unknown_checkpoint = base.copy()
    unknown_checkpoint.target.root = spec.Root(b"\x92" * 32)
    _expect_gossip_ignore(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, unknown_checkpoint),
    )

    wrong_slot = base.copy()
    wrong_slot.target.slot = spec.Slot(1)
    _expect_gossip_reject(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, wrong_slot),
    )

    wrong_branch = base.copy()
    wrong_branch.target = spec.Checkpoint(slot=spec.Slot(2), root=head_b)
    _expect_gossip_reject(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, wrong_branch),
    )

    future_checkpoint = base.copy()
    future_checkpoint.target = spec.Checkpoint(slot=spec.Slot(3), root=future_root)
    _expect_gossip_reject(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, future_checkpoint),
    )

    malformed_empty_finality = base.copy()
    malformed_empty_finality.finality_height = spec.Height(1)
    _expect_gossip_reject(
        spec,
        lambda: spec.validate_attestation_data_gossip(store, state, malformed_empty_finality),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_single_attestation_latest_message_expiry_exact_boundary(spec, state):
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    data = spec.AttestationData(
        slot=spec.GENESIS_SLOT,
        beacon_block_root=anchor_root,
        target=spec.Checkpoint(),
        height=spec.Height(0),
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    committee_index = spec.CommitteeIndex(0)
    committee = spec.get_beacon_committee(state, data.slot, committee_index)
    attester_index = committee[0]
    domain = spec.get_domain(
        state,
        spec.DOMAIN_BEACON_ATTESTER,
        spec.compute_epoch_at_slot(data.slot),
    )
    signature = bls.Sign(
        privkeys[attester_index],
        spec.compute_signing_root(data, domain),
    )
    attestation = spec.SingleAttestation(
        committee_index=committee_index,
        attester_index=attester_index,
        data=data,
        signature=signature,
    )
    committees_per_slot = spec.get_committee_count_per_slot(
        state, spec.compute_epoch_at_slot(data.slot)
    )
    subnet_id = spec.compute_subnet_for_attestation(committees_per_slot, data.slot, committee_index)

    last_live_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS - 1)
    store.time = spec.uint64(
        store.genesis_time + last_live_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    spec.validate_beacon_attestation_gossip(
        get_seen(spec),
        store,
        state,
        attestation,
        spec.compute_time_at_slot_ms(state, last_live_slot),
        subnet_id,
    )

    expiry_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS)
    store.time = spec.uint64(
        store.genesis_time + expiry_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    _expect_gossip_ignore(
        spec,
        lambda: spec.validate_beacon_attestation_gossip(
            get_seen(spec),
            store,
            state,
            attestation,
            spec.compute_time_at_slot_ms(state, expiry_slot),
            subnet_id,
        ),
    )

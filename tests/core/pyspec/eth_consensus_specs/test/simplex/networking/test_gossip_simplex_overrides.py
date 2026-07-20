from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    to_single_attestation,
)
from eth_consensus_specs.test.helpers.block import build_empty_block_for_next_slot
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store_and_block,
)
from eth_consensus_specs.test.helpers.gossip import (
    get_filename,
    get_seen,
    run_validate_gossip,
    wrap_genesis_block,
)
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block


def _yield_anchor(spec, state):
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    signed_anchor = wrap_genesis_block(spec, anchor_block)
    return store, anchor_block, signed_anchor


def _signed_aggregate_and_proof(spec, state, attestation):
    committee_index = spec.get_committee_indices(attestation.committee_bits)[0]
    committee = spec.get_beacon_committee(state, attestation.data.slot, committee_index)
    aggregator_index = next(
        index
        for index in committee
        if spec.is_aggregator(
            state,
            attestation.data.slot,
            committee_index,
            spec.get_slot_signature(state, attestation.data.slot, privkeys[index]),
        )
    )
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
def test_gossip_beacon_attestation__simplex_valid_and_bad_signature(spec, state):
    yield "topic", "meta", "beacon_attestation"
    yield "state", state

    store, anchor_block, signed_anchor = _yield_anchor(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]

    aggregate = get_valid_attestation(
        spec,
        state,
        slot=spec.GENESIS_SLOT,
        beacon_block_root=anchor_root,
        signed=True,
    )
    attestation = to_single_attestation(spec, state, aggregate)
    invalid = attestation.copy()
    invalid.signature = spec.BLSSignature()
    yield get_filename(invalid), invalid
    yield get_filename(attestation), attestation

    committees_per_slot = spec.get_committee_count_per_slot(
        state,
        spec.compute_epoch_at_slot(attestation.data.slot),
    )
    subnet_id = spec.compute_subnet_for_attestation(
        committees_per_slot,
        attestation.data.slot,
        attestation.committee_index,
    )
    current_time_ms = spec.compute_time_at_slot_ms(state, attestation.data.slot)
    yield "current_time_ms", "meta", int(current_time_ms)

    seen = get_seen(spec)
    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        attestation=invalid,
        current_time_ms=current_time_ms,
        subnet_id=subnet_id,
    )
    assert result == "reject"
    assert reason == "invalid attestation signature"

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        attestation=attestation,
        current_time_ms=current_time_ms,
        subnet_id=subnet_id,
    )
    assert result == "valid"
    assert reason is None

    yield (
        "messages",
        "meta",
        [
            {
                "subnet_id": int(subnet_id),
                "offset_ms": 0,
                "message": get_filename(invalid),
                "expected": "reject",
                "reason": "invalid attestation signature",
            },
            {
                "subnet_id": int(subnet_id),
                "offset_ms": 0,
                "message": get_filename(attestation),
                "expected": "valid",
            },
        ],
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_gossip_beacon_aggregate_and_proof__simplex_valid_and_bad_signature(spec, state):
    yield "topic", "meta", "beacon_aggregate_and_proof"
    yield "state", state

    store, anchor_block, signed_anchor = _yield_anchor(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]

    attestation = get_valid_attestation(
        spec,
        state,
        slot=spec.GENESIS_SLOT,
        beacon_block_root=anchor_root,
        signed=True,
    )
    signed_aggregate = _signed_aggregate_and_proof(spec, state, attestation)
    invalid = signed_aggregate.copy()
    invalid.signature = spec.BLSSignature()
    yield get_filename(invalid), invalid
    yield get_filename(signed_aggregate), signed_aggregate

    current_time_ms = spec.compute_time_at_slot_ms(state, attestation.data.slot)
    yield "current_time_ms", "meta", int(current_time_ms)
    seen = get_seen(spec)

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        signed_aggregate_and_proof=invalid,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "invalid aggregator signature"

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        signed_aggregate_and_proof=signed_aggregate,
        current_time_ms=current_time_ms,
    )
    assert result == "valid"
    assert reason is None

    yield (
        "messages",
        "meta",
        [
            {
                "offset_ms": 0,
                "message": get_filename(invalid),
                "expected": "reject",
                "reason": "invalid aggregator signature",
            },
            {
                "offset_ms": 0,
                "message": get_filename(signed_aggregate),
                "expected": "valid",
            },
        ],
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_gossip_beacon_block__simplex_valid_and_bad_bid_parent(spec, state):
    yield "topic", "meta", "beacon_block"
    yield "state", state

    store, anchor_block, signed_anchor = _yield_anchor(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]

    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    invalid = signed_block.copy()
    invalid.message.body.signed_execution_payload_bid.message.parent_block_root = spec.Root(
        b"\x99" * 32
    )
    yield get_filename(invalid), invalid
    yield get_filename(signed_block), signed_block

    # The valid child selects its parent's FULL payload branch.
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    current_time_ms = spec.compute_time_at_slot_ms(state, signed_block.message.slot)
    yield "current_time_ms", "meta", int(current_time_ms)
    seen = get_seen(spec)

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        signed_beacon_block=invalid,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "bid parent root does not match block parent root"

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        signed_beacon_block=signed_block,
        current_time_ms=current_time_ms,
    )
    assert result == "valid"
    assert reason is None

    yield (
        "messages",
        "meta",
        [
            {
                "offset_ms": 0,
                "message": get_filename(invalid),
                "expected": "reject",
                "reason": "bid parent root does not match block parent root",
            },
            {
                "offset_ms": 0,
                "message": get_filename(signed_block),
                "expected": "valid",
            },
        ],
    )

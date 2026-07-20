from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
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
from eth_consensus_specs.utils import bls


@with_simplex_and_later
@spec_state_test
@always_bls
def test_gossip_available_attestation__valid(spec, state):
    yield "topic", "meta", "available_attestation"

    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    anchor_root = anchor_block.hash_tree_root()
    signed_anchor = wrap_genesis_block(spec, anchor_block)
    yield "state", state
    yield get_filename(signed_anchor), signed_anchor
    yield "blocks", "meta", [{"block": get_filename(signed_anchor)}]

    slot = spec.get_current_slot(store)
    committee = spec.get_available_committee(state, slot)
    validator_index = committee[0]
    data = spec.AvailableAttestationData(
        slot=slot,
        beacon_block_root=anchor_root,
        payload_present=False,
    )
    attestation = spec.AvailableAttestation(data=data)
    for position, member in enumerate(committee):
        if member == validator_index:
            attestation.aggregation_bits[position] = True
    domain = spec.get_domain(
        state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Sign(privkeys[validator_index], signing_root)
    invalid = attestation.copy()
    invalid.data.payload_present = True
    yield get_filename(invalid), invalid
    yield get_filename(attestation), attestation

    current_time_ms = spec.compute_time_at_slot_ms(state, slot)
    yield "current_time_ms", "meta", int(current_time_ms)

    seen = get_seen(spec)
    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        attestation=invalid,
        current_time_ms=current_time_ms,
    )
    assert result == "reject"
    assert reason == "same-slot available attestation signals payload presence"

    result, reason = run_validate_gossip(
        spec,
        seen=seen,
        store=store,
        state=state,
        attestation=attestation,
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
                "reason": "same-slot available attestation signals payload presence",
            },
            {
                "offset_ms": 0,
                "message": get_filename(attestation),
                "expected": "valid",
            },
        ],
    )

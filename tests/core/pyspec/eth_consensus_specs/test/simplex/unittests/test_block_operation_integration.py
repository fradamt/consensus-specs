from eth_consensus_specs.test.context import always_bls, spec_state_test, with_simplex_and_later
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    sign_indexed_attestation,
)
from eth_consensus_specs.test.helpers.block import build_empty_block_for_next_slot
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.payload_attestation import (
    prepare_signed_payload_attestation,
)
from eth_consensus_specs.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block
from eth_consensus_specs.utils import bls


def _build_available_attestation(spec, state, slot, root):
    committee = spec.get_available_committee(state, slot)
    validator_index = committee[0]
    data = spec.AvailableAttestationData(
        slot=slot,
        beacon_block_root=root,
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
    attestation.signature = bls.Sign(
        privkeys[validator_index],
        spec.compute_signing_root(data, domain),
    )
    return attestation, validator_index


def _build_round_evidence(spec, state, slot, root, offender):
    attestation_1 = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=root,
            target=spec.Checkpoint(),
            height=state.current_height,
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        ),
    )
    attestation_2 = attestation_1.copy()
    attestation_2.data.height = spec.Height(attestation_1.data.height + 1)
    sign_indexed_attestation(spec, state, attestation_1)
    sign_indexed_attestation(spec, state, attestation_2)
    return spec.RoundDoubleVoteEvidence(
        attestation_1=attestation_1,
        attestation_2=attestation_2,
    )


def _build_e1_slashing(spec, state, slot, root, offender):
    voted_target = spec.Checkpoint(slot=slot, root=root)
    locked_target = spec.Checkpoint(slot=slot, root=spec.Root(b"\xab" * 32))
    vote = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=root,
            target=voted_target,
            height=state.current_height,
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        ),
    )
    commitment = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=root,
            target=spec.Checkpoint(),
            height=spec.Height(0),
            finality_target=locked_target,
            finality_height=state.current_height,
        ),
    )
    sign_indexed_attestation(spec, state, vote)
    sign_indexed_attestation(spec, state, commitment)
    assert spec.is_slashable_attestation_data(vote.data, commitment.data)
    return spec.AttesterSlashing(attestation_1=vote, attestation_2=commitment)


@with_simplex_and_later
@spec_state_test
@always_bls
def test_signed_block_processes_and_mirrors_simplex_operations_once(spec, state):
    data_slot = spec.GENESIS_SLOT
    payment_index = spec.SLOTS_PER_EPOCH + data_slot % spec.SLOTS_PER_EPOCH
    state.builder_pending_payments[payment_index] = spec.BuilderPendingPayment(
        weight=spec.Gwei(7),
        withdrawal=spec.BuilderPendingWithdrawal(
            fee_recipient=spec.ExecutionAddress(b"\x99" * 20),
            amount=spec.Gwei(11),
            builder_index=spec.BuilderIndex(0),
        ),
    )
    pre_state = state.copy()
    store = get_genesis_forkchoice_store(spec, pre_state)
    anchor_root = store.finalized_checkpoint.root
    available, available_validator = _build_available_attestation(
        spec, pre_state, data_slot, anchor_root
    )
    offender = spec.ValidatorIndex(1)
    evidence = _build_round_evidence(spec, pre_state, data_slot, anchor_root, offender)
    e1_offender = spec.ValidatorIndex(2)
    attester_slashing = _build_e1_slashing(spec, pre_state, data_slot, anchor_root, e1_offender)
    finality_attestation = get_valid_attestation(
        spec,
        pre_state,
        slot=data_slot,
        beacon_block_root=anchor_root,
        signed=True,
    )
    payload_voter = spec.get_ptc(pre_state, data_slot)[0]
    payload_attestation = prepare_signed_payload_attestation(
        spec,
        pre_state,
        slot=data_slot,
        beacon_block_root=anchor_root,
        payload_present=False,
        blob_data_available=False,
        attesting_indices=[payload_voter],
    )

    slashed_index = spec.ValidatorIndex(len(state.validators) - 1)
    proposer_slashing = get_valid_proposer_slashing(
        spec,
        pre_state,
        slashed_index=slashed_index,
        slot=data_slot,
        signed_1=True,
        signed_2=True,
    )
    block = build_empty_block_for_next_slot(spec, state)
    block.body.proposer_slashings = [proposer_slashing]
    block.body.attester_slashings = [attester_slashing]
    block.body.attestations = [finality_attestation]
    block.body.payload_attestations = [payload_attestation]
    block.body.available_attestations = [available]
    block.body.round_double_vote_evidence = [evidence]
    signed_block = state_transition_and_sign_block(spec, state, block)

    assert state.validators[slashed_index].slashed
    assert state.validators[e1_offender].slashed
    assert state.round_double_vote_penalized[offender]
    payment = state.builder_pending_payments[payment_index]
    # Body order is proposer slashing before available processing: the
    # financial claim is cleared, so builder-payment support is no longer
    # accumulated, while the independently required reward replay guard is
    # still written.
    assert payment.withdrawal == spec.BuilderPendingWithdrawal()
    assert payment.weight == spec.Gwei(0)
    assert not any(payment.available_participation)
    assert any(payment.timely_head_participation)

    block_time = spec.uint64(store.genesis_time + block.slot * spec.config.SLOT_DURATION_MS // 1000)
    spec.on_tick(store, block_time)
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)

    calls = {
        "attestation": 0,
        "attester_slashing": 0,
        "payload": 0,
        "available": 0,
        "evidence": 0,
    }
    original_attestation = spec.on_attestation
    original_attester_slashing = spec.on_attester_slashing
    original_payload = spec.on_payload_attestation_message
    original_available = spec.on_available_attestation
    original_evidence = spec.on_round_double_vote_evidence

    def count_attestation(*args, **kwargs):
        calls["attestation"] += 1
        return original_attestation(*args, **kwargs)

    def count_attester_slashing(*args, **kwargs):
        calls["attester_slashing"] += 1
        return original_attester_slashing(*args, **kwargs)

    def count_payload(*args, **kwargs):
        calls["payload"] += 1
        return original_payload(*args, **kwargs)

    def count_available(*args, **kwargs):
        calls["available"] += 1
        return original_available(*args, **kwargs)

    def count_evidence(*args, **kwargs):
        calls["evidence"] += 1
        return original_evidence(*args, **kwargs)

    spec.on_attestation = count_attestation
    spec.on_attester_slashing = count_attester_slashing
    spec.on_payload_attestation_message = count_payload
    spec.on_available_attestation = count_available
    spec.on_round_double_vote_evidence = count_evidence
    try:
        spec.on_block(store, signed_block)
    finally:
        spec.on_attestation = original_attestation
        spec.on_attester_slashing = original_attester_slashing
        spec.on_payload_attestation_message = original_payload
        spec.on_available_attestation = original_available
        spec.on_round_double_vote_evidence = original_evidence

    assert calls == {
        "attestation": 1,
        "attester_slashing": 1,
        "payload": 1,
        "available": 1,
        "evidence": 1,
    }
    block_root = signed_block.message.hash_tree_root()
    assert block_root in store.block_states
    assert store.block_states[block_root].round_double_vote_penalized[offender]
    round_ = spec.compute_round_at_slot(data_slot)
    assert offender in store.round_equivocating_indices[round_]
    assert offender in store.equivocating_indices
    assert e1_offender in store.round_equivocating_indices[round_]
    assert e1_offender in store.equivocating_indices
    assert store.payload_votes[data_slot][payload_voter] == payload_attestation.data
    assert store.available_votes[data_slot][available_validator] == available.data

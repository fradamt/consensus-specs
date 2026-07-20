from eth_consensus_specs.test.context import (
    always_bls,
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    run_attestation_processing,
    sign_attestation,
    sign_indexed_attestation,
)
from eth_consensus_specs.test.helpers.attester_slashings import (
    get_valid_attester_slashing_by_indices,
)
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.state import transition_to_slot_via_block
from eth_consensus_specs.utils import bls
from tests.infra.manifest import manifest


def _prepare_recent_attestation_slot(spec, state):
    data_slot = spec.Slot(spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2)) + 1)
    transition_to_slot_via_block(spec, state, data_slot)
    spec.process_slots(state, spec.Slot(data_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY))
    return data_slot, spec.get_block_root_at_slot(state, data_slot)


def _run_attester_slashing_processing(spec, state, attester_slashing, valid=True):
    yield "pre", state
    yield "attester_slashing", attester_slashing

    if not valid:
        expect_assertion_error(lambda: spec.process_attester_slashing(state, attester_slashing))
        yield "post", None
        return

    offenders = set(attester_slashing.attestation_1.attesting_indices) & set(
        attester_slashing.attestation_2.attesting_indices
    )
    spec.process_attester_slashing(state, attester_slashing)
    assert any(state.validators[index].slashed for index in offenders)
    yield "post", state


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_r1_records_current_height_target_participation(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    assert state.current_height_target != spec.Checkpoint()
    attestation.data.target = state.current_height_target
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)
    attesters = spec.get_attesting_indices(state, attestation)

    yield "pre", state
    yield "attestation", attestation
    spec.process_attestation(state, attestation)
    for index in attesters:
        assert state.target_participation[index]
        assert state.timeouts[index]
    yield "post", state


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_r1_other_on_chain_target_does_not_count(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    later_on_chain_target = spec.Checkpoint(slot=data_slot, root=target_root)
    assert state.current_height_target != spec.Checkpoint()
    assert later_on_chain_target != state.current_height_target
    attestation.data.target = later_on_chain_target
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)
    attesters = spec.get_attesting_indices(state, attestation)

    yield "pre", state
    yield "attestation", attestation
    spec.process_attestation(state, attestation)
    for index in attesters:
        assert not state.target_participation[index]
        assert not state.timeouts[index]
    yield "post", state


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_r2_records_timeout(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)
    attesters = spec.get_attesting_indices(state, attestation)

    yield "pre", state
    yield "attestation", attestation
    spec.process_attestation(state, attestation)
    for index in attesters:
        assert not state.target_participation[index]
        assert state.timeouts[index]
    yield "post", state


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_empty_attestation_records_finality_piggyback(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    state.current_height = spec.Height(3)
    state.justified_height = spec.Height(2)
    state.justified_checkpoint = spec.Checkpoint(slot=data_slot, root=target_root)
    assert state.finalized_checkpoint != state.justified_checkpoint

    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.finality_target = state.justified_checkpoint
    attestation.data.finality_height = state.justified_height
    sign_attestation(spec, state, attestation)
    attesters = spec.get_attesting_indices(state, attestation)

    yield "pre", state
    yield "attestation", attestation
    spec.process_attestation(state, attestation)
    for index in attesters:
        assert state.finality_participation[index]
        assert not state.target_participation[index]
        assert not state.timeouts[index]
    yield "post", state


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_rejects_off_chain_target(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.target = spec.Checkpoint(
        slot=data_slot,
        root=spec.Root(b"\x42" * 32),
    )
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)

    yield from run_attestation_processing(spec, state, attestation, valid=False)


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_rejects_target_after_vote_slot(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.target = spec.Checkpoint(
        slot=spec.Slot(data_slot + 1),
        root=spec.Root(b"\x43" * 32),
    )
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)

    yield from run_attestation_processing(spec, state, attestation, valid=False)


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_rejects_empty_finality_target_with_height(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.finality_height = state.justified_height
    assert attestation.data.finality_height != spec.FAR_FUTURE_HEIGHT
    sign_attestation(spec, state, attestation)

    yield from run_attestation_processing(spec, state, attestation, valid=False)


@manifest(handler_name="attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attestation_rejects_finality_target_without_height(spec, state):
    data_slot, target_root = _prepare_recent_attestation_slot(spec, state)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=data_slot,
        beacon_block_root=target_root,
    )
    attestation.data.finality_target = spec.Checkpoint(slot=data_slot, root=target_root)
    assert attestation.data.finality_height == spec.FAR_FUTURE_HEIGHT
    sign_attestation(spec, state, attestation)

    yield from run_attestation_processing(spec, state, attestation, valid=False)


@manifest(handler_name="attester_slashing")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attester_slashing_e1_conflict(spec, state):
    offender = spec.ValidatorIndex(0)
    attester_slashing = get_valid_attester_slashing_by_indices(
        spec,
        state,
        [offender],
        signed_1=True,
        signed_2=True,
    )

    yield from _run_attester_slashing_processing(spec, state, attester_slashing)


@manifest(handler_name="attester_slashing")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_attester_slashing_rejects_signed_non_slashable_votes(spec, state):
    offender = spec.ValidatorIndex(0)
    attester_slashing = get_valid_attester_slashing_by_indices(
        spec,
        state,
        [offender],
        signed_1=True,
        signed_2=False,
    )
    attestation_2 = attester_slashing.attestation_2
    attestation_2.data.finality_target = spec.Checkpoint()
    attestation_2.data.finality_height = spec.FAR_FUTURE_HEIGHT
    sign_indexed_attestation(spec, state, attestation_2)
    assert not spec.is_slashable_attestation_data(
        attester_slashing.attestation_1.data,
        attestation_2.data,
    )
    assert spec.is_valid_indexed_attestation(state, attester_slashing.attestation_1)
    assert spec.is_valid_indexed_attestation(state, attestation_2)

    yield from _run_attester_slashing_processing(
        spec,
        state,
        attester_slashing,
        valid=False,
    )


@manifest(handler_name="available_attestation")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_attestation(spec, state):
    data_slot = state.slot
    spec.process_slots(state, spec.Slot(data_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY))
    committee = spec.get_available_committee(state, data_slot)
    validator_index = committee[0]
    data = spec.AvailableAttestationData(
        slot=data_slot,
        beacon_block_root=spec.get_block_root_at_slot(state, data_slot),
        payload_present=False,
    )
    attestation = spec.AvailableAttestation(data=data)
    for position, member in enumerate(committee):
        if member == validator_index:
            attestation.aggregation_bits[position] = True
    domain = spec.get_domain(
        state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(data_slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Sign(privkeys[validator_index], signing_root)

    yield "pre", state
    yield "available_attestation", attestation
    spec.process_available_attestation(state, attestation)
    yield "post", state


@manifest(handler_name="round_double_vote_evidence")
@with_simplex_and_later
@spec_state_test
@always_bls
def test_round_double_vote_evidence(spec, state):
    offender = spec.ValidatorIndex(0)
    evidence_slot = state.slot
    attestation_1 = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=evidence_slot,
            beacon_block_root=spec.Root(),
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
    evidence = spec.RoundDoubleVoteEvidence(
        attestation_1=attestation_1,
        attestation_2=attestation_2,
    )
    spec.process_slots(state, spec.Slot(evidence_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY))

    yield "pre", state
    yield "round_double_vote_evidence", evidence
    spec.process_round_double_vote_evidence(state, evidence)
    yield "post", state

from eth_consensus_specs.test.context import (
    expect_assertion_error,
    spec_configured_state_test,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    sign_attestation,
)
from eth_consensus_specs.test.helpers.merkle import build_proof


@with_simplex_and_later
@spec_state_test
def test_historical_block_proof_uses_summary_origin_across_period_boundary(spec, state):
    period_length = spec.SLOTS_PER_HISTORICAL_ROOT
    proof_depth = spec.BLOCK_ROOTS_PROOF_DEPTH
    summary_origin = 7

    previous_root = spec.Root(b"\x11" * 32)
    target_root = spec.Root(b"\x22" * 32)
    previous_roots = spec.Vector[spec.Root, spec.SLOTS_PER_HISTORICAL_ROOT]()
    target_roots = spec.Vector[spec.Root, spec.SLOTS_PER_HISTORICAL_ROOT]()
    previous_roots[period_length - 1] = previous_root
    target_roots[0] = target_root

    state.historical_summaries = [
        spec.HistoricalSummary(block_summary_root=previous_roots.hash_tree_root()),
        spec.HistoricalSummary(block_summary_root=target_roots.hash_tree_root()),
    ]
    state.slot = spec.Slot((summary_origin + 2) * period_length)

    target_slot = spec.Slot((summary_origin + 1) * period_length)
    target = spec.Checkpoint(slot=target_slot, root=target_root)
    proof = spec.HistoricalBlockProof(
        slot=target_slot,
        block_root=target_root,
        block_proof=build_proof(
            target_roots.get_backing(),
            2**proof_depth,
        ),
        prev_slot_root=previous_root,
        prev_slot_proof=build_proof(
            previous_roots.get_backing(),
            2**proof_depth + period_length - 1,
        ),
    )

    spec.verify_historical_block_proof(state, target, proof)


@with_simplex_and_later
@spec_state_test
def test_available_attestation_rejects_pre_fork_slot(spec, state):
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    state.slot = fork_slot
    attestation = spec.AvailableAttestation(
        data=spec.AvailableAttestationData(slot=spec.Slot(fork_slot - 1))
    )

    expect_assertion_error(lambda: spec.process_available_attestation(state, attestation))


@with_simplex_and_later
@spec_state_test
def test_finality_attestation_activation_boundary(spec, state):
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    state.slot = spec.Slot(fork_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY)

    pre_fork_attestation = get_valid_attestation(
        spec,
        state,
        slot=spec.Slot(fork_slot - 1),
        signed=True,
    )
    expect_assertion_error(lambda: spec.validate_attestation(state, pre_fork_attestation))

    fork_attestation = get_valid_attestation(
        spec,
        state,
        slot=fork_slot,
        signed=True,
    )
    spec.validate_attestation(state, fork_attestation)


@with_simplex_and_later
@spec_configured_state_test({"SIMPLEX_FORK_EPOCH": 1})
def test_later_fork_accepts_previous_epoch_simplex_attestation(spec, state):
    later_fork_epoch = spec.Epoch(3)
    state.fork.previous_version = spec.config.SIMPLEX_FORK_VERSION
    state.fork.current_version = spec.Version("0x11000000")
    state.fork.epoch = later_fork_epoch
    state.slot = spec.compute_start_slot_at_epoch(later_fork_epoch)

    previous_epoch_slot = spec.Slot(state.slot - 1)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=previous_epoch_slot,
        signed=True,
    )

    assert spec.compute_epoch_at_slot(previous_epoch_slot) == later_fork_epoch - 1
    spec.validate_attestation(state, attestation)


@with_simplex_and_later
@spec_state_test
def test_pending_height_outcome_includes_final_slot_timeout_vote(spec, state):
    # Use an epoch after the startup guard and its final slot, which is also a
    # round boundary under the required epoch-aligned schedule.
    duty_epoch = spec.Epoch(spec.GENESIS_EPOCH + 2)
    next_epoch_start = spec.compute_start_slot_at_epoch(spec.Epoch(duty_epoch + 1))
    final_slot = spec.Slot(next_epoch_start - 1)
    state.slot = final_slot

    attestation = get_valid_attestation(
        spec,
        state,
        slot=final_slot,
        beacon_block_root=spec.Root(),
    )
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)
    attesters = spec.get_attesting_indices(state, attestation)

    active_indices = spec.get_active_validator_indices(state, duty_epoch)
    total_balance = sum(state.validators[index].effective_balance for index in active_indices)
    attester_balance = sum(state.validators[index].effective_balance for index in attesters)
    quorum = (
        total_balance * spec.FINALITY_QUORUM_NUMERATOR + spec.FINALITY_QUORUM_DENOMINATOR - 1
    ) // spec.FINALITY_QUORUM_DENOMINATOR

    # Seed a sub-quorum timeout tally that the final-slot committee completes.
    pre_vote_balance = 0
    for index in active_indices:
        if index in attesters or pre_vote_balance + attester_balance >= quorum:
            continue
        state.timeouts[index] = True
        pre_vote_balance += state.validators[index].effective_balance

    assert pre_vote_balance < quorum
    assert pre_vote_balance + attester_balance >= quorum
    assert not spec.has_timeout_quorum(state)

    pre_height = state.current_height
    spec.process_round(state)
    assert state.pending_height_outcomes == 1
    assert state.current_height == pre_height

    state.slot = next_epoch_start
    spec.process_attestation(state, attestation)
    assert spec.has_timeout_quorum(state)

    spec.process_pending_height_outcome(state)
    assert state.pending_height_outcomes == 0
    assert state.current_height == pre_height + 1
    assert state.current_height_start_slot == next_epoch_start


@with_simplex_and_later
@spec_state_test
def test_empty_slot_settles_pending_outcome_at_next_slot(spec, state):
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))
    for index in spec.get_active_validator_indices(state, spec.get_current_epoch(state)):
        state.timeouts[index] = True
    assert spec.has_timeout_quorum(state)

    pre_slot = state.slot
    pre_height = state.current_height
    state.pending_height_outcomes = 1

    spec.process_slot(state)

    assert state.pending_height_outcomes == 0
    assert state.current_height == pre_height + 1
    assert state.current_height_start_slot == pre_slot + 1

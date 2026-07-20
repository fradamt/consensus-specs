from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth_consensus_specs.utils import bls


def _prepare_inclusion_state(spec, state, inclusion_slot=1):
    spec.process_slots(state, spec.Slot(inclusion_slot))
    return spec.GENESIS_SLOT


def _canonical_root_at(spec, state, slot):
    return spec.get_block_root_at_slot(state, slot)


def _positions_for_distinct_validators(committee, count):
    positions = []
    seen = set()
    for position, validator_index in enumerate(committee):
        if validator_index in seen:
            continue
        positions.append(position)
        seen.add(validator_index)
        if len(positions) == count:
            break
    assert len(positions) == count
    return positions


def _duplicate_positions(committee):
    positions_by_validator = {}
    for position, validator_index in enumerate(committee):
        positions_by_validator.setdefault(validator_index, []).append(position)
    validator_index, positions = max(
        positions_by_validator.items(),
        key=lambda item: len(item[1]),
    )
    assert len(positions) > 1
    return validator_index, positions


def _build_signed_available_attestation(
    spec,
    state,
    slot,
    positions,
    beacon_block_root,
    payload_present=False,
):
    data = spec.AvailableAttestationData(
        slot=slot,
        payload_present=payload_present,
        beacon_block_root=beacon_block_root,
    )
    attestation = spec.AvailableAttestation(data=data)
    for position in positions:
        attestation.aggregation_bits[position] = True

    committee = spec.get_available_committee(state, slot)
    attesting_indices = sorted({committee[position] for position in positions})
    domain = spec.get_domain(
        state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Aggregate(
        [bls.Sign(privkeys[index], signing_root) for index in attesting_indices]
    )
    return attestation


def _payment_index(spec, state, slot):
    if spec.compute_epoch_at_slot(slot) == spec.get_current_epoch(state):
        return spec.SLOTS_PER_EPOCH + slot % spec.SLOTS_PER_EPOCH
    return slot % spec.SLOTS_PER_EPOCH


def _build_payment(spec, amount, builder_index=0):
    return spec.BuilderPendingPayment(
        weight=spec.Gwei(0),
        withdrawal=spec.BuilderPendingWithdrawal(
            fee_recipient=spec.ExecutionAddress(bytes([builder_index + 1]) * 20),
            amount=spec.Gwei(amount),
            builder_index=spec.BuilderIndex(builder_index),
        ),
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_process_available_attestation_valid_signed_aggregate(spec, state):
    data_slot = _prepare_inclusion_state(spec, state)
    committee = spec.get_available_committee(state, data_slot)
    positions = _positions_for_distinct_validators(committee, 2)
    attestation = _build_signed_available_attestation(
        spec,
        state,
        data_slot,
        positions,
        _canonical_root_at(spec, state, data_slot),
    )

    spec.process_available_attestation(state, attestation)

    for index in spec.get_available_attesting_indices(state, attestation):
        assert spec.has_flag(
            state.current_round_participation[index],
            spec.TIMELY_HEAD_FLAG_INDEX,
        )


@with_simplex_and_later
@spec_state_test
def test_available_attestation_duplicate_seats_and_replay_are_idempotent(spec, state):
    data_slot = _prepare_inclusion_state(spec, state)
    committee = spec.get_available_committee(state, data_slot)
    validator_index, duplicate_positions = _duplicate_positions(committee)
    attestation = _build_signed_available_attestation(
        spec,
        state,
        data_slot,
        [duplicate_positions[0]],
        _canonical_root_at(spec, state, data_slot),
    )
    assert spec.get_available_attesting_indices(state, attestation) == {validator_index}
    assert spec.get_available_attesting_positions(state, attestation) == set(duplicate_positions)

    payment_index = _payment_index(spec, state, data_slot)
    state.builder_pending_payments[payment_index] = _build_payment(spec, amount=1)

    spec.process_available_attestation(state, attestation)
    first_balances = list(state.balances)
    first_available_participation = list(
        state.builder_pending_payments[payment_index].available_participation
    )
    first_timely_head_participation = list(
        state.builder_pending_payments[payment_index].timely_head_participation
    )

    spec.process_available_attestation(state, attestation)
    second_payment = state.builder_pending_payments[payment_index]

    assert list(state.balances) == first_balances
    assert list(second_payment.available_participation) == first_available_participation
    assert list(second_payment.timely_head_participation) == first_timely_head_participation
    assert sum(second_payment.available_participation) == len(duplicate_positions)
    assert sum(second_payment.timely_head_participation) == len(duplicate_positions)


@with_simplex_and_later
@spec_state_test
@always_bls
def test_proposer_slashing_preserves_available_reward_replay_guard(spec, state):
    data_slot = _prepare_inclusion_state(spec, state)
    committee = spec.get_available_committee(state, data_slot)
    positions = _positions_for_distinct_validators(committee, 1)
    attestation = _build_signed_available_attestation(
        spec,
        state,
        data_slot,
        positions,
        _canonical_root_at(spec, state, data_slot),
    )
    payment_index = _payment_index(spec, state, data_slot)
    state.builder_pending_payments[payment_index] = _build_payment(spec, amount=1)

    spec.process_available_attestation(state, attestation)
    paid_bitmap = list(state.builder_pending_payments[payment_index].timely_head_participation)
    assert any(paid_bitmap)

    # Move to S + 2 while the same vote remains in the round-based inclusion
    # window, then slash the slot-S proposer. The financial claim is cancelled,
    # but its reward-replay bitmap must survive.
    spec.process_slots(state, spec.Slot(data_slot + 2))
    attesting_indices = spec.get_available_attesting_indices(state, attestation)
    slashed_index = next(
        index
        for index in spec.get_active_validator_indices(state, spec.get_current_epoch(state))
        if index not in attesting_indices
    )
    proposer_slashing = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=slashed_index,
        slot=data_slot,
        signed_1=True,
        signed_2=True,
    )
    spec.process_proposer_slashing(state, proposer_slashing)

    payment = state.builder_pending_payments[payment_index]
    assert payment.withdrawal == spec.BuilderPendingWithdrawal()
    assert list(payment.timely_head_participation) == paid_bitmap

    post_slashing_balances = list(state.balances)
    spec.process_available_attestation(state, attestation)
    assert list(state.balances) == post_slashing_balances


@with_simplex_and_later
@spec_state_test
@always_bls
def test_slashed_builder_payment_with_available_quorum_does_not_settle(spec, state):
    # Put the proposal in the previous-epoch half of the payment ring so the
    # normal settlement helper will inspect it after cancellation.
    spec.process_slots(state, spec.compute_start_slot_at_epoch(spec.Epoch(1)))
    proposal_slot = spec.GENESIS_SLOT
    payment_index = proposal_slot % spec.SLOTS_PER_EPOCH
    payment = _build_payment(spec, amount=1)
    threshold = (
        spec.AVAILABLE_COMMITTEE_SIZE * spec.BUILDER_PAYMENT_THRESHOLD_NUMERATOR
        + spec.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        - 1
    ) // spec.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
    for position in range(threshold):
        payment.available_participation[position] = True
    payment.timely_head_participation[0] = True
    state.builder_pending_payments[payment_index] = payment

    slashed_index = next(
        index
        for index in spec.get_active_validator_indices(state, spec.get_current_epoch(state))
        if spec.is_slashable_validator(state.validators[index], spec.get_current_epoch(state))
    )
    proposer_slashing = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=slashed_index,
        slot=proposal_slot,
        signed_1=True,
        signed_2=True,
    )
    spec.process_proposer_slashing(state, proposer_slashing)

    canceled_payment = state.builder_pending_payments[payment_index]
    assert canceled_payment.withdrawal == spec.BuilderPendingWithdrawal()
    assert sum(canceled_payment.available_participation) == threshold
    assert canceled_payment.timely_head_participation[0]

    pre_withdrawal_count = len(state.builder_pending_withdrawals)
    spec.process_builder_pending_payments(state)
    assert len(state.builder_pending_withdrawals) == pre_withdrawal_count


@with_simplex_and_later
@spec_state_test
def test_available_attestation_timely_head_rewards_attester_seats_and_proposer(spec, state):
    data_slot = _prepare_inclusion_state(spec, state)
    committee = spec.get_available_committee(state, data_slot)
    _, duplicate_positions = _duplicate_positions(committee)
    attestation = _build_signed_available_attestation(
        spec,
        state,
        data_slot,
        [duplicate_positions[0]],
        _canonical_root_at(spec, state, data_slot),
    )
    attesting_positions = spec.get_available_attesting_positions(state, attestation)

    seat_reward = spec.get_available_head_reward_per_seat(state, data_slot)
    assert seat_reward > 0
    proposer_index = spec.get_beacon_proposer_index(state)
    proposer_denominator = (spec.WEIGHT_DENOMINATOR - spec.PROPOSER_WEIGHT) // spec.PROPOSER_WEIGHT
    proposer_reward = len(attesting_positions) * seat_reward // proposer_denominator

    expected_increases = {}
    for position in attesting_positions:
        index = committee[position]
        expected_increases[index] = expected_increases.get(index, 0) + seat_reward
    expected_increases[proposer_index] = expected_increases.get(proposer_index, 0) + proposer_reward
    pre_balances = list(state.balances)

    spec.process_available_attestation(state, attestation)

    for index, pre_balance in enumerate(pre_balances):
        assert state.balances[index] == pre_balance + expected_increases.get(index, 0)


@with_simplex_and_later
@spec_state_test
def test_available_attestation_records_builder_bitmap_when_included_late(spec, state):
    data_slot = _prepare_inclusion_state(spec, state, inclusion_slot=2)
    committee = spec.get_available_committee(state, data_slot)
    _, duplicate_positions = _duplicate_positions(committee)
    attestation = _build_signed_available_attestation(
        spec,
        state,
        data_slot,
        [duplicate_positions[0]],
        _canonical_root_at(spec, state, data_slot),
    )
    payment_index = _payment_index(spec, state, data_slot)
    state.builder_pending_payments[payment_index] = _build_payment(spec, amount=1)

    spec.process_available_attestation(state, attestation)

    payment = state.builder_pending_payments[payment_index]
    assert {
        position
        for position, participated in enumerate(payment.available_participation)
        if participated
    } == set(duplicate_positions)
    assert not any(payment.timely_head_participation)


@with_simplex_and_later
@spec_state_test
def test_builder_payment_available_bitmap_threshold(spec, state):
    threshold = (
        spec.AVAILABLE_COMMITTEE_SIZE * spec.BUILDER_PAYMENT_THRESHOLD_NUMERATOR
        + spec.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        - 1
    ) // spec.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
    assert 0 < threshold <= spec.AVAILABLE_COMMITTEE_SIZE

    below_payment = _build_payment(spec, amount=11, builder_index=0)
    threshold_payment = _build_payment(spec, amount=22, builder_index=1)
    for position in range(threshold - 1):
        below_payment.available_participation[position] = True
    for position in range(threshold):
        threshold_payment.available_participation[position] = True
    state.builder_pending_payments[0] = below_payment
    state.builder_pending_payments[1] = threshold_payment

    pre_withdrawal_count = len(state.builder_pending_withdrawals)
    spec.process_builder_pending_payments(state)

    new_withdrawals = state.builder_pending_withdrawals[pre_withdrawal_count:]
    assert len(new_withdrawals) == 1
    assert new_withdrawals[0] == threshold_payment.withdrawal


@with_simplex_and_later
@spec_state_test
def test_builder_payment_legacy_quorum_is_not_suppressed_by_partial_available_bitmap(spec, state):
    legacy_quorum = spec.get_builder_payment_quorum_threshold(state)
    payment = _build_payment(spec, amount=33, builder_index=0)
    payment.weight = spec.Gwei(legacy_quorum)
    payment.available_participation[0] = True
    assert (
        sum(payment.available_participation) * spec.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        < spec.AVAILABLE_COMMITTEE_SIZE * spec.BUILDER_PAYMENT_THRESHOLD_NUMERATOR
    )
    state.builder_pending_payments[0] = payment

    pre_withdrawal_count = len(state.builder_pending_withdrawals)
    spec.process_builder_pending_payments(state)

    new_withdrawals = state.builder_pending_withdrawals[pre_withdrawal_count:]
    assert new_withdrawals == [payment.withdrawal]

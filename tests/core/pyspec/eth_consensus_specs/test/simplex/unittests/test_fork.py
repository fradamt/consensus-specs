from eth_consensus_specs.test.context import (
    always_bls,
    spec_test,
    with_phases,
    with_state,
)
from eth_consensus_specs.test.helpers.block import apply_empty_block
from eth_consensus_specs.test.helpers.constants import (
    GLOAS,
    SIMPLEX,
)
from eth_consensus_specs.test.helpers.genesis import build_mock_builder
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.state import (
    set_full_participation,
    transition_to,
)
from eth_consensus_specs.utils import bls


def _build_pending_payment(spec, amount, builder_index=0, weight=0):
    return spec.BuilderPendingPayment(
        weight=spec.Gwei(weight),
        withdrawal=spec.BuilderPendingWithdrawal(
            fee_recipient=spec.ExecutionAddress(bytes([builder_index + 1]) * 20),
            amount=spec.Gwei(amount),
            builder_index=spec.BuilderIndex(builder_index),
        ),
    )


def _build_signed_available_attestation(
    spec,
    state,
    slot,
    positions,
    root,
    payload_present=False,
):
    data = spec.AvailableAttestationData(
        slot=slot,
        payload_present=payload_present,
        beacon_block_root=root,
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


def _prepare_payment_bridge_state(spec, simplex, state):
    fork_epoch = spec.Epoch(2)
    activation_slot = spec.compute_start_slot_at_epoch(fork_epoch)
    transition_to(spec, state, spec.Slot(activation_slot - 1))

    # These entries are in Gloas's current-epoch half. Advancing through the
    # fork boundary runs Gloas epoch processing and rotates them into the first
    # half before the Simplex state upgrade.
    final_slot = spec.Slot(activation_slot - 1)
    earlier_slot = spec.Slot(activation_slot - 2)
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH + final_slot % spec.SLOTS_PER_EPOCH] = (
        _build_pending_payment(spec, amount=11, builder_index=0)
    )
    state.builder_pending_payments[spec.SLOTS_PER_EPOCH + earlier_slot % spec.SLOTS_PER_EPOCH] = (
        _build_pending_payment(spec, amount=22, builder_index=1)
    )

    spec.process_slots(state, activation_slot)
    post = simplex.upgrade_to_simplex(state)
    return post, activation_slot, (final_slot, earlier_slot)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
def test_upgrade_to_simplex_initializes_state(spec, phases, state):
    simplex = phases[SIMPLEX]

    # Build a non-genesis Gloas state whose justified root is carried across an
    # empty epoch-boundary slot, matching the legacy checkpoint semantics that
    # Simplex must preserve at height 0.
    justified_epoch = spec.Epoch(2)
    finalized_epoch = spec.Epoch(1)
    proposal_slot = spec.Slot(spec.compute_start_slot_at_epoch(justified_epoch) - 1)
    apply_empty_block(spec, state, proposal_slot)
    fork_epoch = spec.Epoch(justified_epoch + 1)
    transition_to(spec, state, spec.compute_start_slot_at_epoch(fork_epoch))

    state.current_justified_checkpoint = spec.Checkpoint(
        epoch=justified_epoch,
        root=spec.get_block_root(state, justified_epoch),
    )
    state.previous_justified_checkpoint = state.current_justified_checkpoint
    state.finalized_checkpoint = spec.Checkpoint(
        epoch=finalized_epoch,
        root=spec.get_block_root(state, finalized_epoch),
    )

    # Make every reset/conversion observable rather than relying on genesis
    # defaults.
    set_full_participation(spec, state)
    state.builders.append(build_mock_builder(spec, 0, 4 * spec.MIN_ACTIVATION_BALANCE))
    payment_index = 1
    payment = spec.BuilderPendingPayment(
        weight=spec.Gwei(3 * spec.MIN_ACTIVATION_BALANCE),
        withdrawal=spec.BuilderPendingWithdrawal(
            fee_recipient=spec.ExecutionAddress(b"\x42" * 20),
            amount=spec.Gwei(2 * spec.MIN_ACTIVATION_BALANCE),
            builder_index=spec.BuilderIndex(0),
        ),
    )
    state.builder_pending_payments[payment_index] = payment

    assert any(state.previous_epoch_participation)
    assert any(state.current_epoch_participation)

    expected_accounting = state.copy()
    spec.process_inactivity_updates(expected_accounting)
    spec.process_rewards_and_penalties(expected_accounting)
    pre_balances = list(state.balances)
    pre_inactivity_scores = list(state.inactivity_scores)
    post = simplex.upgrade_to_simplex(state)

    # The final Gloas epoch has just rotated into ``previous`` and is still
    # unpaid. Upgrade settles it once with Gloas rules on a copy.
    assert post.balances == expected_accounting.balances
    assert post.inactivity_scores == expected_accounting.inactivity_scores
    assert state.balances == pre_balances
    assert state.inactivity_scores == pre_inactivity_scores

    stable_fields = (
        "genesis_time",
        "genesis_validators_root",
        "slot",
        "latest_block_header",
        "block_roots",
        "state_roots",
        "historical_roots",
        "eth1_data",
        "eth1_data_votes",
        "eth1_deposit_index",
        "validators",
        "randao_mixes",
        "slashings",
        "current_sync_committee",
        "next_sync_committee",
        "latest_execution_payload_bid",
        "next_withdrawal_index",
        "next_withdrawal_validator_index",
        "historical_summaries",
        "deposit_requests_start_index",
        "deposit_balance_to_consume",
        "exit_balance_to_consume",
        "earliest_exit_epoch",
        "consolidation_balance_to_consume",
        "earliest_consolidation_epoch",
        "pending_deposits",
        "pending_partial_withdrawals",
        "pending_consolidations",
        "proposer_lookahead",
        "builders",
        "next_withdrawal_builder_index",
        "execution_payload_availability",
        "builder_pending_withdrawals",
        "latest_block_hash",
        "payload_expected_withdrawals",
        "ptc_window",
    )
    for field in stable_fields:
        assert getattr(post, field) == getattr(state, field)

    assert post.fork.previous_version == state.fork.current_version
    assert post.fork.current_version == simplex.config.SIMPLEX_FORK_VERSION
    assert post.fork.epoch == fork_epoch

    assert post.justified_checkpoint == simplex.Checkpoint(
        slot=simplex.compute_start_slot_at_epoch(justified_epoch),
        root=state.current_justified_checkpoint.root,
    )
    assert post.finalized_checkpoint == simplex.Checkpoint(
        slot=simplex.compute_start_slot_at_epoch(finalized_epoch),
        root=state.finalized_checkpoint.root,
    )
    assert post.justified_height == simplex.Height(0)
    assert post.finalized_height == simplex.Height(0)
    assert post.justified_checkpoint.slot != state.latest_block_header.slot

    assert not hasattr(post, "justification_bits")
    assert not hasattr(post, "previous_justified_checkpoint")
    assert not hasattr(post, "current_justified_checkpoint")
    assert not hasattr(post, "previous_epoch_participation")
    assert not hasattr(post, "current_epoch_participation")
    assert len(post.previous_round_participation) == len(state.validators)
    assert len(post.current_round_participation) == len(state.validators)
    assert not any(post.previous_round_participation)
    assert not any(post.current_round_participation)

    post_payment = post.builder_pending_payments[payment_index]
    assert len(post.builder_pending_payments) == len(state.builder_pending_payments)
    assert post_payment.weight == payment.weight
    assert post_payment.withdrawal.fee_recipient == payment.withdrawal.fee_recipient
    assert post_payment.withdrawal.amount == payment.withdrawal.amount
    assert post_payment.withdrawal.builder_index == payment.withdrawal.builder_index
    assert len(post_payment.available_participation) == simplex.AVAILABLE_COMMITTEE_SIZE
    assert len(post_payment.timely_head_participation) == simplex.AVAILABLE_COMMITTEE_SIZE
    assert not any(post_payment.available_participation)
    assert not any(post_payment.timely_head_participation)

    assert post.current_height == simplex.GENESIS_HEIGHT
    assert not post.current_height_nonjustifiable
    assert post.current_height_start_slot == state.slot
    assert post.current_height_target == simplex.Checkpoint()
    assert post.pending_height_outcomes == 0
    assert len(post.target_participation) == len(state.validators)
    assert not any(post.target_participation)
    assert len(post.timeouts) == len(state.validators)
    assert len(post.finality_participation) == len(state.validators)
    assert len(post.round_double_vote_penalized) == len(state.validators)
    assert not any(post.timeouts)
    assert not any(post.finality_participation)
    assert not any(post.round_double_vote_penalized)

    expected_window = simplex.initialize_available_committee_window(post)
    assert len(post.available_committee_window) == len(expected_window)
    for actual, expected in zip(post.available_committee_window, expected_window, strict=True):
        assert actual == expected

    previous_epoch_window = post.available_committee_window[: simplex.SLOTS_PER_EPOCH]
    assert all(index == 0 for committee in previous_epoch_window for index in committee)

    fork_slot = simplex.compute_start_slot_at_epoch(fork_epoch)
    assert simplex.get_available_committee(post, fork_slot) == simplex.compute_available_committee(
        post, fork_slot
    )


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
def test_upgrade_settles_final_gloas_epoch_participation_once(spec, phases, state):
    simplex = phases[SIMPLEX]
    fork_epoch = spec.Epoch(2)
    activation_slot = spec.compute_start_slot_at_epoch(fork_epoch)
    transition_to(spec, state, spec.Slot(activation_slot - 1))

    # Make the final Gloas epoch's current flags observable. Gloas boundary
    # processing pays the older ``previous`` interval, then rotates this exact
    # pattern into ``previous_epoch_participation`` without paying it.
    final_epoch_flags = [spec.ParticipationFlags(0b0000_0111) for _ in range(len(state.validators))]
    final_epoch_flags[0] = spec.ParticipationFlags(0)
    state.previous_epoch_participation = [
        spec.ParticipationFlags(0) for _ in range(len(state.validators))
    ]
    state.current_epoch_participation = final_epoch_flags
    spec.process_slots(state, activation_slot)

    assert state.previous_epoch_participation == final_epoch_flags
    assert not any(state.current_epoch_participation)
    expected = state.copy()
    spec.process_inactivity_updates(expected)
    spec.process_rewards_and_penalties(expected)
    pre_balances = list(state.balances)
    pre_scores = list(state.inactivity_scores)

    post = simplex.upgrade_to_simplex(state)

    assert post.balances == expected.balances
    assert post.inactivity_scores == expected.inactivity_scores
    assert state.balances == pre_balances
    assert state.inactivity_scores == pre_scores
    assert not any(post.previous_round_participation)
    assert not any(post.current_round_participation)

    # The first Simplex boundary rotates the first native round into
    # ``previous`` but must not reinterpret the already settled Gloas interval
    # represented by the initially empty previous-round array.
    full_flags = [simplex.ParticipationFlags(0b0000_0111) for _ in range(len(post.validators))]
    post.current_round_participation = full_flags
    activation_round = simplex.compute_round_at_slot(activation_slot)
    next_round_start = simplex.compute_start_slot_at_round(simplex.Round(activation_round + 1))
    post.slot = simplex.Slot(next_round_start - 1)
    settled_balances = list(post.balances)
    simplex.process_round(post)

    assert post.balances == settled_balances
    assert post.previous_round_participation == full_flags
    assert not any(post.current_round_participation)

    # One round later, the first native Simplex interval is due and is paid.
    following_round_start = simplex.compute_start_slot_at_round(simplex.Round(activation_round + 2))
    post.slot = simplex.Slot(following_round_start - 1)
    simplex.process_round(post)
    assert post.balances != settled_balances


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_available_receipt_bridges_final_gloas_epoch_payments(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, payment_slots = _prepare_payment_bridge_state(spec, simplex, state)

    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert payment.withdrawal.amount > 0
        assert payment.weight == 0
        assert not any(payment.available_participation)

    simplex.process_slots(post, simplex.Slot(activation_slot + 1))
    activation_root = simplex.get_block_root_at_slot(post, activation_slot)
    receipt = _build_signed_available_attestation(
        simplex,
        post,
        activation_slot,
        range(simplex.AVAILABLE_COMMITTEE_SIZE),
        activation_root,
    )
    assert post.fork.previous_version == simplex.config.GLOAS_FORK_VERSION
    assert simplex.get_current_epoch(post) == post.fork.epoch
    assert receipt.data.slot == simplex.compute_start_slot_at_epoch(post.fork.epoch)
    assert receipt.data.beacon_block_root == simplex.get_block_root_at_slot(post, receipt.data.slot)
    simplex.process_available_attestation(post, receipt)

    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert sum(payment.available_participation) == simplex.AVAILABLE_COMMITTEE_SIZE

    # Transition support is applied only to the legacy first half. There was
    # no activation-slot block, so the regular current-epoch payment entry has
    # no slot-local availability support.
    current_payment = post.builder_pending_payments[
        simplex.SLOTS_PER_EPOCH + activation_slot % simplex.SLOTS_PER_EPOCH
    ]
    assert not any(current_payment.available_participation)

    pre_withdrawal_count = len(post.builder_pending_withdrawals)
    simplex.process_builder_pending_payments(post)
    new_withdrawals = post.builder_pending_withdrawals[pre_withdrawal_count:]
    assert [withdrawal.amount for withdrawal in new_withdrawals] == [
        simplex.Gwei(22),
        simplex.Gwei(11),
    ]


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_available_receipt_nonmatching_head_does_not_bridge(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, payment_slots = _prepare_payment_bridge_state(spec, simplex, state)
    simplex.process_slots(post, simplex.Slot(activation_slot + 1))

    receipt = _build_signed_available_attestation(
        simplex,
        post,
        activation_slot,
        range(simplex.AVAILABLE_COMMITTEE_SIZE),
        simplex.Root(b"\x99" * 32),
    )
    simplex.process_available_attestation(post, receipt)

    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert not any(payment.available_participation)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_available_receipt_below_threshold_does_not_pay(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, payment_slots = _prepare_payment_bridge_state(spec, simplex, state)
    simplex.process_slots(post, simplex.Slot(activation_slot + 1))

    activation_root = simplex.get_block_root_at_slot(post, activation_slot)
    committee = simplex.get_available_committee(post, activation_slot)
    positions_by_validator = {}
    for position, validator_index in enumerate(committee):
        positions_by_validator.setdefault(validator_index, []).append(position)
    selected_positions = min(positions_by_validator.values(), key=len)
    receipt = _build_signed_available_attestation(
        simplex,
        post,
        activation_slot,
        [selected_positions[0]],
        activation_root,
    )
    simplex.process_available_attestation(post, receipt)

    threshold = (
        simplex.AVAILABLE_COMMITTEE_SIZE * simplex.BUILDER_PAYMENT_THRESHOLD_NUMERATOR
        + simplex.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        - 1
    ) // simplex.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert 0 < sum(payment.available_participation) < threshold

    pre_withdrawal_count = len(post.builder_pending_withdrawals)
    simplex.process_builder_pending_payments(post)
    assert len(post.builder_pending_withdrawals) == pre_withdrawal_count


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_receipt_unions_root_support_across_payload_signals(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, payment_slots = _prepare_payment_bridge_state(spec, simplex, state)
    simplex.process_slots(post, simplex.Slot(activation_slot + 1))
    activation_root = simplex.get_block_root_at_slot(post, activation_slot)

    # Keep validator identities disjoint across the two complete data values,
    # while balancing their duplicate-seat weight below the payment threshold.
    positions_by_validator = {}
    for position, validator_index in enumerate(
        simplex.get_available_committee(post, activation_slot)
    ):
        positions_by_validator.setdefault(validator_index, []).append(position)
    position_groups = [[], []]
    for positions in sorted(positions_by_validator.values(), key=len, reverse=True):
        group = 0 if len(position_groups[0]) <= len(position_groups[1]) else 1
        position_groups[group].extend(positions)

    threshold = (
        simplex.AVAILABLE_COMMITTEE_SIZE * simplex.BUILDER_PAYMENT_THRESHOLD_NUMERATOR
        + simplex.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        - 1
    ) // simplex.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
    assert all(0 < len(positions) < threshold for positions in position_groups)

    for payload_present, positions in zip((False, True), position_groups, strict=True):
        receipt = _build_signed_available_attestation(
            simplex,
            post,
            activation_slot,
            positions,
            activation_root,
            payload_present=payload_present,
        )
        simplex.process_available_attestation(post, receipt)

    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert sum(payment.available_participation) == simplex.AVAILABLE_COMMITTEE_SIZE

    pre_withdrawal_count = len(post.builder_pending_withdrawals)
    simplex.process_builder_pending_payments(post)
    assert len(post.builder_pending_withdrawals) > pre_withdrawal_count


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_available_receipt_is_includable_until_payment_rotation(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, payment_slots = _prepare_payment_bridge_state(spec, simplex, state)
    last_activation_epoch_slot = simplex.Slot(activation_slot + simplex.SLOTS_PER_EPOCH - 1)
    simplex.process_slots(post, last_activation_epoch_slot)

    activation_root = simplex.get_block_root_at_slot(post, activation_slot)
    receipt = _build_signed_available_attestation(
        simplex,
        post,
        activation_slot,
        range(simplex.AVAILABLE_COMMITTEE_SIZE),
        activation_root,
    )
    simplex.process_available_attestation(post, receipt)

    for payment_slot in payment_slots:
        payment = post.builder_pending_payments[payment_slot % simplex.SLOTS_PER_EPOCH]
        assert sum(payment.available_participation) == simplex.AVAILABLE_COMMITTEE_SIZE


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_available_receipt_cannot_modify_rotated_postfork_half(spec, phases, state):
    simplex = phases[SIMPLEX]
    post, activation_slot, _ = _prepare_payment_bridge_state(spec, simplex, state)

    # Put a post-fork claim in the activation epoch's second half. At the next
    # boundary it becomes the first-half entry that an unguarded transition
    # receipt would corrupt.
    postfork_payment = simplex.BuilderPendingPayment(
        withdrawal=simplex.BuilderPendingWithdrawal(
            fee_recipient=simplex.ExecutionAddress(b"\x44" * 20),
            amount=simplex.Gwei(44),
            builder_index=simplex.BuilderIndex(0),
        )
    )
    post.builder_pending_payments[simplex.SLOTS_PER_EPOCH] = postfork_payment
    next_epoch_slot = simplex.compute_start_slot_at_epoch(simplex.Epoch(post.fork.epoch + 1))
    simplex.process_slots(post, next_epoch_slot)
    assert post.builder_pending_payments[0].withdrawal.amount == simplex.Gwei(44)
    assert not any(post.builder_pending_payments[0].available_participation)

    activation_root = simplex.get_block_root_at_slot(post, activation_slot)
    receipt = _build_signed_available_attestation(
        simplex,
        post,
        activation_slot,
        range(simplex.AVAILABLE_COMMITTEE_SIZE),
        activation_root,
    )
    simplex.process_available_attestation(post, receipt)

    rotated_payment = post.builder_pending_payments[0]
    assert rotated_payment.withdrawal.amount == simplex.Gwei(44)
    assert not any(rotated_payment.available_participation)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
def test_unweighted_latest_gloas_payment_settles_via_full_parent_after_upgrade(spec, phases, state):
    simplex = phases[SIMPLEX]
    fork_epoch = spec.Epoch(1)
    activation_slot = spec.compute_start_slot_at_epoch(fork_epoch)
    transition_to(spec, state, activation_slot)
    parent_slot = spec.Slot(activation_slot - 1)
    builder_index = spec.BuilderIndex(0)
    amount = spec.Gwei(2 * spec.MIN_ACTIVATION_BALANCE)
    withdrawal = spec.BuilderPendingWithdrawal(
        fee_recipient=spec.ExecutionAddress(b"\x42" * 20),
        amount=amount,
        builder_index=builder_index,
    )
    parent_bid = spec.ExecutionPayloadBid(
        slot=parent_slot,
        builder_index=builder_index,
        value=amount,
        fee_recipient=withdrawal.fee_recipient,
        block_hash=spec.Hash32(b"\x24" * 32),
    )
    state.latest_execution_payload_bid = parent_bid
    payment_index = parent_slot % spec.SLOTS_PER_EPOCH
    state.builder_pending_payments[payment_index] = spec.BuilderPendingPayment(
        weight=spec.Gwei(0),
        withdrawal=withdrawal,
    )

    post = simplex.upgrade_to_simplex(state)
    post_payment = post.builder_pending_payments[payment_index]
    assert post_payment.weight == 0
    assert post_payment.withdrawal.amount == amount
    assert not any(post_payment.available_participation)

    pre_withdrawal_count = len(post.builder_pending_withdrawals)
    simplex.apply_parent_execution_payload(post, simplex.ExecutionRequests())

    assert len(post.builder_pending_withdrawals) == pre_withdrawal_count + 1
    assert post.builder_pending_withdrawals[
        pre_withdrawal_count
    ] == simplex.BuilderPendingWithdrawal(
        fee_recipient=simplex.ExecutionAddress(withdrawal.fee_recipient),
        amount=simplex.Gwei(amount),
        builder_index=simplex.BuilderIndex(builder_index),
    )
    assert post.builder_pending_payments[payment_index] == simplex.BuilderPendingPayment()

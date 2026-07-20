from eth_consensus_specs.test.context import (
    spec_state_test,
    with_config_overrides,
    with_simplex_and_later,
)


def _set_post_startup_slot(spec, state):
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))


def _active_indices(spec, state):
    return list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))


def _configure_exact_three_way_stake(spec, state):
    """Give the active set an exact integral two-thirds stake boundary."""
    active_indices = _active_indices(spec, state)
    remainder = len(active_indices) % spec.FINALITY_QUORUM_DENOMINATOR
    for index in active_indices:
        state.validators[index].effective_balance = spec.EFFECTIVE_BALANCE_INCREMENT
    for index in active_indices[:remainder]:
        state.validators[index].effective_balance = spec.Gwei(0)

    weighted_indices = active_indices[remainder:]
    assert len(weighted_indices) >= spec.FINALITY_QUORUM_DENOMINATOR
    assert len(weighted_indices) % spec.FINALITY_QUORUM_DENOMINATOR == 0
    quorum_count = (
        len(weighted_indices) * spec.FINALITY_QUORUM_NUMERATOR // spec.FINALITY_QUORUM_DENOMINATOR
    )
    quorum_indices = weighted_indices[:quorum_count]

    total_balance = spec.get_total_active_balance(state)
    quorum_balance = sum(state.validators[index].effective_balance for index in quorum_indices)
    assert quorum_balance * spec.FINALITY_QUORUM_DENOMINATOR == (
        total_balance * spec.FINALITY_QUORUM_NUMERATOR
    )
    assert (
        quorum_balance - spec.EFFECTIVE_BALANCE_INCREMENT
    ) * spec.FINALITY_QUORUM_DENOMINATOR < (total_balance * spec.FINALITY_QUORUM_NUMERATOR)
    return quorum_indices


def _configure_guard_matrix(spec, state):
    """Configure four validators to accrue exactly 0, 1, 2, and 3 units."""
    _set_post_startup_slot(spec, state)
    state.current_height = spec.Height(1)
    state.finalized_height = spec.Height(0)
    state.finalized_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x11" * 32),
    )
    state.justified_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x22" * 32),
    )
    best_target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x33" * 32),
    )
    state.current_height_target = best_target
    indices = _active_indices(spec, state)[:4]
    assert len(indices) == 4

    # Validator 0 satisfies every per-validator guard, validator 1 misses
    # finality, validator 2 supplies only a timeout marker, and validator 3
    # supplies no height/finality evidence at all.
    state.timeouts[indices[0]] = True
    state.timeouts[indices[1]] = True
    state.timeouts[indices[2]] = True
    state.target_participation[indices[0]] = True
    state.target_participation[indices[1]] = True
    state.finality_participation[indices[0]] = True

    new_justification = spec.compute_justified_checkpoint(state) != spec.Checkpoint()
    new_height_advance = new_justification or spec.has_timeout_quorum(state)
    new_finalization = spec.has_new_finalization(state)
    assert not new_justification
    assert not new_height_advance
    assert not new_finalization
    return (
        indices,
        new_height_advance,
        new_justification,
        new_finalization,
    )


def _compute_units(spec, state, index, signals):
    new_height_advance, new_justification, new_finalization = signals
    return spec.compute_leak_penalty_units(
        state,
        index,
        new_height_advance,
        new_justification,
        new_finalization,
    )


@with_simplex_and_later
@spec_state_test
def test_weighted_quorum_predicates_accept_exact_two_thirds_only(spec, state):
    _set_post_startup_slot(spec, state)
    state.current_height = spec.Height(1)
    state.finalized_height = spec.Height(0)
    quorum_indices = _configure_exact_three_way_stake(spec, state)
    last_quorum_index = quorum_indices[-1]
    target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\xaa" * 32),
    )
    state.current_height_target = target

    for index in quorum_indices:
        state.target_participation[index] = True
    assert spec.compute_justified_checkpoint(state) == target
    state.target_participation[last_quorum_index] = False
    assert spec.compute_justified_checkpoint(state) == spec.Checkpoint()

    for index in quorum_indices:
        state.timeouts[index] = True
    assert spec.has_timeout_quorum(state)
    state.timeouts[last_quorum_index] = False
    assert not spec.has_timeout_quorum(state)

    state.finalized_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\xbb" * 32),
    )
    state.justified_checkpoint = target
    for index in quorum_indices:
        state.finality_participation[index] = True
    assert spec.has_new_finalization(state)
    state.finality_participation[last_quorum_index] = False
    assert not spec.has_new_finalization(state)


@with_simplex_and_later
@spec_state_test
def test_leak_penalty_units_cover_all_guards_and_target_participation(spec, state):
    (
        indices,
        new_height_advance,
        new_justification,
        new_finalization,
    ) = _configure_guard_matrix(spec, state)
    signals = (
        new_height_advance,
        new_justification,
        new_finalization,
    )

    assert [_compute_units(spec, state, index, signals) for index in indices] == [0, 1, 2, 3]

    # A slashed validator is charged every layer even when its own markers
    # would otherwise exempt it.
    slashed_index = _active_indices(spec, state)[4]
    state.timeouts[slashed_index] = True
    state.target_participation[slashed_index] = True
    state.finality_participation[slashed_index] = True
    state.validators[slashed_index].slashed = True
    assert _compute_units(spec, state, slashed_index, signals) == 3


@with_simplex_and_later
@spec_state_test
def test_nonjustifiable_height_suppresses_target_layer(spec, state):
    _set_post_startup_slot(spec, state)
    state.current_height = spec.Height(spec.K_NONJUSTIFIABLE)
    state.finalized_height = spec.Height(0)
    state.current_height_nonjustifiable = True
    state.finalized_checkpoint = state.justified_checkpoint
    assert spec.is_nonjustifiable_height(state.current_height, state.finalized_height)

    active_indices = _active_indices(spec, state)
    forbidden_target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\xcc" * 32),
    )
    state.current_height_target = forbidden_target
    for index in active_indices:
        state.timeouts[index] = True
        state.target_participation[index] = True

    # Even a full target tally cannot justify at this timeout-only height.
    # A timeout-only voter is therefore not charged the target layer.
    exempt_index = active_indices[0]
    state.target_participation[exempt_index] = False
    assert spec.compute_justified_checkpoint(state) == spec.Checkpoint()
    assert spec.has_timeout_quorum(state)
    signals = (True, False, False)
    assert _compute_units(spec, state, exempt_index, signals) == 0

    # The maximum slashed charge has two applicable layers here because the
    # target layer is disabled protocol-wide.
    state.validators[exempt_index].slashed = True
    assert _compute_units(spec, state, exempt_index, signals) == 2


@with_simplex_and_later
@spec_state_test
def test_inactivity_scores_and_monetary_deltas_scale_with_penalty_units(spec, state):
    (
        indices,
        new_height_advance,
        new_justification,
        new_finalization,
    ) = _configure_guard_matrix(spec, state)
    signals = (
        new_height_advance,
        new_justification,
        new_finalization,
    )
    expected_units = [0, 1, 2, 3]

    # First verify the score update in an actual leak, where recovery does not
    # obscure the per-guard increments.
    score_state = state.copy()
    score_state.slot = spec.compute_start_slot_at_epoch(
        spec.Epoch(spec.GENESIS_EPOCH + spec.MIN_EPOCHS_TO_INACTIVITY_PENALTY + 3)
    )
    score_state.finalized_checkpoint = spec.Checkpoint()
    assert spec.is_in_inactivity_leak(score_state)
    spec.process_inactivity_updates(score_state)
    for index, units in zip(indices, expected_units, strict=True):
        assert score_state.inactivity_scores[index] == spec.config.INACTIVITY_SCORE_BIAS * units

    # Then use a nonzero, exactly representable score to verify monetary
    # penalties and the balance mutations, including the multiplicity factor.
    penalty_denominator = (
        spec.config.INACTIVITY_SCORE_BIAS * spec.INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
    )
    score = penalty_denominator // 1024
    for index in indices:
        state.inactivity_scores[index] = score

    rewards, penalties = spec.get_inactivity_penalty_deltas(state)
    balances_before = state.balances.copy()
    for index, units in zip(indices, expected_units, strict=True):
        base_penalty = state.validators[index].effective_balance * score // penalty_denominator
        assert rewards[index] == 0
        assert penalties[index] == base_penalty * units

    spec.process_inactivity_penalties(state)
    for index, units in zip(indices, expected_units, strict=True):
        base_penalty = state.validators[index].effective_balance * score // penalty_denominator
        assert state.balances[index] == balances_before[index] - base_penalty * units
        assert _compute_units(spec, state, index, signals) == units


def _find_non_final_round_end(spec, epoch):
    epoch_start = spec.compute_start_slot_at_epoch(epoch)
    epoch_end = spec.compute_start_slot_at_epoch(spec.Epoch(epoch + 1))
    for raw_slot in range(int(epoch_start), int(epoch_end - 1)):
        slot = spec.Slot(raw_slot)
        if spec.compute_round_at_slot(spec.Slot(slot + 1)) > spec.compute_round_at_slot(slot):
            return slot
    raise AssertionError("Simplex test preset must contain multiple rounds per epoch")


@with_simplex_and_later
@spec_state_test
@with_config_overrides(
    {
        "ROUND_SCHEDULE": (
            {
                "SLOT": 0,
                "SLOTS_PER_ROUND": 4,
                "START_ROUND": 0,
            },
        ),
    }
)
def test_round_accounting_uses_duty_round_eligibility(spec, state):
    epoch = spec.Epoch(3)
    epoch_start = spec.compute_start_slot_at_epoch(epoch)
    # At the end of the epoch's second round, the previous-round array holds
    # duties from the first round of this same epoch.
    state.slot = spec.Slot(spec.compute_start_slot_at_epoch(spec.Epoch(epoch + 1)) - 1)
    settlement_round = spec.get_previous_round(state)
    assert spec.compute_epoch_at_round(settlement_round) == epoch

    entrant = spec.ValidatorIndex(0)
    leaver = spec.ValidatorIndex(1)
    stable = spec.ValidatorIndex(2)
    state.validators[entrant].activation_epoch = epoch
    state.validators[leaver].exit_epoch = epoch

    state.previous_round_participation = [
        spec.ParticipationFlags(0) for _ in range(len(state.validators))
    ]
    state.previous_round_participation[entrant] = spec.add_flag(
        state.previous_round_participation[entrant], spec.TIMELY_TARGET_FLAG_INDEX
    )
    state.finalized_checkpoint = spec.Checkpoint(
        slot=epoch_start,
        root=spec.Root(b"\x91" * 32),
    )

    rewards, penalties = spec.get_flag_index_deltas(state, spec.TIMELY_TARGET_FLAG_INDEX)
    assert rewards[entrant] > 0
    assert penalties[entrant] == 0
    assert rewards[leaver] == 0
    assert penalties[leaver] == 0
    assert penalties[stable] > 0


@with_simplex_and_later
@spec_state_test
@with_config_overrides(
    {
        "ROUND_SCHEDULE": (
            {
                "SLOT": 0,
                "SLOTS_PER_ROUND": 4,
                "START_ROUND": 0,
            },
        ),
    }
)
def test_round_accounting_uses_duty_epoch_reward_denominator(spec, state):
    duty_epoch = spec.Epoch(3)
    including_epoch = spec.Epoch(duty_epoch + 1)
    # The first round of the new epoch settles the last round of ``duty_epoch``.
    state.slot = spec.Slot(spec.compute_start_slot_at_epoch(including_epoch) + 3)
    settlement_round = spec.get_previous_round(state)
    assert spec.compute_epoch_at_round(settlement_round) == duty_epoch

    participant = spec.ValidatorIndex(0)
    entrant = spec.ValidatorIndex(1)
    leaver = spec.ValidatorIndex(2)
    state.validators[entrant].activation_epoch = including_epoch
    state.validators[entrant].effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    state.validators[leaver].exit_epoch = including_epoch
    state.validators[leaver].effective_balance = spec.EFFECTIVE_BALANCE_INCREMENT

    state.previous_round_participation = [
        spec.ParticipationFlags(0) for _ in range(len(state.validators))
    ]
    state.previous_round_participation[participant] = spec.add_flag(
        state.previous_round_participation[participant], spec.TIMELY_TARGET_FLAG_INDEX
    )
    state.finalized_checkpoint = spec.Checkpoint(
        slot=spec.compute_start_slot_at_epoch(duty_epoch),
        root=spec.Root(b"\x92" * 32),
    )

    rewards, _ = spec.get_flag_index_deltas(state, spec.TIMELY_TARGET_FLAG_INDEX)
    active_balance = spec.get_total_balance(
        state,
        set(spec.get_active_validator_indices(state, duty_epoch)),
    )
    participating_balance = spec.get_total_balance(state, {participant})
    base_reward_per_increment = (
        spec.EFFECTIVE_BALANCE_INCREMENT
        * spec.BASE_REWARD_FACTOR
        // spec.integer_squareroot(active_balance)
    )
    expected_base_reward = (
        state.validators[participant].effective_balance
        // spec.EFFECTIVE_BALANCE_INCREMENT
        * base_reward_per_increment
    )
    assert spec.get_base_reward_at_epoch(state, participant, duty_epoch) == expected_base_reward
    expected_reward = (
        expected_base_reward
        * spec.TIMELY_TARGET_WEIGHT
        * (participating_balance // spec.EFFECTIVE_BALANCE_INCREMENT)
        // (
            (active_balance // spec.EFFECTIVE_BALANCE_INCREMENT)
            * spec.WEIGHT_DENOMINATOR
            * spec.get_rounds_per_epoch_at_slot(spec.compute_start_slot_at_round(settlement_round))
        )
    )
    assert rewards[participant] == expected_reward


@with_simplex_and_later
@spec_state_test
@with_config_overrides(
    {
        "ROUND_SCHEDULE": (
            {
                "SLOT": 0,
                "SLOTS_PER_ROUND": 4,
                "START_ROUND": 0,
            },
        ),
    }
)
def test_height_outcome_cadence_backs_off_from_round_to_epoch_during_leak(spec, state):
    epoch = spec.Epoch(spec.GENESIS_EPOCH + spec.MIN_EPOCHS_TO_INACTIVITY_PENALTY + 3)
    epoch_start = spec.compute_start_slot_at_epoch(epoch)
    epoch_end = spec.compute_start_slot_at_epoch(spec.Epoch(epoch + 1))
    non_final_round_end = _find_non_final_round_end(spec, epoch)

    normal_state = state.copy()
    normal_state.slot = non_final_round_end
    normal_state.finalized_checkpoint = spec.Checkpoint(
        slot=epoch_start,
        root=spec.Root(b"\xdd" * 32),
    )
    normal_state.pending_height_outcomes = 0
    assert not spec.is_in_inactivity_leak(normal_state)
    spec.process_round(normal_state)
    assert normal_state.pending_height_outcomes == 1

    mid_leak_state = state.copy()
    mid_leak_state.slot = non_final_round_end
    mid_leak_state.finalized_checkpoint = spec.Checkpoint()
    mid_leak_state.pending_height_outcomes = 0
    assert spec.is_in_inactivity_leak(mid_leak_state)
    spec.process_round(mid_leak_state)
    assert mid_leak_state.pending_height_outcomes == 0

    epoch_end_leak_state = state.copy()
    epoch_end_leak_state.slot = spec.Slot(epoch_end - 1)
    epoch_end_leak_state.finalized_checkpoint = spec.Checkpoint()
    epoch_end_leak_state.pending_height_outcomes = 0
    assert spec.is_in_inactivity_leak(epoch_end_leak_state)
    spec.process_round(epoch_end_leak_state)
    assert epoch_end_leak_state.pending_height_outcomes == 1

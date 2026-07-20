from eth_consensus_specs.test.context import (
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    sign_attestation,
    sign_indexed_attestation,
)
from eth_consensus_specs.test.helpers.state import transition_to_slot_via_block


def _set_post_startup_slot(spec, state):
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))


def _get_active_indices(spec, state):
    return spec.get_active_validator_indices(state, spec.get_current_epoch(state))


@with_simplex_and_later
@spec_state_test
def test_active_slashed_weight_remains_in_all_quorum_denominators(spec, state):
    _set_post_startup_slot(spec, state)
    active_indices = _get_active_indices(spec, state)
    total_active_balance = spec.get_total_active_balance(state)

    supporters = []
    support = spec.Gwei(0)
    for index in active_indices:
        supporters.append(index)
        support += state.validators[index].effective_balance
        if (
            support * spec.FINALITY_QUORUM_DENOMINATOR
            >= total_active_balance * spec.FINALITY_QUORUM_NUMERATOR
        ):
            break

    pivotal = supporters[-1]
    pivotal_balance = state.validators[pivotal].effective_balance
    support_without_pivotal = support - pivotal_balance
    assert (
        support_without_pivotal * spec.FINALITY_QUORUM_DENOMINATOR
        < total_active_balance * spec.FINALITY_QUORUM_NUMERATOR
    )
    # The fixture's equal active balances make this validator exactly
    # denominator-pivotal: removing its weight from both sides would restore a
    # quorum, while retaining it only in the denominator does not.
    assert (
        support_without_pivotal * spec.FINALITY_QUORUM_DENOMINATOR
        >= (total_active_balance - pivotal_balance) * spec.FINALITY_QUORUM_NUMERATOR
    )

    target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x10" * 32),
    )
    state.current_height_target = target
    for index in supporters:
        state.target_participation[index] = True
        state.timeouts[index] = True
        state.finality_participation[index] = True

    assert spec.compute_justified_checkpoint(state) == target
    assert spec.has_timeout_quorum(state)
    state.justified_checkpoint = target
    assert spec.has_new_finalization(state)

    state.validators[pivotal].slashed = True
    assert spec.get_total_active_balance(state) == total_active_balance
    assert spec.compute_justified_checkpoint(state) == spec.Checkpoint()
    assert not spec.has_timeout_quorum(state)
    assert not spec.has_new_finalization(state)


@with_simplex_and_later
@spec_state_test
def test_target_quorum_advances_height_and_updates_justification(spec, state):
    _set_post_startup_slot(spec, state)
    active_indices = _get_active_indices(spec, state)
    target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x11" * 32),
    )
    previous_height = state.current_height
    previous_finalized_checkpoint = state.finalized_checkpoint
    previous_finalized_height = state.finalized_height
    state.current_height_target = target

    # Model already-validated R1 votes. An R1 vote records both target
    # participation and timeout participation at the current height.
    for index in active_indices:
        state.target_participation[index] = True
        state.timeouts[index] = True

    assert spec.compute_justified_checkpoint(state) == target
    spec.process_justification_and_finalization(state)

    assert state.current_height == previous_height + 1
    assert state.current_height_start_slot == state.slot
    assert state.justified_checkpoint == target
    assert state.justified_height == previous_height
    assert state.finalized_checkpoint == previous_finalized_checkpoint
    assert state.finalized_height == previous_finalized_height
    assert state.current_height_target == spec.Checkpoint()
    assert not any(state.target_participation)
    assert not any(state.timeouts)


@with_simplex_and_later
@spec_state_test
def test_current_height_target_is_first_block_and_never_replaces(spec, state):
    _set_post_startup_slot(spec, state)
    first_block_slot = spec.Slot(state.slot + 1)
    spec.advance_height(state, height_start_slot=first_block_slot)
    assert state.current_height_target == spec.Checkpoint()

    # Empty-slot entry into the height cannot reuse the preceding block.
    spec.process_slot(state)
    assert state.latest_block_header.slot < first_block_slot
    assert state.current_height_target == spec.Checkpoint()

    # Model the first block's post-state. Its header state root is still empty,
    # so the target remains empty until the next process_slot action fills it.
    state.slot = first_block_slot
    state.latest_block_header = spec.BeaconBlockHeader(
        slot=first_block_slot,
        proposer_index=spec.ValidatorIndex(0),
        parent_root=spec.Root(b"\x81" * 32),
        state_root=spec.Root(),
        body_root=spec.Root(b"\x82" * 32),
    )
    expected_header = state.latest_block_header.copy()
    expected_header.state_root = spec.hash_tree_root(state)
    expected_target = spec.Checkpoint(
        slot=first_block_slot,
        root=spec.hash_tree_root(expected_header),
    )
    assert state.current_height_target == spec.Checkpoint()

    spec.process_slot(state)
    assert state.current_height_target == expected_target

    # Neither a later empty slot nor a later block can replace the target.
    state.slot = spec.Slot(first_block_slot + 1)
    spec.process_slot(state)
    assert state.current_height_target == expected_target

    state.slot = spec.Slot(first_block_slot + 2)
    state.latest_block_header = spec.BeaconBlockHeader(
        slot=state.slot,
        proposer_index=spec.ValidatorIndex(1),
        parent_root=expected_target.root,
        state_root=spec.Root(),
        body_root=spec.Root(b"\x83" * 32),
    )
    spec.process_slot(state)
    assert state.current_height_target == expected_target


@with_simplex_and_later
@spec_state_test
def test_process_target_attestation_records_height_progress_and_reward_flag(spec, state):
    target_slot = spec.Slot(
        spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2)) + 1
    )
    transition_to_slot_via_block(spec, state, target_slot)
    spec.process_slots(state, spec.Slot(target_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY))
    target_root = spec.get_block_root_at_slot(state, target_slot)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=target_slot,
        beacon_block_root=target_root,
    )
    assert state.current_height_target != spec.Checkpoint()
    attestation.data.target = state.current_height_target
    attestation.data.height = state.current_height
    sign_attestation(spec, state, attestation)
    attesting_indices = spec.get_attesting_indices(state, attestation)
    proposer = spec.get_beacon_proposer_index(state)
    proposer_balance = state.balances[proposer]

    spec.process_attestation(state, attestation)

    for index in attesting_indices:
        assert state.target_participation[index]
        assert state.timeouts[index]
        assert spec.has_flag(
            state.current_round_participation[index],
            spec.TIMELY_TARGET_FLAG_INDEX,
        )
        assert not spec.has_flag(
            state.current_round_participation[index],
            spec.TIMELY_FINALITY_TARGET_FLAG_INDEX,
        )
    assert state.balances[proposer] > proposer_balance


@with_simplex_and_later
@spec_state_test
def test_finality_piggyback_quorum_updates_finality_without_advancing_height(spec, state):
    _set_post_startup_slot(spec, state)
    state.current_height = spec.Height(3)
    state.justified_height = spec.Height(2)
    state.finalized_height = spec.Height(1)
    state.justified_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x22" * 32),
    )
    state.finalized_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x33" * 32),
    )
    piggyback = spec.AttestationData(
        slot=state.slot,
        beacon_block_root=spec.Root(),
        target=spec.Checkpoint(),
        height=spec.Height(0),
        finality_target=state.justified_checkpoint,
        finality_height=state.justified_height,
    )
    previous_height = state.current_height
    previous_justified_checkpoint = state.justified_checkpoint
    previous_justified_height = state.justified_height

    for index in _get_active_indices(spec, state):
        spec.update_finality_participation(state, index, piggyback)

    assert spec.has_new_finalization(state)
    spec.process_justification_and_finalization(state)

    assert state.finalized_checkpoint == previous_justified_checkpoint
    assert state.finalized_height == previous_justified_height
    assert state.justified_checkpoint == previous_justified_checkpoint
    assert state.justified_height == previous_justified_height
    assert state.current_height == previous_height


@with_simplex_and_later
@spec_state_test
def test_finality_progress_is_keyed_by_height_and_checkpoint(spec, state):
    _set_post_startup_slot(spec, state)
    shared_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x23" * 32),
    )
    state.current_height = spec.Height(2)
    state.justified_height = spec.Height(1)
    state.finalized_height = spec.Height(0)
    state.justified_checkpoint = shared_checkpoint
    state.finalized_checkpoint = shared_checkpoint
    state.finality_participation = spec.Bitlist[spec.VALIDATOR_REGISTRY_LIMIT](
        [False] * len(state.validators)
    )

    # A bootstrap justification may reuse the sentinel block root while
    # advancing its height. The distinct checkpoint pair is still pending and
    # therefore activates both finality participation and the leak's finality
    # layer.
    first_active = _get_active_indices(spec, state)[0]
    assert (
        spec.compute_leak_penalty_units(
            state,
            first_active,
            new_height_advance=True,
            new_justification=True,
            new_finalization=False,
        )
        == 1
    )
    piggyback = spec.AttestationData(
        slot=state.slot,
        beacon_block_root=spec.Root(),
        target=spec.Checkpoint(),
        height=spec.Height(0),
        finality_target=shared_checkpoint,
        finality_height=state.justified_height,
    )
    for index in _get_active_indices(spec, state):
        spec.update_finality_participation(state, index, piggyback)

    assert spec.has_new_finalization(state)
    spec.process_justification_and_finalization(state)

    assert state.finalized_checkpoint == shared_checkpoint
    assert state.finalized_height == spec.Height(1)
    assert state.current_height == spec.Height(2)


@with_simplex_and_later
@spec_state_test
def test_stale_justification_can_be_skipped_for_fresh_honest_finalization(spec, state):
    _set_post_startup_slot(spec, state)
    active_indices = _get_active_indices(spec, state)
    total_active_balance = spec.get_total_active_balance(state)

    # Model Byzantine weight B just below one third. Split the remaining honest
    # validators into A and X so A + B can reveal an old target quorum while A
    # alone cannot finalize it. Because B is below one third, A + X (all honest
    # validators) can form the later fresh quorum without B.
    byzantine = []
    byzantine_balance = spec.Gwei(0)
    remaining = list(active_indices)
    while remaining:
        index = remaining[0]
        balance = state.validators[index].effective_balance
        if (byzantine_balance + balance) * spec.FINALITY_QUORUM_DENOMINATOR >= total_active_balance:
            break
        byzantine.append(remaining.pop(0))
        byzantine_balance += balance

    stale_honest = []
    stale_honest_balance = spec.Gwei(0)
    while remaining and (
        (stale_honest_balance + byzantine_balance) * spec.FINALITY_QUORUM_DENOMINATOR
        < total_active_balance * spec.FINALITY_QUORUM_NUMERATOR
    ):
        index = remaining.pop(0)
        stale_honest.append(index)
        stale_honest_balance += state.validators[index].effective_balance
    fresh_honest = remaining
    fresh_honest_balance = spec.Gwei(
        sum(state.validators[index].effective_balance for index in fresh_honest)
    )

    assert byzantine_balance * spec.FINALITY_QUORUM_DENOMINATOR < total_active_balance
    assert (
        stale_honest_balance * spec.FINALITY_QUORUM_DENOMINATOR
        < total_active_balance * spec.FINALITY_QUORUM_NUMERATOR
    )
    assert (
        stale_honest_balance + byzantine_balance
    ) * spec.FINALITY_QUORUM_DENOMINATOR >= total_active_balance * spec.FINALITY_QUORUM_NUMERATOR
    assert (
        stale_honest_balance + fresh_honest_balance
    ) * spec.FINALITY_QUORUM_DENOMINATOR >= total_active_balance * spec.FINALITY_QUORUM_NUMERATOR

    state.current_height = spec.Height(1)
    state.current_height_nonjustifiable = False
    state.current_height_start_slot = state.slot
    state.justified_checkpoint = state.finalized_checkpoint
    state.justified_height = state.finalized_height
    state.current_height_target = spec.Checkpoint()
    state.target_participation = spec.Bitlist[spec.VALIDATOR_REGISTRY_LIMIT](
        [False] * len(state.validators)
    )
    state.timeouts = spec.Bitlist[spec.VALIDATOR_REGISTRY_LIMIT]([False] * len(state.validators))
    state.finality_participation = spec.Bitlist[spec.VALIDATOR_REGISTRY_LIMIT](
        [False] * len(state.validators)
    )

    stale_target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x44" * 32),
    )
    state.current_height_target = stale_target
    stale_supporters = stale_honest + byzantine
    for index in stale_supporters:
        state.target_participation[index] = True
        state.timeouts[index] = True

    stale_height = state.current_height
    spec.process_justification_and_finalization(state)
    assert state.justified_checkpoint == stale_target
    assert state.justified_height == stale_height

    # Byzantine validators withhold the matching finality piggyback. The honest
    # subset that voted for the stale target is insufficient, so this particular
    # justification need not finalize.
    for index in stale_honest:
        state.finality_participation[index] = True
    spec.process_justification_and_finalization(state)
    assert state.finalized_checkpoint != stale_target

    fresh_target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x55" * 32),
    )
    state.current_height_target = fresh_target
    fresh_supporters = stale_honest + fresh_honest
    fresh_height = state.current_height
    for index in fresh_supporters:
        state.target_participation[index] = True
        state.timeouts[index] = True

    spec.process_justification_and_finalization(state)
    assert state.justified_checkpoint == fresh_target
    assert state.justified_height == fresh_height
    assert state.finalized_checkpoint != stale_target

    # Once all honest validators choose the same fresh round target, their
    # honest-only quorum both justifies it and supplies its later finality vote.
    for index in fresh_supporters:
        state.finality_participation[index] = True
    spec.process_justification_and_finalization(state)
    assert state.finalized_checkpoint == fresh_target
    assert state.finalized_height == fresh_height


@with_simplex_and_later
@spec_state_test
def test_round_double_vote_evidence_rejects_pre_simplex_messages(spec, state):
    offender = spec.ValidatorIndex(0)
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    evidence_slot = spec.Slot(fork_slot - 1)
    state.slot = fork_slot

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

    expect_assertion_error(lambda: spec.process_round_double_vote_evidence(state, evidence))
    assert not state.round_double_vote_penalized[offender]


@with_simplex_and_later
@spec_state_test
def test_round_double_vote_evidence_penalizes_once_and_rejects_replay(spec, state):
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

    # Make the signed evidence eligible for on-chain inclusion.
    state.slot = spec.Slot(evidence_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY)
    proposer = spec.get_beacon_proposer_index(state)
    penalty = spec.get_base_reward(state, offender)
    proposer_reward = spec.Gwei(penalty // spec.PROPOSER_REWARD_QUOTIENT)
    offender_balance = state.balances[offender]
    proposer_balance = state.balances[proposer]

    spec.process_round_double_vote_evidence(state, evidence)

    assert state.round_double_vote_penalized[offender]
    assert state.validators[offender].exit_epoch != spec.FAR_FUTURE_EPOCH
    assert not state.validators[offender].slashed
    if proposer == offender:
        assert state.balances[offender] == offender_balance - penalty + proposer_reward
    else:
        assert state.balances[offender] == offender_balance - penalty
        assert state.balances[proposer] == proposer_balance + proposer_reward

    balances_after_penalty = state.balances.copy()
    exit_epoch_after_penalty = state.validators[offender].exit_epoch
    expect_assertion_error(lambda: spec.process_round_double_vote_evidence(state, evidence))
    assert state.balances == balances_after_penalty
    assert state.validators[offender].exit_epoch == exit_epoch_after_penalty

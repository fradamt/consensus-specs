from eth_consensus_specs.test.context import (
    always_bls,
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation,
    sign_attestation,
)
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _add_child(spec, store, state, parent_root, slot, marker, parent_full=True):
    block = spec.BeaconBlock(
        slot=slot,
        parent_root=parent_root,
        body=spec.BeaconBlockBody(graffiti=bytes([marker]) * 32),
    )
    if not parent_full:
        parent_block_hash = store.blocks[
            parent_root
        ].body.signed_execution_payload_bid.message.block_hash
        mismatching_hash = spec.Hash32(bytes([marker]) * 32)
        assert mismatching_hash != parent_block_hash
        block.body.signed_execution_payload_bid.message.parent_block_hash = mismatching_hash
    root = block.hash_tree_root()
    store.blocks[root] = block
    # These tests exercise fork-choice bookkeeping only. Sharing an immutable
    # state view keeps the deep-tree regression small enough to be useful.
    store.block_states[root] = state
    return root


def _set_store_slot(spec, store, slot):
    store.time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)


def _build_signed_available_attestation(spec, state, slot, position, root):
    data = spec.AvailableAttestationData(
        slot=slot,
        beacon_block_root=root,
        payload_present=False,
    )
    attestation = spec.AvailableAttestation(data=data)
    attestation.aggregation_bits[position] = True

    validator_index = spec.get_available_committee(state, slot)[position]
    domain = spec.get_domain(
        state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Sign(privkeys[validator_index], signing_root)
    return attestation, validator_index


@with_simplex_and_later
@spec_state_test
def test_forkchoice_store_preserves_onchain_round_equivocator_exclusion(spec, state):
    penalized_index = spec.ValidatorIndex(0)
    state.round_double_vote_penalized[penalized_index] = True

    store = get_genesis_forkchoice_store(spec, state)

    assert store.equivocating_indices == {penalized_index}


@with_simplex_and_later
@spec_state_test
def test_forkchoice_finality_ingress_activation_boundary(spec, state):
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    _set_store_slot(spec, store, fork_slot)

    pre_fork_attestation = spec.Attestation(
        data=spec.AttestationData(
            slot=spec.Slot(fork_slot - 1),
            beacon_block_root=anchor_root,
            target=spec.Checkpoint(),
            height=spec.Height(0),
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        )
    )
    expect_assertion_error(lambda: spec.validate_on_attestation(store, pre_fork_attestation))

    fork_attestation = pre_fork_attestation.copy()
    fork_attestation.data.slot = fork_slot
    spec.validate_on_attestation(store, fork_attestation)


@with_simplex_and_later
@spec_state_test
def test_filter_block_tree_beyond_python_recursion_limit(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root

    parent_root = anchor_root
    depth = 1_200
    for slot in range(1, depth + 1):
        parent_root = _add_child(
            spec,
            store,
            state,
            parent_root,
            spec.Slot(slot),
            slot % 256,
        )

    filtered = {}
    assert spec.filter_block_tree(store, anchor_root, filtered)
    assert len(filtered) == depth + 1
    assert parent_root in filtered


@with_simplex_and_later
@spec_state_test
def test_on_block_ancestry_beyond_python_recursion_limit(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root

    parent_root = anchor_root
    depth = 1_200
    for slot in range(1, depth + 1):
        parent_root = _add_child(
            spec,
            store,
            state,
            parent_root,
            spec.Slot(slot),
            slot % 256,
        )

    block = spec.BeaconBlock(
        slot=spec.Slot(depth + 1),
        parent_root=parent_root,
    )
    store.payloads[parent_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=parent_root)
    _set_store_slot(spec, store, block.slot)

    class ReachedStateTransition(Exception):
        pass

    def stop_after_ancestry(*_args, **_kwargs):
        raise ReachedStateTransition

    # Exercise the normal block-ingress ancestry gate without constructing an
    # otherwise unrelated 1,200-block sequence of valid state transitions.
    original_state_transition = spec.state_transition
    spec.state_transition = stop_after_ancestry
    try:
        try:
            spec.on_block(store, spec.SignedBeaconBlock(message=block))
        except ReachedStateTransition:
            pass
        else:
            raise AssertionError("on_block did not reach state transition")
    finally:
        spec.state_transition = original_state_transition


@with_simplex_and_later
@spec_state_test
def test_tick_pins_available_committee_with_long_stalled_justification(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    stalled_justified = store.justified_checkpoint

    target_epoch = spec.Epoch(spec.MIN_SEED_LOOKAHEAD + 3)
    target_slot = spec.compute_start_slot_at_epoch(target_epoch)
    target_time = spec.uint64(
        store.genesis_time + target_slot * spec.config.SLOT_DURATION_MS // 1000
    )

    # A stale justified state is outside its bounded committee lookahead here.
    # The slot-boundary cache must instead process the current walk-head state
    # forward and remain total.
    spec.on_tick(store, target_time)

    assert spec.get_current_slot(store) == target_slot
    assert store.justified_checkpoint == stalled_justified
    assert target_slot in store.available_committees
    assert len(store.available_committees[target_slot]) == spec.AVAILABLE_COMMITTEE_SIZE


@with_simplex_and_later
@spec_state_test
def test_live_confirmation_recovers_across_conflict_without_retracting_user_record(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    root_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB2)

    previous = (root_a, spec.Slot(1))
    store.latest_confirmed_head = previous

    # The externally exposed confirmation record is monotone and therefore
    # ignores a conflicting later candidate.
    assert spec.update_confirmed_head(store, previous, root_b) == previous

    # Healing gates consume the separate live result. With no grade-0 conflict
    # votes, it can switch to the converged branch immediately.
    store.live_confirmed_head = (root_b, spec.Slot(2))
    assert spec.get_safe_confirmed_head(store) == root_b
    assert store.latest_confirmed_head == previous


@with_simplex_and_later
@spec_state_test
def test_same_round_equivocation_compares_complete_attestation_data(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    validator_index = spec.ValidatorIndex(0)

    data_1 = spec.AttestationData(
        slot=spec.GENESIS_SLOT,
        beacon_block_root=anchor_root,
        target=spec.Checkpoint(),
        height=spec.Height(1),
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    data_2 = data_1.copy()
    # Keep the same slot and head. Looking only at latest-message fields would
    # miss this equivocation; the complete signed data differs.
    data_2.height = spec.Height(2)

    spec.update_latest_messages(
        store,
        [validator_index],
        spec.Attestation(data=data_1),
    )
    assert validator_index not in store.equivocating_indices

    spec.update_latest_messages(
        store,
        [validator_index],
        spec.Attestation(data=data_2),
    )

    round_ = spec.compute_round_at_slot(data_1.slot)
    assert validator_index in store.round_equivocating_indices[round_]
    assert validator_index in store.equivocating_indices
    assert not spec.has_unexpired_latest_message(store, validator_index)


@with_simplex_and_later
@spec_state_test
def test_same_slot_wire_attestations_apply_immediately(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    slot = spec.get_current_slot(store)

    empty_vote = get_valid_attestation(
        spec,
        state,
        slot=slot,
        beacon_block_root=anchor_root,
        signed=True,
    )
    justify_vote = empty_vote.copy()
    justify_vote.data.target = spec.Checkpoint(slot=slot, root=anchor_root)
    justify_vote.data.height = spec.Height(1)
    sign_attestation(spec, state, justify_vote)
    attesting_indices = sorted(spec.get_attesting_indices(state, empty_vote))

    spec.on_attestation(store, empty_vote)

    # A valid wire vote enters both live grade inputs in its own slot.
    round_ = spec.compute_round_at_slot(slot)
    for index in attesting_indices:
        assert store.latest_messages[index] == spec.LatestMessage(
            slot=slot,
            root=anchor_root,
        )
        assert store.round_attestations[round_][index] == empty_vote.data

    spec.on_attestation(store, justify_vote)

    # A conflicting complete data value is likewise observed immediately; no
    # slot-change queue mediates either latest-message update.
    for index in attesting_indices:
        assert index in store.round_equivocating_indices[round_]
        assert index in store.equivocating_indices
        assert not spec.has_unexpired_latest_message(store, index)


@with_simplex_and_later
@spec_state_test
def test_future_round_double_vote_evidence_rejected_before_state_derivation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    future_slot = spec.Slot(spec.get_current_slot(store) + spec.LATEST_MESSAGE_EXPIRY_SLOTS)
    data_1 = spec.AttestationData(
        slot=future_slot,
        beacon_block_root=anchor_root,
        target=spec.Checkpoint(),
        height=spec.Height(1),
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    data_2 = data_1.copy()
    data_2.height = spec.Height(2)
    evidence = spec.RoundDoubleVoteEvidence(
        attestation_1=spec.IndexedAttestation(attesting_indices=[0], data=data_1),
        attestation_2=spec.IndexedAttestation(attesting_indices=[0], data=data_2),
    )

    expect_assertion_error(lambda: spec.on_round_double_vote_evidence(store, evidence))
    assert store.checkpoint_states == {}
    assert store.round_equivocating_indices == {}


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_vote_at_two_delta_is_timely_and_strictly_late_vote_is_excluded(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    slot = spec.Slot(1)
    slot_time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)
    spec.on_tick(store, slot_time)

    # Deliver the slot block at exactly Delta, the validator action time.
    action_due_ms = spec.get_slot_component_duration_ms(spec.AVAILABLE_ATTESTATION_DUE_BPS)
    assert action_due_ms == spec.get_attestation_due_ms()
    assert action_due_ms * 4 == spec.config.SLOT_DURATION_MS
    # Block ingress itself has no phase-time gate. In the minimal preset Delta
    # is 1.5 seconds while the inherited Store clock is whole-second, so the
    # validator's millisecond publication schedule is asserted numerically
    # without manufacturing an unrepresentable fork-choice timestamp here.
    block_root = _add_child(spec, store, state, anchor_root, slot, 0xA1)

    committee = list(store.available_committees[slot])
    exact_position = 0
    exact_index = committee[exact_position]
    late_position = next(
        position for position, index in enumerate(committee) if index != exact_index
    )
    exact_vote, exact_index_from_vote = _build_signed_available_attestation(
        spec, state, slot, exact_position, block_root
    )
    late_vote, late_index = _build_signed_available_attestation(
        spec, state, slot, late_position, block_root
    )
    assert exact_index_from_vote == exact_index
    assert late_index != exact_index

    # A vote delivered at exactly 2*Delta is ingested before the equality-time
    # tick and therefore belongs to the frozen timely set.
    confirmation_due_ms = spec.get_available_confirmation_due_ms()
    assert spec.is_at_or_before_available_confirmation_deadline(store)
    assert not spec.is_at_or_after_available_confirmation_deadline(store)
    spec.on_available_attestation(store, exact_vote)
    spec.on_tick_per_high_resolution(store, confirmation_due_ms)

    assert store.frozen_available_votes[slot].votes == {exact_index: exact_vote.data}

    # A strictly later vote remains useful to live Goldfish before view freeze,
    # but cannot enter or mutate the already captured confirmation snapshot.
    assert not spec.is_at_or_before_available_confirmation_deadline(store)
    assert spec.is_at_or_after_available_confirmation_deadline(store)
    spec.on_available_attestation(store, late_vote)

    assert late_index in store.available_votes[slot]
    assert late_index not in store.available_timely_attesters[slot]
    assert store.frozen_available_votes[slot].votes == {exact_index: exact_vote.data}


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_freeze_ignores_post_deadline_equivocation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    slot = spec.Slot(1)
    slot_time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)
    spec.on_tick(store, slot_time)

    root_a = _add_child(spec, store, state, anchor_root, slot, 0xA1)
    root_b = _add_child(spec, store, state, anchor_root, slot, 0xB2)
    vote_a, validator_index = _build_signed_available_attestation(
        spec,
        state,
        slot,
        0,
        root_a,
    )
    vote_b, duplicate_index = _build_signed_available_attestation(
        spec,
        state,
        slot,
        0,
        root_b,
    )
    assert duplicate_index == validator_index

    spec.on_available_attestation(store, vote_a)
    assert validator_index in store.available_timely_attesters[slot]

    # Capture the exact deadline snapshot before the conflicting copy arrives.
    spec.on_tick_per_high_resolution(store, spec.get_available_confirmation_due_ms())
    assert not spec.is_at_or_before_available_confirmation_deadline(store)
    assert spec.is_at_or_after_available_confirmation_deadline(store)
    assert spec.is_before_view_freeze_deadline(store)
    spec.on_available_attestation(store, vote_b)

    assert validator_index in store.available_vote_equivocations[slot]
    assert validator_index not in store.available_timely_equivocations[slot]
    assert store.frozen_available_votes[slot].votes == {
        validator_index: vote_a.data,
    }

    next_slot = spec.Slot(slot + 1)
    next_slot_time = spec.uint64(
        store.genesis_time + next_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    spec.on_tick(store, next_slot_time)

    assert store.frozen_available_votes[slot].votes == {
        validator_index: vote_a.data,
    }


@with_simplex_and_later
@spec_state_test
def test_boundary_catchup_does_not_reconstruct_missed_confirmation_freeze(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    delayed_slot = spec.Slot(1)
    evaluation_slot = spec.Slot(2)
    catchup_slot = spec.Slot(4)
    delayed_root = _add_child(spec, store, state, anchor_root, delayed_slot, 0xD1)
    fast_root = _add_child(
        spec,
        store,
        state,
        delayed_root,
        evaluation_slot,
        0xF2,
        parent_full=False,
    )
    # Expose anchor FULL so ``delayed_root`` is reachable and both payload
    # decisions for ``delayed_root`` so the slot-2 vote can follow its explicit
    # EMPTY-parent child while the slot-1 vote stops at ``delayed_root``.
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    store.payloads[delayed_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=delayed_root)
    validator_index = spec.ValidatorIndex(0)

    # Slot 1 was already frozen. Its unanimous frozen electorate delays
    # confirmation of the slot-1 child until evaluation in slot 2.
    delayed_vote = spec.AvailableAttestationData(
        slot=delayed_slot,
        beacon_block_root=delayed_root,
    )
    store.frozen_available_votes[delayed_slot] = spec.FrozenAvailableVotes(
        committee=[validator_index] * spec.AVAILABLE_COMMITTEE_SIZE,
        votes={validator_index: delayed_vote},
    )

    # Enter slot 2 without ever running its exact 50% confirmation event.
    # Locally classified votes alone are not a frozen snapshot.
    _set_store_slot(spec, store, evaluation_slot)
    fast_vote = spec.AvailableAttestationData(
        slot=evaluation_slot,
        beacon_block_root=fast_root,
    )
    store.available_committees[evaluation_slot] = [validator_index] * spec.AVAILABLE_COMMITTEE_SIZE
    store.available_votes[evaluation_slot] = {validator_index: fast_vote}
    store.available_vote_equivocations[evaluation_slot] = set()
    store.available_timely_attesters[evaluation_slot] = {validator_index}
    store.available_timely_equivocations[evaluation_slot] = set()

    # Jump across two boundaries. Already frozen slot 1 can still be consumed,
    # but the boundary never reconstructs a slot-2 snapshot from the later
    # view, so the missed fast confirmation cannot appear.
    catchup_time = spec.uint64(
        store.genesis_time + catchup_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    spec.on_tick(store, catchup_time)

    assert evaluation_slot not in store.frozen_available_votes
    assert store.latest_confirmed_head == (delayed_root, evaluation_slot)
    assert store.fast_confirmed_head == (anchor_root, spec.GENESIS_SLOT)
    assert store.frozen_available_votes == {}


@with_simplex_and_later
@spec_state_test
def test_tick_prunes_slot_data_and_consumed_tsq_round_state(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    previous_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS + spec.SLOTS_PER_EPOCH)
    old_slot = spec.Slot(previous_slot - 1)
    next_slot = spec.Slot(previous_slot + 1)
    _set_store_slot(spec, store, previous_slot)

    store.payload_votes[old_slot] = {}
    store.payload_vote_equivocations[old_slot] = set()
    store.payload_votes[previous_slot] = {}
    store.payload_vote_equivocations[previous_slot] = set()

    anchor_root = store.finalized_checkpoint.root
    old_round = spec.compute_round_at_slot(spec.GENESIS_SLOT)
    current_round = spec.compute_round_at_slot(next_slot)
    store.round_equivocating_indices[old_round] = {spec.ValidatorIndex(0)}
    store.round_equivocating_indices[current_round] = {spec.ValidatorIndex(1)}
    store.round_attestations[old_round] = {}
    store.round_attestations[current_round] = {}

    expired_sync_round = spec.Round(current_round - 1)
    for round_ in (expired_sync_round, current_round):
        store.frozen_tsq_views[round_] = spec.FrozenTSQView(
            support_round=spec.Round(round_ - 1),
            attestations={},
            equivocating_indices=set(),
        )
        store.tsq_selections[round_] = spec.TSQSelection(
            support_round=spec.Round(round_ - 1),
            simplex_root=anchor_root,
            candidate_roots={anchor_root},
            weights={},
            total_active_balance=spec.Gwei(0),
            candidate_root=anchor_root,
        )
        store.round_proposals[round_] = {anchor_root}
        store.stable_root_decisions[round_] = True
        store.round_proposal_conflicts.add(round_)
    store.view_freeze_slots = {previous_slot, next_slot}

    # Keep this cleanup regression independent of committee derivation.
    store.available_committees[next_slot] = []
    next_slot_time = spec.uint64(
        store.genesis_time + next_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    spec.on_tick_per_slot(store, next_slot_time)

    assert set(store.payload_votes) == {previous_slot, next_slot}
    assert set(store.payload_vote_equivocations) == {previous_slot, next_slot}
    assert old_round not in store.round_equivocating_indices
    assert old_round not in store.round_attestations
    assert current_round in store.round_equivocating_indices
    assert current_round in store.round_attestations
    for round_mapping in (
        store.frozen_tsq_views,
        store.tsq_selections,
        store.round_proposals,
        store.stable_root_decisions,
    ):
        assert expired_sync_round not in round_mapping
        assert current_round in round_mapping
    assert store.round_proposal_conflicts == {current_round}
    assert store.view_freeze_slots == {next_slot}


@with_simplex_and_later
@spec_state_test
def test_available_vote_freeze_is_idempotent(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    slot = spec.get_current_slot(store)
    anchor_root = store.finalized_checkpoint.root
    vote_0 = spec.AvailableAttestationData(slot=slot, beacon_block_root=anchor_root)
    vote_1 = vote_0.copy()
    validator_0 = spec.ValidatorIndex(0)
    validator_1 = spec.ValidatorIndex(1)
    store.available_committees[slot] = [validator_0, validator_1]
    store.available_votes[slot] = {validator_0: vote_0}
    store.available_vote_equivocations[slot] = set()
    store.available_timely_attesters[slot] = {validator_0}
    store.available_timely_equivocations[slot] = set()

    spec.freeze_available_votes(store, slot)
    frozen = store.frozen_available_votes[slot]
    assert frozen.votes == {validator_0: vote_0}

    # Neither a late vote nor a late equivocation report can mutate the
    # time-shifted quorum captured at the deadline.
    store.available_votes[slot][validator_1] = vote_1
    store.available_timely_attesters[slot].add(validator_1)
    store.available_vote_equivocations[slot].add(validator_0)
    spec.freeze_available_votes(store, slot)
    assert store.frozen_available_votes[slot] == frozen


@with_simplex_and_later
@spec_state_test
def test_available_confirmation_seat_threshold_boundaries(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    child_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xC3)
    child = spec.ForkChoiceNode(root=child_root, payload_status=spec.PAYLOAD_STATUS_PENDING)
    validator_0 = spec.ValidatorIndex(0)
    validator_1 = spec.ValidatorIndex(1)

    current_slot = spec.Slot(2)
    previous_slot = spec.Slot(1)
    _set_store_slot(spec, store, current_slot)
    child_vote = spec.AvailableAttestationData(
        slot=previous_slot,
        beacon_block_root=child_root,
    )
    anchor_vote = spec.AvailableAttestationData(
        slot=previous_slot,
        beacon_block_root=anchor_root,
    )

    # Delayed confirmation is a strict relative majority of the frozen
    # participating seats: 256/512 fails, 257/512 passes.
    store.frozen_available_votes[previous_slot] = spec.FrozenAvailableVotes(
        committee=[validator_0] * 256 + [validator_1] * 256,
        votes={validator_0: child_vote, validator_1: anchor_vote},
    )
    assert spec.get_available_confirmation_score(store, child) == 256
    assert not spec.is_available_confirmation_viable(store, child)
    store.frozen_available_votes[previous_slot] = spec.FrozenAvailableVotes(
        committee=[validator_0] * 257 + [validator_1] * 255,
        votes={validator_0: child_vote, validator_1: anchor_vote},
    )
    assert spec.get_available_confirmation_score(store, child) == 257
    assert spec.is_available_confirmation_viable(store, child)

    # Fast confirmation is an absolute 75% of all 512 seats: 383 fails and
    # exactly 384 passes, including duplicate seats held by one validator.
    fast_vote = child_vote.copy()
    fast_vote.slot = current_slot
    store.frozen_available_votes[current_slot] = spec.FrozenAvailableVotes(
        committee=[validator_0] * 383 + [validator_1] * 129,
        votes={validator_0: fast_vote},
    )
    assert spec.get_fast_confirmation_score(store, child) == 383
    assert not spec.is_fast_confirmation_viable(store, child)
    store.frozen_available_votes[current_slot] = spec.FrozenAvailableVotes(
        committee=[validator_0] * 384 + [validator_1] * 128,
        votes={validator_0: fast_vote},
    )
    assert spec.get_fast_confirmation_score(store, child) == 384
    assert spec.is_fast_confirmation_viable(store, child)

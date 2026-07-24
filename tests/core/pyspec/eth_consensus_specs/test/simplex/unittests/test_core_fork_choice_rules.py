from eth_consensus_specs.test.context import spec_state_test, with_simplex_and_later
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store,
    get_viable_for_head_checks,
)


def _add_child(spec, store, state, parent_root, slot, marker, height=None):
    block = spec.BeaconBlock(
        slot=slot,
        parent_root=parent_root,
        body=spec.BeaconBlockBody(graffiti=bytes([marker]) * 32),
    )
    # Make the synthetic child build on the parent's EMPTY decision. The
    # Gloas node graph exposes a FULL parent only after its payload is locally
    # verified, which is outside these bookkeeping tests.
    block.body.signed_execution_payload_bid.message.parent_block_hash = spec.Hash32(
        bytes([marker]) * 32
    )
    root = block.hash_tree_root()
    block_state = state.copy()
    block_state.slot = slot
    if height is not None:
        block_state.current_height = height
    store.blocks[root] = block
    store.block_states[root] = block_state
    return root


def _set_store_slot(spec, store, slot):
    store.time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)


def _checkpoint(spec, store, root):
    return spec.Checkpoint(slot=store.blocks[root].slot, root=root)


def _minimal_quorum_indices(spec, state):
    active_indices = spec.get_active_validator_indices(state, spec.get_current_epoch(state))
    total = spec.get_total_active_balance(state)
    selected = []
    selected_weight = spec.Gwei(0)
    for index in active_indices:
        selected.append(index)
        selected_weight += state.validators[index].effective_balance
        if selected_weight * 3 >= total * 2:
            break
    assert selected_weight * 3 >= total * 2
    assert (selected_weight - state.validators[selected[-1]].effective_balance) * 3 < total * 2
    return selected


def _minimal_weight_indices(spec, state, threshold, excluded=()):
    active_indices = spec.get_active_validator_indices(state, spec.get_current_epoch(state))
    selected = []
    selected_weight = spec.Gwei(0)
    for index in active_indices:
        if index in excluded:
            continue
        selected.append(index)
        selected_weight += state.validators[index].effective_balance
        if selected_weight >= threshold:
            break
    assert selected_weight >= threshold
    assert selected_weight - state.validators[selected[-1]].effective_balance < threshold
    return selected


def _support_slot(spec, support_round=None):
    if support_round is None:
        support_round = spec.GENESIS_ROUND
    return spec.Slot(spec.compute_start_slot_at_round(spec.Round(support_round + 1)) - 1)


def _attestation_data(spec, slot, root, height=None):
    if height is None:
        height = spec.Height(0)
    return spec.AttestationData(
        slot=slot,
        beacon_block_root=root,
        height=height,
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )


def _record_head_votes(spec, store, indices, slot, head_root, height=None):
    data = _attestation_data(spec, slot, head_root, height)
    spec.update_latest_messages(store, indices, spec.Attestation(data=data))
    return data


def _freeze_support_round(spec, store, support_round=None):
    if support_round is None:
        support_round = spec.GENESIS_ROUND
    synchronization_round = spec.Round(support_round + 1)
    freeze_slot = spec.Slot(spec.compute_start_slot_at_round(synchronization_round) - 1)
    _set_store_slot(spec, store, freeze_slot)
    assert spec.compute_round_at_slot(freeze_slot) == support_round
    spec.freeze_tsq_view(store)
    assert synchronization_round in store.frozen_tsq_views
    return synchronization_round


def _freeze_selection(spec, store, synchronization_round):
    proposal_slot = spec.compute_start_slot_at_round(synchronization_round)
    _set_store_slot(spec, store, proposal_slot)
    spec.freeze_tsq_selection(store)
    assert synchronization_round in store.tsq_selections
    return store.tsq_selections[synchronization_round]


def _add_round_proposal(
    spec,
    store,
    state,
    parent_root,
    proposal_round,
    marker,
    slot=None,
    record=True,
):
    if slot is None:
        slot = spec.compute_start_slot_at_round(proposal_round)
    proposal_root = _add_child(
        spec,
        store,
        state,
        parent_root,
        slot,
        marker,
    )
    if record:
        spec.update_round_proposals(store, proposal_root)
    return proposal_root


def _indices_weight(spec, state, indices):
    return spec.Gwei(sum(state.validators[index].effective_balance for index in indices))


@with_simplex_and_later
@spec_state_test
def test_update_justified_lexicographic_max_and_finalized_filter(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    root_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB2)
    checkpoint_a = _checkpoint(spec, store, root_a)
    checkpoint_b = _checkpoint(spec, store, root_b)
    low, high = sorted(
        (checkpoint_a, checkpoint_b),
        key=spec.hash_tree_root,
    )

    spec.update_justified(store, low, spec.Height(1))
    assert store.justified_checkpoint == low
    assert store.justified_height == spec.Height(1)

    # Equal heights use the canonical checkpoint-root key, independent of
    # arrival order, and a later lower key cannot retract the running max.
    spec.update_justified(store, high, spec.Height(1))
    assert store.justified_checkpoint == high
    spec.update_justified(store, low, spec.Height(1))
    assert store.justified_checkpoint == high

    # Height is the primary key.
    spec.update_justified(store, low, spec.Height(2))
    assert store.justified_checkpoint == low
    assert store.justified_height == spec.Height(2)

    # Even an arbitrarily high key is rejected when it conflicts with F.
    store.finalized_checkpoint = checkpoint_a
    store.justified_checkpoint = checkpoint_a
    store.justified_height = spec.Height(1)
    conflicting = checkpoint_b
    spec.update_justified(store, conflicting, spec.Height(99))
    assert store.justified_checkpoint == checkpoint_a
    assert store.justified_height == spec.Height(1)


@with_simplex_and_later
@spec_state_test
def test_update_finalized_requires_candidate_ancestor_of_justified(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1, spec.Height(2))
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2, spec.Height(3))
    root_c2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xC2, spec.Height(3))
    checkpoint_a = _checkpoint(spec, store, root_a)
    checkpoint_a2 = _checkpoint(spec, store, root_a2)
    checkpoint_c2 = _checkpoint(spec, store, root_c2)
    store.justified_checkpoint = checkpoint_a2
    store.justified_height = spec.Height(2)
    store.h_max = spec.Height(3)

    # F' may move to an ancestor of J.
    spec.update_finalized(store, checkpoint_a)
    assert store.finalized_checkpoint == checkpoint_a

    # A sibling descendant of F is not an ancestor of J. This is the
    # direction-sensitive guard that a "descends from J" implementation gets
    # backward.
    spec.update_finalized(store, checkpoint_c2)
    assert store.finalized_checkpoint == checkpoint_a

    # The upper ancestry bound is inclusive: a viable F' equal to J is valid.
    spec.update_finalized(store, checkpoint_a2)
    assert store.finalized_checkpoint == checkpoint_a2


@with_simplex_and_later
@spec_state_test
def test_update_finalized_rejects_nonviable_and_non_descendant_candidates(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1, spec.Height(2))
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2, spec.Height(2))
    root_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB1, spec.Height(4))
    checkpoint_a = _checkpoint(spec, store, root_a)
    checkpoint_a2 = _checkpoint(spec, store, root_a2)
    checkpoint_b = _checkpoint(spec, store, root_b)

    # The competing height-4 leaf raises the viability floor to 3. The
    # monotone-height A branch therefore cannot be finalized even though A is
    # between the current F and J.
    store.justified_checkpoint = checkpoint_a2
    store.justified_height = spec.Height(2)
    store.h_max = spec.Height(4)
    spec.update_finalized(store, checkpoint_a)
    assert store.finalized_checkpoint.root == anchor_root

    # Exercise the defensive lower ancestry bound directly: even if a local
    # store is presented with J on a sibling branch, F cannot switch branches.
    store.h_max = spec.Height(2)
    store.finalized_checkpoint = checkpoint_a
    store.justified_checkpoint = checkpoint_b
    store.justified_height = spec.Height(4)
    spec.update_finalized(store, checkpoint_b)
    assert store.finalized_checkpoint == checkpoint_a


@with_simplex_and_later
@spec_state_test
def test_simplex_root_and_viability_exact_height_boundaries(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    low_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0x11, spec.Height(1))
    frontier_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0x22, spec.Height(2))
    store.justified_checkpoint = _checkpoint(spec, store, frontier_root)
    store.justified_height = spec.Height(2)

    store.h_max = spec.Height(3)
    assert spec.get_simplex_root(store) == frontier_root
    assert spec.get_viability_height_threshold(store) == spec.Height(2)
    assert not spec.is_viable_leaf(store, low_root)
    assert spec.is_viable_leaf(store, frontier_root)

    filtered = spec.get_filtered_block_tree(store)
    assert low_root not in filtered
    assert frontier_root in filtered

    store.h_max = spec.Height(4)
    assert spec.get_simplex_root(store) == store.finalized_checkpoint.root


@with_simplex_and_later
@spec_state_test
def test_grade_one_and_grade_zero_exact_thresholds_and_equivocation_denominator(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    root_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB2)
    indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))[:4]
    assert len(indices) == 4
    assert len({state.validators[index].effective_balance for index in indices}) == 1

    current_slot = spec.Slot(10)
    _set_store_slot(spec, store, current_slot)
    store.latest_messages = {
        indices[0]: spec.LatestMessage(slot=current_slot, root=root_a),
        indices[1]: spec.LatestMessage(slot=current_slot, root=root_a),
        indices[2]: spec.LatestMessage(slot=current_slot, root=root_b),
    }
    blocks = spec.get_filtered_block_tree(store)

    # Exactly two-thirds of the live denominator crosses G1.
    assert spec.get_grade_1_root(store, blocks).root == root_a
    # Exactly one-third on a conflicting branch vetoes G0.
    assert not spec.is_g0_clear(store, root_a)

    # A known equivocator is removed from both numerator and denominator.
    store.equivocating_indices.add(indices[2])
    expected = state.validators[indices[0]].effective_balance * 2
    assert spec.get_total_active_voting_weight(store) == expected
    assert spec.is_g0_clear(store, root_a)

    # With one vote on each branch, neither child reaches two-thirds.
    store.equivocating_indices = set()
    store.latest_messages = {
        indices[0]: spec.LatestMessage(slot=current_slot, root=root_a),
        indices[1]: spec.LatestMessage(slot=current_slot, root=root_b),
    }
    assert spec.get_grade_1_root(store, blocks).root == anchor_root


@with_simplex_and_later
@spec_state_test
def test_finalized_conflicts_are_inert_and_safe_confirmation_has_simplex_floor(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1, spec.Height(10))
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2, spec.Height(11))
    root_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB1, spec.Height(12))

    # Before finalization, B raises the height frontier. A remains finalizable
    # because its descendant reaches h_max - 1.
    store.justified_checkpoint = _checkpoint(spec, store, root_a2)
    store.justified_height = spec.Height(10)
    store.h_max = spec.Height(12)
    spec.update_finalized(store, _checkpoint(spec, store, root_a))
    assert store.finalized_checkpoint.root == root_a

    # Finalization leaves the conflicting branch as raw evidence but removes
    # it from h_max and viability semantics.
    assert root_b in store.blocks
    assert store.h_max == spec.Height(11)
    assert not spec.is_viable_leaf(store, root_b)
    assert root_b not in spec.get_filtered_block_tree(store)

    current_slot = spec.Slot(10)
    _set_store_slot(spec, store, current_slot)
    indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))[:2]
    store.latest_messages = {
        indices[0]: spec.LatestMessage(slot=current_slot, root=root_a2),
        indices[1]: spec.LatestMessage(slot=current_slot, root=root_b),
    }
    expected = state.validators[indices[0]].effective_balance
    assert spec.get_total_active_voting_weight(store) == expected
    accounting_state = store.block_states[store.justified_checkpoint.root]
    conflicting_node = spec.ForkChoiceNode(
        root=root_b,
        payload_status=spec.PAYLOAD_STATUS_PENDING,
    )
    assert spec.get_attestation_score(store, conflicting_node, accounting_state) == spec.Gwei(0)
    assert spec.is_g0_clear(store, root_a2)

    # A stale live confirmation on B cannot push the internal safe head below
    # the Simplex root (J here).
    assert spec.get_simplex_root(store) == root_a2
    store.live_confirmed_head = (root_b, current_slot)
    assert spec.get_safe_confirmed_head(store) == root_a2


@with_simplex_and_later
@spec_state_test
def test_latest_message_expiry_excludes_exact_boundary(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))[:2]
    current_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS)
    _set_store_slot(spec, store, current_slot)
    store.latest_messages = {
        indices[0]: spec.LatestMessage(slot=spec.GENESIS_SLOT, root=anchor_root),
        indices[1]: spec.LatestMessage(slot=spec.Slot(1), root=anchor_root),
    }

    assert not spec.has_unexpired_latest_message(store, indices[0])
    assert spec.has_unexpired_latest_message(store, indices[1])
    assert (
        spec.get_total_active_voting_weight(store) == state.validators[indices[1]].effective_balance
    )


@with_simplex_and_later
@spec_state_test
def test_portable_viable_head_check_starts_at_finalized_when_simplex_root_does(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    finalized_root = store.finalized_checkpoint.root
    justified_root = _add_child(
        spec, store, state, finalized_root, spec.Slot(1), 0xA1, spec.Height(1)
    )
    sibling_root = _add_child(
        spec, store, state, finalized_root, spec.Slot(1), 0xB2, spec.Height(1)
    )
    store.justified_checkpoint = _checkpoint(spec, store, justified_root)
    store.justified_height = spec.Height(1)
    # h_max != justified_height + 1 makes the real walk start from F.
    store.h_max = spec.Height(1)
    assert spec.get_simplex_root(store) == finalized_root

    checks = get_viable_for_head_checks(spec, store)
    checked_roots = {spec.Root(bytes.fromhex(check["root"][2:])) for check in checks}
    assert justified_root in checked_roots
    assert sibling_root in checked_roots


@with_simplex_and_later
@spec_state_test
def test_tsq_quorum_threshold_uses_exact_ceiling(spec, state):
    assert spec.get_tsq_quorum_threshold(spec.Gwei(30)) == spec.Gwei(20)
    assert spec.get_tsq_quorum_threshold(spec.Gwei(31)) == spec.Gwei(21)
    assert spec.get_tsq_quorum_threshold(spec.Gwei(32)) == spec.Gwei(22)


@with_simplex_and_later
@spec_state_test
def test_tsq_high_resolution_handler_runs_only_at_exact_boundaries(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    synchronization_round = spec.Round(1)
    freeze_slot = spec.Slot(spec.compute_start_slot_at_round(synchronization_round) - 1)
    _set_store_slot(spec, store, freeze_slot)

    freeze_due = spec.get_view_freeze_due_ms()
    assert spec.is_before_view_freeze_deadline(store)
    spec.on_tick_per_high_resolution(store, spec.uint64(freeze_due - 1))
    assert spec.is_before_view_freeze_deadline(store)
    assert synchronization_round not in store.frozen_tsq_views
    spec.on_tick_per_high_resolution(store, freeze_due)
    assert freeze_slot in store.view_freeze_slots
    assert not spec.is_before_view_freeze_deadline(store)
    assert synchronization_round in store.frozen_tsq_views

    action_slot = spec.compute_start_slot_at_round(synchronization_round)
    _set_store_slot(spec, store, action_slot)
    spec.freeze_tsq_selection(store)
    action_due = spec.get_slot_component_duration_ms(spec.AVAILABLE_ATTESTATION_DUE_BPS)
    spec.on_tick_per_high_resolution(store, spec.uint64(action_due - 1))
    assert synchronization_round not in store.stable_root_decisions
    spec.on_tick_per_high_resolution(store, action_due)
    assert synchronization_round in store.stable_root_decisions

    confirmation_due = spec.get_available_confirmation_due_ms()
    assert spec.is_at_or_before_available_confirmation_deadline(store)
    assert not spec.is_at_or_after_available_confirmation_deadline(store)
    spec.on_tick_per_high_resolution(store, confirmation_due)
    assert action_slot in store.frozen_available_votes
    assert not spec.is_at_or_before_available_confirmation_deadline(store)
    assert spec.is_at_or_after_available_confirmation_deadline(store)


@with_simplex_and_later
@spec_state_test
def test_tsq_selection_pins_weights_and_slashed_denominator(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    simplex_state = store.block_states[simplex_root]
    active_indices = list(
        spec.get_active_validator_indices(
            simplex_state,
            spec.get_current_epoch(simplex_state),
        )
    )
    assert len(active_indices) >= 2
    slashed_index, retained_index = active_indices[:2]
    simplex_state.validators[slashed_index].slashed = True
    expected_total = spec.get_total_active_balance(simplex_state)
    retained_weight = simplex_state.validators[retained_index].effective_balance

    synchronization_round = _freeze_support_round(spec, store)
    selection = _freeze_selection(spec, store, synchronization_round)

    # The absolute denominator includes active slashed balance, while a
    # slashed validator contributes no support.
    assert selection.total_active_balance == expected_total
    assert slashed_index not in selection.weights
    assert selection.weights[retained_index] == retained_weight

    # The electorate is a round-boundary snapshot, not a live state lookup.
    simplex_state.validators[retained_index].effective_balance = spec.Gwei(0)
    simplex_state.validators[slashed_index].slashed = False
    assert selection.total_active_balance == expected_total
    assert selection.weights[retained_index] == retained_weight
    assert slashed_index not in selection.weights


@with_simplex_and_later
@spec_state_test
def test_tsq_effective_head_projects_only_to_candidate_ancestor(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    finalized_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, finalized_root, spec.Slot(1), 0xA1, spec.Height(1))
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2, spec.Height(2))
    root_b = _add_child(spec, store, state, finalized_root, spec.Slot(1), 0xB1, spec.Height(2))
    store.justified_checkpoint = _checkpoint(spec, store, root_a)
    store.justified_height = spec.Height(1)
    store.h_max = spec.Height(2)

    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    _record_head_votes(spec, store, quorum_indices, support_slot, root_a2)
    synchronization_round = _freeze_support_round(spec, store)
    selection = _freeze_selection(spec, store, synchronization_round)
    assert selection.simplex_root == root_a
    assert selection.candidate_root == root_a2

    # A descendant learned after selection projects back to its deepest pinned
    # ancestor. A sibling branch has no effective root in this selection.
    late_descendant = _add_child(spec, store, state, root_a2, spec.Slot(3), 0xA3, spec.Height(2))
    assert (
        spec.get_tsq_effective_head(
            store,
            selection,
            _attestation_data(spec, support_slot, late_descendant),
        )
        == root_a2
    )
    assert (
        spec.get_tsq_effective_head(
            store,
            selection,
            _attestation_data(spec, support_slot, root_b),
        )
        == spec.Root()
    )


@with_simplex_and_later
@spec_state_test
def test_tsq_intersection_requires_exact_frozen_data_and_excludes_current_only(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    frozen_data = _record_head_votes(spec, store, quorum_indices, support_slot, root_a)
    synchronization_round = _freeze_support_round(spec, store)

    active_indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))
    current_only_index = next(index for index in active_indices if index not in quorum_indices)
    _record_head_votes(spec, store, [current_only_index], support_slot, root_a)
    selection = _freeze_selection(spec, store, synchronization_round)
    frozen_view = store.frozen_tsq_views[synchronization_round]

    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert set(intersection) == set(quorum_indices)
    assert current_only_index not in intersection

    # Even the same head is excluded if another signed field differs. Assign
    # directly here to isolate exact-data intersection from equivocation logic.
    changed_index = quorum_indices[0]
    changed_data = frozen_data.copy()
    changed_data.height = spec.Height(1)
    store.round_attestations[selection.support_round][changed_index] = changed_data
    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert changed_index not in intersection
    assert set(intersection) == set(quorum_indices[1:])


@with_simplex_and_later
@spec_state_test
def test_tsq_post_freeze_second_message_excludes_without_credit(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    first_data = _record_head_votes(spec, store, quorum_indices, support_slot, root_a)
    synchronization_round = _freeze_support_round(spec, store)
    frozen_view = store.frozen_tsq_views[synchronization_round]

    pivotal_index = quorum_indices[-1]
    second_data = first_data.copy()
    second_data.height = spec.Height(1)
    spec.update_latest_messages(
        store,
        [pivotal_index],
        spec.Attestation(data=second_data),
    )
    selection = _freeze_selection(spec, store, synchronization_round)

    assert pivotal_index not in frozen_view.equivocating_indices
    assert pivotal_index in store.round_equivocating_indices[selection.support_round]
    assert selection.candidate_root == simplex_root

    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert pivotal_index not in intersection
    assert spec.get_tsq_support(store, selection, intersection, root_a) < q


@with_simplex_and_later
@spec_state_test
def test_tsq_selects_deepest_quorum_candidate_and_receiver_lock(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    _record_head_votes(spec, store, quorum_indices, support_slot, root_a2)
    synchronization_round = _freeze_support_round(spec, store)
    selection = _freeze_selection(spec, store, synchronization_round)
    frozen_view = store.frozen_tsq_views[synchronization_round]

    assert selection.candidate_root == root_a2
    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert spec.get_deepest_tsq_root(store, selection, intersection) == root_a2

    spec.freeze_stable_root(store)
    assert store.stable_root == root_a2
    assert store.stable_root_proposal_root == spec.Root()


@with_simplex_and_later
@spec_state_test
def test_round_proposals_require_start_slot_timeliness_and_pre_action_receipt(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    round_1 = spec.Round(1)
    round_1_start = spec.compute_start_slot_at_round(round_1)
    _set_store_slot(spec, store, round_1_start)

    timely_root = _add_round_proposal(spec, store, state, simplex_root, round_1, 0xC1)
    non_start_root = _add_round_proposal(
        spec,
        store,
        state,
        simplex_root,
        round_1,
        0xC2,
        slot=spec.Slot(round_1_start + 1),
    )
    assert store.round_proposals[round_1] == {timely_root}
    assert non_start_root not in store.round_proposals[round_1]

    round_2 = spec.Round(2)
    _set_store_slot(spec, store, spec.compute_start_slot_at_round(round_2))
    late_root = _add_round_proposal(
        spec,
        store,
        state,
        simplex_root,
        round_1,
        0xC3,
    )
    assert late_root not in store.round_proposals[round_1]

    # A proposal received only after the action can be recorded, but cannot
    # alter the already frozen round decision.
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    _set_store_slot(spec, store, round_1_start)
    spec.freeze_stable_root(store)
    assert store.stable_root_proposal_root == spec.Root()
    post_action_root = _add_round_proposal(spec, store, state, simplex_root, round_1, 0xC4)
    assert post_action_root in store.round_proposals[round_1]
    spec.freeze_stable_root(store)
    assert store.stable_root_proposal_root == spec.Root()


@with_simplex_and_later
@spec_state_test
def test_tsq_proposal_aligns_unconfirmed_head_but_preserves_live_confirmed_prefix(spec, state):
    def run(proposal_extends_goldfish, goldfish_is_confirmed):
        store = get_genesis_forkchoice_store(spec, state)
        simplex_root = store.finalized_checkpoint.root
        synchronization_round = spec.Round(1)
        proposal_slot = spec.compute_start_slot_at_round(synchronization_round)
        vote_slot = spec.Slot(proposal_slot - 1)
        goldfish_root = _add_child(spec, store, state, simplex_root, vote_slot, 0xA1)
        _set_store_slot(spec, store, vote_slot)
        spec.freeze_tsq_view(store)
        _freeze_selection(spec, store, synchronization_round)

        proposal_parent = goldfish_root if proposal_extends_goldfish else simplex_root
        proposal_root = _add_round_proposal(
            spec,
            store,
            state,
            proposal_parent,
            synchronization_round,
            0xC1 if proposal_extends_goldfish else 0xC2,
        )
        committee = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))[
            :4
        ]
        store.available_committees[vote_slot] = committee
        store.available_votes[vote_slot] = {
            index: spec.AvailableAttestationData(
                slot=vote_slot,
                beacon_block_root=goldfish_root,
            )
            for index in committee
        }
        store.available_vote_equivocations[vote_slot] = set()
        if goldfish_is_confirmed:
            store.live_confirmed_head = (goldfish_root, vote_slot)

        spec.freeze_stable_root(store)
        assert store.stable_root == simplex_root
        return (
            spec.get_head(store).root,
            goldfish_root,
            proposal_root,
            store.stable_root_proposal_root,
        )

    # The TSQ action may replace an unconfirmed ordinary Goldfish head. This is
    # the proposal bootstrap that aligns walks from compatible locks.
    head, goldfish_root, proposal_root, distinguished_root = run(
        proposal_extends_goldfish=False,
        goldfish_is_confirmed=False,
    )
    assert head == proposal_root
    assert goldfish_root != proposal_root
    assert distinguished_root == proposal_root

    # The same proposal cannot replace the action-time live confirmed prefix.
    head, goldfish_root, proposal_root, distinguished_root = run(
        proposal_extends_goldfish=False,
        goldfish_is_confirmed=True,
    )
    assert head == goldfish_root
    assert head != proposal_root
    assert distinguished_root == spec.Root()

    head, goldfish_root, proposal_root, distinguished_root = run(
        proposal_extends_goldfish=True,
        goldfish_is_confirmed=True,
    )
    assert head == proposal_root
    assert goldfish_root != proposal_root
    assert distinguished_root == proposal_root


@with_simplex_and_later
@spec_state_test
def test_tsq_frozen_quorum_prevents_ancestor_regression(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    root_a2 = _add_child(spec, store, state, root_a, spec.Slot(2), 0xA2)
    conflicting_root = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xB1)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    _record_head_votes(spec, store, quorum_indices, support_slot, root_a2)
    synchronization_round = _freeze_support_round(spec, store)

    active_indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))
    remaining_indices = [index for index in active_indices if index not in quorum_indices]
    _record_head_votes(spec, store, remaining_indices, support_slot, conflicting_root)
    selection = _freeze_selection(spec, store, synchronization_round)
    assert selection.candidate_root == root_a2

    frozen_view = store.frozen_tsq_views[synchronization_round]
    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert spec.get_deepest_tsq_root(store, selection, intersection) == root_a2
    spec.freeze_stable_root(store)
    assert store.stable_root == root_a2


@with_simplex_and_later
@spec_state_test
def test_tsq_subquorum_conflicting_branch_cannot_select(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    conflicting_root = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xB1)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    subquorum_indices = quorum_indices[:-1]
    assert _indices_weight(spec, state, subquorum_indices) < q
    _record_head_votes(spec, store, subquorum_indices, support_slot, conflicting_root)

    # Relative G1 follows these votes because they are the whole live grade
    # denominator. Fixed-q TSQ still refuses to select their branch.
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_grade_1_root(store, blocks).root == conflicting_root
    synchronization_round = _freeze_support_round(spec, store)
    selection = _freeze_selection(spec, store, synchronization_round)
    assert selection.candidate_root == simplex_root

    spec.freeze_stable_root(store)
    assert store.stable_root == simplex_root
    assert store.stable_root != conflicting_root


@with_simplex_and_later
@spec_state_test
def test_tsq_relay_transfer_candidate_extends_receiver_lock(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    lock_root = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    candidate_root = _add_child(spec, store, state, lock_root, spec.Slot(2), 0xA2)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    active_indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))
    lock_indices = _minimal_weight_indices(spec, state, q)
    outside_indices = [index for index in active_indices if index not in lock_indices]
    outside_weight = _indices_weight(spec, state, outside_indices)
    deep_threshold = spec.Gwei(q - outside_weight)
    deep_indices = _minimal_weight_indices(
        spec,
        state,
        deep_threshold,
        excluded=outside_indices,
    )
    deep_set = set(deep_indices)
    shallow_indices = [index for index in lock_indices if index not in deep_set]
    assert deep_set <= set(lock_indices)
    assert _indices_weight(spec, state, deep_indices) < q

    # The receiver's public freeze has q support for lock_root but less than q
    # for candidate_root.
    _record_head_votes(spec, store, deep_indices, support_slot, candidate_root)
    _record_head_votes(spec, store, shallow_indices, support_slot, lock_root)
    synchronization_round = _freeze_support_round(spec, store)

    # Messages relayed after the receiver's freeze complete the proposer's
    # current q for candidate_root, but are absent from the receiver's F ∩ U.
    _record_head_votes(spec, store, outside_indices, support_slot, candidate_root)
    selection = _freeze_selection(spec, store, synchronization_round)
    assert selection.candidate_root == candidate_root

    frozen_view = store.frozen_tsq_views[synchronization_round]
    intersection = spec.get_tsq_intersection_heads(store, selection, frozen_view)
    assert spec.get_deepest_tsq_root(store, selection, intersection) == lock_root
    assert store.blocks[candidate_root].parent_root == lock_root


@with_simplex_and_later
@spec_state_test
def test_tsq_action_falls_back_when_lock_conflicts_with_current_simplex_root(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    finalized_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, finalized_root, spec.Slot(1), 0xA1, spec.Height(2))
    root_b = _add_child(spec, store, state, finalized_root, spec.Slot(1), 0xB1, spec.Height(2))
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    _record_head_votes(spec, store, quorum_indices, support_slot, root_a)
    synchronization_round = _freeze_support_round(spec, store)
    selection = _freeze_selection(spec, store, synchronization_round)
    assert selection.candidate_root == root_a

    # A newly authenticated justification moves the live Simplex root to a
    # sibling before the common action. The pinned lock is then inapplicable.
    store.justified_checkpoint = _checkpoint(spec, store, root_b)
    store.justified_height = spec.Height(2)
    store.h_max = spec.Height(3)
    blocks = spec.get_filtered_block_tree(store)
    assert root_a in blocks
    assert root_b in blocks
    assert spec.get_simplex_root(store) == root_b

    spec.freeze_stable_root(store)
    assert store.stable_root == root_b
    assert store.stable_root_proposal_root == spec.Root()


@with_simplex_and_later
@spec_state_test
def test_tsq_action_is_idempotent_after_late_votes_and_proposal(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    simplex_root = store.finalized_checkpoint.root
    root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
    support_slot = _support_slot(spec)
    total = spec.get_total_active_balance(state)
    q = spec.get_tsq_quorum_threshold(total)
    quorum_indices = _minimal_weight_indices(spec, state, q)
    first_data = _record_head_votes(spec, store, quorum_indices, support_slot, root_a)
    synchronization_round = _freeze_support_round(spec, store)
    _freeze_selection(spec, store, synchronization_round)
    proposal_root = _add_round_proposal(
        spec,
        store,
        state,
        root_a,
        synchronization_round,
        0xC1,
    )
    spec.freeze_stable_root(store)
    frozen = (
        store.stable_root,
        store.stable_root_payload_status,
        store.stable_root_proposal_root,
    )
    assert frozen[0] == root_a
    assert frozen[2] == proposal_root

    second_data = first_data.copy()
    second_data.height = spec.Height(1)
    spec.update_latest_messages(
        store,
        [quorum_indices[-1]],
        spec.Attestation(data=second_data),
    )
    _add_round_proposal(
        spec,
        store,
        state,
        root_a,
        synchronization_round,
        0xC2,
    )
    spec.freeze_stable_root(store)
    assert frozen == (
        store.stable_root,
        store.stable_root_payload_status,
        store.stable_root_proposal_root,
    )


@with_simplex_and_later
@spec_state_test
def test_tsq_proposal_selection_unique_equivocating_and_non_descending(spec, state):
    def prepare():
        store = get_genesis_forkchoice_store(spec, state)
        simplex_root = store.finalized_checkpoint.root
        root_a = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xA1)
        root_b = _add_child(spec, store, state, simplex_root, spec.Slot(1), 0xB1)
        support_slot = _support_slot(spec)
        total = spec.get_total_active_balance(state)
        q = spec.get_tsq_quorum_threshold(total)
        quorum_indices = _minimal_weight_indices(spec, state, q)
        _record_head_votes(spec, store, quorum_indices, support_slot, root_a)
        synchronization_round = _freeze_support_round(spec, store)
        _freeze_selection(spec, store, synchronization_round)
        return store, root_a, root_b, synchronization_round

    store, root_a, _root_b, round_ = prepare()
    unique_root = _add_round_proposal(spec, store, state, root_a, round_, 0xC1)
    spec.freeze_stable_root(store)
    assert store.stable_root == root_a
    assert store.stable_root_proposal_root == unique_root

    # A distinct second signed proposal detected by gossip suppresses the
    # first even if the second has not completed block processing.
    store, root_a, _root_b, round_ = prepare()
    _add_round_proposal(spec, store, state, root_a, round_, 0xC5)
    spec.mark_round_proposal_conflict(store, round_)
    spec.freeze_stable_root(store)
    assert store.stable_root == root_a
    assert store.stable_root_proposal_root == spec.Root()

    store, root_a, _root_b, round_ = prepare()
    nonviable_root = _add_round_proposal(spec, store, state, root_a, round_, 0xC2)
    viable_root = _add_round_proposal(spec, store, state, root_a, round_, 0xC3)
    store.block_states[viable_root].current_height = spec.Height(3)
    store.h_max = spec.Height(3)
    filtered = spec.get_filtered_block_tree(store)
    assert nonviable_root not in filtered
    assert viable_root in filtered
    # Proposal equivocation is decided before viability filtering, so the
    # single surviving viable copy still receives no special treatment.
    spec.freeze_stable_root(store)
    assert store.stable_root == root_a
    assert store.stable_root_proposal_root == spec.Root()

    store, root_a, root_b, round_ = prepare()
    _add_round_proposal(spec, store, state, root_b, round_, 0xC4)
    spec.freeze_stable_root(store)
    assert store.stable_root == root_a
    assert store.stable_root_proposal_root == spec.Root()


@with_simplex_and_later
@spec_state_test
def test_freeze_stable_root_preserves_complete_fallback_node(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    proposal_round = spec.Round(1)
    _set_store_slot(spec, store, spec.compute_start_slot_at_round(proposal_round))
    fallback = spec.ForkChoiceNode(
        root=anchor_root,
        payload_status=spec.PAYLOAD_STATUS_EMPTY,
    )

    original_get_grade_1_root = spec.get_grade_1_root
    spec.get_grade_1_root = lambda _store, _blocks: fallback
    try:
        spec.freeze_stable_root(store)
    finally:
        spec.get_grade_1_root = original_get_grade_1_root

    assert store.stable_root == fallback.root
    assert store.stable_root_payload_status == fallback.payload_status
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_stable_root(store, blocks) == fallback


@with_simplex_and_later
@spec_state_test
def test_frozen_stable_root_must_descend_current_simplex_root(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    finalized_root = store.finalized_checkpoint.root
    frozen_root = _add_child(
        spec,
        store,
        state,
        finalized_root,
        spec.Slot(1),
        0xA1,
        spec.Height(2),
    )
    new_simplex_root = _add_child(
        spec,
        store,
        state,
        finalized_root,
        spec.Slot(1),
        0xB1,
        spec.Height(2),
    )
    current_round = spec.Round(1)
    _set_store_slot(spec, store, spec.compute_start_slot_at_round(current_round))

    # Install an already-frozen round decision to isolate how later reads
    # treat a result that no longer descends from the live Simplex root.
    store.stable_root_decisions[current_round] = True
    store.stable_root_round = current_round
    store.stable_root = frozen_root
    store.stable_root_payload_status = spec.PAYLOAD_STATUS_PENDING
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_stable_root(store, blocks).root == frozen_root

    # Move the Simplex root to a sibling while keeping both branches viable.
    # The stale frozen root remains in the filtered tree, so the ancestry guard
    # (rather than mere tree membership) must select the current G1 fallback.
    store.justified_checkpoint = _checkpoint(spec, store, new_simplex_root)
    store.justified_height = spec.Height(2)
    store.h_max = spec.Height(3)
    blocks = spec.get_filtered_block_tree(store)
    assert frozen_root in blocks
    assert new_simplex_root in blocks
    assert spec.get_simplex_root(store) == new_simplex_root
    assert spec.get_stable_root(store, blocks).root == new_simplex_root


@with_simplex_and_later
@spec_state_test
def test_walk_goldfish_and_viability_phases_and_floorless_confirmation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    grade_1_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1, spec.Height(1))
    goldfish = _add_child(spec, store, state, grade_1_root, spec.Slot(2), 0xA2, spec.Height(1))
    current_slot = spec.Slot(3)
    _set_store_slot(spec, store, current_slot)
    indices = list(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))[:3]
    store.latest_messages = {
        index: spec.LatestMessage(slot=current_slot, root=grade_1_root) for index in indices
    }
    store.available_committees[spec.Slot(2)] = indices
    store.available_votes[spec.Slot(2)] = {
        index: spec.AvailableAttestationData(
            slot=spec.Slot(2),
            beacon_block_root=goldfish,
        )
        for index in indices
    }
    store.available_vote_equivocations[spec.Slot(2)] = set()
    store.h_max = spec.Height(1)

    # Phase 1 stops at the grade-1 fallback; phase 2 then follows the
    # previous-slot available majority to its child.
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_stable_root(store, blocks).root == grade_1_root
    assert spec.get_head(store).root == goldfish

    # In a separate adversarial tree, phase 3 is forced down the only viable
    # branch even without a Goldfish majority. The user-facing confirmation
    # walk intentionally remains floorless and may confirm the excluded branch.
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    excluded = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0x31, spec.Height(1))
    frontier = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0x42, spec.Height(2))
    _set_store_slot(spec, store, spec.Slot(2))
    store.h_max = spec.Height(3)
    voter = spec.ValidatorIndex(0)
    frozen_vote = spec.AvailableAttestationData(
        slot=spec.Slot(1),
        beacon_block_root=excluded,
    )
    store.frozen_available_votes[spec.Slot(1)] = spec.FrozenAvailableVotes(
        committee=[voter],
        votes={voter: frozen_vote},
    )

    assert excluded not in spec.get_filtered_block_tree(store)
    assert spec.get_head(store).root == frontier
    assert spec.get_available_confirmation_head(store).root == excluded

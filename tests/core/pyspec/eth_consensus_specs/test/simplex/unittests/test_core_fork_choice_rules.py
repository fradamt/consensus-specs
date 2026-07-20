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
def test_fresh_root_support_freezes_rejected_pointer_and_credits_round_equivocation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    pointer_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    pointer_descendant = _add_child(spec, store, state, pointer_root, spec.Slot(2), 0xA2)
    conflicting_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB1)
    proposal_round = spec.Round(1)
    proposal_slot = spec.compute_start_slot_at_round(proposal_round)
    proposal_root = _add_child(
        spec,
        store,
        state,
        pointer_descendant,
        proposal_slot,
        0xCC,
    )
    _set_store_slot(spec, store, proposal_slot)
    previous_round = spec.Round(proposal_round - 1)
    quorum_indices = _minimal_quorum_indices(spec, state)
    support_data = spec.AttestationData(
        slot=store.blocks[pointer_descendant].slot,
        beacon_block_root=pointer_descendant,
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    assert spec.compute_round_at_slot(support_data.slot) == previous_round
    conflicting_data = support_data.copy()
    conflicting_data.beacon_block_root = conflicting_root

    store.pointer_candidates[proposal_round] = {pointer_root: {proposal_root}}
    for index in quorum_indices[:-1]:
        spec.update_latest_messages(
            store,
            [index],
            spec.Attestation(data=support_data),
        )

    blocks = spec.get_filtered_block_tree(store)
    assert not spec.is_fresh_root(store, pointer_root, previous_round, state)
    # Before the common action, the ordinary grade-1 fallback follows the live
    # supporting votes to their deeper head.
    assert spec.get_stable_root(store, blocks).root == pointer_descendant

    # The pointer is below the absolute threshold at the action, so the whole
    # then-current G1 fallback is frozen for the round.
    spec.freeze_stable_root(store)
    assert store.stable_root_decisions == {proposal_round: True}
    assert store.stable_root == pointer_descendant
    assert store.stable_root_payload_status == spec.PAYLOAD_STATUS_PENDING
    assert store.stable_root_proposal_root == spec.Root()

    # A non-supporting final signer is insufficient until a second distinct
    # vote makes that signer an equivocator, at which point it is credited to
    # every proposed root. This makes the pointer fresh after the action, but
    # cannot change the already frozen fallback stable root.
    last_index = quorum_indices[-1]
    spec.update_latest_messages(
        store,
        [last_index],
        spec.Attestation(data=conflicting_data),
    )
    assert not spec.is_fresh_root(store, pointer_root, previous_round, state)
    spec.update_latest_messages(
        store,
        [last_index],
        spec.Attestation(data=support_data),
    )
    assert store.round_equivocating_indices[previous_round] == {last_index}
    assert spec.is_fresh_root(store, pointer_root, previous_round, state)
    assert spec.get_stable_root(store, blocks).root == pointer_descendant

    # Slashed active weight remains in the absolute denominator but cannot
    # contribute support. The minimal quorum therefore falls below threshold
    # when its pivotal signer is slashed, and recovers when restored.
    total_active_balance = spec.get_total_active_balance(state)
    state.validators[last_index].slashed = True
    assert spec.get_total_active_balance(state) == total_active_balance
    assert not spec.is_fresh_root(store, pointer_root, previous_round, state)
    state.validators[last_index].slashed = False
    assert spec.is_fresh_root(store, pointer_root, previous_round, state)


@with_simplex_and_later
@spec_state_test
def test_update_pointer_candidates_distinguishes_empty_and_rejects_known_equivocation(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    pointer_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    pointer_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB1)
    proposal_round = spec.Round(1)
    proposal_slot = spec.compute_start_slot_at_round(proposal_round)

    def add_proposal(slot, parent_root, pointer, marker):
        block = spec.BeaconBlock(
            slot=slot,
            parent_root=parent_root,
            body=spec.BeaconBlockBody(
                graffiti=bytes([marker]) * 32,
                anchor_root=pointer,
            ),
        )
        root = block.hash_tree_root()
        block_state = state.copy()
        block_state.slot = slot
        store.blocks[root] = block
        store.block_states[root] = block_state
        return root

    proposal_a = add_proposal(proposal_slot, pointer_a, pointer_a, 0xC1)
    proposal_b = add_proposal(proposal_slot, pointer_b, pointer_b, 0xC2)
    proposal_empty = add_proposal(proposal_slot, pointer_a, spec.Root(), 0xC4)
    non_start = add_proposal(spec.Slot(proposal_slot + 1), proposal_a, pointer_b, 0xC3)

    # A proposal delivered outside its own round is ignored.
    _set_store_slot(spec, store, spec.compute_start_slot_at_round(spec.Round(2)))
    spec.update_pointer_candidates(store, proposal_a)
    assert proposal_round not in store.pointer_candidates

    # A same-round proposal from a non-start slot is also ignored.
    _set_store_slot(spec, store, spec.Slot(proposal_slot + 1))
    spec.update_pointer_candidates(store, non_start)
    assert proposal_round not in store.pointer_candidates

    # At the round start, no proposal, an explicit empty pointer, and a
    # non-empty pointer are distinct observations.
    _set_store_slot(spec, store, proposal_slot)
    assert proposal_round not in store.pointer_candidates
    spec.update_pointer_candidates(store, proposal_empty)
    assert store.pointer_candidates[proposal_round] == {spec.Root(): {proposal_empty}}
    spec.update_pointer_candidates(store, proposal_a)
    spec.update_pointer_candidates(store, proposal_b)
    assert set(store.pointer_candidates[proposal_round]) == {
        spec.Root(),
        pointer_a,
        pointer_b,
    }

    # Multiple pointer values known at the action make all of them unusable;
    # the then-current G1 fallback (the finalized root here) is frozen.
    spec.freeze_stable_root(store)
    assert store.stable_root_decisions == {proposal_round: True}
    assert store.stable_root == anchor_root
    assert store.stable_root_payload_status == spec.PAYLOAD_STATUS_PENDING
    assert store.stable_root_proposal_root == spec.Root()
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_stable_root(store, blocks).root == anchor_root


@with_simplex_and_later
@spec_state_test
def test_pointer_decision_filters_proposal_copies_by_current_finalized_root(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    branch_a = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    branch_b = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xB1)
    pointer_root = _add_child(spec, store, state, branch_a, spec.Slot(2), 0xA2)
    proposal_round = spec.Round(1)
    proposal_slot = spec.compute_start_slot_at_round(proposal_round)

    def add_proposal(parent_root, pointer, marker):
        block = spec.BeaconBlock(
            slot=proposal_slot,
            parent_root=parent_root,
            body=spec.BeaconBlockBody(
                graffiti=bytes([marker]) * 32,
                anchor_root=pointer,
            ),
        )
        root = block.hash_tree_root()
        block_state = state.copy()
        block_state.slot = proposal_slot
        store.blocks[root] = block
        store.block_states[root] = block_state
        return root

    proposal_on_a = add_proposal(pointer_root, pointer_root, 0xC1)
    proposal_on_b_same_pointer = add_proposal(branch_b, pointer_root, 0xC2)
    proposal_on_b_other_pointer = add_proposal(branch_b, branch_b, 0xC3)
    _set_store_slot(spec, store, proposal_slot)
    for proposal_root in (
        proposal_on_a,
        proposal_on_b_same_pointer,
        proposal_on_b_other_pointer,
    ):
        spec.update_pointer_candidates(store, proposal_root)

    assert store.pointer_candidates[proposal_round] == {
        pointer_root: {proposal_on_a, proposal_on_b_same_pointer},
        branch_b: {proposal_on_b_other_pointer},
    }

    # Finalizing A before the action makes both B proposal copies inert. The
    # B-only pointer value no longer creates an equivocation, while the
    # same-pointer A copy remains as the deterministic denominator state.
    store.justified_checkpoint = _checkpoint(spec, store, pointer_root)
    store.justified_height = spec.Height(0)
    spec.update_finalized(store, _checkpoint(spec, store, branch_a))
    assert store.finalized_checkpoint.root == branch_a
    previous_round = spec.Round(proposal_round - 1)
    store.round_equivocating_indices[previous_round] = set(_minimal_quorum_indices(spec, state))
    spec.freeze_stable_root(store)
    assert store.stable_root == pointer_root
    assert store.stable_root_proposal_root == proposal_on_a

    # A later finalized advance cannot reopen or replace the frozen decision.
    frozen = (
        store.stable_root,
        store.stable_root_payload_status,
        store.stable_root_proposal_root,
    )
    store.justified_checkpoint = _checkpoint(spec, store, proposal_on_a)
    store.justified_height = spec.Height(1)
    spec.update_finalized(store, _checkpoint(spec, store, proposal_on_a))
    assert store.finalized_checkpoint.root == proposal_on_a
    spec.freeze_stable_root(store)
    assert frozen == (
        store.stable_root,
        store.stable_root_payload_status,
        store.stable_root_proposal_root,
    )


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
def test_freeze_stable_root_accepts_fresh_pointer_once(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    pointer_root = _add_child(spec, store, state, anchor_root, spec.Slot(1), 0xA1)
    proposal_round = spec.Round(1)
    proposal_slot = spec.compute_start_slot_at_round(proposal_round)
    proposal_root = _add_child(
        spec,
        store,
        state,
        pointer_root,
        proposal_slot,
        0xC1,
    )
    _set_store_slot(spec, store, proposal_slot)
    previous_round = spec.Round(proposal_round - 1)
    quorum_indices = _minimal_quorum_indices(spec, state)
    store.round_equivocating_indices[previous_round] = set(quorum_indices)
    store.pointer_candidates[proposal_round] = {pointer_root: {proposal_root}}

    spec.freeze_stable_root(store)
    assert store.stable_root == pointer_root
    assert store.stable_root_payload_status == spec.PAYLOAD_STATUS_PENDING
    assert store.stable_root_proposal_root == proposal_root

    # Even removing the support view after the action cannot recompute the
    # accepted pointer or replace it with a newer G1 fallback.
    store.round_equivocating_indices[previous_round] = set()
    blocks = spec.get_filtered_block_tree(store)
    assert spec.get_stable_root(store, blocks).root == pointer_root


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

    # Install an already-frozen round decision. Pointer acceptance itself is
    # covered separately; this test isolates how later reads treat the result.
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

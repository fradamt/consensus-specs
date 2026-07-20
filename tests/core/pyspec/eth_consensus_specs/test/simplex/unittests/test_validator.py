from eth_consensus_specs.test.context import (
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _add_child(spec, store, state, parent_root, slot, marker):
    block = spec.BeaconBlock(
        slot=slot,
        parent_root=parent_root,
        body=spec.BeaconBlockBody(graffiti=bytes([marker]) * 32),
    )
    root = block.hash_tree_root()
    block_state = state.copy()
    block_state.slot = slot
    block_state.current_height_start_slot = slot
    store.blocks[root] = block
    store.block_states[root] = block_state
    return root, block_state


def _setup_sibling_heads(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    slot = spec.Slot(state.slot + 1)
    root_a, state_a = _add_child(spec, store, state, anchor_root, slot, 0xA1)
    root_b, state_b = _add_child(spec, store, state, anchor_root, slot, 0xB2)
    store.time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)
    target_a = spec.Checkpoint(slot=slot, root=root_a)
    target_b = spec.Checkpoint(slot=slot, root=root_b)
    state_a.current_height_target = target_a
    state_b.current_height_target = target_b
    store.block_states[root_a] = state_a
    store.block_states[root_b] = state_b
    return store, root_a, state_a, target_a, root_b, state_b, target_b


def _freeze_round_vote(spec, store, voted_target_at=None, voted_timeout_at=None):
    frozen_round_vote_at = {}
    head_root = spec.get_head(store).root
    spec.freeze_round_vote(
        store,
        selection_event=spec.RoundSelectionEvent(
            slot=spec.get_current_slot(store),
            head_root=head_root,
            safe_confirmed_root=spec.get_safe_confirmed_head(store),
            finalized_root=store.finalized_checkpoint.root,
        ),
        voted_target_at={} if voted_target_at is None else voted_target_at,
        voted_timeout_at=set() if voted_timeout_at is None else voted_timeout_at,
        voted_finality_at={},
        frozen_round_vote_at=frozen_round_vote_at,
    )
    return frozen_round_vote_at


def _set_latest_head_votes(spec, store, slot, root):
    state = store.block_states[store.justified_checkpoint.root]
    epoch = spec.compute_epoch_at_slot(slot)
    for index in spec.get_active_validator_indices(state, epoch):
        store.latest_messages[index] = spec.LatestMessage(slot=slot, root=root)


def _set_round_selection_time(spec, store, slot):
    # ``Store.time`` is whole seconds; the event token represents the exact
    # possibly-subsecond deadline and its deadline-filtered view.
    due_seconds = (spec.get_attestation_due_ms() + 999) // 1000
    store.time = spec.uint64(
        store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000 + due_seconds
    )


@with_simplex_and_later
@spec_state_test
def test_first_block_is_immediate_target_before_state_update(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    head_root, head_state = _add_child(
        spec,
        store,
        state,
        anchor_root,
        selection_slot,
        0xA0,
    )
    head_state.current_height_target = spec.Checkpoint()
    store.block_states[head_root] = head_state
    _set_round_selection_time(spec, store, selection_slot)
    frozen_round_vote_at = {}

    frozen_vote = spec.freeze_round_vote(
        store,
        selection_event=spec.RoundSelectionEvent(
            slot=selection_slot,
            head_root=head_root,
            safe_confirmed_root=head_root,
            finalized_root=store.finalized_checkpoint.root,
        ),
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
        frozen_round_vote_at=frozen_round_vote_at,
    )

    assert head_state.current_height_target == spec.Checkpoint()
    assert frozen_vote.target == spec.Checkpoint(slot=selection_slot, root=head_root)
    assert frozen_vote.height == head_state.current_height


@with_simplex_and_later
@spec_state_test
def test_e1_lock_repeats_compatible_base_target(spec, state):
    store, root_a, state_a, target_a, _, _, _ = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height

    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: target_a},
        voted_timeout_at=set(),
        voted_finality_at={current_height: target_a},
    )

    assert target == target_a
    assert height == current_height


@with_simplex_and_later
@spec_state_test
def test_unknown_post_anchor_target_requires_same_height_state_match(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    child_slot = spec.Slot(store.blocks[anchor_root].slot + 2)
    child_root, child_state = _add_child(
        spec,
        store,
        state,
        anchor_root,
        child_slot,
        0xA3,
    )
    target_a = spec.Checkpoint(
        slot=spec.Slot(child_slot - 1),
        root=spec.Root(b"\xa1" * 32),
    )
    target_b = spec.Checkpoint(
        slot=target_a.slot,
        root=spec.Root(b"\xb2" * 32),
    )
    child_state.current_height_start_slot = target_a.slot
    child_state.current_height_target = target_a
    store.block_states[child_root] = child_state
    store.time = spec.uint64(store.genesis_time + child_slot * spec.config.SLOT_DURATION_MS // 1000)

    assert target_a.slot > store.finalized_checkpoint.slot
    assert spec.is_current_height_target_on_chain(
        store,
        child_root,
        child_state,
        target_a,
        child_state.current_height,
    )
    target, height = spec.get_attestation_target(
        store,
        child_root,
        child_state,
        child_root,
        child_state,
        target_a,
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == target_a
    assert height == child_state.current_height

    different_target_state = child_state.copy()
    different_target_state.current_height_target = target_b
    assert not spec.is_current_height_target_on_chain(
        store,
        child_root,
        different_target_state,
        target_a,
        child_state.current_height,
    )
    target, height = spec.get_attestation_target(
        store,
        child_root,
        child_state,
        child_root,
        different_target_state,
        target_a,
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == child_state.current_height  # Confirmation-gated timeout.


@with_simplex_and_later
@spec_state_test
def test_frozen_pre_anchor_target_remains_signable_with_equal_slot_boundary(spec, state):
    store, root_a, state_a, target_a, _, _, _ = _setup_sibling_heads(spec, state)
    anchor_slot = spec.Slot(store.blocks[root_a].slot + 1)
    anchor_root, anchor_state = _add_child(
        spec,
        store,
        state_a,
        root_a,
        anchor_slot,
        0xA4,
    )
    anchor_state.current_height_target = target_a
    store.block_states[anchor_root] = anchor_state
    store.finalized_checkpoint = spec.Checkpoint(slot=anchor_slot, root=anchor_root)
    store.time = spec.uint64(
        store.genesis_time + anchor_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    del store.blocks[root_a]

    assert target_a.slot < store.finalized_checkpoint.slot
    assert spec.is_authenticated_checkpoint_on_chain(store, anchor_root, target_a)
    unknown_equal_slot = spec.Checkpoint(
        slot=anchor_slot,
        root=spec.Root(b"\xee" * 32),
    )
    assert not spec.is_authenticated_checkpoint_on_chain(
        store,
        anchor_root,
        unknown_equal_slot,
    )

    round = spec.compute_round_at_slot(anchor_slot)
    frozen_round_vote_at = {
        round: spec.FrozenRoundVote(
            target=target_a,
            height=anchor_state.current_height,
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        )
    }
    original_get_head = spec.get_head
    spec.get_head = lambda _store: spec.ForkChoiceNode(
        root=anchor_root,
        payload_status=spec.PAYLOAD_STATUS_PENDING,
    )
    try:
        data = spec.get_attestation_data(store, frozen_round_vote_at, set())
    finally:
        spec.get_head = original_get_head

    assert data is not None
    assert data.target == target_a
    assert data.height == anchor_state.current_height


@with_simplex_and_later
@spec_state_test
def test_e1_lock_falls_back_to_empty_for_different_same_chain_target(spec, state):
    store, root_a, state_a, target_a, _, _, _ = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height
    older_same_chain_target = store.finalized_checkpoint

    assert spec.is_ancestor(
        store,
        spec.ForkChoiceNode(root=root_a, payload_status=spec.PAYLOAD_STATUS_PENDING),
        spec.ForkChoiceNode(
            root=older_same_chain_target.root,
            payload_status=spec.PAYLOAD_STATUS_PENDING,
        ),
    )
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: older_same_chain_target},
        voted_timeout_at=set(),
        voted_finality_at={current_height: older_same_chain_target},
    )

    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_e1_lock_falls_back_to_empty_for_off_head_target(spec, state):
    store, root_a, state_a, target_a, root_b, state_b, target_b = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height
    assert state_b.current_height == current_height

    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_b,
        state_b,
        target_a,
        voted_target_at={current_height: target_b},
        voted_timeout_at=set(),
        voted_finality_at={current_height: target_b},
    )

    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_timeout_first_history_never_crosses_back_to_target(spec, state):
    store, root_a, state_a, target_a, _, _, _ = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height

    # Even though the ordinary fresh gate admits ``target_a``, a prior timeout
    # makes the validator repeat the timeout instead of crossing vote kind.
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={},
        voted_timeout_at={current_height},
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == current_height

    # The repeat remains confirmation-gated, and an inconsistent durable E1
    # lock always resolves conservatively to the empty vote.
    behind_state = state_a.copy()
    behind_state.current_height = spec.Height(current_height - 1)
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        behind_state,
        target_a,
        voted_target_at={},
        voted_timeout_at={current_height},
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)

    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={},
        voted_timeout_at={current_height},
        voted_finality_at={current_height: target_a},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_nonjustifiable_saved_target_repeat_and_unlocked_timeout_bridge(spec, state):
    store, root_a, state_a, target_a, _, _, target_b = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height
    state_a.current_height_nonjustifiable = True

    # An exact saved target is safe to re-emit. The latched state class makes it
    # marker-only: the branch can time out but can never justify this target.
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: target_a},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == target_a
    assert height == current_height

    # If the target came from a different ordinary branch, an unlocked signer
    # may bridge to timeout. Durable signing then records timeout-first
    # discipline and closes its finality gate at this height.
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: target_b},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == current_height

    # Once a finality commitment locks that incompatible saved target, a
    # timeout would be E1 evidence and the only valid fallback is empty.
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: target_b},
        voted_timeout_at=set(),
        voted_finality_at={current_height: target_b},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_ordinary_incompatible_unlocked_target_bridges_to_timeout(spec, state):
    store, root_a, state_a, target_a, _, _, target_b = _setup_sibling_heads(spec, state)
    current_height = state_a.current_height
    assert not state_a.current_height_nonjustifiable

    # Pre-convergence validators may have saved targets on incompatible ordinary
    # branches. Once the common branch is safe-confirmed into the interval, each
    # unlocked history must be able to contribute its height-progress marker.
    for saved_target in (target_b, store.finalized_checkpoint):
        target, height = spec.get_attestation_target(
            store,
            root_a,
            state_a,
            root_a,
            state_a,
            target_a,
            voted_target_at={current_height: saved_target},
            voted_timeout_at=set(),
            voted_finality_at={},
        )
        assert target == spec.Checkpoint()
        assert height == current_height

    # An E1 finality commitment still forbids the bridge.
    target, height = spec.get_attestation_target(
        store,
        root_a,
        state_a,
        root_a,
        state_a,
        target_a,
        voted_target_at={current_height: target_b},
        voted_timeout_at=set(),
        voted_finality_at={current_height: target_b},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_fresh_proposal_root_depth_and_root_tiebreak(spec, state):
    low_root = spec.Root(b"\x11" * 32)
    high_root = spec.Root(b"\x22" * 32)

    assert spec.select_fresh_proposal_root([]) == spec.Root()
    # Depth is primary even when the shallower root is lexicographically larger.
    assert (
        spec.select_fresh_proposal_root([(high_root, spec.uint64(3)), (low_root, spec.uint64(4))])
        == low_root
    )
    # Equal-depth competing candidates use the root as the deterministic tie.
    assert (
        spec.select_fresh_proposal_root([(low_root, spec.uint64(4)), (high_root, spec.uint64(4))])
        == high_root
    )


@with_simplex_and_later
@spec_state_test
def test_round_vote_requires_common_deadline_event(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    target_root, _ = _add_child(spec, store, state, anchor_root, selection_slot, 0x30)
    late_root, _ = _add_child(spec, store, state, anchor_root, selection_slot, 0x2F)
    _set_latest_head_votes(spec, store, selection_slot, target_root)
    store.live_confirmed_head = (target_root, selection_slot)
    frozen_round_vote_at = {}
    selection_event = spec.RoundSelectionEvent(
        slot=selection_slot,
        head_root=target_root,
        safe_confirmed_root=target_root,
        finalized_root=anchor_root,
    )

    def freeze():
        return spec.freeze_round_vote(
            store,
            selection_event=selection_event,
            voted_target_at={},
            voted_timeout_at=set(),
            voted_finality_at={},
            frozen_round_vote_at=frozen_round_vote_at,
        )

    _set_round_selection_time(spec, store, selection_slot)
    # A token for any other slot cannot create this round's permanent snapshot.
    wrong_event = spec.RoundSelectionEvent(
        slot=spec.Slot(selection_slot + 1),
        head_root=target_root,
        safe_confirmed_root=target_root,
        finalized_root=anchor_root,
    )
    expect_assertion_error(
        lambda: spec.freeze_round_vote(
            store,
            selection_event=wrong_event,
            voted_target_at={},
            voted_timeout_at=set(),
            voted_finality_at={},
            frozen_round_vote_at=frozen_round_vote_at,
        )
    )
    assert frozen_round_vote_at == {}

    # Messages applied after issuance can change the live store, but the event's
    # roots keep selection on the deadline-filtered view.
    _set_latest_head_votes(spec, store, selection_slot, late_root)
    store.live_confirmed_head = (late_root, selection_slot)
    frozen_vote = freeze()
    assert frozen_round_vote_at == {selection_round: frozen_vote}
    assert frozen_vote.target.root == target_root


@with_simplex_and_later
@spec_state_test
def test_round_vote_stays_frozen_while_live_head_advances(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    target_root, target_state = _add_child(spec, store, state, anchor_root, selection_slot, 0x31)
    _set_round_selection_time(spec, store, selection_slot)
    _set_latest_head_votes(spec, store, selection_slot, target_root)
    store.live_confirmed_head = (target_root, selection_slot)

    frozen_round_vote_at = _freeze_round_vote(spec, store)
    frozen_vote = frozen_round_vote_at[selection_round]
    assert frozen_vote.target == spec.Checkpoint(slot=selection_slot, root=target_root)

    duty_slot = spec.Slot(selection_slot + 1)
    child_root, _ = _add_child(spec, store, target_state, target_root, duty_slot, 0x32)
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)
    _set_latest_head_votes(spec, store, duty_slot, child_root)
    store.live_confirmed_head = (child_root, duty_slot)

    data = spec.get_attestation_data(store, frozen_round_vote_at, set())
    assert data is not None
    assert data.slot == duty_slot
    assert data.beacon_block_root == child_root
    assert data.target == frozen_vote.target
    assert data.height == frozen_vote.height


@with_simplex_and_later
@spec_state_test
def test_round_vote_first_selection_wins(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    root_a, _ = _add_child(spec, store, state, anchor_root, selection_slot, 0x35)
    root_b, _ = _add_child(spec, store, state, anchor_root, selection_slot, 0x36)
    _set_round_selection_time(spec, store, selection_slot)

    _set_latest_head_votes(spec, store, selection_slot, root_a)
    store.live_confirmed_head = (root_a, selection_slot)
    frozen_round_vote_at = _freeze_round_vote(spec, store)
    first_vote = frozen_round_vote_at[selection_round]
    assert first_vote.target.root == root_a

    # A later equivocating proposal can change the live view but cannot replace
    # the round-selection result that was already persisted.
    _set_latest_head_votes(spec, store, selection_slot, root_b)
    store.live_confirmed_head = (root_b, selection_slot)
    second_vote = spec.freeze_round_vote(
        store,
        selection_event=spec.RoundSelectionEvent(
            slot=selection_slot,
            head_root=root_b,
            safe_confirmed_root=root_b,
            finalized_root=anchor_root,
        ),
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
        frozen_round_vote_at=frozen_round_vote_at,
    )

    assert second_vote == first_vote
    assert frozen_round_vote_at == {selection_round: first_vote}


@with_simplex_and_later
@spec_state_test
def test_round_vote_conflicting_later_head_fails_closed_without_retarget(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    target_root, _ = _add_child(spec, store, state, anchor_root, selection_slot, 0x41)
    _set_round_selection_time(spec, store, selection_slot)
    _set_latest_head_votes(spec, store, selection_slot, target_root)
    store.live_confirmed_head = (target_root, selection_slot)
    frozen_round_vote_at = _freeze_round_vote(spec, store)
    frozen_target = frozen_round_vote_at[selection_round].target
    assert frozen_target.root == target_root

    duty_slot = spec.Slot(selection_slot + 1)
    conflict_root, _ = _add_child(spec, store, state, anchor_root, duty_slot, 0x42)
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)
    _set_latest_head_votes(spec, store, duty_slot, conflict_root)
    store.live_confirmed_head = (conflict_root, duty_slot)

    frozen_before = frozen_round_vote_at.copy()
    skipped_attestation_at_round = set()
    data = spec.get_attestation_data(store, frozen_round_vote_at, skipped_attestation_at_round)
    assert data is None
    assert frozen_round_vote_at == frozen_before
    assert skipped_attestation_at_round == {selection_round}

    # Returning to the selected chain cannot resurrect the consumed duty.
    _set_latest_head_votes(spec, store, duty_slot, target_root)
    store.live_confirmed_head = (target_root, duty_slot)
    assert (
        spec.get_attestation_data(store, frozen_round_vote_at, skipped_attestation_at_round) is None
    )


@with_simplex_and_later
@spec_state_test
def test_round_vote_survives_height_advance_mid_round(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    target_root, target_state = _add_child(spec, store, state, anchor_root, selection_slot, 0x51)
    _set_round_selection_time(spec, store, selection_slot)
    _set_latest_head_votes(spec, store, selection_slot, target_root)
    store.live_confirmed_head = (target_root, selection_slot)
    frozen_round_vote_at = _freeze_round_vote(spec, store)
    frozen_vote = frozen_round_vote_at[selection_round]

    duty_slot = spec.Slot(selection_slot + 1)
    advanced_root, advanced_state = _add_child(
        spec, store, target_state, target_root, duty_slot, 0x52
    )
    advanced_state.current_height = spec.Height(frozen_vote.height + 1)
    advanced_state.current_height_start_slot = duty_slot
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)
    _set_latest_head_votes(spec, store, duty_slot, advanced_root)
    store.live_confirmed_head = (advanced_root, duty_slot)

    data = spec.get_attestation_data(store, frozen_round_vote_at, set())
    assert data is not None
    assert data.beacon_block_root == advanced_root
    assert data.target == frozen_vote.target
    assert data.height == frozen_vote.height
    assert data.height != advanced_state.current_height


@with_simplex_and_later
@spec_state_test
def test_missing_round_vote_after_restart_fails_closed(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    selection_round = spec.Round(spec.compute_round_at_slot(state.slot) + 1)
    selection_slot = spec.compute_start_slot_at_round(selection_round)
    store.time = spec.uint64(
        store.genesis_time + selection_slot * spec.config.SLOT_DURATION_MS // 1000
    )

    expect_assertion_error(lambda: spec.get_attestation_data(store, {}, set()))


@with_simplex_and_later
@spec_state_test
def test_empty_slot_height_advance_stays_empty_without_safe_interval_block(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    head_root = store.finalized_checkpoint.root
    head_state = state.copy()
    head_state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))
    for index in spec.get_active_validator_indices(head_state, spec.get_current_epoch(head_state)):
        head_state.timeouts[index] = True
    assert spec.has_timeout_quorum(head_state)
    head_state.pending_height_outcomes = 1
    store.block_states[head_root] = head_state

    pre_height = head_state.current_height
    duty_slot = spec.Slot(head_state.slot + 1)
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)

    processed_head_state = spec.get_current_slot_state(store, head_root)
    assert processed_head_state.current_height == pre_height + 1
    assert processed_head_state.current_height_start_slot == duty_slot
    assert store.blocks[head_root].slot < processed_head_state.current_height_start_slot
    assert processed_head_state.current_height_target == spec.Checkpoint()

    target, height = spec.get_attestation_target(
        store,
        head_root,
        processed_head_state,
        head_root,
        store.block_states[head_root],
        spec.Checkpoint(),
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )

    # Processing the head copy reveals the new duty height, but the stored
    # safe-confirmed block has not reached that interval. It therefore cannot
    # contribute a timeout marker to the paper's safe-confirmation certificate.
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)
    assert store.block_states[head_root].current_height == pre_height


@with_simplex_and_later
@spec_state_test
def test_empty_slot_nonjustifiable_height_also_requires_safe_interval_block(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    head_root = store.finalized_checkpoint.root
    head_state = state.copy()
    head_state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))
    next_height = spec.Height(spec.K_NONJUSTIFIABLE)
    head_state.current_height = spec.Height(next_height - 1)
    head_state.finalized_height = spec.Height(0)
    for index in spec.get_active_validator_indices(head_state, spec.get_current_epoch(head_state)):
        head_state.timeouts[index] = True
    assert spec.has_timeout_quorum(head_state)
    head_state.pending_height_outcomes = 1
    store.block_states[head_root] = head_state
    store.h_max = head_state.current_height

    duty_slot = spec.Slot(head_state.slot + 1)
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)
    processed_head_state = spec.get_current_slot_state(store, head_root)
    assert processed_head_state.current_height == next_height
    assert spec.is_nonjustifiable_height(
        processed_head_state.current_height,
        processed_head_state.finalized_height,
    )
    assert processed_head_state.current_height_nonjustifiable

    target, height = spec.get_attestation_target(
        store,
        head_root,
        processed_head_state,
        head_root,
        store.block_states[head_root],
        spec.Checkpoint(),
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )

    # The paper's weaker nonjustifiable caught-up predicate would emit a
    # timeout here even though C is still at ``next_height - 1``. The executable
    # profile preserves the marker certificate and emits an empty vote.
    assert store.block_states[head_root].current_height == next_height - 1
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)


@with_simplex_and_later
@spec_state_test
def test_available_attestation_construction_and_duplicate_seats(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    head_root = store.finalized_checkpoint.root

    # Same-slot votes cannot claim a payload, regardless of the supplied node status.
    same_slot_head = spec.ForkChoiceNode(
        root=head_root,
        payload_status=spec.PAYLOAD_STATUS_FULL,
    )
    same_slot_data = spec.get_available_attestation_data(store, same_slot_head)
    assert not same_slot_data.payload_present

    duty_slot = spec.Slot(spec.get_current_slot(store) + 1)
    store.time = spec.uint64(store.genesis_time + duty_slot * spec.config.SLOT_DURATION_MS // 1000)
    older_full_head = spec.ForkChoiceNode(
        root=head_root,
        payload_status=spec.PAYLOAD_STATUS_FULL,
    )
    data = spec.get_available_attestation_data(store, older_full_head)
    assert data.payload_present

    head_state = spec.get_current_slot_state(store, head_root)
    committee = list(spec.get_available_committee(head_state, duty_slot))
    validator_index = max(set(committee), key=committee.count)
    expected_positions = [i for i, index in enumerate(committee) if index == validator_index]
    assert len(expected_positions) > 1

    aggregation_bits = spec.get_available_attestation_aggregation_bits(
        head_state, duty_slot, validator_index
    )
    actual_positions = [i for i, bit in enumerate(aggregation_bits) if bit]
    assert actual_positions == expected_positions

    signature = spec.get_available_attestation_signature(
        head_state, data, privkeys[validator_index]
    )
    domain = spec.get_domain(
        head_state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(duty_slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    assert bls.Verify(head_state.validators[validator_index].pubkey, signing_root, signature)

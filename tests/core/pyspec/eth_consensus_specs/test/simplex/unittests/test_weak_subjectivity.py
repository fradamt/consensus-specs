from eth_consensus_specs.test.context import (
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.block import (
    build_empty_block,
    build_empty_block_for_next_slot,
)
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block


def _get_exact_checkpoint(spec, signed_block):
    return spec.Checkpoint(
        slot=signed_block.message.slot,
        root=signed_block.message.hash_tree_root(),
    )


def _set_store_epoch(spec, store, epoch):
    slot = spec.compute_start_slot_at_epoch(epoch)
    store.time = spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_accepts_exact_non_boundary_checkpoint(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)

    checkpoint = _get_exact_checkpoint(spec, signed_block)
    assert checkpoint.slot % spec.SLOTS_PER_EPOCH != 0
    assert spec.is_within_weak_subjectivity_period(store, state, checkpoint)


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_rejects_wrong_checkpoint_root(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    checkpoint = _get_exact_checkpoint(spec, signed_block)
    checkpoint.root = spec.Root(b"\xff" * 32)
    assert checkpoint.root != signed_block.message.hash_tree_root()

    expect_assertion_error(
        lambda: spec.is_within_weak_subjectivity_period(store, state, checkpoint)
    )


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_rejects_state_checkpoint_slot_mismatch(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    checkpoint = _get_exact_checkpoint(spec, signed_block)
    checkpoint.slot = spec.Slot(checkpoint.slot + 1)
    assert state.slot != checkpoint.slot

    expect_assertion_error(
        lambda: spec.is_within_weak_subjectivity_period(store, state, checkpoint)
    )


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_rejects_header_checkpoint_slot_mismatch(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    block = build_empty_block_for_next_slot(spec, state)
    signed_block = state_transition_and_sign_block(spec, state, block)
    checkpoint = _get_exact_checkpoint(spec, signed_block)
    state.latest_block_header.slot = spec.Slot(checkpoint.slot - 1)
    assert state.slot == checkpoint.slot
    assert state.latest_block_header.slot != checkpoint.slot

    expect_assertion_error(
        lambda: spec.is_within_weak_subjectivity_period(store, state, checkpoint)
    )


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_last_epoch_slot_and_inclusive_period_boundary(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    checkpoint_slot = spec.Slot(spec.SLOTS_PER_EPOCH - 1)
    block = build_empty_block(spec, state, slot=checkpoint_slot)
    signed_block = state_transition_and_sign_block(spec, state, block)
    checkpoint = _get_exact_checkpoint(spec, signed_block)

    assert checkpoint.slot % spec.SLOTS_PER_EPOCH == spec.SLOTS_PER_EPOCH - 1
    assert state.latest_block_header.slot == checkpoint.slot
    assert spec.is_within_weak_subjectivity_period(store, state, checkpoint)

    checkpoint_epoch = spec.compute_epoch_at_slot(checkpoint.slot)
    ws_period = spec.compute_weak_subjectivity_period(state)
    last_valid_epoch = spec.Epoch(checkpoint_epoch + ws_period)
    _set_store_epoch(spec, store, last_valid_epoch)
    assert spec.is_within_weak_subjectivity_period(store, state, checkpoint)

    _set_store_epoch(spec, store, spec.Epoch(last_valid_epoch + 1))
    assert not spec.is_within_weak_subjectivity_period(store, state, checkpoint)


@with_simplex_and_later
@spec_state_test
def test_weak_subjectivity_rejects_height_zero_boundary_sentinel(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    ws_state = state.copy()

    activation_epoch = spec.Epoch(spec.GENESIS_EPOCH + 1)
    activation_slot = spec.compute_start_slot_at_epoch(activation_epoch)
    ws_state.slot = activation_slot
    ws_state.fork.epoch = activation_epoch

    # Model an empty activation boundary carrying the root of an older block.
    ws_state.latest_block_header.state_root = spec.Root(b"\x42" * 32)
    assert ws_state.latest_block_header.slot < activation_slot
    sentinel = spec.Checkpoint(
        slot=activation_slot,
        root=ws_state.latest_block_header.hash_tree_root(),
    )
    ws_state.finalized_height = spec.Height(0)
    ws_state.finalized_checkpoint = sentinel

    expect_assertion_error(
        lambda: spec.is_within_weak_subjectivity_period(store, ws_state, sentinel)
    )

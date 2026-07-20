from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_presets,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.constants import MAINNET
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _setup_payload_vote_store(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    parent_root = store.finalized_checkpoint.root
    slot = spec.Slot(state.slot + 1)
    block = spec.BeaconBlock(slot=slot, parent_root=parent_root)
    root = block.hash_tree_root()
    block_state = state.copy()
    block_state.slot = slot

    store.blocks[root] = block
    store.block_states[root] = block_state
    store.payloads[root] = spec.ExecutionPayloadEnvelope(beacon_block_root=root)
    store.payload_votes[slot] = {}
    store.payload_vote_equivocations[slot] = set()
    return store, root, block_state


def _set_ptc(spec, state, slot, committee):
    assert len(committee) == spec.PTC_SIZE
    epoch = spec.compute_epoch_at_slot(slot)
    state_epoch = spec.get_current_epoch(state)
    if epoch < state_epoch:
        offset = 0
    else:
        offset = (epoch - state_epoch + 1) * spec.SLOTS_PER_EPOCH
    window_index = offset + slot % spec.SLOTS_PER_EPOCH
    for seat, validator_index in enumerate(committee):
        state.ptc_window[window_index][seat] = validator_index


def _payload_vote(spec, root, slot, *, supported):
    return spec.PayloadAttestationData(
        beacon_block_root=root,
        slot=slot,
        payload_present=supported,
        blob_data_available=supported,
    )


def _signed_payload_vote_message(spec, state, root, slot, validator_index):
    data = _payload_vote(spec, root, slot, supported=True)
    domain = spec.get_domain(
        state,
        spec.DOMAIN_PTC_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    return spec.PayloadAttestationMessage(
        validator_index=validator_index,
        data=data,
        signature=bls.Sign(privkeys[validator_index], signing_root),
    )


def _set_store_time_in_slot(spec, store, slot, offset_ms):
    assert offset_ms % 1000 == 0
    store.time = spec.uint64(
        store.genesis_time + (slot * spec.config.SLOT_DURATION_MS + offset_ms) // 1000
    )


@with_simplex_and_later
@spec_state_test
def test_payload_vote_strict_majority_boundary(spec, state):
    store, root, block_state = _setup_payload_vote_store(spec, state)
    slot = store.blocks[root].slot
    supporter = spec.ValidatorIndex(0)
    opponent = spec.ValidatorIndex(1)
    half = spec.PTC_SIZE // 2
    _set_ptc(
        spec,
        block_state,
        slot,
        [supporter] * half + [opponent] * (spec.PTC_SIZE - half),
    )
    store.payload_votes[slot] = {
        supporter: _payload_vote(spec, root, slot, supported=True),
        opponent: _payload_vote(spec, root, slot, supported=False),
    }

    assert spec.get_payload_participant_count(store, root) == spec.PTC_SIZE
    assert spec.get_payload_full_support(store, root) == half
    assert spec.get_payload_data_available_support(store, root) == half
    assert not spec.is_payload_timely(store, root)
    assert not spec.is_payload_data_available(store, root)
    assert not spec.should_extend_payload(store, root)

    # Move one seat to the supporting identity: half + 1 is a strict majority.
    _set_ptc(
        spec,
        block_state,
        slot,
        [supporter] * (half + 1) + [opponent] * (spec.PTC_SIZE - half - 1),
    )
    assert spec.get_payload_full_support(store, root) == half + 1
    assert spec.get_payload_data_available_support(store, root) == half + 1
    assert spec.is_payload_timely(store, root)
    assert spec.is_payload_data_available(store, root)
    assert spec.should_extend_payload(store, root)


@with_simplex_and_later
@spec_state_test
def test_single_identity_vote_counts_all_duplicate_ptc_seats(spec, state):
    store, root, block_state = _setup_payload_vote_store(spec, state)
    slot = store.blocks[root].slot
    voter = spec.ValidatorIndex(0)
    _set_ptc(spec, block_state, slot, [voter] * spec.PTC_SIZE)
    store.payload_votes[slot] = {
        voter: _payload_vote(spec, root, slot, supported=True),
    }

    assert len(store.payload_votes[slot]) == 1
    assert spec.get_payload_participant_count(store, root) == spec.PTC_SIZE
    assert spec.get_payload_full_support(store, root) == spec.PTC_SIZE
    assert spec.get_payload_data_available_support(store, root) == spec.PTC_SIZE
    assert spec.should_extend_payload(store, root)


@with_simplex_and_later
@spec_state_test
def test_equivocating_identity_credits_all_duplicate_seats_to_payload_decision(spec, state):
    store, root, block_state = _setup_payload_vote_store(spec, state)
    slot = store.blocks[root].slot
    equivocator = spec.ValidatorIndex(0)
    opponent = spec.ValidatorIndex(1)
    equivocation_seats = spec.PTC_SIZE // 2 + 1
    _set_ptc(
        spec,
        block_state,
        slot,
        [equivocator] * equivocation_seats + [opponent] * (spec.PTC_SIZE - equivocation_seats),
    )
    store.payload_votes[slot] = {
        equivocator: _payload_vote(spec, root, slot, supported=False),
        opponent: _payload_vote(spec, root, slot, supported=False),
    }

    assert spec.get_payload_full_support(store, root) == 0
    assert spec.get_payload_data_available_support(store, root) == 0
    assert not spec.should_extend_payload(store, root)

    # The first vote stays identity-keyed. The equivocation marker records the
    # conflicting second vote and gives viability credit to every seat held by
    # that identity for both compatible positive decisions.
    store.payload_vote_equivocations[slot].add(equivocator)
    assert spec.get_payload_participant_count(store, root) == spec.PTC_SIZE
    assert spec.get_payload_full_support(store, root) == equivocation_seats
    assert spec.get_payload_data_available_support(store, root) == equivocation_seats
    assert spec.is_payload_timely(store, root)
    assert spec.is_payload_data_available(store, root)
    assert spec.should_extend_payload(store, root)


@with_simplex_and_later
@with_presets(
    [MAINNET],
    reason="the actual 75% view-freeze point must be representable by Store.time seconds",
)
@spec_state_test
@always_bls
def test_payload_vote_view_freeze_equality_and_next_proposer_override(spec, state):
    store, root, block_state = _setup_payload_vote_store(spec, state)
    slot = store.blocks[root].slot
    validator_index = spec.ValidatorIndex(0)
    _set_ptc(spec, block_state, slot, [validator_index] * spec.PTC_SIZE)
    message = _signed_payload_vote_message(
        spec,
        block_state,
        root,
        slot,
        validator_index,
    )

    freeze_due_ms = spec.get_view_freeze_due_ms()
    assert freeze_due_ms % 1000 == 0
    assert freeze_due_ms + 1000 < spec.config.SLOT_DURATION_MS

    # The wire rule is strictly before the deadline: equality is excluded.
    _set_store_time_in_slot(spec, store, slot, freeze_due_ms)
    assert spec.get_current_slot(store) == slot
    assert not spec.is_before_view_freeze_deadline(store)
    spec.on_payload_attestation_message(store, message)
    assert store.payload_votes[slot] == {}

    # A strictly later ordinary wire delivery remains excluded.
    _set_store_time_in_slot(spec, store, slot, freeze_due_ms + 1000)
    assert not spec.is_before_view_freeze_deadline(store)
    spec.on_payload_attestation_message(store, message)
    assert store.payload_votes[slot] == {}

    # The next proposer may collect the same valid vote after the freeze.
    spec.on_payload_attestation_message(store, message, is_next_proposer=True)
    assert store.payload_votes[slot] == {validator_index: message.data}

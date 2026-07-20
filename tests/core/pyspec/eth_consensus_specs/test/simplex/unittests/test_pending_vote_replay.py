from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import get_valid_attestation
from eth_consensus_specs.test.helpers.block import build_empty_block_for_next_slot
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block
from eth_consensus_specs.utils import bls


def _time_at_slot(spec, store, slot):
    return spec.uint64(store.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000)


def _set_store_slot(spec, store, slot):
    store.time = _time_at_slot(spec, store, slot)


def _build_signed_child(spec, state, marker):
    block_state = state.copy()
    block = build_empty_block_for_next_slot(spec, block_state)
    block.body.graffiti = bytes([marker]) * 32
    signed_block = state_transition_and_sign_block(spec, block_state, block)
    return signed_block, block_state


def _build_available_attestation(spec, state, slot, root, validator_index):
    committee = spec.get_available_committee(state, slot)
    data = spec.AvailableAttestationData(
        slot=slot,
        beacon_block_root=root,
        payload_present=False,
    )
    attestation = spec.AvailableAttestation(data=data)
    for position, member_index in enumerate(committee):
        if member_index == validator_index:
            attestation.aggregation_bits[position] = True

    domain = spec.get_domain(
        state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Sign(privkeys[validator_index], signing_root)
    return attestation


@with_simplex_and_later
@spec_state_test
@always_bls
def test_pending_finality_vote_replays_once_and_invalid_signature_is_skipped(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    signed_block, block_state = _build_signed_child(spec, state, marker=0xA1)
    block = signed_block.message
    block_root = block.hash_tree_root()
    assert block_root not in store.blocks

    committee = list(spec.get_beacon_committee(block_state, block.slot, spec.CommitteeIndex(0)))
    valid_index, invalid_index = list(dict.fromkeys(committee))[:2]
    valid_vote = get_valid_attestation(
        spec,
        block_state,
        slot=block.slot,
        beacon_block_root=block_root,
        signed=True,
        filter_participant_set=lambda _participants: {valid_index},
    )
    invalid_vote = get_valid_attestation(
        spec,
        block_state,
        slot=block.slot,
        beacon_block_root=block_root,
        signed=True,
        filter_participant_set=lambda _participants: {invalid_index},
    )
    invalid_vote.signature = spec.BLSSignature()

    # A block-carried vote whose sibling head is not known locally is retained
    # without attempting branch-relative validation yet.
    spec.on_attestation(store, valid_vote, is_from_block=True)
    spec.on_attestation(store, invalid_vote, is_from_block=True)
    assert store.pending_attestations[block_root] == [valid_vote, invalid_vote]
    assert valid_index not in store.latest_messages
    assert invalid_index not in store.latest_messages

    # Included finality votes affect only subsequent slots. Deliver the named
    # sibling block one slot later and count the pending replay dispatches.
    spec.on_tick(store, _time_at_slot(spec, store, spec.Slot(block.slot + 1)))
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    replayed = []
    original_on_attestation = spec.on_attestation

    def record_replay(*args, **kwargs):
        replayed.append(args[1].hash_tree_root())
        return original_on_attestation(*args, **kwargs)

    spec.on_attestation = record_replay
    try:
        # The bad replay is skip-only: it does not reject an otherwise valid
        # newly arrived block.
        spec.on_block(store, signed_block)
    finally:
        spec.on_attestation = original_on_attestation

    assert replayed.count(valid_vote.hash_tree_root()) == 1
    assert replayed.count(invalid_vote.hash_tree_root()) == 1
    assert block_root not in store.pending_attestations
    assert store.latest_messages[valid_index] == spec.LatestMessage(
        slot=block.slot,
        root=block_root,
    )
    assert invalid_index not in store.latest_messages


@with_simplex_and_later
@spec_state_test
@always_bls
def test_pending_available_vote_replays_once_and_invalid_signature_is_skipped(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    signed_block, block_state = _build_signed_child(spec, state, marker=0xB2)
    block = signed_block.message
    block_root = block.hash_tree_root()
    assert block_root not in store.blocks

    committee = list(dict.fromkeys(spec.get_available_committee(block_state, block.slot)))
    valid_index, invalid_index = committee[:2]
    valid_vote = _build_available_attestation(
        spec,
        block_state,
        block.slot,
        block_root,
        valid_index,
    )
    invalid_vote = _build_available_attestation(
        spec,
        block_state,
        block.slot,
        block_root,
        invalid_index,
    )
    invalid_vote.signature = spec.BLSSignature()

    spec.on_available_attestation(store, valid_vote, is_from_block=True)
    spec.on_available_attestation(store, invalid_vote, is_from_block=True)
    assert store.pending_available_attestations[block_root] == [valid_vote, invalid_vote]

    # Equality is the last live replay slot for an unresolved available vote.
    replay_slot = spec.Slot(block.slot + 1)
    spec.on_tick(store, _time_at_slot(spec, store, replay_slot))
    assert block_root in store.pending_available_attestations
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    replayed = []
    original_on_available_attestation = spec.on_available_attestation

    def record_replay(*args, **kwargs):
        replayed.append(args[1].hash_tree_root())
        return original_on_available_attestation(*args, **kwargs)

    spec.on_available_attestation = record_replay
    try:
        # The invalid aggregate has no vote effect and cannot reject the block
        # that makes its named head known.
        spec.on_block(store, signed_block)
    finally:
        spec.on_available_attestation = original_on_available_attestation

    assert replayed.count(valid_vote.hash_tree_root()) == 1
    assert replayed.count(invalid_vote.hash_tree_root()) == 1
    assert block_root not in store.pending_available_attestations
    assert store.available_votes[block.slot][valid_index] == valid_vote.data
    assert invalid_index not in store.available_votes[block.slot]


@with_simplex_and_later
@spec_state_test
def test_pending_finality_vote_strict_expiry_boundary(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    unknown_root = spec.Root(b"\xcc" * 32)
    vote = spec.Attestation(
        data=spec.AttestationData(
            slot=spec.GENESIS_SLOT,
            beacon_block_root=unknown_root,
            target=spec.Checkpoint(),
            height=spec.Height(0),
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        )
    )
    spec.on_attestation(store, vote, is_from_block=True)

    last_live_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS - 1)
    # Exercise only the two pruning boundaries; the committee cache does not
    # require every intermediate tick for this pending-map rule.
    _set_store_slot(spec, store, spec.Slot(last_live_slot - 1))
    spec.on_tick_per_slot(store, _time_at_slot(spec, store, last_live_slot))
    assert store.pending_attestations[unknown_root] == [vote]

    expiry_slot = spec.Slot(spec.LATEST_MESSAGE_EXPIRY_SLOTS)
    spec.on_tick_per_slot(store, _time_at_slot(spec, store, expiry_slot))
    assert unknown_root not in store.pending_attestations


@with_simplex_and_later
@spec_state_test
def test_pending_available_vote_inclusive_last_slot_and_strictly_after_expiry(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    unknown_root = spec.Root(b"\xdd" * 32)
    vote = spec.AvailableAttestation(
        data=spec.AvailableAttestationData(
            slot=spec.GENESIS_SLOT,
            beacon_block_root=unknown_root,
            payload_present=False,
        )
    )
    spec.on_available_attestation(store, vote, is_from_block=True)

    last_live_slot = spec.Slot(vote.data.slot + 1)
    spec.on_tick_per_slot(store, _time_at_slot(spec, store, last_live_slot))
    assert store.pending_available_attestations[unknown_root] == [vote]

    first_expired_slot = spec.Slot(last_live_slot + 1)
    spec.on_tick_per_slot(store, _time_at_slot(spec, store, first_expired_slot))
    assert unknown_root not in store.pending_available_attestations

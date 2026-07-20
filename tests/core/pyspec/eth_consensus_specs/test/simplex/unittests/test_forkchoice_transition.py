from eth_consensus_specs.test.context import (
    always_bls,
    spec_test,
    with_phases,
    with_state,
)
from eth_consensus_specs.test.helpers.block import build_empty_block, sign_block
from eth_consensus_specs.test.helpers.constants import GLOAS, SIMPLEX
from eth_consensus_specs.test.helpers.execution_payload import (
    build_signed_execution_payload_envelope,
)
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store_and_block,
)
from eth_consensus_specs.test.helpers.gossip import get_seen
from eth_consensus_specs.test.helpers.payload_attestation import (
    prepare_signed_payload_attestation,
)
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block


def _add_block_at_slot(spec, store, state, slot):
    block = build_empty_block(spec, state, slot=slot)
    signed_block = state_transition_and_sign_block(spec, state, block)
    block_time = state.genesis_time + slot * spec.config.SLOT_DURATION_MS // 1000
    spec.on_tick(store, block_time)
    spec.on_block(store, signed_block)
    return signed_block.message.hash_tree_root(), signed_block


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_live_gloas_store_migrates_and_accepts_first_simplex_block(spec, phases, state):
    simplex = phases[SIMPLEX]
    store, _ = get_genesis_forkchoice_store_and_block(spec, state)

    # Sparse proposals make both legacy checkpoint roots differ from their
    # epoch-boundary slots. This makes the migration's exact-slot recovery
    # observable instead of merely checking genesis defaults.
    finalized_root, finalized_block = _add_block_at_slot(spec, store, state, spec.Slot(3))
    justified_root, justified_block = _add_block_at_slot(spec, store, state, spec.Slot(9))
    store.finalized_checkpoint = spec.Checkpoint(epoch=spec.Epoch(1), root=finalized_root)
    store.justified_checkpoint = spec.Checkpoint(epoch=spec.Epoch(2), root=justified_root)
    assert store.blocks[finalized_root].slot != spec.compute_start_slot_at_epoch(spec.Epoch(1))
    assert store.blocks[justified_root].slot != spec.compute_start_slot_at_epoch(spec.Epoch(2))

    # Populate each kind of live Gloas data that the migration promises to
    # retain. The payload goes through the real verification handler.
    voter = spec.ValidatorIndex(0)
    equivocator = spec.ValidatorIndex(1)
    store.latest_messages[voter] = spec.LatestMessage(
        slot=spec.Slot(9),
        root=justified_root,
        payload_present=True,
    )
    store.equivocating_indices.add(equivocator)
    envelope = build_signed_execution_payload_envelope(
        spec,
        store.block_states[finalized_root],
        finalized_root,
        finalized_block,
    )
    spec.on_execution_payload_envelope(store, envelope)
    assert store.payloads[finalized_root] == envelope.message
    late_envelope = build_signed_execution_payload_envelope(
        spec,
        store.block_states[justified_root],
        justified_root,
        justified_block,
    )
    assert justified_root not in store.payloads

    # Make legacy-only caches observably non-empty. They must not be projected
    # into any of the new per-slot or per-round Simplex caches.
    store.payload_timeliness_vote[justified_root][0] = True
    store.payload_data_availability_vote[justified_root][0] = True
    assert store.checkpoint_states
    assert store.unrealized_justifications

    fork_epoch = simplex.Epoch(3)
    activation_slot = simplex.compute_start_slot_at_epoch(fork_epoch)
    activation_time = simplex.uint64(
        store.genesis_time + activation_slot * simplex.config.SLOT_DURATION_MS // 1000
    )
    original_simplex_config = simplex.config
    try:
        simplex.config = simplex.config._replace(SIMPLEX_FORK_EPOCH=fork_epoch)
        spec.on_tick(store, activation_time)
        assert spec.get_current_slot(store) == activation_slot

        pre_roots = set(store.blocks)
        migrated = simplex.upgrade_forkchoice_store_to_simplex(store)

        assert set(migrated.blocks) == pre_roots
        assert set(migrated.block_states) == pre_roots
        assert migrated.justified_checkpoint == simplex.Checkpoint(
            slot=simplex.Slot(9),
            root=justified_root,
        )
        assert migrated.finalized_checkpoint == simplex.Checkpoint(
            slot=simplex.Slot(3),
            root=finalized_root,
        )
        assert migrated.justified_height == simplex.Height(0)
        assert migrated.h_max == simplex.GENESIS_HEIGHT

        for post_state in migrated.block_states.values():
            assert type(post_state) is simplex.BeaconState
            assert post_state.slot == activation_slot
            assert post_state.fork.previous_version == spec.config.GLOAS_FORK_VERSION
            assert post_state.fork.current_version == simplex.config.SIMPLEX_FORK_VERSION
            assert post_state.fork.epoch == fork_epoch
            assert post_state.current_height == simplex.GENESIS_HEIGHT

        assert migrated.latest_messages[simplex.ValidatorIndex(voter)] == simplex.LatestMessage(
            slot=simplex.Slot(9),
            root=justified_root,
        )
        assert not hasattr(
            migrated.latest_messages[simplex.ValidatorIndex(voter)], "payload_present"
        )
        assert migrated.equivocating_indices == {simplex.ValidatorIndex(equivocator)}
        assert migrated.payloads[finalized_root] == envelope.message
        assert migrated.payloads[finalized_root] is not store.payloads[finalized_root]
        assert finalized_root not in migrated.legacy_payload_verification_states
        assert justified_root in migrated.legacy_payload_verification_states
        legacy_verification_state = migrated.legacy_payload_verification_states[justified_root]
        assert type(legacy_verification_state) is spec.BeaconState
        assert legacy_verification_state == store.block_states[justified_root]
        assert legacy_verification_state is not store.block_states[justified_root]
        assert legacy_verification_state.slot == spec.Slot(9)
        assert migrated.block_states[justified_root].slot == activation_slot

        # A Gloas envelope first delivered after the activation tick must still
        # verify against its exact historical immediate post-block state. The
        # upgraded activation-slot branch state cannot satisfy that check.
        simplex.on_execution_payload_envelope(migrated, late_envelope)
        assert migrated.payloads[justified_root] == late_envelope.message
        assert justified_root not in migrated.legacy_payload_verification_states

        assert migrated.checkpoint_states == {}
        assert migrated.round_attestations == {}
        assert migrated.round_equivocating_indices == {}
        assert migrated.pending_attestations == {}
        assert migrated.pending_available_attestations == {}
        legacy_vote_slot = simplex.Slot(activation_slot - 1)
        assert migrated.payload_votes == {legacy_vote_slot: {}, activation_slot: {}}
        assert migrated.payload_vote_equivocations == {
            legacy_vote_slot: set(),
            activation_slot: set(),
        }
        assert migrated.available_votes == {activation_slot: {}}
        assert migrated.available_vote_equivocations == {activation_slot: set()}
        assert migrated.available_timely_attesters == {activation_slot: set()}
        assert migrated.available_timely_equivocations == {activation_slot: set()}
        assert set(migrated.available_committees) == {activation_slot}
        assert (
            len(migrated.available_committees[activation_slot]) == simplex.AVAILABLE_COMMITTEE_SIZE
        )
        assert migrated.frozen_available_votes == {}
        assert migrated.stable_root == simplex.Root()
        assert migrated.stable_root_payload_status == simplex.PAYLOAD_STATUS_PENDING
        assert migrated.stable_root_proposal_root == simplex.Root()
        assert migrated.pointer_candidates == {}
        assert migrated.stable_root_decisions == {}
        assert migrated.stable_root_round == simplex.compute_round_at_slot(
            simplex.Slot(activation_slot - 1)
        )
        for confirmed_head in (
            migrated.latest_confirmed_head,
            migrated.live_confirmed_head,
            migrated.fast_confirmed_head,
        ):
            assert confirmed_head == (finalized_root, activation_slot)

        # Build the first Simplex proposal from the already slot-processed
        # migrated parent. It intentionally claims the verified parent FULL,
        # proving that the retained envelope is usable by the boundary path.
        boundary_pre_state = migrated.block_states[justified_root].copy()
        boundary_block = build_empty_block(
            simplex,
            boundary_pre_state,
            slot=activation_slot,
        )
        boundary_block.body.anchor_root = justified_root
        assert boundary_block.parent_root == justified_root
        boundary_block.body.signed_execution_payload_bid.message.parent_block_hash = (
            migrated.blocks[justified_root].body.signed_execution_payload_bid.message.block_hash
        )
        assert simplex.is_parent_node_full(migrated, boundary_block)
        assert simplex.is_payload_verified(migrated, justified_root)

        boundary_post_state = boundary_pre_state.copy()
        simplex.process_block(boundary_post_state, boundary_block)
        boundary_block.state_root = boundary_post_state.hash_tree_root()
        signed_boundary_block = sign_block(simplex, boundary_post_state, boundary_block)

        # The gossip path must recognize the same already slot-processed live
        # migration boundary as ``on_block``.
        simplex.validate_beacon_block_gossip(
            get_seen(simplex),
            migrated,
            boundary_pre_state,
            signed_boundary_block,
            simplex.compute_time_at_slot_ms(boundary_pre_state, activation_slot),
        )

        simplex.on_block(migrated, signed_boundary_block)
        boundary_root = boundary_block.hash_tree_root()
        assert migrated.blocks[boundary_root] == boundary_block
        assert migrated.block_states[boundary_root] == boundary_post_state
        assert migrated.block_states[boundary_root].slot == activation_slot
        assert migrated.justified_checkpoint.slot == simplex.Slot(9)
        assert migrated.finalized_checkpoint.slot == simplex.Slot(3)
        boundary_round = simplex.compute_round_at_slot(activation_slot)
        assert migrated.pointer_candidates == {
            boundary_round: {justified_root: {boundary_root}}
        }
        # Proposal receipt records the raw pointer candidate but does not select
        # the stable root before the common round-selection action.
        assert migrated.stable_root == simplex.Root()
        assert migrated.stable_root_proposal_root == simplex.Root()
        assert migrated.stable_root_decisions == {}
    finally:
        simplex.config = original_simplex_config


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_live_migration_projects_last_gloas_ptc_decision(spec, phases, state):
    simplex = phases[SIMPLEX]
    store, _ = get_genesis_forkchoice_store_and_block(spec, state)
    fork_epoch = simplex.Epoch(3)
    activation_slot = simplex.compute_start_slot_at_epoch(fork_epoch)
    legacy_vote_slot = simplex.Slot(activation_slot - 1)

    parent_root, signed_parent = _add_block_at_slot(
        spec,
        store,
        state,
        spec.Slot(legacy_vote_slot),
    )
    envelope = build_signed_execution_payload_envelope(
        spec,
        store.block_states[parent_root],
        parent_root,
        signed_parent,
    )
    spec.on_execution_payload_envelope(store, envelope)

    # Gloas stores the vote once per PTC seat. Populate a unanimous live view;
    # duplicate seats must collapse to one identity entry during migration and
    # expand back to their original weight when Simplex reads the decision.
    store.payload_timeliness_vote[parent_root] = [True] * spec.PTC_SIZE
    store.payload_data_availability_vote[parent_root] = [True] * spec.PTC_SIZE
    legacy_ptc = spec.get_ptc(store.block_states[parent_root], spec.Slot(legacy_vote_slot))

    activation_time = simplex.uint64(
        store.genesis_time + activation_slot * simplex.config.SLOT_DURATION_MS // 1000
    )
    original_simplex_config = simplex.config
    try:
        simplex.config = simplex.config._replace(SIMPLEX_FORK_EPOCH=fork_epoch)
        spec.on_tick(store, activation_time)
        migrated = simplex.upgrade_forkchoice_store_to_simplex(store)

        assert set(migrated.payload_votes[legacy_vote_slot]) == set(legacy_ptc)
        for vote in migrated.payload_votes[legacy_vote_slot].values():
            assert vote == simplex.PayloadAttestationData(
                beacon_block_root=parent_root,
                slot=legacy_vote_slot,
                payload_present=True,
                blob_data_available=True,
            )
        assert migrated.payload_vote_equivocations[legacy_vote_slot] == set()
        assert migrated.payload_votes[activation_slot] == {}
        assert simplex.get_payload_participant_count(migrated, parent_root) == simplex.PTC_SIZE
        assert simplex.get_payload_full_support(migrated, parent_root) == simplex.PTC_SIZE
        assert simplex.get_payload_data_available_support(migrated, parent_root) == simplex.PTC_SIZE
        assert simplex.should_extend_payload(migrated, parent_root)
        assert simplex.get_payload_status_tiebreaker(
            migrated,
            simplex.ForkChoiceNode(
                root=parent_root,
                payload_status=simplex.PAYLOAD_STATUS_FULL,
            ),
        ) > simplex.get_payload_status_tiebreaker(
            migrated,
            simplex.ForkChoiceNode(
                root=parent_root,
                payload_status=simplex.PAYLOAD_STATUS_EMPTY,
            ),
        )
    finally:
        simplex.config = original_simplex_config


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@always_bls
def test_activation_block_projects_included_last_gloas_ptc_vote(spec, phases, state):
    simplex = phases[SIMPLEX]
    store, _ = get_genesis_forkchoice_store_and_block(spec, state)
    fork_epoch = simplex.Epoch(3)
    activation_slot = simplex.compute_start_slot_at_epoch(fork_epoch)
    legacy_vote_slot = simplex.Slot(activation_slot - 1)
    parent_root, _ = _add_block_at_slot(spec, store, state, legacy_vote_slot)

    activation_time = simplex.uint64(
        store.genesis_time + activation_slot * simplex.config.SLOT_DURATION_MS // 1000
    )
    original_simplex_config = simplex.config
    try:
        simplex.config = simplex.config._replace(SIMPLEX_FORK_EPOCH=fork_epoch)
        spec.on_tick(store, activation_time)
        migrated = simplex.upgrade_forkchoice_store_to_simplex(store)
        assert migrated.payload_votes[legacy_vote_slot] == {}

        boundary_pre_state = migrated.block_states[parent_root].copy()
        assert boundary_pre_state.slot == activation_slot
        legacy_ptc = simplex.get_ptc(boundary_pre_state, legacy_vote_slot)
        voters = list(dict.fromkeys(legacy_ptc))[:3]
        assert len(voters) == 3
        aggregate = prepare_signed_payload_attestation(
            simplex,
            boundary_pre_state,
            slot=legacy_vote_slot,
            beacon_block_root=parent_root,
            payload_present=True,
            blob_data_available=True,
            attesting_indices=voters,
        )

        boundary_block = build_empty_block(
            simplex,
            boundary_pre_state,
            slot=activation_slot,
        )
        boundary_block.body.payload_attestations.append(aggregate)
        boundary_post_state = boundary_pre_state.copy()
        simplex.process_block(boundary_post_state, boundary_block)
        boundary_block.state_root = boundary_post_state.hash_tree_root()
        signed_boundary_block = sign_block(simplex, boundary_post_state, boundary_block)

        simplex.on_block(migrated, signed_boundary_block)

        assert set(migrated.payload_votes[legacy_vote_slot]) == set(voters)
        assert all(
            vote == aggregate.data for vote in migrated.payload_votes[legacy_vote_slot].values()
        )
        assert migrated.payload_vote_equivocations[legacy_vote_slot] == set()
    finally:
        simplex.config = original_simplex_config

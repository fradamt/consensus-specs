from eth_consensus_specs.test.context import (
    always_bls,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.block import build_empty_block, sign_block
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.gossip import get_seen
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _add_child(spec, store, state, parent_root, slot):
    block = spec.BeaconBlock(slot=slot, parent_root=parent_root)
    root = block.hash_tree_root()
    store.blocks[root] = block
    store.block_states[root] = state
    return root


@with_simplex_and_later
@spec_state_test
def test_legacy_finalized_checkpoint_is_accepted_on_descendant_chain(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root

    # Model an empty activation-boundary slot: the inherited FFG checkpoint
    # names the boundary slot but carries the root of an earlier proposal.
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    store.finalized_checkpoint = spec.Checkpoint(slot=boundary_slot, root=anchor_root)
    head_slot = spec.Slot(boundary_slot + 1)
    head_root = _add_child(spec, store, state, anchor_root, head_slot)

    assert store.blocks[anchor_root].slot < store.finalized_checkpoint.slot
    assert spec.is_checkpoint_on_store_chain(store, head_root, store.finalized_checkpoint)

    # The common attestation-gossip validator must accept a valid descendant;
    # this covers the finalized-ancestry gate shared by finality gossip.
    data = spec.AttestationData(
        slot=head_slot,
        beacon_block_root=head_root,
        target=spec.Checkpoint(slot=head_slot, root=head_root),
        height=spec.GENESIS_HEIGHT,
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    spec.validate_attestation_data_gossip(store, state, data)


@with_simplex_and_later
@spec_state_test
def test_gossip_accepts_unknown_finality_target_and_bounds_its_slot(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    head_slot = spec.Slot(store.blocks[anchor_root].slot + 1)
    head_root = _add_child(spec, store, state, anchor_root, head_slot)

    # The piggyback is read only by exact equality with the including chain's
    # justified checkpoint, so an unretained root -- e.g. one below a
    # checkpoint-sync anchor -- must not be ignored: doing so would drop the
    # ``finality_participation`` update that finalizes it, permanently.
    unknown_root = spec.Root(b"\x51" * 32)
    assert unknown_root not in store.blocks
    data = spec.AttestationData(
        slot=head_slot,
        beacon_block_root=head_root,
        target=spec.Checkpoint(),
        height=spec.GENESIS_HEIGHT,
        finality_target=spec.Checkpoint(slot=head_slot, root=unknown_root),
        finality_height=spec.Height(0),
    )
    spec.validate_attestation_data_gossip(store, state, data)

    # The slot bound is the one rule the piggyback keeps.
    late_data = data.copy()
    late_data.finality_target = spec.Checkpoint(
        slot=spec.Slot(head_slot + 1),
        root=unknown_root,
    )
    try:
        spec.validate_attestation_data_gossip(store, state, late_data)
    except spec.GossipReject:
        pass
    else:
        raise AssertionError("finality target after the vote slot was not rejected")


@with_simplex_and_later
@spec_state_test
def test_finality_gossip_activation_boundary(spec, state):
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    state.slot = fork_slot

    pre_fork_data = spec.AttestationData(
        slot=spec.Slot(fork_slot - 1),
        beacon_block_root=anchor_root,
        target=spec.Checkpoint(),
        height=spec.Height(0),
        finality_target=spec.Checkpoint(),
        finality_height=spec.FAR_FUTURE_HEIGHT,
    )
    try:
        spec.validate_attestation_data_gossip(store, state, pre_fork_data)
    except spec.GossipReject:
        pass
    else:
        raise AssertionError("pre-Simplex finality gossip was not rejected")

    fork_data = pre_fork_data.copy()
    fork_data.slot = fork_slot
    spec.validate_attestation_data_gossip(store, state, fork_data)


@with_simplex_and_later
@spec_state_test
def test_status_v3_accepts_local_legacy_finalized_checkpoint(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    store.finalized_checkpoint = spec.Checkpoint(slot=boundary_slot, root=anchor_root)

    assert store.blocks[anchor_root].slot < boundary_slot
    assert spec.is_status_finalized_checkpoint_compatible(
        store,
        store.finalized_checkpoint.root,
        store.finalized_checkpoint.slot,
    )


@with_simplex_and_later
@spec_state_test
def test_status_v3_accepts_checkpoint_after_local_finality_advances(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    legacy_finalized = spec.Checkpoint(slot=boundary_slot, root=anchor_root)
    store.finalized_checkpoint = legacy_finalized

    advanced_slot = spec.Slot(boundary_slot + 1)
    advanced_root = _add_child(spec, store, state, anchor_root, advanced_slot)
    store.finalized_checkpoint = spec.Checkpoint(slot=advanced_slot, root=advanced_root)

    assert spec.is_status_finalized_checkpoint_compatible(
        store,
        store.finalized_checkpoint.root,
        store.finalized_checkpoint.slot,
    )
    # A peer still advertising the inherited height-0 sentinel remains on the
    # same finalized chain after local finality has advanced beyond it.
    assert spec.is_status_finalized_checkpoint_compatible(
        store,
        legacy_finalized.root,
        legacy_finalized.slot,
    )


@with_simplex_and_later
@spec_state_test
def test_non_store_legacy_checkpoint_remains_invalid(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    head_root = _add_child(spec, store, state, anchor_root, spec.Slot(boundary_slot + 1))
    carried_checkpoint = spec.Checkpoint(slot=boundary_slot, root=anchor_root)

    # Only the store-owned activation sentinel may use boundary-slot semantics;
    # arbitrary new checkpoints remain exact proposal-slot references.
    assert carried_checkpoint != store.finalized_checkpoint
    assert not spec.is_checkpoint_on_store_chain(store, head_root, carried_checkpoint)


@with_simplex_and_later
@spec_state_test
@always_bls
def test_block_gossip_accepts_descendant_of_legacy_finalized_checkpoint(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    store.finalized_checkpoint = spec.Checkpoint(slot=boundary_slot, root=anchor_root)

    block = build_empty_block(spec, state, spec.Slot(boundary_slot + 1))
    assert block.parent_root == anchor_root
    signed_block = sign_block(spec, state, block)
    store.payloads[anchor_root] = spec.ExecutionPayloadEnvelope(beacon_block_root=anchor_root)
    current_time_ms = spec.compute_time_at_slot_ms(state, block.slot)

    spec.validate_beacon_block_gossip(
        get_seen(spec),
        store,
        state,
        signed_block,
        current_time_ms,
    )


@with_simplex_and_later
@spec_state_test
@always_bls
def test_available_gossip_accepts_descendant_of_legacy_finalized_checkpoint(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    boundary_slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
    store.finalized_checkpoint = spec.Checkpoint(slot=boundary_slot, root=anchor_root)
    data_slot = spec.Slot(boundary_slot + 1)
    head_root = _add_child(spec, store, state, anchor_root, data_slot)
    store.time = spec.uint64(store.genesis_time + data_slot * spec.config.SLOT_DURATION_MS // 1000)

    target_state = state.copy()
    spec.process_slots(target_state, boundary_slot)
    committee = spec.get_available_committee(target_state, data_slot)
    validator_index = committee[0]
    data = spec.AvailableAttestationData(
        slot=data_slot,
        beacon_block_root=head_root,
        payload_present=False,
    )
    attestation = spec.AvailableAttestation(data=data)
    attestation.aggregation_bits[0] = True
    domain = spec.get_domain(
        target_state,
        spec.DOMAIN_AVAILABLE_ATTESTER,
        spec.compute_epoch_at_slot(data_slot),
    )
    signing_root = spec.compute_signing_root(data, domain)
    attestation.signature = bls.Sign(privkeys[validator_index], signing_root)
    current_time_ms = spec.compute_time_at_slot_ms(state, data_slot)

    spec.validate_available_attestation_gossip(
        get_seen(spec),
        store,
        state,
        attestation,
        current_time_ms,
    )

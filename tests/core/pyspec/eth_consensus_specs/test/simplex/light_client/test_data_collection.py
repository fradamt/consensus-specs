from eth_consensus_specs.test.context import (
    spec_state_test,
    with_presets,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.constants import MINIMAL
from eth_consensus_specs.test.helpers.light_client_data_collection import (
    add_new_block,
    finish_lc_data_collection_test,
    get_lc_bootstrap_block_id,
    get_light_client_bootstrap,
    get_light_client_finality_update,
    select_new_head,
    setup_lc_data_collection_test,
)

_FORK_EPOCH_CONFIGS = (
    "ALTAIR_FORK_EPOCH",
    "BELLATRIX_FORK_EPOCH",
    "CAPELLA_FORK_EPOCH",
    "DENEB_FORK_EPOCH",
    "ELECTRA_FORK_EPOCH",
    "FULU_FORK_EPOCH",
    "GLOAS_FORK_EPOCH",
    "SIMPLEX_FORK_EPOCH",
)


@with_simplex_and_later
@with_presets([MINIMAL], reason="builds competing branches across three Simplex rounds")
@spec_state_test
def test_simplex_light_client_data_collection_finalization(spec, state):
    """Finalize a branch and expose its bootstrap in the standard data format."""
    original_config = spec.config
    spec.config = spec.config._replace(**dict.fromkeys(_FORK_EPOCH_CONFIGS, spec.GENESIS_EPOCH))
    try:
        test = yield from setup_lc_data_collection_test(spec, state)

        # Import and select a competing pre-finality branch. Its cached data
        # becomes unreachable once branch A finalizes its epoch-boundary block.
        spec_b, _state_b, branch_b = yield from add_new_block(
            test,
            spec,
            state,
            slot=spec.SLOTS_PER_EPOCH - 1,
            num_sync_participants=1,
        )
        yield from select_new_head(test, spec_b, branch_b)
        assert branch_b in test.lc_data_store.cache.data

        # Build branch A through the epoch-1 boundary. Simplex startup suppresses
        # height outcomes through this point, but the genesis block remains the
        # exact target of the still-open startup height.
        spec_a = spec
        state_a = state
        for slot in range(1, spec.SLOTS_PER_EPOCH + 1):
            spec_a, state_a, bid_a = yield from add_new_block(
                test,
                spec_a,
                state_a,
                slot=slot,
                num_sync_participants=1,
            )
            yield from select_new_head(test, spec_a, bid_a)

        def add_startup_timeout(vote_spec, vote_state, data):
            data.target = vote_spec.Checkpoint()
            data.height = vote_state.current_height

        # Clear the startup height by timeout. The block that includes the last
        # committee's vote advances the height and becomes its exact target.
        for slot in range(spec.SLOTS_PER_EPOCH + 1, 2 * spec.SLOTS_PER_EPOCH + 1):
            spec_a, state_a, bid_a = yield from add_new_block(
                test,
                spec_a,
                state_a,
                slot=slot,
                num_sync_participants=1,
                attestation_data_modifier=add_startup_timeout,
            )
            yield from select_new_head(test, spec_a, bid_a)
        target_bid = bid_a
        target = spec.Checkpoint(slot=target_bid.slot, root=target_bid.root)
        assert get_light_client_bootstrap(test, target_bid.root).spec is None

        def add_justification_target(vote_spec, vote_state, data):
            data.target = vote_spec.Checkpoint(slot=target.slot, root=target.root)
            data.height = vote_state.current_height

        # Every following committee votes for the same exact block. The block at
        # the next boundary includes the final committee's vote before settling
        # the due height outcome, thereby justifying the target.
        for slot in range(2 * spec.SLOTS_PER_EPOCH + 1, 3 * spec.SLOTS_PER_EPOCH + 1):
            spec_a, state_a, bid_a = yield from add_new_block(
                test,
                spec_a,
                state_a,
                slot=slot,
                num_sync_participants=1,
                attestation_data_modifier=add_justification_target,
            )
            yield from select_new_head(test, spec_a, bid_a)
        assert state_a.justified_checkpoint == target
        assert state_a.finalized_checkpoint != target

        def add_finality_piggyback(vote_spec, vote_state, data):
            data.finality_target = vote_spec.Checkpoint(
                slot=vote_state.justified_checkpoint.slot,
                root=vote_state.justified_checkpoint.root,
            )
            data.finality_height = vote_state.justified_height

        # The following epoch's committees piggyback finality for that target.
        # Selecting the boundary head advances finalized history, creates the
        # target bootstrap, and prunes the losing branch below finality.
        for slot in range(3 * spec.SLOTS_PER_EPOCH + 1, 4 * spec.SLOTS_PER_EPOCH + 1):
            spec_a, state_a, bid_a = yield from add_new_block(
                test,
                spec_a,
                state_a,
                slot=slot,
                num_sync_participants=1,
                attestation_data_modifier=add_finality_piggyback,
            )
            yield from select_new_head(test, spec_a, bid_a)
        assert state_a.finalized_checkpoint == target
        assert test.latest_finalized_bid == target_bid
        assert (
            get_lc_bootstrap_block_id(get_light_client_bootstrap(test, target_bid.root).data)
            == target_bid
        )
        assert branch_b not in test.lc_data_store.cache.data
        assert branch_b.root not in test.blocks

        # One further signature exposes the newly advanced finality through the
        # best finality update as well as through the bootstrap database.
        spec_a, state_a, bid_a = yield from add_new_block(
            test,
            spec_a,
            state_a,
            num_sync_participants=1,
        )
        yield from select_new_head(test, spec_a, bid_a)
        assert (
            get_light_client_finality_update(test).data.finalized_header.beacon.slot
            == target_bid.slot
        )

        yield from finish_lc_data_collection_test(test)
    finally:
        spec.config = original_config

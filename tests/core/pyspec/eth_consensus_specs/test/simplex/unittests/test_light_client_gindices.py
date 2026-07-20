from eth_consensus_specs.test.context import spec_state_test, spec_test, with_simplex_and_later
from eth_consensus_specs.test.helpers.attestations import state_transition_with_full_block
from eth_consensus_specs.test.helpers.genesis import (
    create_genesis_state,
    create_signed_genesis_block,
)
from eth_consensus_specs.test.helpers.light_client import get_sync_aggregate


@with_simplex_and_later
@spec_test
def test_simplex_light_client_gindices_preserve_historical_layout(spec, phases):
    assert (
        spec.get_generalized_index(spec.BeaconState, "finalized_checkpoint", "root")
        == spec.FINALIZED_ROOT_GINDEX_SIMPLEX
    )
    assert (
        spec.get_generalized_index(spec.BeaconState, "current_sync_committee")
        == spec.CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    )
    assert (
        spec.get_generalized_index(spec.BeaconState, "next_sync_committee")
        == spec.NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    )

    original_config = spec.config
    spec.config = spec.config._replace(
        ELECTRA_FORK_EPOCH=spec.Epoch(2),
        SIMPLEX_FORK_EPOCH=spec.Epoch(4),
    )
    try:
        historical_slot = spec.compute_start_slot_at_epoch(spec.Epoch(3))
        simplex_slot = spec.compute_start_slot_at_epoch(spec.Epoch(4))

        assert (
            spec.finalized_root_gindex_at_slot(historical_slot)
            == spec.FINALIZED_ROOT_GINDEX_ELECTRA
        )
        assert (
            spec.current_sync_committee_gindex_at_slot(historical_slot)
            == spec.CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA
        )
        assert (
            spec.next_sync_committee_gindex_at_slot(historical_slot)
            == spec.NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA
        )

        assert (
            spec.finalized_root_gindex_at_slot(simplex_slot) == spec.FINALIZED_ROOT_GINDEX_SIMPLEX
        )
        assert (
            spec.current_sync_committee_gindex_at_slot(simplex_slot)
            == spec.CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX
        )
        assert (
            spec.next_sync_committee_gindex_at_slot(simplex_slot)
            == spec.NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX
        )
    finally:
        spec.config = original_config


@with_simplex_and_later
@spec_state_test
def test_simplex_light_client_bootstrap_and_update_use_simplex_layout(spec, state):
    original_config = spec.config
    fork_epoch_overrides = dict.fromkeys(
        (
            "ALTAIR_FORK_EPOCH",
            "BELLATRIX_FORK_EPOCH",
            "CAPELLA_FORK_EPOCH",
            "DENEB_FORK_EPOCH",
            "ELECTRA_FORK_EPOCH",
            "FULU_FORK_EPOCH",
            "GLOAS_FORK_EPOCH",
            "SIMPLEX_FORK_EPOCH",
        ),
        spec.GENESIS_EPOCH,
    )
    spec.config = spec.config._replace(**fork_epoch_overrides)
    try:
        trusted_block = create_signed_genesis_block(spec, state)
        trusted_block_root = trusted_block.message.hash_tree_root()
        bootstrap = spec.create_light_client_bootstrap(state, trusted_block)

        assert spec.is_valid_normalized_merkle_branch(
            leaf=state.current_sync_committee.hash_tree_root(),
            branch=bootstrap.current_sync_committee_branch,
            gindex=spec.CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX,
            root=bootstrap.header.beacon.state_root,
        )
        store = spec.initialize_light_client_store(trusted_block_root, bootstrap)

        attested_block = state_transition_with_full_block(
            spec,
            state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
        )
        attested_state = state.copy()
        sync_aggregate, _ = get_sync_aggregate(spec, state)
        block = state_transition_with_full_block(
            spec,
            state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
            sync_aggregate=sync_aggregate,
        )
        update = spec.create_light_client_update(
            state,
            block,
            attested_state,
            attested_block,
            trusted_block,
        )

        assert spec.is_valid_normalized_merkle_branch(
            leaf=attested_state.finalized_checkpoint.root,
            branch=update.finality_branch,
            gindex=spec.FINALIZED_ROOT_GINDEX_SIMPLEX,
            root=update.attested_header.beacon.state_root,
        )
        assert spec.is_valid_normalized_merkle_branch(
            leaf=attested_state.next_sync_committee.hash_tree_root(),
            branch=update.next_sync_committee_branch,
            gindex=spec.NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX,
            root=update.attested_header.beacon.state_root,
        )
        spec.process_light_client_update(
            store,
            update,
            state.slot,
            state.genesis_validators_root,
        )
    finally:
        spec.config = original_config


@with_simplex_and_later
@spec_state_test
def test_simplex_light_client_accepts_historical_gloas_update(spec, state):
    pre_spec = spec.gloas
    original_config = spec.config
    original_pre_config = pre_spec.config
    fork_epoch_overrides = dict.fromkeys(
        (
            "ALTAIR_FORK_EPOCH",
            "BELLATRIX_FORK_EPOCH",
            "CAPELLA_FORK_EPOCH",
            "DENEB_FORK_EPOCH",
            "ELECTRA_FORK_EPOCH",
            "FULU_FORK_EPOCH",
            "GLOAS_FORK_EPOCH",
        ),
        spec.GENESIS_EPOCH,
    )
    spec.config = spec.config._replace(
        **fork_epoch_overrides,
        SIMPLEX_FORK_EPOCH=spec.Epoch(1),
    )
    pre_spec.config = pre_spec.config._replace(**fork_epoch_overrides)
    try:
        pre_state = create_genesis_state(
            pre_spec,
            [int(balance) for balance in state.balances],
            pre_spec.MIN_ACTIVATION_BALANCE,
        )
        trusted_block = create_signed_genesis_block(pre_spec, pre_state)
        trusted_block_root = trusted_block.message.hash_tree_root()
        bootstrap = pre_spec.create_light_client_bootstrap(pre_state, trusted_block)

        assert pre_spec.is_valid_normalized_merkle_branch(
            leaf=pre_state.current_sync_committee.hash_tree_root(),
            branch=bootstrap.current_sync_committee_branch,
            gindex=spec.CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA,
            root=bootstrap.header.beacon.state_root,
        )
        store = spec.initialize_light_client_store(trusted_block_root, bootstrap)

        attested_block = state_transition_with_full_block(
            pre_spec,
            pre_state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
        )
        attested_state = pre_state.copy()
        sync_aggregate, _ = get_sync_aggregate(pre_spec, pre_state)
        block = state_transition_with_full_block(
            pre_spec,
            pre_state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
            sync_aggregate=sync_aggregate,
        )
        update = pre_spec.create_light_client_update(
            pre_state,
            block,
            attested_state,
            attested_block,
            trusted_block,
        )

        assert pre_spec.is_valid_normalized_merkle_branch(
            leaf=attested_state.finalized_checkpoint.root,
            branch=update.finality_branch,
            gindex=spec.FINALIZED_ROOT_GINDEX_ELECTRA,
            root=update.attested_header.beacon.state_root,
        )
        assert pre_spec.is_valid_normalized_merkle_branch(
            leaf=attested_state.next_sync_committee.hash_tree_root(),
            branch=update.next_sync_committee_branch,
            gindex=spec.NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA,
            root=update.attested_header.beacon.state_root,
        )
        spec.process_light_client_update(
            store,
            update,
            pre_state.slot,
            pre_state.genesis_validators_root,
        )
    finally:
        spec.config = original_config
        pre_spec.config = original_pre_config

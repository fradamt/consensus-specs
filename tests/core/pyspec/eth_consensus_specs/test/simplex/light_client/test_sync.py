from eth_utils import encode_hex

from eth_consensus_specs.test.context import (
    spec_state_test,
    with_presets,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import state_transition_with_full_block
from eth_consensus_specs.test.helpers.constants import MINIMAL
from eth_consensus_specs.test.helpers.genesis import (
    create_genesis_state,
    create_signed_genesis_block,
)
from eth_consensus_specs.test.helpers.light_client import get_sync_aggregate
from eth_consensus_specs.test.helpers.light_client_sync import (
    emit_force_update,
    emit_update,
    finish_lc_sync_test,
    LightClientSyncTest,
)
from eth_consensus_specs.test.helpers.state import next_slots


@with_simplex_and_later
@spec_state_test
def test_simplex_light_client_sync(spec, state):
    """Emit a native Simplex bootstrap/update in the standard sync format."""
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
        test = LightClientSyncTest()
        test.steps = []
        test.genesis_validators_root = state.genesis_validators_root
        test.s_spec = spec

        yield (
            "genesis_validators_root",
            "meta",
            encode_hex(test.genesis_validators_root),
        )

        trusted_block = create_signed_genesis_block(spec, state)
        trusted_block_root = trusted_block.message.hash_tree_root()
        yield "trusted_block_root", "meta", encode_hex(trusted_block_root)

        data_epoch = spec.compute_epoch_at_slot(trusted_block.message.slot)
        bootstrap_fork_digest = spec.compute_fork_digest(
            test.genesis_validators_root,
            data_epoch,
        )
        bootstrap = spec.create_light_client_bootstrap(state, trusted_block)
        yield "bootstrap_fork_digest", "meta", encode_hex(bootstrap_fork_digest)
        yield "bootstrap", bootstrap
        yield "store_fork_version", "meta", encode_hex(spec.config.SIMPLEX_FORK_VERSION)

        assert spec.is_valid_normalized_merkle_branch(
            leaf=state.current_sync_committee.hash_tree_root(),
            branch=bootstrap.current_sync_committee_branch,
            gindex=spec.CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX,
            root=bootstrap.header.beacon.state_root,
        )
        test.store = spec.initialize_light_client_store(trusted_block_root, bootstrap)

        finalized_block = state_transition_with_full_block(
            spec,
            state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
        )
        finalized_checkpoint = spec.Checkpoint(
            slot=finalized_block.message.slot,
            root=finalized_block.message.hash_tree_root(),
        )
        # Model an already verified native Simplex certificate in the beacon
        # state. The update below still proves the checkpoint through the
        # attested state's native Simplex generalized index.
        state.justified_checkpoint = finalized_checkpoint
        state.finalized_checkpoint = finalized_checkpoint
        state.justified_height = spec.Height(1)
        state.finalized_height = spec.Height(1)
        state.current_height = spec.Height(2)

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
        update = yield from emit_update(
            test,
            spec,
            state,
            block,
            attested_state,
            attested_block,
            finalized_block,
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
        assert update.finalized_header.beacon.slot > spec.GENESIS_SLOT
        assert test.store.finalized_header.beacon.slot == finalized_block.message.slot
        yield from finish_lc_sync_test(test)
    finally:
        spec.config = original_config


@with_simplex_and_later
@with_presets([MINIMAL], reason="advances through UPDATE_TIMEOUT")
@spec_state_test
def test_simplex_light_client_force_update_timeout(spec, state):
    """Do not force before the timeout, then promote the best valid update."""
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
        test = LightClientSyncTest()
        test.steps = []
        test.genesis_validators_root = state.genesis_validators_root
        test.s_spec = spec
        yield "genesis_validators_root", "meta", encode_hex(test.genesis_validators_root)

        trusted_block = create_signed_genesis_block(spec, state)
        trusted_block_root = trusted_block.message.hash_tree_root()
        yield "trusted_block_root", "meta", encode_hex(trusted_block_root)
        yield (
            "bootstrap_fork_digest",
            "meta",
            encode_hex(spec.compute_fork_digest(test.genesis_validators_root, spec.GENESIS_EPOCH)),
        )
        bootstrap = spec.create_light_client_bootstrap(state, trusted_block)
        yield "bootstrap", bootstrap
        yield "store_fork_version", "meta", encode_hex(spec.config.SIMPLEX_FORK_VERSION)
        test.store = spec.initialize_light_client_store(trusted_block_root, bootstrap)

        attested_block = state_transition_with_full_block(
            spec,
            state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
        )
        attested_state = state.copy()
        sync_aggregate, _ = get_sync_aggregate(spec, state)
        signature_block = state_transition_with_full_block(
            spec,
            state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
            sync_aggregate=sync_aggregate,
        )
        update = yield from emit_update(
            test,
            spec,
            state,
            signature_block,
            attested_state,
            attested_block,
            finalized_block=None,
        )
        assert test.store.best_valid_update == update

        finalized_slot = test.store.finalized_header.beacon.slot
        yield from emit_force_update(test, spec, state)
        assert test.store.finalized_header.beacon.slot == finalized_slot
        assert test.store.best_valid_update == update

        next_slots(
            spec,
            state,
            finalized_slot + spec.UPDATE_TIMEOUT + 1 - state.slot,
        )
        yield from emit_force_update(test, spec, state)
        assert test.store.finalized_header.beacon.slot == attested_block.message.slot
        assert test.store.best_valid_update is None
        yield from finish_lc_sync_test(test)
    finally:
        spec.config = original_config


@with_simplex_and_later
@spec_state_test
def test_simplex_light_client_sync_historical_gloas(spec, state):
    """Process a Gloas bootstrap and update in a native Simplex store."""
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
        test = LightClientSyncTest()
        test.steps = []
        test.genesis_validators_root = pre_state.genesis_validators_root
        test.s_spec = spec
        yield "genesis_validators_root", "meta", encode_hex(test.genesis_validators_root)

        trusted_block = create_signed_genesis_block(pre_spec, pre_state)
        trusted_block_root = trusted_block.message.hash_tree_root()
        yield "trusted_block_root", "meta", encode_hex(trusted_block_root)
        yield (
            "bootstrap_fork_digest",
            "meta",
            encode_hex(
                pre_spec.compute_fork_digest(test.genesis_validators_root, pre_spec.GENESIS_EPOCH)
            ),
        )
        bootstrap = pre_spec.create_light_client_bootstrap(pre_state, trusted_block)
        yield "bootstrap", bootstrap
        yield "store_fork_version", "meta", encode_hex(spec.config.SIMPLEX_FORK_VERSION)
        test.store = spec.initialize_light_client_store(trusted_block_root, bootstrap)

        attested_block = state_transition_with_full_block(
            pre_spec,
            pre_state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
        )
        attested_state = pre_state.copy()
        sync_aggregate, _ = get_sync_aggregate(pre_spec, pre_state)
        signature_block = state_transition_with_full_block(
            pre_spec,
            pre_state,
            fill_cur_epoch=False,
            fill_prev_epoch=False,
            sync_aggregate=sync_aggregate,
        )
        update = yield from emit_update(
            test,
            pre_spec,
            pre_state,
            signature_block,
            attested_state,
            attested_block,
            trusted_block,
        )
        assert pre_spec.is_valid_normalized_merkle_branch(
            leaf=attested_state.finalized_checkpoint.root,
            branch=update.finality_branch,
            gindex=pre_spec.FINALIZED_ROOT_GINDEX_ELECTRA,
            root=update.attested_header.beacon.state_root,
        )
        assert test.store.optimistic_header.beacon.slot == attested_block.message.slot
        yield from finish_lc_sync_test(test)
    finally:
        spec.config = original_config
        pre_spec.config = original_pre_config

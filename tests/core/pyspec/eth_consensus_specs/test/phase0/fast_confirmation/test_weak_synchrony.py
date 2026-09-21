from eth_consensus_specs.test.context import (
    MINIMAL,
    never_bls,
    spec_state_test,
    with_altair_and_later,
    with_presets,
)
from eth_consensus_specs.test.helpers.fast_confirmation import FCRTest


@with_altair_and_later
@spec_state_test
@with_presets([MINIMAL], reason="epoch boundary regression")
@never_bls
def test_certified_banking_preserves_genesis_anchor(spec, state):
    """A certified head must not replace the anchor with the epoch-zero zero root."""
    fcr = FCRTest(spec, seed=1)
    store, fcr_store = fcr.initialize(state)
    anchor = fcr_store.current_epoch_observed_justified_checkpoint.copy()

    fcr.run_slots_with_blocks_and_fast_confirmation(
        number_of_slots=spec.SLOTS_PER_EPOCH, participation_rate=100
    )

    assert spec.has_head_broadcast_certificate(store, spec.get_current_balance_source(fcr_store))
    assert store.unrealized_justifications[fcr.head_root()].epoch == anchor.epoch
    assert fcr_store.current_epoch_observed_justified_checkpoint == anchor
    assert fcr_store.current_epoch_observed_justified_checkpoint.root in store.blocks
    yield from fcr.get_test_artefacts()


@with_altair_and_later
@spec_state_test
@with_presets([MINIMAL], reason="head certificate timing regression")
@never_bls
def test_current_slot_head_has_no_completed_slot_certificate(spec, state):
    """A new head can lack a certificate while its parent already has one."""
    fcr = FCRTest(spec, seed=1)
    store, fcr_store = fcr.initialize(state)
    fcr.run_slots_with_blocks_and_fast_confirmation(
        number_of_slots=2 * spec.SLOTS_PER_EPOCH + 2, participation_rate=100
    )
    balance_source = spec.get_current_balance_source(fcr_store)
    assert spec.has_head_broadcast_certificate(store, balance_source)

    parent = fcr.head_root()
    head = fcr.add_and_apply_block()
    assert fcr.head_root() == head
    assert store.blocks[head].slot == fcr.current_slot()
    assert spec.has_broadcast_certificate(
        store, balance_source, parent, store.blocks[parent].slot, fcr.current_slot() - 1
    )
    assert not spec.has_head_broadcast_certificate(store, balance_source)
    yield from fcr.get_test_artefacts()

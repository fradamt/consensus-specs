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


@with_altair_and_later
@spec_state_test
@with_presets([MINIMAL], reason="duty freshness regression")
@never_bls
def test_duty_freshness_retains_vote_until_next_duty(spec, state):
    """Crossing an epoch alone must not remove a vote before the next duty."""
    fcr = FCRTest(spec, seed=1)
    store, _ = fcr.initialize(state)
    fcr.run_slots_with_blocks_and_fast_confirmation(
        number_of_slots=2 * spec.SLOTS_PER_EPOCH, participation_rate=100
    )
    epoch_start = fcr.current_slot()
    duty_slot = epoch_start + 2
    validator_index = next(iter(spec.get_slot_committee(store, duty_slot)))
    old_message = store.latest_messages[validator_index]
    assert spec.get_latest_message_epoch(old_message) + 1 == fcr.current_epoch()
    assert spec.is_duty_fresh_message(store, validator_index, old_message)

    # No replacement arrives. Keep the old vote through the start of its duty.
    while fcr.current_slot() < duty_slot:
        fcr.next_slot()
        assert spec.is_duty_fresh_message(store, validator_index, old_message)

    # It becomes uncertain only after that duty has completed.
    fcr.next_slot()
    assert store.latest_messages[validator_index] == old_message
    assert not spec.is_duty_fresh_message(store, validator_index, old_message)
    yield from fcr.get_test_artefacts()


@with_altair_and_later
@spec_state_test
@with_presets([MINIMAL], reason="replacement delivery regression")
@never_bls
def test_duty_freshness_accepts_delayed_replacement(spec, state):
    """An omitted replacement excludes the old cell; later delivery restores it."""
    fcr = FCRTest(spec, seed=1)
    store, _ = fcr.initialize(state)
    fcr.run_slots_with_blocks_and_fast_confirmation(
        number_of_slots=2 * spec.SLOTS_PER_EPOCH, participation_rate=100
    )
    duty_slot = fcr.current_slot()
    validator_index = next(iter(spec.get_slot_committee(store, duty_slot)))
    old_message = store.latest_messages[validator_index]
    head = fcr.add_and_apply_block()
    replacements = fcr.attest(block_root=head, attester_indices=[validator_index])
    fcr.recent_attestations = []

    fcr.next_slot()
    assert not spec.is_duty_fresh_message(store, validator_index, old_message)
    fcr.apply_attestations(replacements)
    new_message = store.latest_messages[validator_index]
    assert spec.get_latest_message_epoch(new_message) == fcr.current_epoch()
    assert spec.is_duty_fresh_message(store, validator_index, new_message)
    yield from fcr.get_test_artefacts()


@with_altair_and_later
@spec_state_test
@with_presets([MINIMAL], reason="empty-slot discount across an epoch")
@never_bls
def test_duty_freshness_preserves_then_expires_empty_slot_discount(spec, state):
    """Retain old parent votes across the boundary until newer duties complete."""
    fcr = FCRTest(spec, seed=1)
    store, fcr_store = fcr.initialize(state)
    fcr.run_slots_with_blocks_and_fast_confirmation(
        number_of_slots=2 * spec.SLOTS_PER_EPOCH - 2, participation_rate=100
    )
    parent = fcr.head_root()
    fcr.attest_and_next_slot_with_fast_confirmation(block_root=parent)
    fcr.attest_and_next_slot_with_fast_confirmation(block_root=parent)
    child = fcr.next_slot_with_block_and_fast_confirmation(parent_root=parent)
    balance_source = spec.get_current_balance_source(fcr_store)

    assert spec.compute_slots_since_epoch_start(fcr.current_slot()) == 1
    assert spec.compute_empty_slot_support_discount(store, balance_source, child) > 0

    # Withhold all later replacement votes. Once every new duty has passed,
    # no previous-epoch parent vote may still fund the discount.
    next_epoch = spec.compute_epoch_at_slot(fcr.current_slot()) + 1
    while fcr.current_slot() < spec.compute_start_slot_at_epoch(next_epoch):
        fcr.next_slot()
    assert spec.compute_empty_slot_support_discount(store, balance_source, child) == 0
    yield from fcr.get_test_artefacts()

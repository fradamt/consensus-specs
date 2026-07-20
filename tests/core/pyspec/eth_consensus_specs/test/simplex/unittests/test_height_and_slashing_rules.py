from eth_consensus_specs.test.context import (
    always_bls,
    expect_assertion_error,
    spec_state_test,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attester_slashings import (
    get_valid_attester_slashing_by_indices,
)
from eth_consensus_specs.test.helpers.fork_choice import get_genesis_forkchoice_store
from eth_consensus_specs.test.helpers.keys import privkeys
from eth_consensus_specs.utils import bls


def _set_post_startup_slot(spec, state):
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.GENESIS_EPOCH + 2))


def _set_all_active(spec, state, field, value):
    for index in spec.get_active_validator_indices(state, spec.get_current_epoch(state)):
        field[index] = value


def _get_signed_e2_pair(spec, state, offender, slot):
    attestation_1 = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            target=spec.Checkpoint(slot=slot, root=spec.Root(b"\xa1" * 32)),
            height=spec.Height(7),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        ),
    )
    attestation_2 = attestation_1.copy()
    attestation_2.data.target = spec.Checkpoint(
        slot=slot,
        root=spec.Root(b"\xa2" * 32),
    )
    domain = spec.get_domain(
        state,
        spec.DOMAIN_BEACON_ATTESTER,
        spec.compute_epoch_at_slot(slot),
    )
    for attestation in (attestation_1, attestation_2):
        attestation.signature = bls.Sign(
            privkeys[offender],
            spec.compute_signing_root(attestation.data, domain),
        )
    return attestation_1, attestation_2


@with_simplex_and_later
@spec_state_test
def test_height_event_order_finalizes_then_justifies_before_timeout(spec, state):
    _set_post_startup_slot(spec, state)
    old_finalized = spec.Checkpoint(
        slot=spec.Slot(state.slot - 3),
        root=spec.Root(b"\x11" * 32),
    )
    old_justified = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x22" * 32),
    )
    new_justified = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x33" * 32),
    )
    state.current_height = spec.Height(3)
    state.justified_height = spec.Height(2)
    state.finalized_height = spec.Height(1)
    state.finalized_checkpoint = old_finalized
    state.justified_checkpoint = old_justified
    state.current_height_target = new_justified
    _set_all_active(spec, state, state.finality_participation, value=True)
    _set_all_active(spec, state, state.target_participation, value=True)
    _set_all_active(spec, state, state.timeouts, value=True)

    assert spec.has_new_finalization(state)
    assert spec.compute_justified_checkpoint(state) == new_justified
    assert spec.has_timeout_quorum(state)
    spec.process_justification_and_finalization(state)

    # The independent finality event consumes the old J first. The justify
    # branch then wins over the simultaneously available timeout branch.
    assert state.finalized_checkpoint == old_justified
    assert state.finalized_height == spec.Height(2)
    assert state.justified_checkpoint == new_justified
    assert state.justified_height == spec.Height(3)
    assert state.current_height == spec.Height(4)
    assert state.current_height_target == spec.Checkpoint()
    assert not any(state.target_participation)
    assert not any(state.timeouts)


@with_simplex_and_later
@spec_state_test
def test_nonjustifiable_height_forces_timeout_branch_at_exact_debt_boundary(spec, state):
    _set_post_startup_slot(spec, state)
    assert not spec.is_nonjustifiable_height(spec.Height(8), spec.Height(6))
    assert spec.is_nonjustifiable_height(spec.Height(8), spec.Height(5))
    assert not spec.is_nonjustifiable_height(spec.Height(7), spec.Height(0))

    previous_justified = state.justified_checkpoint
    previous_justified_height = state.justified_height
    state.current_height = spec.Height(8)
    state.finalized_height = spec.Height(5)
    state.current_height_nonjustifiable = True
    target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x44" * 32),
    )
    state.current_height_target = target
    _set_all_active(spec, state, state.target_participation, value=True)
    _set_all_active(spec, state, state.timeouts, value=True)

    # A full target quorum is deliberately inadmissible at a timeout-only
    # height; the independently present timeout quorum advances instead.
    assert spec.compute_justified_checkpoint(state) == spec.Checkpoint()
    assert spec.has_timeout_quorum(state)
    spec.process_justification_and_finalization(state)
    assert state.current_height == spec.Height(9)
    assert not state.current_height_nonjustifiable
    assert state.justified_checkpoint == previous_justified
    assert state.justified_height == previous_justified_height


@with_simplex_and_later
@spec_state_test
def test_nonjustifiable_class_stays_latched_when_finality_reduces_debt(spec, state):
    _set_post_startup_slot(spec, state)
    state.current_height = spec.Height(8)
    state.current_height_nonjustifiable = True
    state.finalized_height = spec.Height(5)
    state.justified_height = spec.Height(6)
    state.finalized_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 3),
        root=spec.Root(b"\x45" * 32),
    )
    state.justified_checkpoint = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x46" * 32),
    )
    forbidden_target = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x47" * 32),
    )
    state.current_height_target = forbidden_target
    _set_all_active(spec, state, state.finality_participation, value=True)
    _set_all_active(spec, state, state.target_participation, value=True)

    assert spec.is_nonjustifiable_height(state.current_height, state.finalized_height)
    assert spec.has_new_finalization(state)
    spec.process_justification_and_finalization(state)

    # Finality-first updates F from height 5 to 6, so recomputing the paper's
    # numeric predicate would now classify H=8 as ordinary. The entry-time
    # latch keeps the already timeout-only height sealed: its full target tally
    # cannot justify and, without a timeout quorum, the height remains open.
    assert state.finalized_height == spec.Height(6)
    assert not spec.is_nonjustifiable_height(state.current_height, state.finalized_height)
    assert state.current_height_nonjustifiable
    assert spec.compute_justified_checkpoint(state) == spec.Checkpoint()
    assert state.current_height == spec.Height(8)


@with_simplex_and_later
@spec_state_test
def test_empty_vote_carries_current_but_not_stale_finality_piggyback(spec, state):
    _set_post_startup_slot(spec, state)
    validator_index = spec.ValidatorIndex(0)
    finalized = spec.Checkpoint(
        slot=spec.Slot(state.slot - 2),
        root=spec.Root(b"\x51" * 32),
    )
    justified = spec.Checkpoint(
        slot=spec.Slot(state.slot - 1),
        root=spec.Root(b"\x52" * 32),
    )
    state.finalized_checkpoint = finalized
    state.finalized_height = spec.Height(1)
    state.justified_checkpoint = justified
    state.justified_height = spec.Height(2)
    empty_vote = spec.AttestationData(
        slot=state.slot,
        beacon_block_root=spec.Root(),
        target=spec.Checkpoint(),
        height=spec.Height(0),
        finality_target=justified,
        finality_height=spec.Height(2),
    )

    assert spec.is_empty_vote(empty_vote)
    assert not spec.is_timeout_vote(empty_vote)
    spec.update_finality_participation(state, validator_index, empty_vote)
    assert state.finality_participation[validator_index]

    state.finality_participation[validator_index] = False
    stale_vote = empty_vote.copy()
    stale_vote.finality_height = spec.Height(1)
    spec.update_finality_participation(state, validator_index, stale_vote)
    assert not state.finality_participation[validator_index]

    stale_vote.finality_height = state.justified_height
    stale_vote.finality_target = finalized
    spec.update_finality_participation(state, validator_index, stale_vote)
    assert not state.finality_participation[validator_index]

    state.finalized_checkpoint = justified
    state.finalized_height = state.justified_height
    spec.update_finality_participation(state, validator_index, empty_vote)
    assert not state.finality_participation[validator_index]


@with_simplex_and_later
@spec_state_test
def test_e1_e2_slashing_timeout_target_and_empty_vote_boundaries(spec, state):
    height = spec.Height(7)
    locked_target = spec.Checkpoint(slot=spec.Slot(3), root=spec.Root(b"\x61" * 32))
    other_target = spec.Checkpoint(slot=spec.Slot(4), root=spec.Root(b"\x62" * 32))
    commitment = spec.AttestationData(
        finality_target=locked_target,
        finality_height=height,
    )
    matching_vote = spec.AttestationData(target=locked_target, height=height)
    conflicting_vote = spec.AttestationData(target=other_target, height=height)
    timeout_vote = spec.AttestationData(target=spec.Checkpoint(), height=height)
    empty_vote = spec.AttestationData(target=spec.Checkpoint(), height=spec.Height(0))

    assert not spec.is_slashable_attestation_data(commitment, matching_vote)
    assert not spec.is_slashable_attestation_data(matching_vote, commitment)
    assert spec.is_slashable_attestation_data(commitment, conflicting_vote)
    assert spec.is_slashable_attestation_data(conflicting_vote, commitment)
    assert spec.is_slashable_attestation_data(commitment, timeout_vote)
    assert spec.is_slashable_attestation_data(timeout_vote, commitment)
    assert not spec.is_slashable_attestation_data(commitment, empty_vote)
    assert spec.is_slashable_attestation_data(matching_vote, conflicting_vote)
    assert spec.is_slashable_attestation_data(conflicting_vote, matching_vote)
    assert not spec.is_slashable_attestation_data(matching_vote, timeout_vote)


@with_simplex_and_later
@spec_state_test
@always_bls
def test_e2_attester_slashing_needs_no_state_target_set(spec, state):
    offender = spec.ValidatorIndex(0)
    attestation_1, attestation_2 = _get_signed_e2_pair(spec, state, offender, state.slot)
    assert spec.is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    assert spec.is_valid_indexed_attestation(state, attestation_1)
    assert spec.is_valid_indexed_attestation(state, attestation_2)

    spec.process_attester_slashing(
        state,
        spec.AttesterSlashing(
            attestation_1=attestation_1,
            attestation_2=attestation_2,
        ),
    )

    assert state.validators[offender].slashed


@with_simplex_and_later
@spec_state_test
@always_bls
def test_round_double_vote_evidence_rejects_e2_pair(spec, state):
    offender = spec.ValidatorIndex(0)
    evidence_slot = state.slot
    attestation_1, attestation_2 = _get_signed_e2_pair(
        spec,
        state,
        offender,
        evidence_slot,
    )
    evidence = spec.RoundDoubleVoteEvidence(
        attestation_1=attestation_1,
        attestation_2=attestation_2,
    )
    state.slot = spec.Slot(evidence_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY)

    expect_assertion_error(lambda: spec.process_round_double_vote_evidence(state, evidence))
    assert not state.round_double_vote_penalized[offender]
    assert not state.validators[offender].slashed


@with_simplex_and_later
@spec_state_test
def test_e1_slashing_cannot_manufacture_pre_fork_duties(spec, state):
    state.fork.epoch = spec.Epoch(1)
    state.fork.current_version = spec.config.SIMPLEX_FORK_VERSION
    fork_slot = spec.compute_start_slot_at_epoch(state.fork.epoch)
    store = get_genesis_forkchoice_store(spec, state)
    state.slot = fork_slot
    offender = spec.ValidatorIndex(0)
    locked_target = spec.Checkpoint(slot=spec.Slot(fork_slot - 1), root=spec.Root(b"\x63" * 32))
    vote = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=spec.Slot(fork_slot - 1),
            target=spec.Checkpoint(),
            height=spec.Height(1),
        ),
    )
    commitment = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=spec.Slot(fork_slot - 1),
            height=spec.Height(2),
            finality_target=locked_target,
            finality_height=spec.Height(1),
        ),
    )
    slashing = spec.AttesterSlashing(
        attestation_1=vote,
        attestation_2=commitment,
    )
    assert spec.is_slashable_attestation_data(vote.data, commitment.data)

    expect_assertion_error(lambda: spec.process_attester_slashing(state, slashing))
    assert not state.validators[offender].slashed

    expect_assertion_error(lambda: spec.on_attester_slashing(store, slashing, is_from_block=True))
    assert offender not in store.equivocating_indices


@with_simplex_and_later
@spec_state_test
@always_bls
def test_standalone_e1_slashing_uses_head_registry_during_finality_stall(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    justified_state = store.block_states[store.justified_checkpoint.root]

    # Model a validator that joined the canonical head registry while the
    # justified checkpoint remained stalled at the older registry.
    head_state = state.copy()
    offender = spec.ValidatorIndex(len(head_state.validators))
    validator = head_state.validators[0].copy()
    validator.pubkey = bls.SkToPk(privkeys[offender])
    head_state.validators.append(validator)
    assert offender >= len(justified_state.validators)

    slashing = get_valid_attester_slashing_by_indices(
        spec,
        head_state,
        [offender],
        signed_1=True,
        signed_2=True,
    )
    assert spec.is_valid_indexed_attestation(head_state, slashing.attestation_1)
    assert spec.is_valid_indexed_attestation(head_state, slashing.attestation_2)

    # Isolate the handler's registry selection: the local fork-choice head has
    # the new validator, while the stalled justified root deliberately does not.
    head_root = spec.Root(b"\x64" * 32)
    store.blocks[head_root] = spec.BeaconBlock(parent_root=store.justified_checkpoint.root)
    store.block_states[head_root] = head_state
    original_get_head = spec.get_head
    spec.get_head = lambda _store: spec.ForkChoiceNode(
        root=head_root,
        payload_status=spec.PAYLOAD_STATUS_PENDING,
    )
    try:
        spec.on_attester_slashing(store, slashing)
    finally:
        spec.get_head = original_get_head

    assert offender in store.equivocating_indices


@with_simplex_and_later
@spec_state_test
@always_bls
def test_old_simplex_evidence_uses_historical_fork_domain(spec, state):
    original_config = spec.config
    spec.config = spec.config._replace(SIMPLEX_FORK_EPOCH=spec.Epoch(1))
    try:
        evidence_epoch = spec.Epoch(1)
        evidence_slot = spec.compute_start_slot_at_epoch(evidence_epoch)
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(3))
        # Simulate the second fork after Simplex. ``state.fork`` alone can no
        # longer recover the Simplex version for the old evidence epoch.
        state.fork.previous_version = spec.Version("0x11000000")
        state.fork.current_version = spec.Version("0x12000000")
        state.fork.epoch = spec.Epoch(3)

        offender = spec.ValidatorIndex(0)
        attestation_1 = spec.IndexedAttestation(
            attesting_indices=[offender],
            data=spec.AttestationData(
                slot=evidence_slot,
                height=spec.Height(1),
                finality_height=spec.FAR_FUTURE_HEIGHT,
            ),
        )
        attestation_2 = attestation_1.copy()
        attestation_2.data.height = spec.Height(2)

        # Sign under the historical Simplex duty domain, independently of the
        # synthetic current/previous Fork pair above.
        domain = spec.compute_domain(
            spec.DOMAIN_BEACON_ATTESTER,
            spec.config.SIMPLEX_FORK_VERSION,
            state.genesis_validators_root,
        )
        for attestation in (attestation_1, attestation_2):
            attestation.signature = bls.Sign(
                privkeys[offender],
                spec.compute_signing_root(attestation.data, domain),
            )

        assert spec.compute_fork_version(evidence_epoch) == spec.config.SIMPLEX_FORK_VERSION
        assert spec.is_valid_indexed_attestation(state, attestation_1)
        assert spec.is_valid_indexed_attestation(state, attestation_2)
        spec.process_round_double_vote_evidence(
            state,
            spec.RoundDoubleVoteEvidence(
                attestation_1=attestation_1,
                attestation_2=attestation_2,
            ),
        )
        assert state.round_double_vote_penalized[offender]
    finally:
        spec.config = original_config


@with_simplex_and_later
@spec_state_test
def test_nonjustifiable_validator_gate_timeout_lock_and_empty_boundaries(spec, state):
    store = get_genesis_forkchoice_store(spec, state)
    anchor_root = store.finalized_checkpoint.root
    current_slot = spec.Slot(1)
    block = spec.BeaconBlock(
        slot=current_slot,
        parent_root=anchor_root,
        body=spec.BeaconBlockBody(graffiti=b"\x71" * 32),
    )
    head_root = block.hash_tree_root()
    head_state = state.copy()
    head_state.slot = current_slot
    head_state.current_height = spec.Height(8)
    head_state.finalized_height = spec.Height(5)
    head_state.current_height_nonjustifiable = True
    store.blocks[head_root] = block
    store.block_states[head_root] = head_state
    store.time = spec.uint64(
        store.genesis_time + current_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    store.h_max = spec.Height(9)
    base_target = spec.Checkpoint(slot=current_slot, root=head_root)

    target, height = spec.get_attestation_target(
        store,
        head_root,
        head_state,
        head_root,
        head_state,
        base_target,
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(8)  # Unlocked and caught up: timeout.

    target, height = spec.get_attestation_target(
        store,
        head_root,
        head_state,
        head_root,
        head_state,
        base_target,
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={spec.Height(8): base_target},
    )
    assert target == base_target
    assert height == spec.Height(8)  # Exact E1 target repeats as a marker only.

    # Model a genuine safe-confirmed ancestor whose stored post-block height 7
    # is below the voted head at height 8.
    behind_state = state.copy()
    behind_state.slot = current_slot
    behind_state.current_height = spec.Height(7)
    store.block_states[anchor_root] = behind_state
    target, height = spec.get_attestation_target(
        store,
        head_root,
        head_state,
        anchor_root,
        behind_state,
        base_target,
        voted_target_at={},
        voted_timeout_at=set(),
        voted_finality_at={},
    )
    assert target == spec.Checkpoint()
    assert height == spec.Height(0)  # Not caught up: empty.

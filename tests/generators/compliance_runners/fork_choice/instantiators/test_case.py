import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from eth_utils import encode_hex
from ruamel.yaml import YAML

from eth_consensus_specs.test.context import (
    spec_state_test,
    spec_test,
    with_altair_and_later,
    with_simplex_and_later,
)
from eth_consensus_specs.test.helpers.attestations import get_valid_attestation
from eth_consensus_specs.test.helpers.block import build_empty_block_for_next_slot
from eth_consensus_specs.test.helpers.constants import SIMPLEX
from eth_consensus_specs.test.helpers.fork_choice import (
    add_payload_vote_checks,
    get_attestation_file_name,
    get_attester_slashing_file_name,
    get_block_file_name,
    get_execution_payload_envelope_file_name,
    get_genesis_forkchoice_store_and_block,
    get_payload_attestation_message_file_name,
    on_tick_and_append_step,
    output_store_checks,
)
from eth_consensus_specs.test.helpers.payload_attestation import (
    prepare_signed_payload_attestation,
)
from eth_consensus_specs.test.helpers.state import state_transition_and_sign_block
from eth_consensus_specs.utils import bls
from tests.generators.compliance_runners.gen_base.gen_typing import (
    TestCase,
    TestCasePart,
    TestCaseResult,
    TestGroup,
)

from .block_cover import gen_block_cover_test_data
from .block_tree import gen_block_tree_test_data
from .helpers import (
    FCTestData,
    filter_out_duplicate_messages,
    make_events,
    ProtocolMessage,
    yield_fork_choice_test_events,
)
from .mutation_operators import MutationOps
from .scheduler import MessageScheduler

BLS_ACTIVE = False
GENERATOR_NAME = "fork_choice_compliance"
SUITE_NAME = "pyspec_tests"
MAX_MUTATION_GROUP_LENGTH = 4
SIMPLEX_NATIVE_CATEGORIES = (
    "topology",
    "invalid_messages",
    "weight_viability_cover",
    "event_mutation",
)
SIMPLEX_NATIVE_SCALE = {
    "tiny": 1,
    "small": 2,
    "standard": 4,
}


def get_available_attestation_file_name(attestation):
    return f"available_attestation_{encode_hex(attestation.hash_tree_root())}"


def get_round_double_vote_evidence_file_name(evidence):
    return f"round_double_vote_evidence_{encode_hex(evidence.hash_tree_root())}"


@dataclass(eq=True, frozen=True)
class FCTestKind:
    pass


@dataclass(eq=True, frozen=True)
class BlockTreeTestKind(FCTestKind):
    with_attester_slashings: bool
    with_invalid_messages: bool


@dataclass(eq=True, frozen=True)
class BlockCoverTestKind(FCTestKind):
    pass


@dataclass
class FCTestDNA:
    kind: FCTestKind
    solution: Any
    variation_seed: int
    mutation_seed: int | None


@dataclass(eq=True, frozen=True)
class MutationGroupCase:
    case_id: int
    mutation_seed: int | None


@dataclass(eq=True, frozen=True)
class MutationGroup:
    test_name: str
    solution_index: int
    test_dna_base: FCTestDNA
    nr_mutations: int


@dataclass(init=False)
class PlainFCTestCase(TestCase):
    test_dna: FCTestDNA
    bls_active: bool
    debug: bool

    def __init__(self, test_dna, bls_active=False, debug=False, **kwds):
        super().__init__(
            fork_name=kwds["fork_name"],
            preset_name=kwds["preset_name"],
            runner_name=kwds["runner_name"],
            handler_name=kwds["handler_name"],
            suite_name=kwds["suite_name"],
            case_name=kwds["case_name"],
        )
        self.test_dna = test_dna
        self.bls_active = bls_active
        self.debug = debug


@dataclass(init=False)
class MutationGroupTestGroup(TestGroup):
    mutation_group: MutationGroup
    group_cases: tuple[MutationGroupCase, ...]
    fork_name: str
    preset_name: str
    bls_active: bool
    debug: bool

    def __init__(
        self,
        mutation_group,
        group_cases,
        fork_name,
        preset_name,
        bls_active=False,
        debug=False,
    ):
        test_cases = [
            PlainFCTestCase(
                test_dna=test_dna,
                bls_active=bls_active,
                debug=debug,
                fork_name=fork_name,
                preset_name=preset_name,
                runner_name=GENERATOR_NAME,
                handler_name=mutation_group.test_name,
                suite_name=SUITE_NAME,
                case_name=case_name,
            )
            for case_name, test_dna in enumerate_test_dnas(mutation_group, group_cases)
        ]
        super().__init__(
            group_name=(
                f"{preset_name}::{fork_name}::{GENERATOR_NAME}::"
                f"{mutation_group.test_name}::{mutation_group.solution_index}::"
                f"{mutation_group.test_dna_base.variation_seed}"
                f"{get_mutation_group_suffix(mutation_group.nr_mutations, group_cases)}"
            ),
            test_cases=test_cases,
            group_fn=self.execute_group,
        )
        self.mutation_group = mutation_group
        self.group_cases = group_cases
        self.fork_name = fork_name
        self.preset_name = preset_name
        self.bls_active = bls_active
        self.debug = debug

    def execute_group(self) -> Iterable[TestCaseResult]:
        bls_active = self.bls_active
        debug = self.debug

        spec, test_data, events = make_test_context(
            self.mutation_group.test_dna_base,
            self.fork_name,
            self.preset_name,
            bls_active=bls_active,
            debug=debug,
        )

        for test_case in self.test_cases:
            mut_seed = test_case.test_dna.mutation_seed
            if mut_seed is None:
                parts_iter = yield_fork_choice_test_events(
                    spec, test_data, events, debug, bls_active=bls_active
                )
            else:
                parts_iter = yield_mutated_test_case_parts(
                    spec, test_data, events, mut_seed, bls_active=bls_active
                )

            yield collect_test_case_result_from_iterator(test_case, parts_iter)


def collect_test_case_result_from_iterator(
    test_case: TestCase,
    parts_iter: Iterable[TestCasePart],
) -> TestCaseResult:
    meta: dict[str, Any] = {}
    outputs: list[TestCasePart] = []

    for name, kind, data in parts_iter:
        if kind == "meta":
            meta[name] = data
        else:
            outputs.append(TestCasePart((name, kind, data)))

    return TestCaseResult(test_case=test_case, meta=meta, case_parts=outputs)


def _make_simplex_available_attestation(
    spec,
    state,
    slot,
    block_root,
    all_positions=False,
    attesting_indices=None,
):
    committee = spec.get_available_committee(state, slot)
    validator_index = committee[0]
    if attesting_indices is not None:
        attesting_indices = set(attesting_indices)
    attestation = spec.AvailableAttestation(
        data=spec.AvailableAttestationData(
            slot=slot,
            beacon_block_root=block_root,
            payload_present=False,
        )
    )
    for position, member in enumerate(committee):
        if all_positions or (
            member == validator_index if attesting_indices is None else member in attesting_indices
        ):
            attestation.aggregation_bits[position] = True
    return attestation


def _make_simplex_round_double_vote_evidence(spec, state, slot, block_root, offender):
    attestation_1 = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=block_root,
            target=spec.Checkpoint(),
            height=state.current_height,
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        ),
    )
    attestation_2 = attestation_1.copy()
    attestation_2.data.height = spec.Height(attestation_1.data.height + 1)
    return spec.RoundDoubleVoteEvidence(
        attestation_1=attestation_1,
        attestation_2=attestation_2,
    )


def _make_simplex_e1_slashing(spec, state, slot, block_root, offender):
    vote = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=block_root,
            target=spec.Checkpoint(slot=slot, root=block_root),
            height=state.current_height,
            finality_target=spec.Checkpoint(),
            finality_height=spec.FAR_FUTURE_HEIGHT,
        ),
    )
    commitment = spec.IndexedAttestation(
        attesting_indices=[offender],
        data=spec.AttestationData(
            slot=slot,
            beacon_block_root=block_root,
            target=spec.Checkpoint(),
            height=spec.Height(0),
            finality_target=spec.Checkpoint(
                slot=slot,
                root=spec.Root(b"\xab" * 32),
            ),
            finality_height=state.current_height,
        ),
    )
    assert spec.is_slashable_attestation_data(vote.data, commitment.data)
    return spec.AttesterSlashing(attestation_1=vote, attestation_2=commitment)


def _make_simplex_payload_vote(spec, state, block_root, supported):
    slot = state.slot
    validator_index = spec.get_ptc(state, slot)[0]
    return spec.PayloadAttestationMessage(
        validator_index=validator_index,
        data=spec.PayloadAttestationData(
            beacon_block_root=block_root,
            slot=slot,
            payload_present=supported,
            blob_data_available=supported,
        ),
    )


def _make_simplex_test_context(fork_name, preset_name):
    @with_simplex_and_later
    @spec_state_test
    def get_spec_and_state(spec, state):
        # A three-item value passes through vector-test post-processing without
        # being mistaken for a named output part.
        yield spec, state, None

    ((spec, state, _),) = get_spec_and_state(
        phase=fork_name,
        preset=preset_name,
        bls_active=BLS_ACTIVE,
    )
    return spec, state


@spec_test
def yield_simplex_fork_choice_test(spec, state):
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    anchor_root = anchor_block.hash_tree_root()

    block_available_attestation = _make_simplex_available_attestation(
        spec,
        state,
        state.slot,
        anchor_root,
    )
    block_evidence = _make_simplex_round_double_vote_evidence(
        spec,
        state,
        state.slot,
        anchor_root,
        spec.ValidatorIndex(0),
    )
    block_e1_slashing = _make_simplex_e1_slashing(
        spec,
        state,
        state.slot,
        anchor_root,
        spec.ValidatorIndex(1),
    )
    block_finality_attestation = get_valid_attestation(
        spec,
        state,
        slot=state.slot,
        beacon_block_root=anchor_root,
        signed=True,
    )
    block_payload_voter = spec.get_ptc(state, state.slot)[0]
    block_payload_attestation = prepare_signed_payload_attestation(
        spec,
        state,
        slot=state.slot,
        beacon_block_root=anchor_root,
        payload_present=False,
        blob_data_available=False,
        attesting_indices=[block_payload_voter],
    )

    block_state = state.copy()
    block = build_empty_block_for_next_slot(spec, block_state)
    block.body.attestations.append(block_finality_attestation)
    block.body.attester_slashings.append(block_e1_slashing)
    block.body.payload_attestations.append(block_payload_attestation)
    block.body.available_attestations.append(block_available_attestation)
    block.body.round_double_vote_evidence.append(block_evidence)
    signed_block = state_transition_and_sign_block(spec, block_state, block)
    block_root = block.hash_tree_root()

    available_attestation = _make_simplex_available_attestation(
        spec,
        block_state,
        block_state.slot,
        block_root,
    )
    evidence = _make_simplex_round_double_vote_evidence(
        spec,
        block_state,
        block_state.slot,
        block_root,
        spec.ValidatorIndex(2),
    )
    payload_vote = _make_simplex_payload_vote(spec, block_state, block_root, supported=True)
    conflicting_payload_vote = _make_simplex_payload_vote(
        spec,
        block_state,
        block_root,
        supported=False,
    )

    yield "anchor_state", state
    yield "anchor_block", anchor_block
    yield get_block_file_name(signed_block), signed_block
    yield get_available_attestation_file_name(available_attestation), available_attestation
    yield get_round_double_vote_evidence_file_name(evidence), evidence
    yield get_payload_attestation_message_file_name(payload_vote), payload_vote
    yield (
        get_payload_attestation_message_file_name(conflicting_payload_vote),
        conflicting_payload_vote,
    )

    test_steps = []
    on_tick_and_append_step(spec, store, store.time, test_steps)

    block_time = spec.uint64(store.genesis_time + block.slot * spec.config.SLOT_DURATION_MS // 1000)
    on_tick_and_append_step(spec, store, block_time, test_steps)
    spec.on_block(store, signed_block)
    test_steps.append({"block": get_block_file_name(signed_block), "valid": True})
    add_payload_vote_checks(store, anchor_root, test_steps)
    output_store_checks(spec, store, test_steps)

    spec.on_available_attestation(store, available_attestation, is_from_block=False)
    test_steps.append(
        {
            "available_attestation": get_available_attestation_file_name(available_attestation),
            "valid": True,
        }
    )
    output_store_checks(spec, store, test_steps)

    spec.on_round_double_vote_evidence(store, evidence, is_from_block=False)
    test_steps.append(
        {
            "round_double_vote_evidence": get_round_double_vote_evidence_file_name(evidence),
            "valid": True,
        }
    )
    output_store_checks(spec, store, test_steps)

    for vote in (payload_vote, conflicting_payload_vote):
        spec.on_payload_attestation_message(store, vote, is_from_block=False)
        test_steps.append(
            {
                "payload_attestation_message": get_payload_attestation_message_file_name(vote),
                "valid": True,
            }
        )
    add_payload_vote_checks(store, block_root, test_steps)
    output_store_checks(spec, store, test_steps, with_viable_for_head_weights=True)

    yield "steps", test_steps


@spec_test
def yield_simplex_missed_confirmation_catchup_test(spec, state):
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)

    block_state = state.copy()
    block_1 = build_empty_block_for_next_slot(spec, block_state)
    signed_block_1 = state_transition_and_sign_block(spec, block_state, block_1)
    block_root_1 = block_1.hash_tree_root()
    available_attestation_1 = _make_simplex_available_attestation(
        spec,
        block_state,
        block_state.slot,
        block_root_1,
        all_positions=True,
    )

    block_2 = build_empty_block_for_next_slot(spec, block_state)
    signed_block_2 = state_transition_and_sign_block(spec, block_state, block_2)
    block_root_2 = block_2.hash_tree_root()
    available_attestation_2 = _make_simplex_available_attestation(
        spec,
        block_state,
        block_state.slot,
        block_root_2,
        all_positions=True,
    )

    yield "anchor_state", state
    yield "anchor_block", anchor_block
    for signed_block in (signed_block_1, signed_block_2):
        yield get_block_file_name(signed_block), signed_block
    for available_attestation in (available_attestation_1, available_attestation_2):
        yield get_available_attestation_file_name(available_attestation), available_attestation

    test_steps = []
    on_tick_and_append_step(spec, store, store.time, test_steps)

    block_1_time = spec.uint64(
        store.genesis_time + block_1.slot * spec.config.SLOT_DURATION_MS // 1000
    )
    on_tick_and_append_step(spec, store, block_1_time, test_steps)
    spec.on_block(store, signed_block_1)
    test_steps.append({"block": get_block_file_name(signed_block_1), "valid": True})
    spec.on_available_attestation(store, available_attestation_1, is_from_block=False)
    test_steps.append(
        {
            "available_attestation": get_available_attestation_file_name(available_attestation_1),
            "valid": True,
        }
    )
    output_store_checks(spec, store, test_steps)

    # Cross into slot 2 without a 50%-of-slot tick in slot 1. The boundary
    # must freeze and evaluate slot 1 before initializing the next slot.
    block_2_time = spec.uint64(
        store.genesis_time + block_2.slot * spec.config.SLOT_DURATION_MS // 1000
    )
    on_tick_and_append_step(spec, store, block_2_time, test_steps)
    spec.on_block(store, signed_block_2)
    test_steps.append({"block": get_block_file_name(signed_block_2), "valid": True})
    spec.on_available_attestation(store, available_attestation_2, is_from_block=False)
    test_steps.append(
        {
            "available_attestation": get_available_attestation_file_name(available_attestation_2),
            "valid": True,
        }
    )
    output_store_checks(spec, store, test_steps)

    # Miss slot 2's deadline and slot 3 entirely. Catch-up must consume fast
    # slot 2 and delayed slots 1/2 before pruning their frozen snapshots.
    catchup_slot = spec.Slot(block_2.slot + 2)
    catchup_time = spec.uint64(
        store.genesis_time + catchup_slot * spec.config.SLOT_DURATION_MS // 1000
    )
    on_tick_and_append_step(spec, store, catchup_time, test_steps)
    assert store.latest_confirmed_head == (block_root_2, spec.Slot(block_2.slot + 1))
    assert store.live_confirmed_head == (block_root_2, spec.Slot(block_2.slot + 1))
    assert store.fast_confirmed_head == (block_root_2, block_2.slot)

    yield "steps", test_steps


def make_simplex_test_group(fork_name, preset_name):
    case_generators = (
        ("block_and_standalone_operations", yield_simplex_fork_choice_test),
        ("missed_confirmation_deadline_catchup", yield_simplex_missed_confirmation_catchup_test),
    )
    test_cases = [
        TestCase(
            fork_name=fork_name,
            preset_name=preset_name,
            runner_name=GENERATOR_NAME,
            handler_name="simplex",
            suite_name=SUITE_NAME,
            case_name=case_name,
        )
        for case_name, _ in case_generators
    ]

    def execute_group():
        for test_case, (_, case_generator) in zip(test_cases, case_generators, strict=True):
            spec, state = _make_simplex_test_context(fork_name, preset_name)
            parts_iter = case_generator(
                spec=spec,
                state=state,
                bls_active=BLS_ACTIVE,
            )
            yield collect_test_case_result_from_iterator(test_case, parts_iter)

    return TestGroup(
        group_name=(f"{preset_name}::{fork_name}::{GENERATOR_NAME}::simplex::{SUITE_NAME}"),
        test_cases=test_cases,
        group_fn=execute_group,
    )


def _build_simplex_child(spec, parent_state, marker):
    post_state = parent_state.copy()
    block = build_empty_block_for_next_slot(spec, post_state)
    block.body.graffiti = bytes([marker]) * 32
    signed_block = state_transition_and_sign_block(spec, post_state, block)
    return post_state, signed_block


def _advance_simplex_state_copy(spec, state, slot):
    advanced_state = state.copy()
    if advanced_state.slot < slot:
        spec.process_slots(advanced_state, slot)
    return advanced_state


def _make_simplex_native_topology_data(spec, state, category, scenario_seed):
    rnd = random.Random(scenario_seed)
    _, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)

    marker_a, marker_b, marker_child = rnd.sample(range(1, 256), 3)
    state_a, signed_block_a = _build_simplex_child(spec, state, marker_a)
    state_b, signed_block_b = _build_simplex_child(spec, state, marker_b)
    state_a_child, signed_block_a_child = _build_simplex_child(
        spec,
        state_a,
        marker_child,
    )

    root_b = signed_block_b.message.hash_tree_root()
    root_a_child = signed_block_a_child.message.hash_tree_root()
    vote_slot = spec.Slot(signed_block_a_child.message.slot + 1)
    vote_state_a = _advance_simplex_state_copy(spec, state_a_child, vote_slot)
    vote_state_b = _advance_simplex_state_copy(spec, state_b, vote_slot)

    if category == "weight_viability_cover" or rnd.choice((True, False)):
        finality_root = root_a_child
        finality_state = vote_state_a
        available_root = root_b
        available_state = vote_state_b
    else:
        finality_root = root_b
        finality_state = vote_state_b
        available_root = root_a_child
        available_state = vote_state_a

    finality_attestation = get_valid_attestation(
        spec,
        finality_state,
        slot=vote_slot,
        beacon_block_root=finality_root,
        signed=True,
    )

    available_committee = spec.get_available_committee(available_state, vote_slot)
    available_members = sorted(set(available_committee))
    if category == "topology":
        rnd.shuffle(available_members)
        available_members = available_members[: max(1, len(available_members) // 2)]
    available_attestation = _make_simplex_available_attestation(
        spec,
        available_state,
        vote_slot,
        available_root,
        attesting_indices=available_members,
    )

    final_slot = spec.Slot(vote_slot + 1)
    final_time = state.genesis_time + final_slot * spec.config.SLOT_DURATION_MS // 1000
    return FCTestData(
        meta={
            "bls_setting": 0,
            "category": category,
            "scenario_seed": scenario_seed,
            "block_parents": "[0, 0, 0, 1]",
        },
        anchor_block=anchor_block,
        anchor_state=state,
        blocks=[
            ProtocolMessage(signed_block_a),
            ProtocolMessage(signed_block_b),
            ProtocolMessage(signed_block_a_child),
        ],
        atts=[ProtocolMessage(finality_attestation)],
        available_atts=[ProtocolMessage(available_attestation)],
        store_final_time=final_time,
    )


def _make_simplex_native_invalid_data(spec, state, scenario_seed):
    rnd = random.Random(scenario_seed)
    _, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    post_state, signed_block = _build_simplex_child(spec, state, rnd.randint(1, 255))
    block_root = signed_block.message.hash_tree_root()

    invalid_block = signed_block.copy()
    invalid_block.message.body.graffiti = bytes([rnd.randint(1, 255)]) * 32
    invalid_block.message.state_root = spec.Root(rnd.randbytes(32))

    invalid_attestation = get_valid_attestation(
        spec,
        post_state,
        slot=post_state.slot,
        beacon_block_root=block_root,
        signed=True,
    )
    for position in range(len(invalid_attestation.aggregation_bits)):
        invalid_attestation.aggregation_bits[position] = False

    invalid_available_attestation = _make_simplex_available_attestation(
        spec,
        post_state,
        post_state.slot,
        block_root,
    )
    invalid_available_attestation.data.payload_present = True

    invalid_evidence = _make_simplex_round_double_vote_evidence(
        spec,
        post_state,
        post_state.slot,
        block_root,
        spec.ValidatorIndex(0),
    )
    invalid_evidence.attestation_2.data = invalid_evidence.attestation_1.data

    invalid_slashing = _make_simplex_e1_slashing(
        spec,
        post_state,
        post_state.slot,
        block_root,
        spec.ValidatorIndex(1),
    )
    invalid_slashing.attestation_2.data = invalid_slashing.attestation_1.data

    invalid_payload_vote = _make_simplex_payload_vote(
        spec,
        post_state,
        block_root,
        supported=True,
    )
    invalid_payload_vote.validator_index = spec.ValidatorIndex(len(post_state.validators))

    final_slot = spec.Slot(post_state.slot + 1)
    final_time = state.genesis_time + final_slot * spec.config.SLOT_DURATION_MS // 1000
    return FCTestData(
        meta={
            "bls_setting": 0,
            "category": "invalid_messages",
            "scenario_seed": scenario_seed,
        },
        anchor_block=anchor_block,
        anchor_state=state,
        blocks=[
            ProtocolMessage(signed_block),
            ProtocolMessage(invalid_block, valid=False),
        ],
        atts=[ProtocolMessage(invalid_attestation, valid=False)],
        slashings=[ProtocolMessage(invalid_slashing, valid=False)],
        payload_atts=[ProtocolMessage(invalid_payload_vote, valid=False)],
        available_atts=[ProtocolMessage(invalid_available_attestation, valid=False)],
        round_evidence=[ProtocolMessage(invalid_evidence, valid=False)],
        store_final_time=final_time,
    )


def make_simplex_native_test_data(spec, state, category, scenario_seed):
    if category == "invalid_messages":
        return _make_simplex_native_invalid_data(spec, state, scenario_seed)
    if category in ("topology", "weight_viability_cover", "event_mutation"):
        return _make_simplex_native_topology_data(spec, state, category, scenario_seed)
    raise ValueError(f"Unknown Simplex native compliance category: {category}")


def get_simplex_native_scenario_seeds(config_path, initial_seed):
    config_name = Path(config_path).parent.name
    scale = SIMPLEX_NATIVE_SCALE.get(config_name, 1)
    seed = 0 if initial_seed is None else initial_seed
    rnd = random.Random(f"simplex:{config_name}:{seed}")
    return rnd.sample(range(1, 1_000_000_000), scale)


def make_simplex_native_test_group(
    fork_name,
    preset_name,
    category,
    scenario_seed,
):
    mutation_seed = scenario_seed ^ 0x5F3759DF if category == "event_mutation" else None
    case_name = f"{category}_{scenario_seed}"
    test_case = TestCase(
        fork_name=fork_name,
        preset_name=preset_name,
        runner_name=GENERATOR_NAME,
        handler_name=f"simplex_{category}",
        suite_name=SUITE_NAME,
        case_name=case_name,
    )

    def execute_group():
        spec, state = _make_simplex_test_context(fork_name, preset_name)
        test_data = make_simplex_native_test_data(spec, state, category, scenario_seed)
        events = make_events(spec, test_data)
        if mutation_seed is not None:
            test_vector = events_to_test_vector(events)
            mutation_ops = MutationOps(
                test_data.anchor_state.genesis_time,
                spec.config.SLOT_DURATION_MS // 1000,
            )
            test_vector, mutations = mutation_ops.rand_mutations(
                test_vector,
                4,
                random.Random(mutation_seed),
            )
            test_data.meta["mutation_seed"] = mutation_seed
            test_data.meta["mutations"] = mutations
            events = convert_test_vector_to_events(test_vector)

        store = spec.get_forkchoice_store(test_data.anchor_state, test_data.anchor_block)
        parts_iter = yield_simplex_native_test_parts(
            spec,
            store,
            test_data,
            events,
            enforce_expected_validity=mutation_seed is None,
            bls_active=BLS_ACTIVE,
        )
        yield collect_test_case_result_from_iterator(test_case, parts_iter)

    return TestGroup(
        group_name=(
            f"{preset_name}::{fork_name}::{GENERATOR_NAME}::"
            f"simplex_{category}::{SUITE_NAME}::{scenario_seed}"
        ),
        test_cases=[test_case],
        group_fn=execute_group,
    )


def get_test_data(spec, state, test_kind, solution, debug, seed):
    if isinstance(test_kind, BlockTreeTestKind):
        with_attester_slashings = test_kind.with_attester_slashings
        with_invalid_messages = test_kind.with_invalid_messages
        sm_links = solution["sm_links"]
        block_parents = solution["block_parents"]
        test_data = gen_block_tree_test_data(
            spec,
            state,
            debug,
            seed,
            sm_links,
            block_parents,
            with_attester_slashings,
            with_invalid_messages,
        )
    elif isinstance(test_kind, BlockCoverTestKind):
        model_params = solution
        test_data, _ = gen_block_cover_test_data(spec, state, model_params, debug, seed)
    else:
        raise ValueError(f"Unknown FC test kind {test_kind}")
    return test_data


def make_test_context(
    test_dna_base: FCTestDNA,
    fork_name: str,
    preset_name: str,
    bls_active: bool = False,
    debug: bool = False,
):
    @with_altair_and_later
    @spec_state_test
    def get_spec_test_data_and_events(spec, state):
        test_kind = test_dna_base.kind
        solution = test_dna_base.solution
        seed = test_dna_base.variation_seed
        test_data = get_test_data(spec, state, test_kind, solution, debug, seed)
        events = make_events(spec, test_data)
        yield (spec, test_data, events)

    ((spec, test_data, events),) = get_spec_test_data_and_events(
        phase=fork_name,
        preset=preset_name,
        bls_active=bls_active,
    )

    return spec, test_data, events


@spec_test
def yield_mutated_test_case_parts(spec, test_data, events, mut_seed):
    store = spec.get_forkchoice_store(test_data.anchor_state, test_data.anchor_block)

    test_vector = events_to_test_vector(events)
    mops = MutationOps(store.time, spec.config.SLOT_DURATION_MS // 1000)
    mutated_vector, mutations = mops.rand_mutations(test_vector, 4, random.Random(mut_seed))

    test_data.meta["mut_seed"] = mut_seed
    test_data.meta["mutations"] = mutations

    mutated_events = convert_test_vector_to_events(mutated_vector)
    return yield_test_parts(spec, store, test_data, mutated_events)


def events_to_test_vector(events) -> list[Any]:
    test_vector = []
    current_time = None
    for event in events:
        event_kind, data, _ = event
        if event_kind == "tick":
            current_time = data
        else:
            if (
                event_kind == "block"
                or event_kind == "attestation"
                or event_kind == "attester_slashing"
                or event_kind == "execution_payload"
                or event_kind == "payload_attestation"
                or event_kind == "available_attestation"
                or event_kind == "round_double_vote_evidence"
            ):
                event_id = data
            else:
                raise AssertionError(event_kind)
            test_vector.append((current_time, (event_kind, event_id)))
    return test_vector


def convert_test_vector_to_events(test_vector):
    events = []
    current_time = None
    for time, (event_kind, data) in test_vector:
        if time != current_time:
            current_time = time
            events.append(("tick", time, None))
        events.append((event_kind, data, None))
    return events


@filter_out_duplicate_messages
def yield_test_parts(
    spec,
    store,
    test_data: FCTestData,
    events,
    enforce_expected_validity=False,
):
    record_recovery_messages = True

    for k, v in test_data.meta.items():
        yield k, "meta", v

    yield "anchor_state", test_data.anchor_state
    yield "anchor_block", test_data.anchor_block

    for message in test_data.blocks:
        block = message.payload
        yield get_block_file_name(block), block

    for message in test_data.atts:
        attestation = message.payload
        yield get_attestation_file_name(attestation), attestation

    for message in test_data.slashings:
        attester_slashing = message.payload
        yield get_attester_slashing_file_name(attester_slashing), attester_slashing

    for message in test_data.envelopes:
        envelope = message.payload
        yield get_execution_payload_envelope_file_name(envelope), envelope

    for message in test_data.payload_atts:
        ptc_message = message.payload
        yield get_payload_attestation_message_file_name(ptc_message), ptc_message

    for message in test_data.available_atts:
        available_attestation = message.payload
        yield get_available_attestation_file_name(available_attestation), available_attestation

    for message in test_data.round_evidence:
        evidence = message.payload
        yield get_round_double_vote_evidence_file_name(evidence), evidence

    test_steps = []
    scheduler = MessageScheduler(spec, store)

    # record first tick
    on_tick_and_append_step(spec, store, store.time, test_steps)

    for kind, data, expected_valid in events:
        if kind == "tick":
            time = data
            if time > store.time:
                applied_events = scheduler.process_tick(time)
                if record_recovery_messages:
                    for event_kind, event_data, recovery in applied_events:
                        if event_kind == "tick":
                            test_steps.append({"tick": int(event_data)})
                        elif event_kind == "block":
                            assert recovery
                            _block_id = get_block_file_name(event_data)
                            test_steps.append({"block": _block_id, "valid": True})
                        elif event_kind == "attestation":
                            assert recovery
                            _attestation_id = get_attestation_file_name(event_data)
                            if _attestation_id not in test_data.atts:
                                yield _attestation_id, event_data
                            test_steps.append({"attestation": _attestation_id, "valid": True})
                        elif event_kind == "execution_payload":
                            assert recovery
                            _payload_id = get_execution_payload_envelope_file_name(event_data)
                            test_steps.append({"execution_payload": _payload_id, "valid": True})
                        elif event_kind == "payload_attestation":
                            assert recovery
                            _payload_attestation_id = get_payload_attestation_message_file_name(
                                event_data
                            )
                            test_steps.append(
                                {
                                    "payload_attestation_message": _payload_attestation_id,
                                    "valid": True,
                                }
                            )
                        elif event_kind == "available_attestation":
                            assert recovery
                            _available_attestation_id = get_available_attestation_file_name(
                                event_data
                            )
                            yield _available_attestation_id, event_data
                            test_steps.append(
                                {
                                    "available_attestation": _available_attestation_id,
                                    "valid": True,
                                }
                            )
                        elif event_kind == "round_double_vote_evidence":
                            assert recovery
                            _evidence_id = get_round_double_vote_evidence_file_name(event_data)
                            yield _evidence_id, event_data
                            test_steps.append(
                                {
                                    "round_double_vote_evidence": _evidence_id,
                                    "valid": True,
                                }
                            )
                        else:
                            raise AssertionError
                else:
                    raise AssertionError
                if time > store.time:
                    # inside a slot
                    on_tick_and_append_step(spec, store, time, test_steps)
                else:
                    assert time == store.time
                    output_store_checks(spec, store, test_steps)
        elif kind == "block":
            block = data
            block_id = get_block_file_name(block)
            valid, applied_events = scheduler.process_block(block)
            if record_recovery_messages:
                if valid:
                    for event_kind, event_data, recovery in applied_events:
                        if event_kind == "block":
                            _block_id = get_block_file_name(event_data)
                            if recovery:
                                test_steps.append({"block": _block_id, "valid": True})
                            else:
                                test_steps.append({"block": _block_id, "valid": True})
                        elif event_kind == "attestation":
                            _attestation_id = get_attestation_file_name(event_data)
                            if recovery:
                                if _attestation_id not in test_data.atts:
                                    yield _attestation_id, event_data
                                test_steps.append({"attestation": _attestation_id, "valid": True})
                            else:
                                raise AssertionError
                                test_steps.append({"attestation": _attestation_id, "valid": True})
                        elif event_kind == "execution_payload":
                            assert recovery
                            _payload_id = get_execution_payload_envelope_file_name(event_data)
                            test_steps.append({"execution_payload": _payload_id, "valid": True})
                        elif event_kind == "payload_attestation":
                            _payload_attestation_id = get_payload_attestation_message_file_name(
                                event_data
                            )
                            assert recovery
                            test_steps.append(
                                {
                                    "payload_attestation_message": _payload_attestation_id,
                                    "valid": True,
                                }
                            )
                        elif event_kind == "available_attestation":
                            assert recovery
                            _available_attestation_id = get_available_attestation_file_name(
                                event_data
                            )
                            yield _available_attestation_id, event_data
                            test_steps.append(
                                {
                                    "available_attestation": _available_attestation_id,
                                    "valid": True,
                                }
                            )
                        elif event_kind == "round_double_vote_evidence":
                            assert recovery
                            _evidence_id = get_round_double_vote_evidence_file_name(event_data)
                            yield _evidence_id, event_data
                            test_steps.append(
                                {
                                    "round_double_vote_evidence": _evidence_id,
                                    "valid": True,
                                }
                            )
                        else:
                            raise AssertionError
                else:
                    assert len(applied_events) == 0
                    test_steps.append({"block": block_id, "valid": valid})
            else:
                raise AssertionError
                test_steps.append({"block": block_id, "valid": valid})
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            output_store_checks(spec, store, test_steps)
        elif kind == "attestation":
            attestation = data
            att_id = get_attestation_file_name(attestation)
            valid = scheduler.process_attestation(attestation, is_from_block=False)
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append({"attestation": att_id, "valid": valid})
            output_store_checks(spec, store, test_steps)
        elif kind == "attester_slashing":
            attester_slashing = data
            slashing_id = get_attester_slashing_file_name(attester_slashing)
            valid = scheduler.process_slashing(attester_slashing)
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append({"attester_slashing": slashing_id, "valid": valid})
            output_store_checks(spec, store, test_steps)
        elif kind == "execution_payload":
            envelope = data
            envelope_id = get_execution_payload_envelope_file_name(envelope)
            valid = scheduler.process_payload(envelope)
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append({"execution_payload": envelope_id, "valid": valid})
            output_store_checks(spec, store, test_steps)
        elif kind == "payload_attestation":
            ptc_message = data
            ptc_message_id = get_payload_attestation_message_file_name(ptc_message)
            valid = scheduler.process_payload_attestation_message(ptc_message, is_from_block=False)
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append({"payload_attestation_message": ptc_message_id, "valid": valid})
            output_store_checks(spec, store, test_steps)
        elif kind == "available_attestation":
            available_attestation = data
            available_attestation_id = get_available_attestation_file_name(available_attestation)
            valid = scheduler.process_available_attestation(
                available_attestation,
                is_from_block=False,
            )
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append(
                {
                    "available_attestation": available_attestation_id,
                    "valid": valid,
                }
            )
            output_store_checks(spec, store, test_steps)
        elif kind == "round_double_vote_evidence":
            evidence = data
            evidence_id = get_round_double_vote_evidence_file_name(evidence)
            valid = scheduler.process_round_double_vote_evidence(evidence, is_from_block=False)
            if enforce_expected_validity and expected_valid is not None:
                assert valid == expected_valid
            test_steps.append(
                {
                    "round_double_vote_evidence": evidence_id,
                    "valid": valid,
                }
            )
            output_store_checks(spec, store, test_steps)
        else:
            raise ValueError(f"not implemented {kind}")
    next_slot_time = (
        store.genesis_time
        + (spec.get_current_slot(store) + 1) * spec.config.SLOT_DURATION_MS // 1000
    )
    on_tick_and_append_step(spec, store, next_slot_time, test_steps)
    output_store_checks(spec, store, test_steps, with_viable_for_head_weights=True)

    yield "steps", test_steps


@spec_test
def yield_simplex_native_test_parts(
    spec,
    store,
    test_data,
    events,
    enforce_expected_validity=False,
):
    """Apply standard vector serialization and the configured BLS switch."""
    yield from yield_test_parts(
        spec,
        store,
        test_data,
        events,
        enforce_expected_validity=enforce_expected_validity,
    )


def prepare_bls():
    bls.use_milagro()


def get_test_kind(test_type, with_attester_slashings, with_invalid_messages):
    if test_type == "block_tree":
        return BlockTreeTestKind(with_attester_slashings, with_invalid_messages)
    elif test_type == "block_cover":
        return BlockCoverTestKind()
    else:
        raise ValueError(f"Unsupported test type: {test_type}")


def _load_yaml(path: str):
    with Path(path).open() as f:
        yaml = YAML(typ="safe")
        return yaml.load(f)


def derive_effective_seed(seed: int, solution_index: int) -> int:
    return random.Random(f"{seed}:{solution_index}").randint(0, 1_000_000_000)


def get_mutation_group_suffix(nr_mutations: int, group_cases: tuple[MutationGroupCase, ...]) -> str:
    group_case_ids = [group_case.case_id for group_case in group_cases]
    full_case_ids = list(range(nr_mutations + 1))
    if group_case_ids == full_case_ids:
        return ""
    joined_case_ids = ",".join(str(case_id) for case_id in group_case_ids)
    return f"::cases={joined_case_ids}"


def iter_mutation_group_chunks(
    mutation_seeds: list[int],
) -> Iterable[tuple[MutationGroupCase, ...]]:
    all_group_cases = [MutationGroupCase(0, None)] + [
        MutationGroupCase(case_id=i, mutation_seed=mutation_seed)
        for i, mutation_seed in enumerate(mutation_seeds, start=1)
    ]
    total_group_length = len(all_group_cases)
    if total_group_length <= MAX_MUTATION_GROUP_LENGTH:
        yield tuple(all_group_cases)
        return

    bucket_count = ceil(total_group_length / MAX_MUTATION_GROUP_LENGTH)
    base_bucket_size = total_group_length // bucket_count
    remainder = total_group_length % bucket_count
    bucket_sizes = [
        base_bucket_size + (1 if bucket_index < remainder else 0)
        for bucket_index in range(bucket_count)
    ]

    remaining_mutation_cases = all_group_cases[1:]
    for bucket_index, bucket_size in enumerate(bucket_sizes):
        if bucket_index == 0:
            mutations_in_bucket = bucket_size - 1
            bucket_cases = remaining_mutation_cases[:mutations_in_bucket]
            yield (all_group_cases[0], *bucket_cases)
        else:
            mutations_in_bucket = bucket_size
            bucket_cases = remaining_mutation_cases[:mutations_in_bucket]
            yield tuple(bucket_cases)
        remaining_mutation_cases = remaining_mutation_cases[mutations_in_bucket:]


def enumerate_mutation_groups(config_dir, test_name, params) -> Iterable[MutationGroup]:
    test_type = params["test_type"]
    instances_path = params["instances"]
    initial_seed = params["seed"]
    nr_variations = params["nr_variations"]
    nr_mutations = params["nr_mutations"]
    with_attester_slashings = params.get("with_attester_slashings", False)
    with_invalid_messages = params.get("with_invalid_messages", False)

    solutions = _load_yaml(str(Path(config_dir) / instances_path))
    test_kind = get_test_kind(test_type, with_attester_slashings, with_invalid_messages)

    seeds = [initial_seed]
    if nr_variations > 1:
        rnd = random.Random(initial_seed)
        seeds = [rnd.randint(1, 10000) for _ in range(nr_variations)]
        seeds[0] = initial_seed

    for i, solution in enumerate(solutions):
        for seed in seeds:
            effective_seed = derive_effective_seed(seed, i)
            yield MutationGroup(
                test_name=test_name,
                solution_index=i,
                test_dna_base=FCTestDNA(test_kind, solution, effective_seed, None),
                nr_mutations=nr_mutations,
            )


def split_mutation_group(mutation_group: MutationGroup) -> Iterable[tuple[MutationGroupCase, ...]]:
    mutation_seeds = [
        mutation_group.test_dna_base.variation_seed + j - 1
        for j in range(1, mutation_group.nr_mutations + 1)
    ]
    yield from iter_mutation_group_chunks(mutation_seeds)


def enumerate_test_dnas(
    mutation_group: MutationGroup, group_cases: tuple[MutationGroupCase, ...]
) -> Iterable[tuple[str, FCTestDNA]]:
    test_name = mutation_group.test_name
    solution_index = mutation_group.solution_index
    test_dna_base = mutation_group.test_dna_base
    seed = test_dna_base.variation_seed

    for group_case in group_cases:
        case_id = group_case.case_id
        mutation_seed = group_case.mutation_seed
        test_dna = FCTestDNA(
            test_dna_base.kind,
            test_dna_base.solution,
            seed,
            mutation_seed,
        )
        case_name = test_name + "_" + str(solution_index) + "_" + str(seed) + "_" + str(case_id)
        yield case_name, test_dna


def enumerate_test_groups(config_path, forks, presets, debug, initial_seed: int | None = None):
    for fork_name in forks:
        if fork_name == SIMPLEX:
            for preset_name in presets:
                yield make_simplex_test_group(fork_name, preset_name)
                for scenario_seed in get_simplex_native_scenario_seeds(
                    config_path,
                    initial_seed,
                ):
                    for category in SIMPLEX_NATIVE_CATEGORIES:
                        yield make_simplex_native_test_group(
                            fork_name,
                            preset_name,
                            category,
                            scenario_seed,
                        )

    forks = [fork_name for fork_name in forks if fork_name != SIMPLEX]
    if not forks:
        return

    config_dir = str(Path(config_path).parent)
    test_gen_config = _load_yaml(config_path)

    seed_generator = random.Random(initial_seed) if initial_seed is not None else None
    for test_name, params in test_gen_config.items():
        if seed_generator is not None:
            params = params | {"seed": seed_generator.randint(0, 1_000_000_000)}
        if debug:
            print(test_name)
        for fork_name in forks:
            for preset_name in presets:
                for mutation_group in enumerate_mutation_groups(config_dir, test_name, params):
                    for group_cases in split_mutation_group(mutation_group):
                        yield MutationGroupTestGroup(
                            mutation_group=mutation_group,
                            group_cases=group_cases,
                            fork_name=fork_name,
                            preset_name=preset_name,
                            bls_active=BLS_ACTIVE,
                            debug=debug,
                        )

import random
from pathlib import Path

from eth_consensus_specs.test.helpers.constants import MINIMAL, SIMPLEX
from tests.generators.compliance_runners.fork_choice.instantiators.mutation_operators import (
    MutationOps,
)
from tests.generators.compliance_runners.fork_choice.instantiators.scheduler import (
    QueueItemKind,
)
from tests.generators.compliance_runners.fork_choice.instantiators.test_case import (
    convert_test_vector_to_events,
    enumerate_test_groups,
    events_to_test_vector,
    SIMPLEX_NATIVE_CATEGORIES,
    SIMPLEX_NATIVE_SCALE,
)

COMPLIANCE_DIR = Path(__file__).parents[1] / "generators" / "compliance_runners" / "fork_choice"


def _enumerate_simplex_groups(config_name, seed):
    config_path = COMPLIANCE_DIR / config_name / "test_gen.yaml"
    return list(
        enumerate_test_groups(
            str(config_path),
            forks=[SIMPLEX],
            presets=[MINIMAL],
            debug=False,
            initial_seed=seed,
        )
    )


def _native_groups(groups):
    return [group for group in groups if "::simplex_" in group.group_name]


def test_simplex_compliance_enumerates_native_categories():
    groups = _enumerate_simplex_groups("tiny", seed=1234)
    group_names = {group.group_name for group in groups}

    # Keep the original deterministic group alongside every native category.
    assert any("::simplex::" in name for name in group_names)
    for category in SIMPLEX_NATIVE_CATEGORIES:
        assert any(f"::simplex_{category}::" in name for name in group_names)


def test_simplex_compliance_group_count_scales_with_config():
    for config_name, scale in SIMPLEX_NATIVE_SCALE.items():
        groups = _enumerate_simplex_groups(config_name, seed=1234)
        assert len(groups) == 1 + scale * len(SIMPLEX_NATIVE_CATEGORIES)


def test_simplex_compliance_seed_changes_native_groups_only():
    first = _enumerate_simplex_groups("small", seed=1)
    second = _enumerate_simplex_groups("small", seed=2)
    first_native = _native_groups(first)
    second_native = _native_groups(second)

    assert {group.group_name for group in first_native} != {
        group.group_name for group in second_native
    }
    assert [group.group_name for group in first if group not in first_native] == [
        group.group_name for group in second if group not in second_native
    ]


def test_simplex_event_kinds_round_trip_through_mutation_plumbing():
    event_kinds = (
        "block",
        "attestation",
        "attester_slashing",
        "execution_payload",
        "payload_attestation",
        "available_attestation",
        "round_double_vote_evidence",
    )
    events = [("tick", 100, None)] + [
        (event_kind, f"message-{index}", True) for index, event_kind in enumerate(event_kinds)
    ]

    vector = events_to_test_vector(events)
    assert [event_kind for _, (event_kind, _) in vector] == list(event_kinds)
    round_tripped = convert_test_vector_to_events(vector)
    assert [kind for kind, _, _ in round_tripped if kind != "tick"] == list(event_kinds)

    mutation_ops = MutationOps(start_time=100, seconds_per_slot=12)
    mutated, mutations = mutation_ops.rand_mutations(vector, 4, random.Random(2025))
    assert mutations
    assert {event_kind for _, (event_kind, _) in mutated}.issubset(set(event_kinds))
    assert {
        QueueItemKind.AVAILABLE_ATTESTATION,
        QueueItemKind.ROUND_DOUBLE_VOTE_EVIDENCE,
    }.issubset(set(QueueItemKind))

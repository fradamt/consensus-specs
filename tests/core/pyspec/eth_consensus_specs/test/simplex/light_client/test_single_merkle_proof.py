from eth_consensus_specs.test.context import (
    spec_state_test,
    with_simplex_and_later,
    with_test_suite_name,
)
from eth_consensus_specs.test.helpers.light_client import (
    latest_current_sync_committee_gindex,
    latest_finalized_root_gindex,
    latest_next_sync_committee_gindex,
)


def _yield_proof(spec, state, leaf, gindex):
    yield "object", state
    branch = spec.compute_merkle_proof(state, gindex)
    yield (
        "proof",
        {
            "leaf": "0x" + leaf.hex(),
            "leaf_index": gindex,
            "branch": ["0x" + root.hex() for root in branch],
        },
    )
    assert spec.is_valid_merkle_branch(
        leaf=leaf,
        branch=branch,
        depth=spec.floorlog2(gindex),
        index=spec.get_subtree_index(gindex),
        root=state.hash_tree_root(),
    )


@with_test_suite_name("BeaconState")
@with_simplex_and_later
@spec_state_test
def test_simplex_current_sync_committee_merkle_proof(spec, state):
    gindex = latest_current_sync_committee_gindex(spec)
    assert gindex == spec.CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    yield from _yield_proof(spec, state, state.current_sync_committee.hash_tree_root(), gindex)


@with_test_suite_name("BeaconState")
@with_simplex_and_later
@spec_state_test
def test_simplex_next_sync_committee_merkle_proof(spec, state):
    gindex = latest_next_sync_committee_gindex(spec)
    assert gindex == spec.NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    yield from _yield_proof(spec, state, state.next_sync_committee.hash_tree_root(), gindex)


@with_test_suite_name("BeaconState")
@with_simplex_and_later
@spec_state_test
def test_simplex_finality_root_merkle_proof(spec, state):
    gindex = latest_finalized_root_gindex(spec)
    assert gindex == spec.FINALIZED_ROOT_GINDEX_SIMPLEX
    yield from _yield_proof(spec, state, state.finalized_checkpoint.root, gindex)

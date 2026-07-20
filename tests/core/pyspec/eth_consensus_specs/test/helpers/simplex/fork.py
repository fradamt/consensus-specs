from eth_consensus_specs.test.helpers.constants import SIMPLEX

SIMPLEX_FORK_TEST_META_TAGS = {
    "fork": SIMPLEX,
}


def run_fork_test(post_spec, pre_state):
    yield "pre", pre_state

    pre_balances = list(pre_state.balances)
    post_state = post_spec.upgrade_to_simplex(pre_state)

    assert pre_state.balances == pre_balances
    assert pre_state.fork.current_version == post_state.fork.previous_version
    assert post_state.fork.current_version == post_spec.config.SIMPLEX_FORK_VERSION
    assert post_state.fork.epoch == post_spec.get_current_epoch(post_state)
    assert post_state.slot == pre_state.slot
    assert post_state.validators == pre_state.validators
    assert len(post_state.previous_round_participation) == len(post_state.validators)
    assert len(post_state.current_round_participation) == len(post_state.validators)
    assert not any(post_state.previous_round_participation)
    assert not any(post_state.current_round_participation)
    assert post_state.current_height_target == post_spec.Checkpoint()
    assert len(post_state.target_participation) == len(post_state.validators)
    assert len(post_state.timeouts) == len(post_state.validators)
    assert len(post_state.finality_participation) == len(post_state.validators)
    assert len(post_state.round_double_vote_penalized) == len(post_state.validators)
    assert post_state.current_height == post_spec.GENESIS_HEIGHT
    assert post_state.justified_height == post_spec.Height(0)
    assert post_state.finalized_height == post_spec.Height(0)

    yield "post", post_state
    return post_state

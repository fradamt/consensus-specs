from eth_consensus_specs.test.context import (
    low_balances,
    misc_balances,
    spec_test,
    with_custom_state,
    with_phases,
    with_state,
)
from eth_consensus_specs.test.helpers.constants import GLOAS, SIMPLEX
from eth_consensus_specs.test.helpers.simplex.fork import (
    run_fork_test,
    SIMPLEX_FORK_TEST_META_TAGS,
)
from eth_consensus_specs.test.helpers.state import next_epoch, next_epoch_via_block
from eth_consensus_specs.test.utils import with_meta_tags


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@with_meta_tags(SIMPLEX_FORK_TEST_META_TAGS)
def test_fork_base_state(spec, phases, state):
    yield from run_fork_test(phases[SIMPLEX], state)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@with_meta_tags(SIMPLEX_FORK_TEST_META_TAGS)
def test_fork_next_epoch(spec, phases, state):
    next_epoch(spec, state)
    yield from run_fork_test(phases[SIMPLEX], state)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@spec_test
@with_state
@with_meta_tags(SIMPLEX_FORK_TEST_META_TAGS)
def test_fork_next_epoch_with_block(spec, phases, state):
    next_epoch_via_block(spec, state)
    yield from run_fork_test(phases[SIMPLEX], state)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@with_custom_state(balances_fn=low_balances, threshold_fn=lambda spec: spec.config.EJECTION_BALANCE)
@spec_test
@with_meta_tags(SIMPLEX_FORK_TEST_META_TAGS)
def test_fork_random_low_balances(spec, phases, state):
    yield from run_fork_test(phases[SIMPLEX], state)


@with_phases(phases=[GLOAS], other_phases=[SIMPLEX])
@with_custom_state(
    balances_fn=misc_balances, threshold_fn=lambda spec: spec.config.EJECTION_BALANCE
)
@spec_test
@with_meta_tags(SIMPLEX_FORK_TEST_META_TAGS)
def test_fork_random_misc_balances(spec, phases, state):
    yield from run_fork_test(phases[SIMPLEX], state)

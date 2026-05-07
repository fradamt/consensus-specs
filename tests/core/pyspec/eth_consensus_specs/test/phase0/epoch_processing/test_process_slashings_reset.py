from eth_consensus_specs.test.context import spec_state_test, with_all_phases_from_to
from eth_consensus_specs.test.helpers.constants import GLOAS, PHASE0
from eth_consensus_specs.test.helpers.epoch_processing import run_epoch_processing_with


def run_process_slashings_reset(spec, state):
    yield from run_epoch_processing_with(spec, state, "process_slashings_reset")


# Gloas removed `process_slashings_reset` (rolling slashings vector replaced
# by per-validator `slashing_epoch`).
@with_all_phases_from_to(PHASE0, GLOAS)
@spec_state_test
def test_flush_slashings(spec, state):
    next_epoch = spec.get_current_epoch(state) + 1
    state.slashings[next_epoch % spec.EPOCHS_PER_SLASHINGS_VECTOR] = 100
    assert state.slashings[next_epoch % spec.EPOCHS_PER_SLASHINGS_VECTOR] != 0

    yield from run_process_slashings_reset(spec, state)

    assert state.slashings[next_epoch % spec.EPOCHS_PER_SLASHINGS_VECTOR] == 0

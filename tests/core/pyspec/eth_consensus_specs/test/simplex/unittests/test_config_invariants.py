from eth_consensus_specs.test.context import (
    single_phase,
    spec_test,
    with_simplex_and_later,
)


@with_simplex_and_later
@spec_test
@single_phase
def test_round_schedule(spec):
    schedule = sorted(spec.config.ROUND_SCHEDULE, key=lambda entry: entry["SLOT"])
    # No two entries share an activation slot
    slots = [entry["SLOT"] for entry in schedule]
    assert len(set(slots)) == len(slots)
    prev_slot = 0
    prev_start_round = 0
    prev_slots_per_round = spec.SLOTS_PER_EPOCH
    for entry in schedule:
        # Activation slots are epoch-aligned and round length divides an epoch
        assert entry["SLOT"] % spec.SLOTS_PER_EPOCH == 0
        assert spec.SLOTS_PER_EPOCH % entry["SLOTS_PER_ROUND"] == 0
        # START_ROUND matches the round index accumulated over preceding eras
        expected_start_round = (
            prev_start_round + (entry["SLOT"] - prev_slot) // prev_slots_per_round
        )
        assert entry["START_ROUND"] == expected_start_round
        prev_slot = entry["SLOT"]
        prev_start_round = entry["START_ROUND"]
        prev_slots_per_round = entry["SLOTS_PER_ROUND"]

from eth_consensus_specs.test.helpers.constants import (
    GLOAS,
)

GLOAS_FORK_TEST_META_TAGS = {
    "fork": GLOAS,
}


def run_fork_test(post_spec, pre_state):
    yield "pre", pre_state

    post_state = post_spec.upgrade_to_gloas(pre_state)

    # Stable fields
    stable_fields = [
        "genesis_time",
        "genesis_validators_root",
        "slot",
        "latest_block_header",
        "block_roots",
        "state_roots",
        "historical_roots",
        "eth1_data",
        "eth1_data_votes",
        "eth1_deposit_index",
        # `validators` is checked in `stable_validator_fields` below; the
        # Validator container has a renamed field (`activation_eligibility_epoch`
        # → `slashing_epoch`) so the wholesale list comparison would fail.
        "balances",
        "randao_mixes",
        "slashings",
        "previous_epoch_participation",
        "current_epoch_participation",
        "justification_bits",
        "previous_justified_checkpoint",
        "current_justified_checkpoint",
        "finalized_checkpoint",
        "inactivity_scores",
        "current_sync_committee",
        "next_sync_committee",
        "next_withdrawal_index",
        "next_withdrawal_validator_index",
        "historical_summaries",
        "deposit_requests_start_index",
        "deposit_balance_to_consume",
        "exit_balance_to_consume",
        "earliest_exit_epoch",
        "consolidation_balance_to_consume",
        "earliest_consolidation_epoch",
        "pending_partial_withdrawals",
        "pending_consolidations",
        "proposer_lookahead",
    ]
    for field in stable_fields:
        assert getattr(pre_state, field) == getattr(post_state, field)

    # Modified fields
    modified_fields = ["fork"]
    for field in modified_fields:
        assert getattr(pre_state, field) != getattr(post_state, field)

    # Deleted fields
    deleted_fields = ["latest_execution_payload_header"]
    for field in deleted_fields:
        assert not hasattr(post_state, field)

    assert len(pre_state.validators) == len(post_state.validators)
    for pre_validator, post_validator in zip(pre_state.validators, post_state.validators):
        stable_validator_fields = [
            "pubkey",
            "withdrawal_credentials",
            "slashed",
            "activation_epoch",
            "exit_epoch",
            "withdrawable_epoch",
        ]
        for field in stable_validator_fields:
            assert getattr(pre_validator, field) == getattr(post_validator, field)
        # `activation_eligibility_epoch` is replaced by `slashing_epoch`.
        # For non-slashed validators it must be FAR_FUTURE_EPOCH; for slashed
        # validators it is derived from withdrawable_epoch (must be a finite
        # epoch consistent with their withdrawable_epoch).
        if post_validator.slashed:
            assert post_validator.slashing_epoch < post_spec.FAR_FUTURE_EPOCH
        else:
            assert post_validator.slashing_epoch == post_spec.FAR_FUTURE_EPOCH
        assert not hasattr(post_validator, "activation_eligibility_epoch")

    assert pre_state.fork.current_version == post_state.fork.previous_version
    assert post_state.fork.current_version == post_spec.config.GLOAS_FORK_VERSION
    assert post_state.fork.epoch == post_spec.get_current_epoch(post_state)

    yield "post", post_state

    return post_state

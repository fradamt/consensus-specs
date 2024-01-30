# EIP7251 - Spec

## Table of contents

<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Constants](#constants)
  - [Withdrawal prefixes](#withdrawal-prefixes)
  - [Domains](#domains)
- [Presets](#presets)
  - [Gwei values](#gwei-values)
  - [Rewards and penalties](#rewards-and-penalties)
  - [Max operations per block](#max-operations-per-block)
  - [State list lengths](#state-list-lengths)
- [Configuration](#configuration)
  - [Validator cycle](#validator-cycle)
- [Containers](#containers)
  - [New containers](#new-containers)
    - [New `PendingBalanceDeposit`](#new-pendingbalancedeposit)
    - [New `PartialWithdrawal`](#new-partialwithdrawal)
    - [New `ExecutionLayerWithdrawRequest`](#new-executionlayerwithdrawrequest)
    - [New `Consolidation`](#new-consolidation)
    - [New `SignedConsolidation`](#new-signedconsolidation)
    - [New `PendingConsolidation`](#new-pendingconsolidation)
  - [Extended Containers](#extended-containers)
    - [`BeaconState`](#beaconstate)
  - [`BeaconBlockBody`](#beaconblockbody)
- [Helpers](#helpers)
  - [Predicates](#predicates)
    - [Updated `is_eligible_for_activation_queue`](#updated-is_eligible_for_activation_queue)
    - [New `has_compounding_withdrawal_credential`](#new-has_compounding_withdrawal_credential)
    - [Updated  `is_fully_withdrawable_validator`](#updated--is_fully_withdrawable_validator)
    - [Updated  `is_partially_withdrawable_validator`](#updated--is_partially_withdrawable_validator)
  - [Beacon state accessors](#beacon-state-accessors)
    - [New `get_validator_excess_balance`](#new-get_validator_excess_balance)
    - [New  `get_churn_limit`](#new--get_churn_limit)
    - [New `get_activation_exit_churn_limit`](#new-get_activation_exit_churn_limit)
    - [New `get_consolidation_churn_limit`](#new-get_consolidation_churn_limit)
  - [Beacon state mutators](#beacon-state-mutators)
    - [Updated  `initiate_validator_exit`](#updated--initiate_validator_exit)
    - [New `compute_exit_epoch_and_update_churn`](#new-compute_exit_epoch_and_update_churn)
    - [New `compute_consolidation_epoch_and_update_churn`](#new-compute_consolidation_epoch_and_update_churn)
    - [Updated `slash_validator`](#updated-slash_validator)
- [Beacon chain state transition function](#beacon-chain-state-transition-function)
  - [Epoch processing](#epoch-processing)
    - [Updated `process_epoch`](#updated-process_epoch)
    - [Updated  `process_registry_updates`](#updated--process_registry_updates)
    - [New `process_pending_balance_deposits`](#new-process_pending_balance_deposits)
    - [Updated `process_effective_balance_updates`](#updated-process_effective_balance_updates)
  - [Block processing](#block-processing)
    - [Operations](#operations)
      - [Updated `process_operations`](#updated-process_operations)
      - [Deposits](#deposits)
        - [Updated  `apply_deposit`](#updated--apply_deposit)
        - [Updated `get_validator_from_deposit`](#updated-get_validator_from_deposit)
      - [Withdrawals](#withdrawals)
        - [New `process_execution_layer_withdraw_request`](#new-process_execution_layer_withdraw_request)
        - [Updated  `get_expected_withdrawals`](#updated--get_expected_withdrawals)
      - [Consolidations](#consolidations)
        - [New ```process_consolidation```](#new-process_consolidation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->

## Introduction

See [a modest proposal](https://notes.ethereum.org/@mikeneuder/increase-maxeb), the [diff view](https://github.com/michaelneuder/consensus-specs/pull/3/files) and 
[security considerations](https://notes.ethereum.org/@fradamt/meb-increase-security).

*Note:* This specification is built upon [Deneb](../../deneb/beacon-chain.md).

## Constants

The following values are (non-configurable) constants used throughout the specification.

### Withdrawal prefixes

| Name | Value |
| - | - |
| `BLS_WITHDRAWAL_PREFIX` | `Bytes1('0x00')` |
| `ETH1_ADDRESS_WITHDRAWAL_PREFIX` | `Bytes1('0x01')` |
| `COMPOUNDING_WITHDRAWAL_PREFIX` | `Bytes1('0x02')` |

### Domains

| Name | Value |
| - | - |
| `DOMAIN_CONSOLIDATION` | `DomainType('0x0B000000')` |

## Presets

### Gwei values

| Name | Value |
| - | - |
| `MIN_ACTIVATION_BALANCE` | `Gwei(2**5 * 10**9)`  (= 32,000,000,000) |
| `MAX_EFFECTIVE_BALANCE_EIP7251` | `Gwei(2**11 * 10**9)` (= 2048,000,000,000) |

### Rewards and penalties

| Name | Value |
| - | - |
| `MIN_SLASHING_PENALTY_QUOTIENT_EIP7251` | `Gwei(2**16)`  (= 65,536) |

### Max operations per block

| Name | Value |
| - | - |
| `MAX_CONSOLIDATIONS` | `uint64(1)` |

### State list lengths

| Name | Value | Unit | Duration |
| - | - | :-: | :-: |
| `PENDING_BALANCE_DEPOSITS_LIMIT` | `uint64(2**27)` (= 134,217,728) | pending balance deposits |  |
| `PENDING_PARTIAL_WITHDRAWALS_LIMIT` | `uint64(2**27)` (= 134,217,728) | pending partial withdrawals |  |
| `PENDING_CONSOLIDATIONS_LIMIT` | `uint64(2**18)` (= 262,144) | pending consolidations |  |


## Configuration

### Validator cycle

| Name | Value |
| - | - |
| `MIN_PER_EPOCH_CHURN_LIMIT_EIP7251` | `Gwei(2**7 * 10**9)` (= 128,000,000,000) | # Equivalent to 4 32 ETH validators
| `MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT` | `Gwei(2**8 * 10**9)` (256,000,000,000) |


## Containers

### New containers

#### New `PendingBalanceDeposit`

```python
class PendingBalanceDeposit(Container):
    index: ValidatorIndex
    amount: Gwei
```

#### New `PartialWithdrawal`

```python
class PartialWithdrawal(Container):
    index: ValidatorIndex
    amount: Gwei
    withdrawable_epoch: Epoch
```
#### New `ExecutionLayerWithdrawRequest`

```python
class ExecutionLayerWithdrawRequest(Container):
    source_address: ExecutionAddress
    validator_pubkey: BLSPubkey
    amount: Gwei
```

#### New `Consolidation`

```python
class Consolidation(Container):
    source_index: ValidatorIndex
    target_index: ValidatorIndex
    epoch: Epoch
```

#### New `SignedConsolidation`
```python
class SignedConsolidation(Container):
    message: Consolidation
    signature: BLSSignature
```

#### New `PendingConsolidation`
```python
class PendingConsolidation(Container):
    source_index: ValidatorIndex
    target_index: ValidatorIndex
```





### Extended Containers

#### `BeaconState`

```python
class BeaconState(Container):
    # Versioning
    genesis_time: uint64
    genesis_validators_root: Root
    slot: Slot
    fork: Fork
    # History
    latest_block_header: BeaconBlockHeader
    block_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT]
    state_roots: Vector[Root, SLOTS_PER_HISTORICAL_ROOT]
    historical_roots: List[Root, HISTORICAL_ROOTS_LIMIT]
    # Eth1
    eth1_data: Eth1Data
    eth1_data_votes: List[Eth1Data, EPOCHS_PER_ETH1_VOTING_PERIOD * SLOTS_PER_EPOCH]
    eth1_deposit_index: uint64
    # Registry
    validators: List[Validator, VALIDATOR_REGISTRY_LIMIT]
    balances: List[Gwei, VALIDATOR_REGISTRY_LIMIT]
    # Randomness
    randao_mixes: Vector[Bytes32, EPOCHS_PER_HISTORICAL_VECTOR]
    # Slashings
    slashings: Vector[Gwei, EPOCHS_PER_SLASHINGS_VECTOR]  # Per-epoch sums of slashed effective balances
    # Participation
    previous_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    current_epoch_participation: List[ParticipationFlags, VALIDATOR_REGISTRY_LIMIT]
    # Finality
    justification_bits: Bitvector[JUSTIFICATION_BITS_LENGTH]  # Bit set for every recent justified epoch
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    # Inactivity
    inactivity_scores: List[uint64, VALIDATOR_REGISTRY_LIMIT]
    # Sync
    current_sync_committee: SyncCommittee
    next_sync_committee: SyncCommittee
    # Execution
    latest_execution_payload_header: ExecutionPayloadHeader
    # Withdrawals
    next_withdrawal_index: WithdrawalIndex
    next_withdrawal_validator_index: ValidatorIndex
    # Deep history valid from Capella onwards
    historical_summaries: List[HistoricalSummary, HISTORICAL_ROOTS_LIMIT]
    # --- New in EIP7251--- #
    deposit_balance_to_consume: Gwei
    exit_balance_to_consume: Gwei
    earliest_exit_epoch: Epoch
    consolidation_balance_to_consume: Gwei
    earliest_consolidation_epoch: Epoch
    pending_balance_deposits: List[PendingBalanceDeposit, 100000]
    pending_partial_withdrawals: List[PartialWithdrawal, 100000]
    pending_consolidations: List[PendingConsolidation, PENDING_CONSOLIDATIONS_LIMIT]
```

### `BeaconBlockBody`

```python
class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    eth1_data: Eth1Data  # Eth1 data vote
    graffiti: Bytes32  # Arbitrary data
    # Operations
    proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS]
    attestations: List[Attestation, MAX_ATTESTATIONS]
    deposits: List[Deposit, MAX_DEPOSITS]
    voluntary_exits: List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]
    sync_aggregate: SyncAggregate
    # Execution
    execution_payload: ExecutionPayload 
    bls_to_execution_changes: List[SignedBLSToExecutionChange, MAX_BLS_TO_EXECUTION_CHANGES]
    blob_kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    consolidations: List[SignedConsolidation, MAX_CONSOLIDATIONS]  # [New in EIP7251]
```

## Helpers

### Predicates

#### Updated `is_eligible_for_activation_queue`

*Note*: Use `>= MIN_ACTIVATION_BALANCE` instead of `== MAX_EFFECTIVE_BALANCE`

```python
def is_eligible_for_activation_queue(validator: Validator) -> bool:
    """
    Check if ``validator`` is eligible to be placed into the activation queue.
    """
    return (
        validator.activation_eligibility_epoch == FAR_FUTURE_EPOCH
        and validator.effective_balance >= MIN_ACTIVATION_BALANCE # [Modified in EIP7251]
    )
```

#### New `has_compounding_withdrawal_credential`

```python
def has_compounding_withdrawal_credential(validator: Validator) -> bool:
    """
    Check if ``validator`` has an 0x02 prefixed "compounding" withdrawal credential.
    """
    return validator.withdrawal_credentials[:1] == COMPOUNDING_WITHDRAWAL_PREFIX
```

#### Updated  `is_fully_withdrawable_validator`

*Note*: now calls `has_compounding_withdrawal_credential` too.  

```python
def is_fully_withdrawable_validator(validator: Validator, balance: Gwei, epoch: Epoch) -> bool:
    """
    Check if ``validator`` is fully withdrawable.
    """
    return (
        (has_eth1_withdrawal_credential(validator) or has_compounding_withdrawal_credential(validator)) # [Modified in EIP7251]
        and validator.withdrawable_epoch <= epoch
        and balance > 0
    )
```

####  Updated  `is_partially_withdrawable_validator`
*Note*: now calls `has_compounding_withdrawal_credential` and gets ceiling from `get_balance_ceiling`.

```python
def is_partially_withdrawable_validator(validator: Validator, balance: Gwei) -> bool:
    """
    Check if ``validator`` is partially withdrawable.
    """
    if not (has_eth1_withdrawal_credential(validator) or has_compounding_withdrawal_credential(validator)):
        return False
    return get_validator_excess_balance(validator, balance) > 0
```


### Beacon state accessors


#### New `get_validator_excess_balance`

```python
def get_validator_excess_balance(validator: Validator, balance: Gwei) -> Gwei:
    """
    Get excess balance for partial withdrawals for ``validator``.
    """
    if has_compounding_withdrawal_credential(validator) and balance > MAX_EFFECTIVE_BALANCE_EIP7251:
        return balance - MAX_EFFECTIVE_BALANCE_EIP7251
    elif has_eth1_withdrawal_credential(validator) and balance > MIN_ACTIVATION_BALANCE:
        return balance - MIN_ACTIVATION_BALANCE
    return Gwei(0)
```

#### New  `get_churn_limit`

*Note*: Updated to return a Gwei amount of amount of churn per epoch.

```python
def get_churn_limit(state: BeaconState) -> Gwei:
    """
    Return the churn limit for the current epoch.
    """
    churn = max(MIN_PER_EPOCH_CHURN_LIMIT_EIP7251, 
                get_total_active_balance(state) // CHURN_LIMIT_QUOTIENT)
    return churn - churn % EFFECTIVE_BALANCE_INCREMENT
```

#### New `get_activation_exit_churn_limit`
```python
def get_activation_exit_churn_limit(state: BeaconState) -> Gwei:
    """
    Return the churn limit for the current epoch dedicated to activations and exits.
    """
    return min(MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT, get_churn_limit(state))
```

#### New `get_consolidation_churn_limit`
```python
def get_consolidation_churn_limit(state: BeaconState) -> Gwei:
    return get_churn_limit(state) - get_activation_exit_churn_limit(state)
```



### Beacon state mutators

#### Updated  `initiate_validator_exit`

*Note*: Modification to make validator exits constrained by the balance
of the exiting validators. 

```python
def initiate_validator_exit(state: BeaconState, index: ValidatorIndex) -> None:
    """
    Initiate the exit of the validator with index ``index``.
    """
    # Return if validator already initiated exit
    validator = state.validators[index]
    if validator.exit_epoch != FAR_FUTURE_EPOCH:
        return

    # Compute exit queue epoch [Modified in EIP 7251]
    exit_queue_epoch = compute_exit_epoch_and_update_churn(state, validator.effective_balance)

    # Set validator exit epoch and withdrawable epoch
    validator.exit_epoch = exit_queue_epoch
    validator.withdrawable_epoch = Epoch(validator.exit_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY)
```

#### New `compute_exit_epoch_and_update_churn`


```python
def compute_exit_epoch_and_update_churn(state: BeaconState, exit_balance: Gwei) -> Epoch:
    earliest_exit_epoch = compute_activation_exit_epoch(get_current_epoch(state))
    per_epoch_churn = get_activation_exit_churn_limit(state)
    # New epoch for exits.
    if state.earliest_exit_epoch < earliest_exit_epoch:
        state.earliest_exit_epoch = earliest_exit_epoch
        state.exit_balance_to_consume = per_epoch_churn

    # Exit fits in the current earliest epoch.
    if exit_balance <= state.exit_balance_to_consume:
        state.exit_balance_to_consume -= exit_balance
    else: # Exit doesn't fit in the current earliest epoch.
        balance_to_process = exit_balance - state.exit_balance_to_consume
        additional_epochs, remainder = divmod(balance_to_process, per_epoch_churn)
        state.earliest_exit_epoch += additional_epochs + 1
        state.exit_balance_to_consume = per_epoch_churn - remainder
    return state.earliest_exit_epoch
```

#### New `compute_consolidation_epoch_and_update_churn`

```python
def compute_consolidation_epoch_and_update_churn(state: BeaconState, consolidation_balance: Gwei) -> Epoch:
    earliest_consolidation_epoch = compute_activation_exit_epoch(get_current_epoch(state))
    per_epoch_consolidation_churn = get_consolidation_churn_limit(state)
    # New epoch for consolidations.
    if state.earliest_consolidation_epoch < earliest_consolidation_epoch:
        state.earliest_consolidation_epoch = earliest_consolidation_epoch
        state.consolidation_balance_to_consume = per_epoch_consolidation_churn
    # Consolidation fits in the current earliest consolidation epoch.
    if consolidation_balance <= state.consolidation_balance_to_consume:
        state.consolidation_balance_to_consume -= consolidation_balance
    else: # Consolidation doesn't fit in the current earliest epoch.
        balance_to_process = consolidation_balance - state.consolidation_balance_to_consume
        additional_epochs, remainder = divmod(balance_to_process, per_epoch_consolidation_churn)
        state.earliest_consolidation_epoch += additional_epochs + 1
        state.consolidation_balance_to_consume = per_epoch_consolidation_churn - remainder
    return state.earliest_consolidation_epoch
```

#### Updated `slash_validator`

```python
def slash_validator(state: BeaconState,
                    slashed_index: ValidatorIndex,
                    whistleblower_index: ValidatorIndex=None) -> None:
    """
    Slash the validator with index ``slashed_index``.
    """
    epoch = get_current_epoch(state)
    initiate_validator_exit(state, slashed_index)
    validator = state.validators[slashed_index]
    validator.slashed = True
    validator.withdrawable_epoch = max(validator.withdrawable_epoch, Epoch(epoch + EPOCHS_PER_SLASHINGS_VECTOR))
    state.slashings[epoch % EPOCHS_PER_SLASHINGS_VECTOR] += validator.effective_balance
    slashing_penalty = validator.effective_balance // MIN_SLASHING_PENALTY_QUOTIENT_EIP7251  # [Modified in EIP7251]
    decrease_balance(state, slashed_index, slashing_penalty)
```


## Beacon chain state transition function

### Epoch processing

#### Updated `process_epoch`
```python
def process_epoch(state: BeaconState) -> None:
    process_justification_and_finalization(state)
    process_inactivity_updates(state)
    process_rewards_and_penalties(state)
    process_registry_updates(state)
    process_slashings(state)
    process_eth1_data_reset(state)
    process_pending_balance_deposits(state) # New in EIP7251
    process_pending_consolidations(state) # New in EIP7251
    process_effective_balance_updates(state)
    process_slashings_reset(state)
    process_randao_mixes_reset(state)
```

#### Updated  `process_registry_updates`
*Note*: changing the dequed validators to depend on the weight of activation up to the
churn limit. 
```python
def process_registry_updates(state: BeaconState) -> None:
    # Process activation eligibility and ejections
    for index, validator in enumerate(state.validators):
        if is_eligible_for_activation_queue(validator):
            validator.activation_eligibility_epoch = get_current_epoch(state) + 1
        if (
            is_active_validator(validator, get_current_epoch(state))
            and validator.effective_balance <= EJECTION_BALANCE
        ):
            initiate_validator_exit(state, ValidatorIndex(index))

    # Activate all eligible validators
    activation_epoch = compute_activation_exit_epoch(get_current_epoch(state))
    for validator in state.validators:
        if is_eligible_for_activation(state, validator):
            validator.activation_epoch = activation_epoch
```

#### New `process_pending_balance_deposits`

```python
def process_pending_balance_deposits(state: BeaconState) -> None:
    state.deposit_balance_to_consume += get_activation_exit_churn_limit(state)
    next_pending_deposit_index = 0
    for pending_balance_deposit in state.pending_balance_deposits:
        if state.deposit_balance_to_consume < pending_balance_deposit.amount:
            break

        state.deposit_balance_to_consume -= pending_balance_deposit.amount
        increase_balance(state, pending_balance_deposit.index, pending_balance_deposit.amount)
        next_pending_deposit_index += 1

    state.pending_balance_deposits = state.pending_balance_deposits[next_pending_deposit_index:]
```

```python
def get_active_balance(state: BeaconState, validator_index: ValidatorIndex) -> Gwei:
    active_balance_ceil = (
        MIN_ACTIVATION_BALANCE 
        if has_eth1_withdrawal_credential(state.validators[validator_index]) 
        else MAX_EFFECTIVE_BALANCE_EIP7251
    )
    return min(state.balances[validator_index], active_balance_ceil)
```

```python
def apply_pending_consolidation(state: BeaconState, pending_consolidation: PendingConsolidation) -> None:
    # Move active balance to target. Excess balance is withdrawable.
    active_balance = get_active_balance(state, pending_consolidation.source_index)
    state.balances[pending_consolidation.source_index] -= active_balance
    state.balances[pending_consolidation.target_index] += active_balance
```

```python
def process_pending_consolidations(state: BeaconState) -> None:
    next_pending_consolidation = 0
    for pending_consolidation in state.pending_consolidations:
        source_validator = state.validators[pending_consolidation.source_index]
        if source_validator.slashed:
            next_pending_consolidation += 1
            continue
        if source_validator.withdrawable_epoch > get_current_epoch(state):
            break

        next_pending_consolidation += 1
        apply_pending_consolidation(state, pending_consolidation)

    state.pending_consolidations = state.pending_consolidations[next_pending_consolidation:]
```


#### Updated `process_effective_balance_updates`

```python
def process_effective_balance_updates(state: BeaconState) -> None:
    # Update effective balances with hysteresis
    for index, validator in enumerate(state.validators):
        balance = state.balances[index]
        HYSTERESIS_INCREMENT = uint64(EFFECTIVE_BALANCE_INCREMENT // HYSTERESIS_QUOTIENT)
        DOWNWARD_THRESHOLD = HYSTERESIS_INCREMENT * HYSTERESIS_DOWNWARD_MULTIPLIER
        UPWARD_THRESHOLD = HYSTERESIS_INCREMENT * HYSTERESIS_UPWARD_MULTIPLIER
        EFFECTIVE_BALANCE_LIMIT = MAX_EFFECTIVE_BALANCE_EIP7251 if has_compounding_withdrawal_credential(validator) else MIN_ACTIVATION_BALANCE # Modified in EIP7251
        if (
            balance + DOWNWARD_THRESHOLD < validator.effective_balance
            or validator.effective_balance + UPWARD_THRESHOLD < balance
        ):
            validator.effective_balance = min(balance - balance % EFFECTIVE_BALANCE_INCREMENT, EFFECTIVE_BALANCE_LIMIT)
```

### Block processing

#### Operations 

##### Updated `process_operations`

```python
def process_operations(state: BeaconState, body: BeaconBlockBody) -> None:
    # Verify that outstanding deposits are processed up to the maximum number of deposits
    assert len(body.deposits) == min(MAX_DEPOSITS, state.eth1_data.deposit_count - state.eth1_deposit_index)
    def for_ops(operations: Sequence[Any], fn: Callable[[BeaconState, Any], None]) -> None:
        for operation in operations:
            fn(state, operation)
    for_ops(body.proposer_slashings, process_proposer_slashing)
    for_ops(body.attester_slashings, process_attester_slashing)
    for_ops(body.attestations, process_attestation)
    for_ops(body.deposits, process_deposit)
    for_ops(body.voluntary_exits, process_voluntary_exit)
    for_ops(body.bls_to_execution_changes, process_bls_to_execution_change) 
    for_ops(body.execution_payload.withdraw_requests, process_execution_layer_withdraw_request) # New in EIP7251
    for_ops(body.consolidations, process_consolidation) # New in EIP7251
```

##### Deposits

###### Updated  `apply_deposit`

*Note*: Updated to cap top-offs at 32 ETH to avoid skipping activation queue.

```python
def apply_deposit(state: BeaconState,
                  pubkey: BLSPubkey,
                  withdrawal_credentials: Bytes32,
                  amount: uint64,
                  signature: BLSSignature) -> None:
    validator_pubkeys = [validator.pubkey for validator in state.validators]
    if pubkey not in validator_pubkeys:
        # Verify the deposit signature (proof of possession) which is not checked by the deposit contract
        deposit_message = DepositMessage(
            pubkey=pubkey,
            withdrawal_credentials=withdrawal_credentials,
            amount=amount,
        )
        domain = compute_domain(DOMAIN_DEPOSIT)  # Fork-agnostic domain since deposits are valid across forks
        signing_root = compute_signing_root(deposit_message, domain)
        # Initialize validator if the deposit signature is valid
        if bls.Verify(pubkey, signing_root, signature):
            state.validators.append(get_validator_from_deposit(pubkey, withdrawal_credentials))
            state.balances.append(0)
            state.previous_epoch_participation.append(ParticipationFlags(0b0000_0000))
            state.current_epoch_participation.append(ParticipationFlags(0b0000_0000))
            state.inactivity_scores.append(uint64(0))
            index = len(state.validators) - 1
            state.pending_balance_deposits.append(PendingBalanceDeposit(index=index, amount=amount))
    else:
        index = ValidatorIndex(validator_pubkeys.index(pubkey))
        state.pending_balance_deposits.append(PendingBalanceDeposit(index=index, amount=amount))
```

###### Updated `get_validator_from_deposit`

```python
def get_validator_from_deposit(pubkey: BLSPubkey, withdrawal_credentials: Bytes32) -> Validator:
    return Validator(
        pubkey=pubkey,
        withdrawal_credentials=withdrawal_credentials,
        activation_eligibility_epoch=FAR_FUTURE_EPOCH,
        activation_epoch=FAR_FUTURE_EPOCH,
        exit_epoch=FAR_FUTURE_EPOCH,
        withdrawable_epoch=FAR_FUTURE_EPOCH,
        effective_balance=0, # [Modified in EIP7251]
    )
```

##### Withdrawals 

###### New `process_execution_layer_withdraw_request`

```python
def process_execution_layer_withdraw_request(
        state: BeaconState,
        execution_layer_withdraw_request: ExecutionLayerWithdrawRequest
    ) -> None:
    amount = execution_layer_withdraw_request.amount
    is_full_exit_request = amount == 0
    # If partial withdrawal queue is full, only full exits are processed 
    if not (is_full_exit_request or len(state.pending_consolidations) < PENDING_PARTIAL_WITHDRAWALS_LIMIT):
        return

    validator_pubkeys = [v.pubkey for v in state.validators]
    validator_index = ValidatorIndex(validator_pubkeys.index(execution_layer_withdraw_request.validator_pubkey))
    validator = state.validators[validator_index]


    # Same conditions as in EIP7002 https://github.com/ethereum/consensus-specs/pull/3349/files#diff-7a6e2ba480d22d8bd035bd88ca91358456caf9d7c2d48a74e1e900fe63d5c4f8R223
    # Verify withdrawal credentials
    is_execution_address = has_eth1_withdrawal_credential(validator) or has_compounding_withdrawal_credential(validator)
    is_correct_source_address = validator.withdrawal_credentials[12:] == execution_layer_withdraw_request.source_address
    if not (is_execution_address and is_correct_source_address):
        return
    # Verify the validator is active
    if not is_active_validator(validator, get_current_epoch(state)):
        return
    # Verify exit has not been initiated, and slashed
    if validator.exit_epoch != FAR_FUTURE_EPOCH:
        return
    # Verify the validator has been active long enough
    if get_current_epoch(state) < validator.activation_epoch + SHARD_COMMITTEE_PERIOD:
        return

    # New condition: only allow partial withdrawals with compounding withdrawal credentials
    if not (is_full_exit_request or has_compounding_withdrawal_credential(validator)):
        return

    pending_balance_to_withdraw = sum(item.amount for item in state.pending_partial_withdrawals if item.index == validator_index)
    # only exit validator if it has no pending withdrawals in the queue
    if is_full_exit_request and pending_balance_to_withdraw == 0:
        initiate_validator_exit(state, validator_index)
    elif state.balances[validator_index] > MIN_ACTIVATION_BALANCE + pending_balance_to_withdraw:
        to_withdraw = min(state.balances[validator_index] - MIN_ACTIVATION_BALANCE - pending_balance_to_withdraw, amount)
        exit_queue_epoch = compute_exit_epoch_and_update_churn(state, to_withdraw)
        withdrawable_epoch = Epoch(exit_queue_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY)
        state.pending_partial_withdrawals.append(PartialWithdrawal(
            index=validator_index,
            amount=to_withdraw,
            withdrawable_epoch=withdrawable_epoch,
        ))
```

######  Updated  `get_expected_withdrawals`

```python
def get_expected_withdrawals(state: BeaconState) -> Sequence[Withdrawal]:
    epoch = get_current_epoch(state)
    withdrawal_index = state.next_withdrawal_index
    validator_index = state.next_withdrawal_validator_index
    withdrawals: List[Withdrawal] = []
    consumed = 0
    for withdrawal in state.pending_partial_withdrawals:
        if withdrawal.withdrawable_epoch > epoch or len(withdrawals) == MAX_WITHDRAWALS_PER_PAYLOAD // 2:
            break

        validator = state.validators[withdrawal.index]
        if validator.exit_epoch == FAR_FUTURE_EPOCH and state.balances[withdrawal.index] > MIN_ACTIVATION_BALANCE:
            withdrawable_balance = min(state.balances[withdrawal.index] - MIN_ACTIVATION_BALANCE, withdrawal.amount)
            withdrawals.append(Withdrawal(
                index=withdrawal_index,
                validator_index=withdrawal.index,
                address=ExecutionAddress(validator.withdrawal_credentials[12:]),
                amount=withdrawable_balance,
            ))
            withdrawal_index += WithdrawalIndex(1)
            consumed += 1

    state.pending_partial_withdrawals = state.pending_partial_withdrawals[consumed:] 

    # Sweep for remaining.
    bound = min(len(state.validators), MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP)
    for _ in range(bound):
        validator = state.validators[validator_index]
        balance = state.balances[validator_index]
        if is_fully_withdrawable_validator(validator, balance, epoch):
            withdrawals.append(Withdrawal(
                index=withdrawal_index,
                validator_index=validator_index,
                address=ExecutionAddress(validator.withdrawal_credentials[12:]),
                amount=balance,
            ))
            withdrawal_index += WithdrawalIndex(1)
        elif is_partially_withdrawable_validator(validator, balance):
            withdrawals.append(Withdrawal(
                index=withdrawal_index,
                validator_index=validator_index,
                address=ExecutionAddress(validator.withdrawal_credentials[12:]),
                amount=get_validator_excess_balance(validator, balance),
            ))
            withdrawal_index += WithdrawalIndex(1)
        if len(withdrawals) == MAX_WITHDRAWALS_PER_PAYLOAD:
            break
        validator_index = ValidatorIndex((validator_index + 1) % len(state.validators))
    return withdrawals
```

##### Consolidations

###### New ```process_consolidation```

```python
def process_consolidation(state: BeaconState, signed_consolidation: SignedConsolidation) -> None:
    # If the pending consolidations queue is full, no consolidations are allowed in the block
    assert len(state.pending_consolidations) < PENDING_CONSOLIDATIONS_LIMIT
    # If there is too little available consolidation churn limit, no consolidations are allowed in the block
    assert get_consolidation_churn_limit(state) > MIN_ACTIVATION_BALANCE
    consolidation = signed_consolidation.message
    # Verify that source != target, so a consolidation cannot be used as an exit.
    assert consolidation.source_index != consolidation.target_index

    source_validator = state.validators[consolidation.source_index]
    target_validator = state.validators[consolidation.target_index]
    # Verify the source and the target are active
    current_epoch = get_current_epoch(state)
    assert is_active_validator(source_validator, current_epoch)
    assert is_active_validator(target_validator, current_epoch)
    # Verify exits for source and target have not been initiated
    assert source_validator.exit_epoch == FAR_FUTURE_EPOCH
    assert target_validator.exit_epoch == FAR_FUTURE_EPOCH
    # Consolidations must specify an epoch when they become valid; they are not valid before then
    assert current_epoch >= consolidation.epoch 

    # Verify the source and the target have Execution layer withdrawal credentials
    assert source_validator.withdrawal_credentials[:1] in (ETH1_ADDRESS_WITHDRAWAL_PREFIX, COMPOUNDING_WITHDRAWAL_PREFIX)
    assert target_validator.withdrawal_credentials[:1] in (ETH1_ADDRESS_WITHDRAWAL_PREFIX, COMPOUNDING_WITHDRAWAL_PREFIX)
    # Verify the same withdrawal address
    assert source_validator.withdrawal_credentials[1:] == target_validator.withdrawal_credentials[1:]

    # Verify consolidation is signed by the source and the target
    domain = compute_domain(DOMAIN_CONSOLIDATION, genesis_validators_root=state.genesis_validators_root)
    signing_root = compute_signing_root(consolidation, domain)
    pubkeys = [source_validator.pubkey, target_validator.pubkey]
    assert bls.FastAggregateVerify(pubkeys, signing_root, signed_consolidation.signature)

    # Initiate source validator exit and append pending consolidation
    active_balance = get_active_balance(state, consolidation.source_index)
    source_validator.exit_epoch = compute_consolidation_epoch_and_update_churn(state, active_balance)
    source_validator.withdrawable_epoch = Epoch(source_validator.exit_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY)
    state.pending_consolidations.append(PendingConsolidation(source_index = consolidation.source_index,
                                                             target_index = consolidation.target_index))
```


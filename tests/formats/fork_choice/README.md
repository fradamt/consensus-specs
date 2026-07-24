# Fork choice tests

The aim of the fork choice tests is to provide test coverage of the various
components of the fork choice.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Test case format](#test-case-format)
  - [`meta.yaml`](#metayaml)
  - [`anchor_state.ssz_snappy`](#anchor_statessz_snappy)
  - [`anchor_block.ssz_snappy`](#anchor_blockssz_snappy)
  - [`steps.yaml`](#stepsyaml)
    - [`on_tick` execution step](#on_tick-execution-step)
    - [`on_attestation` execution step](#on_attestation-execution-step)
    - [`on_block` execution step](#on_block-execution-step)
    - [`on_merge_block` execution step](#on_merge_block-execution-step)
    - [`on_attester_slashing` execution step](#on_attester_slashing-execution-step)
    - [`on_payload_info` execution step](#on_payload_info-execution-step)
    - [`on_execution_payload_envelope` execution step](#on_execution_payload_envelope-execution-step)
    - [`on_payload_attestation_message` execution step](#on_payload_attestation_message-execution-step)
    - [`on_available_attestation` execution step](#on_available_attestation-execution-step)
    - [`on_round_double_vote_evidence` execution step](#on_round_double_vote_evidence-execution-step)
    - [Checks step](#checks-step)
  - [`attestation_<32-byte-root>.ssz_snappy`](#attestation_32-byte-rootssz_snappy)
  - [`block_<32-byte-root>.ssz_snappy`](#block_32-byte-rootssz_snappy)
  - [`execution_payload_envelope_<32-byte-root>.ssz_snappy`](#execution_payload_envelope_32-byte-rootssz_snappy)
  - [`payload_attestation_message_<32-byte-root>.ssz_snappy`](#payload_attestation_message_32-byte-rootssz_snappy)
  - [`available_attestation_<32-byte-root>.ssz_snappy`](#available_attestation_32-byte-rootssz_snappy)
  - [`round_double_vote_evidence_<32-byte-root>.ssz_snappy`](#round_double_vote_evidence_32-byte-rootssz_snappy)
- [Condition](#condition)

<!-- mdformat-toc end -->

## Test case format

### `meta.yaml`

```yaml
description: string    -- Optional. Description of test case, purely for debugging purposes.
bls_setting: int       -- see general test-format spec.
```

### `anchor_state.ssz_snappy`

An SSZ-snappy encoded `BeaconState`, the state to initialize store with
`get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock)`
helper.

### `anchor_block.ssz_snappy`

An SSZ-snappy encoded `BeaconBlock`, the block to initialize store with
`get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock)`
helper.

### `steps.yaml`

The steps to execute in sequence. There may be multiple items of the following
types:

#### `on_tick` execution step

The parameter that is required for executing `on_tick(store, time)`.

```yaml
{
    tick: int       -- to execute `on_tick(store, time)`.
    valid: bool     -- optional, default to `true`.
                       If it's `false`, this execution step is expected to be invalid.
}
```

After this step, the `store` object may have been updated.

#### `on_attestation` execution step

The parameter that is required for executing
`on_attestation(store, attestation)`.

```yaml
{
    attestation: string  -- the name of the `attestation_<32-byte-root>.ssz_snappy` file.
                            To execute `on_attestation(store, attestation)` with the given attestation.
    valid: bool          -- optional, default to `true`.
                            If it's `false`, this execution step is expected to be invalid.
}
```

The file is located in the same folder (see below).

After this step, the `store` object may have been updated.

#### `on_block` execution step

The parameter that is required for executing `on_block(store, block)`.

```yaml
{
    block: string           -- the name of the `block_<32-byte-root>.ssz_snappy` file.
                              To execute `on_block(store, block)` with the given attestation.
    blobs: string           -- optional, the name of the `blobs_<32-byte-root>.ssz_snappy` file.
                               The blobs file content is a `List[Blob, MAX_BLOB_COMMITMENTS_PER_BLOCK]` SSZ object.
    proofs: array of byte48 hex string -- optional, the proofs of blob commitments.
    columns: string        -- optional, array of the names of the `column_<32-byte-root>.ssz_snappy` files.
    valid: bool             -- optional, default to `true`.
                               If it's `false`, this execution step is expected to be invalid.
}
```

The file is located in the same folder (see below).

`blobs` and `proofs` are new fields from Deneb EIP-4844. These fields indicate
the expected values from `retrieve_blobs_and_proofs()` helper inside
`is_data_available()` helper. If these two fields are not provided,
`retrieve_blobs_and_proofs()` returns empty lists.

`columns` is a new field in Fulu EIP-7594. This field indicate the expected
values from `retrieve_column_sidecars` helper inside `is_data_available()`
helper. If this field is an empty array, `retrieve_column_sidecars` should throw
an exception (not enough data sampled). If this field is not provided,
`retrieve_column_sidecars` returns an empty list.

Post-Deneb and pre-Fulu, `columns` should not be present. Post-Fulu `blobs` and
`proofs` should not be present.

After this step, the `store` object may have been updated.

#### `on_merge_block` execution step

Adds `PowBlock` data which is required for executing `on_block(store, block)`.

```yaml
{
    pow_block: string  -- the name of the `pow_block_<32-byte-root>.ssz_snappy` file.
                          To be used in `get_pow_block` lookup
}
```

The file is located in the same folder (see below). PowBlocks should be used as
return values for `get_pow_block(hash: Hash32) -> PowBlock` function if hashes
match.

#### `on_attester_slashing` execution step

The parameter that is required for executing
`on_attester_slashing(store, attester_slashing)`.

```yaml
{
    attester_slashing: string  -- the name of the `attester_slashing_<32-byte-root>.ssz_snappy` file.
                            To execute `on_attester_slashing(store, attester_slashing)` with the given attester slashing.
    valid: bool          -- optional, default to `true`.
                            If it's `false`, this execution step is expected to be invalid.
}
```

The file is located in the same folder (see below).

After this step, the `store` object may have been updated.

#### `on_payload_info` execution step

Optional step for optimistic sync tests.

```yaml
{
    block_hash: string,             -- Encoded 32-byte value of payload's block hash.
    payload_status: {
        status: string,             -- Enum, "VALID" | "INVALID" | "SYNCING" | "ACCEPTED" | "INVALID_BLOCK_HASH".
        latest_valid_hash: string,    -- Encoded 32-byte value of the latest valid block hash, may be `null`.
        validation_error: string,    -- Message providing additional details on the validation error, may be `null`.
    }
}
```

This step sets the
[`payloadStatus`](https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#payloadstatusv1)
value that execution-layer client mock returns in responses to the following
Engine API calls:

- [`engine_newPayloadV1(payload)`](https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#engine_newpayloadv1)
  if `payload.blockHash == payload_info.block_hash`
- [`engine_forkchoiceUpdatedV1(forkchoiceState, ...)`](https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#engine_forkchoiceupdatedv1)
  if `forkchoiceState.headBlockHash == payload_info.block_hash`

*Note*: Status of a payload must be *initialized* via `on_payload_info` before
the corresponding `on_block` execution step.

*Note*: Status of the same payload may be updated for several times throughout
the test.

#### `on_execution_payload_envelope` execution step

The parameter that is required for executing
`on_execution_payload_envelope(store, signed_execution_payload_envelope)`.

```yaml
{
    execution_payload: string  -- the name of the `execution_payload_envelope_<32-byte-root>.ssz_snappy` file.
                                  To execute `on_execution_payload_envelope(store, signed_envelope)` with the given envelope.
    valid: bool                -- optional, default to `true`.
                                  If it's `false`, this execution step is expected to be invalid.
}
```

The file is located in the same folder (see below).

After this step, the `store` object may have been updated.

#### `on_payload_attestation_message` execution step

The parameter that is required for executing
`on_payload_attestation_message(store, ptc_message)`.

```yaml
{
    payload_attestation_message: string  -- the name of the `payload_attestation_message_<32-byte-root>.ssz_snappy` file.
                                            To execute `on_payload_attestation_message(store, ptc_message)` with the given message.
    valid: bool                          -- optional, default to `true`.
                                            If it's `false`, this execution step is expected to be invalid.
}
```

The file is located in the same folder (see below).

This execution step is available for Gloas and later forks.

After this step, the `store` object may have been updated.

#### `on_available_attestation` execution step

The parameter required for executing
`on_available_attestation(store, available_attestation)` in Simplex.

```yaml
{
    available_attestation: string  -- The name of the
                                      `available_attestation_<32-byte-root>.ssz_snappy` file.
    valid: bool                    -- Optional, default `true`; `false` expects rejection.
}
```

After this step, the `store` object may have been updated.

#### `on_round_double_vote_evidence` execution step

The parameter required for executing
`on_round_double_vote_evidence(store, evidence)` in Simplex.

```yaml
{
    round_double_vote_evidence: string  -- The name of the
                                           `round_double_vote_evidence_<32-byte-root>.ssz_snappy`
                                           file.
    valid: bool                         -- Optional, default `true`; `false` expects rejection.
}
```

After this step, the `store` object may have been updated.

#### Checks step

The checks to verify the current status of `store`.

```yaml
checks: {<store_attribute>: value}  -- the assertions.
```

`<store_attribute>` is the field member or property of
[`Store`](../../../specs/phase0/fork-choice.md#store) object that maintained by
client implementation. The fields include:

```yaml
head: {
    slot: int,
    root: string,             -- Encoded 32-byte value from get_head(store).root
    payload_status: int,      -- Gloas and later, the head's payload_status
}
time: int                     -- store.time
genesis_time: int             -- store.genesis_time
justified_checkpoint: {
    epoch: int,               -- Pre-Simplex: store.justified_checkpoint.epoch
    slot: int,                -- Simplex: store.justified_checkpoint.slot (one of epoch/slot)
    root: string,             -- Encoded 32-byte value from store.justified_checkpoint.root
}
finalized_checkpoint: {
    epoch: int,               -- Pre-Simplex: store.finalized_checkpoint.epoch
    slot: int,                -- Simplex: store.finalized_checkpoint.slot (one of epoch/slot)
    root: string,             -- Encoded 32-byte value from store.finalized_checkpoint.root
}
proposer_boost_root: string   -- Encoded 32-byte value from store.proposer_boost_root
viable_for_head_roots_and_weights: [{
    root: string,             -- Encoded 32-byte value of filtered_block_tree leaf blocks/nodes
    weight: int,              -- Integer value of the weight of the block/node
    payload_status: int,      -- Gloas and later, the payload_status of the node
}]
```

Additionally, these fields if `get_proposer_head` is implemented:

```yaml
get_proposer_head: string             -- Encoded 32-byte value from get_proposer_head(store)
payload_timeliness_vote: {            -- [New in Gloas]
    block_root: string,               -- Encoded 32-byte beacon block root
    votes: [bool | null, ...]         -- Votes ordered by PTC positions. Length is `PTC_SIZE`.
}
payload_data_availability_vote: {     -- [New in Gloas]
    block_root: string,               -- Encoded 32-byte beacon block root
    votes: [bool | null, ...]         -- Votes ordered by PTC positions. Length is `PTC_SIZE`.
}
payload_votes: {                      -- [New in Simplex]
    slot: int,                        -- Slot whose identity-keyed votes are checked
    votes: [{
        validator_index: int,
        beacon_block_root: string,    -- Encoded 32-byte beacon block root
        payload_present: bool,
        blob_data_available: bool,
    }, ...]                           -- Sorted by validator_index
}
payload_vote_equivocations: {         -- [New in Simplex]
    slot: int,
    validator_indices: [int, ...]     -- Sorted equivocating validator identities
}
simplex_store: {                      -- [New in Simplex]
    justified_height: int,
    h_max: int,
    latest_messages: [{
        validator_index: int,
        slot: int,
        root: string,
    }, ...],                          -- Sorted by validator_index
    round_attestations: [{
        round: int,
        votes: [{
            validator_index: int,
            data_root: string,        -- hash_tree_root(AttestationData)
        }, ...],
    }, ...],                          -- Rounds and votes sorted numerically
    round_equivocating_indices: [{
        round: int,
        validator_indices: [int, ...],
    }, ...],
    frozen_tsq_views: [{
        round: int,                    -- Synchronization round using this freeze
        support_round: int,
        attestations: [{
            validator_index: int,
            data_root: string,         -- hash_tree_root(AttestationData)
        }, ...],                       -- Sorted by validator_index
        equivocating_indices: [int, ...],
    }, ...],                           -- Sorted by synchronization round
    tsq_selections: [{
        round: int,
        support_round: int,
        simplex_root: string,
        candidate_roots: [string, ...], -- Sorted encoded roots
        weights: [{
            validator_index: int,
            weight: int,
        }, ...],                       -- Sorted by validator_index
        total_active_balance: int,
        candidate_root: string,
    }, ...],                           -- Sorted by round
    round_proposals: [{
        round: int,
        proposal_roots: [string, ...], -- Sorted round-start proposal roots
    }, ...],                           -- Sorted by round
    round_proposal_conflicts: [int, ...], -- Sorted rounds with multiple proposals
    equivocating_indices: [int, ...],
    pending_attestations: [{
        root: string,
        attestation_roots: [string, ...],
    }, ...],
    pending_available_attestations: [{
        root: string,
        attestation_roots: [string, ...],
    }, ...],
    payload_roots: [string, ...],       -- Sorted verified-envelope block roots
    payload_votes: [{
        slot: int,
        votes: [{
            validator_index: int,
            beacon_block_root: string,
            payload_present: bool,
            blob_data_available: bool,
        }, ...],
    }, ...],
    payload_vote_equivocations: [{
        slot: int,
        validator_indices: [int, ...],
    }, ...],
    available_votes: [{
        slot: int,
        votes: [{
            validator_index: int,
            beacon_block_root: string,
            payload_present: bool,
        }, ...],
    }, ...],                          -- Slots and votes sorted numerically
    available_vote_equivocations: [{
        slot: int,
        validator_indices: [int, ...],
    }, ...],
    view_freeze_slots: [int, ...],    -- Sorted exact 75%-completed slots
    available_timely_attesters: [{
        slot: int,
        validator_indices: [int, ...],
    }, ...],
    available_timely_equivocations: [{
        slot: int,
        validator_indices: [int, ...],
    }, ...],
    available_committees: [{
        slot: int,
        validator_indices: [int, ...], -- Seat order and duplicates preserved
    }, ...],
    frozen_available_votes: [{
        slot: int,
        committee: [int, ...],         -- Seat order and duplicates preserved
        votes: [{
            validator_index: int,
            beacon_block_root: string,
            payload_present: bool,
        }, ...],
    }, ...],
    stable_root: string,
    stable_root_proposal_root: string,
    stable_root_round: int,
    latest_confirmed_head: {root: string, slot: int},
    live_confirmed_head: {root: string, slot: int},
    fast_confirmed_head: {root: string, slot: int},
}
```

The `simplex_store` check is an exact snapshot of the listed maps, electorates,
TSQ selections and freezes, proposals, payload availability, and heads,
including empty entries. Rounds, slots, roots, votes, and validator sets are
sorted as indicated; cached committee seat order and duplicate selections are
preserved. It makes fork-choice effects of finality, available-chain, timing,
and equivocation messages portable: an implementation cannot pass a vector by
merely accepting an operation while omitting its Store update or by pruning a
snapshot before its delayed confirmation is consumed. Pure memoization caches
and the live-transition-only historical payload-verification witness are
intentionally outside this portable snapshot.

For example:

```yaml
- checks:
    time: 192
    head: {slot: 32, root: '0xdaa1d49d57594ced0c35688a6da133abb086d191a2ebdfd736fad95299325aeb'}
    justified_checkpoint: {epoch: 3, root: '0xc25faab4acab38d3560864ca01e4d5cc4dc2cd473da053fbc03c2669143a2de4'}
    finalized_checkpoint: {epoch: 2, root: '0x40d32d6283ec11c53317a46808bc88f55657d93b95a1af920403187accf48f4f'}
    proposer_boost_root: '0xdaa1d49d57594ced0c35688a6da133abb086d191a2ebdfd736fad95299325aeb'
    get_proposer_head: '0xdaa1d49d57594ced0c35688a6da133abb086d191a2ebdfd736fad95299325aeb'
    viable_for_head_roots_and_weights: [
      {root: '0x533290b6f44d31c925acd08dfc8448624979d48c40b877d4e6714648866c9ddb', weight: 192000000000},
      {root: '0x5cfb9d9099cdf1d8ab68ce96cdae9f0fa6eef16914a01070580dfdc1d2d59ec3', weight: 544000000000}
    ]
```

*Note*: Each `checks` step may include one or multiple items. Each item has to
be checked against the current store.

### `attestation_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the given attestation.

Each file is an SSZ-snappy encoded `Attestation`.

### `block_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the `BeaconBlock` (the `message` field
of the `SignedBeaconBlock`).

Each file is an SSZ-snappy encoded `SignedBeaconBlock`.

### `execution_payload_envelope_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the given signed envelope.

Each file is an SSZ-snappy encoded `SignedExecutionPayloadEnvelope`.

### `payload_attestation_message_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the given payload attestation message.

Each file is an SSZ-snappy encoded `PayloadAttestationMessage`.

### `available_attestation_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the available attestation. Each file
is an SSZ-snappy encoded `AvailableAttestation` (Simplex only).

### `round_double_vote_evidence_<32-byte-root>.ssz_snappy`

`<32-byte-root>` is the hash tree root of the evidence. Each file is an
SSZ-snappy encoded `RoundDoubleVoteEvidence` (Simplex only).

## Condition

1. Deserialize `anchor_state.ssz_snappy` and `anchor_block.ssz_snappy` to
   initialize the local store object with
   `get_forkchoice_store(anchor_state, anchor_block)` helper.
2. Iterate sequentially through `steps.yaml`
   - For each execution, look up the corresponding ssz_snappy file. Execute the
     corresponding helper function on the current store.
     - For a pre-Simplex `on_block` execution step, execute `on_block` first,
       then explicitly deliver each body attestation and attester slashing to
       its fork-choice handler. From Gloas through pre-Simplex, also expand each
       `PayloadAttestation` into its constituent `PayloadAttestationMessage`
       values and execute each one with
       `on_payload_attestation_message(store, ptc_message, is_from_block=True)`.
       Simplex changes this ownership contract: execute only
       `on_block(store, block)`. Its `on_block` handler owns every block-carried
       fork-choice operation, including finality attestations, attester
       slashings, payload attestations, available attestations, and
       round-double-vote evidence. A runner MUST NOT redeliver any of them.
     - For the `on_execution_payload_envelope` execution step: look up the
       corresponding `execution_payload_envelope_<root>.ssz_snappy` file and
       execute `on_execution_payload_envelope(store, signed_envelope)`.
     - For the `on_payload_attestation_message` execution step: look up the
       corresponding `payload_attestation_message_<root>.ssz_snappy` file and
       execute `on_payload_attestation_message(store, ptc_message)`.
     - For the `on_available_attestation` execution step: look up the
       corresponding `available_attestation_<root>.ssz_snappy` file and execute
       `on_available_attestation(store, available_attestation)`.
     - For the `on_round_double_vote_evidence` execution step: look up the
       corresponding `round_double_vote_evidence_<root>.ssz_snappy` file and
       execute `on_round_double_vote_evidence(store, evidence)`.
   - For each `checks` step, the assertions on the current store must be
     satisfied.

# PeerDASv1 -- Networking

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

<!-- mdformat-toc end -->

## Introduction

This document contains the consensus-layer networking specification for PeerDASv1.

The specification of these changes continues in the same format as the network
specifications of previous upgrades, and assumes them as pre-requisite.

## Modifications in PeerDASv1

### Configuration

| Name                      | Value | Description                                                              |
| --------------------------| ----- | -------------------------------------------------------------------------|
| `COLUMN_SUBNET_COUNT`     | `256` | The number of column subnets used in the gossipsub protocol              |
| `ROW_SUBNET_COUNT`        | `64`  | The number of row subnets used in the gossipsub protocol                 |    


### Containers

#### `KZGCommitmentsSidecar`

```python
class KZGCommitmentsSidecar(Container):
    signed_block_header: SignedBeaconBlockHeader
    kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK]
    kzg_commitments_inclusion_proof: Vector[Bytes32, KZG_COMMITMENTS_INCLUSION_PROOF_DEPTH]
```

### Helpers

##### `verify_cell_sidecar_kzg_proof`

```python
def verify_cell_sidecar_kzg_proof(cell_sidecar: CellSidecar, kzg_commitment: KZGCommitment) -> bool:
    return verify_cell_kzg_proof_batch(
        commitments_bytes=[kzg_commitment],
        cell_indices=[cell_sidecar.column_index],
        cells=[cell_sidecar.cell],
        proofs_bytes=[cell_sidecar.kzg_proof],
    )
```

##### `verify_kzg_commitments_sidecar`

```python
def verify_kzg_commitments_sidecar(sidecar: KZGCommitmentsSidecar) -> bool:
    """
    Verify if the given KZG commitments included in the given beacon block.
    """
    gindex = get_subtree_index(get_generalized_index(BeaconBlockBody, "blob_kzg_commitments"))
    return is_valid_merkle_branch(
        leaf=hash_tree_root(sidecar.kzg_commitments),
        branch=sidecar.kzg_commitments_inclusion_proof,
        depth=KZG_COMMITMENTS_INCLUSION_PROOF_DEPTH,
        index=gindex,
        root=sidecar.signed_block_header.message.body_root,
    )
```

##### `compute_column_subnet`

```python
def compute_column_subnet(column_index: ColumnIndex) -> SubnetID:
    return SubnetID(column_index % COLUMN_SUBNET_COUNT)
```

##### `compute_row_subnet`

```python
def compute_row_subnet(row_index: RowIndex) -> SubnetID:
    return SubnetID(row_index % ROW_SUBNET_COUNT)
```

### MetaData

The `MetaData` stored locally by clients is updated with an additional field to
communicate the custody group count.

```
(
  seq_number: uint64
  attnets: Bitvector[ATTESTATION_SUBNET_COUNT]
  syncnets: Bitvector[SYNC_COMMITTEE_SUBNET_COUNT]
  column_custody_group_count: uint64 # ccgc
  row_custody_group_count: uint64 # rcgc
)
```

Where

- `seq_number`, `attnets`, and `syncnets` have the same meaning defined in the
  Altair document.
- `column_custody_group_count` represents the node's column custody group count. Clients MAY
  reject peers with a value less than `COLUMN_CUSTODY_REQUIREMENT`.
  - `row_custody_group_count` represents the node's row custody group count. Clients MAY
  reject peers with a value less than `ROW_CUSTODY_REQUIREMENT`.

### The gossip domain: gossipsub

Some gossip meshes are upgraded in the Fulu fork to support upgraded types.

#### Topics and messages

##### Global topics

###### `kzg_commitments_sidecar`

This topic is used to propagate kzg commitments sidecars.

The *type* of the payload of this topic is `KZGCommitmentsSidecar`.

The following validations MUST pass before forwarding the
`sidecar: KZGCommitmentsSidecar` on the network, assuming the alias
`block_header = sidecar.signed_block_header.message`:

- _[REJECT]_ The sidecar is valid as verified by
  `verify_kzg_commitments_sidecar(sidecar)`.
- _[IGNORE]_ The sidecar is not from a future slot (with a
  `MAXIMUM_GOSSIP_CLOCK_DISPARITY` allowance) -- i.e. validate that
  `block_header.slot <= current_slot` (a client MAY queue future sidecars for
  processing at the appropriate slot).
- _[IGNORE]_ The sidecar is from a slot greater than the latest finalized slot
  -- i.e. validate that
  `block_header.slot > compute_start_slot_at_epoch(state.finalized_checkpoint.epoch)`
- _[REJECT]_ The proposer signature of `sidecar.signed_block_header`, is valid
  with respect to the `block_header.proposer_index` pubkey.
- _[IGNORE]_ The sidecar's block's parent (defined by
  `block_header.parent_root`) has been seen (via gossip or non-gossip sources)
  (a client MAY queue sidecars for processing once the parent block is
  retrieved).
- _[REJECT]_ The sidecar's block's parent (defined by
  `block_header.parent_root`) passes validation.
- _[REJECT]_ The sidecar is from a higher slot than the sidecar's block's parent
  (defined by `block_header.parent_root`).
- _[REJECT]_ The current finalized_checkpoint is an ancestor of the sidecar's
  block -- i.e.
  `get_checkpoint_block(store, block_header.parent_root, store.finalized_checkpoint.epoch) == store.finalized_checkpoint.root`.
- _[IGNORE]_ The sidecar is the first valid sidecar with valid header signature 
    for the tuple `(block_header.slot, block_header.proposer_index)`
- _[REJECT]_ The sidecar is proposed by the expected `proposer_index` for the
  block's slot in the context of the current shuffling (defined by
  `block_header.parent_root`/`block_header.slot`). If the `proposer_index`
  cannot immediately be verified against the expected shuffling, the sidecar MAY
  be queued for later processing while proposers for the block's branch are
  calculated -- in such a case _do not_ `REJECT`, instead `IGNORE` this message.
  
##### Blob subnets

###### Deprecated `data_column_sidecar_{subnet_id}`

`data_column_sidecar_{subnet_id}` is deprecated.

###### `column_subnet_{subnet_id}`

This topic is used to propagate cell sidecars, each mapping to a `subnet_id`
via `compute_column_subnet(cell_sidecar.column_index)`.

The *type* of the payload of this topic is `CellSidecar`.

The following validations MUST pass before forwarding the
`sidecar: CellSidecar` on the network.

- _[REJECT]_ The sidecar is for the correct subnet -- i.e.
  `compute_column_subnet(sidecar.column_index) == subnet_id`.
- _[IGNORE]_  A `kzg_commitments_sidecar: KZGCommitmentsSidecar`
   has been seen (via gossip or non-gossip sources) with 
  `hash_tree_root(block_header) = sidecar.beacon_block_root`, 
   for `block_header = kzg_commitments_sidecar.signed_block_header.message`
  (a client MAY queue sidecars for processing once the corresponding 
  kzg commitments sidecar is retrieved). In the following, assume the
  alias `kzg_commitments = kzg_commitments_sidecar.kzg_commitments`.
- _[REJECT]_ `sidecar.row_index < len(kzg_commitments)`
- _[REJECT]_ The sidecar kzg proof is valid as verified by
  `verify_cell_sidecar_kzg_proof(sidecar, kzg_commitment)`, where
  `kzg_commitment = kzg_commitments[sidecar.row_index]`

*Note*: Client implementers can choose to batch the kzg proof verification
in `verify_cell_sidecar_kzg_proof` across multiple cells, using the underlying
verification function `verify_cell_kzg_proof_batch`.


###### `row_subnet_{subnet_id}`

This topic is used to propagate cell sidecars, each mapping to 
a `subnet_id` via `compute_row_subnet(cell_sidecar.row_index)`.

The *type* of the payload of this topic is `CellSidecar`.

The following validations MUST pass before forwarding the
`sidecar: CellSidecar` on the network. Except for the first
validation concerning the subnet index, they are *identical*
the validations required in a `column_subnet`

- _[REJECT]_ The sidecar is for the correct subnet -- i.e.
  `compute_row_subnet(sidecar.row_index) == subnet_id`.
- _[IGNORE]_  A `kzg_commitments_sidecar: KZGCommitmentsSidecar`
   has been seen (via gossip or non-gossip sources) with 
  `hash_tree_root(block_header) = sidecar.beacon_block_root`, 
   for `block_header = kzg_commitments_sidecar.signed_block_header.message`
  (a client MAY queue sidecars for processing once the corresponding 
  kzg commitments sidecar is retrieved). In the following, assume the
  alias `kzg_commitments = kzg_commitments_sidecar.kzg_commitments`.
- _[REJECT]_ `sidecar.row_index < len(kzg_commitments)`
- _[REJECT]_ The sidecar kzg proof is valid as verified by
  `verify_cell_sidecar_kzg_proof(sidecar, kzg_commitment)`, where
  `kzg_commitment = kzg_commitments[sidecar.row_index]`

*Note*: As for column subnets, client implementers can choose to batch 
the kzg proof verification in `verify_cell_sidecar_kzg_proof`.

###### Cross-seeding

A `cell_sidecar` belongs to a row subnet, `compute_row_subnet(cell_sidecar.row_index)`,
as well as a column subnet, `compute_column_subnet(cell_sidecar.column_index)`. 
A node that subscribes to both `row_subnet` and `column_subnet` MUST treat a
`cell_sidecar` received in either subnet and belonging to both as if it was also
received in the other subnet, in particular forwarding it to its mesh neighbors
and including it in its gossip messages. Validation only has to be performed once.

###### Distributed Blob Publishing using blobs retrieved from local execution layer client

*Note*: this is currently assuming that `engine_getBlobsV3` has an `indices`
parameter that lets the CL client specify which column indices it wants to
get cells for

Honest nodes SHOULD query `engine_getBlobsV3` as soon as they receive a valid
`beacon_block` or `kzg_commitments_sidecar` from gossip, setting `indices` to
`[i for i in range(NUMBER_OF_COLUMNS) if compute_column_subnet(i) in subnet_ids]`.
where `subnet_ids` are all subnets that they subscribe to. 
If cells and proofs for a blob matching ANY commitment in `kzg_commitments` 
are retrieved, they should be converted to cell sidecars, and imported.

When clients use the local execution layer to retrieve cells and proofs, they
SHOULD skip verification of those cells. When subsequently importing the
resulting cell sidecars, they MUST behave as if the `cell_sidecar` had been
received via gossip. In particular, clients MUST:

- Publish the corresponding `cell_sidecar` on the
  `column_subnet_{subnet_id}` topic
- Update gossip rule related data structures (i.e. update the anti-equivocation
  cache).


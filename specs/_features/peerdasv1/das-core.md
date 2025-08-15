# PeerDASv1 -- Data Availability Sampling Core

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

<!-- mdformat-toc end -->

## Configuration

### Custody setting

| Name                         | Value | Description                                                                       |
| -----------------------------| ----- | --------------------------------------------------------------------------------- |
| `NUMBER_OF_COLUMN_CUSTODY_GROUPS`| `256` | Number of custody groups available for nodes to custody                       |
| `NUMBER_OF_ROW_CUSTODY_GROUPS`   | `64` | Number of custody groups available for nodes to custody                           |
| `COLUMN_CUSTODY_REQUIREMENT` | `8`   | Minimum number of column custody groups an honest node custodies and serves samples from |
| `ROW_CUSTODY_REQUIREMENT`    | `1`   | Minimum number of row custody groups an honest node custodies and serves |

## Preset

### Size parameters

| Name                | Value                                | Description                                   |
| ------------------- | ------------------------------------ | --------------------------------------------- |
| `NUMBER_OF_COLUMNS` | `uint64(CELLS_PER_EXT_BLOB)` (= 256) | Number of columns in the extended data matrix |

### Containers

#### `CellSidecar`

```python
class CellSidecar(Container):
    cell: Cell
    kzg_proof: KZGProof
    row_index: RowIndex
    column_index: ColumnIndex
    beacon_block_root: Root
```

## Helper functions

### `get_custody_groups`

*Note*: modified to take `number_of_custody_groups` as an input,
to be used as a helper for both row and column custody.

```python
def get_custody_groups(
    node_id: NodeID,
    custody_group_count: uint64,
    number_of_custody_groups: uint64
) -> Sequence[CustodyIndex]:
    assert custody_group_count <= number_of_custody_groups

    # Skip computation if all groups are custodied
    if custody_group_count == number_of_custody_groups:
        return [CustodyIndex(i) for i in range(number_of_custody_groups)]

    current_id = uint256(node_id)
    custody_groups: List[CustodyIndex] = []
    while len(custody_groups) < custody_group_count:
        custody_group = CustodyIndex(
            bytes_to_uint64(hash(uint_to_bytes(current_id))[0:8]) % number_of_custody_groups
        )
        if custody_group not in custody_groups:
            custody_groups.append(custody_group)
        if current_id == UINT256_MAX:
            # Overflow prevention
            current_id = uint256(0)
        else:
            current_id += 1

    assert len(custody_groups) == len(set(custody_groups))
    return sorted(custody_groups)
```

### `get_row_custody_groups`

```python
def get_row_custody_groups(node_id: NodeID, custody_group_count: uint64)
    return get_custody_groups(
        node_id=node_id,
        custody_group_count=custody_group_count,
        number_of_custody_groups=NUMBER_OF_ROW_CUSTODY_GROUPS
    )
```

### `get_column_custody_groups`

```python
def get_column_custody_groups(node_id: NodeID, custody_group_count: uint64)
    return get_custody_groups(
        node_id=node_id,
        custody_group_count=custody_group_count,
        number_of_custody_groups=NUMBER_OF_COLUMN_CUSTODY_GROUPS
    )
```

### `compute_columns_for_custody_group`

```python
def compute_columns_for_custody_group(custody_group: CustodyIndex) -> Sequence[ColumnIndex]:
    assert custody_group < NUMBER_OF_CUSTODY_GROUPS
    columns_per_group = NUMBER_OF_COLUMNS // NUMBER_OF_CUSTODY_GROUPS
    return [
        ColumnIndex(NUMBER_OF_CUSTODY_GROUPS * i + custody_group) for i in range(columns_per_group)
    ]
```

### `compute_rows_for_custody_group`

```python
def compute_rows_for_custody_group(custody_group: CustodyIndex, epoch: Epoch) -> Sequence[RowIndex]:
    assert custody_group < NUMBER_OF_ROW_CUSTODY_GROUPS
    max_blobs = get_blob_parameters(epoch).max_blobs_per_block
    rows_per_group = max_blobs // NUMBER_OF_ROW_CUSTODY_GROUPS
    return [
        RowIndex(NUMBER_OF_ROW_CUSTODY_GROUPS * i + custody_group) for i in range(rows_per_group)
    ]
```

### `recover_row`

```python
def recover_row(
    partial_row: Sequence[MatrixEntry],
    row_index: RowIndex,
) -> List[MatrixEntry, CELLS_PER_EXT_BLOB]:
    row = []
    column_indices = [e.column_index for e in partial_row]
    cells = [e.cell for e in partial_row]
    recovered_cells, recovered_proofs = recover_cells_and_kzg_proofs(column_indices, cells)
    for column_index, (cell, proof) in enumerate(zip(recovered_cells, recovered_proofs)):
        row.append(
            MatrixEntry(
                cell=cell,
                kzg_proof=proof,
                row_index=row_index,
                column_index=column_index,
            )
        )
    return row
```

### `compute_cell_sidecar_from_matrix_entry`

```python
def compute_cell_sidecar_from_matrix_entry(
    matrix_entry: MatrixEntry,
    beacon_block_root: Root
) -> CellSidecar:   
    return CellSidecar(
        cell=cell,
        kzg_proof=proof,
        row_index=row_index,
        column_index=column_index,
        beacon_block_root=beacon_block_root,       
)
```

### `recover_matrix`

*Note*: modified to use `recover_row` as a helper

```python
def recover_matrix(
    partial_matrix: Sequence[MatrixEntry], blob_count: uint64
) -> Sequence[MatrixEntry]:
    """
    Recover the full, flattened sequence of matrix entries.

    This helper demonstrates how to apply ``recover_cells_and_kzg_proofs``.
    The data structure for storing cells/proofs is implementation-dependent.
    """
    matrix = []
    for blob_index in range(blob_count):
        partial_row = [e for e in partial_matrix if e.row_index == blob_index]
        row = recover_row(partial_row, blob_index)
        matrix.extend(row)
    return matrix
```

## Custody

Column custody works as in the previous specification, with `NUMBER_OF_CUSTODY_GROUPS`
renamed to `NUMBER_OF_COLUMN_CUSTODY_GROUPS` and `CUSTODY_REQUIREMENT` renamed to 
`COLUMN_CUSTODY_REQUIREMENT`.

Row custody is introduced, and is functionally identical to column custody, but uses
the `_ROW_CUSTODY` parameters `NUMBER_OF_ROW_CUSTODY_GROUPS` and `ROW_CUSTODY_REQUIREMENT`.
Rows are grouped into custody groups, and nodes custodying a row custody group MUST custody
all rows in that group.

Nodes may choose to custody more than the minimum requirements by advertising higher values
in their ENR, via the `column_custody_group_count` and `row_custody_group_count` fields.

Custody group selection uses `get_column_custody_groups(node_id, custody_group_count)`
for columns and `get_row_custody_groups(node_id, custody_group_count, epoch)` for rows. 
Both functions use the same pseudo-random selection mechanism as the original 
`get_custody_groups`, which is retained as a helper for both, ensuring public deterministic
selection based on node ID. However, the row custody function includes an `epoch` parameter,
because the maximum number of rows is determined by the blob schedule for that epoch.

## Reconstruction

If the node obtains 50%+ of all the cells in a row subnet it is subscribed to,
it SHOULD reconstruct the full row via the `recover_row` helper. Nodes MAY delay
this reconstruction allowing time for other cells to arrive over the network.
If delaying reconstruction, nodes may use a random delay in order to desynchronize
reconstruction among nodes, thus reducing overall CPU load.

Once the node obtains a row through reconstruction, it computes all corresponding
cell sidecars via `compute_cell_sidecar_from_matrix_entry`. The node MUST then 
expose the new cell sidecars as if it had received them over the network.
The node MUST send the reconstructed cell sidecars to its topic mesh neighbors
in the corresponding row subnet. For each cell sidecar, it MUST also send it to
the topic mesh neighbors in the corresponding *column* subnet, if it is subscribed
to it. If not, it SHOULD still expose the availability of the cell sidecar as
part of the gossip emission process. 

After exposing the reconstructed cell sidecars to the network, the node MAY
delete any cell sidecar that are not part of the node's custody requirement.

*Note*: A node always maintains a matrix view of the rows and columns they are
following, able to cross-reference and cross-seed in either direction.



# Simplex Light Client -- Sync Protocol

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Constants](#constants)
  - [New constants](#new-constants)
- [Helpers](#helpers)
  - [Modified `finalized_root_gindex_at_slot`](#modified-finalized_root_gindex_at_slot)
  - [Modified `current_sync_committee_gindex_at_slot`](#modified-current_sync_committee_gindex_at_slot)
  - [Modified `next_sync_committee_gindex_at_slot`](#modified-next_sync_committee_gindex_at_slot)

<!-- mdformat-toc end -->

## Introduction

This upgrade extends the
[Electra light-client generalized-index selection](../../../electra/light-client/sync-protocol.md)
for the Simplex `BeaconState` layout. Historical proofs remain keyed to the
layout active at their slot: the frozen Electra constants continue to apply to
Electra, Fulu, and Gloas states, while new constants apply at and after Simplex
activation.

## Constants

### New constants

| Name                                    | Value                                                                        |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| `FINALIZED_ROOT_GINDEX_SIMPLEX`         | `get_generalized_index(BeaconState, 'finalized_checkpoint', 'root')` (= 165) |
| `CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX` | `get_generalized_index(BeaconState, 'current_sync_committee')` (= 84)        |
| `NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX`    | `get_generalized_index(BeaconState, 'next_sync_committee')` (= 85)           |

## Helpers

### Modified `finalized_root_gindex_at_slot`

```python
def finalized_root_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    epoch = compute_epoch_at_slot(slot)

    # [Modified in Simplex]
    if epoch >= SIMPLEX_FORK_EPOCH:
        return FINALIZED_ROOT_GINDEX_SIMPLEX
    if epoch >= ELECTRA_FORK_EPOCH:
        return FINALIZED_ROOT_GINDEX_ELECTRA
    return FINALIZED_ROOT_GINDEX
```

### Modified `current_sync_committee_gindex_at_slot`

```python
def current_sync_committee_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    epoch = compute_epoch_at_slot(slot)

    # [Modified in Simplex]
    if epoch >= SIMPLEX_FORK_EPOCH:
        return CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    if epoch >= ELECTRA_FORK_EPOCH:
        return CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA
    return CURRENT_SYNC_COMMITTEE_GINDEX
```

### Modified `next_sync_committee_gindex_at_slot`

```python
def next_sync_committee_gindex_at_slot(slot: Slot) -> GeneralizedIndex:
    epoch = compute_epoch_at_slot(slot)

    # [Modified in Simplex]
    if epoch >= SIMPLEX_FORK_EPOCH:
        return NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX
    if epoch >= ELECTRA_FORK_EPOCH:
        return NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA
    return NEXT_SYNC_COMMITTEE_GINDEX
```

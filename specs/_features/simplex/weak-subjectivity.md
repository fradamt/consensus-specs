# Simplex -- Weak Subjectivity Guide

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Weak Subjectivity Checkpoint](#weak-subjectivity-checkpoint)
- [Weak Subjectivity Sync](#weak-subjectivity-sync)
  - [Weak Subjectivity Sync Procedure](#weak-subjectivity-sync-procedure)
- [Weak Subjectivity Period](#weak-subjectivity-period)
  - [Modified `is_within_weak_subjectivity_period`](#modified-is_within_weak_subjectivity_period)

<!-- mdformat-toc end -->

## Introduction

This document extends the
[Gloas -- Weak Subjectivity Guide](../../gloas/weak-subjectivity.md). All
behaviors and definitions from the inherited guide carry over unless explicitly
overridden here.

Post-Simplex checkpoints identify an exact block by proposal slot and root. Weak
subjectivity checkpoint validation therefore uses `Checkpoint.slot`; the weak
subjectivity period itself remains measured in epochs.

## Weak Subjectivity Checkpoint

A Simplex Weak Subjectivity Checkpoint identifies an exact post-Simplex block by
its proposal `slot` and `root`. The height-0 justified and finalized checkpoints
created by `upgrade_to_simplex` are state-internal transition sentinels, not
valid Simplex Weak Subjectivity Checkpoints. Their slots retain legacy
epoch-boundary semantics and their roots may name blocks proposed in older
slots.

If a provider supplies only a pre-Simplex checkpoint or a height-0 transition
sentinel, clients MUST use the pre-Simplex `Checkpoint` type, Weak Subjectivity
Sync Procedure, and checkpoint state. In particular, a client MUST NOT decode a
pre-Simplex `block_root:epoch_number` checkpoint as a Simplex
`block_root:slot_number` checkpoint or convert a height-0 transition sentinel
into a Simplex Weak Subjectivity Checkpoint.

As in the inherited guide, any trusted checkpoint may be distributed; the
immediate post-state of that exact post-Simplex block need not itself already
record a nonzero-height finalization. Provider trust and evidence that the
checkpoint is suitable for distribution remain out of band. This guide's
executable check establishes the exact block/state pairing and staleness, not a
subsequent finalization proof.

## Weak Subjectivity Sync

### Weak Subjectivity Sync Procedure

This procedure fully replaces the inherited procedure when the input is a
Simplex Weak Subjectivity Checkpoint:

1. Input a Weak Subjectivity Checkpoint as a CLI parameter in
   `block_root:slot_number` format, where `block_root` (an "0x"-prefixed 32-byte
   hex string) and `slot_number` (an integer) identify a block proposed at that
   exact slot. Height-0 transition sentinels are invalid inputs.

2. Check the Weak Subjectivity requirements:

   - *IF* `slot_number > store.finalized_checkpoint.slot`, then *ASSERT* during
     block sync that a block with root `block_root` and proposal slot
     `slot_number` is in the sync path. Emit a descriptive critical error if
     this assertion fails, then exit the client process.
   - *IF* `slot_number <= store.finalized_checkpoint.slot`, then *ASSERT* that
     the block proposed at exactly `slot_number` in the canonical chain has root
     `block_root`. An empty slot does not inherit the preceding block for this
     check. Emit a descriptive critical error if this assertion fails, then exit
     the client process.

## Weak Subjectivity Period

### Modified `is_within_weak_subjectivity_period`

The supplied state is the post-state of the exact checkpoint block. Immediately
after block processing, `latest_block_header.state_root` may still be zero; fill
it from the supplied state exactly as `process_slot` does before checking the
block root. This exact pairing works for a checkpoint in any proposal slot,
including the last slot of an epoch. A height-0 transition sentinel is rejected,
but a distinct exact post-Simplex block remains eligible while the supplied
state's `finalized_height` is zero, as described above.

```python
def is_within_weak_subjectivity_period(
    store: Store, ws_state: BeaconState, ws_checkpoint: Checkpoint
) -> bool:
    # Clients may choose to validate the input state against the input Weak
    # Subjectivity Checkpoint.
    is_legacy_transition_checkpoint = (
        ws_state.justified_height == Height(0) and ws_checkpoint == ws_state.justified_checkpoint
    ) or (ws_state.finalized_height == Height(0) and ws_checkpoint == ws_state.finalized_checkpoint)
    assert not is_legacy_transition_checkpoint
    assert ws_state.slot == ws_checkpoint.slot
    checkpoint_header = copy(ws_state.latest_block_header)
    assert checkpoint_header.slot == ws_checkpoint.slot
    if checkpoint_header.state_root == Root():
        checkpoint_header.state_root = hash_tree_root(ws_state)
    assert hash_tree_root(checkpoint_header) == ws_checkpoint.root

    ws_period = compute_weak_subjectivity_period(ws_state)
    ws_state_epoch = compute_epoch_at_slot(ws_state.slot)
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    return current_epoch <= ws_state_epoch + ws_period
```

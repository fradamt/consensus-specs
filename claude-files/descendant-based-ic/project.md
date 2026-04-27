# Fresh-Simplex-With-Notarizations Finality Gadget

**Branch**: `descendant-based-ic` (now tracks the
`fresh-simplex-with-notarizations` paper; name retained to preserve git
history). **Status**: Spec implementation pass (2026-04-21). Proof sketches
updated; full proofs live in the overleaf paper
`../overleaf/shared-finality/fresh_simplex_with_notarizations.tex`.

______________________________________________________________________

## Overview

A finality gadget based on the fresh-simplex-with-notarizations protocol. The
model is `n >= 3f+1` BFT with 2/3 quorums for every predicate (justification,
prefix-notarization, finalization). Each validator casts at most one **justify**
(R1) and one **notarize** (R2) attestation per state- height; both are subject
to a **fresh-vote** gate that keys the vote to the current height interval on
the current chain. Finalization is separated from height advance: the justified
checkpoint is confirmed via a piggyback carried on R1 votes (`finality_target`,
`finality_height`); the finality bit-list persists across height advances —
reset only when a new justification fires (the `also_justify=True` branch of
`advance_height`). Finality firing alone (F ← J) does not reset it; the bitlist
then becomes vacuously satisfied because
`finalized_checkpoint == justified_checkpoint`.

The fork-choice store maintains a root `store.root` as the running maximum over
**cert events** — every justification or prefix-notarization observed post-state
on an imported block is offered to `update_fork_choice_root`, which
running-maxes on the lexicographic key `(height, bit, slot, root)`.
`store.finalized_checkpoint` advances via `update_finalized` only when the
candidate strictly extends the current finalized AND is an ancestor-or-self of
`store.root`. No viability filter; no
`filter_block_tree`/`get_filtered_block_tree`/`update_justified`.

______________________________________________________________________

## Key Design Choices

### 1. Two vote kinds (R1 justify, R2 notarize)

`AttestationData.kind ∈ {ATTESTATION_KIND_JUSTIFY, ATTESTATION_KIND_NOTARIZE}`.
R1 is the canonical "justify this target" vote, at most once per state-height
per validator (honest rule). R2 is the fallback when R1 quorums fragment: if the
justify quorum cannot form on a single target, R2 votes contribute to a
cumulative prefix walk that notarizes a common ancestor.

### 2. Fresh-vote gate

`is_viable_attestation_target(state, attestation)` asserts:

- `data.height == state.current_height`
- `data.target.slot >= state.current_height_start_slot` (paper's `s_h`)
- `is_target_on_chain(state, data.target)` — target references an actual block
  at its proposal slot on this chain.

Only fresh votes update `justification_targets[i]` and
`notarization_targets[i]`. Stale-height votes fail the height assertion in
`process_attestation` outright.

### 3. processHeight: three branches

`process_justification_and_finalization` runs three independent checks in order
(at most one height-advance fires per invocation):

1. **Finality**: if `|P| >= 2n/3`, set `F ← J`. Does NOT advance height.
2. **Justify**: if some target `T` reaches
   `|{i : justification_targets[i] = T}| >= 2n/3` (per-target counting), set
   `J ← T`, `N ← T`, reset `P`, advance height.
3. **Prefix-notarize**: if the cumulative walk on `notarization_targets` reaches
   `2n/3` at some on-chain slot `T`, set `N ← T`, advance height (no change to
   `J`, no reset of `P`).

### 4. Store = `(σ, T, F_s, R_s, key_R)`

`Store.root: Root` and `Store.root_key: Tuple[Height, uint8, Slot, Root]` track
`(R_s, key_R)`. `on_block` first runs the state transition, then dispatches
`on_attestation` / `on_available_attestation` for every attestation in the block
body. Only afterwards does it emit two cert events per imported block (one for
the post-state's `justified_checkpoint` with bit 1, one for
`notarized_checkpoint` with bit 0), followed by `update_finalized` and
`maybe_update_justified_checkpoint` (which mirrors the latest bit-1 cert event
onto `store.justified_checkpoint` for weight-accounting consumers).

`get_lmd_ghost_head` walks from `store.root` — no filtered block tree.

### 5. Three-guard leak

Each round evaluates three independent "missed-progress" guards against the
pre-advance state:

- **Notarization missed** (no height advance this round) → penalize validators
  with `notarization_targets[i] == FAR_FUTURE_SLOT`.
- **Justification missed** (no new justification this round) → penalize
  validators whose `justification_targets[i]` differs from the plurality slot
  `T*` (or is unset).
- **Finalization missed** (finalization pending AND no new finalization this
  round) → penalize validators with `finality_participation[i] = False`.

Independent and additive — up to 3 penalty units per round. Slashed validators
always get the maximum.

### 6. E1-only slashing

`is_slashable_attestation_data` retained from the descendant-based branch — it
is already exactly the paper's E1: if you signed `finality_target = T` at
`finality_height = H`, any attestation with `target != T` at `height = H` is
slashable. There is no E2 in this design (validators may cast an R1 at height H
and an R2 at the same height; the R2 honest rule requires the same target, but
this is a honest-behaviour rule only).

______________________________________________________________________

## State Changes

### `BeaconState` (simplex finality fields)

| Field                       | Type                                   | Role                                                          |
| --------------------------- | -------------------------------------- | ------------------------------------------------------------- |
| `justified_height`          | `Height`                               | Height at which `justified_checkpoint` was justified          |
| `current_height`            | `Height`                               | Paper's `h`                                                   |
| `current_height_start_slot` | `Slot`                                 | Paper's `s_h` — slot at which the current height began        |
| `justification_targets`     | `List[Slot, VALIDATOR_REGISTRY_LIMIT]` | Per-validator R1 justify target slot (`FAR_FUTURE_SLOT` = ⊥)  |
| `notarization_targets`      | `List[Slot, VALIDATOR_REGISTRY_LIMIT]` | Per-validator max target across R1/R2 (`FAR_FUTURE_SLOT` = ⊥) |
| `notarized_checkpoint`      | `Checkpoint`                           | Paper's `N`                                                   |
| `finality_participation`    | `Bitlist[VALIDATOR_REGISTRY_LIMIT]`    | Paper's `P`; reset on new justification only                  |

Removed relative to the descendant-based design: `current_height_target_slots`,
`current_height_canonical_target`, `previous_height_canonical_target`,
`previous_height_target_participation`.

State size note: the `justification_targets` + `notarization_targets` pair costs
~16 MB at 1M validators (vs ~8 MB in the descendant-based design) — the paper's
per-kind bookkeeping is required for the prefix-notarization walk. Still smaller
than the original 1rf's full-checkpoint design (~40 MB).

### `AttestationData`

Added fields: `height: Height`, `kind: uint8`, `finality_target: Checkpoint`,
`finality_height: Height`, `payload_present: boolean`. The `source` and `index`
fields (from base Gasper/Electra) are gone. `beacon_block_root` is repurposed as
the LMD head vote; `target` is repurposed as the finality target and now carries
`(slot, root)` instead of `(epoch, root)`.

______________________________________________________________________

## State Transition

- `process_round` → `process_inactivity_updates` →
  `process_rewards_and_penalties` → `process_justification_and_finalization` →
  `process_participation_flag_updates`.
- `process_justification_and_finalization` is the paper's `processHeight` state
  machine (three branches described above). It runs AFTER the leak helpers so
  that they see the pre-advance state.
- `advance_height(state, new_notarized, also_justify)`: always sets
  `notarized_checkpoint ← new_notarized`, `current_height ← current_height + 1`,
  `current_height_start_slot ← latest_block_header.slot`; resets
  `justification_targets` and `notarization_targets`. Additionally on
  `also_justify`, sets `justified_checkpoint ← new_notarized`,
  `justified_height ← current_height` (pre-advance), and resets
  `finality_participation`.

______________________________________________________________________

## Reward and penalty scaling

Because `process_round` runs every round rather than every epoch (one round =
`SLOTS_PER_ROUND` slots, so
`ROUNDS_PER_EPOCH = SLOTS_PER_EPOCH // SLOTS_PER_ROUND` rounds per epoch), the
per-epoch reward and penalty formulas must be rescaled. Without compensation,
validators would receive `ROUNDS_PER_EPOCH`× more rewards per epoch, and the
inactivity leak would compound `ROUNDS_PER_EPOCH²`× faster.

### Rewards (1/ROUNDS_PER_EPOCH)

Flag-index rewards in `get_flag_index_deltas` are divided by `ROUNDS_PER_EPOCH`:

```python
reward = base_reward * weight // (total_weight * ROUNDS_PER_EPOCH)
```

Each round distributes a `1/ROUNDS_PER_EPOCH` fraction of the per-epoch reward
budget; summed over the `ROUNDS_PER_EPOCH` rounds in an epoch, the per-epoch
total is unchanged.

### Inactivity penalty (1/ROUNDS_PER_EPOCH²)

`process_inactivity_updates` runs every round because the three-guard
`compute_leak_penalty_units` is round-specific: it inspects per-validator
`justification_targets` / `notarization_targets` / `finality_participation` at
the current (pre-advance) state. This means `inactivity_scores` evolve at the
round cadence: `+INACTIVITY_SCORE_BIAS * penalty_units` per penalized round,
`-1` per zero-penalty round, `-INACTIVITY_SCORE_RECOVERY_RATE` per round when
not in leak.

Penalties apply in `get_inactivity_penalty_deltas`. Two independent scaling
effects compound:

1. **Penalty application frequency**: penalties fire `ROUNDS_PER_EPOCH` times
   per epoch instead of once. This multiplies the per-epoch penalty weight by
   `ROUNDS_PER_EPOCH`.
2. **Score accumulation speed**: `inactivity_scores[i]` grows
   `ROUNDS_PER_EPOCH`× faster because `+ISB` fires per round, not per epoch.
   Since the penalty is proportional to the score, this multiplies the per-epoch
   penalty weight by another `ROUNDS_PER_EPOCH`.

Both factors must be compensated. The denominator carries `ROUNDS_PER_EPOCH²`:

```python
# get_inactivity_penalty_deltas:
penalty_numerator = effective_balance * inactivity_scores[i]
penalty_denominator = (
    INACTIVITY_SCORE_BIAS * INACTIVITY_PENALTY_QUOTIENT_BELLATRIX * (ROUNDS_PER_EPOCH**2)
)
penalty = (penalty_numerator // penalty_denominator) * penalty_units
```

### Why not rescale the score constants?

One might try to compensate by making `INACTIVITY_SCORE_BIAS` smaller (so scores
grow at the same rate per epoch as before). This is cosmetic:
`INACTIVITY_SCORE_BIAS` appears in both the score increment (`+ISB` per round)
and the denominator (`ISB` factor), so

```
penalty ∝ (ISB * rounds_leaked) / (ISB * INACTIVITY_PENALTY_QUOTIENT * RPE²)
       = rounds_leaked / (INACTIVITY_PENALTY_QUOTIENT * RPE²)
```

Rescaling `ISB` cancels. The irreducible compensation is `RPE²`, placed either
in the denominator (as above) or folded into `INACTIVITY_PENALTY_QUOTIENT`.

### Simulation check

A 100-epoch leak simulation comparing the epoch-based and round-based
formulations on a non-finality trajectory showed 0.87% total-penalty deviation
between the two — pure integer-truncation noise. See
[`claude-files/1rf/project.md`](../1rf/project.md) for the original derivation
and simulation notes.

______________________________________________________________________

## Fork Choice

- `Store` gains `root: Root` and `root_key: Tuple[Height, uint8, Slot, Root]`.
- `update_fork_choice_root` filters by `F_s ⪯ C` then running-maxes the key.
- `update_finalized` advances `F_s` only if the candidate strictly extends `F_s`
  AND is an ancestor-or-self of `store.root`.
- `on_block` runs the state transition, dispatches `on_attestation` /
  `on_available_attestation` for every attestation in the block body, then emits
  two cert events (justified + notarized) per imported block and advances
  finalized.
- `get_lmd_ghost_head` walks from `store.root` — no filtered block tree.

Removed helpers: `get_leaf_justifications`, `update_justified`,
`filter_block_tree`, `get_filtered_block_tree`.

______________________________________________________________________

## Validator duties

- R1 first: on the first voting opportunity per state-height, cast an R1
  (justify) vote for the head's latest block header. Optionally attach a
  finality piggyback if the recorded target at `justified_height` equals the
  justified checkpoint.
- R2 on resubmission: if height has not yet advanced on the next voting
  opportunity, cast an R2 (notarize) vote with the same target (R1 lock).
- Finalize piggyback is only valid on R1 (justify) kind votes.
- E1 avoidance: never sign `(finality_height = H, finality_target = T)` if you
  have voted at `height = H` with a target `!= T`.

______________________________________________________________________

## Proven Properties

Full proofs live in
`overleaf/shared-finality/fresh_simplex_with_notarizations.tex`. Proof sketches
in `proofs.md`. Key results:

- **Lemma: per-height justification uniqueness** (`lem:just-unique-height`):
  under `f < n/3` and the honest rule, at most one target can have a 2/3
  `justification_targets` quorum per height.
- **Lemma: main safety** (`lem:mainsafety`): any chain extending past a
  finalized height contains the finalized block; proof uses quorum intersection
  and E1.
- **Lemma: finalized chain** (`lem:finchain`): finalized blocks form a chain.
- **Theorem: accountable safety** (`thm:safety`): conflicting finalization ⇒
  `≥ n/3` validators slashable via E1.
- **Theorem: fork-choice consistency** (`thm:fcconsistency`):
  `F_s ⪯ R_s ⪯ head`.
- **Lemma: upgrade** (`lem:upgrade`): once `(F, h_F)` has been offered to
  `update_fork_choice_root`, `R_s ≻ F` thereafter (lex on keys).
- **Theorem: local finality acceptance** (`thm:finlive`): after a block is
  imported with post-state `F_B`, `F_s ≻ F_B` eventually (subject to `F_B ⪯ R_s`
  guard).
- **Theorem: non-finality period penalty bound** (`thm:tightness`): per-round
  penalty units accumulate at `≥ (T-1) · n/3` over a non-finality period of
  length `T`.
- **Theorem: leak fairness** (`thm:fairness`): an honest validator's R1 (and, if
  needed, R2) exempts it from the notarization-missed guard at the same
  state-height.

______________________________________________________________________

## Files

| File                                                               | Purpose                                        |
| ------------------------------------------------------------------ | ---------------------------------------------- |
| `specs/_features/simplex/beacon-chain.md`                          | Beacon-chain spec (state, processHeight, leak) |
| `specs/_features/simplex/fork-choice.md`                           | Fork-choice spec (Store, cert events)          |
| `specs/_features/simplex/validator.md`                             | Honest-validator rule (R1/R2, E1 avoidance)    |
| `claude-files/descendant-based-ic/project.md`                      | This file                                      |
| `claude-files/descendant-based-ic/evolution.md`                    | Append-only research changelog                 |
| `claude-files/descendant-based-ic/proofs.md`                       | In-repo proof sketches (paper references)      |
| `../overleaf/shared-finality/fresh_simplex_with_notarizations.tex` | Full paper                                     |

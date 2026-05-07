# Simplex with Height Filter and Timeouts

**Branch**: `simplex-height-filter`. Spawned from `descendant-based-ic` (commit
60c68365c) and migrated to the height-filter-and-timeouts design from
`overleaf/shared-finality/height_filter_and_timeouts.tex` sections 1–4.

______________________________________________________________________

## Overview

A finality gadget for `n >= 3f+1` BFT with a single 2/3 quorum threshold. Each
validator casts at most one **justify** (R1) and one **timeout** (R2)
attestation per state-height. The discriminator is the `target` field of
`AttestationData`: `target != Checkpoint()` is a justification vote;
`target == Checkpoint()` is a timeout vote. Justifications are subject to a
**fresh-vote** gate keying them to the current height's interval on the current
chain; timeout votes only require that their height matches
`state.current_height`.

Finalization is separated from height advance:

- `processHeight` has three branches, checked in this order on each round
  boundary: (i) finality `F ← J` if `|finality_participation| >= 2n/3`; (ii)
  justify, advancing height with `J ← T` and `h_j ← h` if some target has 2/3
  weight on `state.justification_targets`; (iii) timeout cert, advancing height
  with no `J` update if 2/3 of `state.timeouts` are set. The finality bit-list
  resets only when `J` advances (justify branch); the timeout-cert branch leaves
  it untouched.

- The fork-choice store maintains `(σ, T, F, J, h_j, h_max)`. `J` is updated via
  `update_justified` as the lex-max over justification cert events observed
  (key: `(h_j, hash(J))`), filtered by `F ⪯ J`. `F` advances via
  `update_finalized` when the candidate strictly extends `F`, descends from `J`,
  and lies in the **viable subtree** (paper lem:viable-finalized).

- The **viable subtree** is the height filter: a leaf `B` is viable iff
  `state.current_height(B) >= h_max - 1`; an internal block is viable iff some
  descendant leaf is viable. `get_lmd_ghost_head` walks the viable subtree only.

- `get_lmd_ghost_head` cascades: walk from `J` when `h_max == h_j + 1` (the
  cascade gate, paper getConfirmed); otherwise walk from `F` (always viable,
  paper lem:F-viable).

______________________________________________________________________

## Key Design Choices

### 1. Two vote kinds via empty-target convention

`AttestationData.target == Checkpoint()` ⇔ timeout vote (R2);
`AttestationData.target != Checkpoint()` ⇔ justification vote (R1). No explicit
`kind` field. The helper `is_timeout_vote(data)` returns
`data.target == Checkpoint()`.

### 2. Viability gate

`is_viable_attestation_target(state, attestation)` requires
`data.height == state.current_height` for any kind. Timeout votes are viable on
the height match alone (they set `timeouts[i]`). Justification votes
additionally require:

- `data.target.slot >= state.current_height_start_slot` (paper's `s_h`)
- `is_target_on_chain(state, data.target)` — target references an actual block
  at its proposal slot on this chain.

Viable timeout votes set `timeouts[i]`. Viable justification votes set both
`justification_targets[i]` and `timeouts[i]`.

### 3. processHeight three branches

`process_justification_and_finalization` runs:

1. **Finality**: if `|P| >= 2n/3`, set `F ← J`. Does NOT advance height.
2. **Justify**: if some target `T` reaches `2n/3` on `justification_targets`,
   set `J ← T`, `h_j ← h`, reset `P`, advance height (resets
   `justification_targets` and `timeouts`).
3. **Timeout cert**: if `|{i : timeouts[i]}| >= 2n/3`, advance height (resets
   `justification_targets` and `timeouts`; `J`, `h_j`, `P` untouched).

### 4. Store: J + h_j + h_max + F (no R_s, no key_R)

Fields:

- `justified_checkpoint: Checkpoint` — paper's `Σ.J`.
- `justified_height: Height` — paper's `Σ.h_j`.
- `finalized_checkpoint: Checkpoint` — paper's `Σ.F`.
- `h_max: Height` — paper's `Σ.h_max`, the maximum `state.current_height` over
  all known block states.

`on_block` runs the state transition, dispatches `on_attestation` /
`on_available_attestation` for every attestation in the block body, then:

1. `store.h_max = max(store.h_max, state.current_height)` — bump h_max.
2. `update_justified(store, state.justified_checkpoint, state.justified_height)`
   — single justification cert event (no notarize event, no second cert).
3. `update_finalized(store, state.finalized_checkpoint)` — three guards: strict
   extension, ancestor-of-J, viability.

`get_lmd_ghost_head` cascades between `J` (frontier case) and `F`, then walks
the viable subtree under majority-weight gating.

### 5. Three-guard leak with `timeouts[i]` for Layer 1

Leak guards (paper alg:leak-processslot, mapped onto our per-round model):

- **Layer 1 (stall)**: no height advance this round → penalize validators with
  `not state.timeouts[i]`.
- **Layer 2 target (justify-missed)**: no new justification → penalize
  validators whose `justification_targets[i]` differs from the plurality slot
  `T*` (or is unset).
- **Layer 2 finalize (finalize-missed)**: pending finality AND no new
  finalization → penalize validators with `not finality_participation[i]`.

Independent and additive (up to 3 penalty units per round). Slashed validators
always get the maximum.

### 6. E1-only slashing — timeout votes ARE slashable

The slashing condition is exactly the paper's E1
(`is_slashable_attestation_data` matches paper def:slashing). Note the
deliberate consequence:

> A timeout vote (`target = Checkpoint()`) at height `H` conflicts with any
> commitment (`finality_height = H`, `finality_target = T ≠ Checkpoint()`) at
> the same height, since `Checkpoint() ≠ T`.

This means an honest validator must NOT cast a timeout vote at height `H` if it
has already attached a finality piggyback to height `H` (per paper
def:voting-rule, the `getFinalVote` rule re-emits the lock target as another R1
in this case). The validator-side construction encodes this guard.

______________________________________________________________________

## State Changes

### `BeaconState` (simplex finality fields)

| Field                       | Type                                   | Role                                                          |
| --------------------------- | -------------------------------------- | ------------------------------------------------------------- |
| `justified_height`          | `Height`                               | Height at which `justified_checkpoint` was justified          |
| `current_height`            | `Height`                               | Paper's `h`                                                   |
| `current_height_start_slot` | `Slot`                                 | Paper's `s_h` — slot at which the current height began        |
| `justification_targets`     | `List[Slot, VALIDATOR_REGISTRY_LIMIT]` | Per-validator R1 justify target slot (`FAR_FUTURE_SLOT` = ⊥)  |
| `timeouts`                  | `Bitlist[VALIDATOR_REGISTRY_LIMIT]`    | Paper's `timeouts[]`; set by R2 or fresh R1 at current height |
| `finality_participation`    | `Bitlist[VALIDATOR_REGISTRY_LIMIT]`    | Paper's `P`; reset only when a new justification fires        |

Removed relative to fresh-simplex-with-notarizations:
`notarization_targets: List[Slot, VALIDATOR_REGISTRY_LIMIT]`,
`notarized_checkpoint: Checkpoint`.

State size note: per-validator footprint is `slot (8B) + bit + bit` (vs the
prior `slot + slot + bit`) — the timeouts bitlist is ~64× smaller than the prior
notarization slot list, so the simplex tracking footprint shrinks by roughly
half.

### `AttestationData`

```
class AttestationData(Container):
    slot: Slot
    beacon_block_root: Root          # LMD head vote
    target: Checkpoint               # Justification target, or Checkpoint() for timeout vote
    height: Height
    finality_target: Checkpoint      # Piggyback target, or Checkpoint() for none
    finality_height: Height          # Piggyback height, or FAR_FUTURE_HEIGHT
    payload_present: boolean
```

Removed relative to the prior design: `kind: uint8`. The empty-target convention
replaces it.

______________________________________________________________________

## State Transition

- `process_round` → `process_inactivity_updates` →
  `process_rewards_and_penalties` → `process_justification_and_finalization` →
  `process_participation_flag_updates`.
- `process_justification_and_finalization` is the paper's `processHeight` state
  machine (three branches above). Runs AFTER the leak helpers so they see the
  pre-advance state.
- `advance_height(state, justify_target: Optional[Checkpoint] = None)`: always
  bumps `current_height`, sets
  `current_height_start_slot ← latest_block_header.slot`, and resets
  `justification_targets` and `timeouts`. If `justify_target` is not None,
  additionally sets `justified_checkpoint ← justify_target`,
  `justified_height ← current_height` (pre-advance), and resets
  `finality_participation`.

### `process_attestation`

`validate_attestation` enforces wire-format well-formedness (now allows
`target = Checkpoint()`). Per attesting validator:
`update_finality_participation` always runs (so older-height votes can still
carry valid finality piggybacks). If `is_viable_attestation_target` holds, set
`state.timeouts[validator_index] = True`; additionally, if the vote is a
justification (`not is_timeout_vote(data)`), set
`state.justification_targets[validator_index] = data.target.slot` and earn a
TIMELY_TARGET reward.

______________________________________________________________________

## Reward and penalty scaling

(Identical to the prior `descendant-based-ic` and earlier simplex designs;
reproduced here for completeness because it is load-bearing for
`process_round`.)

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
`justification_targets` / `timeouts` / `finality_participation` at the current
(pre-advance) state. This means `inactivity_scores` evolve at the round cadence:
`+INACTIVITY_SCORE_BIAS * penalty_units` per penalized round, `-1` per
zero-penalty round, `-INACTIVITY_SCORE_RECOVERY_RATE` per round when not in
leak.

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

Rescaling `INACTIVITY_SCORE_BIAS` is cosmetic: `ISB` appears in both the score
increment and the penalty denominator, so it cancels. The irreducible
compensation is `RPE²`, placed either in the denominator (as above) or folded
into `INACTIVITY_PENALTY_QUOTIENT`.

______________________________________________________________________

## Fork Choice

- `Store` gains `justified_height: Height` and `h_max: Height`. `root: Root` and
  `root_key` (the prior cert-event running max) are removed.
- `update_justified(store, justified_checkpoint, justified_height)`: single
  cert-event helper. `F-filter` (candidate descends from `F`), then lex
  running-max on `(justified_height, hash_tree_root(justified_checkpoint))`.
- `update_finalized`: three guards — strict extension of `F`, ancestor-of-`J`,
  and viability (`is_viable(store, F'.root)`).
- `is_viable_leaf(store, b)`: `block_states[b].current_height >= h_max - 1`
  (with a `GENESIS_HEIGHT` floor when `h_max == 0`).
- `is_viable(store, b)`: leaves use `is_viable_leaf`; internal blocks check
  recursively for any viable leaf descendant.
- `on_block`: state transition → attestation/available-attestation processing →
  bump `h_max` → `update_justified` → `update_finalized`.
- `get_lmd_ghost_head`: cascade gate `store.h_max == store.justified_height + 1`
  selects walk-from (`store.justified_checkpoint` or
  `store.finalized_checkpoint`); restricts to viable children at every step
  under majority-weight gating.

Removed helpers from the prior design: `update_fork_choice_root`,
`maybe_update_justified_checkpoint`.

______________________________________________________________________

## Validator duties

- **R1 (justify) first**: on the first voting opportunity per state-height, cast
  an R1 (justify) vote with `target` set to the chain's latest block header (or
  to the locked target if a retroactive `voted_finality_at[H]` lock applies).
  Optionally attach a finality piggyback when the recorded target/finality at
  `justified_height` are compatible with `J` and finalization is still pending.
- **R2 (timeout) on resubmission**: if height has not yet advanced on a later
  voting opportunity at the same state-height, cast R2 with
  `target = Checkpoint()` and no finality piggyback. **Self-slash guard**: if
  `voted_finality_at[current_height]` is set (the validator earlier attached a
  finality piggyback to `current_height`), the R2 timeout would self-slash via
  E1 — fall back to another R1 with the locked target (no piggyback).
- Finalize piggyback is only valid on R1 (justification) votes, gated on
  `attestation_data.target != Checkpoint()`.
- E1 avoidance: never sign a vote at `height = H` whose target conflicts with a
  prior commitment `(finality_height = H, finality_target = T)`. By paper
  def:slashing, "conflicts" includes `target = Checkpoint()` whenever
  `T != Checkpoint()`.

______________________________________________________________________

## Proven Properties

Full proofs live in `overleaf/shared-finality/height_filter_and_timeouts.tex`.
Proof sketches in `proofs.md`. Key results (with paper labels):

- **Lemma: per-height justification uniqueness** (`lem:just-unique-height`):
  under `f < n/3` and the honest rule, at most one target can attain a 2/3
  justification quorum per height per chain.
- **Lemma: main safety** (`lem:mainsafety`): unless `>= n/3` slashable, any
  chain extending past a finalized height contains the finalized block.
- **Lemma: finalized chain** (`lem:finchain`): finalized blocks form a chain.
- **Theorem: accountable safety** (`thm:safety`): conflicting finalization ⇒
  `>= n/3` validators slashable via E1.
- **Lemma: F is viable** (`lem:F-viable`): `Σ.F ∈ viable_subtree(Σ)`
  invariantly.
- **Lemma: finalized blocks are viable** (`lem:viable-finalized`): once a block
  witnessing `(F, h_F)` is processed, `F ∈ viable_subtree(Σ)` thereafter.
- **Theorem: fork-choice consistency** (`thm:fcconsistency`): once `Σ.F = F`,
  `getConfirmed(Σ, Ω)` returns a descendant of `F` thereafter.
- **Theorem: lock-in** (paper label `thm:lock-in`): once a block witnessing
  `(F, h_F)` is processed, `Σ.J ⪰ F`, `F ∈ viable_subtree`, and `getConfirmed`
  returns a descendant of `F` thereafter.
- **Theorem: order independence** (`thm:orderindep`): the store state after
  processing a set of blocks depends only on the set, not the order.
- **Lemma: justifications stay at or below `Σ.h_j`** (`lem:no-high-just`): under
  `f < n/3`, every justification fired on any processed chain has `h <= Σ.h_j`
  at all subsequent times.
- **Theorem: non-finality period penalty bound** (`thm:tightness`): per-round
  penalty units accumulate at `>= (T-1) · n/3` over a non-finality period of
  length `T`.
- **Theorem: layer 1 leak fairness** (`thm:fairness`): an honest validator whose
  vote (R1 or R2 per the honest rule) is included on its canonical chain at the
  same state-height is exempt from the Layer 1 (stall) penalty at that
  state-height.

Section 5 of the paper (healing under asynchrony) is not implemented at the
consensus-spec level — it requires syncer / view-merge / virtual-block
infrastructure that is out of scope here.

______________________________________________________________________

## Files

| File                                                         | Purpose                                        |
| ------------------------------------------------------------ | ---------------------------------------------- |
| `specs/_features/simplex/beacon-chain.md`                    | Beacon-chain spec (state, processHeight, leak) |
| `specs/_features/simplex/fork-choice.md`                     | Fork-choice spec (Store, viability, cascade)   |
| `specs/_features/simplex/validator.md`                       | Honest-validator rule (R1/R2, E1 avoidance)    |
| `claude-files/simplex-height-filter/project.md`              | This file                                      |
| `claude-files/simplex-height-filter/evolution.md`            | Append-only research changelog                 |
| `claude-files/simplex-height-filter/proofs.md`               | In-repo proof sketches (paper references)      |
| `../overleaf/shared-finality/height_filter_and_timeouts.tex` | Full paper                                     |

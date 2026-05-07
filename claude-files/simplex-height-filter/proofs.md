# Simplex Height Filter — In-Repo Proof Sketches

Full proofs live in
`../overleaf/shared-finality/height_filter_and_timeouts.tex`. This document
sketches the key lemmas and theorems as they map onto the consensus-specs code
paths. Cross-references use the paper's labels (e.g. `lem:just-unique-height`).

______________________________________________________________________

## 0. Definitions and setup

### 0.1 System model

- `n` = total active voting weight (sum of `effective_balance` of active
  unslashed validators), `f` = adversarial weight, `n >= 3f+1`.
- A **quorum** is any set whose combined weight is `>= 2n/3`.
- An **equivocator** is a validator that signs two attestations whose
  `AttestationData` satisfies the slashing condition E1 (defined in §0.4).
- All quorum checks use live effective balances, read from the state that is
  processing the quorum predicate. The cross-chain balance-drift bound is the
  standard one (≤ one epoch of churn).

### 0.2 Heights, vote kinds, and the fresh-vote gate

Each state tracks `current_height: Height`, `current_height_start_slot: Slot`,
and per-validator records:

- `justification_targets[i]: Slot` — the slot of validator `i`'s last R1
  (justification) vote recorded under the fresh-vote gate at the current height;
  `FAR_FUTURE_SLOT` = no record.
- `timeouts[i]: bool` — `True` iff `i` set the timeout marker at the current
  height (paper's `timeouts[i]`). Set by either a timeout vote
  (`target = Checkpoint()`) at the current height or a height-fresh
  justification vote on this chain. Reset on height advance.
- `finality_participation[i]: bool` — whether the voter's finality piggyback has
  been recorded for the current justified checkpoint.

A vote is encoded by
`AttestationData = (slot, beacon_block_root, target, height, finality_target, finality_height, payload_present)`.
The vote is a **timeout vote** iff `target == Checkpoint()`; otherwise it is a
**justification vote**.

A justification vote `v` is **fresh** on state `σ` iff

```
fresh ≡ (v.height == σ.h)
      ∧ (v.target.slot >= σ.s_h)
      ∧ is_target_on_chain(σ, v.target)
```

where `is_target_on_chain(σ, v.target)` verifies (i) `v.target.slot < σ.slot`,
(ii) `σ.block_roots[v.target.slot] == v.target.root`, and (iii) the slot is an
actual proposal slot (not a carried-forward root). A timeout vote does not pass
this gate; it is recorded iff `v.height == σ.h` (paper processVote, timeout-vote
branch).

Only fresh justification votes update `justification_targets`. A fresh
justification AND a height-matching timeout vote BOTH set `timeouts[i]`. A fresh
justification thus subsumes a timeout: if a justification quorum forms at height
`h`, a timeout quorum also holds.

### 0.3 processHeight state machine

`process_justification_and_finalization` evaluates three predicates in order; at
most one height-advancing branch fires per invocation:

1. **Finality** — if `|P| >= 2n/3` then `F ← J`. Does NOT advance height
   (`has_new_finalization`).
2. **Justify** — if some target `T` has
   `|{i : justification_targets[i] == T, not slashed}| >= 2n/3` then `J ← T`,
   `h_j ← h`, `P ← ∅`, and `advance_height` (`compute_justified_checkpoint`).
3. **Timeout cert** — if `|{i : timeouts[i], not slashed}| >= 2n/3` then
   `advance_height` with no `J` update (`has_timeout_quorum`).

`advance_height` sets `h ← h + 1`, `s_h ← latest_block_header.slot`, and resets
`justification_targets`/`timeouts`. The justify branch additionally sets
`h_j ← h_pre`, `J ← T`, and resets `P`.

### 0.4 Slashing (E1 only)

Given two signed attestations `d_1, d_2` from the same validator, ordered by
height (`d_1.height <= d_2.height`), they are slashable iff:

```
d_2.finality_target != Checkpoint()
AND d_1.height == d_2.finality_height
AND d_1.target != d_2.finality_target
```

(symmetric swap also slashable). Note that `d_1.target = Checkpoint()` (timeout)
IS slashable when conflicting with a same-height
`d_2.finality_target = T ≠ Checkpoint()` — paper def:slashing remark.

### 0.5 Store

The store is `(σ_store, T, F, J, h_j, h_max)`:

- `F = store.finalized_checkpoint`.
- `(J, h_j) = (store.justified_checkpoint, store.justified_height)` — paper's
  `(Σ.J, Σ.h_j)`. Updated by `update_justified` via lex running-max on
  `(h_j, hash_tree_root(J))` with the F-filter (candidate descends from `F`).
- `h_max = store.h_max` — paper's `Σ.h_max`, the maximum `state.current_height`
  over all known block states.
- `σ_store` is the store's collection of block states / latest messages.

`update_finalized(F')` advances `F` iff `F' ⪰ F` (strictly), `F' ⪯ J`, and
`F' ∈ viable_subtree(Σ)`. `on_block` first runs the state transition, then bumps
`h_max ← max(h_max, post_state.current_height)`, then offers the post-state's
`(J, h_j)` to `update_justified` and `F` to `update_finalized`.

`get_lmd_ghost_head` cascades:

```
walk_from = J  if h_max == h_j + 1  else  F
```

then walks the viable subtree under majority-weight gating.

______________________________________________________________________

## 1. Accountable safety

### Lemma: per-height justification uniqueness (lem:just-unique-height)

**Statement.** Under `f < n/3` and the honest voting rule (each honest
validator's `Δ.T(h)` is set at most once at each height, and subsequent non-`⊥`
votes at the same height re-emit the locked target), at most one target can
attain a `2n/3` justification quorum on `justification_targets` per height per
chain.

**Sketch.** Two different quorums `Q_T, Q_{T'}` for distinct targets `T, T'` at
the same height each have `>= 2n/3` justification_targets assignments. Quorum
intersection gives `|Q_T ∩ Q_{T'}| >= n/3 + 1 > f`. Some honest member
`v ∈ Q_T ∩ Q_{T'}` would have to have signed two non-`⊥` justification votes at
the same height with different targets. By the honest rule, the second non-`⊥`
vote at height `h` re-emits `Δ.T(h)`, so both votes carry the same target —
contradicting `T ≠ T'`. □

### Lemma: main safety (lem:mainsafety)

**Statement.** Unless `>= n/3` validators are slashable: if `C` is finalized at
height `h`, then `C ⪯ B` for every block `B` with `σ[B].h > h`.

**Sketch.** Some `B^* ⪯ B` advanced height from `h` to `h + 1` via a quorum `Q`
at height `h` on `B^*`'s chain (justify or timeout). Finality of `C` at `h`
provides a quorum `Q_F` of finalize-bearing votes with `finality_height = h`,
`finality_target = C`. Quorum intersection picks non-slashable `v ∈ Q ∩ Q_F`:
`v`'s contribution to `Q` cannot be a timeout (`target = ⊥` would conflict with
`C ≠ ⊥` finality commitment via E1) and cannot be a justification with
`target ≠ C` (same conflict). Therefore `v.target = C` is fresh on `B^*`'s chain
at height `h`, so freshness gives `C = v.target ⪯ B^* ⪯ B`. □

### Lemma: finalized chain (lem:finchain)

**Statement.** Unless `>= n/3` slashable: any two finalized `(C, h), (C', h')`
are compatible, with `C ⪯ C'` whenever `h <= h'`.

**Sketch.** For `h = h'`: finality of `C` gives `Q_F` at height `h` with
`finality_target = C`; justification of `C'` (precondition for finality) gives a
justification quorum `Q'` at height `h` with target `C'`. Quorum intersection
picks honest `v ∈ Q_F ∩ Q'`: E1 forces `C = C'`. For `h < h'`: apply
lem:mainsafety. □

### Theorem: accountable safety (thm:safety)

Immediate from lem:finchain: any two finalized blocks are compatible. □

______________________________________________________________________

## 2. Fork-choice consistency

### Lemma: F is always viable (lem:F-viable)

**Statement.** `Σ.F ∈ viable_subtree(Σ)` invariantly.

**Sketch.** At genesis, `Σ.F = genesis` and
`h_max = GENESIS_HEIGHT = σ[genesis].h`, so `genesis` is a viable leaf.
Inductively: `Σ.F` only changes via `update_finalized`, whose viability guard
ensures the new value is viable. Block additions that don't bump `h_max`
preserve viability (the threshold doesn't move). The remaining case is an
`h_max` bump in `on_block` with new block `B` whose post-state has
`σ[B].h > h_max`; after the bump `h_max = σ[B].h`, and the `on_block` assertion
`Σ.F ⪯ B` makes `B` — a leaf at the moment of addition with `σ[B].h = h_max` — a
viable leaf descendant of `Σ.F`. □

### Lemma: finalized blocks are viable (lem:viable-finalized)

**Statement.** Unless `>= n/3` slashable: if `F` is finalized at height `h_F`
and a block witnessing `(F, h_F)` (i.e., `σ[B].J = F` and `σ[B].h_j = h_F`) has
been processed, then `F ∈ viable_subtree(Σ)` at all subsequent times.

**Sketch.** The justify branch of `processHeight` fired with target `F` on `B`'s
chain, advancing post-state to `σ[B].h ≥ h_F + 1`; hence `h_max ≥ h_F + 1`
thereafter. Pick any leaf `L ∈ Σ.T` with `σ[L].h = h_max`. Then `L` is viable,
and lem:mainsafety gives `F ⪯ L`. So `L` witnesses `F ∈ viable_subtree(Σ)`. □

### Theorem: fork-choice consistency (thm:fcconsistency)

**Statement.** Once `Σ.F = F`, `getConfirmed(Σ, Ω)` returns a descendant of `F`
thereafter, for every `Ω`.

**Sketch.** By thm:finperm and thm:fleqr, `F ⪯ Σ.F ⪯ Σ.J` thereafter. Both
branches of `getConfirmed` return a descendant of `Σ.J` (cascade gate fires) or
`Σ.F` (otherwise), and both descend from `F`. □

### Lemma: upgrade property (lem:upgrade)

**Statement.** Unless `>= n/3` slashable: if `F` is finalized at height `h_F`
and a block witnessing `(F, h_F)` has been processed, then `Σ.J ⪰ F` thereafter.

**Sketch.** Let `F^0 = Σ.F` at the moment `on_block(B)` offered `(F, h_F)` to
`update_justified`. By rem:fs-invariant, `F^0` is genesis or finalized;
lem:finchain makes `F, F^0` comparable. If `F ⪯ F^0`, thm:fleqr gives
`Σ.J ⪰ Σ.F ⪰ F^0 ⪰ F` and is preserved by thm:finperm/thm:fleqr. Otherwise the
F-filter passes, and the lex test brings `(Σ.h_j, hash(Σ.J)) ≥ (h_F, hash(F))`.
The bound is preserved by lem:Rs-key-monotone. At any later moment, `Σ.J` is
justified at some `h' ≥ h_F` on a processed chain: lem:certchain (h' > h_F) or
E1 quorum intersection (h' = h_F) forces `Σ.J ⪰ F`. □

### Lemma: justifications stay at or below `Σ.h_j` (lem:no-high-just)

**Statement.** Unless `>= n/3` slashable, every justification `(C, h)` fired on
any processed chain has `h ≤ Σ.h_j` thereafter.

**Sketch.** The justification fires inside `on_block(B)`, at which the
descriptor `(C, h)` is offered to `update_justified`. `C` lies on `B`'s chain
(it is the justified target), and so does `Σ.F` (by `Σ.F ⪯ B`), so `Σ.F ∼ C`. If
`Σ.F ⪯ C` the F-filter passes and `update_justified` forces `Σ.h_j ≥ h`. If
`Σ.F ≻ C` strictly: state-height monotonicity gives
`h_F = σ[Σ.F].h ≥ σ[C].h = h`, and `Σ.F ⪯ Σ.J` plus state-height monotonicity
gives `Σ.h_j ≥ h_F ≥ h`. lem:Rs-key-monotone preserves the bound. □

### Theorem: lock-in (paper label thm:lock-in)

**Statement.** Unless `>= n/3` slashable: once `(F, h_F)` is processed,
`Σ.J ⪰ F`, `F ∈ viable_subtree`, and `getConfirmed(Σ, Ω)` returns a descendant
of `F` thereafter.

**Sketch.** `Σ.J ⪰ F` by lem:upgrade. `F ∈ viable_subtree(Σ)` by
lem:viable-finalized. If `h_max = h_j + 1`, `getConfirmed`'s first branch fires
and returns a descendant of `Σ.J ⪰ F`. Otherwise `h_max ≥ h_j + 2 ≥ h_F + 2`,
and the second branch returns a block `T` with
`σ[T].h ≥ h_max - 1 ≥ h_F + 1 > h_F`; lem:mainsafety gives `F ⪯ T`. □

### Theorem: order independence (thm:orderindep)

**Statement.** Unless `>= n/3` slashable: the store state after processing a set
of blocks depends only on which blocks have been processed, not their order.

**Sketch.** Each block's post-state is a deterministic function of itself and
its parent's post-state, so `σ` (and thus `h_max` and `viable_subtree`) depend
only on the processed set. `Σ.F` is the maximum of `{σ[B].F : B processed}`
(lem:finchain + thm:finlive). `Σ.J` is the lex-max over
`{(C, h) : (C, h) offered with h ≥ h_{F_max}}`, deterministic by lem:certchain
and the E1 collapse at equal height. □

______________________________________________________________________

## 3. Liveness and acceptance

### Theorem: local finality acceptance (thm:finlive)

**Statement.** Unless `>= n/3` slashable: if `B` is processed by `on_block` and
`σ[B].F = F'`, then after processing `Σ.F ⪰ F'`.

**Sketch.** If `Σ.F ⪰ F'` already, monotonicity preserves the bound. Otherwise
some ancestor `B' ⪯ B` has `σ[B'].J = F'` and `σ[B'].h_j = h_{F'}`; lem:upgrade
gives `Σ.J ⪰ F'` (ancestor-of-J guard passes), lem:viable-finalized gives
`F' ∈ viable_subtree(Σ)` (viability guard passes). Strict-extension is given.
All three guards in `update_finalized` pass. □

### Theorem: non-finality period penalty bound (thm:tightness)

**Statement.** During a non-finality period of length `T` rounds, every chain
accumulates `>= (T - 1) · n/3` total penalty units across the three guards.

**Sketch.** Three guards (Layer 1 stall, Layer 2 target, Layer 2 finalize) are
independent and additive:

- **Layer 1 (stall)**: fires when no height advance happens. The set of
  validators with `timeouts[i] = false` is `> n/3` (else a 2/3 timeout quorum
  would have advanced height).
- **Layer 2 target**: fires on stable-stuck slots (no `J` advance, gap closed).
  Plurality count `< 2n/3`, so the bumped set
  `{i : justification_targets[i] ≠ T*}` has size `> n/3`.
- **Layer 2 finalize**: fires on pending-stuck slots (`Σ.F ≺ Σ.J`, no
  F-advance). `|P| < 2n/3`, so `|{i : i ∉ P}| > n/3`.

At most one slot is "transition" (justify branch fires, opens gap); every other
slot is pending-stuck or stable-stuck. Sum: `>= (T-1) · n/3`. □

### Theorem: layer 1 leak fairness (thm:fairness)

**Statement.** Assume `f < n/3` and all honest follow def:voting-rule. An honest
validator `i`, with vote `v` produced at the current slot per def:voting-rule
and included on any extension `T'` of its canonical chain still at the same
state-height, sets `timeouts[i] = true` in `σ[T']`. Hence Layer 1 increment to
`i`'s penalty counter on canonical is zero while canonical state-height stays at
`h_c`.

**Sketch.** `getFinalVote` produces `v` in one of three target branches:
first-vote (target = `getConfirmed(Σ, Ω)`), lock (target = `Δ.T(h_c)`, fires
when `Δ.λ(h_c) = true`), or timeout (target = `⊥`, fires when `Δ.T(h_c) ≠ ⊥` and
`Δ.λ(h_c) = false`). Lemma honest-vote-sets-timeout shows the result is either a
timeout vote at `h_c` or a justification with target `≼ T` and
`σ[v.target].h = h_c`. The lock-branch case requires `h_c ≤ h_max - 1` (the
witnessing block bumped `h_max`); since `getConfirmed` returns blocks at
`σ`-height `≥ h_max - 1`, this collapses to `h_c = h_max - 1` and the lock
target equals `Σ.J` by lem:no-high-just plus lem:just-unique-height.

In the timeout sub-case, `processVote` sets `timeouts[i] = true` because
`v.height = σ[T'].h = h_c`. In the non-timeout sub-case, freshness on `T'` holds
(target on chain at the right height), so `processVote` sets both
`justification_targets[i]` and `timeouts[i]`. Either way, `timeouts[i] = true`
in `σ[T']` and persists on canonical descendants until the next height advance.
□

______________________________________________________________________

## 4. Cross-references

- Vote structure: paper def:vote ↔ spec `AttestationData` (no `kind` field;
  `target == Checkpoint()` distinguishes timeout from justification).
- Fresh gate: paper height-freshness paragraph and `processVote` ↔ spec
  `is_viable_attestation_target`, `process_attestation` (which sets
  `timeouts[i]` for timeout votes and both `justification_targets[i]` and
  `timeouts[i]` for viable justification votes inline).
- processHeight: paper alg:state-machine ↔ spec
  `process_justification_and_finalization`, `advance_height`,
  `compute_justified_checkpoint`, `has_timeout_quorum`, `has_new_finalization`.
- Slashing E1 (incl. timeout slashability): paper def:slashing ↔ spec
  `is_slashable_attestation_data`.
- Store: paper def:store, alg:store ↔ spec `Store`, `update_justified`,
  `update_finalized`, `is_viable_leaf`, `is_viable`, `on_block`,
  `get_lmd_ghost_head`.
- Leak: paper alg:leak-processslot ↔ spec `process_inactivity_updates`,
  `compute_leak_penalty_units` (Layer 1 guard checks `not state.timeouts[i]`).
- Voting rule: paper def:voting-rule, alg:voting-rule ↔ spec `validator.md`
  R1/R2 rule (with R2 self-slash guard for finality lock).
- Safety: thm:safety. FC consistency: thm:fcconsistency. Lock-in: thm:lock-in.
  Order independence: thm:orderindep. Tightness: thm:tightness. Fairness:
  thm:fairness.
- Viability: def:viable, lem:F-viable, lem:viable-finalized.
- Justification monotonicity: lem:Rs-key-monotone, lem:no-high-just.

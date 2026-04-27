# Fresh-Simplex-With-Notarizations — In-Repo Proof Sketches

Full proofs live in
`../overleaf/shared-finality/fresh_simplex_with_notarizations.tex`. This
document sketches the key lemmas and theorems as they map onto the
consensus-specs code paths. Cross-references use the paper's labels (e.g.
`lem:just-unique-height`).

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

### 0.2 Heights and the fresh-vote gate

Each state tracks `current_height: Height`, `current_height_start_slot: Slot`,
and three per-validator records:

- `justification_targets[i]: Slot` — the slot of validator `i`'s last R1
  (justify) vote recorded under the fresh-vote gate at the current height;
  `FAR_FUTURE_SLOT` = no record.
- `notarization_targets[i]: Slot` — the highest slot of validator `i`'s votes at
  the current height across both kinds (R1 or R2); `FAR_FUTURE_SLOT` = none.
- `finality_participation[i]: bool` — whether the voter's finality piggyback has
  been recorded for the current justified checkpoint.

A vote `v = (height, slot, target, kind, finality_target, finality_height)` is
**fresh** on state `σ` iff

```
fresh ≡ (v.height == σ.h)
      ∧ (v.target.slot >= σ.s_h)
      ∧ is_target_on_chain(σ, v.target)
```

where `is_target_on_chain(σ, v.target)` in turn verifies (i)
`v.target.slot < σ.slot`, (ii) `σ.block_roots[v.target.slot] == v.target.root`,
and (iii) the slot is an actual proposal slot (not a carried-forward root, i.e.
`σ.block_roots[v.target.slot] != σ.block_roots[v.target.slot - 1]`). Only fresh
votes update `justification_targets` / `notarization_targets`; stale-height
votes are rejected at inclusion time by `process_attestation`.

### 0.3 processHeight state machine

`process_justification_and_finalization` evaluates three predicates in order; at
most one height-advancing branch fires per invocation:

1. **Finality** — if `|P| >= 2n/3` then `F ← J`. Does NOT advance height.
   (`has_new_finalization`)
2. **Justify** — if some target `T` has
   `|{i : justification_targets[i] == T, not slashed}| >= 2n/3` then `J ← T`,
   `N ← T`, `P ← ∅`, and `advance_height`. (`compute_justified_checkpoint`)
3. **Prefix-notarize** — if the cumulative walk on `notarization_targets` down
   from `latest_block_header.slot` to `current_height_start_slot` reaches `2n/3`
   at some on-chain slot `T` then `N ← T` and `advance_height` (no change to
   `J`, no reset of `P`). (`compute_notarized_checkpoint`)

`advance_height` sets `h_n ← h`, `h ← h + 1`, `s_h ← latest_block_header.slot`,
and resets `justification_targets`/`notarization_targets`.

### 0.4 Slashing (E1 only)

Given two signed attestations `d_1, d_2` from the same validator, ordered by
height (`d_1.height <= d_2.height`), they are slashable iff:

```
d_2.finality_target != Checkpoint()
AND d_1.height == d_2.finality_height
AND d_1.target != d_2.finality_target
```

(symmetric swap also slashable). There is no double-target height condition; an
honest validator may cast R1 and R2 at the same height provided they carry the
**same target** (enforced by the R1 lock, a behavioural rule).

### 0.5 Store

The store is `(σ_store, T, F_s, R_s, key_R)`:

- `F_s = store.finalized_checkpoint`.
- `R_s = store.root`, `key_R = store.root_key` — updated by
  `update_fork_choice_root` via lex running-max on `(height, b, slot, root)`
  where `b ∈ {0, 1}` discriminates prefix-notarization (0) from justification
  (1) events.
- `σ_store` is the store's collection of block states / latest messages.

`update_fork_choice_root` first filters by `F_s ⪯ C` (paper def:store), then
running-maxes the key. `update_finalized(C)` advances `F_s` iff
`C.slot > F_s.slot` AND `C` strictly extends `F_s` AND `F_B ⪯ R_s`. `on_block`
emits two cert events per imported block (one bit-1 for
`state.justified_checkpoint`, one bit-0 for `state.notarized_checkpoint`), then
calls `update_finalized(state.finalized_checkpoint)`, then
`maybe_update_justified_checkpoint` to mirror the running bit-1 event onto
`store.justified_checkpoint`.

______________________________________________________________________

## 1. Accountable safety

### Lemma: per-height justification uniqueness (lem:just-unique-height)

**Statement.** Under `f < n/3` and the honest R1 rule (each honest validator
casts at most one R1 vote per state-height), at most one target can attain a
`2n/3` justify quorum on `justification_targets` per height per chain.

**Sketch.** Two different quorums `Q_T, Q_{T'}` for distinct targets `T, T'` at
the same height must each have `>= 2n/3` justification_targets assignments.
Quorum intersection gives `|Q_T ∩ Q_{T'}| >= n/3`. Each honest member of the
intersection would have to have `justification_targets[i] = T` and
`justification_targets[i] = T'` simultaneously, which is impossible
(justification_targets is single-valued). Therefore the intersection is
`<= f < n/3`, contradiction. □

This is used implicitly by the liveness proofs to rule out racing justify
quorums on the same chain.

### Lemma: main safety (lem:mainsafety)

**Statement.** If `C` is finalized at height `H` on some chain, then any other
chain that reaches a height `H' > H` through some sequence of
justify/prefix-notarize advances contains `C` as an ancestor at slot `C.slot`.

**Sketch.** Finalization at `(H, C)` requires:

- **(a)** A justify quorum `J` with `|J| >= 2n/3` at height `H`, each member `i`
  having signed an R1 vote with `target = T_i` and
  `justification_targets[i] = T_i.slot = C.slot`. By lem:just-unique-height (on
  this chain) all `T_i = C`.
- **(b)** A finality quorum `F` with `|F| >= 2n/3`, each member `i` having
  signed an R1 attestation with `(finality_target, finality_height) = (C, H)`
  AND `target = T_i = C` at height `H`.

Now take any chain Y reaching a higher height. Consider the intersection of `F`
with any quorum `Q_Y` that advanced height `H` on Y. The intersection has weight
at least `n/3`, strictly greater than `f`, so it contains an honest validator.
That validator signed `(finality_target = C, finality_height = H)` on F's chain,
and signed a vote at `height = H` on Y's chain. If Y's advance at height `H`
justified a target `T' != C` (or prefix-notarized a non-extension of `C`), the
honest validator's vote at height `H` would have `target != C`, violating E1.
Therefore `Q_Y` advanced through a target extending `C`. By induction along Y's
advance sequence, every block on Y at slot at least `C.slot` extends `C`. □

### Lemma: finalized chain (lem:finchain)

**Statement.** The set of finalized checkpoints forms a chain (totally ordered
by ancestry).

**Sketch.** Given two finalized `C_1, C_2` at heights `H_1 <= H_2`. Applying
lem:mainsafety with `H = H_1`, `C = C_1`, and the chain on which `C_2` was
finalized, we get that the chain containing `C_2` at height `H_2` contains `C_1`
as an ancestor. Therefore `C_1 ⪯ C_2`. □

### Theorem: accountable safety (thm:safety)

**Statement.** Two conflicting finalized checkpoints `C_1, C_2` imply at least
`n/3` validators signed mutually-slashable (E1) pairs.

**Sketch.** "Conflicting" means neither is an ancestor of the other. Without
loss of generality, consider the same height `H` of both (if heights differ,
apply lem:mainsafety to derive a contradiction directly). The two justify
quorums `J_1, J_2` intersect in `>= n/3`. Each intersection member has signed R1
with `target = T_1` on chain 1 AND with `target = T_2 ≠ T_1` on chain 2 — but
those votes are not directly slashable by E1 (E1 requires a finality
commitment). Extend the argument to the finality quorums `F_1, F_2`:
`|F_1 ∩ F_2| >= n/3`, each member signed
`(finality_target = C_1, finality_height = H)` on chain 1 and voted
`target = T_2 ≠ C_1` on chain 2 — this triggers E1. □

______________________________________________________________________

## 2. Fork-choice consistency

### Theorem: fork-choice consistency (thm:fcconsistency)

**Statement.** At all times, `F_s ⪯ R_s ⪯ head` in the store.

**Sketch.** `update_finalized` explicitly guards `F_B ⪯ R_s` before advancing
`F_s`. `update_fork_choice_root` explicitly filters candidates by `F_s ⪯ C`, so
every running-max update keeps `F_s ⪯ R_s` as invariant. `get_lmd_ghost_head`
starts from `R_s` and walks forward (`head ⪰ R_s`). By transitivity,
`F_s ⪯ R_s ⪯ head`. □

### Lemma: upgrade property (lem:upgrade)

**Statement.** If an event `(F, h_F)` with bit `b = 1` has been offered to
`update_fork_choice_root` and passed the `F_s ⪯ F` filter, then thereafter
`key_R >= (h_F, 1, F.slot, F.root)`, and in particular `R_s ≻ F_s` for all
subsequent times where `F_s.slot < F.slot`.

**Sketch.** Key comparison is lexicographic with height primary, bit secondary.
Once `key_R` is raised to `(h_F, 1, F.slot, F.root)`, any smaller tuple (same
height bit-0, lower height, or same key with smaller slot/root) fails to
displace it. Future cert events either strictly increase `key_R` or are
discarded. □

______________________________________________________________________

## 3. Liveness

### Theorem: local finality acceptance (thm:finlive)

**Statement.** If `state.finalized_checkpoint = F_B` is observed in a block's
post-state, then eventually `F_s >= F_B` in every honest node's store (subject
to the `F_B ⪯ R_s` guard in `update_finalized`).

**Sketch.** On each honest node, when a block with post-state finalizing `F_B`
is imported, `on_block` emits the bit-1 cert event `(H_F, 1, F_B)` via
`update_fork_choice_root` (passing `F_s ⪯ F_B` by hypothesis), which
running-maxes `key_R` to at least `(H_F, 1, F_B.slot, F_B.root)`. By
lem:upgrade, `R_s.slot >= F_B.slot` and `R_s` is a descendant of `F_B` (they
differ only by whether `R_s` has subsequently advanced to a later height's cert
event). `update_finalized(F_B)` then succeeds because `F_B ⪯ R_s` holds and
`F_B.slot > F_s.slot` (assuming the new finalized strictly extends the old). □

### Theorem: non-finality period penalty bound (thm:tightness)

**Statement.** During a non-finality period of length `T` rounds, every
non-participating validator accrues at least `(T - 1) · n/3` penalty weight
units total (summed across the three guards).

**Sketch.** The three guards are independent and additive:

- Notarization missed (`notarization_targets[i] = ⊥`): fires whenever no height
  advance happens. Under `f < n/3`, if >= 2n/3 honest validators cast fresh
  votes, either the justify branch fires (on the same target,
  lem:just-unique-height) or the prefix-notarize branch fires (on some on-chain
  ancestor). Therefore a stalled round implies the honest unvoted mass is at
  least `n/3`. Each unvoted honest validator accrues one notarization-missed
  penalty unit.
- Justification missed (`justification_targets[i] ≠ T*`): fires on the first
  round after a stalled justify. Similar argument: < 2n/3 unique R1 assignments
  imply at least `n/3` misaligned or missing.
- Finalization missed (pending `F ≠ J`): fires whenever the finality quorum has
  not yet reached 2n/3 and `F ≠ J`. Accrues `n/3` weight per round during the
  pending window by the same argument applied to `P`.

Summing over `T - 1` rounds gives the stated bound. A fourth "slashed = 3 units"
hard-cap keeps evicted validators maximally penalized. □

### Theorem: leak fairness (thm:fairness)

**Statement.** An honest validator that casts its R1 (and, if needed, R2) at
each state-height is not penalized by the notarization-missed guard at that
height.

**Sketch.** By construction, a fresh R1 vote sets both
`justification_targets[i] := target.slot` and
`notarization_targets[i] := max(notarization_targets[i], target.slot)`. A
subsequent fresh R2 vote sets `notarization_targets[i]` only. In either case,
`notarization_targets[i] ≠ FAR_FUTURE_SLOT` after the vote is recorded, so the
notarization-missed guard does not fire for that validator. The validator's
finality piggyback (R1 only, subject to the E1 invariant) drops the
finalization-missed guard. The justification-missed guard fires only when the
validator's `justification_targets[i]` is off the plurality — under the R1 lock
and head convergence, honest validators vote the plurality target. □

______________________________________________________________________

## 4. Cross-references

- Vote structure: paper def:vote (tex:145-153) ↔ spec `AttestationData`.
- Fresh gate: paper Height freshness paragraph (tex:184-190) and processVote
  (tex:266-279) ↔ spec `is_viable_attestation_target`, `process_attestation`.
- processHeight: paper alg:state-machine (tex:282-305) ↔ spec
  `process_justification_and_finalization`, `advance_height`.
- Slashing E1: paper def:slashing (tex:222-228) ↔ spec
  `is_slashable_attestation_data`.
- Store: paper def:store, alg:store (tex:419-494) ↔ spec `Store`,
  `update_fork_choice_root`, `update_finalized`.
- Leak: paper alg:leak-processslot (tex:642-682) ↔ spec
  `process_inactivity_updates`, `compute_leak_penalty_units`.
- Voting rule: paper def:voting-rule, alg:voting-rule (tex:748-787) ↔ spec
  `validator.md` R1/R2 rule.
- Safety: paper thm:safety (tex:383).
- Fork-choice consistency: paper thm:fcconsistency (tex:540).
- Leak tightness: paper thm:tightness (tex:721).
- Leak fairness: paper thm:fairness (tex:883).

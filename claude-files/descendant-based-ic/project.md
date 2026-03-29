# Descendant-Based IC Consensus Finality Gadget

**Branch**: `descendant-based-ic` (built on `simplex`)
**Status**: Spec implemented, proofs complete, all properties verified

---

## Overview

An IC consensus finality gadget for Ethereum:
- Validators can vote for **multiple targets per height** (no E1 restriction)
- **Single slashing condition (E2)**: finalize commitment requires exclusive voting
- **Descendant-based justification**: 2/3 votes on the same chain = justified (suffix-sum)
- **No timeout mechanism**. Height advances only via justification.
- **Order-independent `update_justified`** + derived fork-choice filter (Prefix-IC formalization)

### What changes from Simplex

| Aspect | Simplex (wildcard) | Descendant-based IC |
|--------|-------------------|---------------------|
| Vote model | One target per height (E1) + timeout | Multiple targets per height, no timeout |
| Slashing | E1 (double-target) + E2 (finalize-target) | **E2 only** |
| Justification | Per-slot counting | **Descendant-based** suffix-sum |
| Height advance | Justification OR timeout-assisted OR pure timeout | **Justification only** |
| Leak | Majority-target + conditional exemption | **Two-layer** (voted-on-chain / voted-above-justified) |
| State fields | target_slots + timeout_bitlist + floor | **target_slots only** |
| Store | `justification_floor_slot` + filtering | **`update_justified`** (order-independent walk) + **`get_filtered_block_tree`** (derived conflict detection) |
| Finalize target | `finalize_target = justified_checkpoint` | `finalize_target = voter's actual target` (descendant) |
| From-block attestations | Epoch check relaxed (adversary transfer) | **Epoch-bounded** (honest suffice for suffix-sum) |

---

## Key Design Decisions

### 1. Descendant-based justification (suffix-sum)

`compute_round_outcome` builds `slot_weights[s]`, iterates highest to lowest,
accumulating a suffix-sum. The highest slot where suffix-sum >= 2/3 is the
justified checkpoint. All votes at that slot and above contributed. On a linear
chain, higher slot = descendant, so this is descendant-based counting.

### 2. E2-only slashing

E2: if `finalize_target = T` at `finalize_height = H`, no vote for any target
other than T at height H. No E1 (no double-target condition). Validators can
freely vote for multiple targets at the same height without slashing risk, as
long as they don't carry a conflicting finalize piggyback.

### 3. Finalize behavioral rule

Honest validators only sign `finalize_target = T` if T was itself justified
(= the justified checkpoint at the time). This prevents locks on side branches:
a validator who voted for a descendant of the justified checkpoint but on a
different branch would otherwise be locked on an off-chain target.

### 4. Two-layer leak

- **Layer 1 (stall)**: exempt if voted on this chain (`target_slots[i] != FAR_FUTURE_SLOT`)
- **Layer 2 (advance)**: exempt if voted above justified (`target_slots[i] > justified_checkpoint.slot`), plus independent finalize check at `current_height == justified_height + 1`

Strict `>` in Layer 2 prevents free rounds (advance without checkpoint update).
Stacking (target + finalize) ensures amortized N/3 penalty units per round.

### 5. Order-independent `update_justified` + derived filter (Prefix-IC formalization)

`get_leaf_justifications` extracts `(justified_checkpoint, justified_height)`
pairs from leaf block states in `store.block_states` — no separate stored
list is needed. By Lemma 1.5 (justified checkpoint slot monotonicity), each
chain's leaf state dominates its ancestors, so only leaf states contribute
distinct candidates.

**`update_justified` (descendant walk):** Starting from
`store.finalized_checkpoint`, repeatedly pick the strict descendant among the
leaf justifications with the highest justified height (tiebreak: slot, root).
Terminates when no descendant exists. Upgrades to the most-progressed branch.
The result is order-independent: `store.block_states` is a map keyed by block
root, so the same blocks produce the same candidates regardless of insertion
order.

**`get_filtered_block_tree` (conflict detection):** Derives `justified_height`
as the max height from `get_leaf_justifications`. Collects checkpoints at that
height, sorts by slot, and performs a single-pass adjacent-pair ancestry check.
When conflicting checkpoints are detected, `filter_block_tree` prunes leaves at
`current_height <= justified_height`, restricting the fork choice to branches
that have advanced past the conflicting height. Under normal conditions (no
conflicts), returns `store.blocks` directly (zero overhead).
`get_lmd_ghost_head` uses the filtered block tree.

**`update_finalized`** is separated: advances finalized if the candidate has a
higher slot and justified descends from it (F <= J guard).

**No stored `justified_height` or `candidate_justified`**: everything is
derived from `store.block_states` on demand. The Store contains only
`justified_checkpoint` and `finalized_checkpoint`. Replaces the old
`update_checkpoints` / `should_update_justified` /
`has_conflicting_justification` machinery.

### 7. Epoch-bounded from-block attestations

From-block attestations skip if their epoch is older than current/previous. In
the IC model with descendant-based justification, honest validators alone
(locked + non-locked > 2n/3) suffice for the suffix-sum. Old adversary vote
transfer is unnecessary.

---

## Proven Properties

### Safety
- **Theorem 1 (Accountable safety)**: conflicting finalization requires >= n/3 slashable
- **Certificate transferability** (Theorem P_CT): justification certificates can be replayed cross-chain

### Liveness
- **Theorem 2 (Amortized N/3)**: >= N/3 penalty units per round during non-finality
- **Lemma 2.6 (Advance-without-update penalized)**: strict `>` guard ensures no free rounds

### Fairness
- **Theorem 3 (Leak fairness)**: under synchrony, honest validators are not penalized on the canonical chain. The behavioral rule ensures locked targets are on-chain; `update_justified` picks the most-progressed descendant and `get_filtered_block_tree` restricts to branches past the conflicting height (all locks expired)

### Store Safety
- **Theorem 4a (Finalization permanence)**: `store.finalized_checkpoint` only advances to descendants
- **Theorem 4b (Fork-choice consistency)**: `get_head` always returns a block descending from finalized (F <= J invariant)
- **Theorem 4c (Pre-finalization lock-in)**: under f < n/3, importing a justified (C, H) locks the fork-choice onto C's chain permanently
- **Theorem 4 (No deadlocks)**: certificate transfer + leak + fork-choice convergence prevent permanent stalls

### Honest Validator Safety
- **Theorem 5 (No self-slashability)**: honest validators following the behavioral rules cannot produce slashable attestation pairs

---

## Files

| File | Purpose |
|------|---------|
| `specs/_features/simplex/beacon-chain.md` | Beacon chain spec |
| `specs/_features/simplex/fork-choice.md` | Fork choice spec |
| `claude-files/descendant-based-ic/proofs.md` | Safety/liveness/fairness/store proofs |
| `claude-files/descendant-based-ic/evolution.md` | Design history (10 phases) |
| `claude-files/descendant-based-ic/project.md` | This file |

# Descendant-Based IC Consensus Finality Gadget

**Branch**: `descendant-based-ic` (built on `simplex`)
**Status**: Spec implemented, proofs complete, all properties verified

---

## Overview

An IC consensus (independent choices) finality gadget for Ethereum:
- Validators can vote for **multiple targets per height** (no E1 restriction)
- **Single slashing condition (E2)**: finalize commitment requires exclusive voting
- **Descendant-based justification**: 2/3 votes on the same chain = justified (suffix-sum)
- **No timeout mechanism**. Height advances only via justification.
- **Conditional fork-choice filter** for conflicting justifications

### What changes from Simplex

| Aspect | Simplex (wildcard) | Descendant-based IC |
|--------|-------------------|---------------------|
| Vote model | One target per height (E1) + timeout | Multiple targets per height, no timeout |
| Slashing | E1 (double-target) + E2 (finalize-target) | **E2 only** |
| Justification | Per-slot counting | **Descendant-based** suffix-sum |
| Height advance | Justification OR timeout-assisted OR pure timeout | **Justification only** |
| Leak | Majority-target + conditional exemption | **Two-layer** (voted-on-chain / voted-above-justified) |
| State fields | target_slots + timeout_bitlist + floor | **target_slots only** |
| Store | `justification_floor_slot` + filtering | **Store-level max** + conditional filter on conflict |
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

### 5. Store-level max

When the candidate and current justified checkpoints are non-conflicting (same
chain): keep the higher-slot checkpoint, advance height to max. When conflicting
(different forks): deterministic tiebreaker (height, slot, root).

### 6. Conflicting-justification fork-choice filter

`Store.has_conflicting_justification` is set when `update_checkpoints` sees
conflicting checkpoints at the same `justified_height`. `filter_block_tree`
prunes leaf blocks at `current_height <= store.justified_height`. The canonical
chain is restricted to chains that advanced past the conflicting height — where
all E2 locks expired. Cleared when `justified_height` advances. Under normal
conditions, never active.

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
- **Theorem 3 (Leak fairness)**: under synchrony, honest validators are not penalized on the canonical chain. Two cases: no conflict (behavioral rule ensures locked targets on-chain), conflict (filter ensures canonical chain past justified height, all locks expired)

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
| `claude-files/descendant-based-ic/evolution.md` | Design history (9 phases) |
| `claude-files/descendant-based-ic/project.md` | This file |

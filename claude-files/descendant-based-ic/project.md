# Descendant-Based IC Consensus Finality Gadget

**Branch**: `descendant-based-ic` (built on `simplex`)
**Status**: Spec implemented, proofs adapted, under review
**Base concept**: IC consensus (Internet Computer) adapted as Ethereum finality gadget with descendant-based justification

---

## Overview

Replace the Simplex wildcard/timeout model with a pure IC consensus approach:
- Validators can vote for **multiple targets per height** (no E1 restriction)
- **Single slashing condition (E2)**: finalize commitment requires exclusive voting
- **Descendant-based justification**: 2/3 votes on the same chain = justified
- **No timeout mechanism**. Height advances only via justification.

### What changes from Simplex

| Aspect | Simplex (wildcard) | Descendant-based IC |
|--------|-------------------|---------------------|
| Vote model | One target per height (E1) + timeout | Multiple targets per height, no timeout |
| Slashing | E1 (double-target) + E2 (finalize-target) | **E2 only** |
| Justification | Per-slot: `slot_weights[T] >= 2/3` | **Descendant-based**: suffix-sum >= 2/3 |
| Height advance | Justification OR timeout-assisted OR pure timeout | **Justification only** |
| Leak | Majority-target + conditional exemption | **Voted on this chain** (simple) |
| State fields | target_slots + timeout_bitlist + floor | **target_slots only** (no timeout, no floor) |
| Store | `justification_floor_slot` + `should_update_justified` | **Store-level max** for non-conflicting ancestors |
| Finalize target | `finalize_target = justified_checkpoint` (exact) | `finalize_target = voter's actual target` (descendant) |

### What gets removed
- `current_height_timeout_participation` bitlist
- `justification_floor_slot`
- E1 slashing condition
- Majority-target computation for leak
- 1/3 minimum target support threshold
- Conditional Layer 1 exemption
- Wildcard/timeout-assisted advance
- Slashed-to-timeout conversion
- Net: -146 lines from spec

---

## Key Design Decisions

### 1. Descendant-based justification (suffix-sum)

On a single chain, all tracked targets are on-chain (verified by `is_target_on_chain`).
Higher slot = descendant (chain is linear). So the suffix-sum over sorted target
slots correctly computes descendant-based support.

The highest slot where suffix-sum >= 2/3 is selected (most specific justified
target). All lower slots also qualify but are less specific.

### 2. Overwrite rule for target_slots

`target_slots[i]` records the highest-slot on-chain target the validator voted
for. On new vote: overwrite if `data.target.slot > current_slot`. This maximizes
each validator's contribution to the suffix-sum.

### 3. E2-only slashing

E2: if `finalize_target = T` at `finalize_height = H`, no vote for any target
other than T at height H.

E1 (height double-target) is removed. Validators CAN vote for multiple targets
at the same height without slashing risk (as long as they don't commit to
finalize at that height).

### 4. Finalize_target = voter's actual target

The voter signs `finalize_target = D` where D is their actual target at
`finalize_height`. On-chain: accepted if `D.slot >= justified_checkpoint.slot`
(D is a descendant of the justified checkpoint). E2 binds: the voter only voted
D at the finalize height.

### 5. Simple leak

- **Stall**: 1 penalty unit if no vote on this chain (target_slots == FAR_FUTURE_SLOT)
- **Advance**: 1 penalty unit if no vote + 1 unit if finalize pending at J+1

Tight bound: if on-chain votes >= 2/3, descendant justification fires (not a
stall). If < 2/3, leaked > 1/3.

### 6. Store-level max

When the store receives a justified checkpoint T' that is a non-conflicting
ancestor of the current store.justified_checkpoint T: keep T (higher slot),
advance store.justified_height. Prevents fork-choice regression.

---

## Safety Argument

If T is finalized at height H, any chain advancing past H must contain T.

**Proof sketch**: The finalize quorum F (>= 2/3) signed finalize_target = D_i
(their actual targets, descendants of T). E2 constrains: they only voted D_i at
H. Any justification quorum Q at H on chain Y (>= 2/3) overlaps with F by >=
1/3. Overlap members voted D_i at H (E2) and their votes were processed on chain
Y (in Q). So D_i is on chain Y. Since D_i descends from T (D_i.slot >= T.slot on
chain A where T was justified), and block tree ancestry is global (same root =
same block everywhere), T is on chain Y.

---

## Open Questions

1. **Advance without justification update**: When descendant-based justification
   fires but justified_slot < justified_checkpoint.slot, the height advances
   without updating the checkpoint. Does this reintroduce the "past J+1" leak
   gap? Under f < n/3, this shouldn't occur (honest vote at or above justified
   slot), but needs formal verification.

2. **Store-level max implementation**: Needs `are_non_conflicting` helper in the
   fork-choice spec. Not yet implemented.

3. **Tests**: No tests written.

---

## Files

| File | Purpose |
|------|---------|
| `specs/_features/simplex/beacon-chain.md` | The spec |
| `claude-files/descendant-based-ic/proofs.md` | Safety/liveness/fairness proofs |
| `claude-files/descendant-based-ic/project.md` | This file |

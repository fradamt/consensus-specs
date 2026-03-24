# Simplex Finality Gadget — Proofs

## 0. Definitions and Setup

### 0.1 System Model

We consider a proof-of-stake system with total active voting weight **n** and adversarial weight at most **f**, where **n >= 3f + 1**. A **quorum** is any set of validators whose combined weight is at least 2n/3. The spec encodes this threshold as `weight * FINALITY_QUORUM_DENOMINATOR >= total * FINALITY_QUORUM_NUMERATOR` with `FINALITY_QUORUM_NUMERATOR = 2` and `FINALITY_QUORUM_DENOMINATOR = 3`.

An **equivocator** is a validator that signs two attestations whose `AttestationData` satisfies the slashing condition (defined in §0.7). The adversary controls at most f weight and may cause any adversarial validator to equivocate. Honest validators follow the protocol.

Weights refer to sums of effective balances. When we write |S| for a set of validators S, we mean the total effective balance of all validators in S, not the count of validators.

**Live balances.** All quorum checks (justification, finalization) use **live effective balances**: `state.validators[i].effective_balance` for per-validator weight, and `get_total_active_balance(state)` for the total. Throughout this document, **n** denotes the total active balance (`get_total_active_balance(state)`), and quorum expressions like "|J| >= 2n/3" mean the sum of `validators[i].effective_balance` for i in J reaches 2/3 of n. The same live balances are used for all quorum checks within a single `compute_round_outcome` call.

### 0.2 Height

**Height** is a `uint64` serving as the finality gadget's counter, decoupled from slots, rounds, and epochs. Heights advance via one condition:

1. **Descendant-based justification**: a quorum decision (the suffix-sum of on-chain target votes at slots >= some slot S reaches 2/3 of total active balance).

The genesis height is `GENESIS_HEIGHT = Height(0)`. The current height is tracked per-state via `state.current_height` and incremented exactly by 1 in each call to `advance_height`.

### 0.3 Chain

A **chain** is a sequence of beacon blocks forming a path from the genesis block. Each chain induces an independent `BeaconState`. We write **chain X** or **chain Y** for particular chains, and **state_X** for the state at a given point on chain X. Two chains **conflict** if neither is a prefix of the other.

The **divergence point** of two conflicting chains X and Y is the last block they share: the block B such that B is on both chains and no child of B is on both chains. Any block at a slot after B's slot is **post-divergence** — it exists on at most one of the two chains. Any block at or before B is **pre-divergence** — it exists on both chains.

### 0.4 Canonical Target

When height H starts on chain X (via `advance_height`), the state records a **canonical target**:

> **CT_X(H)** := `state_X.current_height_canonical_target`

This is a `Checkpoint(slot, root)` referencing an actual block on chain X. It is always set to `latest_block_header.slot` in `advance_height` — the most recent block on this chain. The canonical target is fixed for the duration of height H on chain X. The canonical target is relevant only for rewards (TIMELY_TARGET) and leak scoring (Layer 2 exemption) — it is NOT privileged for justification purposes.

### 0.5 Justification (Descendant-Based)

A checkpoint C at slot S is **justified at height H on chain X** via descendant-based suffix-sum: the total weight of validators (using live `effective_balance`) whose voted target slot is >= S reaches a quorum. On a single chain, higher slot = descendant (the chain is linear and all targets are verified on-chain), so the suffix-sum captures all votes for C or any of its descendants.

Formally, `compute_round_outcome` computes `slot_weights: Dict[Slot, Gwei]` from `current_height_target_slots` using `validators[i].effective_balance` for each active validator. For each slot S with nonzero weight, it computes the suffix-sum: the total weight of all voted slots >= S. Justification fires for the highest slot S whose suffix-sum reaches 2/3:

```
suffix_sums = {S: sum(w for S' >= S, w = slot_weights[S']) for S in slot_weights}
justified_slot = max(S for S in suffix_sums if suffix_sums[S] * 3 >= total * 2)
```

where `total` is `get_total_active_balance(state)`.

**Key properties:**

1. **Heterogeneous quorum.** The justification quorum J for a checkpoint at slot S consists of all validators with `target_slots[i] >= S`. Different validators in J may have voted for different targets (all at slots >= S, all on this chain). The quorum is NOT homogeneous — members share the property of having voted for a descendant of the block at slot S, but may have voted for different specific descendants.

2. **Chain-local semantics.** On-chain `target_slots[i]` records slots of blocks verified to be on-chain (via `is_target_on_chain`). Only blocks that actually exist on the chain (verified by root match and proposal-slot check) can be recorded. On a linear chain, slot >= S implies descendant of the block at S.

3. **Fragmentation recovery.** If 40% vote for the block at slot 100 and 30% vote for the block at slot 90, the suffix-sum at slot 90 is 70% >= 2/3, justifying the block at slot 90. This is the key advantage over per-slot counting: votes for descendants contribute to ancestor justification, preventing fragmentation deadlocks.

4. **Most-specific justified target.** The highest qualifying slot is selected. If the suffix-sum at slot 90 reaches 2/3 and the suffix-sum at slot 100 also reaches 2/3, slot 100 is chosen (more specific, higher on the chain).

**Slot monotonicity.** Justification only updates `state.justified_checkpoint` when the new checkpoint's slot is strictly higher than the current one (`justified_slot > state.justified_checkpoint.slot`). This is the only monotonicity mechanism — there is no separate `justification_floor_slot`. This prevents ancestor regression.

### 0.6 Finalization

A checkpoint C is **finalized** when both of the following have occurred:

**(a) Justification.** C was justified at some height H via descendant-based suffix-sum: a set J of validators with combined weight >= 2n/3 (by live effective balance), where each validator i in J signed an attestation with some `target = D_i` at `height = H` with `D_i.slot >= C.slot`. Their votes were recorded in `current_height_target_slots` as `target_slots[i] = D_i.slot`, and the suffix-sum at C.slot reached 2/3. The quorum is heterogeneous — different members may have voted for different targets, all at slots >= C.slot (descendants of C on this chain).

**(b) Finalization.** A set F of validators with |F| >= 2n/3 each signed an attestation with `finalize_height = H` and `finalize_target = D_i` where `D_i` is the voter's actual target at height H, with `D_i.slot >= C.slot` (descendant of C). These votes were recorded in `finalize_participation`. The finalization check fires when `finalize_weight * 3 >= total * 2`.

Finalization uses an **extended window**: the `finalize_participation` bitlist persists across height advances and is reset only when a new checkpoint is justified. F may contain validators who piggybacked `finalize_height = H` at height H+1, H+2, or any later height, as long as no new justification occurred in between.

The validity constraint `data.finalize_height < data.height` ensures that a validator's finalize piggyback always refers to a strictly earlier height. The sentinel `FAR_FUTURE_HEIGHT` (= `2^64 - 1`) is used when a validator does not wish to cast a finalize vote. The on-chain acceptance constraint requires `data.finalize_target.slot >= state.justified_checkpoint.slot` (the finalize target must be a descendant of the justified checkpoint). E2 binds the voter: at the finalize height, the voter only voted for `finalize_target`.

*Remark (slot comparison = descendancy on linear chain).* The on-chain finalization acceptance check `data.finalize_target.slot >= state.justified_checkpoint.slot` is equivalent to "finalize_target is a descendant of (or equal to) the justified checkpoint" on a linear chain. Since all on-chain targets exist on the same linear chain (verified by `is_target_on_chain`), higher slot implies descendancy. This equivalence is used throughout the safety proofs when we move between slot comparisons and descendancy claims.

### 0.7 Slashing Conditions

The function `is_slashable_attestation_data` defines a single condition. Given two **signed** attestation data from the same validator (ordered by height: `data_1.height <= data_2.height`):

**Condition (E2 — Finalize-target conflict):**
```
data_2.finalize_target != Checkpoint()
AND data_1.height == data_2.finalize_height
AND data_1.target != data_2.finalize_target
```

This is the only slashing condition. It ensures: if a validator signs `finalize_target = T` at `finalize_height = H`, they did not vote for any target other than T at height H.

These are checked against **signed attestation data**, not on-chain state. Evidence can be submitted on any chain.

*Remark.* There is no height double-target condition (the old Condition 1 / E1). Validators CAN vote for multiple targets per height without slashing risk. The overwrite rule (`target_slots[i]` records the highest-slot on-chain target) governs on-chain accounting, but the signed data may contain votes for different targets at the same height. Only E2 constrains behavior: a validator who intends to finalize at a given height must be consistent about what they voted for at that height.

### 0.8 Non-Canonical Targets

When `process_attestation` processes an attestation on chain X at the current height H, if the target T satisfies: (a) T != CT_X(H), (b) T != Checkpoint(), and (c) `is_target_on_chain(state_X, T)` returns True, then T is a **non-canonical target on chain X**. The overwrite rule applies: `current_height_target_slots[i]` records the **highest-slot** on-chain target the validator voted for. If `T.slot > current_height_target_slots[i]`, the slot is overwritten; otherwise the vote is recorded only if the validator has not already voted (guarded by `current_slot == FAR_FUTURE_SLOT`).

Non-canonical target votes contribute to the suffix-sum for descendant-based justification. A non-canonical target at a high slot contributes to the suffix-sum of all ancestor slots. There is no distinction between canonical and non-canonical for justification — any on-chain target can be voted for freely, and the overwrite rule ensures `target_slots` reflects the voter's highest-slot target.

*Remark.* Since there is no E1 (height double-target) slashing condition, validators can sign votes for multiple targets at the same height without slashing risk. The only constraint is E2: if a validator signs `finalize_target = T` at `finalize_height = H`, all their votes at height H must be for T.

### Remark 0.8a (Overwrite Rule and Safety)

The overwrite rule (`target_slots` records the highest-slot on-chain target) does not affect safety. The safety proofs (Theorem 1, Lemma 1.7) use signed attestation data — immutable commitments independent of the on-chain `target_slots` recording. The overwrite rule only affects the justification accounting in `compute_round_outcome` (which slots accumulate weight in the suffix-sum). An adversary cannot exploit the overwrite rule to create conflicting finalizations because E2 operates on signed data: a validator's `finalize_target` commitment is checked against their signed attestations, not against the on-chain `target_slots` value.

### 0.9 Honest Validator Assumptions

An **honest validator** follows these rules:

1. **Finalize consistency.** If an honest validator signs `finalize_target = T` at `finalize_height = H`, then every attestation it signed at height H has `target = T`. This ensures it cannot be slashed via E2. An honest validator tracks which target it voted for at each height it intends to finalize.
2. **One attestation per round.** An honest validator signs at most one `AttestationData` per round.

*Remark.* There is no "target lock per height" — honest validators MAY vote for multiple targets at the same height (e.g., voting for a higher-slot target in a later round, which the overwrite rule records). The only constraint is E2 consistency: if they intend to finalize at a height, they must not have voted for a conflicting target at that height. In practice, honest validators vote for one target per height (the canonical target) and only finalize heights where their vote is unambiguous.

### 0.10 Balance Model

This spec uses **live effective balances**: `state.validators[i].effective_balance` for per-validator weight and `get_total_active_balance(state)` for the total. Effective balances are updated at epoch boundaries (during `process_effective_balance_updates`). Between epoch boundaries, they are constant.

**Cross-chain invariant.** Two chains that share a common finalized checkpoint have the same state up to the divergence point. Any epoch boundary before divergence produced identical effective balance updates on both chains. Balance drift between chains is therefore bounded within one epoch (the first epoch after divergence may diverge; epochs before that are identical). This is key for certificate transferability (Theorem P_CT, §1).

---

## 1. Accountable Safety

### Lemma 1.1 (Finalization Prerequisites)

If C is finalized at height H on some chain, then there exist:

- **(a)** A set **J** with |J| >= 2n/3, where every validator i in J **signed** an attestation with `target = D_i` at `height = H`, with `D_i.slot >= C.slot`. Their votes were included on the chain, and the suffix-sum at C.slot reached 2/3. Every validator in J has `target_slots[i] >= C.slot`. The quorum is heterogeneous: different members may have voted for different targets, all descendants of C on this chain.
- **(b)** A set **F** with |F| >= 2n/3, where every validator i in F **signed** an attestation with `finalize_height = H` and `finalize_target = D_i`, where `D_i` is the voter's actual target at height H, with `D_i.slot >= C.slot` (descendant of C). E2 constrains each member of F: at height H, the voter only voted for `D_i`.

*Proof.* The finalization check fires when `finalize_weight * 3 >= total * 2`, where both weights use live `effective_balance`. Each `finalize_participation[i]` bit was set by `process_attestation` upon verifying a valid signed attestation with `data.finalize_height == state.justified_height` (= H), `data.finalize_target = D_i` with `D_i.slot >= state.justified_checkpoint.slot = C.slot`, and a valid signature. This gives F. E2 binds each member: any other vote at height H with `target != D_i` would be slashable.

The justification that produced `justified_checkpoint = C` used descendant-based suffix-sum: the total weight of validators with `target_slots[i] >= C.slot` reached 2/3 of `get_total_active_balance(state)`. Define J as the set of all such validators. Each validator i in J had `current_height_target_slots[i] = D_i.slot >= C.slot`, set by `process_attestation` from `data.target.slot` after signature verification.

*Remark (heterogeneous quorum).* Unlike per-slot justification, the justification quorum J is NOT homogeneous. Validator i voted for `D_i` and validator j voted for `D_j`, where `D_i.slot >= C.slot` and `D_j.slot >= C.slot`, but potentially `D_i != D_j`. The suffix-sum aggregates all such votes. This heterogeneity is critical for the safety proof structure.

### Remark 1.1a (Heterogeneous Finalization Quorum)

The finalization quorum F is heterogeneous in finalize_targets: each member i signs `finalize_target = D_i` (their actual target at `finalize_height`), where `D_i.slot >= justified_checkpoint.slot`. Different members may have different `D_i` values. The on-chain `finalize_participation` bitlist records participation but not which target was finalized. Safety holds because E2 binds each member to their specific `D_i` at the finalize height — the signed data is what matters, not the bitlist. Specifically, Lemma 1.7 uses E2 to extract each overlap member's unique `D_i` from their signed attestation, independent of the `finalize_participation` bitlist.

### Lemma 1.2 (Height Progression)

For any chain Y to reach height H' > H, chain Y must have advanced through every intermediate height. At each height h in {H, ..., H'-1}, descendant-based justification reached a quorum on chain Y.

*Proof.* Height advances only through `advance_height`, called exclusively from `process_justification_and_finalization` on the justification path. Since `advance_height` increments by exactly 1, reaching H' from h requires H' - h successive calls, each triggered by a justification quorum.

### Lemma 1.3 (No Justification Uniqueness)

**Statement.** With descendant-based justification and no E1 (height double-target) slashing condition, justification uniqueness per height does NOT hold. Two different targets C and D can both be justified at the same height H on different chains, even under f < n/3.

*Example.* Chain X has blocks at slots 90 and 100; chain Y has a different block at slot 95 (post-divergence). On chain X, 70% vote at slots >= 90, justifying slot 90. On chain Y, 70% vote at slots >= 95, justifying slot 95. These are different justified targets at the same height H, with no slashing (validators can vote for different targets at the same height — there is no E1).

*Consequence.* Safety does NOT follow from justification uniqueness. Instead, safety follows from **finalization uniqueness**: two conflicting checkpoints cannot both be finalized without >= n/3 slashable weight. This is established in Lemma 1.7 and Theorem 1, which do not depend on Lemma 1.3.

### Lemma 1.4 (Canonical Target >= Justified)

**Statement.** On any single chain, `current_height_canonical_target.slot >= justified_checkpoint.slot`.

*Proof.* The canonical target is set in `advance_height` to `latest_block_header.slot` — the most recent block on this chain. The latest block is always at a slot >= any previously justified checkpoint's slot, because justified checkpoints are actual blocks that appear on this chain (verified by `is_target_on_chain`), and `latest_block_header` advances monotonically with each new block.

### Lemma 1.5 (Justified Checkpoint Slot Monotonicity)

**Statement.** On any single chain, `state.justified_checkpoint.slot` is monotonically non-decreasing across state transitions.

*Proof.* The only two code paths that update `justified_checkpoint` are in `process_justification_and_finalization`:

- **Late justification**: guarded by `previous_height_canonical_target.slot > state.justified_checkpoint.slot`.
- **Current-height justification**: guarded by `justified_slot > state.justified_checkpoint.slot`.

Both require the new slot to be strictly higher. This is the only monotonicity mechanism — there is no separate `justification_floor_slot`.

*Consequence.* Once a checkpoint at slot S is justified, the justified checkpoint never moves to a slot < S. Combined with Lemma 1.4, the canonical target's slot never decreases, preventing ancestor regression. Under f < n/3, descendant-based justification always produces a justified_slot >= the current `justified_checkpoint.slot` because honest voters (>= 2/3) vote at or above the justified checkpoint.

### Lemma 1.6 (Round Outcome Consistency)

**Statement.** All calls to `compute_round_outcome` within a single `process_round` execution return identical values.

*Proof.* `compute_round_outcome` reads `current_height_target_slots`, `finalize_participation`, `validators` (for `effective_balance`), plus scalar fields. Within `process_round`, it is called by `process_inactivity_updates`, `process_rewards_and_penalties`, and `process_justification_and_finalization`. The first two mutate only `inactivity_scores` and `balances` — they do not mutate `effective_balance` (which is updated only at epoch boundaries during `process_effective_balance_updates`, not during `process_round`). The participation fields and scalars are only mutated by `process_justification_and_finalization` (which resets `finalize_participation` on new justification and calls `advance_height` to reset `target_slots`). All three calls see identical inputs.

### Lemma 1.7 (Any Chain Past a Finalized Height Contains the Finalized Checkpoint)

**Statement.** If C is finalized at height H, any chain Y that progresses past height H must contain C on its chain — otherwise >= n/3 validators are slashable.

*Proof.* By Lemma 1.1, C finalized at H on chain A gives:
- Set F (|F| >= 2n/3), where each member i signed `finalize_height = H` and `finalize_target = D_i` with `D_i.slot >= C.slot`. By E2, each member of F only voted `D_i` at height H (any other vote at H would be slashable).
- Each `D_i` descends from C on chain A (since `D_i.slot >= C.slot` and chain A is linear — higher slot = descendant).

By Lemma 1.2, any chain Y reaching height H' > H progressed through height H via justification: some checkpoint T' is justified at H on Y via descendant-based suffix-sum: a quorum Q (|Q| >= 2n/3) of validators each voted for some target at a slot >= T'.slot on chain Y. By quorum intersection, |F ∩ Q| >= n/3 (under f < n/3).

Each overlap member v in F ∩ Q:
- From F: v signed `finalize_target = D_v` at `finalize_height = H`, and by E2, v only voted `D_v` at height H. Every signed attestation from v at height H has `target = D_v`.
- From Q: v's vote was counted in Q on chain Y, meaning `target_slots_Y[v] >= T'.slot`. Since v only voted `D_v` at height H (E2), the vote counted on chain Y IS `D_v`. So `D_v` is on chain Y (it passed `is_target_on_chain` on chain Y) and `D_v.slot >= T'.slot`.

Now: `D_v` is identified by `Checkpoint(slot, root)` — a specific block. `D_v` exists on chain A (where C was finalized) and on chain Y (where `is_target_on_chain` verified it). Since blocks are identified by root, `D_v` is the SAME block on both chains. On chain A, `D_v` descends from C (chain A is linear, `D_v.slot >= C.slot`). Block ancestry is a property of the block tree, not of individual chains: if `D_v` descends from C in the block tree, it descends from C everywhere. Since chain Y contains `D_v` and `D_v` descends from C, chain Y must also contain C (chains contain all ancestors of their blocks).

Under f < n/3, chain Y contains C.

### Lemma 1.8 (Justifications Descend From Finalized)

**Statement.** If C is finalized at height H, all justifications at height >= H on any chain descend from C — otherwise >= n/3 validators are slashable.

*Proof.* By Lemma 1.7, any chain Y past H contains C. At height H itself, the justified checkpoint has slot >= C.slot (from the suffix-sum that justified C or a descendant of C). At heights > H where a new justification update occurs, Lemma 1.5 (slot monotonicity) requires the new justified slot to be strictly > the previous justified slot >= C.slot, so the new justified slot > C.slot. Since C is on chain Y and chain Y is linear, any checkpoint at slot >= C.slot descends from (or equals) C.

### Theorem 1 (Accountable Safety)

**If C is finalized at height H, then any finalization at height >= H on any chain must be for a checkpoint that descends from C — otherwise >= n/3 weight of validators are slashable.**

*Proof.*

1. By Lemma 1.7, any chain Y that progresses past H contains C.
2. On chain Y, since C is on the chain, all blocks at slots > C.slot descend from C (chain Y is linear).
3. By Lemma 1.5 (slot monotonicity), all justifications at height > H on chain Y have slot > C.slot, so they descend from C.
4. Any finalization at height > H requires prior justification (Lemma 1.1(a)), which by step 3 descends from C.

For finalization at height H itself: by Lemma 1.7, chain Y contains C. The justified checkpoint at H on chain Y has slot >= C.slot (from the suffix-sum). By Lemma 1.8, this justification descends from C.

### Corollary 1 (No Conflicting Finalization)

**Under f < n/3, if C is finalized at height H, no checkpoint conflicting with C (i.e., not on the same chain as C) can be finalized at any height.**

*Proof.* By Theorem 1, any finalization at height >= H must be for a checkpoint descending from C — such checkpoints are on the same chain as C, not conflicting. For height < H: by symmetric argument (apply Theorem 1 from the other finalization's perspective), the finalization at H must descend from the earlier one, so C descends from it — again, not conflicting. Ancestors of C CAN be finalized at earlier heights; this is safe because they are on the same chain (compatible, not conflicting).

### Corollary 2 (Stuck Chains)

**Under f < n/3, any chain that does not contain C is stuck at height H — until certificate transfer (Theorem P_CT) or fork-choice abandonment resolves the stall.**

*Proof.* By Lemma 1.7, any chain past H must contain C. A chain not containing C cannot advance past H via justification. The chain remains stuck at H until one of: (a) the justification certificate for C is transferred to the chain (Theorem P_CT), enabling it to justify C as a non-canonical target; or (b) fork choice abandons the chain (Lemma 4.2).

### Theorem P_CT (Certificate Transferability)

**Statement.** If checkpoint C is justified at height H on chain X — meaning a set J with |J| >= 2n/3 (by live effective balance), where each member i signed `target = D_i` with `D_i.slot >= C.slot` at `height = H` — then any chain Y containing C's block can also justify a checkpoint at height H by processing the same attestations, provided sufficient target blocks exist on chain Y.

**Preconditions:**
1. Chain Y contains the target blocks: for each member i of J whose vote will be counted, `is_target_on_chain(state_Y, D_i)` returns True. At minimum, C's block must be on chain Y.
2. Chain Y is at height H: `state_Y.current_height == H`.
3. The attestation epoch is shared between chains X and Y (same RANDAO, same active set for committee computation — guaranteed if the attestations were produced before the divergence point's epoch, or if the chains share the relevant RANDAO mixes).

*Proof.* The certificate consists of the set of signed attestations from J: each member i of J signed `AttestationData` with `target = D_i` and `height = H` at some slot s_i. We show these attestations are processable on chain Y via `process_attestation`:

**Step 1: Height check.** Each attestation has `data.height == H == state_Y.current_height`. The height check passes.

**Step 2: No round window.** The spec explicitly does not enforce a round-based acceptance window for finality attestations. Any attestation at height H is includable on chain Y regardless of the round in which it was originally signed.

**Step 3: Target on chain.** For each member i, `is_target_on_chain(state_Y, D_i)` must return True. For targets that are pre-divergence blocks (shared between chains X and Y), this holds automatically. For post-divergence targets that only exist on chain X, the vote is rejected on chain Y — but the quorum can still form from the subset of J whose targets exist on chain Y.

**Key insight for certificate transfer:** The most important case is when C itself (the justified checkpoint) is a pre-divergence block shared by both chains. Then D_i for some members may be at higher slots (post-divergence on X, not on Y). But even if only the subset voting for C or pre-divergence descendants of C transfer, the suffix-sum at C.slot on chain Y accumulates weight from: (a) the transferred votes whose targets exist on chain Y, plus (b) any votes already cast on chain Y at slots >= C.slot. Under typical conditions (certificate arrives early, honest validators on chain Y vote for targets at slots >= C.slot), the descendant-based suffix-sum at C.slot reaches 2/3.

**Step 4: Committee membership.** By precondition 3, chains X and Y share the relevant RANDAO mixes and active set, so committee membership is identical.

**Step 5: Target slot recording.** For each validator i in J whose target D_i is on chain Y, `process_attestation` applies the overwrite rule: if `D_i.slot > current_height_target_slots[i]`, the slot is overwritten; otherwise, if `target_slots[i] == FAR_FUTURE_SLOT`, the vote is recorded. Transferred votes for high-slot targets can overwrite earlier votes for lower-slot targets, which only helps the suffix-sum.

**Step 6: Quorum computation.** At the next round boundary, `compute_round_outcome` computes the suffix-sum using live `effective_balance` for each validator. For C.slot, this includes all validators with `target_slots[i] >= C.slot`.

**Balance drift bound.** Chains X and Y share identical state up to the divergence point, including all past epoch-boundary effective balance updates. After divergence, effective balances can diverge only at epoch boundaries. Since the attestations in the certificate were signed before or shortly after divergence, the balance drift between chains is bounded within one epoch: a validator's effective balance on chain Y differs from chain X by at most one epoch's worth of updates. For the quorum check, this means the transferred votes may be weighted slightly differently on chain Y than on chain X — but the quorum threshold can still be met because (a) the original quorum was 2n/3 which has slack over n/3, and (b) balance drift within a single epoch is small relative to validator effective balances.

**Step 7: Justification.** A checkpoint at some slot >= C.slot is justified on chain Y at height H via descendant-based suffix-sum. By slot monotonicity (Lemma 1.5), the justified checkpoint advances. Height advances via `advance_height`.

*Remark (overwrite rule advantage).* The overwrite rule (higher slot wins) is well-suited for certificate transfer. A validator who already voted for a target at slot S' on chain Y can have that vote overwritten by a transferred vote at a higher slot S'' — the higher-slot vote only helps the suffix-sum. This is a key difference from the first-vote-wins model.

*Remark (practical transfer).* The most common scenario for certificate transfer is: chain A finalized C at height H, chain B is stuck at height H. Chain B just reached height H, so few validators have voted on chain B yet. The transferred attestations for targets that exist on chain B are processed, the suffix-sum at C.slot reaches 2/3, and B advances.

---

## 2. Accountable Liveness (Inactivity Leak)

**Notation.** Let W denote total active stake (`get_total_active_balance(state)`). For a set S of validators, write w(S) = sum of `validators[i].effective_balance` over i in S. We say a validator i is *penalized* in a round if `compute_leak_penalty_units(state, i, ...) > 0` and `is_in_inactivity_leak(state) == True`. The penalty count (0, 1, or 2) determines the number of INACTIVITY_SCORE_BIAS increments.

**Theorem 2 (Accountable Liveness — Amortized N/3 Penalty Units Per Round).** During any non-finality period, the total penalty units applied across all rounds is at least N/3 times the number of rounds.

*Remark (two-layer separation).* The two layers use different exemption criteria:
- **Layer 1** (stall): `target_slots[i] != FAR_FUTURE_SLOT` — voted on this chain at all. Any on-chain vote contributes to the suffix-sum that could trigger justification.
- **Layer 2** (advance): `target_slots[i] > justified_checkpoint.slot` — voted above the justified checkpoint (strict >). Only votes above `justified_checkpoint.slot` contribute to checkpoint advancement. Additionally, an independent finalize check fires once per checkpoint: at `current_height == justified_height + 1` with pending finalization.

The target check and finalize check are **independent and additive** — a validator failing both gets 2 penalty units.

### Per-round bounds

**Theorem 2.1 (Stall rounds).** If height has not advanced, either justification occurs or > W/3 is penalized.

*Proof.* Let w_V = weight of validators who voted on this chain. If w_V >= 2W/3, the suffix-sum at the lowest voted slot reaches 2/3 and justification fires — not a stall. If w_V < 2W/3, penalty = W - w_V > W/3.

*Remark (slashed validators).* Slashed validators always receive 2 penalty units unconditionally. The suffix-sum excludes slashed weight. Let w_S = slashed weight, w_V' = non-slashed voted weight. If w_V' >= 2W/3, justification fires. If w_V' < 2W/3, total penalty = 2 * w_S + (W - w_S - w_V') > W/3.

**Theorem 2.2 (Advance without pending finalization).** If height advanced and finalization is not pending (`finalized_checkpoint == justified_checkpoint` or finalize weight >= 2W/3), finality has progressed. No leak.

**Theorem 2.3 (Advance with pending finalization — target check).**

Let w_A = weight of validators with `target_slots[i] > justified_checkpoint.slot`.

- If w_A >= 2W/3: the suffix-sum at the lowest slot above `justified.slot` reaches 2/3. So `justified_slot > justified_checkpoint.slot`, the checkpoint updates — real progress.
- If w_A < 2W/3: target penalty = W - w_A > W/3.

The advance-without-checkpoint-update case (`justified_slot == justified_checkpoint.slot`, everyone voted at the justified slot) is always penalized: w_A = 0, penalty = W.

**Theorem 2.4 (Advance with pending finalization — finalize check).** At `current_height == justified_height + 1` with pending finalization, the finalize check fires independently: penalty = W - w_F where w_F = finalize weight. Since finalization is pending, w_F < 2W/3, so finalize penalty > W/3.

*Note:* The finalize check fires only once per checkpoint (at the first height after justification). This is intentional — validators who cannot safely finalize (because they voted for a different target at the justified height) should receive at most one penalty, not perennial punishment.

### Theorem 2.5 (Amortized N/3 bound)

**Claim.** During any non-finality period, the total penalty units is >= N/3 per round on average.

*Proof.* Consider a period of active leak. Every round is either a stall round or an advance round. We show each round contributes >= N/3 penalty units to the total, with the finalize check's contribution attributed to an earlier round to avoid double-counting.

**Case 1 — Stall round:** Contributes >= N/3 from the target check (Theorem 2.1).

**Case 2 — Advance round where the checkpoint does NOT update** (`justified_slot <= justified_checkpoint.slot`): All validators voted at slots <= `justified_checkpoint.slot`. The strict `>` check fails for all of them. Target penalty >= N/3 (Theorem 2.3).

**Case 3 — Advance round where the checkpoint updates** (`justified_slot > justified_checkpoint.slot`): The target penalty can be zero (if >= 2/3 voted above the old justified slot — which is what caused the checkpoint to update). However, a checkpoint update means that finalization is now pending for this new checkpoint. At the next height advance (some future round R'), the finalize check will fire, contributing >= N/3 penalty units independently of whatever target penalties also apply at R' (Theorem 2.4).

We attribute the finalize penalty units at R' to this checkpoint-update round, not to R'. This is valid because: (a) the finalize check is independent of the target check — the penalty units from each are additive, so there is no double-counting; (b) each checkpoint update produces exactly one finalize-check round, so the attribution is unique.

**No round contributes zero.** Stall rounds and advance-without-update rounds each contribute >= N/3 directly (Cases 1-2). Advance-with-update rounds contribute >= N/3 via the attributed finalize penalty (Case 3). The only scenario where the target check contributes zero at an advance round (w_A >= 2W/3) is when the checkpoint actually updates — real progress — and the finalize penalty for that checkpoint compensates. Total penalty units >= N/3 * (number of rounds).

### Lemma 2.6 (Advance-without-checkpoint-update is always penalized)

**Statement.** When height advances but `justified_slot == justified_checkpoint.slot` (the checkpoint does not update), target penalty > W/3.

*Proof.* Let w_A = weight with `target_slots[i] > justified_checkpoint.slot`. The suffix-sum at `justified_checkpoint.slot` reached 2/3 (justification fired) but no higher slot's suffix-sum did (otherwise `justified_slot` would be higher). So w_A < 2W/3. Penalty = W - w_A > W/3.

*Note.* `justified_slot < justified_checkpoint.slot` is impossible under f < n/3. Honest validators (weight >= 2n/3) vote at slots >= `justified_checkpoint.slot` (the canonical target is at or above the justified checkpoint). The suffix-sum at `justified_checkpoint.slot` >= 2n/3, so `justified_slot >= justified_checkpoint.slot`.

**Equality conditions.** `justified_slot == justified_checkpoint.slot` occurs when: (a) no new blocks since the last justification (the canonical target equals the justified checkpoint), or (b) honest validators split between `justified_checkpoint.slot` and higher slots, but no higher slot's suffix-sum alone reaches 2/3. Height advances but the strict `>` guard blocks the checkpoint update.

**Self-correction under synchrony.** An honest proposer creates a block at a slot S' > `justified_checkpoint.slot`. Honest validators vote at S'. The suffix-sum at S' >= 2n/3. The checkpoint updates. Expected latency: O(1) slots.

---

## 3. Fairness of Inactivity Leak

**Theorem 3 (Leak Fairness).** During synchrony, assuming honest majority among awake validators, honest validators are not penalized on the canonical chain. This must hold even after a period of asynchrony — once the network is synchronous, honest validators stop accumulating penalties.

*Proof.* An honest validator is either **unlocked** (did not sign a finalize piggyback at the current height, or signed one at a previous height whose lock has expired) or **locked** (signed `finalize_target = D_i` at `finalize_height = justified_height`).

**Unlocked validators** can freely vote for the canonical target at the current height. During stalls (Layer 1): they vote on-chain, `target_slots[i] != FAR_FUTURE_SLOT`, exempt. During advances (Layer 2): they vote for the canonical target at a slot > `justified_checkpoint.slot` (under synchrony with an honest proposer), target check passes. They can carry a finalize piggyback for the current justified checkpoint if applicable. **0 ISB.**

**Locked validators** signed `finalize_target = D_i` at `finalize_height = justified_height`. By Lemma 5.1, they only finalize if `D_i` was itself justified — meaning `D_i` was the justified checkpoint at the time they signed. Two cases arise depending on whether conflicting justified checkpoints exist at this height.

**Case A: No conflicting justifications at the justified height.** All justified checkpoints at this height are on the same chain (related by ancestry). `store.justified_checkpoint` is the highest-slot one (by the non-conflicting max in `update_checkpoints`). Therefore `D_i` is an ancestor of (or equal to) `store.justified_checkpoint`, and in particular **`D_i` is on the canonical chain**. `has_conflicting_justification` is `False`, so `get_filtered_block_tree` returns `store.blocks` and `get_head` is unmodified.

Since `D_i` is on the canonical chain, the locked validator's vote for `D_i` is processed on-chain: `target_slots[i] = D_i.slot != FAR_FUTURE_SLOT`. During stalls (Layer 1): exempt (voted on this chain). **0 ISB while stuck at this height.**

When the height advances (Layer 2): the locked validator's target is at `D_i.slot <= justified_checkpoint.slot` (ancestor or equal). The strict `>` target check may fail (at most 1 ISB at the advance round). But this is bounded to a single round — at the new height, the lock expires (E2 is height-specific), and the validator is unlocked.

**Case B: Conflicting justifications at the justified height.** Two conflicting targets T and T' are justified at the same height H on different chains. `update_checkpoints` sets `has_conflicting_justification = True`. `get_filtered_block_tree` invokes `filter_block_tree`, which prunes leaf blocks whose `current_height <= store.justified_height`. The canonical chain (returned by `get_head`) is therefore always a chain that has advanced past the justified height.

On such a chain, `current_height > justified_height`. E2 locks apply at `finalize_height = justified_height` — they are height-specific and expire when the height advances. All validators (locked and unlocked) are free at the current height on the canonical chain. They vote for the canonical target at a slot > `justified_checkpoint.slot`. Target check passes. Finalize check passes (they can safely carry the piggyback). **0 ISB.**

*Note:* The filtered set is never empty. `store.justified_height == H` means some processed block has `current_height > H` (justification fires `advance_height`). That block passes the filter.

**Exception: no-new-blocks edge case.** If no honest proposer has produced a block since the last justification, the canonical target equals `justified_checkpoint.slot`. Even unlocked validators fail the strict `>` target check. This is rare under synchrony (requires all proposer slots adversarial, probability exponentially small in SLOTS_PER_ROUND) and self-correcting (an honest proposer creates a new block in O(1) slots).

### Lemma 3.1 (Layer 2 exemption — target and finalize)

**Setting.** `has_height_progress == True`, synchrony with honest proposer.

**Claim.** An unlocked honest validator passes both the target check and finalize check. A locked honest validator passes the target check if `D_i.slot > justified_checkpoint.slot`, and always passes the finalize check if finalization is pending for a checkpoint they can confirm. When `has_conflicting_justification` is set, the canonical chain has `current_height > justified_height`, so all validators are unlocked (E2 locks expired) and the first clause applies.

*Proof (target check).* Under synchrony with an honest proposer, the canonical target is at a slot strictly above `justified_checkpoint.slot` (honest proposer creates a new block at an increasing slot). An unlocked validator votes for it: `target_slots[i] > justified_checkpoint.slot`. Exempt.

*Proof (finalize check).* The finalize check fires at `current_height == justified_height + 1` with pending finalization. An unlocked validator at this height can carry `finalize_target = voted_target_at[justified_height]`. By Lemma 5.1, this is the justified checkpoint itself. The acceptance check passes (`finalize_target.slot == justified_checkpoint.slot >= justified_checkpoint.slot`). `finalize_participation[i]` is set. Exempt.

### Lemma 3.2 (Layer 1 exemption — voted on this chain)

**Setting.** `has_height_progress == False`.

**Claim.** An honest validator (locked or unlocked) is exempt.

*Proof.* Two cases:

**No conflict (`has_conflicting_justification == False`).** An unlocked validator votes for the canonical target on this chain. `target_slots[i] != FAR_FUTURE_SLOT`. Exempt. A locked validator voted for `D_i` (a justified checkpoint, by Lemma 5.1). Without conflicting justifications, `D_i` is on the canonical chain (ancestor of `store.justified_checkpoint`). `target_slots[i] = D_i.slot != FAR_FUTURE_SLOT`. Exempt.

**Conflict (`has_conflicting_justification == True`).** `filter_block_tree` prunes chains at `current_height <= store.justified_height`. The canonical chain has `current_height > justified_height` (some chain must have advanced — see Case B note in Theorem 3). All validators are unlocked at the current height. They vote for the canonical target. `target_slots[i] != FAR_FUTURE_SLOT`. Exempt.

### Lemma 3.3 (Bounded recovery from wrong target)

**Setting.** An honest validator votes for the wrong target due to a brief fork.

**Claim.** At most two `INACTIVITY_SCORE_BIAS` increments, then recovery.

*Proof.* If the validator voted for the wrong target, `target_slots[i]` records a non-`FAR_FUTURE_SLOT` value at the wrong target's slot.

- **Layer 1** (if stalled): `target_slots[i]` records a voted slot (not `FAR_FUTURE_SLOT`). The Layer 1 check is `target_slots[i] != FAR_FUTURE_SLOT`. The validator is exempt. No ISB hit.
- **Layer 2** (when height advances): The target check is `target_slots[i] > justified_checkpoint.slot`. If the wrong target's slot is > `justified.slot`, the validator is exempt (0 ISB from target). If the wrong target's slot is <= `justified.slot` (voted for an old block), the validator fails the target check (1 ISB from target). If at J+1 with pending finalization: the validator cannot safely finalize if it voted for a different target at the justified height (E2 risk) -> one ISB hit from the finalize check. Maximum: 2 ISB hits (target + finalize).
- **Next height**: `target_slots` resets. Validator votes canonical and finalizes consistently. Exempt.

Net damage: at most two ISB increments (from the target check if voted below justified, plus finalize check at J+1). Recovered over ISB rounds. In the common case (wrong target at a high slot > `justified.slot`), damage is at most one ISB increment (from finalize only).

### Theorem 3b (Fairness Under Partial Synchrony)

**Setting.** During synchrony with honest majority among awake validators. The canonical chain may have been through a period of asynchrony, and non-finality may still be active (the leak is running).

**Claim.** Honest validators are not penalized on the canonical chain.

*Proof.* Two cases based on whether conflicting justifications exist at the justified height.

**Case 1: No conflicting justifications (`has_conflicting_justification == False`).** Two sub-cases based on where the canonical chain stands relative to the justified height.

**1a: The canonical chain is past the justified height** (`current_height > justified_height`). Validators are free at the current height — E2 locks are height-specific and apply only at `finalize_height`, which is at or below `justified_height`. Every honest validator can vote for the canonical target and finalize freely. By Lemma 3.1: target check passes (canonical target is above `justified_checkpoint.slot` with an honest proposer), finalize check passes (they can safely carry the piggyback). **0 ISB.**

**1b: The canonical chain is at the justified height** (`current_height == justified_height`). Some honest validators may be locked by E2. By Lemma 5.1, honest validators only finalize if their target was itself justified — meaning `D_i` IS (or was) the justified checkpoint at the time they signed. Without conflicting justifications, all justified checkpoints at this height are on the same chain. `D_i` is an ancestor of (or equal to) `store.justified_checkpoint`. Therefore `D_i` is on the canonical chain.

Since `D_i` is on the canonical chain, `target_slots[i] = D_i.slot != FAR_FUTURE_SLOT`. Layer 1 (stall) exempts them. If the height advances: `D_i` was the justified checkpoint when they voted, so `D_i.slot >= justified_checkpoint_at_that_time.slot`. Whether the current `justified_checkpoint.slot` has advanced past `D_i.slot` depends on whether a new checkpoint was justified at a higher slot in the interim. If it has: the validator is at a new height (Case 1a applies). If it hasn't: `D_i.slot >= justified_checkpoint.slot`, and the target check passes. **0 ISB.**

**Case 2: Conflicting justifications (`has_conflicting_justification == True`).** Two conflicting targets were justified at the same height (Lemma 1.3). `update_checkpoints` set the flag. `get_filtered_block_tree` invokes `filter_block_tree`, which prunes chains at `current_height <= store.justified_height`. The canonical chain (from `get_head`) is one that has advanced past the justified height. On such a chain, `current_height > justified_height`. All E2 locks at the justified height have expired. All honest validators — whether previously locked or not — are free to vote for the canonical target at the current height. By Lemma 3.1: target check passes, finalize check passes. **0 ISB.**

The filtered set is never empty: `store.justified_height == H` means some processed block has `current_height > H`. When `store.justified_height` later advances past the conflicting height, `update_checkpoints` clears the flag and normal unfiltered operation resumes.

*Remark.* The "only finalize if your target was justified" rule (Lemma 5.1) is essential for Case 1b. Without it, a validator could lock on a target D_i that is a descendant of the justified checkpoint but on a different branch than `store.justified_checkpoint` — an off-chain target, causing unbounded ISB hits. Case 2 does not depend on this property — the filter ensures the canonical chain is past the justified height, making all locks irrelevant.

### Remark 3c (Conflicting Justifications — Resolved by Fork-Choice Filter)

The conflicting-justification scenario (two conflicting targets justified at the same height, Lemma 1.3) is handled by the conditional fork-choice filter implemented in `get_filtered_block_tree` and `filter_block_tree`. When `update_checkpoints` detects a conflicting checkpoint at the current `justified_height`, it sets `store.has_conflicting_justification = True`. The filter prunes chains whose leaf state has `current_height <= store.justified_height`, ensuring the canonical chain has always advanced past the conflicting height. E2 locks are height-specific and expire on advance, so all validators are free on the canonical chain. The flag clears when `justified_height` advances. See Theorem 3 (Case B) and Theorem 3b (Case 2) for the full argument.

---

## 4. Store Safety and No Deadlocks

### Theorem 4a (Local Finalization Permanence)

**Statement (no assumption on f).** Once a node sets `store.finalized_checkpoint = F`, `store.finalized_checkpoint` descends from F at all future times.

*Proof.* Two facts suffice:

1. **`on_block` guard**: every accepted block descends from `store.finalized_checkpoint`. The block's state is computed on that chain, so `state.finalized_checkpoint` is a block on that chain — not conflicting with `store.finalized_checkpoint`.
2. **Slot guard**: `update_checkpoints` requires `finalized_checkpoint.slot > store.finalized_checkpoint.slot` for the update to fire.

Together: the candidate finalized checkpoint is on the same chain as the current one (by 1) and at a strictly higher slot (by 2). On a linear chain, higher slot implies descendant. Therefore `store.finalized_checkpoint` only ever moves forward to descendants.

### Theorem 4b (Local Fork-Choice Consistency)

**Statement (no assumption on f).** If a node has `store.finalized_checkpoint = F`, then `get_head(store)` always returns a block descending from F.

*Proof.* `on_block` asserts that every accepted block descends from `store.finalized_checkpoint`. The block's post-state `justified_checkpoint` is on that chain, so it descends from F. In `update_checkpoints`, the non-conflicting max keeps the higher-slot checkpoint (both descend from F), and the conflicting replacement uses a candidate that descends from F (by `on_block`). Therefore `store.justified_checkpoint` always descends from `store.finalized_checkpoint`.

`get_head(store)` calls `get_lmd_ghost_head(store)`, which uses `get_filtered_block_tree(store)` for its block set. Two cases:

- **No conflict (`has_conflicting_justification == False`):** `get_filtered_block_tree` returns `store.blocks`. The walk starts from `store.justified_checkpoint.root` and visits only descendants. Since justified descends from F, the head descends from F.
- **Conflict (`has_conflicting_justification == True`):** `get_filtered_block_tree` invokes `filter_block_tree`, which returns a subset of `store.blocks` — blocks whose leaf descendants have `current_height > store.justified_height`. All blocks in the store descend from F (by `on_block`'s assertion). A subset also descends from F. The walk starts from `store.justified_checkpoint.root` (which descends from F) and visits only descendants within the filtered set. The head descends from F. (The filtered set is never empty: `store.justified_height == H` means some block has `current_height > H`.)

*Remark.* The store may contain old blocks that predate F (accepted before F was finalized). These are unreachable from the `get_head` walk because `store.justified_checkpoint` descends from F and the walk only visits descendants of the starting point.

### Lemma 4.1 (F <= J Under f < n/3)

**Statement.** Under f < n/3, every justified candidate from a processed block already descends from `store.finalized_checkpoint`.

*Proof.* By Theorem 1: under f < n/3, all justifications at height >= H descend from C. Since `on_block` only accepts blocks from chains descending from finalized, every `state.justified_checkpoint` from a processed block descends from `store.finalized_checkpoint`.

### Lemma 4.1a (Non-Conflicting Max Preserves F <= J)

**Statement.** The non-conflicting max operation in `update_checkpoints` preserves `store.justified_checkpoint` descends from `store.finalized_checkpoint`.

*Proof.* The non-conflicting max keeps the higher-slot checkpoint among the current `store.justified_checkpoint` and the candidate `state.justified_checkpoint`. Both descend from `store.finalized_checkpoint`: the candidate because `on_block` asserts the block descends from finalized (so its state's justified checkpoint is on a chain through F); the current by inductive invariant. On a linear chain, the higher-slot checkpoint descends from the lower-slot one, and both descend from finalized. So the kept checkpoint (higher slot) descends from finalized.

For the conflicting case (candidate and current are on different branches), `should_update_justified` determines whether to replace. The candidate descends from `store.finalized_checkpoint` (by `on_block`'s assertion). In both cases, the invariant is preserved.

### Theorem 4c (Pre-Finalization Fork-Choice Lock-In)

**Statement.** Under f < n/3: if checkpoint F is finalized at height H (globally, on any chain), and a node has voted to finalize F (meaning the node has already imported a block where (F, H) is justified and called `update_checkpoints` with it), then `store.justified_checkpoint` descends from F at all future times — even before the node has locally finalized F. Consequently, F is always part of the node's canonical chain.

*Proof.* Four steps:

1. **Voting to finalize (F, H) implies importing (F, H).** The node signed `finalize_target = D` with `finalize_height = H` where `D.slot >= F.slot` (the finalize acceptance check). This requires the node to have processed a block whose state has `justified_checkpoint` at height H. When that block was processed, `update_checkpoints(store, F_or_descendant, H, ...)` was called.

2. **After importing (F, H), store.justified descends from F.** At the time `update_checkpoints` runs with the justified candidate (F, H):
   - If the store's current justified was at a lower height: `should_update_justified` accepts (height H wins). The candidate descends from F (it IS F, or a descendant). Store.J descends from F.
   - If the store's current justified was at height >= H: by Theorem 1 (accountable safety), any justified checkpoint at height >= H is on a chain containing F (under f < n/3). So the current J is not conflicting with F. The non-conflicting max keeps the higher-slot one. Both descend from F (the current by the safety theorem, the candidate by construction). The result descends from F.

3. **Future updates preserve J >= F.** Any future call to `update_checkpoints` brings a justified candidate from a processed block (via `on_block`). Under f < n/3:
   - Candidates at height >= H: by Theorem 1, on a chain containing F. Not conflicting with F. Whether the max keeps the current or the candidate, both descend from F.
   - Candidates at height < H: dominated by (F, H) in the `should_update_justified` comparison (lower height loses). Store.J stays as is (which descends from F by the inductive invariant).

4. **Canonical chain includes F.** `get_head` calls `get_lmd_ghost_head`, which walks from `store.justified_checkpoint.root` over `get_filtered_block_tree(store)`. Whether the block set is filtered or unfiltered, it is a subset of `store.blocks` (all of which descend from F by `on_block`). Since J descends from F, the walk starts from a descendant of F. Every block in the walk descends from F. The head descends from F. F is on the canonical chain.

*Remark.* This property is what makes finalization practically useful: honest nodes converge on F's chain as soon as they see F's justification, not when finalization completes. The finalization confirmation (finalize piggyback reaching 2/3) is a formality — the fork-choice already committed to F's chain.

### Lemma 4.2 (Liveness After Finalization)

**Statement.** Under f < n/3 and synchrony, if C is finalized at height H, the protocol continues to make progress. Specifically, the fork-choice head is always on a chain that has progressed past H.

*Proof.* The chain that finalized C already progressed past H (finalization requires justification at H, which advances the height). So at least one chain has advanced. By the store mechanics:

1. `store.finalized_checkpoint = C` (Theorem 4a: permanent).
2. `store.justified_checkpoint` descends from C (Theorem 4b: `on_block` ancestry assertion).
3. `get_head` walks from `store.justified_checkpoint`, which is at a height > H.
4. Any chain stuck at height H is below the fork-choice starting point and is never selected.

Validators build on the chain returned by `get_head`, which has already progressed past C. The stuck chains are abandoned.

**Primary resolution: certificate transfer.** If chain B is stuck at height H and contains C, the justification certificate for C transfers to chain B (Theorem P_CT). The transferred votes contribute to the suffix-sum at C.slot on chain B. Height advances via descendant-based justification. Chain B is no longer stuck — it advances on its own without requiring a reorg to chain A.

**Why certificate transfer is the primary mechanism.** The attestations from J are signed data with various targets at `height = H`, all at slots >= C.slot. There is no round-based inclusion window for finality attestations (removed to enable certificate transfer). A proposer on chain B includes the attestations. For each attestation whose target exists on chain B, `is_target_on_chain` returns True. The votes contribute to the suffix-sum, and the suffix-sum at C.slot reaches 2/3.

**Fallback: fork-choice convergence.** If certificate transfer is not possible (e.g., the attestations are not available to chain B's proposers), then fork-choice convergence handles the stall. Once chain A's blocks arrive at chain B's node, the store updates: `store.justified_checkpoint` advances, `get_head` shifts to chain A, and validators build on chain A. The stuck chain B is abandoned.

**Historical targets.** If C's slot falls outside the `block_roots` ring buffer (> ~27 hours), `is_target_on_chain` requires a `HistoricalBlockProof` (Merkle proof against `historical_summaries`), supplied by the proposer.

### Lemma 4.3 (Extended Finalization Window Prevents Flip-Flop)

**Statement.** The adversary cannot strand justified checkpoints indefinitely by rapidly justifying new targets.

*Proof.* The flip-flop attack: justify C at H, justify C' at H+1 (reset finalize_participation before C can be finalized), justify C'' at H+2 (reset again), repeat.

The extended window disrupts this: `finalize_participation` is NOT reset on height advance (only on new justification). If the adversary justifies a new target every height, `finalize_participation` resets each time — but this requires sustaining >= 2/3 descendant-based suffix-sum weight at every height, which requires controlling the justification quorum.

Under the inactivity leak, non-finalizing validators are penalized at J+1 (Theorem 2.3). The adversary's effective balance degrades (> 1/3 leaked per round via stacking penalties), eventually breaking the cycle. Without timeout, the adversary has no "cheap" height-advance mechanism — every advance requires a justification quorum.

### Lemma 4.4 (Processing Order Closes Flip-Flop)

**Statement.** Within `process_justification_and_finalization`, finalization is checked BEFORE new justification, ensuring accumulated finalize votes are acted on before they could be reset.

*Proof.* Code inspection of `process_justification_and_finalization`:

1. **Finalization**: checked first
2. **Late justification**: resets `finalize_participation`
3. **Current justification**: resets `finalize_participation`

If the finalize quorum has reached 2/3, finalization fires at step 1 before step 2 or 3 could reset the bitlist.

Additionally, `process_round` runs `process_inactivity_updates` and `process_rewards_and_penalties` BEFORE `process_justification_and_finalization`, so inactivity scoring correctly reads `finalize_participation` before any reset.

### Theorem 4 (No Deadlocks)

**Statement.** The protocol cannot reach a state where no chain can make progress.

*Proof.* Suppose all chains are permanently stuck. We show this leads to contradiction through a hierarchy of resolution mechanisms.

**Resolution 1: Certificate transfer.** If any checkpoint C was finalized on any chain, the justification certificate for C is transferable to all chains containing C (Theorem P_CT). Under f < n/3, by Corollary 2, any progressing chain must contain C. Certificate transfer contributes to the suffix-sum on the receiving chain, enabling descendant-based justification to advance the height. The stuck chains that contain C advance via justification. No E2 conflict arises because the locked validators' finalize targets are descendants of C, consistent with their votes.

**Resolution 2: Inactivity leak drives justification.** On each chain, the inactivity leak penalizes > 1/3 of stake per round (Theorem 2). For chains where certificate transfer is not immediately effective (e.g., the stuck chain does not yet contain the finalized checkpoint's block), the leak degrades non-participating validators' effective balances. Eventually:

- On the canonical chain (which contains C by fork-choice mechanics), honest validators constitute >= 2/3 of remaining active weight. The suffix-sum at the canonical target's slot reaches 2/3 when honest weight dominates, since all honest validators vote for the same target (or its descendants).
- By Lemma 4.1, fork choice converges. By Lemma 4.3, accumulated finalize votes prevent stranding. By Lemma 4.4, processing order ensures finalization fires.

**The leak's primary role.** The leak's primary function is to drive descendant-based justification on the canonical chain (penalizing non-voters, increasing voted weight toward 2/3). Certificate transfer handles cross-chain catch-up — stuck chains advance via justification of transferred votes contributing to the suffix-sum.

In all cases, at least one chain makes progress. Contradiction with the assumption that all chains are permanently stuck.

---

## 5. No Self-Slashability of Honest Voting

**Theorem 5.** An honest validator following the protocol cannot produce two attestations satisfying the slashing condition (E2) in `is_slashable_attestation_data`.

### Lemma 5.1 (No E2 violation — finalize consistency)

**Statement.** An honest validator who signs `finalize_target = T` at `finalize_height = H` has only voted for target T at height H.

**Honest requirement.** Maintain `voted_target_at: Dict[Height, Checkpoint]`. On producing an attestation at height H with non-empty target T:
- Record voted_target_at[H] = T. (In practice, honest validators vote for one target per height — the canonical target.)

When setting `finalize_height` and `finalize_target`:
- Set `finalize_target = voted_target_at[justified_height]` **only if that target was itself justified** — meaning `voted_target_at[justified_height] == state.justified_checkpoint`. If the validator's target at the justified height is a descendant (or ancestor) of the justified checkpoint but not the justified checkpoint itself, abstain from finalize.
- If voted_target_at[justified_height] is not set, or does not match the justified checkpoint: set `finalize_height = FAR_FUTURE_HEIGHT` (abstain from finalize).

*Proof.* E2 requires `data_2.finalize_target != Checkpoint()` AND `data_1.height == data_2.finalize_height` AND `data_1.target != data_2.finalize_target`. The `voted_target_at` guard ensures that `finalize_target` matches the validator's actual vote at the finalize height. Since honest validators vote for one target per height (the canonical target), all their votes at that height have `target = T = finalize_target`. No pair of attestations can satisfy E2.

*Remark (why "only if justified").* The additional constraint — finalize only if your target was itself justified — prevents locks on side branches. Without it, a validator who voted for a descendant D of the justified checkpoint C could sign `finalize_target = D` (accepted on-chain because D.slot >= C.slot). If D is on a different branch than the store's justified checkpoint C' (where C < C', both non-conflicting), D is not on the canonical chain. The validator is locked on an off-chain target and leaked. With the constraint, the validator only finalizes if D == C, ensuring D is on every chain that justified C.

*Remark (finalize quorum under synchrony).* Under synchrony with an honest proposer, all honest validators vote for the same canonical target. The suffix-sum at that slot reaches 2/3. The justified checkpoint IS the canonical target. All honest voted for the justified checkpoint itself. All can finalize. The "only if justified" constraint does not restrict the finalize quorum under synchrony.

### Lemma 5.2 (Multiple targets at non-finalize heights are safe)

**Statement.** A validator who signs `target = C` and `target = D` (both non-empty, C != D) at height H, but never signs `finalize_height = H`, is not at risk of E2 from those votes.

*Proof.* E2 requires `data_2.finalize_height == data_1.height == H`. If the validator never signs `finalize_height = H` (neither as `finalize_target = C` nor `finalize_target = D`), then no data_2 with `finalize_height = H` exists. E2 cannot fire.

*Remark.* This is the fundamental reason E1 is unnecessary. Without a finalize commitment at height H, voting for multiple targets at H is consequence-free from a slashing perspective. The risk only arises when the validator ALSO signs a finalize piggyback referring to that height.

### Lemma 5.3 (Non-canonical votes are non-slashable)

**Statement.** A validator who signs `target = C` (non-empty) at height H, and whose vote is processed as non-canonical on another chain, is not at risk of E2 from that vote alone.

*Proof.* E2 requires a second attestation with `finalize_target != C` and `finalize_height = H`. An honest validator who voted for C at H and later signs `finalize_target = C` at `finalize_height = H` is safe (C = C, no E2). The only risk is if the validator also voted for a different target D at H AND signs `finalize_target = D` at `finalize_height = H` — then the C vote triggers E2. But by Lemma 5.1, honest validators only finalize heights where they voted for exactly one target, avoiding this scenario.

*Remark.* A validator's vote for target C at slot s contributes to the suffix-sum via `slot_weights[s]`. The on-chain `target_slots[i]` records the highest-slot target (overwrite rule). Slashing conditions operate on signed data — the on-chain aggregation does not affect what was signed.

### Lemma 5.4 (Round double-vote — lighter mechanism)

**Statement.** Round double-vote evidence triggers forced exit, not full slashing. An honest validator never triggers it.

*Proof.* `RoundDoubleVoteEvidence` requires two attestations from the same validator in the same round with different `AttestationData`. An honest validator signs one `AttestationData` per round (§0.9, Rule 2). The penalty is `initiate_validator_exit` plus a fixed deduction — NOT `slash_validator`.

*Putting it together (Theorem 5).* An honest validator following the two behavioral requirements — finalize consistency (5.1) and one attestation per round (5.4) — cannot produce any pair satisfying `is_slashable_attestation_data`. Votes processed as non-canonical on other chains are structurally safe (5.3). Multiple-target votes at non-finalize heights are safe (5.2). The honest validator's state is O(1) per height: one voted_target_at entry.

---

## 6. Resolved Issues

### 6.1 Conflicting-Justification Fork-Choice Filter (Implemented)

**Problem.** In the IC model (no E1), two conflicting targets T and T' can be justified at the same height H on different chains with zero equivocators (Lemma 1.3). A validator locked by E2 (signed `finalize_target = T` at `finalize_height = H`) cannot vote for anything other than T at height H. If the canonical chain descends from T' but has not itself justified at height H (still at `current_height = H`, collecting votes), the locked validator has `target_slots[i] = FAR_FUTURE_SLOT` on that chain (T is not on the canonical chain). Without the filter, they would be penalized by Layer 1 every round for the duration of the stall.

**Precondition.** Two conflicting justified checkpoints at the same height implies a breakdown of normal protocol guarantees — synchrony or honest-majority assumptions did not hold. Under synchrony with f < n/3, honest validators agree on the canonical target, and only one target is justified per height. Conflicting justifications arise only during asynchrony or adversarial manipulation.

**Implementation.** The conditional fork-choice filter is implemented via `store.has_conflicting_justification`, `filter_block_tree`, and `get_filtered_block_tree`:

1. **Detection** (in `update_checkpoints`): when `are_non_conflicting` returns `False` AND `justified_height == store.justified_height` (a conflicting checkpoint at the current justified height), `has_conflicting_justification` is set to `True`. The flag is set regardless of whether the candidate wins or loses `should_update_justified` — the conflict exists either way. This ensures convergence: even if node A saw T first and node B saw T' first, both set the flag when they see the other checkpoint.

2. **Filter** (in `get_filtered_block_tree` / `filter_block_tree`): when the flag is set, `filter_block_tree` recursively walks the block tree from `store.justified_checkpoint.root`, keeping only branches whose leaf state has `current_height > store.justified_height`. Ancestors of viable leaves are kept (the recursive structure ensures this). The filtered set is never empty: `store.justified_height == H` means some processed block has `current_height > H`. When the flag is not set, `get_filtered_block_tree` returns `store.blocks` directly (zero overhead).

3. **Clear** (in `update_checkpoints`): when `store.justified_height` advances past the conflicting height, the flag is cleared. Normal unfiltered operation resumes.

4. **Integration**: `get_lmd_ghost_head` uses `get_filtered_block_tree(store)` for its block set instead of `store.blocks` directly.

**Effect on fairness.** When the filter is active, the canonical chain (from `get_head`) has `current_height > justified_height`. Locked validators' E2 locks are height-specific and expire on advance. All validators are free on the canonical chain. Zero ISB. See Theorem 3 (Case B) and Theorem 3b (Case 2).

**Effect on liveness.** The filter restricts the fork-choice to chains that have demonstrated progress. The filtered set is never empty (at least one chain has advanced). The protocol converges on a progressing chain.

**Normal conditions (no conflict).** The flag is never set. `get_filtered_block_tree` returns `store.blocks`. Zero overhead. The filter is purely reactive to detected breakdowns.

**Comparison with Gasper's `filter_block_tree`.** Gasper applies `filter_block_tree` unconditionally at every fork-choice call, checking justified/finalized alignment for every block. The simplex filter is conditional: it only activates when conflicting justified checkpoints are detected at the same height, and checks a single condition (`current_height > justified_height`). This is strictly simpler and narrower.

**Adversarial exploitation.** The adversary cannot trigger the filter under f < n/3 and synchrony. Creating a conflicting justification requires the adversary to supplement honest votes on a minority chain — the adversary has < n/3, and the minority chain has < n/3 honest. Total < 2n/3. Conflicting justification fails.

# Simplex Finality -- Fork Choice

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Configuration](#configuration)
- [Containers](#containers)
  - [New `FrozenAvailableVotes`](#new-frozenavailablevotes)
  - [New `FrozenTSQView`](#new-frozentsqview)
  - [New `TSQSelection`](#new-tsqselection)
  - [Modified `Store`](#modified-store)
  - [Modified `LatestMessage`](#modified-latestmessage)
- [Helper functions](#helper-functions)
  - [New `upgrade_forkchoice_store_to_simplex`](#new-upgrade_forkchoice_store_to_simplex)
  - [Modified `get_forkchoice_store`](#modified-get_forkchoice_store)
  - [Modified `get_ancestor`](#modified-get_ancestor)
  - [New `update_justified`](#new-update_justified)
  - [New `is_finalized_compatible`](#new-is_finalized_compatible)
  - [New `recompute_h_max`](#new-recompute_h_max)
  - [New `get_viability_height_threshold`](#new-get_viability_height_threshold)
  - [New `is_viable_leaf`](#new-is_viable_leaf)
  - [New `is_viable`](#new-is_viable)
  - [Modified `filter_block_tree`](#modified-filter_block_tree)
  - [Modified `get_filtered_block_tree`](#modified-get_filtered_block_tree)
  - [New `is_in_filtered_block_tree`](#new-is_in_filtered_block_tree)
  - [New `update_finalized`](#new-update_finalized)
  - [New `has_unexpired_latest_message`](#new-has_unexpired_latest_message)
  - [New `get_total_active_voting_weight`](#new-get_total_active_voting_weight)
  - [New `get_view_freeze_due_ms`](#new-get_view_freeze_due_ms)
  - [New `is_before_view_freeze_deadline`](#new-is_before_view_freeze_deadline)
  - [New `get_available_confirmation_due_ms`](#new-get_available_confirmation_due_ms)
  - [New `is_at_or_before_available_confirmation_deadline`](#new-is_at_or_before_available_confirmation_deadline)
  - [New `is_at_or_after_available_confirmation_deadline`](#new-is_at_or_after_available_confirmation_deadline)
  - [New `is_before_attestation_deadline`](#new-is_before_attestation_deadline)
  - [New `is_ptc_decision_node`](#new-is_ptc_decision_node)
  - [Modified `get_supported_node`](#modified-get_supported_node)
  - [New `is_supporting_vote`](#new-is_supporting_vote)
  - [New `get_available_majority_threshold`](#new-get_available_majority_threshold)
  - [New `get_available_vote_payload_status`](#new-get_available_vote_payload_status)
  - [New `get_available_attestation_score`](#new-get_available_attestation_score)
  - [New `is_available_attestation_viable`](#new-is_available_attestation_viable)
  - [New `cache_available_committee`](#new-cache_available_committee)
  - [New `freeze_available_votes`](#new-freeze_available_votes)
  - [New `get_available_confirmation_score`](#new-get_available_confirmation_score)
  - [New `get_available_confirmation_majority_threshold`](#new-get_available_confirmation_majority_threshold)
  - [New `is_available_confirmation_viable`](#new-is_available_confirmation_viable)
  - [New `get_best_available_confirmation_child`](#new-get_best_available_confirmation_child)
  - [New `get_fast_confirmation_score`](#new-get_fast_confirmation_score)
  - [New `is_fast_confirmation_viable`](#new-is_fast_confirmation_viable)
  - [New `get_fast_confirmation_head`](#new-get_fast_confirmation_head)
  - [New `get_attestation_checkpoint_state`](#new-get_attestation_checkpoint_state)
  - [New `get_tsq_quorum_threshold`](#new-get_tsq_quorum_threshold)
  - [New `get_tsq_effective_head`](#new-get_tsq_effective_head)
  - [New `get_tsq_effective_heads`](#new-get_tsq_effective_heads)
  - [New `get_tsq_intersection_heads`](#new-get_tsq_intersection_heads)
  - [New `get_tsq_support`](#new-get_tsq_support)
  - [New `get_deepest_tsq_root`](#new-get_deepest_tsq_root)
  - [New `freeze_tsq_view`](#new-freeze_tsq_view)
  - [New `freeze_tsq_selection`](#new-freeze_tsq_selection)
  - [New `update_round_proposals`](#new-update_round_proposals)
  - [New `mark_round_proposal_conflict`](#new-mark_round_proposal_conflict)
  - [New `freeze_stable_root`](#new-freeze_stable_root)
  - [New `get_simplex_root`](#new-get_simplex_root)
  - [New `get_grade_1_root`](#new-get_grade_1_root)
  - [New `get_stable_root`](#new-get_stable_root)
  - [New `get_safe_confirmed_head`](#new-get_safe_confirmed_head)
  - [Modified `get_safe_execution_block_hash`](#modified-get_safe_execution_block_hash)
  - [New `update_confirmed_head`](#new-update_confirmed_head)
  - [New `update_confirmation_heads`](#new-update_confirmation_heads)
  - [New `get_available_confirmation_head`](#new-get_available_confirmation_head)
  - [New `get_payload_participant_count`](#new-get_payload_participant_count)
  - [New `get_payload_full_support`](#new-get_payload_full_support)
  - [New `get_payload_data_available_support`](#new-get_payload_data_available_support)
  - [New `is_payload_verified`](#new-is_payload_verified)
  - [Modified `is_payload_timely`](#modified-is_payload_timely)
  - [Modified `is_payload_data_available`](#modified-is_payload_data_available)
  - [Modified `should_extend_payload`](#modified-should_extend_payload)
  - [Modified `should_build_on_full`](#modified-should_build_on_full)
  - [Modified `update_latest_messages`](#modified-update_latest_messages)
  - [New `apply_attestation_latest_messages`](#new-apply_attestation_latest_messages)
  - [Modified `get_attestation_score`](#modified-get_attestation_score)
  - [New `is_g0_clear`](#new-is_g0_clear)
  - [Modified `get_weight`](#modified-get_weight)
  - [Modified `get_head`](#modified-get_head)
  - [Modified `get_proposer_head`](#modified-get_proposer_head)
  - [New `is_attestation_from_store_active_simplex_fork`](#new-is_attestation_from_store_active_simplex_fork)
  - [New `is_valid_from_block_attestation`](#new-is_valid_from_block_attestation)
  - [New `is_valid_from_block_available_attestation_precheck`](#new-is_valid_from_block_available_attestation_precheck)
  - [New `is_valid_from_block_available_attestation`](#new-is_valid_from_block_available_attestation)
  - [Modified `validate_on_attestation`](#modified-validate_on_attestation)
  - [New `validate_on_available_attestation`](#new-validate_on_available_attestation)
- [Handlers](#handlers)
  - [Modified `on_tick_per_slot`](#modified-on_tick_per_slot)
  - [New `on_tick_per_high_resolution`](#new-on_tick_per_high_resolution)
  - [Modified `on_block`](#modified-on_block)
  - [Modified `on_execution_payload_envelope`](#modified-on_execution_payload_envelope)
  - [Modified `on_attester_slashing`](#modified-on_attester_slashing)
  - [Modified `on_payload_attestation_message`](#modified-on_payload_attestation_message)
  - [Modified `on_attestation`](#modified-on_attestation)
  - [New `on_available_attestation`](#new-on_available_attestation)
  - [New `on_round_double_vote_evidence`](#new-on_round_double_vote_evidence)
- [Removed inherited mechanisms](#removed-inherited-mechanisms)

<!-- mdformat-toc end -->

## Introduction

This is the fork choice specification for simplex-based finality. It modifies
the fork choice to use the justified and finalized checkpoints from the
height-filter-and-timeouts simplex finality gadget instead of Casper FFG, and
removes the unrealized justification/finalization machinery.

The fork choice operates in three layers. Layer 1 is the finality gadget: it
maintains `store.justified_checkpoint` (paper's `Σ.J`) as the lex-max over
justification cert events (paper's `updateJustified`), and advances
`store.finalized_checkpoint` via `update_finalized` when the incoming checkpoint
strictly extends the current finalized, is an ancestor of or equal to
`store.justified_checkpoint`, and is in the **viable subtree**. The store also
tracks `h_max` (the highest `state.current_height` in the current finalized
subtree) which drives the **height filter**: only blocks whose state-height is
at least `h_max - 1` (or whose descendants reach that bound) are viable. Layer 2
is the stabilization layer: every valid finality attestation received from the
network immediately updates its validator's expiring **latest head vote**; block
inclusion is not required. Each usable round applies time-shifted-quorum (TSQ)
synchronization to the preceding round's signed heads. A validator freezes its
preceding-round view at the public view-freeze boundary. Before the next
round-start proposal, it pins a proposal-independent candidate tree and
electorate. At the common available-vote action, it selects the deepest root
with absolute quorum support from the same signed heads in both its frozen and
current views. A signer with two distinct round messages contributes no TSQ
support. The selected root is used only when it remains viable and descends from
the action state's Simplex root; otherwise the action freezes its current
grade-1 fallback. The proposal carries no synchronization root and never
installs one. Layer 3 is the Goldfish available-chain layer: per-slot
available-committee attestations and the availability confirmations derived from
them.

The fork-choice head is computed by the **walk** (`get_head`), in three phases:
a **stable root** frozen at the common action — the fixed-quorum TSQ lock when
it passes the action-state checks, else the then-current two-thirds
latest-head-vote descent from the Simplex root — then the **Goldfish** descent
from the stable root within the viable subtree, then the **viability descent**
down to the height frontier. A unique round-start proposal receives
proposal-specific treatment in the round-start slot only when it descends both
the stable root and the action-time live available-confirmed head. It may align
split unconfirmed heads but cannot replace that action-time live prefix. This is
not a floor at the historical `latest_confirmed_head`: using that non-retracting
record would let a pre-stabilization conflicting confirmation permanently block
healing. Persistence of an earlier user-facing confirmation therefore still
requires the paper's continuing Goldfish/common-prefix premises. The paper
models every consumer through that walk. This executable profile deliberately
makes confirmation a separate, floorless rule over the unfiltered accepted tree
from live finality: user confirmation must not disappear merely because finality
height or SG grades move. Confirmation is split in two: the user-facing,
non-retracting **available confirmation** (`store.latest_confirmed_head`), never
gated by finality-gadget or grade state, and the recoverable live confirmation
(`store.live_confirmed_head`) from which internal **safe confirmation**
(`get_safe_confirmed_head`) selects the deepest G0-clear block, with the Simplex
root as an unconditional floor. Finality-vote gates read the latter, so a
pre-stabilization conflicting confirmation cannot permanently strand a validator
after the live available chain converges.

The grade thresholds are uniform at every height: the two-thirds grade-1
fallback and the one-third conflict veto (`is_g0_clear`) are the only SG grade
objects. A processed height-fresh target vote also sets the validator's timeout
marker, and a later empty vote does not clear it; votes for a lock target
therefore remain part of a height-advance quorum on a chain containing that
lock.

**Proof scope.** The paper and this executable profile both freeze the vote
kind, height, target, and finality piggyback at one common first-slot deadline,
then emit duties across the round with a live head field. This profile gives
that abstract event concrete boundary ordering and restart-persistent local
state. Height/leak events are marked at round boundaries and consumed after the
next block's operations (at epoch cadence during a leak). The evolving active
set follows the inherited Ethereum weak-subjectivity model. The paper leaves the
Ethereum committee-randomness mapping open, while this profile pins a
balance-weighted available committee from each node's boundary head. The
intended `h + 2` safe-confirmation argument and post-GST healing therefore
remain research obligations under explicit committee- and round-start
convergence assumptions; they are not established merely by the executable
functions. Proposers still include ordinary finality votes to form justification
and finalization certificates. Cross-node agreement of the
available-confirmation deadline snapshot under adversarial straggler delivery is
likewise a timing/reconciliation research obligation; the local freeze below
does not prove that two nodes saw the same pre-deadline messages. In particular,
the paper's TSQ synchronization lemmas apply here only under their explicit
freeze, relay, common-state, and counted-proposal construction premises. Under
those premises, an honest round-start proposal produces one common canonical
head and validators freeze one common interval-first target before their
distributed duties. A Byzantine equivocating round-start proposer need not do
so; proposer fairness alone also does not prove the counted synchronization
opportunity. This executable Store pins the TSQ selection tree and weights but
does not retain the paper's entire later action state: a later authenticated
Simplex-root change can make the frozen root inert, and later walks use the live
viable tree. Applying the paper's same-round bound therefore additionally
requires that this action state remain common through the dependent actions. The
executable schedule also still combines the SG head with the FG message instead
of providing the paper's separate post-confirmation SG pre-vote.

The fixed quorum is a positive selection threshold, unlike `G0`'s one-third
conflict veto. A subquorum adversary cannot use TSQ to select a conflicting
root, including in the fault range between one third and one half. TSQ
synchronization liveness nevertheless requires responsive honest weight at least
the fixed quorum.

*Note*: This specification is built upon Gloas (EIP-7732 ePBS fork choice).

## Configuration

| Name                                      | Value                  | Description                                                                                                                                                                                                                                                                                                |
| ----------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LATEST_MESSAGE_EXPIRY_SLOTS`             | `uint64(2**7)` (= 128) | Staleness bound for the latest head votes used by the SG grades (`get_attestation_score` and `get_total_active_voting_weight`): a validator's `latest_message` is ignored once its slot is at least this many slots in the past. Both grade numerator and denominator use the same live-message predicate. |
| `AVAILABLE_ATTESTATION_DUE_BPS`           | `uint64(2500)`         | basis points; available-committee validators publish by 25% of `SLOT_DURATION_MS`, after the assumed first 25%-of-slot delivery phase for the beacon block.                                                                                                                                                |
| `AVAILABLE_CONFIRMATION_DUE_BPS`          | `uint64(5000)`         | basis points; 50% of `SLOT_DURATION_MS`. Dual role: inclusive in-slot cutoff for an available vote to count as *timely*, and the time at which the previous slot's available-confirmation rule is run. Messages delivered exactly at the cutoff are processed before the freeze/tick evaluation.           |
| `FAST_CONFIRMATION_COMMITTEE_NUMERATOR`   | `uint64(3)`            | Numerator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                              |
| `FAST_CONFIRMATION_COMMITTEE_DENOMINATOR` | `uint64(4)`            | Denominator for the fast-confirmation absolute threshold: at least 75% of `AVAILABLE_COMMITTEE_SIZE` seats in the current slot.                                                                                                                                                                            |
| `VIEW_FREEZE_DUE_BPS`                     | `uint64(7500)`         | basis points; 75% of `SLOT_DURATION_MS`. In the final slot of a round, the previous-round SG-TSQ view is frozen here for the next round. Later valid messages remain in the current view but cannot enter the completed freeze.                                                                            |

## Containers

### New `FrozenAvailableVotes`

*Note*: The node-local frozen electorate for Goldfish available confirmation
(paper Definition: Goldfish available chain and confirmation): the slot's
available committee, pinned from the node's boundary walk head, and the first
votes that this node classified as timely and non-equivocating at
`AVAILABLE_CONFIRMATION_DUE_BPS`. Captured once per slot by
`freeze_available_votes`, the snapshot's numerator and denominator do not
retract locally when later messages arrive. This is not a cross-node agreement
claim: an adversarial straggler may arrive on opposite sides of the deadline at
different nodes. The confirmation-head walk is not frozen; it consumes these
scores over the live block tree from the live finalized root, and therefore may
produce a different candidate after the tree or finalized root changes.
`update_confirmed_head` separately makes the stored user-facing record monotone
by accepting only descendant candidates.

```python
@dataclass
class FrozenAvailableVotes:
    # [New in Simplex]
    committee: Sequence[ValidatorIndex]
    votes: Dict[ValidatorIndex, AvailableAttestationData]
```

### New `FrozenTSQView`

*Note*: A local immutable copy of one support round's first validated
attestation data and known round equivocators at the public TSQ freeze. The
high-resolution scheduler records it exactly once. A missed freeze is not
reconstructed from later messages.

```python
@dataclass
class FrozenTSQView:
    # [New in Simplex]
    support_round: Round
    attestations: Dict[ValidatorIndex, AttestationData]
    equivocating_indices: Set[ValidatorIndex]
```

### New `TSQSelection`

*Note*: The proposal-independent TSQ state pinned at the next round boundary,
before any proposal for that round is admitted. `candidate_roots` is the
then-viable block tree rooted at `simplex_root`. `weights` contains the fixed
active, unslashed contribution of each validator; `total_active_balance` retains
active slashed balance in the absolute quorum denominator. `candidate_root` is
the deepest quorum-supported root in the local proposer-time view, or
`simplex_root` when no strict descendant qualifies.

```python
@dataclass
class TSQSelection:
    # [New in Simplex]
    support_round: Round
    simplex_root: Root
    candidate_roots: Set[Root]
    weights: Dict[ValidatorIndex, Gwei]
    total_active_balance: Gwei
    candidate_root: Root
```

### Modified `Store`

*Note*: `justified_checkpoint` (paper's `Σ.J`) and `justified_height` (paper's
`Σ.h_j`) jointly track the lex-max justification cert event ever observed, under
the lex key `(h_j, hash(J))`. `h_max` (paper's `Σ.h_max`) tracks the maximum
`state.current_height` in the current finalized subtree; it drives the height
filter / viable subtree (paper Definition: viable subtree).
`finalized_checkpoint` is the paper's `Σ.F`. Weight-accounting consumers such as
`get_total_active_voting_weight` and `get_attestation_score` read
`store.block_states[store.justified_checkpoint.root]` as a weight-accounting
base state. Blocks and latest messages that conflict with the current finalized
root remain stored as evidence, but do not affect SG grades, G0 clearance, or
the height-filter frontier.

```python
@dataclass
class Store:
    time: uint64
    genesis_time: uint64
    # [Modified in Simplex]
    justified_checkpoint: Checkpoint  # paper's Σ.J
    # [New in Simplex]
    justified_height: Height  # paper's Σ.h_j
    finalized_checkpoint: Checkpoint
    # [New in Simplex]
    h_max: Height  # paper's Σ.h_max
    # Monotone set of validators known to have equivocated. It is seeded from
    # the on-chain round-double-vote penalty bitmap at checkpoint sync; pruning
    # the per-round evidence cache must not rehabilitate a proven equivocator.
    equivocating_indices: Set[ValidatorIndex]
    # Multiple distinct validated round-start proposals suppress
    # proposal-specific treatment for their round without changing the
    # independently chosen TSQ lock. They need not share a proposer identity.
    round_proposal_conflicts: Set[Round]
    # Slots whose exact 75% high-resolution view-freeze event completed.
    # Wire ingress consults this marker instead of whole-second ``Store.time``.
    view_freeze_slots: Set[Slot]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    # Committee-resolution states keyed by (voted head, duty epoch). Using the
    # known head state and processing it forward avoids an ancestor walk below
    # an exact-slot checkpoint-sync anchor.
    checkpoint_states: Dict[Tuple[Root, Epoch], BeaconState] = field(default_factory=dict)
    # Latest valid finality-attestation head vote received from the network,
    # per validator. These expiring messages directly supply both numerator
    # and denominator of the SG grade thresholds; inclusion is not required.
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    # [New in Simplex]
    # First valid finality-attestation data received from each validator in
    # each round, plus validators seen signing distinct data in that round.
    # This bounded local view supplies TSQ inputs independently of grades.
    # An equivocation excludes the signer instead of contributing support.
    round_attestations: Dict[Round, Dict[ValidatorIndex, AttestationData]] = field(
        default_factory=dict
    )
    round_equivocating_indices: Dict[Round, Set[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # Immutable support-round freezes and proposal-independent next-round
    # selection snapshots.
    frozen_tsq_views: Dict[Round, FrozenTSQView] = field(default_factory=dict)
    tsq_selections: Dict[Round, TSQSelection] = field(default_factory=dict)
    # Block-carried votes may name a valid sibling block that has not arrived
    # yet. Retain them until that head arrives (or the vote expires) so
    # fork-choice effects do not depend permanently on block arrival order.
    pending_attestations: Dict[Root, list[Attestation]] = field(default_factory=dict)
    pending_available_attestations: Dict[Root, list[AvailableAttestation]] = field(
        default_factory=dict
    )
    # [New in Simplex]
    # Last confirmed head (root, slot-confirmed-at) from the available-confirmation rule,
    # first evaluated by the exact high-resolution confirmation event.
    latest_confirmed_head: Tuple[Root, Slot] = (Root(), Slot(0))
    # [New in Simplex]
    # Current delayed available-confirmation result. Unlike the user-facing
    # monotone record above, this may switch branches as successive frozen
    # electorates converge; safe-confirmation/healing gates read this value.
    live_confirmed_head: Tuple[Root, Slot] = (Root(), Slot(0))
    # [New in Simplex]
    # Immediate confirmed head (root, slot-confirmed-at) from the 75%-absolute
    # fast-confirmation rule, first evaluated by the exact high-resolution
    # confirmation event.
    fast_confirmed_head: Tuple[Root, Slot] = (Root(), Slot(0))
    # [New in Simplex]
    # The stable root selected at the current round's common action: either the
    # fixed-quorum TSQ lock or the then-current G1 fallback. The complete node
    # identity is fixed for the round.
    stable_root: Root = Root()
    stable_root_payload_status: PayloadStatus = PAYLOAD_STATUS_PENDING
    stable_root_proposal_root: Root = Root()
    stable_root_round: Round = Round(0)
    # All distinct round-start proposal roots observed before or after the
    # decision. Exactly one action-timely proposal may start the first-slot
    # available-vote walk after the stable-root and confirmed-prefix checks;
    # conflicting proposals never change the independently selected stable
    # root.
    round_proposals: Dict[Round, Set[Root]] = field(default_factory=dict)
    stable_root_decisions: Dict[Round, boolean] = field(default_factory=dict)
    # Verified execution payload envelopes (gloas model); membership is the
    # local-availability signal consulted by the payload gates.
    payloads: Dict[Root, ExecutionPayloadEnvelope] = field(default_factory=dict)
    # Exact Gloas immediate post-block states retained only for pre-activation
    # roots whose payload envelope was unresolved at live migration. Advancing
    # those states to the fork slot would destroy the historical verification
    # context needed by a valid late envelope.
    # ``Any`` is used because these values are previous-fork BeaconState
    # instances, not the Simplex BeaconState assembled by this module.
    legacy_payload_verification_states: Dict[Root, Any] = field(default_factory=dict)
    # [Modified in Simplex]
    # Per-slot PTC first-seen votes keyed by validator identity. Duplicate PTC
    # seats are counted at read time against the slot's PTC committee.
    payload_votes: Dict[Slot, Dict[ValidatorIndex, PayloadAttestationData]] = field(
        default_factory=dict
    )
    payload_vote_equivocations: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # Per-slot available-attestation tracking for Goldfish, keyed by validator
    # identity. Duplicate committee seats are counted at read time against the
    # slot's available committee.
    available_votes: Dict[Slot, Dict[ValidatorIndex, AvailableAttestationData]] = field(
        default_factory=dict
    )
    available_vote_equivocations: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # First wire votes received before the available-confirmation deadline.
    available_timely_attesters: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # Equivocations observed before the available-confirmation deadline for a
    # slot's timely wire voters. Kept separately from the live Goldfish
    # equivocation set so a late conflicting copy cannot alter the deadline
    # snapshot captured at the exact boundary.
    available_timely_equivocations: Dict[Slot, Set[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # The available committee pinned from this node's head view at each slot
    # boundary. Live Goldfish scoring and the later freeze use the same entry;
    # they never try to query a justified state beyond its bounded committee
    # lookahead during a finality stall.
    available_committees: Dict[Slot, Sequence[ValidatorIndex]] = field(default_factory=dict)
    # [New in Simplex]
    # Per-slot available-confirmation freezes (committee and timely,
    # non-equivocating votes as of the slot's confirmation deadline), captured
    # by ``freeze_available_votes`` and read by the confirmation rules.
    frozen_available_votes: Dict[Slot, FrozenAvailableVotes] = field(default_factory=dict)
```

### Modified `LatestMessage`

*Note*: `payload_present` is removed. A stored latest message is a finality LMD
vote for a beacon block, which makes no payload decision; the payload status is
supplied explicitly at the support check (see `get_supported_node`). The
available / Goldfish layer builds its own transient `LatestMessage` and passes
the status it decides.

```python
@dataclass(eq=True, frozen=True)
class LatestMessage:
    # [Modified in Simplex]
    # Removed `payload_present`
    slot: Slot
    root: Root
```

## Helper functions

### New `upgrade_forkchoice_store_to_simplex`

*Note*: A running node MUST migrate its Gloas fork-choice store at the exact
start of `SIMPLEX_FORK_EPOCH`, before accepting any Simplex block or message.
This is distinct from checkpoint sync through `get_forkchoice_store`: a live
migration has branch states, latest messages, locally verified payloads, and
known equivocators that must survive the fork.

Every known Gloas branch state is first advanced, without a block, to the fork
slot under the Gloas transition and is then passed through `upgrade_to_simplex`.
Consequently, the first Simplex block at the fork slot starts from an already
slot-processed state; `on_block` has an explicit path for that one boundary
case. Historical Gloas blocks remain opaque entries in the block tree because
all fork-choice consumers use only their common `slot`/`parent_root`/payload-bid
fields. Their associated states are Simplex states at the common boundary, ready
to process a Simplex descendant.

The function below is deliberately eager so that the executable reference has
one concrete `Store` representation. A client need not materialize every local
branch at the activation tick. Fork-choice storage is node-local, and its size
is not consensus-bounded during a finality stall; eagerly computing the full
available-committee window once per retained branch could therefore create an
activation-time resource cliff. An implementation MAY retain each unconverted
Gloas branch state behind a deterministic conversion thunk, but it must retain
every block and live latest message preserved by the eager function. In
particular, a live message on a finalized-conflicting root remains raw evidence
and cannot be discarded merely because its root is outside the finalized
subtree, but it is inert in the grade denominator and G0 clearance. The
implementation MUST materialize the same branch-state result—advance a copy with
`gloas.process_slots(state, activation_slot)`, then pass it to
`upgrade_to_simplex`—before any Simplex handler, committee lookup, or
fork-choice read uses that branch. The state selected by the activation-slot
walk must be materialized before the activation committee is pinned. Conversion
results may be shared or memoized only when every input affecting the advanced
state and committee selection is identical; in particular, branch-specific
RANDAO, registry, and effective-balance inputs preclude unconditional sharing.
The lazy representation must be observationally equivalent to the eager
function.

The legacy Store checkpoints carry epoch-boundary slots even when their roots
name earlier proposals. The live store has the block tree needed to recover the
actual proposal slot, so the migrated fork-choice checkpoints use that exact
slot. They are height-0 initialization anchors, not Simplex certificate events.
The incompatible legacy epoch-FFG caches are cleared; new per-round structures
begin empty. Live LMD head votes are converted by dropping only their Gloas
payload bit, so the grade layer does not unnecessarily forget the pre-fork view.
Normal expiry applies to those votes. The one still-actionable Gloas PTC view is
different: votes for the slot immediately before activation determine whether
the first Simplex proposer extends that parent's payload. Migration projects
those positional arrays into Simplex's identity-keyed map. Duplicate PTC seats
merge into one identity entry; conflicting data across retained roots marks that
identity as equivocating. Older PTC views cannot affect a new previous-slot
decision and are discarded.

An unresolved execution payload envelope is also allowed to arrive after the
activation tick. For every pre-fork root without an already verified envelope,
the migration therefore retains a copy of the exact Gloas immediate post-block
state in `legacy_payload_verification_states`; the activation-slot state cannot
verify its historical block root, slot, fork domain, or committed bid. This
auxiliary state is removed after successful verification. Implementations may
instead retain an equivalent immutable verification witness, and may discard it
once the inherited envelope-ingress lower bound makes that root ineligible. The
witness must otherwise survive ordinary fork-choice persistence and restart;
discarding it while retaining an unresolved eligible block would make envelope
acceptance depend on whether activation happened during the current process.

```python
def upgrade_forkchoice_store_to_simplex(pre: gloas.Store) -> Store:
    activation_slot = compute_start_slot_at_epoch(SIMPLEX_FORK_EPOCH)
    activation_time = uint64(pre.genesis_time + SLOT_DURATION_MS * activation_slot // 1000)
    # A live migration from Gloas necessarily has a preceding slot whose PTC
    # view may still decide the first Simplex parent's payload.
    assert activation_slot > GENESIS_SLOT
    legacy_vote_slot = Slot(activation_slot - 1)

    # Migration is a one-time boundary action, before any fork-slot ingress.
    assert gloas.get_current_slot(pre) == activation_slot
    assert pre.time == activation_time
    assert set(pre.blocks) == set(pre.block_states)
    assert pre.justified_checkpoint.root in pre.blocks
    assert pre.finalized_checkpoint.root in pre.blocks
    assert all(message.root in pre.blocks for message in pre.latest_messages.values())

    blocks = {root: copy(block) for root, block in pre.blocks.items()}
    block_states: Dict[Root, BeaconState] = {}
    for root, pre_state in pre.block_states.items():
        branch_state = copy(pre_state)
        assert branch_state.slot <= activation_slot
        if branch_state.slot < activation_slot:
            gloas.process_slots(branch_state, activation_slot)
        block_states[root] = upgrade_to_simplex(branch_state)

    justified_root = pre.justified_checkpoint.root
    finalized_root = pre.finalized_checkpoint.root
    justified_checkpoint = Checkpoint(
        slot=blocks[justified_root].slot,
        root=justified_root,
    )
    finalized_checkpoint = Checkpoint(
        slot=blocks[finalized_root].slot,
        root=finalized_root,
    )

    # Project the only Gloas PTC view that remains live after activation. Gloas
    # stores one value per committee position, whereas Simplex stores one value
    # per validator identity and resolves duplicate-seat weight at read time.
    migrated_payload_votes: Dict[ValidatorIndex, PayloadAttestationData] = {}
    migrated_payload_equivocations: Set[ValidatorIndex] = set()
    for root in sorted(blocks):
        if blocks[root].slot != legacy_vote_slot:
            continue
        if (
            root not in pre.payload_timeliness_vote
            or root not in pre.payload_data_availability_vote
        ):
            continue
        timely_votes = pre.payload_timeliness_vote[root]
        availability_votes = pre.payload_data_availability_vote[root]
        ptc = gloas.get_ptc(pre.block_states[root], legacy_vote_slot)
        assert len(timely_votes) == len(availability_votes) == len(ptc)
        for position, validator_index in enumerate(ptc):
            payload_present = timely_votes[position]
            blob_data_available = availability_votes[position]
            # The Gloas ingress handler writes both arrays atomically.
            assert (payload_present is None) == (blob_data_available is None)
            if payload_present is None:
                continue
            data = PayloadAttestationData(
                beacon_block_root=root,
                slot=legacy_vote_slot,
                payload_present=payload_present,
                blob_data_available=blob_data_available,
            )
            if validator_index not in migrated_payload_votes:
                migrated_payload_votes[validator_index] = data
            elif migrated_payload_votes[validator_index] != data:
                migrated_payload_equivocations.add(validator_index)

    store = Store(
        time=pre.time,
        genesis_time=pre.genesis_time,
        justified_checkpoint=justified_checkpoint,
        justified_height=Height(0),
        finalized_checkpoint=finalized_checkpoint,
        h_max=GENESIS_HEIGHT,
        equivocating_indices=set(pre.equivocating_indices),
        blocks=blocks,
        block_states=block_states,
        checkpoint_states={},
        latest_messages={
            index: LatestMessage(slot=message.slot, root=message.root)
            for index, message in pre.latest_messages.items()
        },
        round_attestations={},
        round_equivocating_indices={},
        frozen_tsq_views={},
        tsq_selections={},
        pending_attestations={},
        pending_available_attestations={},
        latest_confirmed_head=(finalized_root, activation_slot),
        live_confirmed_head=(finalized_root, activation_slot),
        fast_confirmed_head=(finalized_root, activation_slot),
        stable_root=Root(),
        stable_root_payload_status=PAYLOAD_STATUS_PENDING,
        stable_root_proposal_root=Root(),
        # Leave the activation round without a precomputed decision. It may use
        # the ordinary G1 fallback until a complete TSQ freeze and next-round
        # selection are available.
        stable_root_round=compute_round_at_slot(Slot(activation_slot - 1)),
        round_proposals={},
        round_proposal_conflicts=set(),
        stable_root_decisions={},
        payloads={root: copy(payload) for root, payload in pre.payloads.items()},
        legacy_payload_verification_states={
            root: copy(pre_state)
            for root, pre_state in pre.block_states.items()
            if root not in pre.payloads
        },
        payload_votes={
            legacy_vote_slot: migrated_payload_votes,
            activation_slot: {},
        },
        payload_vote_equivocations={
            legacy_vote_slot: migrated_payload_equivocations,
            activation_slot: set(),
        },
        available_votes={activation_slot: {}},
        available_vote_equivocations={activation_slot: set()},
        view_freeze_slots=set(),
        available_timely_attesters={activation_slot: set()},
        available_timely_equivocations={activation_slot: set()},
        available_committees={},
        frozen_available_votes={},
    )
    cache_available_committee(store, activation_slot)
    return store
```

### Modified `get_forkchoice_store`

*Note*: The anchor is treated as a pre-justified, pre-finalized block at height
`0`, while its state-height is `GENESIS_HEIGHT == Height(1)`. Thus
`justified_height` is initialized to `Height(0)` and `h_max` to
`GENESIS_HEIGHT`, matching the paper convention that genesis is pre-justified at
height `0` and starts the finality gadget at state-height `1`. The empty
initialization of `payload_votes` means no initial strict-majority payload
support; the first post-anchor payload decision resolves through the tiebreak
path until PTC votes are recorded. `payloads` starts empty (matching gloas): the
anchor block's payload is implicitly treated as available because the payload
gates are only consulted post-anchor — they short-circuit to `False` for
non-anchor roots until an envelope arrives via `on_execution_payload_envelope`,
and the pre-finalized anchor never needs the timely/DA gate.
`latest_confirmed_head` and `fast_confirmed_head` are seeded to the anchor
`(root, slot)`. `equivocating_indices` is seeded from the anchor state's
one-time round-double-vote penalty bitmap so checkpoint-synced nodes preserve
the same permanent grade exclusion as nodes that observed the evidence live.

```python
def get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock) -> Store:
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    justified_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    finalized_checkpoint = Checkpoint(slot=anchor_state.slot, root=anchor_root)
    anchor_slot = anchor_state.slot
    return Store(
        time=uint64(anchor_state.genesis_time + SLOT_DURATION_MS * anchor_slot // 1000),
        genesis_time=anchor_state.genesis_time,
        justified_checkpoint=justified_checkpoint,
        # [New in Simplex]
        # Genesis is pre-justified at height 0
        justified_height=Height(0),
        finalized_checkpoint=finalized_checkpoint,
        # [New in Simplex]
        # Genesis state-height; bumped by on_block thereafter
        h_max=GENESIS_HEIGHT,
        equivocating_indices={
            ValidatorIndex(index)
            for index, penalized in enumerate(anchor_state.round_double_vote_penalized)
            if penalized
        },
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={},
        # [New in Simplex]
        round_attestations={},
        round_equivocating_indices={},
        frozen_tsq_views={},
        tsq_selections={},
        pending_attestations={},
        pending_available_attestations={},
        # [New in Simplex]
        latest_confirmed_head=(anchor_root, anchor_slot),
        # [New in Simplex]
        live_confirmed_head=(anchor_root, anchor_slot),
        # [New in Simplex]
        fast_confirmed_head=(anchor_root, anchor_slot),
        # [New in Simplex]
        # No stable root yet (Root() means no stable root has been selected).
        stable_root=Root(),
        stable_root_payload_status=PAYLOAD_STATUS_PENDING,
        stable_root_proposal_root=Root(),
        stable_root_round=GENESIS_ROUND,
        round_proposals={},
        round_proposal_conflicts=set(),
        stable_root_decisions={},
        # [Modified in Simplex]
        # gloas payloads model: starts empty; populated by on_execution_payload_envelope.
        payloads={},
        # A checkpoint-sync anchor has no live pre-fork ingress backlog.
        legacy_payload_verification_states={},
        payload_votes={anchor_slot: {}},
        payload_vote_equivocations={anchor_slot: set()},
        available_votes={anchor_slot: {}},
        available_vote_equivocations={anchor_slot: set()},
        view_freeze_slots=set(),
        available_timely_attesters={anchor_slot: set()},
        available_timely_equivocations={anchor_slot: set()},
        available_committees={anchor_slot: get_available_committee(anchor_state, anchor_slot)},
        # [New in Simplex]
        # No available-confirmation freeze captured yet.
        frozen_available_votes={},
    )
```

### Modified `get_ancestor`

*Note*: A finality stall can make the distance to the finalized checkpoint
arbitrarily large. The inherited recursive walk is therefore replaced by an
iterative walk; payload-status propagation is unchanged.

```python
def get_ancestor(store: Store, node: ForkChoiceNode, slot: Slot) -> ForkChoiceNode:
    current = node
    while store.blocks[current.root].slot > slot:
        block = store.blocks[current.root]
        current = ForkChoiceNode(
            root=block.parent_root,
            payload_status=get_parent_payload_status(store, block),
        )
    return current
```

### New `update_justified`

*Note*: Paper's `updateJustified`. For a justification cert event `(J', h')` —
accept the event only if the candidate descends from the current finalized
checkpoint (`F-filter`), then update
`(store.justified_checkpoint, store.justified_height)` iff the candidate's lex
key `(h', hash_tree_root(J'))` strictly exceeds the current store key
`(store.justified_height, hash_tree_root(store.justified_checkpoint))`. The
tiebreaker is `hash_tree_root(Checkpoint)` — deterministic and uniform across
clients — where the paper keys on the block hash. A letter deviation, intended:
the running max only needs some fixed injective key on candidates, and the
checkpoint's hash tree root is the canonical one here.

Height `0` is reserved for the migrated/checkpoint-sync initialization anchor.
It is not a Simplex justification certificate and therefore never participates
in the lexicographic update.

```python
def update_justified(
    store: Store, justified_checkpoint: Checkpoint, justified_height: Height
) -> None:
    """
    [New in Simplex] Paper's updateJustified. Filter candidates by F ⪯ J',
    then lex running-max on ``(h_j, hash_tree_root(J))``.
    """
    if justified_height == Height(0):
        return
    if justified_checkpoint == Checkpoint():
        return  # No justification yet (empty checkpoint)
    if justified_checkpoint.root not in store.blocks:
        return
    # F-filter: candidate must descend from (or equal) the current finalized checkpoint.
    if justified_checkpoint.root != store.finalized_checkpoint.root:
        if (
            get_ancestor(
                store,
                ForkChoiceNode(
                    root=justified_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING
                ),
                store.finalized_checkpoint.slot,
            ).root
            != store.finalized_checkpoint.root
        ):
            return
    new_key = (justified_height, hash_tree_root(justified_checkpoint))
    current_key = (store.justified_height, hash_tree_root(store.justified_checkpoint))
    if new_key > current_key:
        store.justified_checkpoint = justified_checkpoint
        store.justified_height = justified_height
```

### New `is_finalized_compatible`

```python
def is_finalized_compatible(store: Store, root: Root) -> bool:
    """
    [New in Simplex] Return whether ``root`` is comparable by ancestry with the
    current finalized root. Finalized-conflicting data remains stored as raw
    evidence but is inert in fork-choice scoring.
    """
    if root not in store.blocks:
        return False
    node = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    finalized = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    return is_ancestor(store, node, finalized) or is_ancestor(store, finalized, node)
```

### New `recompute_h_max`

```python
def recompute_h_max(store: Store) -> None:
    """
    [New in Simplex] Recompute the height frontier over the current finalized
    subtree. Blocks on branches cut off by finalization remain in ``store`` but
    cannot keep the height filter artificially high.
    """
    finalized = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    heights = [
        state.current_height
        for root, state in store.block_states.items()
        if root in store.blocks
        and is_ancestor(
            store,
            ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING),
            finalized,
        )
    ]
    assert len(heights) > 0
    store.h_max = max(heights)
```

### New `get_viability_height_threshold`

```python
def get_viability_height_threshold(store: Store) -> Height:
    """
    [New in Simplex] Return the minimum state-height required by the viable-tree
    height filter.
    """
    return Height(store.h_max - 1) if store.h_max > 0 else GENESIS_HEIGHT
```

### New `is_viable_leaf`

```python
def is_viable_leaf(store: Store, block_root: Root) -> bool:
    """
    [New in Simplex] A leaf of ``store.blocks`` is viable iff its state-height
    is at least ``store.h_max - 1`` (paper Definition: viable subtree).
    """
    finalized = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    node = ForkChoiceNode(root=block_root, payload_status=PAYLOAD_STATUS_PENDING)
    if not is_ancestor(store, node, finalized):
        return False
    block_state = store.block_states[block_root]
    return block_state.current_height >= get_viability_height_threshold(store)
```

### New `is_viable`

```python
def is_viable(store: Store, block_root: Root) -> bool:
    """
    [New in Simplex] A block is viable iff some leaf descendant of it in
    ``store.blocks`` is a viable leaf (paper Definition: viable subtree).
    """
    viable_blocks: Dict[Root, BeaconBlock] = {}
    return filter_block_tree(store, block_root, viable_blocks)
```

### Modified `filter_block_tree`

```python
def filter_block_tree(store: Store, block_root: Root, blocks: Dict[Root, BeaconBlock]) -> bool:
    """
    [Modified in Simplex] Add ``block_root`` and its viable descendants to
    ``blocks`` iff the branch contains a leaf whose state-height satisfies the
    viable-tree height filter. The post-order walk is iterative so a finality
    stall cannot exhaust the Python call stack, and the child index avoids a
    full ``store.blocks`` scan at every tree level.
    """
    finalized = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    start = ForkChoiceNode(root=block_root, payload_status=PAYLOAD_STATUS_PENDING)
    if not is_ancestor(store, start, finalized):
        return False

    children: Dict[Root, List[Root]] = {root: [] for root in store.blocks}
    for root, block in store.blocks.items():
        if root != block.parent_root and block.parent_root in children:
            children[block.parent_root].append(root)

    reachable: List[Root] = []
    pending = [block_root]
    while pending:
        root = pending.pop()
        reachable.append(root)
        pending.extend(children[root])

    viable_roots: Set[Root] = set()
    for root in reversed(reachable):
        child_roots = children[root]
        if any(child in viable_roots for child in child_roots) or (
            len(child_roots) == 0 and is_viable_leaf(store, root)
        ):
            viable_roots.add(root)
            blocks[root] = store.blocks[root]

    return block_root in viable_roots
```

### Modified `get_filtered_block_tree`

```python
def get_filtered_block_tree(store: Store) -> Dict[Root, BeaconBlock]:
    """
    [Modified in Simplex] Retrieve the viable subtree rooted at the finalized
    checkpoint. The Simplex root may be either ``store.finalized_checkpoint``
    or ``store.justified_checkpoint``, so filtering from finalized keeps both
    possible roots available.
    """
    blocks: Dict[Root, BeaconBlock] = {}
    filter_block_tree(store, store.finalized_checkpoint.root, blocks)
    return blocks
```

### New `is_in_filtered_block_tree`

```python
def is_in_filtered_block_tree(
    store: Store, blocks: Dict[Root, BeaconBlock], node: ForkChoiceNode
) -> bool:
    """
    [New in Simplex] Return whether ``node`` is in the filtered viable subtree.
    Payload-decision nodes share their root with the parent block, so they are in
    the filtered tree only if the block itself satisfies the height bound or if
    that payload-status branch has a filtered child block.
    """
    if node.root not in blocks:
        return False
    if not is_ptc_decision_node(store, node):
        return True
    if store.block_states[node.root].current_height >= get_viability_height_threshold(store):
        return True
    return len(get_node_children(store, blocks, node)) > 0
```

### New `update_finalized`

*Note*: Paper's `updateFinalized`. Advance `store.finalized_checkpoint` only if
the candidate strictly extends the current finalized checkpoint, is an ancestor
of or equal to `store.justified_checkpoint` (paper thm:fleqr: `Σ.F ⪯ F' ⪯ Σ.J`),
AND is in the viable subtree (paper viability guard, lem:viable-finalized).

```python
def update_finalized(store: Store, finalized_checkpoint: Checkpoint) -> None:
    """
    [New in Simplex] Advance Σ.F if candidate strictly extends Σ.F, is an
    ancestor of or equal to Σ.J, and lies in the viable subtree.
    """
    if finalized_checkpoint.slot <= store.finalized_checkpoint.slot:
        return
    if finalized_checkpoint.root not in store.blocks:
        return
    # F' must descend from the current finalized checkpoint.
    if (
        get_ancestor(
            store,
            ForkChoiceNode(root=finalized_checkpoint.root, payload_status=PAYLOAD_STATUS_PENDING),
            store.finalized_checkpoint.slot,
        ).root
        != store.finalized_checkpoint.root
    ):
        return
    # F' must be ancestor-or-self of Σ.J.
    if store.justified_checkpoint.root != finalized_checkpoint.root:
        if store.justified_checkpoint.root not in store.blocks:
            return
        if (
            get_ancestor(
                store,
                ForkChoiceNode(
                    root=store.justified_checkpoint.root,
                    payload_status=PAYLOAD_STATUS_PENDING,
                ),
                finalized_checkpoint.slot,
            ).root
            != finalized_checkpoint.root
        ):
            return
    # Viability guard: F' must be in the viable subtree.
    if not is_viable(store, finalized_checkpoint.root):
        return
    store.finalized_checkpoint = finalized_checkpoint
    recompute_h_max(store)
```

### New `has_unexpired_latest_message`

*Note*: Latest-message expiry. A validator's `latest_message` is counted by
`get_attestation_score` only while its slot is within
`LATEST_MESSAGE_EXPIRY_SLOTS` of the current slot, i.e. while
`message.slot > current_slot - LATEST_MESSAGE_EXPIRY_SLOTS`. The grade-1
fallback and G0 veto use this same predicate in both numerator and denominator.

```python
def has_unexpired_latest_message(store: Store, index: ValidatorIndex) -> bool:
    """
    [New in Simplex] Whether ``index`` has a non-equivocating ``latest_message``
    whose slot is within ``LATEST_MESSAGE_EXPIRY_SLOTS`` of the current slot.
    """
    if index not in store.latest_messages:
        return False
    if index in store.equivocating_indices:
        return False
    if not is_finalized_compatible(store, store.latest_messages[index].root):
        return False
    # Unexpired iff message.slot > current_slot - LATEST_MESSAGE_EXPIRY_SLOTS,
    # written additively to avoid underflow at low slots.
    return store.latest_messages[index].slot + LATEST_MESSAGE_EXPIRY_SLOTS > get_current_slot(store)
```

### New `get_total_active_voting_weight`

*Note*: This is the relative denominator `D` shared by the two-thirds grade-1
descent and the one-third G0 conflict veto. It counts exactly the active,
unslashed validators whose latest network-received head vote is live and
finalized-compatible. Unlike `get_total_balance`, it has no minimum-balance
floor: with no live messages the grade denominator is exactly zero.

```python
def get_total_active_voting_weight(store: Store) -> Gwei:
    """
    [New in Simplex] Return the effective-balance weight of active, unslashed
    validators with a live, non-equivocating, finalized-compatible latest head
    vote. Finalized-conflicting messages remain available as evidence but are
    excluded from both grade numerator and denominator.
    """
    state = store.block_states[store.justified_checkpoint.root]
    return Gwei(
        sum(
            state.validators[index].effective_balance
            for index in get_active_validator_indices(state, get_current_epoch(state))
            if not state.validators[index].slashed and has_unexpired_latest_message(store, index)
        )
    )
```

### New `get_view_freeze_due_ms`

```python
def get_view_freeze_due_ms() -> uint64:
    """[New in Simplex] Return the in-slot SG-TSQ view-freeze boundary."""
    return get_slot_component_duration_ms(VIEW_FREEZE_DUE_BPS)
```

### New `is_before_view_freeze_deadline`

```python
def is_before_view_freeze_deadline(store: Store) -> bool:
    """
    [New in Simplex] Return whether the exact current-slot 75% view-freeze
    event has not completed.
    """
    return get_current_slot(store) not in store.view_freeze_slots
```

### New `get_available_confirmation_due_ms`

```python
def get_available_confirmation_due_ms() -> uint64:
    """[New in Simplex] Return the in-slot timely cutoff for available-confirmation votes."""
    return get_slot_component_duration_ms(AVAILABLE_CONFIRMATION_DUE_BPS)
```

### New `is_at_or_before_available_confirmation_deadline`

```python
def is_at_or_before_available_confirmation_deadline(store: Store) -> bool:
    """
    [New in Simplex] Return whether the exact current-slot confirmation freeze
    has not completed. Messages delivered exactly at the deadline are processed
    before that event and therefore still return ``True``.
    """
    return get_current_slot(store) not in store.frozen_available_votes
```

### New `is_at_or_after_available_confirmation_deadline`

```python
def is_at_or_after_available_confirmation_deadline(store: Store) -> bool:
    """
    [New in Simplex] Return whether the exact current-slot
    available-confirmation freeze completed.
    """
    return get_current_slot(store) in store.frozen_available_votes
```

### New `is_before_attestation_deadline`

```python
def is_before_attestation_deadline(store: Store) -> bool:
    """[New in Simplex] Return whether current local time is before the attestation deadline."""
    seconds_since_genesis = store.time - store.genesis_time
    time_into_slot_ms = seconds_to_milliseconds(seconds_since_genesis) % SLOT_DURATION_MS
    return time_into_slot_ms < get_attestation_due_ms()
```

### New `is_ptc_decision_node`

```python
def is_ptc_decision_node(store: Store, node: ForkChoiceNode) -> bool:
    """Return whether ``node`` is a previous-slot payload decision (EMPTY/FULL)."""
    return node.payload_status != PAYLOAD_STATUS_PENDING and store.blocks[
        node.root
    ].slot + 1 == get_current_slot(store)
```

### Modified `get_supported_node`

*Note*: `LatestMessage` no longer carries `payload_present`, so the supported
node's payload status is passed in explicitly. A finality vote makes no payload
decision — its caller passes `PAYLOAD_STATUS_PENDING` (it stabilizes the beacon
block and the payloads already in its chain, leaving the tip payload to the
available / Goldfish layer, which passes the status its vote decides).

```python
def get_supported_node(message: LatestMessage, payload_status: PayloadStatus) -> ForkChoiceNode:
    return ForkChoiceNode(root=message.root, payload_status=payload_status)
```

### New `is_supporting_vote`

*Note*: Gloas removed `is_supporting_vote`; simplex reintroduces it. The payload
status the vote supports is supplied by the caller (see `get_supported_node`).

```python
def is_supporting_vote(
    store: Store,
    node: ForkChoiceNode,
    message: LatestMessage,
    payload_status: PayloadStatus,
) -> bool:
    return is_ancestor(store, get_supported_node(message, payload_status), node)
```

### New `get_available_majority_threshold`

```python
def get_available_majority_threshold(store: Store) -> uint64:
    """
    Return the majority threshold for previous-slot available-attestation gating.
    A child's score must exceed this value to be viable.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT:
        return uint64(0)
    previous_slot = Slot(current_slot - 1)
    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot entry.
    if previous_slot not in store.available_votes:
        return uint64(0)
    previous_votes = store.available_votes[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the committee pinned at the slot boundary so denominator and child score
    # use exactly the same electorate.
    previous_committee = store.available_committees.get(previous_slot, [])
    participant_count = uint64(
        len([index for index in previous_committee if index in previous_votes])
    )
    return participant_count // 2
```

### New `get_available_vote_payload_status`

*Note*: The payload status an available vote supports, mirroring the base
`get_supported_node` derivation: a vote whose block precedes its slot decides
`FULL`/`EMPTY` by `payload_present`; a same-slot vote is `PENDING`.

```python
def get_available_vote_payload_status(
    store: Store, data: AvailableAttestationData
) -> PayloadStatus:
    block = store.blocks[data.beacon_block_root]
    if block.slot < data.slot:
        if data.payload_present:
            return PAYLOAD_STATUS_FULL
        return PAYLOAD_STATUS_EMPTY
    return PAYLOAD_STATUS_PENDING
```

### New `get_available_attestation_score`

```python
def get_available_attestation_score(store: Store, child: ForkChoiceNode) -> uint64:
    """
    Return the available-attestation score for ``child``: non-equivocating
    supporting votes plus total equivocations from the previous slot's
    available committee.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, child):
        return uint64(0)

    previous_slot = Slot(current_slot - 1)
    # available_votes is seeded per-slot by on_tick_per_slot; a checkpoint-sync
    # anchor evaluated before its first tick has no previous-slot entry.
    if previous_slot not in store.available_votes:
        return uint64(0)
    previous_votes = store.available_votes[previous_slot]
    previous_equivocations = store.available_vote_equivocations[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; iterate the pinned previous-slot
    # committee to resolve seat multiplicity (a validator holding k seats
    # contributes k, and all k seats are excluded when it equivocates).
    previous_committee = store.available_committees.get(previous_slot, [])
    score = uint64(0)
    for member_index in previous_committee:
        if member_index in previous_equivocations:
            score += 1  # Equivocator counted for viability
            continue
        if member_index not in previous_votes:
            continue
        vote = previous_votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, child, message, payload_status):
            score += 1
    return score
```

### New `is_available_attestation_viable`

```python
def is_available_attestation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in the ordinary Layer 3 Goldfish walk.
    PTC decision nodes and non-round-start current-slot proposals pass through;
    a round-start proposal is considered separately after the ordinary walk.
    Other children require available-attestation score exceeding the majority
    threshold.
    """
    if is_ptc_decision_node(store, child):
        return True
    if store.blocks[child.root].slot == get_current_slot(store):
        block_round = compute_round_at_slot(store.blocks[child.root].slot)
        return store.blocks[child.root].slot != compute_start_slot_at_round(block_round)
    return get_available_attestation_score(store, child) > get_available_majority_threshold(store)
```

### New `cache_available_committee`

*Note*: The paper leaves the common-randomness / branch-relative available
committee construction as an open item. This executable profile pins the local
committee once, at the start of each slot, from the node's then-current walk
head. Under the synchrony premise used by view merge, honest nodes have the same
head by the committee-selection boundary and therefore pin the same committee.
The cache makes that premise explicit and keeps committee evaluation total when
justification stalls beyond `MIN_SEED_LOOKAHEAD`; it is not itself a proof of
committee convergence before stabilization.

```python
def cache_available_committee(store: Store, slot: Slot) -> None:
    """
    [New in Simplex] Pin ``slot``'s available committee from the walk head at
    the slot boundary. Idempotent: later blocks or votes cannot change it.
    """
    if slot in store.available_committees:
        return
    head = get_head(store)
    state = copy(store.block_states[head.root])
    if state.slot < slot:
        process_slots(state, slot)
    store.available_committees[slot] = get_available_committee(state, slot)
```

### New `freeze_available_votes`

*Note*: The node-local per-slot available-confirmation freeze. The snapshot
contains the committee pinned at the slot boundary, first wire votes received
before the confirmation deadline, and equivocations observed before that
deadline for those voters. Deadline-specific equivocation tracking is separate
from live Goldfish tracking, so a conflicting copy received after the exact
freeze cannot change the snapshot. A missed high-resolution event creates no
snapshot for that slot; the slot-boundary handler never reconstructs one from a
later view. Once captured, the local numerator and denominator do not retract.
This does not imply identical snapshots at different nodes: adversarial
stragglers can cross the deadline in one view but not another, and proving or
repairing cross-node structural consistency remains a research obligation. The
confirmation-head walk is also outside the freeze; it starts from the live
finalized root over the live block tree.

```python
def freeze_available_votes(store: Store, slot: Slot) -> None:
    """
    [New in Simplex] Capture the available-confirmation freeze for ``slot``: the
    slot's available committee and the node's deadline-classified first votes.
    Idempotent: the first capture wins.
    """
    if slot in store.frozen_available_votes:
        return
    committee = store.available_committees.get(slot, [])
    votes = store.available_votes.get(slot, {})
    timely_equivocations = store.available_timely_equivocations.get(slot, set())
    timely_attesters = store.available_timely_attesters.get(slot, set())
    frozen_votes = {
        index: votes[index]
        for index in votes
        if index in timely_attesters and index not in timely_equivocations
    }
    store.frozen_available_votes[slot] = FrozenAvailableVotes(
        committee=committee, votes=frozen_votes
    )
```

### New `get_available_confirmation_score`

*Note*: `store.frozen_available_votes[slot]` is the node's per-slot
available-confirmation snapshot, capturing its deadline-classified votes and the
pinned committee (see `freeze_available_votes`). The available-confirmation rule
reads the previous slot's snapshot, while fast confirmation reads the current
slot's snapshot immediately. Later local messages are never counted. The
snapshot covers this numerator and its matching denominator
(`get_available_confirmation_majority_threshold`), but neither cross-node
snapshot equality nor the live-tree walk consuming the scores.

```python
def get_available_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    Return delayed available-confirmation support for ``node`` from the
    previous slot's freeze: timely, non-equivocating available attesters as of
    that slot's confirmation deadline.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    previous_slot = Slot(current_slot - 1)
    # The freeze is captured by the exact high-resolution event; a missed event
    # or a checkpoint-sync anchor before its first event has no previous-slot
    # freeze.
    if previous_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[previous_slot]
    # [Modified in Simplex]
    # Votes are keyed by validator identity; iterate the frozen committee to
    # resolve seat multiplicity. Timeliness and equivocation exclusions are
    # baked into the frozen votes, so a post-deadline equivocation report
    # cannot retro-mutate the score.
    count = uint64(0)
    for member_index in freeze.committee:
        if member_index not in freeze.votes:
            continue
        vote = freeze.votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, node, message, payload_status):
            count += 1
    return count
```

### New `get_available_confirmation_majority_threshold`

*Note*: Within one local snapshot, the available-confirmation relative quorum
must use the same frozen electorate for BOTH numerator and denominator. This
denominator therefore counts the previous slot's frozen electorate, exactly
matching the numerator's electorate in `get_available_confirmation_score`. It is
distinct from `get_available_majority_threshold` (the all-votes threshold gating
the Goldfish head), whose base-branch semantics is unchanged. Matching the two
quantities prevents local post-freeze retraction; it does not establish
cross-node agreement under adversarial straggler timing.

```python
def get_available_confirmation_majority_threshold(store: Store) -> uint64:
    """
    [New in Simplex] Return the relative-majority threshold for delayed
    available confirmation over the frozen electorate: the previous slot's
    timely, non-equivocating available attesters (seat-counted). Numerator and
    this denominator read the same local snapshot.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT:
        return uint64(0)
    previous_slot = Slot(current_slot - 1)
    # The freeze is captured by the exact high-resolution event; a missed event
    # or a checkpoint-sync anchor before its first event has no previous-slot
    # freeze.
    if previous_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[previous_slot]
    participant_count = uint64(0)
    for member_index in freeze.committee:
        if member_index in freeze.votes:
            participant_count += 1
    return participant_count // 2
```

### New `is_available_confirmation_viable`

```python
def is_available_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    Return whether ``child`` is viable in delayed available confirmation:
    PTC decision nodes always pass through; other children require
    available-confirmation score exceeding the frozen-electorate majority
    threshold.
    """
    if is_ptc_decision_node(store, child):
        return True
    # [Modified in Simplex]
    # Numerator and denominator both read the same node-local per-slot
    # snapshot, so later messages cannot retract that node's evaluation.
    return get_available_confirmation_score(
        store, child
    ) > get_available_confirmation_majority_threshold(store)
```

### New `get_best_available_confirmation_child`

```python
def get_best_available_confirmation_child(
    store: Store,
    blocks: Dict[Root, BeaconBlock],
    head: ForkChoiceNode,
) -> Optional[ForkChoiceNode]:
    """
    [New in Simplex] Return the best child for delayed available confirmation.
    ``blocks`` is the unfiltered accepted block tree: child filtering here is
    only the local confirmation threshold, not the height/tree viability
    filter.
    """
    children = [
        child
        for child in get_node_children(store, blocks, head)
        if is_available_confirmation_viable(store, child)
    ]
    if len(children) == 0:
        return None
    return max(
        children,
        key=lambda child: (
            get_available_confirmation_score(store, child),
            child.root,
            get_payload_status_tiebreaker(store, child),
        ),
    )
```

### New `get_fast_confirmation_score`

```python
def get_fast_confirmation_score(store: Store, node: ForkChoiceNode) -> uint64:
    """
    [New in Simplex] Return immediate fast-confirmation support for ``node``
    from the current slot's freeze: timely, non-equivocating available
    attesters as of the slot's confirmation deadline.
    """
    current_slot = get_current_slot(store)
    if current_slot == GENESIS_SLOT or is_ptc_decision_node(store, node):
        return uint64(0)

    # The freeze is captured by the exact high-resolution event; a missed event
    # or a checkpoint-sync anchor before its first event has no current-slot
    # freeze.
    if current_slot not in store.frozen_available_votes:
        return uint64(0)
    freeze = store.frozen_available_votes[current_slot]
    # [New in Simplex]
    # Votes are keyed by validator identity; iterate the frozen committee to
    # resolve seat multiplicity. Timeliness and equivocation exclusions are
    # baked into the frozen votes.
    count = uint64(0)
    for member_index in freeze.committee:
        if member_index not in freeze.votes:
            continue
        vote = freeze.votes[member_index]
        message = LatestMessage(slot=vote.slot, root=vote.beacon_block_root)
        payload_status = get_available_vote_payload_status(store, vote)
        if is_supporting_vote(store, node, message, payload_status):
            count += 1
    return count
```

### New `is_fast_confirmation_viable`

```python
def is_fast_confirmation_viable(store: Store, child: ForkChoiceNode) -> bool:
    """
    [New in Simplex] Return whether ``child`` is viable in immediate fast
    confirmation: PTC decision nodes always pass through; other children require
    an absolute 75% of ``AVAILABLE_COMMITTEE_SIZE`` seats.
    """
    if is_ptc_decision_node(store, child):
        return True
    return (
        get_fast_confirmation_score(store, child) * FAST_CONFIRMATION_COMMITTEE_DENOMINATOR
        >= AVAILABLE_COMMITTEE_SIZE * FAST_CONFIRMATION_COMMITTEE_NUMERATOR
    )
```

### New `get_fast_confirmation_head`

```python
def get_fast_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    [New in Simplex] Return the immediate fast-confirmation head for the
    current slot, from the current slot's frozen available votes and an
    absolute 75% committee-seat threshold, over all accepted descendants of
    the finalized root.
    """
    # User-facing confirmation is floorless and never gated by finality-gadget
    # or grade state: it walks the unfiltered accepted block tree from the
    # finalized root, with no height-filter viability bound.
    blocks = store.blocks
    head = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )

    # Fast confirmation. Among fast-viable children pick by confirmation
    # score, then root, then payload-status tiebreaker. At the 75%-absolute
    # threshold, at most one block child can cross.
    while True:
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_fast_confirmation_viable(store, child)
        ]
        if len(children) == 0:
            return head
        head = max(
            children,
            key=lambda child: (
                get_fast_confirmation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
```

### New `get_attestation_checkpoint_state`

*Note*: Committee membership and signing domains are epoch-based, so an
attestation is verified using its own known head-chain state, processed forward
to the duty epoch boundary when necessary. If the head is already inside that
epoch, its cached committee inputs are valid for the whole epoch. This avoids
walking below an exact-slot checkpoint-sync anchor when that anchor is later
than the epoch's first slot. Callers still require the named head block to be no
later than the attestation.

```python
def get_attestation_checkpoint_state(store: Store, data: AttestationData) -> BeaconState:
    """
    [New in Simplex] Return (and cache) the state used to resolve committees and
    verify signatures for an attestation: its known head-chain state, processed
    to the start of the duty epoch if it is older.
    """
    attestation_epoch = compute_epoch_at_slot(data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    checkpoint_key = (data.beacon_block_root, attestation_epoch)
    if checkpoint_key not in store.checkpoint_states:
        base_state = copy(store.block_states[data.beacon_block_root])
        if base_state.slot < epoch_boundary_slot:
            process_slots(base_state, epoch_boundary_slot)
        store.checkpoint_states[checkpoint_key] = base_state
    return store.checkpoint_states[checkpoint_key]
```

### New `get_tsq_quorum_threshold`

*Note*: TSQ uses the same absolute finality quorum rounding at every selection
and action. There is no lower positive synchronization threshold.

```python
def get_tsq_quorum_threshold(total_active_balance: Gwei) -> Gwei:
    """[New in Simplex] Return ``ceil(2 * W / 3)`` for the fixed electorate."""
    return Gwei(
        (total_active_balance * FINALITY_QUORUM_NUMERATOR + FINALITY_QUORUM_DENOMINATOR - 1)
        // FINALITY_QUORUM_DENOMINATOR
    )
```

### New `get_tsq_effective_head`

*Note*: Project one signed support-round head backward to its deepest ancestor
in the pinned candidate tree. The projection never moves a vote forward or
across a fork. `Root()` means that the message has no effective head in this
selection.

```python
def get_tsq_effective_head(store: Store, selection: TSQSelection, data: AttestationData) -> Root:
    """[New in Simplex] Project ``data`` into ``selection.candidate_roots``."""
    root = data.beacon_block_root
    while root in store.blocks:
        if root in selection.candidate_roots:
            return root
        block = store.blocks[root]
        if block.slot == GENESIS_SLOT:
            break
        root = block.parent_root
    return Root()
```

### New `get_tsq_effective_heads`

*Note*: The bounded executable representation retains the first validated
complete data per signer and an absorbing same-round equivocation marker. A
marked signer supplies no TSQ support. This is conservative relative to
filtering all copies against a later finalized root before deciding whether the
signer equivocated: it may remove faulty support, but cannot create support or
affect the quorum of responsive honest signers.

```python
def get_tsq_effective_heads(
    store: Store,
    selection: TSQSelection,
    attestations: Dict[ValidatorIndex, AttestationData],
    equivocating_indices: Set[ValidatorIndex],
) -> Dict[ValidatorIndex, Root]:
    """[New in Simplex] Return one effective TSQ head per usable signer."""
    effective_heads: Dict[ValidatorIndex, Root] = {}
    for index, data in attestations.items():
        if index in equivocating_indices or selection.weights.get(index, Gwei(0)) == Gwei(0):
            continue
        effective_head = get_tsq_effective_head(store, selection, data)
        if effective_head != Root():
            effective_heads[index] = effective_head
    return effective_heads
```

### New `get_tsq_intersection_heads`

*Note*: A receiver counts a signer only when the same complete signed data is
present in both its immutable frozen view and its current view, and neither view
marks the signer as an equivocator. A current-only message is not enough. A
distinct copy learned after the freeze excludes the signer through the
current-view marker without changing the frozen view.

```python
def get_tsq_intersection_heads(
    store: Store, selection: TSQSelection, frozen_view: FrozenTSQView
) -> Dict[ValidatorIndex, Root]:
    """[New in Simplex] Return effective heads usable in ``F ∩ U``."""
    current_attestations = store.round_attestations.get(selection.support_round, {})
    current_equivocators = store.round_equivocating_indices.get(selection.support_round, set())
    intersection: Dict[ValidatorIndex, AttestationData] = {}
    excluded = frozen_view.equivocating_indices | current_equivocators
    for index, frozen_data in frozen_view.attestations.items():
        if index not in excluded and current_attestations.get(index) == frozen_data:
            intersection[index] = frozen_data
    return get_tsq_effective_heads(store, selection, intersection, excluded)
```

### New `get_tsq_support`

```python
def get_tsq_support(
    store: Store,
    selection: TSQSelection,
    effective_heads: Dict[ValidatorIndex, Root],
    root: Root,
) -> Gwei:
    """[New in Simplex] Return fixed-electorate subtree support for ``root``."""
    if root not in selection.candidate_roots:
        return Gwei(0)
    candidate = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    return Gwei(
        sum(
            selection.weights[index]
            for index, effective_head in effective_heads.items()
            if is_ancestor(
                store,
                ForkChoiceNode(
                    root=effective_head,
                    payload_status=PAYLOAD_STATUS_PENDING,
                ),
                candidate,
            )
        )
    )
```

### New `get_deepest_tsq_root`

*Note*: Starting from the pinned Simplex root, descend through the unique direct
candidate-tree child with fixed-quorum support. Two conflicting children cannot
both qualify because `2 * q > W` and each usable signer contributes once. When
no child qualifies, the current root is returned; in particular, no support is
required to use the pinned Simplex root as the default.

```python
def get_deepest_tsq_root(
    store: Store,
    selection: TSQSelection,
    effective_heads: Dict[ValidatorIndex, Root],
) -> Root:
    """[New in Simplex] Return the deepest fixed-quorum TSQ root."""
    q = get_tsq_quorum_threshold(selection.total_active_balance)
    root = selection.simplex_root
    if q == Gwei(0):
        return root
    while True:
        children = [
            candidate_root
            for candidate_root in selection.candidate_roots
            if store.blocks[candidate_root].parent_root == root
            and get_tsq_support(store, selection, effective_heads, candidate_root) >= q
        ]
        assert len(children) <= 1
        if len(children) == 0:
            return root
        root = children[0]
```

### New `freeze_tsq_view`

*Note*: The high-resolution scheduler calls this exactly at
`VIEW_FREEZE_DUE_BPS` in the last slot of a round, after processing messages
delivered at the boundary. The snapshot is assigned to the following
synchronization round. A missed event or restart does not authorize
reconstruction from the later live cache.

```python
def freeze_tsq_view(store: Store) -> None:
    """[New in Simplex] Freeze the current support round for the next round."""
    slot = get_current_slot(store)
    support_round = compute_round_at_slot(slot)
    synchronization_round = Round(support_round + 1)
    assert compute_round_at_slot(Slot(slot + 1)) == synchronization_round
    if synchronization_round in store.frozen_tsq_views:
        return
    store.frozen_tsq_views[synchronization_round] = FrozenTSQView(
        support_round=support_round,
        attestations={
            index: copy(data)
            for index, data in store.round_attestations.get(support_round, {}).items()
        },
        equivocating_indices=set(store.round_equivocating_indices.get(support_round, set())),
    )
```

### New `freeze_tsq_selection`

*Note*: Called at a round's first-slot boundary before admitting any proposal
for that round. It pins the viable candidate tree and the active-balance
electorate from the local Simplex-root chain, then computes the proposer-time
candidate from the live support-round view. The snapshot is immutable. If the
preceding public freeze is unavailable, no selection is created and the later
action falls back to `G1`.

```python
def freeze_tsq_selection(store: Store) -> None:
    """[New in Simplex] Pin the current round's proposal-independent TSQ state."""
    slot = get_current_slot(store)
    round = compute_round_at_slot(slot)
    if slot != compute_start_slot_at_round(round) or round in store.tsq_selections:
        return
    if round not in store.frozen_tsq_views:
        return

    blocks = get_filtered_block_tree(store)
    simplex_root = get_simplex_root(store)
    simplex_node = ForkChoiceNode(
        root=simplex_root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    candidate_roots = {
        root
        for root in blocks
        if is_ancestor(
            store,
            ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING),
            simplex_node,
        )
    }

    state = copy(store.block_states[simplex_root])
    if state.slot < slot:
        process_slots(state, slot)
    epoch = get_current_epoch(state)
    weights = {
        index: state.validators[index].effective_balance
        for index in get_active_validator_indices(state, epoch)
        if not state.validators[index].slashed
    }
    selection = TSQSelection(
        support_round=store.frozen_tsq_views[round].support_round,
        simplex_root=simplex_root,
        candidate_roots=candidate_roots,
        weights=weights,
        total_active_balance=get_total_active_balance(state),
        candidate_root=simplex_root,
    )
    current_heads = get_tsq_effective_heads(
        store,
        selection,
        store.round_attestations.get(selection.support_round, {}),
        store.round_equivocating_indices.get(selection.support_round, set()),
    )
    selection.candidate_root = get_deepest_tsq_root(store, selection, current_heads)
    store.tsq_selections[round] = selection
```

### New `update_round_proposals`

*Note*: Called by `on_block`. It records every round-start proposal root
received during its own round. Collection does not affect the independent TSQ
lock and cannot change an already completed action.

```python
def update_round_proposals(store: Store, block_root: Root) -> None:
    """[New in Simplex] Record a current round-start proposal."""
    block = store.blocks[block_root]
    block_round = compute_round_at_slot(block.slot)
    if compute_round_at_slot(get_current_slot(store)) != block_round:
        return
    if block.slot != compute_start_slot_at_round(block_round):
        return
    proposals = store.round_proposals.setdefault(block_round, set())
    proposals.add(block_root)
    if len(proposals) > 1:
        store.round_proposal_conflicts.add(block_round)
```

### New `mark_round_proposal_conflict`

*Note*: Gossip calls this after fully validating one distinct second signed
round-start block from the same proposer and slot, even if that block has not
yet completed `on_block`. `update_round_proposals` also records the conflict
when it processes two distinct round-start proposals whose branch-relative
proposer identities differ. Either case disables proposal-specific treatment but
never changes the independently computed TSQ lock.

```python
def mark_round_proposal_conflict(store: Store, round: Round) -> None:
    """[New in Simplex] Record multiple validated round-start proposals."""
    store.round_proposal_conflicts.add(round)
```

### New `freeze_stable_root`

*Note*: The high-resolution duty scheduler calls this helper once at the common
first-slot available-vote action, immediately before it snapshots the
`head_root` placed in `RoundSelectionEvent`. The helper freezes the round's
stable root: the deepest fixed-quorum lock from the frozen/current support-view
intersection when it passes the live action-state checks, or the grade-1
fallback as it exists at that action. A later vote, proposal, equivocation, or
ancestry arrival cannot change the stored decision.

The proposal does not determine the lock. After choosing the stable root, the
action separately distinguishes a proposal only when exactly one round-start
proposal was received by the action and it is viable and descends from both the
stable root and the action-time live available-confirmed head. Two proposal
copies distinguish neither. During the round-start slot, `get_head` starts the
available-vote walk at that distinguished proposal. From the next slot onward,
the proposal is an ordinary prior block and must win through actual available
votes.

```python
def freeze_stable_root(store: Store) -> None:
    """[New in Simplex] Select and fix the current round's stable root."""
    round = compute_round_at_slot(get_current_slot(store))
    if round in store.stable_root_decisions:
        return

    blocks = get_filtered_block_tree(store)
    stable_root = get_grade_1_root(store, blocks)
    frozen_proposal_root = Root()
    selection = store.tsq_selections.get(round)
    frozen_view = store.frozen_tsq_views.get(round)
    if (
        selection is not None
        and frozen_view is not None
        and selection.support_round == frozen_view.support_round
    ):
        intersection_heads = get_tsq_intersection_heads(store, selection, frozen_view)
        lock_root = get_deepest_tsq_root(store, selection, intersection_heads)
        lock = ForkChoiceNode(
            root=lock_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        simplex_node = ForkChoiceNode(
            root=get_simplex_root(store),
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        if is_in_filtered_block_tree(store, blocks, lock) and is_ancestor(
            store, lock, simplex_node
        ):
            stable_root = lock

    # Count every fully processed proposal received by the action before
    # applying local viability. Two distinct proposals distinguish neither,
    # even when only one remains in the filtered tree.
    proposal_roots = store.round_proposals.get(round, set())
    if round not in store.round_proposal_conflicts and len(proposal_roots) == 1:
        proposal_root = next(iter(proposal_roots))
        proposal = ForkChoiceNode(
            root=proposal_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        confirmed_root = store.live_confirmed_head[0]
        confirmed = ForkChoiceNode(
            root=confirmed_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        if (
            proposal_root in blocks
            and proposal_root in store.block_states
            and store.blocks[proposal_root].slot == compute_start_slot_at_round(round)
            and confirmed_root in store.blocks
            and is_ancestor(store, proposal, stable_root)
            and is_ancestor(store, proposal, confirmed)
        ):
            frozen_proposal_root = proposal_root

    store.stable_root = stable_root.root
    store.stable_root_payload_status = stable_root.payload_status
    store.stable_root_proposal_root = frozen_proposal_root
    store.stable_root_round = round
    store.stable_root_decisions[round] = True
```

### New `get_simplex_root`

*Note*: Paper Definition: Simplex root — the current J/F-selected root:
`store.justified_checkpoint` when it sits at the height frontier, otherwise the
always-viable `store.finalized_checkpoint` (paper lem:F-viable).

```python
def get_simplex_root(store: Store) -> Root:
    """
    [New in Simplex] Return the Simplex root: ``store.justified_checkpoint``
    when ``store.h_max == store.justified_height + 1``, else
    ``store.finalized_checkpoint``.
    """
    if store.h_max == store.justified_height + 1:
        return store.justified_checkpoint.root
    return store.finalized_checkpoint.root
```

### New `get_grade_1_root`

*Note*: Paper Definition: grade-1 root `G1` — the walk's fallback root, used
when a TSQ snapshot is unavailable or its selected lock fails the action-state
checks. From the Simplex root, descend while a viable child holds at least
two-thirds of the live latest-head-vote weight. The threshold is relative to
`get_total_active_voting_weight`, unlike TSQ's fixed absolute electorate. Every
validator has at most one live latest message in a local view, so child supports
sum to at most the denominator and the descent is unique whenever it steps. This
descent is over direct beacon-block children and deliberately skips the ePBS
payload-decision nodes: finality head votes stabilize beacon blocks at
`PAYLOAD_STATUS_PENDING`, while Goldfish resolves payload status in phase 2. No
block inclusion is involved.

```python
def get_grade_1_root(store: Store, blocks: Dict[Root, BeaconBlock]) -> ForkChoiceNode:
    """
    [New in Simplex] Return the grade-1 fallback root: the latest-head-vote
    descent under the two-thirds threshold, restricted to the viable subtree.
    """
    root = get_simplex_root(store)
    head = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
    voting_weight = get_total_active_voting_weight(store)
    if voting_weight == Gwei(0):
        return head
    state = store.block_states[store.justified_checkpoint.root]
    while True:
        # SG grades are over beacon-block heads, not virtual EMPTY/FULL payload
        # decisions. Descend through direct block children and leave payload
        # selection to the Goldfish phase of get_head.
        children = [
            ForkChoiceNode(root=child_root, payload_status=PAYLOAD_STATUS_PENDING)
            for child_root, child_block in blocks.items()
            if child_block.parent_root == head.root
        ]
        grade_1_children = [
            child
            for child in children
            if is_in_filtered_block_tree(store, blocks, child)
            and get_attestation_score(store, child, state) * 3 >= voting_weight * 2
        ]
        if len(grade_1_children) == 0:
            return head
        # Unique: at most one child can hold >= 2/3 of live head-vote weight.
        head = grade_1_children[0]
```

### New `get_stable_root`

*Note*: Phase 1 of the walk (paper Definition: the walk). Before the common
action it returns the live grade-1 fallback. After `freeze_stable_root`, every
read in the round returns the stable root selected at that action without
re-evaluating either TSQ support or the fallback.

```python
def get_stable_root(store: Store, blocks: Dict[Root, BeaconBlock]) -> ForkChoiceNode:
    """
    [New in Simplex] Return the current round's stable root after selection,
    else the live grade-1 fallback before selection.
    """
    current_round = compute_round_at_slot(get_current_slot(store))
    if current_round in store.stable_root_decisions and store.stable_root_round == current_round:
        frozen_root = ForkChoiceNode(
            root=store.stable_root,
            payload_status=store.stable_root_payload_status,
        )
        simplex_node = ForkChoiceNode(
            root=get_simplex_root(store),
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        # TSQ support is never re-evaluated within the round. However, a
        # later authenticated state change can make the frozen root unavailable
        # below the current Simplex root (in particular after finalization).
        # Such a root is semantically inert; use the current grade-1 fallback.
        if frozen_root.root in blocks and is_ancestor(store, frozen_root, simplex_node):
            return frozen_root
    return get_grade_1_root(store, blocks)
```

### New `get_safe_confirmed_head`

*Note*: Paper Definition: safe confirmation. Above the Simplex root, a block is
*safe-confirmed* iff it is availability-confirmed (an ancestor of
`store.live_confirmed_head`) and G0-clear — no conflicting block holds at least
one-third latest-head-vote support. The Simplex root itself is safe by
construction and is the unconditional floor. The safe-confirmed head is the
deepest such block. Above the floor it is well defined because G0-clearance is
monotone along a chain: a block conflicting with an ancestor also conflicts with
the block, so every ancestor of a G0-clear block is G0-clear. Safe confirmation
is an *internal*, recoverable notion read by the finality-vote gates (validator
spec). The user-facing non-retracting record (`store.latest_confirmed_head`) is
untouched by it and is never gated by grade or finality-gadget state. Separating
the two prevents a conflicting result seen before stabilization from permanently
blocking healing after the live available chain converges.

```python
def get_safe_confirmed_head(store: Store) -> Root:
    """
    [New in Simplex] Return the safe-confirmed head: the deepest G0-clear
    ancestor of the available-confirmed head, floored at the Simplex root.
    """
    simplex_root = get_simplex_root(store)
    simplex_node = ForkChoiceNode(
        root=simplex_root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    head = store.live_confirmed_head[0]
    if head not in store.blocks:
        return simplex_root
    head_node = ForkChoiceNode(root=head, payload_status=PAYLOAD_STATUS_PENDING)
    if not is_ancestor(store, head_node, simplex_node):
        return simplex_root
    # By monotonicity, the first G0-clear block on the walk up from the
    # available-confirmed head is the deepest G0-clear ancestor. The walk
    # stops at the Simplex root, which is safe by construction. In particular,
    # finalized-conflicting raw evidence can never push safe confirmation below
    # the finalized/justified root from which the walk starts.
    while head != simplex_root and not is_g0_clear(store, head):
        head = store.blocks[head].parent_root
    return head
```

### Modified `get_safe_execution_block_hash`

*Note*: Simplex removes the legacy `FastConfirmationStore`. As in Gloas, only
the parent execution payload of a safe-confirmed beacon block is considered
safe, but the safe beacon root now comes from `get_safe_confirmed_head`.

```python
def get_safe_execution_block_hash(store: Store) -> Hash32:
    safe_root = get_safe_confirmed_head(store)
    safe_block = store.blocks[safe_root]
    return safe_block.body.signed_execution_payload_bid.message.parent_block_hash
```

### New `update_confirmed_head`

*Note*: Confirmation is non-retracting. A later frozen electorate that cannot
re-confirm an already confirmed descendant does not erase that result. The
stored head advances only to a descendant; a conflicting candidate is ignored.
The finalized-conflict fallback is defensive and should be unreachable under the
finality safety invariant.

```python
def update_confirmed_head(
    store: Store,
    previous: Tuple[Root, Slot],
    candidate: Root,
) -> Tuple[Root, Slot]:
    previous_root, _ = previous
    current_slot = get_current_slot(store)
    if previous_root not in store.blocks:
        return candidate, current_slot
    previous_node = ForkChoiceNode(root=previous_root, payload_status=PAYLOAD_STATUS_PENDING)
    finalized_node = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    if not is_ancestor(store, previous_node, finalized_node):
        return candidate, current_slot
    if candidate == previous_root:
        return previous
    candidate_node = ForkChoiceNode(root=candidate, payload_status=PAYLOAD_STATUS_PENDING)
    if is_ancestor(store, candidate_node, previous_node):
        return candidate, current_slot
    return previous
```

### New `update_confirmation_heads`

This helper evaluates both confirmation rules for the slot named by
`get_current_slot(store)`. It is called at the confirmation deadline and once
more immediately before leaving a slot. The boundary call only re-evaluates
snapshots already captured by exact high-resolution events; it never creates a
missing snapshot. Re-evaluation is safe because snapshots are immutable and the
confirmation walk intentionally reads the live block tree.

```python
def update_confirmation_heads(store: Store) -> None:
    current_slot = get_current_slot(store)
    confirmation_candidate = get_available_confirmation_head(store).root
    store.live_confirmed_head = (confirmation_candidate, current_slot)
    store.latest_confirmed_head = update_confirmed_head(
        store,
        store.latest_confirmed_head,
        confirmation_candidate,
    )
    store.fast_confirmed_head = update_confirmed_head(
        store,
        store.fast_confirmed_head,
        get_fast_confirmation_head(store).root,
    )
```

### New `get_available_confirmation_head`

*Note*: `update_confirmation_heads` first calls this at the exact
high-resolution confirmation event and may call it again before leaving the
slot, using the previous slot's node-local frozen scores. The walk is computed
over **all** currently accepted descendants of the current finalized root, not
the viability-filtered subtree. Consequently, freezing the scores does not
freeze this candidate: later block arrival or a finalized-root change may change
the walk. The helper stores the candidate directly in
`store.live_confirmed_head` and separately passes it through
`update_confirmed_head` for the monotone user-facing record. The internal,
grade-gated notion the finality-vote gates read is `get_safe_confirmed_head`.

```python
def get_available_confirmation_head(store: Store) -> ForkChoiceNode:
    """
    [New in Simplex] Return the delayed available-confirmation head for slot
    ``n`` when called in slot ``n+1``, from the previous slot's frozen
    available votes, over all accepted descendants of the finalized root.
    """
    # User-facing confirmation is floorless and never gated by finality-gadget
    # or grade state: it walks the unfiltered accepted block tree from the
    # finalized root, with no height-filter viability bound.
    blocks = store.blocks
    head = ForkChoiceNode(
        root=store.finalized_checkpoint.root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )

    # Delayed available confirmation. Among children meeting the local
    # confirmation threshold, pick by score, then root, then payload-status
    # tiebreaker -- matching get_head's disambiguation so that at a
    # payload-decision (EMPTY/FULL) node the better-supported payload wins
    # rather than the inherited EMPTY-first child order.
    while True:
        child = get_best_available_confirmation_child(store, blocks, head)
        if child is None:
            return head
        head = child
```

### New `get_payload_participant_count`

```python
def get_payload_participant_count(store: Store, root: Root) -> uint64:
    """Return the number of PTC seats with a vote in the block's slot."""
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    return uint64(len([index for index in ptc if index in payload_votes]))
```

### New `get_payload_full_support`

```python
def get_payload_full_support(store: Store, root: Root) -> uint64:
    """
    Return payload FULL support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``payload_present == True`` count.
    Equivocating participants in the slot are included for viability.
    """
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    equivocations = store.payload_vote_equivocations.get(vote_slot, set())
    full_support_count = uint64(0)
    for ptc_member_index in ptc:
        if ptc_member_index not in payload_votes:
            continue
        vote = payload_votes[ptc_member_index]
        if ptc_member_index in equivocations or (
            vote.beacon_block_root == root and vote.payload_present
        ):
            full_support_count += 1
    return full_support_count
```

### New `get_payload_data_available_support`

```python
def get_payload_data_available_support(store: Store, root: Root) -> uint64:
    """
    Return payload data-availability support for ``root`` in its slot.
    Non-equivocating votes for ``root`` with ``blob_data_available == True``
    count. Equivocating participants in the slot are included for viability.
    """
    # [Modified in Simplex]
    # Votes are keyed by validator identity; resolve seat multiplicity against
    # the block's own PTC committee.
    vote_slot = store.blocks[root].slot
    ptc = get_ptc(store.block_states[root], vote_slot)
    payload_votes = store.payload_votes.get(vote_slot, {})
    equivocations = store.payload_vote_equivocations.get(vote_slot, set())
    data_available_support_count = uint64(0)
    for ptc_member_index in ptc:
        if ptc_member_index not in payload_votes:
            continue
        vote = payload_votes[ptc_member_index]
        if ptc_member_index in equivocations or (
            vote.beacon_block_root == root and vote.blob_data_available
        ):
            data_available_support_count += 1
    return data_available_support_count
```

### New `is_payload_verified`

*Note*: Adopted verbatim from gloas. Membership in `store.payloads` is the
local-availability gate consulted by the payload-decision helpers below.

```python
def is_payload_verified(store: Store, root: Root) -> bool:
    """
    Return whether the execution payload envelope for the beacon block with
    root ``root`` has been locally delivered and verified via
    ``on_execution_payload_envelope``.
    """
    return root in store.payloads
```

### Modified `is_payload_timely`

```python
def is_payload_timely(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload FULL support.
    """
    # [Modified in Simplex]
    # Local-availability gate now reads ``store.payloads`` via ``is_payload_verified``.
    if not is_payload_verified(store, root):
        return False

    participant_count = get_payload_participant_count(store, root)
    full_support_count = get_payload_full_support(store, root)
    return full_support_count > participant_count // 2
```

### Modified `is_payload_data_available`

```python
def is_payload_data_available(store: Store, root: Root) -> bool:
    """
    Return whether ``root`` has strict-majority payload data-availability support.
    """
    # [Modified in Simplex]
    # Local-availability gate now reads ``store.payloads`` via ``is_payload_verified``.
    if not is_payload_verified(store, root):
        return False

    participant_count = get_payload_participant_count(store, root)
    data_available_support_count = get_payload_data_available_support(store, root)
    return data_available_support_count > participant_count // 2
```

### Modified `should_extend_payload`

```python
def should_extend_payload(store: Store, root: Root) -> bool:
    # [Modified in Simplex]
    # Strict majority required for both payload presence and data availability.
    return is_payload_timely(store, root) and is_payload_data_available(store, root)
```

### Modified `should_build_on_full`

*Note*: gloas's version calls `payload_timeliness` /
`payload_data_availability`, which read the removed
`store.payload_timeliness_vote` / `payload_data_availability_vote`. Simplex
decides build-on-full from its own strict-majority `should_extend_payload`,
keeping the proposer's choice consistent with the fork-choice payload decision.

```python
def should_build_on_full(store: Store, head: ForkChoiceNode) -> bool:
    assert head.payload_status != PAYLOAD_STATUS_PENDING
    if store.blocks[head.root].slot + 1 != get_current_slot(store):
        return head.payload_status == PAYLOAD_STATUS_FULL
    if head.payload_status == PAYLOAD_STATUS_EMPTY:
        return False
    # [Modified in Simplex]
    # Strict-majority decision via simplex's own payload counters.
    return should_extend_payload(store, head.root)
```

### Modified `update_latest_messages`

*Note*: Before updating the expiring SG latest message, retain the first valid
attestation data seen from each signer in its round and mark the signer if any
distinct valid data is later received in that round. This bounded round-local
view supplies both proposer-time and receiver TSQ inputs. It is updated from
gossip and block delivery, just like the grade input. A marked signer is
excluded from TSQ support.

```python
def update_latest_messages(
    store: Store, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation
) -> None:
    # [Modified in Simplex]
    # First update the bounded round-local view used by TSQ. Keep one signed
    # data value per validator plus an absorbing equivocation bit. A marked
    # signer contributes no TSQ support.
    round = compute_round_at_slot(attestation.data.slot)
    round_attestations = store.round_attestations.setdefault(round, {})
    round_equivocators = store.round_equivocating_indices.setdefault(round, set())
    for i in attesting_indices:
        if i not in round_attestations:
            round_attestations[i] = attestation.data
        elif round_attestations[i] != attestation.data:
            round_equivocators.add(i)
            # The grades exclude every validator known to have signed two
            # distinct valid attestations, even when both copies carry the
            # same head field and therefore would evade the same-slot/root
            # check below.
            store.equivocating_indices.add(i)

    # Every valid finality attestation also immediately supplies its signers'
    # SG latest head vote. Inclusion is not required. Globally known grade
    # equivocators remain excluded from the grade numerator and denominator.
    slot = attestation.data.slot
    beacon_block_root = attestation.data.beacon_block_root
    non_equivocating_attesting_indices = [
        i for i in attesting_indices if i not in store.equivocating_indices
    ]
    for i in non_equivocating_attesting_indices:
        if i in store.latest_messages and slot == store.latest_messages[i].slot:
            if beacon_block_root != store.latest_messages[i].root:
                store.equivocating_indices.add(i)
            continue
        if i not in store.latest_messages or slot > store.latest_messages[i].slot:
            store.latest_messages[i] = LatestMessage(
                slot=slot,
                root=beacon_block_root,
            )
```

### New `apply_attestation_latest_messages`

```python
def apply_attestation_latest_messages(
    store: Store,
    attesting_indices: Sequence[ValidatorIndex],
    attestation: Attestation,
) -> None:
    """
    Apply a valid finality attestation to the grade inputs immediately upon
    receipt, including when received in its own slot (paper Definition: live
    latest head vote).
    """
    assert attestation.data.slot <= get_current_slot(store)
    update_latest_messages(store, attesting_indices, attestation)
```

### Modified `get_attestation_score`

*Note*: The base fork choice provides `get_attestation_score`; simplex overrides
it to add the latest-message expiry filter (`has_unexpired_latest_message`).
Only unexpired, non-equivocating supporting latest messages contribute. The
grade-1 fallback and G0 veto consume this score directly; Goldfish continues to
use available attestations.

```python
def get_attestation_score(
    store: Store,
    node: ForkChoiceNode,
    state: BeaconState,
    window_slots: uint64 = LATEST_MESSAGE_EXPIRY_SLOTS,
) -> Gwei:
    """
    [Modified in Simplex] Effective-balance weight of unexpired, non-equivocating
    latest messages supporting ``node`` cast within the last ``window_slots``
    slots (default: the whole unexpired set).
    """
    current_slot = get_current_slot(store)
    unslashed_and_active_indices = [
        i
        for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    return Gwei(
        sum(
            state.validators[i].effective_balance
            for i in unslashed_and_active_indices
            if (
                has_unexpired_latest_message(store, i)
                and store.latest_messages[i].slot + window_slots >= current_slot
                # Finality votes are beacon-block votes at PAYLOAD_STATUS_PENDING.
                and is_supporting_vote(
                    store, node, store.latest_messages[i], PAYLOAD_STATUS_PENDING
                )
            )
        )
    )
```

### New `is_g0_clear`

*Note*: Paper Definition: G0 clearance. Both the one-third numerator and its
relative denominator are computed over the same network-received live,
finalized-compatible latest head votes. Finalized-conflicting blocks and votes
remain stored as evidence but do not participate. With no live votes the check
is vacuously clear; the user-facing available confirmation remains unaffected.

```python
def is_g0_clear(store: Store, target_root: Root) -> bool:
    """
    [New in Simplex] Return whether no block conflicting with ``target_root``
    has at least one-third of live latest-head-vote weight.
    """
    voting_weight = get_total_active_voting_weight(store)
    if voting_weight == Gwei(0):
        return True
    state = store.block_states[store.justified_checkpoint.root]
    target = ForkChoiceNode(root=target_root, payload_status=PAYLOAD_STATUS_PENDING)
    for root in store.blocks:
        if not is_finalized_compatible(store, root):
            continue
        node = ForkChoiceNode(root=root, payload_status=PAYLOAD_STATUS_PENDING)
        conflicts = not is_ancestor(store, node, target) and not is_ancestor(store, target, node)
        if conflicts and get_attestation_score(store, node, state) * 3 >= voting_weight:
            return False
    return True
```

### Modified `get_weight`

```python
def get_weight(
    store: Store, node: ForkChoiceNode, window_slots: uint64 = LATEST_MESSAGE_EXPIRY_SLOTS
) -> Gwei:
    # [Modified in Simplex]
    # Returns 0 for payload-decision nodes; no proposer boost. Counts only votes
    # cast within the last ``window_slots`` slots (default: whole unexpired set).
    if is_ptc_decision_node(store, node):
        return Gwei(0)
    state = store.block_states[store.justified_checkpoint.root]
    return get_attestation_score(store, node, state, window_slots)
```

### Modified `get_head`

*Note*: The walk (paper Definition: the walk), in three phases. (1) *Stable
root*: the fixed-quorum TSQ lock when it passes the action-state checks,
otherwise the grade-1 fallback (`get_stable_root`). (2) *Goldfish*: in the
round-start slot, a unique proposal distinguished at the action starts the walk;
it was already checked to descend both the stable root and the action-time live
available-confirmed head. This lets the proposal align compatible locks without
overriding that action-time live prefix. Otherwise, and from the next slot
onward, follow the available chain from the stable root by previous-slot
participant majority. (3) *Viability descent*: continue without the majority
gate until the head's state-height reaches the height-filter bound `h_max - 1`.
Proposals, available votes, and finality-vote construction read this walk.
Unlike the paper's one-walk abstraction, `get_available_confirmation_head` and
`get_fast_confirmation_head` deliberately read their separate floorless,
unfiltered confirmation walk; see the Introduction. This deviation is exercised
by a test in which confirmation supports a branch excluded by the viability
filter.

```python
def get_head(store: Store) -> ForkChoiceNode:
    # [Modified in Simplex]
    # Get filtered block tree that only includes viable branches
    blocks = get_filtered_block_tree(store)

    # Phase 1 -- stable root: the locally selected TSQ lock, else grade 1.
    head = get_stable_root(store, blocks)

    # Phase 2 -- at the round-start available-vote action, a unique eligible
    # proposal starts Goldfish from the frozen stable root. This may replace an
    # unconfirmed ordinary head, which is the alignment step. The action
    # already required the proposal to preserve the live confirmed prefix.
    current_round = compute_round_at_slot(get_current_slot(store))
    current_round_start = compute_start_slot_at_round(current_round)
    if (
        get_current_slot(store) == current_round_start
        and current_round in store.stable_root_decisions
        and store.stable_root_round == current_round
        and store.stable_root_proposal_root in blocks
    ):
        proposal = ForkChoiceNode(
            root=store.stable_root_proposal_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        if is_ancestor(store, proposal, head):
            head = proposal

    # Follow ordinary Goldfish from the stable root or the distinguished
    # proposal. From the next slot onward the proposal has no special status
    # and can be selected only through actual available votes.
    while True:
        children = get_node_children(store, blocks, head)
        viable_children = [
            child
            for child in children
            if is_in_filtered_block_tree(store, blocks, child)
            and is_available_attestation_viable(store, child)
        ]
        if len(viable_children) == 0:
            break
        head = max(
            viable_children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )

    # Phase 3 -- viability descent: the phase-2 descent continued without its
    # majority gate (paper Definition: viability descent), until the head's
    # state-height reaches the height-filter bound. A viable child exists
    # until the bound is met; the emptiness guard only keeps the walk total.
    while store.block_states[head.root].current_height < get_viability_height_threshold(store):
        children = [
            child
            for child in get_node_children(store, blocks, head)
            if is_in_filtered_block_tree(store, blocks, child)
        ]
        if len(children) == 0:
            break
        head = max(
            children,
            key=lambda child: (
                get_available_attestation_score(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )
    return head
```

### Modified `get_proposer_head`

*Note*: In a round-start slot with a pinned TSQ selection, an honest proposer
first finds the deeper of the pinned candidate and its live available-confirmed
head when they are compatible. This is the TSQ base. If the ordinary `head_root`
descends from that base, the proposer preserves the deeper healthy suffix; a
conflicting unconfirmed head does not veto synchronization. The ordinary head
remains the fallback when the selection is missing, either required root is
unknown or too new, or the required roots conflict; such a round is not counted
for TSQ liveness.

```python
def get_proposer_head(store: Store, head_root: Root, slot: Slot) -> Root:
    # [Modified in Simplex]
    round = compute_round_at_slot(slot)
    if slot != compute_start_slot_at_round(round):
        return head_root
    selection = store.tsq_selections.get(round)
    if selection is None:
        return head_root

    confirmed_root = store.live_confirmed_head[0]
    required_roots = (selection.candidate_root, confirmed_root)
    if any(root not in store.blocks or store.blocks[root].slot >= slot for root in required_roots):
        return head_root

    tsq_base_root = max(required_roots, key=lambda root: store.blocks[root].slot)
    tsq_base = ForkChoiceNode(
        root=tsq_base_root,
        payload_status=PAYLOAD_STATUS_PENDING,
    )
    if not all(
        is_ancestor(
            store,
            tsq_base,
            ForkChoiceNode(
                root=root,
                payload_status=PAYLOAD_STATUS_PENDING,
            ),
        )
        for root in required_roots
    ):
        return head_root

    # Preserve a healthy ordinary suffix when it already extends the TSQ base.
    # A conflicting unconfirmed head does not veto synchronization.
    if head_root in store.blocks and store.blocks[head_root].slot < slot:
        head = ForkChoiceNode(
            root=head_root,
            payload_status=PAYLOAD_STATUS_PENDING,
        )
        if is_ancestor(store, head, tsq_base):
            return head_root
    return tsq_base_root
```

### New `is_attestation_from_store_active_simplex_fork`

```python
def is_attestation_from_store_active_simplex_fork(store: Store, data: AttestationData) -> bool:
    """Return whether ``data`` is from a slot with Simplex duties in this store."""
    # Synthetic feature tests may carry an activation epoch only in the anchor
    # state's Fork. A production pre-fork or later-fork finalized state instead
    # falls back to SIMPLEX_FORK_EPOCH through the shared predicate.
    return is_attestation_from_active_simplex_fork(
        store.block_states[store.finalized_checkpoint.root], data
    )
```

### New `is_valid_from_block_attestation`

*Note*: The skip-only validator for block-included finality attestations, called
by `on_attestation` on the from-block path. It never asserts: any failure skips
the attestation's effects and leaves the block accepted, so a block's
fork-choice acceptance can never depend on the local view of the attestations it
carries. (An assert here would split block acceptance across views — the
head-chain committee resolution and aggregate-signature check can differ across
diverged forks, so only skip-only attribution keeps acceptance
view-independent.) The justification target is not validated here because the
latest-message update consumes only the head field. Ordinary finality processing
has already validated the attestation against the including chain.

The signature check makes latest-message attribution correct. The state
transition verified the aggregate under the *including* chain, while fork choice
resolves the committee on the attestation's own head chain. On diverged forks
the two samplings can disagree, and unverified attribution could assign a head
vote or an equivocation to a non-signer. Re-verifying under the
head-chain-resolved committee makes the attributed indices actual signers. The
committee-structure walk also length-guards the bits/committee mapping, which on
mismatched committee sizes would otherwise raise.

```python
def is_valid_from_block_attestation(store: Store, attestation: Attestation) -> bool:
    """
    [New in Simplex] Return whether a block-included finality attestation may
    update latest messages: well-formed data and a valid aggregate signature
    under the committee resolved on the attestation's own head chain. Checked,
    never asserted: a failure skips this fork-choice effect and never rejects
    the block.
    """
    data = attestation.data
    if not is_attestation_from_store_active_simplex_fork(store, data):
        return False
    # The named head block must not be later than the attestation.
    if store.blocks[data.beacon_block_root].slot > data.slot:
        return False
    # The justification target is deliberately NOT validated here. This path
    # consumes only the signed head field for latest-message fork choice;
    # ordinary finality processing of the including block is separate.
    # Attestations can only affect fork choice of subsequent slots.
    if get_current_slot(store) < data.slot + 1:
        return False
    # Committee structure (Electra pattern) under the head-chain checkpoint
    # state: on a diverged fork the committee sizes can differ from the
    # including chain's, so the bits/committee mapping is length-guarded.
    checkpoint_state = get_attestation_checkpoint_state(store, data)
    data_epoch = compute_epoch_at_slot(data.slot)
    committee_offset = 0
    for committee_index in get_committee_indices(attestation.committee_bits):
        if committee_index >= get_committee_count_per_slot(checkpoint_state, data_epoch):
            return False
        committee_offset += len(get_beacon_committee(checkpoint_state, data.slot, committee_index))
    if len(attestation.aggregation_bits) != committee_offset:
        return False
    # Attribution is signature-verified under the resolving state: the state
    # transition verified this aggregate under the including chain, whose
    # committee sampling can disagree with the head chain's on diverged forks.
    return is_valid_indexed_attestation(
        checkpoint_state, get_indexed_attestation(checkpoint_state, attestation)
    )
```

### New `is_valid_from_block_available_attestation_precheck`

*Note*: The state-free half of the skip-only validator for block-included
available (Goldfish) attestations. `on_available_attestation` calls it on the
from-block path *before* deriving the head-chain checkpoint state, so that
derivation is never reached for an attestation that will be skipped. This
mirrors the finality path, where `is_valid_from_block_attestation` runs its
head-slot precheck before `get_attestation_checkpoint_state`: the head-slot
check (named head not later than the attestation) is the shield that keeps the
epoch-boundary `get_ancestor` walk in the derivation from descending below the
anchor on a checkpoint-synced store, where it would otherwise `KeyError` — a
raise the skip-only from-block path must never make. It carries the from-block
form of every well-formedness check that `validate_on_available_attestation`
asserts on the wire path — head-block slot, same-slot payload flag, and
committee size — as return-`False` skips, so those checks can never reject the
block.

```python
def is_valid_from_block_available_attestation_precheck(
    store: Store, attestation: AvailableAttestation
) -> bool:
    """
    [New in Simplex] State-free well-formedness prechecks for a block-included
    available attestation, run before the head-chain checkpoint state is
    derived. Checked, never asserted: a failure skips the attestation, never
    rejects the block. The head-slot check is the shield that keeps the
    epoch-boundary ``get_ancestor`` derivation from walking below the anchor.
    """
    data = attestation.data
    # The named head block (known here: the unknown-head from-block case is
    # skipped in ``on_available_attestation``) must not be later than the
    # attestation, and a same-slot attestation cannot signal payload
    # availability. Wire attestations assert these in
    # ``validate_on_available_attestation``; on the from-block path they skip.
    block_slot = store.blocks[data.beacon_block_root].slot
    if block_slot > data.slot:
        return False
    if block_slot == data.slot and data.payload_present:
        return False
    # Bits must match the fixed AVAILABLE_COMMITTEE_SIZE committee: the
    # from-block form of the wire ``bits == AVAILABLE_COMMITTEE_SIZE`` check and
    # defensive parity with the finality helper's committee-structure guard. The
    # available committee size is fixed, so unlike the finality committee it
    # cannot diverge across forks.
    if len(attestation.aggregation_bits) != AVAILABLE_COMMITTEE_SIZE:
        return False
    return True
```

### New `is_valid_from_block_available_attestation`

*Note*: The signature half of the skip-only validator for block-included
available (Goldfish) attestations — the analog of
`is_valid_from_block_attestation` for the per-slot available-committee votes.
`on_available_attestation` calls it on the from-block path, *after* the
state-free `is_valid_from_block_available_attestation_precheck` and the
checkpoint-state derivation, before attributing votes and equivocation marks.
Like the finality helper it never asserts: any failure skips the attestation's
effects and leaves the block accepted, so a block's fork-choice acceptance can
never depend on the local view of the available attestations it carries.

The signature check is what makes vote/equivocation attribution correct: the
state transition verified the aggregate under the *including* chain
(`process_available_attestation`), while attribution resolves the available
committee on the attestation's own head chain — on RANDAO-diverged forks the two
samplings can disagree, and unverified attribution would let one included
aggregate mark non-signers (two such same-slot inclusions could manufacture
equivocation marks against honest validators). Re-verifying under the
head-chain-resolved committee makes the attributed indices actual signers. The
bits/committee length agreement that `get_available_attesting_indices` relies on
is guaranteed by the precheck's committee-size skip, which runs first.

```python
def is_valid_from_block_available_attestation(
    state: BeaconState, attestation: AvailableAttestation
) -> bool:
    """
    [New in Simplex] Return whether a block-included available attestation
    carries a valid aggregate signature under the available committee resolved
    on the attestation's own head chain (``state``, the head-chain checkpoint
    state). Well-formedness is checked separately by
    ``is_valid_from_block_available_attestation_precheck`` before ``state`` is
    derived. Checked, never asserted: a failure skips the attestation, never
    rejects the block.
    """
    data = attestation.data
    # Attribution is signature-verified under the resolving state: the state
    # transition verified this aggregate under the including chain, whose
    # committee sampling can disagree with the head chain's on diverged forks.
    attesting_indices = get_available_attesting_indices(state, attestation)
    pubkeys = [state.validators[i].pubkey for i in sorted(attesting_indices)]
    domain = get_domain(state, DOMAIN_AVAILABLE_ATTESTER, compute_epoch_at_slot(data.slot))
    signing_root = compute_signing_root(data, domain)
    return bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)
```

### Modified `validate_on_attestation`

*Note*: Wire-only. Wire attestations must name known blocks (the head, and the
justification target when one is set), be no later than the current slot, and
remain inside the configured latest-message window. A valid same-slot vote
enters the latest-message view immediately, as required by the paper. From-block
attestations never reach this function: they take the skip-only
`is_valid_from_block_attestation` path in `on_attestation`. Timeout votes and
empty votes use `Checkpoint()` as the target and still carry a head vote.

```python
def validate_on_attestation(store: Store, attestation: Attestation) -> None:
    data = attestation.data
    # A Simplex-shaped wire vote cannot claim a pre-activation duty.
    assert is_attestation_from_store_active_simplex_fork(store, data)
    # Attestation must be for a known block
    assert data.beacon_block_root in store.blocks
    # Block must not be in the future
    block_slot = store.blocks[data.beacon_block_root].slot
    assert block_slot <= data.slot
    if data.target != Checkpoint():
        # [Modified in Simplex]
        # A wire justification target must name a known real block at its
        # actual proposal slot.
        assert data.target.root in store.blocks
        assert store.blocks[data.target.root].slot == data.target.slot
        # Target slot may precede attestation slot (height-based finality)
        assert data.target.slot <= data.slot

    # Network-received head votes enter without block inclusion and remain
    # admissible for their configured live window, including in their own slot.
    current_slot = get_current_slot(store)
    assert data.slot <= current_slot
    assert data.slot + LATEST_MESSAGE_EXPIRY_SLOTS > current_slot
```

### New `validate_on_available_attestation`

*Note*: Wire-only. Wire available attestations must be for the current slot and
a known block, carry exactly `AVAILABLE_COMMITTEE_SIZE` bits, and (when
same-slot) not signal payload availability, asserting on violation. From-block
attestations never reach these asserts: they take the skip-only
`is_valid_from_block_available_attestation_precheck` path in
`on_available_attestation`, which applies the same well-formedness checks as
return-`False` skips, so a block's acceptance can never depend on the local view
of the available attestations it carries.

```python
def validate_on_available_attestation(
    store: Store, attestation: AvailableAttestation, is_from_block: bool
) -> None:
    if not is_from_block:
        # Wire votes are only accepted for the current slot
        # (view-merge synchronization window).
        assert attestation.data.slot == get_current_slot(store)
        # Attestations must be for a known block.
        assert attestation.data.beacon_block_root in store.blocks
        # Attestations must not be for blocks in the future.
        block_slot = store.blocks[attestation.data.beacon_block_root].slot
        assert block_slot <= attestation.data.slot
        head_state = store.block_states[attestation.data.beacon_block_root]
        if head_state.fork.current_version == SIMPLEX_FORK_VERSION:
            assert attestation.data.slot >= compute_start_slot_at_epoch(head_state.fork.epoch)
        # Available attestation bits must match the fixed committee size.
        assert len(attestation.aggregation_bits) == AVAILABLE_COMMITTEE_SIZE
        # Same-slot attestation cannot signal payload availability
        if block_slot == attestation.data.slot:
            assert not attestation.data.payload_present
```

## Handlers

### Modified `on_tick_per_slot`

```python
def on_tick_per_slot(store: Store, time: uint64) -> None:
    # [Modified in Simplex]
    # No epoch boundary pull-up; initializes per-slot vote tracking.
    previous_slot = get_current_slot(store)
    tick_slot = Slot((time - store.genesis_time) * 1000 // SLOT_DURATION_MS)
    if tick_slot > previous_slot:
        # Re-evaluate while ``get_current_slot`` still names the slot being
        # left: fast confirmation reads that slot, and delayed confirmation
        # reads its predecessor. A missed high-resolution freeze is not
        # reconstructed from the later boundary view.
        update_confirmation_heads(store)
    store.time = time
    current_slot = get_current_slot(store)
    if current_slot > previous_slot:
        # [New in Simplex]
        # The outer ``on_tick`` handler invokes this routine once for every
        # crossed slot.
        store.payload_votes[current_slot] = {}
        store.payload_vote_equivocations[current_slot] = set()
        store.available_votes[current_slot] = {}
        store.available_vote_equivocations[current_slot] = set()
        store.available_timely_attesters[current_slot] = set()
        store.available_timely_equivocations[current_slot] = set()
        # At a round boundary, pin the proposal-independent TSQ candidate tree
        # and electorate before any proposal for the new round is admitted.
        # A missing prior high-resolution freeze deliberately yields no
        # selection and the later action falls back to G1.
        freeze_tsq_selection(store)
        # Pin the new slot's electorate from the boundary head. Goldfish reads
        # only this entry for live scoring and later copies it into the freeze.
        cache_available_committee(store, current_slot)
        # Goldfish consumes only current/previous-slot live data; confirmed
        # results retain their own frozen snapshot and monotone stored head.
        oldest_available_slot = (
            GENESIS_SLOT if current_slot == GENESIS_SLOT else Slot(current_slot - 1)
        )
        for mapping in (
            store.payload_votes,
            store.payload_vote_equivocations,
            store.available_votes,
            store.available_vote_equivocations,
            store.available_timely_attesters,
            store.available_timely_equivocations,
            store.available_committees,
            store.frozen_available_votes,
        ):
            for tracked_slot in list(mapping):
                if tracked_slot < oldest_available_slot:
                    mapping.pop(tracked_slot)
        # TSQ reads the preceding round, but equivocation detection must retain
        # first-data history for the entire latest-message ingress window.
        # Otherwise a late conflicting copy can arrive after its first copy was
        # pruned and evade global grade exclusion.
        oldest_live_slot = (
            GENESIS_SLOT
            if current_slot < LATEST_MESSAGE_EXPIRY_SLOTS
            else Slot(current_slot - LATEST_MESSAGE_EXPIRY_SLOTS + 1)
        )
        oldest_round = compute_round_at_slot(oldest_live_slot)
        tracked_rounds = set(store.round_attestations) | set(store.round_equivocating_indices)
        for tracked_round in tracked_rounds:
            if tracked_round < oldest_round:
                store.round_attestations.pop(tracked_round, None)
                store.round_equivocating_indices.pop(tracked_round, None)
        # TSQ snapshots and the stable root are consumed only by their own
        # synchronization round. Proposal blocks and signed messages remain in
        # their ordinary stores.
        current_round = compute_round_at_slot(current_slot)
        for round_mapping in (
            store.frozen_tsq_views,
            store.tsq_selections,
            store.round_proposals,
            store.stable_root_decisions,
        ):
            for tracked_round in list(round_mapping):
                if tracked_round < current_round:
                    round_mapping.pop(tracked_round)
        store.round_proposal_conflicts = {
            tracked_round
            for tracked_round in store.round_proposal_conflicts
            if tracked_round >= current_round
        }
        # Drop unresolved finality heads once their latest-message window has
        # closed, and unresolved available heads once their one-slot Goldfish
        # use has passed.
        for root, attestations in list(store.pending_attestations.items()):
            live = [
                attestation
                for attestation in attestations
                if attestation.data.slot + LATEST_MESSAGE_EXPIRY_SLOTS > current_slot
            ]
            if live:
                store.pending_attestations[root] = live
            else:
                store.pending_attestations.pop(root)
        for root, attestations in list(store.pending_available_attestations.items()):
            live = [
                attestation
                for attestation in attestations
                if attestation.data.slot + 1 >= current_slot
            ]
            if live:
                store.pending_available_attestations[root] = live
            else:
                store.pending_available_attestations.pop(root)
        store.view_freeze_slots = {
            tracked_slot for tracked_slot in store.view_freeze_slots if tracked_slot >= current_slot
        }
    # [New in Simplex]
    # Confirmation rules: after the exact high-resolution confirmation event
    # has captured the current slot's node-local snapshot, derive the
    # available-confirmation candidate from the previous slot's snapshot and
    # the fast-confirmation candidate from the current slot's snapshot. Repeat
    # ticks use the same frozen scores, but the confirmation walks still read
    # the live tree from the live finalized root. Thus the live candidate can
    # change; only update_confirmed_head's user-facing record is locally
    # non-retracting.
    # At equality, clients MUST process every inbound message timestamped at the
    # deadline before invoking this tick. Thus ``<=`` ingress and ``>=``
    # evaluation are deterministic: a vote delivered exactly at 2*Delta enters
    # the freeze, while a strictly later vote does not.
    if is_at_or_after_available_confirmation_deadline(store):
        freeze_available_votes(store, get_current_slot(store))
        update_confirmation_heads(store)
```

### New `on_tick_per_high_resolution`

*Note*: `Store.time` has whole-second resolution, so the ordinary `on_tick`
cannot represent every in-slot deadline. The client scheduler calls this handler
exactly at each configured millisecond boundary, after processing messages
delivered at that boundary. A missed freeze or action is not reconstructed from
a later view; a later slot boundary may still re-evaluate a confirmation rule
from an immutable snapshot that was captured on time. Before the slot-boundary
`on_tick`, clients process prior-round messages delivered at or before that
boundary; only after `on_tick_per_slot` has pinned the new round's selection may
they admit its proposal.

```python
def on_tick_per_high_resolution(store: Store, time_into_slot_ms: uint64) -> None:
    """[New in Simplex] Process exact in-slot TSQ boundaries."""
    assert time_into_slot_ms < SLOT_DURATION_MS
    slot = get_current_slot(store)
    round = compute_round_at_slot(slot)
    start_slot = compute_start_slot_at_round(round)
    last_slot = Slot(start_slot + get_slots_per_round_at_slot(start_slot) - 1)

    if time_into_slot_ms == get_available_confirmation_due_ms():
        freeze_available_votes(store, slot)
        update_confirmation_heads(store)
    if time_into_slot_ms == get_view_freeze_due_ms():
        store.view_freeze_slots.add(slot)
        if slot == last_slot:
            freeze_tsq_view(store)
    if slot == start_slot and time_into_slot_ms == get_slot_component_duration_ms(
        AVAILABLE_ATTESTATION_DUE_BPS
    ):
        freeze_stable_root(store)
```

### Modified `on_block`

*Note*: During live activation, `upgrade_forkchoice_store_to_simplex` has
already advanced every branch state through the empty-slot processing for the
fork slot before applying the state upgrade. If a block is proposed in that same
slot, only block processing remains. The boundary branch below performs the
signature and post-state-root checks normally supplied by `state_transition`;
every later block follows the ordinary transition path.

```python
def on_block(store: Store, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    assert block.parent_root in store.block_states

    # [Modified in Simplex]
    # gloas payloads model: never cache a post-payload state. If the parent is
    # full, only assert the parent envelope was locally verified; the parent's
    # executed payload is folded into this child during its own state transition
    # (process_parent_execution_payload). Always copy the parent's block state.
    if is_parent_node_full(store, block):
        assert is_payload_verified(store, block.parent_root)
    state = copy(store.block_states[block.parent_root])

    current_slot = get_current_slot(store)
    assert current_slot >= block.slot

    # (a) Assert block descends from finalized: slot guard rejects equivocating forks,
    #     ancestry check enforces linear descent from the current finalized root.
    assert block.slot > store.finalized_checkpoint.slot
    assert (
        store.finalized_checkpoint.root
        == get_ancestor(
            store,
            ForkChoiceNode(root=block.parent_root, payload_status=PAYLOAD_STATUS_PENDING),
            store.finalized_checkpoint.slot,
        ).root
    )

    # (b) State transition
    block_root = hash_tree_root(block)
    if state.slot == block.slot:
        # The only valid equal-slot pre-state is the already slot-processed
        # live fork-transition state for the first Simplex slot.
        assert block.slot == compute_start_slot_at_epoch(SIMPLEX_FORK_EPOCH)
        assert state.fork.current_version == SIMPLEX_FORK_VERSION
        assert state.fork.epoch == SIMPLEX_FORK_EPOCH
        assert state.latest_block_header.slot < block.slot
        assert verify_block_signature(state, signed_block)
        process_block(state, block)
        assert block.state_root == hash_tree_root(state)
    else:
        state_transition(state, signed_block, True)  # noqa: FBT003

    store.blocks[block_root] = block
    store.block_states[block_root] = state

    # Replay block-carried sibling votes that were waiting for this head. The
    # from-block handlers are skip-only, so a vote that does not verify under
    # the newly available head branch has no effect and cannot invalidate this
    # block.
    for pending_attestation in store.pending_attestations.pop(block_root, []):
        on_attestation(store, pending_attestation, is_from_block=True)
    for pending_attestation in store.pending_available_attestations.pop(block_root, []):
        on_available_attestation(store, pending_attestation, is_from_block=True)

    # [Modified in Simplex]
    # on_block is the sole owner of block-carried fork-choice effects. The
    # state transition already verified each E1/E2 slashing; mirror its signer
    # attribution without asking runners to redeliver the operation.
    for attester_slashing in block.body.attester_slashings:
        on_attester_slashing(store, attester_slashing, is_from_block=True)

    notify_ptc_messages(store, state, block.body.payload_attestations)

    # [Modified in Simplex]
    # Process finality attestations, delivering their signed head votes to the
    # same latest-message store used by the gossip path.
    for attestation in block.body.attestations:
        on_attestation(store, attestation, is_from_block=True)

    # [Modified in Simplex]
    # Process available attestations (per-slot Goldfish tracking)
    for available_attestation in block.body.available_attestations:
        on_available_attestation(store, available_attestation, is_from_block=True)

    # [New in Simplex]
    # Consensus processing has verified this evidence and its signatures.
    # Mirror the signer attribution into the fork-choice equivocation views.
    for evidence in block.body.round_double_vote_evidence:
        on_round_double_vote_evidence(store, evidence, is_from_block=True)

    # [New in Simplex]
    # Bump h_max so the viability guard sees the new maximum
    # before update_finalized evaluates it.
    store.h_max = max(store.h_max, state.current_height)

    # [New in Simplex]
    # Single justification cert event: lex-max on
    # ``(h_j, hash_tree_root(J))`` with the F-filter.
    update_justified(store, state.justified_checkpoint, state.justified_height)

    # [New in Simplex]
    # Advance Σ.F if the block's finalized checkpoint improves on the stored
    # one, is an ancestor of or equal to Σ.J, and lies in the viable subtree.
    # Height 0 is the pre-Simplex initialization sentinel, not a Simplex
    # finalization event. In particular, do not replace the exact proposal slot
    # recovered by live Store migration with a legacy epoch-boundary slot.
    if state.finalized_height > Height(0):
        update_finalized(store, state.finalized_checkpoint)

    # [New in Simplex]
    # Record every round-start proposal root. The common action computes its
    # TSQ lock independently, then may distinguish one proposal that descends
    # from that lock.
    update_round_proposals(store, block_root)
```

### Modified `on_execution_payload_envelope`

Live activation can race an execution payload envelope for a Gloas block. The
ordinary Simplex `block_states` entry has already been advanced and upgraded at
that point, so it no longer reconstructs the historical block root or supplies
the historical slot and signature domain. For such an unresolved root, verify
against the immutable Gloas post-block state retained by
`upgrade_forkchoice_store_to_simplex`. New Simplex-era block roots continue to
use their ordinary post-block state. The retained legacy state is local
verification metadata and is released once the envelope succeeds.

```python
def on_execution_payload_envelope(
    store: Store, signed_envelope: SignedExecutionPayloadEnvelope
) -> None:
    """
    Run ``on_execution_payload_envelope`` upon receiving a new execution payload envelope.
    """
    envelope = signed_envelope.message
    root = envelope.beacon_block_root

    # The corresponding beacon block root needs to be known.
    assert root in store.block_states

    # Check if blob data is available. If not, this payload MAY be queued and
    # subsequently considered when blob data becomes available.
    assert is_data_available(root)

    # [Modified in Simplex]
    # A pre-fork envelope delivered after live migration must be checked in its
    # exact historical Gloas post-block context, not the upgraded fork-slot
    # branch state stored for processing Simplex descendants.
    if root in store.legacy_payload_verification_states:
        state = store.legacy_payload_verification_states[root]
        gloas.verify_execution_payload_envelope(state, signed_envelope, EXECUTION_ENGINE)
    else:
        state = store.block_states[root]
        verify_execution_payload_envelope(state, signed_envelope, EXECUTION_ENGINE)

    store.payloads[root] = envelope
    store.legacy_payload_verification_states.pop(root, None)
```

### Modified `on_attester_slashing`

Block-carried E1/E2 slashings have already been fully validated by the state
transition. Their fork-choice effect is signer attribution only, so the
from-block path mirrors the intersection without a second signature check. A
standalone delivery retains the inherited structural and signature validation,
but performs it against the current fork-choice head state rather than the
justified state. During a finality stall the justified registry can predate an
otherwise valid attester, while the head state is the same registry view used by
gossip validation. In either path, Simplex activation guards prevent evidence
from manufacturing a pre-fork duty.

```python
def on_attester_slashing(
    store: Store,
    attester_slashing: AttesterSlashing,
    is_from_block: bool = False,
) -> None:
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2
    if is_from_block:
        # The state transition already validated both signatures and indices.
        # Any post-fork state is sufficient for the activation guards below.
        state = store.block_states[store.justified_checkpoint.root]
    else:
        # Match the registry view used by standalone gossip validation. Using
        # the justified state is not total during a long finality stall: a
        # validator may have joined the head registry after that checkpoint.
        head = get_head(store)
        state = store.block_states[head.root]
    assert is_attestation_from_active_simplex_fork(state, attestation_1.data)
    assert is_attestation_from_active_simplex_fork(state, attestation_2.data)
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data)

    if not is_from_block:
        assert all(index < len(state.validators) for index in attestation_1.attesting_indices)
        assert all(index < len(state.validators) for index in attestation_2.attesting_indices)
        assert is_valid_indexed_attestation(state, attestation_1)
        assert is_valid_indexed_attestation(state, attestation_2)

    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    assert len(indices) > 0
    store.equivocating_indices.update(indices)
    round_1 = compute_round_at_slot(attestation_1.data.slot)
    round_2 = compute_round_at_slot(attestation_2.data.slot)
    if round_1 == round_2:
        store.round_equivocating_indices.setdefault(round_1, set()).update(indices)
```

### Modified `on_payload_attestation_message`

*Note*: Payload votes use first-vote + equivocation tracking with view-merge
freeze handling. Non-proposer wire votes after freeze are ignored; the next
proposer may continue collecting via `is_next_proposer=True`.

*Note*: The from-block PTC path (`is_from_block=True`) attributes votes and
equivocation marks without re-verifying the signature, the same
unverified-attribution pattern that `is_valid_from_block_available_attestation`
closes on the available path. Here the vote is attributed to a single
`validator_index` against the PTC resolved on the block's own post-state
(`store.block_states[data.beacon_block_root]`), not the including chain, so the
committee is already head-chain-resolved; only the signature is unchecked. This
handler is inherited from gloas and its from-block hardening is a base-spec
concern (the same gap exists in gloas, unmodified here), so it is flagged for
the base spec rather than fixed on this branch.

```python
def on_payload_attestation_message(
    store: Store,
    ptc_message: PayloadAttestationMessage,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    data = ptc_message.data
    assert data.beacon_block_root in store.block_states

    state = store.block_states[data.beacon_block_root]
    ptc = get_ptc(state, data.slot)

    # PTC votes can only affect the block slot they are assigned to. At the
    # live activation boundary, a retained Gloas block at A - 1 has a migrated
    # branch state already advanced to A, so comparing against ``state.slot``
    # would silently discard its still-includable aggregate from the activation
    # block.
    if data.slot != store.blocks[data.beacon_block_root].slot:
        return
    assert ptc_message.validator_index in ptc

    if not is_from_block:
        # [Modified in Simplex]
        # Wire votes accepted only for the current slot,
        # with view-freeze gating (proposer may override via ``is_next_proposer``).
        assert data.slot == get_current_slot(store)
        if not is_next_proposer and not is_before_view_freeze_deadline(store):
            return
        assert is_valid_indexed_payload_attestation(
            state,
            IndexedPayloadAttestation(
                attesting_indices=[ptc_message.validator_index],
                data=data,
                signature=ptc_message.signature,
            ),
        )

    vote_slot = data.slot
    if vote_slot not in store.payload_votes:
        return

    payload_votes = store.payload_votes[vote_slot]
    equivocations = store.payload_vote_equivocations[vote_slot]
    validator_index = ptc_message.validator_index
    # [Modified in Simplex]
    # Votes are keyed by validator identity: equivocation is same validator +
    # same slot + different data, regardless of committee position or branch
    # family. Seat multiplicity (a validator may hold multiple PTC seats under
    # balance-weighted selection) is resolved at read time against the slot's
    # PTC committee.
    if validator_index in equivocations:
        return
    if validator_index not in payload_votes:
        payload_votes[validator_index] = data
    elif payload_votes[validator_index] != data:
        equivocations.add(validator_index)
```

### Modified `on_attestation`

*Note*: Every valid finality attestation received over gossip enters the grade
inputs immediately, including in its own slot, and is forwarded immediately. A
block-included attestation is another delivery path for the same signed vote:
after skip-only validation and signature verification on the attestation's head
chain, it calls `update_latest_messages`. Inclusion creates no separate vote
object and has no special grade semantics. An included vote whose head is
unknown locally is retained in the bounded pending map and replayed if that head
arrives before the latest-message window closes.

```python
def on_attestation(store: Store, attestation: Attestation, is_from_block: bool = False) -> None:
    """[Modified in Simplex]"""
    data = attestation.data
    if is_from_block:
        # [New in Simplex]
        # Skip-only delivery path: an unknown head is queued for bounded replay;
        # failed validation has no fork-choice effect and never changes
        # acceptance of the including block.
        if data.beacon_block_root not in store.blocks:
            store.pending_attestations.setdefault(data.beacon_block_root, []).append(attestation)
            return
        if not is_valid_from_block_attestation(store, attestation):
            return
        # The committee is resolved, and the aggregate verified, on the
        # attestation's own head chain, so these indices are actual signers.
        target_state = get_attestation_checkpoint_state(store, data)
        attesting_indices = get_attesting_indices(target_state, attestation)
        update_latest_messages(store, sorted(attesting_indices), attestation)
        return

    validate_on_attestation(store, attestation)

    # Derive the checkpoint state for signature verification and attesting
    # indices (epoch boundary on the attestation's own head chain).
    target_state = get_attestation_checkpoint_state(store, data)

    # Verify signature against beacon committee
    assert is_valid_indexed_attestation(
        target_state, get_indexed_attestation(target_state, attestation)
    )

    attesting_indices = get_attesting_indices(target_state, attestation)
    apply_attestation_latest_messages(store, sorted(attesting_indices), attestation)
```

### New `on_available_attestation`

*Note*: Available attestations track per-slot per-committee-member votes for the
Goldfish fork choice layer. Non-proposers ignore wire votes after freeze;
proposers may continue collecting via `is_next_proposer=True`.

```python
def on_available_attestation(
    store: Store,
    attestation: AvailableAttestation,
    is_from_block: bool = False,
    is_next_proposer: bool = False,
) -> None:
    """[New in Simplex]"""
    # Retain a block-carried sibling vote until its head arrives. The pending
    # map is pruned after the vote's one-slot Goldfish use has passed.
    if is_from_block and attestation.data.beacon_block_root not in store.blocks:
        store.pending_available_attestations.setdefault(
            attestation.data.beacon_block_root, []
        ).append(attestation)
        return

    if not is_from_block and not is_next_proposer and not is_before_view_freeze_deadline(store):
        # Late wire vote: ignored for view-merge.
        return

    validate_on_available_attestation(store, attestation, is_from_block)

    # [New in Simplex]
    # State-free well-formedness prechecks BEFORE deriving the head-chain
    # checkpoint state, mirroring the finality path (is_valid_from_block_attestation
    # runs its head-slot shield before get_attestation_checkpoint_state). The
    # head-slot precheck is the shield: without it a from-block attestation whose
    # named head is later than the attestation would reach the epoch-boundary
    # get_ancestor walk below, which KeyErrors on a checkpoint-synced store when
    # the boundary is below the anchor. On the skip-only from-block path that
    # must never raise, so the state is never derived for an attestation that
    # will be skipped.
    if is_from_block and not is_valid_from_block_available_attestation_precheck(store, attestation):
        return

    # Derive the head-chain duty-epoch state for signature verification and
    # committee positions. Processing forward (rather than walking to an epoch
    # boundary below the trusted anchor) keeps checkpoint-sync handling total.
    attestation_epoch = compute_epoch_at_slot(attestation.data.slot)
    epoch_boundary_slot = compute_start_slot_at_epoch(attestation_epoch)
    checkpoint_key = (attestation.data.beacon_block_root, attestation_epoch)
    if checkpoint_key not in store.checkpoint_states:
        base_state = copy(store.block_states[attestation.data.beacon_block_root])
        if base_state.slot < epoch_boundary_slot:
            process_slots(base_state, epoch_boundary_slot)
        store.checkpoint_states[checkpoint_key] = base_state

    target_state = store.checkpoint_states[checkpoint_key]

    if is_from_block:
        # [New in Simplex]
        # From-block available attestations were signature-verified only under
        # the including chain during ``process_available_attestation``; on
        # RANDAO-diverged forks the head-chain committee resolved here can
        # disagree, so re-verify the aggregate under ``target_state`` before
        # attributing votes/equivocations. Skip-only (checked, never asserted):
        # a failure drops the attestation's effects but never rejects the block
        # (attribution runs after the block is already in the store).
        if not is_valid_from_block_available_attestation(target_state, attestation):
            return
    else:
        # Verify signature against available committee
        attesting_indices = get_available_attesting_indices(target_state, attestation)
        pubkeys = [target_state.validators[i].pubkey for i in sorted(attesting_indices)]
        domain = get_domain(target_state, DOMAIN_AVAILABLE_ATTESTER, attestation_epoch)
        signing_root = compute_signing_root(attestation.data, domain)
        assert bls.FastAggregateVerify(pubkeys, signing_root, attestation.signature)

    # Store individual votes for Goldfish tracking.
    vote_slot = attestation.data.slot
    if vote_slot not in store.available_votes:
        return
    available_votes = store.available_votes[vote_slot]
    available_vote_equivocations = store.available_vote_equivocations[vote_slot]
    available_timely_attesters = store.available_timely_attesters[vote_slot]
    available_timely_equivocations = store.available_timely_equivocations[vote_slot]
    current_slot = get_current_slot(store)

    # [Modified in Simplex]
    # Votes are keyed by validator identity: equivocation is same validator +
    # same slot + different data, regardless of committee position or branch
    # family. Seat multiplicity is resolved at read time against the slot's
    # available committee.
    for member_index in get_available_attesting_indices(target_state, attestation):
        # Ignore further votes once the member has equivocated.
        if member_index in available_vote_equivocations:
            continue
        if member_index not in available_votes:
            # First vote from this committee member for this slot
            available_votes[member_index] = attestation.data
            if (
                vote_slot == current_slot
                and not is_from_block
                and is_at_or_before_available_confirmation_deadline(store)
            ):
                available_timely_attesters.add(member_index)
        elif available_votes[member_index] != attestation.data:
            # Second (different) vote — record as equivocation
            available_vote_equivocations.add(member_index)
            # Only an equivocation fully observed before the confirmation
            # deadline excludes the member from that deadline's snapshot. A
            # later conflicting copy still affects live Goldfish, but cannot
            # retroactively contaminate the exact deadline snapshot.
            if vote_slot == current_slot and is_at_or_before_available_confirmation_deadline(store):
                available_timely_equivocations.add(member_index)
```

### New `on_round_double_vote_evidence`

*Note*: Round-double-vote evidence affects both TSQ eligibility (the signer is
excluded from that round's support) and SG grade eligibility (the global known
equivocator set). Block-carried evidence has already been fully verified by the
state transition. Wire evidence is signature-checked against the checkpoint
state of each attestation's own head branch before any signer is attributed.

```python
def on_round_double_vote_evidence(
    store: Store,
    evidence: RoundDoubleVoteEvidence,
    is_from_block: bool = False,
) -> None:
    attestation_1 = evidence.attestation_1
    attestation_2 = evidence.attestation_2
    # Evidence is indefinitely valid only for messages from the Simplex era.
    assert is_attestation_from_store_active_simplex_fork(store, attestation_1.data)
    assert is_attestation_from_store_active_simplex_fork(store, attestation_2.data)
    round = compute_round_at_slot(attestation_1.data.slot)
    if is_from_block:
        # The state transition asserted these conditions.
        assert round == compute_round_at_slot(attestation_2.data.slot)
        assert attestation_1.data != attestation_2.data
    else:
        assert round == compute_round_at_slot(attestation_2.data.slot)
        assert attestation_1.data != attestation_2.data
        # Apply the ordinary finality-message ingress window before any copied
        # head state is processed toward the duty epoch. Without this bound,
        # valid signatures for an arbitrarily future round could force an
        # unbounded process_slots loop and retain unbounded future-round keys.
        current_slot = get_current_slot(store)
        for attestation in (attestation_1, attestation_2):
            assert attestation.data.slot <= current_slot
            assert attestation.data.slot + LATEST_MESSAGE_EXPIRY_SLOTS > current_slot
        assert attestation_1.data.beacon_block_root in store.block_states
        assert attestation_2.data.beacon_block_root in store.block_states
        # Apply the same temporal shield as ordinary wire attestations before
        # deriving a duty-epoch state. Invalid wire evidence is rejected
        # without walking below a checkpoint-sync anchor.
        assert store.blocks[attestation_1.data.beacon_block_root].slot <= attestation_1.data.slot
        assert store.blocks[attestation_2.data.beacon_block_root].slot <= attestation_2.data.slot
        state_1 = get_attestation_checkpoint_state(store, attestation_1.data)
        state_2 = get_attestation_checkpoint_state(store, attestation_2.data)
        assert is_valid_indexed_attestation(state_1, attestation_1)
        assert is_valid_indexed_attestation(state_2, attestation_2)

    offenders = set(attestation_1.attesting_indices) & set(attestation_2.attesting_indices)
    assert len(offenders) > 0
    store.round_equivocating_indices.setdefault(round, set()).update(offenders)
    store.equivocating_indices.update(offenders)
```

## Removed inherited mechanisms

Simplex removes the inherited unrealized-FFG pull-up, proposer boost/reorg,
block-timeliness, and target-epoch validation paths. Accordingly, the Simplex
spec builder excludes their containers and functions rather than exposing
consensus no-ops. In particular, clients MUST NOT carry forward the inherited
`get_voting_source`, `update_unrealized_checkpoints`, `compute_pulled_up_tip`,
`record_block_timeliness`, `update_proposer_boost_root`, proposer-boost/reorg
predicates, `is_finalization_ok`, or
`validate_target_epoch_against_current_time`.

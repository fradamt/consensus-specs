"""
Gloas FCR: the empty-slot support discount respects the parent payload status.

Two correct observer stores, ``source`` and ``receiver``, receive the same
messages from correct validators. Faulty validators (exactly two of the eight
members of every slot committee, i.e. ``CONFIRMATION_BYZANTINE_THRESHOLD``
percent of every committee interval) send different messages to the two
stores. Each correct validator belongs to one of two groups for the whole run
and uses the store of that group as its own view: group ``S`` uses ``source``
and group ``R`` uses ``receiver``. Every correct attestation is for the signing
validator's own ``get_head`` at signing time, with ``data.index`` set by the
Gloas validator rule. All blocks, envelopes and attestations are real, built
and signed with the test helpers and processed by the unchanged fork-choice
handlers.

Schedule (minimal preset, epoch 0 only):

- slot 1: correct proposer publishes ``P``. The envelope of ``P`` reaches
  ``source`` at the start of slot 2 and ``receiver`` in slot 2 after the
  attestation deadline (the end-of-slot relay used by the G3 model).
- slots 2, 3: no block (faulty proposers stay silent). ``S`` members vote
  ``P`` FULL, ``R`` members vote ``P`` EMPTY. Faulty members vote ``P`` FULL
  to ``source`` and ``P`` EMPTY to ``receiver``.
- slot 4: correct ``S`` proposer publishes ``c`` on ``P`` FULL.
- slots 4, 5: correct members are ``S`` and vote ``c``. Faulty members vote
  ``c`` to ``source`` and ``P`` EMPTY to ``receiver``.
- slot 6: at the slot start the source FCR does not confirm ``c``. The faulty slot-6
  proposer publishes ``B`` on ``P`` EMPTY. ``B`` is timely at ``receiver`` and
  late at ``source``. The receiver head is ``B``. Correct slot-6 members are
  ``R`` and vote ``B``.
- slot 7: the receiver head is still ``B``, which excludes the unconfirmed ``c``.

The modified discount counts duty-fresh parent votes whose payload status matches
``get_parent_payload_status(c)`` or is PENDING. With the same message schedule
the source FCR does not confirm ``c``.
"""

from eth_consensus_specs.test.context import (
    always_bls,
    MINIMAL,
    spec_state_test,
    with_gloas_and_later,
    with_presets,
)
from eth_consensus_specs.test.helpers.attestations import (
    get_valid_attestation_at_slot,
)
from eth_consensus_specs.test.helpers.block import (
    build_block_and_payload,
)
from eth_consensus_specs.test.helpers.execution_payload import (
    compute_and_sign_execution_payload_envelope,
)
from eth_consensus_specs.test.helpers.fork_choice import (
    get_genesis_forkchoice_store_and_block,
)
from eth_consensus_specs.test.helpers.genesis import (
    get_sample_genesis_execution_payload,
)
from eth_consensus_specs.test.helpers.state import (
    state_transition_and_sign_block,
)
from eth_consensus_specs.utils import bls

P_SLOT = 1
SKIPPED_SLOTS = (2, 3)
C_SLOT = 4
B_SLOT = 6
LAST_SLOT = 7
FAULTY_PER_COMMITTEE = 2


def _status_name(spec, status):
    return {
        spec.PAYLOAD_STATUS_EMPTY: "EMPTY",
        spec.PAYLOAD_STATUS_FULL: "FULL",
        spec.PAYLOAD_STATUS_PENDING: "PENDING",
    }[status]


def _slot_committee(spec, state, slot):
    members = set()
    count = spec.get_committee_count_per_slot(state, spec.compute_epoch_at_slot(slot))
    for index in range(count):
        members.update(int(i) for i in spec.get_beacon_committee(state, slot, index))
    return members


def _proposer(spec, state, slot):
    st = state.copy()
    spec.process_slots(st, slot)
    return int(spec.get_beacon_proposer_index(st))


class _Run:
    def __init__(self, spec, genesis_state, log):
        self.spec = spec
        self.genesis_state = genesis_state
        self.log = log
        self.stores = {}
        for name in ("source", "receiver"):
            store, _ = get_genesis_forkchoice_store_and_block(spec, genesis_state)
            self.stores[name] = store
        self.fcr = spec.get_fast_confirmation_store(self.stores["source"])
        self.pending = {"source": [], "receiver": []}
        self.names = {}
        self.honest_votes = 0

    # Time -----------------------------------------------------------------
    def time_at(self, slot, seconds=0):
        return (
            self.genesis_state.genesis_time
            + slot * self.spec.config.SLOT_DURATION_MS // 1000
            + seconds
        )

    def tick(self, slot, seconds=0):
        for store in self.stores.values():
            self.spec.on_tick(store, self.time_at(slot, seconds))

    # Blocks and envelopes -------------------------------------------------
    def build(self, parent_root, slot, *, parent_full, graffiti):
        spec = self.spec
        src = self.stores["source"]
        parent_state = src.block_states[parent_root].copy()
        if parent_full:
            parent_payload = src.payloads[parent_root].payload
        else:
            parent_payload = get_sample_genesis_execution_payload(
                spec, eth1_block_hash=self.genesis_state.eth1_data.block_hash
            )
        requests = spec.ExecutionRequests()
        block, payload = build_block_and_payload(
            spec, parent_state, slot, parent_payload=parent_payload, execution_requests=requests
        )
        block.body.graffiti = graffiti.encode().ljust(32, b"\x00")
        if parent_full:
            block.body.parent_execution_requests = src.payloads[parent_root].execution_requests
        signed_block = state_transition_and_sign_block(spec, parent_state, block)
        root = spec.hash_tree_root(block)
        envelope = compute_and_sign_execution_payload_envelope(
            spec, parent_state, root, signed_block, payload, requests
        )
        self.names[root] = graffiti
        return root, signed_block, envelope

    def deliver_block(self, name, signed_block):
        self.spec.on_block(self.stores[name], signed_block)

    def deliver_envelope(self, name, envelope):
        self.spec.on_execution_payload_envelope(self.stores[name], envelope)
        assert self.spec.is_payload_verified(self.stores[name], envelope.message.beacon_block_root)

    # Attestations ----------------------------------------------------------
    def attestation(self, slot, members, root, index):
        spec = self.spec
        st = self.stores["source"].block_states.get(root)
        if st is None:
            st = self.stores["receiver"].block_states[root]
        st = st.copy()
        if st.slot < slot:
            spec.process_slots(st, slot)
        members = set(members)
        return get_valid_attestation_at_slot(
            st,
            spec,
            slot,
            participation_fn=lambda _s, _i, committee: set(committee) & members,
            beacon_block_root=root,
            payload_index=index,
        )

    def head_vote(self, name, slot):
        """Vote of a correct validator whose view is store ``name`` (Gloas validator rule)."""
        spec = self.spec
        store = self.stores[name]
        head = spec.get_head(store)
        block = store.blocks[head.root]
        if block.slot == slot:
            index = 0
        else:
            index = 1 if head.payload_status == spec.PAYLOAD_STATUS_FULL else 0
        return head.root, index

    def honest_attest(self, name, slot, members):
        if not members:
            return None
        root, index = self.head_vote(name, slot)
        att = self.attestation(slot, members, root, index)
        for target in self.pending.values():
            target.append(att)
        self.honest_votes += len(members)
        return root, index

    def faulty_attest(self, slot, members, per_store):
        for name, (root, index) in per_store.items():
            self.pending[name].append(self.attestation(slot, members, root, index))

    def apply_pending(self):
        for name, atts in self.pending.items():
            for att in atts:
                self.spec.on_attestation(self.stores[name], att, is_from_block=False)
            self.pending[name] = []

    def label(self, root):
        return self.names.get(root, "genesis")


def _run_scenario(spec, state):
    # Signatures of blocks, envelopes and attestations are verified.
    assert bls.bls_active
    log = []
    run = _Run(spec, state, log)
    src, rcv = run.stores["source"], run.stores["receiver"]
    genesis_root = src.finalized_checkpoint.root
    run.names[genesis_root] = "genesis"

    # Committees, proposers, faulty set and correct groups -------------------
    committees = {s: _slot_committee(spec, state, s) for s in range(LAST_SLOT + 1)}
    proposers = {s: _proposer(spec, state, s) for s in range(1, LAST_SLOT + 1)}
    silent_or_faulty = {proposers[s] for s in (*SKIPPED_SLOTS, 5, B_SLOT, LAST_SLOT)}
    c_proposer = proposers[C_SLOT]
    assert c_proposer not in silent_or_faulty
    faulty = set()
    for s in range(LAST_SLOT + 1):
        committee = sorted(committees[s])
        chosen = [v for v in committee if v in silent_or_faulty]
        assert len(chosen) <= FAULTY_PER_COMMITTEE, (s, chosen)
        for v in committee:
            if len(chosen) == FAULTY_PER_COMMITTEE:
                break
            if v not in chosen and v != c_proposer:
                chosen.append(v)
        faulty.update(chosen)
    all_validators = set(range(len(state.validators)))
    assert set().union(*committees.values()) == all_validators
    for s in range(LAST_SLOT + 1):
        assert len(committees[s] & faulty) == FAULTY_PER_COMMITTEE
        assert len(committees[s]) == 8
    # Faulty stake is at most CONFIRMATION_BYZANTINE_THRESHOLD percent of
    # every committee interval (committees are disjoint in the epoch).
    for lo in range(LAST_SLOT + 1):
        for hi in range(lo, LAST_SLOT + 1):
            members = set().union(*(committees[s] for s in range(lo, hi + 1)))
            assert 100 * len(
                members & faulty
            ) <= spec.config.CONFIRMATION_BYZANTINE_THRESHOLD * len(members)
    honest = all_validators - faulty
    group_r = set()
    for s in SKIPPED_SLOTS:
        candidates = sorted(committees[s] & honest - {c_proposer})
        group_r.update(candidates[:4])
    group_r.update(committees[B_SLOT] & honest - {c_proposer})
    group_s = honest - group_r

    def hon(slot, group):
        return committees[slot] & group

    def byz(slot):
        return committees[slot] & faulty

    balance = spec.get_current_balance_source(run.fcr)
    unit = int(spec.get_total_active_balance(balance)) // int(spec.SLOTS_PER_EPOCH) // 8
    c_root = None
    p_root = None
    records = {}

    def record(slot, phase):
        p_full = spec.ForkChoiceNode(root=p_root, payload_status=spec.PAYLOAD_STATUS_FULL)
        p_empty = spec.ForkChoiceNode(root=p_root, payload_status=spec.PAYLOAD_STATUS_EMPTY)
        path_status = spec.get_ancestor(rcv, spec.get_head(rcv), spec.Slot(P_SLOT)).payload_status
        item = {
            "slot": slot,
            "phase": phase,
            "confirmed": run.label(run.fcr.confirmed_root),
            "source_head": run.label(spec.get_head(src).root),
            "receiver_head": run.label(spec.get_head(rcv).root),
            "receiver_path_at_P": _status_name(spec, path_status),
            "rcv_full": int(spec.get_weight(rcv, p_full)) / unit,
            "rcv_empty": int(spec.get_weight(rcv, p_empty)) / unit,
            "src_full": int(spec.get_weight(src, p_full)) / unit,
            "src_empty": int(spec.get_weight(src, p_empty)) / unit,
        }
        if c_root is not None:
            item.update(
                discount=int(spec.get_support_discount(src, balance, c_root)) / unit,
                support=int(
                    spec.get_attestation_score(src, spec.get_node_for_root(c_root), balance)
                )
                / unit,
                threshold=int(spec.compute_safety_threshold(src, c_root, balance)) / unit,
                receiver_descends_from_c=bool(
                    spec.is_ancestor(rcv, spec.get_head(rcv), spec.get_node_for_root(c_root))
                ),
            )
        records[(slot, phase)] = item
        log.append(item)
        return item

    # Slot 0 --------------------------------------------------------------
    run.tick(0)
    spec.on_fast_confirmation(run.fcr)
    run.tick(0, 1)
    run.honest_attest("source", 0, hon(0, honest))
    run.faulty_attest(0, byz(0), {"source": (genesis_root, 0), "receiver": (genesis_root, 0)})

    # Slot 1: P ----------------------------------------------------------
    run.tick(1)
    run.apply_pending()
    spec.on_fast_confirmation(run.fcr)
    p_root, p_block, p_envelope = run.build(genesis_root, P_SLOT, parent_full=False, graffiti="P")
    run.deliver_block("source", p_block)
    run.deliver_block("receiver", p_block)
    assert spec.get_parent_payload_status(src, p_block.message) == spec.PAYLOAD_STATUS_EMPTY
    run.tick(1, 1)
    assert run.honest_attest("source", 1, hon(1, honest)) == (p_root, 0)
    run.faulty_attest(1, byz(1), {"source": (p_root, 0), "receiver": (p_root, 0)})

    # Slots 2, 3: skipped ------------------------------------------------
    for slot in SKIPPED_SLOTS:
        run.tick(slot)
        run.apply_pending()
        spec.on_fast_confirmation(run.fcr)
        if slot == SKIPPED_SLOTS[0]:
            # Envelope of P: source at slot start, before the deadline.
            run.deliver_envelope("source", p_envelope)
        run.tick(slot, 1)  # attestation signing, before get_attestation_due_ms
        assert (run.time_at(slot, 1) - run.time_at(slot)) * 1000 < spec.get_attestation_due_ms()
        assert run.honest_attest("source", slot, hon(slot, group_s)) == (p_root, 1)
        assert run.honest_attest("receiver", slot, hon(slot, group_r)) == (p_root, 0)
        run.faulty_attest(slot, byz(slot), {"source": (p_root, 1), "receiver": (p_root, 0)})
        if slot == SKIPPED_SLOTS[0]:
            # Envelope of P: receiver after the deadline, before the slot ends.
            run.tick(slot, 5)
            run.deliver_envelope("receiver", p_envelope)
        record(slot, "late")

    # Slot 4: c on P FULL ------------------------------------------------
    run.tick(C_SLOT)
    run.apply_pending()
    spec.on_fast_confirmation(run.fcr)
    head = spec.get_head(src)
    assert head.root == p_root
    assert head.payload_status == spec.PAYLOAD_STATUS_FULL
    assert spec.should_build_on_full(src, head, spec.Slot(C_SLOT))
    c_root, c_block, c_envelope = run.build(p_root, C_SLOT, parent_full=True, graffiti="c")
    assert int(c_block.message.proposer_index) == c_proposer
    assert c_proposer in group_s
    run.deliver_block("source", c_block)
    run.deliver_block("receiver", c_block)
    assert spec.get_parent_payload_status(src, c_block.message) == spec.PAYLOAD_STATUS_FULL
    record(C_SLOT, "start")
    run.tick(C_SLOT, 1)
    assert run.honest_attest("source", C_SLOT, hon(C_SLOT, group_s)) == (c_root, 0)
    assert not hon(C_SLOT, group_r)
    run.faulty_attest(C_SLOT, byz(C_SLOT), {"source": (c_root, 0), "receiver": (p_root, 0)})
    run.tick(C_SLOT, 3)
    run.deliver_envelope("source", c_envelope)
    run.deliver_envelope("receiver", c_envelope)

    # Slot 5 -----------------------------------------------------------
    run.tick(5)
    run.apply_pending()
    spec.on_fast_confirmation(run.fcr)
    record(5, "start")
    run.tick(5, 1)
    assert run.honest_attest("source", 5, hon(5, group_s)) == (c_root, 1)
    assert not hon(5, group_r)
    run.faulty_attest(5, byz(5), {"source": (c_root, 1), "receiver": (p_root, 0)})

    # Slot 6: confirmation, then B on P EMPTY ----------------------------
    run.tick(B_SLOT)
    run.apply_pending()
    spec.on_fast_confirmation(run.fcr)
    record(B_SLOT, "start")
    b_root, b_block, b_envelope = run.build(p_root, B_SLOT, parent_full=False, graffiti="B")
    assert int(b_block.message.proposer_index) in faulty
    run.deliver_block("receiver", b_block)  # timely: proposer boost
    assert rcv.proposer_boost_root == b_root
    assert spec.get_parent_payload_status(rcv, b_block.message) == spec.PAYLOAD_STATUS_EMPTY
    record(B_SLOT, "after_B")
    run.tick(B_SLOT, 1)
    assert not hon(B_SLOT, group_s)
    assert run.honest_attest("receiver", B_SLOT, hon(B_SLOT, group_r)) == (b_root, 0)
    run.faulty_attest(B_SLOT, byz(B_SLOT), {"source": (c_root, 1), "receiver": (b_root, 0)})
    run.tick(B_SLOT, 3)
    run.deliver_block("source", b_block)  # late at source: no boost
    assert src.proposer_boost_root != b_root
    run.deliver_envelope("source", b_envelope)
    run.deliver_envelope("receiver", b_envelope)

    # Slot 7 -----------------------------------------------------------
    run.tick(LAST_SLOT)
    run.apply_pending()
    spec.on_fast_confirmation(run.fcr)
    record(LAST_SLOT, "start")

    meta = {
        "faulty": len(faulty),
        "validators": len(state.validators),
        "honest_votes": run.honest_votes,
        "group_r": len(group_r),
        "group_s": len(group_s),
        "proposers": {s: (v, "faulty" if v in faulty else "correct") for s, v in proposers.items()},
    }
    return records, log, meta


def _print(title, log, meta):
    print(
        f"\n{title}: validators={meta['validators']} faulty={meta['faulty']} "
        f"(per slot committee 2/8) honest_votes={meta['honest_votes']} unit=1 validator (32 ETH)"
    )
    print(f"  proposers={meta['proposers']} group_S={meta['group_s']} group_R={meta['group_r']}")
    for item in log:
        print("  " + " ".join(f"{k}={v}" for k, v in item.items()))


@with_gloas_and_later
@with_presets([MINIMAL], reason="committee sizes are fixed for the minimal preset")
@spec_state_test
@always_bls
def test_fcr_does_not_confirm_block_excluded_by_parent_empty_branch(spec, state):
    records, log, meta = _run_scenario(spec, state)
    _print("payload-aware discount", log, meta)
    for item in log:
        assert item["confirmed"] != "c", item
    confirmed = records[(B_SLOT, "start")]
    assert confirmed["support"] <= confirmed["threshold"]
    after_b = records[(B_SLOT, "after_B")]
    assert after_b["receiver_head"] == "B"
    assert not after_b["receiver_descends_from_c"]
    assert after_b["receiver_path_at_P"] == "EMPTY"
    final = records[(LAST_SLOT, "start")]
    assert final["confirmed"] != "c"
    assert final["receiver_head"] == "B"
    assert not final["receiver_descends_from_c"]
    assert final["receiver_path_at_P"] == "EMPTY"

# Weak-synchrony fast confirmation: test expectation changes

This branch replaces the strong fast-confirmation rule with the
weak-synchrony rule. The weak rule has two ingredients that the strong rule
does not have, and lacks two that the strong rule has:

- It requires a broadcast certificate on the confirmed carrier: completed-slot
  votes from the block's slot through the previous slot.
- It banks the observed justified checkpoint from the certified carrier, and
  only when that checkpoint's epoch is strictly newer than the banked one.
- It has no equivocation discount: slashed or equivocating validators are not
  removed from the adversarial budget.
- It has no empty-slot discount.

When this branch was created, 44 of the 182 altair-minimal fast-confirmation
tests failed. They were repaired under two policies, decided per test:

- Policy one (26 tests): the scenario is still valid under the weak rule, but
  the fixture lacked the certificate evidence that a real network would
  produce. The fixture now supplies that evidence and the original assertion
  is unchanged.
- Policy two (18 tests): the strong outcome is not the weak outcome. The
  expected value was changed to the weak rule's result. Every such change is
  a deliberate behavioural difference, listed below with its cause.

Classes: `certificate_only` (policy one), `discount_only` (weak rule declines
because the equivocation or empty-slot discount is absent),
`discount_and_certificate` (both strong ingredients absent; the store resets
to the finalized root or banks the first certified previous-epoch carrier),
`banking` (the weak rule banks the certified carrier instead of the strong
finality fallback).

Notable policy-two cases:

- `slashing_non_supporters_helps`: slashing validators who did not support
  the block no longer lowers the threshold. Adversarial weight is unchanged.
- `slashing_supporters_does_not_hurt`: with no discount, the block with
  slashed supporters is expected not to be confirmed.
- `restarts_to_gu_and_confirms_beyond_gu`: the strong restart-and-confirm
  root is declined; certified progress is retained.

## Table

| Test | Class | Policy | Fixture or expectation change | Why correct under weak FCR |
|---|---|---|---|---|
| current_epoch_12 | certificate_only | one | Builder supplies fresh head-UJ evidence. | The head now carries the broadcast certificate required by the weak guard; original strong assertion remains. |
| current_epoch_13 | certificate_only | one | Same fresh head-UJ evidence. | Same certified-carrier fact; original assertion remains. |
| current_epoch_14 | certificate_only | one | Same fresh head-UJ evidence. | Same certified-carrier fact; original assertion remains. |
| current_epoch_19 | certificate_only | one | Same fresh head-UJ evidence. | Same certified-carrier fact; original assertion remains. |
| no_restart_if_head_gu_is_stale | banking | two | Expect stale observed UJ and certified-carrier banking; retain no-restart finality assertion. | Weak banking carries the observed checkpoint on the carrier; the strong head/observed inequality is not a weak-rule invariant. |
| previous_epoch_012 | certificate_only | one | Builder supplies fresh VS/UJ evidence on previous head, block, and head. | The broadcast certificate is present on every relevant carrier; strong expectation remains. |
| previous_epoch_013 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_014 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_015 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_018 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_019 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_021 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_022 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_023 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_024 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_027 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_028 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_030 | discount_and_certificate | two | Expect weak reset to finalized root. | Both the certificate gate and discount are absent; the store banks finalized rather than the strong candidate. |
| previous_epoch_031 | discount_and_certificate | two | Expect the first previous-epoch certified carrier. | The weak carrier is the first canonical previous-epoch block, slot 16; strong latest-candidate banking is unavailable. |
| previous_epoch_032 | discount_and_certificate | two | Expect the first previous-epoch certified carrier. | Same store fact: certificate carrier exists, but the combined strong path does not. |
| previous_epoch_033 | discount_and_certificate | two | Expect the first previous-epoch certified carrier. | Same. |
| previous_epoch_034 | discount_and_certificate | two | Expect weak reset to finalized root. | Missing certificate and discount leave only finalized banking. |
| previous_epoch_035 | discount_and_certificate | two | Expect the first previous-epoch certified carrier. | The weak certified carrier is slot 16; latest strong candidate is not available. |
| previous_epoch_036 | discount_only | two | Set `is_one_confirmed=False`; retain the original scenario in the matrix. | The missing discount makes the weak rule decline; the store predicate is false. |
| previous_epoch_037 | discount_only | two | Set `is_one_confirmed=False`; retain the original scenario in the matrix. | Same missing-discount store fact. |
| previous_epoch_038 | discount_and_certificate | two | Expect weak reset to finalized root. | Neither strong ingredient is available. |
| previous_epoch_039 | discount_and_certificate | two | Expect the first previous-epoch certified carrier. | Weak banking stops at the first certified carrier. |
| previous_epoch_040 | discount_only | two | Set `is_one_confirmed=False`; retain the original scenario in the matrix. | The missing discount makes the weak rule decline. |
| previous_epoch_041 | certificate_only | one | Builder supplies fresh VS/UJ evidence on previous head, block, and head. | Broadcast certificate evidence makes the strong assertion sound. |
| previous_epoch_042 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_043 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_044 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_045 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_046 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_049 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_050 | certificate_only | one | Same four freshness facts. | Same. |
| previous_epoch_083 | certificate_only | one | Same four freshness facts, including the carrier route. | The carrier now has the real-network certificate evidence missing from the original fixture. |
| resets_when_bcand_not_descendant_of_gu_via_first_received_uj | banking | two | Expect certified descendant-carrier banking, not reset-to-finalized; assert the carrier certificate. | Weak banking follows the certified carrier, so the strong stale-GU non-descendant premise is no longer the selected store fact. |
| restarts_to_gu_and_confirms_beyond_gu | discount_only | two | Expect decline of the strong restart-and-confirm root while retaining certified progress. | Missing equivocation discount prevents the strong restart confirmation. |
| slashing_non_supporters_helps | discount_only | two | Expect adversarial weight unchanged and `is_one_confirmed` false. | Weak rule does not remove equivocator weight from the safety budget. |
| slashing_supporters_does_not_hurt | discount_only | two | Expect supporter-slashing block not to be confirmed. | Weak rule has no strong equivocation discount. |
| reconfirmation_fails_for_block_without_uj_checkpoint_in_chain | banking | two | Expect non-finalized certified-carrier banking and assert its broadcast certificate. | The weak rule banks the certified carrier instead of the strong finality fallback. |
| reconfirmation_passes_with_empty_slots_prior_first_block | discount_only | two | Expect no head confirmation at the two intermediate checks; retain eventual epoch-boundary confirmation. | Empty-slot discount is unavailable at those calls, while later fresh full-epoch support still confirms. |
| reset_to_finality_but_no_restart_to_gu_because_gu_too_old_epoch | certificate_only | one | Raise epoch-2 participation from 20% to 75% to add the stale GU attestation certificate; retain strict old-GU and finalized-root assertions. | Real attestations produce a stale GU carrier while still withholding finality advancement; no soundness gate is weakened. |

Source: orchestration lane z4 on 22 September 2026, altair-minimal
generator suite 182 passed, 9 skipped after the change; fulu and gloas
normal targets unchanged.

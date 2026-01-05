# Gloas Spec Development Notes

## BeaconAttestation payload_present field

`BeaconAttestationData` includes a `payload_present: boolean` field that indicates whether the attested block had its execution payload available.

### Validation

The `payload_present` field must be validated in both beacon-chain and fork-choice:

- **Same-slot attestations**: If the attestation is for a block proposed at the same slot (`is_block_from_slot` returns true), `payload_present` must be `False` (payload availability is not yet determined)
- **Previous-slot attestations**: `payload_present` must match `state.execution_payload_availability[slot_index]`

### Helper function

`is_block_from_slot(state, root, slot)` checks if a given block root corresponds to a block actually proposed at the given slot (not a skip slot). Used by both regular attestation processing and beacon attestation validation.

### Fork-choice integration

`get_beacon_attestation_score` takes a `ForkChoiceNode` and uses `is_supporting_vote` to properly account for payload status when counting beacon attestation votes. `BeaconAttestationData` is converted to `LatestMessage` inline for compatibility with `is_supporting_vote`.

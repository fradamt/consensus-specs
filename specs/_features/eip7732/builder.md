# EIP-7732 -- Honest Builder

*Note*: This document is a work-in-progress for researchers and implementers.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Builders attributions](#builders-attributions)
  - [Constructing the payload bid](#constructing-the-payload-bid)
  - [Constructing the `BlobSidecar`s](#constructing-the-blobsidecars)
  - [Constructing the execution payload envelope](#constructing-the-execution-payload-envelope)
  - [Honest payload withheld messages](#honest-payload-withheld-messages)

<!-- mdformat-toc end -->

## Introduction

This is an accompanying document which describes the expected actions of a
"builder" participating in the Ethereum proof-of-stake protocol.

With the EIP-7732 Fork, the protocol includes new staked participants of the
protocol called *Builders*. While Builders are a subset of the validator set,
they have extra attributions that are optional. Validators may opt to not be
builders and as such we collect the set of guidelines for those validators that
want to act as builders in this document.

## Builders attributions

Builders can submit bids to produce execution payloads. They can broadcast these
bids in the form of `SignedExecutionPayloadHeader` objects, these objects encode
a commitment to reveal an execution payload in exchange for a payment. When
their bids are chosen by the corresponding proposer, builders are expected to
broadcast an accompanying `SignedExecutionPayloadEnvelope` object honoring the
commitment.

Thus, builders tasks are divided in two, submitting bids, and submitting
payloads.

### Constructing the payload bid

Builders can broadcast a payload bid for the current or the next slot's proposer
to include. They produce a `SignedExecutionPayloadHeader` as follows.

01. Set `header.parent_block_hash` to the current head of the execution chain
    (this can be obtained from the beacon state as `state.last_block_hash`).
02. Set `header.parent_block_root` to be the head of the consensus chain (this
    can be obtained from the beacon state as
    `hash_tree_root(state.latest_block_header)`. The `parent_block_root` and
    `parent_block_hash` must be compatible, in the sense that they both should
    come from the same `state` by the method described in this and the previous
    point.
03. Construct an execution payload. This can be performed with an external
    execution engine with a call to `engine_getPayloadV4`.
04. Set `header.block_hash` to be the block hash of the constructed payload,
    that is `payload.block_hash`.
05. Set `header.gas_limit` to be the gas limit of the constructed payload, that
    is `payload.gas_limit`.
06. Set `header.builder_index` to be the validator index of the builder
    performing these actions.
07. Set `header.slot` to be the slot for which this bid is aimed. This slot
    **MUST** be either the current slot or the next slot.
08. Set `header.value` to be the value that the builder will pay the proposer if
    the bid is accepted. The builder **MUST** have enough balance to fulfill
    this bid and all pending payments.
09. Set `header.kzg_commitments_root` to be the `hash_tree_root` of the
    `blobsbundle.commitments` field returned by `engine_getPayloadV4`.
10. Set `header.fee_recipient` to be an execution address to receive the
    payment. This address can be obtained from the proposer directly via a
    request or can be set from the withdrawal credentials of the proposer. The
    burn address can be used as a fallback.

After building the `header`, the builder obtains a `signature` of the header by
using

```python
def get_execution_payload_header_signature(
    state: BeaconState, header: ExecutionPayloadHeader, privkey: int
) -> BLSSignature:
    domain = get_domain(state, DOMAIN_BEACON_BUILDER, compute_epoch_at_slot(header.slot))
    signing_root = compute_signing_root(header, domain)
    return bls.Sign(privkey, signing_root)
```

The builder assembles then
`signed_execution_payload_header = SignedExecutionPayloadHeader(message=header, signature=signature)`
and broadcasts it on the `execution_payload_header` global gossip topic.

### Constructing the `BlobSidecar`s

*[Modified in EIP7732]*

The `BlobSidecar` container is modified indirectly because the constant
`KZG_COMMITMENT_INCLUSION_PROOF_DEPTH` is modified. The function
`get_blob_sidecars` is modified because the KZG commitments are no longer
included in the beacon block but rather in the `ExecutionPayloadEnvelope`, the
builder has to send the commitments as parameters to this function.

```python
def get_blob_sidecars(
    signed_block: SignedBeaconBlock,
    blobs: Sequence[Blob],
    blob_kzg_commitments: List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK],
    blob_kzg_proofs: Sequence[KZGProof],
) -> Sequence[BlobSidecar]:
    block = signed_block.message
    block_header = BeaconBlockHeader(
        slot=block.slot,
        proposer_index=block.proposer_index,
        parent_root=block.parent_root,
        state_root=block.state_root,
        body_root=hash_tree_root(block.body),
    )
    signed_block_header = SignedBeaconBlockHeader(
        message=block_header, signature=signed_block.signature
    )
    sidecars: List[BlobSidecar] = []
    for index, blob in enumerate(blobs):
        proof = compute_merkle_proof(
            blob_kzg_commitments,
            get_generalized_index(List[KZGCommitment, MAX_BLOB_COMMITMENTS_PER_BLOCK], index),
        )
        proof += compute_merkle_proof(
            block.body,
            get_generalized_index(
                BeaconBlockBody,
                "signed_execution_payload_header",
                "message",
                "blob_kzg_commitments_root",
            ),
        )
        sidecars.append(
            BlobSidecar(
                index=index,
                blob=blob,
                kzg_commitment=blob_kzg_commitments[index],
                kzg_proof=blob_kzg_proofs[index],
                signed_block_header=signed_block_header,
                kzg_commitment_inclusion_proof=proof,
            )
        )
    return sidecars
```

### Constructing the execution payload envelope

When the proposer publishes a valid `SignedBeaconBlock` containing a signed
commitment by the builder, the builder is later expected to broadcast the
corresponding `SignedExecutionPayloadEnvelope` that fulfills this commitment.
See below for a special case of an *honestly withheld payload*.

To construct the `execution_payload_envelope` the builder must perform the
following steps. We alias `block` to be the corresponding beacon block and alias
`header` to be the committed `ExecutionPayloadHeader` in
`block.body.signed_execution_payload_header.message`.

1. Set the `payload` field to be the `ExecutionPayload` constructed when
   creating the corresponding bid. This payload **MUST** have the same block
   hash as `header.block_hash`.
2. Set the `execution_requests` field to be the `ExecutionRequests` associated
   with `payload`.
3. Set the `builder_index` field to be the validator index of the builder
   performing these steps. This field **MUST** be `header.builder_index`.
4. Set `beacon_block_root` to be `hash_tree_root(block)`.
5. Set `slot` to be `block.slot`.
6. Set `blob_kzg_commitments` to be the `commitments` field of the blobs bundle
   constructed when constructing the bid. This field **MUST** have a
   `hash_tree_root` equal to `header.blob_kzg_commitments_root`.

After setting these parameters, the builder should run
`process_execution_payload(state, signed_envelope, verify=False)` and this
function should not trigger an exception.

6. Set `state_root` to `hash_tree_root(state)`.

After preparing the `envelope` the builder should sign the envelope using:

```python
def get_execution_payload_envelope_signature(
    state: BeaconState, envelope: ExecutionPayloadEnvelope, privkey: int
) -> BLSSignature:
    domain = get_domain(state, DOMAIN_BEACON_BUILDER, compute_epoch_at_slot(state.slot))
    signing_root = compute_signing_root(envelope, domain)
    return bls.Sign(privkey, signing_root)
```

The builder assembles then
`signed_execution_payload_envelope = SignedExecutionPayloadEnvelope(message=envelope, signature=signature)`
and broadcasts it on the `execution_payload` global gossip topic.

### Honest payload withheld messages

An honest builder that has seen a `SignedBeaconBlock` referencing his signed
bid, but that block was not timely and thus it is not the head of the builder's
chain, may choose to withhold their execution payload. For this the builder
should act as if no block was produced and not broadcast the payload.

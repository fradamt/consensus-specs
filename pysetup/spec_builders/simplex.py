from pysetup.constants import SIMPLEX

from .base import BaseSpecBuilder


class SimplexSpecBuilder(BaseSpecBuilder):
    fork: str = SIMPLEX

    @classmethod
    def deprecate_constants(cls) -> set[str]:
        # Retain the frozen historical values but do not re-verify their
        # Electra BeaconState expressions against the changed Simplex layout.
        return {
            "FINALIZED_ROOT_GINDEX_ELECTRA",
            "CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA",
            "NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA",
        }

    @classmethod
    def deprecate_containers(cls) -> set[str]:
        # Simplex replaces the legacy epoch-FFG fast-confirmation subsystem
        # with the available/safe-confirmation heads in its fork-choice Store.
        return {
            "FastConfirmationStore",
        }

    @classmethod
    def deprecate_functions(cls) -> set[str]:
        return {
            # Legacy epoch-FFG state/fork-choice helpers. Simplex replaces the
            # checkpoint and participation models these functions consume.
            "calculate_committee_fraction",
            "compute_proposer_score",
            "compute_pulled_up_tip",
            "get_attestation_participation_flag_indices",
            "get_proposer_reorg_cutoff_ms",
            "get_proposer_score",
            "get_voting_source",
            "is_ffg_competitive",
            "is_finalization_ok",
            "is_head_late",
            "is_head_weak",
            "is_parent_strong",
            "is_proposer_equivocation",
            "is_proposing_on_time",
            "is_shuffling_stable",
            "record_block_timeliness",
            "should_apply_proposer_boost",
            "should_override_forkchoice_update",
            "store_target_checkpoint_state",
            "update_checkpoints",
            "update_proposer_boost_root",
            "update_unrealized_checkpoints",
            "validate_target_epoch_against_current_time",
            "weigh_justification_and_finalization",
            # Legacy fast confirmation. Simplex has a distinct
            # latest_confirmed_head / fast_confirmed_head subsystem.
            "adjust_committee_weight_estimate_to_ensure_safety",
            "compute_adversarial_weight",
            "compute_empty_slot_support_discount",
            "compute_honest_ffg_support_for_current_target",
            "compute_safety_threshold",
            "estimate_committee_weight_between_slots",
            "find_latest_confirmed_descendant",
            "get_adversarial_weight",
            "get_ancestor_roots",
            "get_block_epoch",
            "get_block_slot",
            "get_block_support_between_slots",
            "get_checkpoint_for_block",
            "get_current_balance_source",
            "get_current_target",
            "get_current_target_score",
            "get_equivocation_score",
            "get_fast_confirmation_store",
            "get_latest_confirmed",
            "get_node_for_root",
            "get_previous_balance_source",
            "get_pulled_up_head_state",
            "get_slot_committee",
            "get_support_discount",
            "is_confirmed_chain_safe",
            "is_full_validator_set_covered",
            "is_one_confirmed",
            "is_start_slot_at_epoch",
            "on_fast_confirmation",
            "update_fast_confirmation_variables",
            "will_current_target_be_justified",
            "will_no_conflicting_checkpoint_be_justified",
            # Gloas payload vote arrays are replaced by identity-keyed Simplex
            # payload votes and is_payload_{timely,data_available}.
            "payload_data_availability",
            "payload_timeliness",
            # Superseded fork transition.
            "upgrade_to_gloas",
        }

    @classmethod
    def imports(cls, preset_name: str):
        return f"""
from eth_consensus_specs.gloas import {preset_name} as gloas
"""

    @classmethod
    def hardcoded_ssz_dep_constants(cls) -> dict[str, str]:
        # Simplex's BeaconState drops justification_bits and the
        # previous/current_justified_checkpoint pair (replaced by a single
        # justified_checkpoint), shifting its light-client gindices:
        # finalized_checkpoint 20 -> 18, sync committees 22/23 -> 20/21. These
        # are new era constants; the inherited Electra constants must remain
        # frozen for historical pre-Simplex proofs.
        return {
            "FINALIZED_ROOT_GINDEX_SIMPLEX": "GeneralizedIndex(165)",
            "CURRENT_SYNC_COMMITTEE_GINDEX_SIMPLEX": "GeneralizedIndex(84)",
            "NEXT_SYNC_COMMITTEE_GINDEX_SIMPLEX": "GeneralizedIndex(85)",
        }

from pysetup.constants import SIMPLEX

from .base import BaseSpecBuilder


class SimplexSpecBuilder(BaseSpecBuilder):
    fork: str = SIMPLEX

    @classmethod
    def imports(cls, preset_name: str):
        return f"""
from eth_consensus_specs.gloas import {preset_name} as gloas
"""

    @classmethod
    def hardcoded_ssz_dep_constants(cls) -> dict[str, str]:
        # Simplex's BeaconState drops justification_bits and the
        # previous/current_justified_checkpoint pair (replaced by a single
        # justified_checkpoint), shifting the Electra-era light-client gindices:
        # finalized_checkpoint 20 -> 18, sync committees 22/23 -> 20/21. These
        # are the recomputed values over the simplex BeaconState tree.
        return {
            "FINALIZED_ROOT_GINDEX_ELECTRA": "GeneralizedIndex(165)",
            "CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA": "GeneralizedIndex(84)",
            "NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA": "GeneralizedIndex(85)",
        }

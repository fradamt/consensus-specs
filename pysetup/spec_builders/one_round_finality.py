from ..constants import ONE_ROUND_FINALITY
from .base import BaseSpecBuilder


class OneRoundFinalitySpecBuilder(BaseSpecBuilder):
    fork: str = ONE_ROUND_FINALITY

    @classmethod
    def imports(cls, preset_name: str):
        return f"""
from eth2spec.gloas import {preset_name} as gloas
"""

    @classmethod
    def hardcoded_ssz_dep_constants(cls) -> dict[str, str]:
        return {
            "FINALIZED_ROOT_GINDEX_ELECTRA": "GeneralizedIndex(165)",
            "CURRENT_SYNC_COMMITTEE_GINDEX_ELECTRA": "GeneralizedIndex(85)",
            "NEXT_SYNC_COMMITTEE_GINDEX_ELECTRA": "GeneralizedIndex(86)",
        }

    @classmethod
    def deprecate_constants(cls) -> set[str]:
        return set(
            [
                "TIMELY_SOURCE_FLAG_INDEX",
                "TIMELY_SOURCE_WEIGHT",
            ]
        )

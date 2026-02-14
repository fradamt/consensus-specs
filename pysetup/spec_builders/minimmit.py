from ..constants import MINIMMIT
from .base import BaseSpecBuilder


class MinimmitSpecBuilder(BaseSpecBuilder):
    fork: str = MINIMMIT

    @classmethod
    def imports(cls, preset_name: str):
        return f"""
from eth2spec.fulu import {preset_name} as fulu
"""

    @classmethod
    def deprecate_constants(cls) -> set[str]:
        return set(
            [
                "TIMELY_SOURCE_FLAG_INDEX",
                "TIMELY_SOURCE_WEIGHT",
            ]
        )

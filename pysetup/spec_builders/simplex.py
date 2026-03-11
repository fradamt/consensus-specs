from ..constants import SIMPLEX
from .base import BaseSpecBuilder


class SimplexSpecBuilder(BaseSpecBuilder):
    fork: str = SIMPLEX

    @classmethod
    def imports(cls, preset_name: str):
        return f"""
from eth2spec.fulu import {preset_name} as fulu
"""

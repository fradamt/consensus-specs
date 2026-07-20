from __future__ import annotations

import shutil

from eth_consensus_specs.test.helpers.constants import SIMPLEX
from tests.generators.compliance_runners.gen_base.output import dump_test_case_result
from tests.infra.dumper import Dumper


def test_generate_compliance_group(test_group, comptests_output_dir):
    dumper = Dumper()

    for test_case in test_group.test_cases:
        test_case.set_output_dir(comptests_output_dir)
        if test_case.dir.exists():
            shutil.rmtree(test_case.dir)

    for test_case_result in test_group.group_fn():
        dump_test_case_result(test_case_result, dumper)
        if test_case_result.test_case.fork_name == SIMPLEX:
            # The dedicated Simplex path is intentionally small enough to
            # replay during generation. This keeps its non-FFG format and
            # runner integration covered as one executable contract.
            from tests.generators.compliance_runners.fork_choice.runner.test_run import (  # noqa: PLC0415
                ComplianceTestInfo,
                run_test,
            )

            test_case = test_case_result.test_case
            assert test_case.dir is not None
            run_test(
                ComplianceTestInfo(
                    preset=test_case.preset_name,
                    fork=test_case.fork_name,
                    test_dir=test_case.dir,
                )
            )

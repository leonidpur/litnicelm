# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Host power condition and LitniceLM steady-state parsing, without a GPU."""

import unittest

from ..device.host_power import HostPowerCondition
from ..engines.litnicelm import LitniceLMEngine
from ..run import SPEC
from ..spec import EnvironmentSpec
from ..verticals.train_speed import TrainSpeedVertical

GOOD = {"ac_online": True, "platform_profile": "balanced"}


class HostPowerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.condition = HostPowerCondition(EnvironmentSpec(clock_mhz=1500))

    def test_matching_host_is_not_flagged(self) -> None:
        self.assertEqual(self.condition.mismatches(GOOD), [])
        self.assertEqual(self.condition.row_flags(GOOD), {"host_state_flagged": False})

    def test_battery_and_profile_are_flagged(self) -> None:
        state = {"ac_online": False, "platform_profile": "power-saver"}
        self.assertEqual(len(self.condition.mismatches(state)), 2)
        self.assertTrue(self.condition.row_flags(state)["host_state_flagged"])

    def test_unknown_ac_fails_when_ac_required(self) -> None:
        state = {"ac_online": "unavailable", "platform_profile": "balanced"}
        self.assertTrue(self.condition.row_flags(state)["host_state_flagged"])

    def test_ac_not_required(self) -> None:
        condition = HostPowerCondition(
            EnvironmentSpec(clock_mhz=1500, require_ac_power=False)
        )
        self.assertEqual(condition.mismatches({**GOOD, "ac_online": False}), [])


OUTPUT = """\
cuda_cublas_plugin: cuDNN sdpa forward plan 2/3 eng1 (63.4 us; heuristic first choice 115.5 us; selection 151.250 ms)
cuda_cublas_plugin: cuDNN sdpa backward plan 1/2 eng7 (149.5 us; heuristic first choice 166.9 us; selection 98.750 ms)
[SteadyState] after 5 warm-up optimizer steps: total=2.000 s steps=200 avg=10.000 ms
"""


class SteadyStateParseTest(unittest.TestCase):
    def test_steady_state_and_plan_selection(self) -> None:
        metrics = LitniceLMEngine._steady_state_metrics(SPEC, OUTPUT)
        self.assertAlmostEqual(metrics["plan_selection_seconds"], 0.25)
        self.assertEqual(metrics["steady_warmup_steps"], 5)
        self.assertEqual(metrics["steady_steps"], 200)
        self.assertAlmostEqual(metrics["steady_step_ms"], 10.0)
        self.assertAlmostEqual(
            metrics["steady_tokens_per_s"], 200 * SPEC.workload.tokens_per_step / 2.0
        )

    def test_without_cudnn_or_steady_line(self) -> None:
        self.assertEqual(
            LitniceLMEngine._steady_state_metrics(SPEC, "no such lines\n"),
            {"plan_selection_seconds": 0.0},
        )

    def test_cross_check_uses_steady_value(self) -> None:
        check = TrainSpeedVertical._timing_cross_check(
            {
                "tokens_per_s": 90.0,
                "steady_tokens_per_s": 100.0,
                "differential_tokens_per_s": 101.0,
            }
        )
        self.assertAlmostEqual(check["throughput_cross_check_difference_pct"], 1.0)
        self.assertFalse(check["throughput_cross_check_flagged"])


if __name__ == "__main__":
    unittest.main()

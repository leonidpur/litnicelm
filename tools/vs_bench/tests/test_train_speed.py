# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Differential throughput and its cross-check, without a GPU."""

import unittest

from ..engines.base import EngineResult
from ..verticals.train_speed import TrainSpeedVertical


def endpoint(steps: int, wall: float) -> EngineResult:
    return EngineResult(
        "PyTorch",
        "train_speed",
        {"steps": steps, "tokens_per_step": 4096, "process_wall_seconds": wall},
    )


class DifferentialTest(unittest.TestCase):
    def test_valid_differential(self) -> None:
        metrics = TrainSpeedVertical._differential_metrics((endpoint(10, 1.0), endpoint(30, 2.0)))
        self.assertTrue(metrics["differential_valid"])
        self.assertAlmostEqual(metrics["differential_step_ms"], 50.0)
        self.assertAlmostEqual(metrics["differential_tokens_per_s"], 4096 * 20)

    def test_longer_run_not_slower_is_invalid_not_fatal(self) -> None:
        metrics = TrainSpeedVertical._differential_metrics(
            (endpoint(119, 21.31), endpoint(357, 20.83))
        )
        self.assertFalse(metrics["differential_valid"])
        self.assertNotIn("differential_tokens_per_s", metrics)
        self.assertEqual(metrics["differential_low_wall_seconds"], 21.31)

    def test_invalid_differential_flags_cross_check(self) -> None:
        check = TrainSpeedVertical._timing_cross_check({"tokens_per_s": 100.0})
        self.assertEqual(check, {"throughput_cross_check_flagged": True})

    def test_valid_cross_check(self) -> None:
        check = TrainSpeedVertical._timing_cross_check(
            {"tokens_per_s": 100.0, "differential_tokens_per_s": 102.0}
        )
        self.assertAlmostEqual(check["throughput_cross_check_difference_pct"], 2.0)
        self.assertFalse(check["throughput_cross_check_flagged"])


if __name__ == "__main__":
    unittest.main()

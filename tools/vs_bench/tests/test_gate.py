# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Gate comparison, results loading and report rendering, without a GPU."""

import json
from pathlib import Path
import tempfile
import unittest

from ..gate.compare import (
    GateComparison,
    GateKey,
    GateRow,
    compare,
    load_litnicelm_rows,
    load_pytorch_references,
    new_cells,
)
from ..gate.report import render_gate_report

BF16 = GateKey("bf16", "fused_inplace")
FP32 = GateKey("fp32", "fused_inplace")


def row(key: GateKey, *samples: float) -> GateRow:
    return GateRow(key, samples, f"{key.precision} ops")


def speed_row(engine: str, precision: str, label: str, options: str, value: float) -> dict:
    return {
        "engine": engine,
        "operation": "train_speed",
        "precision": precision,
        "variant_label": label,
        "variant_options": options,
        "tokens_per_s": value,
        "attention_cuda_kernels": f"{label} kernels",
    }


class ResultsFile:
    """A temporary results JSON with one train_speed vertical."""

    def __init__(self, rows: list[dict], status: str = "complete") -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "results.json"
        self.path.write_text(
            json.dumps({"status": status, "verticals": [{"name": "train_speed", "rows": rows}]})
        )

    def close(self) -> None:
        self._directory.cleanup()


class ComparisonTest(unittest.TestCase):
    def test_median_of_samples(self) -> None:
        self.assertEqual(row(BF16, 3.0, 1.0, 2.0).median_tokens_per_s, 2.0)

    def test_drop_of_exactly_the_limit_passes(self) -> None:
        item = GateComparison(row(BF16, 99.5), row(BF16, 100.0), 0.5)
        self.assertAlmostEqual(item.change_pct, -0.5)
        self.assertTrue(item.passed)

    def test_drop_beyond_the_limit_fails(self) -> None:
        self.assertFalse(GateComparison(row(BF16, 99.49), row(BF16, 100.0), 0.5).passed)

    def test_gain_passes(self) -> None:
        item = GateComparison(row(BF16, 110.0), row(BF16, 100.0), 0.5)
        self.assertAlmostEqual(item.change_pct, 10.0)
        self.assertTrue(item.passed)

    def test_missing_cell_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "fp32/fused_inplace"):
            compare({BF16: row(BF16, 1.0)}, {BF16: row(BF16, 1.0), FP32: row(FP32, 1.0)})

    def test_new_cell_is_not_gated(self) -> None:
        current = {BF16: row(BF16, 1.0), FP32: row(FP32, 1.0)}
        result = compare(current, {BF16: row(BF16, 1.0)})
        self.assertEqual([item.current.key for item in result], [BF16])
        self.assertEqual(
            [item.key for item in new_cells(current, {BF16: row(BF16, 1.0)})], [FP32]
        )


class LoadingTest(unittest.TestCase):
    def test_litnicelm_rows_grouped_by_precision_and_variant(self) -> None:
        results = ResultsFile(
            [
                speed_row("LitniceLM", "bf16", "fused_inplace", "attention_mode=fused_inplace", 1.0),
                speed_row("LitniceLM", "bf16", "fused_inplace", "attention_mode=fused_inplace", 3.0),
                speed_row("PyTorch", "bf16", "eager", "compile_mode=none", 9.0),
            ]
        )
        self.addCleanup(results.close)
        rows = load_litnicelm_rows(results.path)
        self.assertEqual(list(rows), [BF16])
        self.assertEqual(rows[BF16].samples_tokens_per_s, (1.0, 3.0))
        self.assertEqual(rows[BF16].op_breakdown, "fused_inplace kernels")

    def test_incomplete_results_rejected(self) -> None:
        results = ResultsFile(
            [speed_row("LitniceLM", "bf16", "fused_inplace", "", 1.0)], status="incomplete"
        )
        self.addCleanup(results.close)
        with self.assertRaisesRegex(ValueError, "not 'complete'"):
            load_litnicelm_rows(results.path)

    def test_best_eager_and_compiled_pytorch(self) -> None:
        results = ResultsFile(
            [
                speed_row("PyTorch", "bf16", "eager / end", "loss_read=end, compile_mode=none", 100.0),
                speed_row("PyTorch", "bf16", "eager / batch", "loss_read=per_batch, compile_mode=none", 90.0),
                speed_row("PyTorch", "bf16", "compiled / default", "compile_mode=default", 200.0),
                speed_row("PyTorch", "bf16", "compiled / ro", "compile_mode=reduce-overhead", 300.0),
            ]
        )
        self.addCleanup(results.close)
        reference = load_pytorch_references(results.path)["bf16"]
        self.assertEqual(
            (reference.best_eager_label, reference.best_eager_tokens_per_s), ("eager / end", 100.0)
        )
        self.assertEqual(
            (reference.best_compiled_label, reference.best_compiled_tokens_per_s),
            ("compiled / ro", 300.0),
        )

    def test_missing_compiled_pytorch_raises(self) -> None:
        results = ResultsFile([speed_row("PyTorch", "bf16", "eager", "compile_mode=none", 1.0)])
        self.addCleanup(results.close)
        with self.assertRaisesRegex(ValueError, "no compiled PyTorch rows for bf16"):
            load_pytorch_references(results.path)


class ReportTest(unittest.TestCase):
    def test_failed_cell_and_verdict(self) -> None:
        report = render_gate_report(
            compare({BF16: row(BF16, 90.0)}, {BF16: row(BF16, 100.0)}), {}
        )
        self.assertIn("| -10.00% | FAIL |", report)
        self.assertIn("Gate FAILED: 1 of 1 cells", report)
        self.assertIn("before: bf16 ops", report)

    def test_new_cell_listed_as_not_gated(self) -> None:
        report = render_gate_report(
            compare({BF16: row(BF16, 100.0)}, {BF16: row(BF16, 100.0)}),
            {},
            (row(FP32, 50.0),),
        )
        self.assertIn("| fp32 | fused_inplace | — | 50 |", report)
        self.assertIn("new, not gated", report)


if __name__ == "__main__":
    unittest.main()

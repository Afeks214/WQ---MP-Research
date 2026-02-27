import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_sweep_auto import (
    SymbolInventory,
    _apply_sweep_abort_guardrails,
    _selection_probe_window,
    _supports_manual_candidates,
    compute_quality_score,
    inspect_symbol_file,
    select_symbols_deterministic,
)


@unittest.skipIf(pd is None, "pandas not available")
class TestSweepAutoSelection(unittest.TestCase):
    def test_abort_guardrails_are_applied_deterministically(self) -> None:
        cfg = {
            "symbols": ["SPY", "QQQ"],
            "harness": {
                "failure_rate_abort_threshold": 0.02,
                "failure_count_abort_threshold": 50,
                "parallel_backend": "process_pool",
                "parallel_workers": 4,
            },
        }
        out = _apply_sweep_abort_guardrails(cfg)
        self.assertEqual(float(out["harness"]["failure_rate_abort_threshold"]), 1.0)
        self.assertEqual(int(out["harness"]["failure_count_abort_threshold"]), 1_000_000)
        self.assertEqual(str(out["harness"]["parallel_backend"]), "process_pool")
        self.assertEqual(int(out["harness"]["parallel_workers"]), 4)

        out_quick = _apply_sweep_abort_guardrails(cfg, force_serial=True)
        self.assertEqual(str(out_quick["harness"]["parallel_backend"]), "serial")
        self.assertEqual(int(out_quick["harness"]["parallel_workers"]), 1)
        # Source config must remain unchanged.
        self.assertEqual(float(cfg["harness"]["failure_rate_abort_threshold"]), 0.02)
        self.assertEqual(int(cfg["harness"]["failure_count_abort_threshold"]), 50)

    def test_manual_candidates_support_detection(self) -> None:
        self.assertTrue(_supports_manual_candidates())

    def test_deterministic_selection_tie_break(self) -> None:
        base_score = compute_quality_score(
            row_count=1_000,
            nan_rate_ohlcv=0.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=True,
        )
        rows = [
            SymbolInventory(
                symbol="BBB",
                path="/tmp/BBB.parquet",
                file_size_bytes=1,
                row_count=1_000,
                detected_columns="{}",
                has_required_aliases=True,
                nan_rate_ohlcv=0.0,
                duplicate_timestamp_count=0,
                monotonic_timestamp_ok=True,
                quality_score=base_score,
                excluded=False,
                exclusion_reason="",
            ),
            SymbolInventory(
                symbol="AAA",
                path="/tmp/AAA.parquet",
                file_size_bytes=1,
                row_count=1_000,
                detected_columns="{}",
                has_required_aliases=True,
                nan_rate_ohlcv=0.0,
                duplicate_timestamp_count=0,
                monotonic_timestamp_ok=True,
                quality_score=base_score,
                excluded=False,
                exclusion_reason="",
            ),
            SymbolInventory(
                symbol="CCC",
                path="/tmp/CCC.parquet",
                file_size_bytes=1,
                row_count=1_000,
                detected_columns="{}",
                has_required_aliases=True,
                nan_rate_ohlcv=0.0,
                duplicate_timestamp_count=0,
                monotonic_timestamp_ok=True,
                quality_score=base_score,
                excluded=False,
                exclusion_reason="",
            ),
        ]

        chosen, ranked, _excluded = select_symbols_deterministic(
            rows,
            target_symbols=2,
            min_symbols=1,
        )

        self.assertEqual(chosen, ["AAA", "BBB"])
        self.assertEqual([r.symbol for r in ranked], ["AAA", "BBB", "CCC"])

    def test_exclusion_reason_when_required_columns_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "BAD.parquet"
            df = pd.DataFrame(
                {
                    "foo": [1, 2],
                    "bar": [3, 4],
                }
            )
            try:
                df.to_parquet(p, index=False)
            except Exception as exc:  # pragma: no cover - environment dependency
                self.skipTest(f"parquet writer unavailable: {exc}")

            row = inspect_symbol_file(p)
            self.assertTrue(row.excluded)
            self.assertFalse(row.has_required_aliases)
            self.assertIn("missing_required_columns", row.exclusion_reason)

    def test_min_symbols_fail_closed(self) -> None:
        row = SymbolInventory(
            symbol="SPY",
            path="/tmp/SPY.parquet",
            file_size_bytes=1,
            row_count=100,
            detected_columns="{}",
            has_required_aliases=True,
            nan_rate_ohlcv=0.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=True,
            quality_score=100.0,
            excluded=False,
            exclusion_reason="",
        )

        with self.assertRaises(RuntimeError) as ctx:
            select_symbols_deterministic(
                [row],
                target_symbols=20,
                min_symbols=2,
            )

        self.assertIn("Insufficient valid symbols", str(ctx.exception))

    def test_selection_probe_window_uses_config_date_range(self) -> None:
        cfg = {
            "data": {
                "start": "2023-02-01T00:00:00Z",
                "end": "2023-12-31T23:59:59Z",
            }
        }
        start_ts, end_ts, probe_year = _selection_probe_window(cfg)
        self.assertIsNotNone(start_ts)
        self.assertIsNotNone(end_ts)
        self.assertEqual(int(probe_year), 2023)

    def test_selection_probe_window_falls_back_to_2024(self) -> None:
        start_ts, end_ts, probe_year = _selection_probe_window({})
        self.assertIsNone(start_ts)
        self.assertIsNone(end_ts)
        self.assertEqual(int(probe_year), 2024)


if __name__ == "__main__":
    unittest.main()

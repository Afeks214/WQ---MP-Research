import unittest

from scripts.run_sweep_auto import (
    SymbolDQProbe,
    SymbolInventory,
    compute_quality_score,
    select_symbols_deterministic,
)


class TestDQSelectionRanking(unittest.TestCase):
    def _inv(self, symbol: str, score: float) -> SymbolInventory:
        return SymbolInventory(
            symbol=symbol,
            path=f"/tmp/{symbol}.parquet",
            file_size_bytes=1,
            row_count=1000,
            detected_columns="{}",
            has_required_aliases=True,
            nan_rate_ohlcv=0.0,
            duplicate_timestamp_count=0,
            monotonic_timestamp_ok=True,
            quality_score=float(score),
            excluded=False,
            exclusion_reason="",
        )

    def _dq(self, symbol: str, coverage: float, median_dqs: float) -> SymbolDQProbe:
        reject_ratio = max(0.0, 1.0 - float(coverage))
        return SymbolDQProbe(
            symbol=symbol,
            coverage_ratio=float(coverage),
            median_dqs=float(median_dqs),
            reject_ratio=float(reject_ratio),
            total_days=100,
            accept_days=int(round(coverage * 100.0)),
            degrade_days=0,
            reject_days=int(round((1.0 - coverage) * 100.0)),
            probe_error="",
        )

    def test_coverage_ranked_before_quality(self) -> None:
        high_quality_low_coverage = self._inv("AAA", score=compute_quality_score(10000, 0.0, 0, True))
        lower_quality_high_coverage = self._inv("BBB", score=compute_quality_score(5000, 0.0, 0, True))

        chosen, ranked, _ = select_symbols_deterministic(
            [high_quality_low_coverage, lower_quality_high_coverage],
            target_symbols=1,
            min_symbols=1,
            dq_metrics_by_symbol={
                "AAA": self._dq("AAA", coverage=0.60, median_dqs=0.70),
                "BBB": self._dq("BBB", coverage=0.99, median_dqs=0.95),
            },
        )

        self.assertEqual(chosen, ["BBB"])
        self.assertEqual([r.symbol for r in ranked], ["BBB", "AAA"])

    def test_tie_break_by_symbol_is_deterministic(self) -> None:
        score = compute_quality_score(1000, 0.0, 0, True)
        rows = [self._inv("ZZZ", score), self._inv("AAA", score)]
        dq = {
            "ZZZ": self._dq("ZZZ", coverage=0.95, median_dqs=0.90),
            "AAA": self._dq("AAA", coverage=0.95, median_dqs=0.90),
        }

        chosen, ranked, _ = select_symbols_deterministic(
            rows,
            target_symbols=2,
            min_symbols=1,
            dq_metrics_by_symbol=dq,
        )

        self.assertEqual([r.symbol for r in ranked], ["AAA", "ZZZ"])
        self.assertEqual(chosen, ["AAA", "ZZZ"])


if __name__ == "__main__":
    unittest.main()

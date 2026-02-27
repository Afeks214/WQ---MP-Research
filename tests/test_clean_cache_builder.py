from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from scripts.build_clean_cache_from_bundle import (
    build_clean_cache_from_bundle,
    choose_preferred_part,
)


class TestCleanCacheBuilder(unittest.TestCase):
    def _write_part(self, path: Path, start: str, periods: int, freq: str) -> None:
        idx = pd.date_range(start=start, periods=periods, freq=freq, tz="America/New_York").tz_convert("UTC")
        close = pd.Series(range(periods), dtype="float64") + 100.0
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000.0,
            },
            index=idx,
        )
        df.to_parquet(path)

    def _make_bundle(self, root: Path) -> Path:
        market = root / "MarketData"
        (market / "SPY").mkdir(parents=True, exist_ok=True)
        (market / "QQQ").mkdir(parents=True, exist_ok=True)

        self._write_part(market / "SPY" / "part-2024-1Min.parquet", "2024-01-02 09:31:00", 10, "1min")
        self._write_part(market / "SPY" / "part-2024-5Min.parquet", "2024-01-02 09:35:00", 5, "5min")
        self._write_part(market / "QQQ" / "part-2023-1Min.parquet", "2023-01-03 09:31:00", 10, "1min")

        bundle = root / "bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted((root / "MarketData").rglob("*.parquet")):
                zf.write(p, p.relative_to(root))
        return bundle

    def test_choose_preferred_part_prioritizes_2024_1min(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_bundle(root)
            chosen = choose_preferred_part(root / "MarketData" / "SPY", target_year=2024)
            self.assertEqual(chosen.name, "part-2024-1Min.parquet")

    def test_build_cache_atomic_swap_and_timestamp_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "data" / "alpaca").mkdir(parents=True, exist_ok=True)
            old_clean = repo / "data" / "alpaca" / "clean"
            old_clean.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}).to_parquet(old_clean / "OLD.parquet", index=False)

            src = Path(td) / "src"
            src.mkdir(parents=True, exist_ok=True)
            bundle = self._make_bundle(src)
            extract_dir = Path(td) / "extracted"

            manifest = build_clean_cache_from_bundle(
                bundle_zip=bundle,
                extract_dir=extract_dir,
                repo_root=repo,
                target_year=2024,
            )

            clean = repo / "data" / "alpaca" / "clean"
            self.assertTrue((clean / "SPY.parquet").exists())
            self.assertTrue((clean / "QQQ.parquet").exists())

            backups = sorted((repo / "data" / "alpaca").glob("clean_backup_*"))
            self.assertTrue(len(backups) >= 1)
            self.assertTrue((backups[-1] / "OLD.parquet").exists())

            out = pd.read_parquet(clean / "SPY.parquet")
            self.assertIn("timestamp", out.columns)
            self.assertIn("open", out.columns)
            self.assertEqual(out["timestamp"].isna().sum(), 0)
            self.assertIn("symbols_built", manifest)


if __name__ == "__main__":
    unittest.main()

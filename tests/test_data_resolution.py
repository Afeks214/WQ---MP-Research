from __future__ import annotations

from pathlib import Path

import pandas as pd

from weightiz.shared.config.models import DataConfigModel
from weightiz.shared.io.data_resolution import in_memory_date_filter_loader


def test_in_memory_loader_accepts_utc_datetime_index_without_timestamp_column(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-02 14:31:00+00:00", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100.0, 200.0, 300.0],
        },
        index=idx,
    )
    path = tmp_path / "indexed.parquet"
    df.to_parquet(path)

    loader = in_memory_date_filter_loader(
        DataConfigModel(
            root=str(tmp_path),
            format="parquet",
            timestamp_column=None,
            start=pd.Timestamp("2025-01-02T00:00:00Z"),
            end=pd.Timestamp("2025-01-02T23:59:59Z"),
        )
    )
    out = loader(str(path), "America/New_York")

    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.shape == (3, 5)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]

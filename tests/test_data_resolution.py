from __future__ import annotations

from pathlib import Path

import pandas as pd

import weightiz.shared.io.data_resolution as data_resolution
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


def test_in_memory_loader_handles_read_only_boolean_masks(tmp_path: Path, monkeypatch) -> None:
    ts = pd.date_range("2025-01-02 14:31:00+00:00", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100.0, 200.0, 300.0],
        }
    )
    path = tmp_path / "readonly-mask.parquet"
    df.to_parquet(path, index=False)

    original_asarray = data_resolution.np.asarray
    call_state = {"locked": False}

    def _asarray_readonly_once(*args, **kwargs):
        arr = original_asarray(*args, **kwargs)
        if not call_state["locked"] and getattr(arr, "dtype", None) == bool and getattr(arr, "ndim", 0) == 1:
            arr = arr.copy()
            arr.setflags(write=False)
            call_state["locked"] = True
        return arr

    monkeypatch.setattr(data_resolution.np, "asarray", _asarray_readonly_once)

    loader = in_memory_date_filter_loader(
        DataConfigModel(
            root=str(tmp_path),
            format="parquet",
            timestamp_column="timestamp",
            start=pd.Timestamp("2025-01-02T00:00:00Z"),
            end=pd.Timestamp("2025-01-02T23:59:59Z"),
        )
    )
    out = loader(str(path), "America/New_York")
    assert out.shape == (3, 5)

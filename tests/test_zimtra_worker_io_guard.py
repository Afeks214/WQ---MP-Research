import pytest

from sweep_runner import _install_worker_io_guard


def test_worker_io_guard_blocks_raw_reads() -> None:
    _install_worker_io_guard()
    import pandas as pd

    with pytest.raises(RuntimeError, match="WORKER_IO_VIOLATION"):
        pd.read_parquet("dummy.parquet")

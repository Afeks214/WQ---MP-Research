from __future__ import annotations

import pytest

import weightiz.module5.orchestrator as h


def test_workers_cannot_write_strategy_ledger(tmp_path, monkeypatch):
    out = tmp_path / "strategy_results.parquet"

    # Avoid parquet engine dependency in this unit test.
    monkeypatch.setattr(h, "_atomic_write_parquet", lambda df, path: path.write_text("ok", encoding="utf-8"))

    h._WORKER_PROCESS = True
    with pytest.raises(RuntimeError, match="LEDGER_WRITE_FORBIDDEN_IN_WORKER"):
        h._ledger_write([], out)

    h._WORKER_PROCESS = False
    h._ledger_write([], out)
    assert out.exists()

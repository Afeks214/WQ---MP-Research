from __future__ import annotations

import numpy as np

from weightiz.module6.io import load_module5_run
from weightiz.module6.ledger import materialize_canonical_ledgers
from weightiz.module6.matrices import build_matrix_store
from weightiz.module6.runtime import open_matrix_store
from tests.module6_testkit import build_synthetic_module5_run, make_test_config


def test_matrix_store_shapes_and_availability_mapping(tmp_path):
    run_dir = build_synthetic_module5_run(tmp_path)
    cfg = make_test_config()
    loaded = load_module5_run(run_dir, cfg)
    ledgers = materialize_canonical_ledgers(loaded, run_dir / "ledgers", cfg)
    store = build_matrix_store(ledgers=ledgers, run=loaded, output_dir=run_dir / "matrix_out", config=cfg)
    opened = open_matrix_store(store)
    assert opened["R_exec"].shape == opened["A"].shape
    assert opened["R_raw"].shape == opened["A"].shape
    assert opened["state_codes"].shape == opened["A"].shape
    assert opened["buying_power_min"].shape == opened["A"].shape
    assert store.calendar_index_path.exists()
    codes = np.asarray(opened["state_codes"])
    a = np.asarray(opened["A"])
    assert np.all(a[(codes == 1) | (codes == 2)])
    assert not np.any(a[(codes == 3) | (codes == 6)])

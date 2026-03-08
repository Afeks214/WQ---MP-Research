from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from run_research import _load_config
from weightiz_module4_strategy_funnel import Module4Config


@unittest.skipIf(yaml is None, "pyyaml not available")
class TestModule4ConfigSchemaCompat(unittest.TestCase):
    def _write_cfg(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        with tmp:
            tmp.write(yaml.safe_dump(payload, sort_keys=False))
        return Path(tmp.name)

    def _base_cfg(self) -> dict:
        return {
            "run_name": "schema_compat_test",
            "symbols": ["SPY", "QQQ"],
            "data": {
                "root": "./data/minute",
                "format": "parquet",
                "timestamp_column": "timestamp",
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-01-10T23:59:59Z",
            },
            "engine": {
                "mode": "sealed",
                "B": 64,
                "x_min": -6.0,
                "dx": 0.05,
                "seed": 17,
            },
            "module2_configs": [{}],
            "module3_configs": [{}],
            "module4_configs": [{}],
            "harness": {
                "parallel_backend": "serial",
                "parallel_workers": 1,
                "wf_train_sessions": 5,
                "wf_test_sessions": 2,
                "wf_step_sessions": 2,
                "cpcv_slices": 2,
                "cpcv_k_test": 1,
                "daily_return_min_days": 2,
            },
            "stress_scenarios": [
                {
                    "scenario_id": "baseline",
                    "name": "baseline",
                    "missing_burst_prob": 0.0,
                    "missing_burst_min": 0,
                    "missing_burst_max": 0,
                    "jitter_sigma_bps": 0.0,
                    "slippage_mult": 1.0,
                    "enabled": True,
                }
            ],
            "candidates": {"mode": "auto_grid", "specs": []},
        }

    def test_cell6_keys_validate_and_map_delta_threshold(self) -> None:
        cfg = self._base_cfg()
        cfg["module4_configs"] = [
            {
                "entry_threshold": 0.2,
                "strategy_type": "breakout",
                "score_gate": "sbreak",
                "score_gate_rule": "gte",
                "deviation_signal": "delta",
                "deviation_rule": "abs_gt",
                "entry_model": "model_a",
                "exit_model": "model_b",
                "origin_level": "POC",
                "direction": "long",
                "delta_th": 0.61,
                "dev_th": 1.2,
                "tp_mult": 1.5,
                "atr_stop_mult": 1.1,
            }
        ]
        path = self._write_cfg(cfg)
        parsed = _load_config(path)
        m4 = parsed.module4_configs[0]
        self.assertEqual(m4.strategy_type, "breakout")
        self.assertAlmostEqual(m4.delta_th, 0.61)
        self.assertAlmostEqual(m4.entry_threshold, 0.61)

    def test_legacy_entry_threshold_unchanged_without_delta_th(self) -> None:
        cfg = self._base_cfg()
        cfg["module4_configs"] = [{"entry_threshold": 0.73, "exit_threshold": 0.3}]
        path = self._write_cfg(cfg)
        parsed = _load_config(path)
        m4 = parsed.module4_configs[0]
        self.assertAlmostEqual(m4.entry_threshold, 0.73)

    def test_new_decision_layer_fields_validate(self) -> None:
        cfg = self._base_cfg()
        cfg["module4_configs"] = [
            {
                "window_selection_mode": "multi_window",
                "fixed_window_index": 0,
                "anchor_window_index": 1,
                "max_volatility": 2.0,
                "max_spread": 0.05,
                "min_liquidity": 0.2,
                "regime_confidence_min": 0.6,
                "conviction_scale": 1.25,
                "conviction_clip": 0.9,
                "max_abs_weight": 0.8,
                "enable_degraded_bridge_mode": True,
            }
        ]
        path = self._write_cfg(cfg)
        parsed = _load_config(path)
        m4 = parsed.module4_configs[0]
        self.assertEqual(m4.window_selection_mode, "multi_window")
        self.assertAlmostEqual(m4.regime_confidence_min, 0.6)
        self.assertAlmostEqual(m4.max_abs_weight, 0.8)

    def test_runtime_and_loader_module4_schema_keys_match(self) -> None:
        runtime_fields = set(Module4Config.__dataclass_fields__.keys())
        loader_fields = set(type(_load_config(self._write_cfg(self._base_cfg())).module4_configs[0]).model_fields.keys())
        self.assertEqual(runtime_fields, loader_fields)

    def test_regime_classifier_fields_parse_with_locked_defaults(self) -> None:
        cfg = self._base_cfg()
        cfg["module4_configs"] = [
            {
                "trend_spread_min": 0.07,
                "trend_poc_drift_min_abs": 0.42,
                "neutral_poc_drift_max_abs": 0.12,
                "shape_skew_min_abs": 0.5,
                "regime_confidence_min": 0.66,
                "eps": 1e-10,
            }
        ]
        path = self._write_cfg(cfg)
        parsed = _load_config(path)
        m4 = parsed.module4_configs[0]
        self.assertAlmostEqual(m4.trend_spread_min, 0.07)
        self.assertAlmostEqual(m4.trend_poc_drift_min_abs, 0.42)
        self.assertAlmostEqual(m4.neutral_poc_drift_max_abs, 0.12)
        self.assertAlmostEqual(m4.shape_skew_min_abs, 0.5)
        self.assertAlmostEqual(m4.regime_confidence_min, 0.66)
        self.assertAlmostEqual(m4.eps, 1e-10)

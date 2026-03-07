#!/usr/bin/env python3
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import yaml


def main() -> None:
    out_dir = Path('configs/_generated')
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path('configs/sweep_20x100.yaml')
    out_path = out_dir / 'sweep_20x432.yaml'

    if not src_path.exists():
        raise SystemExit(f"Missing baseline config: {src_path}")

    base = yaml.safe_load(src_path.read_text(encoding='utf-8'))
    if not isinstance(base, dict):
        raise SystemExit('Baseline YAML root must be a mapping')

    m2_base_list = base.get('module2_configs')
    if not isinstance(m2_base_list, list) or not m2_base_list:
        raise SystemExit('module2_configs must be a non-empty list in baseline')
    m2_base = copy.deepcopy(m2_base_list[0])

    m3_base_list = base.get('module3_configs')
    if not isinstance(m3_base_list, list) or not m3_base_list:
        raise SystemExit('module3_configs must be a non-empty list in baseline')

    # Canonical M3: prefer 30-minute block if present, else first entry.
    m3_base = None
    for row in m3_base_list:
        if isinstance(row, dict) and int(row.get('block_minutes', -1)) == 30:
            m3_base = copy.deepcopy(row)
            break
    if m3_base is None:
        m3_base = copy.deepcopy(m3_base_list[0])

    m4_base_list = base.get('module4_configs')
    if not isinstance(m4_base_list, list) or not m4_base_list:
        raise SystemExit('module4_configs must be a non-empty list in baseline')
    m4_base = copy.deepcopy(m4_base_list[0])

    # Tier A+ sets (deterministic order required by spec)
    entry_threshold_vals = [0.50, 0.55, 0.60, 0.65]
    trend_poc_drift_min_abs_vals = [0.25, 0.40, 0.55]
    neutral_poc_drift_max_abs_vals = [0.10, 0.15]
    shape_skew_min_abs_vals = [0.25, 0.35, 0.50]
    top_k_intraday_vals = [2, 4, 6]

    # Build M2 list (2) by copying baseline and changing only va_threshold.
    m2_list = []
    for va in [0.65, 0.70]:
        row = copy.deepcopy(m2_base)
        row['va_threshold'] = float(va)
        m2_list.append(row)

    # Build M3 list (1) using canonical baseline.
    m3_list = [m3_base]

    # Build M4 list (216) in exact required nested order.
    m4_list = []
    for entry_threshold, trend_poc_drift_min_abs, neutral_poc_drift_max_abs, shape_skew_min_abs, top_k_intraday in itertools.product(
        entry_threshold_vals,
        trend_poc_drift_min_abs_vals,
        neutral_poc_drift_max_abs_vals,
        shape_skew_min_abs_vals,
        top_k_intraday_vals,
    ):
        row = copy.deepcopy(m4_base)
        row['entry_threshold'] = float(entry_threshold)
        row['trend_poc_drift_min_abs'] = float(trend_poc_drift_min_abs)
        row['neutral_poc_drift_max_abs'] = float(neutral_poc_drift_max_abs)
        row['shape_skew_min_abs'] = float(shape_skew_min_abs)
        row['top_k_intraday'] = int(top_k_intraday)
        m4_list.append(row)

    # Uniqueness proof for M4 cartesian rows.
    m4_hashes = {json.dumps(row, sort_keys=True, separators=(',', ':')) for row in m4_list}

    n2 = len(m2_list)
    n3 = len(m3_list)
    n4 = len(m4_list)
    total = n2 * n3 * n4

    assert n2 == 2, f"Expected M2=2, got {n2}"
    assert n3 == 1, f"Expected M3=1, got {n3}"
    assert n4 == 216, f"Expected M4=216, got {n4}"
    assert total == 432, f"Expected total=432, got {total}"
    assert len(m4_hashes) == 216, f"Expected unique M4 hashes=216, got {len(m4_hashes)}"

    out = copy.deepcopy(base)
    out['run_name'] = 'sweep_20x432'
    out['symbols'] = ['SPY', 'QQQ', 'GLD', 'EEM', 'HYG', 'IWM', 'XLU', 'XLK', 'XLE', 'TLT']
    out.setdefault('data', copy.deepcopy(base.get('data', {})))
    out['data']['start'] = '2023-01-01T00:00:00+00:00'
    out['data']['end'] = '2026-02-27T23:59:59+00:00'
    out['module2_configs'] = m2_list
    out['module3_configs'] = m3_list
    out['module4_configs'] = m4_list

    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding='utf-8')

    print(f"M2={n2}, M3={n3}, M4={n4}, total={total}")
    print(f"M4_unique_hash_count={len(m4_hashes)}")
    print(f"output_yaml={out_path}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure repo root is importable when script is executed as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.hpc_market_profile_parity import compute_market_profile_features
from engine.profile_sanity_plots import generate_profile_sanity_plots


def _canonical_hash(df: pd.DataFrame) -> str:
    ordered = df.sort_values([c for c in ("timestamp", "symbol") if c in df.columns], kind="mergesort")
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-year profile export with diagnostics")
    parser.add_argument("--parquet", required=True, help="Input parquet path")
    parser.add_argument("--symbol", required=False, default="", help="Optional symbol label if absent")
    parser.add_argument("--workers", required=False, type=int, default=1, help="Reserved compatibility flag")
    parser.add_argument("--output", required=True, help="Output features parquet path")
    parser.add_argument("--determinism-check", action="store_true", help="Run deterministic hash check")
    parser.add_argument(
        "--diagnostics-dir",
        default="artifacts/profile_diagnostics",
        help="Directory for diagnostics PNG outputs",
    )
    parser.add_argument(
        "--metadata-output",
        default="",
        help="Optional metadata JSON output path (default: sibling run_metadata.json)",
    )
    args = parser.parse_args()

    inp = Path(args.parquet).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    diagnostics_dir = Path(args.diagnostics_dir).expanduser().resolve()

    if not inp.exists():
        raise RuntimeError(f"Input parquet not found: {inp}")

    df = pd.read_parquet(inp)
    if "symbol" not in df.columns and args.symbol:
        df = df.copy()
        df["symbol"] = str(args.symbol)

    features = compute_market_profile_features(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)

    summary = generate_profile_sanity_plots(features_path=out, output_dir=diagnostics_dir)

    if args.determinism_check:
        hash1 = _canonical_hash(features)
        hash2 = _canonical_hash(pd.read_parquet(out))
        if hash1 != hash2:
            raise RuntimeError("DETERMINISM_HASH_MISMATCH: parquet roundtrip hash mismatch")

    meta_path = (
        Path(args.metadata_output).expanduser().resolve()
        if args.metadata_output
        else out.with_name("run_metadata.json")
    )

    metadata: dict[str, Any] = {
        "output_path": str(out),
        "row_count": int(len(features)),
        "nan_count": int(features.isna().sum().sum()),
        "inf_count": int(
            np.isinf(
                features.select_dtypes(include=["number"]).to_numpy(dtype=np.float64, copy=True)
            ).sum()
        ),
        "determinism_hash": _canonical_hash(features),
        "diagnostics": summary,
        "module_validation_status": "not_run_in_this_runner",
        "forensic_validation_status": "diagnostics_ok" if summary.get("plots_generated") or summary.get("reason") == "matplotlib_missing" else "diagnostics_failed",
    }

    _write_json(meta_path, metadata)

    print(f"[RUNNER] features_written={out}")
    print(f"[RUNNER] diagnostics_dir={diagnostics_dir}")
    print(f"[RUNNER] metadata={meta_path}")
    print(f"[RUNNER] plots_generated={summary.get('plots_generated')}")
    if summary.get("reason"):
        print(f"[RUNNER] diagnostics_reason={summary['reason']}")


if __name__ == "__main__":
    main()

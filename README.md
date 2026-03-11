# Weightiz

Canonical Weightiz source now lives under `src/weightiz`. Supported execution is package-first and server-safe.

## Install

```bash
python3 -m pip install -e .
```

## Supported Runners

```bash
python3 -m weightiz.cli.run_research --config configs/server/compute-small.yaml
python3 -m weightiz.cli.run_module5 --config configs/server/compute-small.yaml
python3 -m weightiz.cli.run_module6 --config configs/module6/default.yaml --run-dir <module5-run-dir>
python3 -m weightiz.cli.validate_artifacts --run-dir <module5-run-dir> --require-module6
```

## Repository Layout

- `src/weightiz/`: canonical Python package
- `configs/`: checked-in run profiles, including `configs/server/` and `configs/module6/`
- `data/`: input data roots
- `artifacts/`: runtime outputs
- `logs/`: launcher records and runtime logs
- `tmp/`: transient scratch space
- `infra/environment/`: environment examples for server execution

## Notes

- Root-level business logic and the old dashboard-style Module 6 have been removed from the supported path.
- `pyproject.toml` is the canonical project and tool configuration hub.
- Historical and archive scripts remain under `scripts/` and `scripts/_archive/`, but the supported execution model is `python -m weightiz.cli...`.

# Repository Normalization Notes

## Canonical Code

- Canonical Python code now lives under `src/weightiz/`.
- Supported runners now live under `src/weightiz/cli/`.
- The real Module 6 is the decomposed research and portfolio engine under `src/weightiz/module6/`.

## Removed From Main

- Root-level `weightiz.cli.run_research` runner wrapper
- Root-level `weightiz.module5.worker_io_guard` stub
- Root-level canonical business-logic modules replaced by package locations under `src/weightiz/`
- Old dashboard-style Module 6 files and dashboard tests

## Server Execution Contract

- `pip install -e .`
- `python -m weightiz.cli.run_research --config configs/server/compute-small.yaml`
- `python -m weightiz.cli.run_module6 --config configs/module6/default.yaml --run-dir <module5-run-dir>`

## Runtime Paths

- `configs/` holds checked-in run profiles
- `artifacts/` holds run output
- `logs/` holds launcher records
- `tmp/` holds transient scratch output
- `infra/environment/server.env.example` documents server-oriented environment variables

## Temporary Compatibility

- `setup.py` remains as a minimal packaging compatibility shim for editable install environments that still expect it.

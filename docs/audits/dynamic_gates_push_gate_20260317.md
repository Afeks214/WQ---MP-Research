# Dynamic Gates Push Gate Lock (2026-03-17)

- source_branch: `main-mp`
- source_commits:
  - `2b08b28` weightiz: correct harness gate calibration seam and add regression guard
  - `4dfb8dc` weightiz: add forensic audit for dynamic gate closure
- destination_branch: `main-mp`

## Validation suites run
- `PYTHONPATH=src python3 -m pytest -q tests/test_gate_calibrator.py tests/test_module6_policy_contract.py tests/test_module6_e2e.py::test_run_research_fails_closed_when_module6_blocks tests/test_e2e_config_integrity.py` -> `22 passed`
- `PYTHONPATH=src python3 -m pytest -q tests/test_stage_a_cloud_campaign.py tests/test_canonical_single_path.py tests/test_architecture_pipeline.py tests/test_cli_server_paths.py` -> `14 passed`
- Representative config model validation (`RunConfigModel.model_validate` for canonical configs) -> all valid

## Push-set files
- `src/weightiz/cli/run_research.py`
- `src/weightiz/shared/config/builders.py`
- `src/weightiz/shared/config/models.py`
- `src/weightiz/shared/gate_calibrator.py`
- `tests/test_gate_calibrator.py`
- `docs/audits/dynamic_gates_forensic_audit_20260317.md`
- `docs/audits/dynamic_gates_push_gate_20260317.md`

## Declaration
- READY_TO_PUSH: **YES**
- No unrelated local changes are included in the committed push-set.

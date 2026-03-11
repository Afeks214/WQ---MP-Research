PYTHON ?= python3

.PHONY: install test packaging-smoke smoke-research

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

packaging-smoke:
	$(PYTHON) -m pytest -q tests/test_package_smoke.py

smoke-research:
	$(PYTHON) -m weightiz.cli.run_research --config configs/server/compute-small.yaml

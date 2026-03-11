PYTHON ?= python3

.PHONY: install packaging-smoke

install:
	$(PYTHON) -m pip install -e .

packaging-smoke:
	$(PYTHON) -m pytest -q tests/test_package_smoke.py

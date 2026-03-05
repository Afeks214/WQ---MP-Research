#!/usr/bin/env bash
set -euo pipefail

export PATH="$(pwd)/.venv/bin:$PATH"

if command -v python3 >/dev/null 2>&1; then
PYTHON_BIN=python3
else
echo "Python3 not found"
exit 1
fi

echo "Interpreter:"
$PYTHON_BIN --version

echo "File descriptor limit:"
ulimit -n

echo "Running compile gate..."

$PYTHON_BIN -m py_compile \
run_research.py \
sweep_runner.py \
strategy_engine.py \
weightiz_profile_engine.py

echo "Compile OK"

echo "Running pytest..."

$PYTHON_BIN -m pytest -q

echo "Pytest OK"

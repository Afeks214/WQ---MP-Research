#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/Weightiz-Research/WQ---MP-Research"
SESSION_NAME="weightiz_research"

cd "${REPO_ROOT}"

export PATH="${HOME}/miniforge3/bin:${HOME}/miniforge3/condabin:/opt/homebrew/opt/python@3.11/bin:${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

for fam in sprinters surfers snipers marathoners; do
  mkdir -p "artifacts/sweep_family_${fam}"
  if [ ! -f "configs/_generated/sweep_family_${fam}.yaml" ]; then
    echo "Missing config: configs/_generated/sweep_family_${fam}.yaml"
    exit 1
  fi
done

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" -n "families"
tmux split-window -h -t "${SESSION_NAME}:0"
tmux split-window -v -t "${SESSION_NAME}:0.0"
tmux split-window -v -t "${SESSION_NAME}:0.1"
tmux select-layout -t "${SESSION_NAME}:0" tiled

tmux send-keys -t "${SESSION_NAME}:0.0" "cd '${REPO_ROOT}' && python3 -u run_research.py --config configs/_generated/sweep_family_sprinters.yaml 2>&1 | tee -a artifacts/sweep_family_sprinters/run.log" C-m
tmux send-keys -t "${SESSION_NAME}:0.1" "cd '${REPO_ROOT}' && python3 -u run_research.py --config configs/_generated/sweep_family_surfers.yaml 2>&1 | tee -a artifacts/sweep_family_surfers/run.log" C-m
tmux send-keys -t "${SESSION_NAME}:0.2" "cd '${REPO_ROOT}' && python3 -u run_research.py --config configs/_generated/sweep_family_snipers.yaml 2>&1 | tee -a artifacts/sweep_family_snipers/run.log" C-m
tmux send-keys -t "${SESSION_NAME}:0.3" "cd '${REPO_ROOT}' && python3 -u run_research.py --config configs/_generated/sweep_family_marathoners.yaml 2>&1 | tee -a artifacts/sweep_family_marathoners/run.log" C-m

echo "tmux session started: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Pane map:"
echo "  0.0 sprinters"
echo "  0.1 surfers"
echo "  0.2 snipers"
echo "  0.3 marathoners"

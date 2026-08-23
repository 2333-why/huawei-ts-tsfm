#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== experiment lock =="
if [[ -d logs/.full_experiment.lock ]]; then
  echo "host=$(cat logs/.full_experiment.lock/hostname 2>/dev/null || echo legacy-or-unknown)"
  echo "pid=$(cat logs/.full_experiment.lock/pid 2>/dev/null || echo unknown)"
else
  echo "not present"
fi

echo "== processes =="
pgrep -af "run_all_datasets|luoyang_pipeline|ylj_pipeline|run_teacher|run_kd" || true

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader || true

LATEST_LOG="$(ls -1t logs/dual_dataset_full_run_*.log 2>/dev/null | head -1 || true)"
echo "== latest full-run log =="
echo "${LATEST_LOG:-not found}"
if [[ -n "$LATEST_LOG" ]]; then
  tail -n 80 "$LATEST_LOG"
fi

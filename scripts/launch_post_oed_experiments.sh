#!/usr/bin/env bash
# Launch the post-OED parametric experiments using whichever OED winner is
# already on disk. Usage:
#   bash scripts/launch_post_oed_experiments.sh <run_dir>
#
# Defaults to the gpt-5.5 codex run dir if no arg given. Generic — works
# for any OED run-dir as long as open_ended_discovery/history.json and
# baseline_metrics.json exist.
set -euo pipefail

RUN_DIR="${1:-/home/somasn/Desktop/cfd-scientist-arch-change/runs/open_ended_turbulence_model_codex}"
LOG="${2:-runs/post_oed_$(basename "$RUN_DIR").log}"

cd /home/somasn/Desktop/cfd-scientist-arch-change
mkdir -p "$(dirname "$LOG")"

echo "[launch] run_dir=$RUN_DIR  log=$LOG"
nohup python scripts/run_post_oed_experiments.py \
  --run-dir "$RUN_DIR" \
  --cases-dir-name paper_cases \
  > "$LOG" 2>&1 &
echo "Post-OED PID: $!"

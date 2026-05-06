#!/usr/bin/env bash
cd /home/somasn/Desktop/cfd-scientist-arch-change
mkdir -p runs
nohup python scripts/orchestrator_run.py \
  --topic "Investigate turbulence-model sensitivity in a 2D backward-facing step flow at Re_h=25,400, based on step height. Use a simple structured blockMesh with fewer than 10,000 cells, run the case in serial only, and compare several turbulence models such as k-epsilon, k-omega SST, and Spalart-Allmaras under the same flow conditions. For each model, identify and compare the reattachment length, pressure drop, and main recirculation-zone features, then summarize which model gives the most physically plausible separated-flow behavior and explain the observed trend." \
  --starter-dir /home/somasn/Desktop/cfd-scientist-arch-change/starter \
  --out-dir /home/somasn/Desktop/cfd-scientist-arch-change/runs/bfs_codex \
  --provider openai-codex --model gpt-5.4 \
  --no-ask-clarifications \
  > runs/bfs_codex.log 2>&1 &
echo "BFS Codex PID: $!"

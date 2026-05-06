#!/usr/bin/env bash
cd /home/somasn/Desktop/cfd-scientist-arch-change
mkdir -p runs
nohup python scripts/orchestrator_run.py \
  --topic "Open-ended discovery: find a novel SA model modification for periodic hill flow at Re=5600 that beats baseline SA and available literature on Cf prediction. Base case and DNS reference data are in the starter folder. Propose, implement, and test new model terms not in the literature." \
  --starter-dir /home/somasn/Desktop/cfd-scientist-arch-change/starter \
  --out-dir /home/somasn/Desktop/cfd-scientist-arch-change/runs/open_ended_turbulence_model_sonnet_46 \
  --open-ended-budget 20 \
  --provider claude-code --model claude-sonnet-4-6 \
  --no-ask-clarifications \
  > runs/oed_claude_code.log 2>&1 &
echo "OED Claude-Code PID: $!"

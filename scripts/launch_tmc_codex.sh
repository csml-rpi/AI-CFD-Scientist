#!/usr/bin/env bash
cd /home/somasn/Desktop/cfd-scientist-arch-change
mkdir -p runs
nohup python scripts/orchestrator_run.py \
  --topic "Implement a custom SA turbulence model for periodic hill flow using the equation and materials in the starter folder; baseline simulation files are provided in the starter folder. compare against the baseline SA model and reference data in starter. Cf should be the main comparison metric, you can use others as well as sub metrics" \
  --starter-dir /home/somasn/Desktop/cfd-scientist-arch-change/starter_turbulence_model_change \
  --out-dir /home/somasn/Desktop/cfd-scientist-arch-change/runs/turbulence_model_change_codex \
  --provider openai-codex --model gpt-5.4 \
  --no-ask-clarifications \
  > runs/tmc_codex.log 2>&1 &
echo "TMC Codex PID: $!"

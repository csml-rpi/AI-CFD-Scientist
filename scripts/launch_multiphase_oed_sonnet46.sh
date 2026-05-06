#!/usr/bin/env bash
cd /home/somasn/Desktop/cfd-scientist-arch-change
mkdir -p runs
nohup python scripts/orchestrator_run.py \
  --topic "Open-ended discovery: improve the built-in OpenFOAM Lagrangian droplet evaporation model for an isolated n-heptane droplet in hot quiescent nitrogen so that the predicted evaporation rate constant K agrees with the Verwey & Birouk (2023) fiber-free experimental correlation K0(T) = 3.6552e-4 * T[K] - 0.1078 [mm^2/s] across T_inf in {473, 573, 673, 873, 973} K. Propose, implement, and test novel blowing-corrected transfer-coefficient formulations that beat the built-in model and published baselines on mean over-prediction ratio across all five temperatures. Reference materials (starter case, experimental paper, built-in model source) are in the starter folder." \
  --starter-dir /home/somasn/Desktop/cfd-scientist-arch-change/starter_multiphase \
  --out-dir /home/somasn/Desktop/cfd-scientist-arch-change/runs/multiphase_droplet_evap_oed_sonnet46 \
  --open-ended-budget 10 \
  --provider claude-code --model claude-sonnet-4-6 \
  --no-ask-clarifications \
  > runs/multiphase_oed_sonnet46.log 2>&1 &
echo "Multiphase OED Sonnet-4.6 PID: $!"

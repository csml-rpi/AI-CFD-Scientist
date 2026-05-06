#!/bin/bash
set -e
# Prefer conda when available (matches LangChain/OpenFOAM agent stack), e.g.:
#   conda activate cfd-scientist && pip install -r requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Activate: source .venv/bin/activate (or: conda activate cfd-scientist)"

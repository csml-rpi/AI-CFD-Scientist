#!/usr/bin/env bash
# Copy this file, fill in your values, and source it before running: source env.sh

# LLM provider + model
export CFD_SCIENTIST_LLM_PROVIDER="bedrock"
export CFD_SCIENTIST_MODEL="arn:aws:bedrock:us-west-2:567316078106:inference-profile/us.anthropic.claude-sonnet-4-6" # sonnet 4.6
export CFD_SCIENTIST_VALIDATOR_MODEL="arn:aws:bedrock:us-west-2:567316078106:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0" # haiku 4.5

# AWS credentials (Bedrock)
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."
# export AWS_DEFAULT_REGION="us-west-2"

# Other providers (uncomment as needed)
# export CFD_SCIENTIST_LLM_PROVIDER="anthropic"
# export CFD_SCIENTIST_MODEL="claude-sonnet-4-6"
# export ANTHROPIC_API_KEY="..."

# export CFD_SCIENTIST_LLM_PROVIDER="openai"
# export CFD_SCIENTIST_MODEL="gpt-4o"
# export OPENAI_API_KEY="..."

# export CFD_SCIENTIST_LLM_PROVIDER="gemini"
# export CFD_SCIENTIST_MODEL="gemini-1.5-pro"
# export OPENAI_API_KEY="..."
# export OPENAI_BASE_URL="..."

# Paths (defaults shown; override if needed)
# export CFD_PROMPTS_PATH="./prompts/prompts.yaml"
# export FOAM_AGENT_MAIN="./Foam-Agent/foambench_main.py"
# export WM_PROJECT_DIR="/opt/openfoam10"

# Literature / ideation
# export S2_API_KEY="..."
# export BRAVE_SEARCH_API_KEY="..."
# export CFD_IDEATION_ENABLE_LITERATURE=1
# export CFD_IDEATION_MAX_PAPERS=40
# export CFD_IDEATION_MAX_EXPERIMENTS=10
# export CFD_IDEATION_NOVELTY_THRESHOLD=0.62

# Orchestration
# export CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL=50
# export CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT=10

# Tuning
export BEDROCK_READ_TIMEOUT=90
export CFD_IDEATION_MAX_EXPERIMENTS=3
export CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL=5

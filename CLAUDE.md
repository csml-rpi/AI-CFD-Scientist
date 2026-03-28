# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end agentic CFD research pipeline orchestrated by LangGraph: from a research topic to literature-aware ideation, OpenFOAM runs via Foam-Agent, interpreter diagnostics, cross-case analysis, and a LaTeX paper draft.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run full pipeline (no OpenFOAM execution)
cfd-scientist run-topic --topic "Your topic" --out-dir ./output

# Run with real OpenFOAM execution
cfd-scientist run-topic --topic "Your topic" --out-dir ./output --execute

# Ideation only
cfd-scientist ideate --topic "Your topic" --out ideation.json

# Resume after partial Foam-Agent runs
cfd-scientist resume-topic --out-dir ./output

# Restart from interpreter stage (skip ideation/hypothesis/foam)
cfd-scientist restart-topic --out-dir ./output

# Run as module (alternative to CLI)
python -m cfd_langgraph.workflow.main run-topic --topic "..." --out-dir ./output

# Tests (no framework configured; run individually)
python test_validation.py
python test_bedrock_model.py
```

## Architecture

### Pipeline Flow (LangGraph State Machine)

```
ideate → expand_and_init → [prepare_next_sim → generate_requirement → precheck ⇄ revise_requirement → foam_run → append_log]* → final_artifacts_gate → interpret_batch → [rerun loop]* → analysis_and_writer → save_pipeline_log
```

The graph uses `add_conditional_edges()` for routing decisions (loop/exit, valid/invalid, rerun needed, etc.).

### Key Modules

- **`src/cfd_langgraph/workflow/graph.py`** — `CFDWorkflow` class: builds the LangGraph state machine, defines all nodes and conditional routing. Central orchestration point.
- **`src/cfd_langgraph/workflow/main.py`** — CLI entry point (`main()`), argument parsing, command dispatch.
- **`src/cfd_langgraph/agents/`** — Each agent wraps an LLM with domain-specific prompting. Key agents: `HypothesisAgent` (idea→Foam-Agent prompt with validation loop), `InterpreterAgent` (PyVista viz + rerun decisions), `AnalysisAgent` (cross-case figures), `WriterAgent` (LaTeX generation).
- **`src/cfd_langgraph/llm/factory.py`** — Multi-provider LLM factory. Supports Bedrock, OpenAI, Anthropic, Gemini. Provider selected via `CFD_SCIENTIST_LLM_PROVIDER` env var or inferred from model string.
- **`src/cfd_langgraph/foam/runner.py`** — Runs Foam-Agent as a subprocess with timeout (default 2h), streams stdout/stderr, passes env vars through.
- **`src/cfd_langgraph/config.py`** — `Settings` dataclass populated from environment variables.
- **`src/cfd_langgraph/prompts/loader.py`** — Loads prompt templates from `prompts/prompts.yaml`.
- **`src/cfd_langgraph/ideation.py`** — Literature-aware idea generation with novelty gating (string similarity threshold).
- **`src/cfd_langgraph/literature.py`** — Semantic Scholar + OpenAlex API clients.
- **`src/cfd_langgraph/viz_creator.py`** — Generates PyVista/matplotlib visualization scripts for CFD results.

### Agent Pattern

Each agent: takes `(model: str, prompt_loader: PromptLoader)`, creates LLM via `create_langchain_llm()`, uses `ChatPromptTemplate` for structured prompting, returns domain-specific outputs.

### State

`WorkflowState` (TypedDict) carries everything between nodes: topic, simulations list, current sim index, hypothesis text, run results, interpreter output, rerun queue, and `pipeline_log` dict.

## Environment Variables

**Required:**
- `CFD_SCIENTIST_LLM_PROVIDER` — `bedrock`, `openai`, `anthropic`, `gemini`, or `openai-codex`
- `CFD_SCIENTIST_MODEL` — model identifier or Bedrock ARN
- Provider-specific auth (AWS creds for Bedrock, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

**Paths:**
- `CFD_PROMPTS_PATH` — prompt YAML (default: `./prompts/prompts.yaml`)
- `FOAM_AGENT_MAIN` — Foam-Agent script (default: `./Foam-Agent/foambench_main.py`)
- `WM_PROJECT_DIR` — OpenFOAM install (default: `/opt/openfoam10`)

**Tuning:**
- `CFD_IDEATION_MAX_EXPERIMENTS` (default 10), `CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL` (default 50)
- `CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT` (default 10)
- `CFD_IDEATION_NOVELTY_THRESHOLD` (default 0.62)
- `BEDROCK_READ_TIMEOUT` (default 300s)

## Key Patterns

- **Validation loop**: `HypothesisAgent.generate_validated_requirement()` generates a Foam-Agent prompt, runs LLM semantic QA (solver consistency, BCs, no viz text), repairs if invalid, retries up to 3 times.
- **Rerun loop**: Interpreter checks viz quality + requirement satisfaction; if rerun needed, `RerunAnalysisAgent` revises the requirement, re-queues for Foam-Agent (up to N rounds).
- **Vision LLM**: Interpreter and analysis agents can base64-encode PyVista PNGs inline for vision-capable models.
- **Token tracking**: Global `TokenStatsCallbackHandler` in `llm/token_stats.py` accumulates usage across all providers.
- **Foam-Agent env mapping**: `CFD_SCIENTIST_LLM_PROVIDER` maps to `FOAMAGENT_MODEL_PROVIDER`; `CFD_SCIENTIST_MODEL` maps to `FOAMAGENT_MODEL_VERSION`.

# cfd-scientist-langchain

LangChain modular re-architecture of `~/Documents/cfd-scientist` with the **same agent roles**, prompt source, and Foam-Agent integration points.

## Goals
- Preserve existing behavior/prompting from `cfd-scientist/prompts/prompts.yaml`
- Modularize into composable agents and workflow orchestration
- Add literature-aware ideation (Semantic Scholar + optional Brave web search)
- Keep execution optional (no auto-run in this scaffold)

## Agent parity
- IdeationAgent
- HypothesisAgent
- AnalysisAgent
- WriterAgent
- LiteratureSurveyAgent (Semantic Scholar + web search)

## Prompt parity
By default, this project reads prompts from:
- `/home/somasn/Documents/openclaw/2026-02-26/cfd-scientist-langchain/prompts/prompts.yaml`

You can override with env var:
- `CFD_PROMPTS_PATH=/path/to/prompts.yaml`

## Foam-Agent parity
Runner points to:
- `~/Documents/cfd-scientist/Foam-Agent/foambench_main.py`
(or override via `FOAM_AGENT_MAIN` env var)

## Install
```bash
cd /home/somasn/Documents/openclaw/2026-02-26/cfd-scientist-langchain
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
```bash
export CFD_SCIENTIST_MODEL="us.anthropic.claude-sonnet-4-20250514-v1:0"
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Codex provider (API only)
Use OpenAI API-key auth for codex models.

```bash
export CFD_SCIENTIST_MODEL="codex/gpt-5-codex"
export OPENAI_API_KEY="..."
```

You can also set the model directly without the `codex/` prefix:

```bash
export CFD_SCIENTIST_MODEL="gpt-5-codex"
export OPENAI_API_KEY="..."
```

Optional:
```bash
export CFD_PROMPTS_PATH="/home/somasn/Documents/openclaw/2026-02-26/cfd-scientist-langchain/prompts/prompts.yaml"
export FOAM_AGENT_MAIN="/home/somasn/Documents/openclaw/2026-02-26/cfd-scientist-langchain/Foam-Agent/foambench_main.py"
export S2_API_KEY="your_semantic_scholar_api_key"      # literature agent
export BRAVE_SEARCH_API_KEY="your_brave_search_api_key" # optional web supplement
```

## CLI examples
```bash
python -m cfd_langchain.workflow.main ideate \
  --topic "Study the effect of fuel velocity and inlet box sizes in 2D small pool fire." \
  --out ideation_with_lit.json

python -m cfd_langchain.workflow.main run-topic \
  --topic "Your CFD research topic" \
  --out-dir ./runs/topic_run_001
# add --execute to actually run Foam-Agent
# add --allow-non-executed-artifacts to generate analysis/paper without --execute
```

Current implemented workflow commands:
- `ideate` (literature-aware): retrieves prior studies first, then generates idea JSON.
- `run-topic`: end-to-end flow (topic -> ideation -> hypothesis+LLM validation/repair -> Foam-Agent run/plan -> interpreter rerun loop -> analysis -> writer).

Env knobs for ideation:
```bash
export CFD_IDEATION_ENABLE_LITERATURE=1
export CFD_IDEATION_MAX_PAPERS=12
export CFD_IDEATION_MAX_WEB_RESULTS=5
export CFD_IDEATION_MAX_EXPERIMENTS=50
export CFD_IDEATION_NOVELTY_THRESHOLD=0.62
export CFD_IDEATION_NOVELTY_MAX_RETRIES=3
export CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL=50
export CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT=2
export S2_API_KEY="..."              # Semantic Scholar
export BRAVE_SEARCH_API_KEY="..."    # Optional web supplement
```

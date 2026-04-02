# CFD Scientist — Findings & Issues Log

A running log of bugs, bottlenecks, and findings discovered during development and test runs.
Add new entries at the top (newest first).

---

## 2026-04-02 — Hypothesis stage hangs on Bedrock (slow / no timeout) ✓ Fixed

**Symptom:** `cfd-scientist run-topic` stalls at `[CFD-WORKFLOW] START hypothesis :: exp_001` for many minutes with no output.

**Root cause:** Two full LLM calls are made per experiment during hypothesis generation:
1. `generate_user_requirement` — generates the Foam-Agent prompt
2. `llm_validate_requirement` — LLM-based QA check (solver consistency, BCs, time controls, etc.)

Both use the same large model (Sonnet 4.6 via Bedrock cross-region inference profile). `BEDROCK_READ_TIMEOUT` defaults to 300s, so a throttled or slow Bedrock call blocks for up to 5 minutes before timing out. With 3+ experiments this can add up to 15–30 min before any progress.

**Files involved:**
- `src/cfd_langgraph/agents/hypothesis_agent.py:104` — `llm_validate_requirement` call
- `src/cfd_langgraph/llm/factory.py:15` — `_bedrock_read_timeout()` default of 300s

**Workaround:** Lower the read timeout at runtime:
```bash
BEDROCK_READ_TIMEOUT=90 cfd-scientist run-topic --topic "..." --out-dir ./output --execute
```

**Fix implemented (2026-04-02):**
- Added `CFD_SCIENTIST_VALIDATOR_MODEL` env var (`config.py`, `hypothesis_agent.py`, `graph.py`)
- When set, the validate+repair loop uses this lighter model; generation still uses the main model
- Recommended: set to Haiku ARN for fast validation

**Still open:**
- Add streaming to hypothesis LLM calls so progress is visible
- Retry with exponential backoff on `ThrottlingException` rather than waiting on a stalled socket

---

## 2026-03-28 — OpenFOAM 10 API incompatibilities cause high solver failure rate

**Symptom:** Only 1 of 10 experiments completed solver run (exp_002). Others failed at startup or ran only briefly.

**Root cause:** Multiple OpenFOAM 10 API changes not in Foam-Agent's knowledge base:
- `fireFoam` solver deprecated → replaced by `buoyantReactingFoam`
- `reactingMixture` → `multiComponentMixture`
- `method standard` → `method chemistryModel` in `chemistryProperties`
- Reactions dict syntax: list `()` → dictionary `{}`
- `calculated` BC incompatible for pressure field

Foam-Agent's review loop catches and fixes these iteratively, but the 25-iteration limit is sometimes exhausted before all issues in a case are resolved simultaneously.

**Files involved:**
- `src/cfd_langgraph/foam/runner.py` — subprocess runner with timeout
- Foam-Agent submodule knowledge base (no OpenFOAM 10 Foundation changelog)

**Workaround:** None automatic. Foam-Agent sometimes resolves all issues within the retry budget (proven by exp_002 succeeding).

**Potential fixes (not yet implemented):**
- Prepend OpenFOAM 10 migration notes to Foam-Agent's system prompt
- Apply all config fixes atomically (not file-by-file) to avoid cascading failures
- Increase Foam-Agent review loop iteration limit

---

## 2026-03-28 — Analysis & Paper stages skipped when `--execute` not passed

**Symptom:** Pipeline runs through interpreter but then stops; analysis and writer agents never run.

**Root cause:** `final_artifacts_gate` node requires `--execute` flag to be set. Without it, the artifact gate blocks the transition to `interpret_batch` and onward stages.

**Files involved:**
- `src/cfd_langgraph/workflow/graph.py` — `final_artifacts_gate` conditional routing

**Workaround:** Always pass `--execute` when a full end-to-end run is intended. Use `cfd-scientist restart-topic` to re-enter the pipeline from the interpreter stage without re-running Foam-Agent.

---

## 2026-03-28 — Bedrock vision timeouts in interpreter stage

**Symptom:** `InterpreterAgent` LLM calls hit read timeouts when base64-encoded PyVista PNGs are included inline.

**Root cause:** Vision payloads are large; Bedrock cross-region inference profiles add latency on top of model inference time. Default 300s timeout is sometimes insufficient.

**Files involved:**
- `src/cfd_langgraph/agents/` — interpreter and analysis agents
- `src/cfd_langgraph/llm/factory.py:15` — `BEDROCK_READ_TIMEOUT`

**Workaround:** Increase timeout: `BEDROCK_READ_TIMEOUT=600`

---

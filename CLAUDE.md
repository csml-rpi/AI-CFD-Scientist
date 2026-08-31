# CFD Scientist

## Environment — always activate first
**Recommended:** `cfd-scientist` conda env (correct LangChain / project deps):
```bash
conda activate cfd-scientist
```
Alternative (venv from repo): `source .venv/bin/activate`

## Model providers

`CFD_SCIENTIST_LLM_PROVIDER` + `CFD_SCIENTIST_MODEL` select the model for the CLI
and every script it shells out to. All of these can drive the manager (they
implement `bind_tools`; `deep_agent.py` refuses one that does not).

```bash
# API-billed
export CFD_SCIENTIST_LLM_PROVIDER=gemini    CFD_SCIENTIST_MODEL=gemini-3.7-flash   # + GOOGLE_GENAI_USE_VERTEXAI=true
export CFD_SCIENTIST_LLM_PROVIDER=bedrock   CFD_SCIENTIST_MODEL=us.anthropic.claude-sonnet-4-6

# subscription-billed, no API key — uses the local CLI's own OAuth cache
export CFD_SCIENTIST_LLM_PROVIDER=claude-code   CFD_SCIENTIST_MODEL=claude-sonnet-4-6   # ~/.claude, via claude-agent-sdk
export CFD_SCIENTIST_LLM_PROVIDER=openai-codex  CFD_SCIENTIST_MODEL=codex               # ~/.codex/auth.json
```

`CFD_SCIENTIST_EFFORT` sets reasoning effort for every stage. The accepted set
differs by provider and an unsupported value is refused rather than silently
downgraded: `openai-codex` takes none/minimal/low/medium/high/xhigh,
`claude-code` takes low/medium/high/max.

`CFD_SCIENTIST_MODEL=codex` resolves to whatever `~/.codex/config.toml` sets, so
it tracks the Codex CLI rather than pinning a name. Sign in with `claude` /
`codex login`; an expired token is reported up front, not as an HTTP 401.

Both subscription providers use native tool calling (Claude via an in-process
SDK MCP server, Codex via Responses `tools`), so neither depends on parsing
tool calls out of prose. One difference worth knowing: Codex issues parallel
tool calls, `claude-code` issues one per turn, so manager fan-outs serialize on
that provider.

## FoamAgent

The execution loop (parse, RAG, decompose, write, Allrun, run, review/retry) is
ported into `src/cfd_langgraph/foam_native/`. Nothing on the CLI path imports
the vendored `Foam-Agent/` package any more — verified by tracing imports across
every script the CLI shells out to.

What remains is *data*: the prebuilt FAISS tutorial indices. Point at them with

```bash
export CFD_SCIENTIST_FAISS_DIR=/path/to/faiss     # default: <repo>/database/faiss
```

falling back to `Foam-Agent/database/faiss/` if neither exists. Copy the ~34 MB
`Qwen_Qwen3-Embedding-0.6B/` index into `<repo>/database/faiss/` and the
`Foam-Agent` symlink can go. Without any index, case writing still works — it
degrades to reading a tutorial from `$WM_PROJECT_DIR/tutorials/`.

Two legacy execution paths were removed rather than ported: the
`run_case_scripted` tool, and the OED class-derivation parameter sweep (which
now refuses explicitly instead of running an unmodified case and reporting a
score for parameters it never applied). Runtime-coefficient models — what every
current proposer emits — are unaffected.

## Routing

The skill recipes live under `cfd-skills/`. After the Nov 2026 ARIS/DS-style refactor every recipe is self-contained — expert prompts from `prompts/prompts.yaml` are embedded verbatim, and scripts are an optional fast-path rather than the primary mechanism. The `skills/` dir keeps `cfd-orchestrator` (router), `cfd-foamagent-runtime` (canonical FoamAgent execution contract), `cfd-mesh-independence` (deeper protocol commentary), plus thin aliases at `cfd-research` and `cfd-code-mod`.

- any new topic/user request
  → `skills/cfd-orchestrator/SKILL.md` FIRST (top-level router)

- study/sensitivity/turbulence/backward step/channel/experiment/run
  → `cfd-skills/cfd-pipeline/SKILL.md` (general path)

- implement/modify/viscosity model/turbulence model change/Bingham/
  power law/Carreau/custom model/source term/fvOption
  → `cfd-skills/cfd-code-modify/SKILL.md` FIRST, then continue from Step 9 in `cfd-skills/cfd-pipeline/SKILL.md`

- mesh independence/GCI/grid convergence/Richardson
  → `cfd-skills/cfd-mesh-gate/SKILL.md` (protocol embedded; deeper commentary in `skills/cfd-mesh-independence/SKILL.md`)

- analyze/plot/post-process/visualize results
  → `cfd-skills/cfd-analyze/SKILL.md` (text + tables); follow with `cfd-skills/cfd-viz/SKILL.md mode=full` if figures are wanted

- write paper/LaTeX/manuscript
  → `cfd-skills/cfd-paper/SKILL.md`

- open-ended discovery / find a novel model / beat baseline
  → `cfd-skills/cfd-open-discovery/SKILL.md`

For all simulation execution, the FoamAgent contract remains canonical:
  → `skills/cfd-foamagent-runtime/SKILL.md` (wraps `scripts/foam_run.py`)

## Required orchestration behavior
- Topic-first flow: literature (Semantic Scholar with `S2_API_KEY`, respect user max papers), then idea/hypothesis generation.
- If request is simulation-only: generate exactly user-requested number of FoamAgent requirements and run all.
- After each run, interpreter must create PyVista visuals and decide PROCEED vs RERUN/REVISE.
- For reruns, prioritize similar successful cases and revise only the failing case requirement/config while preserving study intent.
- For source-code/model-change requests, follow the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol embedded verbatim inside `cfd-skills/cfd-code-modify/SKILL.md` (full text mirrored from `openfoam_literature_change_agent_prompt_v2.txt`).
- For code-mod path, never edit OpenFOAM installation/source dirs; create/compile only case-local custom libraries in `{case}/customModels/`.
- For mesh-independence requests, follow `cfd-skills/cfd-mesh-gate/SKILL.md` (STEPS A–E embedded: near-wall ~10%, away-from-wall ~5%, same physics/numerics, QoI/y+/quality comparison, 5% threshold assessment, escalate to GCI if needed).
- Mesh-independence is mandatory for all study types: start from a base mesh, refine and compare, continue until QoIs stabilize, then use that selected mesh for all further simulations (including code-mod studies).
- Preserve existing interpreter/analysis/viz/writer prompt behavior from `prompts/prompts.yaml` in the skill-based orchestration. The YAML is the authoritative source; skills embed verbatim copies for self-containment.
- Foam-Agent workflow is executed via the FoamAgent runtime skill (see `skills/cfd-foamagent-runtime/SKILL.md`) so orchestration is skill-driven while preserving Foam-Agent RAG/planning/writing/review prompts and sequence.
- Long CFD runs are expected; use long wait windows and avoid premature timeout decisions.
- For very slow/timeout cases, apply conservative CFL-aware timestep tuning (small `deltaT` increase with `adjustTimeStep` and bounded `maxCo`) before declaring failure.
- Once all experiments are complete, run cross-case analysis (with viz creator) and then writer agent for manuscript drafting/review.

## Two execution modes

CFD Scientist can be driven two ways. Both share artifact contracts (`lit.json`, `requirements.json`, `selected_mesh_spec.json`, `analysis.json`, etc.) and the same expert prompts.

- **Skill-driven (this file's routing path)** — every recipe lives in markdown SKILLs under `cfd-skills/` and `skills/`. Self-contained, framework-agnostic, ARIS/DS-style. Run from any LLM agent (Claude Code, Cursor, Codex CLI). Scripts are an *optional* fast-path inside each skill; the agent recipe is the primary path.
- **LangGraph orchestrator (Python pipeline)** — `python scripts/orchestrator_run.py …` runs the same pipeline end-to-end with checkpointing and resume. The Python source under `scripts/` and `src/cfd_langgraph/` is unchanged by the skill refactor.

Both modes can be mixed (Hybrid). See top-level `README.md`.

## Key paths
| Purpose | Path |
|---|---|
| Skill recipes (per-stage, ARIS/DS-style)   | `cfd-skills/`            |
| Skill router + FoamAgent runtime + aliases | `skills/`                |
| LangGraph pipeline scripts                 | `scripts/`               |
| LangGraph agents source                    | `src/cfd_langgraph/`     |
| Run outputs                                | `runs/`                  |
| Foam-Agent vendored framework              | `Foam-Agent/`            |
| Expert prompts (authoritative)             | `prompts/prompts.yaml`   |
| OPENFOAM 10 code-mod protocol (mirrored)   | `openfoam_literature_change_agent_prompt_v2.txt` |
| Pipeline reference                         | `AGENTS.md`              |
| Project README + install                   | `README.md`              |

## Pipeline reference
See `AGENTS.md` for the complete orchestrator-agnostic stage-by-stage pipeline.

For end-user-facing install + usage docs, see `README.md` at the repo root.

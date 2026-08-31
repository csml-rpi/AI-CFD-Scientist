# CFD Scientist

End-to-end **agentic CFD research pipeline** — from a research topic to literature-aware ideation, mesh-independence-checked OpenFOAM runs (via Foam-Agent), interpreter checks with diagnostics, cross-case analysis, and a LaTeX paper draft.

Run it three ways:

- **Interactive CLI (recommended)** — `cfd-scientist-cli`. A [deepagents](https://github.com/langchain-ai/deepagents) manager agent on LangGraph drives the whole study, delegating work to subagents. You can talk to it mid-run, Ctrl-C pauses cleanly at the next tool-call boundary, and every study is SQLite-checkpointed so it resumes exactly where it stopped.
- **LangGraph Python orchestrator** — one command, hands-off, checkpointed. Fixed stage sequence, no conversation.
- **Skill-driven (ARIS / DeepScientist style)** — markdown SKILLs invokable from any LLM agent (Claude Code, Cursor, Codex CLI, custom). Self-contained: every skill embeds its expert prompts verbatim and walks the agent end-to-end.

All three share artifact contracts (`lit.json`, `requirements.json`, `selected_mesh_spec.json`, `analysis.json`, etc.), so a partial run from one can be picked up by another.

---

## Pipeline overview

1. **User topic** → research question in CFD.
2. **Literature** (Semantic Scholar ± OpenAlex / arXiv / web) → `lit.json`.
3. **Hypothesis / Ideation** → testable hypotheses + study skeleton.
4. **Requirements** → one Foam-Agent user-requirement string per case → `requirements.json`.
5. **Baseline + metric setup** → run unmodified base case; LLM-author + verify the comparator script that scores all metrics.
6. **Mesh-independence gate** (mandatory) → per physics group: baseline + refined mesh, 5% threshold (10% near-wall), escalate to Richardson/GCI when needed → `selected_mesh_spec.json`.
7. **Foam-Agent runs** → planner + RAG + input writer + reviewer loop, one case at a time, with CFL-aware retries.
8. **Interpreter** → PyVista figures + vision-LLM call → `decision.json` (PROCEED / REVISE / RERUN).
9. **Cross-case analysis** → QoI table, trends, correlations, conclusions → `analysis.json`.
10. **Paper writer** → planner → batch PyVista figures with VLM QA → LaTeX draft → reviewer loop (up to 10 iterations).
11. **Stage-gate audit** → `scripts/stage_gate_audit.py` verifies every gate; the run is complete only once it writes a signed `audit_passed.json`. See [Verifying a run](#verifying-a-run--the-stage-gate-audit).

For open-ended model discovery, steps 3–9 are replaced by a search loop: propose candidates from a quality-diversity archive → build and evaluate each on every graded case → score → record → repeat until the budget is spent or the archive saturates. See [Open-ended discovery](#open-ended-discovery-how-the-search-works).

The orchestrator-agnostic stage-by-stage spec is in [`AGENTS.md`](AGENTS.md). The Claude Code / agent routing rules are in [`CLAUDE.md`](CLAUDE.md). The skill recipes are documented in [`cfd-skills/README.md`](cfd-skills/README.md).

---

## Modes — pick what fits

### Mode A — Interactive CLI (recommended)

A manager agent built with **deepagents** on **LangGraph** runs the study and calls tools;
you supervise it in a terminal UI. Start a study or resume one:

```bash
conda activate cfd-scientist

# New study
cfd-scientist-cli run \
  --topic "Improve k-omega-SST separation prediction on periodic hills, \
           starting from starter_oed_turbulence/" \
  --out-dir runs/my_study

# Long or multi-paragraph topic — read it verbatim from a file
cfd-scientist-cli run --topic-file my_topic.md --out-dir runs/my_study

# Resume — picks up from the LangGraph checkpoint, nothing repeated
cfd-scientist-cli resume --out-dir runs/my_study
```

Without `pip install -e .`, use the thin entrypoint instead:
`python scripts/cfd_cli.py run --topic ... --out-dir ...`

There is no `--starter-dir`: name the starter folder in the topic text and the manager reads
it from there. The bundled starters (each with reference data) are `starter_viscosity_change/`,
`starter_turbulence_model_change/`, `starter_oed_turbulence/` and `starter_multiphase/`; omit
any mention to let FoamAgent generate the base case from scratch.

Other flags: `--num-candidates` (candidates per discovery round) and `--max-parallel-cases`
(override the auto-calibrated concurrency cap). Omit `--topic` to be prompted for it; omit
`--out-dir` for `runs/study_<timestamp>`.

**What the CLI gives you over the non-interactive modes:**

- **Talk to it mid-run.** Type at the prompt at any time; the manager picks it up at the next
  step. Redirect the search, add a constraint, ask what it is doing.
- **Clean interrupts.** Ctrl-C (or `esc`) pauses at the next *tool-call boundary* — whatever is
  executing finishes normally, so work in flight is kept. A second Ctrl-C force-quits.
- **Durable resume.** Every step is checkpointed to `state/checkpoints.sqlite`
  (`langgraph-checkpoint-sqlite`), so `resume` continues from where it stopped.
- **A human approval gate on hypotheses** — see [Hypothesis generation](#hypothesis-generation-and-the-approval-gate).
- **Real concurrency.** Independent cases run as concurrent subagent calls, scheduled against a
  hardware-calibrated limit by a shared coordinator.

### Mode B — LangGraph Python orchestrator (unattended, fixed sequence)

One command, end-to-end, with checkpointing and `--resume-from`. This is what you want for multi-hour production runs.

```bash
conda activate cfd-scientist

# Full pipeline
python scripts/orchestrator_run.py \
  --topic "LES of backward-facing step at Re=5100, compare against Le-Moin DNS" \
  --out-dir runs/bfs_les \
  --provider claude-code --model claude-sonnet-4-6 \
  --starter-dir starter_turbulence_model_change
# --starter-dir is optional. The bundled starters (each with reference data) are
# starter_viscosity_change/, starter_turbulence_model_change/,
# starter_oed_turbulence/, starter_multiphase/. Omit it to let FoamAgent
# generate the base case from scratch.

# Resume from a specific stage (literature, hypothesis, requirements, code_mod,
# mesh_gate_resume, baseline_synthesis, experiments, analysis, paper_review,
# reference_verify, analysis_without_viz_full)
python scripts/orchestrator_run.py --topic "..." --out-dir runs/bfs_les \
  --resume-from paper_review

# Or use the legacy CLI
cfd-scientist run-topic \
  --topic "Lid-driven cavity at Re=100 and Re=400" \
  --out-dir ./runs/cavity \
  --execute
```

Use this when you want one command to do everything, automatic resume on failure, and hands-off long runs.

### Mode C — Skill-driven (ARIS / DeepScientist style)

Invoke individual skills from any LLM agent. Each `cfd-skills/cfd-<stage>/SKILL.md` is self-contained: the expert prompts (HypothesisAgent, IdeationAgent, ResultsInterpreterAgent, WriterAgent, PaperReviewerAgent, RunValidityAgent, MetricProposer, ComparatorAuthor, ComparatorVerifier, MetricSetupAgent, MetricSetupVerifier from `prompts/prompts.yaml`, plus the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol) are embedded verbatim. Scripts are an *optional fast-path*; the agent recipe is primary.

```text
# Top-level chain — start at the router; it picks the route and walks the whole
# chain (literature → … → paper → stage-gate audit) to an audited paper.
/cfd-orchestrator topic="LES of backward-facing step Re=5100" out-dir=runs/bfs_skill

# Or stage-by-stage
/cfd-literature   topic="..." out-dir=runs/bfs_skill
/cfd-hypothesis   out-dir=runs/bfs_skill
/cfd-requirements out-dir=runs/bfs_skill n_cases=4
/cfd-mesh-gate    out-dir=runs/bfs_skill
/cfd-experiment   out-dir=runs/bfs_skill case_id=case_001
/cfd-interpret    out-dir=runs/bfs_skill case_id=case_001
/cfd-analyze      out-dir=runs/bfs_skill
/cfd-paper        out-dir=runs/bfs_skill

# Code modification + study
/cfd-code-modify  out-dir=runs/bingham case_path=runs/bingham/case_001
# then: /cfd-mesh-gate, /cfd-experiment, etc.

# Open-ended model discovery
/cfd-open-discovery out-dir=runs/oed topic="novel SA mod for periodic hill Re=5600 beating baseline on Cf" \
  starter-dir=starter_oed_turbulence budget=20
```

Use this when you want manual control, are integrating into another agent framework, or want to run only part of the pipeline ad-hoc. See [`cfd-skills/README.md`](cfd-skills/README.md) for the complete skill catalog and contracts.

### Mode D — Hybrid

Run Mode A or B for the main pipeline; invoke skills (Mode C) ad-hoc against the same `out-dir`. Because every mode reads and writes the same JSON contracts:

- The orchestrator stopped at `analysis_done`? Invoke `/cfd-paper` against the same `out-dir`.
- Use a skill to hand-craft one stage's output, then resume the orchestrator with `--resume-from <next_stage>`.
- Use the orchestrator for long unattended runs; use skills for interactive exploration on the same artifacts.

---

## How the agent is built

The interactive CLI is a **deepagents** graph (`create_deep_agent`) running on **LangGraph**,
with SQLite checkpointing. One *manager* agent owns the study and holds the planning tools;
the heavy, repetitive work goes to *subagents*, each with its own isolated context window.

| Component | Role |
|---|---|
| **manager** | Owns the study: literature → hypotheses → requirements → mesh gate → experiments → analysis → paper. Holds the planning and bookkeeping tools. |
| **`case-runner`** subagent | Runs exactly one CFD case end-to-end (write → Allrun → solve → review) and reports back. |
| **`oed-candidate-runner`** subagent | Runs exactly one discovery candidate: compile a model, evaluate it on every graded case, score it. |

The manager fans out N `task` calls in one message, so N cases or candidates run
**concurrently**, each in its own context window. Every subagent gets prompt-caching middleware
and the same interrupt wiring as the manager.

**Hardware-aware concurrency.** A shared `CaseCoordinator` runs the first case of a physics group
alone to benchmark it, then computes
`min(cores available / cores per case, memory available / memory per case)` with a safety margin
and schedules every later case against that limit. The cap is enforced by the coordinator, so it
holds however many calls the model issues.

**Scoped filesystem access.** Subagents work through purpose-built tools that validate every
path rather than general read/write tools, so graded benchmark inputs stay read-only and each
candidate writes only inside its own directory.

---

## Hypothesis generation and the approval gate

The hypothesis stage does more than ask a model for ideas:

1. **Literature-grounded ideation** — the study objective is distilled into a bibliographic
   search query (the physics question, separated from the paths, metric names and case counts
   that belong to the pipeline), papers are retrieved, and ideas are generated against them.
2. **Critique** — each idea is checked for physical plausibility and implementability.
3. **Ranking** — survivors are ranked and written to `hypotheses_ranked.json`.
4. **Human approval gate** — the manager proposes a shortlist via
   `advance_with_approved_hypotheses` and the CLI **interrupts**, so you can approve the
   shortlist, edit it down, or send it back. The requirements and experiments stages read
   `hypotheses_approved.json`, so the study runs the experiments you signed off on.

---

## Open-ended discovery: how the search works

For "find me a better model" studies the pipeline replaces steps 3–9 with a search loop the
manager drives directly, one tool call at a time, so it stays interruptible and checkpointed
throughout.

**A quality-diversity archive.** Candidates are niched on a 2-D grid of **mechanism family ×
strategy**, so the search covers distinct ideas rather than concentrating on whichever one is
currently ahead. Selection is a PUCT-style rule over the archive — exploit the best-scoring
niche, plus an exploration bonus that decays with visits — with staleness damping on niches that
stop improving and an exploration floor that opens a new family at a set interval. Each niche
keeps its elite, and new candidates are proposed conditioned on it, so a promising mechanism is
refined rather than rediscovered.

**Strategy is a searched dimension, not a prompt assertion.** *How* a model's coefficients are
arrived at is niched alongside *what* the model changes. The four buckets:

**`analytic`** — derive the functional form and choose every coefficient by reasoning from
physics, then compile once and evaluate. No data is fitted; the coefficients are literals in the
source, so every number in the model has an argument behind it. One compile, one pass over the
graded cases.

**`sweep`** — take an already-compiled model that exposes its coefficients at runtime and vary
them across runs, keeping the best. No recompile between trials, so a trial costs one solve. Maps
the objective along a coefficient axis and establishes how much leverage that coefficient has.

**`solver_fit`** — *a-posteriori*, or solver-in-the-loop. The coefficients are chosen by an
optimiser that calls the solver: each objective evaluation runs a subset of cases to convergence
and scores them with the study's own metric, and the optimiser searches for the coefficient
vector that minimises it. This optimises the quantity the study is graded on directly. The
candidate's plan names the optimiser and the case subset, and the agent sizes the search against
its measured per-case cost and its wall-clock budget before it starts.

**`offline_fit`** — *a-priori*. The correction is fitted to stored high-fidelity fields (LES/DNS
velocity, turbulent kinetic energy, Reynolds stresses) **before** it meets the solver:
regression, symbolic regression, or a small network mapping local flow invariants to the
correction. The fitted form is then compiled into the model and evaluated. Objective evaluations
read stored fields rather than running CFD, so a large fit is affordable — tens of thousands of
sampled cells across every training case, in seconds. The plan names which stored fields are the
target and what is being fitted.

Because strategy is part of the niche key, the archive *measures* which approach pays off on a
given problem instead of the prompt asserting one, and each strategy is time-fenced against its
own kind so cheap and expensive approaches can compete on equal terms.

**Every candidate is scored on the full declared case set**, so scores are comparable across the
archive. Candidates are evaluated on private replicas of the graded cases, never on the cases
themselves, and the graded cases stay pristine.

**Cost is measured.** Solver launches are counted per candidate and charged against the study
budget, so the archive knows what each idea actually cost.

### Candidate lifecycle management

Model-building agents run for a long time and can be stopped by a wall clock, a turn cap or a
provider outage. The loop manages that lifecycle:

- **Per-strategy time fencing.** A candidate's wall-clock budget is an outlier bound over the
  durations candidates of *that strategy* have actually taken, so each approach is measured
  against its own kind and a slow strategy gets the time its work needs.
- **The agent knows its deadline.** It is given its wall-clock budget up front and sizes any fit
  against it — evaluations × cases × measured seconds per case — scaling the work to fit before
  it starts.
- **Diagnosis before acceptance.** An agent that ends early is reviewed by a model that reads the
  build result, the compiled model source, the case's activation dictionary and the fit's own
  ledger. It returns one of `complete` / `repair` / `extend` / `abandon` together with a
  judgement on whether the compiled model is finished, and that verdict governs what happens
  next — including refusing to score a model it judges incomplete.
- **Repair and extension.** Two of each per candidate, recorded before the work runs. A repair
  carries out diagnosed steps against the candidate's own files under a hard prohibition on
  touching the mesh, boundary conditions, physics, endTime or the closure under test. An
  extension continues the previous attempt, with the earlier work on disk to build on, rather
  than restarting it.
- **Bounded execution.** Shell commands run in a bubblewrap sandbox with a read-only host, a
  writable candidate directory and its own PID namespace, so everything a command starts is
  bounded by the call that started it.

---

## Install

### System prerequisites

- **OpenFOAM 10** — for any real CFD run. Install per upstream instructions; set `WM_PROJECT_DIR` to your install root.
- **Python ≥ 3.10** — the LangGraph pipeline targets 3.10+; `cfd-scientist` conda env standardizes on 3.11.
- **`pdflatex` + `bibtex`** — for `cfd-paper` PDF compilation. On Debian/Ubuntu: `sudo apt install texlive-latex-extra texlive-bibtex-extra`. On macOS: `brew install --cask mactex` (or `basictex` for a slimmer install).
- **`wmake`** — comes with OpenFOAM; needed by `cfd-code-modify`.
- **GPU/headless rendering for PyVista** — on headless servers, install OSMesa or EGL backends so `pyvista.Plotter(off_screen=True)` can render. Debian/Ubuntu: `sudo apt install libosmesa6 libegl1`.
- **`xvfb`** (optional) — useful when running PyVista in CI/Docker without GPU. The skill recipes call `pv.start_xvfb()` defensively.

### Python environment

**Recommended (matches FoamAgent's stack):**

```bash
conda create -y -n cfd-scientist python=3.11
conda activate cfd-scientist
pip install -r requirements.txt
pip install -e .   # installs the cfd-scientist CLI
```

**venv alternative:**

```bash
./setup_env.sh   # creates .venv and installs requirements.txt
source .venv/bin/activate
pip install -e .
```

**Pip-only (no editable install):**

```bash
pip install -r requirements.txt
# then run via: python -m cfd_langgraph.workflow.main <cmd> ...
```

### Foam-Agent

The Foam-Agent framework is vendored under `Foam-Agent/`. Modes A and B call it through `scripts/foam_run.py`. Mode C drives the same planner → write → Allrun → run → review loop **agent-natively** via `cfd-skills/cfd-foamagent/SKILL.md` (FoamAgent prompts embedded verbatim); its only scripted dependency is FAISS retrieval through `scripts/rag_query.py`. Build the RAG index once with `python Foam-Agent/init_database.py`. For full FoamAgent install/usage, see `Foam-Agent/README.md`.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `S2_API_KEY` | Semantic Scholar API key (literature stage). Public endpoint works without it but is rate-limited. | unset |
| `WM_PROJECT_DIR` | OpenFOAM install root. Required for any real CFD run. | unset |
| `CFD_PROMPTS_PATH` | Path to `prompts.yaml`. | `./prompts/prompts.yaml` |
| `FOAM_AGENT_MAIN` | Foam-Agent entrypoint. | `./Foam-Agent/foambench_main.py` |
| `CFD_SCIENTIST_LLM_PROVIDER` | LLM provider. One of `bedrock`, `openai`, `anthropic`, `claude-code`, `openai-codex`, `gemini`. | inferred from model id |
| `CFD_SCIENTIST_MODEL` | Model identifier for the chosen provider. | provider default |
| `CFD_SCIENTIST_EFFORT` | Reasoning effort for every stage. Accepted values differ by provider; an unsupported one is refused, not downgraded. | provider default |
| `CFD_ORCH_TIMELINE_PATH` | Run timeline path (single-source observability). | per-run default |
| `CFD_IDEATION_ENABLE_LITERATURE` | `1` to enable literature in ideation. | `1` |
| `CFD_IDEATION_MAX_PAPERS` | Cap on retrieved papers. | `12` |
| `CFD_IDEATION_MAX_EXPERIMENTS` | Cap on proposed experiments. | `50` |
| `CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL` | Cap on total experiments. | `50` |
| `CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT` | Per-case rerun cap. | `2` |

### LLM provider — subscription or API

`CFD_SCIENTIST_LLM_PROVIDER` + `CFD_SCIENTIST_MODEL` select the model for the CLI and every
script it shells out to. Two families are supported.

**Subscription-billed (no API key).** These reuse the OAuth cache of a CLI you already log into,
so a Claude Pro/Max or ChatGPT Plus/Pro subscription drives the whole pipeline at no per-token
cost:

```bash
# Claude Code — credentials from ~/.claude, via claude-agent-sdk
export CFD_SCIENTIST_LLM_PROVIDER=claude-code
export CFD_SCIENTIST_MODEL=claude-sonnet-4-6      # run `claude` once to sign in

# OpenAI Codex — credentials from ~/.codex/auth.json
export CFD_SCIENTIST_LLM_PROVIDER=openai-codex
export CFD_SCIENTIST_MODEL=codex                  # run `codex login` once
unset OPENAI_API_KEY OPENAI_BASE_URL              # OAuth only; an API key here overrides it
```

`CFD_SCIENTIST_MODEL=codex` resolves to whatever `~/.codex/config.toml` is set to, so it tracks
the Codex CLI rather than pinning a name. Both providers use **native tool calling** — Claude via
an in-process SDK MCP server, Codex via the Responses API `tools` field — so neither depends on
parsing tool calls out of prose. One practical difference: Codex issues parallel tool calls while
`claude-code` issues one per turn, so manager fan-outs serialize on the latter.

An expired subscription token is reported up front rather than surfacing later as an HTTP 401,
and a token refreshed on disk mid-run is picked up without restarting the study.

**Reasoning effort** applies to every stage. The accepted set differs by provider and an
unsupported value is refused rather than silently downgraded:

```bash
export CFD_SCIENTIST_EFFORT=high    # openai-codex: none|minimal|low|medium|high|xhigh
                                    # claude-code:  low|medium|high|max
```

**API-billed providers** below. Any of them can drive the manager — `deep_agent.py` refuses a
model that does not implement `bind_tools`.

**Bedrock:**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="bedrock"
export CFD_SCIENTIST_MODEL="us.anthropic.claude-sonnet-4-6"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"
```

**OpenAI (text + vision):**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="openai"
export CFD_SCIENTIST_MODEL="gpt-4o"   # vision-capable for the VLM steps
export OPENAI_API_KEY="..."
```

**Anthropic (direct API):**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="anthropic"
export CFD_SCIENTIST_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_API_KEY="..."
```

**Gemini (OpenAI-compatible endpoint):**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="gemini"
export CFD_SCIENTIST_MODEL="gemini-1.5-pro"
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."  # Gemini OpenAI-compatible proxy
```

The interpreter / analysis / vision-QA steps need a **vision-capable** model (they read PNGs). If you set a text-only model, those steps will fail.

---

## Repository layout

| Path | What's there |
|---|---|
| `scripts/` | Per-stage workers + shared tooling — `cfd_cli.py` (Mode A entrypoint for pip-only installs), `orchestrator_run.py` (Mode B driver), `lit.py`, `hypothesis.py`, `requirements.py`, `foam_run.py`, `rag_query.py`, `viz.py`, `interpret.py`, `analyze.py`, `paper_unified.py`, `open_ended_discovery.py`, `oed_extensions.py`, `oed_search_archive.py` (the quality-diversity archive), `code_mod_agentic.py` (the sandboxed model-building agent), … plus the cross-mode **audit tooling**: `stage_gate_audit.py` (per-run gate), `audit_bundle.py` (batch verifier), `sanitize_bundle.py` (release hygiene). Mode C calls the audit tooling + `foam_run.py` / `rag_query.py`; only `orchestrator_run.py` is Mode-B-only. |
| `src/cfd_langgraph/` | The Python package: agents, prompts loader, workflow graph. |
| `src/cfd_langgraph/cli/` | The interactive CLI (`repl.py`) — terminal UI, interrupt handling, approval gates. |
| `src/cfd_langgraph/manager/` | The deepagents manager: `deep_agent.py` (graph + system prompt), `subagents.py` (`case-runner`, `oed-candidate-runner`), `tools.py` (every tool the agents call), `control.py` (interrupts + filesystem permissions). |
| `src/cfd_langgraph/scheduling/` | `CaseCoordinator` and the concurrency calibration that bounds how many solvers run at once. |
| `src/cfd_langgraph/llm/` | Provider factory, subscription OAuth (Claude Code / Codex), prompt-caching middleware. |
| `cfd-skills/` | Mode C's per-stage skill recipes (ARIS/DS-style; expert prompts embedded). 16 SKILLs — see [`cfd-skills/README.md`](cfd-skills/README.md). |
| `skills/` | Skill router + FoamAgent runtime contract + thin aliases (`cfd-orchestrator`, `cfd-foamagent-runtime`, `cfd-mesh-independence`, `cfd-research`, `cfd-code-mod`). |
| `.claude/skills/` | **Local, gitignored** — per-checkout symlinks so Claude Code discovers the skills. Regenerate per `cfd-skills/README.md`; never committed. |
| `prompts/prompts.yaml` | Authoritative source for all expert prompts. Embedded verbatim in `cfd-skills/`. |
| `openfoam_literature_change_agent_prompt_v2.txt` | OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol. Mirrored verbatim inside `cfd-skills/cfd-code-modify/SKILL.md`. |
| `Foam-Agent/` | Vendored FoamAgent framework (RAG + planner + reviewer for case generation). |
| `runs/` | Per-study output directories. |
| `starter_*/` | Per-flow starter case templates with reference DNS/experimental data — `starter_viscosity_change/`, `starter_turbulence_model_change/`, `starter_oed_turbulence/`, `starter_multiphase/`. Read-only inputs. |
| `AGENTS.md` | Orchestrator-agnostic stage-by-stage pipeline spec. |
| `CLAUDE.md` | Routing rules for Claude Code / similar agents. |
| `pyproject.toml`, `requirements.txt`, `setup_env.sh` | Packaging + setup. |

---

## Outputs (`out-dir`)

```
runs/<study>/
├─ state.json                      # routing + current_stage + status (reconciled by the audit)
├─ checkpoints/<stage>_done.json   # per-stage completion markers (the real progress truth)
├─ timeline.json                   # append-only event log
├─ lit.json                        # cfd-literature
├─ hypotheses.json                 # cfd-hypothesis
├─ requirements.json               # cfd-requirements
├─ benchmark_data.json             # cfd-pipeline (optional)
├─ reference_data_manifest.json    # cfd-pipeline (optional)
├─ baseline_case/                  # cfd-pipeline / baseline_setup
├─ baseline_metrics.json
├─ metric_specs.json               # cfd-pipeline / metric_setup
├─ comparators/compute_metrics.py
├─ selected_mesh_spec.json         # cfd-mesh-gate
├─ mesh_independence_context.json
├─ mesh_gate/<group>/{baseline,refined,refined_v2}/   # per-level cases
├─ cases/case_NNN/                 # per-experiment OpenFOAM cases
│  ├─ run_result.json              # status (success only with a clean End log, no FATAL)
│  ├─ decision.json
│  ├─ runtime_dependencies.json    # declared .so dependencies (code-mod cases)
│  ├─ figs/*.png                   # diagnostic figures
│  └─ vision_analysis.json
├─ analysis.json                   # cfd-analyze
├─ cross_experiment_analysis/      # cfd-cross-analyze (aggregate.csv, *.png, interpretation.md)
├─ paper_unified_plan.json         # cfd-paper planner
├─ paper_figs/*.png                # cfd-paper figures (full mode)
├─ paper/main.tex
├─ paper/refs.bib
├─ paper/main.pdf                  # FINAL PDF
├─ review.json                     # cfd-paper reviewer's last verdict
├─ open_ended_discovery/           # OED only
│  ├─ history.json                 # one record per iteration (honest status)
│  ├─ best.json
│  ├─ baseline_metric_vector.json
│  ├─ bound_comparators.json
│  ├─ candidates/<id>/case
│  └─ comparators/*.py
├─ oed_artifact.json               # post-OED handoff (or regular code_mod)
├─ bridge.json, manifest.json      # OED → analyze/paper bridge (cfd-open-discovery Step 14)
└─ audit_passed.json               # written ONLY by stage_gate_audit.py on rc=0 — the
                                   #   authoritative "run complete" signal
```

---

## Verifying a run — the stage-gate audit

Completion is **not** "the agent said it finished". A run is complete only when `scripts/stage_gate_audit.py` exits `rc == 0` and has written a signed `<out-dir>/audit_passed.json` (`audit_signature: "stage_gate_audit.py:v1"`). In Mode C the orchestrator skill runs this as its mandatory final loop; in Modes A and B it is the post-run check.

```bash
# Audit one run (skill mode runs this automatically as its last step)
python scripts/stage_gate_audit.py --out-dir runs/<study>

# Verify every task in a batch carries a valid audit record
python scripts/audit_bundle.py --bundle runs/<batch>

# Before sharing/zipping a batch: portable paths + reproducibility manifest
python scripts/sanitize_bundle.py --bundle runs/<batch> --apply
```

The audit hard-fails on a failed `run_result.json` whose rerun budget is not documented as exhausted, a `FOAM FATAL ERROR` in any solver log, incomplete OED iteration bookkeeping, a mesh-gate self-exemption lacking a live-verified DOI citation, and paper-quality gaps (missing governing/closure equations, near-duplicate or filler paragraphs, inconsistent reference values, …). It also reconciles a stale `state.json` against the checkpoints on disk. The full gate list lives in `skills/cfd-orchestrator/SKILL.md` Step 5 and `cfd-skills/cfd-paper-writer/SKILL.md`.

---

## Long-run policy

- CFD runs are slow but legitimate. Steady RANS may take 30–120 min; transient hours; OED budgets multi-hour.
- Default `max_time_limit` per case is 2 h; raise to 6 h+ for paper-quality production. The orchestrator and the skills both honor this.
- Don't declare timeout prematurely: monitor by tailing `<case_dir>/log.<solver>` and checking mtime + Time line progress.
- If a run stalls or diverges, apply CFL-aware retry (`adjustTimeStep yes; maxCo 0.7; small deltaT bump`) before declaring failure (see `cfd-skills/cfd-experiment/SKILL.md` Step 6).

---

## Troubleshooting

- **"No module named 'cfd_langgraph'"** — run from repo root and `pip install -e .`, or set `PYTHONPATH` to the repo root.
- **Bedrock errors** — check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` and the model id (e.g. `us.anthropic.claude-sonnet-4-6`).
- **OAuth (openai-codex) silently uses API billing** — ensure `OPENAI_API_KEY` is unset (`unset OPENAI_API_KEY OPENAI_BASE_URL`); the OAuth code path lives in `~/.codex/auth.json`.
- **Codex CLI rejects the model** — `npm install -g @openai/codex@latest`; older versions reject newer model ids.
- **Foam-Agent not found** — ensure `Foam-Agent/foambench_main.py` exists or set `FOAM_AGENT_MAIN`.
- **No figures / PyVista errors** — ensure Foam-Agent wrote results under `<case_dir>/`, that `pyvista` and `matplotlib` are installed, and that an off-screen OpenGL backend (OSMesa/EGL) is available on headless systems.
- **PDF compilation fails** — `pdflatex` is missing; install `texlive-latex-extra` (Debian/Ubuntu) or `mactex` (macOS).
- **HTTP 503 in `paper_unified.py` / `batch_paper_viz`** — transient upstream API outage during per-figure VLM QA; resume with `--resume-from paper_review` once the upstream is healthy. No state is lost.
- **OED post-bridge produces a degenerate 2-case plan** — `oed_artifact.json.provenance == "regular_code_mod"` is being treated as a real OED winner. The gate is documented in `cfd-skills/cfd-open-discovery/SKILL.md` (Step 13); the fix is to honor `provenance != "regular_code_mod" && best_iteration > 0` before firing the bridge.
- **The audit fails a run I thought was finished** — the stage-gate audit hard-fails a failed `run_result.json` unless its rerun budget is documented as exhausted, a `FOAM FATAL ERROR` in any solver log, a `success` case with no clean `End` line, incomplete OED bookkeeping (H33), or a mesh-gate self-exemption without a verifiable DOI (H31). Read the printed failure list and fix each gap — that is the intended behaviour, not a regression. `audit_passed.json` only appears when every gate passes.
- **`audit_bundle.py` reports "audit_passed.json absent"** — the task's chain stopped before the final audit. Re-run that task through `Skill cfd-orchestrator` Step 5 (or `scripts/stage_gate_audit.py`) until it reaches `rc=0`.

---

## Contributing — keeping the modes in sync

Every mode shares `prompts/prompts.yaml` as the authoritative prompt source. The skill mode embeds verbatim copies for self-containment.

When you edit a prompt:
1. Edit `prompts/prompts.yaml` (Modes A and B pick it up automatically).
2. Find the embedded copies — `grep -rn "from prompts/prompts.yaml" cfd-skills/` lists every reference.
3. Update each embedded copy. The CI / pre-commit hook will eventually verify these match (TODO).

When you change the OPENFOAM 10 code-mod protocol:
1. Edit `openfoam_literature_change_agent_prompt_v2.txt`.
2. Update the verbatim mirror in `cfd-skills/cfd-code-modify/SKILL.md`.

---

## License and citation
MIT. 

```
@article{somasekharan2026ai,
  title={AI CFD Scientist: Toward Open-Ended Computational Fluid Dynamics Discovery with Physics-Aware AI Agents},
  author={Somasekharan, Nithin and Pathak, Rabi and Dhanakoti, Manushri and Zhang, Tingwen and Yue, Ling and Zhu, Andy and Pan, Shaowu},
  journal={arXiv preprint arXiv:2605.06607},
  year={2026}
}
```

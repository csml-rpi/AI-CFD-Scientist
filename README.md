# CFD Scientist

End-to-end **agentic CFD research pipeline** — from a research topic to literature-aware ideation, mesh-independence-checked OpenFOAM runs (via Foam-Agent), interpreter checks with diagnostics, cross-case analysis, and a LaTeX paper draft.

Run it two ways:

- **LangGraph Python orchestrator** — one command, hands-off, checkpointed, the long-running default.
- **Skill-driven (ARIS / DeepScientist style)** — markdown SKILLs invokable from any LLM agent (Claude Code, Cursor, Codex CLI, custom). Self-contained: every skill embeds its expert prompts verbatim and walks the agent end-to-end.
- **Or both, mixed** — the modes share artifact contracts (`lit.json`, `requirements.json`, `selected_mesh_spec.json`, `analysis.json`, etc.), so a partial run from one can be picked up by the other.

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

For open-ended model discovery, replace steps 3–9 with the `cfd-open-discovery` loop: propose candidate → cfd-code-modify → cfd-experiment → score → repeat until budget.

The orchestrator-agnostic stage-by-stage spec is in [`AGENTS.md`](AGENTS.md). The Claude Code / agent routing rules are in [`CLAUDE.md`](CLAUDE.md). The skill recipes are documented in [`cfd-skills/README.md`](cfd-skills/README.md).

---

## Three modes — pick what fits

### Mode A — LangGraph Python orchestrator (default for unattended runs)

One command, end-to-end, with checkpointing and `--resume-from`. This is what you want for multi-hour production runs.

```bash
conda activate cfd-scientist

# Full pipeline
python scripts/orchestrator_run.py \
  --topic "LES of backward-facing step at Re=5100, compare against Le-Moin DNS" \
  --out-dir runs/bfs_les \
  --provider claude-code --model claude-sonnet-4-6 \
  --starter-dir starter

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

### Mode B — Skill-driven (ARIS / DeepScientist style)

Invoke individual skills from any LLM agent. Each `cfd-skills/cfd-<stage>/SKILL.md` is self-contained: the expert prompts (HypothesisAgent, IdeationAgent, ResultsInterpreterAgent, WriterAgent, PaperReviewerAgent, RunValidityAgent, MetricProposer, ComparatorAuthor, ComparatorVerifier, MetricSetupAgent, MetricSetupVerifier from `prompts/prompts.yaml`, plus the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol) are embedded verbatim. Scripts are an *optional fast-path*; the agent recipe is primary.

```text
# Top-level chain
/cfd-pipeline topic="LES of backward-facing step Re=5100" out-dir=runs/bfs_skill

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
  starter-dir=starter/periodic_hill budget=20
```

Use this when you want manual control, are integrating into another agent framework, or want to run only part of the pipeline ad-hoc. See [`cfd-skills/README.md`](cfd-skills/README.md) for the complete skill catalog and contracts.

### Mode C — Hybrid

Run Mode A for the main pipeline; invoke skills (Mode B) ad-hoc against the same `out-dir`. Because both modes read/write the same JSON contracts:

- The orchestrator stopped at `analysis_done`? Invoke `/cfd-paper` against the same `out-dir`.
- Use a skill to hand-craft one stage's output, then resume the orchestrator with `--resume-from <next_stage>`.
- Use the orchestrator for long unattended runs; use skills for interactive exploration on the same artifacts.

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

The Foam-Agent framework is vendored under `Foam-Agent/`. The Python pipeline calls it through `scripts/foam_run.py`; the skill mode calls the same script (this is the one place a script is unavoidable, because Foam-Agent *is* the framework — see `cfd-skills/cfd-experiment/SKILL.md`). For full FoamAgent install/usage, see `Foam-Agent/README.md`.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `S2_API_KEY` | Semantic Scholar API key (literature stage). Public endpoint works without it but is rate-limited. | unset |
| `WM_PROJECT_DIR` | OpenFOAM install root. Required for any real CFD run. | unset |
| `CFD_PROMPTS_PATH` | Path to `prompts.yaml`. | `./prompts/prompts.yaml` |
| `FOAM_AGENT_MAIN` | Foam-Agent entrypoint. | `./Foam-Agent/foambench_main.py` |
| `CFD_SCIENTIST_LLM_PROVIDER` | LLM provider. One of `bedrock`, `openai`, `anthropic`, `claude-code`, `openai-codex`, `gemini`. | inferred from model id |
| `CFD_SCIENTIST_MODEL` | Model identifier for the chosen provider. | provider default |
| `CFD_ORCH_TIMELINE_PATH` | Run timeline path (single-source observability). | per-run default |
| `CFD_IDEATION_ENABLE_LITERATURE` | `1` to enable literature in ideation. | `1` |
| `CFD_IDEATION_MAX_PAPERS` | Cap on retrieved papers. | `12` |
| `CFD_IDEATION_MAX_EXPERIMENTS` | Cap on proposed experiments. | `50` |
| `CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL` | Cap on total experiments. | `50` |
| `CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT` | Per-case rerun cap. | `2` |

### LLM provider — concrete examples

**Bedrock (legacy default):**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="bedrock"
export CFD_SCIENTIST_MODEL="us.anthropic.claude-sonnet-4-6"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"
```

**OpenAI Codex (OAuth, current default for skill orchestration):**
```bash
export CFD_SCIENTIST_LLM_PROVIDER="openai-codex"
export CFD_SCIENTIST_MODEL="gpt-5.5"
# OAuth comes from ~/.codex/auth.json (run `codex login` once).
# Make sure OPENAI_API_KEY is unset when using OAuth:
unset OPENAI_API_KEY OPENAI_BASE_URL
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
| `scripts/` | LangGraph pipeline scripts (`orchestrator_run.py`, `lit.py`, `hypothesis.py`, `requirements.py`, `foam_run.py`, `viz.py`, `interpret.py`, `analyze.py`, `paper_unified.py`, `open_ended_discovery.py`, `oed_extensions.py`, etc.). All Python source for Mode A. |
| `src/cfd_langgraph/` | Mode A's Python package: agents, prompts loader, workflow graph. |
| `cfd-skills/` | Mode B's per-stage skill recipes (ARIS/DS-style; expert prompts embedded). 12 SKILLs. |
| `skills/` | Skill router + FoamAgent runtime contract + thin aliases (`cfd-orchestrator`, `cfd-foamagent-runtime`, `cfd-mesh-independence`, `cfd-research`, `cfd-code-mod`). |
| `prompts/prompts.yaml` | Authoritative source for all expert prompts. Embedded verbatim in `cfd-skills/`. |
| `openfoam_literature_change_agent_prompt_v2.txt` | OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol. Mirrored verbatim inside `cfd-skills/cfd-code-modify/SKILL.md`. |
| `Foam-Agent/` | Vendored FoamAgent framework (RAG + planner + reviewer for case generation). |
| `runs/` | Per-study output directories. |
| `starter/` | Per-flow starter case templates (BFS, periodic hill, channel, ...) with reference DNS data. |
| `AGENTS.md` | Orchestrator-agnostic stage-by-stage pipeline spec. |
| `CLAUDE.md` | Routing rules for Claude Code / similar agents. |
| `pyproject.toml`, `requirements.txt`, `setup_env.sh` | Packaging + setup. |

---

## Outputs (`out-dir`)

```
runs/<study>/
├─ state.json                      # current routing, stage, checkpoint
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
│  ├─ run_result.json
│  ├─ decision.json
│  ├─ figs/*.png                   # diagnostic figures
│  └─ vision_analysis.json
├─ analysis.json                   # cfd-analyze
├─ paper_unified_plan.json         # cfd-paper planner
├─ paper_figs/*.png                # cfd-paper figures (full mode)
├─ paper/main.tex
├─ paper/references.bib
├─ paper/paper_draft.pdf           # FINAL PDF
├─ review.json                     # cfd-paper reviewer's last verdict
└─ open_ended_discovery/           # OED only
   ├─ history.json
   ├─ best.json
   ├─ baseline_metric_vector.json
   ├─ bound_comparators.json
   ├─ candidates/<id>/case
   └─ comparators/*.py
oed_artifact.json                  # post-OED handoff (or regular code_mod)
```

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

---

## Contributing — keeping the two modes in sync

Both modes share `prompts/prompts.yaml` as the authoritative prompt source. The skill mode embeds verbatim copies for self-containment.

When you edit a prompt:
1. Edit `prompts/prompts.yaml` (Mode A picks it up automatically).
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

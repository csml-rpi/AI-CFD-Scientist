# CFD Scientist

End-to-end **agentic CFD research pipeline** orchestrated by **LangGraph**: from a research topic to literature-aware ideation, Foam-Agent runs, interpreter checks with diagnostics, analysis, and a LaTeX paper draft (Sakana AI Scientist v2–style).

---

## Pipeline overview

1. **User topic** → research question in CFD.
2. **Ideation agent** → literature (Semantic Scholar ± Brave), novelty check, list of experiments.
3. **Hypothesis agent** → turns experiment ideas into **Foam-Agent prompts** (no viz text); LLM checks logic (solver, dt/endTime, BCs, etc.) and repairs until valid.
4. **Foam-Agent** → runs OpenFOAM cases (planner, input writer, runner, reviewer, etc.).
5. **Interpreter agent** → after each run: load data with PyVista, generate diagnostic plots (contours, slices, streamlines, etc.), check syntax/viz health; if needed, revise requirement and **rerun via Foam-Agent**.
6. **Analysis agent** → after all experiments: PyVista-based viz, cross-case analysis, conclusions; saves figures and `analysis_report.md`.
7. **Writer agent** → LaTeX paper: claim–evidence table, reproducibility appendix, failure/negative results, mandatory AI-disclosure sentence.

---

## Requirements

- **Python ≥ 3.10**
- **OpenFOAM** (for real runs; only needed when using `--execute`)
- **AWS credentials** (for default Bedrock model) or **OpenAI/Anthropic API key** if using another model

---

## Install

```bash
cd /path/to/cfd-scientist
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .           # install cfd-scientist CLI
```

Or with pip only (no editable install):

```bash
pip install -r requirements.txt
# then run via: python -m cfd_langgraph.workflow.main <cmd> ...
```

---

## Configure

### Required for pipeline (Bedrock default)

```bash
export CFD_SCIENTIST_MODEL="us.anthropic.claude-sonnet-4-6"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
# AWS region for Bedrock (e.g. us-west-2)
export AWS_DEFAULT_REGION="us-west-2"
```

### Optional paths

- `CFD_PROMPTS_PATH` – path to `prompts.yaml` (default: `./prompts/prompts.yaml`)
- `FOAM_AGENT_MAIN` – path to Foam-Agent entrypoint (default: `./Foam-Agent/foambench_main.py`)
- `WM_PROJECT_DIR` – OpenFOAM install path (for Foam-Agent when using `--execute`)

### Optional: literature and ideation

- `S2_API_KEY` – Semantic Scholar API key (optional; public API works without it, rate-limited)
- `BRAVE_SEARCH_API_KEY` – optional web search for literature
- `CFD_IDEATION_ENABLE_LITERATURE=1`
- `CFD_IDEATION_MAX_PAPERS=12`
- `CFD_IDEATION_MAX_EXPERIMENTS=50`
- `CFD_WORKFLOW_MAX_EXPERIMENTS_TOTAL=50`
- `CFD_WORKFLOW_MAX_RERUNS_PER_EXPERIMENT=2`

### Other model providers

- **OpenAI:** `export CFD_SCIENTIST_MODEL="gpt-4o"` and `OPENAI_API_KEY="..."`
- **Anthropic (direct):** `export CFD_SCIENTIST_MODEL="claude-3-5-sonnet-20241022"` and `ANTHROPIC_API_KEY="..."`

---

## How to run the pipeline

### 1. Ideation only (no runs)

Generate an idea JSON from a topic (with literature if keys are set):

```bash
cfd-scientist ideate \
  --topic "Study the effect of fuel velocity and inlet box sizes in 2D small pool fire." \
  --out ideation.json
```

- `--out -` prints JSON to stdout.
- Output includes `idea`, `literature_used`, novelty and experiment-count info.

### 2. Full pipeline (recommended)

Single command from topic to analysis + paper:

```bash
cfd-scientist run-topic \
  --topic "Your CFD research topic" \
  --out-dir ./runs/my_topic
```

- **Without `--execute`:** ideation → hypothesis (Foam-Agent prompts) → **plan-only** Foam-Agent (no OpenFOAM run) → no interpreter viz/rerun → no analysis/paper unless you add `--allow-non-executed-artifacts`.
- **With `--execute`:** real Foam-Agent runs → interpreter (PyVista diagnostics + rerun loop) → analysis (figures + report) → writer (LaTeX draft).

**Execute Foam-Agent and produce analysis + paper:**

```bash
cfd-scientist run-topic \
  --topic "Lid-driven cavity at Re=100 and Re=400" \
  --out-dir ./runs/cavity \
  --execute
```

**Generate analysis and paper even when not executing** (e.g. using existing run data):

```bash
cfd-scientist run-topic \
  --topic "Same topic" \
  --out-dir ./runs/cavity \
  --allow-non-executed-artifacts
```

### 3. Run as Python module (no CLI install)

```bash
python -m cfd_langgraph.workflow.main ideate --topic "Your topic" --out -
python -m cfd_langgraph.workflow.main run-topic --topic "Your topic" --out-dir ./runs/out --execute
```

---

## Output layout (`--out-dir`)

After `run-topic` with `--execute` (and optionally `--allow-non-executed-artifacts`):

- `pipeline_log.json` – full log (ideation, sims, run history, interpreter, analysis path, paper path).
- `analysis_report.md` – cross-case analysis.
- `paper_draft.tex` – LaTeX draft (Sakana AI Scientist v2–style).
- `paper_draft.pdf` – compiled PDF (via pdflatex; reviewer loop runs up to 10 times until pass).
- Per simulation (e.g. `sim_001/`):
  - `user_requirement.txt` – Foam-Agent prompt used.
  - `foam_output/` – OpenFOAM case output.
  - `viz_interpreter/` – diagnostic plots (interpreter).
  - `viz_analysis/` – publication-style figures (analysis agent).

---

## Prompts and Foam-Agent

- **Prompts:** default path `./prompts/prompts.yaml`. Override with `CFD_PROMPTS_PATH`.
- **Foam-Agent:** default `./Foam-Agent/foambench_main.py`. Override with `FOAM_AGENT_MAIN`. When you use `--execute`, the pipeline calls this script with `--prompt_path` and `--output`; OpenFOAM must be available (e.g. `WM_PROJECT_DIR` set) for real runs.

---

## Troubleshooting

- **“No module named 'cfd_langgraph'”** – run from repo root and `pip install -e .`, or set `PYTHONPATH` to the repo root.
- **Bedrock errors** – check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` and model id `us.anthropic.claude-sonnet-4-6`.
- **Foam-Agent not found** – ensure `Foam-Agent/foambench_main.py` exists or set `FOAM_AGENT_MAIN`.
- **No figures / PyVista errors** – ensure Foam-Agent wrote results (e.g. VTK) under `foam_output/` and that `pyvista` and `matplotlib` are installed. On headless servers (no physical display or GPU), PyVista/VTK also require an off-screen OpenGL backend such as OSMesa/EGL or a software Mesa stack; make sure the container or system has these libraries so `pyvista.Plotter(off_screen=True)` can render without X/GUI.
- **PDF compilation fails** – ensure `pdflatex` is installed (e.g. `texlive-latex-base` or full TeX Live). The paper agent compiles LaTeX to PDF and runs a reviewer loop (max 10 tries).

---

## License and citation

See repository license. If you use this pipeline in research, cite the repo and any Foam-Agent / OpenFOAM references as appropriate. The writer agent adds a mandatory sentence that the draft was generated with an automated CFD Scientist (AI-assisted) pipeline.

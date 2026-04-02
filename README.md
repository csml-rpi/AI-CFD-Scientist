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

### Set up Foam-Agent
```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Environment variables

Copy `env.sh`, fill in your values, and source it before running:

```bash
source env.sh
```

Supported providers: `bedrock` (default), `openai`, `openai-codex`, `anthropic`, `gemini`. The interpreter and analysis stages require a vision-capable model. When using `--execute`, the same provider/model is passed through to Foam-Agent automatically.

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
  --out-dir /absolute/path/to/runs/my_topic
```

Note: Use an **absolute path** for `--out-dir`. Later stages (including Foam-Agent subprocess calls) may run with a different working directory; absolute `--out-dir` avoids “file not found” issues for `user_requirement.txt` and other artifacts.

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

### 3. Resume or restart a partial run

If a run was interrupted after Foam-Agent finished (e.g. Ctrl-C during interpreter):

```bash
# Resume from where it left off (re-enters the interpreter/rerun loop)
cfd-scientist resume-topic --out-dir ./runs/my_topic

# Restart from the interpreter stage, skipping ideation/hypothesis/foam entirely
cfd-scientist restart-topic --out-dir ./runs/my_topic
```

### 4. Run as Python module (no CLI install)

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
- **Hypothesis stage is very slow** – the validate+repair loop uses the main model by default (up to ~9 calls per experiment). Set `CFD_SCIENTIST_VALIDATOR_MODEL` to a fast model (e.g. Haiku) to reduce this significantly. Also consider lowering `BEDROCK_READ_TIMEOUT` from 300s to 90s for text-only stages.
- **Foam-Agent not found** – ensure `Foam-Agent/foambench_main.py` exists or set `FOAM_AGENT_MAIN`.
- **No figures / PyVista errors** – ensure Foam-Agent wrote results (e.g. VTK) under `foam_output/` and that `pyvista` and `matplotlib` are installed. On headless servers (no physical display or GPU), PyVista/VTK also require an off-screen OpenGL backend such as OSMesa/EGL or a software Mesa stack; make sure the container or system has these libraries so `pyvista.Plotter(off_screen=True)` can render without X/GUI.
- **PDF compilation fails** – ensure `pdflatex` is installed (e.g. `texlive-latex-base` or full TeX Live). The paper agent compiles LaTeX to PDF and runs a reviewer loop (max 10 tries).

---

## License and citation

See repository license. If you use this pipeline in research, cite the repo and any Foam-Agent / OpenFOAM references as appropriate. The writer agent adds a mandatory sentence that the draft was generated with an automated CFD Scientist (AI-assisted) pipeline.

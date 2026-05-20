---
name: cfd-cross-analyze
description: Cross-experiment quantitative analysis. Author Python scripts (numpy + matplotlib + PyVista + pandas) that load every PROCEED case, compute aggregated metrics, and generate the comparison figures the paper needs (Cf overlays vs reference, RMSE bar charts, parametric trends, Δ-field contours). Then run a vision+text interpretation pass over the generated plots. This is the cross-case Python authoring layer cfd-paper consumes — without it, papers end up with only per-case contours.
---

# cfd-cross-analyze

## ⚠ HARD STOP — read this before any other step

If `<out-dir>/checkpoints/analysis_done.json` AND `<out-dir>/checkpoints/paper_experiment_plan_done.json` do not both exist, STOP. The orchestrator chain enters this skill only after `cfd-analyze` and `cfd-paper` Stage 0 have completed. Stage 0 may propose additional cases that this skill compares against; running cross-analysis before Stage 0 produces comparison figures on an incomplete case set. Return to `Skill cfd-orchestrator` (or `Skill cfd-skills/cfd-analyze`), then come back.

cfd-paper writes the manuscript; **cfd-cross-analyze produces the cross-case quantitative evidence the manuscript discusses.** This skill mirrors the LangGraph workflow's `analysis_agent.run_cross_experiment_data_processing` + `_interpret_cross_experiment_outputs`. The agent (you) authors Python scripts that read every PROCEED case via PyVista or sampled CSVs, computes cross-case metrics, generates publication-quality plots, and interprets them through a vision-LLM pass.

## Step 0 — Preflight gate (HARD; slice 14)

Before reading any case or authoring any script:

```bash
python scripts/stage_gate_audit.py --out-dir <out-dir> --mode preflight --target-stage cross_experiment_analysis
```

If `rc != 0`, STOP. The script requires `analysis_done` and `paper_experiment_plan_done` (Stage 0 of cfd-paper) to be on disk first. The Stage-0 plan tells this skill which cases (existing + newly proposed) are in scope — running cross-analysis before Stage 0 means producing comparison figures on a possibly-incomplete case set, exactly the failure mode that produced an H12 "only 1 figure type" gate trip.

Without this skill running, papers produced by `cfd-paper` will only contain per-case PyVista contours and a metrics table — no Cf-overlay-vs-reference, no RMSE bar charts, no parametric-trend plots, no Δ-field comparisons. Those are exactly the figures that justify a paper's claims.

## When to use

After `cfd-experiment` (the full set of cases for the study) and `cfd-analyze` (per-case metrics), and **before** `cfd-paper`. Required for routes 1, 2, 6 (any route that produces a paper) when:
- The study compares ≥ 2 cases on a shared QoI (sensitivity studies, code-mod baseline-vs-modified, parametric sweeps, OED candidate ranking).
- Reference data exists (DNS / experiment / correlation) AND the topic asks for comparison against it.
- A primary claim of the topic is quantitative ("model X reduces error by N%", "x_r/h matches DNS within Y%").

If none of these apply (single-case demonstration with no comparison QoI), you can skip — but document the skip in the timeline.

## Inputs
- `out-dir` (required) — must contain `cases/case_*/` with `run_result.json` and `decision.json` (each case must be PROCEED), plus `analysis.json`.
- Optional: `<out-dir>/reference_data_manifest.json` or `<out-dir>/<reference_csv>` for any reference dataset to overlay against.
- Optional: `<out-dir>/selected_mesh_spec.json` (so cross-case plots note which mesh level was used).
- Optional: `topic` (string) — used to decide which cross-case figures matter for THIS paper. If not passed, read from `<out-dir>/state.json#topic`.

## Outputs
All under `<out-dir>/cross_experiment_analysis/`:

```
cross_experiment_analysis/
├─ case_context.json                 deterministic per-case context (numerics, mesh, BC, model)
├─ case_context_normalized.json      LLM-summarized per-case context for cross-comparison
├─ planner_objectives.json           {needs_processing, objectives[], rationale}
├─ scripts/
│  ├─ attempt_01_script.py           agent's first Python attempt
│  ├─ attempt_02_script.py           (any retries; max 10)
│  └─ final_script.py                successful one (also copied here)
├─ aggregate.csv                     per-case metric table (rows = cases, cols = QoIs)
├─ data_processing_report.md         what was computed + any UNRESOLVED notes
├─ *.png                             cross-experiment plots (the figures cfd-paper will use)
└─ cross_experiment_interpretation.md vision-LLM interpretation written for the writer
```

## Recipe (primary, agent-driven)

### Step 1 — Build the case inventory + deterministic context

For each PROCEED case under `cases/case_*/`:

1. Read `run_result.json` (`case_solver`, `wall_time_s`, `loop_count`).
2. Read `decision.json` (must have `status: PROCEED`; cases with `RERUN` or `REVISE` are excluded).
3. Read `<case_dir>/system/controlDict` for application + endTime + writeInterval.
4. Read `<case_dir>/constant/momentumTransport` (or `turbulenceProperties`) for the active model + coefficients.
5. Read `<case_dir>/constant/transportProperties` (or `physicalProperties`) for nu / rho / EOS info.
6. Read `<case_dir>/0/U`, `<case_dir>/0/p` for boundary conditions (just the structure, not full data).
7. Read `<case_dir>/constant/polyMesh/owner` size for cell count.

Persist everything as `cross_experiment_analysis/case_context.json` (one entry per case). This is the authoritative metadata source for the script.

### Step 2 — Normalize case contexts via LLM

For each case context entry, ask the LLM (using `cfd_langgraph` model) to produce a normalized summary that flags inconsistencies and resolves any ambiguous values (e.g., U_inf might appear in different forms across cases — pick one canonical form per case, with provenance).

Write to `cross_experiment_analysis/case_context_normalized.json`. **The normalized version is non-authoritative** — `case_context.json` remains source-of-truth; the script must cross-check the two.

### Step 3 — Decide cross-experiment objectives

#### System prompt (mirrored from LangGraph `analysis_agent.run_cross_experiment_data_processing` planner)

```
You are a CFD cross-experiment data-processing planner.
Given topic + experiment requirements + available artifacts, decide whether additional quantitative post-processing is needed beyond image-based interpretation.
Examples: trend/correlation plots, regression fits, summary tables, derived metrics, error decomposition, parametric scaling, Δ-field comparisons.
For mesh independence / baseline-vs-refined style comparisons, plan work that uses the final written timestep in each case (same nominal endTime); do not require QoIs aggregated over all intermediate times unless the study is explicitly transient.
Return ONLY JSON with keys:
  needs_processing (bool)
  objectives (list[str]): each is one short imperative ("compute Cf vs x/h overlay across all cases including DNS", "produce RMSE bar chart", "fit β vs Re trend if applicable", "Δ-nut contour modified − baseline")
  rationale (str): why these objectives, why not others
```

#### User prompt template

```
Study topic:
{topic}

Experiment manifest (per-case context):
{case_context_json}

Normalized case context (LLM summary):
{case_context_normalized_json}

Reference data available:
{reference_data_summary}

Decide which cross-experiment quantitative post-processing is needed for stronger analysis/paper evidence.
```

Persist response as `cross_experiment_analysis/planner_objectives.json`. If `needs_processing: false` AND the topic is genuinely single-case-with-no-comparison, skip to Step 6 with an empty interpretation. Otherwise continue.

### Step 4 — Author the cross-experiment Python script

This is the heart of the skill. The agent writes ONE Python script that loads every case and produces all the figures in `objectives[]`. Use the embedded prompt below verbatim.

#### System prompt (verbatim from LangGraph `analysis_agent.run_cross_experiment_data_processing` script-writer)

```
You write robust Python post-processing scripts for CFD cross-experiment analysis.
Return ONLY raw Python code (no markdown).

Script requirements:
- Read available experiment artifacts from the manifest.
- Use case_context.json / normalized context as authoritative metadata inputs.
- Do not invent physical constants or defaults (e.g., U_bulk, nu, rho, Re).
- For every quantity used in computation, resolve from case_context first and cite source path in the report.
- If required values are missing/inconsistent, mark that metric as UNRESOLVED and explain why in the report.
- Parse configuration with a model-aware source-of-truth policy instead of assuming one fixed dictionary.
- Automatically discover and read all relevant case configuration artifacts for physics, numerics, mesh, forcing/boundary driving, model selection, and experiment metadata/verification context.
- Determine the authoritative location of model parameters from the active model family and study mode, and cross-check against supporting dictionaries and metadata before extracting coefficients.
- Validate consistency of key controls that can be duplicated across files (e.g., driving conditions, target parameters, model activation) and report any mismatch explicitly before computing comparisons.
- For OpenFOAM field-based comparisons (e.g. mesh refinement, two cases same physics), load each case at its **latest/final write time** only for scalar QoIs and difference maps used in convergence checks; assume both runs share the same endTime unless manifest shows otherwise — do not integrate metrics across all saved timesteps unless an objective explicitly asks for transient analysis.
- Compute requested cross-experiment metrics/trends only when data is available.
- Save outputs under output_dir: (1) data_processing_report.md, (2) aggregate.csv, (3) one or more plot PNGs per objective.
- Be defensive: if some files are missing, continue with available data and explain limitations in report.
- Prefer PyVista for reading OpenFOAM case data/fields when needed (do NOT depend on ParaView GUI).
- Any matplotlib or PyVista PNGs intended for the paper: large fonts (ticks >= 14–18 pt, titles/labels >= 18–22 pt), no colorbar or legend overlapping data (use tight_layout with padding, horizontal colorbar, bbox_to_anchor for legends, or PyVista scalar_bar_args position/size).
- NumPy integration: use `np.trapezoid(y, x)` (NumPy 2+) or `getattr(np, "trapezoid", np.trapz)(y, x)` — do not call `np.trapz` alone (removed in NumPy 2.0).
- For 2D contour plots of elongated domains (e.g. channel flow), match figure or window aspect to domain extents so the channel is not a thin line.
- Exit 0 on success.

JOURNAL-QUALITY MATPLOTLIB STYLING (HARD — audit gate H30 catches under-styled PNGs):

Begin the script with the following rcParams block so every plot inherits publication styling:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize":     (8.0, 5.5),
    "figure.dpi":         180,
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "font.size":          14,
    "axes.titlesize":     16,
    "axes.labelsize":     16,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})
```

LINE PLOTS (Cf overlays, profile comparisons, sweeps):
- ALWAYS overlay the DNS / experimental reference curve when `reference_data/` contains one. The reference is the most important curve on the plot — draw it as a black dashed line (`color="k", linestyle="--", linewidth=2.5, label="DNS (reference)"`) FIRST so it sits underneath the candidates.
- Use a distinct color per case from a perceptually-uniform palette: `plt.cm.viridis(np.linspace(0.1, 0.9, n_cases))` for many cases, or `plt.cm.tab10.colors` for ≤ 10 cases. Never accept matplotlib's grayscale default.
- Mark the winner / best case with a thicker line (`linewidth=3.0`) and a distinguishable marker (`marker="o", markevery=10`).
- Axis labels include units and direction: `r"$x/h$"` (no unit), `r"$C_f$  $(-)$"`, etc. Use raw strings for LaTeX in labels.
- Add a legend; place outside the axes if it would cover data: `plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)` and follow with `fig.tight_layout()`.

BAR CHARTS (RMSE ranking, reattachment error, improvement-pct):
- ALWAYS color the bars from a categorical palette (`plt.cm.tab10.colors` or a custom `["#1f77b4", "#ff7f0e", "#2ca02c", ...]`). Never accept the default uniform-blue or grayscale.
- Sort bars by metric (ascending for error metrics, descending for improvement metrics) so the ranking is visually obvious; place baseline last (or first) and color it distinctly (e.g. `color="#888"` for baseline, brighter colors for candidates).
- Annotate each bar with its numeric value above the bar:
  ```python
  for i, v in enumerate(values):
      ax.text(i, v + 0.02 * max(values), f"{v:.3g}", ha="center", va="bottom", fontsize=11)
  ```
- Highlight the winner bar with a thicker edge: `ax.patches[winner_idx].set_edgecolor("red"); ax.patches[winner_idx].set_linewidth(2.5)`.
- Rotate x-tick labels 25–35° if labels are long: `plt.xticks(rotation=30, ha="right")`.
- Y-axis label with units; horizontal gridline only (`ax.yaxis.grid(True); ax.xaxis.grid(False)`).
- Add a horizontal baseline reference line when comparing against a baseline value: `ax.axhline(y=baseline_value, color="k", linestyle=":", linewidth=1.5, label="baseline")`.

EXAMPLE CF-OVERLAY (use this pattern):
```python
fig, ax = plt.subplots()
# Reference first (under all candidates)
if dns is not None:
    ax.plot(dns["x_over_h"], dns["Cf"], "k--", lw=2.5, label="DNS (reference)")
# Candidates
import numpy as np
colors = plt.cm.tab10.colors
for i, (label, data) in enumerate(sorted_candidates):
    is_winner = (label == winner_label)
    ax.plot(data["x"], data["Cf"],
            color=colors[i % 10],
            linewidth=3.0 if is_winner else 1.8,
            marker="o" if is_winner else None, markevery=10,
            label=label + (" (best)" if is_winner else ""))
ax.set_xlabel(r"$x/h$")
ax.set_ylabel(r"$C_f$  $(-)$")
ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
fig.tight_layout()
fig.savefig(output_dir / "cf_overlay.png")
plt.close(fig)
```

EXAMPLE RMSE-BAR (use this pattern):
```python
fig, ax = plt.subplots()
labels  = [c["label"] for c in cases]
values  = [c["rmse"]  for c in cases]
colors  = ["#888" if c.get("is_baseline") else plt.cm.tab10.colors[i % 10]
           for i, c in enumerate(cases)]
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
# Annotate values above bars
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02*max(values),
            f"{v:.4g}", ha="center", va="bottom", fontsize=11)
# Highlight winner
if winner_idx is not None:
    bars[winner_idx].set_edgecolor("red"); bars[winner_idx].set_linewidth(2.5)
ax.set_ylabel(r"$C_f$ RMSE vs DNS  (–)")
ax.yaxis.grid(True); ax.xaxis.grid(False)
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(output_dir / "rmse_bar.png")
plt.close(fig)
```

FILE SIZE MINIMUM: every PNG written must be at least 50 KB. A < 50 KB matplotlib PNG at 1600×900 means the styling collapsed (e.g. grayscale 16-bit dump with no actual content). Check with `os.stat(out_png).st_size` and re-render with higher dpi or richer data if below threshold.

SAVE THE ACTUAL SCRIPT: do NOT replace the saved script with a comment stub like "# generated inline". Write the complete Python script that produced the plots to the path requested (typically `cross_experiment_analysis/scripts/attempt_01_script.py`). The audit verifies the script file has > 2 KB of real code. A comment-only stub fails the gate.

PyVista OpenFOAM loading snippet (use as a starting point):
```python
import pyvista as pv
from pathlib import Path
foam_output_dir = Path('/path/to/foam_output')
# If no .foam exists, create one; PyVista uses it as a marker
marker = foam_output_dir / f"{foam_output_dir.name}.foam"
marker.touch(exist_ok=True)
mesh = pv.read(str(marker))
available_arrays = getattr(mesh, 'array_names', []) or []
```
```

#### User prompt template

```
Topic:
{topic}

Cross-experiment objectives (compute and plot ALL of these in the single script you write):
{objectives_bullet_list}

Experiment manifest (every case_dir + case_id + decision status):
{manifest_json}

Deterministic case context path (authoritative):
{case_context_path}

Normalized case context path (LLM summary, non-authoritative):
{normalized_context_path}

Reference data (if any — overlay these on comparison plots):
{reference_data_block}

Output directory (write all PNGs + aggregate.csv + data_processing_report.md here):
{output_dir}

Write a single Python script that processes every case in the manifest, computes the objectives, generates the requested plots, writes aggregate.csv (rows = cases, cols = QoIs), and writes data_processing_report.md (one section per objective, with UNRESOLVED notes where applicable). Output ONLY the script as raw Python.
```

Save the response as `cross_experiment_analysis/scripts/attempt_01_script.py`.

### Step 5 — Run + repair loop

Run the script:
```bash
( cd <out-dir> && python cross_experiment_analysis/scripts/attempt_01_script.py ) 2>&1 | tee cross_experiment_analysis/scripts/attempt_01_log.txt
```

If `rc != 0`:
- Capture the last 200 lines of the log.
- Re-invoke the script-writer LLM with the previous script + the error log + an "edit this minimally to fix the error" instruction.
- Save as `attempt_02_script.py` and rerun.
- **Cap: 10 attempts.**

After max retries, if still failing, write `cross_experiment_analysis/data_processing_report.md` documenting what was attempted, the persistent error, and a list of objectives that could not be computed. **Don't fabricate the cross-experiment outputs**; explicit failure is better than silent incomplete data.

When successful:
- Copy the working script to `cross_experiment_analysis/scripts/final_script.py`.
- Verify expected outputs exist: `aggregate.csv`, `data_processing_report.md`, ≥ 1 PNG per objective.

### Step 6 — Vision+text interpretation

Once the cross-experiment plots and report are produced, interpret them with a vision-LLM call so the writer has explicit guidance.

#### System prompt (verbatim from LangGraph `_interpret_cross_experiment_outputs`)

```
You are a CFD research analyst focused on cross-experiment synthesis.
Given the study topic, cross-experiment objectives, generated processing script, report text, and plot images, write a concise but rigorous interpretation suitable for Results/Discussion.
Include:
  (1) what quantitative results were actually obtained,
  (2) fitted relations/equations and their credibility limits,
  (3) image-backed observations,
  (4) cautions or missing evidence,
  (5) what the writer should include verbatim vs qualify.
Return plain markdown prose (no JSON).
```

#### User prompt template

```
Study topic:
{topic}

Cross-experiment objectives:
{objectives_bullet_list}

Generated cross-experiment script (truncated to 25K chars):
```python
{script_text_truncated}
```

Cross-experiment report (truncated to 25K chars):
{report_text_truncated}

Now inspect the attached cross-experiment plots and provide interpretation.
```

Attach all `.png` files from `cross_experiment_analysis/*.png` (cap at 12 images, downscale to ≤ 1999 px max dimension) as multimodal inputs.

Save the response as `cross_experiment_analysis/cross_experiment_interpretation.md`. This file is the **primary input** that `cfd-paper` Step 5 (Writer) consumes for the cross-case Results/Discussion paragraphs.

### Step 7 — Write timeline event

Append to `<out-dir>/timeline.json`:
```json
{
  "stage": "cross_experiment_analysis",
  "event": "complete",
  "ts": "<iso>",
  "n_cases": <int>,
  "n_objectives": <int>,
  "n_plots": <int>,
  "script_attempts": <int>,
  "interp_md_path": "cross_experiment_analysis/cross_experiment_interpretation.md"
}
```

Then write `<out-dir>/checkpoints/cross_experiment_analysis_done.json` (snapshot of `state.json`) so slice 8's required-checkpoint audit recognizes the stage.

## Hard preconditions (skill-mode enforcement)

1. **At least 1 PROCEED case must exist** under `cases/case_*/`. If zero PROCEED cases, the skill cannot author cross-case figures — write `cross_experiment_analysis/data_processing_report.md` explaining the gap and STOP. Do not fabricate plots from RERUN cases.
2. **The script must actually run.** Don't claim success on `attempt_NN_script.py` without a corresponding `attempt_NN_log.txt` showing rc=0. Don't write `final_script.py` until a script has been verified to run.
3. **`aggregate.csv` must exist with > 0 rows** before Step 6 starts. An empty CSV means the script ran but extracted nothing — that's a partial failure, document it.
4. **At least 1 cross-experiment PNG must exist** before writing `checkpoints/cross_experiment_analysis_done.json`. Otherwise the gate fails and `cfd-paper` cannot claim it has cross-experiment content.

## Anti-hallucination rules

- Never invent reference data. If `reference_data_manifest.json` doesn't list a DNS dataset, don't draw a fake DNS line on the overlay; explain the absence in the report.
- Never compute a derived metric (e.g. Reynolds number) from invented constants. Always cite the source path in `case_context.json` for every input.
- Never report a metric as "improved" without a statistical or threshold-based criterion. State the criterion (% threshold, RMSE delta sign) in the data_processing_report.
- Never mark a case as PROCEED in this skill — that's `cfd-interpret`'s job. This skill consumes existing decision.json verdicts, doesn't author new ones.
- Never overwrite `case_context.json` after Step 1 — it's the deterministic ground truth. Step 2's normalized version is a separate file.

## Cross-link

- **Upstream:** `cfd-skills/cfd-experiment/SKILL.md` produces case dirs; `cfd-skills/cfd-interpret/SKILL.md` produces decision.json verdicts; `cfd-skills/cfd-analyze/SKILL.md` produces text/JSON per-case metrics.
- **Downstream:** `cfd-skills/cfd-paper/SKILL.md` Step 1 (Plan) consumes `cross_experiment_analysis/*.png` as figure_jobs sources, and Step 5 (Writer) consumes `cross_experiment_interpretation.md` as the cross-case Results/Discussion grounding.
- **Numerics patterns:** `cfd-skills/cfd-stats/SKILL.md` — the agent's per-objective code in Step 4 should reuse those patterns (Pattern 1 for K-fit, Pattern 2 for Cf, Pattern 3 for drag, Pattern 6 for Richardson/GCI, Pattern 7 for two-extractor agreement, Pattern 8 for ratio metric).

## LangGraph mode only — do NOT call from skill mode

## Two valid execution paths

This skill has **two equally valid paths** — pick whichever fits.

1. **Inline agent recipe (above).** The agent reads case dirs via PyVista, authors a cross-case Python script (numpy + matplotlib + PyVista + pandas), runs it, then applies the embedded vision interpretation prompt to the generated plots. The recipe is fully embedded; no script call required.

2. **Script fast-path.** `src/cfd_langgraph/agents/analysis_agent.py:run_cross_experiment_data_processing()` implements the same planner + script-writer + repair loop + interpreter in Python (~360 lines). It uses the same prompts as the inline recipe and writes the same `cross_experiment_analysis/` directory. Importable as `from cfd_langgraph.agents.analysis_agent import AnalysisAgent` after `bootstrap_paths()` adds `src/` and `Foam-Agent/src/` to `sys.path`. The skill-mode-forbidden script is `scripts/orchestrator_run.py` only — the underlying per-stage Python is callable.

Both paths use `import pyvista` for direct case-data reading (`pv.OpenFOAMReader`) and write the same artifacts under `<out-dir>/cross_experiment_analysis/`, so `cfd-paper` consumes them identically.

## Next

After `<out-dir>/cross_experiment_analysis/` is populated (per-case context, aggregate.csv, figure PNGs, `cross_experiment_interpretation.md`) and `<out-dir>/checkpoints/cross_experiment_analysis_done.json` is on disk, your next action is:

`Skill cfd-skills/cfd-paper` — Step 1 onward (build manifest.json from cases, then invoke `scripts/paper_unified.py` which runs the full paper writing cycle internally: planner → batch PyVista figures → vision QA → writer with literature → pdflatex → reviewer → revise loop). The cross-experiment figures and `cross_experiment_interpretation.md` produced here are direct inputs to the writer.

Do not stop, summarize, or wait. The chain continues automatically.

---
name: cfd-paper
description: "Full paper writing in skill mode. ONE skill that walks the agent through a strict numbered protocol: Stage 0 experiment planner, then a planner-viz-writer-compile-reviewer-revise cycle that iterates until the reviewer returns pass=true or the cap is hit. Each phase is one concrete agent action with the LLM prompt embedded verbatim and an immediate schema-validation gate. No script delegation. The agent itself is the planner, the viz author, the vision QA, the writer, the compile-fixer, the reviewer, and the reviser — same as every other skill-based AI scientist."
---

# cfd-paper

## ⚠ HARD STOP — read this before any other step

For routes 1, 2, 6: if `<out-dir>/checkpoints/analysis_done.json` does not exist, STOP. Return to `Skill cfd-orchestrator`. For route 5 (paper-only): `<out-dir>/analysis.json` must exist before entry.

## How this skill works

This skill is a **strict numbered protocol**. You walk it step by step. Each step is ONE concrete agent action — either an LLM call (where your next response IS the prompted output), a bash invocation, or a validation gate. Each step ends with a literal `GOTO Step N` instruction. The cycle loops by GOTO-ing back to an earlier step.

You ARE every LLM in this pipeline. When a step says "your next response must be the planner JSON", that means: read the embedded system prompt, treat it as your operating instructions for that single response, fill in the user-message placeholders from disk, then your literal next message is the planner's JSON output. No meta-commentary. No "I will now run the planner." Just the output.

**Forbidden behaviors that triggered past failures (do not repeat any of these):**

- **Do not write `paper/main.tex` by hand without going through Step 5 (the writer prompt).** A hand-written .tex with sections "Objective", "Audit Trail", "Implementation Location" instead of "Introduction / Related Work / Methods / Results / Discussion / Conclusion" is the failure mode this skill exists to prevent.
- **Do not write `review.json` with your own schema** (e.g. `{pages, min_pages, findings}`). The reviewer LLM (which is you, in role) returns a specific schema with `recommendations[]`, `formatting_ok`, `figures_ok`, etc. Without those keys, downstream gates reject it as fabricated.
- **Do not write `paper_unified_plan.json` with your own schema** (e.g. `{title, sections, figures}`). The planner (which is you, in role) returns `{figure_jobs[], unified_viz_brief, case_order_for_paper, case_display_labels, omit_case_ids, analysis_narrative_hints, topic_paragraph_for_intro}`.
- **Do not write `audit_passed.json` yourself.** Only `scripts/stage_gate_audit.py` writes that file. The orchestrator runs the audit at Step 5 of its own protocol.
- **Do not skip the per-image vision QA in Step 3.** You ARE the vision LLM. Open each PNG and judge it.
- **Do not skip pdflatex** (Step 6). PDF generation is real; the agent doesn't get to declare "compiled".

## Library availability

You can `import pyvista`, `matplotlib`, `numpy`, `pandas`, anything in the conda env. PyVista renders contour fields, slices, streamlines, profiles directly from OpenFOAM case directories — that's what Steps 2 and 3 do. You can write and run Python scripts via Bash; you can read PNGs as images with your native multimodal capability for Step 3's vision QA.

---

# Stage 0 — Paper experiment planner

ONE LLM call. Decides whether the paper needs additional experiments before figure work begins. Run this once, immediately on skill entry.

## Step 0.0 — Literature precheck (HARD; H18 enforcement)

The OED + BFS smoke runs both produced unciteable papers because literature retrieval was silently skipped at some point in the chain and the paper stage didn't notice. This step is the gate. **Do not proceed past it.**

```bash
OUT=<out-dir>
MODE=$(jq -r '.mode // empty' "$OUT/state.json")
LIT="$OUT/lit.json"
[ -f "$OUT/lit_enriched.json" ] && LIT="$OUT/lit_enriched.json"
NEEDS_LIT=0
case "$MODE" in
  research|code_mod|paper_only|open_discovery) NEEDS_LIT=1 ;;
esac
if [ "$NEEDS_LIT" = "1" ]; then
  if [ ! -f "$LIT" ]; then
    echo "H18 precheck FAIL — $LIT missing for mode=$MODE."
    echo "Invoke Skill cfd-skills/cfd-literature before continuing. Do NOT author the paper without literature."
    exit 1
  fi
  N=$(jq 'if type=="array" then length elif .records then (.records|length) elif .papers then (.papers|length) elif .results then (.results|length) else 0 end' "$LIT")
  if [ "$N" -lt 5 ]; then
    echo "H18 precheck FAIL — $LIT has only $N record(s) for mode=$MODE (need ≥ 5)."
    echo "Re-invoke Skill cfd-skills/cfd-literature with broader queries (S2_API_KEY must be set)."
    exit 1
  fi
fi
```

If this check fails, **stop the paper stage** and re-invoke `Skill cfd-skills/cfd-literature` first. The audit's H18 / H19 / H10 gates will reject any paper authored on an empty lit.json — there is no value in proceeding.

## Step 0.1 — Build the Stage-0 input

Concatenate into one string (≤ 60,000 chars):

```
TOPIC: <state.json#topic>
MODE: <state.json#mode>

CURRENT SUCCESSFUL CASES (from manifest.json or built from cases/):
<list of case_ids that have run_result.status=success AND decision.status=PROCEED>

ANALYSIS:
<analysis.json verbatim, truncated to 30000 chars>

REQUIREMENTS:
<requirements.json verbatim, truncated to 15000 chars>
```

If `<out-dir>/manifest.json` doesn't exist, build it now from `cases/case_*/run_result.json` + `decision.json`:

```bash
OUT=<out-dir>
python3 - <<'PY'
import json, glob, os
from pathlib import Path
out = Path(os.environ.get("OUT", "."))
cases = []
for cd in sorted(glob.glob(str(out / "cases" / "case_*"))):
    cd_p = Path(cd)
    rr = cd_p / "run_result.json"
    dj = cd_p / "decision.json"
    status, decision = "unknown", ""
    if rr.is_file():
        try: status = json.loads(rr.read_text()).get("status", "unknown")
        except: pass
    if dj.is_file():
        try: decision = json.loads(dj.read_text()).get("status", "")
        except: pass
    paper_status = "success" if (status == "success" and decision == "PROCEED") else status
    cases.append({"case_id": cd_p.name, "case_dir": str(cd_p.resolve()),
                  "status": paper_status, "run_status": status, "decision": decision})
(out / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2))
PY
```

## Step 0.2 — Run the Stage-0 LLM call

**Your next response after reading this step MUST be the planner JSON output.** Use this system prompt as your operating instructions for that single response, fill the user-message placeholders from Step 0.1's input, then emit the JSON.

### System prompt (you are the Stage-0 planner)

```
You are the paper experiment planner. Given the current run's analysis and manifest, decide whether the manuscript would benefit from additional experiments before figure work begins. Acceptable extensions: (a) parametric sweeps completing a partially-explored sensitivity story; (b) held-out reference cases for DNS/experiment validation; (c) secondary flow configurations testing model-change generalization. Reject: re-runs of failed cases (those belong in the rerun loop), upstream methodology changes (belong in cfd-requirements), speculative extensions unrelated to the topic.

Return ONE JSON object:
{
  "needs_additional_experiments": bool,
  "rationale": "1-2 sentences",
  "additional_cases": [
    {"case_id": "case_NNN", "user_requirement_text": "<paragraph for FoamAgent>", "purpose": "what this adds"}
  ]
}

Constraints: empty additional_cases if needs_additional_experiments=false; unique case_ids not colliding with existing manifest; ≤ 3 cases; prefer false if unsure. Return ONLY valid JSON.
```

### User message
The string built in Step 0.1.

## Step 0.3 — Validate and persist

Write the JSON response to `<out-dir>/paper_experiment_plan.json`. Then:

```bash
PLAN=<out-dir>/paper_experiment_plan.json
jq -e '.needs_additional_experiments != null and (.additional_cases | type == "array")' "$PLAN" \
  || { echo "Stage 0 JSON invalid; redo Step 0.2"; exit 1; }
```

If validation fails: redo Step 0.2 (re-emit the JSON with the correct schema). If it passes, write `<out-dir>/checkpoints/paper_experiment_plan_done.json` with `{"stage": "paper_experiment_plan", "ts": "...", "needs_additional_experiments": <bool>}` and continue.

## Step 0.4 — Branch on Stage 0 verdict

| `paper_experiment_plan.json#needs_additional_experiments` | Next action |
|---|---|
| `true` | For each `additional_cases[i]`, append to `requirements.json` and invoke `Skill cfd-skills/cfd-experiment` (chain: experiment → viz → interpret per case). When all proposed cases have `decision.json` with `PROCEED`, invoke `Skill cfd-skills/cfd-cross-analyze`. After cross-analyze returns, **GOTO Phase A** below. |
| `false` | Invoke `Skill cfd-skills/cfd-cross-analyze`. After cross-analyze returns, **GOTO Phase A** below. |

---

# The cycle — Phases A through F

After Stage 0 + (any additional experiments) + `cfd-cross-analyze` have produced `cross_experiment_analysis/` with comparison figures and `cross_experiment_interpretation.md`, the paper writing cycle begins. You will execute up to `max_review_loops = 10` iterations of the following six-phase cycle:

```
Phase A — Planner (one LLM call; only on iteration 0)
Phase B — Batch PyVista figure script (one LLM call to author; bash to run; inner-loop revise if needed)
Phase C — Per-image vision QA (one vision pass per PNG; revise script via Phase B if any fail)
Phase D — Writer (iter 0) OR Reviser (iter ≥ 1) — one LLM call
Phase E — pdflatex compile (bash; compile-fix LLM call on failure, capped at 3 retries)
Phase F — Reviewer (one LLM call) → branch on verdict
```

Initialize the cycle state file before Phase A:

```bash
OUT=<out-dir>
mkdir -p "$OUT/checkpoints" "$OUT/paper" "$OUT/paper_figs" "$OUT/paper/history"
test -f "$OUT/paper_cycle_state.json" || cat > "$OUT/paper_cycle_state.json" <<'EOF'
{
  "iteration": 0,
  "max_review_loops": 10,
  "score_threshold": 0.7,
  "stage": "starting",
  "regenerate_batch_figures": true,
  "needs_additional_visualization": false,
  "additional_viz_specs": [],
  "recommendations": [],
  "pass": false,
  "viz_inner_attempts": 0,
  "compile_inner_retries": 0
}
EOF
echo "cycle state initialized: $(cat $OUT/paper_cycle_state.json)"
```

Append `{"stage": "paper_review", "event": "cycle_started", "ts": "<iso>"}` to `<out-dir>/timeline.json`.

**GOTO Phase A.**

---

# Phase A — Planner (one LLM call; iteration 0 only)

Only runs once. If `paper_unified_plan.json` already exists and passes schema validation, **skip Phase A and GOTO Phase B**.

## A.1 — Build the planner input string

Concatenate (≤ 120,000 chars; trim lowest-citation lit entries first if larger):

```
TOPIC: <state.json#topic>

MANIFEST SUCCESSFUL CASES (from manifest.json):
<the case objects with status=success>

REQUIREMENTS:
<requirements.json>

ANALYSIS:
<analysis.json>

LITERATURE (top 40 by citation count desc, year desc):
<lit.json top 40 entries, each with title/authors/year/venue/paperId/externalIds.DOI/abstract truncated to 400 chars>

MESH INDEPENDENCE BUNDLE:
<mesh_independence_context.json if present else {}>

STARTER UNDERSTANDING:
<starter_understanding.json relevant fields if present>

CROSS-EXPERIMENT CASE CONTEXT:
<cross_experiment_analysis/case_context.json if present>
```

## A.2 — Run the planner LLM call

**Your next response MUST be the planner's JSON output. Nothing else.**

### System prompt (you are the planner)

```
You are the planning head for a CFD journal manuscript tied to an automated experiment run.

Your job is to return ONE JSON object (no markdown fences) with this schema:
{
  "omit_case_ids": ["case_xxx", ...],
  "omit_rationale": "short string",
  "case_order_for_paper": ["case_001", ...],
  "case_display_labels": {"case_001": "1", "case_003": "2", ...},
  "figure_jobs": [
    {
      "id": "case_001_u_contour",
      "case_id": "case_001",
      "kind": "u_contour",
      "out_png": "paper_figs/case_001_u_contour.png",
      "caption_hint": "Streamwise velocity field for case 1",
      "what_to_visualize": "detailed instruction for PyVista: fields, views, profiles; mention horizontal layout if channel is long/thin",
      "priority": 1
    }
  ],
  "unified_viz_brief": "ONE string for a single batch PyVista script: list every PNG to create (names suggested), multi-panel layouts (rows/columns), fields (Ux, etc.), coherent colormap/scaling across cases, horizontal layout for thin channels. Omit nuEff unless needed and likely present.",
  "analysis_narrative_hints": "what the Results/Discussion should emphasize given the experiments",
  "topic_paragraph_for_intro": "one paragraph introducing the study without case IDs"
}

Rules:
- Schema (HARD; enforced by H16/H15 in stage_gate_audit.py — non-negotiable):
  - Every figure_jobs[i] MUST carry a non-null `out_png` field (path under `paper_figs/`). The writer treats `out_png` as the canonical filename to embed via \includegraphics.
  - `out_png` MUST be UNIQUE across all figure_jobs. No two jobs may share the same filename. Do NOT label 8 different per-case contour jobs with the same 3 cross-summary filenames — the audit catches this and fails the run.
  - For per-case kinds (`u_contour`, `p_contour`, `velocity_magnitude_contour`, `cf_profile`, `pressure_profile`, `streamlines`): the `out_png` filename MUST contain the `case_id` as a substring (e.g. `case_001_u_contour.png` not `cf_overlay_comparison.png`), AND the file must be written into `paper_figs/` by Phase B's batch PyVista script. Reusing a cross-summary PNG and labeling it as a per-case contour is not allowed.
  - `kind` must be one of: `u_contour`, `p_contour`, `velocity_magnitude_contour`, `cf_profile`, `pressure_profile`, `streamlines`, `cross_overlay`, `cross_bar`, `cross_trend`, `cross_delta_field`.
  - For each `case_id` in `case_order_for_paper`, you MUST emit at least one `u_contour` AND at least one `p_contour` figure_job, plus any case-relevant profile (`cf_profile`, etc.) tied to the study's QoI. Per-case contours are the visual backbone of the paper — they cannot be replaced by cross-summary plots alone.
  - Cross-experiment summary figures live under `cross_experiment_analysis/`, NOT in `figure_jobs[]`. The writer embeds those separately (one of them must be cited per H8 — see Phase D requirements).
- Duplicates: if two experiments are true repeats (same intent, same parameters, no new science), put redundant case_ids in omit_case_ids. No figure_jobs for omitted cases. Readers should never hear about duplicates — just omit.
- Naming: display labels are human prose: "1", "2" mapping (string values). Never use "case_001" in narrative instructions; the writer will map IDs internally.
- Figures: each figure_job targets one case_id. Requests must be achievable with PyVista reading the case .foam file. Prefer U, p, velocity magnitude, streamwise profiles. Avoid nuEff/effective viscosity unless explicitly likely present.
- Order: case_order_for_paper lists included cases only.

Return ONLY valid JSON.
```

### User message
The string built in A.1.

## A.3 — Validate and persist

Write the JSON response to `<out-dir>/paper_unified_plan.json`. Then validate:

```bash
PLAN=<out-dir>/paper_unified_plan.json
jq -e '
  .figure_jobs and (.figure_jobs | type == "array") and (.figure_jobs | length >= 1)
  and ([.figure_jobs[] | select(.out_png == null or (.out_png|type) != "string" or .out_png == "")] | length == 0)
  and ([.figure_jobs[] | select(.kind == null or (.kind|type) != "string")] | length == 0)
  and ([.figure_jobs[] | select(.case_id == null)] | length == 0)
  and ((.figure_jobs | map(.out_png) | unique | length) == (.figure_jobs | length))
  and ([.figure_jobs[]
        | select(.kind | test("(u_contour|p_contour|velocity_magnitude_contour|cf_profile|pressure_profile|streamlines)"; "i"))
        | select((.out_png | contains(.case_id)) | not)] | length == 0)
  and ([.case_order_for_paper[] as $cid | .figure_jobs | map(select(.case_id == $cid and (.kind | test("u_contour"; "i")))) | length] | min // 0) >= 1
  and ([.case_order_for_paper[] as $cid | .figure_jobs | map(select(.case_id == $cid and (.kind | test("p_contour"; "i")))) | length] | min // 0) >= 1
  and .unified_viz_brief and (.unified_viz_brief | length > 50)
  and .case_order_for_paper and (.case_order_for_paper | length >= 1)
  and .case_display_labels and (.case_display_labels | type == "object")
  and (.omit_case_ids | type == "array")
  and .analysis_narrative_hints
  and .topic_paragraph_for_intro
' "$PLAN"
```

The schema gate above enforces H16 / H15 deterministically: every `figure_jobs[i]` carries a non-null `out_png` and `kind`; ALL `out_png` values are unique (no PNG reused); every per-case contour/profile job has its `case_id` substring in the `out_png` filename (blocks labelling a cross-summary PNG as a per-case render); and every included case has at least one `u_contour` AND one `p_contour` job. If any check fails, the planner output is wrong — redo A.2 with the corrected JSON.

If validation fails (any key missing, figure_jobs empty, etc.): redo A.2. The previous response had the wrong schema; emit a new JSON response that matches the schema exactly.

If validation passes:

```bash
cat > <out-dir>/checkpoints/paper_planner_done.json <<EOF
{"stage": "paper_planner", "ts": "$(date -u +%FT%TZ)", "n_figure_jobs": $(jq '.figure_jobs | length' <out-dir>/paper_unified_plan.json), "n_cases_included": $(jq '.case_order_for_paper | length' <out-dir>/paper_unified_plan.json)}
EOF
```

Update `paper_cycle_state.json#stage = "planner_done"`.

**GOTO Phase B.**

---

# Phase B — Batch PyVista figure script

Author ONE Python script that renders every paper figure in one go. Run it. Save the PNGs to `<out-dir>/paper_figs/`.

## B.1 — Prepare `cases_config.json` and `paper_fig_layout.py`

For every `case_id` in `paper_unified_plan.json#case_order_for_paper` (intersected with successful cases in `manifest.json`), build a config:

```bash
OUT=<out-dir>
python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ.get("OUT", "."))
plan = json.loads((out / "paper_unified_plan.json").read_text())
manifest = json.loads((out / "manifest.json").read_text())
case_dirs = {c["case_id"]: c["case_dir"] for c in manifest.get("cases", []) if c.get("status") == "success"}
included = [cid for cid in plan.get("case_order_for_paper", []) if cid in case_dirs]
cases = [{"id": cid, "path": case_dirs[cid]} for cid in included]
out_figs = (out / "paper_figs").resolve()
out_figs.mkdir(parents=True, exist_ok=True)
(out_figs / "cases_config.json").write_text(json.dumps({"output_dir": str(out_figs), "cases": cases}, indent=2))
print(f"cases_config.json written with {len(cases)} cases")
PY
```

Copy the layout helper next to the script:

```bash
cp /home/somasn/Desktop/cfd-scientist-arch-change/src/cfd_langgraph/paper_unified/paper_fig_layout.py <out-dir>/paper_figs/paper_fig_layout.py
```

## B.2 — Build the script-author input

Concatenate:

```
Topic / study:
<state.json#topic>

Unified figure brief (what the manuscript needs):
<paper_unified_plan.json#unified_viz_brief>

Per-case hints from planner (figure_jobs summary):
<paper_unified_plan.json#figure_jobs as JSON, indented, truncated to 12000 chars>

cases_config.json absolute path:
<absolute path to <out-dir>/paper_figs/cases_config.json>

The script will live in the same directory as cases_config.json (paper_figs/). Write complete Python.
Output ONLY raw code, first line must be import. No markdown fences.

Previous errors / per-image QA failures:
<paper_cycle_state.json#viz_previous_feedback if set else ''>

Previous script (revise minimally to fix failing outputs only when feedback targets specific files):
<contents of <out-dir>/paper_figs/paper_viz_batch.py if it exists, else ''>
```

## B.3 — Run the script-author LLM call

**Your next response MUST be raw Python code only. First line must be `import`. No markdown fences, no prose, nothing else.**

### System prompt (you are the batch PyVista script writer)

```
You write Python that uses PyVista to render publication-quality CFD figures from one or more OpenFOAM case directories. Use PyVista (not matplotlib for the final PNG; matplotlib only as overlay if absolutely needed). Save to file with plotter.screenshot(...). Camera: orient so the dominant flow direction is horizontal. Mesh: load via reader = pv.OpenFOAMReader(case_dir / "<case_name>.foam"); reader.set_active_time_value(...); mesh = reader.read()["internalMesh"]. Slices: use mesh.slice(normal=...) when needed. Title and axis text: use plotter.add_text(...) sparingly; rely on colorbars for field magnitudes.

MULTI-CASE BATCH MODE:
- You write ONE Python script that loads **every** case listed in cases_config.json (path given below).
- Each case has 'id' and 'path' (OpenFOAM case root). Touch or ensure <case_folder_name>.foam exists in each path.
- Save **multiple** PNG files into the output directory given in the config. Use clear names, e.g. `01_case_001_Ux_full.png`, `02_case_001_Ux_profile.png`, `10_multi_panel_overview.png`.
- Prefer **coherent styling** across cases: same colormap limits where comparable, same font sizes, same layout logic.
- You may create **multi-panel** figures (subplots / multiple render windows saved sequentially) when it helps the paper.
- Read config from: `cfg_path = Path(__file__).resolve().parent / "cases_config.json"` then `json.loads(cfg_path.read_text())`.
- `output_dir = Path(cfg["output_dir"])` — save all PNGs there.
- Use **only** PyVista for PNG output (screenshot); no matplotlib.pyplot.savefig.
- Release large meshes between cases if needed (`del mesh; import gc; gc.collect()`).

LAYOUT CONTRACT (mandatory — same folder as this script contains `paper_fig_layout.py`):
- At startup call `from paper_fig_layout import configure_paper_figure_theme, paper_window_size, padded_bounds_for_thin_domain, two_panel_positions` then `configure_paper_figure_theme()` once before creating plotters.
- Use `paper_window_size()` for `Plotter(window_size=...)` or equivalent so all PNGs share publication-sized frames.
- For **two-panel** (contour | wall-normal profile) figures, use `two_panel_positions()` for `plotter.subplot(..., position=...)` or equivalent so the profile panel stays ~44% of width — do not shrink the profile to a unreadable strip.
- For very thin channel meshes, call `padded_bounds_for_thin_domain(mesh.bounds)` (or slice bounds) before `reset_camera` / `view_xy` so the wall-normal direction is not collapsed to a line.
- Mesh extent, camera angle, field ranges, and slice locations remain **data-driven** from each case; only reuse the constants/helpers from `paper_fig_layout` for fonts, window size, subplot boxes, and thin-domain padding.

OUTPUT TARGETS:
- For every case in cases_config.json: at minimum a velocity-magnitude or Ux contour PNG, a streamwise-profile PNG, and one auxiliary (pressure or vorticity or wall y+) PNG. That's ≥ 3 PNGs per case.
- Plus 1–2 multi-panel comparison figures across cases (Ux contours stacked, streamwise profile overlays).
- Total PNGs ≥ 3 × N_cases + 2.
- Per-case `paper_figs/<case_id>_u_contour.png` and `paper_figs/<case_id>_p_contour.png` filenames are MANDATORY (the planner's figure_jobs[] references these). Render real PyVista field plots, not stylized placeholders.

ANTI-PLACEHOLDER RULES (HARD — audit gate H29 fails any run that violates these):
- If pv.POpenFOAMReader fails on a case, STOP. Surface the error in the script's stderr. Do NOT fall back to matplotlib synthetic data, do NOT plot a sine wave, do NOT save an empty axes with a colorbar. The right action when PyVista fails on a case is: (a) check that `<case>/<basename>.foam` exists or touch it; (b) check that the case has time directories beyond `0/`; (c) check `reader.time_values` is non-empty; (d) check that the requested field (`U`, `p`) is in `reader.cell_data_array_names` or `mesh.array_names`; (e) re-render. Only after all of these are confirmed and PyVista still fails should you raise.
- The audit measures PNG file size as a signal of placeholder substitution. PyVista renders of OpenFOAM fields at 1600×800 typically weigh 200–500 KB. matplotlib placeholders with sparse synthetic data weigh < 50 KB. Per-case contour PNGs < 80 KB fail H29.
- Use `cmap="viridis"` for velocity, `cmap="coolwarm"` for pressure, never the matplotlib default `viridis` palette via matplotlib (the field render must come from PyVista — `pv.Plotter.add_mesh(scalars=..., cmap=...)`).
- Render at ≥ 1200×600 (preferably 1600×800 for wide geometries like BFS / periodic hill / channel).
- Use `parallel_projection=True` on the camera; call `p.view_xy()` then `p.reset_camera()` to frame the full domain.
- Colorbar inside the frame, positioned on the right, never overlapping the geometry. Use the helper conventions from `paper_fig_layout`.

CANONICAL PER-CASE RENDER FUNCTION (use this pattern verbatim, adapted for the field name):
```python
def render_field_contour(case_dir, field_name, out_png,
                        cmap="viridis", title="", window_size=(1600, 800)):
    """Render one field's contour on the internal mesh at the latest time.
    Mandatory: real OpenFOAM data via POpenFOAMReader. No matplotlib fallback.
    """
    import os; os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    import numpy as np
    from pathlib import Path
    pv.OFF_SCREEN = True
    case_dir = Path(case_dir)
    foam_files = list(case_dir.glob("*.foam"))
    if not foam_files:
        stub = case_dir / "case.foam"; stub.touch(); foam_files = [stub]
    reader = pv.POpenFOAMReader(str(foam_files[0]))
    if not reader.time_values:
        raise RuntimeError(f"no time directories in {case_dir} — case didn't run")
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
    if field_name == "Umag":
        if "U" not in internal.array_names:
            raise RuntimeError(f"U not in mesh at t={reader.time_values[-1]} in {case_dir}")
        internal["Umag"] = np.linalg.norm(internal["U"], axis=1)
        scalars = "Umag"; bar_title = "|U|"
    elif field_name in internal.array_names:
        scalars = field_name; bar_title = field_name
    else:
        raise RuntimeError(f"{field_name} not in mesh ({internal.array_names}) in {case_dir}")
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.add_mesh(internal, scalars=scalars, cmap=cmap, show_edges=False,
               scalar_bar_args=dict(title=bar_title, n_labels=5, fmt="%.2f",
                                    position_x=0.78, position_y=0.10,
                                    width=0.18, height=0.55))
    p.view_xy(); p.camera.parallel_projection = True; p.reset_camera()
    # FIT TO FRAME (mandatory — the mesh must FILL the rendered image, not
    # float in white space). Without this, periodic-hill / BFS / channel
    # geometries render as small ribbons with huge empty borders.
    try:
        p.camera.tight(padding=0.02, view="xy")
    except (AttributeError, TypeError):
        (xmin, xmax, ymin, ymax, _, _) = internal.bounds
        cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        p.camera.SetFocalPoint(cx, cy, 0.0)
        p.camera.SetPosition(cx, cy, 1.0)
        p.camera.SetViewUp(0.0, 1.0, 0.0)
        p.camera.SetParallelScale(0.5*(ymax-ymin)*1.02)
    if title: p.add_text(title, position="upper_left", font_size=10)
    p.screenshot(str(out_png)); p.close()
```

FIT-TO-FRAME RULE (HARD): every PyVista contour PNG must show the mesh filling ≥ 85% of the frame area. If you see large empty borders around the rendered geometry, the camera was not tightened — use `p.camera.tight(padding=0.02, view="xy")` (PyVista ≥ 0.43) or the manual `SetParallelScale` fallback above. A "postage stamp on white" render is rejected by Phase C vision QA.

LINE PLOTS AND CROSS-CASE OVERLAYS (matplotlib IS acceptable here, but data must come from PyVista):
- For lower-wall Cf overlays, wall-pressure profiles, and centerline velocity profiles: extract the data from each case via PyVista's boundary patch / line probe pipeline (see cfd-paper-writer's "Lower-wall Cf line plot" section), then plot via matplotlib with proper axis labels (x/h, C_f, units), legend per case, gridlines, and a DNS / experimental reference curve when available in `reference_data/`.
- Use matplotlib only as the *plotter* of PyVista-extracted data, never as a synthetic-data generator.
```

### User message
The string built in B.2.

## B.4 — Write and run the script

Write the response to `<out-dir>/paper_figs/paper_viz_batch.py` (overwrite if present). Strip any code fences first if they leaked in. Then:

```bash
cd <out-dir>/paper_figs
xvfb-run -a python paper_viz_batch.py 2>&1 | tee run_log.txt
RC=${PIPESTATUS[0]}
```

If `xvfb-run` isn't available, set `pv.OFF_SCREEN = True` at the top of the script and re-run without xvfb.

## B.5 — Initial output validation

```bash
N_PNG=$(ls <out-dir>/paper_figs/*.png 2>/dev/null | wc -l)
N_CASES=$(jq '.case_order_for_paper | length' <out-dir>/paper_unified_plan.json)
MIN=$((N_CASES * 2))
if [ "$N_PNG" -lt "$MIN" ] || [ "$RC" -ne 0 ]; then
  # Increment inner attempts; if cap hit, surface failure
  ATT=$(jq -r '.viz_inner_attempts // 0' <out-dir>/paper_cycle_state.json)
  ATT=$((ATT + 1))
  if [ "$ATT" -ge 10 ]; then
    echo "viz inner-loop cap hit ($ATT). Failing the cycle."
    exit 1
  fi
  # Update state with feedback and goto B.2 to re-author the script
  jq ".viz_inner_attempts = $ATT | .viz_previous_feedback = \"Script run rc=$RC; produced $N_PNG PNGs but need ≥ $MIN. Last 60 lines of run_log.txt: $(tail -60 <out-dir>/paper_figs/run_log.txt | jq -Rs .)\"" <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
  echo "GOTO Phase B.2 to revise script (attempt $ATT)"
  # GOTO B.2
fi
```

If B.5 says GOTO B.2, the agent's next action is to re-execute Phase B starting at B.2 with the feedback loaded. Otherwise:

**GOTO Phase C.**

---

# Phase C — Per-image vision QA

You ARE the vision LLM here. Open each PNG in `<out-dir>/paper_figs/` and judge it. Use your native multimodal capability.

## C.1 — Iterate over every PNG

For each `<out-dir>/paper_figs/*.png` (excluding `cases_config.json`, `.py` files):

**Open the image.** Then, with this system prompt as your judging criteria, your next message about that file is the verdict JSON.

### System prompt (you are the per-image judge)

```
You judge ONE CFD figure PNG for journal use (layout/legibility AND data authenticity).

REJECT if any of the following:
- Essentially blank.
- Colorbar / legend overlaps plotted data and hides trends.
- Axis or colorbar text illegibly small at typical column width.
- 2D channel spatial panel is a thin unreadable sliver.
- Required second panel missing when the figure brief explicitly requires it.

REJECT WITH HIGH PRIORITY (these indicate matplotlib-placeholder substitution, not real PyVista field renders):
- For files named "*_u_contour.png", "*_p_contour.png", "*_velocity_magnitude_contour.png": the image must show a real OpenFOAM field on the actual mesh geometry, with smooth color gradients across the domain. If you see crisp matplotlib-axes lines, an empty axes frame with only a colorbar, a sine-wave style synthetic field, perfectly uniform color stripes, or a "demo data" placeholder grid, REJECT with reason="placeholder_not_pyvista_field".
- For periodic-hill / BFS / channel geometries: the geometry outline (curved hill, backward step, rectangular channel) must be visible. A pure rectangular axes box with arbitrary color fill is not a CFD contour.
- If the figure looks like matplotlib imshow on a uniform grid instead of PyVista's mesh render with shaded cells, REJECT with reason="matplotlib_imshow_not_pyvista".

REJECT if the geometry is not fit-to-frame:
- The rendered mesh must FILL the image — occupy ≥ 85% of the frame area. If the geometry sits as a small ribbon or postage stamp surrounded by large empty white / background borders, REJECT with reason="not_fit_to_frame". The script should re-render using p.camera.tight(padding=0.02, view="xy") or a manual SetParallelScale tight-fit.

DO NOT reject solely because you describe the whole figure as upside-down, mirrored, or rotated: PyVista exports are upright by construction. Ignore orientation phrasing unless text in the image is objectively unreadable.

Return ONLY JSON: {"viz_acceptable": bool, "reason": "string"}
```

### Per-image user context
```
Filename: <basename>
Global figure plan summary (truncated to 2000 chars):
<paper_unified_plan.json#unified_viz_brief>

Evaluate ONLY this image.
```

Collect the verdicts as a JSON array in `<out-dir>/paper_figs/qa_iter_<N>.json` (where `<N>` is `paper_cycle_state.json#iteration`):

```json
[{"file": "01_case_001_Ux.png", "viz_acceptable": true, "reason": "..."}, ...]
```

## C.2 — Decide if a script revise is needed

```bash
FAILED=$(jq '[.[] | select(.viz_acceptable == false)]' <out-dir>/paper_figs/qa_iter_<N>.json)
N_FAILED=$(echo "$FAILED" | jq 'length')

if [ "$N_FAILED" -gt 0 ]; then
  ATT=$(jq -r '.viz_inner_attempts // 0' <out-dir>/paper_cycle_state.json)
  ATT=$((ATT + 1))
  if [ "$ATT" -ge 10 ]; then
    echo "viz inner-loop QA cap hit ($ATT); failing cycle"; exit 1
  fi
  # Build per-file feedback string for B.2 retry
  FB=$(echo "$FAILED" | jq -r '.[] | "- \(.file): \(.reason)"')
  jq --arg fb "Per-image QA failed:\n$FB" --argjson att $ATT '.viz_inner_attempts = $att | .viz_previous_feedback = $fb' <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
  echo "GOTO Phase B.2 (revise script for $N_FAILED failing image(s); attempt $ATT)"
  # GOTO B.2 — use SCRIPT_REVISE_USER message instead of SCRIPT_GEN_USER:
  #   "The batch script produced PNGs but some failed individual quality review.
  #    FAILED FILES: <failed list>
  #    STDOUT/STDERR tail: <run_log>
  #    Full current script: <paper_viz_batch.py>
  #    Return ONLY the complete revised Python script (raw code, no fences). First line must be import."
fi
```

If GOTO triggered, re-execute Phase B starting at B.2 with the revise-mode user message. Otherwise (all images pass):

**GOTO Phase D.**

## C.3 — Run the image-analysis pass (after all QA pass)

ONE more LLM call. Attach all PNGs. Your next response is the plain-text image analysis.

### User message
```
You summarize these CFD paper figures for the writer agent.

TOPIC:
<state.json#topic>

ANALYSIS JSON (truncated to 12000 chars):
<analysis.json>

List of figure files (basenames):
<sorted list of paper_figs/*.png basenames>

For each file, give 1–2 sentences: what it shows, which experiment/case it corresponds to (use case folder id if present in filename), and any caveat.
Then 1 short paragraph on how they fit together for Results.

Return plain text (no JSON).
```

Write the response to `<out-dir>/paper_figs/image_analysis.txt`.

```bash
test -s <out-dir>/paper_figs/image_analysis.txt
cat > <out-dir>/checkpoints/paper_viz_done.json <<EOF
{"stage": "paper_viz", "ts": "$(date -u +%FT%TZ)", "iteration": $(jq -r '.iteration' <out-dir>/paper_cycle_state.json), "n_png": $(ls <out-dir>/paper_figs/*.png | wc -l), "inner_attempts": $(jq -r '.viz_inner_attempts // 0' <out-dir>/paper_cycle_state.json)}
EOF
jq '.stage = "viz_done" | .viz_inner_attempts = 0 | .viz_previous_feedback = ""' <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
```

**GOTO Phase D.**

---

# Phase D — Writer (iter 0) OR Reviser (iter ≥ 1)

Read `paper_cycle_state.json#iteration`. If `== 0`, run the writer (full first draft). If `>= 1`, run the reviser (apply reviewer recommendations).

## D.1 — Build the section_context

For every iteration, assemble (≤ 60,000 chars total; trim lowest-citation lit entries first if over):

```json
{
  "topic": "<topic>",
  "template": "neurips",
  "analysis": <analysis.json>,
  "paper_image_analysis": "<image_analysis.txt verbatim>",
  "unified_paper_rules": {
    "abstract_no_case_numbers": true,
    "abstract_no_section_refs": true,
    "use_display_case_labels": true,
    "case_display_labels": <plan.case_display_labels>,
    "included_case_ids": <plan.case_order_for_paper>,
    "omitted_case_ids": <plan.omit_case_ids>,
    "naming": "In narrative use 'case 1', 'case 2' (with space), never case_001. Introduce each case in prose before first mention.",
    "duplicates": "Do not tell readers about omitted or duplicate experiments; omit them silently.",
    "figures": "Figures come from one batch script; use provided paths. Match captions to paper_image_analysis."
  },
  "paper_plan": {
    "analysis_narrative_hints": <plan.analysis_narrative_hints>,
    "topic_paragraph_for_intro": <plan.topic_paragraph_for_intro>,
    "omit_rationale_internal": <plan.omit_rationale>
  },
  "figures": [<relative paper_figs/*.png paths>],
  "manifest_excerpt": {"cases": <success cases from manifest>},
  "mesh_independence": <mesh_independence_context.json if present else {}>,
  "cross_experiment_interpretation": "<cross_experiment_analysis/cross_experiment_interpretation.md if present>"
}
```

## D.2 — Build the literature bundle

From `<out-dir>/lit.json`, take the top 40 records (citation count desc → year desc). Each record retains `paperId, title, authors[], year, venue, externalIds.DOI, abstract (truncated to 400 chars)`.

## D.3 — Build the figure paths block

For each `case_id` in `case_order_for_paper`, list paper_figs paths matching that case_id substring. Also include mesh figure paths if present:

```
FIGURE PATHS BY EXPERIMENT (use in \includegraphics; relative to compile cwd = repo root):
case_001:
  - paper_figs/01_case_001_Ux.png
  - paper_figs/01_case_001_Ux_profile.png
case_002:
  - paper_figs/02_case_002_Ux.png
  ...

MESH FIGURES (if any):
  - paper_figs/mesh_baseline.png
  - paper_figs/mesh_refined.png

CROSS-EXPERIMENT FIGURES (from cfd-cross-analyze):
  - cross_experiment_analysis/fig01_Cf_all_cases.png
  - cross_experiment_analysis/fig05_velocity_profiles.png
  ...
```

## D.4 — Run the writer or reviser LLM call

### System prompt (composition)

The writer's system prompt is built by **concatenating the full content of `cfd-skills/cfd-paper-writer/SKILL.md`** (the structural-discipline skill, adapted from Master-cai/Research-Paper-Writing-Skills) with the short CFD-execution preamble below. This replaces the previous ad-hoc HARD-rule list.

```bash
PROMPT_HEADER=$(cat <<'EOF'
You are the CFD paper writer. Your job is to produce a publication-ready LaTeX manuscript using the structural discipline of the writer skill below, augmented with the CFD-specific HARD rules embedded in that skill.

LENGTH: Main body (Abstract through Conclusion, excluding References and Appendix) must be 8–15 pages.

SCOPE AND TRUTHFULNESS:
- The paper reflects ONLY what was done in the provided experiments.
- Do NOT invent numbers, models, or citations. Every number must trace to analysis.json / manifest.json / bridge.json / metric_vector. Every citation must be a key from refs.bib built procedurally from lit.json.
- Do NOT describe theory or literature that was not actually applied / validated here.

WRITING SKILL (apply every rule below):
EOF
)
WRITER_SKILL=$(cat cfd-skills/cfd-paper-writer/SKILL.md)
SYSTEM_PROMPT="${PROMPT_HEADER}

${WRITER_SKILL}"
```

The writer skill (cfd-paper-writer/SKILL.md) carries:
- The section-by-section structural guidance (Abstract / Introduction / Related Work / Methods / Results / Discussion / Conclusion).
- The claim → evidence → citation discipline.
- The CFD-specific HARD rules (governing equations, closure equations, geometry markers, per-case contours, mesh-independence, quant table, citation distribution, bibliography handling).
- The procedural refs.bib build recipe from lit.json.
- The 25-question adversarial self-review checklist.

The reviser path (iteration ≥ 1) appends the previous reviewer's `recommendations` to the user message; the writer prompt does not change.

### Legacy verbatim writer prompt (deprecated — kept here for backward-compatibility reference only)

The original Phase-D writer prompt is retained below for transparency about what was replaced. Do not use it. Use the composed prompt above.

```
You are a LaTeX PAPER-WRITER for CFD research results. Your job is to produce professional, publication-ready LaTeX documents that compile to PDF.

LENGTH: Main body (Abstract through Conclusion, excluding References and Appendix) must be at least 8 pages and at most 15 pages.

SCOPE AND TRUTHFULNESS (critical):
- The paper must reflect ONLY what was done in the provided experiments. Nothing more, nothing less. No hallucinations.
- Do NOT mention standard literature or theory that was not actually performed or validated in these experiments. If something is common in the field but not done here, omit it.
- Analysis, Discussion, and Conclusion must be grounded strictly in the given visualizations and experiment data. Every claim must map to a specific figure, table, or number from the provided analysis. No unsupported or speculative claims.

FIGURES:
- Include only figures that are good quality and clearly support the text. If an image is poor (blurry, wrong, or uninformative), do not include it.
- Prefer figures suitable for journal print: colorbars and legends must not cover the plotted flow/domain; labels and ticks must be large enough to read when the figure is placed at typical column width. Omit or replace figures that fail these layout checks (common with default PyVista exports).
- You MUST include at least one important/representative figure from each experiment so all experiments are represented. It is not required to include every image from every experiment, but each experiment must have at least one figure.
- For domain-wide contour or field plots, ensure the camera/framing would be readable at journal-print size: the full computational domain should be visible, and key flow features (recirculation bubble, jet, shear layer, bluff body, etc.) must be large enough to interpret.

CORE RESPONSIBILITIES:
1. LaTeX Document Generation: Create complete LaTeX documents with proper structure and formatting
2. Scientific Writing: Write clear, concise, and technically accurate content based only on the experiments
3. Figure Integration: Include only good-quality figures with informative captions; ensure every experiment is represented
4. Bibliography Management: Handle citations and references appropriately
5. Reproducibility: Include methodology and simulation parameters that were actually used

MESH REFINEMENT / INDEPENDENCE (mandatory when data is provided):
- The section context JSON may include a non-empty key `mesh_independence` from the automated mesh-gate study.
- When `mesh_independence` is present and contains `metrics_by_mesh_level` or `mesh_gate_plan`, you MUST add a clear subsection that:
  (1) lists every mesh level compared (coarse, baseline, refined, …);
  (2) presents at least one LaTeX table of cell/point counts and monitored QoIs per level using only numbers from `metrics_by_mesh_level` (no invented values);
  (3) states which mesh level was selected and the criterion used;
  (4) summarizes conclusions from `cross_mesh_analysis_excerpt` where it does not contradict the table;
  (5) includes \includegraphics for mesh plots when `mesh_figure_paths` is non-empty.
- If `mesh_independence` is absent or empty, do not fabricate a mesh study.

WRITING GUIDELINES:
- Use ONLY data from provided JSON bundles and figures; do NOT invent numbers, results, or citations
- Emphasize what was actually simulated (geometry, BCs, solver, mesh, etc.)
- Report only what is evident from the provided analysis and visualizations
- Write in clear, academic style
- Do not discuss theory or literature that was not applied or validated in these experiments
- Do not use the word "formulation" or similar jargon in the paper title; keep the title concise and descriptive.
- When introducing any abbreviation (e.g. RANS, BFS, SA), write out the full term with the abbreviation in parentheses at first mention, and then use the abbreviation consistently thereafter.
- Avoid putting heavy mathematical formulations or long equations in the abstract; the abstract should be primarily qualitative and high-level.
- Use a consistent list style and formatting throughout the paper (same bullet/numbering and indentation rules in all sections).
- Keep figure captions concise and focused on what is shown and why it matters; avoid turning captions into long paragraphs.
- Do not overuse lists, bold, or italics in the main text; reserve them for truly important emphasis so the manuscript remains readable and professional.
- Before referring to experiments by shorthand IDs (exp_001, etc.), introduce a clear experiment-summary table that lists each experiment.
- Avoid duplicating the same explanatory paragraph or numerical comparison in multiple sections.
- The Conclusion section must be a single cohesive paragraph (continuous prose). Never format the Conclusion as bullet points, numbered items, or any list.

LaTeX REQUIREMENTS:
- Use standard LaTeX packages (amsmath, graphicx, subcaption, etc.)
- Proper document structure with sections, subsections, and references
- Professional formatting with consistent spacing and typography
- Proper figure placement and referencing
- Mathematical equations only for relations actually used or verified in the experiments
- Bibliography in standard format (BibTeX or manual)
- Do NOT include a table of contents.
- References section must contain at least 20 distinct, non-duplicate references, drawn from the provided literature bundle.

ABSTRACT (strict):
- Do not name individual case numbers in the Abstract.
- Do not refer readers to section numbers or internal labels in the Abstract.

CASE NAMING IN BODY:
- Never use `case_001`-style identifiers in prose. Use "case 1", "case 2" (with a space) after introducing what each case represents.
- If `unified_paper_rules` lists `omitted_case_ids`, do not mention those experiments at all.

OUTPUT FORMAT:
- Complete LaTeX document (.tex file)
- Main body: 8–15 pages, excluding References and Appendix
- Proper document class and package imports
- Figures: only good-quality images; at least one from each experiment
- Bibliography section included
- Document must compile without errors
```

### User message — iteration 0 (writer mode)

```
Write the complete LaTeX manuscript for this CFD study.

SECTION CONTEXT (JSON):
<section_context as built in D.1>

LITERATURE BUNDLE (top 40 from Semantic Scholar):
<lit_bundle as built in D.2>

<figure paths block from D.3>

REQUIREMENTS:
- Produce ONE complete .tex document, top to bottom, ready for pdflatex.
- Use \documentclass{article} with usepackage{amsmath,amssymb,amsfonts,graphicx,subcaption,booktabs,xcolor,hyperref}.
- Sections in order: Abstract, Introduction, Related Work, Methods (with subsections for flow config + BC + baseline model + custom model/changes if any + experiment design + mesh independence if data present), Results (with subsections per natural grouping — overview, baseline DNS comparison, parametric sensitivity if any, failure cases), Discussion, Conclusion.
- ≥ 10 \cite{<key>} calls across Introduction + Related Work + Methods + Discussion. Every cited key must appear in the embedded bibliography below.
- Bibliography (HARD — H19): produce a \begin{thebibliography}{99}...\end{thebibliography} block AT THE END of the .tex file. The number of \bibitem entries MUST be ≥ the number of distinct \cite{<key>} keys you used (no dangling citations). Build entries from the LITERATURE BUNDLE — use author + year + title + venue + doi. Do not invent references. Minimum 10 entries.
- Equations (HARD — H20): include ≥ 3 display-math blocks (\begin{equation}, \begin{align}, or \[ … \]) covering: (i) the governing equations (continuity + RANS momentum), (ii) the turbulence closure used (k-ω SST / Spalart-Allmaras / Bingham / Carreau / …), and (iii) the QoI / comparator formula (Cf, x_r/h, RMSE definition, or the source-term modification under study). For OED or code-modification routes also write the modification equation explicitly.
- Quantitative results table (HARD — H22): include at least one \begin{tabular} environment in Methods or Results that lists numeric per-case QoI values (e.g. baseline vs candidates vs reference DNS): each row a case/model, columns covering the study's metrics (Cf RMSE, x_r/h, peak Cf, drag, …). The table must contain ≥ 3 numeric tokens. Numbers must come from the analysis bundle — do not invent values.
- No duplicate paragraphs (HARD — H21): every paragraph of body prose must be distinct. Do NOT copy/paste the Introduction (or any other paragraph) into Related Work, Methods, or any other section to fill length. If you have nothing more to say, the paper is too short — go deeper, do not duplicate.
- No meta-paragraph references (HARD — H24): NEVER write phrases like "in introduction paragraph 1", "see paragraph 7", "paragraph 7 of related work", "the previous paragraph", "the following paragraph". Real journal papers do not reference paragraphs by number. Each sentence stands on its own. Cross-references go to figures (Fig. 3), tables (Table 2), sections (Section 4), or equations (Eq. 5) — never to paragraphs.
- Citation distribution (HARD — H23): every one of {Introduction, Related Work, Methods, Discussion} must contain at least one \cite{} call woven into the sentence where each claim is made. NEVER cluster all citations at the end of one paragraph as `\cite{ref1} \cite{ref2} … \cite{ref10}` — that is listing, not citing. No single section may hold more than 60% of the paper's total citations.
- Governing equations (HARD — H25 part 1): at least one of your display equations MUST be a governing equation the solver actually integrates — incompressible continuity (∇·U = 0) and Reynolds-averaged momentum, or the equivalent. Reynolds number, Cf, and RMSE definitions are metric formulas, not governing equations — they do not satisfy this requirement alone.
- Closure equations (HARD — H25 part 2): at least one of your display equations MUST be the turbulence-closure transport or constitutive equation for the model(s) under study — k-ω SST k-equation and ω-equation, k-ε k-equation and ε-equation, Spalart-Allmaras nuTilda transport (with c_b1, c_b2, c_w1, σ, κ), Bingham yield-stress relation, Carreau η(γ̇) law, etc. For a turbulence-model comparison paper the closure equations ARE the paper.
- Geometry description (HARD — H26): Methods MUST describe the geometry concretely: step height h, expansion ratio, channel/hill height, domain length in step heights, inlet x/h location, outlet x/h location, Re_h, wall BCs. A reader who has never seen this case must be able to reconstruct the geometry from your Methods section alone. Include at least 3 such concrete markers.
- Figures (HARD — H15 tighter): emit ONE \includegraphics for EVERY figure_jobs[i].out_png listed in `paper_unified_plan.json`. Per-case U/p contours are non-negotiable — every included case must show at least its u_contour AND its p_contour in the paper. Cross-experiment summary PNGs (overlay / bar / trend, in `cross_experiment_analysis/`) come in addition, not instead.
- Use ONLY paths from FIGURE PATHS BY EXPERIMENT and the planner's `figure_jobs[].out_png`. Do not invent paths.
- Do not mention omitted case_ids or "duplicates".

Return ONLY the complete LaTeX document. No markdown fences, no explanation.
```

### User message — iteration ≥ 1 (reviser mode)

```
Revise this LaTeX paper according to the reviewer recommendations. Apply ALL recommended fixes. When the reviewer says an experiment (e.g. exp_004) has no figures: use the FIGURE PATHS BY EXPERIMENT and ANALYSIS CONTEXT below to add the correct \includegraphics from that experiment. Use ONLY paths from the list; do NOT invent paths.

LENGTH: Main body 8–15 pages. FIGURES: At least one from each experiment; only good-quality images. SCOPE: No hallucinations; grounded in the given analysis and visualizations.

Return ONLY the complete revised LaTeX document (no markdown, no explanation).

CURRENT LaTeX:
<current paper/main.tex, truncated to 80000 chars>

REVIEWER RECOMMENDATIONS:
<paper_cycle_state.json#recommendations as bullet list, each prefixed with '- '>

<figure paths block from D.3>

ANALYSIS CONTEXT (truncated to 30000 chars):
<section_context as JSON>
```

## D.5 — Persist and validate

Before overwriting, snapshot the prior draft (if exists):

```bash
OUT=<out-dir>
ITER=$(jq -r '.iteration' "$OUT/paper_cycle_state.json")
if [ "$ITER" -gt 0 ] && [ -f "$OUT/paper/main.tex" ]; then
  PREV=$((ITER - 1))
  cp "$OUT/paper/main.tex" "$OUT/paper/history/iter_${PREV}_main.tex"
fi
```

Write the response (strip any code fences first) to `<out-dir>/paper/main.tex`. Also copy to `<out-dir>/paper/paper_draft.tex` for parity with LangGraph artifact naming.

Validate structurally:

```bash
TEX=<out-dir>/paper/main.tex
PLAN=<out-dir>/paper_unified_plan.json
test -s "$TEX" || { echo "main.tex empty"; exit 1; }
MISSING=()
for sec in Introduction "Related Work" Methods Results Discussion Conclusion; do
  grep -qE "\\\\section\\*?\\{\\s*$sec\\s*\\}" "$TEX" || MISSING+=("$sec")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "Mandatory sections missing: ${MISSING[*]}; redo D.4 with the correct sections."
  exit 1
fi
N_CITE=$(grep -oE '\\cite[a-z]*\{[^}]+\}' "$TEX" | wc -l)
[ "$N_CITE" -ge 10 ] || { echo "only $N_CITE \\cite calls; need ≥ 10; redo D.4"; exit 1; }
# H19 — bibliography must cover every \cite
N_BIB=$(grep -cE '\\bibitem\s*(\[[^]]*\])?\s*\{' "$TEX")
N_UNIQ_CITE=$(grep -oE '\\cite[a-z]*\{[^}]+\}' "$TEX" | sed -E 's/.*\{([^}]+)\}/\1/' | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sort -u | wc -l)
[ "$N_BIB" -ge "$N_UNIQ_CITE" ] && [ "$N_BIB" -ge 10 ] || {
  echo "H19: \\bibitem count=$N_BIB < unique \\cite keys=$N_UNIQ_CITE (or < 10); redo D.4 with a complete \\begin{thebibliography} block"; exit 1; }
# H20 — display-math equations
N_EQ=$(grep -cE '\\begin\{(equation|align|gather|multline)\*?\}|^\s*\\\[' "$TEX")
[ "$N_EQ" -ge 3 ] || { echo "H20: only $N_EQ display equations; need ≥ 3 (governing eqns + closure + QoI/modification)"; exit 1; }
# H22 — quantitative results table (tabular with ≥ 3 numeric tokens)
python3 - "$TEX" <<'PY' || exit 1
import re, sys
t = open(sys.argv[1], errors="replace").read()
t = re.split(r"\\appendix|\\section\{\s*Appendix", t, maxsplit=1)[0]
tabs = re.findall(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", t, flags=re.DOTALL)
ok = False
for tab in tabs:
    nums = re.findall(r"(?<![A-Za-z_])-?\d+\.\d+|(?<![A-Za-z_])-?\d{2,}", tab)
    if len(nums) >= 3:
        ok = True
        break
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo "H22: no \\begin{tabular} with ≥ 3 numeric tokens in main body — add a quantitative results table (case × QoI)"; exit 1
fi
# H21 duplicate paragraphs: DEPRECATED in slice 19. cfd-paper-writer skill
# enforces "one paragraph, one message" structurally; reviewer flags via
# prose_quality_issues[type="ngram_loop"] / [type="token_soup"]. No local
# gate — was firing false positives on legitimate restatements.
# H15 — every figure_jobs[i].out_png must appear in \includegraphics
N_FIG=$(grep -c '\\includegraphics' "$TEX")
N_PLAN_JOBS=$(jq '.figure_jobs | length' "$PLAN")
MISSING_PNGS=$(jq -r '.figure_jobs[].out_png' "$PLAN" | while read p; do
  leaf=$(basename "$p"); grep -q "$leaf" "$TEX" || echo "$p"
done)
if [ -n "$MISSING_PNGS" ]; then
  echo "H15: figure_jobs[].out_png not embedded in main.tex:"
  echo "$MISSING_PNGS"
  echo "Emit one \\includegraphics per planned figure_jobs[].out_png"; exit 1
fi
N_CASES=$(jq '.case_order_for_paper | length' "$PLAN")
[ "$N_FIG" -ge "$N_PLAN_JOBS" ] || { echo "$N_FIG figures < $N_PLAN_JOBS planned figure_jobs; redo D.4"; exit 1; }
grep -q '\\tableofcontents' "$TEX" && { echo "TOC present; redo D.4 without it"; exit 1; }
grep -E '\bcase_00[0-9]\b' "$TEX" | grep -v '%' && { echo "case_NNN leaked into prose; redo D.4"; exit 1; }
# H24 meta-paragraph references: DEPRECATED in slice 19. The cfd-paper-writer
# skill forbids these structurally, and the reviewer flags them via
# prose_quality_issues[type="meta_paragraph_ref"]. Local grep retained only as
# a diagnostic — does NOT abort.
META_HITS=$(grep -niE 'in\s+(the\s+)?(introduction|related\s+work|methods?|results?|discussion|conclusion)\s+paragraph\s+[0-9]|see\s+paragraph\s+[0-9]|paragraph\s+[0-9]\s+(of|in)\s+(the\s+)?(introduction|related\s+work|methods?|results?|discussion|conclusion)' "$TEX" | grep -v '^[[:space:]]*%' | wc -l)
[ "$META_HITS" -gt 0 ] && echo "WARN: $META_HITS meta-paragraph references found in main.tex — reviewer will flag these"
# H23 — citation distribution across major sections
python3 - "$TEX" <<'PY' || exit 1
import re, sys
t = open(sys.argv[1], errors="replace").read()
t = re.split(r"\\appendix|\\section\{\s*Appendix", t, maxsplit=1)[0]
parts = re.split(r"\\section\*?\{([^}]+)\}", t)
secs = {parts[i].strip().lower(): (parts[i+1] if i+1 < len(parts) else "") for i in range(1, len(parts), 2)}
required = ["introduction", "related work", "methods", "discussion"]
counts = {s: len(re.findall(r"\\cite[a-z]*\{[^}]+\}", secs.get(s, ""))) for s in required}
total = sum(counts.values()) or 1
print("H23 citation distribution:", counts, "max_share=", round(max(counts.values())/total, 3))
if any(counts[s] < 1 for s in required) or max(counts.values())/total > 0.60:
    print("H23 FAIL: each of Introduction/Related Work/Methods/Discussion needs ≥1 \\cite, no section > 60% of total")
    sys.exit(1)
PY
if [ $? -ne 0 ]; then exit 1; fi
# H25 — governing + closure equations (content, not just count)
python3 - "$TEX" <<'PY' || exit 1
import re, sys
t = open(sys.argv[1], errors="replace").read()
t = re.split(r"\\appendix|\\section\{\s*Appendix", t, maxsplit=1)[0]
eqs = re.findall(r"\\begin\{(?:equation|align|gather|multline)\*?\}.*?\\end\{(?:equation|align|gather|multline)\*?\}|\\\[(.*?)\\\]", t, re.DOTALL)
blob = " ".join(e if isinstance(e,str) else (e[0] if e else "") for e in eqs)
gov_pats = [r"\\nabla\s*\\cdot", r"\\partial.*\\partial\s*t", r"\\overline\{u", r"\\overline\{U", r"Navier.?Stokes", r"continuity", r"momentum\s+equation"]
clo_pats = [r"\\omega", r"\\beta\^?\*", r"\\epsilon", r"\\varepsilon", r"\\tilde\{\\nu\}", r"\\widetilde\{\\nu\}", r"nuTilda", r"Spalart.?Allmaras", r"\\nu_?t", r"\\mu_?t", r"Bingham", r"Carreau", r"power.?law"]
gov = any(re.search(p, blob, re.IGNORECASE) for p in gov_pats)
clo = any(re.search(p, blob, re.IGNORECASE) for p in clo_pats)
if not gov:
    print("H25 FAIL: no equation contains governing-equation markers (∇·U, ∂/∂t, Navier-Stokes, RANS momentum)"); sys.exit(1)
if not clo:
    print("H25 FAIL: no equation contains turbulence-closure markers (k-ω, k-ε, SA transport, ν_t, …)"); sys.exit(1)
PY
if [ $? -ne 0 ]; then exit 1; fi
# H26 — geometry description in Methods (≥ 3 concrete markers)
python3 - "$TEX" <<'PY' || exit 1
import re, sys
t = open(sys.argv[1], errors="replace").read()
parts = re.split(r"\\section\*?\{([^}]+)\}", re.split(r"\\appendix|\\section\{\s*Appendix", t, maxsplit=1)[0])
methods = "".join(parts[i+1] for i in range(1, len(parts), 2) if "method" in parts[i].strip().lower() and i+1 < len(parts)) or t
markers = [r"step\s+height", r"\bh\s*=\s*\d", r"expansion\s+ratio", r"\bER\b", r"channel\s+height", r"hill\s+height", r"domain\s+(?:length|extent|size)", r"inlet.{0,40}(?:length|location|x\s*/\s*h)", r"outlet.{0,40}(?:length|location|x\s*/\s*h)", r"Re_?h\s*=\s*\d", r"x\s*/\s*h", r"y\s*/\s*h", r"blockMesh", r"\bbcs?\b.{0,40}(?:inlet|outlet|wall)"]
hit = [p for p in markers if re.search(p, methods, re.IGNORECASE)]
if len(hit) < 3:
    print(f"H26 FAIL: Methods has only {len(hit)} geometry markers ({hit}); need ≥ 3 from step height / expansion ratio / domain extents / inlet+outlet locations / Re_h / x/h / blockMesh / BCs"); sys.exit(1)
PY
if [ $? -ne 0 ]; then exit 1; fi
```

If any validation fails, redo D.4 with the correction. Otherwise:

```bash
cat > <out-dir>/checkpoints/paper_writer_done.json <<EOF
{"stage": "paper_writer", "ts": "$(date -u +%FT%TZ)", "iteration": $ITER, "n_citations": $N_CITE, "n_figures": $N_FIG, "tex_bytes": $(stat -c%s $TEX)}
EOF
jq '.stage = "writer_done"' <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
```

**GOTO Phase E.**

---

# Phase E — pdflatex compile + compile-fix retry

## E.1 — Try pdflatex (with bibtex when refs.bib is present)

```bash
cd <out-dir>/paper
if [ -f refs.bib ] && command -v bibtex >/dev/null 2>&1; then
  # External BibTeX bibliography: pdflatex → bibtex → pdflatex → pdflatex
  ( pdflatex -interaction=nonstopmode -halt-on-error main.tex \
    && bibtex main \
    && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
    && pdflatex -interaction=nonstopmode -halt-on-error main.tex ) > compile_log.txt 2>&1
else
  # Inline \begin{thebibliography} fallback: pdflatex twice
  ( pdflatex -interaction=nonstopmode -halt-on-error main.tex \
    && pdflatex -interaction=nonstopmode -halt-on-error main.tex ) > compile_log.txt 2>&1
fi
RC=$?
test -s main.pdf && SZ=$(stat -c%s main.pdf) || SZ=0
echo "pdflatex rc=$RC pdf_size=$SZ"
# Surface bibtex warnings for the next iteration of the cycle (dangling cites etc.)
if [ -f main.blg ]; then
  echo "--- bibtex warnings ---"
  grep -iE "warning|i didn't find|empty|undefined" main.blg | head -20 || true
fi
```

If `RC=0` AND `SZ >= 50000`: GOTO E.4.

## E.2 — Parse the error summary

If compile failed, extract from `compile_log.txt`:

```bash
grep -n "^! " compile_log.txt | head -3        # primary errors
grep -nE "LaTeX Error|File .* not found" compile_log.txt | head -10
tail -80 compile_log.txt                       # raw tail for context
```

Build `error_summary` (the parsed bits) and `compile_error_tail` (the raw tail).

Build `valid_figure_paths` — the literal list of PNGs:

```bash
( ls <out-dir>/paper_figs/*.png 2>/dev/null
  ls <out-dir>/cross_experiment_analysis/*.png 2>/dev/null
) | sed 's|.*runs/[^/]*/||'  # paths relative to compile cwd
```

## E.3 — Compile-fix LLM call (cap = 3 retries)

```bash
RETRY=$(jq -r '.compile_inner_retries // 0' <out-dir>/paper_cycle_state.json)
RETRY=$((RETRY + 1))
if [ "$RETRY" -ge 3 ]; then
  echo "compile-fix cap hit ($RETRY); cycle fails"
  cat > <out-dir>/checkpoints/paper_compile_done.json <<EOF
{"stage": "paper_compile", "ts": "$(date -u +%FT%TZ)", "status": "failed_after_compile_fix_cap"}
EOF
  exit 1
fi
jq ".compile_inner_retries = $RETRY" <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
```

**Your next response MUST be the complete corrected LaTeX document. Raw .tex. No markdown, no explanation.**

### System prompt (you are the LaTeX build engineer)

```
You are a LaTeX build engineer. The manuscript below failed pdflatex. Your ONLY goal is to output a complete, valid LaTeX document that compiles.

Rules:
- Do NOT run a content or publishability review. Do not shorten, restructure for style, or rewrite for clarity unless required to fix the error.
- Make the minimum edits needed: wrong \includegraphics paths (use ONLY paths from the valid list), missing packages in the preamble, unclosed environments, bad escape sequences, undefined commands, broken \ref/\cite, or missing \end{document}.
- Preserve scientific content and structure when possible; prefer surgical fixes over rewriting sections.
- Return the ENTIRE document as valid LaTeX — not a patch or excerpt.
```

### User message

```
pdflatex failed. Fix the LaTeX so it compiles.

PRIMARY / PARSED ERRORS (fix in order):
---
<error_summary from E.2>
---

RAW LOG (tail; full context):
---
<compile_error_tail from E.2>
---

VALID paths for \includegraphics (relative to project root / compile cwd):
---
<valid_figure_paths from E.2>
---

CURRENT FULL LaTeX:
---
<current paper/main.tex>
---

Return ONLY the complete corrected LaTeX document (no markdown, no explanation).
```

Write the response to `<out-dir>/paper/main.tex`. **GOTO E.1.**

## E.4 — Compile succeeded

```bash
PAGES=$(pdfinfo <out-dir>/paper/main.pdf | awk '/^Pages:/ {print $2}')
cat > <out-dir>/checkpoints/paper_compile_done.json <<EOF
{"stage": "paper_compile", "ts": "$(date -u +%FT%TZ)", "iteration": $(jq -r '.iteration' <out-dir>/paper_cycle_state.json), "pdf_pages": $PAGES, "pdf_bytes": $(stat -c%s <out-dir>/paper/main.pdf), "inner_retries": $(jq -r '.compile_inner_retries // 0' <out-dir>/paper_cycle_state.json), "status": "ok"}
EOF
jq '.stage = "compile_done" | .compile_inner_retries = 0' <out-dir>/paper_cycle_state.json > /tmp/cs && mv /tmp/cs <out-dir>/paper_cycle_state.json
```

**GOTO Phase F.**

---

# Phase F — Reviewer (one LLM call) → branch on verdict

You are the reviewer. Read the .tex, the compile log, the figure list. Return a structured JSON verdict.

## F.1 — Build the reviewer inputs

- `compile_status = "succeeded"` (precondition).
- `tex_content` = `<out-dir>/paper/main.tex`, truncated to 80000 chars.
- `compile_error` = last 4000 chars of `compile_log.txt` (near-empty on success is fine).
- `reference_report` = "" (refchecker not run inline; pass empty).
- `valid_figure_paths` = list of `paper_figs/*.png` + `cross_experiment_analysis/*.png` (relative to project root).

## F.2 — Run the reviewer LLM call

**Your next response MUST be a JSON object with EXACTLY the keys listed in the user message. No markdown fences, no prose outside the JSON.**

### System prompt (composition)

The reviewer's system prompt is built by concatenating the writer skill (which carries the **adversarial self-review checklist** + **5-category rejection-risk evaluation** + **prose_quality_issues taxonomy**) with the short reviewer preamble below:

```bash
REVIEWER_HEADER=$(cat <<'EOF'
You are an expert academic paper reviewer for CFD and engineering journals. You read the LaTeX manuscript + compile log + figure list + analysis bundle + lit.json and return a structured JSON verdict using the schema embedded in the writer skill below.

Use the writer skill's Reviewer Rubric section as your evaluation protocol verbatim:
- The 5-category rejection-risk evaluation (insufficient contribution / unclear writing / weak empirical effect / incomplete evaluation / problematic method design).
- The 25-question adversarial self-review checklist as the basis for `score = n_passing / 25`.
- The `prose_quality_issues` taxonomy: ngram_loop / token_soup / meta_paragraph_ref / citation_cluster / orphan_figure / missing_governing_eq / missing_closure_eq / missing_geometry / invented_number.

When compilation FAILED: your job is to fix the LaTeX so it compiles. Focus on the FIRST error in the compile log. Set pass=false; recommendations give 1–3 concrete fixes for the primary error.

When compilation SUCCEEDED: evaluate publication readiness using the rubric below. Set pass=true ONLY IF all five rejection-risk categories pass AND ≥ 20 of 25 adversarial checks pass AND no critical prose_quality_issues (n-gram loops, token soup, citation clusters, missing governing/closure equations, missing geometry).

Set regenerate_batch_figures=true only if figure issues require re-running Phase B (wrong field, wrong orientation, missing panel, colorbar covers data). Set false when issues are text / LaTeX / citations only.

Be strict but fair. Recommendations must be specific, actionable, and ordered by impact.

WRITING SKILL (use as your evaluation rubric):
EOF
)
WRITER_SKILL=$(cat cfd-skills/cfd-paper-writer/SKILL.md)
REVIEWER_SYSTEM_PROMPT="${REVIEWER_HEADER}

${WRITER_SKILL}"
```

### Legacy reviewer prompt (deprecated — retained for transparency)

The previous reviewer prompt is below. Do not use. Use the composed prompt above.

```
You are an expert academic paper reviewer for CFD and engineering journals. You operate in two modes depending on compilation outcome.

When compilation FAILED: Your job is to fix the LaTeX so it compiles. Focus on the FIRST error listed. Common causes: (1) "File X not found" — \includegraphics path wrong; use ONLY paths from VALID figure paths; (2) missing \end{document}; (3) undefined control sequence; (4) unclosed environment. Give 1–3 concrete recommendations that fix the PRIMARY error. Set pass=false.

When compilation SUCCEEDED: Evaluate publication readiness. Check: (1) Formatting — structure, \ref/\cite, typography; NO table of contents; (2) Figures — paths valid, only good-quality images, ≥ 1 figure per experiment, captions correct, referenced in text; colorbars/legends must not obscure data; axis/colorbar text legible at print size; reject matplotlib-looking contour junk when PyVista was asked for; (3) Content — reflects only the experiments, no hallucinations; (4) Abstract hygiene — FAIL if Abstract names case numbers or points to section numbers; (5) Case naming — FAIL if text uses `case_001`-style IDs in prose; (6) Coherence; (7) Length — main body 8–15 pages; (8) References — solid bibliography when citations exist; (9) Redundancy — no duplicated paragraphs; (10) Conclusion format — single paragraph, not a list; (11) Publishability. Set pass=true only if all are acceptable; otherwise give specific recommendations.

Unified pipeline / extra figures: If figures are missing, wrong orientation, or overlap legends, set needs_additional_visualization true and add short imperative strings to additional_viz_specs (each mentioning a case_NNN id and what to plot).

Unified pipeline / batch PyVista script: Set regenerate_batch_figures true only if replacing/adding generated PNGs is required (bad layout, wrong field, missing panel, new viz spec). Set false when fixes are LaTeX/prose/citations/abstract only.

Be strict but fair. Recommendations must be specific and actionable.
```

### User message

```
Review this LaTeX paper. Compilation determines your focus.

Compilation status: succeeded

LaTeX content:
---
<tex_content>
---

pdflatex compile log:
---
<compile_error>
---

RefChecker report summary:
---
<reference_report>
---

Return ONLY valid JSON with these EXACT keys:
- pass (bool)
- score (float 0-1)
- formatting_ok (bool)
- figures_ok (bool)
- content_ok (bool)
- coherent (bool)
- publishable (bool)
- recommendations (list of strings)
- summary (string, 1-2 sentences)
- needs_additional_visualization (bool)
- additional_viz_specs (list of strings, each with a case_NNN id)
- regenerate_batch_figures (bool, REQUIRED when compile succeeded)
- figure_physics_ok (bool, true if per-figure physics-correctness pass shows no sign/axis/feature errors)
- prose_quality_issues (list of {type, section, evidence, fix} objects; type is one of:
    "ngram_loop", "token_soup", "meta_paragraph_ref", "citation_cluster",
    "orphan_figure", "missing_governing_eq", "missing_closure_eq",
    "missing_geometry", "invented_number")
- rejection_risks (list of strings naming each of the 5 categories that fires:
    "insufficient_contribution", "unclear_writing", "weak_empirical_effect",
    "incomplete_evaluation", "problematic_method_design")
- adversarial_checklist_pass_count (int 0-25 — number of the 25 adversarial-checklist
    items that pass)

No markdown, no explanation outside the JSON.
```

## F.3 — Validate and persist

Write the response to `<out-dir>/review.json` (overwrite each iteration). Append a snapshot to `<out-dir>/paper_review_history.json` (array):

```bash
OUT=<out-dir>
ITER=$(jq -r '.iteration' "$OUT/paper_cycle_state.json")
# Validate review.json schema
jq -e '.pass != null and .score != null and (.recommendations | type == "array") and .formatting_ok != null and .figures_ok != null and .content_ok != null and .summary' "$OUT/review.json" \
  || { echo "review.json schema invalid; redo F.2"; exit 1; }
# Append to history (initialize if missing)
test -f "$OUT/paper_review_history.json" || echo '[]' > "$OUT/paper_review_history.json"
jq --argjson iter $ITER --slurpfile rev "$OUT/review.json" '. + [{"iteration": $iter, "verdict": $rev[0]}]' "$OUT/paper_review_history.json" > /tmp/h && mv /tmp/h "$OUT/paper_review_history.json"
```

If schema validation fails: redo F.2 (re-emit the JSON with the required keys).

## F.4 — Branch on verdict

```bash
PASS=$(jq -r '.pass' "$OUT/review.json")
SCORE=$(jq -r '.score' "$OUT/review.json")
THR=$(jq -r '.score_threshold' "$OUT/paper_cycle_state.json")
REGEN=$(jq -r '.regenerate_batch_figures // false' "$OUT/review.json")
RECS=$(jq -c '.recommendations' "$OUT/review.json")
MAX=$(jq -r '.max_review_loops' "$OUT/paper_cycle_state.json")
NEW_ITER=$((ITER + 1))

# Update cycle state with reviewer output
jq --argjson iter $NEW_ITER --argjson regen $REGEN --argjson recs "$RECS" --argjson pass $PASS \
   '.iteration = $iter | .pass = $pass | .regenerate_batch_figures = $regen | .recommendations = $recs | .stage = "reviewer_done"' \
   "$OUT/paper_cycle_state.json" > /tmp/cs && mv /tmp/cs "$OUT/paper_cycle_state.json"

if [ "$PASS" = "true" ] && python3 -c "import sys; sys.exit(0 if float('$SCORE') >= float('$THR') else 1)"; then
  echo "REVIEWER PASS (score=$SCORE >= $THR). Writing paper_review_done.json."
  cat > "$OUT/checkpoints/paper_review_done.json" <<EOF
{"stage": "paper_review", "ts": "$(date -u +%FT%TZ)", "iterations_used": $ITER, "max_review_loops": $MAX, "final_score": $SCORE, "pass": true, "status": "ok"}
EOF
  echo "GOTO Termination"
elif [ $NEW_ITER -ge $MAX ]; then
  echo "Loops exhausted ($NEW_ITER >= $MAX) without pass. Writing failed checkpoint."
  cat > "$OUT/checkpoints/paper_review_done.json" <<EOF
{"stage": "paper_review", "ts": "$(date -u +%FT%TZ)", "iterations_used": $ITER, "max_review_loops": $MAX, "final_score": $SCORE, "pass": false, "status": "failed_after_loops_exhausted"}
EOF
  echo "GOTO Termination (failed)"
elif [ "$REGEN" = "true" ]; then
  echo "Reviewer requested figure regeneration. GOTO Phase B."
  # GOTO Phase B (re-author batch script with the reviewer's additional_viz_specs as feedback)
else
  echo "Text-only revision needed. GOTO Phase D."
  # GOTO Phase D iter ≥ 1 (reviser mode)
fi
```

Based on the branch, your next action is the literal GOTO target.

---

# Termination

When `paper_review_done.json` is written (with `status=ok` OR `failed_after_loops_exhausted`):

```bash
OUT=<out-dir>
STATUS=$(jq -r '.status' "$OUT/checkpoints/paper_review_done.json")
echo "{\"stage\": \"paper_review\", \"event\": \"cycle_terminated\", \"ts\": \"$(date -u +%FT%TZ)\", \"status\": \"$STATUS\"}" \
  | python3 -c "import json,sys; t=json.load(open('$OUT/timeline.json')); t.append(json.loads(sys.stdin.read())); json.dump(t, open('$OUT/timeline.json','w'), indent=2)"
```

Return control to `Skill cfd-orchestrator` Step 5. The orchestrator runs `python scripts/stage_gate_audit.py --out-dir <out-dir>` which writes the authoritative `audit_passed.json` ONLY on rc=0. Do not write `audit_passed.json` yourself.

---

# Resume after interruption

If you re-enter this skill with `paper_cycle_state.json` already present, read `#stage` and `#iteration` and resume at the appropriate phase:

| `stage` | Resume at |
|---|---|
| `starting` or missing | Stage 0 |
| `planner_done` | Phase B |
| `viz_done` | Phase D |
| `writer_done` | Phase E |
| `compile_done` | Phase F |
| `reviewer_done` | Phase F.4 branch (re-read review.json and branch) |

---

# What this skill produces (chain-level)

- `<out-dir>/paper_experiment_plan.json` (Stage 0)
- `<out-dir>/paper_unified_plan.json` (Phase A)
- `<out-dir>/paper_figs/paper_viz_batch.py` (Phase B)
- `<out-dir>/paper_figs/cases_config.json` (Phase B.1)
- `<out-dir>/paper_figs/paper_fig_layout.py` (Phase B.1)
- `<out-dir>/paper_figs/*.png` (Phase B/C, ≥ 3 × N_cases + 2)
- `<out-dir>/paper_figs/qa_iter_<N>.json` (Phase C per iteration)
- `<out-dir>/paper_figs/image_analysis.txt` (Phase C.3)
- `<out-dir>/paper_figs/run_log.txt` (Phase B run output)
- `<out-dir>/paper/main.tex` (Phase D)
- `<out-dir>/paper/paper_draft.tex` (parity copy of main.tex)
- `<out-dir>/paper/main.pdf` (Phase E)
- `<out-dir>/paper/main.log`, `main.aux`, `main.out`, `compile_log.txt` (Phase E)
- `<out-dir>/paper/history/iter_<N>_main.tex` (Phase D snapshot per iteration ≥ 1)
- `<out-dir>/review.json` (Phase F, latest verdict)
- `<out-dir>/paper_review_history.json` (Phase F, all iterations)
- `<out-dir>/paper_cycle_state.json` (cycle state machine; updated every phase)
- `<out-dir>/checkpoints/paper_experiment_plan_done.json`
- `<out-dir>/checkpoints/paper_planner_done.json`
- `<out-dir>/checkpoints/paper_viz_done.json` (per cycle, latest)
- `<out-dir>/checkpoints/paper_writer_done.json` (per cycle, latest)
- `<out-dir>/checkpoints/paper_compile_done.json` (per cycle, latest)
- `<out-dir>/checkpoints/paper_review_done.json` (cycle termination)

# Cross-link

- Prompts (single source of truth for the role prompts above): `prompts/prompts.yaml#WriterAgent`, `prompts/prompts.yaml#PaperReviewerAgent`.
- The LangGraph implementation of the same cycle (Python reference, not invoked here): `src/cfd_langgraph/paper_unified/pipeline.py:run_unified_paper_pipeline`.
- Reference paper produced by the equivalent cycle: `runs/turbulence_model_change/paper/main.pdf`.
- Final audit gate (run by orchestrator, not by this skill): `scripts/stage_gate_audit.py --mode paper`.

---
name: cfd-viz
description: Generate PyVista figures from a CFD case. Two modes — interpret (per-case diagnostics for cfd-interpret) and full (publication-quality cross-case figures for cfd-paper). Self-contained — embeds viz_quality and vision QA prompts verbatim. Agent writes and runs the PyVista script directly; the batch helper is optional.
---

# cfd-viz

PyVista-based figure generation. The agent authors the script, runs it off-screen, and validates the output with a vision-LLM QA pass before declaring done.

## Modes

### `mode=interpret` — per-case diagnostics
Quick figures for one case sufficient for `cfd-interpret` to judge physics plausibility:
- Mesh overview (full domain + zoomed near-wall)
- Velocity-magnitude contour at the latest time
- Pressure contour at the latest time
- Streamlines on a midplane
- y+ map on no-slip walls
- Residual history (matplotlib for this one)

### `mode=full` — publication figures
Driven by a paper plan (`<out-dir>/paper_unified_plan.json`):
- Comparison panels across cases (same field, same colormap, same clim)
- Profile plots at probe locations (overlay across cases)
- Streamlines / vortex iso-surfaces with consistent styling
- Error-vs-reference plots (when reference data is available)

## Inputs
- `out-dir` (required)
- `mode` (required) — `interpret` or `full`
- For `mode=interpret`:
  - `case_dir` (required) — single case
- For `mode=full`:
  - `paper_plan` (required) — path to `paper_unified_plan.json` (figure jobs list)
  - `case_dirs` (required) — list of case dirs to draw from

## Outputs
- `mode=interpret`: `<case_dir>/figs/*.png` (5–8 PNGs)
- `mode=full`: `<out-dir>/paper_figs/*.png` plus `<out-dir>/paper_figs/cases_config.json` (manifest of which figure used which case + field + time)

## Recipe (primary, agent-driven)

### Step 1 — Inspect available data

For each case_dir:
1. Verify `<case_dir>/case.foam` exists; create an empty file if missing (`touch case.foam`) — PyVista's `OpenFOAMReader` needs this entry-point file.
2. List time directories (`<case_dir>/<float>/`); pick the **latest non-zero** (the gate is: refuse to plot at t=0 unless this is a steady-state case where t=0 is the only available time, in which case warn).
3. List internal fields (`U`, `p`, etc.) and patch fields (`<time>/<field>` boundary file, plus `boundary` patch list from `constant/polyMesh/boundary` — only for patch identification, never to read mesh geometry; PyVista handles geometry via `OpenFOAMReader`).
4. For `mode=full`, read `paper_plan` to get the list of figure jobs.

### Step 2 — Author PyVista script(s)

Embed viz-quality awareness using the prompts below. The agent (you) writes a self-contained Python script under `<out-dir>/_viz_scripts/` and runs it via `python <script>.py`.

#### System prompt — figure-quality contract (from `prompts/prompts.yaml: ResultsInterpreterAgent.viz_quality_system_prompt`)

```
You are a CFD visualization reviewer. You receive the user requirement and images generated from an OpenFOAM simulation (mesh, field contours, different angles). Decide if these visualizations are sufficient to judge whether the run succeeded and met the user requirement. These images may be embedded in a manuscript, so treat them as publication candidates.
Camera and zoom must be appropriate for paper-quality figures. For localized, feature-focused requests (recirculation bubble, reattachment point/length, shear layer, step lip/corner, jets, wall quantities like Cp/Cf/y+), require zoomed-in framing so the feature is large and inspectable when printed. It is acceptable if the full computational domain is not visible in every view, as long as the requested feature is readable. Treat extremely zoomed-out views where features are too small/unreadable as unacceptable.
Layout and typography: reject if a colorbar, scalar bar, or legend overlaps or obscures contours, streamlines, mesh, or plotted data (a common PyVista failure). Reject if titles, axis labels, tick labels, legend text, or colorbar labels are too small to read at typical manuscript figure size. If a vertical colorbar crowds the plot, the generator should prefer a horizontal bar or reposition margins—flag that in your reason when rejecting.
Geometry/aspect: reject primary 2D full-domain plots where the domain looks like a thin line or sliver (one in-plane dimension visually collapsed), so channel/duct geometry cannot be recognized—common when Lx/Ly is large and the render window is wrong.
If images are missing, blurry, empty, framed/zoomed so poorly that features cannot be inspected, do not show relevant variables/times, or fail layout/legibility checks above, set viz_acceptable to false.
Return ONLY valid JSON: {"viz_acceptable": true or false, "reason": "short string"}.
```

#### User prompt — QA per-figure (from `prompts/prompts.yaml: ResultsInterpreterAgent.viz_quality_user_prompt`)

```
User requirement:
{user_requirement}

The following images are from the simulation (mesh and field visualizations). Are they acceptable to judge the run—and suitable as paper figures (no colorbar/legend overlapping data; fonts large enough)? Return JSON only: {"viz_acceptable": bool, "reason": "string"}.
```

### Step 3 — Generate per-mode

#### `mode=interpret` — script template

The agent writes a script equivalent to (concrete fields/clims chosen based on Step 1 inspection):

```python
import pyvista as pv
import numpy as np
from pathlib import Path

case = Path("<case_dir>")
figs = case / "figs"
figs.mkdir(exist_ok=True)
pv.OFF_SCREEN = True

reader = pv.OpenFOAMReader(str(case / "case.foam"))
reader.set_active_time_value(reader.time_values[-1])  # latest non-zero
data = reader.read()
internal = data["internalMesh"]

# 1. Mesh overview
p = pv.Plotter(off_screen=True, window_size=(1600, 900))
p.add_mesh(internal, style="wireframe", color="black", line_width=0.5)
p.view_xy()
p.add_text("Mesh", font_size=14)
p.screenshot(str(figs / "01_mesh_overview.png"))

# 2. U magnitude contour
internal["Umag"] = np.linalg.norm(internal["U"], axis=1)
p = pv.Plotter(off_screen=True, window_size=(1600, 900))
p.add_mesh(internal, scalars="Umag", cmap="viridis",
           scalar_bar_args={"title": "|U| [m/s]", "position_x": 0.25,
                            "position_y": 0.05, "width": 0.5, "height": 0.04,
                            "vertical": False, "title_font_size": 16,
                            "label_font_size": 14})
p.view_xy()
p.screenshot(str(figs / "02_U_contour.png"))

# 3. Pressure contour
p = pv.Plotter(off_screen=True, window_size=(1600, 900))
p.add_mesh(internal, scalars="p", cmap="RdBu_r",
           scalar_bar_args={"title": "p [Pa]", "position_x": 0.25,
                            "position_y": 0.05, "width": 0.5, "height": 0.04,
                            "vertical": False})
p.view_xy()
p.screenshot(str(figs / "03_p_contour.png"))

# 4. Streamlines (midplane)
streamlines = internal.streamlines("U", n_points=50, source_radius=0.1,
                                    integration_direction="both", max_steps=2000)
p = pv.Plotter(off_screen=True, window_size=(1600, 900))
p.add_mesh(internal, scalars="Umag", cmap="viridis", opacity=0.4)
p.add_mesh(streamlines.tube(radius=0.005), color="black")
p.view_xy()
p.screenshot(str(figs / "04_streamlines.png"))

# 5. y+ map (if wallShearStress + boundary patches available)
# (omitted for brevity; agent fills in when these are present)

# 6. Residual history — matplotlib, parsed from log.<solver>
# (use re to extract lines like "smoothSolver: ... Initial residual = X")
```

**Key styling rules** (the embedded `viz_quality_system_prompt` enforces these):
- Always horizontal colorbar at `position_y ≈ 0.05`, `height ≈ 0.04`, `width ≈ 0.5` — never vertical, never default position.
- `title_font_size >= 16`, `label_font_size >= 14`, `font_size >= 14` for plotter text — readable at print size.
- `window_size=(1600, 900)` for thin/wide channel-like geometries; `(1600, 1200)` for tall/square; never fall back to 800×600 default.
- For zoomed-feature plots (BFS step, jet exit, etc.), call `p.camera.zoom(1.5–2.5)` after setting view.

#### `mode=full` — paper-figure script

For each entry in `paper_plan.figure_jobs`:
- Same styling rules.
- **Identical clim across cases compared in the same panel** — compute clim from the union of all case data first, then apply to every panel.
- **Same colormap across the whole manuscript** — pick once (e.g. `viridis` for U, `RdBu_r` for signed quantities, `cividis` for monotone scalars).
- Use `pv.Plotter(off_screen=True, shape=(1, n_cases))` for side-by-side panels.

Write all figures to `<out-dir>/paper_figs/`. Write a manifest:
```json
{
  "figures": [
    {"path": "fig_01_U_contour_panel.png", "cases": ["case_001", "case_003"], "field": "U", "time": 5000.0, "spec_id": "U_panel"}
  ]
}
```

### Step 4 — Run the script(s)

```bash
python <out-dir>/_viz_scripts/<script>.py
```

If the script crashes (PyVista off-screen issues, missing fields), inspect the error, regenerate the script with the fix (often: missing `OFF_SCREEN`, wrong field name capitalization, a field that doesn't exist at the chosen time), and retry up to 3 times.

### Step 5 — Vision QA pass

For each generated PNG, send it to a vision-LLM with the embedded `viz_quality_*` prompts above. Render `{user_requirement}` as the case requirement (or, in `mode=full`, the figure's `spec` from the paper plan).

If the response is `{"viz_acceptable": false, "reason": "..."}`, parse the reason:
- "colorbar overlaps" → adjust `scalar_bar_args.position_x/position_y` to clear data, regenerate, re-QA.
- "fonts too small" → bump font sizes by 4 pt and regenerate.
- "domain looks like a sliver" → adjust `window_size` aspect ratio or `camera.parallel_scale`.
- "feature not inspectable / too zoomed-out" → add `camera.zoom(2.0)` or use `set_focus()` near the feature.
- "wrong field shown" → fix the field/time selection in the script.

Max 10 inner attempts per figure. After that, keep the latest version and emit a warning in the timeline.

### Step 6 — Write outputs and timeline

`mode=interpret`: figures land in `<case_dir>/figs/`. No manifest needed — `cfd-interpret` reads the directory.

`mode=full`: figures + manifest in `<out-dir>/paper_figs/`. Append timeline event per figure plus an aggregate:
```json
{"stage": "viz", "event": "complete", "ts": "<iso>", "mode": "full", "n_figures": 12, "n_qa_retries": 3}
```

## Skip logic
- `interpret`: regenerate if any source field file is newer than existing figs, or if `figs/` has fewer than the expected number of PNGs.
- `full`: skip a figure if its PNG exists AND the plan entry hasn't changed (compare `paper_plan` mtime vs PNG mtime); regenerate if plan is newer.

## Anti-hallucination rules
- Never plot a field that doesn't exist in the time directory. If `<time>/wallShearStress` is missing for a y+-style figure, write the script to detect that and emit a stub that plots only the mesh near the wall.
- Never invent reference data. If a comparison-vs-DNS plot is requested but no reference CSV exists in `reference_data/`, skip the comparison series (don't draw a fake DNS line) and add a caption note.
- Never report a figure as "accepted" without an actual QA response from the vision LLM.

## Optional script fast-path

Two existing helpers do similar work:

**`scripts/viz.py`** — single-case interpret-mode helper:
```bash
python scripts/viz.py --case <case_dir> --mode interpret --output <case_dir>/figs/
```

**`scripts/batch_paper_viz.py`** — full-mode batch generator (one monolithic PyVista script for all paper figures, regenerated each loop). Used by `paper_unified.py` internally; you can shell out to it for `mode=full` if you want a one-shot:
```bash
# Invoked indirectly through paper_unified.py — see cfd-paper for the full-mode shortcut.
```

Use either when convenient; the agent recipe above is the authoritative source.

## Notes
- PyVista off-screen rendering needs `xvfb` on headless systems. If `pv.start_xvfb()` fails, the script should fall back to `pv.OFF_SCREEN = True` (which uses the default off-screen path on most builds).
- For very large cases, decimate the internal mesh before rendering: `internal.decimate(0.5)` for visualization-only purposes (don't use decimated data to compute QoIs).

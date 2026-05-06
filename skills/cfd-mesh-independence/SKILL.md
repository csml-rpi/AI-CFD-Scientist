---
name: cfd-mesh-independence
description: Mesh independence study using rapid mesh-sensitivity assessment
  with Richardson extrapolation and GCI. Follows the mesh sensitivity
  protocol exactly.
allowed-tools: Bash, Read, Write
---

# Mesh independence study

## Protocol (from mesh sensitivity assessment specification)

This skill implements the following mesh sensitivity protocol exactly:

OBJECTIVE: Determine whether the current baseline mesh is sufficiently
stable for the current analysis objective, and select the mesh to be used
for all downstream simulations.

STEP A — Identify reference scales
Before modifying any mesh:
1. Identify the most appropriate characteristic length scale L_ref for
   the geometry. Choose the dominant physical length: chord, diameter,
   body height, hydraulic diameter, gap/clearance scale, or other
   clearly justified reference dimension.
2. Identify primary quantities of interest (QoI) for this case. These
   may include: force/moment coefficients, pressure drop, mass-flow
   rate, separation/reattachment location, wake properties,
   heat-transfer quantities, or other geometry-specific metrics.

STEP B — Create comparison mesh (10% near-wall, 5% away-from-wall)
Start with one modified mesh relative to baseline:
- Near-wall region: wall-adjacent boundary-layer mesh attached to all
  no-slip solid surfaces. Reduce local mesh size by ~10% relative to
  current values. Include comparable reduction in first-layer thickness
  and consistent scaling of wall-normal layered mesh. Keep same number
  of layers and approximately same growth ratio unless adjustment is
  needed for mesh quality. If using wall functions, ensure refinement
  does not push y+ outside intended range.
- Away-from-wall region: outer/core/wake flow region outside wall-
  adjacent layers. Reduce characteristic mesh size by ~5% relative to
  baseline. These are approximate guidance, not strict rules.
- Preserve: current mesh topology, domain extent, blocking strategy,
  and mesh-generation methodology.
- If no inflation/prism layers exist: define near-wall as all cells
  within d_w <= 0.05 * L_ref from any no-slip wall. Apply consistently.

STEP C — Run both meshes
Use IDENTICAL: physical models, turbulence models, material properties,
BCs, reference values, solver settings, schemes, relaxation, time-step,
convergence criteria.
Steady: ensure both solutions are converged — judge by BOTH residual
reduction AND stabilization of integral quantities.
Unsteady: ensure transients decayed, compare statistically converged
results with same averaging procedure and window.

STEP D — Monitor and compare
Select representative monitoring locations: probe points, lines, planes,
or surface stations in regions important to flow physics (near-wall,
stagnation, separated shear layers, wakes, recirculation, jet cores,
mixing regions, far field).

Compare between baseline and modified meshes:
- Converged velocity and pressure at monitoring locations
- Wall skin-friction coefficient C_f where applicable
- Pressure coefficient C_p where applicable
- Heat-flux or Nusselt-number distributions where relevant
- All key global/integral quantities: lift, drag, moments, pressure
  loss, flow split, mass-flow rate, or other case-specific outputs

Report for both meshes:
- Total cell count
- Near-wall y+ statistics where relevant
- Any important mesh-quality indicators
- Confirmation that refined mesh remains valid and acceptable quality

STEP E — Quantify and assess
- Percentage difference between baseline and modified meshes for ALL
  monitored quantities
- Identify any locations/regions/global metrics where change exceeds 5%
  relative to converged baseline (or another tolerance if clearly
  justified)
- Assess whether surface distributions show meaningful differences
- State conclusion: whether current mesh appears sufficiently
  mesh-insensitive, or whether a more systematic multi-level study
  is required
- If further refinement needed: identify which regions/quantities are
  driving that conclusion
- Continue refinement progression until an acceptable mesh-insensitive level
  is reached for primary QoIs, then mark that as `selected_mesh_spec`.

## Implementation steps

## Orchestrator — base case strategy (literature-first)
Mesh independence compares two (or more) **meshes** for the **same**
physics setup. The baseline geometry and solver stack must come from a
sound case:
1. Search literature (`scripts/lit.py`) for benchmark backward-step /
   channel / airfoil cases that publish mesh details and results. If a
   public OpenFOAM or community repository matches the QoI and geometry,
   you may use it as the baseline **provided** you can reproduce meshing
   constraints in the requirement text.
2. If no external case is clearly superior, generate the baseline with
   `scripts/foam_run.py` so FoamAgent+RAG produces a consistent tutorial-
   grounded case, then encode the baseline mesh description in the
   requirement for Step 2.
3. The refined mesh (Step 3) must be the same topology methodology with
   controlled refinement per STEP B — do not change physics or solver
   between runs.

### Step 1 — Literature search
python scripts/lit.py \
  --topic "{topic} mesh independence CFD" \
  --limit 10 \
  --output runs/{study_name}/mesh_study/lit.json

### Step 2 — Baseline requirement
Write the user requirement for the baseline case.
Run baseline through FoamAgent:
python scripts/foam_run.py \
  --requirement "{baseline_requirement}" \
  --output-dir runs/{study_name}/mesh_study/baseline/
Read baseline/run_result.json.

### Step 3 — Modified mesh requirement
Write the modified mesh requirement following STEP B above.
The requirement must specify:
- Which characteristic length L_ref was chosen and why
- The ~10% near-wall reduction applied
- The ~5% away-from-wall reduction applied
- Explicit cell counts or mesh size values

python scripts/foam_run.py \
  --requirement "{modified_mesh_requirement}" \
  --output-dir runs/{study_name}/mesh_study/refined/

### Step 4 — Generate comparison figures
python scripts/viz.py \
  --cases runs/{study_name}/mesh_study/baseline/ runs/{study_name}/mesh_study/refined/ \
  --mode full \
  --output runs/{study_name}/mesh_study/comparison_figs/

### Step 5 — Analyze following STEP D and STEP E
python scripts/analyze.py \
  --cases runs/{study_name}/mesh_study/baseline/ runs/{study_name}/mesh_study/refined/ \
  --metrics {qoi_metrics} \
  --output runs/{study_name}/mesh_study/mesh_analysis.json

### Step 6 — Additional refinement levels (if needed) + Richardson/GCI
If STEP E indicates continued sensitivity, add further refinement levels
until QoIs stabilize, then optionally quantify with 3-level Richardson/GCI:
Run coarse, medium, fine meshes through foam_run.py.
Then compute:
  r = refinement ratio = (cell_count_fine/cell_count_coarse)^(1/N_dim)
  p = log((f_medium - f_coarse)/(f_fine - f_medium)) / log(r)
  f_exact = f_fine + (f_fine - f_medium) / (r^p - 1)
  GCI_fine = 1.25 * abs((f_fine - f_medium)/f_fine) / (r^p - 1)

Accept if GCI_fine < 3% for validation cases, < 10% for exploratory.
Generate mesh convergence plot with matplotlib.

### Step 7 — Report
Summarize following the exact reporting requirements from STEP D and E:
- Cell counts for both meshes
- y+ statistics
- Percentage differences for all QoIs
- Whether 5% threshold exceeded anywhere
- Conclusion on mesh sufficiency
- `selected_mesh_spec` to use for production runs (and why it was chosen)

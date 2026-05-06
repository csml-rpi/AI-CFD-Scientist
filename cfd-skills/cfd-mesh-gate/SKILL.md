---
name: cfd-mesh-gate
description: Mandatory mesh-independence study before any experiments. Per physics group, run baseline + refined meshes (10% near-wall, 5% away-from-wall), compare QoIs against the 5% threshold, escalate to GCI if needed, lock the selected mesh. Self-contained protocol embedded in this file (merged from skills/cfd-mesh-independence). Outputs selected_mesh_spec.json and mesh_independence_context.json.
---

# cfd-mesh-gate

Mesh independence is **mandatory** for every CFD Scientist study (per `CLAUDE.md`). This skill is the canonical implementation. The full mesh-sensitivity protocol — STEPS A–E — is embedded below; the per-level case execution goes through `cfd-experiment` (which itself wraps `scripts/foam_run.py` with `--mesh-gate-role`); the analysis (QoI comparison, percent-difference, GCI) is fully agent-driven.

## When to use
After `cfd-requirements` (or after `cfd-code-modify`) and before bulk `cfd-experiment` calls. Every pathway — general, code-mod, OED — passes through this gate.

## Inputs
- `out-dir` (required) — must contain `requirements.json`
- `starter-dir` (optional) — starter case for baseline geometry/numerics
- `qoi_tolerance_percent` (optional, default 5.0) — threshold for declaring mesh-insensitivity
- `near_wall_tolerance_percent` (optional, default 10.0) — looser threshold for wall-bounded QoIs

## Outputs
- `<out-dir>/selected_mesh_spec.json` — per-physics-group selected mesh:
  ```json
  {
    "version": 2,
    "groups": {
      "<group_id>": {
        "selected_level": "baseline|refined_v1|refined_v2|...",
        "stable_name": "<dirname>",
        "baseline_case": "<path>",
        "analysis_path": "<path>/mesh_analysis.json",
        "levels": [{"level": 0, "cells": <n>, "qois": {...}}, ...],
        "requirement_suffix": "MESH POLICY: …"
      }
    },
    "case_to_group": {"case_001": "<group_id>", ...}
  }
  ```
- `<out-dir>/mesh_independence_context.json` — paper-ready context (per-group cells, QoIs, figure paths) consumed by `cfd-paper`'s Writer
- `<out-dir>/mesh_gate/<group>/mesh_analysis.json` per group

## Mesh sensitivity protocol (recipe — STEPS A–E)

The protocol below is the canonical mesh-sensitivity assessment for CFD Scientist. Follow it exactly.

### STEP A — Identify reference scales

Before modifying any mesh, for **each physics group** (defined in Step 1 below):

1. Identify the most appropriate characteristic length scale `L_ref` for the geometry. Choose the dominant physical length: chord, diameter, body height, hydraulic diameter, gap/clearance scale, or another clearly justified reference dimension. State your choice and the reason in the per-group `mesh_analysis.json`.

2. Identify primary quantities of interest (QoI). Examples: force/moment coefficients, pressure drop, mass-flow rate, separation/reattachment location, wake properties, heat-transfer quantities, or other geometry-specific metrics. List 3–6 QoIs per group.

### STEP B — Create comparison mesh (10% near-wall, 5% away-from-wall)

Start with one modified mesh relative to baseline:

- **Near-wall region:** wall-adjacent boundary-layer mesh attached to all no-slip solid surfaces. Reduce local mesh size by ~10% relative to baseline. Include comparable reduction in first-layer thickness and consistent scaling of wall-normal layered mesh. Keep same number of layers and approximately same growth ratio unless adjustment is needed for mesh quality. If using wall functions, ensure refinement does not push y+ outside intended range.

- **Away-from-wall region:** outer/core/wake flow region outside wall-adjacent layers. Reduce characteristic mesh size by ~5% relative to baseline.

- **Preserve:** current mesh topology, domain extent, blocking strategy, and mesh-generation methodology.

- **If no inflation/prism layers exist:** define near-wall as all cells within `d_w <= 0.05 * L_ref` from any no-slip wall. Apply consistently.

These percentages are guidance, not strict rules — what matters is that the refined mesh is meaningfully finer than baseline in both regions while preserving topology.

### STEP C — Run both meshes

Use IDENTICAL: physical models, turbulence models, material properties, BCs, reference values, solver settings, schemes, relaxation, time-step, convergence criteria. Only the mesh changes.

- **Steady:** ensure both solutions are converged — judge by BOTH residual reduction AND stabilization of integral quantities.
- **Unsteady:** ensure transients have decayed; compare statistically converged results with the same averaging procedure and window.

### STEP D — Monitor and compare

Select representative monitoring locations: probe points, lines, planes, or surface stations in regions important to flow physics (near-wall, stagnation, separated shear layers, wakes, recirculation, jet cores, mixing regions, far field).

Compare between baseline and modified meshes:
- Converged velocity and pressure at monitoring locations
- Wall skin-friction coefficient `Cf` where applicable
- Pressure coefficient `Cp` where applicable
- Heat-flux or Nusselt-number distributions where relevant
- All key global/integral quantities: lift, drag, moments, pressure loss, flow split, mass-flow rate, or other case-specific outputs

For both meshes, report:
- Total cell count
- Near-wall y+ statistics where relevant
- Mesh-quality indicators (skewness, non-orthogonality, aspect ratio)
- Confirmation that the refined mesh remains valid and acceptable quality

### STEP E — Quantify and assess

For each monitored quantity:
- Compute percent difference between baseline and modified meshes.
- Identify any locations/regions/global metrics where change exceeds **5%** relative to converged baseline (or another tolerance if clearly justified — `near_wall_tolerance_percent` allows looser threshold for wall-bounded QoIs).
- Assess whether surface distributions show meaningful differences.
- State conclusion: whether the current mesh appears sufficiently mesh-insensitive, or whether a more systematic multi-level study is required.
- If further refinement is needed: identify which regions/quantities are driving that conclusion.
- Continue refinement progression until an acceptable mesh-insensitive level is reached for primary QoIs, then mark that as `selected_mesh_spec`.

## Implementation steps

### Step 1 — Group cases by physics
Cases that share solver + turbulence model + custom-code class form a **physics group**. Examples:
- `sa_builtin_baseline` — built-in Spalart-Allmaras, no custom code
- `custom_sa_mods` — a case-local compiled custom SA modification (one mesh study per code-mod variant only if the variant changes the BL physics significantly; otherwise share with the baseline group)
- `kOmegaSST_with_wallfunction` vs `kOmegaSST_lowRe` — different y+ targets ⇒ different groups

For each case in `requirements.json`, decide its group (the agent uses `solver` + `turbulence model` + `customModels lib name` as the key). Write the grouping to `<out-dir>/mesh_gate_resume.json`:
```json
{
  "groups": {
    "<group_id>": {
      "case_ids": ["case_001", "case_003"],
      "physics_signature": {"solver": "simpleFoam", "turbulence": "SpalartAllmaras", "custom_lib": null},
      "L_ref_basis": "step height = 0.038 m",
      "qois": ["Cf_along_lower_wall", "x_reattach", "Cp_along_lower_wall", "y_plus_max"]
    }
  }
}
```

### Step 2 — Build baseline + refined requirement texts
For each group:

**Baseline requirement** — derive from the first case in the group's `case_ids`. Strip any case-specific parameter sweep value (we want the baseline to be a single canonical setup); keep solver, turbulence model, BCs, geometry, and the *baseline* mesh size from the original requirement (or from the starter case if `starter-dir` was given).

**Refined requirement** — copy the baseline; modify per STEP B:
- Add an explicit "MESH-GATE REFINEMENT: this case is the refined-level mesh for the mesh-independence study; reduce near-wall cell size by ~10% (incl. first-layer thickness and layer-stack scaling) and reduce away-from-wall cell size by ~5%, preserving topology and domain extent."
- State explicit cell counts (or nx×ny×nz), the chosen `L_ref`, and confirm y+ stays in target range.

### Step 3 — Execute baseline + refined
Use `cfd-experiment` for each level:

```text
/cfd-experiment out-dir=<out-dir> requirement_text=<baseline_req> case_dir=<out-dir>/mesh_gate/<group>/baseline mesh_gate_role=baseline
/cfd-experiment out-dir=<out-dir> requirement_text=<refined_req>  case_dir=<out-dir>/mesh_gate/<group>/refined  mesh_gate_role=refined
```

Or directly via the underlying script (the gate uses the existing `--mesh-gate-role` flag in `foam_run.py`):
```bash
python scripts/foam_run.py \
  --requirement "<baseline req text>" \
  --output-dir <out-dir>/mesh_gate/<group>/baseline \
  --mesh-gate-role baseline \
  --timeline <out-dir>/timeline.json

python scripts/foam_run.py \
  --requirement "<refined req text>" \
  --output-dir <out-dir>/mesh_gate/<group>/refined \
  --mesh-gate-role refined \
  --timeline <out-dir>/timeline.json
```

### Step 4 — Extract QoIs (agent-driven)
For each level dir, agent computes:
- **Cell count** — parse `log.checkMesh` ("Cells: N"), or run `checkMesh` if missing.
- **y+ statistics** — parse `postProcessing/yPlus*/<time>/yPlus.dat` (min/mean/max).
- **Per-QoI numbers** — depends on QoI type:
  - Cf: read `<time>/wallShearStress` boundary file → magnitude / (0.5 ρ U_ref²) → write `Cf_along_<patch>.csv` (x, Cf)
  - Cp: read `<time>/p` boundary file → (p − p_ref) / (0.5 ρ U_ref²)
  - x_reattach (separation/reattachment): find first sign-change of Cf along lower wall *downstream of* the step lip, skipping the trapped-bubble first crossing. Use the strongest-derivative crossing if multiple.
  - x_separation: first sign-change going downstream into the recirculation
  - Integrated quantities (lift, drag, mass flow): from `postProcessing/forces/<time>/forces.dat` or equivalent
  - Profile metrics (peak |U|, RMS): sample along a probe line; use `pyvista.OpenFOAMReader` for spatial sampling

The agent writes a single `<group>/mesh_analysis.json` with per-level QoI vectors.

### Step 5 — Compute deltas (agent-driven, STEP E)

For each QoI:
```
delta_pct = 100 * |QoI_refined - QoI_baseline| / |QoI_baseline|
```

Classify:
- **Converged**: all QoIs change by less than `qoi_tolerance_percent` (5% default), and y+ stays in target range, and mesh quality on the refined mesh is acceptable. → SELECT BASELINE.
- **Marginally not-converged**: one QoI exceeds threshold by a small margin (≤ 2× threshold). → run a second refinement (`refined_v2`).
- **Strongly not-converged**: a QoI changes by ≥ 2× threshold or trend is monotone-divergent. → run `refined_v2`, fit Richardson extrapolation, compute GCI:

  ```
  r = (cells_fine / cells_coarse) ** (1 / N_dim)
  p = log((f_medium - f_coarse) / (f_fine - f_medium)) / log(r)
  f_exact = f_fine + (f_fine - f_medium) / (r**p - 1)
  GCI_fine = 1.25 * abs((f_fine - f_medium) / f_fine) / (r**p - 1)
  ```
  Accept the level where `GCI_fine < 5%` for validation cases, `< 10%` for exploratory. Bound the safety factor to 1.25 for 3-level studies.

If after two further refinements GCI is still > 10%, **stop and warn the user**. Do not silently continue — the study likely needs different physics/numerics, not just more mesh.

### Step 6 — Write `selected_mesh_spec.json` and `mesh_independence_context.json`

`selected_mesh_spec.json` — see schema at top.

`mesh_independence_context.json` — paper-ready structure (consumed by `cfd-paper`'s Writer; the `WriterAgent.system_prompt` mandates a mesh-independence subsection when this is non-empty):
```json
{
  "groups": {
    "<group_id>": {
      "selected_stable_name": "<level dirname>",
      "selected_level_path": "<path>",
      "L_ref_basis": "...",
      "metrics_by_mesh_level": {
        "baseline":    {"cells": 6500,  "Cf_RMS": 0.0048, "x_reattach": 4.71, "y_plus_max": 0.92},
        "refined":     {"cells": 9200,  "Cf_RMS": 0.0046, "x_reattach": 4.78, "y_plus_max": 0.86},
        "refined_v2":  {"cells": 14800, "Cf_RMS": 0.0046, "x_reattach": 4.79, "y_plus_max": 0.85}
      },
      "mesh_gate_plan": ["baseline", "refined", "refined_v2"],
      "selection_note": "Cf_RMS and x_reattach changed < 1% between refined and refined_v2; baseline-to-refined change was 4.2% — selected refined for production.",
      "cross_mesh_analysis_excerpt": "<one paragraph from agent's STEP E analysis>",
      "mesh_figure_paths": ["<abs path>/mesh_gate/<group>/figs/mesh_comparison.png"]
    }
  }
}
```

### Step 7 — Inject mesh policy into requirements

For each downstream case, the gate appends a `requirement_suffix` to the case's `user_requirement_text` so FoamAgent uses the locked mesh. Standard suffix:

```
MESH POLICY: Use the locked mesh from group <group_id> (<n> cells, <stable_name>).
Do NOT specify mesh_cells_x / mesh_cells_y in experiment parameters — the mesh is authoritative from the mesh-gate study.
If a case genuinely requires a different resolution, state it explicitly (e.g. 'refine near-wall by 10% from base mesh').
```

The injection happens at consumption time (in `cfd-experiment`'s Step 1), not by mutating `requirements.json` — that file remains the agent-generated study specification. The mapping `case_to_group` in `selected_mesh_spec.json` is what `cfd-experiment` reads.

## Skip if already done
If `selected_mesh_spec.json` exists with all groups having a non-empty `selected_level` AND each group's `analysis_path` file exists, skip and emit `mesh_gate_skipped_existing`.

**Important:** if `selected_level` is empty for any group, the gate is **incomplete** — do not skip; finish it.

## Optional shortcut — let the orchestrator run only this stage
If you're already in the LangGraph repo and `requirements.json` exists, the orchestrator's resume-from feature is the fastest path:
```bash
python scripts/orchestrator_run.py \
  --topic "<study topic>" --out-dir <out-dir> \
  --resume-from mesh_gate
```
Same artifact contracts (`selected_mesh_spec.json`, `mesh_independence_context.json`).

## Failure mode
If a baseline or refined run fails (compilation error, divergence), do NOT pick a mesh based on a single level. Apply the conservative CFL retry policy from `cfd-experiment` and try again before escalating. If the failure is consistent (e.g. baseline OK, refined diverges only at the higher resolution), warn the user; the issue is usually numerics, not mesh.

## Timeline
```json
{"stage": "mesh_gate", "event": "complete", "ts": "<iso>", "groups": [...], "selected_levels": {...}}
```

## Notes
- The mesh gate is per-physics-group, **not** per-parameter-sweep value. If you have 6 BFS cases at 6 expansion ratios with the same turbulence model, that's ONE group ⇒ one mesh study, applied to all 6.
- Do **not** re-do mesh independence per OED candidate. The gate runs once on baseline physics; OED candidates inherit the locked mesh.
- For deep-dive mesh studies (Richardson, GCI conventions, more refinement levels), the long-form protocol reference is `skills/cfd-mesh-independence/SKILL.md` — it covers the same STEPS A–E with extra commentary.

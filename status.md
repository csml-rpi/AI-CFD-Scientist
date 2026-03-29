# CFD Scientist — Pipeline Status

**Date:** 2026-03-28
**Topic:** "Study the effect of fuel velocity and inlet box sizes in 2D small pool fire."
**Branch:** `tingwen`
**Model:** Bedrock Opus 4.6 (ARN inference profile)

---

## Pipeline Stages Completed

| Stage | Status | Notes |
|-------|--------|-------|
| Ideation | Completed | 10 experiments designed (3x3 parameter sweep + 1 mesh refinement study) |
| Hypothesis | Completed | User requirements generated for all 10 experiments |
| Foam-Agent (case generation) | Completed | OpenFOAM cases generated for all 10 experiments |
| Foam-Agent (solver execution) | Partial | 4 of 10 experiments produced solver output |
| Interpreter | Completed | All 10 experiments evaluated; all flagged for rerun |
| Rerun loop | Ran 3 rounds | Some experiments improved but many still failing |
| Analysis & Paper | Skipped | Artifact gate blocked — `--execute` was not passed in the original run |

## Experiment Results

| Experiment | Case Name | Inlet Width | Fuel Velocity | Solver Ran | Time Reached | Status |
|------------|-----------|-------------|---------------|------------|--------------|--------|
| exp_001 | W5cm_V001 | 0.05 m | 0.01 m/s | No | 0 s | Failed at startup |
| exp_002 | W5cm_V005 | 0.05 m | 0.05 m/s | Yes | 9.95 s | Solver completed (End marker found) |
| exp_003 | W5cm_V010 | 0.05 m | 0.10 m/s | No | 0 s | Failed — missing `fuel` keyword |
| exp_004 | W10cm_V001 | 0.10 m | 0.01 m/s | Yes | 0.4 s | Solver ran but didn't complete |
| exp_005 | W10cm_V005 | 0.10 m | 0.05 m/s | No | 0 s | Failed at startup |
| exp_006 | W10cm_V010 | 0.10 m | 0.10 m/s | Yes | 0.1 s | Solver ran but didn't complete |
| exp_007 | W20cm_V001 | 0.20 m | 0.01 m/s | Yes | 0.85 s | Solver ran but didn't complete |
| exp_008 | W20cm_V005 | 0.20 m | 0.05 m/s | No | 0 s | Never executed (earlier run attempt) |
| exp_009 | W20cm_V010 | 0.20 m | 0.10 m/s | No | 0 s | Never executed (earlier run attempt) |
| exp_010 | W10cm_V005_meshRefine | 0.10 m | 0.05 m/s | No | 0 s | Failed — fine mesh (0.5mm) issues |

**Best result:** exp_002 (W5cm_V005) — solver ran to completion (t=9.95s, `End` marker found).

## Root Causes of Failures

1. **`fireFoam` deprecated in OpenFOAM 10** — replaced by `buoyantReactingFoam`. The Foam-Agent review loop caught this and switched solvers, but not all experiments were fixed in time.

2. **OpenFOAM 10 API changes** — Multiple config incompatibilities:
   - `reactingMixture` → `multiComponentMixture`
   - `method standard` → `method chemistryModel` in chemistryProperties
   - Reactions dict syntax changed from list `()` to dictionary `{}`
   - Boundary condition `calculated` incompatible for pressure field

3. **Foam-Agent review loop limits** — Max 25 iterations per experiment. Some experiments exhausted retries before resolving all config issues simultaneously.

4. **Bedrock timeouts** — Vision LLM calls for PyVista visualization hit read timeouts during interpreter stage.

## What Works

- **Ideation:** Successfully generates literature-aware experiment designs with parameter sweeps.
- **Hypothesis:** Produces valid Foam-Agent prompts (user requirements) from experiment ideas.
- **Foam-Agent case generation:** Creates complete OpenFOAM case directories (mesh, BCs, solver settings, chemistry, radiation).
- **Foam-Agent review loop:** Identifies and fixes OpenFOAM errors iteratively. Successfully resolved complex multi-file compatibility issues (proven by exp_002 completing).
- **Interpreter:** Correctly diagnoses simulation failures and identifies root causes.
- **Rerun loop:** Revises requirements based on interpreter feedback.

## What Needs Improvement

- **OpenFOAM version awareness:** Foam-Agent's knowledge base should include OpenFOAM 10 Foundation API changes to avoid the `fireFoam` → `buoyantReactingFoam` migration issues upfront.
- **Multi-file fix atomicity:** The review loop sometimes fixes one file at a time, causing cascading failures. All config changes need to be applied simultaneously.
- **Solver wall-time:** Most successful runs didn't complete the full simulation time (only exp_002 reached `End`). The 1-hour Foam-Agent timeout may be insufficient for fine-mesh reacting flow cases.
- **Analysis & Paper stages:** Not tested — the `--execute` flag was not passed in the original `run-topic` call, so the artifact gate blocked these stages. Use `--execute` or `--allow-non-executed-artifacts` to reach them.

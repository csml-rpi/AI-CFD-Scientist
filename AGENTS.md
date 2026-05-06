# CFD Scientist — AGENTS.md
# Orchestrator-agnostic pipeline
# Works with: Claude Code, OpenCode, Codex CLI, any ACP-compatible agent

## Environment
**Recommended:** `conda activate cfd-scientist` (matches Foam-Agent + LangChain stack).

Alternative: `source .venv/bin/activate` after `./setup_env.sh`.

## Stage 0 — Route by topic intent (mandatory)
Given user topic + optional controls (`max_papers`, `n_experiments`, `mesh_source`), choose one path:

1) **Standard CFD study** (no OpenFOAM source-code change requested)
   - Continue with Stage 1 -> Stage 7.

2) **OpenFOAM model/source-code change requested** (viscosity/turbulence/source term/fvOption/custom model)
   - Execute code-mod pipeline first (see `skills/cfd-code-mod/SKILL.md` and `openfoam_literature_change_agent_prompt_v2.txt`).
   - Then continue from Stage 2 (hypothesis/requirements/runs/interpret/analysis/paper) using the modified model setup.
   - Hard rule: no edits to OpenFOAM installation/source directories; all code and compilation must be case-local under `{case}/customModels/`.

3) **Mesh independence / mesh sensitivity requested**
   - Execute mesh pipeline (see `skills/cfd-mesh-independence/SKILL.md`) using a baseline case from FoamAgent, literature, or GitHub.
   - Keep physics/numerics identical between baseline and refined meshes.

If topic is ambiguous, ask one clarifying question, then proceed.
Always use `skills/cfd-orchestrator/SKILL.md` as the top-level router in skill-based operation.
Mandatory rule for all routes: perform mesh-independence gate first (baseline -> refined progression) and use the selected mesh for all subsequent simulations.

## Stage 1 — Literature (S2 first)
Input:  topic string
Script: python scripts/lit.py --topic "{topic}" --limit {max_papers} --output {run_dir}/lit.json
Output: lit.json — papers with title, abstract, authors, year, doi
Notes:
- Use Semantic Scholar (`S2_API_KEY`) first; respect user-provided maximum number of papers.
- Default `max_papers` to 20 if user does not specify.
- Timeline: pass `--timeline {run_dir}/timeline.json` to log paper titles/count.

## Stage 2 — Hypothesis
Input:  lit.json, topic
Script: python scripts/hypothesis.py --literature {run_dir}/lit.json --topic "{topic}" --output {run_dir}/hypotheses.json
Output: hypotheses.json — list of testable CFD hypotheses
Note:   Uses HypothesisAgent with prompts/prompts.yaml internally

## Stage 3 — Requirements (N experiments)
Input:  hypotheses.json
Script: python scripts/requirements.py --hypotheses {run_dir}/hypotheses.json --output {run_dir}/requirements.json
Output: requirements.json — per-experiment user requirement strings for FoamAgent
Notes:
- Number of generated requirements must match user-requested experiment count `N` (bounded by workflow limits).
- Each requirement must be FoamAgent-executable and free of visualization-only instructions.
- Timeline: include requirement text per experiment in timeline.

## Stage 4 — Case setup + Run (Foam-Agent runtime skill)
Input:  requirements.json, optional custom_mesh_path
Skill: `skills/cfd-foamagent-runtime/SKILL.md` (internally calls `scripts/foam_run.py`)
Output: {case_dir}/ — complete OpenFOAM case with results
        {case_dir}/run_result.json — {status, case_dir, error_logs, loop_count}
Note:   Internally runs the COMPLETE FoamAgent flow in this exact order:
        1. generate_simulation_plan() — RAG + FAISS retrieval + subtask decomposition
        2. meshing — copy_custom_mesh/prepare_standard_mesh/handle_gmsh_mesh
        3. initial_write() — INITIAL_WRITE_SYSTEM_PROMPT + tutorial reference + sequential file gen
        4. build_allrun() — Allrun script generation
        5. run_allrun_and_collect_errors() — execute simulation
        6. reviewer loop: review_error_logs() + generate_rewrite_plan() + rewrite_files()
        7. Repeat 5-6 until no errors or max_loop
        This script preserves ALL FoamAgent prompts and service logic exactly.
        Run one case at a time. Use --parallel N via parallel shell calls for multiple cases.
        In skill-based orchestration, Foam-Agent stages are executed as orchestrated skills (not a separate external orchestration API).
        Preserve existing prompts for interpreter/analysis/viz/writer from `prompts/prompts.yaml`.
        Visualization is handled by viz creator/interpreter stages, not by Foam-Agent runtime skill.
        CFD runs may take multiple hours; wait with long time windows and do not fail early.
        Timeline: include run start, reviewer-loop attempts, slow-progress timestep updates, and run completion.

## Stage 5 — Interpret results (per case)
Input:  case directory from Stage 4
Script A: python scripts/viz.py --case {case_dir} --mode interpret --output {case_dir}/figs/
Script B: python scripts/interpret.py --case {case_dir} --figs {case_dir}/figs/ --output {case_dir}/decision.json
Output: decision.json — {status: RERUN|REVISE|PROCEED, reason, suggested_changes}
Note:   Uses ResultsInterpreterAgent with prompts/prompts.yaml internally
Policy:
- Interpreter must use PyVista-generated figures to decide whether flow behavior is physically sensible for the requirement.
- If not acceptable, trigger rerun loop:
  1. Select most similar successful case by requirement similarity.
  2. Reuse the failing case as base input structure (do not copy generated outputs blindly).
  3. Revise user requirement using interpreter feedback + working-case diff cues.
  4. Re-run FoamAgent and re-interpret.
  5. Stop when case reaches PROCEED or max reruns is hit.
- If run repeatedly times out or progresses too slowly, apply conservative CFL-aware timestep tuning before rerun:
  - increase `deltaT` in small steps,
  - enforce `adjustTimeStep yes`,
  - keep `maxCo` bounded (e.g., <= 0.7) and `maxDeltaT` bounded,
  - stop escalation once stable/no-error behavior is reached or retry cap is hit.
- Timeline: record interpreter decision and reason for each trial.

## Stage 6 — Analysis
Input:  all successful case directories
Script A: python scripts/viz.py --cases {dirs} --mode full --output {run_dir}/figs/
Script B: python scripts/analyze.py --cases {dirs} --metrics {metrics} --output {run_dir}/analysis.json
Output: analysis.json, figs/
Note:   Uses AnalysisAgent with prompts/prompts.yaml internally
Policy:
- Use viz creator again for publication-quality cross-case comparisons.
- Produce discussion, conclusions, key correlations/trends requested by user.

## Stage 7 — Paper
Input:  analysis.json, figs/, lit.json; optional `mesh_independence_context.json` when mesh-gate ran (orchestrator writes it from `selected_mesh_spec.json` + `mesh_gate/mesh_analysis.json` and passes `--mesh-independence` to paper_utils).
Script A: python scripts/paper_utils.py --analysis {run_dir}/analysis.json --figs {run_dir}/figs/ --literature {run_dir}/lit.json --output {run_dir}/paper/ --template neurips [--mesh-independence {run_dir}/mesh_independence_context.json]
Script B: python scripts/reviewer.py --paper {run_dir}/paper/ --output {run_dir}/review.json
Note:   Uses WriterAgent and PaperReviewerAgent with prompts/prompts.yaml internally; Writer must include mesh-refinement table/subsection when `mesh_independence` is present in section context.

## Code modification pipeline (before Stage 1 if model implementation requested)
See skills/cfd-code-mod/SKILL.md

## Mesh independence pipeline
See skills/cfd-mesh-independence/SKILL.md

## Mesh sensitivity protocol (rapid two-mesh assessment)
When mesh-independence is requested, follow this protocol (implemented by the mesh skill):
- Identify and justify characteristic length scale `L_ref` and primary QoIs.
- Build one refined comparison mesh from current baseline:
  - Near-wall region: ~10% local refinement (including first layer and layer stack scaling when present).
  - Away-from-wall region: ~5% refinement.
  - Preserve topology/domain/blocking/meshing method as closely as possible.
  - If no inflation/prism layers exist, define near-wall cells using wall distance `d_w <= 0.05 * L_ref` unless justified otherwise.
- Run baseline and refined with identical models/BCs/numerics/convergence treatment.
- Compare local fields and surface/global metrics (e.g., U, p, Cf, Cp, heat-transfer metrics, lift/drag/pressure loss/mass-flow as applicable).
- Report cell counts, y+ and key mesh-quality indicators for both.
- Quantify percent differences for all monitored metrics.
- Flag if any important region/metric exceeds 5% change versus converged baseline (or justified tolerance).
- Conclude whether mesh is sufficiently insensitive or whether multi-level (e.g., Richardson/GCI) study is required.

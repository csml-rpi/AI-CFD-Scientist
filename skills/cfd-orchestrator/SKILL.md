---
name: cfd-orchestrator
description: Top-level CFD scientist orchestration that routes topic requests into standard research, code-modification, mesh-independence, OED, analysis-only, or paper-only pipelines. After the Nov 2026 ARIS/DS-style refactor, all recipe content lives under cfd-skills/ — this skill remains as the canonical router.
allowed-tools: Bash, Read, Write
---

# CFD orchestrator (skill-first router)

## Purpose
Top-level orchestrator for all user CFD topics. This skill **routes**; the actual recipes live in `cfd-skills/`.

## Environment activation (mandatory before any study)
For every new terminal/session, run:
```bash
conda activate cfd-scientist
```
If activation fails because the env does not exist, stop and ask the user to run one-time repo setup (`setup_env.sh` or manual `conda create -n cfd-scientist python=3.11 && conda activate cfd-scientist && pip install -r requirements.txt && pip install -e .`) before continuing.

## Non-negotiable constraints
- Do NOT edit OpenFOAM installation directories (`$WM_PROJECT_DIR`, `$FOAM_SRC`, solver source).
- For code modifications, work only inside case-local paths:
  - `<case_path>/customModels/<ClassName>/...`
  - dictionary activation in case files only.
- Follow the code-mod protocol defined in `cfd-skills/cfd-code-modify/SKILL.md` (full OPENFOAM 10 LITERATURE CHANGE AGENT v2 embedded there).
- Keep existing prompt behavior for interpreter / analysis / viz / writer / reviewer. Each is embedded verbatim in the relevant `cfd-skills/cfd-<stage>/SKILL.md`. The YAML at `prompts/prompts.yaml` remains the authoritative source.
- Use Foam-Agent workflow via `skills/cfd-foamagent-runtime/SKILL.md` for case execution.
- Visualization is handled by `cfd-skills/cfd-viz/SKILL.md` (interpret-mode) and `cfd-skills/cfd-paper/SKILL.md` (full-mode), not by the Foam-Agent runtime skill.

## Routing
Given user topic and optional references (papers/equations), route to exactly one:

1. **Standard CFD research path** — keywords: study, sensitivity, turbulence, backward step, channel, experiment, run.
   - Use `cfd-skills/cfd-pipeline/SKILL.md` (general path).

2. **Code-modification path** — keywords: implement, modify, viscosity model, turbulence model change, Bingham, power law, Carreau, custom model, source term, fvOption.
   - First use `cfd-skills/cfd-code-modify/SKILL.md`.
   - Then continue from Step 9 of `cfd-skills/cfd-pipeline/SKILL.md` (mesh-gate → experiments → analyze → paper).

3. **Mesh-independence path** — keywords: mesh independence, GCI, grid convergence, Richardson.
   - Use `cfd-skills/cfd-mesh-gate/SKILL.md`.
   - For deeper protocol commentary (Richardson/GCI conventions), see `skills/cfd-mesh-independence/SKILL.md`.

4. **Analysis-only path** — keywords: analyze, plot, post-process, visualize results.
   - Skip directly to `cfd-skills/cfd-analyze/SKILL.md` against the existing `runs/<study>/`. Do **not** re-run upstream stages. Required inputs: per-case dirs under `cases/` with `run_result.json` already present. If figures are also wanted, follow with `cfd-skills/cfd-viz/SKILL.md mode=full`.

5. **Paper-only path** — keywords: write paper, LaTeX, manuscript, draft the paper.
   - Skip directly to `cfd-skills/cfd-paper/SKILL.md` against the existing run dir. Required inputs: `analysis.json` + `manifest.json`. If they're missing, fall back to the analysis path first.

6. **Open-ended discovery path** — keywords: open-ended, discover a novel model, beat baseline, find the best model.
   - Use `cfd-skills/cfd-open-discovery/SKILL.md`. The discovery loop owns its own propose/compile/run/score iteration; mesh-gate runs once at the start.

If ambiguous, ask one concise clarifying question.

For all simulation runs in any route, invoke `skills/cfd-foamagent-runtime/SKILL.md` per case requirement (the FoamAgent runtime contract is preserved here — full Foam-Agent flow runs internally to `scripts/foam_run.py`).

## Mandatory mesh gate for all paths
Mesh independence is mandatory regardless of research direction:
- run a baseline case
- run at least one controlled refined mesh
- compare QoIs and local/global indicators (5% threshold; 10% near-wall)
- if changes are still meaningful, continue refinement progression
- once QoIs become sufficiently insensitive, lock that mesh specification
- use the selected mesh for all downstream simulations (standard or code-mod)

Full protocol: `cfd-skills/cfd-mesh-gate/SKILL.md` (STEPS A–E embedded).

## Timeline logging (mandatory)
Maintain a run timeline JSON for orchestrator execution progress.
- Set one timeline path for the study:
  - `export CFD_ORCH_TIMELINE_PATH="runs/<study_name>/timeline.json"`
- Pass `--timeline "$CFD_ORCH_TIMELINE_PATH"` to stage scripts when invoking the optional script fast-paths.
- Record at minimum:
  - literature: topic, paper count, top titles
  - hypotheses/experiments generated
  - requirements generated
  - per-case run start, reviewer-loop attempts, slow-progress / timestep updates, final status
  - interpreter decision per case (PROCEED/REVISE/RERUN + reason)

Each `cfd-skills/cfd-<stage>/SKILL.md` documents its own timeline-event shape.

## Model/provider selection
Default (when nothing is set): `--provider openai-codex --model gpt-5.5`. The `openai-codex` provider authenticates via OAuth from `~/.codex/auth.json` and posts to `https://chatgpt.com/backend-api/codex/responses` (no API key billing).

Override via env vars (set before run):
```bash
export CFD_SCIENTIST_LLM_PROVIDER="bedrock"             # alternatives: openai, anthropic, claude-code, openai-codex, gemini
export CFD_SCIENTIST_MODEL="us.anthropic.claude-sonnet-4-6"
```
Or pass through orchestrator CLI:
- `--provider bedrock|openai|anthropic|claude-code|openai-codex|gemini`
- `--model <model_id>`

If running skills directly without `orchestrator_run.py`, each per-stage script accepts the same `--provider` / `--model` flags. If you want OAuth (no API key billing), make sure `OPENAI_API_KEY` is **unset** in the shell before launch — `env -u OPENAI_API_KEY -u OPENAI_BASE_URL python scripts/lit.py …`.

## Input references (papers/equations)
- If user provides a paper/equation, include it in:
  - literature topic synthesis (`cfd-skills/cfd-literature`)
  - payload/formula normalization for code-mod path (`cfd-skills/cfd-code-modify`)
  - requirement constraints for simulation path (`cfd-skills/cfd-requirements`)
- Never assume missing symbols/units; request clarification when needed (the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol explicitly returns `NEEDS_INFO` rather than guess).

## Long-running CFD policy
- Treat 3-4 hour runs as normal.
- Use long wait windows (>= 6h default) before timeout decisions.
- Do not declare failure solely due to elapsed wall-clock if solver is progressing.

## Slow-progress adaptive timestep policy (CFL-aware)
If a run is timing out or progressing too slowly:
1. Increase `deltaT` conservatively (small step, e.g. 1.1–1.2×).
2. Ensure `adjustTimeStep yes` with controlled `maxCo` (e.g. ≤ 0.7).
3. Keep `maxDeltaT` bounded.
4. Re-run and verify stability/no divergence.
5. Repeat until stable or capped by orchestrator retry limits.

Do not make aggressive timestep jumps that violate CFL constraints. Detail in `cfd-skills/cfd-experiment/SKILL.md` Step 6.

## Cross-link
- Pipeline reference: `AGENTS.md` (orchestrator-agnostic stage-by-stage spec)
- Skill recipes: `cfd-skills/README.md`
- Project setup + modes: `README.md` (top-level)

---
name: cfd-foamagent-runtime
description: LangGraph-mode runtime contract for scripts/foam_run.py — preserves Foam-Agent RAG and prompt behavior. For skill-mode case generation, use cfd-skills/cfd-foamagent/SKILL.md instead (fully self-sufficient, agent-driven, no foam_run.py wrapper).
allowed-tools: Bash, Read, Write
---

# Foam-Agent runtime contract (LangGraph mode)

## Two modes — pick the right one

This skill documents the **LangGraph-mode** contract for `scripts/foam_run.py`. It is what `scripts/orchestrator_run.py` invokes per case requirement. **For skill-mode (agent-driven, no Python wrapper), use `cfd-skills/cfd-foamagent/SKILL.md` instead** — that skill embeds the FoamAgent planner/writer/reviewer prompts verbatim and the agent runtime drives the loop natively via Read/Write/Bash.

| Mode | Driver | Use when |
|---|---|---|
| Skill mode | `cfd-skills/cfd-foamagent/SKILL.md` (agent-driven, embedded prompts, RAG via `scripts/rag_query.py`) | Skill-driven workflow (Claude Code / Codex / Cursor); fresh case generation that needs RAG |
| LangGraph mode | `scripts/foam_run.py` (this contract) | Unattended long runs through `scripts/orchestrator_run.py`; existing Python pipeline |

Both modes use the **same prompts** (verbatim from `Foam-Agent/src/services/`) and produce the **same** `run_result.json`. Skill mode's only Python dependency is FAISS retrieval; LangGraph mode wraps the entire loop in a Python supervisor.

## Purpose (LangGraph mode)
Execute one CFD case using Foam-Agent workflow stages, controlled by the top-level orchestrator skill instead of a separate external orchestration API.

## Hard requirements
- Preserve Foam-Agent workflow order and behavior.
- Preserve Foam-Agent prompt logic for planning/input writing/reviewing.
- Preserve RAG retrieval usage.
- Use existing tested prompts only; do not replace prompt text:
  - Foam-Agent prompts from `Foam-Agent/src/services/*` (including `INITIAL_WRITE_SYSTEM_PROMPT`, `REVIEWER_SYSTEM_PROMPT`).
  - CFD-scientist agent prompts from `prompts/prompts.yaml` via `src/cfd_langgraph/prompts/loader.py`.
- Do not edit OpenFOAM installation directories.
- Keep runs case-local under the chosen output directory.

## Workflow stages (must stay in this order)
1. **Plan + RAG**: `generate_simulation_plan()` using tutorial/case retrieval.
2. **Mesh routing**: standard mesh / custom mesh / gmsh mesh handling.
3. **Initial write**: generate OpenFOAM files with existing input-writer prompts.
4. **Allrun build**: generate `Allrun`.
5. **Run**: execute case (`run_allrun_and_collect_errors` or HPC path).
6. **Review loop**: review errors, generate rewrite plan, rewrite files.
7. **Repeat run+review** until success or loop cap.

Visualization is NOT part of this skill. Visualization is handled later by viz
creator/interpreter stages in the orchestrator pipeline.

## Intermediate artifact checklist (required)
After each stage, verify artifacts exist before advancing:

1) Plan + RAG
- case plan object returned by planner (`case_name`, `case_solver`, `subtasks`).
- tutorial/reference fields available when retrieval finds matches.

2) Mesh routing
- case directory exists.
- mesh prep command outputs/logs recorded for selected mesh path.

3) Initial write
- OpenFOAM tree created: `0/`, `constant/`, `system/`.
- generated file map/structure persisted from input writer.

4) Allrun build
- `Allrun` exists and is executable.
- run command sequence is present in `Allrun`.

5) Run
- `Allrun.out` and `Allrun.err` exist (or HPC log equivalents).
- solver logs (`log.*`) present.

6) Review loop iteration
- review analysis text exists.
- rewrite plan exists.
- rewritten files are persisted.

7) Completion
- `run_result.json` exists with status, error logs, and loop count.
- case directory is ready for downstream interpreter/viz stages.
- timeline contains `foam_run_start`, zero or more `foam_run_reviewer_loop`,
  optional `foam_run_slow_progress`, and `foam_run_complete`.

## Canonical command
For each case requirement:
```bash
python scripts/foam_run.py \
  --requirement "{requirement_text}" \
  --output-dir {case_dir} \
  --max-loop {loop_cap<=10} \
  --max-time-limit {seconds} \
  --timeline "$CFD_ORCH_TIMELINE_PATH"
```

## Long-run policy
- CFD cases can run for hours; use long wait windows.
- Treat timeout/slow-progress as recoverable when possible.
- Allow CFL-aware timestep adjustments (small deltaT increases with bounded maxCo).

## Outputs
- Case directory with OpenFOAM files/results.
- `run_result.json` with status/error logs/loop count.

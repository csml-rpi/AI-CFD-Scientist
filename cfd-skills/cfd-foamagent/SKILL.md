---
name: cfd-foamagent
description: Generate one OpenFOAM case from a natural-language requirement using a markdown-driven FoamAgent loop. Self-contained — embeds parser/planner/writer/reviewer prompts verbatim from Foam-Agent/src/services/. The agent runtime drives the loop natively via Read/Write/Bash. FAISS RAG retrieval uses scripts/rag_query.py; pdflatex, xvfb-run, inline Python, jq, and all other tools are also available — the only forbidden script across the entire system is scripts/orchestrator_run.py.
---

# cfd-foamagent

Skill-mode replacement for the `scripts/foam_run.py` wrapper. The agent walks through FoamAgent's planner→write→Allrun→run→review loop natively — no Python supervisor, no LangGraph DAG, no `nodes/`-stage machinery.

The single Python touchpoint is FAISS retrieval. RAG over the OpenFOAM tutorial corpus is the one part of FoamAgent that genuinely needs an embedding model + index lookup; everything else is the agent runtime applying the prompts below to its own Read/Write/Bash tools.

## When to use vs other modes

| Path | Use when |
|---|---|
| `cfd-experiment` **sweep mode** | You have a working baseline + want to vary parameters (typical for code-mod sweeps, sensitivity studies). **No FoamAgent needed.** |
| **This skill (cfd-foamagent)** | Fresh case generation from a natural-language requirement. The first case of a new physics group. Cases that genuinely need RAG-retrieval to ground writer/reviewer in similar tutorials. |
| LangGraph mode (`scripts/foam_run.py`) | Unattended long runs; the existing Python pipeline. Skill mode is a peer, not a replacement. |

## Inputs
- `case_dir` (required) — where the case will be created (always inside `<out-dir>/`)
- `user_requirement` (required) — natural-language description ("simulate flow over a backward-facing step at Re=5100 using simpleFoam with kOmegaSST")
- `mesh_type` (optional) — `standard_mesh` (default, blockMesh-based), `custom_mesh` (user supplies polyMesh), `gmsh` (.geo file), `copied_case` (case copied from baseline, mesh already on disk)

> **Starter directory is READ-ONLY** (per `skills/cfd-orchestrator/SKILL.md` non-negotiable constraints). When `mesh_type=custom_mesh` or `mesh_type=copied_case` and the source is a starter case, copy the case into `<case_dir>/` first via `cp -r`, then edit the copy. Never `sed -i`, `> file`, or otherwise mutate any path under the starter directory.
- `max_loop` (optional, default 10) — review-rewrite-rerun cap
- `case_stats` (optional) — `{case_domain: [...], case_category: [...], case_solver: [...]}`. If not supplied, the agent infers from the FoamAgent FAISS corpus or from `Foam-Agent/database/raw/openfoam_case_stats.json` if it exists.

## Outputs
- `<case_dir>/` — full OpenFOAM case (mesh, fields, logs, postProcessing)
- `<case_dir>/run_result.json`:
```json
{
  "status": "success | failed | timeout",
  "case_dir": "...",
  "case_name": "backward_facing_step",
  "case_solver": "simpleFoam",
  "loop_count": 1,
  "wall_time_s": 1234.5,
  "error_logs": []
}
```
- Optional sidecar: `<case_dir>/.foamagent_state.json` capturing per-loop summary.

## Loop overview

```
[1] Parse requirement       — apply parse_system_prompt → case_info {name, domain, category, solver}
[2] RAG retrieval           — scripts/rag_query.py against three indices → tutorial_reference, allrun_reference, dir_structure
[3] Decompose to subtasks   — apply decompose_system_prompt → list of (file_name, folder_name)
[4] Mesh routing            — choose standard / custom / gmsh / copied path
[5] For each subtask:       — apply INITIAL_WRITE_SYSTEM_PROMPT → write file
[6] Allrun                  — apply command_system_prompt → write Allrun, chmod +x
[7] Run                     — bash <case>/Allrun, capture log
[8] If failure, REVIEW:     — apply REVIEWER_SYSTEM_PROMPT → review_analysis
                            — apply planner_system_prompt → rewrite plan {target_files: [...]}
                            — apply EDIT_WRITE_SYSTEM_PROMPT to each → rewrite files
                            — re-run [7]; loop max max_loop times
[9] Write run_result.json
```

The agent **is** the LLM, so each "apply X_PROMPT" step means: the agent acts under the system-prompt constraints below as it produces the artifact. The prompts are the agent's mental model for that step.

---

## Stage 1 — Parse the requirement

### System prompt (verbatim from `Foam-Agent/src/services/plan.py: parse_system_prompt`)

```
Please transform the following user requirement into a standard case description using a structured format.
The key elements should include case name, case domain, case category, and case solver.
Note: case domain must be one of {case_domain_list}.
Note: case category must be one of {case_category_list}.
Note: case solver must be one of {case_solver_list}.
```

### User prompt
```
User requirement: {user_requirement}.
```

### Output schema
```json
{
  "case_name":     "<string>",
  "case_domain":   "<one of case_domain_list>",
  "case_category": "<one of case_category_list>",
  "case_solver":   "<one of case_solver_list>"
}
```

Persist as `<case_dir>/.foamagent_state.json#case_info`.

---

## Stage 2 — RAG retrieval (FAISS via scripts/rag_query.py)

Query three FAISS indices against the parsed `case_info` + `user_requirement`:

```bash
# 1. similar full case content (writer + reviewer reference)
python scripts/rag_query.py \
  --db openfoam_tutorials_details \
  --query "{case_solver} {user_requirement}" \
  --top 3 \
  --output <case_dir>/.foamagent_state.tutorials.json

# 2. directory structure of similar cases (planner reference)
python scripts/rag_query.py \
  --db openfoam_tutorials_structure \
  --query "{case_solver} {case_domain} {case_category}" \
  --top 3 \
  --output <case_dir>/.foamagent_state.dir.json

# 3. Allrun scripts of similar cases
python scripts/rag_query.py \
  --db openfoam_allrun_scripts \
  --query "{case_solver} {user_requirement}" \
  --top 3 \
  --output <case_dir>/.foamagent_state.allrun.json
```

**If FAISS indices don't exist** (`scripts/rag_query.py` returns rc=2), build them first:
```bash
python Foam-Agent/init_database.py
```

**If RAG retrieval fails** (rc=3, e.g. embedding model missing), the agent can fall back to:
- Reading a tutorial directly from `$WM_PROJECT_DIR/tutorials/<solver>/<case>/` (find one by `find $WM_PROJECT_DIR/tutorials/<solver> -maxdepth 2 -type d | head -5`).
- Quality is lower but the writer still works; flag `rag_fallback: true` in `run_result.json`.

---

## Stage 3 — Decompose into subtasks

### System prompt (verbatim from `Foam-Agent/src/services/plan.py: decompose_system_prompt`)

```
You are an experienced Planner specializing in OpenFOAM projects.
Your task is to break down the following user requirement into a series of smaller, manageable subtasks.
For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored.
Your final output must strictly follow the JSON schema below and include no additional keys or information:

{
  "subtasks": [
    { "file_name": "<string>", "folder_name": "<string>" }
    // ... more subtasks
  ]
}

Make sure that your output is valid JSON and strictly adheres to the provided schema.
Make sure you generate all the necessary files for the user's requirements.
```

### User prompt
```
User Requirement: {user_requirement}

Reference Directory Structure (similar case): {dir_structure}

{dir_counts_str}

Make sure you generate all the necessary files for the user's requirements.
Do not include any gmsh files like .geo etc. in the subtasks.
Only include blockMesh or snappyHexMesh if the user hasn't requested for gmsh mesh or user isn't using an external uploaded custom mesh.
Please generate the output as structured JSON.
```

Where:
- `{dir_structure}` — the top hit from `openfoam_tutorials_structure` (read from `.foamagent_state.dir.json`)
- `{dir_counts_str}` — small summary like `0/ has 4 files, constant/ has 6 files, system/ has 5 files`

### Output
List of `{file_name, folder_name}` pairs covering: the boundary fields under `0/`, the physical/numerical settings under `constant/` and `system/`, and any custom mesh/scheme overrides.

Persist as `<case_dir>/.foamagent_state.json#subtasks`.

---

## Stage 4 — Mesh routing

Decide based on `mesh_type`:

| `mesh_type` | Action |
|---|---|
| `standard_mesh` (default) | Include a `system/blockMeshDict` subtask. Allrun will run `blockMesh`. |
| `custom_mesh` | User supplied `<case_dir>/constant/polyMesh/`. Skip `blockMeshDict`. Allrun should run only the solver (and optionally `checkMesh`). |
| `gmsh` | User supplied `<case_dir>/<case>.geo` or `<case>.msh`. Allrun must run `gmshToFoam` first. **For .geo → .msh conversion**, see `Foam-Agent/src/services/mesh.py` (the `GMSH_PYTHON_SYSTEM_PROMPT` and the `gmshToFoam` invocation; agent reads that file directly via Read tool). |
| `copied_case` | The case was copied from a baseline; `constant/polyMesh/` already exists. Skip all mesh-generation subtasks. Reviewer must NOT propose remeshing. |

For `gmsh` mesh, also apply `BOUNDARY_EXTRACTION_SYSTEM_PROMPT` (in `mesh.py`) to extract boundary names from the Gmsh output for `0/<field>` boundary blocks.

---

## Stage 5 — Write each subtask file

For each `(file_name, folder_name)` in `subtasks`, apply the writer prompt below.

### System prompt (verbatim from `Foam-Agent/src/services/input_writer.py: INITIAL_WRITE_SYSTEM_PROMPT`)

```
You are an expert in OpenFOAM simulation and numerical modeling.
Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory.
Ensure all required values are present and match with the files content already generated.
Before finalizing the output, ensure:
- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).
- Cross-check field names between different files to avoid mismatches.
- Ensure units and dimensions are correct for all physical variables.
- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {case_solver}.
Provide only the code—no explanations, comments, or additional text.
```

### User prompt template

```
User requirement: {user_requirement}

Tutorial reference (similar case content):
{tutorial_reference}

Already-written files in this case (for cross-referencing):
{written_files_ctx}

Generate <file_name>{file_name}</file_name> in <folder_name>{folder_name}</folder_name>.
Output only the file body — no markdown fences, no explanations.
```

Where:
- `{tutorial_reference}` — concatenation of top-3 hits from `openfoam_tutorials_details` (each tagged `<similar_case_N>...</similar_case_N>`)
- `{written_files_ctx}` — list of files the agent has already written in this case, with their content (gives the writer cross-file consistency context)

### Iteration order

Write files in this order so each step has the prior context it needs:
1. `system/controlDict` (sets the application name everything else aligns to)
2. `system/fvSchemes`, `system/fvSolution`
3. `constant/transportProperties` (or `constant/physicalProperties`, `constant/cloudProperties`, etc., depending on solver)
4. `constant/momentumTransport` (for RANS) or `constant/turbulenceProperties` (legacy naming)
5. `system/blockMeshDict` (if `mesh_type=standard_mesh`)
6. `0/<field>` files (one per field declared in `controlDict`/`transportProperties`/turbulence model)
7. Any extra subtasks not covered above

After each write, run `foamDictionary <case_dir>/<folder>/<file>` if available to verify it parses; if it errors, the writer made a syntax mistake — re-write that one file with the error attached.

---

## Stage 6 — Allrun

### System prompt (verbatim from `Foam-Agent/src/services/input_writer.py: command_system_prompt`)

```
You are an expert in OpenFOAM. The user will provide a list of available commands.
Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure.
Return only the list of commands—no explanations, comments, or additional text.
```

### Mesh-type-conditional appendices (verbatim)

If `mesh_type == "copied_case"`, append:
```
CRITICAL: The case was copied from a baseline and already contains constant/polyMesh (or equivalent mesh).
Do NOT list blockMesh, snappyHexMesh, surfaceFeatureExtract, cartesianMesh, or other mesh generators
unless the user requirement explicitly demands remeshing.
Prefer: optional checkMesh (non-fatal), then the solver from controlDict.
Do not run decomposePar unless the user requires parallel execution.
```

If `mesh_type == "custom_mesh"`, append:
```
If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used).
```

### User prompt
```
Available OpenFOAM commands for the Allrun script: {commands}
Case directory structure: {dir_structure}
User case information: {case_info}
Reference Allrun scripts from similar cases: {allrun_reference}
Generate only the required OpenFOAM command list — no extra text.
```

Where `{allrun_reference}` is the concatenation of top-3 hits from `openfoam_allrun_scripts`.

### After generation

```bash
chmod +x <case_dir>/Allrun
```

The Allrun must contain the standard FoamAgent header (`#!/bin/sh` + `cd "${0%/*}"` + `. "$WM_PROJECT_DIR/bin/tools/RunFunctions"`) plus the generated command list using `runApplication` / `runParallel`.

---

## Stage 7 — Run

```bash
( cd <case_dir> && ./Allrun ) 2>&1 | tee <case_dir>/Allrun.out
echo "EXIT_CODE=$?" >> <case_dir>/Allrun.out
```

Capture solver logs (`<case_dir>/log.<solver>`). Wall-clock the run; record `wall_time_s`.

For long runs (transient cases > 1 hour), the agent should NOT block-wait synchronously for the entire duration. Instead:
- Launch with `nohup ./Allrun > Allrun.out 2>&1 &`
- Periodically `tail -50 log.<solver>` and check that the latest `Time =` line is advancing.
- Declare stall only when log mtime is older than ~10 minutes AND no new `Time =` line has been written.

---

## Stage 8 — Review loop

Triggered when:
- Allrun exit code is non-zero, OR
- A solver log exists but ends without `End` (segfault / FPE / divergence), OR
- Any `log.*` contains `FOAM FATAL ERROR` or `FOAM FATAL IO ERROR` (even if the run appeared to continue), OR
- The latest `Time =` is `0` after Allrun's expected duration (silent solver crash).

### 8a. Reviewer — analyze errors

#### System prompt (verbatim from `Foam-Agent/src/services/review.py: REVIEWER_SYSTEM_PROMPT`)

```
You are an expert in OpenFOAM simulation and numerical modeling.
Your task is to review the provided error logs and diagnose the underlying issues.
You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error.
When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log.
Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally.
Propose ideas on how to resolve the errors, but do not modify any files directly.
Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions.
The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name.
```

#### User prompt template

```
<similar_case_reference>{tutorial_reference}</similar_case_reference>

{similar_case_advice_block}

<foamfiles>{foamfiles_xml}</foamfiles>

<error_logs>{error_logs}</error_logs>

<user_requirement>{user_requirement}</user_requirement>

{history_text}
```

Where:
- `{foamfiles_xml}` — every foamfile in the case wrapped as `<foamfile><file_name>...</file_name><folder_name>...</folder_name><content>...</content></foamfile>`. Truncate any single file > 30 KB; cap total context at 300 KB.
- `{error_logs}` — concatenation of `Allrun.err`, `log.<solver>` tail, and any other `log.*` files. Last 200 lines of each.
- `{history_text}` — if this is loop iteration > 1, list of prior reviewer analyses (so the reviewer doesn't repeat itself).

Output: free-text **review_analysis** explaining the error and proposing fixes (no file edits yet).

### 8b. Rewrite planner — pick the files to edit

#### System prompt (verbatim from `Foam-Agent/src/services/review.py: planner_system_prompt`)

```
You are an OpenFOAM debugging planner.
Given current foam files, error logs and reviewer analysis, create a minimal rewrite plan.
Output MUST be strict JSON only, with this exact schema:
{"target_files": [{"file": "relative/path", "changes": "change1; change2"}]}.
Rules:
1) Do not use markdown, backticks, or comments.
2) Use double quotes for all strings.
3) In changes, use short plain text actions separated by semicolons.
4) Do not include parentheses, backticks, or quote characters inside changes text.
5) Do not include run steps; only file edits.
```

#### User prompt template

```
<foamfiles>{foamfiles_xml}</foamfiles>
<error_logs>{error_logs}</error_logs>
<review_analysis>{review_analysis}</review_analysis>
<user_requirement>{user_requirement}</user_requirement>
Return strict JSON now with key target_files only.
```

Output: `{target_files: [{file: "...", changes: "..."}]}`. Persist as `<case_dir>/.foamagent_state.json#last_rewrite_plan`.

### 8c. Apply rewrites — edit the targeted files

For each `target_files[i]`, apply the EDIT writer prompt:

#### System prompt (verbatim from `Foam-Agent/src/services/input_writer.py: EDIT_WRITE_SYSTEM_PROMPT`)

```
You are an expert in OpenFOAM simulation and numerical modeling.
An existing OpenFOAM case was copied from a baseline; you must EDIT one file in place.
File: <file_name>{file_name}</file_name> in <folder_name>{folder_name}</folder_name>.
Apply the MINIMAL changes needed to satisfy the user requirement. Preserve mesh-related entries
(e.g. polyMesh paths, fvSchemes/fvSolution structure) unless the requirement explicitly demands remeshing or solver changes that require it.
Solver context from planner: {case_solver}.
Return only the complete updated file body — no markdown fences, no commentary.
```

#### User prompt template
```
Required changes for this file: {changes}

Current file contents:
---
{current_content}
---

Other foamfiles in the case (for cross-referencing):
{written_files_ctx}

Return only the complete updated file body.
```

After all targeted files are rewritten, **return to Stage 7** (re-run Allrun). If the same file appears in `target_files` for 3 consecutive loops, the loop is stuck — bail with `status: failed, reason: "rewrite_loop_stuck"`.

### 8d. Loop cap

Maximum `max_loop` iterations of [Run → Review → Rewrite → Run]. If exhausted without success, write `run_result.json` with `status: failed, loop_count: max_loop, error_logs: [<last error tail>]`.

---

## Stage 9 — Write run_result.json

`status: "success"` requires ALL of: Allrun exit 0, the solver log (`log.<solver>`) ends with the `End` line, AND **no** `log.*` contains `FOAM FATAL ERROR` / `FOAM FATAL IO ERROR`. A FATAL — even one the run appeared to survive — forces `status: "failed"`: a FATAL is never an admissible documented success, and the stage-gate audit hard-fails any `success` case sitting over a FATAL log.

On a clean completion:
```json
{
  "status": "success",
  "case_dir": "<abs path>",
  "case_name": "<from case_info>",
  "case_solver": "<from case_info>",
  "loop_count": <iterations actually used>,
  "max_loop": <the reviewer-loop cap>,
  "wall_time_s": <total time including all reviewer iterations>,
  "error_logs": []
}
```

On failure, set `status: failed`, populate `error_logs` with the FATAL block(s) and the last error tail from each `log.*`, and include both `loop_count` and `max_loop` (so the audit can confirm the reviewer-loop budget was exhausted). On wall-clock cap, set `status: timeout`.

Append a timeline event:
```json
{"stage": "experiment", "event": "complete", "ts": "<iso>", "case_id": "<>", "status": "success|failed|timeout", "loop_count": N, "wall_time_s": X}
```

---

## Anti-hallucination rules

- **Never** edit files outside `<case_dir>` (no `$WM_PROJECT_DIR`, no Foam-Agent source, no other case dirs).
- **Never** write `polyMesh/` files by hand. If mesh is wrong, regenerate via `blockMesh` / `gmshToFoam` / re-import — don't text-edit mesh data.
- **Never** propose changes to `user_requirement` parameters during review. The reviewer prompt explicitly forbids this — find a different fix.
- **Never** skip the `End` and FATAL check after Allrun. A non-zero exit isn't enough; some solvers exit 0 after a silent crash, and some keep running past a `FOAM FATAL ERROR`. Check the actual solver log for the `End` line AND scan every `log.*` for `FOAM FATAL ERROR` before writing `status: "success"`.
- **Never** hardcode the solver application in Allrun. Read it from `system/controlDict.application` — that's the only authority.

## Edge-case prompts not embedded here

For brevity, three less-common prompt families live in their FoamAgent source files; the agent should `Read` them directly when needed:

- `Foam-Agent/src/services/mesh.py`:
  - `BOUNDARY_SYSTEM_PROMPT` (line 143) — boundary update for Gmsh-imported cases
  - `CONTROLDICT_SYSTEM_PROMPT` (line 152) — controlDict adjustment for Gmsh
  - `GMSH_PYTHON_SYSTEM_PROMPT` (line 159) — generate Python to convert .geo → .msh
  - `BOUNDARY_EXTRACTION_SYSTEM_PROMPT` (line 199)
  - `GMSH_PYTHON_ERROR_CORRECTION_SYSTEM_PROMPT` (line 216)
- `Foam-Agent/src/services/run_hpc.py` — HPC submission wrappers (Slurm, etc.)
- `Foam-Agent/src/services/visualization.py` — handled by `cfd-skills/cfd-viz` instead

The agent reads these on demand via Read; they don't need to be re-embedded here.

## Cross-link

- Sweep mode (parameter variation against a known baseline): `cfd-skills/cfd-experiment/SKILL.md` — much faster path when you have a working baseline.
- QoI extractors (post-run analysis): `cfd-skills/cfd-stats/SKILL.md`.
- Code modifications (custom OpenFOAM models): `cfd-skills/cfd-code-modify/SKILL.md` — runs *before* this skill so the compiled `.so` is ready to be activated by the case's `controlDict.libs`.
- LangGraph parity: `skills/cfd-foamagent-runtime/SKILL.md` documents the contract for `scripts/foam_run.py` (the LangGraph mode) — the two paths share the same prompts and produce the same `run_result.json`.

## Two valid execution paths

1. **Inline agent recipe (above).** The agent uses Read/Write/Bash directly to drive FoamAgent's planner → writer → reviewer loop, with embedded prompts. RAG retrieval via `scripts/rag_query.py`.

2. **Script fast-path.** `scripts/foam_run.py` runs the same loop with a Python supervisor and the same prompts. Invoke via Bash with the standard FoamAgent CLI args. The skill-mode-forbidden script is `scripts/orchestrator_run.py` only — `foam_run.py` is per-case and callable from any skill flow.

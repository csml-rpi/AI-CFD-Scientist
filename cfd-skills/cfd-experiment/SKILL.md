---
name: cfd-experiment
description: Run one CFD case end-to-end through Foam-Agent (RAG → plan → mesh → input write → Allrun → reviewer loop). The simulation execution itself goes through scripts/foam_run.py because that *is* Foam-Agent's framework — but this skill embeds RunValidityAgent prompts (Allrun preflight + runtime investigation) verbatim so the agent can drive pre-flight checks and CFL-aware retries without external code. Writes <case_dir>/run_result.json.
---

# cfd-experiment

Execute one OpenFOAM case via Foam-Agent. **The case-running engine is `scripts/foam_run.py`** — it wraps Foam-Agent's RAG/planner/writer/reviewer pipeline, and reimplementing that in markdown would mean reimplementing Foam-Agent. So this skill keeps the script call as the actual execution mechanism (in the same spirit as DeepScientist's baseline runner), but everything **around** the run — pre-flight validation, runtime triage, CFL retry decisions — is fully self-contained agent work driven by the embedded `RunValidityAgent` prompts below.

## Inputs
- `out-dir` (required) — run dir (or mesh_gate sub-dir for mesh-gate use)
- One of:
  - `case_id` — pulls requirement from `<out-dir>/requirements.json`
  - `requirement_text` — direct requirement string (used by `cfd-mesh-gate`, `cfd-open-discovery`)
- `case_dir` (optional) — override the default `<out-dir>/cases/<case_id>/`
- `mesh_policy` (auto) — if `<out-dir>/selected_mesh_spec.json` exists for this case's group, append the locked-mesh `requirement_suffix` to the requirement before running
- `max_loop` (optional, default 10) — Foam-Agent reviewer-loop cap
- `max_time_limit` (optional, default 7200 seconds) — wall-clock cap for the run
- `provider`, `model` (optional) — LLM provider/model

## Output
- `<case_dir>/` — full OpenFOAM case (mesh, fields, logs, postProcessing)
- `<case_dir>/run_result.json`:
  ```json
  {
    "status": "success|failed|timeout",
    "case_dir": "...",
    "case_name": "case_001",
    "case_solver": "simpleFoam",
    "error_logs": [],
    "loop_count": 1,
    "mesh_type": "standard_mesh|custom_mesh|gmsh",
    "wall_time_s": 1234.5
  }
  ```

## Recipe (primary, agent-driven around the foam_run.py call)

### Step 1 — Resolve the requirement
1. If `case_id` was given, read `<out-dir>/requirements.json` and find the entry; take its `user_requirement_text`.
2. Else use `requirement_text` directly.
3. If `<out-dir>/selected_mesh_spec.json` exists AND this case is mapped to a group via `case_to_group`, append the group's `requirement_suffix` (the mesh-policy paragraph from `cfd-mesh-gate`) to the requirement text — so FoamAgent uses the locked mesh. Skip this injection only when this skill is being called *by* `cfd-mesh-gate` itself (which is by definition determining the mesh).

### Step 2 — Pre-flight check
Before launching, verify there is no stale half-finished run in `<case_dir>` that would confuse the reviewer loop:
- If `<case_dir>/run_result.json` exists with `status: success`, **skip and return** (use `force=true` to override).
- If `<case_dir>` exists but has no `run_result.json`, look at directory mtime: if older than 24h, archive it to `<case_dir>.stale_<ts>` and start fresh; if recent, ask the user before clobbering.

### Step 3 — Launch via Foam-Agent
This is the only place a Python script is mandatory in the skill flow.

```bash
python scripts/foam_run.py \
  --requirement "<full requirement text>" \
  --output-dir <case_dir> \
  --max-loop <max_loop> \
  --max-time-limit <max_time_limit> \
  --timeline <out-dir>/timeline.json \
  --provider <provider> --model <model>
```

`scripts/foam_run.py` runs the **complete Foam-Agent pipeline** internally, in this order:
1. `generate_simulation_plan()` — RAG + FAISS retrieval over tutorial cases + subtask decomposition
2. Mesh routing — `copy_custom_mesh` / `prepare_standard_mesh` / `handle_gmsh_mesh`
3. `initial_write()` — generate all OpenFOAM files using FoamAgent's `INITIAL_WRITE_SYSTEM_PROMPT` and tutorial reference
4. `build_allrun()` — generate `Allrun`
5. `run_allrun_and_collect_errors()` — execute simulation
6. Reviewer loop — `review_error_logs` → `generate_rewrite_plan` → `rewrite_files`, repeat up to `max_loop`

This script preserves all Foam-Agent prompts and service logic exactly (including `INITIAL_WRITE_SYSTEM_PROMPT`, `REVIEWER_SYSTEM_PROMPT`, etc. defined under `Foam-Agent/src/services/`). Do not try to reimplement any of those — the Foam-Agent framework owns them.

For multiple cases in parallel, launch the script multiple times in the background (with `nohup` or `&`) and wait. Foam-Agent handles concurrency only if you give each case its own `--output-dir`.

### Step 4 — Post-launch: pre-flight Allrun audit

While Foam-Agent's own reviewer loop catches most issues, we add an extra agent-driven Allrun audit using the `RunValidityAgent.allrun_preflight` prompts. Run this once after Foam-Agent generates `Allrun` but before the solver finishes. (In practice: when `<case_dir>/Allrun` first appears, do this audit; if it returns BROKEN, write the corrected `Allrun` and let Foam-Agent's reviewer loop pick up the change.)

#### System prompt (from `prompts/prompts.yaml: RunValidityAgent.allrun_preflight_system_prompt`)

```
You are an OpenFOAM Allrun auditor. Given Allrun contents, decide if it
executes a flow solver to completion. Return STRICT JSON only:
  {"verdict": "OK"|"BROKEN",
   "flow_solver_runs": true|false,
   "reason": "<short>",
   "corrected_allrun": "<FULL Allrun text or empty>"}

Rules:
  - Preserve all existing pre-solver setup commands (blockMesh, snappy,
    decomposePar, mapping, post-processing helpers, etc.); only fix the
    solver-invocation lines.
  - Accept multi-solver Allruns (e.g. potentialFoam + simpleFoam). It is
    OK as long as a flow solver runs to completion.
  - Accept BOTH `runApplication simpleFoam` and `runParallel simpleFoam`.
  - If the only solver invocation is commented out, that is BROKEN — emit
    a corrected Allrun that uncomments / restores it. Provide a serial
    fallback when parallel decomposition is not staged.
  - The corrected Allrun must be the COMPLETE replacement file (including
    the `#!/bin/sh` and `cd ${0%/*}` boilerplate when present).
  - If the Allrun is fine, set verdict=OK and corrected_allrun="".
  - Output JSON only, no prose, no markdown fences.
```

#### User prompt (from `prompts/prompts.yaml: RunValidityAgent.allrun_preflight_user_prompt`)

````
Allrun contents:
```
{allrun_text}
```

controlDict.application = {application}

case_dir listing (compact):
{case_listing}
````

Render with:
- `{allrun_text}` = full file contents of `<case_dir>/Allrun`
- `{application}` = the value of `application` from `<case_dir>/system/controlDict`
- `{case_listing}` = `ls -la <case_dir>` output, trimmed to relevant lines

If the verdict is `BROKEN`, write the `corrected_allrun` content to `<case_dir>/Allrun` (with execute permission) and continue; Foam-Agent's reviewer loop will pick up the change on its next iteration.

### Step 5 — Runtime triage (when run fails or stalls)

If `foam_run.py` returns non-zero, OR if `run_result.json` reports `status: failed`, OR if the run wall-clock exceeds `max_time_limit` without producing any solver progress, invoke the runtime-investigation prompt below to classify the root cause.

#### System prompt (from `prompts/prompts.yaml: RunValidityAgent.investigate_runtime_system_prompt`)

```
You are an OpenFOAM run-validity investigator. A previous OED iteration was
flagged RUN_INVALID by the run-validity gate (the flow solver did not
advance the case to a meaningful time). Read the diagnostic bundle, the
Allrun, the OpenFOAM log tails, and the original action JSON. Classify the
root cause and decide whether the fix belongs in the harness (Allrun /
runner / preflight bug) or in the model (the proposed model modification
is wrong, e.g. divergence, blow-up, BC bug).

Output STRICT JSON only:
  {
    "root_cause_class": "code_mod_source_bug"|"allrun_bug"|"of_version"|"oom"|"divergence"|"mesh"|"bc"|"other",
    "explanation": "<short>",
    "patch_target": "harness"|"model",
    "patch": {
      "files": [{"path": "Allrun", "new_content": "<FULL file contents>", "rationale": "<why>"}],
      "rerun_strategy": "rerun_same_model"|"downgrade_to_revise"
    },
    "confidence": <float 0..1>
  }

Rules:
  - If patch_target="harness", provide the FULL replacement contents for
    each affected file (no diffs).
  - If patch_target="model", patch.files MAY be empty and rerun_strategy
    should be "downgrade_to_revise" so the planner proposes a corrected
    code_mod next iteration.
  - Do not modify files under the OpenFOAM installation. All edits are
    case-local.
```

#### User prompt (from `prompts/prompts.yaml: RunValidityAgent.investigate_runtime_user_prompt`)

````
DIAGNOSTIC BUNDLE (run_validity_diagnostic.json):
{diag_json}

ALLRUN CONTENTS:
```
{allrun_text}
```

ORIGINAL ACTION JSON:
```
{action_json}
```

LOG TAILS (last lines of any log.* file):
{log_tails}
````

Render with:
- `{diag_json}` — assemble: max_time, baseline_final_time, log presence, brief reason ("solver crashed at t=…", "deltaT collapsed to <1e-12", "only time=0 written"), file count, run_result.json content
- `{allrun_text}` — full Allrun
- `{action_json}` — for OED: the candidate action JSON; for general experiments: the requirement-text JSON
- `{log_tails}` — last 50 lines of any `log.*` file

#### Reaction policy
- `patch_target == "harness"` → write each replacement file from `patch.files`, then re-run `foam_run.py` against the same case dir (Foam-Agent's reviewer loop will iterate).
- `patch_target == "model"` and `rerun_strategy == "downgrade_to_revise"` → return `status: failed` with `reason="model bug — needs requirement revision"` and let `cfd-interpret` route to REVISE.

### Step 6 — CFL-aware retry policy
Independent of root-cause class, if the failure looks numerical (`divergence`, `oom`, `bc` with timestep clue), apply **conservative CFL tuning** before giving up:

1. Open `<case_dir>/system/controlDict`. Set:
   - `adjustTimeStep yes;`
   - `maxCo` between `0.5` and `1.0` (start at 0.7)
   - `maxDeltaT` bounded (e.g. ≤ initial deltaT × 10)
2. Slightly increase initial `deltaT` (1.1×–1.2×) — too small a deltaT can stall on stiff transients.
3. Keep `endTime` and physics unchanged.
4. Re-run `foam_run.py`. **Max 3 retries per case** at this skill level. Beyond that, return `status: failed`.

Do **not** make aggressive timestep jumps that would violate CFL. Do **not** silently downgrade physics (e.g. switching turbulence model) — that's a `cfd-interpret REVISE` decision, not a CFL retry.

### Step 7 — Write run_result.json
If `foam_run.py` already wrote it, fine. Otherwise assemble from logs:
- `status` from exit code + log tail
- `case_solver` from `controlDict.application`
- `error_logs` from `log.*` tails when failed
- `loop_count` from Foam-Agent's reviewer-loop record (in `<case_dir>/.foamagent_state.json` if present)
- `wall_time_s` from process timing

Append to timeline:
```json
{"stage": "experiment", "event": "complete", "case_id": "case_001", "status": "success", "wall_time_s": 1234.5, "loop_count": 1}
```

## Long-run policy
- CFD runs are slow but legitimate. Steady RANS may take 30–120 min; transient hours.
- **Do not declare timeout prematurely.** Default `max_time_limit` is 2 h; for paper-quality production runs use 6 h or more.
- Monitor by tailing `<case_dir>/log.<solver>` periodically; declare stall only when log mtime is older than ~10 minutes AND no new Time line has been written.

## Skip if already done
If `<case_dir>/run_result.json` exists with `status: success`, skip with `experiment_skipped_existing`. To force a re-run, delete `run_result.json` (and ideally archive the case).

## Mesh-gate cooperation
When invoked from `cfd-mesh-gate`, the skill will receive `requirement_text` directly with the per-level mesh spec, and a `mesh_gate_role` (`baseline|refined|coarse`). Pass it through as `--mesh-gate-role <role>` to `foam_run.py` so its existing logic for stable per-level dirnames kicks in. Do **not** apply step 1's mesh-policy injection in this case (the gate is *determining* the policy).

## Optional script fast-path
There is no separate one-shot for this skill — the script call in Step 3 is already the canonical execution path. The "agent-driven" content is the *wrapping*: pre-flight Allrun audit, runtime investigation, CFL retry — none of which require external Python.

## Cross-link
- The Foam-Agent execution contract is documented in `skills/cfd-foamagent-runtime/SKILL.md` (the staged workflow + intermediate-artifact checklist). Read it once if you need to understand what happens *inside* `foam_run.py`.

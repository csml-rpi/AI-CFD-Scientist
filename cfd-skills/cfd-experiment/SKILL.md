---
name: cfd-experiment
description: Run CFD cases. Two modes — sweep (ARIS-style, agent-driven, no FoamAgent handoff) for parameter variants on a known-good baseline; single-case (FoamAgent-driven via scripts/foam_run.py) for fresh case generation that needs RAG. Embeds RunValidityAgent prompts (Allrun preflight + runtime investigation) verbatim. Writes <case_dir>/run_result.json.
---

# cfd-experiment

## ⚠ HARD STOP — read this before any other step

If `<out-dir>/checkpoints/mesh_gate_done.json` does **not** exist (or for `code_mod`, if `code_mod_done.json` also does not exist), STOP. Running cases on the starter mesh before mesh-gate has selected the locked mesh is forbidden by `Skill cfd-orchestrator`'s "mesh-gate before experiments" constraint. Return to `Skill cfd-orchestrator` and start over. The one exception: this skill is invoked recursively by `cfd-mesh-gate` itself with `mesh-gate-role` set; that path is keyed on the call coming from inside `<out-dir>/mesh_gate/<group>/` and bypasses this stop.

## Step 0 — Preflight gate (HARD; slice 14)

Before launching any case (sweep or single-case), run:

```bash
python scripts/stage_gate_audit.py --out-dir <out-dir> --mode preflight --target-stage experiments
```

If `rc != 0`, STOP. The script enforces that `literature_done`, `baseline_setup_done`, `metric_setup_done`, `hypothesis_done`, `requirements_done`, `mesh_gate_done` (and `code_mod_done` on route 2) are all on disk first. Running cases on the starter mesh before `cfd-mesh-gate` has selected the locked mesh is forbidden by `skills/cfd-orchestrator/SKILL.md` "Order constraint — mesh-gate must come BEFORE experiments". This preflight check is what enforces it at stage entry instead of after the fact.

Inside `cfd-mesh-gate`, the per-level runs invoke this skill with `mesh-gate-role` and bypass the experiments-stage preflight (mesh-gate is itself a predecessor of `experiments`). The bypass is keyed on the presence of `<out-dir>/mesh_gate/<group>/` in the call path, not on a flag the agent can fake.

Execute one or many OpenFOAM cases. Two modes:

| Mode | When to use | Engine |
|---|---|---|
| **Sweep mode** (ARIS-style) | You have a working baseline case + want to vary parameters across N variants × M conditions (typical for code-mod sweeps, sensitivity studies, OED follow-up). | Agent-driven — copy baseline, patch one dictionary entry, run solver via Bash, parse log. **No FoamAgent handoff.** |
| **Single-case mode** (FoamAgent-driven) | You're generating a fresh case from a natural-language requirement and need RAG retrieval over tutorial cases (typical for the first case of a new physics group). | `scripts/foam_run.py` — wraps FoamAgent's RAG/planner/writer/reviewer pipeline. Slice 4 of the skill migration will fold the FoamAgent prompts into a `cfd-foamagent` skill so this handoff also goes away. |

The wrapping concerns — pre-flight Allrun validation, runtime triage, CFL-aware retries — are agent-driven in **both** modes via the embedded `RunValidityAgent` prompts further down.

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

> **Starter directory is READ-ONLY** (per `skills/cfd-orchestrator/SKILL.md` non-negotiable constraints). All case mutations happen inside `<out-dir>/` — sweep mode's `shutil.copytree(BASE, cd)` is the right pattern; never `sed -i` or mutate the starter source directly.

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

## Sweep mode (ARIS-style — no FoamAgent handoff)

A purely scripted parameter sweep, in the shape ARIS-style frameworks use for parameter studies (typical wall-clock: tens of minutes for 30 cases vs hours for an LLM-supervised loop). The pattern is genuinely simple. Adopt it for any study where:

- A working baseline case already exists (mesh + boundary conditions + numerics validated).
- The variation across cases is **parametric**: changing one or a few dictionary entries (model name, coefficient, ambient condition, fluid property, etc.).
- The QoI extraction is well-known (pick the appropriate pattern in `cfd-skills/cfd-stats/SKILL.md`).

Do **not** use sweep mode when you need to generate a topology/mesh/boundary-set the framework hasn't seen — that's single-case mode (next section).

### Sweep-mode pattern

For each `(variant × condition)`:
1. **Copy** the baseline case to `<case_dir>` (`cp -r baseline_case/case <case_dir>`). Because custom libraries are built case-local (see `cfd-code-modify` — `customModels/<Class>/` lives inside the case), `cp -r` carries the compiled `.so` with the copy; no global library is needed.
2. **Patch** the dictionary entries that vary (e.g. `constant/cloudProperties.phaseChangeModel`, `0/T.internalField`, `system/controlDict.endTime`). **Do not** patch fields you didn't intend to change.
3. **Activate** any compiled custom library with a **case-relative** `libs` entry — `libs ("customModels/<Class>/lib<Class>.so");` in `system/controlDict` — and point the relevant model selector at the new class name. Use the case-relative path, never a bare `lib<Class>.so` name: the bare form depends on `FOAM_USER_LIBBIN` / the global library search path and breaks reproducibility when the case is moved or shared.
4. **Run** the application named in `system/controlDict` (`reactingFoam`, `simpleFoam`, etc.) via Bash. Stream stdout to `log.<solver>`.
5. **Parse** the log for the QoI using a `cfd-stats` pattern (e.g. Pattern 1 for droplet K, Pattern 2 for Cf, Pattern 3 for drag).
6. **Append** one row to `<out-dir>/results.csv` with: variant, condition, QoI, ratio-vs-reference, sample count, t_end.

That's the entire pattern. No reviewer loop, no RAG, no LLM-graded retries.

### Sweep-mode template

The agent writes this Python directly via Bash (not as a wrapper script). The template is study-agnostic — every domain-specific bit is concentrated in three small functions (`configure`, `extract_qoi`, `qoi_ref`) that the agent fills in for the topic at hand.

```python
# sweep_template.py — drop into <out-dir> via Write tool, then `python sweep_template.py`
# Replace every <PLACEHOLDER> with the topic-specific value before running.
import re, csv, shutil, subprocess
from pathlib import Path

OUT  = Path("<out-dir>").resolve()
BASE = OUT / "baseline_case/case"        # known-good baseline
CASES_DIR = OUT / "cases"
CASES_DIR.mkdir(exist_ok=True)

# Define what varies. Each entry is one case to run.
# Keys are study-specific — the example below shows a model × condition sweep,
# but you can sweep any combination of dictionary entries.
SWEEP = [
    {"id": "<variant_a>_<cond_1>", "variant": "<variant_a>", "condition": <cond_1>, "coeffs": {...}},
    {"id": "<variant_a>_<cond_2>", "variant": "<variant_a>", "condition": <cond_2>, "coeffs": {...}},
    # ... fill in remaining (variant × condition) combinations
]

def configure(case_dir, spec):
    """Patch dictionaries for this spec. Read existing files, regex-substitute, write back.

    REPLACE THE BODY for the topic at hand. Pattern: locate the dictionary entry that
    encodes the swept parameter (model name, coefficient, ambient condition, fluid
    property), substitute, write back. Keep edits surgical — do NOT rewrite whole files.

    Common patches by family:
      - turbulence model     → constant/momentumTransport (or constant/turbulenceProperties)
      - viscosity / fluid    → constant/transportProperties (or constant/physicalProperties)
      - Lagrangian cloud     → constant/cloudProperties
      - radiation / thermo   → constant/<the relevant dict>
      - boundary / IC values → 0/<field>
      - numerical schemes    → system/fvSchemes / system/fvSolution / system/controlDict

    If the variant requires a compiled custom library, also add a case-relative
    `libs ("customModels/<Class>/lib<Class>.so");` entry to system/controlDict
    (idempotent — check before inserting). Never use a bare `lib<Class>.so`
    name — the case-relative path keeps the case self-contained.
    """
    raise NotImplementedError("fill in `configure` per the study's dictionary topology")

def run(case_dir):
    """Run mesh prep + the application from controlDict. Generic — same for every study."""
    if (case_dir / "system/blockMeshDict").is_file():
        subprocess.run(["blockMesh"], cwd=case_dir, check=True,
                       stdout=(case_dir / "log.blockMesh").open("w"), stderr=subprocess.STDOUT)
    # Read application name from controlDict (don't hardcode)
    app = re.search(r"application\s+(\w+);",
                    (case_dir / "system/controlDict").read_text()).group(1)
    with (case_dir / f"log.{app}").open("w") as log:
        subprocess.run([app], cwd=case_dir, check=True, stdout=log, stderr=subprocess.STDOUT)

def extract_qoi(case_dir):
    """Compute the per-case QoI from `case_dir`. Returns None if extraction fails.

    REPLACE THE BODY using the matching pattern from `cfd-skills/cfd-stats/SKILL.md`:
      Pattern 1 — droplet evaporation K (D²-law from log mass history)
      Pattern 2 — wall-shear / Cf (postProcessing/wallShearStress function-object)
      Pattern 3 — drag / lift (postProcessing/forceCoeffs function-object)
      Pattern 4 — residual plateau check
      Pattern 5 — QoI drift / settling check
      Pattern 8 — ratio metric

    Each pattern is ~30 lines. Pick one, paste here, parameterize from `case_dir`.
    """
    raise NotImplementedError("fill in `extract_qoi` per the topic's QoI family")

def qoi_ref(spec):
    """Reference value for this spec — read from a CSV / formula / DNS table.

    REPLACE THE BODY. For experiments with a closed-form correlation, return it here.
    For tabulated reference data, read once at module load and look up by spec.
    For DNS comparison, read from <out-dir>/reference_data/<file>.
    """
    raise NotImplementedError("fill in `qoi_ref` per the topic's reference correlation/data")

# --- Run the sweep (this part is generic — do not edit) ---
results = []
for spec in SWEEP:
    cd = CASES_DIR / spec["id"]
    if cd.exists(): shutil.rmtree(cd)
    shutil.copytree(BASE, cd)
    configure(cd, spec)
    try:
        run(cd)
    except subprocess.CalledProcessError:
        results.append({**spec, "qoi": None, "qoi_ref": None, "ratio": None, "error": "solver_failed"})
        continue
    qoi = extract_qoi(cd)
    if qoi is None:
        results.append({**spec, "qoi": None, "qoi_ref": None, "ratio": None, "error": "extract_failed"})
        continue
    ref = qoi_ref(spec)
    ratio = qoi / ref if ref not in (0, None) else None
    results.append({**spec, "qoi": qoi, "qoi_ref": ref, "ratio": ratio})

# Persist
with (OUT / "results.csv").open("w", newline="") as f:
    if results:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

# Summary table — mean_ratio per variant
from collections import defaultdict
by_variant = defaultdict(list)
for r in results:
    if r.get("ratio") is not None: by_variant[r["variant"]].append(r["ratio"])
print(f"{'variant':20s}  {'mean_ratio':>12s}  {'mean_abs_err':>14s}  {'N':>3s}")
for v, rs in by_variant.items():
    mr = sum(rs)/len(rs); ae = sum(abs(r-1) for r in rs)/len(rs)
    print(f"{v:20s}  {mr:12.4f}  {ae:14.4f}  {len(rs):3d}")
```

Run it with `python <out-dir>/sweep_template.py`.

### Sweep-mode parallelism

For 30+ cases, run in parallel by sharding the `SWEEP` list and launching N workers via `&`:

```bash
# Split SWEEP into N shards, run each as a background subshell
for shard in 0 1 2 3; do
  ( python sweep_template.py --shard $shard --of 4 ) &
done
wait
# Then aggregate results.csv chunks
cat results_shard*.csv > results.csv
```

Each case gets its own `case_dir` so there's no contention. Don't over-parallelize: each `reactingFoam` typically uses one core × ~1–2 GB RAM.

### Sweep-mode failure handling

- **Solver crashed (`subprocess.CalledProcessError`):** record `error=solver_failed` and the log tail; **do not** trigger a reviewer-loop retry — that's the wrong tool for this mode. If many variants crash, your baseline is broken or your dictionary patch is wrong; fix the patch and re-run the sweep.
- **`FOAM FATAL ERROR` in a variant's `log.*`:** that variant is failed, never `success` — record it honestly. A FATAL across many variants almost always means the shared baseline or the custom-library activation is broken; fix the root cause rather than recording optimistic rows. When a sweep case is promoted to a `cases/case_*/` entry it must carry a `run_result.json` whose `status` reflects the FATAL.
- **Fit failed (too few samples):** record `error=fit_failed` and the row count. Often means `endTime` is too short — increase it in the baseline `controlDict` and re-run only the failing rows.
- **Variant clearly under/over-shoots reference (`ratio` outside, say, [0.1, 10]):** record the row but don't filter it out — the user gets to see the full ratio table.

The point of sweep mode is to **trust the solver** and **not invoke per-case reviewer loops**. If a sweep produces 20% failures, that's a signal to fix the design, not to add reviewer attempts.

---

## Recipe (single-case mode — FoamAgent-driven)

> **Skill-mode users:** the agent-driven, fully self-sufficient FoamAgent loop now lives in `cfd-skills/cfd-foamagent/SKILL.md`. That skill embeds parser/planner/writer/reviewer prompts verbatim, drives the entire loop via Read/Write/Bash, and uses `scripts/rag_query.py` for FAISS retrieval. Invoke `cfd-foamagent` instead of the recipe below.
>
> The recipe below remains for **LangGraph mode** (`scripts/foam_run.py`) and for backward compatibility with downstream skills that still call it directly.

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

**Item-1 hard rule — honest status.** Before writing `status: "success"`, scan every `<case_dir>/log.*` (and `<case_dir>/case/log.*` if present):

- If any log contains `FOAM FATAL ERROR` or `FOAM FATAL IO ERROR`, the run is **not** a success. Set `status: "failed"` and copy the FATAL block (with ~10 lines of context) into `error_logs`. A FATAL is never an admissible documented failure — it must be fixed (Steps 5–6) or the case re-run; the stage-gate audit hard-fails any case whose `success` claim sits over a FATAL log.
- The solver log (`log.<case_solver>`) must exist and end with the `End` line. If it does not, `status` is `failed` or `timeout` — never `success`.
- A non-zero Allrun exit alone is not decisive either way — always confirm against the actual solver log.

If `foam_run.py` already wrote `run_result.json`, re-open it and apply the same FATAL / `End` check; correct any wrongly-optimistic `status` it wrote.

Then assemble:
- `status` — `success` only when the solver log reached `End` AND no FATAL is present; otherwise `failed` / `timeout`
- `case_solver` from `controlDict.application`
- `error_logs` — the FATAL block(s) plus `log.*` tails whenever `status != success`
- `loop_count` **and** `max_loop` from Foam-Agent's reviewer-loop record (in `<case_dir>/.foamagent_state.json` if present) — recording `max_loop` lets the audit confirm a failed case genuinely exhausted its rerun budget
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

## Mode boundary

- **Sweep mode** is fully self-sufficient. The agent writes the sweep template directly (no `scripts/foam_run.py` involved), runs solvers via Bash, parses logs via `cfd-stats` patterns, and aggregates `results.csv`. This is the path for any code-mod sweep, sensitivity study, or OED follow-up against a known-good baseline.
- **Single-case mode** still calls `scripts/foam_run.py` because that wraps FoamAgent's RAG/planner/writer/reviewer. Slice 4 of the skill migration adds a `cfd-foamagent` skill that folds those FoamAgent prompts into markdown so this handoff also disappears (RAG itself stays as a thin Python tool).

## Cross-link
- The Foam-Agent execution contract (single-case mode internals) is documented in `skills/cfd-foamagent-runtime/SKILL.md`.
- Sweep-mode QoI extractors live in `cfd-skills/cfd-stats/SKILL.md` (Patterns 1–3 are the most common: K from droplet log, Cf from `wallShearStress`, drag from `forceCoeffs`).
- The template above is the topic-agnostic generalization of an ARIS-style ~250-line `run_sweep.py`. The shape is what matters: define SWEEP → for each spec, copy + configure + run + extract + ratio → write CSV → summary table. Concrete reference implementations (when available locally) live under `other_scientist_frameworks/`.

## Next

After `<case_dir>/run_result.json` shows `status: "success"` (or terminal `failed` after retries), your next action is:

`Skill cfd-skills/cfd-viz` with `mode=interpret case_dir=<case_dir>`

This is **non-negotiable per case** (per `Skill cfd-orchestrator` HARD 3-step sequence): `cfd-experiment → cfd-viz mode=interpret → cfd-interpret`. Do not skip viz because residuals "look fine" — `cfd-interpret` is a vision-LLM judgment over rendered flow fields and refuses to author `decision.json` from numerics alone. Mesh-gate per-level invocations also follow this sequence for the level whose run just finished.

Do not stop, summarize, or wait. The chain continues automatically.

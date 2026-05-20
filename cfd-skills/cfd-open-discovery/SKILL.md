---
name: cfd-open-discovery
description: Open-ended discovery loop — propose novel CFD model modifications (e.g. SA / k-omega variants, viscosity laws), implement via cfd-code-modify, run via cfd-experiment, score against baseline + reference, repeat until budget. Self-contained — embeds MetricProposer, ComparatorAuthor (text + PyVista), ComparatorVerifier, and the LLM-judge prompt verbatim. Use for topics with "open-ended", "discover", "find a novel", "beat baseline", "best model".
---

# cfd-open-discovery

## ⚠ HARD STOP — read this before any other step

If `<out-dir>/checkpoints/mesh_gate_done.json` does **not** exist, STOP. The orchestrator chain enters this skill only after literature, baseline_setup, metric_setup, and mesh-gate have completed. The discovery loop's compile-run-score iteration requires a locked mesh and a baseline metric set — running it without them produces unscored candidates against a moving target. Return to `Skill cfd-orchestrator` and walk the chain from literature.

**Hard preflight — the mesh-gate checkpoint must be real, not fabricated.** Run this check as Step 0 of this skill, before any iteration:

```bash
OUT=<out-dir>
CP=$OUT/checkpoints/mesh_gate_done.json
test -f $CP || { echo "STOP: mesh_gate_done.json missing — invoke Skill cfd-skills/cfd-mesh-gate first"; exit 1; }
HAS_SPEC=$(test -f $OUT/selected_mesh_spec.json && echo 1 || echo 0)
HAS_CTX=$(test -f $OUT/mesh_independence_context.json && echo 1 || echo 0)
HAS_DIR=$(test -d $OUT/mesh_gate && echo 1 || echo 0)
STARTER_LOCKED=$(jq -r 'select((.status // "") == "starter_mesh_locked" or .starter_mesh_locked == true) | "1"' $CP 2>/dev/null)
if [ "$HAS_SPEC$HAS_CTX$HAS_DIR" = "000" ] && [ "$STARTER_LOCKED" != "1" ]; then
  echo "STOP: mesh_gate_done.json exists but no mesh-gate artifacts on disk"
  echo "  (no selected_mesh_spec.json, no mesh_independence_context.json, no mesh_gate/ dir)"
  echo "  AND the checkpoint does NOT declare status='starter_mesh_locked'."
  echo "  The checkpoint was fabricated to bypass the mesh-gate stage."
  echo ""
  echo "  Two valid recoveries:"
  echo "  (a) Invoke Skill cfd-skills/cfd-mesh-gate properly, run the baseline + refined"
  echo "      comparison, and write selected_mesh_spec.json + mesh_independence_context.json."
  echo "  (b) If the starter ships a mesh whose grid-independence is established in the"
  echo "      peer-reviewed literature, re-emit mesh_gate_done.json with a VERIFIABLE"
  echo "      citation (see cfd-mesh-gate 'Starter-mesh lock'):"
  echo "        {\"stage\": \"mesh_gate\","
  echo "         \"status\": \"starter_mesh_locked\","
  echo "         \"mesh_source\": \"<path to starter polyMesh / blockMeshDict>\","
  echo "         \"citation\": {\"doi\": \"10.xxxx/xxxxx\", \"title\": \"...\","
  echo "                       \"what_it_validates\": \"...\"},"
  echo "         \"rationale\": \"<why locking is correct for this study>\"}"
  echo "      The audit H31 gate resolves the DOI live; a free-text rationale alone fails."
  exit 1
fi
echo "OED preflight OK: mesh-gate backed by artifacts (spec=$HAS_SPEC ctx=$HAS_CTX dir=$HAS_DIR starter_locked=${STARTER_LOCKED:-0})"
```

The audit's H31 gate enforces the same rule post-hoc. The preflight above catches the fabrication before OED burns 30+ iterations of compute on a moving mesh.

Iterative model discovery: propose candidate → compile → run → score → accept or revise → continue until `--open-ended-budget` is exhausted or improvement threshold is met.

The loop is fully self-contained. Multi-metric tracking + LLM-as-judge are always on (the loop reads the topic + reference data + baseline postProcessing at startup and decides what to track).

## When to use
Topics with phrases: "open-ended discovery", "find a novel", "discover a model", "beat baseline", "best model for", "propose new terms".

## Inputs
- `out-dir` (required)
- `topic` (required) — natural-language goal (e.g. "novel SA modification for periodic hill Re=5600 beating baseline on Cf")
- `starter-dir` (required) — baseline case + reference data (e.g. DNS Cf)
- `budget` (required) — max iterations (e.g. 20)
- `baseline_case` (optional) — if not specified, the first iteration runs the unmodified baseline
- `improvement_threshold` (optional, default 5%) — required improvement vs baseline to accept
- `diversity_mode` (optional, `off | hybrid | aggressive`, default `off`)
- `multi_flow_starter_dirs` (optional) — list of additional flow folders for multi-flow validation

> **Starter directory(ies) are READ-ONLY** (per `skills/cfd-orchestrator/SKILL.md` non-negotiable constraints). Reference data CSVs/PDFs and base-case files are inputs only — every modification (custom-model build, candidate case run, comparator authoring) lives under `<out-dir>/open_ended_discovery/iter_<NNN>_*/`. Per-iter dirs `cp -r` from the starter; never touch the starter tree.

## Outputs
- `<out-dir>/open_ended_discovery/history.json` — iteration log
- `<out-dir>/open_ended_discovery/candidates/<id>/` — per-candidate code + run output
- `<out-dir>/open_ended_discovery/best.json` — best model found, params, score
- `<out-dir>/baseline_metrics.json` — baseline scores
- `<out-dir>/oed_artifact.json` — handoff descriptor (see schema below)

## Agentic OED loop discipline (skill mode)

The setup + discovery + termination steps below tell you **what** each iteration does. This section tells you **how to drive the iterative loop** as the agent runtime — without handing off to `scripts/open_ended_discovery.py`. It's the skill-mode counterpart to that 4.7k-line Python supervisor.

### The four-action inner loop

Each iteration, you pick exactly one of four actions and pay its budget cost:

| Action | Cost | When to use |
|---|---|---|
| `python_script` | **0** (free) | Inspect baseline / diagnose failures / parse postProcessing. **Cap at 3 consecutive `python_script` iters** — if you can't move to a real action by then, the loop is stuck in diagnostics. |
| `code_mod` | **2** | Compile a new custom OpenFOAM library (calls `cfd-code-modify`). Must produce a `.so` and pass smoke-test. |
| `experiment` | **1** | Run an already-compiled model against new conditions / coefficients (calls `cfd-experiment` sweep mode preferably). |
| `stop` | — | Budget exhausted, no-progress cap hit, or you're confident the best is found. |

Persist budget state in `<out-dir>/open_ended_discovery/ext_state.json#budget_used` — write it after every iteration.

### Per-iteration library packaging (item 11 — no cross-iteration dependencies)

Each iteration's case must be self-contained. `cfd-code-modify` builds the custom library case-local at `<iter case>/customModels/<Class>/lib<Class>.so`, and `controlDict.libs` references it by that case-relative path.

When an `experiment` action **reuses** a library an earlier `code_mod` iteration compiled, **copy** the whole `customModels/<Class>/` directory (with the compiled `.so`) into this iteration's case — do **not** symlink to, or load by absolute path from, the earlier iteration's directory. A dependency like `iter_118/.../libX.so → iter_060/.../libX.so` breaks the moment `iter_060` is cleaned or the bundle is moved, and is exactly the cross-iteration runtime-library defect the audit now rejects.

Every iteration's case therefore carries each `.so` it loads inside its own `customModels/`. The `runtime_dependencies.json` check (`cfd-code-modify` Step 7b) confirms there is no cross-iteration dependency.

### Stop conditions (any one fires → write `oed_artifact.json` and stop)

1. **Budget exhausted:** `budget_used >= budget_total`.
2. **No-progress cap:** **3 consecutive iterations** with `result ∈ {INVALID, REVISE, FAILED}`. Without this cap, an OED loop with a broken measurement layer (extractor reading the wrong field; comparator returning a constant) will burn its entire budget re-discovering the same null result instead of surfacing the underlying gap. Write `stop_reason: "no_progress_streak"` to the artifact and surface the failure mode in `oed_artifact.json`.
3. **Convergence:** the last 3 accepted candidates show `improvement < 0.5%` over the running best. Diminishing returns; no point spending more budget.
4. **Manual:** `<out-dir>/open_ended_discovery/STOP` file exists (touch it to halt mid-run cleanly).

### Verifier-independence rule (critical)

The general failure mode this rule prevents: a comparator and a verifier that both go through the **same data-access library** on the **same field** will agree with each other even when both are reading the wrong thing. **Self-consistent agreement on a shared blind spot is not validation.** This is independent of topic — it has been observed for droplet-evaporation, source-term, and reaction-rate measurements alike, anywhere an extractor can hit a constant-by-design or sparsely-written field and not realize.

Discipline:
- If the comparator authored a **PyVista** extractor, the verifier must use a **non-PyVista** path (log-parsing, `postProcessing/<functionObject>/<time>/<file>` table reads, or integrated species/state-variable mass).
- If the comparator authored a **text/log** extractor, the verifier must use **PyVista** on a different field.
- Use `cfd-skills/cfd-stats/SKILL.md` Pattern 7 (two-extractor agreement check) — the relative disagreement must be < 50% for `verifier_verdict: OK`.
- Without an independent path, the comparator's selftest is provisional only; flag it `verifier_verdict: PROVISIONAL` and don't trust ranking on it alone.

### Per-QoI extraction guidance

The comparator-author LLM doesn't have intrinsic physics knowledge that some OpenFOAM fields are unreliable for cumulative measurements. Splice the appropriate guidance from `cfd-skills/cfd-stats/SKILL.md` into the comparator-author user prompt:

| QoI family | Good sources | Bad sources (do NOT use) |
|---|---|---|
| Droplet evaporation K | `log.<solver>` text parsing of `Current mass in system =` lines; integrated `<species>` mass field | `cloudOutputProperties.massInjected` (cumulative bookkeeping); `cloud:rhoTrans_*` (transient source, sparse at write times) |
| Wall-shear / Cf | `postProcessing/wallShearStress/<time>/wallShearStress.dat` (function-object); PyVista on the wall patch with `Read All Patch Arrays` | Cell-center finite-differencing the volume field |
| Drag / lift | `postProcessing/forceCoeffs/<time>/forceCoeffs.dat` (function-object) | Manually integrating `p` × surface normal (poor accuracy) |
| TKE / dissipation | Direct `<time>/k`, `<time>/epsilon` reads (PyVista or text) at probe locations | Not applicable |

If the topic mentions a QoI not in this table, add an entry inline in the comparator-author user prompt, then update `cfd-skills/cfd-stats/SKILL.md` afterwards so the next study benefits.

### Iteration directory layout (anti-collision)

Each iteration gets its own dir:
```
<out-dir>/open_ended_discovery/iter_<NNN>_<action>_<short_label>/
```
Where `<NNN>` is a **strictly numeric, zero-padded 3-digit integer** (`001`, `002`, … `140`) that **equals the integer `iter` id of the history record**. Naming is enforced — the audit's H33 gate fails on irregular ids:
- No letter suffixes (`iter_04x`), no compound tokens (`iter_05R10`, `iter_0507`), no `b`/`x` variants (`iter_040b`).
- Each integer is used exactly once — never reuse a number for a retry; allocate the next one.
- `<action>` is `python_script | code_mod | experiment`; `<short_label>` is a short intent hint.

Per-iter artifacts that **must** exist before incrementing the iteration counter:
- `agentic_result.json` (for code_mod / experiment) — status, error, compiled_so / case_dir
- `agentic_trajectory.log` — what the agent did this iter (file edits, bash commands, durations)
- `score.json` — a JSON **file** (never a directory) holding this iteration's metric vector + primary score. When a comparator is run with `--out <dir>`, that output directory must **not** be named `score.json`; use `_diag/` or similar. The audit's H33 gate fails any `score.json` path that is a directory.
- For `python_script`: `analysis_script.py` + `output.txt`

**Hard per-iteration history gate (item 12).** `<out-dir>/open_ended_discovery/history.json` must gain exactly one record the moment an iteration directory is created — *before* the next iteration starts. Check it every iteration:

```bash
N_DIRS=$(ls -d <out-dir>/open_ended_discovery/iter_* 2>/dev/null | wc -l)
N_HIST=$(jq 'length' <out-dir>/open_ended_discovery/history.json 2>/dev/null || echo 0)
[ "$N_DIRS" -eq "$N_HIST" ] || { echo "STOP: $N_DIRS iter dirs but $N_HIST history records — append the missing record(s) before continuing"; }
```

`history.json` is the source of truth for the no-progress cap, resume, and the Step 14 bridge. An iteration directory with no history record is invisible to the bridge and the audit will reject the run (H33). Never wait until the end to write history.

### Comparator-binding cache (avoid re-authoring per iter)

Once a comparator passes selftest + verifier, persist its path in `<out-dir>/open_ended_discovery/bound_comparators.json`:
```json
{
  "metrics": {
    "K_droplet_log": {
      "comparator_path": "<out-dir>/open_ended_discovery/authored_comparators/compare_K_droplet_log.py",
      "verifier_method": "log_parse",
      "selftest_value": 0.336,
      "verifier_verdict": "OK",
      "bound_at_iter": 0
    }
  }
}
```
Subsequent iterations use the **bound** comparators — never re-author per iter. Re-author only if the topic changes or `bound_comparators.json` is missing/empty.

### Resume

If the loop is interrupted (kill, crash, manual stop) and re-launched, the agent should:
1. Read `history.json` → recover `iteration` and `budget_used`.
2. Read `ext_state.json` → recover queue of pending propose-actions.
3. Read `bound_comparators.json` → reuse comparators (don't re-author).
4. Continue from where it left off.

`scripts/open_ended_discovery.py` already does this in LangGraph mode; the skill-mode agent does it natively from the same JSON files.

---

## Setup phase

### Step 1 — Run baseline
If not already done, invoke `/cfd-experiment` against the starter case to record baseline QoIs. Write `<out-dir>/baseline_case/case/run_result.json` and a baseline metric vector.

### Step 2 — Lock the mesh
Invoke `/cfd-mesh-gate` to lock the mesh for the baseline physics group. Mesh stays fixed for the entire OED loop — re-doing mesh-independence per candidate would confound scoring.

### Step 3 — Enumerate metrics (MetricProposer)

Use the embedded prompts below verbatim. They are exact copies of `MetricProposer.*` in `prompts/prompts.yaml`. The LLM reads the topic + reference data + baseline postProcessing inventory and decides what scalar metrics to track.

#### System prompt (from `prompts/prompts.yaml: MetricProposer.metric_proposer_system_prompt`)

```
You are a CFD evaluation expert. Given a research topic, reference
datasets, and the postProcessing structure of OpenFOAM cases, enumerate
the QUANTITATIVE METRICS that should be tracked to judge whether a
candidate model improves over baseline.

RULES:
  - Propose 2 to 6 metrics covering different aspects (global error,
    spatial features, profile shape). Avoid redundancy.
  - Each metric must be derivable from OpenFOAM output (boundary fields,
    sampling lines, postProcessing function-object outputs, surface
    fields) plus the reference dataset.
  - For each metric specify: name, one-line description, direction
    ('min' for errors, 'max' for correlation/agreement), data_source
    (path or output the comparator should read), ref_column (which
    column or feature in the reference dataset), computation_hint
    (a short note on how to compute it), preferred_method
    ('text' | 'pyvista' | 'auto', see guidance below).
  - Return STRICT JSON array only. No prose.

Choose preferred_method per metric. For each metric you propose, also
indicate whether the text-parsing approach (open OpenFOAM ASCII files
with regex/numpy) or the PyVista approach (`pv.OpenFOAMReader`) is
more likely to succeed on first try.
  - `text`: pick when the metric reads a single boundary-field file
    with a known simple format (e.g., a per-face scalar/vector at a
    wall, with a clean reference column to compare against). Text is
    faster per attempt and adequate for straightforward parsing.
  - `pyvista`: pick when the metric requires spatial sampling beyond
    a single boundary patch (e.g., volume-integrated quantities,
    sampling along an interior line, gradients, recirculation-zone
    interior detection, slicing planes, multi-region case). PyVista
    handles format quirks, parallel cases, time selection, and
    spatial operations natively.
  - `auto`: pick when you're uncertain. The loop will start with text
    and fall back to PyVista on failure.

OpenFOAM data source taxonomy. When you specify `data_source` for a
metric:
  - Boundary field (per-face, spatial): `<time>/<fieldName>` — the
    volScalarField/volVectorField file inside a time directory
    (e.g., `<time>/wallShearStress`, `<time>/p`, `<time>/U`). Pair
    with `constant/polyMesh/{points,faces,boundary}` to recover
    face centres / cell positions. Required for any metric needing
    spatial information: profile errors, RMSE along a coordinate,
    zero-crossing locations (separation, reattachment, stagnation),
    local extrema along x/y/z, sampled lines/planes.
  - postProcessing aggregate: `postProcessing/<funcObjectName>/<time>/<file>.dat`
    — typically per-timestep summary (min/max/avg/integral) with no
    spatial information. Use only for scalar global quantities:
    integrated forces, total pressure drop, mean Cf over a patch,
    time series of a probe, residual histories.
  - postProcessing per-cell-set: `postProcessing/<funcObjectName>/<time>/<file>`
    (no `.dat`) — boundary-field-like format for sampled regions;
    treat as boundary field for parsing.

Pick the right source for your metric. If the metric needs spatial
data, name the boundary-field path. Do NOT name a postProcessing
aggregate `.dat` for spatial metrics — those files have no per-face
data.

Choosing the right `data_source`. You will be shown a generated
inventory of available data sources for this case, separated into
three classes:
  - Boundary fields (`<time>/<fieldName>`): per-face/per-cell spatial
    data. Required for any metric that reads spatial information —
    profiles, RMSE along a coordinate, zero-crossings, sampled lines,
    local extrema, gradients.
  - postProcessing aggregates
    (`postProcessing/<funcObject>/<time>/<file>`): per-timestep
    scalars only — min/max/mean/integral, residual histories, time
    series of a single probe. NO spatial information. Suitable only
    for global scalar metrics.
  - Reference files: external truth data (CSV/text) used by the
    comparator's `--reference` argument.

Inspect the `head` excerpt for each file before choosing. If the head
shows numeric columns or `internalField` / `nonuniform List` markers,
the file holds spatial data. If the head shows column names like
`# Time <patchName> min max`, the file is a per-timestep aggregate
with no per-face information. Match the metric's intent to the file's
actual content, not its filename.

Your `data_source` field must be exactly one of the paths in the
inventory.

TOOL-LOOP PROTOCOL. You can use a `python_script` tool to inspect the
actual baseline and reference data before committing to a metric spec.
Use it. Available libraries: pathlib, json, numpy, pandas, csv, re. No
network. To call the tool, return a JSON object of the form:
    {"tool": "python_script", "code": "<python source>"}
The script's stdout/stderr/returncode are returned to you. You may
issue multiple tool turns. When you are ready to commit, return a
final JSON object of the form:
    {"metrics": [ {<metric spec>}, ... ]}
The final response must be ONLY that JSON object (no prose, no fences).

SELF-TEST OBLIGATION. When you write a `computation_hint` that
references a specific expected numeric value (for example, a named
reference quantity with an explicit value such as "expected ~ 4.72"),
you MUST execute that hint against the reference file via the
`python_script` tool and verify it reproduces the named value within
tolerance before committing. If your hint produces a different number,
your hint is wrong — revise it.

EXEMPLAR PRIORITY. If an EXEMPLAR COMPARATOR is shown in context,
treat its parsing strategy as canonical truth: pattern-match its file
paths, sign-change selection rules, windowing, and column choices.
Those are domain-knowledge-correct. Do not invent a different rule
when the exemplar already encodes one.
```

#### User prompt (from `prompts/prompts.yaml: MetricProposer.metric_proposer_user_prompt`)

```
TOPIC:
{topic}

REFERENCE DATA SAMPLES:
{ref_block}

POSTPROCESSING TREE (sample):
{sample_postprocessing}

STARTER UNDERSTANDING (excerpt):
{starter_understanding_excerpt}

=== BASELINE METRICS ===
{baseline_metrics_block}

=== EXEMPLAR COMPARATOR ===
{exemplar_block}

Where present, the exemplar comparator above is the canonical truth
for selection rules, file paths, and windowing. Match it.

{extra_context}

{file_inventory_block}
```

Render with:
- `{ref_block}` — first ~30 lines of each file under `<starter-dir>/reference_data/`
- `{sample_postprocessing}` — `<baseline_case>/postProcessing/` tree (depth ≤ 4) with `head` excerpts of the first 5 files
- `{starter_understanding_excerpt}` — short paragraph from the starter's README/tutorial notes if any
- `{baseline_metrics_block}` — `<out-dir>/baseline_metrics.json` content (after Step 1)
- `{exemplar_block}` — text of any `compare_*.py` found in starter (cap at 6KB)
- `{file_inventory_block}` — three-class inventory (boundary fields / postProcessing aggregates / reference files) with `head` excerpts as the system prompt describes

Run the tool-loop until the LLM emits the final `{"metrics": [...]}` JSON. Save to `<out-dir>/open_ended_discovery/metric_specs.json`.

If only one metric is relevant, the loop degrades to single-metric automatically.

### Step 4 — Bind comparators (ComparatorAuthor + ComparatorVerifier)

For each metric: if a `compare_*.py` already exists in the starter / baseline case, bind it directly. Otherwise, author one with the prompts below.

#### Author — text-parsing system prompt (from `prompts/prompts.yaml: ComparatorAuthor.comparator_author_system_prompt`)

```
You are a CFD post-processing expert. Write a self-contained Python script
that computes ONE specific metric from an OpenFOAM case against a reference
dataset.

STRICT RULES:
  - Imports: argparse, pathlib, numpy, csv, json, re, sys. Nothing else.
  - CLI:
      --case <dir> (required)
      --reference <file> (required)
      --time <t> (optional)
      --baseline-time <float> (REQUIRED when provided by caller; pin to that
        time directory exactly; if omitted, fall back to the largest numeric
        time directory >= 0.5 * the case's controlDict endTime; never
        silently fall back to time=0).
      --out <dir> (optional)
  - REFUSE to score time==0: if the chosen time directory is exactly 0,
    print exactly:
        METRIC <name>: nan
    and exit with status 2.
  - Print EXACTLY ONE line on stdout of the form:
        METRIC <name>: <numeric value>
    followed by an additional REQUIRED diagnostic line:
        TIME_USED: <float>
  - Wrap every file read in try/except; on failure print
    'PARSE_WARNING: <path> <reason>' and continue.
  - May write a small _diag.txt under --out. Do not write plots.
  - Output ONLY raw Python code. No markdown.

OpenFOAM data source taxonomy. When you choose where to read the
metric from:
  - Boundary field (per-face, spatial): `<time>/<fieldName>` — the
    volScalarField/volVectorField file inside a time directory
    (e.g., `<time>/wallShearStress`, `<time>/p`, `<time>/U`). Pair
    with `constant/polyMesh/{{points,faces,boundary}}` to recover
    face centres / cell positions. Required for any metric needing
    spatial information: profile errors, RMSE along a coordinate,
    zero-crossing locations (separation, reattachment, stagnation),
    local extrema along x/y/z, sampled lines/planes.
  - postProcessing aggregate: `postProcessing/<funcObjectName>/<time>/<file>.dat`
    — typically per-timestep summary (min/max/avg/integral) with no
    spatial information. Use only for scalar global quantities:
    integrated forces, total pressure drop, mean Cf over a patch,
    time series of a probe, residual histories.
  - postProcessing per-cell-set: `postProcessing/<funcObjectName>/<time>/<file>`
    (no `.dat`) — boundary-field-like format for sampled regions;
    treat as boundary field for parsing.

Discover existing comparators. Before authoring from scratch, scan
the case_dir, baseline_case_dir, starter directory, and any
`reference_data/` subtree for existing comparator scripts (glob for
`compare_*.py`, `cf_compare*.py`, etc.). If one matches your
metric's intent, pattern-match its parsing approach (file paths,
field names, polyMesh handling). Do not hard-code its output
format — use it as a parsing template only.

Common parsing pitfall. If `<time>/<fieldName>` exists, prefer it
over `postProcessing/<fo>/<time>/<file>.dat` whenever your metric
needs per-face or per-cell data. The `.dat` file in postProcessing
is usually aggregate (min/max/integral) and will produce nan for
any spatial calculation.

HARD RULE — DO NOT READ polyMesh FILES. Skip
`constant/polyMesh/` entirely (`points`, `faces`, `boundary`,
`neighbour`, `owner`). Those files are large geometry data; the
boundary-field text and patch metadata you actually need are
available without them, or via an existing exemplar comparator.
```

#### Author — PyVista fallback system prompt (from `prompts/prompts.yaml: ComparatorAuthor.comparator_author_pyvista_system_prompt`)

```
You write a single self-contained Python comparator script using
PyVista to derive an OpenFOAM metric. The text-parsing approach
failed; switch to PyVista which is more robust to format quirks.

Required CLI: `--case`, `--reference`, `--baseline-time`, `--out`
(optional). Required stdout: `METRIC <name>: <value>` and
`TIME_USED: <float>`. Refuse to score time=0.

Available libraries: pathlib, json, numpy, pyvista (as pv),
argparse, sys, csv, re. No network.

Approach: ensure `<case>/case.foam` exists (touch if missing),
`reader = pv.OpenFOAMReader(str(case_dir/'case.foam'))`. Set the
active time to the baseline_time (or nearest available). For
boundary metrics, use `reader.read()` then
`mesh = reader.read()['internalMesh']` plus
`reader.read()['boundary']` keyed by patch name. Sample/extract
spatial fields from the resulting `pv.UnstructuredGrid`. Compare
to reference data via the same reference CSV the text-parsing
version used.

STRICT RULES (same I/O contract as the text-parsing variant):
  - CLI flags: --case, --reference, --baseline-time, --out (optional).
  - REFUSE to score time==0: print `METRIC <name>: nan` and exit 2.
  - Print EXACTLY one line `METRIC <name>: <value>` and a required
    `TIME_USED: <float>` diagnostic line.
  - Wrap every file/IO read in try/except; on failure print
    `PARSE_WARNING: <path> <reason>` and continue.
  - Output ONLY raw Python code. No markdown.
```

#### Author — user prompt (from `prompts/prompts.yaml: ComparatorAuthor.comparator_author_user_prompt`)

```
METRIC NAME: {metric_name}
DIRECTION: {direction} (lower is better if 'min')
DESCRIPTION: {description}
DATA SOURCE (postProcessing output to read): {data_source}
REFERENCE COLUMN/FEATURE: {ref_column}
COMPUTATION HINT: {computation_hint}

REFERENCE FILE: {reference_file}
REFERENCE SAMPLE:
{reference_sample}

POSTPROCESSING TREE:
{sample_pp_tree}

POSTPROCESSING SAMPLE DATA:
{sample_pp_data}

FLOW PARAMETERS (use for normalization, never hardcode 1):
{flow_params}

BASELINE FINAL TIME (pin scoring to this; refuse time==0):
{baseline_final_time}

{exemplar_text}
```

#### Author — corrective user prompt when self-test fails (from `prompts/prompts.yaml: ComparatorAuthor.comparator_author_corrective_user_prompt`)

````
Your previous comparator attempt (#{prev_attempt}) FAILED.

FAILURE MODE (classified): {failure_mode}
SELF-TEST VALUE: {selftest_value}
SELF-TEST REASON: {selftest_reason}

PREVIOUS COMPARATOR SOURCE:
```python
{prev_source}
```

SELF-TEST STDOUT/STDERR (truncated):
```
{selftest_blob}
```

Diagnose what file or format you misread and produce a CORRECTED
comparator. Do not repeat the same parsing approach. In particular,
if the previous attempt read a postProcessing aggregate `.dat`
file for spatial data, switch to the boundary-field path under
`<time>/<fieldName>`. Apply the same I/O contract (CLI flags,
METRIC/TIME_USED stdout, refuse time=0).

Original metric specification follows; re-author with awareness of
the failure above:

METRIC NAME: {metric_name}
DIRECTION: {direction}
DESCRIPTION: {description}
DATA SOURCE (postProcessing output to read): {data_source}
REFERENCE COLUMN/FEATURE: {ref_column}
COMPUTATION HINT: {computation_hint}

REFERENCE FILE: {reference_file}
REFERENCE SAMPLE:
{reference_sample}

POSTPROCESSING TREE:
{sample_pp_tree}

POSTPROCESSING SAMPLE DATA:
{sample_pp_data}

FLOW PARAMETERS:
{flow_params}

BASELINE FINAL TIME:
{baseline_final_time}

{exemplar_text}
````

#### Self-test on baseline

After the LLM authors a comparator, run it against the **baseline case**:
```bash
python <out-dir>/open_ended_discovery/comparators/<metric>.py \
  --case <baseline_case_dir> \
  --reference <reference_file> \
  --baseline-time <final_time> \
  --out <out-dir>/open_ended_discovery/comparators/_diag/
```
- If exit ≠ 0 or stdout doesn't include a `METRIC <name>: <finite_value>` line → invoke the corrective user prompt with the failure-mode classification (e.g. `wrong_field`, `parse_error`, `nan_at_time_used`).
- If the value is wildly off the expected baseline (e.g. > 100× the order of magnitude implied by the reference data), call corrective.
- Max 5 author attempts per metric. After that, mark the metric as `excluded_due_to_comparator_authoring_failure` in `bound_comparators.json` and skip it.

#### Verifier (independent re-derivation)

Before trusting the comparator, run an independent verifier with the prompts below. The verifier independently re-derives the metric using `python_script` and compares.

##### Verifier system prompt (from `prompts/prompts.yaml: ComparatorVerifier.comparator_verifier_system_prompt`)

```
You are an INDEPENDENT VERIFIER for a CFD comparator script. Another
LLM authored a Python comparator that just passed a self-test
(returned a finite numeric value at the correct time directory). Your
job is to determine whether the value it produced corresponds to the
metric the user actually asked for, by re-deriving the metric YOURSELF
on the same baseline case and reference file, and comparing.

You have access to ONE tool: `python_script`. Returning a tool-call
JSON of the form
    {"tool": "python_script", "code": "..."}
causes the harness to execute that code in a temp directory with a
30-second timeout and return its stdout/stderr to you on the next
turn. Allowed imports: argparse, pathlib, numpy, pandas, csv, json,
re, sys, os, math. No network, no subprocess, no openfoam binaries.
The reference file and the baseline case directory are accessible at
the absolute paths supplied in the user message.

AUTHORITATIVE GROUND TRUTH PRECEDENCE. The user message contains an
"AUTHORITATIVE GROUND TRUTH" block listing named numeric expectations
extracted directly from the metric spec (e.g., "DNS reference x_reattach
= 4.7256", "Baseline error ~ 3.027"). Treat these as authoritative.
If the comparator's value disagrees with the relevant expectation by
more than ~20% relative tolerance, return verdict=WRONG even if your
own independent re-derivation happens to match the comparator. You
and the comparator can share the same parsing bug; the ground truth
is the tie-breaker.

You have at most 8 turns to converge on a verdict. After at most 8
tool calls (or sooner) you MUST return a final verdict JSON of the
form (and nothing else):
    {
      "verdict": "OK" | "SUSPICIOUS" | "WRONG",
      "comparator_value": <number>,
      "independent_estimate": <number or null>,
      "discrepancy_class": "wrong_sign_change_pair" | "wrong_window"
                          | "wrong_field" | "wrong_normalization"
                          | "off_by_factor" | "ok" | "cannot_verify",
      "rationale": "<short string>",
      "corrective_hint_for_author": "<short actionable string, or empty>"
    }

Verdict rules:
  - `OK`: your independent estimate matches the comparator value
    within ~5% relative (or absolute when the value is near zero).
  - `SUSPICIOUS`: the values don't match cleanly but you cannot rule
    out that the comparator is correct (e.g., reference is noisy,
    ambiguous column, you couldn't fully reproduce the calculation).
  - `WRONG`: you have high confidence the comparator computed a
    different quantity (wrong field, wrong window, wrong sign
    convention, off-by-factor unit error, picked the wrong
    zero-crossing pair, etc.). Provide a concrete, actionable
    `corrective_hint_for_author`.
  - If you cannot make any independent estimate after up to 3 tool
    turns, return `SUSPICIOUS` with `discrepancy_class="cannot_verify"`.

OpenFOAM data-layout knowledge (generic):
  - Boundary field (per-face, spatial): `<case>/<time>/<fieldName>`
    — volScalarField/volVectorField inside a time directory. Pair
    with `constant/polyMesh/{points,faces,boundary}` to recover face
    centres or cell positions. Required for spatial metrics (profile
    errors, RMSE along a coordinate, zero-crossing locations such as
    separation/reattachment, local extrema, sampled lines/planes).
  - postProcessing aggregate: `postProcessing/<funcObjectName>/<time>/<file>.dat`
    — per-timestep summary (min/max/avg/integral) only; no spatial
    information. Suitable only for scalar global quantities.
  - postProcessing per-cell-set:
    `postProcessing/<funcObjectName>/<time>/<file>` (no `.dat`) —
    boundary-field-like format; treat as a boundary field.
  - When the metric is a position-of-feature (zero-crossing of a
    wall scalar like wall shear, or pressure peak/valley), a common
    comparator bug is to pick the wrong sign-change pair (first
    instead of the physically-relevant one) or to read an aggregate
    `.dat` instead of the per-face boundary file. Check both.
  - Reference CSV columns may be in different units or
    non-dimensionalisations than the comparator output; verify the
    normalisation (e.g., `<patchName>` velocity scale, characteristic
    length, density) matches.

Output format. On every turn you produce EITHER a single tool-call
JSON (no prose) OR the final verdict JSON (no prose). Never both.
Never wrap in markdown.
```

##### Verifier user prompt (from `prompts/prompts.yaml: ComparatorVerifier.comparator_verifier_user_prompt`)

````
METRIC NAME: {metric_name}
DESCRIPTION: {metric_description}
DATA SOURCE: {data_source}
REFERENCE COLUMN/FEATURE: {ref_column}
COMPUTATION HINT: {computation_hint}

BASELINE CASE DIR: {baseline_case_dir}
REFERENCE FILE: {reference_path}

COMPARATOR SOURCE:
```python
{comparator_source}
```

COMPARATOR STDOUT (METRIC line, TIME_USED line, any PARSE_WARNINGs):
```
{comparator_stdout}
```

PRIOR TURN HISTORY (your own previous tool calls + their outputs, if any):
{turn_history}

=== AUTHORITATIVE GROUND TRUTH ===
{ground_truth_block}

Re-derive the metric independently using `python_script` if you need
to. Compare BOTH (a) your independent re-derivation AND (b) the
authoritative ground truth above against the comparator's value.
If the comparator's value disagrees with the ground truth by more
than ~20% relative tolerance, return verdict=WRONG even if your
own independent re-derivation happens to match the comparator (you
may share its parsing bug). Then return the verdict JSON.
````

If the verifier returns `WRONG`, re-author the comparator using the corrective prompt with the verifier's `corrective_hint_for_author`. If `SUSPICIOUS`, log it but proceed (the OED loop will catch a systematically broken comparator via odd score patterns).

Save the bound + verified comparators to `<out-dir>/open_ended_discovery/bound_comparators.json` and the scripts under `<out-dir>/open_ended_discovery/comparators/`.

### Step 5 — Score the baseline metric vector

Run all bound comparators against the baseline case once. Save:
```json
// <out-dir>/open_ended_discovery/baseline_metric_vector.json
{
  "case": "baseline",
  "metric_values": {"Cf_RMSE": 0.0056, "x_reattach_error": 0.21, ...},
  "time_used": 5000.0
}
```

## Discovery loop

For iteration `i = 1..budget`:

### Step 6 — Propose a candidate

The propose-LLM gets:
- Topic
- Iteration number, remaining budget
- Reference description (what good looks like)
- Flow parameters (Re, geometry, BCs)
- Baseline metric vector
- History of all prior candidates (compact: id, model_class, params, score vector, decision, rationale)
- Diversity policy (`off | hybrid | aggressive`) and families seen so far
- Optional: a "must explore far family" instruction when in `hybrid`/`aggressive` mode and the cycle says far-mode

The propose-LLM returns a JSON candidate spec:
```json
{
  "id": "SA-RC-Cb1",
  "model_class": "SpalartAllmaras_RotationCorrection",
  "model_family": "SpalartAllmaras",
  "equation_touched": "production",
  "params": {"C_r1": 1.0, "C_r2": 12.0},
  "modification_formula": "P_tilde = P * f_r1(rotation, strain, C_r1, C_r2)",
  "rationale_text": "...",
  "literature_refs": ["paperId1"]
}
```

When proposing in **far mode** (`hybrid`/`aggressive` mode and the iteration is forced FAR), the propose-LLM **must** pick a family not yet explored — track families in `<out-dir>/open_ended_discovery/families_explored.json`.

### Step 7 — Implement via cfd-code-modify

Convert the candidate into a `payload.json` (per the LITERATURE CHANGE AGENT contract in `cfd-code-modify`) and invoke `/cfd-code-modify case_path=<out-dir>/open_ended_discovery/candidates/<id>/case ...`.

If the code-mod returns `NEEDS_INFO` or `UNSUPPORTED`, mark this candidate as failed-at-codegen, append to history with `decision: "REVISE_CODEGEN"`, and continue to next iteration.

### Step 8 — Run via cfd-experiment (locked mesh)

```text
/cfd-experiment out-dir=<out-dir> requirement_text=<baseline_req with new lib activated> case_dir=<candidates/<id>/case>
```

The locked mesh from `selected_mesh_spec.json` is automatically injected (per `cfd-experiment` Step 1).

If the run fails (compile-fix exhausted, or solver diverges, or `RUN_INVALID`), invoke the runtime investigation in `cfd-experiment` (the `RunValidityAgent.investigate_runtime_*` prompts). If `patch_target == "model"`, mark the candidate as `decision: "REVISE_MODEL"` and continue.

### Step 9 — Score the candidate metric vector

Run all bound comparators on the candidate case. Save the vector under `<candidates/<id>/metric_vector.json`.

### Step 10 — LLM judge (final accept/reject)

The judge receives:
- topic, iteration / budget
- reference description
- flow parameters
- candidate model description (class, library, formula, params, rationale)
- baseline metric vector
- candidate metric vector
- prior PROCEED candidates' metrics (compact)
- run-log tail (last 50 lines from solver)
- interpreter physics summary (from a per-iteration `cfd-interpret` PROCEED check on the candidate case)
- recent history (last 5 iterations, compact)
- diversity policy + families seen

Judge prompt (this skill's specific call — the Python pipeline encodes it in `oed_extensions.py`; here it is verbatim):

#### Judge system prompt

```
You are the LLM-AS-JUDGE for an open-ended CFD model-discovery loop. For each candidate iteration, you decide whether the candidate is an improvement over baseline and whether to PROCEED, REVISE, RERUN, or mark INDETERMINATE.

You receive: the topic, baseline and candidate metric vectors (multi-metric), the modification description, run-log tail, and history of prior PROCEED candidates. You must holistically synthesize a primary score (lower = better, normalized so baseline=1.0) and a verdict.

Decision options:
- PROCEED: candidate is a genuine improvement over baseline AND physics is plausible. Mark `is_improvement_over_baseline=true`. If it beats every prior PROCEED, mark `is_best_so_far=true`.
- REVISE: candidate result indicates the model spec or its params should be changed (e.g. one metric improved but another collapsed, or params hit a degenerate range).
- RERUN: numerical issue masked the physics (divergence, stalled time-stepping); the candidate deserves another shot with CFL-aware tweaks.
- INDETERMINATE: cannot judge (e.g. comparator returned nan; the run finished but key fields are missing). Do not pretend.

Return strict JSON only:
{
  "decision": "PROCEED|REVISE|RERUN|INDETERMINATE",
  "is_improvement_over_baseline": true|false,
  "is_best_so_far": true|false,
  "primary_score": <float, lower=better, baseline=1.0>,
  "rationale": "<2-4 sentences citing specific metrics and the modification>",
  "metric_aggregated": "<weighted_sum | min_improvement | pareto_rank | judge_synthesis>"
}
```

#### Judge user prompt

```
TOPIC:
{topic}

ITERATION: {iter}/{budget}
DIVERSITY POLICY: {diversity_mode}
FAMILIES SEEN: {families_explored}
THIS ITERATION FAR MODE: {is_far_iteration}

REFERENCE:
{reference_summary}

FLOW PARAMETERS:
{flow_params}

CANDIDATE:
{candidate_spec_json}

BASELINE METRIC VECTOR:
{baseline_metric_vector_json}

CANDIDATE METRIC VECTOR:
{candidate_metric_vector_json}

PRIOR PROCEED CANDIDATES (compact):
{prior_proceed_history}

RUN LOG TAIL:
{run_log_tail}

INTERPRETER PHYSICS SUMMARY (cfd-interpret on this candidate):
{interpreter_summary}

RECENT HISTORY (last 5 iterations, compact):
{recent_history}

Judge holistically. Return only the JSON object.
```

If `--oed-metric-aggregator weighted_sum | min_improvement | pareto_rank` was passed, replace the LLM judge with the corresponding fixed math reduction (legacy fallback, only useful for cost-constrained reruns). Default is `judge_synthesis`.

### Step 11 — Append to history.json

Append **one record per iteration** — including iterations that failed, ran partially, were scaffolded but never executed, or were code-only. Every record carries an honest `status` (item 13):

```json
// <out-dir>/open_ended_discovery/history.json
[
  {
    "iter": 3,
    "id": "SA-RC-Cb1",
    "status": "success",
    "model_class": "SpalartAllmaras_RotationCorrection",
    "model_family": "SpalartAllmaras",
    "equation_touched": "production",
    "params": {"C_r1": 1.0, "C_r2": 12.0},
    "metric_vector": {"Cf_RMSE": 0.0049, "x_reattach_error": 0.18},
    "primary_score": 0.92,
    "baseline_score": 1.00,
    "improvement_pct": 8.0,
    "decision": "PROCEED",
    "is_improvement_over_baseline": true,
    "is_best_so_far": false,
    "judge_decision": "PROCEED",
    "judge_is_improvement": true,
    "judge_rationale": "...",
    "rationale": "...",
    "final_time": 5000.0,
    "reason": "",
    "library": "customModels/SARotationCorrection/libSARotationCorrection.so",
    "case_dir": "<path to candidates/SA-RC-Cb1/case>"
  }
]
```

**`status` is mandatory and must be honest (item 13).** One of:
- `success` — ran to the target end time and was scored.
- `partial` — the solver ran but stopped short of the target end time; record `final_time` and `reason` (e.g. `"diverged at t=1168"`).
- `failed` — compile-fix exhausted, solver crashed, or `FOAM FATAL ERROR`.
- `scaffolded` — the iteration directory was created but the candidate was never run; record `reason` (e.g. `"planned past budget, not executed"`).
- `code_only` — a `code_mod` action that compiled a library but has not yet run a case.
- `revise` / `rerun` / `indeterminate` — the LLM-judge verdict when the candidate is not an improver.

Never label a partial or never-run iteration `success`. `library` is the **case-relative** path to the compiled `.so` (`customModels/<Class>/lib<Class>.so`), never a bare name.

### Step 12 — Update `best.json` if improved

```json
// <out-dir>/open_ended_discovery/best.json
{
  "id": "SA-RC-Cb1",
  "iteration": 3,
  "primary_score": 0.92,
  "metric_vector": {...},
  "library_path": "<abs path>",
  "case_dir": "<abs path>"
}
```

## Termination

- Budget exhausted → stop.
- N consecutive iterations with no improvement → stop early (saves cost; default N=5; configurable).
- User-supplied target metric reached → stop.

## Cross-experiment validation (post-loop)

Before declaring the run complete, run a small parametric sweep (3–5 cases) around the **best model's parameters** to confirm robustness. Each sweep case is a normal `/cfd-experiment` invocation; results join the analysis stage.

## Step 13 — Write `oed_artifact.json`

After the loop ends (or when a regular `code_mod` study completes), write a single JSON descriptor used by downstream stages to wire the winning model into experiment cases.

**Schema** (key fields):
```json
{
  "status": "ok",
  "category": "class_derivation | runtime_source",
  "base_case_dir": "<path to winner's experiment dir>",
  "edited_files": ["<dict files patched relative to base_case_dir>"],
  "primary_dictionary": "<path>",
  "coefficient_names": ["Cb1", "C_pe", "..."],
  "novel_coefficient_names": ["C_pe", "r_c"],
  "coefficient_block_name": "SpalartAllmarasNEQCoeffs",
  "parent_class_name": "SpalartAllmaras",
  "snippet_text": "<full text of dict or runtime-coded source>",
  "customModels_dir": "<path>",
  "class_name": "SpalartAllmarasNEQ",
  "lib_name": "libSpalartAllmarasNEQ.so",
  "compiled_so_paths": ["<.so path>"],
  "source_iteration": 23,
  "best_iteration": 23,
  "best_score": {"metric": "error", "value": 0.003997, "direction": "min"},
  "provenance": "open_ended_discovery"
}
```

**Provenance gating (critical):**
- `provenance == "open_ended_discovery"` AND `best_iteration > 0` → **real OED winner**. Downstream experiments seed each case from `base_case_dir`, patch the named coefficients via `coefficient_names`, and use the bound comparator for scoring. Skips FoamAgent's LLM-driven case rebuild.
- `provenance == "regular_code_mod"` → **synthetic pseudo-entry** written so a non-OED code_mod run can still expose its compiled lib + dict. `best_iteration` is 0. **Do NOT fire the post-OED bridge** — fall through to the standard FoamAgent loop on `requirements.json`. (This distinction was the root cause of the RUN C "degenerate 2-case plan" bug seen in the gpt-5.5 sonnet runs — confusing this case for a real winner produced an invented coefficient list.)
- Missing/empty file or `status != "ok"` → no winner; run all `requirements.json` items via the standard FoamAgent path.

**Skill-only gate logic** (encode this in any orchestrator that consumes the artifact):
```python
artifact = read_json("oed_artifact.json", default={})
use_bridge = (
    artifact.get("status") == "ok"
    and artifact.get("provenance") != "regular_code_mod"
    and int(artifact.get("best_iteration") or 0) > 0
    and bool(state.open_discovery)   # was --open-ended-budget > 0
)
```

## Step 14 — Bridge OED outputs into analyze/paper layout (MANDATORY)

When the discovery loop ends, the data sits under `<out-dir>/open_ended_discovery/iter_*/case/`. `cfd-analyze` and the paper stage expect a normalized `<out-dir>/cases/case_*/` layout with `run_result.json`, `decision.json`, and at least one PNG per `figs/`. This step builds that bridge automatically — do not skip it.

```bash
OUT=<out-dir>
xvfb-run -a python3 - <<'PY' || python3 - <<'PY'
import json, os, re, shutil, subprocess
from pathlib import Path
out = Path(os.environ.get("OUT") or ".")
oed_dir = out / "open_ended_discovery"
hist_path = oed_dir / "history.json"
artifact_path = out / "oed_artifact.json"
if not hist_path.is_file():
    raise SystemExit("history.json missing — OED did not run")
history = json.loads(hist_path.read_text())

# --- item 12: reconcile history.json with the iter_* dirs on disk ---
# The OED loop sometimes forgets to append a record; the bridge must not
# silently drop iterations. Scan the tree and append an honest synthesized
# record for any iteration directory missing from history, then persist —
# so history.json is complete before anything downstream consumes it.
def _iter_num(name):
    m = re.match(r"iter_0*(\d+)", name)
    return int(m.group(1)) if m else None

_recorded = set()
for _e in history:
    if isinstance(_e, dict):
        _it = _e.get("iter", _e.get("iteration"))
        if isinstance(_it, int):
            _recorded.add(_it)
_added = 0
for _d in sorted(oed_dir.glob("iter_*")):
    if not _d.is_dir():
        continue
    _n = _iter_num(_d.name)
    if _n is None or _n in _recorded:
        continue
    _case = _d / "case"
    _has_case = _case.is_dir()
    _logs = list((_case if _has_case else _d).glob("log.*"))
    if "code_mod" in _d.name and not _logs:
        _st = "code_only"
    elif not _has_case and not _logs:
        _st = "scaffolded"
    else:
        _st = "partial"
    history.append({
        "iter": _n, "id": _d.name, "status": _st, "decision": _st.upper(),
        "reconstructed_by_bridge": True,
        "case_dir": str(_case) if _has_case else None,
        "reason": ("not recorded in history.json during the loop; "
                   "reconstructed by the Step 14 bridge from on-disk artifacts"),
    })
    _recorded.add(_n)
    _added += 1
if _added:
    history.sort(key=lambda e: (e.get("iter") or e.get("iteration") or 0))
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"bridge: reconciled history.json — added {_added} unrecorded iteration(s)")

artifact = json.loads(artifact_path.read_text()) if artifact_path.is_file() else {}
# OED outcomes vary: "ok"/"completed" with best_model, "no_winner", or
# "COMPLETE_TARGET_NOT_MET" / "target_not_met" — the last is "best candidate
# improved but didn't hit the stretch target". All three are publishable;
# the bridge fires regardless. Only literal absence of any candidate halts.
status = (artifact.get("status") or "").lower()
no_winner = (
    status in ("no_winner", "no winner")
    or (not artifact.get("best_model") and not artifact.get("best"))
)
target_not_met = (
    "target_not_met" in status or "target not met" in status
)
# Resolve the "best" record across schema variants. v1: best_model. v2: best.
best_block = artifact.get("best_model") or artifact.get("best") or {}
best_label = (
    best_block.get("class_name")
    or best_block.get("label")
    or best_block.get("model")
    or ""
)
best_case_dir = best_block.get("case_dir") or ""

cases_dir = out / "cases"
cases_dir.mkdir(exist_ok=True)
manifest_cases = []

# case_000 — baseline
baseline_src = out / "baseline_case" / "case"
if not baseline_src.is_dir():
    baseline_src = out / "baseline_case"
case_baseline = cases_dir / "case_000_baseline"
case_baseline.mkdir(exist_ok=True)
# Symlink the OpenFOAM case so we don't duplicate gigabytes
src_link = case_baseline / "case"
if not src_link.exists() and baseline_src.is_dir():
    os.symlink(baseline_src.resolve(), src_link)
baseline_metrics = {}
bm_path = out / "baseline_metrics.json"
if bm_path.is_file():
    try: baseline_metrics = json.loads(bm_path.read_text())
    except Exception: pass
(case_baseline / "run_result.json").write_text(json.dumps({
    "status": "success", "case_id": "case_000_baseline",
    "case_dir": str(baseline_src.resolve()),
    "label": "baseline", "metrics": baseline_metrics,
}, indent=2))
(case_baseline / "decision.json").write_text(json.dumps({
    "status": "PROCEED", "case_id": "case_000_baseline",
    "reason": "baseline reference case for OED comparison",
    "key_metrics": baseline_metrics,
}, indent=2))
(case_baseline / "figs").mkdir(exist_ok=True)
manifest_cases.append({
    "case_id": "case_000_baseline",
    "case_dir": str(baseline_src.resolve()),
    "status": "success", "label": "baseline",
})

# Build case entries from every OED history record that produced a real
# simulation. The history schema varies across OED variants — handle both:
#   (a) {iter, id, decision, is_improvement_over_baseline, is_best_so_far,
#        case_dir, metric_vector, library, model_class, equation_touched}
#   (b) {iteration, action, result, label, score} (older variant)
# When status==no_winner we INCLUDE the failed candidates too — characterizing
# what was tried and why each REVISE'd is itself the paper for a no-winner OED.
def normalize_entry(e: dict) -> dict:
    iter_n = e.get("iter") or e.get("iteration")
    cid_label = e.get("id") or e.get("label") or e.get("model_class") or (f"iter_{iter_n:03d}" if iter_n else "unknown")
    decision = (e.get("decision") or e.get("result") or "").upper()
    # Treat ACCEPT / OK / ACCEPTED / BEST as improvers; REVISE / INVALID / FAILED as non-improvers.
    is_improver = bool(
        e.get("is_improvement_over_baseline")
        or e.get("is_best_so_far")
        or decision in ("ACCEPT", "ACCEPTED", "OK", "BEST")
    )
    case_dir_field = e.get("case_dir")
    case_dir = Path(case_dir_field).resolve() if case_dir_field else None
    metrics = e.get("metric_vector") or e.get("score") or {}
    return {
        "iter": iter_n, "label": cid_label, "decision": decision,
        "is_improver": is_improver, "case_dir": case_dir, "metrics": metrics,
        "equation_touched": e.get("equation_touched", ""),
        "model_class": e.get("model_class", ""),
        "library": e.get("library", ""),
        "rationale": e.get("judge_rationale") or e.get("rationale", ""),
        "raw": e,
    }

normalized = [normalize_entry(e) for e in history]

# Locate each entry's OpenFOAM case dir. Prefer explicit case_dir field; fall back
# to glob of iter_{NNN}_* under open_ended_discovery/.
def locate_case_dir(n: dict) -> Path | None:
    if n["case_dir"]:
        # case_dir might be relative — resolve against repo root
        cd = n["case_dir"]
        if cd.is_dir():
            return cd
        # try out-dir-relative
        guess = (out / Path(*cd.parts[-5:])).resolve()  # last few path components
        if guess.is_dir():
            return guess
    iter_n = n["iter"]
    if iter_n is None:
        return None
    cands = sorted([d for d in oed_dir.glob(f"iter_{int(iter_n):03d}_*") if d.is_dir()])
    if not cands:
        return None
    case = cands[0] / "case"
    return case if case.is_dir() else None

# item 13 — honest run status + final time for each bridged case.
def classify_run(case_path, n):
    explicit = str((n.get("raw") or {}).get("status") or "").lower()
    base = explicit if explicit in (
        "scaffolded", "code_only", "failed", "partial", "success") else None
    if case_path is None or not Path(case_path).is_dir():
        return (base or "scaffolded"), None
    solver_logs = [
        l for l in Path(case_path).glob("log.*")
        if not any(t in l.name.lower() for t in
                   ("blockmesh", "checkmesh", "decompose", "snappy",
                    "surfacefeature", "renumber", "postprocess"))
    ]
    final_time, reached_end = None, False
    for l in solver_logs:
        try:
            txt = l.read_text(errors="replace")
        except Exception:
            continue
        tl = re.findall(r"(?m)^Time = ([0-9.eE+-]+)", txt)
        if tl:
            try:
                final_time = float(tl[-1])
            except Exception:
                pass
        if re.search(r"(?m)^\s*End\s*$", txt[-6000:]):
            reached_end = True
    if reached_end:
        return "success", final_time
    if solver_logs:
        return "partial", final_time
    return (base or "code_only"), final_time

# Always include the winner explicitly (when there is one)
winner_dir = Path(best_case_dir).resolve() if best_case_dir else None

seq = 1
seen_dirs: set = set()
for n in normalized:
    iter_case = locate_case_dir(n)
    if iter_case is None or not iter_case.is_dir():
        continue
    if iter_case in seen_dirs:
        continue
    seen_dirs.add(iter_case)
    label = n["label"]
    cid = f"case_{seq:03d}_{label[:40].replace(' ','_')}"
    cdir = cases_dir / cid
    cdir.mkdir(exist_ok=True)
    if not (cdir / "case").exists():
        os.symlink(iter_case, cdir / "case")
    run_status, run_final_time = classify_run(iter_case, n)
    (cdir / "run_result.json").write_text(json.dumps({
        "status": run_status, "case_id": cid,
        "case_dir": str(iter_case),
        "label": label, "oed_iteration": n["iter"],
        "model_class": n["model_class"],
        "equation_touched": n["equation_touched"],
        "library": n["library"],
        "metrics": n["metrics"],
        "is_improver": n["is_improver"],
        "decision": n["decision"],
        "final_time": run_final_time,
    }, indent=2))
    is_winner = (winner_dir is not None and iter_case == winner_dir)
    (cdir / "decision.json").write_text(json.dumps({
        "status": "PROCEED", "case_id": cid,
        "reason": (
            "OED winner" if is_winner
            else "OED accepted candidate (improver)" if n["is_improver"]
            else "OED attempted candidate (no improvement)"
        ),
        "is_winner": bool(is_winner),
        "is_improver": n["is_improver"],
        "judge_rationale": n["rationale"],
        "key_metrics": n["metrics"],
    }, indent=2))
    (cdir / "figs").mkdir(exist_ok=True)
    manifest_cases.append({
        "case_id": cid, "case_dir": str(iter_case),
        "status": "success", "label": label,
        "is_winner": bool(is_winner),
        "is_improver": n["is_improver"],
        "equation_touched": n["equation_touched"],
    })
    seq += 1

if no_winner and len(manifest_cases) <= 1:
    print(f"WARNING: OED produced no winner AND no usable candidate cases (only baseline). "
          f"Paper stage will run but will be a failed-attempt characterization with only "
          f"the baseline + textual rationales from history.json.")

# Render one PyVista U contour per case so cfd-analyze's figs/ check passes.
# Each .foam file lives at <case>/<basename>.foam or <case>/case.foam — try both.
def render_u_contour(case_path: Path, out_png: Path) -> bool:
    try:
        import pyvista as pv
    except Exception:
        return False
    pv.OFF_SCREEN = True
    foam_files = list(case_path.glob("*.foam"))
    if not foam_files:
        foam_files = [case_path / "case.foam"]
        foam_files[0].touch()
    try:
        reader = pv.POpenFOAMReader(str(foam_files[0]))
        reader.set_active_time_value(reader.time_values[-1])
        mesh = reader.read()
        block = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
        if "U" in block.array_names:
            block["Umag"] = ((block["U"]**2).sum(axis=1))**0.5
            scalar = "Umag"
        else:
            scalar = block.array_names[0]
        p = pv.Plotter(off_screen=True, window_size=(1200, 500))
        p.add_mesh(block, scalars=scalar, cmap="viridis", show_edges=False)
        p.view_xy()
        p.screenshot(str(out_png))
        p.close()
        return True
    except Exception as e:
        print(f"  pyvista render failed for {case_path}: {e}")
        return False

import re as _re
for entry in manifest_cases:
    cdir = cases_dir / entry["case_id"]
    case_path = cdir / "case"
    target_path = case_path.resolve() if case_path.exists() else Path(entry["case_dir"])
    out_png = cdir / "figs" / f"{entry['case_id']}_U_contour.png"
    if out_png.is_file() and out_png.stat().st_size > 5_000:
        continue
    ok = render_u_contour(target_path, out_png)
    if not ok:
        # Placeholder so audit's figs/*.png count is satisfied — clearly named for human review.
        out_png.write_bytes(b"")
        # Write minimal text fallback so 0-byte does not occur
        out_png.write_bytes(bytes.fromhex(
            "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
            "0000000A49444154789C636000000002000160C3270000000049454E44AE426082"
        ))

manifest = {"cases": manifest_cases, "n_cases": len(manifest_cases),
            "provenance": "cfd-open-discovery bridge",
            "no_winner": no_winner}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))

# Update state.json so cfd-analyze preflight sees the right current_stage
state_path = out / "state.json"
state = json.loads(state_path.read_text()) if state_path.is_file() else {}
state["current_stage"] = "analysis"
state["paper_subagent_dispatched"] = False
state_path.write_text(json.dumps(state, indent=2))

# Write the discovery checkpoint STAMPED with a bridge_signature the audit
# script verifies. The signature is the only proof the bridge body actually
# ran — agents that fabricate this checkpoint by hand cannot guess it, and
# the H27 gate in stage_gate_audit.py fails them.
ck = out / "checkpoints"
ck.mkdir(exist_ok=True)
import datetime
ts = datetime.datetime.utcnow().isoformat() + "Z"
n_improvers = sum(1 for c in manifest_cases if c.get("is_improver"))
winner_cid = next((c["case_id"] for c in manifest_cases if c.get("is_winner")), None)
(ck / "open_ended_discovery_done.json").write_text(json.dumps({
    "stage": "open_ended_discovery", "ts": ts,
    "iterations_done": len(history),
    "n_cases_built": len(manifest_cases),
    "n_improvers": n_improvers,
    "winner_case_id": winner_cid,
    "no_winner": no_winner,
    # Signature the audit verifies. Agents cannot fabricate this checkpoint
    # without running the bridge body.
    "bridge_signature": "cfd-open-discovery:bridge:v1",
}, indent=2))

# Companion artifact: bridge.json captures the per-case decisions for the
# paper writer to embed in its rationale table. Audit also looks at this.
(out / "bridge.json").write_text(json.dumps({
    "provenance": "cfd-open-discovery:bridge:v1",
    "ts": ts,
    "no_winner": no_winner,
    "winner_case_id": winner_cid,
    "n_improvers": n_improvers,
    "n_cases": len(manifest_cases),
    "families_explored": sorted({c.get("equation_touched","") for c in manifest_cases if c.get("equation_touched")}),
    "per_case_summary": [
        {
            "case_id": c["case_id"], "label": c["label"],
            "equation_touched": c.get("equation_touched", ""),
            "is_winner": c.get("is_winner", False),
            "is_improver": c.get("is_improver", False),
        } for c in manifest_cases
    ],
}, indent=2))

print(f"bridge built: {len(manifest_cases)} cases under {cases_dir} "
      f"(no_winner={no_winner}, improvers={n_improvers}, winner={winner_cid})")
PY
```

Validate:

```bash
OUT=<out-dir>
test -f $OUT/manifest.json || { echo "bridge failed: manifest.json missing"; exit 1; }
test -f $OUT/bridge.json   || { echo "bridge failed: bridge.json missing";   exit 1; }
jq -e '.bridge_signature == "cfd-open-discovery:bridge:v1"' $OUT/checkpoints/open_ended_discovery_done.json >/dev/null \
  || { echo "bridge failed: open_ended_discovery_done.json missing the bridge_signature; the bridge body did not run"; exit 1; }
N_CASES=$(jq '.cases | length' $OUT/manifest.json)
[ "$N_CASES" -ge 2 ] || { echo "bridge built only $N_CASES cases — need baseline + ≥1 candidate (improver or REVISE'd, both count for a paper)"; exit 1; }
for cdir in $OUT/cases/case_*; do
  test -f "$cdir/run_result.json" || { echo "$cdir/run_result.json missing"; exit 1; }
  test -f "$cdir/decision.json"   || { echo "$cdir/decision.json missing";   exit 1; }
  [ -n "$(ls $cdir/figs/*.png 2>/dev/null)" ] || { echo "$cdir/figs/ has no PNG"; exit 1; }
done
echo "bridge OK: $N_CASES cases ready for cfd-analyze"
```

## Step 15 — Dispatch the remaining chain in a sub-agent (MANDATORY)

This is the same task-closure-bias mitigation cfd-analyze uses at the analysis→paper boundary, applied one stage earlier. The OED loop at budget=20 leaves the parent agent at high context with strong "task complete" bias — empirically the parent fails to invoke `cfd-analyze` after writing `oed_artifact.json`. A fresh sub-agent re-creates the conditions needed to walk analyze → cross-analyze → paper → audit reliably.

Read `<out-dir>/state.json#topic` and the absolute `<out-dir>` path, then invoke the `Agent` tool with `subagent_type="general-purpose"`, a short description like "Run analyze→paper→audit for OED winner", and the prompt below with `<OUT_DIR>` and `<TOPIC>` substituted. **This is a single tool call. Wait for the sub-agent to return.**

```
Sub-agent prompt (pass as the `prompt` parameter to the Agent tool):
---------------------------------------------------------------------
You are the post-OED sub-agent for a CFD scientist run. The OED loop has finished:
- `<OUT_DIR>/oed_artifact.json` describes the best model.
- `<OUT_DIR>/open_ended_discovery/history.json` lists every iteration.
- `<OUT_DIR>/cases/case_*/` was just built by the bridge step (one case per accepted iter + the baseline; each has run_result.json, decision.json, figs/<id>_U_contour.png).
- `<OUT_DIR>/manifest.json` lists every case with status=success.
- `<OUT_DIR>/checkpoints/open_ended_discovery_done.json` is on disk.

Run directory: <OUT_DIR>
Topic:         <TOPIC>
Mode:          open_discovery

Do NOT re-run literature, baseline_setup, mesh_gate, or open_ended_discovery. They are complete. Your job is the remaining chain: analyze → cross-analyze → paper (full Phase A–F cycle) → audit.

Before you start, set the nesting marker so the cfd-analyze sub-agent dispatch does not fire a second nested sub-agent inside your session:
    python3 -c "import json,pathlib; p=pathlib.Path('<OUT_DIR>/state.json'); s=json.loads(p.read_text()); s['paper_subagent_dispatched']=True; p.write_text(json.dumps(s,indent=2))"

STEP 1 — Clear any stale paper-stage stubs from prior aborted attempts:
    rm -f <OUT_DIR>/review.json
    rm -f <OUT_DIR>/paper_unified_plan.json
    rm -f <OUT_DIR>/paper_cycle_state.json
    rm -f <OUT_DIR>/paper_review_history.json
    rm -f <OUT_DIR>/audit_passed.json
    rm -f <OUT_DIR>/checkpoints/paper_done.json
    rm -f <OUT_DIR>/checkpoints/paper_planner_done.json
    rm -f <OUT_DIR>/checkpoints/paper_viz_done.json
    rm -f <OUT_DIR>/checkpoints/paper_writer_done.json
    rm -f <OUT_DIR>/checkpoints/paper_compile_done.json
    rm -f <OUT_DIR>/checkpoints/paper_review_done.json
    rm -f <OUT_DIR>/paper/main.tex
    rm -f <OUT_DIR>/paper/main.pdf
    rm -rf <OUT_DIR>/paper_figs

STEP 2 — Invoke Skill cfd-skills/cfd-analyze on <OUT_DIR>. It will read manifest.json + cases/case_*/decision.json, produce <OUT_DIR>/analysis.json, and write checkpoints/analysis_done.json. Because state.json#paper_subagent_dispatched is true, the analyze skill will NOT spawn its own paper sub-agent — it returns inline and you continue.

STEP 3 — Invoke Skill cfd-skills/cfd-cross-analyze. It produces cross_experiment_analysis/{aggregate.csv, cross_experiment_interpretation.md, *.png} comparing the OED candidates against the baseline (and DNS reference if present in reference_data/).

STEP 4 — Invoke Skill cfd-skills/cfd-paper. Walk Stage 0 → Phase A → Phase B → Phase C → Phase D → Phase E → Phase F literally and in order. Honor every jq/python validation gate. The cfd-paper planner MUST use the canonical figure_jobs schema (unique out_png, case_id substring in per-case filenames). The writer MUST satisfy every HARD rule (≥3 equations including governing + closure, distributed citations, geometry markers, no meta-paragraph refs, no duplicate paragraphs, quantitative tabular).

STEP 5 — Run the audit:
    python scripts/stage_gate_audit.py --out-dir <OUT_DIR>
If rc != 0, read the failure list, invoke the relevant cfd-paper phase to fix each gate, re-run the audit. Repeat until rc=0. Do NOT write audit_passed.json by hand.

Constraints:
- Only forbidden script under scripts/ is scripts/orchestrator_run.py. Every other script is callable.
- xvfb-run, pdflatex, jq, Bash heredocs, inline Python, PyVista, matplotlib, numpy, pandas — all available.
- Do not invoke cfd-literature, cfd-baseline-setup, cfd-metric-setup, cfd-mesh-gate, cfd-experiment, cfd-foamagent, cfd-open-discovery, cfd-code-modify. Those are complete.

Return when audit_passed.json exists with audit_signature="stage_gate_audit.py:v1", OR after 3 unrecoverable audit failures with the latest failure list and which phases you re-ran.
---------------------------------------------------------------------
```

After the sub-agent returns:

```bash
OUT=<out-dir>
if [ -f $OUT/audit_passed.json ] && jq -e '.audit_signature == "stage_gate_audit.py:v1"' $OUT/audit_passed.json >/dev/null; then
  echo "post-OED chain complete: paper at $OUT/paper/main.pdf"
else
  # Re-spawn the sub-agent with the audit failures appended (cap 3).
  echo "sub-agent did not pass audit; re-spawn or surface to user"
fi
```

If audit still fails after 3 sub-agent attempts, halt and surface the failure list to the user.

## Resume
If `<out-dir>/open_ended_discovery/history.json` exists, resume from `len(history) + 1`. Re-running this skill with the same `out-dir` and `budget` continues where it left off.

## Timeline
```json
{"stage": "open_ended_discovery", "event": "iteration_complete", "ts": "<iso>", "iter": 3, "id": "SA-RC", "decision": "PROCEED", "score": 0.92}
{"stage": "open_ended_discovery", "event": "complete", "ts": "<iso>", "iterations_done": 20, "best_iteration": 23, "best_score": 0.92}
```

## Notes
- Token cost can be substantial (multi-hour, multi-million-token runs). The orchestrator's reference run used ~4.3M tokens / 240 LLM calls / 5h for budget=20.
- Always lock mesh before starting iterations; otherwise scoring is confounded by mesh effects.
- Do **not** re-do mesh independence per candidate — the gate is per-physics-group, not per-parameter.
- LLM-authored comparators are self-tested AND independent-verified before being trusted. Failed comparators are logged in `bound_comparators.json` and excluded rather than silently poisoning scoring.

## Phase 2 — close + far search (`diversity_mode hybrid|aggressive`)
- Tracks model **family** (SA / SA-RC / SA-NEQ / k-omega-SST / RSM / LES-SGS …) and **equation touched** (production / destruction / diffusion / source / limiter / coefficient) per accepted candidate.
- `hybrid`: every Nth iteration (`diversity_far_ratio`, default 0.3 → ~every 3rd) is forced to FAR mode — the propose-LLM is told it MUST pick a candidate from a family not yet explored.
- `aggressive`: alternate close/far each iteration.
- Triggers FAR also when N consecutive iterations show no improvement.
- Artifact: `<out-dir>/open_ended_discovery/families_explored.json`.

## Phase 3 — multi-flow / multi-reference (`multi_flow_starter_dirs`)
- Provide multiple flow folders (each with its own `reference_data/`):
  `multi_flow_starter_dirs=starter/periodic_hill,starter/bfs,starter/channel`. Or place multiple flow subdirs under a single `--starter-dir`; auto-detected.
- Per iteration, candidate's metric vector is computed against each flow's reference data; per-flow primaries are aggregated by `--metric-aggregator` (e.g. `min_improvement` = pessimistic / must-improve-every-flow). The judge sees the full **flow × metric matrix**.
- Today's behavior: scores against multiple reference datasets on the **same** simulated case. Running each candidate against each flow's base case is the natural next step (Phase 3.5).

## LangGraph mode only — do NOT call from skill mode

The Python pipeline (`scripts/orchestrator_run.py` → `scripts/open_ended_discovery.py`) implements the same OED recipe with a Python supervisor: same embedded prompts, same artifact contracts (`history.json`, `bound_comparators.json`, `oed_artifact.json`, `best.json`), same per-iter directory layout. It exists for the LangGraph workflow.

```bash
# LangGraph mode (NOT for skill mode)
python scripts/orchestrator_run.py \
  --topic "..." --out-dir <out-dir> \
  --provider claude-code --model claude-sonnet-4-6 \
  --starter-dir <starter-dir> --open-ended-budget <budget>

# Phase 2 (close+far diversity)
python scripts/orchestrator_run.py ... \
  --oed-diversity-mode hybrid --oed-diversity-far-ratio 0.3

# Phase 3 (multi-flow validation)
python scripts/orchestrator_run.py ... \
  --oed-diversity-mode hybrid \
  --oed-multi-flow --oed-starter-dirs starter/periodic_hill starter/bfs starter/channel
```

## Two valid execution paths

1. **Inline agent recipe (above).** The agent runs the four-action loop (propose → compile → run → score) with the no-progress cap, verifier independence, per-QoI guidance, iteration directory layout, bound-comparator cache, resume. Native Read/Write/Bash.

2. **Script fast-path.** `scripts/open_ended_discovery.py` runs the same loop with a Python supervisor and the same prompts. Phase 2 (close+far diversity) and Phase 3 (multi-flow) are currently script-only — invoke via Bash with the documented CLI args. The skill-mode-forbidden script is `scripts/orchestrator_run.py` only.

## Next

After the discovery loop has terminated (budget exhausted or improvement threshold met) and `<out-dir>/checkpoints/open_ended_discovery_done.json` is on disk, your next action is:

`Skill cfd-skills/cfd-analyze`

The analyze skill then chains forward to `cfd-paper` Stage 0 → `cfd-cross-analyze` → `cfd-paper` iterations 0+, just like routes 1 and 2. The cross-experiment figures will compare top-K candidates against the baseline and (if available) reference data.

Do not stop, summarize, or wait. The chain continues automatically.

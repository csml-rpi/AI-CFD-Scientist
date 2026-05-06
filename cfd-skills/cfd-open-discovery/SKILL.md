---
name: cfd-open-discovery
description: Open-ended discovery loop — propose novel CFD model modifications (e.g. SA / k-omega variants, viscosity laws), implement via cfd-code-modify, run via cfd-experiment, score against baseline + reference, repeat until budget. Self-contained — embeds MetricProposer, ComparatorAuthor (text + PyVista), ComparatorVerifier, and the LLM-judge prompt verbatim. Use for topics with "open-ended", "discover", "find a novel", "beat baseline", "best model".
---

# cfd-open-discovery

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

## Outputs
- `<out-dir>/open_ended_discovery/history.json` — iteration log
- `<out-dir>/open_ended_discovery/candidates/<id>/` — per-candidate code + run output
- `<out-dir>/open_ended_discovery/best.json` — best model found, params, score
- `<out-dir>/baseline_metrics.json` — baseline scores
- `<out-dir>/oed_artifact.json` — handoff descriptor (see schema below)

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

```json
// <out-dir>/open_ended_discovery/history.json
[
  {
    "iter": 3,
    "id": "SA-RC-Cb1",
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
    "library": "libSpalartAllmarasRotationCorrection.so",
    "case_dir": "<path to candidates/SA-RC-Cb1/case>"
  }
]
```

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

## Optional script fast-path

```bash
python scripts/orchestrator_run.py \
  --topic "..." --out-dir <out-dir> \
  --provider claude-code --model claude-sonnet-4-6 \
  --starter-dir <starter-dir> --open-ended-budget <budget>

# Add close+far diversity (search policy choice — explicit flag)
python scripts/orchestrator_run.py ... \
  --oed-diversity-mode hybrid --oed-diversity-far-ratio 0.3

# Add multi-flow validation
python scripts/orchestrator_run.py ... \
  --oed-diversity-mode hybrid \
  --oed-multi-flow --oed-starter-dirs starter/periodic_hill starter/bfs starter/channel
```

The orchestrator's OED loop (`scripts/open_ended_discovery.py`) implements the same recipe — same prompts, same artifacts, same gating. Phase 1/2/3 flags are currently Python-orchestrator-only; the skill-only path runs the loop in markdown using the prompts above.

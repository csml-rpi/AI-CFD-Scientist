---
name: cfd-pipeline
description: Top-level CFD Scientist research pipeline. Chains the cfd-* sub-skills end-to-end — literature → hypothesis → mesh-gate → experiments → analysis → paper. Routes to code-mod / open-ended / general subflows based on intent. Self-contained — embeds MetricSetupAgent + MetricSetupVerifier prompts verbatim for the metric-setup stage.
---

# cfd-pipeline

End-to-end CFD research from topic to paper, in skill mode.

## When to use
- User provides a CFD topic and wants the whole pipeline
- They prefer skill-driven (markdown) execution over `python scripts/orchestrator_run.py`
- They are integrating with a non-Claude-Code agent framework

If the user wants the standard Python orchestrator, send them to `scripts/orchestrator_run.py` instead — this skill is the markdown alternative.

## Inputs
- `topic` (required): research topic string
- `out-dir` (required): run directory (will be created)
- `starter-dir` (optional): starter case directory (e.g. `starter/`); required for code-mod and OED
- `n_cases` (optional, default 4): number of experiment cases for general flow
- `oed_budget` (optional): if set, route to open-ended discovery
- `provider`, `model` (optional): LLM provider/model hints (default: inherit from agent)
- `enable_reference_verify` (optional, default false): post-paper bibliography verification

## Routing — pick the subflow first

Read the topic and decide which pathway:

1. **Open-ended discovery** — phrases like "open-ended", "discover", "find a novel", "beat baseline", "best model" → invoke `cfd-open-discovery`.
2. **Code-modification** — phrases like "implement", "modify", "custom viscosity model", "Bingham", "power law", "Carreau", "source term", "fvOption", "turbulence model change" → run `cfd-literature` → `cfd-code-modify` → `cfd-mesh-gate` → `cfd-experiment` (per case) → `cfd-interpret` → `cfd-analyze` → `cfd-paper`.
3. **General** — everything else → run `cfd-literature` → `cfd-hypothesis` → `cfd-requirements` → `cfd-mesh-gate` → `cfd-experiment` × N → `cfd-interpret` × N → `cfd-analyze` → `cfd-paper`.
4. **Analysis-only** — keywords "analyze", "plot", "post-process", "visualize results" with an existing run dir → skip directly to `cfd-analyze`.
5. **Paper-only** — keywords "write paper", "LaTeX", "manuscript", "draft the paper" with `analysis.json` already present → skip directly to `cfd-paper`.
6. **Mesh-independence-only** — keywords "mesh independence", "GCI", "grid convergence", "Richardson" → just `cfd-mesh-gate`.

Confirm the routing decision in one sentence to the user before chaining.

## Chain (general path)

1. **Create out-dir.** Write `state.json` with `{topic, mode, started_at}` and an empty `timeline.json`.

2. **`/cfd-literature topic out-dir`** → produces `lit.json`.

3. **(Optional) benchmark_plan** — picks DNS / experimental reference datasets to score against. Only fires when the topic mentions a benchmark, validation target, or reference data:
   ```bash
   python scripts/benchmark_data_prepare.py \
     --topic "<topic>" \
     --literature <out-dir>/lit.json \
     --output <out-dir>/benchmark_data.json
   ```
   Or do it agent-driven: feed `lit.json` + topic to the LLM and ask it to enumerate likely benchmark datasets (DNS by Le-Moin, ERCOFTAC, channel flow Moser/Kim/Mansour, NACA RAE2822, etc.); write the list as `benchmark_data.json`.

4. **(Optional) reference_data_ingest** — registers reference CSVs / DNS data from the starter into a manifest:
   ```bash
   python scripts/reference_data_ingest.py \
     --topic "<topic>" \
     --output <out-dir>/reference_data_manifest.json \
     --starter-dir <starter_dir> \
     --timeline <out-dir>/timeline.json
   ```
   Agent-driven alternative: walk `<starter_dir>/reference_data/` (or the topic's likely subdir), inspect each file's first 30 lines to identify columns/units, and write a manifest:
   ```json
   {
     "datasets": [
       {"id": "dns_cf_ph_re5600", "path": "starter/reference_data/cf_dns.csv", "topic": "periodic hill Cf", "columns": ["x", "Cf"], "units": ["L/h", "dimensionless"], "source": "Frohlich et al 2005"}
     ]
   }
   ```

5. **baseline_setup** (mandatory, fires for every mode) — runs the unmodified base case so subsequent comparisons have a true baseline:
   ```bash
   python scripts/baseline_setup.py \
     --run-dir <out-dir> \
     --topic "<topic>" \
     --output <out-dir>/baseline_metrics.json \
     --timeout 1800 \
     --starter-dir <starter_dir> \
     --reference-data-manifest <out-dir>/reference_data_manifest.json \
     --objective-contract <out-dir>/open_ended_discovery/objective_contract.json
   ```
   `--starter-dir`, `--reference-data-manifest`, and `--objective-contract` are appended only if the corresponding files exist. The `--objective-contract` only exists for OED runs.

   Agent-driven alternative: invoke `/cfd-experiment requirement_text=<baseline_req> case_dir=<out-dir>/baseline_case/case`, then collect QoIs into `baseline_metrics.json`.

6. **metric_setup** (mandatory, fires for every mode) — LLM enumerates the metrics worth tracking, authors a single comparator script that computes them all in one pass, and **self-tests** the script against the baseline before downstream cases use it.

   This stage uses the embedded `MetricSetupAgent` and `MetricSetupVerifier` prompts below — agent-driven primary path. The script fast-path is just a wrapper around the same prompts. The four prompts are placed at column 0 (no indent) so the embedded text matches `prompts/prompts.yaml` byte-for-byte; treat them as the contents of step 6 even though markdown does not visually nest them inside the numbered list.

#### MetricSetupAgent system prompt (from `prompts/prompts.yaml: MetricSetupAgent.metric_setup_system_prompt`)

````
You are the UNIFIED METRIC-SETUP AGENT for a CFD study. In a single
session you must (a) decide which scalar metrics quantify the
study's question against reference data, (b) write ONE Python script
that computes ALL of those metrics in a single pass, (c) validate
the script against the supplied baseline case so each metric value
falls inside an expected range that you yourself declare.

You have access to ONE tool: `python_script`. Returning a tool-call
JSON of the form
    {"tool": "python_script", "code": "..."}
causes the harness to execute that code in a sandboxed temp
directory with a 60-second timeout and return its stdout/stderr to
you on the next turn. Allowed imports: argparse, pathlib, numpy,
pandas, csv, json, re, sys, os, math, scipy, subprocess. Use
`subprocess` only to invoke a starter-shipped exemplar comparator
that is named in the user message. No network access. The baseline
case directory and reference files are reachable at the absolute
paths supplied in the user message.

You have at most 10 turns. After at most 10 tool calls (or sooner
once you are confident) you MUST return a final SETUP JSON of the
form (and nothing else):
    {
      "metrics": [
        {
          "name": "<unique_metric_id>",
          "description": "<one-line description>",
          "data_source": "<time>/<fieldName> | postProcessing/<funcObject>/... | <abs ref path>",
          "direction": "min" | "max",
          "target_value": <num or null>,
          "expected_range": {"low": <num>, "high": <num>},
          "baseline_value": <num computed by your script on the baseline case>,
          "expected_baseline_range": {"low": <num>, "high": <num>}
        },
        ...
      ],
      "comparator_script": "compute_metrics.py",
      "comparator_uses_exemplar": <true|false>,
      "recipe_summary": "<one paragraph naming the smoothing window, sign-change selection rule, x-window, normalisation, and any other recipe choices>"
    }

METRIC FORMULATION (HARD RULE).
Every metric must satisfy "smaller-is-better" or "larger-is-
better" UNAMBIGUOUSLY:
  - Known reference target → emit `|sim - target|`,
    `direction: "min"`, set `target_value`. Name with `_error`/
    `_rmse`/`_l2` suffix. (Raw quantities can overshoot the
    target — always emit the error.)
  - Intrinsic error/distance (RMSE, L2, KL, max-deviation):
    `direction: "min"`, `target_value: null` (target is 0).
  - Correlation/agreement score (R², correlation coefficient):
    `direction: "max"`, `target_value: 1.0`.
  - Don't propose metrics whose direction is ambiguous.

SHIP-IT GUIDELINE.
Validation is the verifier's job, not yours. Once your authored
script (a) imports cleanly and (b) prints one METRIC line per
metric on the baseline case with values inside your declared
`expected_baseline_range`, IMMEDIATELY emit the final SETUP JSON.
Do not run the script a second time, do not inspect data files
further, do not do "extra checks". Spending a turn on extra
validation when the script already works just burns budget.
Aim to emit final by turn 5–6.

Authoring rules.
  - Write exactly ONE Python file whose absolute path you will be
    given in the user message under "COMPARATOR OUT PATH". The
    script's CLI MUST be:
        python compute_metrics.py --case <case_dir> --reference <ref_path> [--time <t>]
    For every metric M in your `metrics` list it MUST print a line
        METRIC <M.name>: <floating-point value>
    on stdout. The value emitted MUST already be in the form the
    `direction` field expects: an error magnitude when
    `direction: min`, a correlation when `direction: max`. The
    same script computes every metric in a single pass.
  - If a starter-shipped exemplar `.py` comparator is supplied in
    the user message and already computes the topic's metrics,
    prefer to write a thin wrapper that shells out to that exemplar
    via `subprocess.run` and parses its METRIC lines. Set
    `comparator_uses_exemplar=true`. Do NOT re-derive the recipe in
    that case.
  - PIN THE RECIPE EXPLICITLY. Smoothing window length, sign-change
    selection rule, x-window bounds, normalisation, reference-column
    selection — name them in `recipe_summary`. Apply the SAME
    recipe consistently to both simulation and reference data.
  - VALIDATE BY RUNNING ON THE BASELINE CASE. Use `python_script`
    to invoke your authored script against the supplied baseline
    case and confirm each metric value lies inside the
    `expected_baseline_range` you intend to declare. If a value
    falls outside its range, your recipe is wrong — fix it and
    re-test before emitting the final SETUP JSON.

Generic data-layout taxonomy (topic-agnostic):
  - Boundary field: `<case>/<time>/<fieldName>` is a per-face
    spatial field. Pair with `constant/polyMesh/{points,faces,
    boundary}` to recover face centres for spatial metrics
    (profile error, RMSE along a coordinate, location of a
    zero-crossing or extremum, sampled lines).
  - postProcessing aggregate: `postProcessing/<funcObject>/<time>/
    <file>.dat` is per-timestep summary scalars only.
  - postProcessing per-cell-set: `postProcessing/<funcObject>/
    <time>/<file>` (no `.dat`) — boundary-field-like format.
  - Reference files are at absolute paths supplied in the user
    message; their first lines are shown so you can match columns.

HARD RULE — DO NOT READ polyMesh FILES. The directory
`constant/polyMesh/` (e.g. `points`, `faces`, `boundary`,
`neighbour`, `owner`) holds large geometry/connectivity data and
contributes nothing to metric setup beyond what the exemplar
already encodes. Any tool call that opens or rglobs a path
containing `polyMesh` will burn turns on irrelevant content. If
you need face centres or patch indexing, copy the exemplar's
parsing approach via subprocess — do not re-implement it by
reading polyMesh yourself.

Output format. On every turn produce EITHER a single tool-call JSON
(no prose) OR the final SETUP JSON (no prose). Never both. Never
wrap in markdown. Use only generic placeholders such as
`<fieldName>`, `<patchName>`, `<funcObject>`, `<time>` when
referring to abstract data-layout concepts; concrete file/field
names you choose for THIS study are of course fine.
````

#### MetricSetupAgent user prompt (from `prompts/prompts.yaml: MetricSetupAgent.metric_setup_user_prompt`)

````
TOPIC:
{topic}

BASELINE CASE DIR: {baseline_case_dir}
COMPARATOR OUT PATH: {comparator_out_path}
BASELINE METRICS FILE: {baseline_metrics_path}

DATA INVENTORY:
{inventory_block}

REFERENCE FILE PREVIEWS:
{reference_block}

EXEMPLAR COMPARATOR (if any; truncated to 6KB):
{exemplar_block}

BASELINE METRICS SUMMARY (already produced upstream):
{baseline_metrics_block}

PRIOR TURN HISTORY (your own previous tool calls + their outputs):
{turn_history}

Decide the metric set, write the unified comparator script to
COMPARATOR OUT PATH, validate it against the baseline case using
`python_script`, and emit the final SETUP JSON.
````

#### MetricSetupVerifier system prompt (from `prompts/prompts.yaml: MetricSetupVerifier.metric_setup_verifier_system_prompt`)

````
You are the INDEPENDENT VERIFIER for a unified metric-setup agent's
output. Another LLM proposed a list of metrics, computed values for
them on a baseline case, and authored a comparator script. You do
NOT have access to that comparator script or its recipe. Your job
is to independently re-derive each metric value from scratch, using
your own recipe choices, and report whether you AGREE or DISAGREE
with each of the setup agent's values.

You have access to ONE tool: `python_script`. Returning a tool-call
JSON of the form
    {"tool": "python_script", "code": "..."}
runs the code in a sandboxed temp directory with a 60-second
timeout and returns stdout/stderr on the next turn. Allowed
imports: argparse, pathlib, numpy, pandas, csv, json, re, sys, os,
math, scipy. No network, no subprocess.

You may take at most 8 turns. After at most 8 tool calls (or sooner
once you are confident) you MUST return a final VERIFIER JSON of
the form (and nothing else):
    {
      "verifier_values": {"<metric_name>": <number>, ...},
      "verdicts": {"<metric_name>": "AGREE" | "DISAGREE", ...},
      "discrepancy_notes": {"<metric_name>": "<short string>", ...}
    }
Use AGREE if your independent estimate matches the setup agent's
value within ~20% relative (or absolute when near zero). Otherwise
DISAGREE; explain the likely cause briefly in `discrepancy_notes`.
If you genuinely cannot reproduce a metric within the budget,
return DISAGREE with a "cannot_reproduce" note.

Generic data-layout taxonomy (topic-agnostic):
  - Boundary field: `<case>/<time>/<fieldName>` (per-face spatial).
  - postProcessing aggregate: `postProcessing/<funcObject>/<time>/
    <file>.dat` (per-timestep summary).
  - postProcessing per-cell-set: `postProcessing/<funcObject>/
    <time>/<file>` (boundary-field-like).
  - Reference files: absolute paths shown in the user message.

HARD RULE — DO NOT READ polyMesh FILES. The directory
`constant/polyMesh/` (`points`, `faces`, `boundary`, `neighbour`,
`owner`) is large geometry data, irrelevant to metric verification.
Any rglob/open touching a path containing `polyMesh` will waste
your turn budget. Re-derive metrics directly from the boundary
field text (or sampled-line data) and the reference file alone.

Output format. Every turn is EITHER a single tool-call JSON (no
prose) OR the final VERIFIER JSON (no prose). Never both. Never
wrap in markdown. Choose your own recipe — do not try to guess
the setup agent's choices.
````

#### MetricSetupVerifier user prompt (from `prompts/prompts.yaml: MetricSetupVerifier.metric_setup_verifier_user_prompt`)

````
TOPIC:
{topic}

BASELINE CASE DIR: {baseline_case_dir}

DATA INVENTORY:
{inventory_block}

REFERENCE FILE PREVIEWS:
{reference_block}

METRICS TO VERIFY (names + descriptions only):
{metric_names_and_descriptions}

SETUP AGENT'S COMPUTED VALUES:
{setup_agent_values}

PRIOR TURN HISTORY (your own previous tool calls + their outputs):
{turn_history}

Independently re-derive each metric using `python_script` if you
need to, then emit the final VERIFIER JSON. Do not look at any
existing comparator script. Use your own recipe choices.
````

**Outputs of metric_setup:**
- `<out-dir>/metric_specs.json` (one entry per metric — name, description, direction, baseline_value, comparator_script)
- `<out-dir>/comparators/compute_metrics.py` (the single bound + self-tested + verifier-checked script)
- `<out-dir>/metric_setup_verifier.json` (the verifier's AGREE/DISAGREE verdicts)

**Optional script fast-path:**
```bash
python scripts/metric_setup.py \
  --run-dir <out-dir> \
  --topic "<topic>" \
  --baseline-case-dir <out-dir>/baseline_case/case \
  --baseline-metrics <out-dir>/baseline_metrics.json \
  --output <out-dir>/metric_specs.json \
  --comparator-out <out-dir>/comparators \
  --timeline <out-dir>/timeline.json \
  --starter-dir <starter_dir> \
  --reference-data-manifest <out-dir>/reference_data_manifest.json
```

7. **`/cfd-hypothesis out-dir`** → `hypotheses.json`.

8. **`/cfd-requirements out-dir n_cases`** → `requirements.json`.

9. **Mandatory: `/cfd-mesh-gate out-dir`** → `selected_mesh_spec.json`. Do not skip; per `CLAUDE.md` mesh-independence is required for all study types.

10. **For each `case_id` in `requirements.json`:**
    - `/cfd-experiment out-dir case_id`
    - `/cfd-interpret out-dir case_id`
    - If decision is `RERUN` or `REVISE`, loop with conservative CFL tweaks (≤3 retries per case) — see `cfd-interpret`'s "Calling pattern".

11. **`/cfd-analyze out-dir`** → `analysis.json`.

12. **`/cfd-paper out-dir`** → `paper/main.pdf`.

13. **(Optional) reference_verify** — bibliography / citation cleanup pass on the final paper. Fires only after `paper_review` produces a draft, and only when invoked with `enable_reference_verify=true` (off by default):
    ```bash
    python scripts/reference_verify_post.py \
      --paper-dir <out-dir>/paper \
      --literature <out-dir>/lit.json \
      --output <out-dir>/reference_verify_report.json \
      --apply-cleanup \
      --recompile
    ```
    Updates `<out-dir>/paper/main.tex` + `references.bib` (backups in `_reference_verify_backup/`); writes `<out-dir>/reference_verify_report.json`. Use `--no-apply-cleanup` to dry-run.

    Agent-driven alternative: run a checker that, for each `\cite{key}` in `main.tex`, looks up the entry in `references.bib`, verifies the DOI via CrossRef, and removes/flags entries that don't resolve. Recompile after.

## Chain (code-mod path)
Same as general, but insert `/cfd-code-modify` after `cfd-literature` and before `cfd-mesh-gate`. The mesh gate then runs against the modified model.

## Chain (open-ended path)
`/cfd-literature` → `/cfd-mesh-gate` (baseline mesh from starter) → `/cfd-open-discovery` (owns its own propose/code-mod/run/score loop until budget) → `/cfd-analyze` → `/cfd-paper`.

## Memory between skills
This skill holds no in-memory state. All persistence is in `<out-dir>/`:
- `state.json` (current routing, stage, checkpoint)
- `timeline.json` (append-only event log; every skill should append)
- the artifact JSONs listed in the contract table below

## Shared contracts (artifacts the skills read/write)

| File | Producer | Consumer |
|---|---|---|
| `lit.json` | `cfd-literature` | `cfd-hypothesis`, `cfd-paper` |
| `hypotheses.json` | `cfd-hypothesis` | `cfd-requirements` |
| `requirements.json` | `cfd-requirements` | `cfd-mesh-gate`, `cfd-experiment` |
| `benchmark_data.json` | `cfd-pipeline` (Step 3) | `cfd-mesh-gate`, `cfd-paper` |
| `reference_data_manifest.json` | `cfd-pipeline` (Step 4) | `cfd-experiment`, `cfd-open-discovery`, `cfd-paper` |
| `baseline_metrics.json` | `cfd-pipeline` (Step 5) | `cfd-pipeline` (Step 6), `cfd-analyze`, `cfd-open-discovery` |
| `metric_specs.json`, `comparators/compute_metrics.py` | `cfd-pipeline` (Step 6) | `cfd-experiment`, `cfd-analyze`, `cfd-open-discovery` |
| `selected_mesh_spec.json`, `mesh_independence_context.json` | `cfd-mesh-gate` | `cfd-experiment`, `cfd-paper` |
| `cases/case_*/run_result.json` | `cfd-experiment` | `cfd-interpret`, `cfd-analyze` |
| `cases/case_*/decision.json` | `cfd-interpret` | `cfd-experiment` (rerun), `cfd-analyze` |
| `analysis.json` | `cfd-analyze` | `cfd-paper` |
| `paper_unified_plan.json` | `cfd-paper` (planner) | `cfd-paper` (viz/writer) |
| `paper/main.tex`, `paper/paper_draft.pdf`, `paper_figs/*.png` | `cfd-paper` | — |
| `oed_artifact.json` | `cfd-open-discovery` (or code-mod path) | post-OED bridge in experiments |
| `timeline.json` | all skills (append-only event log) | observability |

If interrupted, re-invoke `/cfd-pipeline` with the same `out-dir` and `topic`; each sub-skill detects its named output already exists and skips, so the chain resumes naturally.

## Long runs
CFD cases can take hours. Do not declare timeout prematurely; FoamAgent steady runs may legitimately need 30–120 min. See `cfd-experiment` for CFL-aware retry policy.

## Outputs (final)
- `<out-dir>/paper/paper_draft.pdf` — manuscript
- `<out-dir>/paper_figs/*.png` — figures
- `<out-dir>/analysis.json` — cross-case metrics
- `<out-dir>/cases/case_*/` — per-case OpenFOAM output
- `<out-dir>/timeline.json` — full event log

## Cross-mode handoff
Output `<out-dir>` is fully compatible with the Python orchestrator. To take over with the orchestrator after partial skill execution:
```bash
python scripts/orchestrator_run.py --topic "..." --out-dir <same out-dir> --resume-from <next_stage>
```
Supported `--resume-from` stages: `literature, hypothesis, requirements, code_mod, mesh_gate_resume, baseline_synthesis, experiments, analysis, paper_review, reference_verify, analysis_without_viz_full`.

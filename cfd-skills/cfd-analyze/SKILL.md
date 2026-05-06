---
name: cfd-analyze
description: Cross-case analysis — collect all PROCEED-flagged cases, compute QoI tables, identify trends and correlations, produce analysis.json that cfd-paper consumes. Self-contained — embeds ResultsInterpreterAgent.system_prompt + user_prompt verbatim. Note: paper-figure generation is owned by cfd-paper's unified loop, not here.
---

# cfd-analyze

Aggregate per-case results into a study-level analysis artifact. Text + tables + correlations; figures are `cfd-paper`'s job.

## Inputs
- `out-dir` (required) — must have `cases/case_*/run_result.json` and `decision.json`
- Optional: reference data (DNS, experimental) for QoI comparison — typically under `<starter-dir>/reference_data/` or `<out-dir>/reference_data_manifest.json`
- `metrics` (optional) — comma-separated metric names. If unset, the agent picks 3–6 based on the topic (Cd, Cl, Cf_RMSE_vs_DNS, x_reattach, x_separation, peak_U, etc.)

## Output
`<out-dir>/analysis.json`:
```json
{
  "cases_included": ["case_001", "case_003"],
  "cases_excluded": [{"case_id": "case_002", "reason": "decision=REVISE max_retries"}],
  "metrics": {
    "Cf_RMSE_vs_DNS":  {"case_001": 0.0043, "case_003": 0.0049},
    "x_reattach":      {"case_001": 4.71, "case_003": 4.85}
  },
  "trends": ["case_003 (refined wall treatment) shows 8% lower Cf_RMSE than case_001 baseline"],
  "correlations": [{"x": "y_plus_max", "y": "Cf_RMSE_vs_DNS", "r": -0.92, "n": 4}],
  "conclusions": "...",
  "discussion": "...",
  "best_case": "case_001",
  "reference_data_used": {"dns_cf": "<path or doi>"}
}
```

## Recipe (primary, agent-driven)

### Step 1 — Collect cases
1. Walk `<out-dir>/cases/` for `case_*/decision.json`. Keep cases where `status == "PROCEED"`. List excluded cases with reasons.
2. For each kept case, load:
   - `requirements.json` entry (parameter values for this case)
   - `decision.json.key_metrics` (per-case interpreter output)
   - `vision_analysis.json` if present (deeper per-case observations)
   - `<case_dir>/postProcessing/**` — the agent walks the tree to find function-object outputs (forces, residuals, probes, sampling lines)
   - `<case_dir>/<latest_time>/` — for any spatial QoI computation

### Step 2 — Pick metrics
If `metrics` was passed in, use those. Otherwise the agent infers based on the topic and the presence of reference data:
- BFS / step flow → `Cf_along_lower_wall`, `x_reattach`, `Cp_along_lower_wall`, `peak_recirc_U`
- Channel flow → `Cf`, `peak_U`, `centerline_U_profile_RMSE`, `friction_factor`
- Airfoil / external bluff body → `Cd`, `Cl`, `Cm`, `wake_centerline_U_decay`
- Code-mod / OED → primary objective (e.g. `Cf_RMSE_vs_DNS`) + secondary diagnostics

For each metric, compute the per-case value:
- Read postProcessing aggregates (`forces.dat` etc.) for global scalars
- Read boundary fields (`<time>/<field>`) for spatial profiles → compute RMSE vs reference
- Run small numpy/scipy reductions for derived quantities

If reference data is available, compute error metrics:
- RMSE: `sqrt(mean((sim − ref)**2))`
- MAE: `mean(|sim − ref|)`
- Pearson r between sim and ref over the comparison axis

### Step 3 — Build the QoI table
Pandas DataFrame: rows = cases, columns = metrics. Save as `<out-dir>/metrics.csv` (optional but useful — paper writer can pull it in).

### Step 4 — Trend + correlation analysis (agent-driven)

For each metric column:
- Sort by the parameter that varies across cases (the sweep axis, e.g. Re or expansion ratio).
- Note monotone trends.
- Compute Pearson r with every other metric column AND the sweep parameter.
- Keep correlations with `|r| >= 0.7` and `n >= 3`.

### Step 5 — LLM call for discussion + conclusions

Use the embedded prompts below verbatim. They are exact copies of `ResultsInterpreterAgent.system_prompt` and `user_prompt` in `prompts/prompts.yaml` (the simpler block at lines 230–245, distinct from the longer interpretation prompts which are per-case). These short prompts are the right tool for cross-case textual synthesis.

#### System prompt (from `prompts/prompts.yaml: ResultsInterpreterAgent.system_prompt`)

```
You are a CFD results interpreter for OpenFOAM solver runs.
You receive only (1) the user requirement for the run and (2) the last 20 lines of the solver log.
Your job: based on the user requirement and the solver log tail, say whether the run succeeded or failed, and give a short interpretation (e.g. convergence, errors, next steps).
Use ONLY the user requirement and the solver log provided. Do not invent data.
Return valid JSON only (no markdown fences, no extra text).
```

> Note: the canonical `ResultsInterpreterAgent.system_prompt` is the per-case run-success prompt above. For *cross-case* analysis (this skill's actual job), pair its constraint discipline ("Use ONLY the data provided. Do not invent.") with the cross-case task description below. The Python pipeline's `scripts/analyze.py` does the same — it reuses the role with a specialized cross-case user message.

#### User prompt — cross-case (this skill's specific call)

System message: the system prompt above, plus this addendum:
```
For this turn, you are doing CROSS-CASE analysis. You receive a metric table (one row per case, one column per QoI), per-case interpreter summaries, and reference data when available. Your job: identify trends, correlations, and the best-performing case. Output strict JSON with keys: trends (array of strings), correlations (array of {x, y, r, n}), conclusions (string), discussion (string, 2–4 paragraphs), best_case (case_id). Do not invent numbers. Do not reference a metric that isn't in the table.
```

User message:
```
TOPIC:
{topic}

QOI TABLE (rows = cases, columns = metrics; values are floats; "—" = missing):
{metrics_table_csv}

CASE PARAMETERS (what each case varies):
{case_parameters_table}

PER-CASE INTERPRETER SUMMARIES:
{per_case_summaries}

PRECOMPUTED CORRELATIONS (kept |r| >= 0.7):
{correlations_json}

REFERENCE DATA AVAILABLE:
{reference_data_summary}

Return only the JSON object as specified.
```

Render with:
- `{topic}` — from `state.json` or pipeline arg
- `{metrics_table_csv}` — pandas DataFrame to CSV string (cap at ~50 rows; for OED runs this is usually small)
- `{case_parameters_table}` — small table showing what each case varies (Re, turbulence model, ...)
- `{per_case_summaries}` — concatenated `decision.json.key_metrics` + `decision.json.reason` for each kept case
- `{correlations_json}` — the array from step 4
- `{reference_data_summary}` — one-paragraph description: source, columns, range, units

### Step 6 — Validate and write

Validate the JSON:
- `best_case` is in `cases_included`.
- Every metric mentioned in `trends` or `correlations` exists as a column in the QoI table.
- `discussion` is non-empty and references at least one figure-able quantity.

Write `<out-dir>/analysis.json` (indent 2). Append:
```json
{"stage": "analyze", "event": "complete", "ts": "<iso>", "cases_included": <n>, "cases_excluded": <n>, "best_case": "case_001"}
```

## Scope boundary with cfd-paper
- `cfd-analyze` produces the **textual cross-case story + metrics table**.
- `cfd-paper` produces the **figures** (via `cfd-viz mode=full`) + LaTeX manuscript + reviewer loop.

Splitting allows the user to run `/cfd-analyze` standalone (e.g. for a quick QoI table) without triggering the full paper pipeline. Many useful runs end here.

## Skip if already done
If `analysis.json` exists and `cases_included` matches the current PROCEED set in `cases/`, skip with `analyze_skipped_existing`. The set match is exact — adding even one new PROCEED case forces a re-analyze.

## Anti-hallucination rules
- Numbers in `trends` and `discussion` MUST come from the QoI table — every "X is 8% lower than Y" claim must be re-derivable from the table.
- Correlations require `n >= 3`. Don't report `r` for a 2-case set.
- `best_case` must be the case that minimizes (or maximizes, for `direction=max`) the primary objective. If the primary objective is ambiguous (no clear "min vs DNS" target), say so in `discussion` and pick the case with best aggregate rank across all metrics.

## Optional script fast-path
```bash
python scripts/analyze.py \
  --cases <out-dir>/cases/case_*/ \
  --metrics "Cf_RMSE,x_reattach,Cd" \
  --output <out-dir>/analysis.json \
  --timeline <out-dir>/timeline.json
```
Same artifact contract.

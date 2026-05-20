---
name: cfd-analyze
description: Per-case + cross-case TEXT/JSON analysis — collect all PROCEED-flagged cases, compute QoI tables, identify trends and correlations, produce analysis.json that downstream skills consume. Self-contained — embeds ResultsInterpreterAgent.system_prompt + user_prompt verbatim. Scope boundary — this skill produces text + tables only. Cross-case figure generation (Cf overlays, RMSE bars, Δ-fields, parametric trends) is owned by cfd-cross-analyze. Per-case PyVista renders are owned by cfd-viz mode=interpret. Paper figure planning + LaTeX writing is owned by cfd-paper.
---

# cfd-analyze

## ⚠ HARD STOP — read this before any other step

For routes 1, 2, 6: if `<out-dir>/checkpoints/experiments_done.json` does **not** exist, STOP. Return to `Skill cfd-orchestrator` and walk the chain from literature. Authoring `analysis.json` with cases that lack `figs/*.png` and a vision-derived `decision.json` is the failure mode that produced the slim 1-page papers in earlier slices.

Route 4 (analysis-only) enters this skill directly with cases already on disk — that path is legitimate; preflight passes if every `cases/case_*/` has `run_result.json` + `figs/*.png` + a non-numerics `decision.json`.

Aggregate per-case results into a study-level analysis artifact. Text + tables + correlations; figures are `cfd-paper`'s job.

## Step 0 — Preflight gate (HARD; slice 14)

Before reading any case dir, run:

```bash
python scripts/stage_gate_audit.py --out-dir <out-dir> --mode preflight --target-stage analysis
```

If `rc != 0`, STOP. On routes 1/2/6 the script requires the full upstream chain through `experiments_done`. Authoring `analysis.json` with `experiments_done.json` missing — or with cases that lack `figs/*.png` and a vision-derived `decision.json` — is the failure mode that produced the slim "1-page main body" papers. Run `cfd-experiment` (per-case loop with `cfd-viz mode=interpret` + `cfd-interpret`) for every case first.

Route 4 (analysis-only) skips the upstream chain by design: preflight will pass if `routing_done.json` exists and the cases on disk already have valid `run_result.json`/`figs/`/`decision.json`.

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

## Next — paper stage runs in a sub-agent (mandatory for routes 1, 2, 6)

`<out-dir>/analysis.json` and `<out-dir>/checkpoints/analysis_done.json` are now on disk. The substantive science is complete. **The paper stage must run, and it runs in a fresh sub-agent** — not in your current session.

Why: long end-to-end chains accumulate context, retry history, and "task closure" bias by the time they reach the paper stage. Empirically the same paper skill produces a fabricated stub in long sessions and a real paper in fresh sessions. Spawning a sub-agent with focused scope re-creates the fresh-session conditions automatically.

### Read the route

```bash
MODE=$(jq -r '.mode' <out-dir>/state.json)
case "$MODE" in
  research|code_mod|open_discovery) NEEDS_PAPER=1 ;;
  paper_only)                       NEEDS_PAPER=1 ;;
  mesh_gate|analysis_only)          NEEDS_PAPER=0 ;;
  *)                                NEEDS_PAPER=1 ;;
esac
```

- `NEEDS_PAPER=0` (routes 3, 4) → chain ends here. The orchestrator's final audit (Step 5) is your next action; do NOT spawn a sub-agent for the paper stage.
- `NEEDS_PAPER=1` (routes 1, 2, 5, 6) → continue below, unless we are already inside a parent sub-agent.

### Skip the dispatch if already inside a sub-agent

```bash
NESTED=$(jq -r '.paper_subagent_dispatched // false' <out-dir>/state.json)
if [ "$NESTED" = "true" ]; then
  echo "analyze: paper_subagent_dispatched=true — returning inline so the parent sub-agent continues with cfd-cross-analyze → cfd-paper directly. No nested sub-agent."
  exit 0
fi
```

The `cfd-open-discovery` skill's Step 15 sets `paper_subagent_dispatched=true` before invoking its own end-of-chain sub-agent (analyze → cross-analyze → paper → audit). When that parent sub-agent then invokes `cfd-analyze`, this guard fires and analyze returns inline — the parent continues the chain itself rather than spawning a second nested sub-agent.

### Your next action — invoke the Agent tool

Read `<out-dir>/state.json#topic` and the absolute `<out-dir>` path. Then invoke the `Agent` tool with `subagent_type="general-purpose"`, a short description, and the prompt below with `<OUT_DIR>` and `<TOPIC>` and `<MODE>` substituted with the actual values from state.json. **This is a single tool call.** Wait for the sub-agent to return.

```
Sub-agent prompt (pass as the `prompt` parameter to the Agent tool):
---------------------------------------------------------------------
You are the paper-stage sub-agent for a CFD scientist run. The upstream chain (literature, setup, hypothesis, requirements, mesh-gate, experiments, analysis, and any code-mod if applicable) has completed. Do NOT re-run any upstream stage — all relevant checkpoints already exist under <OUT_DIR>/checkpoints/.

Run directory: <OUT_DIR>
Topic:         <TOPIC>
Mode:          <MODE>

Your only job is the paper stage. Walk it end to end via the cfd-paper skill cycle, then prove completion via the audit script.

STEP 1 — Clear any prior fabricated paper-stage stubs from earlier aborted attempts so the skill does not mistake them for completed work:
    rm -f <OUT_DIR>/review.json
    rm -f <OUT_DIR>/paper_unified_plan.json
    rm -f <OUT_DIR>/paper_cycle_state.json
    rm -f <OUT_DIR>/paper_review_history.json
    rm -f <OUT_DIR>/audit_passed.json
    rm -f <OUT_DIR>/checkpoints/paper_done.json
    rm -f <OUT_DIR>/checkpoints/review_done.json
    rm -f <OUT_DIR>/checkpoints/paper_planner_done.json
    rm -f <OUT_DIR>/checkpoints/paper_viz_done.json
    rm -f <OUT_DIR>/checkpoints/paper_writer_done.json
    rm -f <OUT_DIR>/checkpoints/paper_compile_done.json
    rm -f <OUT_DIR>/checkpoints/paper_review_done.json
    rm -f <OUT_DIR>/paper/main.tex
    rm -f <OUT_DIR>/paper/main.pdf
    rm -rf <OUT_DIR>/paper_figs

STEP 2 — Invoke Skill cfd-skills/cfd-paper. Walk its numbered protocol literally and in order:
    Stage 0 (paper experiment planner — one LLM call → paper_experiment_plan.json)
    → invoke Skill cfd-skills/cfd-cross-analyze (produces cross_experiment_analysis/*.png + cross_experiment_interpretation.md)
    → Phase A (planner LLM call → paper_unified_plan.json matching the planner's required schema)
    → Phase B (author paper_figs/paper_viz_batch.py, run with xvfb-run python, produce paper_figs/*.png; revise script inner-loop if needed, cap = 10)
    → Phase C (per-image vision QA on every PNG; if any fail, GOTO Phase B.2 to revise script)
    → Phase D (writer LLM call → paper/main.tex with mandatory sections Introduction, Related Work, Methods, Results, Discussion, Conclusion AND >= 10 \cite{} calls grounded in lit.json AND >= 1 \includegraphics per included case)
    → Phase E (pdflatex; compile-fix LLM call if it errors; cap = 3)
    → Phase F (reviewer LLM call → review.json with full reviewer schema: pass, score, recommendations[], formatting_ok, figures_ok, content_ok, summary, regenerate_batch_figures, ...)
    → If reviewer pass=true && score>=0.7: write paper_review_done.json{status:"ok"}; cycle done.
    → Else if iteration < 10: GOTO Phase B (regen figures) or Phase D (text revise) per the reviewer's regenerate_batch_figures flag.
    → Else: write paper_review_done.json{status:"failed_after_loops_exhausted"}; halt.

Run every Bash validation gate inside the cycle (the jq -e checks after each phase). If any gate fails, redo that phase. Do not advance past a gate. Do not write paper_unified_plan.json, review.json, paper/main.tex, or audit_passed.json by hand — they have specific schemas the audit verifies; hand-written stubs are detected and fail.

STEP 3 — After the cycle terminates, run the audit. This is non-negotiable.
    python scripts/stage_gate_audit.py --out-dir <OUT_DIR>
If rc != 0, read the failures, re-invoke the relevant cycle phase to fix each one, and re-run the audit. Repeat until rc=0 and audit_passed.json is written by the script. Do NOT write audit_passed.json yourself.

Constraints — apply globally:
- The only forbidden Python file in skill mode is scripts/orchestrator_run.py.
- Every other script under scripts/ (stage_gate_audit.py, lit.py, viz.py, interpret.py, analyze.py, rag_query.py, foam_run.py, paper_unified.py, etc.) is callable.
- xvfb-run, pdflatex, jq, Bash heredocs, inline Python, import pyvista, import matplotlib, import numpy, import pandas — all available and expected.
- There is NO "only one script allowed" rule.

Return when audit_passed.json exists at <OUT_DIR>/audit_passed.json (rc=0 outcome) OR after 3 unrecoverable audit failures with the latest failure list and which phases you re-ran.
---------------------------------------------------------------------
```

### After the sub-agent returns

```bash
if [ -f <out-dir>/audit_passed.json ]; then
  jq -e '.audit_signature == "stage_gate_audit.py:v1"' <out-dir>/audit_passed.json \
    && echo "paper stage complete; chain done"
else
  # The sub-agent did not pass the audit. Read the audit output, and either:
  #  (a) re-invoke the Agent tool with the same prompt + the failure list appended, OR
  #  (b) if you have re-invoked the Agent tool 3 times without rc=0, halt and surface to user.
  echo "paper sub-agent did not reach audit pass; re-spawn or halt"
fi
```

If the sub-agent returned without `audit_passed.json` and you have re-spawned ≤ 2 times, re-invoke the `Agent` tool with the same prompt structure plus an appended block:

```
PREVIOUS AUDIT FAILURES (fix these specifically before re-running audit):
<paste the failure list from `python scripts/stage_gate_audit.py --out-dir <OUT_DIR>` output>
```

Otherwise (3rd spawn failed, or you have other reason to halt), surface to the user and stop.

Once `audit_passed.json` is present and matches the v1 signature, the orchestrator's Step 5 is satisfied automatically — you do not need to re-run the audit yourself; the sub-agent's final audit call wrote it. Report the paper path (`<out-dir>/paper/main.pdf`) to the user. Chain complete.

For route 4 (analysis-only) when the user did NOT request a paper: the chain ends at `analysis_done.json`. Do not spawn a sub-agent. The orchestrator's final audit runs normally.

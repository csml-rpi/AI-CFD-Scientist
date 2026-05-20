---
name: cfd-interpret
description: Per-case decision agent — PROCEED, REVISE, or RERUN. Reads run_result.json + diagnostic figures, judges physics plausibility through a vision-LLM pass, returns decision.json. Self-contained — embeds ResultsInterpreterAgent.interpretation_* and vision_* prompts verbatim.
---

# cfd-interpret

## ⚠ HARD STOP — read this before any other step

If `<case_dir>/figs/` does not contain at least one `.png`, STOP. The orchestrator chain enters this skill only after `cfd-viz mode=interpret` has produced per-case PyVista renders. Return to `Skill cfd-skills/cfd-viz` with `mode=interpret case_dir=<case_dir>`, then come back. Authoring `decision.json` from `run_result.json` + numerics alone is forbidden — vision-LLM judgment over rendered flow fields is the entire point of this skill.

After `cfd-experiment` runs a case, decide whether the result is acceptable, needs a tweaked rerun (numerical), or needs the requirement revised (physics).

## Inputs
- `out-dir` (required)
- `case_id` (required)
- Implicit: `<case_dir>/run_result.json`, `<case_dir>/figs/*.png`. **`figs/` is a HARD precondition** — the skill MUST invoke `cfd-viz mode=interpret` first when `figs/` is empty, and MUST hard-fail with `RERUN` if viz can't produce figures. Numerics-only verdicts are forbidden. See "Hard preconditions" section below.

## Output
`<case_dir>/decision.json`:
```json
{
  "status": "PROCEED|REVISE|RERUN",
  "reason": "<short physical reasoning>",
  "suggested_changes": "<for REVISE: specific requirement edits; for RERUN: numerical tweaks>",
  "confidence": 0.0,
  "key_metrics": {"reattachment_x": 4.7, "Cd": 0.42},
  "simulation_success": true,
  "requirement_met": true,
  "issues": ""
}
```

## Decision rules

- **PROCEED** — physics looks sensible, requirement is satisfied. Examples: separation in expected location, residuals converged, y+ in target range, fields look smooth.
- **REVISE** — flow is unphysical or violates the requirement (reattachment far off, pressure profile wrong, mass not conserved, geometry/scenario in figures doesn't match what the requirement asked for). Action: regenerate the requirement (`cfd-requirements` style) and rerun.
- **RERUN** — numerical issue despite valid physics: timeout, divergence, NaN, oscillating residuals. Action: apply CFL-aware tweak in `cfd-experiment`, keep physics identical, rerun.

## Hard preconditions (skill-mode enforcement)

This skill is **vision-first**. The embedded `interpretation_*` prompts below are multimodal — they expect rendered case figures attached to the LLM call as image inputs. **Authoring a `decision.json` from numerics alone (residuals, log tail, extracted metrics) is forbidden.**

Hard rules:

1. **`<case_dir>/figs/` MUST contain at least one `.png` before this skill writes `decision.json`.** If `figs/` is empty or missing, the skill MUST invoke `cfd-viz mode=interpret` first (Step 1 below) and only proceed to Step 2 once at least one figure exists.
2. **If `cfd-viz mode=interpret` returns without producing any PNG**, hard-fail with `decision.json: {"status": "RERUN", "reason": "viz failed — cannot interpret without figures", "viz_failed": true, "no_figures": true}`. **Do NOT silently fall back to a numerics-only verdict.** A numerics-only judgment is a different (weaker) check; it cannot stand in for the vision-LLM check this skill is for.
3. **If the vision-LLM call (Step 3 below) cannot be made** (e.g. provider down, auth missing): `decision.json: {"status": "RERUN", "reason": "vision-LLM call failed; retry when provider is healthy", "vision_call_failed": true}`. Same principle — don't fabricate a verdict from text outputs.

These rules close the failure mode where a clean-converging RANS run gets a textual PROCEED on residuals alone, bypassing the geometry/scenario / unphysical-recirculation / wrong-patch-sampling / y+-violation checks that the vision pass is the only place to catch.

## Recipe (primary, agent-driven)

### Step 1 — Ensure figures (HARD PRECONDITION)
**Mandatory before Step 2.** Check `<case_dir>/figs/`:
- If it contains zero `.png` files (or doesn't exist), **invoke `cfd-viz mode=interpret`** before continuing:
  ```text
  /cfd-viz out-dir=<out-dir> mode=interpret case_dir=<case_dir>
  ```
- After `cfd-viz` returns, re-check `<case_dir>/figs/`. If still empty, apply hard rule #2 above (write `RERUN` decision.json with `viz_failed=true` and stop).

`cfd-viz mode=interpret` produces 5–8 diagnostic PNGs sufficient for the vision-LLM physics check (mesh overview + zoomed near-wall, U-magnitude contour at latest time, pressure contour at latest time, streamlines on midplane, y+ map on no-slip walls, residual history). The PyVista call ensures these are real flow-field renders, not metrics-derived gnuplot plots.

### Step 2 — Assemble context
Read:
- The case's `user_requirement_text` (from `<out-dir>/requirements.json` lookup by `case_id`, or from the requirement that drove the run if mesh-gate / OED).
- `<case_dir>/run_result.json` — status, error_logs, loop_count, wall_time.
- The last 50 lines of `<case_dir>/log.<solver>` (residual history).
- The list of figure paths in `<case_dir>/figs/`.

### Step 3 — Vision-LLM call (interpretation prompts)

Use the embedded prompts below verbatim. They are exact copies of `ResultsInterpreterAgent.interpretation_*` in `prompts/prompts.yaml`.

#### System prompt (from `prompts/prompts.yaml: ResultsInterpreterAgent.interpretation_system_prompt`)

```
You are a CFD results interpreter. You receive the user requirement and visualization images from an OpenFOAM run.
Scope constraint: evaluate ONLY the current experiment represented by the provided images. Do not infer or assume anything about other experiments/cases unless those other-case results are explicitly visible in the provided images.
Your tasks: (1) Did the simulation run successfully (no crashes, completed)? (2) Did the results satisfy the user requirement? (3) VERY IMPORTANT: check that the geometry and flow scenario described in the user requirement (e.g. jet, cavity, channel, bluff-body, backward-facing step, etc.) actually matches what is shown in the images (domain shape, inlet/outlet locations, obstacles, etc.). If the geometry/scenario in the images does not match the requested scenario, treat this as "requirement NOT met". (4) What issues exist, if any (e.g. divergence, unphysical fields, wrong setup)? (5) Should the run be redone (rerun_required)? (6) Extract a concise set of salient quantitative or categorical metrics that matter most for the study (e.g. reattachment length, drag coefficient, centerline decay rate, qualitative verdict on recirculation bubble size or jet spread) and report them in a dedicated key_metrics field.
Mesh guidance: A higher mesh resolution / higher cell count than requested (e.g., more cells than a "<10000 cells" target) is NOT, by itself, grounds to reject the simulation. Only request rerun if the results are unphysical, the geometry/flow scenario does not match, or there are execution/numerical issues visible in the images/logs that prevent evaluating the requirement.
Be concise. Return ONLY valid JSON with keys: simulation_success (bool), requirement_met (bool), issues (string or list), rerun_required (bool), summary (string), reasons (string), key_metrics (object with a few named entries capturing the most important metrics or qualitative judgements). If results are good and requirement is met, set rerun_required false and explain why. If not, set rerun_required true and give reasons, explicitly mentioning any geometry/scenario mismatch.
```

#### User prompt (from `prompts/prompts.yaml: ResultsInterpreterAgent.interpretation_user_prompt`)

```
User requirement:
{user_requirement}

The following images are visualizations from the run (mesh, fields at different times/angles). Evaluate only this current experiment; do not use or assume other experiments unless they are explicitly shown in the provided images. Did the simulation satisfy the requirement? Any issues? Set rerun_required true only if results are not acceptable. Return JSON only.
```

Render with `{user_requirement}` = the case's full requirement text. Attach all images from `<case_dir>/figs/*.png` as multimodal inputs to the LLM call.

### Step 4 — Optional deep vision pass

For tricky cases (run completed but interpretation is ambiguous, or the user is generating a paper and needs richer per-case analysis), follow up with the deeper vision-analyzer prompt below. This produces a structured per-figure flow-field analysis used by `cfd-paper`'s Writer for narrative content. Use it only when:
- The interpretation step returned `requirement_met=true` AND `simulation_success=true` (so we want to enrich, not block).
- The downstream caller is `cfd-analyze` or `cfd-paper`.
- Otherwise skip; the basic interpretation prompts above are sufficient for the PROCEED/REVISE/RERUN decision.

#### System prompt (from `prompts/prompts.yaml: ResultsInterpreterAgent.vision_system_prompt`)

```
You are a CFD VISUALIZATION ANALYZER specializing in interpreting Computational Fluid Dynamics simulation results.
Your job is to analyze CFD visualization images and provide detailed technical insights about the flow field,
convergence, and physical phenomena for the simulation (e.g. internal flow, external flow, jets, cavities, multiphase, etc.).

CORE RESPONSIBILITIES:
1. **Flow Analysis**: Identify flow features such as vortices, recirculation zones, boundary layers, shear layers, jets, wakes, etc., as relevant to the problem type.
2. **Physical Validation**: Verify if the flow behavior is physically reasonable for the given geometry, boundary conditions, and flow regime.
3. **Convergence Assessment**: Evaluate if the solution appears converged based on visual indicators.
4. **Parameter Effects**: Analyze how the case parameters (e.g. inlet velocity, geometry, Reynolds number) affect the flow field.
5. **Quality Assessment**: Identify potential issues with the simulation or visualization.
6. **Problem-Specific Analysis**: Focus on flow phenomena and geometry effects relevant to the type of simulation.

ANALYSIS FRAMEWORK:
- **Contour Analysis**: Interpret velocity magnitude, pressure, and other field contours in the domain.
- **Streamline Analysis**: Analyze flow direction, vortex formation, and recirculation patterns.
- **Vector Field Analysis**: Understand velocity vectors and flow patterns in the domain.
- **Boundary Layer Analysis**: Assess wall effects and boundary layer development where relevant.
- **Vortex / Feature Analysis**: Examine vortex formation, shear layers, jets, or other dominant features as appropriate.
- **Centerline / Profile Analysis**: Assess velocity or scalar profiles along relevant lines (centerline, inlet-to-outlet, etc.) when applicable.
- **Geometry Effects**: Analyze how domain aspect ratio and topology affect flow patterns.

CFD EXPERTISE:
- Recognize flow phenomena (vortices, recirculation, boundary layers, jets, shocks, etc.) appropriate to the problem type.
- Identify numerical artifacts (oscillations, checkerboarding, spurious oscillations, etc.).
- Understand different visualization types (contours, vectors, streamlines, isosurfaces, etc.).
- Relate visual patterns to physical flow behavior for the given parameters.
- Compare results with expected flow behavior for the given conditions.
- Adapt analysis approach based on flow regime (laminar, transitional, turbulent).

OUTPUT REQUIREMENTS:
- Provide clear, technical analysis of the flow field.
- Identify any physical or numerical issues specific to the simulation type.
- Relate observations to the simulation parameters and context.
- Suggest improvements if problems are identified.
- Use proper CFD terminology and concepts.
- Consider validation criteria appropriate to the flow problem.

RULES:
- Focus on technical CFD analysis of the flow, not general image description.
- Use specific CFD terminology appropriate to the type of simulation (internal/external flow, compressible/incompressible, etc.).
- Relate observations to the simulation context provided.
- Be quantitative when possible (e.g., "strong recirculation", "well-developed shear layer", "clear jet spreading").
- Identify both positive aspects and potential issues in the flow patterns.
- Adapt analysis depth and focus based on Reynolds number, geometry, and problem type.
```

#### User prompt (from `prompts/prompts.yaml: ResultsInterpreterAgent.vision_user_prompt`)

````
You are analyzing a CFD visualization image from an OpenFOAM simulation.

SIMULATION CONTEXT:
{simulation_context}

ANALYSIS TASKS:
1. **Flow Field Description**: Describe the main flow patterns and phenomena visible for this specific problem type
2. **Physical Validation**: Assess if the flow behavior is physically reasonable for the given geometry, boundary conditions, and flow regime
3. **Convergence Assessment**: Evaluate if the solution appears converged based on visual indicators appropriate to the problem type
4. **Parameter Effects**: Analyze how the parameter value affects the flow field compared to other simulations in the study
5. **Quality Issues**: Identify any potential problems or artifacts specific to this type of simulation
6. **Research Relevance**: Assess how well the visualization supports the research question and experiment goals
7. **Problem-Specific Analysis**: Provide insights relevant to the specific type of flow problem (internal/external flow, compressible/incompressible, multiphase, heat transfer, etc.)

OUTPUT FORMAT:
Return a JSON object with the following structure. You may ADD additional fields that are relevant to the specific problem type, and you may EXPAND any section with problem-specific details:
```json
{{
  "flow_field_analysis": {{
    "main_patterns": ["description of primary flow patterns for this problem type"],
    "flow_features": ["description of key flow features (vortices, shocks, boundary layers, etc.)"],
    "separation_points": ["locations of flow separation if applicable"],
    "recirculation_zones": ["description of recirculation regions if present"],
    "problem_specific_features": ["features specific to this type of flow problem"],
    "additional_observations": "any other flow field observations relevant to this problem"
  }},
  "physical_validation": {{
    "realistic_flow": true|false,
    "expected_behavior": "description of expected flow for this problem type and conditions",
    "anomalies": ["list of unexpected or unrealistic features"],
    "problem_specific_validation": "validation criteria specific to this flow type",
    "domain_specific_checks": "additional validation checks relevant to this flow type"
  }},
  "convergence_assessment": {{
    "appears_converged": true|false,
    "indicators": ["convergence indicators observed for this problem type"],
    "concerns": ["potential convergence issues"],
    "problem_specific_convergence": "convergence criteria relevant to this simulation type",
    "specialized_convergence_metrics": "convergence metrics specific to this problem type"
  }},
  "parameter_effects": {{
    "parameter_impact": "how the parameter value affects the flow",
    "trends": ["observable trends in the flow field"],
    "sensitivity": "assessment of parameter sensitivity",
    "problem_specific_effects": "parameter effects relevant to this flow type",
    "parameter_insights": "additional insights about parameter effects for this problem"
  }},
  "quality_assessment": {{
    "overall_quality": "excellent|good|fair|poor",
    "issues": ["list of quality issues or artifacts"],
    "recommendations": ["suggestions for improvement"],
    "problem_specific_quality": "quality criteria relevant to this simulation type",
    "specialized_quality_metrics": "quality metrics specific to this problem type"
  }},
  "research_relevance": {{
    "supports_hypothesis": true|false,
    "contribution": "how this visualization contributes to the research goals",
    "key_findings": ["main insights relevant to the research question"],
    "problem_specific_insights": ["insights specific to this type of flow problem"],
    "research_implications": "broader research implications for this problem type"
  }},
  "technical_insights": [
    "insight_1",
    "insight_2",
    "insight_3"
  ],
  "problem_specific_analysis": {{
    "specialized_observations": "observations specific to this flow type",
    "domain_expertise_insights": "insights that leverage domain expertise for this problem",
    "additional_recommendations": "recommendations specific to this problem type"
  }}
}}
```

FLEXIBILITY GUIDELINES:
- You may ADD new fields to any section if they provide valuable insights for this specific problem
- You may EXPAND existing fields with problem-specific details
- You may include ADDITIONAL SECTIONS if they are relevant to the specific flow type
- Focus on providing the most relevant and insightful analysis for this particular problem
- Don't feel constrained by the template - adapt it to serve the analysis needs
- Use your domain expertise to provide the most valuable insights for this specific flow type
````

Render `{simulation_context}` as a short paragraph: topic + requirement summary + the parameter values that distinguish this case from others. Save the resulting JSON as `<case_dir>/vision_analysis.json` so `cfd-paper` can pick it up later.

### Step 5 — Map result to decision

Take the JSON from Step 3:
- `simulation_success == false` → `RERUN`. `suggested_changes` = "apply CFL-aware retry: adjustTimeStep yes, maxCo 0.7, slight deltaT increase".
- `simulation_success == true && requirement_met == false`:
  - If issues mention geometry/scenario mismatch, divergent fields, or unphysical results that aren't fixable by numerics → `REVISE`. `suggested_changes` = a concrete edit to the requirement text addressing the mismatch (e.g. "step is in wrong location: change inlet to x=0 and step lip to x=4h").
  - Otherwise (rerun_required=true with numerical complaint only) → `RERUN`.
- `simulation_success == true && requirement_met == true` → `PROCEED`.

Set `confidence` from a simple proxy: high (0.9) when simulation_success, requirement_met, and key_metrics has ≥ 2 entries; medium (0.7) when one metric or scenario looks marginal; low (0.4) when interpreter wrote "ambiguous" or "unclear" anywhere.

### Step 6 — Write `decision.json`

```json
{
  "status": "PROCEED|REVISE|RERUN",
  "reason": "<short physical reasoning>",
  "suggested_changes": "<for REVISE/RERUN; empty for PROCEED>",
  "confidence": 0.9,
  "key_metrics": {...},
  "simulation_success": true,
  "requirement_met": true,
  "issues": "",
  "raw_interpreter_response": {...}
}
```

Append to timeline:
```json
{"stage": "interpret", "event": "decision", "ts": "<iso>", "case_id": "case_001", "status": "PROCEED", "confidence": 0.9}
```

## Retry budget

Per case, limit the rerun/revise loop to **3 attempts** at this skill level (the orchestrator may use 5 in some configs). Beyond that, write `decision.json` with `status: REVISE`, `reason: "max retries exceeded — needs human"`, and let the calling skill (typically `cfd-pipeline` or `cfd-open-discovery`) decide whether to escalate or skip the case.

## Calling pattern (used by cfd-pipeline)

```text
attempts = 0
while attempts < 3:
  /cfd-experiment case_id
  /cfd-interpret  case_id
  read decision.json
  if status == PROCEED: break
  if status == REVISE:  rewrite requirement; archive case; continue
  if status == RERUN:   apply CFL tweak in controlDict; continue
  attempts += 1
if status != PROCEED: mark case "failed_with_max_retries"
```

## Anti-hallucination rules
- Never invent metric values that aren't visible in the figures or computable from the case dir. If you can't see reattachment in the figures, leave `key_metrics.reattachment_x` out — don't guess.
- The vision LLM's `realistic_flow=false` is enough to set REVISE, but the `reasons` field must cite a specific image observation (e.g. "Figure U_contour shows recirculation extending beyond x=10h, contradicting expected reattachment at x=4-7h"). If the LLM gave no concrete observation, request one more iteration before deciding.

## Scope boundary with cfd-rerun-selector

The rerun-selector logic — picking a similar successful case as a template when `REVISE` requires rewriting the requirement — lives in the calling pipeline (`cfd-pipeline` step 10's loop), not in this skill. This skill produces the decision; the pipeline handles the rerun mechanics.

## LangGraph mode only — do NOT call from skill mode

## Two valid execution paths

1. **Inline agent recipe (above).** The agent reads `figs/*.png` directly via its native multimodal capability, applies the embedded `interpretation_*` prompts as judgment criteria, and writes `decision.json`.

2. **Script fast-path.** `scripts/interpret.py` (full retry loop) and `scripts/quick_interpret.py` (single-shot, faster) implement the same logic. Same prompts, same `decision.json` schema. Invoke via Bash with `--case-dir`, `--out-dir`, etc. The skill-mode-forbidden script is `scripts/orchestrator_run.py` only.

## Next

After `<case_dir>/decision.json` is on disk, walk the per-case loop:

```bash
# Find the next case (in requirements.json order) that does not yet have a decision.json.
NEXT=$(jq -r '.[].case_id' "<out-dir>/requirements.json" 2>/dev/null | while read c; do
  [ -f "<out-dir>/cases/$c/decision.json" ] || { echo "$c"; break; }
done | head -1)
```

| Condition | Next action |
|---|---|
| `$NEXT` non-empty | `Skill cfd-skills/cfd-experiment` for case `$NEXT` (chain continues: experiment → viz → interpret → next case). |
| `$NEXT` empty (every case has `decision.json`) | Write `<out-dir>/checkpoints/experiments_done.json` after the orchestrator's per-case file-system gate passes, then invoke `Skill cfd-skills/cfd-analyze`. |
| `decision.json#status == REVISE` (after max revisions) or `RERUN` (after max retries) | Halt the per-case loop, surface to the user, and stop. |

Do not stop, summarize, or wait between cases — the chain continues automatically.

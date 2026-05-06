---
name: cfd-hypothesis
description: Convert literature + topic into a list of testable, FoamAgent-executable CFD hypotheses. Self-contained — embeds the HypothesisAgent system + user prompts verbatim from prompts/prompts.yaml. Outputs hypotheses.json. Use after cfd-literature in the general pipeline.
---

# cfd-hypothesis

Generate concrete, testable CFD hypotheses from `lit.json` and the topic. Each hypothesis is later expanded by `cfd-requirements` into a FoamAgent user-requirement string.

## Inputs
- `out-dir` (required) — must contain `lit.json` and `state.json` with the topic
- `n_hypotheses` (optional, default 5)
- `provider`, `model` (optional) — LLM hints; inherit from agent if unset

## Output
`<out-dir>/hypotheses.json` — JSON array:
```json
[
  {
    "hypothesis_id": "h_001",
    "statement": "string",
    "rationale": "string",
    "supporting_papers": ["paperId1", "paperId2"],
    "expected_outcome": "string",
    "constraints": {
      "solver": "<openfoam_solver>",
      "Re": <number>,
      "geometry": "<geometry_id>",
      "turbulence_model": "<model or null>",
      "mesh_cells_max": <number or null>,
      "CFL_target": <number or null>,
      "endTime": <number or null>
    }
  }
]
```

## Recipe (primary, agent-driven)

### Step 1 — Load context
1. Read `<out-dir>/lit.json` (the array from `cfd-literature`).
2. Read the topic: from `<out-dir>/state.json` if it exists, else from the calling skill's `topic` argument.
3. Build `literature_context` — concatenate, for each paper, a block of:
   ```
   [paperId] Title (year, cit=N)
     Abstract: <abstract or "—">
   ```
   Cap at the top ~10–12 papers by citation count to keep the prompt within context.

### Step 2 — Render and call

Use the embedded prompts below verbatim. The `HypothesisAgent` block in `prompts/prompts.yaml` is the ground truth; what follows is its complete text (kept in sync with that file — search for `HypothesisAgent:` there to verify if you suspect drift).

#### System prompt (from `prompts/prompts.yaml: HypothesisAgent.hypothesis_system_prompt`)

```
You are HYPOTHESIS-TO-FOAM, a converter that transforms OpenFOAM CFD study ideas into user requirements for Foam-Agent.

MISSION:
Convert ONE simulation from the study JSON into a single, complete user requirement paragraph that Foam-Agent can execute.

INPUT:
- idea_json: Complete CFD study with cases, solver, target_CFL, post-processing details, etc.
- selected_simulation: One simulation from a case to convert

CONVERSION RULES:
1) GEOMETRY & DOMAIN
   - Use case.dimensions [x, y, z] for domain dimensions
   - Use case.topology (2d|3d) to determine mesh type
   - For 2D simulations: z_cells = 1, front/back = empty
   - For 3D simulations: use full 3D mesh

2) PHYSICS & SOLVER
   - Use study.solver
   - Calculate kinematic viscosity from Reynolds number: ν = (L × U) / Re where applicable
   - Use study.target_CFL for time step calculation when provided
   - Use standard fluid properties as appropriate (e.g. density, viscosity)

3) CASE PARAMETERS
   - Use selected_simulation.parameter_value as the parameter(s) over which the simulation is to be executed (e.g. inlet velocity, geometry size, Reynolds number, turbulence model, etc. as defined in the study)
   - VERY IMPORTANT: Respect what the *topic/study description* says should be swept across experiments, and keep all other aspects consistent across the experiment set.
   - Example A (geometry/BC sweep): if the topic says to sweep over different geometries or boundary-condition layouts (e.g. several backward-facing step expansion ratios or different inlet conditions) at fixed turbulence model, then change ONLY the geometry/BCs according to the study JSON and keep solver, turbulence model, and other physics the same across experiments.
   - Example B (turbulence-model sweep): if the topic says to sweep over turbulence models (e.g. $k$–$\\varepsilon$, $k$–$\\omega$ SST, Spalart–Allmaras) at fixed geometry and BCs, then change ONLY the turbulence model (and related near-wall treatments) and keep geometry, domain dimensions, and boundary-condition types/locations identical across experiments.

4) BOUNDARY CONDITIONS
   - Specify all boundary conditions in detail
   - For 2D: frontAndBack = empty
   - For 3D: specify wall types (no-slip, slip, inlet, outlet, etc.) as appropriate for the problem

5) SIMULATION SETTINGS
   - Use study.target_CFL to calculate time step
   - End time and write interval: use study controls or reasonable defaults
   - Mesh: structured mesh with appropriate cell count

6) POST-PROCESSING CONTEXT
   - You may mention measurable outputs (e.g., residuals, centerline metrics) only as context.
   - DO NOT include any explicit visualization instructions in the Foam-Agent requirement paragraph.
   - DO NOT include commands starting with "Visualize...".

7) UNITS & ASSUMPTIONS
  - SI units everywhere. No placeholders (no “TBD”).
  - If you must invent a value, pick a safe, conservative default and tag with “#ASSUMPTION(reason=value)”.

8) RUN TOPIC CONTEXT
   - The user launched this run with a high-level topic (provided below). Embed this in the requirement naturally.
   - Add one short phrase or sentence so the requirement is clearly part of that broader study, e.g. "This simulation is part of the broader study to {topic}." or "This case supports a parametric study of {topic}."
   - Do not copy the topic verbatim; paraphrase or summarize as needed so it reads naturally in the requirement.
   - CONSTRAINTS FROM TOPIC: Read the topic for any specific simulation instructions and ensure the generated requirement satisfies them. Examples:
     * If the topic mentions a maximum mesh size or cell count (e.g. "mesh of less than or equal to 10,000 cells", "not more than 5000 cells"), the requirement MUST specify a mesh that satisfies that limit (e.g. state total cell count or nx×ny×nz that stays at or below the limit).
     * If the topic mentions CFL (e.g. "CFL less than 0.5", "target CFL 0.5"), the requirement MUST specify deltaT and/or mesh spacing so that the stated CFL constraint is satisfied.
     * If the topic mentions time step, end time, write interval, or other run-control limits, reflect those in the requirement.
   - If the topic does not mention such constraints, use reasonable defaults; if it does, the generated requirement must explicitly satisfy them.

OUTPUT FORMAT:
- Return EXACTLY ONE paragraph of plain English
- No lists, JSON, YAML, or Markdown
- Must include: geometry/dimensions, key parameters (as defined by the case), topology, solver, boundary conditions, mesh, time settings
- Must NOT include visualization instructions or visualization command text.
- Use SI units everywhere
- If values are missing, use reasonable defaults and mark with "#ASSUMPTION"


Rules:
- Keep geometry simple (blockMesh for rectangular domains).
- Target mesh ≤ 10,000 cells for fast execution when no other constraint is given. Use reasonable cell counts (e.g. 100x100 for 2D).
- Provide concrete values for:
    – domain dimensions and key geometry parameters
    – inlet/flow parameters as defined by the case
    – fluid properties (ρ, ν)
    – run control (startTime, endTime, deltaT, writeInterval)
- If a value is missing, insert a *reasonable* default and mark it with
  “# ASSUMPTION”.


Here is the experiment to be converted into a simulation requirement: 
{experiment_concept}

Output format
Return **exactly one paragraph of plain text**.
It should contain *no* YAML/JSON/Markdown fences.
The paragraph must read like a human user’s request to Foam‑Agent.

```

#### User prompt (from `prompts/prompts.yaml: HypothesisAgent.hypothesis_user_prompt`)

```
Convert this simulation into a Foam-Agent user requirement.

Run topic (embed in the requirement naturally and satisfy any simulation constraints it states—e.g. max cells, CFL, mesh size, time step):
{run_topic}

Study: {study_id}
Description: {description}
Case: {case_name}

Selected Simulation:
- Simulation ID: {simulation_id}
- Parameter values (use keys from the case, e.g. velocity, dimensions, Re): {parameter_values}
- Description: {simulation_description}

Broader experiment-set context (for consistency across experiments):
{all_experiment_ideas}

Current experiment idea (focus this requirement on this case):
{current_experiment_idea}

Previous experiment requirement (maintain style/consistency; change only what this case needs):
{previous_requirement}

Important: Do NOT include any visualization request in the generated requirement.

Return ONLY the user requirement paragraph for Foam-Agent.

```

> The HypothesisAgent prompts above are designed for the *requirement-string* expansion that `cfd-requirements` does. For *hypothesis enumeration* — which is what this skill produces — pair the system prompt's spirit with the smaller hypothesis-only request below. (The Python pipeline's `scripts/hypothesis.py` does the same: it reuses the HypothesisAgent role to brainstorm a study, then `scripts/requirements.py` reuses it to expand each entry.)

#### Hypothesis-enumeration request (this skill's specific call)

System message: the **HypothesisAgent system prompt above**, plus this addendum:
```
For this turn, your task is NOT yet to write a Foam-Agent requirement paragraph. Instead, propose {n_hypotheses} testable CFD hypotheses for the topic below, grounded in the literature provided. Each hypothesis must be falsifiable through one or two OpenFOAM runs within reasonable wall time. Output strict JSON only — no prose, no markdown fences.
```

User message:
```
TOPIC:
{topic}

LITERATURE CONTEXT (top papers from cfd-literature):
{literature_context}

Generate {n_hypotheses} hypotheses. Hypotheses must be NON-OVERLAPPING — each tests a distinct effect. Hypotheses must specify a solver, geometry, and at least one quantitative constraint (Re, CFL, mesh size, end time, or turbulence model).

Return ONLY a JSON array of objects with keys: hypothesis_id, statement, rationale, supporting_papers (array of paperIds), expected_outcome, constraints (object).
```

### Step 3 — Parse and validate
1. Strip any accidental markdown fences (`json … `).
2. `json.loads` the response; on failure, retry once with the LLM, attaching the parse error.
3. For each hypothesis, validate:
   - `hypothesis_id` matches `^h_\d{3}$` — renumber if not.
   - `constraints.solver` is non-empty (use a sensible default like `simpleFoam` for steady incompressible only if the LLM left it blank AND log a `hypothesis_default_filled` warning to timeline).
   - `supporting_papers` IDs all exist in `lit.json` — drop any that don't.
   - At least one quantitative constraint is set (Re / CFL / mesh_cells_max / endTime).
4. Discard any hypothesis missing required fields after one retry.

### Step 4 — Write
Write `<out-dir>/hypotheses.json` (indent 2). Append to timeline:
```json
{"stage": "hypothesis", "event": "complete", "ts": "<iso>", "hypotheses_count": <n>}
```

## Skip if already done
If `hypotheses.json` exists with ≥1 entry, skip and emit `hypothesis_skipped_existing`. Delete the file to force regeneration.

## Anti-hallucination rules
- `supporting_papers` MUST reference real `paperId`s from `lit.json` — do not invent IDs. If a hypothesis has no real support, leave the array empty.
- Do not contradict the topic. If the user said "RANS turbulence sweep at Re=5100", every hypothesis's `constraints.Re` must be 5100 unless the user is explicitly asking for a Reynolds-number sweep.
- Do not promise quantities the simulation cannot produce (e.g. instantaneous turbulent kinetic energy budget terms from a steady RANS run).

## Optional script fast-path
```bash
python scripts/hypothesis.py \
  --literature <out-dir>/lit.json \
  --topic "<topic>" \
  --output <out-dir>/hypotheses.json \
  --timeline <out-dir>/timeline.json
```
The script implements the same recipe; same artifact contract. Use whichever is more convenient for the calling agent's environment.

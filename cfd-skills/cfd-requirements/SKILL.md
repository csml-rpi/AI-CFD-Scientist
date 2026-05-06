---
name: cfd-requirements
description: Convert hypotheses (or a direct simulation request) into FoamAgent-ready user-requirement strings, one per case. Self-contained — embeds the IdeationAgent + HypothesisAgent expert prompts verbatim. Use after cfd-hypothesis (general flow) or directly when the user gave a simulation-only request. Outputs requirements.json.
---

# cfd-requirements

Produce exactly N concrete FoamAgent `user_requirement_text` strings, one per case. Each is a single self-contained English paragraph specifying geometry, solver, BCs, mesh, and run-control — directly executable by `cfd-experiment` / `cfd-foamagent-runtime` without further user input.

## Inputs
- `out-dir` (required)
- `n_cases` (optional, default 4) — exactly N. If user said "run X experiments", use X.
- One of:
  - `<out-dir>/hypotheses.json` exists (general flow), OR
  - `<out-dir>/state.json` has a `simulation_request` field (simulation-only flow)
- `<out-dir>/lit.json` (recommended for novelty checks; optional)

## Output
`<out-dir>/requirements.json` — JSON array:
```json
[
  {
    "case_id": "case_001",
    "experiment_id": "exp_001",
    "study_id": "<from study/topic>",
    "description": "<short>",
    "user_requirement_text": "<one English paragraph for FoamAgent>"
  }
]
```

## Recipe (primary, agent-driven)

### Step 1 — Build the study skeleton
1. If `hypotheses.json` exists, load it. Otherwise read `simulation_request` from `state.json`.
2. Read the topic (from `state.json` or the calling skill's argument).
3. Read `lit.json` (top ~10 by citation) to ground novelty.

### Step 2 — Render and call the IdeationAgent (study generation)

Use the embedded prompts below verbatim. They are exact copies of `IdeationAgent` in `prompts/prompts.yaml`. The first two are the standard path; the third is for re-generation when the first attempt is too similar to known prior work.

#### System prompt (from `prompts/prompts.yaml: IdeationAgent.initial_idea_prompt`)

```
You are an experienced AI researcher in Computational Fluid Dynamics (CFD).
Your job is to propose ONE impactful but realistically executable CFD research idea.

IMPORTANT: You will be given a literature summary from Semantic Scholar/OpenAlex/arXiv/web.
You MUST synthesize prior studies, identify what is known and what is missing,
then propose a non-overlapping set of experiments that is novel and feasible.

Constraints:
- Output MUST be valid JSON and return ONLY JSON (no prose, no markdown).
- Experiments must be non-overlapping (no duplicates in parameter/setup tuple).
- Total experiments must be <= {max_experiments} (can be less).
- Use SI units and concrete values (no TBD placeholders).

Return ONLY JSON in this GENERIC schema:
{
  "study_id": "string",
  "description": "string",
  "solver": "string",
  "target_CFL": number,
  "objective": "string",
  "experiments": [
    {
      "experiment_id": "exp_001",
      "name": "string",
      "topology": "string (2d|3d)",
      "dimensions": [number, number, number],
      "parameters": {
        "<variable_1>": number|string,
        "<variable_2>": number|string
      },
      "controls": {
        "end_time": number,
        "write_interval": number
      },
      "notes": "string"
    }
  ],
  "post": {
    "objective": "string"
  }
}
```

#### User prompt — initial (from `prompts/prompts.yaml: IdeationAgent.literature_aware_user_prompt`)

```
Research topic: {research_topic}

PRIOR STUDIES (retrieved from literature/web):
{literature_context}

Instructions:
1) Reason about patterns in prior work and identify at least 2 concrete gaps.
2) Propose ONE idea as a non-overlapping experiment set.
3) Ensure proposal is not a duplicate of listed prior studies.
4) Number of experiments must be <= {max_experiments}.
5) Output ONLY valid JSON matching the required schema.
```

#### User prompt — novelty retry (from `prompts/prompts.yaml: IdeationAgent.novelty_retry_user_prompt`)

```
Your previous idea appears too similar to prior studies or has overlapping/too many experiments.

PRIOR STUDIES:
{literature_context}

Previous idea (for reference only, do not repeat):
{previous_idea}

Generate a NEW non-overlapping experiment set with clear novelty and <= {max_experiments} experiments.
Output ONLY valid JSON in the required schema.
```

### Step 3 — Run the call

1. Render `literature_context` as in `cfd-hypothesis`: the top ~10 papers as `[paperId] Title (year, cit=N)\n  Abstract: <abstract>`.
2. Render the system prompt with `{max_experiments}` = `n_cases` (the IdeationAgent treats `<= max_experiments` as the cap; we will enforce *exactly* `n_cases` in step 5).
3. If `hypotheses.json` exists, prepend a short `Hypothesis context: <list of hypothesis statements>` paragraph to the user message so the study aligns with the hypotheses.
4. Send to LLM. Expect a JSON object with `study_id`, `description`, `solver`, `target_CFL`, `objective`, `experiments` (array), `post`.
5. If parse fails, retry once with the parse error appended. If `len(experiments) != n_cases`, do **one** retry with the novelty_retry_user_prompt to coax the right count. After that, truncate or pad (see step 5).

### Step 4 — Expand each experiment to a FoamAgent requirement paragraph

For each experiment in the study, send a follow-up call using the `HypothesisAgent.hypothesis_system_prompt` + `hypothesis_user_prompt` (embedded verbatim in `cfd-hypothesis/SKILL.md` — see that file for the exact text; do not re-embed here to avoid drift). The expansion turns each compact experiment object into a single self-contained English paragraph for FoamAgent.

Variables to fill in `hypothesis_user_prompt`:
- `{run_topic}` — the user's topic
- `{study_id}` — from study JSON
- `{description}` — from study JSON
- `{case_name}` — derive: `case_001`, `case_002`, …
- `{simulation_id}` — `experiment_id` from study JSON
- `{parameter_values}` — `experiment.parameters` rendered as `key=value, key=value`
- `{simulation_description}` — `experiment.notes` or `experiment.name`
- `{all_experiment_ideas}` — short list of every experiment's name + parameters (for cross-case consistency)
- `{current_experiment_idea}` — full JSON of *this* experiment
- `{previous_requirement}` — the requirement text from the previous case (empty for case_001)

Run case_001 first, then feed each output back as `{previous_requirement}` for the next; this preserves stylistic consistency across the case set (the Python pipeline does the same).

### Step 5 — Enforce exactly N cases
- If the model produced `< n_cases`, raise an error and re-call `IdeationAgent.literature_aware_user_prompt` with `{max_experiments}` set higher; truncate the response back to `n_cases`.
- If the model produced `> n_cases`, keep the first N (in JSON order — this matches the Python pipeline).
- Hard-fail rather than silently produce a different count. The user said "X experiments" → ship X.

### Step 6 — Sanity-check each requirement
For each generated paragraph, verify:
- Mentions a concrete solver (`simpleFoam`, `pimpleFoam`, `icoFoam`, …).
- Mentions concrete BCs for every patch implied by the geometry (inlet, outlet, walls, frontAndBack for 2D).
- Mentions cell count (or nx×ny×nz) — this is required by FoamAgent.
- Mentions `endTime`, `deltaT`, `writeInterval`.
- Does **not** contain any `Visualize…` instruction.
- Respects topic constraints: if the topic said "≤10000 cells", the requirement's cell count is ≤10000; if it said "CFL ≤ 0.5", the deltaT/cell-spacing combo satisfies it.

If any check fails, regenerate that one requirement with the failed constraint listed in `{previous_requirement}` as a "DO NOT VIOLATE: …" line.

### Step 7 — Write `requirements.json`
Each entry: `{case_id, experiment_id, study_id, description, user_requirement_text}`.

Append to timeline:
```json
{"stage": "requirements", "event": "complete", "ts": "<iso>", "n_cases": <n>}
```

## Skip if already done
If `requirements.json` exists with `>= n_cases` entries, skip with `requirements_skipped_existing`. Delete the file to regenerate.

## Constraints
- Number of cases must match `n_cases` exactly. If user said "5 experiments", produce 5 — not 4 or 6.
- Each case must be **executable as-is** by FoamAgent without further user input.
- If `starter-dir` is present, the requirement may say "based on the starter case at `starter/`" — but it must still specify any deviations explicitly.

## Anti-hallucination rules
- Do not invent fluid properties beyond the topic. If the topic doesn't specify ρ or ν, use water-at-20°C (`ρ=1000, ν=1e-6`) for incompressible / air-at-20°C (`ρ=1.225, ν=1.5e-5`) for external flow, and tag with `#ASSUMPTION`.
- Do not invent benchmark data values inside the requirement text. The requirement is for FoamAgent to *run* the simulation, not to repeat known answers.
- Do not contradict the user's stated mesh size or CFL constraint, ever.

## Optional script fast-path
```bash
python scripts/requirements.py \
  --hypotheses <out-dir>/hypotheses.json \
  --output <out-dir>/requirements.json \
  --n <n_cases> \
  --timeline <out-dir>/timeline.json
```
Same artifact contract. Uses the same prompts internally.

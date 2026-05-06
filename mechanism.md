# CFD Scientist — Pathway Mechanism

This document is the canonical spec for how cfd-scientist should behave end-to-end. It supersedes any prior assumptions in code or prompts. Any change to orchestrator behavior must respect this mechanism.

## 1. Three nested pathways

There are three pathways. They nest, with the inner ones reusable by the outer:

```
[ OED (open-ended discovery)              ]   outermost
  └─[ code_mod                            ]   middle (used by OED + standalone)
       └─[ pure parameter sweep           ]   innermost (used by code_mod, OED, + standalone)
```

- **pure parameter sweep** — innermost. Run experiments on a single (possibly built-in) model, varying physical/numerical parameters, fitting a correlation or characterizing a trend. No model code change. No discovery loop.
- **code_mod** — middle. User provides equations / model change. System hypothesizes, implements, compiles, tests the change. After the change is healthy it falls through to a parameter sweep that *demonstrates* the modified model for the paper.
- **OED (open-ended discovery)** — outermost. Many candidate ideas explored under a budget. Once a winner is identified, fall through to code_mod (implement+compile+test the winner), then to a parameter sweep that *demonstrates the discovered model's superiority* for the paper.

The orchestrator picks the pathway by asking an LLM to classify the topic at the start of the run. The classifier already exists (`_llm_classify_topic_mode`) and supports `pure_sweep | code_mod | mesh_focus | oed-style (standard for now)`. Keep that classifier as the single routing gate; downstream stages must trust the chosen pathway.

## 2. Starter handling — single rule

Starter contents may include (a) PDFs / gradients / support materials, (b) a baseline OpenFOAM case, (c) both, or (d) nothing.

- **If starter exists**: ingest it once. Build `starter_understanding.json` (PDFs, params, geometry, formula, reference data) and the `starter_case_seed`. All downstream stages may read these.
- **If starter does not exist** (`--no-starter` or empty starter dir): **do not read, fabricate, or synthesize any starter content.** Skip the starter-understanding LLM call. Skip `starter_study_brief` and any starter-flavored stage. Do not write a placeholder file. Hypothesis stages run on topic + lit only.
- **If any pathway needs a baseline case but starter is absent** (pure_sweep / code_mod / OED with `--no-starter`, or starter that has no runnable case): the orchestrator runs the **`baseline_synthesis`** stage (see §3.1). It distils a canonical-baseline requirement from topic + first synthesised requirement and asks Foam-Agent (RAG over OpenFOAM tutorials + LLM scaffolding + precheck/run/fix) to produce `<run_dir>/canonical_base_case/`. That directory then serves the same role a starter case would: it is the seed for mesh-gate (and, in code_mod runs, the host directory for `customModels/`).

This rule is one-line: starter ingestion is gated by starter presence; no other branch may second-guess it.

## 3.1 baseline_synthesis — bridges the no-starter gap

Stage that runs **after `requirements`** and **before `code_mod` / `mesh_gate`**, only when:

- `<run_dir>/canonical_base_case/system/controlDict` does NOT exist, AND
- `state.starter_seed_case_dir` does NOT point to a real case, AND
- `--disable-mesh-gate` is not set (no seed needed if mesh-gate is skipped).

Behaviour:

1. **Distil a canonical-baseline requirement.** LLM call: topic + first sweep requirement → a single representative case spec (one mid-Re point, simplest setup, built-in models only, ~5k–9k cells default, transient if topic requires). Persisted at `<run_dir>/canonical_baseline_requirement.txt`.
2. **Hand off to Foam-Agent.** Call `scripts/foam_run.py` with that requirement, output to `<run_dir>/canonical_base_case/`. Foam-Agent already does FAISS-RAG match → scaffold `0/`, `constant/`, `system/` → precheck → run → fix loop. No new RAG plumbing needed in cfd-scientist.
3. **Verify.** Confirm `system/controlDict` exists. On success, write `state.canonical_base_case_dir`. On failure, log a clear warning (`mesh_gate may stub out`) and let mesh-gate's existing empty-selection path handle it; don't silently mask.
4. **Idempotent.** Returns immediately if a starter case is present or canonical_base_case already exists (e.g. on resume).

Mesh-gate's `resolve_mesh_seed_path` already checks `<run_dir>/canonical_base_case/system/controlDict`, so no mesh-gate code changes are needed — it picks the synthesised baseline up automatically.

## 3. Mesh-gate — runs before any paper-grade experiments

Mesh-gate is a mandatory pre-experiment study. Its job: find the converged mesh resolution for the configuration that downstream experiments will use.

- Mesh-gate runs **before** any parameter-sweep / paper experiments.
- Each **model** carries its own mesh-independence selection. Pure sweep, code_mod, and OED variants each anchor mesh-gate on the model they are about to run.
  - pure sweep: mesh-gate runs once on the baseline (built-in) model.
  - code_mod: mesh-gate runs **after** the modified model is compiled and verified, with that model loaded.
  - OED: mesh-gate runs **after** the discovered winning model is implemented via code_mod, with that model loaded.
- The selected mesh from mesh-gate becomes the base mesh for all subsequent experiments using that model.
- Mesh-gate context is informational ("selected mesh level / cell count / y+ / quality") and must never be packaged as a code-mod or "custom-compiled-model" signal to downstream LLMs.

## 4. Hypothesis is multi-stage

The hypothesis agent runs **more than once** in code_mod and OED pathways. Each call is fed only the context that actually exists at that point — never more, never less.

| Stage | Pathway(s) | Context fed | Output |
|---|---|---|---|
| H1 — initial | all | topic + literature + (starter if present) | first ideas / candidate experiments |
| H2 — post code_mod | code_mod, OED | + code_mod_context (real implementation only) | experiments that exercise the modified model |
| H3 — post mesh_gate | code_mod, OED, pure_sweep | + mesh_gate_context (selected mesh) | experiments fixed to the converged mesh |
| H4 — post OED | OED only | + OED winner context | experiments that demonstrate winner's superiority |

Rules for hypothesis context assembly:

- Each context block is a separate input. **Never merge starter / code_mod / mesh_gate into one anonymous "context" string** that gets dressed up as a code-mod preamble downstream. (This was the leak that caused gpt-5.5 to invent `transport_model: compiled_custom_laminar_newtonian` in pure_sweep.)
- The "a custom OpenFOAM model has already been implemented and compiled" preamble may only fire when there is an **actual** code-mod artifact on disk (compiled `.so`, source under `customModels/`) AND `mode == code_mod` (or `oed` post-implementation).
- Mesh-gate context is rendered as plain mesh selection text. It does not imply a custom model.
- Hypothesis must read prior stage outputs and update experiments accordingly — not regenerate from scratch each call.

## 5. Pathway flow detail

### 5.1 pure parameter sweep

```
classify -> pure_sweep
↓
[skip starter stages if --no-starter]
↓
literature
↓
H1(topic + lit + starter?)
↓
mesh_gate (baseline model)
↓
H3(... + mesh_gate)
↓
experiments (Foam-Agent scaffold + run, no code change)
↓
interpret → analyze → paper
```

### 5.2 code_mod

```
classify -> code_mod
↓
literature
↓
H1(topic + lit + starter?)
↓
[if starter has baseline → use it; else system synthesizes minimal baseline]
↓
code_mod loop: implement equations → compile → test → iterate
↓
[code_mod_context now real]
↓
mesh_gate (with the modified model loaded)
↓
H3(topic + lit + starter? + code_mod_context + mesh_gate)
↓
parameter sweep experiments to demonstrate the modified model
↓
interpret → analyze → paper
```

### 5.3 OED

```
classify -> oed
↓
literature
↓
H1(topic + lit + starter?)
↓
OED loop (under budget): propose model variants, score, iterate
↓
[OED winner identified]
↓
code_mod loop: implement winner → compile → test
↓
[code_mod_context + OED context now real]
↓
mesh_gate (with the discovered model loaded)
↓
H4(topic + lit + starter? + code_mod_context + mesh_gate + OED winner)
↓
parameter sweep experiments to demonstrate winner's superiority
↓
interpret → analyze → paper
```

## 6. Anti-clash rules across pathways

These are invariants the code must enforce. Violating any of them is the bug class that caused the recent `compiled_custom_laminar_newtonian` and `custom_laminar_jet` leaks.

1. **pure_sweep must never receive a code_mod preamble.** No "a custom model has been compiled" framing, no `customModels/` paths in the LLM context, no `transport_model: compiled_custom_*` framing.
2. **The `IMPORTANT — A custom OpenFOAM model has already been implemented` system-prompt block fires only when a real code-mod artifact exists.** Not "when context is non-empty."
3. **Mesh-gate context is mesh selection only.** It must not be passed in as `code_mod_context` to any downstream LLM.
4. **Starter context only injects when starter actually exists.** `--no-starter` is binary: nothing starter-flavored leaks anywhere. No `starter_study_brief`, no `starter_understanding.json`, no `starter_case_seed` reads.
5. **Routing decision is authoritative.** Once `_llm_classify_topic_mode` picks a pathway, downstream stages must trust it. No silent re-routing based on artifact presence.
6. **Hypothesis stages compose context, not concatenate it.** Separate fields (`starter_ctx`, `code_mod_ctx`, `mesh_gate_ctx`, `oed_ctx`) — never one combined blob with one preamble.
7. **No downstream "sanitizer" is allowed to do the job of correct context routing.** Filtering for forbidden keys after the fact is a bandaid. Fix the prompt so the LLM is never told to emit them.

## 7. What's already correct — keep as-is

These pieces are working and should not be churned:

- `_llm_classify_topic_mode` correctly classifies into `code_mod | mesh_focus | pure_sweep | standard`.
- The code-mod loop (`scripts/code_mod_agentic.py`) — implement, compile, test — works.
- Mesh-independence protocol (near-wall ~10%, away ~5%, 5% QoI threshold, GCI escalation) per `skills/cfd-mesh-independence/SKILL.md`.
- Foam-Agent scaffolding/RAG/precheck/runtime sequence per `skills/cfd-foamagent-runtime/SKILL.md`.
- OED budget extension with hard 1.5× ceiling (`_BUDGET_HARD_CAP_MULTIPLIER`).
- Sandbox `error_summary` surfacing real gcc errors to the LLM.
- OED loop's 3-attempt parse-failure retry.
- `--no-starter` flag plumbing through `_build_execution_plan` and the per-stage gates.
- Provider routing (`openai-codex` OAuth via `~/.codex/auth.json`, `claude-code`, `claude-sonnet-4-6`).

## 8. What's broken vs this spec — fix list

Concrete deltas vs current code, in priority order. Each is a separate change so a regression bisects cleanly.

1. **Decompose `combined_ctx` in `scripts/hypothesis.py:155–225`.** Pass `starter_ctx`, `code_mod_ctx`, `mesh_gate_ctx` as **named** parameters into `run_literature_aware_ideation`. Stop renaming `combined_ctx` to `code_mod_context` in the call.
2. **Gate the "custom model already implemented" preamble** (`hypothesis.py:68–76`) on `code_mod_ctx` AND `mode in {code_mod, oed_post_impl}` — not on the merged blob being non-empty.
3. **Render mesh_gate as a separate user-prompt block** ("Selected mesh: …"), not as part of `code_mod_context`.
4. **Remove the `_CODE_MOD_ONLY_KEYS` sanitizer** in `orchestrator_run.py:2098–2114` once 1–3 are in. Sanitizers are bandaids; the prompt should never have asked for those keys.
5. **`starter_understanding.json` build is fully gated by starter presence.** Verify no path writes it under `--no-starter`. (Currently gated; just verify after 1–4.)
6. **Hypothesis stage 2/3/4 plumbing.** Make sure the orchestrator can re-invoke hypothesis with growing context after code_mod completes, after mesh_gate completes, and after OED completes — instead of one monolithic call up front. Code_mod and OED pathways need this; pure_sweep needs only H1 + H3.
7. **OED → code_mod handoff.** When OED picks a winner, package the winner spec into a code_mod input and let the code_mod loop run. Keep OED context as a separate named field for H4.

## 9. Verification

After any change to orchestration / hypothesis / mesh-gate / starter handling, run all three pathways end-to-end as smoke tests:

- pure_sweep: a clean topic with `--no-starter` (e.g. the planar jet Re sweep). Expect: no `customModels/` directory, no `transport_model: compiled_custom_*` anywhere, sweep finishes.
- code_mod: a topic that says "implement this Carreau viscosity model" with starter. Expect: code-mod loop runs, compiles, mesh-gate runs with the new model, sweep demonstrates it.
- OED: a topic asking to discover a new SA modification with starter. Expect: OED iterates under budget, picks winner, code_mod implements it, mesh-gate runs, paper sweep demonstrates superiority.

If any of the three smoke runs leaks fields or stages from the wrong pathway, the change has violated section 6.

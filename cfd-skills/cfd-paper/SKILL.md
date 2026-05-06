---
name: cfd-paper
description: End-to-end paper writing — figure planning → batch PyVista figure generation with per-image VLM QA → LaTeX drafting → compile → reviewer → revise. Self-contained — embeds WriterAgent (system + user), PaperReviewerAgent (review system + user, compile_fix system + user), citation system + user, and vision QA prompts verbatim. Up to max_review_loops outer iterations. Consumes analysis.json + lit.json; produces paper/main.pdf, paper_figs/, review.json.
---

# cfd-paper

The paper-writing pipeline is **a loop**, not a single pass. Each outer iteration plans figures, generates them, validates them with a vision model, drafts/revises the LaTeX, compiles, runs a reviewer agent. It exits when the reviewer passes or after the iteration cap.

## Inputs
- `out-dir` (required) — must contain `analysis.json`, `lit.json`, and case dirs
- `template` (optional, default `neurips`) — `neurips`, `iclr`, `icml`, `ieee`, `acm`, `arxiv`
- `score_threshold` (optional, default 0.7) — reviewer score required to pass
- `max_review_loops` (optional, default 10)

## Outputs
- `<out-dir>/paper_unified_plan.json` — figure jobs + section plan (planner output)
- `<out-dir>/paper_figs/` — PNGs + `cases_config.json` manifest
- `<out-dir>/paper/main.tex` (and per-section files under `<out-dir>/paper/sections/` if you split)
- `<out-dir>/paper/references.bib`
- `<out-dir>/paper/paper_draft.pdf` (final compiled PDF)
- `<out-dir>/review.json` — reviewer agent's last verdict

## Recipe (primary, agent-driven outer loop)

The outer loop iterates `r = 0..max_review_loops`. Each iteration: plan → batch viz → image QA → write → compile → review.

### Iteration 0 — first pass

#### Step 1 — Plan

Produce `<out-dir>/paper_unified_plan.json`:
```json
{
  "title": "...",
  "sections": ["Abstract", "Introduction", "Methods", "Mesh independence", "Results", "Discussion", "Conclusion", "References", "Appendix"],
  "figure_jobs": [
    {
      "spec_id": "U_panel",
      "description": "Side-by-side velocity-magnitude contour for cases 1 and 3 at final time",
      "cases": ["case_001", "case_003"],
      "field": "U",
      "representation": "contour",
      "panel_layout": "1x2",
      "claim_supported": "Refined wall treatment reduces shear-layer thickness by ~12%",
      "expected_caption": "..."
    },
    {
      "spec_id": "Cf_vs_dns",
      "description": "Cf along lower wall vs DNS, all PROCEED cases overlaid",
      "cases": ["case_001", "case_002", "case_003"],
      "field": "Cf",
      "representation": "line_overlay",
      "claim_supported": "..."
    }
  ],
  "claim_to_evidence": {
    "claim_1": ["fig:U_panel", "tab:metrics"],
    "claim_2": ["fig:Cf_vs_dns"]
  },
  "include_mesh_subsection": true,
  "experiment_summary_table": [
    {"case_id": "case_001", "label": "case 1", "varying": "wall function vs low-Re", "cells": 9200},
    ...
  ]
}
```

The plan is the contract for the rest of the pipeline. Build it from `analysis.json`, `lit.json`, and (when present) `mesh_independence_context.json`.

#### Step 2 — Batch viz + image QA

Invoke `/cfd-viz mode=full out-dir=<out-dir> paper_plan=<out-dir>/paper_unified_plan.json case_dirs=...`. That skill embeds the `viz_quality_*` prompts and runs the per-image vision-QA loop until each figure passes (max 10 inner attempts per figure).

After completion, `<out-dir>/paper_figs/` contains validated PNGs + `cases_config.json` listing each figure's `(spec_id, path, cases, field, time)`.

#### Step 3 — Image-aware narrative analysis

For each generated figure, call the vision LLM with the embedded `vision_*` prompts (already documented in `cfd-interpret/SKILL.md`) to produce a one-paragraph narrative observation. Save to `<out-dir>/paper_figs/<spec_id>_analysis.txt`. The Writer pulls these for Results and Discussion section text.

#### Step 4 — Citation enrichment

The bibliography needs to draw from `lit.json` AND from venue-specific best-known references (DBLP / CrossRef-enriched). Use the embedded prompts below to iteratively gather citations.

##### System prompt (from `prompts/prompts.yaml: WriterAgent.citation_system_prompt`)

````
You are an expert research assistant helping to collect relevant citations for a CFD research paper.

Your task is to identify and suggest relevant academic papers that should be cited in the research paper.
Focus on papers that are directly related to the research topic, methodology, or background.

GUIDELINES:
- Suggest papers that are highly relevant to the research topic
- Include foundational papers in the field
- Include recent papers that show current state-of-the-art
- Focus on papers from reputable journals and conferences
- Avoid suggesting papers that are too general or unrelated

RESPONSE FORMAT:
Return your response in JSON format with the following structure:
```json
{
  "action": "add" or "stop",
  "citations": [
    {
      "title": "Paper Title",
      "authors": "Author Names",
      "year": "Year",
      "venue": "Journal/Conference",
      "doi": "DOI if available",
      "relevance": "Brief explanation of relevance"
    }
  ]
}
```

Use "action": "stop" when you believe sufficient citations have been collected.
````

##### User prompt (from `prompts/prompts.yaml: WriterAgent.citation_user_prompt`)

```
You are collecting citations for a CFD research paper.

RESEARCH CONTEXT:
{citation_context}

ROUND: {round}/{total_rounds}

Based on the research context above, suggest relevant papers that should be cited.
Focus on papers that are directly related to the research methodology, background, or results.

Consider:
1. Foundational papers in CFD and OpenFOAM
2. Papers related to the specific problem being studied
3. Recent papers showing current state-of-the-art
4. Papers that validate or compare with your methodology

Return your response in the specified JSON format.

```

Render `{citation_context}` as: topic + analysis summary + figure list + initial bibliography from `lit.json`. Run for up to `total_rounds=3` rounds; merge the suggestions into a candidate bibliography. Then verify each via DOI/CrossRef (skip the verification on the first pass if running in air-gapped mode; flag for the user).

Write `<out-dir>/paper/references.bib` with at least **20 distinct, non-duplicate** references (the `WriterAgent.system_prompt` requires this minimum).

#### Step 5 — Write LaTeX (Writer call)

Use the embedded prompts below verbatim. They are exact copies of `WriterAgent.*` in `prompts/prompts.yaml`.

##### System prompt (from `prompts/prompts.yaml: WriterAgent.system_prompt`)

```
You are a LaTeX PAPER-WRITER for CFD research results. Your job is to produce professional, publication-ready LaTeX documents that compile to PDF.

LENGTH: Main body (Abstract through Conclusion, excluding References and Appendix) must be at least 8 pages and at most 15 pages.

SCOPE AND TRUTHFULNESS (critical):
- The paper must reflect ONLY what was done in the provided experiments. Nothing more, nothing less. No hallucinations.
- Do NOT mention standard literature or theory that was not actually performed or validated in these experiments. If something is common in the field but not done here, omit it.
- Analysis, Discussion, and Conclusion must be grounded strictly in the given visualizations and experiment data. Every claim must map to a specific figure, table, or number from the provided analysis. No unsupported or speculative claims.

FIGURES:
- Include only figures that are good quality and clearly support the text. If an image is poor (blurry, wrong, or uninformative), do not include it.
- Prefer figures suitable for journal print: colorbars and legends must not cover the plotted flow/domain; labels and ticks must be large enough to read when the figure is placed at typical column width. Omit or replace figures that fail these layout checks (common with default PyVista exports).
- You MUST include at least one important/representative figure from each experiment so all experiments are represented. It is not required to include every image from every experiment, but each experiment must have at least one figure.
- For domain-wide contour or field plots, ensure the camera/framing would be readable at journal-print size: the full computational domain should be visible, and key flow features (recirculation bubble, jet, shear layer, bluff body, etc.) must be large enough to interpret.

CORE RESPONSIBILITIES:
1. **LaTeX Document Generation**: Create complete LaTeX documents with proper structure and formatting
2. **Scientific Writing**: Write clear, concise, and technically accurate content based only on the experiments
3. **Figure Integration**: Include only good-quality figures with informative captions; ensure every experiment is represented
4. **Bibliography Management**: Handle citations and references appropriately
5. **Reproducibility**: Include methodology and simulation parameters that were actually used

MESH REFINEMENT / INDEPENDENCE (mandatory when data is provided):
- The section context JSON may include a non-empty key `mesh_independence` from the automated mesh-gate study.
- When `mesh_independence` is present and contains `metrics_by_mesh_level` or `mesh_gate_plan`, you MUST add a clear subsection (typically under Methods, or standalone before Results) that:
  (1) lists every mesh level compared (coarse, baseline, refined, …) using `mesh_gate_plan` and/or folder names in `metrics_by_mesh_level`;
  (2) presents at least one **LaTeX table** of cell/point counts and monitored QoIs per level using only numbers from `metrics_by_mesh_level` (no invented values);
  (3) states which mesh level was **selected** for all subsequent simulations (`selected_stable_name`, `selected_level_path`) and the **criterion** used (e.g. percent change in QoIs between successive refinements, as described in `selection_note` or `cross_mesh_analysis_excerpt`);
  (4) summarizes conclusions from `cross_mesh_analysis_excerpt` where it does not contradict the table;
  (5) includes `\\includegraphics` for mesh plots when `mesh_figure_paths` is non-empty, using the same path style as other figures in the context `figures` list (typically absolute paths to PNGs under the run directory).
- If `mesh_independence` is absent or empty, do not fabricate a mesh study; you may give one sentence that mesh-convergence details were not supplied in this bundle.

WRITING GUIDELINES:
- Use ONLY data from provided JSON bundles and figures; do NOT invent numbers, results, or citations
- Emphasize what was actually simulated (geometry, BCs, solver, mesh, etc.)
- Report only what is evident from the provided analysis and visualizations
- Write in clear, academic style
- Do not discuss theory or literature that was not applied or validated in these experiments
- Do not use the word "formulation" or similar jargon in the paper title; keep the title concise and descriptive.
- When introducing any abbreviation (e.g. RANS, BFS, SA), write out the full term with the abbreviation in parentheses at first mention, and then use the abbreviation consistently thereafter.
- Avoid putting heavy mathematical formulations or long equations in the abstract; the abstract should be primarily qualitative and high-level.
- Use a consistent list style and formatting throughout the paper (same bullet/numbering and indentation rules in all sections).
- Keep figure captions concise and focused on what is shown and why it matters; avoid turning captions into long paragraphs.
- Do not overuse lists, bold, or italics in the main text; reserve them for truly important emphasis so the manuscript remains readable and professional.
- Before referring to experiments by shorthand IDs (exp_001, exp_002, etc.), introduce a clear experiment-summary table that lists each experiment (ID, high-level description, key varying parameter such as turbulence model, wall treatment, and total cell count). Reference this table in the text so that readers know what each experiment label means.
- Avoid duplicating the same explanatory paragraph or numerical comparison in multiple sections: fully explain important caveats (e.g. geometry mismatch, post-processing artefact) once in the most appropriate section, then refer back briefly rather than repeating the explanation verbatim elsewhere.
- The Conclusion section must be a single cohesive paragraph (continuous prose). Never format the Conclusion as bullet points, numbered items, or any list.

LaTeX REQUIREMENTS:
- Use standard LaTeX packages (amsmath, graphicx, subcaption, etc.)
- Proper document structure with sections, subsections, and references
- Professional formatting with consistent spacing and typography
- Proper figure placement and referencing
- Mathematical equations only for relations actually used or verified in the experiments
- Bibliography in standard format (BibTeX or manual)
- Do NOT include a table of contents (no \\tableofcontents or “Contents” section); typical CFD journal papers do not use a TOC in the main manuscript.
- References section must contain at least 20 distinct, non-duplicate references, drawn from the provided Semantic Scholar / web literature bundles and candidate citations.

ABSTRACT (strict):
- Do not name individual case numbers (no “case 1”, “case_001”, exp IDs) in the Abstract.
- Do not refer readers to section numbers or internal labels (no “see Section X”) in the Abstract.

CASE NAMING IN BODY:
- Never use `case_001`-style identifiers in prose. Use “case 1”, “case 2” (with a space) after you have introduced what each case represents in ordinary language first.
- If `unified_paper_rules` in section context lists `omitted_case_ids`, do not mention those experiments at all (no “duplicate omitted” language).

OUTPUT FORMAT:
- Complete LaTeX document (.tex file)
- Main body: 8–15 pages, excluding References and Appendix
- Proper document class and package imports
- Figures: only good-quality images; at least one from each experiment
- Bibliography section included
- Document must compile without errors

```

##### User prompt (from `prompts/prompts.yaml: WriterAgent.user_prompt`)

```
You are writing a specific section of a LaTeX research paper for OpenFOAM CFD simulation results.

SECTION CONTEXT:
{section_context}

REQUIREMENTS:
1. Write content for the specified section only
2. Use proper LaTeX formatting and syntax
3. Include relevant equations, figures, and citations where appropriate
4. Ensure scientific accuracy and completeness
5. Focus on the specific content needed for this section
6. Use clear, academic writing style
7. Include proper mathematical notation when needed
8. Reference figures and tables appropriately

WRITING GUIDELINES:
- Use ONLY data from provided experiment data and analysis results
- Emphasize what was actually simulated (geometry, BCs, solver, mesh, CFL, turbulence model)
- Report metrics with proper units and effect sizes
- Write in clear, academic style suitable for CFD/engineering publications
- Include comprehensive methodology details for reproducibility
- Acknowledge limitations and assumptions clearly

Return ONLY the LaTeX content for this section, without any markdown formatting or code blocks.

```

The Writer is **section-by-section**: invoke once per section in `paper_unified_plan.sections`. For each, build `{section_context}` containing:
- `section_name` (e.g. "Methods")
- `topic`
- the paper plan (so the section knows what the rest of the paper claims)
- `analysis.json`
- `manifest.json` (case statuses)
- For "Mesh independence" / "Methods": `mesh_independence_context.json` (if present)
- `figures`: list of figure paths the section may use, with their captions
- `references`: the candidate bibliography
- `unified_paper_rules`: any case-omission policy
- `experiment_summary_table` (from the plan)

Stitch the section outputs into `<out-dir>/paper/main.tex` with the venue-specific document class header (`\documentclass{article}` for arxiv, `\documentclass[10pt]{neurips_2024}` for neurips, etc.).

#### Step 6 — Compile

```bash
cd <out-dir>/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If pdflatex returns non-zero, run the **compile-fix loop** with the embedded prompts below.

##### Compile-fix system prompt (from `prompts/prompts.yaml: WriterAgent.compile_fix_system_prompt`)

```
You are a LaTeX build engineer. The manuscript below failed pdflatex. Your ONLY goal is to output a complete, valid LaTeX document that compiles.

Rules:
- Do NOT run a content or publishability review. Do not shorten, restructure for style, or rewrite for clarity unless required to fix the error.
- Make the minimum edits needed: wrong \\includegraphics paths (use ONLY paths from the valid list), missing packages in the preamble, unclosed environments, bad escape sequences, undefined commands, broken \\ref/\\cite, or missing \\end{document}.
- Preserve scientific content and structure when possible; prefer surgical fixes over rewriting sections.
- Return the ENTIRE document as valid LaTeX — not a patch or excerpt.

```

##### Compile-fix user prompt (from `prompts/prompts.yaml: WriterAgent.compile_fix_user_prompt`)

```
pdflatex failed. Fix the LaTeX so it compiles.

PRIMARY / PARSED ERRORS (fix in order):
---
{error_summary}
---

RAW LOG (tail; full context):
---
{compile_error_tail}
---

VALID paths for \\includegraphics (relative to project root / compile cwd when listed):
---
{valid_figure_paths}
---

CURRENT FULL LaTeX:
---
{tex_content}
---

Return ONLY the complete corrected LaTeX document (no markdown, no explanation).

```

Apply the returned full document, recompile. **Max 3 compile-fix attempts**. After that, return failure to the outer loop.

#### Step 7 — Review

Use the embedded reviewer prompts below verbatim.

##### System prompt (from `prompts/prompts.yaml: PaperReviewerAgent.system_prompt`)

```
You are an expert academic paper reviewer for CFD and engineering journals. You operate in two modes depending on compilation outcome.

**When compilation FAILED:** Your job is to fix the LaTeX so it compiles. The error summary below contains the PRIMARY errors from pdflatex — focus on the FIRST error listed; it is usually the root cause. Common causes: (1) "File X not found" — the \\includegraphics path is wrong; use ONLY paths from the VALID figure paths list if provided; (2) missing \\end{{document}}; (3) undefined control sequence; (4) unclosed environment. Do NOT assume the file is truncated unless the error explicitly says so. Give 1–3 concrete recommendations that fix the PRIMARY error. Set pass=false.

**When compilation SUCCEEDED:** Your job is to evaluate the paper for publication readiness. Check: (1) Formatting — structure, \\ref/\\cite, typography; ensure there is NO table of contents (no \\tableofcontents or “Contents” section), as typical CFD journal manuscripts do not include one; (2) Figures — paths valid, only good-quality images included, at least one figure from each experiment, captions and labels correct, referenced in text; when you can inspect figures or PDF previews, also check publication layout: colorbars/legends must not obscure plotted data, and axis/colorbar text must be legible at typical print size (flag figures_ok=false and recommend regenerating plots if not); figures for long thin channels should use horizontal/wide layout if they waste vertical space; reject matplotlib-looking contour junk when the pipeline asked for PyVista paper figures; (3) Content — reflects only the experiments, no hallucinations, analysis/discussion/conclusion grounded in the given visualizations; (4) **Abstract hygiene** — FAIL if the Abstract names specific case numbers (e.g. case 1, case_001) or points to section numbers (“see Section 3”); the Abstract must stand alone; (5) **Case naming** — FAIL if the text uses `case_001`-style IDs in prose; narrative should use “case 1”, “case 2” after introducing what each case is; (6) Coherence — flow, conclusions supported by evidence; (7) Length — main body 8–15 pages (excluding References and Appendix); flag if too short or too long; (8) References — prefer a solid bibliography when citations exist; if the manuscript intentionally uses a short reference list, note it but do not fail solely on count; (9) Redundancy — watch for duplicated paragraphs; (10) Conclusion format — single paragraph, not a list; (11) Publishability. Set pass=true only if all are acceptable; otherwise give specific recommendations.

**Unified pipeline / extra figures:** If figures are missing, wrong orientation, or overlap legends, set `needs_additional_visualization` true and add short imperative strings to `additional_viz_specs` (each should mention a `case_NNN` id and what to plot, e.g. “case_003: horizontal PyVista Ux profile mid-channel, colorbar below plot”).

**Unified pipeline / batch PyVista script:** The paper stage can re-run one batch script that regenerates all PNGs. Set `regenerate_batch_figures` explicitly whenever possible: **true** only if replacing or adding generated PNGs is required (bad layout, wrong field, missing panel, new viz spec). Set **false** when fixes are LaTeX/prose/citations/abstract only so the pipeline does not thrash on incidental words like “figure” in a sentence. If you omit `regenerate_batch_figures`, the orchestrator falls back to keyword heuristics (less reliable).

Be strict but fair. Recommendations must be specific and actionable.

```

##### User prompt (from `prompts/prompts.yaml: PaperReviewerAgent.user_prompt`)

```
Review this LaTeX paper. Compilation determines your focus.

Compilation status: {compile_status}

LaTeX content:
---
{tex_content}
---

pdflatex compile log (full stdout/stderr if available; may be empty on success):
---
{compile_error}
---

RefChecker report summary (optional; may be empty if not run/available):
---
{reference_report}
---

Return ONLY valid JSON with these exact keys:
- pass (bool): if compilation failed, false until it compiles; if compilation succeeded, true only when the paper is publishable and all dimensions are acceptable
- score (float 0-1): overall quality (0 if compilation failed)
- formatting_ok (bool)
- figures_ok (bool)
- content_ok (bool)
- coherent (bool)
- publishable (bool)
- recommendations (list of strings): specific, actionable fixes — compilation fixes (syntax, packages, paths) when compile failed; content/formatting improvements when compile succeeded
- summary (string): 1-2 sentence assessment (what went wrong or what is good)
- needs_additional_visualization (bool): true if new or replacement PyVista figures are required before the paper is acceptable
- additional_viz_specs (list of strings): each line is one viz instruction including target case_id substring `case_NNN`
- regenerate_batch_figures (bool): **required** when compilation succeeded — true if the batch `paper_viz_batch.py` must re-run to refresh PNGs; false if LaTeX/text-only edits suffice (even if recommendations mention “Figure 3” in prose). When compilation failed, you may omit this key.

No markdown, no explanation outside the JSON.

```

Save the response as `<out-dir>/review.json`.

### Iterations 1+ — revise

Read `review.json`:
- `pass == true && score >= score_threshold && compile_ok` → **EXIT**: paper is done. Final PDF is `<out-dir>/paper/paper_draft.pdf`.
- Otherwise:
  - `regenerate_batch_figures == true` OR `needs_additional_visualization == true` → augment `paper_unified_plan.figure_jobs` with `additional_viz_specs`, jump back to Step 2 (batch viz).
  - Otherwise (text/citation/abstract fixes only) → re-invoke the Writer (Step 5) with the recommendations as a "revision instructions" block in `{section_context}`; recompile; re-review.

### Termination
- Pass condition: `review.pass and compile_ok and review.score >= score_threshold`.
- Otherwise stop after `max_review_loops` iterations.
- On failure: write the latest PDF anyway, with `review.json.pass=false` and a `review.json.summary` explaining what's missing.

## Mesh-independence inclusion
If `<out-dir>/mesh_independence_context.json` exists, the planner MUST include a mesh-independence subsection (the Writer's system prompt mandates this). Include `mesh_independence` as a key inside `{section_context}` for the Methods (or Mesh independence) section. The `WriterAgent.system_prompt` block above details the exact required content.

## Stateless resume
Re-running this skill with the same `<out-dir>` resumes from the latest checkpoint. If `paper/paper_draft.pdf` exists AND `review.json.pass == true`, the loop is done; rerun only if you change inputs.

## AI disclosure
The Writer prompts include a mandatory AI-disclosure footnote (see `WriterAgent.system_prompt`). Do not strip it from generated output.

## Anti-hallucination rules (echoed from the Writer prompt)
- Use ONLY data from `analysis.json`, `manifest.json`, figures, and `references.bib` candidates from `lit.json`. Do not invent numbers, results, or citations.
- Every claim in Discussion/Conclusion must map to a specific figure, table, or number.
- If a value is missing, write "TBD" rather than guessing.
- Do not strip the AI-disclosure footnote.

## Optional script fast-path

The orchestrator's actual default is one big script that does all of the above:

```bash
python scripts/paper_unified.py \
  --repo-root "$(pwd)" \
  --run-dir <out-dir> \
  --topic "<topic>" \
  --paper-dir <out-dir>/paper \
  --analysis <out-dir>/analysis.json \
  --manifest <out-dir>/manifest.json \
  --requirements <out-dir>/requirements.json \
  --literature <out-dir>/lit.json \
  --review-output <out-dir>/review.json \
  --template <template> \
  --max-review-loops <max_review_loops> \
  --mesh-independence <out-dir>/mesh_independence_context.json
```

This runs the full plan/viz/QA/write/compile/review loop in one process. Use it when you're inside the LangGraph repo and have an OpenAI/Anthropic key plus `pdflatex` available.

> **Known failure mode:** the `batch_paper_viz` sub-loop calls a vision model once per figure for image-quality QA. If the upstream service throws HTTP 503 (transient API outage), `paper_unified.py` exits rc=1. Resume with `--resume-from paper_review` once the upstream is healthy — no other state is lost.

**Legacy two-step (still functional, not what the orchestrator uses now):**
```bash
python scripts/paper_utils.py \
  --analysis <out-dir>/analysis.json \
  --figs <out-dir>/figs/ \
  --literature <out-dir>/lit.json \
  --output <out-dir>/paper/ \
  --template neurips

python scripts/reviewer.py \
  --paper <out-dir>/paper/ \
  --output <out-dir>/review.json
```

## Scope vs cfd-analyze
- `cfd-analyze` — text & metrics only (cross-case discussion, QoI table, conclusions JSON)
- `cfd-paper` — figures + LaTeX + review loop (consumes `analysis.json` output)

## Timeline
```json
{"stage": "paper_review", "event": "iteration_complete", "ts": "<iso>", "iter": 2, "compile_ok": true, "review_pass": false, "score": 0.62}
{"stage": "paper_review", "event": "complete", "ts": "<iso>", "iter": 4, "compile_ok": true, "review_pass": true, "score": 0.78}
```

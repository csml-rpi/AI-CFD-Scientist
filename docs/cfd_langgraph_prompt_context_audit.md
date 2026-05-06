# CFD Scientist Agents – Prompt Context Audit Report

**Scope:** All agents under `src/cfd_langgraph/` (CFD Scientist only; Foam-Agent excluded).  
**Goal:** Identify places where prompts may contain **huge context** (large JSON dumps, long logs, full run results, etc.) that could slow or break LLM calls.

---

## Summary

| Agent / Module        | Risk   | Location / call                         | What can be large |
|----------------------|--------|-----------------------------------------|-------------------|
| **AnalysisAgent**    | **HIGH** | `analyze_text_bundle()`                 | `bundle_text` = full `simulations` JSON |
| **WriterAgent**      | **HIGH** | `write_paper_with_literature()`        | `section_context`, `lit_bundle`, `viz_bundle`, `citations` |
| **LiteratureSurveyAgent** | **HIGH** | `survey()`                         | `s2` (up to 20 papers), `web` (8 results) – full objects in prompt |
| **Ideation**         | **MODERATE** | `run_ideation()`                  | `literature_context` (many papers), retry: `previous_idea` JSON |
| **HypothesisAgent**  | **LOW**   | `generate_user_requirement()`, etc.   | Single idea + simulation; `req` is one paragraph |
| **ResultsInterpreterAgent** | **LOW** | `interpret()`, `_text_only_interpret()` | Already audited: user_req + 20-line log or user_req + images |
| **IdeationAgent**    | **LOW**   | `generate_candidates()`               | Only static `task` from prompts (no user payload) |

---

## 1. AnalysisAgent – HIGH

**File:** `src/cfd_langgraph/agents/analysis_agent.py`

**Method:** `analyze_text_bundle(batch_name, bundle_text, extra_context)`

**Prompt content:**
- `batch_name` – short string.
- `extra_context` – short (e.g. `"Topic: ..."`).
- **`bundle_text`** – passed straight into the prompt as `"Bundle:\n{bundle_text}\n\n"`.

**Call site (workflow):** `graph.py` (analysis_and_writer):

```python
summary_text = json.dumps(pipeline_log.get("simulations", []), indent=2)
analysis_text = self.analysis.analyze_text_bundle(
    batch_name="cfd_topic_batch",
    bundle_text=summary_text,
    extra_context=f"Topic: {topic}",
)
```

**Why it’s large:** `pipeline_log["simulations"]` holds **all** cases with full run history (run_result, interpreter output, viz summaries, etc.). With many experiments this can be hundreds of KB of JSON.

**Recommendation:** Truncate or summarize before calling the LLM, e.g.:
- Keep only essential fields per case (simulation_id, case_name, status, rerun_required, short summary).
- Or cap total character length (e.g. last N chars or first N cases + “... (N more)”).
- Or build a short narrative summary in code and pass that instead of the full JSON.

---

## 2. WriterAgent – HIGH

**File:** `src/cfd_langgraph/agents/writer_agent.py`

**Methods:**
- **`write_section(section_context)`** – only `section_context` is in the prompt. Not used in the current workflow; if ever called with a long section, context would be large.
- **`write_paper_with_literature(topic, section_context, ideation_literature_bundle, visualization_bundle)`**

**Prompt content in `write_paper_with_literature`:**
- `topic` – short.
- **`section_context`** – built in workflow as:
  - `TOPIC:` + topic
  - `IDEA:` + **`json.dumps(state.get('idea', {}), indent=2)`**
  - `ANALYSIS:` + **full `analysis_text`** (output of `analyze_text_bundle`, can be long)
  - Plus a short line. So `section_context` can be **very large** (full idea + full analysis).
- **`lit_bundle`** – **`json.dumps(lit_bundle)`** in the prompt. Either ideation’s `literature_used` or result of `LiteratureSurveyAgent.survey()` (papers + synthesis). Can be large.
- **`citations`** – **`json.dumps(citations)`**. Multiple rounds of citations.
- **`viz_bundle`** – **`json.dumps(visualization_bundle or [])`**. One entry per simulation with viz summaries and metadata.

**Why it’s large:** One prompt contains full idea JSON, full analysis text, full literature bundle, full citation list, and full visualization bundle. This is the heaviest single prompt in the pipeline.

**Recommendation:**
- Truncate or summarize `section_context`: e.g. idea in one short paragraph, analysis in a fixed max length.
- For `lit_bundle` and `viz_bundle`: pass summaries (e.g. titles + one-line descriptions) or cap number of items / total chars.
- Consider a two-phase design: (1) short “outline” call with summarized context, (2) section-by-section generation with only the relevant slice of context.

---

## 3. LiteratureSurveyAgent – HIGH

**File:** `src/cfd_langgraph/agents/literature_agent.py`

**Method:** `survey(idea_text, max_papers=20)`

**Prompt content:**
- `idea` – `idea_text` (short).
- **`s2`** – full list of Semantic Scholar results (up to 20 papers). Each item can include title, abstract, year, venue, url, etc. Passed as `{s2}` so the **entire list is stringified** into the prompt.
- **`web`** – full list of web search results (e.g. 8). Same: full objects in the prompt.

**Why it’s large:** 20 abstracts plus metadata can easily be 30k+ characters in one prompt.

**Recommendation:**
- Truncate each paper to title + first N chars of abstract (e.g. 200–300).
- Or pass a preformatted string (like ideation’s `build_literature_context`) with fixed max length per item and cap total length.

---

## 4. Ideation – MODERATE

**File:** `src/cfd_langgraph/ideation.py`

**Flow:** `run_ideation()` builds `literature_context` and calls the LLM; on retry it adds `previous_idea` to the prompt.

**Prompt content:**
- **`literature_context`** – from `build_literature_context(lit_items)`. Each item: title, year, venue, URL, and **snippet[:300]**. With ~20 papers this is on the order of 10k–15k characters. Bounded but non-trivial.
- **Retry path:** `previous_idea` is **`json.dumps(idea_json, ...)`** – full idea JSON. Usually one idea, so size is moderate unless the idea has many experiments with large parameter blocks.

**Why it’s moderate:** Literature context is capped per snippet (300 chars) but scales with number of papers. Retry adds full idea JSON once.

**Recommendation:**
- Optionally cap total `literature_context` length or number of papers (e.g. top 10).
- On retry, pass a shortened idea (e.g. description + experiment count + one line per experiment) instead of full JSON.

---

## 5. HypothesisAgent – LOW

**File:** `src/cfd_langgraph/agents/hypothesis_agent.py`

**Methods:**
- `generate_user_requirement(idea, simulation)` – payload is one idea dict and one simulation dict (study_id, description, case_name, experiment_concept, etc.). Single experiment, so size is small.
- `llm_validate_requirement(req)` – only `req` (one requirement paragraph). Small.
- `repair_requirement(req, issues, guidance)` – `req` plus lists of issues and guidance. Typically short.

**Verdict:** No huge context; inputs are single-idea, single-simulation, or short text.

---

## 6. ResultsInterpreterAgent – LOW (already audited)

**File:** `src/cfd_langgraph/agents/interpreter_agent.py`

- **Vision path:** Only short `user_requirement` + base64 images.
- **Text-only path:** Only short `user_requirement` + last 20 lines of solver log.
- idea_json, experiment_spec, and experiment_results are **never** sent to the model.

**Verdict:** No huge context.

---

## 7. IdeationAgent – LOW

**File:** `src/cfd_langgraph/agents/idea_agent.py`

**Method:** `generate_candidates(num_calls)`

**Prompt content:** Only `task` from prompts (static template). No user-provided payload in the prompt.

**Verdict:** No huge context.

---

## 8. WriterAgent.collect_citations – MODERATE

**File:** `src/cfd_langgraph/agents/writer_agent.py`

**Method:** `collect_citations(citation_context, total_rounds)`

**Prompt content:** `citation_context` is the **topic** string (from `write_paper_with_literature` call: `topic`). So citation_context is short. Plus `round` and `total_rounds`. No large payload.

**Verdict:** Low; only topic string.

---

## Recommendations Summary

1. **AnalysisAgent.analyze_text_bundle:** Do not pass full `simulations` JSON. Summarize or truncate (e.g. per-case one-liner + key metrics, or strict char limit).
2. **WriterAgent.write_paper_with_literature:** Reduce size of `section_context`, `lit_bundle`, and `viz_bundle` (summaries, caps, or structured truncation).
3. **LiteratureSurveyAgent.survey:** Truncate each S2/web item (e.g. title + abstract[:300]) and/or cap total prompt length.
4. **Ideation:** Optionally cap literature items and shorten `previous_idea` on retry.

Implementing these will reduce token usage, latency, and risk of context overflows while keeping behavior aligned with the pipeline’s goals.

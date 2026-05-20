---
name: cfd-literature
description: Retrieve CFD literature for a topic via Semantic Scholar (with optional OpenAlex / arXiv / web supplements) and write a normalized lit.json. Use when the user wants a literature survey for a CFD study, or when cfd-pipeline / cfd-hypothesis / cfd-paper needs lit.json upstream. Self-contained — no Python helper required, but a one-shot script fast-path is documented at the bottom.
---

# cfd-literature

Build the canonical `lit.json` artifact that every downstream skill (`cfd-hypothesis`, `cfd-requirements`, `cfd-paper`) consumes.

This skill is **self-contained**. The agent makes the HTTP calls itself, normalizes the response, and writes the artifact. The Python helper is only an optional one-shot.

## Hard rules (slice 14)

1. **Semantic Scholar is mandatory.** `lit.json` must contain at least one record sourced from `https://api.semanticscholar.org/graph/v1/paper/search` (i.e. at least one record with `"source": "semanticscholar"`). The OpenAlex / arXiv / web supplements (Step 5) are *additive* — they do not substitute for the S2 call. Authoring `lit.json` without making the S2 HTTP call is forbidden.
2. **`S2_API_KEY` MUST be used when set.** Before making any S2 HTTP request, run `echo "${S2_API_KEY:+set}"` (or `printenv S2_API_KEY | head -c 4`). If the variable is non-empty, the request **MUST** include the `x-api-key: $S2_API_KEY` header. Falling back to the public (unauthenticated) endpoint while `S2_API_KEY` is set is a defect — it artificially trips the public rate limit even though the user provisioned a key. The only acceptable path to the unauthenticated endpoint is `S2_API_KEY` being genuinely unset/empty in the shell. If you see 429 from S2 and you have a key set, the request was misconfigured — fix the header, do not retry without it.
3. **Provenance is recorded.** Every record carries a `"source"` field (`"semanticscholar"`, `"openalex"`, `"arxiv"`, or `"web"`). Records without a `source` field are treated as fabricated by downstream gates.
4. **Empty-result handling is explicit, not silent.** If S2 returns zero useful records after filtering, write `lit.json` as `[]` AND emit `literature_empty_result` to the timeline AND surface the empty result to the agent in the response so the user sees it. Do NOT silently advance to `cfd-hypothesis` with an empty lit set — the downstream H10 paper-stage citation gate would still demand citations the agent has no source for.
5. **`checkpoints/literature_done.json` requires `lit.json` to exist** (even if `[]`) AND the timeline event to be present. The orchestrator-skill stage-gate `--mode preflight --target-stage hypothesis` checks `checkpoints/literature_done.json` directly; if the agent writes the checkpoint without S2 being called, the gate cannot detect it, so a downstream `cfd-paper` Stage 0.5 enrichment + H10 audit acts as the final defense.

## Inputs
- `topic` (required): research topic string
- `out-dir` (required): run directory; created if missing
- `max_papers` (optional, default 20): respect the user's limit if they specified one
- `s2_api_key` (optional): read from `S2_API_KEY` env var if not passed

## Output
`<out-dir>/lit.json` — JSON array of paper records:
```json
[
  {
    "paperId": "string",
    "title": "string",
    "abstract": "string or null",
    "year": 2024,
    "citationCount": 42,
    "venue": "string or null",
    "externalIds": {"DOI": "10.xxx/...", "ArXiv": "..."},
    "url": "https://...",
    "authors": [{"authorId": "...", "name": "..."}],
    "source": "semanticscholar | openalex | arxiv | web"
  }
]
```

## Recipe (primary, agent-driven)

### Step 1 — Build the query
Read the user's topic verbatim. Do **not** rephrase or "improve" it — the user's wording is the ground truth. If the topic is multi-clause (e.g. "LES of backward-facing step Re=5100 with comparison to DNS"), keep it as-is for Semantic Scholar; the API does fuzzy keyword matching.

If the user provided a specific maximum number of papers, use that exactly. Otherwise default to 20.

### Step 2 — Call Semantic Scholar Graph API
Endpoint: `GET https://api.semanticscholar.org/graph/v1/paper/search`

Query parameters:
- `query` = the topic string (URL-encoded)
- `limit` = `min(max_papers, 100)` (S2 hard caps at 100 per request; paginate with `offset` if you need more)
- `fields` = `paperId,title,abstract,year,citationCount,venue,externalIds,url,authors`

**Authentication — read this first, do NOT skip:**

Before issuing the request, check whether `S2_API_KEY` is set:

```bash
if [ -n "${S2_API_KEY:-}" ]; then
  echo "S2_API_KEY is set; authenticated endpoint MUST be used"
else
  echo "S2_API_KEY is not set; public endpoint is the only option"
fi
```

- **If `S2_API_KEY` is non-empty:** include `x-api-key: $S2_API_KEY` on **every** request to `api.semanticscholar.org`. This is mandatory (Hard rule #2). The authenticated endpoint has dramatically higher rate limits and is the path the user provisioned. Example with `curl`:

  ```bash
  curl -sS -H "x-api-key: $S2_API_KEY" \
    "https://api.semanticscholar.org/graph/v1/paper/search?query=<urlencoded>&limit=<n>&fields=paperId,title,abstract,year,citationCount,venue,externalIds,url,authors"
  ```

  Example with Python `requests`:

  ```python
  import os, requests
  headers = {"x-api-key": os.environ["S2_API_KEY"]}
  r = requests.get(
      "https://api.semanticscholar.org/graph/v1/paper/search",
      params={"query": topic, "limit": limit, "fields": fields},
      headers=headers,
      timeout=30,
  )
  ```

  If you call S2 without the header while the key is set, that is the defect this rule exists to prevent. Re-issue the request with the header.

- **Only if `S2_API_KEY` is unset/empty:** use the public endpoint (no `x-api-key` header). It is rate-limited to ≈100 requests / 5 min — back off and retry with exponential delay on 429.

**Never** use the public endpoint when `S2_API_KEY` is set. There is no scenario where stripping the header is the correct choice while the user has a key provisioned.

### Step 3 — Filter for CFD relevance
S2 search is loose; many results will be off-topic. Drop a paper when:
- `abstract` is null AND title doesn't contain any CFD/fluid keyword (`flow`, `turbulence`, `Reynolds`, `LES`, `RANS`, `DNS`, `OpenFOAM`, `mesh`, `boundary layer`, `viscosity`, …) — only filter on title when the abstract is unavailable.
- `year < 1980` (very old; usually not what the user wants unless the topic explicitly mentions a classical reference).
- The paper is a duplicate of one already kept (match on `paperId`, then on lowercased title).

Don't be aggressive. When in doubt, keep the paper.

### Step 4 — Sort
Primary: `citationCount` descending. Secondary: `year` descending. This biases toward foundational + recent.

### Step 5 — Optional supplements
If after filtering you have fewer than `max_papers / 2` results, supplement from:
- **OpenAlex** — `GET https://api.openalex.org/works?search=<topic>&per-page=25` — same field-mapping (title, abstract, publication_year, cited_by_count, doi, authors).
- **arXiv** — `GET http://export.arxiv.org/api/query?search_query=all:<topic>&max_results=25` (Atom XML; parse `entry/title`, `entry/summary`, `entry/published`, `entry/id`).

Tag supplemented entries with `"source": "openalex"` or `"source": "arxiv"`. Do not invent records.

### Step 6 — Write `lit.json`
- Truncate to `max_papers`.
- Sort once more (citation desc).
- Write JSON, indent 2, UTF-8.

### Step 7 — Append to timeline
Open `<out-dir>/timeline.json` (create as `[]` if missing) and append:
```json
{"stage": "literature", "event": "complete", "ts": "<iso8601>", "paper_count": <n>, "topic": "<topic>", "with_api_key": true|false}
```

## Skip if already done
If `<out-dir>/lit.json` exists and is a non-empty array, skip the search and append a `literature_skipped_existing` event. The user can force regeneration by deleting `lit.json` first.

## Anti-hallucination rules
- **Never invent a paper.** If the API returns nothing useful, write an empty `[]` and emit a timeline event `literature_empty_result` so downstream skills can react. Do **not** fabricate titles, DOIs, or abstracts to "fill in" the array.
- **Never invent citation counts.** If the field is missing in the response, leave it `null` (not 0).
- **Never invent authors.** Trust the API response shape exactly.

## Optional script fast-path
If `scripts/lit.py` is present (it is, in this repo's LangGraph pipeline) and you would prefer to delegate the HTTP/parse work, you may run:
```bash
python scripts/lit.py \
  --topic "<topic>" \
  --limit <max_papers> \
  --output <out-dir>/lit.json \
  --timeline <out-dir>/timeline.json
```
Same artifact contract. The Python script reads `S2_API_KEY` from the environment and attaches the `x-api-key` header automatically (`scripts/lit.py:148`), so the auth contract from Hard rule #2 is preserved. Make sure the variable is exported in the shell before invoking:

```bash
export S2_API_KEY=<your_key>
python scripts/lit.py --topic "..." --output <out-dir>/lit.json --timeline <out-dir>/timeline.json
```

Use this script when running inside the conda env and the script is reachable; use the agent recipe otherwise (e.g. when running from a different agent framework or when `cfd-skills/` is symlinked into a non-Python project).

## Notes
- Citation enrichment for the bibliography (`cfd-paper`) happens later via DBLP/CrossRef in `cfd-paper`'s citation step. This skill produces only the discovery dataset.
- For multi-flow OED (`--oed-multi-flow`) you may want to run `cfd-literature` once per flow with the flow-specific topic — that produces multiple `lit.json` files; downstream skills handle either case.

## Next

After `lit.json` and `checkpoints/literature_done.json` exist on disk, your next action is to invoke the next skill in the chain. Do **not** stop, summarize, or wait — this is a literal handoff.

Read `<out-dir>/state.json#mode`, then invoke from this table:

| `mode` | Next skill |
|---|---|
| `research`, `code_mod`, `open_discovery` | `Skill cfd-skills/cfd-pipeline` — run only Steps 5 (baseline_setup) and 6 (metric_setup); write `baseline_metrics.json`, `metric_definitions.json`, and the matching `checkpoints/*_done.json`. Its own `## Next` then chains onward. |

`mode == analysis_only / paper_only / mesh_gate` does not pass through this skill, so no branch is needed here.

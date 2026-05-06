---
name: cfd-literature
description: Retrieve CFD literature for a topic via Semantic Scholar (with optional OpenAlex / arXiv / web supplements) and write a normalized lit.json. Use when the user wants a literature survey for a CFD study, or when cfd-pipeline / cfd-hypothesis / cfd-paper needs lit.json upstream. Self-contained — no Python helper required, but a one-shot script fast-path is documented at the bottom.
---

# cfd-literature

Build the canonical `lit.json` artifact that every downstream skill (`cfd-hypothesis`, `cfd-requirements`, `cfd-paper`) consumes.

This skill is **self-contained**. The agent makes the HTTP calls itself, normalizes the response, and writes the artifact. The Python helper is only an optional one-shot.

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

Header (if `S2_API_KEY` is set):
- `x-api-key: $S2_API_KEY`

Without an API key, the public endpoint works but with much lower rate limits (≈100 requests / 5 min). Back off and retry with exponential delay if you hit 429.

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
Same artifact contract. The Python script does the same Semantic Scholar call + filtering. Use this when running inside the conda env and the script is reachable; use the agent recipe otherwise (e.g. when running from a different agent framework or when `cfd-skills/` is symlinked into a non-Python project).

## Notes
- Citation enrichment for the bibliography (`cfd-paper`) happens later via DBLP/CrossRef in `cfd-paper`'s citation step. This skill produces only the discovery dataset.
- For multi-flow OED (`--oed-multi-flow`) you may want to run `cfd-literature` once per flow with the flow-specific topic — that produces multiple `lit.json` files; downstream skills handle either case.

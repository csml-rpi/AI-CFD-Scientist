#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import requests
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from timeline_logger import append_timeline_event, resolve_timeline_path


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def to_record(paper: Any) -> Dict[str, Any]:
    authors = [a.name for a in (getattr(paper, "authors", []) or []) if getattr(a, "name", None)]
    doi = getattr(paper, "externalIds", {}).get("DOI") if getattr(paper, "externalIds", None) else None
    return {
        "title": getattr(paper, "title", "") or "",
        "abstract": getattr(paper, "abstract", "") or "",
        "authors": authors,
        "year": getattr(paper, "year", None),
        "doi": doi,
        "url": getattr(paper, "url", "") or "",
        "citationCount": getattr(paper, "citationCount", 0) or 0,
    }


_STOP_WORDS = frozenset({
    "a", "an", "the", "in", "of", "for", "and", "or", "to", "with",
    "on", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "using", "based", "via", "into", "between", "over", "under", "through",
})


def _shorten_query(topic: str, max_keywords: int = 6) -> str:
    """S2 search needs short keyword queries (~5-6 content words)."""
    cleaned = topic.split(":")[0] if ":" in topic else topic
    cleaned = cleaned.replace(",", " ").replace(";", " ").replace("(", " ").replace(")", " ")
    words = cleaned.split()
    keywords = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
    if len(keywords) <= max_keywords:
        return " ".join(keywords).strip()
    return " ".join(keywords[:max_keywords]).strip()


def _generate_multi_queries(topic: str, max_queries: int = 4) -> List[str]:
    """Split a long topic into several shorter, targeted keyword queries.

    Issuing 3-4 small queries (5-6 keywords each) tends to be more
    rate-limit-friendly and retrieves a broader set of relevant papers than
    one giant query. Strategy (generic across any topic):
      - Query 1: the primary shortened form (legacy behaviour)
      - Query 2: second half of keywords (complementary coverage)
      - Query 3: noun-phrase pairs extracted from the topic
      - Query 4: the shortest form (top 3 keywords) for maximum recall
    Deduplicates before returning.
    """
    # Use the FULL topic for keyword extraction — don't strip the pre-colon
    # label (e.g. "Open-ended discovery:") early because for some topics the
    # real content lives after the colon.
    cleaned = topic.replace(":", " ").replace(",", " ").replace(";", " ")
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    words = cleaned.split()
    kws = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    queries: List[str] = []
    if len(kws) >= 1:
        queries.append(" ".join(kws[:6]).strip())                         # primary
    if len(kws) > 6:
        queries.append(" ".join(kws[6:12]).strip())                       # secondary
    if len(kws) >= 4:
        # noun-phrase pairs (adjacent pairs after stopword removal)
        pairs = " ".join(" ".join(kws[i:i + 2]) for i in range(0, min(len(kws), 8), 2))
        if pairs and pairs not in queries:
            queries.append(pairs)
    if len(kws) >= 3:
        short3 = " ".join(kws[:3]).strip()
        if short3 and short3 not in queries:
            queries.append(short3)

    seen: set = set()
    uniq: List[str] = []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
        if len(uniq) >= max_queries:
            break
    return uniq or [topic[:200]]


def collect_papers_via_requests(topic: str, limit: int) -> List[Dict[str, Any]]:
    """Issue multiple short keyword queries instead of one long one.

    Deduplicates results by DOI/title across queries. Stops early when `limit`
    papers have been collected. Resilient to per-query 429s — moves on to the
    next query instead of burning retries on a single rate-limited endpoint.
    """
    out: List[Dict[str, Any]] = []
    seen_dois: set = set()
    seen_titles: set = set()
    queries = _generate_multi_queries(topic)
    print(f"[LIT] Using {len(queries)} queries: {queries}")

    for q_idx, query in enumerate(queries, 1):
        if len(out) >= limit:
            break
        print(f"[LIT] === Query {q_idx}/{len(queries)}: {query!r} ===")
        out_before = len(out)
        res = _collect_papers_single_query(query, limit - len(out))
        for rec in res:
            doi = (rec.get("doi") or "").strip().lower()
            title = (rec.get("title") or "").strip().lower()
            if doi and doi in seen_dois:
                continue
            if not doi and title and title in seen_titles:
                continue
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)
            out.append(rec)
            if len(out) >= limit:
                break
        print(f"[LIT] Query {q_idx} added {len(out) - out_before} unique paper(s); total now {len(out)}.")
    print(f"[LIT] Finished {len(queries)} queries: {len(out)} unique papers.")
    return out


def _collect_papers_single_query(query: str, limit: int) -> List[Dict[str, Any]]:
    """Original single-query collector, renamed. Used by multi-query above."""
    out: List[Dict[str, Any]] = []
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    page_size = 5
    offset = 0
    total_available: Optional[int] = None
    while len(out) < limit:
        want = min(page_size, limit - len(out))
        if total_available is not None and offset >= total_available:
            print(f"[LIT] Reached end of S2 result set (total={total_available}).")
            break
        params = {
            "query": query,
            "limit": want,
            "offset": offset,
            "fields": "title,abstract,year,venue,url,citationCount,authors,externalIds",
        }
        print(f"[LIT] Requesting page offset={offset} limit={want} ...")
        resp = requests.get(base, params=params, headers=headers, timeout=120)
        print(f"[LIT] API response status={resp.status_code} for offset={offset}")
        if not resp.ok:
            # Do not discard papers already retrieved (S2 often returns 400 when offset exceeds total, or 429).
            if out:
                print(
                    f"[LIT] Non-success HTTP {resp.status_code}; stopping pagination and "
                    f"keeping {len(out)} paper(s) already retrieved.",
                    file=sys.stderr,
                )
                break
            resp.raise_for_status()
        payload = resp.json() or {}
        if total_available is None and isinstance(payload.get("total"), int):
            total_available = int(payload["total"])
            print(f"[LIT] API reports total matching papers ≈ {total_available}.")
        data = payload.get("data", []) or []
        if not data:
            print("[LIT] No more papers returned by API.")
            break
        for rec in data:
            out.append(rec)
            title = rec.get("title") or "(untitled)"
            year = rec.get("year")
            cc = rec.get("citationCount", 0) or 0
            print(f"[LIT] Loaded paper {len(out)}/{limit}: {title} | year={year} | citations={cc}")
            if len(out) >= limit:
                break
        offset += len(data)
    if len(out) < limit and out:
        print(
            f"[LIT] Partial literature set: saved {len(out)} paper(s) "
            f"(requested limit={limit}; API may have fewer matches or pagination stopped early).",
            file=sys.stderr,
        )
    return out


def collect_papers(sch: Any, topic: str, limit: int) -> List[Any]:
    """Collect papers with per-paper progress output."""
    out: List[Any] = []
    # Prefer paged collection so progress can be printed as records arrive.
    offset = 0
    page_size = 5
    while len(out) < limit:
        want = min(page_size, limit - len(out))
        try:
            res = sch.search_paper(topic, limit=want, offset=offset)
        except TypeError:
            # Backward compatibility for client versions without offset support.
            if offset > 0:
                break
            res = sch.search_paper(topic, limit=limit)
        if res is None:
            break
        page_items: List[Any] = []
        if isinstance(res, list):
            page_items = res
        else:
            for p in res:
                page_items.append(p)
                if len(page_items) >= want:
                    break
        if not page_items:
            break
        for p in page_items:
            out.append(p)
            title = getattr(p, "title", "") or "(untitled)"
            year = getattr(p, "year", None)
            cc = getattr(p, "citationCount", 0) or 0
            print(f"[LIT] Loaded paper {len(out)}/{limit}: {title} | year={year} | citations={cc}")
            if len(out) >= limit:
                break
        offset += len(page_items)
    return out


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Search Semantic Scholar literature.")
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--limit", default=20, type=int)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--timeline", default="", type=str)
    args = parser.parse_args()
    timeline_path = resolve_timeline_path(args.timeline)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retries = 3
    paper_records: List[Dict[str, Any]] = []
    for attempt in range(1, retries + 1):
        try:
            print(f"Searching papers for topic: {args.topic} (attempt {attempt}/{retries})")
            paper_records = collect_papers_via_requests(args.topic, args.limit)
            print(f"[LIT] Finished loading {len(paper_records)} paper(s).")
            break
        except Exception as exc:
            if attempt == retries:
                print(f"Failed after retries: {exc}", file=sys.stderr)
                return 1
            sleep_s = 2 ** (attempt - 1)
            print(f"API error: {exc}; retrying in {sleep_s}s")
            time.sleep(sleep_s)

    records = paper_records
    records.sort(key=lambda r: r.get("citationCount", 0), reverse=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {
            "stage": "literature",
            "topic": args.topic,
            "limit": int(args.limit),
            "paper_count": len(records),
            "paper_titles": [r.get("title", "") for r in records[:20]],
            "output_path": str(out_path),
        },
    )
    print(f"Saved {len(records)} papers to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
        "source": "semanticscholar",
    }


_STOP_WORDS = frozenset({
    "a", "an", "the", "in", "of", "for", "and", "or", "to", "with",
    "on", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "using", "based", "via", "into", "between", "over", "under", "through",
})

_INTENT_WORDS = frozenset({
    "analysis", "cfd", "compare", "comparison", "discover", "discovery",
    "effect", "effects", "ended", "find", "investigate", "modification",
    "novel", "open", "open-ended", "simulation", "study", "using",
})


def _topic_keywords(topic: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_+./-]*", str(topic or ""))
    return [
        word for word in words
        if word.lower() not in _STOP_WORDS
        and word.lower() not in _INTENT_WORDS
        and len(word) > 1
    ]


def _shorten_query(topic: str, max_keywords: int = 6) -> str:
    """S2 search needs short keyword queries (~5-6 content words)."""
    keywords = _topic_keywords(topic)
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
    kws = _topic_keywords(topic)

    queries: List[str] = []
    if len(kws) >= 1:
        queries.append(" ".join(kws[:6]).strip())                         # primary
    if len(kws) > 6:
        queries.append(" ".join(kws[-6:]).strip())                        # complementary tail
    if len(kws) >= 4:
        # Join anchors from both ends so this is not just query 1 with two
        # extra words (the previous "pair" construction reconstructed the
        # same prefix and added almost no retrieval diversity).
        anchors = " ".join((kws[:3] + kws[-3:])).strip()
        if anchors and anchors not in queries:
            queries.append(anchors)
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

    if limit <= 0:
        return []

    def add_unique(records: List[Dict[str, Any]]) -> int:
        before = len(out)
        for rec in records:
            if len(out) >= limit:
                break
            doi = (rec.get("doi") or "").strip().lower()
            title = (rec.get("title") or "").strip().lower()
            if (doi and doi in seen_dois) or (title and title in seen_titles):
                continue
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)
            out.append(rec)
            if len(out) >= limit:
                break
        return len(out) - before

    # One failing query must not throw away the papers the others found (it
    # used to: an error on query 3 restarted the search from query 1). A
    # non-retryable HTTP error skips that query; a service that is still down
    # after every retry ends the search with whatever was already collected.
    query_errors: List[str] = []

    def run_query(query: str, quota: int, offset_start: int = 0) -> List[Dict[str, Any]]:
        try:
            return _collect_papers_single_query(query, quota, offset_start=offset_start)
        except ServiceUnavailable:
            raise
        except requests.RequestException as exc:
            print(f"[LIT] Query {query!r} failed: {exc}; moving on to the next query.", file=sys.stderr)
            query_errors.append(str(exc))
            return []

    # Give every query an initial share before any one query can consume the
    # whole paper budget. This is the part the former implementation missed:
    # it asked query 1 for `limit` results and normally never reached query 2.
    initial_quota = max(1, math.ceil(limit / len(queries)))
    try:
        for q_idx, query in enumerate(queries, 1):
            if len(out) >= limit:
                break
            print(f"[LIT] === Query {q_idx}/{len(queries)}: {query!r} ===")
            added = add_unique(run_query(query, initial_quota))
            print(f"[LIT] Query {q_idx} added {added} unique paper(s); total now {len(out)}.")

        # Duplicates or sparse result sets may leave the balanced first pass
        # short. Continue each query after its initial offset until the global
        # budget is full or every query is exhausted.
        if len(out) < limit:
            for q_idx, query in enumerate(queries, 1):
                remaining = limit - len(out)
                if remaining <= 0:
                    break
                print(f"[LIT] Fill pass query {q_idx}/{len(queries)}: {query!r}")
                added = add_unique(run_query(query, remaining, offset_start=initial_quota))
                print(f"[LIT] Fill pass added {added} unique paper(s); total now {len(out)}.")
    except ServiceUnavailable as exc:
        if not out:
            raise
        print(f"[LIT] {exc} Keeping the {len(out)} paper(s) already retrieved.", file=sys.stderr)
    if not out and query_errors:
        raise RuntimeError(f"Every Semantic Scholar query failed; last error: {query_errors[-1]}")
    print(f"[LIT] Finished {len(queries)} queries: {len(out)} unique papers.")
    return out


# Semantic Scholar answers overload with 429 and 5xx, and the same request
# usually succeeds seconds later: on 2026-09-11 an identical search returned
# 500 and then 200 eighteen seconds on. The old 1 s / 2 s retries restarted the
# whole multi-query search and gave up inside ~10 s, so three live studies read
# "0 papers" for queries as broad as "deep learning" and looped on the call.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_WAITS_S = (5, 10, 20, 40, 60)


class ServiceUnavailable(RuntimeError):
    """Semantic Scholar kept failing after every retry — not an empty result."""


def _get_with_retry(url: str, params: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
    """GET one page, waiting and retrying the SAME page on transient failures."""
    last = ""
    for attempt, wait in enumerate((*_RETRY_WAITS_S, None), 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=120)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code not in _RETRY_STATUS:
                return resp
            last = f"HTTP {resp.status_code} from Semantic Scholar"
            retry_after = str(resp.headers.get("Retry-After") or "")
            if wait is not None and retry_after.isdigit():
                wait = max(wait, min(int(retry_after), 120))
        if wait is None:
            break
        print(f"[LIT] {last}; retrying the same request in {wait}s (attempt {attempt})", flush=True)
        time.sleep(wait)
    raise ServiceUnavailable(
        f"Semantic Scholar service error, not an empty result: {last} after "
        f"{len(_RETRY_WAITS_S) + 1} attempts over ~{sum(_RETRY_WAITS_S)} s."
    )


def _collect_papers_single_query(query: str, limit: int, offset_start: int = 0) -> List[Dict[str, Any]]:
    """Original single-query collector, renamed. Used by multi-query above."""
    out: List[Dict[str, Any]] = []
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    # Every request is one more chance to hit a transient 5xx, so ask for a
    # useful page rather than five records at a time (the API allows 100).
    page_size = 20
    offset = max(0, int(offset_start))
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
        resp = _get_with_retry(base, params, headers)
        print(f"[LIT] API response status={resp.status_code} for offset={offset}")
        if not resp.ok:
            # Do not discard papers already retrieved (S2 returns 400 when the
            # offset runs past the end of the result set).
            if out or offset > 0:
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
            rec["source"] = "semanticscholar"
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
    # Retries happen per request inside the collector, so a failure here has
    # already waited out the transient window; retrying the whole search again
    # would only multiply that wait.
    paper_records: List[Dict[str, Any]] = []
    try:
        print(f"Searching papers for topic: {args.topic}")
        paper_records = collect_papers_via_requests(args.topic, args.limit)
        print(f"[LIT] Finished loading {len(paper_records)} paper(s).")
    except Exception as exc:
        print(f"Failed after retries: {exc}", file=sys.stderr)
        return 1

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

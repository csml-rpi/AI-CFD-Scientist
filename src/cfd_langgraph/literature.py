from __future__ import annotations

import requests
from dataclasses import dataclass, asdict
from typing import List
from urllib.parse import quote_plus


@dataclass
class LiteratureItem:
    source: str
    title: str
    url: str
    year: str | None = None
    venue: str | None = None
    snippet: str | None = None


class LiteratureClient:
    def __init__(self, s2_api_key: str | None = None, brave_api_key: str | None = None):
        self.s2_api_key = s2_api_key
        self.brave_api_key = brave_api_key

    def search_semantic_scholar(self, query: str, limit: int = 10) -> List[LiteratureItem]:
        """Works with or without API key."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "title,year,venue,url,abstract",
        }
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception:
            return []

        out: List[LiteratureItem] = []
        for p in data:
            out.append(
                LiteratureItem(
                    source="semantic_scholar",
                    title=p.get("title", ""),
                    url=p.get("url", ""),
                    year=str(p.get("year")) if p.get("year") else None,
                    venue=p.get("venue"),
                    snippet=(p.get("abstract") or "")[:600],
                )
            )
        return out

    def search_openalex(self, query: str, limit: int = 10) -> List[LiteratureItem]:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": max(1, min(limit, 50)),
            "sort": "relevance_score:desc",
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception:
            return []

        out: List[LiteratureItem] = []
        for r in results:
            title = r.get("display_name", "")
            year = r.get("publication_year")
            venue = None
            host = r.get("primary_location", {}) if isinstance(r.get("primary_location"), dict) else {}
            src = host.get("source", {}) if isinstance(host.get("source"), dict) else {}
            if src:
                venue = src.get("display_name")
            url = r.get("id", "")
            abs_idx = r.get("abstract_inverted_index")
            snippet = ""
            if isinstance(abs_idx, dict) and abs_idx:
                inv = {}
                for token, pos_list in abs_idx.items():
                    for p in pos_list:
                        inv[p] = token
                snippet = " ".join(inv[k] for k in sorted(inv.keys()))[:600]

            out.append(
                LiteratureItem(
                    source="openalex",
                    title=title,
                    url=url,
                    year=str(year) if year else None,
                    venue=venue,
                    snippet=snippet,
                )
            )
        return out

    def search_arxiv(self, query: str, limit: int = 10) -> List[LiteratureItem]:
        url = (
            "http://export.arxiv.org/api/query?search_query=all:"
            + quote_plus(query)
            + f"&start=0&max_results={max(1, min(limit, 30))}"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return []

        out: List[LiteratureItem] = []
        entries = text.split("<entry>")[1:]
        for e in entries:
            def grab(tag: str) -> str:
                a = f"<{tag}>"
                b = f"</{tag}>"
                if a in e and b in e:
                    return e.split(a, 1)[1].split(b, 1)[0].strip().replace("\n", " ")
                return ""

            title = grab("title")
            summary = grab("summary")
            link = ""
            if 'rel="alternate"' in e and 'href="' in e:
                try:
                    link = e.split('rel="alternate"', 1)[1].split('href="', 1)[1].split('"', 1)[0]
                except Exception:
                    link = ""
            published = grab("published")
            year = published[:4] if len(published) >= 4 else None
            if title:
                out.append(
                    LiteratureItem(
                        source="arxiv",
                        title=title,
                        url=link,
                        year=year,
                        venue="arXiv",
                        snippet=summary[:600],
                    )
                )
        return out

    def search_brave_web(self, query: str, limit: int = 5) -> List[LiteratureItem]:
        if not self.brave_api_key:
            return []

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key,
        }
        params = {"q": query, "count": max(1, min(limit, 20))}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
        except Exception:
            return []

        out: List[LiteratureItem] = []
        for r in results:
            out.append(
                LiteratureItem(
                    source="brave_web",
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=(r.get("description") or "")[:500],
                )
            )
        return out

    def collect(self, query: str, max_papers: int = 20, max_web_results: int = 5) -> List[LiteratureItem]:
        papers = self.search_semantic_scholar(query=query, limit=max_papers)

        if len(papers) < max_papers:
            need = max_papers - len(papers)
            papers.extend(self.search_openalex(query=query, limit=need))
        if len(papers) < max_papers:
            need = max_papers - len(papers)
            papers.extend(self.search_arxiv(query=query, limit=need))

        papers = papers[:max_papers]
        web = self.search_brave_web(query=query, limit=max_web_results)

        seen = set()
        merged: List[LiteratureItem] = []
        for item in papers + web:
            key = (item.url or "").strip().lower() or item.title.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def to_json_ready(items: List[LiteratureItem]):
        return [asdict(x) for x in items]

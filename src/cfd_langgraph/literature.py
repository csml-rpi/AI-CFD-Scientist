from __future__ import annotations

import requests
from dataclasses import dataclass, asdict
from typing import List


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
        if not self.s2_api_key:
            return []

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "title,year,venue,url,abstract",
        }
        headers = {"x-api-key": self.s2_api_key}

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

    def collect(self, query: str, max_papers: int = 12, max_web_results: int = 5) -> List[LiteratureItem]:
        papers = self.search_semantic_scholar(query=query, limit=max_papers)
        web = self.search_brave_web(query=query, limit=max_web_results)

        # Deduplicate by URL/title
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

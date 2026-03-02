from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from langchain_core.prompts import ChatPromptTemplate

from cfd_langchain.llm.factory import create_langchain_llm


class LiteratureSurveyAgent:
    """Finds close/relevant literature for a CFD idea using Semantic Scholar + web search snippets.

    Env vars:
    - S2_API_KEY (optional, but recommended)
    - BRAVE_SEARCH_API_KEY (optional, for web supplement)
    """

    S2_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, model: str):
        self.model = model
        self.llm = create_langchain_llm(model=model, temperature=0.1)
        self.s2_api_key = os.environ.get("S2_API_KEY")
        self.brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    def _s2_headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.s2_api_key:
            h["x-api-key"] = self.s2_api_key
        return h

    def search_semantic_scholar(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{self.S2_BASE}/paper/search"
        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "title,abstract,year,venue,url,citationCount,authors,externalIds",
        }
        r = requests.get(url, params=params, headers=self._s2_headers(), timeout=30)
        r.raise_for_status()
        return (r.json() or {}).get("data", [])

    def search_web(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        if not self.brave_api_key:
            return []
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key,
        }
        params = {"q": query, "count": max(1, min(count, 20))}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return (((r.json() or {}).get("web") or {}).get("results") or [])

    def survey(self, idea_text: str, max_papers: int = 20) -> Dict[str, Any]:
        s2 = self.search_semantic_scholar(idea_text, limit=max_papers)
        web = self.search_web(f"{idea_text} CFD OpenFOAM", count=8)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a CFD literature survey expert. Rank closest literature, identify novelty overlap, and point to missing baselines.",
            ),
            (
                "human",
                "Idea:\n{idea}\n\nSemanticScholar:\n{s2}\n\nWeb:\n{web}\n\nReturn concise JSON with keys: close_literature, gaps, recommended_baselines, risk_assessment.",
            ),
        ])
        chain = prompt | self.llm
        synthesis = chain.invoke({"idea": idea_text, "s2": s2, "web": web}).content

        return {
            "idea": idea_text,
            "semantic_scholar": s2,
            "web_results": web,
            "synthesis": synthesis,
        }

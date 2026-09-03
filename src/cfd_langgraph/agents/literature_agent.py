from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm

# Retry on 429 (rate limit) or timeout; max 4 attempts, exponential backoff
S2_RETRY_ATTEMPTS = 4
S2_RETRY_BASE_DELAY = 10
S2_REQUEST_TIMEOUT = 600  # seconds (10 minutes)


class LiteratureSurveyAgent:
    """Finds close/relevant literature for a CFD idea using Semantic Scholar + web search snippets.

    Env vars:
    - S2_API_KEY (optional; Semantic Scholar works without it, public rate limit applies)
    - BRAVE_SEARCH_API_KEY (optional, for web supplement)
    """

    S2_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, model: str):
        self.model = model
        self.llm = create_langchain_llm(model=model, temperature=0.0)
        self.s2_api_key = os.environ.get("S2_API_KEY")
        self.brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    def _s2_headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.s2_api_key:
            h["x-api-key"] = self.s2_api_key
        return h

    def search_semantic_scholar(self, query: str, limit: int = 40) -> List[Dict[str, Any]]:
        url = f"{self.S2_BASE}/paper/search"
        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "title,abstract,year,venue,url,citationCount,authors,externalIds",
        }
        last_error = None
        for attempt in range(S2_RETRY_ATTEMPTS):
            try:
                r = requests.get(
                    url,
                    params=params,
                    headers=self._s2_headers(),
                    timeout=S2_REQUEST_TIMEOUT,
                )
                if r.status_code == 429:
                    last_error = requests.exceptions.HTTPError("429 Rate limit (Semantic Scholar)", response=r)
                    if attempt < S2_RETRY_ATTEMPTS - 1:
                        delay = S2_RETRY_BASE_DELAY * (2 ** attempt)
                        time.sleep(delay)
                    continue
                r.raise_for_status()
                return (r.json() or {}).get("data", [])
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < S2_RETRY_ATTEMPTS - 1:
                    delay = S2_RETRY_BASE_DELAY * (2 ** attempt)
                    print("[Literature] Semantic Scholar timeout, retrying in %ds (attempt %d/%d)..." % (delay, attempt + 1, S2_RETRY_ATTEMPTS))
                    time.sleep(delay)
                continue
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code == 429 and attempt < S2_RETRY_ATTEMPTS - 1:
                    delay = S2_RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
            except requests.exceptions.RequestException as e:
                last_error = e
                raise
        if last_error:
            raise last_error
        return []

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

    def survey(self, idea_text: str, max_papers: int = 40) -> Dict[str, Any]:
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

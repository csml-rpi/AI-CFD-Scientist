from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class HypothesisRankAgent:
    """The "Rank" step of propose -> critique -> rank.

    Compares every critique-surviving candidate against the others in one
    pass and returns them ordered strongest-first, each with a rationale.
    This is a group comparison rather than a full O(n^2) pairwise Elo
    tournament — a practical simplification of the head-to-head ranking in
    Google DeepMind's AI co-scientist (arXiv 2502.18864); worth revisiting as
    a true pairwise tournament if candidate counts grow large enough that a
    single-pass comparison stops being reliable.
    """

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.prompts = prompt_loader.section("IdeationRankAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.1)

    def rank(self, candidates: List[Dict[str, Any]], research_topic: str = "") -> List[Dict[str, Any]]:
        if len(candidates) <= 1:
            for i, c in enumerate(candidates):
                c.setdefault("rank", i + 1)
                c.setdefault("rank_rationale", "Only surviving candidate.")
            return candidates

        sys_t = self.prompts.get("rank_system_prompt", "")
        usr_t = self.prompts.get("rank_user_prompt", "")
        if not sys_t or not usr_t:
            raise ValueError("Missing IdeationRankAgent prompts")

        payload = {
            "research_topic": research_topic,
            "candidates_json": json.dumps(
                [{"candidate_id": c["candidate_id"], "idea": c.get("idea", {})} for c in candidates],
                ensure_ascii=False,
                indent=2,
            ),
        }
        prompt = ChatPromptTemplate.from_messages([("system", sys_t), ("human", usr_t)])
        raw = (prompt | self.llm).invoke(payload).content

        try:
            parsed = json.loads(strip_json_fences(raw))
            order = parsed.get("ranking", []) if isinstance(parsed, dict) else []
        except Exception:
            order = []

        by_id = {c["candidate_id"]: c for c in candidates}
        ranked: List[Dict[str, Any]] = []
        seen = set()
        for entry in order:
            cid = entry.get("candidate_id") if isinstance(entry, dict) else None
            if cid in by_id and cid not in seen:
                c = by_id[cid]
                c["rank"] = len(ranked) + 1
                c["rank_rationale"] = entry.get("rationale", "")
                ranked.append(c)
                seen.add(cid)

        # Anything the LLM's response dropped still gets returned, at the bottom,
        # rather than silently disappearing.
        for c in candidates:
            if c["candidate_id"] not in seen:
                c["rank"] = len(ranked) + 1
                c["rank_rationale"] = "Not addressed in ranking response."
                ranked.append(c)

        return ranked

from __future__ import annotations

import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class IdeationAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("IdeationAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.65)

    def generate_candidates(self, num_calls: int = 1) -> List[Dict[str, Any]]:
        system = "You are an experienced AI researcher in Computational Fluid Dynamics (CFD)."
        user = self.prompts.get("initial_idea_prompt", "")
        if not user:
            raise ValueError("Missing IdeationAgent.initial_idea_prompt")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{task}"),
        ])
        chain = prompt | self.llm

        out: List[Dict[str, Any]] = []
        for _ in range(num_calls):
            txt = strip_json_fences(chain.invoke({"task": user}).content)
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1 and e > s:
                txt = txt[s : e + 1]
            out.append(json.loads(txt))
        return out

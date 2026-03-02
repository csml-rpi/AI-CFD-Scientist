from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from cfd_langchain.llm.factory import create_langchain_llm
from cfd_langchain.prompts.loader import PromptLoader
from cfd_langchain.agents.literature_agent import LiteratureSurveyAgent
from cfd_langchain.utils import strip_json_fences


AI_SCIENTIST_STYLE_CHECKLIST = """
Incorporate modern autonomous-science writing best practices inspired by AI Scientist-style workflows:
1) Explicit novelty positioning vs closest prior work.
2) Claim-evidence mapping (each major claim tied to an experiment/figure/result).
3) Transparent negative results and failure analysis.
4) Ablation/sensitivity discussion and robustness caveats.
5) Reproducibility-first reporting (exact config, seeds, environment, solver/version).
6) Limitations and concrete next-experiment proposals.
7) Avoid overclaiming; calibrate conclusions to evidence strength.
""".strip()


class WriterAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("WriterAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.2)
        self.lit_agent = LiteratureSurveyAgent(model=model)

    def write_section(self, section_context: str) -> str:
        system = self.prompts.get(
            "system_prompt", "You are a LaTeX paper writer for CFD."
        )
        user = self.prompts.get(
            "user_prompt",
            "SECTION CONTEXT:\n{section_context}\nReturn LaTeX section only.",
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"section_context": section_context}).content

    def collect_citations(
        self, citation_context: str, total_rounds: int = 2
    ) -> List[Dict[str, Any]]:
        sys_t = self.prompts.get("citation_system_prompt", "")
        usr_t = self.prompts.get("citation_user_prompt", "")
        if not sys_t or not usr_t:
            return []

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_t),
                ("human", usr_t),
            ]
        )
        chain = prompt | create_langchain_llm(self.model, temperature=0.1)

        collected: List[Dict[str, Any]] = []
        for r in range(1, max(1, total_rounds) + 1):
            raw = chain.invoke(
                {
                    "citation_context": citation_context,
                    "round": r,
                    "total_rounds": total_rounds,
                }
            ).content
            try:
                parsed = json.loads(strip_json_fences(raw))
            except Exception:
                continue
            cites = parsed.get("citations", []) if isinstance(parsed, dict) else []
            for c in cites:
                if isinstance(c, dict):
                    collected.append(c)
            if isinstance(parsed, dict) and parsed.get("action") == "stop":
                break

        # dedup by title
        seen = set()
        uniq = []
        for c in collected:
            t = (c.get("title") or "").strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            uniq.append(c)
        return uniq

    def write_paper_with_literature(
        self,
        topic: str,
        section_context: str,
        ideation_literature_bundle: Optional[List[Dict[str, Any]]] = None,
        visualization_bundle: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Prefer workflow ideation literature to keep end-to-end context aligned.
        lit_bundle: Any = ideation_literature_bundle or []
        if not lit_bundle:
            try:
                lit_bundle = self.lit_agent.survey(idea_text=topic, max_papers=20)
            except Exception as e:
                lit_bundle = {
                    "idea": topic,
                    "semantic_scholar": [],
                    "web_results": [],
                    "synthesis": "",
                    "fetch_error": str(e),
                }

        try:
            citations = self.collect_citations(citation_context=topic, total_rounds=2)
        except Exception:
            citations = []

        system = self.prompts.get(
            "system_prompt",
            "You are a LaTeX paper writer for CFD.",
        )
        user = (
            "Write a complete publication-ready LaTeX paper draft using ALL provided context and artifacts.\n\n"
            "Topic:\n{topic}\n\n"
            "Context (including analysis summaries):\n{section_context}\n\n"
            "Literature survey bundle:\n{lit_bundle}\n\n"
            "Candidate citations:\n{citations}\n\n"
            "Visualization bundle (figures and plot descriptions):\n{viz_bundle}\n\n"
            "Additional mandatory style checklist:\n{checklist}\n\n"
            "Requirements:\n"
            "- Use evidence-grounded claims only.\n"
            "- Include a Related Work section with concrete comparison positioning.\n"
            "- Include an explicit Claim-Evidence table (can be LaTeX tabular) that ties claims to specific experiments, analysis findings, and figures.\n"
            "- Include Failure Cases / Negative Results section.\n"
            "- Include Reproducibility appendix.\n"
            "- When appropriate, refer to generated figures using LaTeX-style references (e.g., Figure~\\ref{fig:...}) and ensure the text interpretation matches the analysis and visualization evidence.\n"
            "Return ONLY LaTeX."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke(
            {
                "topic": topic,
                "section_context": section_context,
                "lit_bundle": json.dumps(lit_bundle),
                "citations": json.dumps(citations),
                "checklist": AI_SCIENTIST_STYLE_CHECKLIST,
                "viz_bundle": json.dumps(visualization_bundle or []),
            }
        ).content

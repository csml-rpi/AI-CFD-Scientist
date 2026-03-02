"""Writer agent: LaTeX paper generation with Sakana AI Scientist v2–aligned features.

Includes: standard sections (Abstract–Conclusion), claim–evidence table,
figure–text alignment, multi-round citations, reproducibility appendix,
failure/negative results section, and mandatory AI-disclosure sentence.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.agents.literature_agent import LiteratureSurveyAgent
from cfd_langgraph.utils import strip_json_fences


# Writer features aligned with Sakana AI Scientist v2: standard sections, claim–evidence,
# VLM-style figure–text alignment, multi-round citations, reproducibility, and mandatory disclosure.
AI_SCIENTIST_V2_STYLE_CHECKLIST = """
Sakana AI Scientist v2–style manuscript requirements (peer-review oriented):

Structure:
1) Standard sections in order: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion, plus appendices as needed.
2) Explicit novelty positioning vs closest prior work in Related Work / Introduction.
3) Claim–evidence mapping: include an explicit Claim–Evidence table (LaTeX tabular) tying each major claim to specific experiments, figures, and quantitative results. No unsupported claims.
4) Transparent negative results and failure analysis: dedicate a subsection (e.g. Failure Cases / Negative Results) to failed or inconclusive runs, syntax/execution issues, and what was learned.
5) Ablation/sensitivity discussion and robustness caveats where applicable.
6) Limitations and concrete next-experiment proposals (what would you run next and why).

Reproducibility (reproducibility-first reporting):
7) Reproducibility appendix or section: exact solver name and version, OpenFOAM/case setup, mesh/resolution, boundary conditions, time step and end time, and how to run the case (e.g. Foam-Agent or allRun script). No vague “as in code” without specifics.
8) Environment and config: mention WM_PROJECT_DIR / OpenFOAM version if relevant, and any env vars or paths a reader would need.

Figures and consistency (VLM-style alignment):
9) Every figure must be referenced in the main text; figure captions must accurately describe what is shown (field, slice/contour type, time step, case). Text interpretation must match the actual analysis and visualization evidence—no hallucinated numbers or mismatched descriptions.
10) Use LaTeX \\ref{fig:...} for all figures; ensure numbering and captions are complete.

Ethics and calibration:
11) Avoid overclaiming; calibrate conclusions to evidence strength. Do not state implications not supported by the experiments.
12) Mandatory disclosure: in the Abstract or Methods section, include a single clear sentence that this draft was generated with an automated CFD Scientist pipeline (or “AI-assisted workflow”) and that results and figures come from the provided experiments and analysis.
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

        # Multi-round citation (AI Scientist v2–style) for better coverage.
        try:
            citations = self.collect_citations(citation_context=topic, total_rounds=3)
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
            "Candidate citations (use these; add \\cite{} where appropriate):\n{citations}\n\n"
            "Visualization bundle (figures and plot descriptions—every figure listed here must appear in the paper with a caption and be referenced in the text):\n{viz_bundle}\n\n"
            "Mandatory style checklist (Sakana AI Scientist v2–aligned):\n{checklist}\n\n"
            "Requirements:\n"
            "- Structure: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion; add Reproducibility appendix and Claim–Evidence table.\n"
            "- Use evidence-grounded claims only; every claim must map to an experiment, figure, or number in the analysis.\n"
            "- Include a Failure Cases / Negative Results subsection.\n"
            "- Include mandatory disclosure in Abstract or Methods: one sentence that this draft was generated with an automated CFD Scientist (AI-assisted) pipeline and that results/figures come from the provided experiments and analysis.\n"
            "- Reference every figure with \\ref{fig:...}; ensure captions and in-text descriptions match the actual visualization data (no hallucinated values).\n"
            "Return ONLY LaTeX (no markdown, no explanation outside comments)."
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
                "checklist": AI_SCIENTIST_V2_STYLE_CHECKLIST,
                "viz_bundle": json.dumps(visualization_bundle or []),
            }
        ).content

"""Paper reviewer agent: evaluates LaTeX paper quality for formatting, figures, content, and publishability."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class PaperReviewerAgent:
    """Reviews a LaTeX paper for formatting, figure quality, content coherence, and publishability."""

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("PaperReviewerAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.1)

    def review(
        self,
        tex_content: str,
        compile_ok: bool = True,
        compile_error: str = "",
        reference_report: str = "",
        valid_figure_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Review the paper and return pass/fail with recommendations.

        Returns:
            {
                "pass": bool,
                "score": float (0-1),
                "formatting_ok": bool,
                "figures_ok": bool,
                "content_ok": bool,
                "coherent": bool,
                "publishable": bool,
                "recommendations": list[str],
                "summary": str,
            }
        """
        system = self.prompts.get(
            "system_prompt",
            "You are an expert academic paper reviewer for CFD/engineering journals.",
        )
        user = self.prompts.get(
            "user_prompt",
            "Review this LaTeX paper.\n\nCompilation: {compile_status}\n\nLaTeX content:\n{tex_content}\n\n"
            "Conclusion must be a single paragraph; if Conclusion is formatted as bullets/numbered list, mark fail and recommend rewriting as one paragraph.\n\n"
            "Return JSON with: pass (bool), score (0-1), formatting_ok, figures_ok, content_ok, coherent, publishable (all bool), "
            "recommendations (list of specific fixes), summary (brief).",
        )
        if compile_ok:
            compile_status = "OK"
        else:
            compile_status = f"FAILED (fix these so LaTeX compiles):\n{compile_error}"
            if valid_figure_paths:
                compile_status += "\n\nVALID figure paths to use in \\includegraphics (relative to project root):\n" + "\n".join("- " + p for p in valid_figure_paths[:30])
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user),
        ])
        chain = prompt | self.llm
        tex_for_review = tex_content or ""

        out = chain.invoke({
            "compile_status": compile_status,
            # Required by the prompt template (see prompts.yaml).
            # Always provide the variable to avoid LangChain KeyError.
            "compile_error": compile_error or "",
            # Optional extra input used by the prompt template (see prompts.yaml).
            "reference_report": reference_report or "",
            "tex_content": tex_for_review,
        })
        raw = getattr(out, "content", str(out))

        try:
            parsed = json.loads(strip_json_fences(raw))
        except Exception:
            return {
                "pass": False,
                "score": 0.0,
                "formatting_ok": False,
                "figures_ok": False,
                "content_ok": False,
                "coherent": False,
                "publishable": False,
                "recommendations": ["Reviewer output parse failed; re-run review."],
                "summary": "Could not parse reviewer response.",
            }

        return {
            "pass": bool(parsed.get("pass", False)),
            "score": float(parsed.get("score", 0.0)),
            "formatting_ok": bool(parsed.get("formatting_ok", False)),
            "figures_ok": bool(parsed.get("figures_ok", False)),
            "content_ok": bool(parsed.get("content_ok", False)),
            "coherent": bool(parsed.get("coherent", False)),
            "publishable": bool(parsed.get("publishable", False)),
            "recommendations": list(parsed.get("recommendations", [])) or [],
            "summary": str(parsed.get("summary", "")),
        }

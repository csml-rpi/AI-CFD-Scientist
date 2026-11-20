"""Paper reviewer agent: evaluates LaTeX paper quality for formatting, figures, content, and publishability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import extract_json_object, strip_json_fences
from cfd_langgraph.llm.reply import reply_text


# The final audit's H13 reads review.json#figure_physics_ok, a judgement only
# a reader of the figures can make; nothing produced it, so every paper failed
# H13 as "missing" (shelf_demo_dafoam_20261019_1057, shelf_demo_hystrath_20261022_1030).
FIGURE_PHYSICS_SYSTEM = (
    "You check CFD paper figures for PHYSICAL consistency (not layout). For each image, using its file name, "
    "the paper's text and captions: does the field look physically plausible for the flow described (flow "
    "direction and inlet/outlet, no-slip or slip walls, stagnation and high pressure where the flow is brought "
    "to rest, shocks ahead of bodies in supersonic flow, wakes and recirculation where expected, value ranges "
    "and units consistent with the stated conditions)? Does each caption and the text describe what the image "
    "actually shows (right field, right case, right quantity)? A density field shown in place of pressure is "
    "fine only if the caption says so. Set figure_physics_ok=false if any figure is physically implausible or "
    "misdescribed; name the file and the problem in reason.\n"
    "Return ONLY JSON: {\"figure_physics_ok\": bool, \"reason\": \"one or two sentences\", "
    "\"per_figure\": [{\"file\": \"name.png\", \"ok\": bool, \"issue\": \"string\"}]}"
)


class PaperReviewerAgent:
    """Reviews a LaTeX paper for formatting, figure quality, content coherence, and publishability."""

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("PaperReviewerAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.0)

    def review(
        self,
        tex_content: str,
        compile_ok: bool = True,
        compile_error: str = "",
        reference_report: str = "",
        valid_figure_paths: Optional[List[str]] = None,
        study_brief: str = "",
    ) -> Dict[str, Any]:
        """
        Review the paper and return pass/fail with recommendations.

        ``study_brief`` is the user's request for the study. Without it the
        review asked for a mesh study and longer runs the user had ruled out,
        and scored the paper down for their absence
        (nn_in_solver_demo_20261022_1129).

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
        if str(study_brief or "").strip():
            system = system + (
                "\n\nTHE USER'S BRIEF FOR THIS STUDY (what was asked, and what the user ruled out or "
                "limited):\n"
                # data, never template variables
                + str(study_brief).strip()[:12000].replace("{", "{{").replace("}", "}}")
                + "\n\nWork the brief rules out, or "
                "that exceeds the compute and time limits it sets, is not missing work: do not ask for "
                "it and do not lower the score for its absence. Ask only that the paper states such a "
                "limit plainly where it bears on a claim.")
        # The pipeline writes and re-inserts the figure paths: asked each round
        # for relative paths, the writer was told to keep the exact ones, and the
        # score fell for it every time (shelf_demo_hystrath_20261022_1030).
        system = system + (
            "\n\nThe \\includegraphics file paths are set by the pipeline (absolute, so the paper "
            "compiles from any folder); do not ask to change them and do not score them.")
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
        raw = reply_text(out)

        try:
            parsed = json.loads(extract_json_object(raw))
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
                "needs_additional_visualization": False,
                "additional_viz_specs": [],
                "regenerate_batch_figures": True,
            }

        rb = parsed.get("regenerate_batch_figures")
        if rb is None and compile_ok and not bool(parsed.get("figures_ok", True)):
            rb = True

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
            "needs_additional_visualization": bool(parsed.get("needs_additional_visualization", False)),
            "additional_viz_specs": list(parsed.get("additional_viz_specs", [])) or [],
            "regenerate_batch_figures": rb,
        }

    def judge_figure_physics(
        self,
        image_paths: List[Path],
        tex_content: str,
        max_images: int = 16,
    ) -> Dict[str, Any]:
        """{figure_physics_ok: bool | None, figure_physics_reason: str,
        figure_physics_per_figure: list}. None means not judged (no figures,
        or the reply could not be read): the audit then reports it missing
        rather than passing it."""
        from cfd_langgraph.viz_creator import _images_to_blocks

        paths = [Path(p) for p in image_paths if Path(p).is_file()]
        if not paths:
            return {"figure_physics_ok": None, "figure_physics_reason": "no rendered figures to judge",
                    "figure_physics_per_figure": []}
        shown = paths[:max_images]
        content: List[Any] = [{
            "type": "text",
            "text": (
                "PAPER (LaTeX, truncated):\n" + (tex_content or "")[:40_000]
                + "\n\nThe images follow, each after its file name."
            ),
        }]
        for p in shown:
            blocks = _images_to_blocks([p], max_images=1)
            if blocks:
                content.append({"type": "text", "text": f"File: {p.name}"})
                content.extend(blocks)
        system = self.prompts.get("figure_physics_system_prompt", FIGURE_PHYSICS_SYSTEM)
        try:
            out = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
            parsed = json.loads(extract_json_object(reply_text(out)))
        except Exception as exc:  # noqa: BLE001
            return {"figure_physics_ok": None, "figure_physics_reason": f"figure physics check failed: {exc}",
                    "figure_physics_per_figure": []}
        ok = parsed.get("figure_physics_ok") if isinstance(parsed, dict) else None
        reason = str(parsed.get("reason", "")).strip() if isinstance(parsed, dict) else ""
        if len(paths) > len(shown):
            reason += f" (judged {len(shown)} of {len(paths)} figures)"
        return {
            "figure_physics_ok": ok if isinstance(ok, bool) else None,
            "figure_physics_reason": reason,
            "figure_physics_per_figure": list(parsed.get("per_figure") or []) if isinstance(parsed, dict) else [],
        }

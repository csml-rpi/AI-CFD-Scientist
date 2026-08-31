from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.ideation import build_literature_context
from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class HypothesisCritiqueAgent:
    """The "Critique" step of propose -> critique -> rank.

    Checks one candidate idea (already novelty-gated by ``run_ideation_batch``)
    for physical plausibility and whether it can actually become an
    executable FoamAgent case — the two things a single novelty check doesn't
    catch. Modeled on the reflection/critic pattern from Google DeepMind's
    AI co-scientist (arXiv 2502.18864).
    """

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.prompts = prompt_loader.section("IdeationCritiqueAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.15)

    def critique(
        self,
        idea_json: Dict[str, Any],
        lit_items: List[Dict[str, Any]],
        research_topic: str = "",
        case_context: str = "",
    ) -> Dict[str, Any]:
        sys_t = self.prompts.get("critique_system_prompt", "")
        usr_t = self.prompts.get("critique_user_prompt", "")
        if not sys_t or not usr_t:
            raise ValueError("Missing IdeationCritiqueAgent prompts")

        if case_context:
            # The implementability criterion asks whether the idea specifies
            # geometry, BCs, solver and parameters. When the study runs on an
            # existing case those are already decided, and a reviewer blind to
            # that rejects every candidate for "missing geometry/mesh/BC
            # details" — measured on a real run, 6 of 6 rejected for exactly
            # that, leaving the study with nothing to approve.
            sys_t = (
                sys_t
                + "\n\nFIXED CASE SETUP — this study runs on an existing case, described "
                "below. Its geometry, mesh, boundary conditions, solver, numerics and flow "
                "parameters are GIVEN. Do NOT reject an idea for failing to restate them, and "
                "do NOT treat them as details that would have to be guessed. Judge only what "
                "the idea itself adds on top of this case.\n"
                + case_context
            )

        prompt = ChatPromptTemplate.from_messages([("system", sys_t), ("human", usr_t)])
        chain = prompt | self.llm
        raw = chain.invoke(
            {
                "research_topic": research_topic,
                "idea_json": json.dumps(idea_json, ensure_ascii=False, indent=2),
                "literature_context": build_literature_context(lit_items),
            }
        ).content

        try:
            parsed = json.loads(strip_json_fences(raw))
            if not isinstance(parsed, dict):
                raise ValueError("not dict")
        except Exception:
            parsed = {
                "verdict": "reject",
                "issues": ["Critique response was not parseable JSON."],
                "raw": raw,
            }

        verdict = str(parsed.get("verdict", "reject") or "reject").strip().lower()
        plausible = parsed.get("plausible") is True
        feasible = parsed.get("feasible_for_foamagent") is True
        # A prose verdict cannot override contradictory boolean gates.  The
        # old implementation accepted {verdict: pass, plausible: false},
        # allowing an explicitly unphysical idea into the ranking stage.
        parsed["verdict"] = "pass" if verdict == "pass" and plausible and feasible else "reject"
        parsed["plausible"] = plausible
        parsed["feasible_for_foamagent"] = feasible
        parsed.setdefault("distinct_from_literature", True)
        if not isinstance(parsed.get("issues"), list):
            parsed["issues"] = [str(parsed.get("issues") or "Invalid critique issues field.")]
        parsed.setdefault("reason", "")
        return parsed

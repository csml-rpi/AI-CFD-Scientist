from __future__ import annotations

from cfd_langgraph.utils import structured_output

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm.caching import cacheable_human_message
from . import prompts as P
from cfd_langgraph.llm.reply import reply_text


def review_errors(
    llm: Any,
    *,
    tutorial_reference: str,
    foamfiles_xml: str,
    error_logs: str,
    user_requirement: str,
    history_text: str = "",
    similar_case_advice_block: str = "",
) -> str:
    """FoamAgent Stage 8a (verbatim prompt): diagnose the error logs and
    propose fixes in free text — does not touch any files itself."""
    system = P.with_version_facts(P.REVIEWER_SYSTEM_PROMPT)
    # The reviewer runs once per retry round — up to max_loop times for a
    # stubborn case — and re-sends the same tutorial reference and advice
    # block every round. Those lead the prompt, and the system prompt above
    # takes no format arguments at all, so everything up to <foamfiles> is a
    # genuinely stable prefix. Split there; the concatenation is identical to
    # the single formatted string it replaces.
    stable, _, tail = P.REVIEWER_USER_PROMPT.format(
        tutorial_reference=tutorial_reference,
        similar_case_advice_block=similar_case_advice_block,
        foamfiles_xml=foamfiles_xml,
        error_logs=error_logs,
        user_requirement=user_requirement,
        history_text=history_text,
    ).partition(P.REVIEW_CACHE_SPLIT_MARKER)
    raw = reply_text(llm.invoke([
        SystemMessage(content=system),
        cacheable_human_message(llm, stable, P.REVIEW_CACHE_SPLIT_MARKER + tail),
    ]))
    return (raw or "").strip()


class _TargetFile(BaseModel):
    file: str
    changes: str
    # A file the case should not have at all. Asked for the full new content
    # of a file the plan meant to remove, the model returned nothing, five
    # times, and the empty turn ended the whole case run (four runs lost in
    # shelf_demo_dafoam_20261011_1437 and shelf_demo_hystrath_20261011_1602).
    delete: bool = False


class _RewritePlan(BaseModel):
    target_files: List[_TargetFile] = Field(default_factory=list)


def plan_rewrite(
    llm: Any,
    *,
    foamfiles_xml: str,
    error_logs: str,
    review_analysis: str,
    user_requirement: str,
) -> List[Dict[str, str]]:
    """FoamAgent Stage 8b (verbatim prompt): pick the minimal set of files to
    edit, and what to change in each, given the reviewer's analysis."""
    system = P.with_version_facts(P.REWRITE_PLANNER_SYSTEM_PROMPT)
    user = P.REWRITE_PLANNER_USER_PROMPT.format(
        foamfiles_xml=foamfiles_xml, error_logs=error_logs,
        review_analysis=review_analysis, user_requirement=user_requirement,
    )
    out: _RewritePlan = structured_output(llm, _RewritePlan).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return [{"file": t.file, "changes": t.changes, "delete": t.delete} for t in out.target_files]


class _SetupItem(BaseModel):
    requirement: str
    file: str = ""
    found: str = ""
    status: str
    fix: str = ""


class _SetupCheck(BaseModel):
    items: List[_SetupItem] = Field(default_factory=list)


SETUP_PROBLEMS = ("mismatch", "missing")


def check_setup(llm: Any, *, foamfiles_xml: str, user_requirement: str, solver_notes: str = "") -> List[Dict[str, str]]:
    """Every value the requirement fixes, where the case sets it, and whether
    it matches: one item per value, status match/mismatch/missing/unclear."""
    user = P.SETUP_CHECK_USER_PROMPT.format(user_requirement=user_requirement, foamfiles_xml=foamfiles_xml,
                                            solver_notes=solver_notes.strip() or "(none)")
    out: _SetupCheck = structured_output(llm, _SetupCheck).invoke(
        [SystemMessage(content=P.with_version_facts(P.SETUP_CHECK_SYSTEM_PROMPT)), HumanMessage(content=user)]
    )
    items = []
    for item in out.items:
        status = item.status.strip().lower()
        items.append({"requirement": item.requirement, "file": item.file, "found": item.found,
                      "status": status if status in ("match", "mismatch", "missing", "unclear") else "unclear",
                      "fix": item.fix})
    return items

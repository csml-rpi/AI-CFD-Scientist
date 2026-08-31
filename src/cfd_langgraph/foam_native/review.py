from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm.caching import cacheable_human_message
from . import prompts as P


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
    system = P.REVIEWER_SYSTEM_PROMPT
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
    raw = llm.invoke([
        SystemMessage(content=system),
        cacheable_human_message(llm, stable, P.REVIEW_CACHE_SPLIT_MARKER + tail),
    ]).content
    return (raw or "").strip()


class _TargetFile(BaseModel):
    file: str
    changes: str


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
    system = P.REWRITE_PLANNER_SYSTEM_PROMPT
    user = P.REWRITE_PLANNER_USER_PROMPT.format(
        foamfiles_xml=foamfiles_xml, error_logs=error_logs,
        review_analysis=review_analysis, user_requirement=user_requirement,
    )
    out: _RewritePlan = llm.with_structured_output(_RewritePlan).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return [{"file": t.file, "changes": t.changes} for t in out.target_files]

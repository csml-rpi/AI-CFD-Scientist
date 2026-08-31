from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm.caching import cacheable_human_message
from . import prompts as P


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def write_file_initial(
    llm: Any,
    *,
    file_name: str,
    folder_name: str,
    user_requirement: str,
    tutorial_reference: str,
    written_files_ctx: str,
    case_solver: str,
) -> str:
    """FoamAgent Stage 5 (verbatim prompt): write one case file from scratch."""
    system = P.INITIAL_WRITE_SYSTEM_PROMPT.format(case_solver=case_solver)
    # Split at the end of the tutorial reference: the requirement and the
    # tutorial are byte-identical for every file written in this case, and
    # together they are by far the largest part of the prompt. Everything
    # after — the growing written-files context and the per-file instruction
    # — differs per call. The two halves are concatenated by the provider, so
    # the rendered prompt is character-for-character what it was before.
    stable, _, tail = P.INITIAL_WRITE_USER_PROMPT.format(
        user_requirement=user_requirement, tutorial_reference=tutorial_reference,
        written_files_ctx=written_files_ctx, file_name=file_name, folder_name=folder_name,
    ).partition(P.WRITE_CACHE_SPLIT_MARKER)
    raw = llm.invoke([
        SystemMessage(content=system),
        cacheable_human_message(llm, stable, P.WRITE_CACHE_SPLIT_MARKER + tail),
    ]).content
    return _strip_code_fences(raw)


def edit_file(
    llm: Any,
    *,
    file_name: str,
    folder_name: str,
    changes: str,
    current_content: str,
    written_files_ctx: str,
    case_solver: str,
) -> str:
    """FoamAgent Stage 8c (verbatim prompt): minimally edit one existing file
    during the review/rewrite loop."""
    system = P.EDIT_WRITE_SYSTEM_PROMPT.format(case_solver=case_solver)
    user = P.EDIT_WRITE_USER_PROMPT.format(
        file_name=file_name, folder_name=folder_name,
        changes=changes, current_content=current_content, written_files_ctx=written_files_ctx,
    )
    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    return _strip_code_fences(raw)

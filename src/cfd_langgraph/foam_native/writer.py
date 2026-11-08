from __future__ import annotations

from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm.caching import cacheable_human_message
from . import prompts as P
from cfd_langgraph.llm.reply import reply_text


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
    system = P.with_version_facts(P.INITIAL_WRITE_SYSTEM_PROMPT.format(case_solver=case_solver))
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
    raw = reply_text(llm.invoke([
        SystemMessage(content=system),
        cacheable_human_message(llm, stable, P.WRITE_CACHE_SPLIT_MARKER + tail),
    ]))
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
    system = P.with_version_facts(P.EDIT_WRITE_SYSTEM_PROMPT.format(case_solver=case_solver))
    user = P.EDIT_WRITE_USER_PROMPT.format(
        file_name=file_name, folder_name=folder_name,
        changes=changes, current_content=current_content, written_files_ctx=written_files_ctx,
    )
    raw = reply_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]))
    return _strip_code_fences(raw)


class _Replacement(BaseModel):
    old: str
    new: str


class _Patch(BaseModel):
    replacements: List[_Replacement] = Field(default_factory=list)


def apply_replacements(content: str, replacements: List[Any]) -> tuple[str, List[str]]:
    """``content`` with each replacement applied in turn, and what could not be
    applied: an "old" that is empty, absent, or found more than once."""
    problems: List[str] = []
    for i, r in enumerate(replacements, start=1):
        old, new = getattr(r, "old", ""), getattr(r, "new", "")
        count = content.count(old) if old else 0
        if count != 1:
            problems.append(f"replacement {i}: its old text occurs {count} times in the file "
                            f"(it must occur exactly once): {old[:120]!r}")
            continue
        content = content.replace(old, new, 1)
    return content, problems


def edit_file_by_patches(
    llm: Any,
    *,
    file_name: str,
    folder_name: str,
    changes: str,
    current_content: str,
    written_files_ctx: str,
    case_solver: str,
    attempts: int = 3,
) -> str:
    """Edit a file too long to send back whole, by exact replacements the
    model names and the code applies. The replacements that apply are kept;
    the model is shown the file with them made and asked again for the ones
    that did not. Asked for the whole list again, a reply of 37 replacements
    on a 103 KB DAFoam script failed on its 37th, every time, and the edit
    fell back to a whole rewrite that dropped all 37
    (nn_airfoil_s814_20261026_1537, case_003). After ``attempts`` replies
    with something still unapplied the edit fails, and the file stays as it
    was."""
    from cfd_langgraph.utils import structured_output

    system = P.with_version_facts(P.EDIT_PATCH_SYSTEM_PROMPT.format(case_solver=case_solver))

    def ask(content: str) -> str:
        return P.EDIT_PATCH_USER_PROMPT.format(
            file_name=file_name, folder_name=folder_name, changes=changes,
            current_content=content, written_files_ctx=written_files_ctx,
        )

    content = current_content
    user = ask(content)
    feedback = ""
    problems: List[str] = []
    for _ in range(max(1, attempts)):
        out: _Patch = structured_output(llm, _Patch).invoke(
            [SystemMessage(content=system), HumanMessage(content=user + feedback)])
        if not out.replacements:
            problems = ["no replacements were returned"]
            feedback = "\n\nYour previous reply held no replacements. Return the replacements that make the changes."
            continue
        new_content, problems = apply_replacements(content, out.replacements)
        if not problems:
            return new_content
        applied = len(out.replacements) - len(problems)
        if applied:
            content = new_content
            user = ask(content)
            feedback = (f"\n\n{applied} of your previous replacements were applied: the file above has "
                        "them. These could not be applied:\n" + "\n".join(f"- {p}" for p in problems)
                        + "\nReturn replacements for the changes still to be made only, every old text "
                        "copied exactly from the file above.")
        else:
            feedback = ("\n\nYour previous replacements could not be applied, so nothing was changed:\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + "\nReturn the full list again, every old text copied exactly from the current file.")
    raise ValueError("the edit's replacements could not be applied: " + "; ".join(problems)[:600])

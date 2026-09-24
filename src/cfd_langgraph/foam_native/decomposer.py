from __future__ import annotations

from cfd_langgraph.utils import structured_output

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from . import prompts as P


class _Subtask(BaseModel):
    file_name: str
    folder_name: str


class _Subtasks(BaseModel):
    subtasks: List[_Subtask] = Field(default_factory=list)


def decompose_subtasks(
    llm: Any,
    user_requirement: str,
    dir_structure: str,
    dir_counts_str: str = "",
) -> List[Dict[str, str]]:
    """FoamAgent Stage 3 (verbatim prompt): break the requirement into
    (file_name, folder_name) subtasks — the OpenFOAM input files this case needs."""
    system = P.DECOMPOSE_SYSTEM_PROMPT
    user = P.DECOMPOSE_USER_PROMPT.format(
        user_requirement=user_requirement, dir_structure=dir_structure, dir_counts_str=dir_counts_str,
    )
    out: _Subtasks = structured_output(llm, _Subtasks).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return [{"file_name": s.file_name, "folder_name": s.folder_name} for s in out.subtasks]

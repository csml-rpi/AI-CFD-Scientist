from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from . import prompts as P


class _CaseInfo(BaseModel):
    case_name: str = Field(description="Short descriptive case name")
    case_domain: str = Field(description="Must be one of the provided case_domain_list")
    case_category: str = Field(description="Must be one of the provided case_category_list")
    case_solver: str = Field(description="Must be one of the provided case_solver_list")


def parse_requirement(
    llm: Any,
    user_requirement: str,
    *,
    case_domain_list: Optional[List[str]] = None,
    case_category_list: Optional[List[str]] = None,
    case_solver_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """FoamAgent Stage 1 (verbatim prompt, see prompts.py): turn a
    natural-language requirement into {case_name, case_domain,
    case_category, case_solver}."""
    domains = case_domain_list or P.DEFAULT_CASE_DOMAIN_LIST
    categories = case_category_list or P.DEFAULT_CASE_CATEGORY_LIST
    solvers = case_solver_list or P.DEFAULT_CASE_SOLVER_LIST
    system = P.PARSE_SYSTEM_PROMPT.format(
        case_domain_list=domains, case_category_list=categories, case_solver_list=solvers,
    )
    user = P.PARSE_USER_PROMPT.format(user_requirement=user_requirement)
    out: _CaseInfo = llm.with_structured_output(_CaseInfo).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return {
        "case_name": out.case_name or "case",
        "case_domain": out.case_domain if out.case_domain in domains else domains[0],
        "case_category": out.case_category if out.case_category in categories else categories[0],
        "case_solver": out.case_solver if out.case_solver in solvers else solvers[0],
    }

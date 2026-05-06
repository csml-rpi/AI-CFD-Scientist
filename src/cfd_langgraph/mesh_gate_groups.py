"""LLM-driven mesh refinement groups: which physics families need their own mesh study."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MeshPhysicsGroup(BaseModel):
    """One mesh-independence family (same transport/viscosity class, shared best mesh)."""

    group_id: str = Field(
        ...,
        description="Slug a-z0-9_ only, e.g. custom_transport, newtonian_builtin, model_variant_b",
    )
    label: str = Field(..., description="Short human label for logs and manifests")
    case_ids: List[str] = Field(
        default_factory=list,
        description="Every case_id from the requirements list that should use this group's selected mesh",
    )
    mesh_study_baseline_requirement: str = Field(
        ...,
        description=(
            "Full Foam-Agent requirement for the mesh-gate BASELINE run for this group only: "
            "lock BCs/solver/numerics to match downstream cases in this group; fix physics model "
            "(custom vs built-in Newtonian etc.) exactly as those cases will use. Mesh will be "
            "coarsened/refined in later steps — do not embed stale case_snapshot JSON."
        ),
    )
    mesh_seed_source: str = Field(
        default="canonical_base_case",
        description="canonical_base_case | starter_seed — where to copy the starting case tree from",
    )


class MeshGateGroupPlan(BaseModel):
    groups: List[MeshPhysicsGroup] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Brief audit trail for humans")


_GROUP_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _sanitize_group_id(raw: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", (raw or "").lower()).strip("_") or f"group_{idx}"
    if not _GROUP_ID_RE.match(s):
        s = f"group_{idx}"
    return s[:64]


def _compact_requirements_for_llm(requirements: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    lines: List[str] = []
    n = 0
    for r in requirements:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("case_id", ""))
        desc = str(r.get("description", ""))[:400]
        txt = str(r.get("user_requirement_text", ""))[:1200]
        lines.append(f"### {cid}\ndescription: {desc}\nrequirement_excerpt:\n{txt}\n")
        n += len(lines[-1])
        if n > max_chars:
            lines.append("(…truncated…)")
            break
    return "\n".join(lines)


def plan_mesh_refinement_groups_llm(
    *,
    model: str,
    topic: str,
    requirements: List[Dict[str, Any]],
    code_mod_context: str,
) -> MeshGateGroupPlan:
    """Partition cases into physics groups; each group gets its own mesh-gate baseline text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from cfd_langgraph.llm.factory import create_langchain_llm

    case_ids = [str(r.get("case_id")) for r in requirements if isinstance(r, dict) and r.get("case_id")]
    case_ids = [c for c in case_ids if c]
    compact = _compact_requirements_for_llm(requirements)

    system = (
        "You plan **mesh refinement/independence studies** for a CFD experiment batch.\n\n"
        "Each **group** shares one viscosity/turbulence **physics implementation** (e.g. same custom "
        "OpenFOAM transport vs built-in Newtonian). Mesh sensitivity must be evaluated **under that "
        "same physics**, because the best mesh can differ between models.\n\n"
        "Rules:\n"
        "- Read every case_id and its requirement excerpt.\n"
        "- **Partition** case_ids across groups so each case appears in **exactly one** group.\n"
        "- If all cases share one physics, return **one** group containing every case_id.\n"
        "- If some cases use a **custom / code-mod** transport and others use **built-in Newtonian** "
        "(or another built-in), use **at least two** groups (custom vs built-in, etc.).\n"
        "- For **three** distinct model implementations, use **three** groups (plus optional baseline-only group if needed).\n"
        "- `mesh_study_baseline_requirement`: self-contained Foam prompt for the **baseline** mesh-gate "
        "run for that group — same domain/BCs/solver intent as those cases, **physics locked** to that group. "
        "Explicitly forbid changing physics in mesh-only steps. No pasted case_snapshot blobs.\n"
        "- `mesh_seed_source`: prefer **canonical_base_case** when custom libraries live there; use "
        "**starter_seed** only when the group's physics matches the untouched starter and canonical would be wrong.\n"
        "- `group_id`: short slug a-z0-9_ (unique).\n"
        "Return structured output matching the schema."
    )

    user = (
        f"TOPIC:\n{topic[:4000]}\n\n"
        f"ALL case_ids (must partition): {json.dumps(case_ids)}\n\n"
        f"CODE-MOD / IMPLEMENTATION CONTEXT (may be empty):\n{code_mod_context[:8000]}\n\n"
        f"REQUIREMENTS (by case):\n{compact}\n\n"
        "Respond with the structured MeshGateGroupPlan only."
    )

    llm = create_langchain_llm(model=model, temperature=0.1)
    try:
        structured = llm.with_structured_output(MeshGateGroupPlan)
        out = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if isinstance(out, MeshGateGroupPlan) and out.groups:
            return out
    except Exception:
        pass

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = getattr(resp, "content", str(resp))
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return MeshGateGroupPlan.model_validate(data)
    except Exception:
        pass

    # Fallback: single group, all cases
    fb_req = ""
    if requirements and isinstance(requirements[0], dict):
        fb_req = str(requirements[0].get("user_requirement_text") or "")[:4000]
    if not fb_req:
        fb_req = f"Channel / study baseline matching topic: {topic[:500]}"
    return MeshGateGroupPlan(
        groups=[
            MeshPhysicsGroup(
                group_id="all_cases",
                label="default single physics",
                case_ids=list(case_ids),
                mesh_study_baseline_requirement=(
                    "Mesh-gate baseline: use the seeded OpenFOAM case. Preserve BCs, solver, and transport "
                    "models exactly as in the study requirements for these cases. MESH-ONLY changes in follow-on "
                    "steps.\n\n" + fb_req
                ),
                mesh_seed_source="canonical_base_case",
            )
        ],
        reasoning="LLM planner failed or returned empty; fallback single group for all case_ids.",
    )


def normalize_group_plan(plan: MeshGateGroupPlan, all_case_ids: List[str]) -> MeshGateGroupPlan:
    """Sanitize ids and assign any missing case_ids to the first group."""
    seen: set[str] = set()
    groups: List[MeshPhysicsGroup] = []
    for i, g in enumerate(plan.groups or []):
        gid = _sanitize_group_id(g.group_id, i)
        while gid in seen:
            gid = f"{gid}_{i}"
        seen.add(gid)
        src = (g.mesh_seed_source or "canonical_base_case").strip().lower()
        if src not in ("canonical_base_case", "starter_seed"):
            src = "canonical_base_case"
        groups.append(
            MeshPhysicsGroup(
                group_id=gid,
                label=g.label[:200] or gid,
                case_ids=list(dict.fromkeys(g.case_ids or [])),
                mesh_study_baseline_requirement=g.mesh_study_baseline_requirement,
                mesh_seed_source=src,
            )
        )
    if not groups:
        return plan

    covered = {cid for g in groups for cid in g.case_ids}
    missing = [c for c in all_case_ids if c not in covered]
    if missing:
        groups[0].case_ids = list(dict.fromkeys(list(groups[0].case_ids) + missing))
    return MeshGateGroupPlan(groups=groups, reasoning=plan.reasoning)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def resolve_mesh_seed_path(
    *,
    mesh_seed_source: str,
    run_dir: Path,
    state_path: Path,
) -> str:
    src = (mesh_seed_source or "").strip().lower()
    st = _read_json(state_path)

    if src == "starter_seed" and isinstance(st, dict):
        raw = st.get("starter_seed_case_dir") or ""
        if isinstance(raw, str) and raw.strip():
            p = Path(raw.strip()).expanduser().resolve()
            if p.exists() and (p / "system" / "controlDict").is_file():
                return str(p)

    canon = run_dir / "canonical_base_case"
    if canon.exists() and (canon / "system" / "controlDict").is_file():
        return str(canon.resolve())

    if isinstance(st, dict):
        raw = st.get("starter_seed_case_dir") or ""
        if isinstance(raw, str) and raw.strip():
            p = Path(raw.strip()).expanduser().resolve()
            if p.exists() and (p / "system" / "controlDict").is_file():
                return str(p)
    return ""

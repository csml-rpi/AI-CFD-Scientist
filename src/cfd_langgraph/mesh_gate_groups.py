"""LLM-driven mesh refinement groups: which physics families need their own mesh study."""

from __future__ import annotations

from cfd_langgraph.utils import structured_output

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

    llm = create_langchain_llm(model=model, temperature=0.0)
    try:
        structured = structured_output(llm, MeshGateGroupPlan)
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


# ---------------------------------------------------------------------------
# Pairwise mesh-convergence judgment.
#
# Ported (not just wrapped) from scripts/orchestrator_run.py's
# ``_llm_mesh_gate_pair_convergence`` / ``_heuristic_mesh_gate_pair_fallback`` /
# ``_merge_mesh_gate_metrics`` so a clean, in-process caller (this repo's new
# manager tools) doesn't need the flat sys.path hack that script relies on
# (``from timeline_logger import ...``) just to reuse three self-contained
# functions. orchestrator_run.py's own copies are left untouched — this is a
# second, public copy for the same logic, not a redirect, so the existing
# Mode A pipeline is never at risk from this change.
#
# There is no formal Richardson-extrapolation/GCI-number computation anywhere
# in this codebase (checked): "converged" is decided by an LLM judging raw
# QoIs and their percent change between a parent (coarser) and child (finer)
# mesh against a ~5% rule for physically trustworthy quantities, with a
# non-LLM heuristic fallback and a metric-name magnitude floor to avoid
# treating noise-dominated QoIs as mesh sensitivity.
# ---------------------------------------------------------------------------


def merge_mesh_gate_metrics(current: List[str], suggested: List[str], max_metrics: int = 10) -> List[str]:
    out: List[str] = []
    for m in suggested + current:
        s = str(m).strip()
        if s and s not in out:
            out.append(s)
    return out[:max_metrics]


def heuristic_mesh_gate_pair_fallback(
    q_a: Dict[str, Any],
    q_b: Dict[str, Any],
    pct_changes: Dict[str, float],
    *,
    magnitude_floor: float = 1e-9,
) -> Dict[str, Any]:
    """If the LLM is unavailable: never trust relative % when both sides are ~zero/noise.

    Only apply the 5% rule to metrics whose magnitude exceeds a small floor.
    If none qualify, stop refinement (converged=True) to avoid an infinite
    spiral on garbage QoIs.
    """
    reliable_pct: Dict[str, float] = {}
    for k, p in pct_changes.items():
        va, vb = q_a.get(k), q_b.get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        fa, fb = float(va), float(vb)
        if max(abs(fa), abs(fb)) >= magnitude_floor:
            reliable_pct[k] = float(p)
    if not reliable_pct:
        return {
            "converged": True,
            "reason": (
                "Heuristic fallback: no paired QoI exceeded the magnitude floor; "
                "relative percent change is not meaningful — stopping refinement spiral."
            ),
            "qoi_reliability": "unreliable",
            "recommended_metrics_for_retry": [],
            "source": "heuristic_fallback",
        }
    conv = all(v <= 5.0 for v in reliable_pct.values())
    return {
        "converged": conv,
        "reason": f"Heuristic fallback: applied 5% rule only to {list(reliable_pct.keys())} (|Q|>={magnitude_floor}).",
        "qoi_reliability": "mixed",
        "recommended_metrics_for_retry": [],
        "source": "heuristic_fallback",
    }


def llm_mesh_gate_pair_convergence(
    llm: Any,
    *,
    parent_label: str,
    child_label: str,
    q_a: Dict[str, Any],
    q_b: Dict[str, Any],
    pct_changes: Dict[str, float],
    metrics_requested: List[str],
    topic_excerpt: str,
    requirement_excerpt: str,
    metric_attempt_index: int,
    max_metric_attempts: int,
) -> Dict[str, Any]:
    """LLM judges mesh sensitivity for one parent -> child pair using raw QoIs
    and naive percent deltas. May recommend different metrics for a re-run of
    analyze.py when values are noise-dominated."""

    class _MeshGatePairDecision(BaseModel):
        converged: bool = Field(
            description=(
                "True if the coarser (parent) mesh is adequate: for QoIs you judge trustworthy, "
                "parent vs child differ by at most ~5% (standard mesh-independence target), "
                "OR QoIs are unreliable/noise-dominated and further refinement should stop."
            )
        )
        reason: str = Field(description="Short engineering rationale referencing magnitudes, not only %.")
        qoi_reliability: str = Field(
            description="One of: good, mixed, unreliable — are extracted QoIs trustworthy for a % comparison?"
        )
        recommended_metrics_for_retry: List[str] = Field(
            default_factory=list,
            description=(
                "If qoi_reliability is unreliable and a different metric set could help "
                "(e.g. early-time bulk U, pressure_drop_proxy, wall shear), suggest metric names "
                "for scripts/analyze.py. Empty if no retry or attempt budget exhausted."
            ),
        )

    payload_preview = {
        "parent": parent_label,
        "child": child_label,
        "metrics_requested": metrics_requested,
        "qoi_parent": {k: q_a.get(k) for k in sorted(q_a.keys()) if not str(k).startswith("_")},
        "qoi_child": {k: q_b.get(k) for k in sorted(q_b.keys()) if not str(k).startswith("_")},
        "naive_percent_change_parent_to_child": pct_changes,
        "metric_attempt_index": metric_attempt_index,
        "max_metric_attempts": max_metric_attempts,
    }
    system_prompt = (
        "You are a senior CFD engineer judging mesh sensitivity between two OpenFOAM runs "
        "(parent = coarser mesh, child = finer mesh).\n"
        "You receive raw QoIs from an automated PyVista/LLM extraction and naive relative percent changes "
        "computed as |v_child - v_parent| / max(|v_parent|, 1e-12) * 100.\n\n"
        "Rules:\n"
        "- If expected velocities are O(1) m/s but reported QoIs are ~1e-12 or smaller, treat them as "
        "noise / wrong time window / missing driving force — NOT as mesh sensitivity. "
        "Percent changes between noise floors are meaningless (often 10^4-10^5 %).\n"
        "- Prefer metrics anchored in physics: sustained bulk or centreline speed, pressure drop proxy, "
        "wall shear, integrated flow rate — evaluated at the final timestep (developed steady state when applicable).\n"
        "- If current metrics are unreliable, set qoi_reliability=unreliable and suggest "
        "recommended_metrics_for_retry (e.g. bulk_velocity_Ux, pressure_drop_proxy, wall_shear_mean).\n"
        "- If metric_attempt_index >= max_metric_attempts - 1, do NOT suggest retries; "
        "set converged=true if further refinement would only chase numerical noise, and explain.\n"
        "- 5% convergence rule (for trustworthy QoIs only): when qoi_reliability is good or mixed, "
        "if all key physics QoIs you rely on change by <= 5% from parent to child, that IS an acceptable "
        "mesh convergence outcome — set converged=true and say so in reason.\n"
        "- If any trustworthy key QoI changes by > 5%, set converged=false so refinement can continue "
        "(unless the change is clearly a numerical artifact — then explain).\n"
        "- converged=true means: stop refining (parent mesh is the selected level for this gate). "
        "converged=false means: trustworthy QoIs still differ by more than ~5% — continue refinement.\n"
    )
    user_prompt = (
        f"Study topic (excerpt):\n{topic_excerpt[:3000]}\n\n"
        f"Requirement (excerpt):\n{requirement_excerpt[:6000]}\n\n"
        f"Pair data (JSON):\n{json.dumps(payload_preview, indent=2)[:14000]}\n"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = structured_output(llm, _MeshGatePairDecision)
        out: _MeshGatePairDecision = structured.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return {
            "converged": bool(out.converged),
            "reason": str(out.reason or "").strip(),
            "qoi_reliability": str(out.qoi_reliability or "mixed").strip().lower(),
            "recommended_metrics_for_retry": [
                str(x).strip() for x in (out.recommended_metrics_for_retry or []) if str(x).strip()
            ],
            "source": "llm",
        }
    except Exception as exc:
        fb = heuristic_mesh_gate_pair_fallback(q_a, q_b, pct_changes)
        fb["reason"] = f"LLM mesh-gate decision failed ({exc}); {fb.get('reason', '')}"
        fb["source"] = "heuristic_after_llm_error"
        return fb

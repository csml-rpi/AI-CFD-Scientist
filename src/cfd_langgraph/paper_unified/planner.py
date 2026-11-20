"""LLM plan: omit duplicate experiments, label cases for prose, plan paper figures."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import strip_json_fences
from cfd_langgraph.llm.reply import reply_text
from cfd_langgraph.paper_unified.figure_contract import (
    FIGURE_KINDS,
    PER_CASE_KINDS,
    normalize_figure_jobs,
)


PLANNER_SYSTEM = """You are the planning head for a CFD journal manuscript tied to an automated experiment run.

Your job is to return ONE JSON object (no markdown fences) with this schema:
{
  "omit_case_ids": ["case_xxx", ...],
  "omit_rationale": "short string",
  "case_order_for_paper": ["case_001", ...],
  "case_display_labels": {"case_001": "1", "case_003": "2", ...},
  "figure_jobs": [
    {
      "case_id": "case_001",
      "kind": "u_contour",
      "field": "U",
      "what_to_visualize": "detailed instruction for PyVista: fields, views, profiles; mention horizontal layout if channel is long/thin",
      "priority": 1
    }
  ],
  "unified_viz_brief": "ONE string for a single batch PyVista script: list every PNG to create (names suggested), multi-panel layouts (rows/columns), fields (Ux, etc.), coherent colormap/scaling across cases, horizontal layout for thin channels. Omit nuEff unless needed and likely present.",
  "analysis_narrative_hints": "what the Results/Discussion should emphasize given the experiments",
  "topic_paragraph_for_intro": "one paragraph introducing the study without case IDs"
}

Rules:
- **Duplicates / repetitions:** If two experiments are true repeats (same intent, same parameters, no new science), put the redundant case_ids in omit_case_ids. Do NOT plan figure_jobs for omitted cases. Readers should never hear that something was a duplicate—just omit.
- **Naming:** Display labels must be human prose: "case 1", "case 2" mapping (values are strings like "1", "2"). Never use "case_001" in narrative instructions for the writer; the writer will map IDs internally.
- **Figures:** Each figure_job targets one case_id (OpenFOAM case folder). Requests must be achievable with PyVista reading the case .foam file. Do not ask for nuEff/effective viscosity plots unless that field is likely present; prefer U, p, velocity magnitude, streamwise profiles.
- **Kinds:** every figure_job has a `kind` from this list only: {kinds}. Per-case kinds ({per_case_kinds}) need a case_id; `case_comparison` is a figure drawn from several cases (case_id null).
- **Mandatory per-case contours:** for EVERY included case whose manifest status is success, plan one `u_contour` job (velocity) and one `p_contour` job (pressure). Add other kinds as the study needs.
- **Field names differ by solver** (U, p, U_Ar, p_Ar, rhoN_Ar, ...). Set `field` to a name listed for that case under CASE FIELDS: the velocity-like field for u_contour, the pressure-like field for p_contour. Never assume U and p exist.
- **File names:** do not set `id` or `out_png`; the pipeline assigns them (paper_figs/<case_id>_<kind>.png) and the figure script must write exactly those files.
- **Order:** case_order_for_paper lists included cases only (exclude omitted).
- If unsure about duplicates, prefer keeping distinct parameter sweeps and only omit obvious exact duplicates.

Return ONLY valid JSON.
"""
# Filled with str.replace, not str.format: the schema above is full of braces.
PLANNER_SYSTEM = (
    PLANNER_SYSTEM.replace("{kinds}", ", ".join(FIGURE_KINDS))
    .replace("{per_case_kinds}", ", ".join(PER_CASE_KINDS))
)


def plan_paper_stage(model: str, planner_input: str) -> Dict[str, Any]:
    llm = create_langchain_llm(model=model, temperature=0.0)
    msgs = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=planner_input[:120_000]),
    ]
    resp = llm.invoke(msgs)
    raw = reply_text(resp)
    raw = strip_json_fences(raw if isinstance(raw, str) else str(raw))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    # sane defaults
    data.setdefault("omit_case_ids", [])
    data.setdefault("case_order_for_paper", [])
    data.setdefault("case_display_labels", {})
    data.setdefault("figure_jobs", [])
    data.setdefault("analysis_narrative_hints", "")
    data.setdefault("omit_rationale", "")
    data.setdefault("topic_paragraph_for_intro", "")
    data.setdefault("unified_viz_brief", "")
    return data


def fallback_plan(
    manifest_cases: List[Dict[str, Any]],
    fields_by_case: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """If LLM fails: include all success cases, with the u_contour and
    p_contour job each needs (figure_contract), on the case's own field names."""
    fields_by_case = fields_by_case or {}
    order: List[str] = []
    jobs: List[Dict[str, Any]] = []
    labels: Dict[str, str] = {}
    n = 0
    for c in manifest_cases:
        if not isinstance(c, dict):
            continue
        if str(c.get("status", "")).lower() != "success":
            continue
        cid = str(c.get("case_id", ""))
        if not cid:
            continue
        n += 1
        order.append(cid)
        labels[cid] = str(n)
        summary = fields_by_case.get(cid, {})
        u = summary.get("velocity_field") or "U"
        p = summary.get("pressure_field") or "p"
        jobs.append(
            {
                "case_id": cid,
                "kind": "u_contour",
                "field": u,
                "what_to_visualize": (
                    f"Paper-quality contour of the velocity field {u} (magnitude or streamwise component; "
                    "say which) at the latest time on the case's main plane. Use horizontal layout (wide "
                    "along streamwise) if the domain is a long thin channel. PyVista screenshot only."
                ),
                "priority": n,
            }
        )
        jobs.append(
            {
                "case_id": cid,
                "kind": "p_contour",
                "field": p,
                "what_to_visualize": (
                    f"Paper-quality contour of the pressure-like field {p} at the latest time on the same "
                    "plane and framing as the velocity contour. PyVista screenshot only."
                ),
                "priority": n,
            }
        )
    plan = {
        "omit_case_ids": [],
        "omit_rationale": "fallback: no LLM plan",
        "case_order_for_paper": order,
        "case_display_labels": labels,
        "figure_jobs": jobs,
        "unified_viz_brief": (
            "One batch script: for each case in order, save the velocity and pressure contours listed under "
            "REQUIRED OUTPUT FILES (horizontal aspect for long channels); use consistent colormap limits "
            "across comparable cases. PyVista screenshots only."
        ),
        "analysis_narrative_hints": "",
        "topic_paragraph_for_intro": "",
    }
    return normalize_figure_jobs(plan, manifest_cases, fields_by_case)

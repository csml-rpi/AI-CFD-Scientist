"""LLM plan: omit duplicate experiments, label cases for prose, plan paper figures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import strip_json_fences


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
- **Order:** case_order_for_paper lists included cases only (exclude omitted).
- If unsure about duplicates, prefer keeping distinct parameter sweeps and only omit obvious exact duplicates.

Return ONLY valid JSON.
"""


def plan_paper_stage(model: str, planner_input: str) -> Dict[str, Any]:
    llm = create_langchain_llm(model=model, temperature=0.0)
    msgs = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=planner_input[:120_000]),
    ]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp))
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


def fallback_plan(manifest_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """If LLM fails: include all success cases, one generic figure job each."""
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
        jobs.append(
            {
                "case_id": cid,
                "what_to_visualize": (
                    "Paper-quality figure: load latest time; streamwise velocity Ux contour on the 2D plane "
                    "OR wall-normal profile of Ux at mid-length. Use horizontal layout (wide along streamwise) "
                    "if the domain is a long thin channel. PyVista screenshot only. No nuEff unless present."
                ),
                "priority": n,
            }
        )
    return {
        "omit_case_ids": [],
        "omit_rationale": "fallback: no LLM plan",
        "case_order_for_paper": order,
        "case_display_labels": labels,
        "figure_jobs": jobs,
        "unified_viz_brief": (
            "One batch script: for each case in order, save Ux contour (horizontal aspect for long channel) "
            "and optional mid-channel Ux profile; use consistent viridis limits across comparable cases; "
            "optional one multi-panel overview figure. PyVista screenshots only. Descriptive filenames with case id."
        ),
        "analysis_narrative_hints": "",
        "topic_paragraph_for_intro": "",
    }

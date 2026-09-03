#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from timeline_logger import append_timeline_event, resolve_timeline_path

from langchain_core.messages import HumanMessage, SystemMessage


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def literature_file_to_ideation_items(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map lit.json entries into the shape expected by ideation.build_literature_context."""
    out: List[Dict[str, Any]] = []
    for p in records:
        if not isinstance(p, dict):
            continue
        authors = p.get("authors") or []
        if isinstance(authors, list):
            auth_s = ", ".join(str(a) for a in authors[:8])
        else:
            auth_s = str(authors)
        snippet = (p.get("abstract") or "")[:800]
        out.append(
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue") or "preprint / unknown",
                "source": "semantic_scholar_lit_json",
                "url": p.get("url", "") or "",
                "snippet": snippet,
                "doi": p.get("doi"),
                "citationCount": p.get("citationCount"),
                "authors": auth_s,
            }
        )
    return out


def run_literature_aware_ideation(
    research_topic: str,
    lit_items: List[Dict[str, Any]],
    settings: Any,
    *,
    starter_ctx: str = "",
    code_mod_ctx: str = "",
    mesh_gate_ctx: str = "",
) -> Dict[str, Any]:
    """Run the IdeationAgent with the three context blocks routed correctly.

    Per mechanism.md §4 and §6: starter / code-mod / mesh-gate contexts are
    composed, never concatenated into a single anonymous "context" blob. The
    "custom model already implemented" preamble fires only when there is an
    actual code-mod context AND the routing mode authorizes it (code_mod or
    oed_post_impl). Mesh-gate and starter contexts are rendered as plain
    informational user-prompt blocks and never imply a custom model.
    """
    from cfd_langgraph.ideation import build_literature_context, load_prompts, _normalize_to_experiments_schema
    from cfd_langgraph.llm.factory import create_langchain_llm
    from cfd_langgraph.utils import extract_json_object, strip_json_fences

    prompts = load_prompts(settings.prompts_path)
    ideation_prompts = prompts["IdeationAgent"]
    literature_context = build_literature_context(lit_items)
    system_prompt = ideation_prompts["initial_idea_prompt"]

    _mode = (getattr(settings, "_mode", None) or "").strip()
    _code_mod_authorized = _mode in {"code_mod", "oed_post_impl"}

    if code_mod_ctx and _code_mod_authorized:
        system_prompt += (
            "\n\nIMPORTANT — A custom OpenFOAM model has already been implemented and compiled "
            "for this study.  Your experiments MUST use this custom model (not built-in "
            "alternatives like Carreau-Yasuda or power-law unless explicitly comparing against "
            "them as a secondary objective).  Design experiments that exercise and validate the "
            "implemented model.  Below is the full implementation context:\n\n"
            f"{code_mod_ctx}\n"
        )

    # Mode-specific constraint. Pure-sweep studies must NOT invent custom models
    # or invoke any code-modification path. Only physical parameter values may
    # vary across experiments.
    if _mode == "pure_sweep":
        system_prompt += (
            "\n\nMODE: pure_sweep — STRICT RULES:\n"
            "- This is a PARAMETER SWEEP only. Vary ONE physical parameter (e.g. Reynolds "
            "number, viscosity, geometry size) across experiments to characterise a trend or "
            "fit a correlation.\n"
            "- DO NOT propose any custom model, custom viscosity model, custom turbulence "
            "model, custom transport model, or source-term modification. Use ONLY built-in "
            "OpenFOAM models (e.g. `Newtonian` viscosity, the standard solvers).\n"
            "- DO NOT include any field whose value names a custom or compiled model "
            "(e.g. `transport_model: compiled_custom_*`). Restrict `parameters` to "
            "physical/numerical values only.\n"
            "- DO NOT frame this as 'baseline vs. improved' — there is no improved model. "
            "Each experiment is just a different parameter setting.\n"
        )

    user_prompt = ideation_prompts.get(
        "literature_aware_user_prompt",
        "Research topic: {research_topic}\n\nPrior studies:\n{literature_context}\nMax experiments: {max_experiments}",
    ).format(
        research_topic=research_topic,
        literature_context=literature_context,
        max_experiments=settings.ideation_max_experiments,
    )

    # Append starter / mesh-gate context as separate informational blocks on
    # the user prompt. Neither implies a code-mod.
    if starter_ctx:
        user_prompt += (
            "\n\n--- STARTER CONTEXT (authoritative reference values from the "
            "starter folder; do not invent additional model fields) ---\n"
            f"{starter_ctx}\n"
        )
    if mesh_gate_ctx:
        user_prompt += (
            "\n\n--- MESH-GATE CONTEXT (selected mesh from the mesh-independence "
            "study; informational only, NOT a code-mod signal) ---\n"
            f"{mesh_gate_ctx}\n"
        )

    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        idea_json = json.loads(extract_json_object(content))
    except Exception as exc:
        raise ValueError(f"Ideation JSON parse failed: {exc}") from exc
    if not isinstance(idea_json, dict):
        raise ValueError("Ideation output is not a JSON object")
    return _normalize_to_experiments_schema(idea_json, settings.ideation_max_experiments)


def main() -> int:
    root = bootstrap_paths()
    parser = argparse.ArgumentParser(description="Generate experiment-design hypotheses from literature-aware ideation.")
    parser.add_argument("--literature", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--timeline", default="", type=str)
    parser.add_argument("--code-mod-context", default="", type=str,
                        help="Path to a text file containing code-mod context (model source, dictionaries)")
    parser.add_argument("--mesh-gate-resume", default="", type=str,
                        help="Path to mesh_gate_resume.json — injects selected mesh info into ideation context")
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json — LLM-extracted flow params, formula, reference data")
    parser.add_argument("--mode", default="", type=str,
                        help="Study mode (pure_sweep|standard|code_mod|mesh_focus). When pure_sweep, the "
                             "ideation prompt is constrained to plain parameter sweeps with no custom models.")
    args = parser.parse_args()
    timeline_path = resolve_timeline_path(args.timeline)

    lit_path = Path(args.literature)
    if not lit_path.exists():
        print(f"Literature file not found: {lit_path}", file=sys.stderr)
        return 1
    raw_lit = json.loads(lit_path.read_text(encoding="utf-8"))
    if not isinstance(raw_lit, list):
        print("Literature JSON must be a list of paper records", file=sys.stderr)
        return 1

    from cfd_langgraph.config import get_settings

    settings = get_settings()
    # Stash the mode on settings for run_ideation_pipeline to read
    try:
        setattr(settings, "_mode", args.mode or "")
    except Exception:
        pass

    code_mod_ctx = ""
    if args.code_mod_context:
        cm_path = Path(args.code_mod_context)
        if cm_path.is_file():
            code_mod_ctx = cm_path.read_text(encoding="utf-8", errors="ignore")
            print(f"[HYP] Loaded code-mod context from {cm_path} ({len(code_mod_ctx)} chars)")
        else:
            print(f"[HYP] WARNING: code-mod-context path not found: {cm_path}")

    mesh_gate_ctx = ""
    if args.mesh_gate_resume:
        mg_path = Path(args.mesh_gate_resume)
        if mg_path.is_file():
            try:
                mg = json.loads(mg_path.read_text(encoding="utf-8"))
                lines = ["MESH-GATE STUDY RESULTS — selected mesh per physics group:"]
                for gid, ginfo in (mg.get("groups") or {}).items():
                    lines.append(f"\n--- Group: {gid} ---")
                    lines.append(ginfo.get("summary", ""))
                mesh_gate_ctx = "\n".join(lines)
                print(f"[HYP] Loaded mesh-gate resume ({len(mesh_gate_ctx)} chars, {len(mg.get('groups', {}))} group(s))")
            except Exception as e:
                print(f"[HYP] WARNING: failed to read mesh-gate resume: {e}")
        else:
            print(f"[HYP] WARNING: mesh-gate-resume path not found: {mg_path}")

    lit_items = literature_file_to_ideation_items(raw_lit[:20])
    print(f"[HYP] Loaded literature items for ideation: {len(lit_items)}")

    # Load starter understanding (flow params, formula, reference data)
    starter_ctx = ""
    if args.starter_understanding:
        su_path = Path(args.starter_understanding)
        if su_path.is_file():
            try:
                su = json.loads(su_path.read_text(encoding="utf-8"))
                parts = ["STARTER FOLDER UNDERSTANDING — use these authoritative values in all hypotheses:"]
                fp = su.get("flow_parameters") or {}
                if fp:
                    parts.append(f"Flow parameters: {json.dumps(fp, ensure_ascii=False)}")
                geo = su.get("geometry") or su.get("flow_parameters", {}).get("geometry") or ""
                if geo and isinstance(geo, str):
                    parts.append(f"Geometry: {geo}")
                formula = su.get("formula_or_model_spec") or ""
                if formula:
                    parts.append(f"Model/formula to implement: {formula[:1200]}")
                ref = su.get("reference_data") or {}
                if ref:
                    desc = ref.get("description") or ""
                    quantities = ref.get("quantities") or ""
                    usage = ref.get("usage_guidance") or ""
                    if desc:
                        parts.append(f"Reference data available: {desc}")
                    if quantities:
                        parts.append(f"Reference quantities: {quantities}")
                    if usage:
                        parts.append(f"How to use reference data: {usage}")
                starter_ctx = "\n".join(parts)
                print(f"[HYP] Loaded starter_understanding ({len(starter_ctx)} chars, "
                      f"Re={fp.get('Re')}, nu={fp.get('nu')}, Ub={fp.get('Ub')})")
            except Exception as e:
                print(f"[HYP] WARNING: failed to read starter-understanding: {e}")
        else:
            print(f"[HYP] WARNING: starter-understanding path not found: {su_path}")

    # Per mechanism.md §6: pass the three contexts separately. Never merge into
    # a single anonymous blob — that's what caused the "compiled_custom_*"
    # leak in pure_sweep runs.
    try:
        idea_json = run_literature_aware_ideation(
            args.topic,
            lit_items,
            settings,
            starter_ctx=starter_ctx,
            code_mod_ctx=code_mod_ctx,
            mesh_gate_ctx=mesh_gate_ctx,
        )
    except Exception as exc:
        print(f"Ideation failed ({exc}); using fallback study skeleton.", file=sys.stderr)
        idea_json = {
            "study_id": "fallback_study",
            "description": f"Exploratory CFD study for: {args.topic}",
            "solver": "simpleFoam",
            "target_CFL": 0.5,
            "post": {"objective": "compare baseline and perturbed setups"},
            "experiments": [
                {
                    "experiment_id": "exp_001",
                    "name": "baseline",
                    "topology": "2d",
                    "dimensions": [1.0, 0.5, 0.01],
                    "parameters": {"inlet_velocity": 1.0},
                    "controls": {"end_time": 0.5, "write_interval": 0.05},
                    "notes": "Baseline channel-like setup",
                },
            ],
        }

    experiments = idea_json.get("experiments") if isinstance(idea_json.get("experiments"), list) else []
    if not experiments:
        print("No experiments in ideation output.", file=sys.stderr)
        return 1
    print(f"[HYP] Ideation generated experiments: {len(experiments)}")
    for idx, exp in enumerate(experiments, 1):
        if not isinstance(exp, dict):
            continue
        exp_id = exp.get("experiment_id") or exp.get("name") or f"exp_{idx:03d}"
        exp_name = str(exp.get("name", exp_id))
        desc = str(exp.get("notes", "") or exp.get("description", "") or "")
        print(f"[HYP] Idea {idx}/{len(experiments)} id={exp_id} name={exp_name}")
        if desc:
            print(f"[HYP]   description: {desc[:220]}")
        params = exp.get("parameters", {}) if isinstance(exp.get("parameters"), dict) else {}
        if params:
            print(f"[HYP]   parameters: {json.dumps(params, ensure_ascii=False)[:320]}")

    output: List[Dict[str, Any]] = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        sim_id = exp.get("experiment_id") or exp.get("name") or f"exp_{len(output)+1:03d}"
        description = str(exp.get("notes", "") or exp.get("description", "") or "")
        output.append(
            {
                "hypothesis_id": sim_id,
                "experiment_id": sim_id,
                "study_id": idea_json.get("study_id", ""),
                "description": description or f"Experiment {sim_id}",
                "hypothesis_text": description or f"Test experiment design for {sim_id}",
                "parameter_value": exp.get("parameters", {}) if isinstance(exp.get("parameters"), dict) else {},
                "valid": True,
                "idea_experiment": exp,
            }
        )
    print(f"[HYP] Final hypothesis records written: {len(output)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {
            "stage": "hypothesis",
            "topic": args.topic,
            "hypothesis_count": len(output),
            "experiments": [
                {
                    "hypothesis_id": o.get("hypothesis_id"),
                    "experiment_id": o.get("experiment_id"),
                    "valid": bool(o.get("valid", False)),
                    "description": o.get("description", ""),
                }
                for o in output
            ],
            "output_path": str(out_path),
        },
    )
    print(f"Hypotheses generated: {len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

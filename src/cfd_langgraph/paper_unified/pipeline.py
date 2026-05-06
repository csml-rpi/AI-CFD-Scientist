"""Unified loop: batch PyVista script → per-image QA → figure analysis → paper → review (outer loop can regen script)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cfd_langgraph.agents.writer_agent import WriterAgent
from cfd_langgraph.agents.paper_reviewer_agent import PaperReviewerAgent
from cfd_langgraph.config import get_settings
from cfd_langgraph.paper_unified.batch_paper_viz import (
    analyze_figures_for_paper,
    build_unified_brief_from_plan,
    reviewer_requests_script_refresh,
    run_batch_paper_viz_loop,
    write_cases_config,
)
from cfd_langgraph.paper_unified.context import build_planner_payload, load_json
from cfd_langgraph.paper_unified.planner import fallback_plan, plan_paper_stage
from cfd_langgraph.paper_utils import compile_tex_to_pdf
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.refchecker_integration import run_refchecker_on_tex
from cfd_langgraph.utils import strip_latex_fences


def _case_path_map(manifest: Dict[str, Any]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for c in manifest.get("cases") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id", ""))
        p = c.get("case_path")
        if cid and isinstance(p, str) and p.strip():
            out[cid] = Path(p).expanduser().resolve()
    return out


def _success_case_ids(manifest: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for c in manifest.get("cases") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("status", "")).lower() == "success":
            cid = str(c.get("case_id", ""))
            if cid:
                ids.append(cid)
    return ids


def run_unified_paper_pipeline(
    *,
    repo_root: Path,
    run_dir: Path,
    topic: str,
    paper_dir: Path,
    analysis_path: Path,
    manifest_path: Path,
    requirements_path: Path,
    lit_path: Path,
    review_path: Path,
    mesh_independence_path: Optional[Path],
    paper_template: str,
    model: str,
    starter_understanding_path: Optional[Path] = None,
    max_review_loops: int = 10,
    score_threshold: float = 0.82,
    max_viz_inner_attempts: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Outer loop (max_review_loops): optionally regenerate batch viz script + figures, write/revise paper, compile+review.
    Inner loop (max_viz_inner_attempts): one script, per-image VLM, targeted revision.
    """
    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    paper_dir = paper_dir.resolve()
    manifest = load_json(manifest_path, {})
    requirements = load_json(requirements_path, [])
    if not isinstance(requirements, list):
        requirements = []
    analysis = load_json(analysis_path, {})
    mesh_bundle: Dict[str, Any] = {}
    if mesh_independence_path and mesh_independence_path.is_file():
        mb = load_json(mesh_independence_path, {})
        if isinstance(mb, dict):
            mesh_bundle = mb
    literature = load_json(lit_path, [])
    if not isinstance(literature, list):
        literature = [literature] if literature else []

    # Load starter_understanding for paper planner context (flow params, formula, reference data)
    starter_note = ""
    su_path = starter_understanding_path or (run_dir / "starter_understanding.json")
    if su_path and su_path.is_file():
        try:
            su = load_json(su_path, {})
            if isinstance(su, dict):
                parts: list[str] = []
                fp = su.get("flow_parameters") or {}
                if fp:
                    parts.append(f"Flow parameters: {json.dumps(fp, ensure_ascii=False)}")
                formula = su.get("formula_or_model_spec") or ""
                if formula:
                    parts.append(f"Model/formula: {formula[:800]}")
                ref = su.get("reference_data") or {}
                if ref.get("description"):
                    parts.append(f"Reference data: {ref['description']}")
                if ref.get("quantities"):
                    parts.append(f"Reference quantities: {ref['quantities']}")
                if ref.get("usage_guidance"):
                    parts.append(f"Usage guidance: {ref['usage_guidance']}")
                if ref.get("data_excerpt"):
                    parts.append(f"Reference excerpt:\n{str(ref['data_excerpt'])[:2000]}")
                starter_note = "\n".join(parts)
                if verbose:
                    print(f"[PAPER] Loaded starter_understanding ({len(starter_note)} chars)")
        except Exception as _e:
            if verbose:
                print(f"[PAPER] WARNING: could not load starter_understanding: {_e}")

    planner_text = build_planner_payload(
        run_dir=run_dir,
        topic=topic,
        manifest=manifest if isinstance(manifest, dict) else {},
        requirements=requirements,
        analysis=analysis if isinstance(analysis, dict) else {},
        mesh_bundle=mesh_bundle,
        code_mod_note="",
        starter_understanding_note=starter_note,
    )
    plan = plan_paper_stage(model, planner_text)
    success_manifest_cases = [
        c
        for c in (manifest.get("cases") or [])
        if isinstance(c, dict) and str(c.get("status", "")).lower() == "success"
    ]
    if not plan.get("figure_jobs"):
        plan = fallback_plan(success_manifest_cases)
    plan_path = run_dir / "paper_unified_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    paths_by_id = _case_path_map(manifest if isinstance(manifest, dict) else {})
    omit = {str(x) for x in (plan.get("omit_case_ids") or [])}
    included_order = [c for c in (plan.get("case_order_for_paper") or []) if c not in omit]
    if not included_order:
        included_order = [c for c in _success_case_ids(manifest if isinstance(manifest, dict) else {}) if c not in omit]

    unified_brief = build_unified_brief_from_plan(plan)
    figure_jobs_summary = json.dumps(plan.get("figure_jobs") or [], indent=2)[:12_000]

    paper_figs_root = run_dir / "paper_figs"
    cases_for_cfg = [
        {"id": cid, "path": str(paths_by_id[cid])}
        for cid in included_order
        if cid in paths_by_id
    ]
    config_path = write_cases_config(paper_figs_root, cases_for_cfg)

    display_labels = plan.get("case_display_labels") if isinstance(plan.get("case_display_labels"), dict) else {}

    settings = get_settings()
    prompts = PromptLoader(settings.prompts_path)
    writer = WriterAgent(model=model, prompt_loader=prompts)
    reviewer = PaperReviewerAgent(model=model, prompt_loader=prompts)

    paper_dir.mkdir(parents=True, exist_ok=True)
    tex_path = paper_dir / "paper_draft.tex"
    work_dir = repo_root
    review_info: Dict[str, Any] = {
        "tries": 0,
        "reviews": [],
        "plan_path": str(plan_path),
        "viz_meta": [],
    }

    script_text = ""
    fig_paths: List[str] = []
    image_analysis = ""
    viz_bundle: List[Dict[str, Any]] = []
    section_context = ""
    paper_tex = ""
    extra_script_feedback = ""

    recs_prev: List[str] = []
    needs_viz_prev = False
    review_regenerate_batch: Optional[bool] = None
    last_compile_ok = False
    last_pdf: Optional[str] = None
    last_err = ""
    last_rev: Dict[str, Any] = {}

    for outer in range(max(1, max_review_loops)):
        review_info["tries"] = outer + 1
        regen_viz = outer == 0 or reviewer_requests_script_refresh(
            recs_prev,
            needs_viz_prev,
            regenerate_batch_figures=review_regenerate_batch,
        )

        if regen_viz:
            if verbose:
                print(f"[paper_unified] outer {outer + 1}: batch PyVista script (inner QA max {max_viz_inner_attempts})")
            script_text, png_paths, vmeta = run_batch_paper_viz_loop(
                model=model,
                repo_root=repo_root,
                paper_figs_dir=paper_figs_root,
                topic=topic,
                unified_brief=unified_brief,
                figure_jobs_summary=figure_jobs_summary,
                config_path=config_path,
                max_inner_attempts=max_viz_inner_attempts,
                verbose=verbose,
                previous_script=script_text,
                extra_feedback=extra_script_feedback,
            )
            review_info["viz_meta"].append(vmeta)
            extra_script_feedback = ""

            fig_paths = [str(p.resolve()) for p in png_paths]
            if mesh_bundle:
                for p in mesh_bundle.get("mesh_figure_paths") or []:
                    if isinstance(p, str) and Path(p).is_file():
                        fig_paths.append(str(Path(p).resolve()))

            image_analysis = analyze_figures_for_paper(
                model=model,
                topic=topic,
                analysis=analysis if isinstance(analysis, dict) else {},
                image_paths=png_paths,
                verbose=verbose,
            )

            viz_bundle = [
                {
                    "simulation_id": "paper_batch",
                    "case_name": "all_experiments",
                    "visualization": {"images": fig_paths},
                }
            ]

            unified_context: Dict[str, Any] = {
                "topic": topic,
                "template": paper_template,
                "analysis": analysis,
                "paper_image_analysis": image_analysis,
                "unified_paper_rules": {
                    "abstract_no_case_numbers": True,
                    "abstract_no_section_refs": True,
                    "use_display_case_labels": True,
                    "case_display_labels": display_labels,
                    "included_case_ids": included_order,
                    "omitted_case_ids": sorted(omit),
                    "naming": "In narrative use 'case 1', 'case 2' (with space), never case_001. Introduce each case in prose before first mention.",
                    "duplicates": "Do not tell readers about omitted or duplicate experiments; omit them silently.",
                    "figures": "Figures come from one batch script; use provided paths. Match captions to paper_image_analysis.",
                },
                "paper_plan": {
                    "analysis_narrative_hints": plan.get("analysis_narrative_hints", ""),
                    "topic_paragraph_for_intro": plan.get("topic_paragraph_for_intro", ""),
                    "omit_rationale_internal": plan.get("omit_rationale", ""),
                },
                "figures": fig_paths,
                "manifest_excerpt": {
                    "cases": [
                        c
                        for c in (manifest.get("cases") or [])
                        if isinstance(c, dict) and c.get("case_id") in set(included_order)
                    ]
                },
                "mesh_independence": mesh_bundle,
            }
            section_context = json.dumps(unified_context, indent=2)

            if outer == 0:
                paper_tex = writer.write_paper_with_literature(
                    topic=topic,
                    section_context=section_context,
                    ideation_literature_bundle=literature,
                    visualization_bundle=viz_bundle,
                    max_literature_papers=40,
                    verbose=verbose,
                )
                paper_tex = strip_latex_fences(paper_tex)
            else:
                recs = list(recs_prev) or ["Refresh manuscript for new batch figures and paper_image_analysis."]
                paper_tex = writer.revise_paper(
                    paper_tex,
                    recs,
                    visualization_bundle=viz_bundle,
                    work_dir=work_dir,
                    analysis_context=section_context,
                    verbose=verbose,
                )
                paper_tex = strip_latex_fences(paper_tex)
        else:
            if verbose:
                print(f"[paper_unified] outer {outer + 1}: text-only revision (reuse figures)")
            recs = list(recs_prev) or [""]
            if not recs or recs == [""]:
                recs = ["Address reviewer feedback."]
            paper_tex = writer.revise_paper(
                paper_tex,
                recs,
                visualization_bundle=viz_bundle,
                work_dir=work_dir,
                analysis_context=section_context,
                verbose=verbose,
            )
            paper_tex = strip_latex_fences(paper_tex)

        if verbose:
            print(f"[paper_unified] outer {outer + 1}: compile + review")
        tex_path.write_text(paper_tex, encoding="utf-8")
        ref_summary = ""
        try:
            _, ref_summary = run_refchecker_on_tex(
                tex_path=tex_path,
                out_dir=paper_dir / "refchecker_reports",
                attempt=outer + 1,
                verbose=verbose,
            )
        except Exception:
            ref_summary = ""

        def _fig_paths_for_review() -> List[str]:
            rels: List[str] = []
            for p in fig_paths:
                try:
                    rels.append(str(Path(p).relative_to(work_dir)))
                except ValueError:
                    rels.append(p)
            return rels

        compile_ok, pdf_path, compile_err = compile_tex_to_pdf(tex_path, work_dir=work_dir)
        last_compile_ok = compile_ok
        last_pdf = pdf_path
        last_err = compile_err or ""

        review = reviewer.review(
            tex_content=paper_tex,
            compile_ok=compile_ok,
            compile_error=last_err,
            reference_report=ref_summary,
            valid_figure_paths=_fig_paths_for_review(),
        )
        review_info["reviews"].append(review)
        last_rev = review
        score = float(review.get("score", 0.0))
        passed = (
            bool(review.get("pass", False))
            and compile_ok
            and score >= float(score_threshold)
        )
        recs_prev = list(review.get("recommendations") or [])
        needs_viz_prev = bool(review.get("needs_additional_visualization", False))
        specs = list(review.get("additional_viz_specs") or [])
        if "regenerate_batch_figures" in review:
            review_regenerate_batch = bool(review["regenerate_batch_figures"])

        if passed:
            if verbose:
                print(
                    f"[paper_unified] accepted: pass=True score={score} "
                    f"(threshold={score_threshold})"
                )
            break

        _rb = review["regenerate_batch_figures"] if "regenerate_batch_figures" in review else None
        if reviewer_requests_script_refresh(
            recs_prev,
            needs_viz_prev,
            regenerate_batch_figures=_rb,
        ):
            fb_parts: List[str] = []
            if specs:
                fb_parts.append("Additional visualization specs:\n" + "\n".join(specs[:25]))
            if recs_prev:
                fb_parts.append(
                    "Paper reviewer recommendations (figures / layout / narrative):\n"
                    + "\n".join(str(r) for r in recs_prev[:40])
                )
            extra_script_feedback = "\n\n".join(fb_parts) if fb_parts else ""
        else:
            extra_script_feedback = ""

        if outer >= max_review_loops - 1 and verbose:
            print("[paper_unified] max outer loops reached")

    if not last_compile_ok:
        for cr in range(10):
            if verbose:
                print(f"[paper_unified] compile recovery {cr + 1}/10")
            paper_tex = writer.revise_paper_for_compilation_only(
                paper_tex,
                last_err,
                visualization_bundle=viz_bundle,
                work_dir=work_dir,
                verbose=verbose,
            )
            paper_tex = strip_latex_fences(paper_tex)
            tex_path.write_text(paper_tex, encoding="utf-8")
            compile_ok, pdf_path, compile_err = compile_tex_to_pdf(tex_path, work_dir=work_dir)
            last_compile_ok = compile_ok
            last_pdf = pdf_path
            last_err = compile_err or ""
            if compile_ok:
                break

    main_tex = paper_dir / "main.tex"
    main_tex.write_text(paper_tex, encoding="utf-8")
    (paper_dir / "sections").mkdir(parents=True, exist_ok=True)
    (paper_dir / "sections" / "body.tex").write_text(paper_tex, encoding="utf-8")
    (paper_dir / "references.bib").write_text("", encoding="utf-8")
    if last_pdf and Path(last_pdf).exists():
        import shutil

        shutil.copy2(last_pdf, paper_dir / "main.pdf")

    last_rev = review_info["reviews"][-1] if review_info["reviews"] else {}
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(
            {
                "overall_score": float(last_rev.get("score", 0.0)),
                "major_issues": last_rev.get("recommendations", []),
                "raw": {"review_info": review_info, "last_review": last_rev},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "ok": bool(last_compile_ok),
        "paper_dir": str(paper_dir),
        "review_path": str(review_path),
        "plan_path": str(plan_path),
        "last_score": float(last_rev.get("score", 0.0)),
        "compile_ok": last_compile_ok,
    }

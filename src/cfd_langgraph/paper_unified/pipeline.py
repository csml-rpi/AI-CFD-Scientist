"""Unified loop: batch PyVista script → per-image QA → figure analysis → paper → review (outer loop can regen script)."""

from __future__ import annotations

import json
import os
import time
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
from cfd_langgraph.paper_unified.figure_contract import (
    case_field_summary,
    figure_jobs_brief,
    includegraphics_targets,
    missing_embeds,
    normalize_figure_jobs,
    split_rendered,
)
from cfd_langgraph.paper_unified.planner import fallback_plan, plan_paper_stage
from cfd_langgraph.paper_utils import compile_tex_to_pdf, expand_tex_inputs
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.refchecker_integration import run_refchecker_on_tex
from cfd_langgraph.utils import strip_latex_fences
from cfd_langgraph.case_records import case_records


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
    start_from_current: bool = False,
    results_files: Optional[List[Path]] = None,
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
            # The gate's numbers are for comparing its mesh levels: printed
            # beside the study's results, a hyStrath gate's half-domain drag
            # (20.2 N/m) read as contradicting the reported full-cylinder
            # 39.6 N/m (shelf_demo_hystrath_20261022_1030).
            mesh_bundle["values_note"] = (
                "The quantities here are the mesh gate's own measurements, made to compare its mesh levels "
                "with each other: convergence is judged on their relative change between levels. They follow "
                "the gate's extractor's own conventions (a half domain without the symmetry factor, its own "
                "reference values) and are not the study's results. Report the study's values from the "
                "cases' records and the analysis; use these only for the relative changes between levels.")
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

    paths_by_id = _case_path_map(manifest if isinstance(manifest, dict) else {})
    manifest_cases = [c for c in (manifest.get("cases") or []) if isinstance(c, dict)]
    # Which velocity/pressure fields each case actually wrote, for the
    # planner and for the figure jobs' `field` (see figure_contract).
    fields_by_case: Dict[str, Dict[str, Any]] = {}
    for cid, cpath in paths_by_id.items():
        try:
            fields_by_case[cid] = case_field_summary(cpath)
        except Exception as exc:  # noqa: BLE001 -- an unreadable case is planned without field names
            fields_by_case[cid] = {"error": str(exc)}

    planner_text = build_planner_payload(
        run_dir=run_dir,
        topic=topic,
        manifest=manifest if isinstance(manifest, dict) else {},
        requirements=requirements,
        analysis=analysis if isinstance(analysis, dict) else {},
        mesh_bundle=mesh_bundle,
        code_mod_note="",
        starter_understanding_note=starter_note,
        case_fields=fields_by_case,
    )
    plan = plan_paper_stage(model, planner_text)
    success_manifest_cases = [
        c for c in manifest_cases if str(c.get("status", "")).lower() == "success"
    ]
    if not plan.get("figure_jobs"):
        plan = fallback_plan(success_manifest_cases, fields_by_case)
    else:
        # The planner's jobs get their id/kind/field/out_png here, and any
        # successful case it left without a u_contour or p_contour gets one.
        plan = normalize_figure_jobs(plan, manifest_cases, fields_by_case)
    # Revising the paper as it stands: the required per-case contours, plus
    # the planned figures it already embeds. Every planned figure was forced
    # back into a compact paper the manager had written, which had been cut
    # down from a 60-figure draft (shelf_demo_hystrath_20261022_1030).
    current_tex = paper_dir / "main.tex"
    if start_from_current and current_tex.is_file():
        embedded = {Path(t).name for t in includegraphics_targets(expand_tex_inputs(current_tex))}
        kept_jobs = [j for j in (plan.get("figure_jobs") or []) if isinstance(j, dict)
                     and (j.get("kind") in ("u_contour", "p_contour")
                          or Path(str(j.get("out_png") or "")).name in embedded)]
        dropped = len(plan.get("figure_jobs") or []) - len(kept_jobs)
        if dropped:
            plan["figure_jobs"] = kept_jobs
            plan["figure_jobs_dropped_for_current_paper"] = dropped
        # The paper as it stands names each case by its own number. The
        # planner renumbered the included cases 1..n, so the figures said
        # "case 2" where the text said "case 3", and its brief still listed
        # the outputs dropped above (shelf_demo_hystrath_20261022_1030).
        planner_labels = plan.get("case_display_labels") if isinstance(plan.get("case_display_labels"), dict) else {}
        own_labels = {}
        for cid in plan.get("case_order_for_paper") or list(planner_labels):
            number = str(cid).rsplit("_", 1)[-1].lstrip("0")
            own_labels[str(cid)] = number if number.isdigit() else str(planner_labels.get(cid, cid))
        if planner_labels and own_labels != {str(k): str(v) for k, v in planner_labels.items()}:
            plan["case_display_labels"] = own_labels
            plan["unified_viz_brief"] = (
                str(plan.get("unified_viz_brief") or "")
                + "\n\nCASE NUMBERING: where this brief says 'case N' it means: "
                + "; ".join(f"case {planner_labels[c]} = {c}" for c in own_labels if c in planner_labels)
                + ". Label every figure with the case's own number, the numbering the paper uses: "
                + "; ".join(f"{c} is case {n}" for c, n in own_labels.items()) + ".")
        if dropped:
            plan["unified_viz_brief"] = (
                str(plan.get("unified_viz_brief") or "")
                + "\n\nTHIS REVISION renders only the files under REQUIRED OUTPUT FILES. Any other output "
                "named above is not wanted now: do not write it.")
    plan_path = run_dir / "paper_unified_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    figure_jobs: List[Dict[str, Any]] = [j for j in (plan.get("figure_jobs") or []) if isinstance(j, dict)]
    required_pngs = [(run_dir / str(j["out_png"])).resolve() for j in figure_jobs if j.get("out_png")]

    omit = {str(x) for x in (plan.get("omit_case_ids") or [])}
    included_order = [c for c in (plan.get("case_order_for_paper") or []) if c not in omit]
    if not included_order:
        included_order = [c for c in _success_case_ids(manifest if isinstance(manifest, dict) else {}) if c not in omit]

    unified_brief = build_unified_brief_from_plan(plan) + "\n\n" + figure_jobs_brief(figure_jobs, run_dir)
    figure_jobs_summary = json.dumps(figure_jobs, indent=2)[:12_000]

    paper_figs_root = run_dir / "paper_figs"
    # every case a figure job names must be loadable by the script
    cfg_ids = list(included_order) + [
        str(j["case_id"]) for j in figure_jobs
        if j.get("case_id") and str(j["case_id"]) not in included_order
    ]
    cases_for_cfg = [
        {"id": cid, "path": str(paths_by_id[cid])}
        for cid in dict.fromkeys(cfg_ids)
        if cid in paths_by_id
    ]
    config_path = write_cases_config(
        paper_figs_root,
        cases_for_cfg,
        figure_jobs=[
            {**{k: j.get(k) for k in ("id", "kind", "case_id", "field", "field_note", "what_to_visualize")},
             "out_png_abs": str((run_dir / str(j["out_png"])).resolve())}
            for j in figure_jobs if j.get("out_png")
        ],
    )
    rendered_pngs: List[Path] = []
    missing_pngs: List[str] = []
    cross = analysis.get("cross_experiment_processing") if isinstance(analysis, dict) else None
    analysis_pngs = [Path(str(p)).resolve() for p in ((cross or {}).get("images") or [])
                     if isinstance(cross, dict) and Path(str(p)).is_file()]
    records_by_case = {cid: case_records(paths_by_id[cid]) for cid in included_order if cid in paths_by_id}
    study_results: Dict[str, str] = {}
    named_pngs = [Path(f).resolve() for f in results_files or []
                  if Path(f).suffix.lower() == ".png" and Path(f).is_file()]
    for f in [f for f in results_files or [] if Path(f).suffix.lower() != ".png"]:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if Path(f).suffix.lower() == ".json":
            try:
                text = json.dumps(json.loads(text), separators=(",", ":"))
            except ValueError:
                pass
        study_results[str(Path(f).name)] = text if len(text) <= 20000 else text[:20000] + f"\n... ({len(text)} characters in all)"

    display_labels = plan.get("case_display_labels") if isinstance(plan.get("case_display_labels"), dict) else {}

    settings = get_settings()
    prompts = PromptLoader(settings.prompts_path)
    writer = WriterAgent(model=model, prompt_loader=prompts)
    reviewer = PaperReviewerAgent(model=model, prompt_loader=prompts)

    paper_dir.mkdir(parents=True, exist_ok=True)
    tex_path = paper_dir / "paper_draft.tex"
    start_tex = ""
    if start_from_current:
        current = paper_dir / "main.tex"
        start_tex = expand_tex_inputs(current) if current.is_file() else ""
        if not start_tex.strip():
            raise FileNotFoundError(f"start_from_current: {current} is missing or empty")
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
    best: Dict[str, Any] = {"score": -1.0, "rank": (-1, -1.0), "tex": "", "review": {}, "round": 0, "pdf": ""}

    def _rank(rev: Dict[str, Any]) -> tuple:
        # The checks the final audit holds a paper to come before the score:
        # kept on score alone, a 0.72 round whose figures failed the physics
        # check was chosen over a 0.66 round whose figures passed
        # (shelf_demo_hystrath_20261022_1030).
        gates_ok = rev.get("figure_physics_ok") is not False and not rev.get("missing_embeds")
        return (int(gates_ok), float(rev.get("score", 0.0)))
    last_pdf: Optional[str] = None
    last_err = ""
    last_rev: Dict[str, Any] = {}

    # The run is killed at the write_paper tool's time limit. Saved only at
    # the end, a killed run kept no paper and no review although five review
    # rounds had finished (shelf_demo_hystrath_20261011_1602): each reviewed
    # version is now saved as it is reviewed, and no round starts that would
    # not finish within CFD_SCIENTIST_PAPER_BUDGET_S.
    try:
        budget_s = float(os.environ.get("CFD_SCIENTIST_PAPER_BUDGET_S") or 0)
    except ValueError:
        budget_s = 0.0
    deadline = time.monotonic() + budget_s if budget_s > 0 else None
    last_round_s = 0.0

    def _save_outputs() -> None:
        main_tex = paper_dir / "main.tex"
        main_tex.write_text(paper_tex, encoding="utf-8")
        (paper_dir / "sections").mkdir(parents=True, exist_ok=True)
        (paper_dir / "sections" / "body.tex").write_text(paper_tex, encoding="utf-8")
        (paper_dir / "references.bib").write_text("", encoding="utf-8")
        if last_pdf and Path(last_pdf).exists():
            import shutil

            shutil.copy2(last_pdf, paper_dir / "main.pdf")
        rev = review_info["reviews"][-1] if review_info["reviews"] else {}
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            json.dumps(
                {
                    "overall_score": float(rev.get("score", 0.0)),
                    "major_issues": rev.get("recommendations", []),
                    # top level, where the audit's H13 reads it
                    "figure_physics_ok": rev.get("figure_physics_ok"),
                    "figure_physics_reason": rev.get("figure_physics_reason", ""),
                    "figures": {
                        "planned": len(figure_jobs),
                        "rendered": [p.name for p in rendered_pngs],
                        "missing_out_png": list(missing_pngs),
                        "missing_embeds": list(rev.get("missing_embeds") or []),
                    },
                    "raw": {"review_info": review_info, "last_review": rev},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _embed_required(tex: str, round_no: int) -> str:
        """One targeted revision when the draft left planned figures out. The
        writer is told the files, but a draft can still drop some; the audit
        (H15) needs every rendered out_png embedded."""
        gone = missing_embeds(tex, rendered_pngs)
        if not gone:
            return tex
        review_info.setdefault("embed_fixes", []).append({"round": round_no, "missing": gone})
        if verbose:
            print(f"[paper_unified] {len(gone)} planned figure(s) not embedded; asking the writer to add them")
        fixed = writer.revise_paper(
            tex,
            [
                f"Embed the figure {p} with its own \\includegraphics{{{p}}} (exact path, with .png) "
                f"in a figure environment with a caption that says what it shows, and reference it in the text."
                for p in gone
            ],
            visualization_bundle=viz_bundle,
            work_dir=work_dir,
            analysis_context=section_context,
            verbose=verbose,
            required_figures=[str(p) for p in rendered_pngs],
        )
        return strip_latex_fences(fixed)

    for outer in range(max(1, max_review_loops)):
        if deadline is not None and outer > 0 and time.monotonic() + last_round_s > deadline:
            review_info["stopped_for_time"] = True
            if verbose:
                print(f"[paper_unified] stopping before round {outer + 1}: not enough time left "
                      f"(the last round took {last_round_s / 60:.0f} min)")
            break
        round_started = time.monotonic()
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
                deadline=deadline,
                required_pngs=required_pngs,
            )
            review_info["viz_meta"].append(vmeta)
            extra_script_feedback = ""

            # Which planned figures exist now. A missing one stays in the plan,
            # marked, so the audit reports it rather than the plan hiding it.
            rendered_pngs, missing_pngs = split_rendered(figure_jobs, run_dir)
            # The cross-case analysis's figures, and figures the manager named,
            # are offered to the writer, not required: never given, an NN paper
            # left out the training history (nn_in_solver_demo_20261022_1129);
            # all required, a hyStrath paper had to embed the analysis's 40
            # (shelf_demo_hystrath_20261022_1030).
            offered_pngs = [p for p in analysis_pngs + named_pngs if p not in rendered_pngs]
            for j in figure_jobs:
                j["rendered"] = (run_dir / str(j.get("out_png", ""))).is_file()
            plan["missing_out_png"] = list(missing_pngs)
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            if missing_pngs and verbose:
                print(f"[paper_unified] {len(missing_pngs)} planned figure(s) not written: {missing_pngs[:6]}")

            # The planned figures (and the analysis's), not every PNG the
            # script left: given all 40 a hyStrath figure script wrote, the
            # writer embedded them, the manuscript grew a 60-figure appendix,
            # and the manager wrote the paper by hand instead
            # (shelf_demo_hystrath_20261022_1030).
            fig_paths = [str(p.resolve()) for p in rendered_pngs + offered_pngs]
            if mesh_bundle:
                for p in mesh_bundle.get("mesh_figure_paths") or []:
                    if isinstance(p, str) and Path(p).is_file():
                        fig_paths.append(str(Path(p).resolve()))

            image_analysis = analyze_figures_for_paper(
                model=model,
                topic=topic,
                analysis=analysis if isinstance(analysis, dict) else {},
                image_paths=rendered_pngs + offered_pngs,
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
                    "required_figures": [str(p) for p in rendered_pngs],
                    "required_figures_rule": "Embed every required_figures file with its own \\includegraphics (exact path).",
                },
                "paper_plan": {
                    "analysis_narrative_hints": plan.get("analysis_narrative_hints", ""),
                    "topic_paragraph_for_intro": plan.get("topic_paragraph_for_intro", ""),
                    "omit_rationale_internal": plan.get("omit_rationale", ""),
                },
                "figures": fig_paths,
                # Listed among the analysis's 40, the manager's matched plots
                # were left out of every draft (shelf_demo_hystrath_20261022_1030).
                **({"study_result_figures": [str(p) for p in named_pngs],
                    "study_result_figures_rule": (
                        "study_result_figures are the figures the study itself made from its results "
                        "(cross-case comparisons). Embed each where the result it shows is reported, "
                        "unless it repeats a figure already embedded.")}
                   if named_pngs else {}),
                "manifest_excerpt": {
                    "cases": [
                        c
                        for c in (manifest.get("cases") or [])
                        if isinstance(c, dict) and c.get("case_id") in set(included_order)
                    ]
                },
                "mesh_independence": mesh_bundle,
                "case_records": records_by_case,
                **({"study_results": study_results,
                    "study_results_rule": ("study_results holds study-level results a script computed "
                                           "from the cases' records; report from them, as from case_records.")}
                   if study_results else {}),
                # Over the analysis text too: written before the records reached it,
                # the cross-case interpretation said the final coefficients were not
                # exposed, and the draft repeated it (nn_in_solver_demo_20261022_1129).
                "case_records_rule": (
                    "case_records holds each included case's own result files as the case saved "
                    "them, and it outranks the analysis text: where the analysis says a quantity is "
                    "missing, not supplied or not exposed but case_records hold it, report it from "
                    "case_records. Say a quantity is missing only when neither holds it."),
            }
            section_context = json.dumps(unified_context, indent=2)

            if outer == 0 and start_tex:
                # The paper as it stands, reviewed before anything rewrites it:
                # a manager's corrections from the case records were lost when
                # the next write_paper drafted the paper again from scratch, and
                # review.json kept reviewing the draft it had replaced
                # (nn_in_solver_demo_20261022_1129).
                paper_tex = start_tex
            elif outer == 0:
                paper_tex = writer.write_paper_with_literature(
                    topic=topic,
                    section_context=section_context,
                    ideation_literature_bundle=literature,
                    visualization_bundle=viz_bundle,
                    max_literature_papers=40,
                    verbose=verbose,
                    required_figures=[str(p) for p in rendered_pngs],
                )
                paper_tex = strip_latex_fences(paper_tex)
            else:
                # The script and the writer both answered the same review: the
                # figures went from one panel to two while the captions went
                # from two to one, a round behind each other
                # (shelf_demo_hystrath_20261022_1030).
                recs = [
                    "The figures were re-rendered after the review below. Where a recommendation says what "
                    "a figure shows, it describes the previous image, not the one now on disk: write every "
                    "caption and every sentence about a figure from paper_image_analysis, which describes "
                    "the figures as they are now."
                ] + (list(recs_prev) or ["Refresh manuscript for new batch figures and paper_image_analysis."])
                paper_tex = writer.revise_paper(
                    paper_tex,
                    recs,
                    visualization_bundle=viz_bundle,
                    work_dir=work_dir,
                    analysis_context=section_context,
                    verbose=verbose,
                    required_figures=[str(p) for p in rendered_pngs],
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
                required_figures=[str(p) for p in rendered_pngs],
            )
            paper_tex = strip_latex_fences(paper_tex)

        paper_tex = _embed_required(paper_tex, outer + 1)

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
            study_brief=topic,
        )
        # Judged from the rendered figures themselves (a vision call), not from
        # the LaTeX alone; None (not judged) is kept as None, never as a pass.
        if compile_ok:
            review.update(reviewer.judge_figure_physics(rendered_pngs, paper_tex))
        else:
            review.update({
                "figure_physics_ok": None,
                "figure_physics_reason": "not judged: the paper did not compile",
                "figure_physics_per_figure": [],
            })
        review["missing_embeds"] = missing_embeds(paper_tex, rendered_pngs)
        review["missing_out_png"] = list(missing_pngs)
        review_info["reviews"].append(review)
        last_rev = review
        score = float(review.get("score", 0.0))
        passed = (
            bool(review.get("pass", False))
            and compile_ok
            and score >= float(score_threshold)
            and review.get("figure_physics_ok") is not False
            and not review["missing_embeds"]
        )
        # The best compiled version so far, kept: a later round's revision
        # can lower the score, and the paper saved was always the last round's
        # (nn_in_solver_demo_20261022_1129: 0.78, then a rewrite that lost
        # sections and equations was what the study kept).
        # Its figures are kept with it: a later round re-renders them under the
        # same names, and the kept text and review then described images that
        # were no longer on disk (shelf_demo_hystrath_20261022_1030: the manager
        # chased a figure review of a single-panel render replaced by a two-panel one).
        if compile_ok and _rank(review) > best["rank"]:
            import shutil

            best.update(score=score, rank=_rank(review), tex=paper_tex, review=dict(review), round=outer + 1, pdf="",
                        pngs=list(rendered_pngs))
            if pdf_path and Path(pdf_path).is_file():
                kept = paper_dir / ".best_round.pdf"
                shutil.copy2(pdf_path, kept)
                best["pdf"] = str(kept)
            figs_dir = paper_dir / ".best_round_figs"
            shutil.rmtree(figs_dir, ignore_errors=True)
            figs_dir.mkdir(parents=True)
            best["fig_copies"] = []
            for i, p in enumerate(rendered_pngs):
                if p.is_file():
                    copy = figs_dir / f"{i:03d}_{p.name}"
                    shutil.copy2(p, copy)
                    best["fig_copies"].append((copy, p))
            # and the script that drew them: the next write_paper runs it as
            # it stands, and a later round's rewrite redrew the kept figures
            # (shelf_demo_hystrath_20261022_1030)
            batch_script = paper_figs_root / "paper_viz_batch.py"
            if batch_script.is_file():
                copy = figs_dir / batch_script.name
                shutil.copy2(batch_script, copy)
                best["fig_copies"].append((copy, batch_script))
        _save_outputs()
        last_round_s = time.monotonic() - round_started
        recs_prev = list(review.get("recommendations") or [])
        if review.get("figure_physics_ok") is False:
            recs_prev.append("Figure physics check failed: " + str(review.get("figure_physics_reason", "")))
        recs_prev.extend(
            f"Embed the planned figure {p} with its own \\includegraphics (exact path)."
            for p in review["missing_embeds"]
        )
        # a planned figure that was never written asks for the script again
        needs_viz_prev = bool(review.get("needs_additional_visualization", False)) or bool(missing_pngs)
        if missing_pngs:
            review.setdefault("additional_viz_specs", [])
            review["additional_viz_specs"] = list(review["additional_viz_specs"]) + [
                f"write the planned figure at exactly {run_dir / p}" for p in missing_pngs
            ]
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
            review_info["compile_recovery"] = {"rounds": cr + 1, "compiled": compile_ok}
            if compile_ok:
                break
        # Reviewed again once it compiles: review.json kept the uncompiled
        # round's "did not compile, score 0.0" beside a PDF that had compiled,
        # and the manager reads review.json (nn_in_solver_demo_20261022_1129).
        if last_compile_ok:
            rels = []
            for p in fig_paths:
                try:
                    rels.append(str(Path(p).relative_to(work_dir)))
                except ValueError:
                    rels.append(p)
            review = reviewer.review(
                tex_content=paper_tex, compile_ok=True, compile_error="",
                reference_report="", valid_figure_paths=rels, study_brief=topic,
            )
            review.update(reviewer.judge_figure_physics(rendered_pngs, paper_tex))
            review["missing_embeds"] = missing_embeds(paper_tex, rendered_pngs)
            review["missing_out_png"] = list(missing_pngs)
            review["after_compile_recovery"] = True
            review_info["reviews"].append(review)

    final = review_info["reviews"][-1] if review_info["reviews"] else {}
    final_score = float(final.get("score", 0.0)) if last_compile_ok else -1.0
    final_rank = _rank(final) if last_compile_ok else (-1, -1.0)
    if best["tex"] and best["rank"] > final_rank:
        paper_tex = best["tex"]
        tex_path.write_text(paper_tex, encoding="utf-8")
        import shutil

        for copy, original in best.get("fig_copies") or []:
            shutil.copy2(copy, original)
        rendered_pngs = list(best.get("pngs") or rendered_pngs)
        last_compile_ok = True
        last_pdf = best["pdf"] or last_pdf
        review_info["kept_best_round"] = best["round"]
        review_info["reviews"].append({**best["review"], "kept_best_round": best["round"]})
        if verbose:
            print(f"[paper_unified] kept round {best['round']} (score {best['score']}): the last round "
                  f"scored {final_score if final_score >= 0 else 'nothing, as it did not compile'}")
    _save_outputs()
    last_rev = review_info["reviews"][-1] if review_info["reviews"] else {}

    return {
        "ok": bool(last_compile_ok),
        "paper_dir": str(paper_dir),
        "review_path": str(review_path),
        "plan_path": str(plan_path),
        "last_score": float(last_rev.get("score", 0.0)),
        "compile_ok": last_compile_ok,
    }

"""Writer agent: LaTeX paper generation with Sakana AI Scientist v2–aligned features.

Includes: standard sections (Abstract–Conclusion), claim–evidence table,
figure–text alignment, multi-round citations, reproducibility appendix,
failure/negative results section, and mandatory AI-disclosure sentence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.agents.literature_agent import LiteratureSurveyAgent
from cfd_langgraph.agents.paper_reviewer_agent import PaperReviewerAgent
from cfd_langgraph.paper_utils import compile_tex_to_pdf, extract_pdflatex_errors

_COMPILE_ERR_TAIL_CHARS = 14_000
from cfd_langgraph.utils import extract_json_object, strip_json_fences, strip_latex_fences
from cfd_langgraph.refchecker_integration import run_refchecker_on_tex


def _get_figure_paths_for_review(
    visualization_bundle: Optional[List[Dict[str, Any]]],
    work_dir: Optional[Path],
) -> List[str]:
    """Extract valid figure paths (relative to work_dir) for reviewer."""
    if not visualization_bundle or not work_dir:
        return []
    paths: List[str] = []
    for v in visualization_bundle:
        vis = v.get("visualization", {}) if isinstance(v, dict) else {}
        for p in vis.get("images", []) or []:
            if isinstance(p, str) and p.strip():
                try:
                    rel = str(Path(p).relative_to(Path(work_dir)))
                    if rel not in paths:
                        paths.append(rel)
                except ValueError:
                    paths.append(p)
    return paths


# Writer features: experiment-only, no hallucination, grounded to visualizations, 8–15 pages.
AI_SCIENTIST_V2_STYLE_CHECKLIST = """
Manuscript requirements (experiment-grounded, no hallucination):

Length:
0) Main body (Abstract–Conclusion): at least 8 pages, at most 15 pages, excluding References and Appendix.

Scope and truthfulness:
1) The paper must reflect ONLY the provided experiments. No hallucinations. Do not mention standard literature or theory that was not actually performed or validated in these experiments.
2) Analysis, Discussion, and Conclusion must be grounded strictly in the given visualizations and experiment data. Every claim must map to a specific figure, table, or number from the provided analysis.

Figures:
3) Include only good-quality figures. If an image is poor (blurry, wrong, uninformative), omit it. If an image doesn't make sense or doesn't provide any information, exclude it.
3b) Prefer figures suitable for journal print: colorbars and legends must not cover the flow field or data; text must be large enough to read at column width. Omit figures that fail this (common with default PyVista exports).
4) Include at least one important figure from each experiment so all experiments are represented. Not every image from every experiment is required, but each experiment must appear in at least one figure.
5) Every figure must be referenced in the main text; captions must accurately describe what is shown. No hallucinated numbers or mismatched descriptions. Use LaTeX \\ref{fig:...} for all figures.

Structure:
6) Standard sections: Abstract, Introduction, Related Work (only if relevant to what was done), Methods, Results, Discussion, Conclusion; Reproducibility appendix and Claim–Evidence table.
6b) If the section context JSON includes a non-empty `mesh_independence` object from the workflow, include an explicit **mesh refinement / mesh independence** subsection with a table of QoIs per mesh level and state which mesh was selected for production runs; use only provided numbers.
7) Claim–evidence table: tie each major claim to specific experiments, figures, and numbers. No unsupported claims.
8) Failure Cases / Negative Results subsection if any run failed or was inconclusive.
9) Reproducibility appendix: solver, OpenFOAM/case setup, mesh, BCs, time step and end time, how to run—only what was actually used.

Abstract (strict):
10a) Abstract must NOT name case numbers (no “case 1”, “case_003”, exp_ IDs) and must NOT reference section numbers (“Section 3”, etc.).

Case naming (body):
10b) In prose use “case 1”, “case 2” (space, not underscore) only after introducing what each case is; never use case_001-style tokens in narrative. If context lists omitted duplicate experiments, say nothing about them.

Disclosure:
10) Mandatory: one sentence in Abstract or Methods that this draft was generated with an automated CFD Scientist (AI-assisted) pipeline and that results and figures come from the provided experiments and analysis.
11) Introduce the set of experiments with a clear table early in Methods/Results before referring to them by number; the table should summarise for each experiment the key varying parameter(s) (e.g. turbulence model, wall treatment), mesh size, and any other essential configuration.
12) Avoid repeating the same explanatory paragraph or caveat (e.g. geometry mismatch, post-processing artefact, expansion-ratio difference) in multiple sections. Provide a single, well-placed, detailed explanation and refer back to it briefly elsewhere instead of duplicating text.
13) The Conclusion section must be written as one cohesive paragraph (continuous prose). Do NOT use bullet points, numbered lists, or itemized/enumerated formatting in Conclusion.
""".strip()


class WriterAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("WriterAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.0)
        self.lit_agent = LiteratureSurveyAgent(model=model)
        self.reviewer = PaperReviewerAgent(model=model, prompt_loader=prompt_loader)

    def write_section(self, section_context: str) -> str:
        system = self.prompts.get(
            "system_prompt", "You are a LaTeX paper writer for CFD."
        )
        user = self.prompts.get(
            "user_prompt",
            "SECTION CONTEXT:\n{section_context}\nReturn LaTeX section only.",
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"section_context": section_context}).content

    def collect_citations(
        self, citation_context: str, total_rounds: int = 2
    ) -> List[Dict[str, Any]]:
        sys_t = self.prompts.get("citation_system_prompt", "")
        usr_t = self.prompts.get("citation_user_prompt", "")
        if not sys_t or not usr_t:
            return []

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_t),
                ("human", usr_t),
            ]
        )
        chain = prompt | create_langchain_llm(self.model, temperature=0.0)

        collected: List[Dict[str, Any]] = []
        for r in range(1, max(1, total_rounds) + 1):
            raw = chain.invoke(
                {
                    "citation_context": citation_context,
                    "round": r,
                    "total_rounds": total_rounds,
                }
            ).content
            try:
                parsed = json.loads(extract_json_object(raw))
            except Exception:
                continue
            cites = parsed.get("citations", []) if isinstance(parsed, dict) else []
            for c in cites:
                if isinstance(c, dict):
                    collected.append(c)
            if isinstance(parsed, dict) and parsed.get("action") == "stop":
                break

        # dedup by title
        seen = set()
        uniq = []
        for c in collected:
            t = (c.get("title") or "").strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            uniq.append(c)
        return uniq

    def write_paper_with_literature(
        self,
        topic: str,
        section_context: str,
        ideation_literature_bundle: Optional[List[Dict[str, Any]]] = None,
        visualization_bundle: Optional[List[Dict[str, Any]]] = None,
        max_literature_papers: int = 40,
        verbose: bool = False,
    ) -> str:
        # Prefer workflow ideation literature to keep end-to-end context aligned.
        lit_bundle: Any = ideation_literature_bundle or []
        if not lit_bundle:
            if verbose:
                print("[Writer] Literature survey (Semantic Scholar, max %d papers)..." % max_literature_papers)
            try:
                lit_bundle = self.lit_agent.survey(idea_text=topic, max_papers=max_literature_papers)
            except Exception as e:
                if verbose:
                    print("[Writer] Literature survey failed: %s" % e)
                lit_bundle = {
                    "idea": topic,
                    "semantic_scholar": [],
                    "web_results": [],
                    "synthesis": "",
                    "fetch_error": str(e),
                }

        # Multi-round citation (AI Scientist v2–style) for better coverage.
        try:
            citations = self.collect_citations(citation_context=topic, total_rounds=3)
        except Exception:
            citations = []

        system = self.prompts.get(
            "system_prompt",
            "You are a LaTeX paper writer for CFD.",
        )
        user = (
            "Write a complete publication-ready LaTeX paper draft using ALL provided context and artifacts.\n\n"
            "Topic:\n{topic}\n\n"
            "Context (including analysis summaries):\n{section_context}\n\n"
            "Literature survey bundle:\n{lit_bundle}\n\n"
            "Candidate citations (use these; add \\cite{{}} where appropriate):\n{citations}\n\n"
            "Visualization bundle (figures and paths): Include only good-quality images. You must include at least one figure from each experiment; omit poor or uninformative images. Every figure included must have a caption and be referenced in the text. Do not include figures that do not clearly support the narrative.\n{viz_bundle}\n\n"
            "Mandatory style checklist (Sakana AI Scientist v2–aligned):\n{checklist}\n\n"
            "Requirements:\n"
            "- Structure: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion; add Reproducibility appendix and Claim–Evidence table.\n"
            "- Use evidence-grounded claims only; every claim must map to an experiment, figure, or number in the analysis.\n"
            "- When cross-experiment artifacts are present in context (e.g., fitted formulas/correlations, regression outputs, cross-case tables, or cross_experiment_analysis plots), include and discuss them explicitly in Results/Discussion with matching equations/figure references.\n"
            "- Include a Failure Cases / Negative Results subsection.\n"
            "- Include mandatory disclosure in Abstract or Methods: one sentence that this draft was generated with an automated CFD Scientist (AI-assisted) pipeline and that results/figures come from the provided experiments and analysis.\n"
            "- Reference every figure with \\ref{{fig:...}}; ensure captions and in-text descriptions match the actual visualization data (no hallucinated values).\n"
            "- Conclusion format: write Conclusion as a single paragraph only; never as bullet points or numbered lists.\n"
            "Return ONLY LaTeX (no markdown, no explanation outside comments)."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user),
            ]
        )
        chain = prompt | self.llm
        out = chain.invoke(
            {
                "topic": topic,
                "section_context": section_context,
                "lit_bundle": json.dumps(lit_bundle),
                "citations": json.dumps(citations),
                "checklist": AI_SCIENTIST_V2_STYLE_CHECKLIST,
                "viz_bundle": json.dumps(visualization_bundle or []),
            }
        )
        result = getattr(out, "content", str(out))
        if verbose:
            print("[Writer] Paper draft generated.")
        return result

    def revise_paper(
        self,
        tex_content: str,
        recommendations: List[str],
        visualization_bundle: Optional[List[Dict[str, Any]]] = None,
        work_dir: Optional[Path] = None,
        analysis_context: Optional[str] = None,
        verbose: bool = False,
    ) -> str:
        """Revise the LaTeX paper based on reviewer recommendations. analysis_context (topic, idea, analysis report) is provided so the writer knows what experiments exist and what figures are available per experiment."""
        system = self.prompts.get(
            "system_prompt",
            "You are a LaTeX paper writer for CFD.",
        )
        fig_paths_block = ""
        if visualization_bundle and work_dir:
            by_exp: Dict[str, List[str]] = {}
            for v in visualization_bundle:
                sim_id = (v.get("simulation_id") or v.get("case_name") or "unknown") if isinstance(v, dict) else "unknown"
                vis = v.get("visualization", {}) if isinstance(v, dict) else {}
                imgs = vis.get("images", [])
                paths_for_exp: List[str] = []
                for p in imgs:
                    if isinstance(p, str) and p.strip():
                        try:
                            rel = str(Path(p).relative_to(Path(work_dir)))
                            if rel not in paths_for_exp:
                                paths_for_exp.append(rel)
                        except ValueError:
                            paths_for_exp.append(p)
                if paths_for_exp:
                    by_exp.setdefault(sim_id, []).extend(paths_for_exp[:15])
            if by_exp:
                lines = []
                for sid, plist in sorted(by_exp.items()):
                    lines.append(f"{sid}:")
                    for p in plist:
                        lines.append(f"  - {p}")
                fig_paths_block = (
                    "\n\nFIGURE PATHS BY EXPERIMENT (use in \\includegraphics; relative to project root):\n"
                    + "\n".join(lines)
                )
        analysis_block = ""
        if analysis_context:
            analysis_block = (
                "\n\nANALYSIS CONTEXT (topic, experiments, available figures—use this to add missing experiment figures or correct descriptions):\n"
                "---\n"
                f"{analysis_context[:30000]}\n"
                "---\n"
            )
        user = (
            "Revise this LaTeX paper according to the reviewer recommendations. "
            "Apply ALL recommended fixes. When the reviewer says an experiment (e.g. exp_004) has no figures: use the FIGURE PATHS BY EXPERIMENT and ANALYSIS CONTEXT below to add the correct \\includegraphics from that experiment. Use ONLY paths from the list; do NOT invent paths. "
            "LENGTH: Main body 8–15 pages. FIGURES: At least one from each experiment; only good-quality images. Exclude any image that doesn't make sense or doesn't provide useful information. SCOPE: No hallucinations; grounded in the given analysis and visualizations.\n\n"
            "Return ONLY the complete revised LaTeX document (no markdown, no explanation).\n\n"
            "CURRENT LaTeX:\n{tex_content}\n\n"
            "REVIEWER RECOMMENDATIONS:\n{recommendations}"
            "{fig_paths}"
            "{analysis_context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user),
        ])
        chain = prompt | self.llm
        if verbose:
            print("[Writer] Revising paper (%d recommendations)..." % len(recommendations))
        out = chain.invoke({
            "tex_content": tex_content[:80000],
            "recommendations": "\n".join(f"- {r}" for r in recommendations),
            "fig_paths": fig_paths_block,
            "analysis_context": analysis_block,
        })
        if verbose:
            print("[Writer] Revision done.")
        return getattr(out, "content", str(out))

    def revise_paper_for_compilation_only(
        self,
        tex_content: str,
        compile_error: str,
        visualization_bundle: Optional[List[Dict[str, Any]]] = None,
        work_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> str:
        """Minimal LaTeX edits so pdflatex succeeds (no publishability review)."""
        system = self.prompts.get("compile_fix_system_prompt", "").strip()
        if not system:
            system = (
                "You are a LaTeX build engineer. The document failed pdflatex. "
                "Output a complete LaTeX document that compiles. "
                "Only fix build errors (paths, packages, environments, escapes, undefined commands). "
                "Do not rewrite for style or length. Return ONLY full LaTeX."
            )
        user_t = self.prompts.get("compile_fix_user_prompt", "").strip()
        if not user_t:
            user_t = (
                "Fix compilation.\n\nERRORS:\n{error_summary}\n\nLOG TAIL:\n{compile_error_tail}\n\n"
                "VALID FIGURE PATHS:\n{valid_figure_paths}\n\nLaTeX:\n{tex_content}\n\n"
                "Return ONLY complete LaTeX."
            )
        paths = _get_figure_paths_for_review(visualization_bundle, work_dir)
        paths_block = "\n".join(f"- {p}" for p in paths[:80]) if paths else "(none listed — keep existing paths that match the project layout)"
        err = compile_error or ""
        summary = extract_pdflatex_errors(err, max_errors=8)
        tail = err[-_COMPILE_ERR_TAIL_CHARS:] if len(err) > _COMPILE_ERR_TAIL_CHARS else err
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user_t),
            ]
        )
        chain = prompt | self.llm
        if verbose:
            print("[Writer] Compile-only revision (minimize edits, no reviewer pass)...")
        out = chain.invoke(
            {
                "tex_content": tex_content[:80000],
                "error_summary": summary,
                "compile_error_tail": tail,
                "valid_figure_paths": paths_block,
            }
        )
        if verbose:
            print("[Writer] Compile-only revision done.")
        return getattr(out, "content", str(out))

    def write_paper_with_literature_and_review(
        self,
        topic: str,
        section_context: str,
        out_dir: Path,
        work_dir: Path | None = None,
        ideation_literature_bundle: Optional[List[Dict[str, Any]]] = None,
        visualization_bundle: Optional[List[Dict[str, Any]]] = None,
        max_literature_papers: int = 40,
        max_review_tries: int = 10,
        max_compile_recovery_tries: int = 10,
        verbose: bool = False,
    ) -> Tuple[str, Path | None, Dict[str, Any]]:
        """
        Write paper, compile to PDF, and run reviewer loop until pass or max tries.

        If the review budget is exhausted and pdflatex still fails, runs an extra
        compile-only fix loop (no publishability review) to try to produce a PDF.

        Returns:
            (final_tex, pdf_path, review_info)
            pdf_path is None if compilation never succeeded.
        """
        if verbose:
            print("[Writer] === Paper writing (with literature and review loop, max %d tries) ===" % max_review_tries)

        paper_tex = self.write_paper_with_literature(
            topic=topic,
            section_context=section_context,
            ideation_literature_bundle=ideation_literature_bundle,
            visualization_bundle=visualization_bundle,
            max_literature_papers=max_literature_papers,
            verbose=verbose,
        )
        paper_tex = strip_latex_fences(paper_tex)

        tex_path = out_dir / "paper_draft.tex"
        work_dir = work_dir or out_dir
        review_info: Dict[str, Any] = {"tries": 0, "reviews": []}
        last_compile_ok = False
        last_pdf_path: Path | None = None
        last_compile_err = ""

        for attempt in range(max_review_tries):
            review_info["tries"] = attempt + 1
            n = attempt + 1
            if verbose:
                print("[Writer] --- Attempt %d / %d ---" % (n, max_review_tries))

            tex_path.write_text(paper_tex, encoding="utf-8")

            reference_report_summary = ""
            try:
                # Validate generated citations/references before pdflatex.
                # This helps the reviewer catch missing/incorrect BibTeX entries.
                _, reference_report_summary = run_refchecker_on_tex(
                    tex_path=tex_path,
                    out_dir=out_dir / "refchecker_reports",
                    attempt=n,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[Writer] refchecker integration failed: {e}")
                reference_report_summary = ""

            if verbose:
                print("[Writer] Compiling LaTeX (pdflatex)...")
            compile_ok, pdf_path, compile_err = compile_tex_to_pdf(
                tex_path, work_dir=work_dir
            )
            last_compile_ok = compile_ok
            last_pdf_path = pdf_path
            last_compile_err = compile_err or ""

            if compile_ok:
                if verbose:
                    print("[Writer] Compilation: SUCCESS -> %s" % pdf_path)
            else:
                if verbose:
                    print("[Writer] Compilation: FAILED")
                    # Print full pdflatex log when available so users can inspect all errors.
                    print("[Writer] Compilation errors (full log):\n%s" % (compile_err or ""))

            if verbose:
                print("[Writer] Reviewing paper...")
            # Pass the full pdflatex log (if any) to the reviewer so it can see all errors.
            err_for_review = compile_err
            review = self.reviewer.review(
                tex_content=paper_tex,
                compile_ok=compile_ok,
                compile_error=err_for_review,
                reference_report=reference_report_summary,
                valid_figure_paths=_get_figure_paths_for_review(visualization_bundle, work_dir),
            )
            review_info["reviews"].append(review)

            passed = bool(review.get("pass", False))
            score = review.get("score", 0)
            recs = review.get("recommendations", [])
            summary = review.get("summary", "")

            if verbose:
                print("[Writer] Review: %s (score=%.2f) %s" % (
                    "PASS" if passed else "FAIL",
                    score,
                    ("- %s" % summary) if summary else "",
                ))
                if recs:
                    print("[Writer] Recommendations (%d):" % len(recs))
                    for i, r in enumerate(recs[:10], 1):
                        print("[Writer]   %d. %s" % (i, (r[:120] + "..." if len(r) > 120 else r)))
                    if len(recs) > 10:
                        print("[Writer]   ... and %d more" % (len(recs) - 10))

            if compile_ok and passed:
                if verbose:
                    print("[Writer] === Done: paper accepted after %d attempt(s) ===" % n)
                return paper_tex, pdf_path, review_info

            recommendations = recs
            if not recommendations:
                recommendations = [summary or "Improve overall quality."]

            if attempt < max_review_tries - 1:
                if verbose:
                    print("[Writer] Revising paper and retrying...")
                paper_tex = self.revise_paper(
                    paper_tex,
                    recommendations,
                    visualization_bundle=visualization_bundle,
                    work_dir=work_dir,
                    analysis_context=section_context,
                    verbose=verbose,
                )
                paper_tex = strip_latex_fences(paper_tex)
            else:
                if verbose:
                    print("[Writer] Max tries (%d) reached; returning last version." % max_review_tries)

        # Review budget exhausted: if the last build failed, try compile-only fixes (no reviewer pass).
        recovery_cap = max(0, int(max_compile_recovery_tries))
        if not last_compile_ok and recovery_cap > 0:
            review_info["compile_recovery"] = {
                "attempts": 0,
                "succeeded": False,
                "max_tries": recovery_cap,
            }
            if verbose:
                print(
                    "[Writer] Review loop ended with compilation failure; "
                    "starting compile-only recovery (max %d tries, no publishability review)."
                    % recovery_cap
                )
            for cr in range(recovery_cap):
                review_info["compile_recovery"]["attempts"] = cr + 1
                if verbose:
                    print("[Writer] --- Compile recovery %d / %d ---" % (cr + 1, recovery_cap))
                paper_tex = self.revise_paper_for_compilation_only(
                    paper_tex,
                    last_compile_err,
                    visualization_bundle=visualization_bundle,
                    work_dir=work_dir,
                    verbose=verbose,
                )
                paper_tex = strip_latex_fences(paper_tex)
                tex_path.write_text(paper_tex, encoding="utf-8")
                compile_ok, pdf_path, compile_err = compile_tex_to_pdf(
                    tex_path, work_dir=work_dir
                )
                last_compile_ok = compile_ok
                last_pdf_path = pdf_path
                last_compile_err = compile_err or ""
                if compile_ok:
                    review_info["compile_recovery"]["succeeded"] = True
                    if verbose:
                        print(
                            "[Writer] Compile recovery succeeded -> %s" % (pdf_path or "")
                        )
                    return paper_tex, pdf_path, review_info
            review_info["compile_recovery"]["succeeded"] = False
            if verbose:
                print("[Writer] Compile recovery exhausted; PDF may be missing.")

        pdf_path = last_pdf_path
        if pdf_path is None or not Path(pdf_path).exists():
            fallback = out_dir / "paper_draft.pdf"
            pdf_path = fallback if fallback.is_file() else None
        return paper_tex, pdf_path, review_info

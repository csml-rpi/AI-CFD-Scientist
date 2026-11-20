"""The paper pipeline produces what the final audit's figure checks read.

shelf_demo_dafoam_20261019_1057 and shelf_demo_hystrath_20261022_1030 both
failed H13/H15/H16/H28 because nothing in the paper pipeline wrote the fields
those checks read. Covered here:

  1. every planned figure job has id, kind (from the fixed list) and out_png,
     and each successful case gets a u_contour and a p_contour, on the
     velocity/pressure field the case actually wrote (U, U_Ar, p_Ar, ...)
  2. the fallback plan does the same; normalising twice changes nothing
  3. out_png values are unique, under paper_figs/, and carry the case id
  4. the writer's prompt lists every out_png to embed
  5. the reviewer returns figure_physics_ok, and review.json carries it
  6. end to end, with the LLM steps faked: the audit's H13/H15/H16/H28 pass
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai", CFD_SCIENTIST_MODEL="fake/model",
                  OPENAI_API_KEY="fake", OPENAI_BASE_URL="http://127.0.0.1:9/v1")

FAILURES = []


def check(label, got, want=True):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        FAILURES.append(label)


from cfd_langgraph.paper_unified import figure_contract as fc  # noqa: E402
from cfd_langgraph.paper_unified import planner  # noqa: E402
import stage_gate_audit as audit  # noqa: E402


def _field(path: Path, cls: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    object      {path.name};\n}}\n"
    )


def _make_cases(root: Path) -> list:
    """Four cases: a plain one (U, p), a DSMC one (U_Ar, p_Ar, no U/p), one
    with no pressure at all (rhoN_Ar) and an unreconstructed parallel one."""
    plain = root / "cases" / "case_001"
    _field(plain / "0" / "U", "volVectorField")
    _field(plain / "200" / "U", "volVectorField")
    _field(plain / "200" / "p", "volScalarField")
    _field(plain / "200" / "phi", "surfaceScalarField")
    dsmc = root / "cases" / "case_002"
    _field(dsmc / "0.0005" / "U_Ar", "volVectorField")
    _field(dsmc / "0.0005" / "p_Ar", "volScalarField")
    _field(dsmc / "0.0005" / "rhoN_Ar", "volScalarField")
    nop = root / "cases" / "case_003"
    _field(nop / "0.001" / "U_Ar", "volVectorField")
    _field(nop / "0.001" / "rhoN_Ar", "volScalarField")
    par = root / "cases" / "case_004"
    _field(par / "0" / "U", "volVectorField")
    _field(par / "processor0" / "50" / "U", "volVectorField")
    _field(par / "processor0" / "50" / "p_rgh", "volScalarField")
    failed = root / "cases" / "case_005"
    _field(failed / "0" / "U", "volVectorField")
    return [
        {"case_id": "case_001", "status": "success", "case_path": str(plain)},
        {"case_id": "case_002", "status": "success", "case_path": str(dsmc)},
        {"case_id": "case_003", "status": "success", "case_path": str(nop)},
        {"case_id": "case_004", "status": "success", "case_path": str(par)},
        {"case_id": "case_005", "status": "failed", "case_path": str(failed)},
    ]


def _per_case_ok(plan, cases):
    jobs = plan["figure_jobs"]
    out = True
    for c in cases:
        if c["status"] != "success":
            continue
        kinds = {j["kind"] for j in jobs if j.get("case_id") == c["case_id"]}
        out = out and {"u_contour", "p_contour"} <= kinds
    return out


with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    cases = _make_cases(run)
    fields = {c["case_id"]: fc.case_field_summary(Path(c["case_path"])) for c in cases}

    print("== 1. the case's own field names")
    check("plain case: U", fields["case_001"]["velocity_field"], "U")
    check("plain case: p (not phi)", fields["case_001"]["pressure_field"], "p")
    check("plain case: latest time", fields["case_001"]["time"], "200")
    check("DSMC case: U_Ar", fields["case_002"]["velocity_field"], "U_Ar")
    check("DSMC case: p_Ar", fields["case_002"]["pressure_field"], "p_Ar")
    check("no pressure: a density stands in", fields["case_003"]["pressure_field"], "rhoN_Ar")
    check("and the note says so", "no pressure field" in fields["case_003"]["pressure_note"])
    check("parallel case read from processor0", fields["case_004"]["pressure_field"], "p_rgh")
    check("failed case: only initial conditions", fields["case_005"]["solution_written"], False)

    print("== 1b. a planner's plan, normalised")
    llm_plan = {
        "omit_case_ids": [],
        "case_order_for_paper": ["case_001", "case_002", "case_003", "case_004"],
        "case_display_labels": {},
        "figure_jobs": [
            {"case_id": "case_001", "kind": "u_contour", "field": "U", "what_to_visualize": "Ux"},
            {"case_id": "case_001", "kind": "Pressure Contour", "what_to_visualize": "p"},
            {"case_id": "case_002", "kind": "u_contour", "field": "U", "what_to_visualize": "wrong field"},
            {"case_id": "case_002", "kind": "field_contour", "field": "Ttra_Ar", "what_to_visualize": "T"},
            {"case_id": "case_002", "kind": "field_contour", "field": "Ttra_Ar", "what_to_visualize": "T zoom"},
            {"case_id": None, "kind": "comparison", "what_to_visualize": "all cases"},
            {"case_id": "case_005", "kind": "u_contour", "what_to_visualize": "failed, nothing written"},
            {"case_id": "case_999", "kind": "u_contour", "what_to_visualize": "not a case"},
            {"case_id": "case_003", "kind": "made_up_kind", "out_png": "fig1.png", "id": "x"},
        ],
        "unified_viz_brief": "",
        "analysis_narrative_hints": "",
        "topic_paragraph_for_intro": "",
    }
    plan = fc.normalize_figure_jobs(json.loads(json.dumps(llm_plan)), cases, fields)
    jobs = plan["figure_jobs"]
    check("every job has id, kind, out_png",
          all(j.get("id") and j.get("kind") and j.get("out_png") for j in jobs))
    check("every kind is in the fixed list", all(j["kind"] in fc.FIGURE_KINDS for j in jobs))
    check("one u_contour and one p_contour per successful case", _per_case_ok(plan, cases))
    check("'Pressure Contour' read as p_contour",
          [j["kind"] for j in jobs if j["case_id"] == "case_001"], ["u_contour", "p_contour"])
    check("a field the case lacks is replaced by its own",
          [j["field"] for j in jobs if j["case_id"] == "case_002" and j["kind"] == "u_contour"], ["U_Ar"])
    check("DSMC p_contour on p_Ar",
          [j["field"] for j in jobs if j["case_id"] == "case_002" and j["kind"] == "p_contour"], ["p_Ar"])
    p3 = [j for j in jobs if j["case_id"] == "case_003" and j["kind"] == "p_contour"][0]
    check("density stand-in carries its note", ("rhoN_Ar", "no pressure field" in p3.get("field_note", "")),
          ("rhoN_Ar", True))
    check("a failed case with nothing written is dropped", any(j["case_id"] == "case_005" for j in jobs), False)
    check("an unknown case is dropped", any(j["case_id"] == "case_999" for j in jobs), False)
    check("a planner's own out_png is not kept", any(j["out_png"] == "fig1.png" for j in jobs), False)
    check("the cross-case figure has no case id and its own name",
          [(j["kind"], j["out_png"]) for j in jobs if not j["case_id"]],
          [("case_comparison", "paper_figs/cross_01_case_comparison.png")])
    check("the per-case contour is paper_figs/<case_id>_<kind>.png",
          [j["out_png"] for j in jobs if j["case_id"] == "case_004" and j["kind"] == "u_contour"],
          ["paper_figs/case_004_u_contour.png"])

    print("== 2. fallback plan, and normalising twice")
    fb = planner.fallback_plan([c for c in cases if c["status"] == "success"], fields)
    check("fallback: every job has id, kind, out_png",
          all(j.get("id") and j.get("kind") and j.get("out_png") for j in fb["figure_jobs"]))
    check("fallback: u and p contour per case", _per_case_ok(fb, cases))
    check("fallback: DSMC fields", sorted(j["field"] for j in fb["figure_jobs"] if j["case_id"] == "case_002"),
          ["U_Ar", "p_Ar"])
    check("fallback keeps the audit's plan keys",
          all(k in fb for k in ["figure_jobs", "unified_viz_brief", "case_order_for_paper", "case_display_labels",
                                "omit_case_ids", "analysis_narrative_hints", "topic_paragraph_for_intro"]))
    again = fc.normalize_figure_jobs(json.loads(json.dumps(plan)), cases, fields)
    check("normalising twice changes nothing",
          [(j["id"], j["out_png"]) for j in again["figure_jobs"]], [(j["id"], j["out_png"]) for j in jobs])

    print("== 3. out_png unique, under paper_figs/, carrying the case id")
    leaves = [Path(j["out_png"]).name for j in jobs]
    check("no two jobs share an out_png", len(leaves), len(set(leaves)))
    check("two jobs on the same field get different names",
          sorted(j["out_png"] for j in jobs if j.get("field") == "Ttra_Ar"),
          ["paper_figs/case_002_field_contour_Ttra_Ar.png", "paper_figs/case_002_field_contour_Ttra_Ar_2.png"])
    check("per-case names carry the case id",
          all(j["case_id"] in Path(j["out_png"]).name for j in jobs if j["kind"] in fc.PER_CASE_KINDS))
    check("contract's audit kinds match stage_gate_audit.PER_CASE_KINDS",
          set(fc.AUDIT_PER_CASE_KINDS), set(audit.PER_CASE_KINDS))
    for j in jobs:
        (run / j["out_png"]).parent.mkdir(parents=True, exist_ok=True)
        (run / j["out_png"]).write_bytes(b"png")
    ok, n_valid, n_total, reasons = audit._figure_jobs_have_out_png(plan, run)
    check("the audit's H16 accepts the plan once the files exist", (ok, reasons), (True, []))
    have, missing = fc.split_rendered(jobs, run)
    check("all rendered", (len(have), missing), (len(jobs), []))
    (run / jobs[0]["out_png"]).unlink()
    have, missing = fc.split_rendered(jobs, run)
    check("a missing one is reported, not hidden", missing, [jobs[0]["out_png"]])
    check("and the audit fails it", audit._figure_jobs_have_out_png(plan, run)[0], False)
    (run / jobs[0]["out_png"]).write_bytes(b"png")

    print("== 3b. which figures a LaTeX source embeds")
    tex = ("\\includegraphics[width=0.9\\linewidth]{/a/paper_figs/case_001_u_contour.png}\n"
           "\\includegraphics*{paper_figs/case_001_p_contour.png} \\includegraphics {x/y.png}\n"
           "% \\includegraphics without braces\n")
    check("targets read from \\includegraphics",
          fc.includegraphics_targets(tex),
          ["/a/paper_figs/case_001_u_contour.png", "paper_figs/case_001_p_contour.png", "x/y.png"])
    req = [run / "paper_figs/case_001_u_contour.png", run / "paper_figs/case_002_u_contour.png"]
    check("missing embeds found by file name", fc.missing_embeds(tex, req), [str(req[1])])

print("== 4. the writer's prompt lists every out_png")
from cfd_langgraph.agents.writer_agent import WriterAgent  # noqa: E402

seen_prompts = []


def _fake_llm(prompt_value):
    seen_prompts.append(prompt_value.to_string())
    return "\\documentclass{article}\\begin{document}x\\end{document}"


w = object.__new__(WriterAgent)
w.prompts = {}
w.llm = _fake_llm
w.collect_citations = lambda **kw: []
required = ["/run/paper_figs/case_001_u_contour.png", "/run/paper_figs/case_001_p_contour.png"]
w.write_paper_with_literature(topic="t", section_context="{}", ideation_literature_bundle=[{"title": "x"}],
                              visualization_bundle=[], required_figures=required)
check("first draft: every out_png listed", all(p in seen_prompts[-1] for p in required))
check("first draft: told to use \\includegraphics", "REQUIRED FIGURES" in seen_prompts[-1])
w.revise_paper("tex", ["fix"], visualization_bundle=[], work_dir=Path("/run"), required_figures=required)
check("revision: every out_png listed", all(p in seen_prompts[-1] for p in required))
w.revise_paper("tex", ["fix"], visualization_bundle=[], work_dir=Path("/run"))
check("no list, no block", "REQUIRED FIGURES" in seen_prompts[-1], False)

print("== 5. the reviewer's figure_physics_ok")
from cfd_langgraph.agents.paper_reviewer_agent import PaperReviewerAgent  # noqa: E402


class _Reply:
    def __init__(self, text):
        self.content = text


class _FakeVLM:
    def __init__(self, text):
        self.text = text
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return _Reply(self.text)


with tempfile.TemporaryDirectory() as tmp:
    png = Path(tmp) / "case_001_u_contour.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    r = object.__new__(PaperReviewerAgent)
    r.prompts = {}
    r.llm = _FakeVLM('Looks fine. {"figure_physics_ok": true, "reason": "wake behind body", "per_figure": []}')
    got = r.judge_figure_physics([png], "tex")
    check("a bool verdict with its reason", (got["figure_physics_ok"], got["figure_physics_reason"]),
          (True, "wake behind body"))
    sent = r.llm.seen[1].content
    check("the image itself is sent", any(isinstance(b, dict) and b.get("type") == "image_url" for b in sent))
    check("named by its file", any(isinstance(b, dict) and b.get("text") == f"File: {png.name}" for b in sent))
    r.llm = _FakeVLM("no json here")
    check("an unreadable reply is None, not a pass", r.judge_figure_physics([png], "tex")["figure_physics_ok"], None)
    check("no figures is None", r.judge_figure_physics([], "tex")["figure_physics_ok"], None)

print("== 6. end to end, LLM steps faked: the audit's figure checks")
from cfd_langgraph.paper_unified import pipeline as pp  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp) / "study"
    run.mkdir()
    cases = _make_cases(run)
    manifest = {"cases": cases}
    (run / "manifest.json").write_text(json.dumps(manifest))
    for name, val in (("analysis.json", {}), ("requirements.json", []), ("lit.json", [])):
        (run / name).write_text(json.dumps(val))

    # the planner names no kinds at all: the contract must still hold
    pp.plan_paper_stage = lambda model, text: {
        "omit_case_ids": [], "case_order_for_paper": ["case_001", "case_002"], "case_display_labels": {},
        "figure_jobs": [{"case_id": "case_001", "what_to_visualize": "velocity"}],
        "unified_viz_brief": "b", "analysis_narrative_hints": "", "topic_paragraph_for_intro": "",
        "planner_saw_fields": "U_Ar" in text,
    }
    seen_viz = {}

    def _fake_viz(**kw):
        seen_viz.update(kw)
        out = []
        for p in kw["required_pngs"]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"png")
            out.append(p)
        return "script", out, {}

    pp.run_batch_paper_viz_loop = _fake_viz
    pp.analyze_figures_for_paper = lambda **kw: "analysis"
    pp.run_refchecker_on_tex = lambda **kw: (None, "")
    pp.compile_tex_to_pdf = lambda tex_path, work_dir: (True, None, "")
    pp.get_settings = lambda: type("S", (), {"prompts_path": ROOT / "prompts" / "prompts.yaml"})()

    def _doc(paths):
        figs = "".join(
            f"\\begin{{figure}}\\includegraphics[width=\\linewidth]{{{p}}}\\caption{{c}}\\end{{figure}}\n"
            for p in paths
        )
        return "\\documentclass{article}\\begin{document}\n" + figs + "\\end{document}\n"

    writer_calls = []

    class _FakeWriter:
        def __init__(self, model, prompt_loader):
            pass

        def write_paper_with_literature(self, required_figures=None, **kw):
            writer_calls.append(("write", list(required_figures or [])))
            return _doc((required_figures or [])[:-1])  # drops the last figure

        def revise_paper(self, tex, recs, required_figures=None, **kw):
            writer_calls.append(("revise", list(required_figures or []), list(recs)))
            return _doc(required_figures or [])

        def revise_paper_for_compilation_only(self, tex, err, **kw):
            return tex

    class _FakeReviewer:
        def __init__(self, model, prompt_loader):
            pass

        def review(self, **kw):
            return {"pass": True, "score": 0.9, "formatting_ok": True, "figures_ok": True, "content_ok": True,
                    "coherent": True, "publishable": True, "recommendations": [], "summary": "ok",
                    "regenerate_batch_figures": False}

        def judge_figure_physics(self, image_paths, tex):
            return {"figure_physics_ok": True, "figure_physics_reason": f"{len(image_paths)} figures plausible",
                    "figure_physics_per_figure": []}

    pp.WriterAgent = _FakeWriter
    pp.PaperReviewerAgent = _FakeReviewer
    out = pp.run_unified_paper_pipeline(
        repo_root=ROOT, run_dir=run, topic="t", paper_dir=run / "paper",
        analysis_path=run / "analysis.json", manifest_path=run / "manifest.json",
        requirements_path=run / "requirements.json", lit_path=run / "lit.json",
        review_path=run / "review.json", mesh_independence_path=None, paper_template="neurips",
        model="fake", max_review_loops=2, max_viz_inner_attempts=1, verbose=False,
    )
    plan = json.loads((run / "paper_unified_plan.json").read_text())
    review = json.loads((run / "review.json").read_text())
    check("the planner was shown each case's fields", plan.get("planner_saw_fields"), True)
    check("the viz step got every out_png as a required path",
          sorted(p.name for p in seen_viz["required_pngs"]),
          sorted(Path(j["out_png"]).name for j in plan["figure_jobs"]))
    cfg = json.loads(Path(seen_viz["config_path"]).read_text())
    check("cases_config.json lists the jobs with absolute out_png",
          sorted(Path(j["out_png_abs"]).name for j in cfg["figure_jobs"]),
          sorted(Path(j["out_png"]).name for j in plan["figure_jobs"]))
    check("the brief names the exact files", all(str(run / j["out_png"]) in seen_viz["unified_brief"]
                                                for j in plan["figure_jobs"]))
    check("the writer was given every rendered out_png",
          len(writer_calls[0][1]), len(plan["figure_jobs"]))
    check("a dropped figure got a targeted revision", writer_calls[1][0] == "revise"
          and "Embed the figure" in writer_calls[1][2][0])
    check("rendered recorded in the plan", all(j.get("rendered") for j in plan["figure_jobs"]))
    check("review.json has figure_physics_ok at the top", review.get("figure_physics_ok"), True)
    check("with its reason", "plausible" in review.get("figure_physics_reason", ""))

    a = audit._compute_paper_audit(run, route=1)
    check("H16 figure_jobs_have_out_png", (a["figure_jobs_have_out_png"], a["figure_jobs_invalid_reasons"]),
          (True, []))
    check("H15 figure_jobs_includegraphics_ok", a["figure_jobs_includegraphics_ok"], True)
    check("H28 per_case_coverage_ok", (a["per_case_coverage_ok"], a["per_case_coverage_missing"]), (True, []))
    check("H13 figure_physics_ok", a["figure_physics_ok"], True)

    # Starting from the paper as it stands: no new draft, the current paper
    # (with what it \\inputs) is what gets reviewed.
    (run / "paper" / "sections").mkdir(parents=True, exist_ok=True)
    hand = _doc([str(run / j["out_png"]) for j in plan["figure_jobs"]]).replace(
        "\\begin{document}", "\\begin{document}HAND-CORRECTED ")
    (run / "paper" / "sections" / "body.tex").write_text(hand)
    (run / "paper" / "main.tex").write_text("\\input{sections/body.tex}\n")
    writer_calls.clear()
    reviewed = []
    _FakeReviewer.review = lambda self, **kw: (reviewed.append(kw["tex_content"]) or {
        "pass": True, "score": 0.9, "recommendations": [], "regenerate_batch_figures": False})
    # the planner renumbers the cases it includes; the paper as it stands does not
    first_planner = pp.plan_paper_stage
    pp.plan_paper_stage = lambda model, text: {
        **first_planner(model, text), "case_order_for_paper": ["case_002", "case_001"],
        "case_display_labels": {"case_002": "1", "case_001": "2"}, "unified_viz_brief": "cases 1-2"}
    pp.run_unified_paper_pipeline(
        repo_root=ROOT, run_dir=run, topic="t", paper_dir=run / "paper",
        analysis_path=run / "analysis.json", manifest_path=run / "manifest.json",
        requirements_path=run / "requirements.json", lit_path=run / "lit.json",
        review_path=run / "review.json", mesh_independence_path=None, paper_template="neurips",
        model="fake", max_review_loops=2, max_viz_inner_attempts=1, verbose=False, start_from_current=True,
    )
    pp.plan_paper_stage = first_planner
    current_plan = json.loads((run / "paper_unified_plan.json").read_text())
    check("revising the current paper, each case keeps its own number",
          current_plan["case_display_labels"], {"case_002": "2", "case_001": "1"})
    check("and the figure brief says which planner number is which case",
          "case 1 = case_002" in current_plan["unified_viz_brief"]
          and "case_002 is case 2" in current_plan["unified_viz_brief"])
    check("start_from_current writes no new draft", [c[0] for c in writer_calls if c[0] == "write"], [])
    check("and reviews the paper as it stands, its \\input expanded",
          bool(reviewed) and "HAND-CORRECTED" in reviewed[0])

    # A later round that scores lower does not replace the best one.
    scores = iter([0.8, 0.5])
    _FakeReviewer.review = lambda self, **kw: {
        "pass": False, "score": next(scores), "recommendations": ["tighten"], "regenerate_batch_figures": False}
    # the later round also re-renders a figure under the same name
    fig = run / plan["figure_jobs"][0]["out_png"]
    fig_bytes = fig.read_bytes()
    batch_script = run / "paper_figs" / "paper_viz_batch.py"
    batch_script.write_text("# round 1 script\n")

    def _worse(self, tex, recs, required_figures=None, **kw):
        fig.write_bytes(b"RE-RENDERED")
        batch_script.write_text("# rewritten script\n")
        return tex.replace("HAND-CORRECTED", "WORSE-REWRITE")

    _FakeWriter.revise_paper = _worse
    pp.run_unified_paper_pipeline(
        repo_root=ROOT, run_dir=run, topic="t", paper_dir=run / "paper",
        analysis_path=run / "analysis.json", manifest_path=run / "manifest.json",
        requirements_path=run / "requirements.json", lit_path=run / "lit.json",
        review_path=run / "review.json", mesh_independence_path=None, paper_template="neurips",
        model="fake", max_review_loops=2, max_viz_inner_attempts=1, verbose=False, start_from_current=True,
    )
    kept = (run / "paper" / "main.tex").read_text()
    check("the best-scoring round is the paper kept", "HAND-CORRECTED" in kept and "WORSE-REWRITE" not in kept)
    check("and review.json reviews it", json.loads((run / "review.json").read_text())["overall_score"], 0.8)
    check("with the figures that round was reviewed with", fig.read_bytes() == fig_bytes)
    check("and the script that drew them", batch_script.read_text(), "# round 1 script\n")

    # A higher score whose figures fail the physics check does not displace
    # a round whose figures pass: the audit holds the paper to that check.
    scores = iter([0.6, 0.8])
    physics = iter([True, False])
    _FakeReviewer.review = lambda self, **kw: {
        "pass": False, "score": next(scores), "recommendations": ["tighten"], "regenerate_batch_figures": False}
    _FakeReviewer.judge_figure_physics = lambda self, image_paths, tex: {
        "figure_physics_ok": next(physics), "figure_physics_reason": "r", "figure_physics_per_figure": []}
    pp.run_unified_paper_pipeline(
        repo_root=ROOT, run_dir=run, topic="t", paper_dir=run / "paper",
        analysis_path=run / "analysis.json", manifest_path=run / "manifest.json",
        requirements_path=run / "requirements.json", lit_path=run / "lit.json",
        review_path=run / "review.json", mesh_independence_path=None, paper_template="neurips",
        model="fake", max_review_loops=2, max_viz_inner_attempts=1, verbose=False, start_from_current=True,
    )
    kept = (run / "paper" / "main.tex").read_text()
    check("a round whose figures pass outranks a higher score whose figures fail",
          ("HAND-CORRECTED" in kept, json.loads((run / "review.json").read_text())["overall_score"]), (True, 0.6))
    _FakeReviewer.judge_figure_physics = lambda self, image_paths, tex: {
        "figure_physics_ok": True, "figure_physics_reason": f"{len(image_paths)} figures plausible",
        "figure_physics_per_figure": []}

    # A study-level results file named by the manager reaches the writer.
    (run / "study_results.json").write_text(json.dumps({"Cd_continuum": 1.906, "Cd_dsmc": 1.795}))
    _FakeReviewer.review = lambda self, **kw: {"pass": True, "score": 0.9, "recommendations": [],
                                               "regenerate_batch_figures": False}
    contexts = []
    _FakeWriter.write_paper_with_literature = lambda self, required_figures=None, **kw: (
        contexts.append(kw.get("section_context", "")) or _doc(required_figures or []))
    pp.run_unified_paper_pipeline(
        repo_root=ROOT, run_dir=run, topic="t", paper_dir=run / "paper",
        analysis_path=run / "analysis.json", manifest_path=run / "manifest.json",
        requirements_path=run / "requirements.json", lit_path=run / "lit.json",
        review_path=run / "review.json", mesh_independence_path=None, paper_template="neurips",
        model="fake", max_review_loops=1, max_viz_inner_attempts=1, verbose=False,
        results_files=[run / "study_results.json"],
    )
    check("a named study results file reaches the writer",
          bool(contexts) and "Cd_dsmc" in contexts[0] and "study_results" in contexts[0])

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

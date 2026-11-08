"""
One PyVista script generates all paper PNGs; per-image VLM QA; targeted script revision.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import extract_json_object, strip_json_fences
from cfd_langgraph.viz_creator import _PAPER_PYVISTA_ONLY_SYSTEM, _images_to_blocks
from cfd_langgraph.llm.reply import reply_text

SCRIPT_GEN_SYSTEM = (
    _PAPER_PYVISTA_ONLY_SYSTEM
    + "\n\nMULTI-CASE BATCH MODE:\n"
    "- You write ONE Python script that loads **every** case listed in cases_config.json (path given below).\n"
    "- Each case has 'id' and 'path' (OpenFOAM case root). Touch or ensure <case_folder_name>.foam exists in each path.\n"
    "- Save **multiple** PNG files into the output directory given in the config.\n"
    "- REQUIRED FILE NAMES: `cfg[\"figure_jobs\"]` lists the planned figures; each has `out_png_abs`, `kind`, "
    "`case_id`, `field` and `what_to_visualize`. Save every one of them at **exactly** `out_png_abs` (do not rename, "
    "do not add prefixes). `field` is the case's own field name (e.g. U_Ar, p_Ar): read that field, never assume "
    "U or p exist. Any extra figure you add needs its own new file name.\n"
    "- Prefer **coherent styling** across cases: same colormap limits where comparable, same font sizes, same layout logic.\n"
    "- You may create **multi-panel** figures (subplots / multiple render windows saved sequentially) when it helps the paper.\n"
    "- Read config from: `cfg_path = Path(__file__).resolve().parent / \"cases_config.json\"` then `json.loads(cfg_path.read_text())`.\n"
    "- `output_dir = Path(cfg[\"output_dir\"])` — save all PNGs there.\n"
    "- Use **only** PyVista for PNG output (screenshot); no matplotlib.pyplot.savefig.\n"
    "- Release large meshes between cases if needed (`del mesh; import gc; gc.collect()`).\n"
    "\n"
    "LAYOUT CONTRACT (mandatory — same folder as this script contains `paper_fig_layout.py`):\n"
    "- At startup call `from paper_fig_layout import configure_paper_figure_theme, paper_window_size, "
    "padded_bounds_for_thin_domain, two_panel_positions` then `configure_paper_figure_theme()` once before creating plotters.\n"
    "- Use `paper_window_size()` for `Plotter(window_size=...)` or equivalent so all PNGs share publication-sized frames.\n"
    "- For **two-panel** (contour | wall-normal profile) figures, use `two_panel_positions()` for `plotter.subplot(..., position=...)` "
    "or equivalent so the profile panel stays ~44% of width — do not shrink the profile to a unreadable strip.\n"
    "- For very thin channel meshes, call `padded_bounds_for_thin_domain(mesh.bounds)` (or slice bounds) before `reset_camera` / "
    "`view_xy` so the wall-normal direction is not collapsed to a line.\n"
    "- Mesh extent, camera angle, field ranges, and slice locations remain **data-driven** from each case; only reuse the "
    "constants/helpers from `paper_fig_layout` for fonts, window size, subplot boxes, and thin-domain padding.\n"
)

SCRIPT_GEN_USER = """Topic / study:
{topic}

Unified figure brief (what the manuscript needs):
{unified_brief}

Per-case hints from planner (figure_jobs summary):
{figure_jobs_summary}

cases_config.json absolute path:
{config_path}

The script will live in the same directory as cases_config.json (paper_figs/). Write complete Python.
Output ONLY raw code, first line must be import. No markdown fences.

{reference_block}
Previous errors / per-image QA failures:
{previous_feedback}

Previous script (if any — revise minimally to fix failing outputs only when feedback targets specific files):
{previous_script}
"""

SCRIPT_REVISE_USER = """The batch script produced PNGs but some failed individual quality review.

FAILED FILES (fix only what is needed for these; keep other outputs working):
{failure_block}

STDOUT/STDERR tail from last run:
{run_log}

Full current script:
{previous_script}

Return ONLY the complete revised Python script (raw code, no fences). First line must be import.
"""

SINGLE_IMG_CHECK_SYSTEM = (
    "You judge ONE CFD figure PNG for journal use (layout/legibility only, not physics truth).\n"
    "REJECT if: essentially blank; colorbar/legend overlaps plotted data in a way that hides trends; axis or colorbar text "
    "illegibly small; 2D channel spatial panel is a thin unreadable sliver; required second panel (e.g. wall-normal profile) "
    "missing when the figure brief explicitly requires it.\n"
    "DO NOT reject solely because you describe the whole figure as upside-down, mirrored, or rotated: PyVista/Matplotlib "
    "exports are upright by construction. Ignore orientation phrasing unless text in the image is objectively unreadable "
    "(e.g. characters drawn inverted relative to the page).\n"
    "Return ONLY JSON: {\"viz_acceptable\": bool, \"reason\": \"string\"}"
)

SINGLE_IMG_CHECK_USER = (
    "Filename: {filename}\n"
    "Global figure plan summary:\n{brief}\n\n"
    "Evaluate ONLY this image. Return JSON viz_acceptable and reason."
)

ANALYSIS_USER = """You summarize these CFD paper figures for the writer agent.

TOPIC:
{topic}

ANALYSIS JSON (truncated):
{analysis_excerpt}

List of figure files (basenames):
{filenames}

For each file, give 1–2 sentences: what it shows, which experiment/case it corresponds to (use case folder id if present in filename), and any caveat.
Then 1 short paragraph on how they fit together for Results.

Return plain text (no JSON).
"""


def build_unified_brief_from_plan(plan: Dict[str, Any]) -> str:
    u = plan.get("unified_viz_brief")
    if isinstance(u, str) and u.strip():
        return u.strip()
    lines = ["Generate all paper figures in one batch, with consistent styling."]
    jobs = plan.get("figure_jobs") or []
    if isinstance(jobs, list):
        for j in jobs:
            if not isinstance(j, dict):
                continue
            cid = j.get("case_id", "")
            w = j.get("what_to_visualize", "")
            lines.append(f"- {cid}: {w}")
    return "\n".join(lines)


def write_cases_config(
    output_dir: Path,
    cases: List[Dict[str, str]],
    figure_jobs: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "cases_config.json"
    p.write_text(
        json.dumps(
            {"output_dir": str(output_dir.resolve()), "cases": cases, "figure_jobs": list(figure_jobs or [])},
            indent=2,
        ),
        encoding="utf-8",
    )
    return p.resolve()


def ensure_paper_fig_layout_module(dest_dir: Path) -> Path:
    """Copy `paper_fig_layout.py` next to the batch script so `import paper_fig_layout` works at runtime."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / "paper_fig_layout.py"
    dst = dest_dir / "paper_fig_layout.py"
    shutil.copy2(src, dst)
    return dst


def _programmatic_image_checks(image_path: Path) -> Tuple[bool, str]:
    """
    Cheap pre-VLM gates: corrupt file, tiny canvas, nearly blank image.
    Returns (ok, reason_if_not_ok).
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return True, ""

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
    except Exception as exc:
        return False, f"not a readable image ({exc})"

    if w < 480 or h < 240:
        return False, f"image too small ({w}x{h}); use paper_window_size / larger screenshot"

    arr = np.asarray(im, dtype=np.float32)
    gray = arr.mean(axis=2)
    if not np.isfinite(gray).all():
        return False, "non-finite pixels in image"

    std = float(gray.std())
    mean = float(gray.mean())
    if std < 1.2:
        return False, f"nearly blank or flat image (std={std:.3f})"
    if mean > 252.0 and std < 4.0:
        return False, "mostly white canvas with negligible ink"

    # Thin horizontal "sliver": very wide but vertical band of non-white pixels is tiny
    active = gray < (mean + 0.5 * std)
    frac_rows = float(active.any(axis=1).mean()) if h > 8 else 1.0
    if frac_rows < 0.035 and w > 2.2 * h:
        return False, "spatial panel looks like a thin horizontal sliver (check bounds / aspect)"

    return True, ""


def _run_script(script_path: Path, cwd: Path) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=900,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return -1, "", str(e)


def _check_one_image(
    *,
    image_path: Path,
    llm: Any,
    brief: str,
) -> Tuple[bool, str]:
    blocks = _images_to_blocks([image_path], max_images=1)
    if not blocks:
        return False, "Could not read image bytes"
    user = SINGLE_IMG_CHECK_USER.format(filename=image_path.name, brief=brief[:6000])
    content: List[Any] = [{"type": "text", "text": user}]
    content.extend(blocks)
    msgs = [
        SystemMessage(content=SINGLE_IMG_CHECK_SYSTEM),
        HumanMessage(content=content),
    ]
    try:
        r = llm.invoke(msgs)
        raw = reply_text(r)
    except Exception as e:
        return True, f"VLM check error, accepting: {e}"
    try:
        parsed = json.loads(extract_json_object(raw if isinstance(raw, str) else str(raw)))
        ok = bool(parsed.get("viz_acceptable", False))
        return ok, str(parsed.get("reason", ""))
    except Exception:
        return True, "Unparseable VLM response; accepting"


def run_batch_paper_viz_loop(
    *,
    model: str,
    repo_root: Path,
    paper_figs_dir: Path,
    topic: str,
    unified_brief: str,
    figure_jobs_summary: str,
    config_path: Path,
    max_inner_attempts: int = 10,
    verbose: bool = True,
    previous_script: str = "",
    extra_feedback: str = "",
    deadline: Optional[float] = None,
    required_pngs: Optional[List[Path]] = None,
) -> Tuple[str, List[Path], Dict[str, Any]]:
    """
    Returns (final_script_text, list of png paths, meta dict with attempts, failures history).
    Each outer attempt: generate OR revise script → run → per-image VLM; on failures, next iteration revises.
    No new attempt starts when another as long as the last would end past
    ``deadline`` (time.monotonic()): the figures of the last attempt are kept.
    ``required_pngs`` are the plan's out_png files: one the script did not
    write fails the attempt like a QA reject, so the next revision writes it.
    """
    llm = create_langchain_llm(model=model, temperature=0.0)
    paper_figs_dir = paper_figs_dir.resolve()
    repo_root = repo_root.resolve()
    ensure_paper_fig_layout_module(paper_figs_dir)
    script_path = paper_figs_dir / "paper_viz_batch.py"
    meta: Dict[str, Any] = {"inner_attempts": 0, "failures_log": [], "programmatic_failures": []}
    required = [Path(r).resolve() for r in (required_pngs or [])]
    last_script = previous_script.strip()
    pending_qa_block = ""  # if set, use SCRIPT_REVISE_USER
    run_log_tail = ""

    # The figures and script already here, kept until an attempt does better.
    entry_dir = paper_figs_dir / ".entry_figures"
    shutil.rmtree(entry_dir, ignore_errors=True)
    entry_dir.mkdir(parents=True)
    for old in paper_figs_dir.glob("*.png"):
        shutil.copy2(old, entry_dir / old.name)
    entry_script = script_path.read_text(encoding="utf-8") if script_path.is_file() else ""

    def _present(found) -> int:
        """How many of the required figures are among ``found`` (all PNGs when none are required)."""
        names = {f.name for f in found}
        return sum(r.name in names for r in required) if required else len(names)

    def _run_and_check(attempt: int) -> Tuple[bool, List[Path], str, str]:
        """Run the script on disk and QA its figures: (passed, pngs, feedback, log tail)."""
        for old in paper_figs_dir.glob("*.png"):
            try:
                old.unlink()
            except OSError:
                pass

        rc, out, err = _run_script(script_path, cwd=repo_root)
        log_tail = (out + "\n" + err)[-6000:]
        pngs = sorted(paper_figs_dir.glob("*.png"))
        if rc != 0 or not pngs:
            meta["failures_log"].append({"attempt": attempt, "phase": "run", "error": log_tail[:2000]})
            if verbose:
                print(f"[batch_paper_viz] attempt {attempt}: run failed rc={rc} pngs={len(pngs)}")
            return False, pngs, f"Script failed (exit {rc}) or produced no PNGs. Fix the script.\n{log_tail[:4000]}", log_tail

        failures: List[Tuple[str, str]] = []
        # A script that named its PNGs its own way left every planned out_png
        # missing, and the audit could not match one figure to its job
        # (shelf_demo_hystrath_20261022_1030 wrote 06_01_case_001_... names).
        for r in required:
            if not r.is_file():
                failures.append((r.name, f"[required] not written: save this figure at exactly {r}"))
        for p in pngs:
            prog_ok, prog_reason = _programmatic_image_checks(p)
            if not prog_ok:
                msg = f"[programmatic] {prog_reason}"
                failures.append((p.name, msg))
                meta["programmatic_failures"].append({"file": p.name, "reason": msg})
                if verbose:
                    print(f"[batch_paper_viz] QA reject: {p.name} — {msg[:120]}")
                continue
            ok, reason = _check_one_image(image_path=p, llm=llm, brief=unified_brief)
            if not ok:
                failures.append((p.name, reason))
                if verbose:
                    print(f"[batch_paper_viz] QA reject: {p.name} — {reason[:120]}")

        if not failures:
            if verbose:
                print(f"[batch_paper_viz] all {len(pngs)} images passed QA (attempt {attempt})")
            meta["missing_required"] = []
            return True, pngs, "", log_tail

        meta["failures_log"].append({"attempt": attempt, "phase": "qa", "failures": failures})
        if verbose:
            print(f"[batch_paper_viz] attempt {attempt}: {len(failures)} image(s) failed QA; will revise script")
        return False, pngs, "\n".join(f"- {name}: {rs}" for name, rs in failures), log_tail

    # A script already on disk with no script handed in -- one the manager
    # repaired between write_paper calls -- runs first as it is; only when it
    # fails does an attempt rewrite it, starting from it.
    if not last_script and entry_script.strip():
        last_script = entry_script.strip()
        passed, pngs, pending_qa_block, run_log_tail = _run_and_check(0)
        if passed:
            return entry_script, pngs, meta

    last_took = 0.0
    for attempt in range(1, max(1, max_inner_attempts) + 1):
        # A paper run killed at its time limit mid-attempt had deleted the
        # last attempt's figures and saved nothing (shelf_demo_hystrath_20261011_1602).
        if deadline is not None and attempt > 1 and time.monotonic() + last_took > deadline:
            meta["stopped_for_time"] = True
            if verbose:
                print(f"[batch_paper_viz] stopping before attempt {attempt}: not enough time left")
            break
        started = time.monotonic()
        meta["inner_attempts"] = attempt

        if pending_qa_block:
            user = SCRIPT_REVISE_USER.format(
                failure_block=pending_qa_block[:8000],
                run_log=run_log_tail,
                previous_script=last_script[:100_000] if last_script else "",
            )
        else:
            ref_block = ""
            if last_script:
                ref_block = "Refine or extend the previous script as needed; preserve working parts.\n\n"
            user = SCRIPT_GEN_USER.format(
                topic=topic[:4000],
                unified_brief=unified_brief[:12000],
                figure_jobs_summary=figure_jobs_summary[:8000],
                config_path=str(config_path),
                reference_block=ref_block,
                previous_feedback=extra_feedback or "(none)",
                previous_script=last_script or "(none - generate from scratch)",
            )

        msgs = [SystemMessage(content=SCRIPT_GEN_SYSTEM), HumanMessage(content=user)]
        try:
            resp = llm.invoke(msgs)
            script_text = reply_text(resp)
        except Exception as e:
            meta["failures_log"].append({"attempt": attempt, "phase": "llm", "error": str(e)})
            if verbose:
                print(f"[batch_paper_viz] attempt {attempt}: LLM error {e}")
            last_took = time.monotonic() - started
            time.sleep(min(2.0, 0.25 * (1 + random.random())))
            continue

        script_text = strip_json_fences(script_text if isinstance(script_text, str) else str(script_text))
        lines = script_text.lstrip().splitlines()
        if lines and lines[0].strip().lower() in {"python", "bash", "sh"}:
            script_text = "\n".join(lines[1:])
        try:
            from cfd_langgraph.foam_load import install_beside

            install_beside(script_path.parent)
        except Exception:  # noqa: BLE001 -- the script can still try its own loader
            pass
        script_path.write_text(script_text, encoding="utf-8")
        last_script = script_text
        passed, pngs, pending_qa_block, run_log_tail = _run_and_check(attempt)
        if passed:
            return script_text, pngs, meta
        last_took = time.monotonic() - started
        time.sleep(min(2.0, 0.25 * (1 + random.random())))

    # Every attempt failed. A re-render that wrote fewer of the planned figures
    # than the round before had deleted that round's figures, and the paper
    # then compiled against none of its 20 (nn_in_solver_demo_20261022_1129):
    # the figures and script this call started with come back instead.
    if _present(entry_dir.glob("*.png")) > _present(paper_figs_dir.glob("*.png")):
        for old in paper_figs_dir.glob("*.png"):
            old.unlink()
        for kept in entry_dir.glob("*.png"):
            shutil.copy2(kept, paper_figs_dir / kept.name)
        if entry_script:
            script_path.write_text(entry_script, encoding="utf-8")
            last_script = entry_script
        meta["restored_entry_figures"] = True
        if verbose:
            print("[batch_paper_viz] no attempt did better than the figures already there; kept those")
    pngs = sorted(paper_figs_dir.glob("*.png"))
    meta["missing_required"] = [str(r) for r in required if not r.is_file()]
    return last_script, pngs, meta


def analyze_figures_for_paper(
    *,
    model: str,
    topic: str,
    analysis: Dict[str, Any],
    image_paths: List[Path],
    verbose: bool = False,
) -> str:
    llm = create_langchain_llm(model=model, temperature=0.0)
    names = [p.name for p in image_paths[:48]]
    excerpt = json.dumps(analysis, indent=2)[:25_000]
    user = ANALYSIS_USER.format(
        topic=topic[:3000],
        analysis_excerpt=excerpt,
        filenames=json.dumps(names, indent=2),
    )
    try:
        out = llm.invoke([HumanMessage(content=user)])
        return reply_text(out)[:24_000]
    except Exception as e:
        if verbose:
            print(f"[batch_paper_viz] image analysis LLM failed: {e}")
        return ""


def reviewer_requests_script_refresh(
    recommendations: List[str],
    needs_viz: bool,
    *,
    regenerate_batch_figures: Optional[bool] = None,
) -> bool:
    """
    Decide whether to re-run the batch PyVista script.

    If the reviewer sets ``regenerate_batch_figures`` explicitly, that dominates
    keyword heuristics (except ``needs_additional_visualization``, which always
    forces a refresh).
    """
    if needs_viz:
        return True
    if regenerate_batch_figures is True:
        return True
    if regenerate_batch_figures is False:
        return False
    keys = (
        "figure",
        "plot",
        "image",
        "png",
        "contour",
        "colorbar",
        "pyvista",
        "visualization",
        "graphic",
    )
    for r in recommendations:
        t = str(r).lower()
        if any(k in t for k in keys):
            return True
    return False

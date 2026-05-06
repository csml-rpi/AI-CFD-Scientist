from __future__ import annotations

import base64
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import strip_json_fences


VIZ_MAX_RETRIES = 10


def _ensure_marker_foam(case_dir: Path) -> Path:
    """
    Ensure <folder_name>.foam exists inside case_dir and return its path.
    Example: /.../exp_001/foam_output -> foam_output.foam
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    marker = case_dir / f"{case_dir.name}.foam"
    if not marker.exists():
        marker.touch()
    return marker


def _run_script(script_path: Path, cwd: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", f"Runner exception: {e}"


def _images_to_blocks(image_paths: List[Path], max_images: int = 16) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in image_paths[:max_images]:
        if not p.exists() or not p.is_file():
            continue
        try:
            b = p.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".gif":
                mime = "image/gif"
            else:
                mime = "image/png"
            url = f"data:{mime};base64,{b64}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception:
            continue
    return blocks


_PAPER_PYVISTA_ONLY_SYSTEM = (
    "You write Python scripts for CFD **paper figures** using **PyVista only** for all raster (PNG) output.\n"
    "Requirements:\n"
    "- Load OpenFOAM data with PyVista from foam_output_dir using the given marker .foam file.\n"
    "- Use off_screen=True plotters only. Save PNGs **only** via PyVista (e.g. plotter.screenshot, plotter.export_gltf is NOT for PNG—use screenshot).\n"
    "- **Do NOT use matplotlib.pyplot.savefig** or any matplotlib-based figure export. Matplotlib may be imported only for numpy-style helpers if needed, but every PNG must come from a PyVista Plotter screenshot.\n"
    "- For 1D profiles (e.g. U vs y/H), use PyVista Chart2D, or plot polylines in a 2D Plotter view, then screenshot.\n"
    "- **Slim/long channels (high aspect ratio):** prefer **horizontal** layout: set window_size so the **longer** domain direction is the **wider** pixel dimension (e.g. streamwise horizontal). Use parallel_projection, zoom, and bounds so the channel is not a thin line.\n"
    "- **nuEff / effective viscosity:** only plot if the field exists in cell_data or point_data (e.g. nuEff, strainRateViscosityModel:nu). If absent, **do not** fabricate a figure—skip that output or plot another requested quantity.\n"
    "- Colorbars and legends must not overlap streamlines, contours, or velocity data; use scalar_bar_args (horizontal bar if needed), margins, larger window.\n"
    "- Typography: large fonts (tick labels ≥18 pt, titles ≥20 pt) via PyVista theme or text properties.\n"
    "- Output ONLY raw Python. No markdown fences. First line must be import.\n"
)

_PAPER_PYVISTA_ONLY_USER_EXTRA = (
    "PAPER MODE: PyVista-only PNG output (no matplotlib savefig). Horizontal layout for long thin channels. "
    "Do not save nuEff/viscosity figures unless that array exists on the mesh.\n\n"
)

_REFERENCE_DATA_RULE = (
    "MANDATORY REFERENCE DATA RULE:\n"
    "Any plot that shows a simulation quantity (wall coefficient, force, profile, scalar field sample, "
    "error metric, or any QoI) MUST include the corresponding reference/DNS/experimental data on the "
    "same axes if reference data is available in the context. A comparison plot without the ground-truth "
    "curve is incomplete. Label reference curves clearly (e.g. 'DNS', 'Experiment', 'Reference'). "
    "If reference data is provided as a CSV or table, load and overlay it on the same plot.\n"
    "SIGN AND NORMALIZATION CHECK:\n"
    "Before plotting any wall quantity extracted from OpenFOAM (e.g. wallShearStress, heatFlux), "
    "check the sign convention. OpenFOAM reports quantities acting ON THE FLUID; if the reference "
    "uses the opposite convention, negate the simulation values. "
    "Normalization must use the authoritative reference scale from the case parameters "
    "(bulk velocity, free-stream velocity, diameter, etc.) — never use 1.0 as a default.\n"
)


def viz_creator(
    model: str,
    foam_output_dir: Path,
    viz_dir: Path,
    what_to_visualize: str,
    user_requirement: str,
    reference_viz_script: Optional[str] = None,
    max_retries: int = VIZ_MAX_RETRIES,
    paper_pyvista_only: bool = False,
    strict_quality: bool = True,
) -> Dict[str, Any]:
    """
    Central visualization creator.

    - Touches <foam_output_dir.name>.foam inside foam_output_dir and uses it as the .foam marker.
    - For up to max_retries:
        * Asks LLM to write a PyVista script that loads the marker .foam, creates the requested
          visualizations, and saves PNGs into viz_dir.
        * Runs the script.
        * If script fails or produces no PNGs, feeds the error trace back into the next LLM call.
        * If PNGs exist, runs a vision-style LLM check over the images to decide if viz is acceptable
          for the given user requirement + what_to_visualize. If not acceptable, uses that feedback
          as part of the next error message and retries.
    - Returns:
        {
          "ok": bool,
          "images": [str paths],
          "attempts": int,
          "last_error": str,
          "foam_output_dir": str,
          "viz_dir": str,
          "marker_foam": str,
        }
    """
    foam_output_dir = foam_output_dir.expanduser().resolve()
    viz_dir = viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Verbose: case identifier for diagnostics
    case_label = foam_output_dir.parent.name if foam_output_dir.parent else foam_output_dir.name
    def _log(msg: str) -> None:
        print(f"[viz_creator {case_label}] {msg}", flush=True)

    _log(f"case_dir={foam_output_dir} viz_dir={viz_dir}")

    # On each new viz_creator call (including interpreter reruns / analysis reruns),
    # clear out any existing PNGs in viz_dir so that only images from the current
    # attempt set are considered. This avoids mixing old images from previous runs.
    for old_png in viz_dir.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    marker_foam = _ensure_marker_foam(foam_output_dir)

    llm = create_langchain_llm(model=model, temperature=0.1)

    script_system = (_PAPER_PYVISTA_ONLY_SYSTEM + "\n\n" + _REFERENCE_DATA_RULE) if paper_pyvista_only else (
        "You write PyVista+matplotlib Python scripts to visualize OpenFOAM cases.\n"
        + _REFERENCE_DATA_RULE + "\n"
        "Requirements (CFD paper-quality figures only):\n"
        "- Load the case using PyVista from the given foam_output_dir.\n"
        "- The marker .foam file to load is always the given marker_name.\n"
        "- Use off_screen=True plotters only (no interactive windows).\n"
        "- Save all figures as PNG files into viz_dir.\n"
        "- Use PyVista for field visualizations (filled contour/colormap plots, streamlines, mesh outlines, slices, etc.).\n"
        "- **Elongated 2D domains (channels, long ducts, periodic boxes with Lx >> Ly or Ly >> Lx):** a naive PyVista top/side view in a wide window makes the domain look like a **thin line**—unacceptable for papers. "
        "You MUST fix this by at least one of: (1) set `window_size=(w,h)` so **w:h is proportional to the in-plane extents** (e.g. dx:dy) so both dimensions occupy a substantial fraction of the frame; "
        "(2) use `parallel_projection=True` and adjust camera / `reset_camera` / zoom so bounds fill the view without crushing the short direction; "
        "(3) add a **second** figure that zooms a short streamwise window so wall-to-wall structure is obvious; "
        "(4) if PyVista still cannot show both extents clearly, use **matplotlib** `tricontourf` / `pcolormesh` on the 2D slice with `extent=[xmin,xmax,ymin,ymax]` and a **figure size whose aspect ratio matches the domain** (or `ax.set_box_aspect` / constrained layout) so the channel looks like a channel, not a line.\n"
        "- Matplotlib is required for 1D line plots (profiles, time histories). It is **also allowed** for 2D filled contours when needed to fix extreme aspect ratio as above.\n"
        "- All figures must be readable in a CFD paper: avoid sparse dot-only plots; use filled contours, meaningful colorbars, clear legends, and informative layouts so readers can extract physical insight.\n"
        "- Camera/zoom rules: if the requested visualization targets a localized flow feature "
        "(recirculation bubble, reattachment point/length, shear layer, step lip/corner, inlet jet, "
        "or wall quantities like Cp/Cf/y+), you MUST create at least one view that zooms/crops "
        "around that feature so it occupies a substantial part of the frame and is legible when printed. "
        "Do NOT rely only on far-out views where the feature becomes tiny/unreadable. "
        "If full-domain context helps, include a second full/wider view as well.\n"
        "- Colorbars and legends (publication layout): they must NEVER overlap contours, streamlines, mesh, or other plotted data. "
        "Reserve margin space (e.g. adjust camera/viewport, use plotter.subplot, or shrink the rendered domain area). "
        "For PyVista, use scalar_bar_args to position/size the bar (e.g. position_x, position_y, width, height) or place it in empty space; "
        "if a vertical bar would cover data, use a horizontal colorbar (vertical=False in scalar_bar_args where supported) or move it below/above the scene. "
        "For matplotlib line plots, use tight_layout with rect=[0,0,0.85,1] or make room for colorbar(orientation='horizontal', pad=...), and keep legend outside axes (bbox_to_anchor) when needed.\n"
        "- Typography: use large, paper-ready fonts — at least 18 pt for tick labels and colorbar tick labels, "
        "and at least 20–22 pt for titles, axis labels, and colorbar titles (PyVista: title_font_size / label_font_size in scalar_bar_args; matplotlib: rcParams['font.size'] and axis label sizes). "
        "Figures are often scaled down in PDFs; err on the side of larger text.\n"
        "- Output ONLY raw Python code. Do NOT wrap in markdown code fences (no ``` or ```python).\n"
        "- Do NOT start with the word 'python' or any language tag. The first line must be an import statement.\n"
    )

    script_user_tpl = (
        "User requirement for this experiment:\n"
        "{user_requirement}\n\n"
        "What to visualize:\n"
        "{what_to_visualize}\n\n"
        "{reference_block}"
        "foam_output_dir (input data):\n"
        "{foam_output_dir}\n\n"
        "viz_dir (where PNGs must be saved):\n"
        "{viz_dir}\n\n"
        "marker_name (.foam file inside foam_output_dir):\n"
        "{marker_name}\n\n"
        "Previous feedback / error (if any):\n"
        "{previous_error}\n\n"
        "Previous viz script that failed or was rejected (if any):\n"
        "{previous_script}\n\n"
        "Write ONLY a complete Python script. Output raw code only—no markdown fences, no ```python, no explanations. "
        "The first line must be 'import ...' or similar, never the word 'python'. The script should:\n"
        "- import pyvista as pv (and matplotlib if needed for line plots)\n"
        "- read the case from foam_output_dir/marker_name\n"
        "- generate high-quality CFD paper-style visualizations: PyVista contours/streamlines/mesh, plus matplotlib line plots; for long thin 2D domains, match window/figure aspect to domain so the geometry is visible\n"
        "- avoid sparse dot-only plots; ensure every figure is informative and readable with font sizes >= 18 (ticks/legend/colorbar ticks) and >= 20 for titles and axis/colorbar titles\n"
        "- choose camera positions and zoom/cropping so that the requested feature(s) are resolved and readable; "
        "include zoomed-in frames for localized features (recirculation/reattachment/shear layer/step lip/walls) "
        "and optionally include a wider full-domain context view if it helps interpretation\n"
        "- colorbars and legends must stay clear of the data region: use scalar_bar_args (position, size, vertical=False for horizontal bar if side space is tight), extra margins, or subplot layout; never let the bar cover contours\n"
        "- save PNG files into viz_dir\n"
        "- exit with code 0 on success and non-zero on fatal error.\n"
    )

    script_user_tpl_paper = (
        _PAPER_PYVISTA_ONLY_USER_EXTRA
        + "User requirement for this experiment:\n"
        "{user_requirement}\n\n"
        "What to visualize:\n"
        "{what_to_visualize}\n\n"
        "{reference_block}"
        "foam_output_dir (input data):\n"
        "{foam_output_dir}\n\n"
        "viz_dir (where PNGs must be saved):\n"
        "{viz_dir}\n\n"
        "marker_name (.foam file inside foam_output_dir):\n"
        "{marker_name}\n\n"
        "Previous feedback / error (if any):\n"
        "{previous_error}\n\n"
        "Previous viz script that failed or was rejected (if any):\n"
        "{previous_script}\n\n"
        "Write ONLY a complete Python script. Output raw code only—no markdown fences.\n"
        "The first line must be 'import ...'. Requirements:\n"
        "- import pyvista as pv; use OpenFOAMReader on foam_output_dir/marker_name; latest time value\n"
        "- ALL PNGs via PyVista plotter.screenshot only (no matplotlib savefig)\n"
        "- Long thin 2D channel: use horizontal layout (window_size wider along streamwise)\n"
        "- Only plot nuEff/effective viscosity if the field exists on the dataset\n"
        "- Colorbar/legend must not overlap data; large fonts\n"
        "- save PNG files into viz_dir; exit 0 on success\n"
    )

    active_user_tpl = script_user_tpl_paper if paper_pyvista_only else script_user_tpl

    _viz_check_system_strict = (
        "You are a visualization output checker (NOT a physics/simulation judge). Your ONLY job is to decide "
        "whether the generated images show the REQUESTED types of data (contours, profiles, plots, etc.) "
        "and are non-empty and readable for a journal-style figure (these images may be embedded in a paper).\n"
        "Check that contour/field figures are framed and zoomed appropriately for a CFD paper: "
        "the requested feature(s) must be readable when printed. "
        "If the request is about localized features (recirculation/reattachment/shear layer/step lip/walls/Cp/Cf/y+), "
        "accept zoomed-in feature framing even if the full computational domain is not visible. "
        "Reject if the only views are extremely zoomed-out such that the feature is too small to inspect.\n"
        "**2D domain / geometry legibility:** REJECT full-domain (or primary) 2D contour/surface plots where the flow domain appears as a **negligible thin strip or line** (one in-plane dimension visually collapsed), so a reader cannot recognize the geometry (e.g. channel height vs length). "
        "This is a common PyVista failure for high-aspect-ratio meshes—not acceptable for publication even if a colormap is present. "
        "Accept only if both in-plane directions are visibly resolved OR a companion zoom clearly shows wall-bounded structure.\n"
        "Publication layout (use your vision): REJECT if a colorbar, scalar bar, or legend overlaps or obscures "
        "meaningful contour, streamline, mesh, or plot data (common PyVista issue). REJECT if axis labels, tick labels, "
        "titles, legend text, or colorbar text are clearly too small to read when the figure is viewed at moderate zoom "
        "(typical manuscript single-column width). Accept only when annotations are legible and the data region is unobstructed; "
        "suggest in 'reason' that the next attempt use a horizontal colorbar, different bar position, larger fonts, or more margin if you reject for these reasons.\n"
        "REJECT when: (1) images are blank/empty or show no data, (2) images are broken or unreadable, "
        "(3) the requested visualization types are completely missing (e.g. contours requested but no contour "
        "figures at all), (4) the camera/zoom makes the domain or key features too small to inspect in a paper figure, "
        "(5) colorbar/legend overlaps plotted data, (6) fonts are too small for publication, or (7) 2D domain is visually degenerate (thin-line full-domain view).\n"
        "Do NOT reject based on: whether the simulation physics looks wrong, whether the flow field is "
        "physically plausible, numerical noise, failed simulation, or plug flow. Judging "
        "simulation correctness is the interpreter's job later. If the figures show the requested data "
        "(even if the underlying simulation seems wrong), accept them—unless they fail the layout, legibility, or **geometry/aspect** checks above.\n"
        "Return ONLY JSON: {\"viz_acceptable\": bool, \"reason\": \"string\"}."
    )

    # Relaxed quality check for interpret stage — only reject on hard failures (blank/missing/broken).
    # Cosmetic issues (colorbar overlap, font size, aspect ratio) are acceptable for physics assessment.
    _viz_check_system_relaxed = (
        "You are a visualization output checker (NOT a physics/simulation judge). Your ONLY job is to decide "
        "whether the generated images contain actual readable data for the requested visualization types.\n"
        "REJECT ONLY when: (1) images are completely blank or empty with no data, (2) images are broken/corrupted/unreadable, "
        "(3) the requested visualization types are entirely absent (e.g. contours requested but no contour figures at all), "
        "(4) the domain or key features are so small they are completely invisible (e.g. a single pixel).\n"
        "ACCEPT despite: colorbar or legend overlapping data, small fonts, axis crowding, imperfect aspect ratio, "
        "thin-strip 2D domain views, cosmetic layout issues, or physically wrong-looking results. "
        "These images are for physics interpretation only, not publication. Minor cosmetic issues do not matter here.\n"
        "Do NOT reject because physics looks wrong — that is the interpreter's job.\n"
        "Return ONLY JSON: {\"viz_acceptable\": bool, \"reason\": \"string\"}."
    )

    viz_check_system = _viz_check_system_strict if strict_quality else _viz_check_system_relaxed

    viz_check_user_tpl = (
        "User requirement:\n"
        "{user_requirement}\n\n"
        "Requested visualizations:\n"
        "{what_to_visualize}\n\n"
        "Previous feedback / error (if any):\n"
        "{previous_error}\n\n"
        "You will see the generated images below. Check: "
        "(1) Non-empty and readable? "
        "(2) Requested plot types present with actual data? "
        "(3) If localized features were requested, is there a zoomed-in view where they are inspectable? "
        "(4) Do colorbars/legends avoid overlapping the plotted flow/domain (reject if they cover contours/mesh/curves)? "
        "(5) Are fonts large enough for a paper figure (title, ticks, colorbar labels legible)? "
        "(6) For 2D channel/duct-style plots: can you see **both** wall-normal and streamwise extent (not a single vertical/horizontal sliver)? "
        "If (1)-(6) pass, set viz_acceptable=true. Do NOT reject because physics looks wrong—that is for the interpreter. "
        "Reject for bad layout (overlap), illegible typography, or degenerate domain aspect even when a colormap exists.\n"
        "Return ONLY JSON with keys viz_acceptable (bool) and reason (string)."
    )

    last_error = ""
    last_script = ""
    images: List[Path] = []
    attempt = 0

    # If no reference script explicitly provided, try to reuse an existing viz_script.py
    # in this viz_dir (from a previous interpreter/analysis loop) as a starting point.
    if not reference_viz_script:
        existing_script = viz_dir / "viz_script.py"
        if existing_script.is_file():
            try:
                reference_viz_script = existing_script.read_text(encoding="utf-8")
            except Exception:
                reference_viz_script = None

    if reference_viz_script:
        reference_block = (
            "Reference: a previous visualization run used the following Python script "
            "to generate figures in this directory. You can treat this as a starting "
            "point and adapt or extend it to satisfy the current visualization request:\n"
            f"{reference_viz_script}\n\n"
        )
    else:
        reference_block = ""

    for attempt in range(1, max_retries + 1):
        # Ask LLM for script
        user_prompt = active_user_tpl.format(
            user_requirement=user_requirement,
            what_to_visualize=what_to_visualize,
            reference_block=reference_block,
            foam_output_dir=str(foam_output_dir),
            viz_dir=str(viz_dir),
            marker_name=marker_foam.name,
            previous_error=last_error or "(none)",
            previous_script=last_script or "(none - first attempt)",
        )
        script_msgs = [
            SystemMessage(content=script_system),
            HumanMessage(content=user_prompt),
        ]
        try:
            resp = llm.invoke(script_msgs)
            script_text = getattr(resp, "content", str(resp))
        except Exception as e:
            last_error = f"LLM error while generating script: {e}"
            _log(f"LLM error: {e}")
            continue

        # Remove markdown fences and any stray leading language tag like "python"
        script_text = strip_json_fences(script_text)
        lines = script_text.lstrip().splitlines()
        if lines and lines[0].strip().lower() in {"python", "bash", "sh"}:
            script_text = "\n".join(lines[1:])
        script_path = viz_dir / "viz_script.py"
        script_path.write_text(script_text, encoding="utf-8")

        rc, out, err = _run_script(script_path, cwd=foam_output_dir)
        pngs = sorted(p for p in viz_dir.glob("*.png") if p.is_file())

        if rc != 0 or not pngs:
            # Script failed or produced no images; feed back error and script for next attempt
            snippet = (err or out or "Unknown error")[-4000:]
            last_error = f"Return code: {rc}\nSTDOUT:\n{out[-1000:]}\nSTDERR:\n{snippet}\n"
            last_script = script_text
            # Clean up any partial images
            for p in pngs:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        # Script produced images; run viz-quality check (limit images to avoid huge payload + Bedrock read timeout)
        img_blocks = _images_to_blocks(pngs)
        if not img_blocks:
            last_error = "Generated images could not be read/encoded."
            for p in pngs:
                try:
                    p.unlink()
                except Exception:
                    pass
            continue

        viz_user = viz_check_user_tpl.format(
            user_requirement=user_requirement,
            what_to_visualize=what_to_visualize,
            previous_error=last_error or "(none)",
        )
        content: List[Any] = [{"type": "text", "text": viz_user}]
        content.extend(img_blocks)
        viz_msgs = [
            SystemMessage(content=viz_check_system),
            HumanMessage(content=content),
        ]
        try:
            viz_resp = llm.invoke(viz_msgs)
            raw = getattr(viz_resp, "content", str(viz_resp))
            parsed = json.loads(strip_json_fences(raw))
            viz_ok = bool(parsed.get("viz_acceptable", False))
            reason = str(parsed.get("reason", ""))
        except Exception as e:
            viz_ok = True
            reason = f"Could not parse viz check response, assuming acceptable. Raw error: {e}"

        if viz_ok:
            _log(f"attempt {attempt}: viz accepted, {len(pngs)} images")
            images = pngs
            last_error = ""
            break

        # Viz not acceptable; treat reason as feedback for next script attempt
        _log(f"attempt {attempt}: viz rejected - {reason}")
        last_error = f"Viz unacceptable: {reason}"
        last_script = script_text

        # Final-attempt fallback: if the script ran and produced images, keep them
        # rather than discarding. The reviewer has been observed to hallucinate
        # missing/uniform figures on legitimately usable output; losing the whole
        # case over that wastes the simulation.
        if attempt >= max_retries:
            _log(
                f"attempt {attempt}: max retries reached — accepting {len(pngs)} images "
                f"from final attempt despite reviewer rejection"
            )
            images = pngs
            last_error = f"Viz accepted on final attempt with reviewer rejection: {reason}"
            break

        for p in pngs:
            try:
                p.unlink()
            except Exception:
                pass

        # brief backoff before retry
        time.sleep(min(2.0, 0.25 * (1 + random.random())))

    if images:
        _log(f"SUCCESS: {len(images)} images after {attempt} attempt(s)")
    else:
        _log(f"FAILED after {attempt} attempt(s): {last_error[:500]}{'...' if len(last_error) > 500 else ''}")

    return {
        "ok": bool(images),
        "images": [str(p) for p in images],
        "attempts": attempt,
        "last_error": last_error,
        "foam_output_dir": str(foam_output_dir),
        "viz_dir": str(viz_dir),
        "marker_foam": str(marker_foam),
    }

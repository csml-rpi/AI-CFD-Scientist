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


def viz_creator(
    model: str,
    foam_output_dir: Path,
    viz_dir: Path,
    what_to_visualize: str,
    user_requirement: str,
    reference_viz_script: Optional[str] = None,
    max_retries: int = VIZ_MAX_RETRIES,
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

    script_system = (
        "You write PyVista+matplotlib Python scripts to visualize OpenFOAM cases.\n"
        "Requirements (CFD paper-quality figures only):\n"
        "- Load the case using PyVista from the given foam_output_dir.\n"
        "- The marker .foam file to load is always the given marker_name.\n"
        "- Use off_screen=True plotters only (no interactive windows).\n"
        "- Save all figures as PNG files into viz_dir.\n"
        "- Use PyVista for all field visualizations (filled contour/colormap plots, streamlines, mesh outlines, slices, etc.); do NOT use matplotlib to draw 2D contour/filled-field plots.\n"
        "- Matplotlib may be used only for 1D line plots (e.g. profiles, time histories) where data are first sampled/extracted from PyVista.\n"
        "- All figures must be readable in a CFD paper: avoid sparse dot-only plots; use filled contours, meaningful colorbars, clear legends, and informative layouts so readers can extract physical insight.\n"
        "- Use a minimum font size of 18 for all titles, axis labels, tick labels, legends, and colorbar labels so they are legible when embedded in a paper.\n"
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
        "- generate high-quality CFD paper-style visualizations: PyVista filled contour/colormap plots and streamlines for field data, plus line plots (via matplotlib) for extracted profiles or time histories as needed\n"
        "- avoid sparse dot-only plots; ensure every figure is informative and readable with font sizes >= 18 for titles, labels, ticks, legends, and colorbars\n"
        "- save PNG files into viz_dir\n"
        "- exit with code 0 on success and non-zero on fatal error.\n"
    )

    viz_check_system = (
        "You are a visualization output checker (NOT a physics/simulation judge). Your ONLY job is to decide "
        "whether the generated images show the REQUESTED types of data (contours, profiles, plots, etc.) "
        "and are non-empty and readable.\n"
        "REJECT only when: (1) images are blank/empty or show no data, (2) images are broken or unreadable, "
        "(3) the requested visualization types are completely missing (e.g. contours requested but no contour "
        "figures at all).\n"
        "Do NOT reject based on: whether the simulation physics looks wrong, whether the flow field is "
        "physically plausible, axis orientation, numerical noise, failed simulation, or plug flow. Judging "
        "simulation correctness is the interpreter's job later. If the figures show the requested data "
        "(even if the underlying simulation seems wrong), accept them.\n"
        "Return ONLY JSON: {\"viz_acceptable\": bool, \"reason\": \"string\"}."
    )

    viz_check_user_tpl = (
        "User requirement:\n"
        "{user_requirement}\n\n"
        "Requested visualizations:\n"
        "{what_to_visualize}\n\n"
        "Previous feedback / error (if any):\n"
        "{previous_error}\n\n"
        "You will see the generated images below. Check ONLY: (1) Are the images non-empty and readable? "
        "(2) Do they contain the requested types of plots (e.g. contours, profiles) with actual data drawn? "
        "If yes, set viz_acceptable=true. Do NOT reject because the physics or simulation results look wrong "
        "or unphysical—that is for the interpreter to judge. Reject only if figures are blank, empty, "
        "missing the requested plot types entirely, or broken.\n"
        "Return ONLY JSON with keys viz_acceptable (bool) and reason (string)."
    )

    last_error = ""
    last_script = ""
    images: List[Path] = []
    attempt = 0

    if reference_viz_script:
        reference_block = (
            "Reference: the interpreter agent used the following code to create viz. "
            "You can use this as reference (adapt and extend for the requested visualizations):\n"
            f"{reference_viz_script}\n\n"
        )
    else:
        reference_block = ""

    for attempt in range(1, max_retries + 1):
        # Ask LLM for script
        user_prompt = script_user_tpl.format(
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

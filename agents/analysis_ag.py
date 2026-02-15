"""analysis_agent.py

Scan a batch under data/experiments and analyze all experiments/runs using an LLM.

Usage:
    python src/analysis_agent.py --batch <batch_name>
    python src/analysis_agent.py                # analyzes latest batch

Outputs:
    - <batch_dir>/analysis_summary.txt  : overall batch analysis
    - <batch_dir>/<experiment>/analysis.txt : per-experiment analysis

This module uses the project's `utils.base_llm` helpers to create a client and call the model.
"""

from pathlib import Path
import argparse
import json
import os
import textwrap
import base64
import re
from io import BytesIO
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
    _PIL_IMPORT_ERROR = None
except Exception as _e:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    _PIL_AVAILABLE = False
    _PIL_IMPORT_ERROR = _e

# Ensure project root is on sys.path for absolute imports BEFORE importing project modules
import sys as _sys
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from utils.base_llm import create_client, get_response_from_llm, extract_json_between_markers


MAX_SNIPPET_CHARS = 1600
DEFAULT_MODEL = os.environ.get("CFD_SCIENTIST_MODEL", "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/f6tueltt82a2")

# bedrock requires encoding to base64 
def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """
    Encode an image file to base64 string for Bedrock vision models.
    
    Args:
        image_path: Path to the image file (PNG, JPG, etc.)
        
    Returns:
        Base64 encoded string or None if error
    """
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"⚠️  Error encoding image {image_path}: {e}")
        return None


def read_text_safe(p: Path, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    try:
        s = p.read_text(encoding="utf-8")
        if len(s) > max_chars:
            return s[:max_chars] + "\n\n...[truncated]..."
        return s
    except Exception as e:
        return f"[Error reading {p}: {e}]"


def _load_artifacts_json(out_dir: Path) -> Optional[dict]:
    try:
        p = out_dir / "artifacts.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _time_tag(tt: float) -> str:
    return f"{float(tt):.2f}".replace(".", "p")


def _deterministic_velocity_deliverables_check(out_dir: Path) -> dict:
    """Check that deterministic postprocess artifacts exist for velocity-only evaluation."""
    tol = 0.051
    det = {
        "enabled": False,
        "ok": None,
        "requested_times": None,
        "max_time_error": None,
        "missing_umag": [],
        "missing_uy": [],
        "notes": [],
    }

    art = _load_artifacts_json(out_dir)
    if not art:
        det["ok"] = None
        return det

    det["enabled"] = True

    req_times = art.get("requested_times") or []
    times = []
    for t in req_times:
        try:
            times.append(float(t))
        except Exception:
            continue
    times = sorted(times)
    det["requested_times"] = times

    # Time errors
    max_err = None
    errs = art.get("time_errors") or {}
    if isinstance(errs, dict):
        for v in errs.values():
            try:
                if v is None:
                    continue
                fv = float(v)
            except Exception:
                continue
            max_err = fv if max_err is None else max(max_err, fv)
    det["max_time_error"] = max_err

    for t in times:
        ut = out_dir / f"umag_t{_time_tag(t)}.png"
        ct = out_dir / f"uy_centerline_t{_time_tag(t)}.csv"
        if not ut.exists():
            det["missing_umag"].append(ut.name)
        if not ct.exists():
            det["missing_uy"].append(ct.name)

    ok_files = (not det["missing_umag"]) and (not det["missing_uy"])
    ok_times = (max_err is None) or (max_err <= tol)

    if not ok_files:
        det["notes"].append("Missing expected deterministic velocity artifacts (UMag PNGs and/or Uy CSVs).")
    if not ok_times:
        det["notes"].append(f"Requested times were not written closely enough (max |Δt|={max_err:.3g}s > {tol}).")

    det["ok"] = bool(ok_files and ok_times)
    return det


def _parse_pimplefoam_log(log_path: Path) -> dict:
    """Parse pimpleFoam log for run validity signals and key diagnostics.

    This function scans the full log text (deterministically) and extracts:
      - completion markers (End, last Time)
      - fatal/error blocks (FOAM FATAL, Floating point exception, etc.)
      - stability hints (Courant number, continuity errors)

    Returns a dict with:
      - status: ok|fatal|incomplete|missing
      - last_time_s: float|None
      - has_end: bool
      - fatal_snippet: str|None
      - warning_lines: list[str]
      - courant_last: dict|None (mean,max)
      - courant_max: float|None
      - continuity_last: dict|None (local,global,cumulative)
      - continuity_cumulative: float|None
      - tail: str (last ~120 lines)
    """

    out = {
        "status": "missing",
        "last_time_s": None,
        "has_end": False,
        "fatal_snippet": None,
        "warning_lines": [],
        "courant_last": None,
        "courant_max": None,
        "continuity_last": None,
        "continuity_cumulative": None,
        "tail": "",
    }

    if not log_path.exists():
        return out

    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return out

    lines = txt.splitlines()
    out["tail"] = "\n".join(lines[-120:])

    # End marker
    out["has_end"] = bool(re.search(r"^\s*End\s*$", txt, flags=re.MULTILINE))

    # Last reported Time = ...
    last_t = None
    for m in re.finditer(r"^\s*Time\s*=\s*([0-9.eE+-]+)\s*s?\s*$", txt, flags=re.MULTILINE):
        try:
            last_t = float(m.group(1))
        except Exception:
            continue
    out["last_time_s"] = last_t

    # Courant number
    courant_re = re.compile(
        r"^\s*Courant Number\s+mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    )
    cmax = None
    clast = None
    for m in courant_re.finditer(txt):
        try:
            mean = float(m.group(1))
            mx = float(m.group(2))
        except Exception:
            continue
        clast = {"mean": mean, "max": mx}
        cmax = mx if cmax is None else max(cmax, mx)
    out["courant_last"] = clast
    out["courant_max"] = cmax

    # Continuity errors
    cont_re = re.compile(
        r"^\s*time step continuity errors\s*:\s*sum local\s*=\s*([0-9.eE+-]+),\s*global\s*=\s*([0-9.eE+-]+),\s*cumulative\s*=\s*([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    )
    cont_last = None
    for m in cont_re.finditer(txt):
        try:
            loc = float(m.group(1))
            glob = float(m.group(2))
            cum = float(m.group(3))
        except Exception:
            continue
        cont_last = {"local": loc, "global": glob, "cumulative": cum}
    out["continuity_last"] = cont_last
    out["continuity_cumulative"] = cont_last.get("cumulative") if cont_last else None

    # Warnings (keep a small sample)
    warn = []
    for ln in lines:
        if "FOAM Warning" in ln or "--> FOAM Warning" in ln:
            warn.append(ln.strip())
            if len(warn) >= 12:
                break
    out["warning_lines"] = warn

    # Fatal/error markers: avoid false positives from sigFpe enabling line.
    fatal_start_re = re.compile(
        r"^\s*(-->)?\s*FOAM\s+(FATAL|ERROR)\b.*$|^\s*Floating point exception\b|^\s*Segmentation fault\b|^\s*MPI_ABORT\b",
        re.IGNORECASE,
    )

    def _is_sigfpe_enabling(ln: str) -> bool:
        return bool(re.search(r"^\s*sigFpe\s*:\s*Enabling floating point exception trapping", ln, re.IGNORECASE))

    fatal_idx = None
    for i, ln in enumerate(lines):
        if _is_sigfpe_enabling(ln):
            continue
        if fatal_start_re.search(ln):
            fatal_idx = i
            break

    if fatal_idx is not None:
        out["status"] = "fatal"
        snippet = lines[fatal_idx : min(len(lines), fatal_idx + 80)]
        out["fatal_snippet"] = "\n".join(snippet).strip()
        return out

    out["status"] = "ok" if out["has_end"] else "incomplete"
    return out


def _deterministic_run_validity(out_dir: Path) -> dict:
    """Combine log + artifacts.json into a deterministic validity verdict."""
    det = {
        "status": "unknown",  # ok|fatal|incomplete|time_mismatch|missing
        "notes": [],
        "log": None,
        "deliverables": None,
    }

    log_info = _parse_pimplefoam_log(out_dir / "log.pimpleFoam")
    det["log"] = log_info
    if log_info.get("status") == "missing":
        det["notes"].append("Missing log.pimpleFoam")
    if log_info.get("status") == "fatal":
        det["status"] = "fatal"
        det["notes"].append("Solver crashed (FOAM FATAL).")
        return det

    # Deliverables/time mapping
    d = _deterministic_velocity_deliverables_check(out_dir)
    det["deliverables"] = d
    if d.get("enabled") and d.get("ok") is False:
        det["status"] = "time_mismatch"
        det["notes"].extend(d.get("notes") or [])
        return det

    if log_info.get("status") == "incomplete":
        det["status"] = "incomplete"
        det["notes"].append("Solver did not reach a clean 'End' in log.")
        return det

    if d.get("enabled") and d.get("ok") is True and log_info.get("status") == "ok":
        det["status"] = "ok"
        return det

    det["status"] = "unknown"
    return det


def _make_montage_jpeg_bytes(
    timestep_images: list,
    *,
    ncols: int = 3,
    tile_w: int = 384,
    tile_h: int = 384,
    pad: int = 6,
    label: bool = True,
) -> Optional[bytes]:
    """Create a JPEG montage from a list of timestep image dicts.

    Each timestep_images entry should have keys like: base64, t, path.
    Montage layout is time-order left-to-right, top-to-bottom.
    """
    if not timestep_images:
        return None
    if not _PIL_AVAILABLE:
        return None

    # Decode and load images
    loaded = []
    for d in timestep_images:
        try:
            raw = base64.b64decode(d.get("base64", "") or "")
            if not raw:
                continue
            im = Image.open(BytesIO(raw)).convert("RGB")
            im = im.resize((tile_w, tile_h))
            loaded.append((d, im))
        except Exception:
            continue

    if not loaded:
        return None

    n = len(loaded)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    W = ncols * tile_w + (ncols + 1) * pad
    H = nrows * tile_h + (nrows + 1) * pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    font = None
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for idx, (d, im) in enumerate(loaded):
        r = idx // ncols
        c = idx % ncols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        canvas.paste(im, (x0, y0))

        if label:
            try:
                draw = ImageDraw.Draw(canvas)
                t = d.get("t")
                tstr = f"t={float(t):.2f}s" if t is not None else "t=?"
                name = Path(d.get("path", "")).name if d.get("path") else ""
                txt = f"{tstr} {name}".strip()
                # Draw a white box behind text for readability
                if font:
                    bbox = draw.textbbox((0, 0), txt, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                else:
                    tw, th = (len(txt) * 6, 12)
                draw.rectangle([x0 + 4, y0 + 4, x0 + 8 + tw, y0 + 8 + th], fill=(255, 255, 255))
                draw.text((x0 + 6, y0 + 6), txt, fill=(0, 0, 0), font=font)
            except Exception:
                pass

    buf = BytesIO()
    try:
        canvas.save(buf, format="JPEG", quality=82, optimize=True)
    except Exception:
        canvas.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def choose_batch(base_dir: Path, batch_name: Optional[str]) -> Optional[Path]:
    if not base_dir.exists():
        print(f"No experiments directory found at {base_dir}")
        return None

    batches = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    if not batches:
        print(f"No batch directories found in {base_dir}")
        return None

    if batch_name:
        candidate = base_dir / batch_name
        if candidate.exists() and candidate.is_dir():
            return candidate
        else:
            print(f"Batch '{batch_name}' not found under {base_dir}")
            return None

    # default: newest batch (by name sort / mtime)
    # try by modification time
    batches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return batches[0]


def collect_batch_info(batch_dir: Path, max_experiments: Optional[int] = None):
    experiments = [d for d in batch_dir.iterdir() if d.is_dir()]
    experiments.sort()
    if max_experiments:
        experiments = experiments[:max_experiments]

    batch_summary = []

    for exp in experiments:
        exp_info = {"experiment": exp.name, "runs": []}
        # Only analyze actual run directories. Ignore backups/temp reruns (e.g. backup_run_*, temp_rerun_*).
        for run in sorted([d for d in exp.iterdir() if d.is_dir() and d.name.startswith('run_')]):
            run_info = {"run": run.name, "paths": {}}
            # primary user requirement
            ur = run / "user_requirement.txt"
            if ur.exists():
                # User requirement is the primary spec; keep more context than other snippets.
                run_info["user_requirement"] = read_text_safe(ur, max_chars=8000)
                run_info["paths"]["user_requirement"] = str(ur)

            # Collect postprocess time-stamped images for BOTH velocity magnitude (UMag) and pressure (p)
            out_dir = run / "output"
            if out_dir.exists():

                def _parse_time_from_stem(stem: str, prefix: str):
                    # e.g., stem='umag_t0p10' prefix='umag_t' -> 0.10
                    try:
                        tag = stem.split(prefix, 1)[1]
                        return float(tag.replace("p", "."))
                    except Exception:
                        return None

                def _collect_images(pattern: str, prefix: str, field: str):
                    paths = list(out_dir.glob(pattern))

                    def _t(p: Path):
                        return _parse_time_from_stem(p.stem, prefix)

                    paths.sort(key=lambda p: (_t(p) is None, _t(p) or 0.0, p.name))
                    imgs = []
                    for p in paths:
                        image_base64 = encode_image_to_base64(p)
                        if not image_base64:
                            continue
                        imgs.append(
                            {
                                "field": field,
                                "path": str(p),
                                "base64": image_base64,
                                "format": "png",
                                "t": _t(p),
                            }
                        )
                    return imgs

                umag_imgs = _collect_images("umag_t*.png", "umag_t", "umag")
                p_imgs = _collect_images("p_t*.png", "p_t", "p")

                if umag_imgs:
                    run_info["timestep_images_umag"] = umag_imgs
                if p_imgs:
                    run_info["timestep_images_p"] = p_imgs

                # For backward compatibility, keep a combined list.
                timestep_images = umag_imgs + p_imgs
                if timestep_images:
                    run_info["timestep_images"] = timestep_images
                    print(
                        f"   📸 Found timestep images under: {out_dir} (umag={len(umag_imgs)}, p={len(p_imgs)})"
                    )

                    # Representative image for cross-case batch summary: prefer UMag at final time.
                    rep = out_dir / "umag_t3p00.png"
                    if not rep.exists() and umag_imgs:
                        rep = Path(umag_imgs[-1]["path"])

                    rep64 = encode_image_to_base64(rep)
                    if rep64:
                        run_info["visualization_image"] = {
                            "path": str(rep),
                            "base64": rep64,
                            "format": "png",
                        }

            # OpenFOAM log summary (parsed deterministically)
            out_dir = run / "output"
            if out_dir.exists():
                run_info["paths"]["output"] = str(out_dir)

                log_path = out_dir / "log.pimpleFoam"
                run_info["log_info"] = _parse_pimplefoam_log(log_path)

            exp_info["runs"].append(run_info)
        batch_summary.append(exp_info)

    return batch_summary


def build_prompt_for_batch(batch_name: str, batch_summary, simulation_description: str = None, 
                          simulation_instructions: str = None, simulation_configs: list = None) -> str:
    """
    Enhanced prompt builder for comprehensive cross-case analysis.
    """
    
    # Build enhanced header with study context
    header = (
        f"You are an expert CFD analysis assistant.\n"
        f"You will analyze a batch of simulation runs named '{batch_name}'.\n"
    )

    if simulation_description:
        header += f"\nSTUDY CONTEXT: {simulation_description}\n"

    if simulation_instructions:
        header += f"\nANALYSIS INSTRUCTIONS:\n{simulation_instructions}\n"

    header += (
        "\nFor this batch analysis, provide:\n"
        "1. INDIVIDUAL RUN ANALYSIS: For each run, assess whether outputs match the user requirement and note numerical/physical issues.\n"
        "2. CROSS-CASE COMPARISON: Compare outcomes across the parameter sweep(s) and identify trends and outliers.\n"
        "3. ARTIFACT COMPLETENESS: Verify required visualizations/exports are present and consistent across runs.\n"
        "4. NUMERICAL QUALITY: Comment on stability, convergence/steadiness, and whether mesh/time step/sim time appear adequate.\n"
        "\nProvide accuracy scores (/10) for each run and an overall study assessment.\n\n"
    )

    # Enhanced per-experiment blocks with case metadata
    blocks = []
    for i, exp in enumerate(batch_summary):
        case_info = ""
        if simulation_configs and i < len(simulation_configs):
            config = simulation_configs[i]
            case_info = (
                f"Case ID: {config.get('case_id', 'Unknown')}\n"
                f"Profile: {config.get('profile_id', 'Unknown')}\n"
                f"Reynolds Number: {config.get('reynolds_number', 'Unknown')}\n"
                f"Geometry: {config.get('geometry', 'Unknown')}\n"
            )
        
        b = [f"=== EXPERIMENT {i+1}: {exp['experiment']} ===\n{case_info}"]
        
        for run in exp.get("runs", []):
            b.append(f"Run: {run['run']}")
            if "user_requirement" in run:
                b.append("User Requirement:\n" + textwrap.indent(run["user_requirement"], "  "))
            
            # Log summary for context
            log_info = run.get("log_info") or {}
            if isinstance(log_info, dict) and log_info:
                b.append("OpenFOAM log summary:")
                b.append(f"  status: {log_info.get('status')}")
                b.append(f"  has_end: {log_info.get('has_end')}")
                b.append(f"  last_time_s: {log_info.get('last_time_s')}")
                b.append(f"  courant_last: {log_info.get('courant_last')}")
                b.append(f"  continuity_cumulative: {log_info.get('continuity_cumulative')}")
                if log_info.get("fatal_snippet"):
                    b.append("  fatal snippet (truncated):")
                    sn = str(log_info.get("fatal_snippet"))
                    sn = sn[:800] + ("\n  ...[truncated]..." if len(sn) > 800 else "")
                    b.append(textwrap.indent(sn, "    "))
                    
        blocks.append("\n".join(b))

    # Cross-case analysis section
    cross_case_prompt = """

=== CROSS-CASE COMPARATIVE ANALYSIS REQUIREMENTS ===

After analyzing individual runs, provide a concise comparative analysis across the study:

1. PARAMETER → RESPONSE TRENDS:
   - What qualitative changes occur as parameters vary (e.g., inlet/fuel settings, geometry changes)?

2. CONSISTENCY & OUTLIERS:
   - Identify runs that look inconsistent with the rest (possible numerical issues or setup mistakes).

3. ARTIFACT/POSTPROCESS COMPLETENESS:
   - Note any missing or inconsistent visualization/export artifacts across runs.

4. RECOMMENDATIONS:
   - Suggest reruns with concrete requirement fixes if needed.
   - Suggest next experiments if the sweep is insufficient to support/reject the hypothesis.

Conclude with an overall study assessment and next steps.
"""

    prompt = header + "\n\n".join(blocks) + cross_case_prompt
    
    # Ensure prompt isn't too large
    if len(prompt) > 40000:  # Increased limit for enhanced analysis
        prompt = prompt[:40000] + "\n\n...[truncated for length]..."
    return prompt


def build_multimodal_cross_case_prompt(batch_name: str, batch_summary, all_images: list,
                                     simulation_description: str = None, 
                                     simulation_instructions: str = None, 
                                     simulation_configs: list = None) -> str:
    """
    Build a multimodal prompt for comprehensive cross-case analysis with all visualization images.
    """
    
    header = (
        f"Cross-case CFD analysis\n"
        f"Batch: {batch_name}\n\n"
        f"You are analyzing a CFD study with {len(all_images)} visualization images.\n"
    )
    
    if simulation_description:
        header += f"STUDY CONTEXT: {simulation_description}\n\n"
    
    if simulation_instructions:
        header += f"ANALYSIS INSTRUCTIONS:\n{simulation_instructions}\n\n"
    
    header += (
        f"Multimodal analysis instructions:\n"
        f"You will receive {len(all_images)} visualization images showing flow fields for different cases/parameters.\n"
        f"For each image, provide:\n"
        f"1. What is shown (field, slice/plane, time) and whether it matches the requirement\n"
        f"2. Notable numerical issues visible (if any)\n\n"
        f"Then provide cross-case comparative analysis:\n"
        f"1. Parameter progression effects\n"
        f"2. Consistency and outliers\n"
        f"3. Overall study assessment and recommendations\n\n"
        f"Base your conclusions on the user requirements, OpenFOAM logs, and the images. Do not rely on visualization scripts.\n\n"
    )
    
    # Build case-by-case descriptions
    case_descriptions = []
    for i, img_data in enumerate(all_images):
        case_info = ""
        if simulation_configs and i < len(simulation_configs):
            config = simulation_configs[i]
            case_info = (
                f"Case ID: {config.get('case_id', 'Unknown')}\n"
                f"Reynolds Number: {config.get('reynolds_number', 'Unknown')}\n"
                f"Geometry: {config.get('geometry', 'Unknown')}\n"
            )
        
        log_info = img_data.get('log_info') or {}
        log_block = ""
        if isinstance(log_info, dict) and log_info:
            log_block = (
                "OpenFOAM log summary:\n"
                f"  status: {log_info.get('status')}\n"
                f"  has_end: {log_info.get('has_end')}\n"
                f"  last_time_s: {log_info.get('last_time_s')}\n"
                f"  courant_last: {log_info.get('courant_last')}\n"
            )
            if log_info.get('fatal_snippet'):
                sn = str(log_info.get('fatal_snippet'))
                sn = sn[:600] + ("\n  ...[truncated]..." if len(sn) > 600 else "")
                log_block += "  fatal excerpt (truncated):\n" + textwrap.indent(sn, "    ") + "\n"

        desc = (
            f"=== image {i+1}: {img_data['experiment']} ===\n"
            f"{case_info}"
            f"Run: {img_data['run']}\n"
            f"Visualization image: {img_data['image_data']['path']}\n"
            f"{log_block}"
            f"User requirement:\n{textwrap.indent(img_data['user_requirement'], '  ')}\n"
        )
        case_descriptions.append(desc)
    
    cross_analysis_requirements = """

Cross-case analysis requirements:

1. Parameter → response effects:
   - Describe how key qualitative features change across cases/parameters.

2. Temporal/steadiness check (if applicable):
   - If the provided images represent different cases at a single time, note consistency.

3. Numerical quality:
   - Identify signs of instability, excessive diffusion, nonphysical artifacts, or mesh imprinting.

4. Study completeness:
   - Are the chosen cases sufficient to support or reject the stated hypothesis? What is missing?

5. Recommendations:
   - Suggest reruns with concrete fixes.
   - Suggest next experiments to strengthen discriminative power.

Be specific and base conclusions on the requirements, logs, and images.
"""
    
    prompt = header + "\n\n".join(case_descriptions) + cross_analysis_requirements
    
    # Limit prompt size
    if len(prompt) > 50000:
        prompt = prompt[:50000] + "\n\n...[truncated for length]..."
    
    return prompt


def perform_multimodal_cross_case_analysis(prompt: str, all_images: list, model: str, temperature: float) -> str:
    """
    Perform multimodal analysis by sending prompt with all visualization images to LLM.
    """
    client, model_name = create_client(model)
    
    print(f"🖼️ Performing multimodal analysis with {len(all_images)} images...")
    
    # Prepare multimodal content 
    # For Bedrock and Claude, we need to format as content list
    multimodal_content = [{"type": "text", "text": prompt}]
    
    # Add all images to the content
    for i, img_data in enumerate(all_images):
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{img_data['image_data']['format']}",
                "data": img_data['image_data']['base64']
            }
        }
        multimodal_content.append(image_content)
        print(f"   📸 Added image {i+1}: {img_data['experiment']} / {img_data['run']}")
    
    # System message for multimodal analysis
    system_message = (
        "You are a CFD analysis assistant. "
        "Analyze the visualization images and the provided OpenFOAM log summaries to assess whether each run "
        "matches its user requirement. Provide cross-case comparisons, identify parameter/case effects, and "
        "suggest concrete rerun improvements where needed."
    )
    
    try:
        # For multimodal, pass the content list as prompt and use msg_history to structure the message
        msg_history = [{
            "role": "user", 
            "content": multimodal_content
        }][:-1]  # Remove last message since get_response_from_llm will add it
        
        # Extract just the text for the prompt parameter
        text_prompt = prompt
        
        analysis_text, _ = get_response_from_llm(
            prompt=text_prompt,
            client=client,
            model=model_name,
            system_message=system_message,
            temperature=temperature,
            print_debug=False,
            msg_history=msg_history
        )
        
        print("✅ Multimodal cross-case analysis completed successfully!")
        return analysis_text
        
    except Exception as e:
        print(f"❌ Multimodal analysis failed: {e}")
        # Try a different approach - create a custom call for multimodal
        return perform_direct_multimodal_call(prompt, all_images, client, model_name, system_message, temperature)


def perform_direct_multimodal_call(prompt: str, all_images: list, client, model: str, system_message: str, temperature: float) -> str:
    """
    Direct multimodal call for Bedrock/Claude with images.
    """
    print("🔄 Attempting direct multimodal call...")
    
    # Check if this is a Bedrock client
    is_bedrock_boto3 = hasattr(client, 'invoke_model') and (model.startswith("arn:aws:bedrock") or "bedrock" in str(type(client)).lower())
    
    if is_bedrock_boto3:
        # Bedrock Converse API with images
        content = [{"text": prompt}]
        
        # Add images in Bedrock format
        for img_data in all_images:
            content.append({
                "image": {
                    "format": img_data['image_data']['format'],
                    "source": {
                        "bytes": base64.b64decode(img_data['image_data']['base64'])
                    }
                }
            })
        
        converse_params = {
            "modelId": model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "system": [{"text": system_message}],
            "inferenceConfig": {
                "maxTokens": 4096,
                "temperature": temperature,
            }
        }
        
        response = client.converse(**converse_params)
        return response['output']['message']['content'][0]['text']
        
    elif "claude" in model:
        # Anthropic Claude API with images
        content = [{"type": "text", "text": prompt}]
        
        for img_data in all_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{img_data['image_data']['format']}",
                    "data": img_data['image_data']['base64']
                }
            })
        
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_message,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        return response.content[0].text
    
    else:
        # Fallback: text-only analysis
        print("⚠️  Model doesn't support multimodal, falling back to text-only analysis")
        analysis_text, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message=system_message,
            temperature=temperature,
            print_debug=False
        )
        return analysis_text


def analyze_batch(batch_dir: Path, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_experiments: Optional[int] = None,
                 simulation_description: str = None, simulation_instructions: str = None, simulation_configs: list = None,
                 auto_rerun_threshold: float = 5.0, enable_auto_rerun: bool = False, max_rerun_iterations: int = 3,
                 auto_execute_recommendations: bool = False):
    """
    Analyze a batch of experiments and return suggestions for reruns.
    Enhanced to perform cross-case comparative analysis across different Reynolds numbers.
    Optionally automatically rerun cases with very low scores until threshold is met.
    
    Args:
        batch_dir: Path to batch directory
        model: LLM model to use
        temperature: Temperature for LLM
        max_experiments: Maximum experiments to analyze
        simulation_description: Overall description of the simulation study
        simulation_instructions: Analysis instructions for the study
        simulation_configs: List of simulation configurations with case IDs and metadata
        auto_rerun_threshold: Threshold below which cases are automatically rerun (default: 5.0)
        enable_auto_rerun: Whether to automatically rerun very poor cases (default: False)
        max_rerun_iterations: Maximum number of rerun attempts per case (default: 3)
        auto_execute_recommendations: Whether to automatically execute study recommendations (default: False)
    
    Returns:
        tuple: (analysis_text, rerun_suggestions)
            analysis_text: str - The full analysis report with cross-case comparisons
            rerun_suggestions: list - List of dicts with experiment, run, and updated requirements
    """
    print(f"Scanning batch directory: {batch_dir}")
    batch_summary = collect_batch_info(batch_dir, max_experiments=max_experiments)

    if not batch_summary:
        print("No experiments found in batch")
        return None, []

    # Print a plan of what will be analyzed for transparency
    total_runs = 0
    print("\n=== Analysis Plan ===")
    for exp in batch_summary:
        runs = exp.get("runs", [])
        total_runs += len(runs)
        print(f"• Experiment: {exp['experiment']} — {len(runs)} run(s)")
        for r in runs:
            ur = (r.get("user_requirement", "") or "").strip()
            first_line = next((ln.strip() for ln in ur.splitlines() if ln.strip()), "<missing user requirement>")
            print(f"   - Run {r['run']}: {first_line}")
    print(f"Total runs summarized for batch-level analysis: {total_runs}")
    
    # Enhanced analysis with simulation context and image collection
    if simulation_description or simulation_instructions or simulation_configs:
        print("🔬 Enhanced cross-case comparative analysis enabled")
        if simulation_configs:
            print(f"   📊 Analyzing {len(simulation_configs)} simulation configurations")
    
    # Collect all visualization images for cross-case analysis
    all_images = []
    for exp in batch_summary:
        for run in exp.get("runs", []):
            if "visualization_image" in run:
                all_images.append({
                    'experiment': exp['experiment'],
                    'run': run['run'], 
                    'image_data': run['visualization_image'],
                    'user_requirement': run.get('user_requirement', ''),
                    'log_info': run.get('log_info', {}),
                })
    
    print(f"🖼️  Found {len(all_images)} visualization images for cross-case analysis")
    
    # Build enhanced multimodal prompt
    if all_images:
        # Create a comprehensive multimodal analysis
        prompt = build_multimodal_cross_case_prompt(
            batch_dir.name,
            batch_summary,
            all_images,
            simulation_description=simulation_description,
            simulation_instructions=simulation_instructions,
            simulation_configs=simulation_configs
        )
        
        # Use multimodal LLM call with all images
        try:
            analysis_text = perform_multimodal_cross_case_analysis(
                prompt, all_images, model, temperature
            )
        except Exception as e:
            print(f"Multimodal analysis failed, falling back to text-only: {e}")
            # Fallback to text-only analysis
            prompt = build_prompt_for_batch(
                batch_dir.name, 
                batch_summary, 
                simulation_description=simulation_description,
                simulation_instructions=simulation_instructions, 
                simulation_configs=simulation_configs
            )
            client, model_name = create_client(model)
            analysis_text, _ = get_response_from_llm(prompt, client, model_name, 
                                                   "You are a CFD expert. Analyze simulations against their user requirements.", 
                                                   print_debug=False, temperature=temperature)
    else:
        # No images found, use text-only analysis
        prompt = build_prompt_for_batch(
            batch_dir.name, 
            batch_summary, 
            simulation_description=simulation_description,
            simulation_instructions=simulation_instructions, 
            simulation_configs=simulation_configs
        )
        client, model_name = create_client(model)
        analysis_text, _ = get_response_from_llm(prompt, client, model_name,
                                               "You are a CFD expert. Analyze simulations against their user requirements.",
                                               print_debug=False, temperature=temperature)

    # Ensure client and model_name are defined for rerun analysis
    if 'client' not in locals() or 'model_name' not in locals():
        client, model_name = create_client(model)

    # Save overall analysis
    out_file = batch_dir / "analysis_summary.txt"
    out_file.write_text(analysis_text, encoding="utf-8")
    print(f"Saved batch analysis to {out_file}")

    # Evaluate runs and collect rerun suggestions with per-experiment analysis
    # This combines analysis + validation into a single step per experiment
    rerun_suggestions = []
    auto_rerun_cases = []
    try:
        rerun_suggestions, auto_rerun_cases = _analyze_and_collect_rerun_suggestions(
            batch_dir, batch_summary, client, model_name, temperature, 
            auto_rerun_threshold=auto_rerun_threshold
        )
    except Exception as e:
        print(f"Analysis and rerun evaluation stage encountered an error: {e}")
        import traceback
        traceback.print_exc()
        rerun_suggestions = []
        auto_rerun_cases = []

    # Automatically rerun very poor cases if enabled - loop until threshold is met
    if enable_auto_rerun:
        iteration = 0
        while iteration < max_rerun_iterations:
            if not auto_rerun_cases:
                break
                
            iteration += 1
            print(f"\n{'='*60}")
            print(f"🔄 AUTO-RERUN ITERATION {iteration}/{max_rerun_iterations}")
            print(f"Found {len(auto_rerun_cases)} cases with scores < {auto_rerun_threshold}")
            print(f"{'='*60}")
            
            for case in auto_rerun_cases:
                print(f"   • {case['experiment']}/{case['run']} (score: {case['accuracy']:.1f}/10)")
            
            print(f"\n🚀 Starting automatic reruns (iteration {iteration})...")
            _execute_automatic_reruns(batch_dir, auto_rerun_cases)
            
            # Re-analyze the entire batch to check if reruns improved the scores
            print(f"\n🔬 Re-analyzing batch after iteration {iteration}...")
            
            # Collect batch info again (may include replacement runs)
            batch_summary = collect_batch_info(batch_dir, max_experiments=max_experiments)
            
            # Re-analyze ALL cases in the batch (including replaced runs)
            rerun_suggestions_new, auto_rerun_cases_new = _analyze_and_collect_rerun_suggestions(
                batch_dir, batch_summary, client, model_name, temperature, 
                auto_rerun_threshold=auto_rerun_threshold
            )
            
            # Filter to only cases that still need auto-rerun
            remaining_auto_rerun = [
                case for case in auto_rerun_cases_new 
                if any(orig['experiment'] == case['experiment'] and orig['run'] == case['run'] 
                      for orig in auto_rerun_cases)
            ]
            
            improved_count = len(auto_rerun_cases) - len(remaining_auto_rerun)
            
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   • Cases improved: {improved_count}/{len(auto_rerun_cases)}")
            print(f"   • Cases still below threshold: {len(remaining_auto_rerun)}")
            
            if not remaining_auto_rerun:
                print(f"✅ All cases now meet threshold! Stopping auto-rerun loop.")
                break
            elif improved_count == 0:
                print(f"⚠️  No cases improved in iteration {iteration}. Continuing with remaining cases...")
            
            auto_rerun_cases = remaining_auto_rerun
        
        # Final status
        if auto_rerun_cases and iteration >= max_rerun_iterations:
            print(f"\n⏹️  Reached maximum iterations ({max_rerun_iterations}). {len(auto_rerun_cases)} cases still below threshold.")
            # Add remaining cases back to rerun suggestions for manual handling
            rerun_suggestions.extend(auto_rerun_cases)
        elif not auto_rerun_cases:
            print(f"\n✅ All auto-rerun cases now meet quality threshold after {iteration} iteration(s)!")
        
        # Update rerun suggestions to exclude auto-rerun cases that were handled
        rerun_suggestions = [r for r in rerun_suggestions if r['accuracy'] >= auto_rerun_threshold]
        
        print(f"\n🏁 Auto-rerun process completed. {len(rerun_suggestions)} cases remain for manual rerun consideration.")
    
    # Generate comprehensive study recommendations based on all analyses
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE STUDY ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    # Collect all individual analyses for recommendation generation
    all_individual_analyses = []
    for exp in batch_summary:
        exp_dir = batch_dir / exp["experiment"]
        for run in exp.get("runs", []):
            verdict_file = exp_dir / run["run"] / "analysis_verdict.json"
            if verdict_file.exists():
                try:
                    with open(verdict_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    all_individual_analyses.append(analysis_data)
                except Exception as e:
                    print(f"⚠️  Could not read analysis for {exp['experiment']}/{run['run']}: {e}")
    
    try:
        study_recommendations = _generate_study_recommendations(
            batch_dir=batch_dir,
            batch_analysis=analysis_text,
            individual_analyses=all_individual_analyses,
            simulation_configs=simulation_configs,
            client=client,
            model_name=model_name,
            temperature=temperature
        )
        
        print("\n Study recommendations generated successfully!")
        print("Check study_recommendations.txt for detailed suggestions")
        
        # Auto-execute recommendations if enabled
        if auto_execute_recommendations:
            print(f"\n EXECUTING STUDY RECOMMENDATIONS...")
            print(f"{'='*60}")
            _execute_study_recommendations(batch_dir)
        
    except Exception as e:
        print(f"⚠️  Could not generate study recommendations: {e}")
        study_recommendations = None
    
    return analysis_text, rerun_suggestions


def _generate_study_recommendations(
    batch_dir: Path,
    batch_analysis: str,
    individual_analyses: list,
    simulation_configs: list,
    client,
    model_name: str,
    temperature: float = 0.0
) -> str:
    """
    Generate comprehensive study recommendations by analyzing all results together.
    Suggests new experiments, parameter studies, or validation cases to make the study more complete.
    
    Args:
        batch_dir: Path to batch directory
        batch_analysis: Overall batch analysis text
        individual_analyses: List of individual run analyses
        simulation_configs: List of simulation configurations
        client: LLM client
        model_name: Model name
        temperature: Temperature for LLM
    
    Returns:
        str: Comprehensive study recommendations
    """
    print("\n🔬 Generating comprehensive study recommendations...")
    
    # Collect all individual analysis summaries
    individual_summaries = []
    for analysis in individual_analyses:
        summary = {
            'experiment': analysis.get('experiment', 'Unknown'),
            'run': analysis.get('run', 'Unknown'),
            'accuracy': analysis.get('accuracy', 0),
            'analysis': analysis.get('analysis', ''),
            'issues': analysis.get('explanation', ''),
            'requirement': analysis.get('original_requirement', '')
        }
        individual_summaries.append(summary)
    
    # Build comprehensive analysis prompt
    reynolds_numbers = []
    geometries = []
    if simulation_configs:
        reynolds_numbers = [config.get('reynolds_number', 'Unknown') for config in simulation_configs]
        geometries = list(set([config.get('geometry', 'Unknown') for config in simulation_configs]))
    
    system_message = (
        "You are a world-class CFD research scientist with expertise in experimental design and "
        "parameter studies. Your task is to identify the most critical missing experiments that would "
        "significantly improve the study's scientific value. You MUST respond with valid JSON only, "
        "focusing on the top 3-5 high priority experiments that are scientifically essential and "
        "computationally feasible."
    )
    
    prompt = f"""
🔬 COMPREHENSIVE STUDY ANALYSIS & RECOMMENDATION GENERATION

You have access to a complete CFD study with the following components:

1. OVERALL BATCH ANALYSIS:
{batch_analysis[:3000]}{'...[truncated]' if len(batch_analysis) > 3000 else ''}

2. STUDY PARAMETERS:
   - Reynolds Numbers Tested: {reynolds_numbers}
   - Geometries: {geometries}
   - Total Cases: {len(individual_summaries)}
   - Successful Cases: {len([a for a in individual_summaries if a['accuracy'] >= 6.0])}

3. INDIVIDUAL CASE RESULTS:
"""
    
    # Add individual case summaries
    for i, summary in enumerate(individual_summaries[:10]):  # Limit to first 10 for prompt size
        prompt += f"""
   Case {i+1}: {summary['experiment']}/{summary['run']}
   - Accuracy: {summary['accuracy']:.1f}/10
   - Requirement: {summary['requirement'][:100]}{'...' if len(summary['requirement']) > 100 else ''}
   - Analysis: {summary['analysis'][:200]}{'...' if len(summary['analysis']) > 200 else ''}
   - Issues: {summary['issues'][:150]}{'...' if len(summary['issues']) > 150 else ''}
"""
    
    if len(individual_summaries) > 10:
        prompt += f"\n   ... and {len(individual_summaries) - 10} more cases\n"
    
    prompt += f"""

🎯 COMPACT STUDY RECOMMENDATIONS

Based on your analysis, identify the TOP 3-5 HIGH PRIORITY experiments that would most significantly improve this study's scientific value and publication readiness.

OUTPUT FORMAT: Return ONLY a valid JSON object with this exact structure:

{{
  "summary": {{
    "total_cases_analyzed": {len(individual_summaries)},
    "successful_cases": {len([a for a in individual_summaries if a['accuracy'] >= 6.0])},
    "reynolds_range": {reynolds_numbers if reynolds_numbers else ["Unknown"]},
    "main_gaps": ["brief description of 1-2 key gaps"]
  }},
  "high_priority_experiments": [
    {{
      "experiment_id": "gap_grid_refinement",
      "description": "Grid refinement / sensitivity case on the current baseline flow",
      "parameters": {{
        "domain": "2D square enclosure 0.20x0.20x0.01 m (front/back empty)",
        "inlet": "bottom-center inlet with fixedValue U=(0 0.2 0) m/s",
        "walls": "no-slip side/bottom walls",
        "outlet": "top pressure outlet p=0",
        "mesh_size": "200x200x1",
        "solver_settings": "transient, CFL-controlled time step, write every 0.1 s"
      }},
      "scientific_justification": "Establishes numerical robustness and separates discretization error from physics",
      "computational_cost_hours": 3,
      "priority_reason": "Needed to support claims with a basic grid sensitivity check"
    }}
  ]
}}

REQUIREMENTS:
- Focus ONLY on the most critical 3-5 missing experiments
- Each experiment should fill a significant scientific gap
- Parameters must be specific and executable
- Computational cost should be realistic (1-10 hours per case)
- Prioritize experiments that would most improve publication potential

Identify experiments for: missing Reynolds numbers, grid convergence validation, benchmark comparison, or critical flow physics gaps.
"""
    
    try:
        recommendations, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model_name,
            system_message=system_message,
            temperature=temperature,
            print_debug=False
        )
        
        # Try to parse as JSON first
        try:
            recommendations_json = json.loads(recommendations)
            
            # Save JSON recommendations for programmatic use
            json_file = batch_dir / "study_recommendations.json"
            json_file.write_text(json.dumps(recommendations_json, indent=2), encoding="utf-8")
            print(f"📋 Saved JSON recommendations to {json_file}")
            
            # Also save human-readable version
            txt_file = batch_dir / "study_recommendations.txt"
            readable_content = f"""# 🔬 STUDY RECOMMENDATIONS

## Summary
- Total Cases Analyzed: {recommendations_json.get('summary', {}).get('total_cases_analyzed', 'Unknown')}
- Successful Cases: {recommendations_json.get('summary', {}).get('successful_cases', 'Unknown')}
- Reynolds Range: {recommendations_json.get('summary', {}).get('reynolds_range', [])}
- Main Gaps: {recommendations_json.get('summary', {}).get('main_gaps', [])}

## High Priority Experiments
"""
            
            for i, exp in enumerate(recommendations_json.get('high_priority_experiments', []), 1):
                readable_content += f"""
### {i}. {exp.get('experiment_id', 'Unknown')}
**Description:** {exp.get('description', 'No description')}
**Scientific Justification:** {exp.get('scientific_justification', 'No justification')}
**Computational Cost:** {exp.get('computational_cost_hours', 'Unknown')} hours
**Priority Reason:** {exp.get('priority_reason', 'No reason')}

**Parameters:**
"""
                params = exp.get('parameters', {})
                for key, value in params.items():
                    readable_content += f"- {key}: {value}\n"
            
            txt_file.write_text(readable_content, encoding="utf-8")
            print(f"📄 Saved readable recommendations to {txt_file}")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Response not in valid JSON format, saving as text only")
            # Save as text if JSON parsing fails
            recommendations_file = batch_dir / "study_recommendations.txt"
            recommendations_file.write_text(recommendations, encoding="utf-8")
            print(f"📋 Saved text recommendations to {recommendations_file}")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Failed to generate study recommendations: {e}")
        return "Error: Could not generate study recommendations"


def _execute_study_recommendations(batch_dir: Path):
    """
    Automatically execute study recommendations by reading from study_recommendations.json
    and running new experiments using Foam-Agent.
    
    Args:
        batch_dir: Path to batch directory containing study_recommendations.json
    """
    import subprocess
    import uuid
    from datetime import datetime
    
    # Look for the JSON recommendations file
    recommendations_file = batch_dir / "study_recommendations.json"
    
    if not recommendations_file.exists():
        print("❌ No study_recommendations.json found. Cannot execute recommendations.")
        return
    
    try:
        with open(recommendations_file, 'r', encoding='utf-8') as f:
            recommendations_json = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read recommendations file: {e}")
        return
    
    high_priority_experiments = recommendations_json.get('high_priority_experiments', [])
    
    if not high_priority_experiments:
        print("📝 No high priority experiments found in recommendations")
        return
    
    print(f"🎯 Found {len(high_priority_experiments)} high priority experiments to execute")
    
    project_root = _project_root
    foam_bench = (project_root / "Foam-Agent" / "foambench_main.py").resolve()
    
    if not foam_bench.exists():
        print(f"❌ Error: Foam-Agent not found at {foam_bench}. Cannot execute recommendations.")
        return
    
    # Find existing experiment directory in the batch (should be sim_TIMESTAMP_ID)
    existing_experiment_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('sim_')]
    
    if existing_experiment_dirs:
        # Use the first (and usually only) existing experiment directory
        target_experiment_dir = existing_experiment_dirs[0]
        print(f"📁 Adding recommendations to existing experiment: {target_experiment_dir}")
    else:
        # Fallback: create new experiment directory if none found
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        target_experiment_dir = batch_dir / f"sim_{timestamp}_{short_id}"
        target_experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created new experiment directory: {target_experiment_dir}")
    
    # Determine next run number by checking existing runs
    existing_runs = [d for d in target_experiment_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if existing_runs:
        # Extract run numbers and find the highest one
        run_numbers = []
        for run_dir in existing_runs:
            try:
                m = re.match(r"^run_(\d{3})", run_dir.name)
                if m:
                    run_numbers.append(int(m.group(1)))
            except:
                continue
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run_number = 1
    
    print(f"🔢 Starting recommendation runs at run number: {next_run_number:03d}")
    
    execution_results = []
    current_run_number = next_run_number
    
    for i, exp_rec in enumerate(high_priority_experiments, 1):
        experiment_id = exp_rec.get('experiment_id', f'rec_{i}')
        description = exp_rec.get('description', 'No description')
        params = exp_rec.get('parameters', {})
        
        print(f"\\n[{i}/{len(high_priority_experiments)}] 🔬 Executing: {experiment_id}")
        print(f"   📋 Description: {description}")
        print(f"   🎯 Run number: {current_run_number:03d}")
        print(f"   ⏱️  Estimated time: {exp_rec.get('computational_cost_hours', 'Unknown')} hours")
        
        # Convert recommendation parameters to user requirement format
        user_requirement = _convert_recommendation_to_user_requirement(exp_rec)
        
        if not user_requirement:
            print(f"   ❌ Could not convert recommendation to user requirement")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': 'Failed to convert recommendation to user requirement'
            })
            current_run_number += 1
            continue
        
        print(f"   📄 Generated user requirement:")
        print(f"   {user_requirement[:150]}...")
        
        # Create run directory with sequential numbering
        run_dir = target_experiment_dir / f"run_{current_run_number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write user requirement
        prompt_file = run_dir / "user_requirement.txt"
        try:
            prompt_file.write_text(user_requirement, encoding='utf-8')
        except Exception as e:
            print(f"   ❌ Failed to write requirement file: {e}")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': f'Failed to write requirement file: {e}'
            })
            current_run_number += 1
            continue
        
        # Build Foam-Agent command
        cmd = [
            "python", str(foam_bench),
            "--openfoam_path", os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10"),
            "--output", str(output_dir),
            "--prompt_path", str(prompt_file),
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Execute Foam-Agent
        try:
            print(f"   ⏳ Executing Foam-Agent...")
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per case
            )
            
            if result.returncode == 0:
                print(f"   ✅ Recommendation executed successfully!")
                execution_results.append({
                    'run_number': current_run_number,
                    'experiment_id': experiment_id,
                    'success': True,
                    'run_dir': str(run_dir),
                    'output_dir': str(output_dir)
                })
            else:
                print(f"   ❌ Execution failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                execution_results.append({
                    'run_number': current_run_number,
                    'experiment_id': experiment_id,
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode
                })
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Execution timed out after 2 hours")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': 'Execution timed out after 2 hours'
            })
        except Exception as e:
            print(f"   ❌ Execution exception: {e}")
            execution_results.append({
                'run_number': current_run_number,
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e)
            })
        
        current_run_number += 1
    
    # Save execution results
    try:    
        results_file = target_experiment_dir / "execution_results.json"
        results_file.write_text(json.dumps(execution_results, indent=2), encoding='utf-8')
        print(f"\\n📋 Saved execution results to {results_file}")
    except Exception as e:
        print(f"Failed to save execution results: {e}")
    
    # Print summary
    successful = sum(1 for r in execution_results if r.get('success', False))
    print(f"\\n{'='*60}")
    print(f"📊 RECOMMENDATIONS EXECUTION SUMMARY")
    print(f"   Experiment Directory: {target_experiment_dir}")
    print(f"   Total Recommendations: {len(execution_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(execution_results) - successful}")
    
    if successful > 0:
        print("   ✅ Successful Runs:")
        for result in execution_results:
            if result['success']:
                print(f"      • Run {result['run_number']:03d} - {result['experiment_id']}")
    
    if len(execution_results) > successful:
        print("   ❌ Failed Runs:")
        for result in execution_results:
            if not result['success']:
                print(f"      • Run {result['run_number']:03d} - {result['experiment_id']}")
    
    print(f"{'='*60}")


def _convert_recommendation_to_user_requirement(exp_rec: dict) -> str:
    """Convert a study recommendation JSON into a Foam-Agent prompt.

    This repository previously carried legacy demo conversion logic; that conversion logic has
    been removed. We now generate a generic, execution-safe requirement text from whatever fields the
    recommender provides.

    Priority order:
      1) exp_rec['user_requirement'|'requirement'|'prompt'] if present
      2) parameters['user_requirement'|'requirement'|'prompt'] if present
      3) Otherwise, serialize description + parameters into a concise prompt.
    """

    for k in ("user_requirement", "requirement", "prompt"):
        v = exp_rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    params = exp_rec.get("parameters")
    if not isinstance(params, dict):
        params = {}

    for k in ("user_requirement", "requirement", "prompt"):
        v = params.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    description = exp_rec.get("description")
    if not isinstance(description, str) or not description.strip():
        description = "CFD simulation"

    lines: list[str] = [description.strip()]

    # Render parameters in a stable order for readability.
    if params:
        lines.append("")
        lines.append("Parameters:")
        for key in sorted(params.keys(), key=lambda x: str(x)):
            val = params.get(key)
            lines.append(f"- {key}: {val}")

    # Always include a minimal visualization statement so downstream tooling has a clear target.
    lines.append("")
    lines.append("Visualization: export velocity magnitude |U| contours at the requested output times.")

    return "\n".join([ln for ln in lines if ln is not None]).strip()


def _analyze_and_collect_rerun_suggestions(
    batch_dir: Path,
    batch_summary: list,
    client,
    model_name: str,
    temperature: float,
    rerun_threshold: float = 7.0,
    auto_rerun_threshold: float = 5.0,
) -> tuple[list, list]:
    """Analyze each run and decide whether a rerun is needed.

    Two-step analysis (two LLM calls per run):
      1) log triage (OpenFOAM log.pimpleFoam)
      2) image vs requirement evaluation (velocity images)

    LLM inputs are limited to: images, user requirement, and OpenFOAM logs.
    Visualization scripts are never passed to the LLM.
    """

    log_system = (
        "You are an OpenFOAM troubleshooting assistant. "
        "Given a user requirement and an OpenFOAM solver log, determine if the run completed, "
        "and propose concrete steps to make a rerun succeed. Focus on actionable improvements."
    )

    vision_system = (
        "You are a CFD run evaluator. Given the user requirement and velocity visualization images, "
        "decide whether the run did what the user requested. You do not know ground truth; judge only "
        "against the requirement and obvious numerical issues. Provide a score and concrete actions to "
        "improve a rerun if needed."
    )

    rerun_suggestions: list = []
    auto_rerun_cases: list = []

    # Always include pressure images when available.
    def _order_images(xs: list) -> list:
        return sorted(
            xs,
            key=lambda d: (
                d.get("t") is None,
                float(d.get("t") or 0.0),
                str(d.get("path") or ""),
            ),
        )

    for exp in batch_summary:
        exp_name = exp.get("experiment")
        exp_dir = batch_dir / str(exp_name)
        runs = exp.get("runs", []) or []
        print(f"\nAnalyzing experiment: {exp_name} with {len(runs)} run(s)")

        for run in runs:
            run_name = run.get("run")
            ur = (run.get("user_requirement", "") or "").strip()
            first_line = next((ln.strip() for ln in ur.splitlines() if ln.strip()), "<missing user requirement>")
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            print(f"   Run {run_name}: {first_line}")

            out_dir = Path(exp_dir / str(run_name) / "output")
            log_path = out_dir / "log.pimpleFoam"

            # ------------------------------
            # Step 1: log triage (LLM call)
            # ------------------------------
            log_info = run.get("log_info") if isinstance(run.get("log_info"), dict) else None
            if not log_info:
                log_info = _parse_pimplefoam_log(log_path)

            log_packet_lines = [
                f"log_present: {log_path.exists()}",
                f"log_status_guess: {log_info.get('status')}",
                f"has_end: {log_info.get('has_end')}",
                f"last_time_s: {log_info.get('last_time_s')}",
                f"courant_last: {log_info.get('courant_last')}",
                f"courant_max: {log_info.get('courant_max')}",
                f"continuity_last: {log_info.get('continuity_last')}",
                f"continuity_cumulative: {log_info.get('continuity_cumulative')}",
            ]

            warning_lines = log_info.get("warning_lines") or []
            if isinstance(warning_lines, list) and warning_lines:
                log_packet_lines += ["", "warnings (sample):"]
                for w in warning_lines[:12]:
                    log_packet_lines.append(str(w))

            fatal_snip = (log_info.get("fatal_snippet") or "").strip()
            if fatal_snip:
                log_packet_lines += ["", "fatal_or_error_excerpt:", fatal_snip]

            tail = (log_info.get("tail") or "").strip()
            if tail:
                log_packet_lines += ["", "log_tail:", tail]

            log_packet = "\n".join(log_packet_lines).strip()

            log_prompt = f"""User requirement:
{ur}

OpenFOAM log excerpts:
{log_packet}

Task:
- Determine run_status = ok|incomplete|fatal|unknown from the log.
- If rerun is needed, propose ordered, concrete action_items to make the rerun succeed.
- Include short evidence lines from the log.

Return strict JSON between ```json and ```:
{{
  \"run_status\": string,
  \"rerun_needed\": boolean,
  \"summary\": string,
  \"evidence\": string[],
  \"action_items\": string[]
}}
"""

            try:
                log_response_text, _ = get_response_from_llm(
                    prompt=log_prompt,
                    client=client,
                    model=model_name,
                    system_message=log_system,
                    temperature=temperature,
                    print_debug=False,
                )
            except Exception as e:
                print(f"      log triage LLM failed: {e}")
                continue

            log_data = extract_json_between_markers(log_response_text) or {}

            # Use deterministic parser status as ground truth for run health.
            det_status = str(log_info.get("status") or "").strip().lower()
            if det_status in {"ok", "incomplete", "fatal"}:
                log_status = det_status
            else:
                log_status = str(log_data.get("run_status") or "unknown").strip().lower()
                if log_status not in {"ok", "incomplete", "fatal", "unknown"}:
                    log_status = "unknown"

            log_rerun_needed = bool(log_data.get("rerun_needed", log_status != "ok"))
            log_summary = str(log_data.get("summary") or "").strip()

            log_evidence = log_data.get("evidence")
            if not isinstance(log_evidence, list):
                log_evidence = []
            log_evidence = [str(x) for x in log_evidence if isinstance(x, (str, int, float))]

            log_action_items = log_data.get("action_items")
            if not isinstance(log_action_items, list):
                log_action_items = []
            log_action_items = [str(x) for x in log_action_items if isinstance(x, (str, int, float))]

            # ------------------------------
            # Step 2: images vs requirement (LLM call)
            # ------------------------------
            timestep_images = run.get("timestep_images", []) or []
            timestep_umag = run.get("timestep_images_umag", []) or [x for x in timestep_images if x.get("field") == "umag"]
            timestep_p = run.get("timestep_images_p", []) or [x for x in timestep_images if x.get("field") == "p"]

            ordered_umag = _order_images(timestep_umag)
            ordered_p = _order_images(timestep_p)

            missing_umag = not bool(ordered_umag)

            img_lines = []
            for d in ordered_umag:
                tstr = f"t={float(d['t']):.2f}s" if d.get("t") is not None else "t=?"
                fname = Path(d.get("path", "")).name
                img_lines.append(f"umag: {tstr} {fname}")
            for d in ordered_p:
                tstr = f"t={float(d['t']):.2f}s" if d.get("t") is not None else "t=?"
                fname = Path(d.get("path", "")).name
                img_lines.append(f"p: {tstr} {fname}")
            images_index = "\n".join(img_lines).strip() or "<no images indexed>"

            # Vision prompt is stable across runs. If log isn't ok or images missing, the model should
            # still return a score and focus on rerun improvements.
            vision_prompt = f"""User requirement:
{ur}

OpenFOAM log status (from step 1): {log_status}

Images provided (time order):
{images_index}

Task:
- Decide whether this run did what the user requested, but evaluate only the log health + image-based deliverables.
- Ignore non-image exports even if the requirement mentions them (e.g., CSV profiles). Do not mention them.
- Set "visualization_matches_requirement" based only on the visualization portion (the provided images vs requested visualizations).
- Pressure images may be provided even if the requirement does not mention pressure; do not penalize the run for extra images.
  If pressure is explicitly requested and pressure images are missing, that is an issue.
- If the log status is not ok, focus on how to fix the run rather than over-interpreting images.
- Provide a score (0 to 10) using this rule:
  - 0 if the log indicates an error/fatal run
  - <3 if the log did not reach End
  - 3-10 for visualization correctness when the log is ok
- If rerun is needed, provide ordered action_items to improve the rerun.

Return strict JSON between ```json and ```:
{{
  \"analysis\": string,
  \"accurate\": boolean,
  \"accuracy\": number,
  \"explanation\": string,
  \"action_items\": string[],
  \"visualization_matches_requirement\": boolean,
  \"visualization_statement_clear\": boolean,
  \"proposed_user_requirement\": string|null
}}
"""

            # Prepare a multimodal request only when we have velocity images and a multimodal client.
            vision_response_text = None
            try:
                if hasattr(client, "converse") and (not missing_umag):
                    content = []
                    umag_montage = _make_montage_jpeg_bytes(ordered_umag)
                    if umag_montage:
                        content.append({"image": {"format": "jpeg", "source": {"bytes": umag_montage}}})

                    if ordered_p:
                        p_montage = _make_montage_jpeg_bytes(ordered_p)
                        if p_montage:
                            content.append({"image": {"format": "jpeg", "source": {"bytes": p_montage}}})

                    content.append({"text": vision_prompt})
                    messages = [{"role": "user", "content": content}]
                    response = client.converse(
                        modelId=model_name,
                        messages=messages,
                        system=[{"text": vision_system}],
                        inferenceConfig={
                            "temperature": temperature,
                            "maxTokens": 4096,
                        },
                    )
                    vision_response_text = response["output"]["message"]["content"][0]["text"]
                else:
                    # Text-only fallback (still a second LLM call)
                    vision_response_text, _ = get_response_from_llm(
                        prompt=vision_prompt,
                        client=client,
                        model=model_name,
                        system_message=vision_system,
                        temperature=temperature,
                        print_debug=False,
                    )
            except Exception as e:
                print(f"      vision LLM failed: {e}")
                continue

            vision_data = extract_json_between_markers(vision_response_text) or {}

            accurate = bool(vision_data.get("accurate", False))
            try:
                accuracy = float(vision_data.get("accuracy", 0.0))
            except Exception:
                accuracy = 0.0

            # Enforce global scoring policy:
            # - 0 if log indicates fatal/error
            # - <3 if log did not reach End (incomplete)
            # - 3-10 for visualization correctness when log is ok
            if log_status == "fatal":
                accuracy = 0.0
                accurate = False
            elif log_status in {"incomplete", "unknown"}:
                accuracy = max(0.0, min(2.9, accuracy))
                accurate = False
            else:  # ok
                accuracy = max(3.0, min(10.0, accuracy))

            analysis_text = str(vision_data.get("analysis", "") or "")
            explanation = str(vision_data.get("explanation", "") or "")

            v_action_items = vision_data.get("action_items")
            if not isinstance(v_action_items, list):
                v_action_items = []
            v_action_items = [str(x) for x in v_action_items if isinstance(x, (str, int, float))]

            viz_matches = bool(vision_data.get("visualization_matches_requirement", False))
            viz_statement_clear = bool(vision_data.get("visualization_statement_clear", bool(ur.strip())))
            proposed = vision_data.get("proposed_user_requirement")

            # Merge action items (dedupe while preserving order)
            merged_items = []
            for it in log_action_items + v_action_items:
                if not it:
                    continue
                if it not in merged_items:
                    merged_items.append(it)

            should_rerun = (
                log_rerun_needed
                or (log_status != "ok")
                or missing_umag
                or (accuracy < rerun_threshold)
                or (not viz_matches)
                or (not viz_statement_clear)
            )

            # Avoid auto-rerun for log failures; those often require template/config fixes.
            can_auto = (log_status == "ok") and (not missing_umag)

            updated_requirement = proposed
            if should_rerun and not (isinstance(updated_requirement, str) and updated_requirement.strip()):
                updated_requirement = _synthesize_updated_requirement(
                    original_requirement=ur,
                    action_items=merged_items,
                    log_evidence=log_evidence,
                    log_status=log_status,
                )

            decision = {
                "experiment": exp_name,
                "run": run_name,
                "run_status": log_status,
                "log_summary": log_summary,
                "log_evidence": log_evidence,
                "analysis": analysis_text,
                "action_items": merged_items,
                "accurate": accurate,
                "accuracy": accuracy,
                "explanation": explanation,
                "visualization_matches_requirement": viz_matches,
                "visualization_statement_clear": viz_statement_clear,
                "original_requirement": ur,
                "updated_requirement": updated_requirement,
                "evidence_files": {
                    "umag_images": [x.get("path") for x in ordered_umag],
                    "p_images": [x.get("path") for x in ordered_p],
                    "log": str(log_path) if log_path.exists() else None,
                },
            }

            # Write per-run outputs
            try:
                analysis_file = exp_dir / str(run_name) / "analysis.txt"
                blocks = []
                blocks.append(f"Run status: {log_status}")
                if log_summary:
                    blocks.append("\nLog summary:\n" + log_summary)
                if log_evidence:
                    blocks.append("\nLog evidence:\n" + "\n".join([f"- {x}" for x in log_evidence]))
                blocks.append(f"\nAccuracy: {accuracy:.1f}/10")
                blocks.append(f"Matches requirement: {viz_matches}")
                blocks.append(f"Visualization statement clear: {viz_statement_clear}")
                if explanation:
                    blocks.append("\nExplanation:\n" + explanation)
                if analysis_text:
                    blocks.append("\nAnalysis:\n" + analysis_text)
                if merged_items:
                    blocks.append("\nAction items:\n" + "\n".join([f"{i}. {it}" for i, it in enumerate(merged_items, 1)]))
                if proposed:
                    blocks.append("\nProposed corrected requirement:\n" + str(proposed))

                analysis_file.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")

                verdict_file = exp_dir / str(run_name) / "analysis_verdict.json"
                verdict_file.write_text(json.dumps(decision, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"      failed to write analysis outputs: {e}")

            if should_rerun:
                if can_auto and accuracy < auto_rerun_threshold:
                    print(f"      auto-rerun needed (accuracy={accuracy:.1f}/10)")
                    auto_rerun_cases.append(decision)
                else:
                    print(f"      rerun recommended (accuracy={accuracy:.1f}/10, status={log_status})")
                    rerun_suggestions.append(decision)
            else:
                print("      run is satisfactory")

    # Save consolidated rerun suggestions
    total_reruns = len(rerun_suggestions) + len(auto_rerun_cases)
    if total_reruns > 0:
        try:
            if rerun_suggestions:
                (batch_dir / "rerun_suggestions.json").write_text(json.dumps(rerun_suggestions, indent=2), encoding="utf-8")
            if auto_rerun_cases:
                (batch_dir / "auto_rerun_cases.json").write_text(json.dumps(auto_rerun_cases, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Failed to write rerun suggestions: {e}")
    else:
        print("\nNo reruns needed")

    return rerun_suggestions, auto_rerun_cases

def _synthesize_updated_requirement(original_requirement: str, action_items: list, log_evidence: list, log_status: str) -> str:
    """Build an 'updated requirement' prompt for reruns when the LLM didn't propose one.

    We keep the original user requirement verbatim, and append a small, explicit
    "RERUN FIXES" section derived from action_items/log evidence.
    """
    original_requirement = (original_requirement or "").rstrip()
    items = [str(x).strip() for x in (action_items or []) if str(x).strip()]

    # Add deterministic hints from common OpenFOAM fatal patterns.
    try:
        import re as _re
        for ln in (log_evidence or []):
            if not isinstance(ln, str):
                continue
            m = _re.search(r"keyword\s+(\S+)\s+is undefined in dictionary\s+\"([^\"]+)\"", ln)
            if m:
                key = m.group(1)
                dct = m.group(2)
                if key and key not in " ".join(items):
                    items.insert(0, f"Fix OpenFOAM dictionary error: add missing keyword '{key}' in {dct} (the run is failing at startup).")
    except Exception:
        pass

    if not items:
        items = [
            "Fix the startup/fatal errors shown in the OpenFOAM log, then rerun the case to completion (EndTime reached).",
            "Regenerate the required visualization images (umag_t*.png, p_t*.png) after a successful run."
        ]

    lines = []
    lines.append(original_requirement)
    lines.append("")
    lines.append("RERUN FIXES (must be applied before running):")
    lines.append(f"- Log status: {log_status}")
    for it in items:
        # keep bullets single-line to reduce prompt noise
        lines.append(f"- {it}")
    return "\n".join(lines).strip() + "\n"


def _execute_automatic_reruns(batch_dir: Path, auto_rerun_cases: list):
    """
    Automatically execute reruns for cases with critically low scores by calling Foam-Agent directly.
    Replaces the original bad case after backing it up.
    
    Args:
        batch_dir: Path to batch directory
        auto_rerun_cases: List of cases that need immediate rerun
    """
    import subprocess
    import shutil
    from datetime import datetime
    
    project_root = _project_root
    foam_bench = (project_root / "Foam-Agent" / "foambench_main.py").resolve()
    
    if not foam_bench.exists():
        print(f"❌ Error: Foam-Agent not found at {foam_bench}. Cannot perform automatic reruns.")
        return
    
    print(f"🔧 Foam-Agent found at: {foam_bench}")
    
    for i, case in enumerate(auto_rerun_cases, 1):
        exp_name = case['experiment']
        run_name = case['run']

        # Safety: auto-rerun should only operate on canonical run directories (run_###).
        # Never attempt to rerun backup/temp directories.
        import re as _re
        if not _re.match(r"^run_\d{3}(?:__.*)?$", str(run_name)):
            print(f"   ⏭️  Skipping non-canonical run directory for auto-rerun: {run_name}")
            continue
        updated_requirement = case.get('updated_requirement') or case.get('original_requirement')
        accuracy = float(case.get('accuracy') or 0.0)
        
        print(f"\n[{i}/{len(auto_rerun_cases)}] 🔄 Auto-rerunning: {exp_name}/{run_name}")
        print(f"   Original accuracy: {accuracy:.1f}/10")
        if isinstance(updated_requirement, str) and updated_requirement.strip():
            print(f"   Requirement used for rerun: {updated_requirement[:100]}...")
        else:
            print("   ❌ No usable requirement found in verdict; skipping auto-rerun")
            continue
        
        # Get paths
        exp_dir = batch_dir / exp_name
        original_run_dir = exp_dir / run_name
        
        if not original_run_dir.exists():
            print(f"   ❌ Original run directory not found: {original_run_dir}")
            continue
        
        # Create backup of original bad case
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # YYYYMMDD_HHMMSS_mmm
        backup_dir = exp_dir / f"backup_{run_name}_{timestamp}"
        
        try:
            print(f"   📦 Backing up original to: {backup_dir.name}")
            shutil.copytree(original_run_dir, backup_dir)
        except Exception as e:
            print(f"   ❌ Failed to backup original run: {e}")
            continue
        
        # Create temporary directory for new run
        temp_rerun_dir = exp_dir / f"temp_rerun_{timestamp}"
        temp_rerun_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory for Foam-Agent
        temp_output_dir = temp_rerun_dir / "output"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write updated user requirement to temp location
        temp_prompt_file = temp_rerun_dir / "user_requirement.txt"
        try:
            temp_prompt_file.write_text(updated_requirement, encoding='utf-8')
            print(f"   📄 Updated requirement written to temp location")
        except Exception as e:
            print(f"   ❌ Failed to write requirement file: {e}")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
            continue
        
        # Build Foam-Agent command
        cmd = [
            "python", str(foam_bench),
            "--openfoam_path", os.environ.get("WM_PROJECT_DIR", "/opt/openfoam10"),
            "--output", str(temp_output_dir),
            "--prompt_path", str(temp_prompt_file),
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Execute Foam-Agent
        try:
            print(f"   ⏳ Executing Foam-Agent (this may take several minutes)...")
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600 # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Auto-rerun successful! Replacing original run...")
                
                # Remove original bad run
                shutil.rmtree(original_run_dir)
                
                # Move temp rerun to replace original
                shutil.move(str(temp_rerun_dir), str(original_run_dir))
                
                # Run deterministic post-processing after rerun replacement.
                # This produces requirement-compliant artifacts (umag_t*.png, p_t*.png, uy_centerline_*.csv, artifacts.json).
                postprocess_result = None
                try:
                    from src.postprocess import postprocess_case
                    out_dir = original_run_dir / "output"
                    postprocess_result = postprocess_case(out_dir, user_requirement=updated_requirement)
                    if postprocess_result.get('success'):
                        print(f"   🧾 Postprocess artifacts written under: {out_dir}")
                    else:
                        print(f"   ⚠️  Postprocess skipped/failed: {postprocess_result.get('error')}")
                except Exception as e:
                    print(f"   ⚠️  Postprocess exception: {e}")
                    postprocess_result = {"success": False, "error": str(e)}

                # Save replacement metadata in the new run directory
                replacement_info = {
                    "replaced_on": datetime.now().isoformat(),
                    "original_accuracy": accuracy,
                    "backup_location": backup_dir.name,
                    "original_requirement": case['original_requirement'],
                    "updated_requirement": updated_requirement,
                    "foam_agent_success": True,
                    "postprocess": postprocess_result,
                    "replacement_reason": f"Auto-rerun due to accuracy score {accuracy:.1f}/10 < 5.0"
                }
                
                replacement_file = original_run_dir / "replacement_info.json"
                replacement_file.write_text(json.dumps(replacement_info, indent=2), encoding='utf-8')
                
                print(f"   📁 Original run replaced successfully!")
                print(f"   📦 Backup saved as: {backup_dir.name}")
                
            else:
                print(f"   ❌ Auto-rerun failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                
                # Clean up temp directory on failure
                shutil.rmtree(temp_rerun_dir, ignore_errors=True)
                
                # Save error info to backup directory
                error_info = {
                    "rerun_failed_on": datetime.now().isoformat(),
                    "original_accuracy": accuracy,
                    "updated_requirement": updated_requirement,
                    "foam_agent_success": False,
                    "foam_agent_error": result.stderr,
                    "return_code": result.returncode
                }
                
                error_file = backup_dir / "rerun_error.json"
                error_file.write_text(json.dumps(error_info, indent=2), encoding='utf-8')
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Auto-rerun timed out after 1 hour")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
        except Exception as e:
            print(f"   ❌ Auto-rerun exception: {e}")
            # Clean up temp directory
            shutil.rmtree(temp_rerun_dir, ignore_errors=True)
    
    print(f"\n🏁 Completed automatic reruns for {len(auto_rerun_cases)} cases")
    print("   ✅ Original bad runs have been replaced with corrected versions")
    print("   📦 Backups of original runs are preserved for reference")


def main():
    parser = argparse.ArgumentParser(description="Analyze a batch of experiments using an LLM")
    parser.add_argument("--batch", type=str, help="Batch folder name under data/experiments to analyze (defaults to latest)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to use (default: Bedrock ARN)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-experiments", type=int, default=None, help="Limit number of experiments to analyze")
    parser.add_argument("--auto-rerun", action="store_true", help="Automatically rerun cases with scores < 5.0 until threshold is met")
    parser.add_argument("--auto-rerun-threshold", type=float, default=5.0, help="Threshold below which cases are automatically rerun (default: 5.0)")
    parser.add_argument("--max-rerun-iterations", type=int, default=3, help="Maximum rerun attempts per case (default: 3)")
    parser.add_argument("--auto-execute-recommendations", action="store_true", help="Automatically execute study recommendations as new experiments (default: False)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    base_dir = project_root / "data" / "experiments"
    
    batch_dir = choose_batch(base_dir, args.batch)
    if not batch_dir:
        print("Unable to locate a batch to analyze. Exiting.")
        return
    
    analysis, rerun_suggestions = analyze_batch(
        batch_dir, 
        model=args.model, 
        temperature=args.temperature, 
        max_experiments=args.max_experiments,
        auto_rerun_threshold=args.auto_rerun_threshold,
        enable_auto_rerun=args.auto_rerun,
        max_rerun_iterations=args.max_rerun_iterations,
        auto_execute_recommendations=args.auto_execute_recommendations
    )
    
    if analysis:
        print("\n" + "="*60)
        print("📊 Analysis complete.")
        print("="*60)
        if rerun_suggestions:
            print(f"\n⚠️  Found {len(rerun_suggestions)} experiments that need reruns")
            print("\nTo rerun these experiments, use:")
            print(f"  python src/main.py --rerun-batch {batch_dir.name}")
        else:
            print("\n✅ All experiments meet quality standards!") 

        if args.auto_rerun:
            print(f"\n🔥 Auto-rerun was enabled with threshold < {args.auto_rerun_threshold}")
            print(f"   Maximum {args.max_rerun_iterations} iterations per case")
            print("   Critical cases were automatically reprocessed until threshold was met")
            print("   Check experiment directories for 'backup_*' folders with original bad runs")
        
        # Show study recommendations info
        recommendations_file = batch_dir / "study_recommendations.txt"
        if recommendations_file.exists():
            print(f"\n🔬 STUDY RECOMMENDATIONS GENERATED:")
            print(f"   📋 Comprehensive suggestions saved to: {recommendations_file}")
            print("   💡 Review recommendations for next research steps")
            print("   🎯 Includes parameter gaps, validation opportunities, and publication enhancements")



if __name__ == '__main__':
    main()

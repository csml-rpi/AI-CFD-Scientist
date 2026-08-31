#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def _ensure_foam_marker(case_dir: Path) -> Path:
    marker = case_dir / f"{case_dir.name}.foam"
    if not marker.exists():
        marker.touch()
    return marker


def _openfoam_mesh_as_single_block(mesh: Any) -> Any:
    """Reduce MultiBlock / composite OpenFOAM output to a single DataSet if possible."""
    import pyvista as pv  # type: ignore

    if isinstance(mesh, pv.MultiBlock):
        try:
            combined = mesh.combine()
            if combined.n_cells > 0:
                return combined
        except Exception:
            pass
        for i in range(len(mesh.keys())):
            try:
                b = mesh[i]
                if b is not None and getattr(b, "n_cells", 0) > 0:
                    return b
            except Exception:
                continue
    return mesh


def _set_reader_latest_time(reader: Any) -> None:
    tv = getattr(reader, "time_values", None) or []
    try:
        vals = [float(t) for t in list(tv)]
    except Exception:
        vals = []
    if not vals:
        return
    latest = max(vals)
    try:
        reader.set_active_time_value(latest)
    except Exception:
        reader.set_active_time_value(tv[-1])


def _extract_from_pyvista_robust(case_dir: Path) -> Dict[str, Any]:
    """
    Deterministic QoIs from PyVista: latest timestep, centreline streamwise velocity sample,
    and volume |U| on mesh DoFs (diagnostic only).
    """
    out: Dict[str, Any] = {}
    try:
        import numpy as np  # type: ignore
        import pyvista as pv  # type: ignore
    except Exception:
        return out

    foam = _ensure_foam_marker(case_dir)
    try:
        reader = pv.OpenFOAMReader(str(foam))
        _set_reader_latest_time(reader)
        data = reader.read()
        mesh = _openfoam_mesh_as_single_block(data)
        if mesh is None:
            return out

        out["mesh_n_cells"] = int(getattr(mesh, "n_cells", 0))
        out["mesh_n_points"] = int(getattr(mesh, "n_points", 0))
        out["pyvista_time_used"] = float(getattr(reader, "active_time_value", 0.0) or 0.0)

        U = None
        if "U" in mesh.cell_data:
            U = mesh.cell_data["U"]
        elif "U" in mesh.point_data:
            U = mesh.point_data["U"]
        if U is not None:
            umag = np.linalg.norm(np.asarray(U), axis=1)
            out["Umag_mean"] = float(np.mean(umag))
            out["Umag_max"] = float(np.max(umag))

        # Wall quantities, straight off the boundary patches. These are ordinary
        # fields sitting in the same reader as U — a wall patch commonly carries
        # wallShearStress and yPlus — but this extractor only ever opened U, so
        # a study asking for a wall metric (Cf) got nothing and the mesh gate
        # silently judged convergence on centreline velocity instead.
        try:
            boundary = data["boundary"] if "boundary" in data.keys() else None
        except Exception:
            boundary = None
        if boundary is not None:
            for patch_name in list(boundary.keys()):
                try:
                    patch = boundary[patch_name]
                except Exception:
                    continue
                fields = getattr(patch, "cell_data", {}) or {}
                if "wallShearStress" not in fields:
                    continue
                wss = np.asarray(fields["wallShearStress"])
                if wss.ndim != 2 or wss.shape[1] < 1 or wss.size == 0:
                    continue
                tau_x = wss[:, 0]
                out[f"{patch_name}_wallShearStress_x_mean"] = float(np.mean(tau_x))
                out[f"{patch_name}_wallShearStress_x_min"] = float(np.min(tau_x))
                out[f"{patch_name}_wallShearStress_x_max"] = float(np.max(tau_x))
                # Cf needs the case's reference velocity, which is a property of
                # the study rather than of the mesh; the LLM extractor derives it.
                # What is deterministic here is the shear itself.
                if "yPlus" in fields:
                    yplus = np.asarray(fields["yPlus"]).ravel()
                    if yplus.size:
                        out[f"{patch_name}_yPlus_mean"] = float(np.mean(yplus))
                        out[f"{patch_name}_yPlus_max"] = float(np.max(yplus))

        b = mesh.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = b
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        p0 = [xmin, cy, cz]
        p1 = [xmax, cy, cz]
        try:
            line = mesh.sample_over_line(p0, p1, resolution=400)
            if line.n_points > 0 and "U" in line.point_data:
                Ul = np.asarray(line.point_data["U"])
                ux = Ul[:, 0] if Ul.ndim == 2 and Ul.shape[1] >= 1 else Ul.ravel()
                out["centreline_Ux_mean"] = float(np.mean(ux))
                out["centreline_Ux_max"] = float(np.max(ux))
                out["centreline_Ux_min"] = float(np.min(ux))
                idx = np.where(np.diff(np.sign(ux)) != 0)[0]
                if len(idx) > 0:
                    pts = np.asarray(line.points)
                    x = pts[:, 0]
                    i0 = int(idx[0])
                    x0, x1 = float(x[i0]), float(x[i0 + 1])
                    u0, u1 = float(ux[i0]), float(ux[i0 + 1])
                    if abs(u1 - u0) > 1e-20:
                        out["reattachment_length"] = float(x0 - u0 * (x1 - x0) / (u1 - u0))
        except Exception:
            pass
    except Exception:
        return {}
    return out


def _extract_from_sampleline(case_dir: Path) -> Dict[str, Any]:
    """Legacy: OpenFOAM postProcessing/sampleLine text files (optional)."""
    out: Dict[str, Any] = {}
    sroot = case_dir / "postProcessing" / "sampleLine"
    if not sroot.exists():
        return out
    time_dirs = sorted(
        [p for p in sroot.iterdir() if p.is_dir()],
        key=lambda p: float(p.name) if p.name.replace(".", "", 1).isdigit() else -1.0,
    )
    if not time_dirs:
        return out
    latest = time_dirs[-1]
    xy = latest / "channelCentreLine.xy"
    if not xy.is_file():
        return out
    try:
        import numpy as np  # type: ignore

        data = np.loadtxt(str(xy), comments="#")
        if getattr(data, "ndim", 0) == 1:
            data = data.reshape(1, -1)
        if data.shape[1] >= 2:
            ux = data[:, 1]
            out["centreline_Ux_mean"] = float(np.mean(ux))
            out["centreline_Ux_max"] = float(np.max(ux))
            out["centreline_Ux_min"] = float(np.min(ux))
            idx = np.where(np.diff(np.sign(ux)) != 0)[0]
            if len(idx) > 0 and data.shape[1] >= 1:
                x = data[:, 0]
                i0 = int(idx[0])
                x0, x1 = float(x[i0]), float(x[i0 + 1])
                u0, u1 = float(ux[i0]), float(ux[i0 + 1])
                if abs(u1 - u0) > 1e-15:
                    out["reattachment_length"] = float(x0 - u0 * (x1 - x0) / (u1 - u0))
    except Exception:
        pass
    return out


def _extract_from_wall_shear(case_dir: Path) -> Dict[str, Any]:
    """
    Generic postProcessing QoI extractor.
    Catalogues all postProcessing data files, asks the LLM to identify and compute
    key scalar metrics. Works for any QoI: wall shear, heat flux, drag, viscosity
    profile statistics, outlet flux, pressure drop, etc.
    """
    pp_dir = case_dir / "postProcessing"
    if not pp_dir.exists():
        return {}

    # Collect all data files (small enough to sample)
    file_samples: Dict[str, str] = {}
    for p in sorted(pp_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix in {".png", ".jpg", ".pdf"}:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:800]
            file_samples[str(p.relative_to(case_dir))] = txt
        except Exception:
            pass

    if not file_samples:
        return {}

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from cfd_langgraph.llm.factory import create_langchain_llm
        from cfd_langgraph.config import get_settings
        from cfd_langgraph.utils import strip_json_fences

        sys_msg = (
            "You are a CFD post-processing expert. Given samples of postProcessing output files "
            "from an OpenFOAM case, extract the most physically meaningful scalar QoI statistics.\n"
            "Return a flat JSON object mapping descriptive snake_case keys to float values.\n"
            "Examples of keys: wall_shear_mean, wall_shear_max, heat_flux_mean, drag_coeff, "
            "pressure_drop, outlet_flux_mean, viscosity_rms, reattachment_length, y_plus_max.\n"
            "Only include keys you can actually compute from the data shown.\n"
            "Do NOT assume the QoI is wall shear stress — read the file contents and decide.\n"
            "Return ONLY raw JSON. No markdown. No commentary."
        )
        samples_text = json.dumps(file_samples, ensure_ascii=False)[:8000]
        user_msg = f"postProcessing file samples:\n{samples_text}"

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        txt = strip_json_fences(str(getattr(raw, "content", raw)).strip())
        result = json.loads(txt)
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


# Why the last llm_pyvista batch gave up. Kept so the caller can report it —
# a silent fallback to the deterministic extractor looks identical to success
# until a study-specific metric turns up missing several minutes later.
_LAST_BATCH_ERROR = ""


def _run_python_script(script_path: Path, cwd: Path, timeout_s: int = 600) -> tuple[int, str, str]:
    """Run a generated script. The path is resolved because ``cwd`` is usually
    the script's own directory: a relative path would be resolved against it a
    second time and doubled, which no file matches."""
    script_path = Path(script_path).resolve()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return -1, "", str(e)


def _llm_pyvista_batch_qoi(
    case_paths: List[Path],
    metrics: List[str],
    model: str,
    work_dir: Path,
    max_retries: int = 4,
    metric_hints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ask an LLM to write one PyVista script that loads all cases, samples fields, and writes QoIs to JSON.
    Returns map case_path_resolved_str -> qoi dict.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from cfd_langgraph.llm.factory import create_langchain_llm
    from cfd_langgraph.utils import strip_json_fences

    # Absolute, because the script is executed with cwd set to this directory.
    # A relative work_dir makes `python <relative script path>` resolve against
    # that same directory and doubles it —
    #   <work_dir>/<work_dir>/qoi_batch_script.py — which no file matches.
    # This is the identical mistake already fixed in
    # `agents/analysis_agent.py`; it existed in two places and only one was
    # repaired. It hid for four mesh-gate runs because every direct test used
    # an absolute --output under /tmp and so never doubled, while the gate
    # passes a repo-relative path and always did.
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_json = work_dir / "qoi_batch_out.json"
    script_path = work_dir / "qoi_batch_script.py"
    case_strs = [str(p.resolve()) for p in case_paths]
    out_json_path_str = str(out_json.resolve())

    foam_markers: Dict[str, str] = {}
    for cp in case_paths:
        marker = _ensure_foam_marker(cp.resolve())
        foam_markers[str(cp.resolve())] = str(marker)

    llm = create_langchain_llm(model=model, temperature=0.1)

    system_prompt = (
        "You write a single self-contained Python script for PyVista-based OpenFOAM QoI extraction.\n"
        "Rules:\n"
        "- Use only: pathlib, json, numpy, pyvista (as pv). No network.\n"
        "- For EACH case directory in CASE_DIRS, the .foam marker file has already been created; "
        "use the marker path from FOAM_MARKERS dict. Load via pv.OpenFOAMReader(marker_path). "
        "Select the **latest** simulation time (max of time_values; set_active_time_value).\n"
        "- **Mesh convergence / cross-case comparison:** Extract every requested QoI from that **final "
        "timestep only** — do not loop over or average intermediate write times. For each case use its own "
        "latest time; record it in `pyvista_time_used` for audit.\n"
        "\n"
        "SIMULATION DATA COMES FROM THE MESH. Read the case with PyVista and take every simulated "
        "quantity from the loaded data — not from OpenFOAM's postProcessing/ text files and not from "
        "log files; you do not need them.\n"
        "You MAY read files the metric definition below explicitly names — a reference/DNS data file, "
        "or a case dictionary such as constant/transportProperties for a constant. Read those with "
        "numpy/pathlib. What you must not do is invent a value for a constant the definition tells "
        "you where to find.\n"
        "\n"
        "WHAT THE READER GIVES YOU. reader.read() returns a MultiBlock, typically with an 'internalMesh' "
        "block and a 'boundary' block:\n"
        "  data['internalMesh']            -> volume fields, e.g. U, p, nut, nuTilda\n"
        "  data['boundary']                -> a MultiBlock of patches, addressable BY NAME\n"
        "  list(data['boundary'].keys())   -> e.g. ['inlet','outlet','topWall','bottomWall','defaultFaces']\n"
        "  data['boundary']['bottomWall']  -> that patch, carrying its own cell_data\n"
        "Wall quantities are ORDINARY FIELDS on the patch, not something to approximate: a wall patch "
        "commonly exposes wallShearStress, yPlus, U, p, nut alongside each other. Inspect "
        "`sorted(patch.cell_data.keys())` and use what is there. Patch face coordinates come from "
        "`patch.cell_centers().points`.\n"
        "\n"
        "THE REQUESTED METRICS ARE THE JOB. You are given METRICS — the exact quantities this study is "
        "judged on. Produce every one of them. They are not a menu and not examples; a study that asked "
        "for a metric cannot proceed on a different one. Derive them from the fields above, including "
        "standard definitions, for example a skin-friction coefficient from wall shear stress and the "
        "case's reference velocity (Cf = -2 * wallShearStress_x / Ub**2, with Ub read from the case's "
        "constant/transportProperties or its documented value). Emit a scalar per case: a summary "
        "(mean/RMS/extremum) of a profile is fine, but it must be that metric, computed from that field.\n"
        "Return null for a requested metric ONLY if the underlying field genuinely does not exist in the "
        "case. In that case also add a key '<metric>__why_null' with a one-line reason naming what you "
        "looked for and which patches/fields you found — a bare null with no explanation is a defect.\n"
        "\n"
        "- Also report `mesh_n_cells` and `mesh_n_points` for every case.\n"
        "- read() and combine MultiBlock meshes when needed (mesh.combine() or first non-empty block).\n"
        "- At the end, write ONLY valid JSON to OUT_JSON path with schema:\n"
        '  {"results": [{"case": "<exact path string from CASE_DIRS>", "qoi": {str->number}} ... ]}\n'
        "- use float values; every listed case must appear exactly once in results, same string as in CASE_DIRS.\n"
        "- Exit with status 0 on success; on failure write {\"error\": \"...\"} to OUT_JSON and sys.exit(1).\n"
        "Output ONLY raw Python. No markdown fences. First non-empty line must be an import.\n"
    )

    # The study's own definition of each metric, decided once and carried here
    # so the script never re-derives it. Two runs of this extractor on the same
    # case produced Cf = -2.755e-04 and -3.939e-07 — the second having quietly
    # used Ub = 1.0 instead of the case's 0.028, a factor of 1276.
    hint_block = ""
    if metric_hints:
        lines = "".join(
            f"  {h.get('name')}: {h.get('computation_hint') or h.get('description') or ''}\n"
            for h in metric_hints
        )
        hint_block = (
            "\nHOW EACH METRIC IS DEFINED FOR THIS STUDY — follow exactly, do not re-derive, "
            "do not substitute a default for any constant:\n" + lines + "\n"
        )

    global _LAST_BATCH_ERROR
    last_err = ""
    last_script = ""
    for attempt in range(1, max_retries + 1):
        user_prompt = (
            "Implement one Python script that extracts QoIs for every OpenFOAM case listed below.\n"
            f"CASE_DIRS = {case_strs!r}\n"
            f"FOAM_MARKERS = {foam_markers!r}\n"
            f"REQUESTED_METRIC_NAMES = {metrics!r}\n"
            f"{hint_block}"
            f"OUT_JSON = pathlib.Path({out_json_path_str!r})\n\n"
            "Start from `import json, pathlib`, `import numpy as np`, `import pyvista as pv`.\n"
            "Iterate over CASE_DIRS; for each, load via pv.OpenFOAMReader(FOAM_MARKERS[case_dir]).\n"
            "The .foam markers already exist — do NOT touch/create them.\n\n"
            f"Previous error:\n{last_err or '(none)'}\n\n"
            f"Previous script (for repair):\n{last_script[:12000] if last_script else '(none)'}\n"
        )
        try:
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            script_text = getattr(resp, "content", str(resp))
        except Exception as e:
            last_err = f"LLM invoke failed: {e}"
            time.sleep(min(2.0, 0.5 * attempt))
            continue

        script_text = strip_json_fences(script_text.strip())
        lines = script_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            script_text = "\n".join(lines[1:])
        lines = script_text.lstrip().splitlines()
        if lines and lines[0].strip().lower() in {"python", "bash", "sh"}:
            script_text = "\n".join(lines[1:])

        script_path.write_text(script_text, encoding="utf-8")
        if out_json.exists():
            try:
                out_json.unlink()
            except Exception:
                pass

        rc, out, err = _run_python_script(script_path, cwd=work_dir, timeout_s=900)
        if rc != 0 or not out_json.is_file():
            last_err = f"rc={rc}\nSTDOUT:\n{out[-4000:]}\nSTDERR:\n{err[-4000:]}"
            last_script = script_text
            time.sleep(min(2.0, 0.5 * attempt))
            continue

        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as e:
            last_err = f"Invalid JSON in OUT_JSON: {e}"
            last_script = script_text
            continue

        if "error" in payload and payload["error"]:
            last_err = str(payload["error"])
            last_script = script_text
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            last_err = "results not a list"
            last_script = script_text
            continue

        qmap: Dict[str, Dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            c = str(item.get("case", "")).strip()
            qoi = item.get("qoi")
            if not c or not isinstance(qoi, dict):
                continue
            qclean: Dict[str, Any] = {}
            for k, v in qoi.items():
                if v is None:
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
                    qclean[str(k)] = float(v) if isinstance(v, int) else v
            qmap[str(Path(c).resolve())] = qclean

        if len(qmap) < len(case_strs):
            missing = [c for c in case_strs if c not in qmap]
            last_err = f"Missing cases in LLM output: {missing}"
            last_script = script_text
            continue

        return qmap

    _LAST_BATCH_ERROR = last_err or "(no error recorded)"
    return {}


def extract_metrics(
    case_dir: Path,
    metrics: List[str],
    *,
    qoi_source: str = "pyvista",
    precomputed_qoi: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    qoi_source:
      - pyvista: PyVista robust only (no postProcessing parsing).
      - llm_pyvista: use precomputed_qoi from batch LLM run (must be provided for that case).
      - legacy: old path (PyVista weak + postProcessing grep + sampleLine/wallShear).
    """
    data: Dict[str, Any] = {
        "case": str(case_dir),
        "metrics": {},
        "post_files": [],
        "qoi": {},
        "analysis_source": qoi_source,
    }

    if qoi_source == "llm_pyvista" and precomputed_qoi is not None:
        merged = dict(_extract_from_pyvista_robust(case_dir))
        merged.update(precomputed_qoi)
        data["qoi"] = merged
        data["analysis_source"] = "llm_pyvista"
        return data

    if qoi_source == "legacy":
        data["qoi"].update(_extract_from_pyvista_robust(case_dir))
        post_dir = case_dir / "postProcessing"
        if post_dir.exists():
            for f in post_dir.rglob("*"):
                if f.is_file():
                    data["post_files"].append(str(f))
        data["qoi"].update(_extract_from_sampleline(case_dir))
        # LLM-driven: scans all postProcessing output, extracts any scalar QoIs it finds
        pp_qoi = _extract_from_wall_shear(case_dir)
        data["qoi"].update(pp_qoi)
        # Promote any extracted QoIs that match requested metrics
        for m in metrics:
            for key, val in pp_qoi.items():
                if m.lower().replace("_", "") in key.lower().replace("_", ""):
                    data["metrics"].setdefault(m, []).append(float(val))
        data["analysis_source"] = "legacy"
        return data

    data["qoi"].update(_extract_from_pyvista_robust(case_dir))
    data["analysis_source"] = "pyvista"
    return data


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Analyze multiple CFD cases.")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--metrics", type=str, default="Cd,Cl,y_plus")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--benchmark-data", type=str, default="")
    parser.add_argument("--reference-manifest", type=str, default="",
                        help="Path to reference_data_manifest.json; CSV/tabular reference files are read and included in benchmark context.")
    parser.add_argument("--topic", type=str, default="")
    parser.add_argument(
        "--cross-objectives-json",
        type=str,
        default="",
        help="Optional analysis plan JSON containing cross_case_objectives.",
    )
    parser.add_argument(
        "--qoi-source",
        choices=["pyvista", "llm_pyvista", "legacy"],
        default="pyvista",
        help="pyvista=robust OpenFOAM reader only; llm_pyvista=LLM-written PyVista batch script; legacy=old postProcessing path.",
    )
    parser.add_argument(
        "--metric-spec",
        default="",
        help="JSON file of study metric specs (name + computation_hint). Carried verbatim into "
             "the extractor prompt so constants are never re-derived.",
    )
    args = parser.parse_args()

    from cfd_langgraph.agents.analysis_agent import AnalysisAgent
    from cfd_langgraph.config import get_settings

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved = [Path(c).resolve() for c in args.cases]
    llm_map: Dict[str, Dict[str, Any]] = {}
    if args.qoi_source == "llm_pyvista":
        work_dir = out_path.parent / f".qoi_llm_{out_path.stem}"
        hints = None
        if args.metric_spec:
            try:
                hints = json.loads(Path(args.metric_spec).read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[analyze] could not read --metric-spec: {exc}", file=sys.stderr)
        llm_map = _llm_pyvista_batch_qoi(
            resolved, metrics, get_settings().model, work_dir, metric_hints=hints
        )
        if not llm_map:
            print(
                "[analyze] llm_pyvista batch failed after all attempts; falling back to the "
                "deterministic pyvista extractor, which cannot compute study-specific metrics. "
                f"Last error:\n{_LAST_BATCH_ERROR[:1500]}",
                file=sys.stderr,
            )

    raw: List[Dict[str, Any]] = []
    for p in resolved:
        key = str(p)
        if args.qoi_source == "llm_pyvista" and key in llm_map:
            raw.append(
                extract_metrics(
                    p,
                    metrics,
                    qoi_source="llm_pyvista",
                    precomputed_qoi=llm_map[key],
                )
            )
        elif args.qoi_source == "llm_pyvista":
            raw.append(extract_metrics(p, metrics, qoi_source="pyvista"))
        else:
            raw.append(extract_metrics(p, metrics, qoi_source=args.qoi_source))

    benchmark = {}
    if args.benchmark_data:
        bpath = Path(args.benchmark_data)
        if bpath.exists():
            benchmark = json.loads(bpath.read_text(encoding="utf-8"))

    # Augment benchmark with the LLM interpretation from the reference data manifest.
    # The LLM has already read every reference file (any format) and produced a
    # structured description with quantities, usage guidance, and verbatim data excerpt.
    if args.reference_manifest:
        rmp = Path(args.reference_manifest)
        if rmp.exists():
            try:
                rm = json.loads(rmp.read_text(encoding="utf-8"))
                llm_interp = rm.get("llm_interpretation", {})
                if llm_interp and llm_interp.get("status") not in {"no_readable_files", "llm_failed"}:
                    benchmark["reference_data_interpretation"] = llm_interp
                    print(
                        f"[analyze] reference data loaded — quantities: "
                        f"{llm_interp.get('quantities', [])}"
                    )
                else:
                    print("[analyze] warning: no usable LLM interpretation in reference manifest",
                          file=sys.stderr)
            except Exception as e:
                print(f"[analyze] warning: failed to parse reference manifest: {e}", file=sys.stderr)

    bundle = {"metrics": raw, "benchmark": benchmark}
    agent = AnalysisAgent(model=get_settings().model)
    cross_objectives: List[str] = []
    if args.cross_objectives_json:
        cp = Path(args.cross_objectives_json)
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    objs = payload.get("cross_case_objectives", [])
                    if isinstance(objs, list):
                        cross_objectives = [str(x).strip() for x in objs if str(x).strip()]
            except Exception:
                pass

    # Run cross-experiment processing so analysis stage can generate true across-case figures/tables.
    experiments_for_cross: List[Dict[str, Any]] = []
    for p in resolved:
        experiments_for_cross.append(
            {
                "simulation_id": p.name,
                "case_name": p.name,
                "description": "",
                "user_requirement": "",
                "sim_dir": str(p),
                "foam_output_dir": str(p),
            }
        )
    cross_topic = args.topic or "CFD cross-experiment analysis"
    if cross_objectives:
        cross_topic = cross_topic + "\n\nCross-case objectives:\n" + "\n".join(f"- {o}" for o in cross_objectives)
    cross_proc = agent.run_cross_experiment_data_processing(
        topic=cross_topic,
        experiments=experiments_for_cross,
        out_dir=out_path.parent,
        verbose=False,
        max_retries=10,
    )

    ref_rule = (
        "MANDATORY VISUALIZATION RULE: Every plot that shows simulation QoI values "
        "(wall quantities, force coefficients, profiles, error metrics, or any scalar "
        "extracted from the simulation) MUST also plot the reference/DNS/experimental "
        "data on the same axes. The reference data is in benchmark['reference_data_interpretation']. "
        "A plot without the ground-truth curve alongside simulation results is incomplete "
        "and unacceptable — it makes model comparison impossible. "
        "Label reference curves clearly (e.g. 'DNS', 'Experiment', 'Reference'). "
        "SIGN/NORMALIZATION CHECK: Before plotting, verify that simulation and reference "
        "QoI share the same sign convention and normalization. If the simulation curve is "
        "globally inverted relative to reference (negative correlation), the sign convention "
        "differs — negate the simulation curve and document the reason. "
        "If magnitudes differ by orders of magnitude, the normalization scale is wrong — "
        "use the authoritative reference velocity/length/density from the case parameters."
    )
    analysis_text = agent.analyze_text_bundle(
        batch_name="cfd_analysis",
        bundle_text=json.dumps(bundle, indent=2),
        extra_context=(
            f"metrics={metrics} qoi_source={args.qoi_source} "
            f"cross_case_objectives={cross_objectives} "
            f"cross_processing={json.dumps(cross_proc, ensure_ascii=False)[:6000]}\n\n"
            f"{ref_rule}"
        ),
    )

    result = {
        "metrics": raw,
        "benchmark": benchmark,
        "analysis": analysis_text,
        "qoi_source": args.qoi_source,
        "cross_case_objectives": cross_objectives,
        "cross_experiment_processing": cross_proc,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(analysis_text[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

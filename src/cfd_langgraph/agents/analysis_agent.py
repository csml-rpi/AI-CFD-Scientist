from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.viz_creator import viz_creator

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def _downscale_image_bytes(b: bytes, max_dimension: int, fmt: str = "PNG") -> bytes:
    """Downscale image bytes so no dimension exceeds max_dimension. Returns bytes."""
    if Image is None:
        return b
    img = Image.open(io.BytesIO(b))
    w, h = img.size
    if w <= max_dimension and h <= max_dimension:
        return b
    scale = min(max_dimension / w, max_dimension / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if fmt.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _image_paths_to_blocks(
    image_paths: List[Path],
    max_images: int = 12,
    max_dimension: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build content blocks for vision LLM: image_url (base64). Optionally downscale if max_dimension set."""
    blocks: List[Dict[str, Any]] = []
    for p in image_paths[:max_images]:
        if not p.exists() or not p.is_file():
            continue
        try:
            b = p.read_bytes()
            if max_dimension is not None and Image is not None:
                ext = p.suffix.lower()
                fmt = "PNG" if ext == ".png" else "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
                b = _downscale_image_bytes(b, max_dimension, fmt)
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/gif"
            url = f"data:{mime};base64,{b64}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception:
            continue
    return blocks


def _is_image_dimension_error(exc: BaseException) -> bool:
    """Check if exception is Bedrock's image-dimension-too-large error."""
    msg = str(exc).lower()
    return "2000" in str(exc) and ("dimension" in msg or "pixels" in msg)

def _strip_json_fences(text: str) -> str:
    """
    Strip common markdown code fences around JSON, e.g.:
    ```json
    {...}
    ```
    Returns best-effort raw JSON string.
    """
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence line
    first_nl = s.find("\n")
    if first_nl == -1:
        return s.strip("`").strip()
    s2 = s[first_nl + 1 :]
    # Drop a trailing fence if present
    end = s2.rfind("```")
    if end != -1:
        s2 = s2[:end]
    return s2.strip()


class AnalysisAgent:
    """Max distinct visualization types to suggest per experiment (tweak as needed)."""
    MAX_EXP_VIZ = 10

    def __init__(self, model: str):
        self.model = model
        self.llm = create_langchain_llm(model=model, temperature=0.0)

    @staticmethod
    def _experiment_idea_text(ex: Dict[str, Any]) -> str:
        """Best-effort compact idea text from experiment_idea/case_data/description."""
        idea = ex.get("experiment_idea", None) or ex.get("case_data", None)
        if isinstance(idea, dict):
            parts: List[str] = []
            name = str(idea.get("name", "") or "").strip()
            notes = str(idea.get("notes", "") or "").strip()
            params = idea.get("parameters", {}) if isinstance(idea.get("parameters", {}), dict) else {}
            if name:
                parts.append(f"name={name}")
            if notes:
                parts.append(f"notes={notes[:600]}")
            if params:
                try:
                    parts.append("parameters=" + json.dumps(params, ensure_ascii=False)[:1200])
                except Exception:
                    parts.append(f"parameters={str(params)[:1200]}")
            if parts:
                return "; ".join(parts)
        if isinstance(idea, str) and idea.strip():
            return idea.strip()[:1800]
        return str(ex.get("description", "") or "")[:1000]

    def analyze_text_bundle(self, batch_name: str, bundle_text: str, extra_context: Optional[str] = None) -> str:
        system = "You are a CFD expert specializing in simulation diagnostics and scientific analysis."
        user = (
            "Analyze this CFD batch named {batch_name}.\n"
            "Context:\n{extra_context}\n\n"
            "Bundle:\n{bundle_text}\n\n"
            "Return detailed per-case + cross-case analysis, observations, conclusions, and publication-ready figure suggestions."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user),
        ])
        chain = prompt | self.llm
        return chain.invoke({"batch_name": batch_name, "bundle_text": bundle_text, "extra_context": extra_context or ""}).content

    def save_analysis(self, out_path: Path, text: str, topic: Optional[str] = None) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if topic:
            text = f"# Study Topic\n\n{topic}\n\n---\n\n{text}"
        out_path.write_text(text, encoding="utf-8")

    def generate_plots_from_foam_data(
        self,
        foam_data_path: Path,
        request_text: str,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """
        Generate visualizations using the central viz_creator.
        foam_data_path: path to .foam marker (e.g. case.foam); its parent is foam_output_dir.
        request_text: what to visualize (e.g. publication-quality contours, slices, etc.).
        out_dir: where viz_script.py and PNGs are saved (e.g. images_for_paper).
        """
        foam_output_dir = foam_data_path.parent
        result = viz_creator(
            model=self.model,
            foam_output_dir=foam_output_dir,
            viz_dir=out_dir,
            what_to_visualize=request_text,
            user_requirement=request_text,
        )
        return {
            "ok": result.get("ok", False),
            "images": result.get("images", []),
            "foam_data": str(foam_data_path),
            "viz_dir": result.get("viz_dir", str(out_dir)),
            "attempts": result.get("attempts", 0),
            "error": result.get("last_error", ""),
            "results": [{"ok": True, "output": p} for p in result.get("images", [])],
        }

    def generate_viz_for_experiment(
        self,
        foam_output_dir: Path,
        viz_dir: Path,
        user_requirement: str,
        what_to_visualize: str,
        reference_viz_script: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate visualizations for one experiment using the central viz_creator.
        Use this when you have the experiment's user requirement and a specific
        viz plan (e.g. from analysis or LLM). Optionally pass interpreter-created
        viz code as reference_viz_script.
        """
        result = viz_creator(
            model=self.model,
            foam_output_dir=foam_output_dir,
            viz_dir=viz_dir,
            what_to_visualize=what_to_visualize,
            user_requirement=user_requirement,
            reference_viz_script=reference_viz_script,
        )
        return {
            "ok": result.get("ok", False),
            "images": result.get("images", []),
            "attempts": result.get("attempts", 0),
            "error": result.get("last_error", ""),
            "results": [{"ok": True, "output": p} for p in result.get("images", [])],
        }

    def decide_visualizations(
        self,
        experiments: List[Dict[str, Any]],
        topic: str,
    ) -> str:
        """
        Ask LLM: given user requirements for each experiment, decide what visualizations
        are needed for the paper (e.g. velocity contour, pressure contour, line plots, streamlines).
        experiments: list of {simulation_id, case_name, user_requirement}.
        Returns a short spec string to pass to viz_creator (what_to_visualize).
        """
        max_viz = getattr(self, "MAX_EXP_VIZ", 10)
        system = (
            "You are a CFD expert preparing figures for a paper. Given the experiment descriptions "
            "and user requirements below, decide exactly what visualizations are needed for the analysis. "
            "List specific types: e.g. velocity magnitude contour at selected times, pressure contour, "
            "centreline velocity profile (line plot), cross-stream profiles, streamlines, etc. "
            "CRITICAL: If the requested analysis focuses on localized flow features (recirculation bubble, "
            "reattachment region, shear layer, step lip/corner, jets, wall quantities like Cp/Cf/y+, etc.), "
            "your visualization spec MUST include zoomed-in crops/frames around those features so they are "
            "readable in a journal figure. If needed, also include a separate full-domain/context view, but "
            "do NOT rely only on a tiny full-domain view where the feature becomes illegible."
            f"CRITICAL: suggest at most {max_viz} distinct visualization types per experiment; "
            "this is a strict upper bound and you must NOT exceed it. "
            "Be concise but precise so a script writer can implement them. Output only the visualization "
            "specification (no code, no markdown)."
        )
        parts = [f"Study topic: {topic}\n"]
        for ex in experiments:
            idea_text = self._experiment_idea_text(ex)
            parts.append(
                f"Experiment {ex.get('simulation_id', '?')} ({ex.get('case_name', '')}):\n"
                f"Idea/context: {idea_text}\n"
                f"{ex.get('user_requirement', '')[:2000]}\n"
            )
        parts.append(f"\nWhat visualizations are needed for the paper? List at most {max_viz} distinct types per experiment.")
        user = "\n".join(parts)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{user}")])
        out = (prompt | self.llm).invoke({"user": user})
        return getattr(out, "content", str(out)).strip()

    def create_analysis_viz_for_experiments(
        self,
        experiments: List[Dict[str, Any]],
        viz_spec: str,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        For each experiment: load interpreter viz script as reference, call viz_creator
        with viz_spec and that reference, save outputs to sim_dir/analysis_viz.
        experiments: list of {simulation_id, case_name, user_requirement, sim_dir: Path, foam_output_dir: Path}.
        Returns list of {simulation_id, case_name, visualization: {...}, images: [...], viz_dir: str}.
        """
        results: List[Dict[str, Any]] = []
        for ex in experiments:
            sim_id = ex.get("simulation_id", "unknown")
            case_name = ex.get("case_name", sim_id)
            user_req = ex.get("user_requirement", "")
            sim_dir = ex.get("sim_dir")
            foam_output_dir = ex.get("foam_output_dir")
            if not sim_dir or not foam_output_dir:
                results.append({
                    "simulation_id": sim_id,
                    "case_name": case_name,
                    "visualization": {"ok": False, "error": "missing sim_dir or foam_output_dir"},
                    "images": [],
                    "viz_dir": "",
                })
                continue
            sim_dir = Path(sim_dir)
            foam_output_dir = Path(foam_output_dir)
            viz_dir = sim_dir / "analysis_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)

            reference_code = ""
            ref_script = foam_output_dir / "interpreter_viz" / "viz_script.py"
            if ref_script.is_file():
                reference_code = ref_script.read_text(encoding="utf-8")

            result = viz_creator(
                model=self.model,
                foam_output_dir=foam_output_dir,
                viz_dir=viz_dir,
                what_to_visualize=viz_spec,
                user_requirement=user_req,
                reference_viz_script=reference_code or None,
            )
            viz_summary = {
                "ok": result.get("ok", False),
                "images": result.get("images", []),
                "viz_dir": result.get("viz_dir", str(viz_dir)),
                "attempts": result.get("attempts", 0),
                "error": result.get("last_error", ""),
            }
            results.append({
                "simulation_id": sim_id,
                "case_name": case_name,
                "visualization": viz_summary,
                "images": viz_summary["images"],
                "viz_dir": viz_summary["viz_dir"],
            })
        return results

    def run_analysis_with_images(
        self,
        experiments_with_images: List[Dict[str, Any]],
        topic: str,
        max_images_per_experiment: int = 8,
        verbose: bool = False,
    ) -> str:
        """
        Invoke vision LLM with all experiment images and user requirements; get back
        analysis text (what do you see, what is happening between experiments).
        On Bedrock image-dimension error (>2000px), retries with downscaled images.
        experiments_with_images: list of {simulation_id, case_name, user_requirement, image_paths: List[Path]}.
        """
        system = (
            "You are a CFD expert writing the analysis section for a paper. You will see the user "
            "requirements for each experiment and the generated visualizations. Write a concise but "
            "thorough analysis: (1) What do you see in each experiment (flow features, trends)? "
            "(2) How do results compare across experiments? (3) Any conclusions or recommendations. "
            "Write in clear prose suitable for a paper Methods/Results or Analysis section."
        )
        max_retries = 4
        max_dimensions = [None, 1999, 1500, 1000]  # None = no limit; then progressively smaller

        for attempt in range(max_retries):
            max_dim = max_dimensions[attempt]
            content_parts: List[Any] = [{"type": "text", "text": f"Study topic: {topic}\n\n"}]
            for ex in experiments_with_images:
                sim_id = ex.get("simulation_id", "?")
                case_name = ex.get("case_name", "")
                user_req = (ex.get("user_requirement") or "")[:1500]
                idea_text = (ex.get("experiment_idea_text") or "")[:1800]
                paths = ex.get("image_paths") or []
                content_parts.append({
                    "type": "text",
                    "text": (
                        f"--- Experiment: {sim_id} ({case_name}) ---\n"
                        f"Experiment idea/context: {idea_text}\n"
                        f"User requirement: {user_req}\n\n"
                    ),
                })
                blocks = _image_paths_to_blocks(
                    [Path(p) for p in paths],
                    max_images=max_images_per_experiment,
                    max_dimension=max_dim,
                )
                content_parts.extend(blocks)
            content_parts.append({
                "type": "text",
                "text": "\n--- End of figures ---\nProvide the analysis as requested above.",
            })
            messages = [SystemMessage(content=system), HumanMessage(content=content_parts)]
            try:
                out = self.llm.invoke(messages)
                return getattr(out, "content", str(out)).strip()
            except Exception as e:
                if _is_image_dimension_error(e) and attempt < max_retries - 1:
                    if verbose:
                        print(
                            f"[Analysis] Image dimension error (attempt {attempt + 1}), "
                            f"retrying with max_dimension={max_dimensions[attempt + 1]}...",
                            flush=True,
                        )
                    continue
                raise

    def _build_experiment_inventory(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact manifest of available artifacts per experiment for cross-experiment data processing."""
        manifest: List[Dict[str, Any]] = []
        for ex in experiments:
            sim_id = str(ex.get("simulation_id", ""))
            if not sim_id:
                continue
            sim_dir = Path(ex.get("sim_dir")) if ex.get("sim_dir") else None
            foam_dir = Path(ex.get("foam_output_dir")) if ex.get("foam_output_dir") else None
            item: Dict[str, Any] = {
                "simulation_id": sim_id,
                "case_name": ex.get("case_name", sim_id),
                "description": ex.get("description", ""),
                "experiment_idea": self._experiment_idea_text(ex),
                "user_requirement": ex.get("user_requirement", ""),
                "sim_dir": str(sim_dir) if sim_dir else "",
                "foam_output_dir": str(foam_dir) if foam_dir else "",
                "time_dirs": [],
                "files_0": [],
                "files_system": [],
                "files_constant": [],
            }
            if foam_dir and foam_dir.is_dir():
                # Numeric time folders
                times: List[str] = []
                for d in foam_dir.iterdir():
                    if not d.is_dir():
                        continue
                    try:
                        float(d.name)
                        times.append(d.name)
                    except Exception:
                        continue
                item["time_dirs"] = sorted(times, key=lambda x: float(x))[:50]
                # Key config/input files
                for sub, key in [("0", "files_0"), ("system", "files_system"), ("constant", "files_constant")]:
                    p = foam_dir / sub
                    if p.is_dir():
                        names = []
                        for f in sorted(p.iterdir()):
                            if sub == "constant" and f.name == "polyMesh":
                                continue
                            if f.is_file():
                                names.append(f.name)
                        item[key] = names[:80]
            manifest.append(item)
        return manifest

    def _interpret_cross_experiment_outputs(
        self,
        topic: str,
        proc_dir: Path,
        objectives: List[str],
        script_text: str,
        report_text: str,
        image_paths: List[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Vision+text interpretation for cross-experiment processing artifacts.
        Reads the generated script/report and inspects cross-experiment plots.
        """
        system = (
            "You are a CFD research analyst focused on cross-experiment synthesis.\n"
            "Given the study topic, cross-experiment objectives, generated processing script, report text, "
            "and plot images, write a concise but rigorous interpretation suitable for Results/Discussion.\n"
            "Include: (1) what quantitative results were actually obtained, (2) fitted relations/equations "
            "and their credibility limits, (3) image-backed observations, (4) cautions or missing evidence, "
            "(5) what the writer should include verbatim vs qualify.\n"
            "Return plain markdown prose (no JSON)."
        )
        content_parts: List[Any] = [
            {"type": "text", "text": f"Study topic:\n{topic}\n\n"},
            {"type": "text", "text": "Cross-experiment objectives:\n" + ("\n".join(f"- {o}" for o in objectives) if objectives else "- (none)") + "\n\n"},
            {"type": "text", "text": "Generated cross-experiment script (truncated):\n```python\n" + (script_text or "")[:25000] + "\n```\n\n"},
            {"type": "text", "text": "Cross-experiment report (truncated):\n" + (report_text or "")[:25000] + "\n\n"},
            {"type": "text", "text": "Now inspect attached cross-experiment plots and provide interpretation.\n"},
        ]
        blocks = _image_paths_to_blocks([Path(p) for p in image_paths], max_images=12, max_dimension=1999)
        content_parts.extend(blocks)
        content_parts.append({"type": "text", "text": "\nEnd of artifacts. Write cross-experiment interpretation now."})

        try:
            out = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=content_parts)])
            text = str(getattr(out, "content", str(out))).strip()
        except Exception as e:
            text = (
                "Cross-experiment interpretation generation failed.\n\n"
                f"Error: {e}\n\n"
                "Fallback summary: use data_processing_report.md directly in writer context."
            )
        interp_path = proc_dir / "cross_experiment_interpretation.md"
        try:
            interp_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        if verbose:
            print(f"[Analysis] cross_experiment: interpretation saved -> {interp_path}", flush=True)
        return {"interpretation_text": text, "interpretation_path": str(interp_path)}

    def run_cross_experiment_data_processing(
        self,
        topic: str,
        experiments: List[Dict[str, Any]],
        out_dir: Path,
        verbose: bool = False,
        max_retries: int = 10,
    ) -> Dict[str, Any]:
        """
        Decide if additional cross-experiment quantitative post-processing is needed.
        If needed, generate and execute a Python script with a repair loop.
        """
        proc_dir = out_dir / "cross_experiment_analysis"
        proc_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._build_experiment_inventory(experiments)

        print(
            f"[Analysis] cross_experiment: starting (experiments={len(experiments)}, dir={proc_dir})",
            flush=True,
        )

        planner_system = (
            "You are a CFD cross-experiment data-processing planner.\n"
            "Given topic + experiment requirements + available artifacts, decide whether additional quantitative "
            "post-processing is needed beyond image-based interpretation.\n"
            "Examples: trend/correlation plots, regression fits, summary tables, derived metrics.\n"
            "Return ONLY JSON with keys: needs_processing (bool), objectives (list[str]), rationale (str)."
        )
        planner_user = (
            f"Study topic:\n{topic}\n\n"
            f"Experiment manifest:\n{json.dumps(manifest, ensure_ascii=False)}\n\n"
            "Decide if cross-experiment processing is needed for stronger analysis/paper evidence."
        )
        needs_processing = False
        objectives: List[str] = []
        rationale = ""
        plan_raw = ""
        planner_raw_path = proc_dir / "planner_raw.txt"
        planner_json_path = proc_dir / "planner_parsed.json"
        try:
            plan_raw = (self.llm.invoke([
                SystemMessage(content=planner_system),
                HumanMessage(content=planner_user),
            ])).content
            plan_raw = str(plan_raw)
            try:
                planner_raw_path.write_text(plan_raw, encoding="utf-8")
            except Exception:
                pass
            plan = json.loads(_strip_json_fences(plan_raw))
            needs_processing = bool(plan.get("needs_processing", False))
            objectives = [str(x) for x in (plan.get("objectives", []) or []) if str(x).strip()]
            rationale = str(plan.get("rationale", "") or "")
            try:
                planner_json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            needs_processing = False
            objectives = []
            rationale = "Planner output parse failed; skipped extra processing."
            try:
                if plan_raw:
                    planner_raw_path.write_text(str(plan_raw), encoding="utf-8")
            except Exception:
                pass
            raw_s = str(plan_raw or "")
            if raw_s:
                head = raw_s[:500].replace("\n", "\\n")
                tail = raw_s[-800:].replace("\n", "\\n")
                print(
                    "[Analysis] cross_experiment: planner parse failed; saved planner_raw.txt; "
                    f"error={type(e).__name__}: {e}; head={head!r}; tail={tail!r}",
                    flush=True,
                )
            else:
                print(
                    "[Analysis] cross_experiment: planner parse failed; no raw output captured; "
                    f"error={type(e).__name__}: {e}",
                    flush=True,
                )

        rationale_short = (rationale or "")[:200] + ("..." if len(rationale or "") > 200 else "")
        print(
            f"[Analysis] cross_experiment: planner -> needs_processing={needs_processing}"
            + (f"; objectives={len(objectives)}" if objectives else "")
            + (f" | {rationale_short!r}" if rationale_short else ""),
            flush=True,
        )
        if verbose and objectives:
            for i, obj in enumerate(objectives[:8], 1):
                print(f"[Analysis] cross_experiment:   objective {i}: {obj[:160]}", flush=True)
            if len(objectives) > 8:
                print(f"[Analysis] cross_experiment:   ... and {len(objectives) - 8} more", flush=True)

        if not needs_processing:
            report_path = proc_dir / "data_processing_report.md"
            report_text = (
                "# Cross-Experiment Data Processing\n\n"
                "No additional script-based cross-experiment processing was required.\n\n"
                f"Reason: {rationale or 'not needed by planner'}\n"
            )
            report_path.write_text(report_text, encoding="utf-8")
            interp = self._interpret_cross_experiment_outputs(
                topic=topic,
                proc_dir=proc_dir,
                objectives=objectives,
                script_text="",
                report_text=report_text,
                image_paths=[],
                verbose=verbose,
            )
            print(
                f"[Analysis] cross_experiment: no Python script generated; report={report_path}",
                flush=True,
            )
            return {
                "ok": True,
                "needs_processing": False,
                "objectives": objectives,
                "rationale": rationale,
                "script_path": None,
                "report_path": str(report_path),
                "images": [],
                "attempts": 0,
                "error": "",
                "interpretation_text": interp.get("interpretation_text", ""),
                "interpretation_path": interp.get("interpretation_path"),
            }

        script_system = (
            "You write robust Python post-processing scripts for CFD cross-experiment analysis.\n"
            "Return ONLY raw Python code (no markdown).\n"
            "Script requirements:\n"
            "- Read available experiment artifacts from the manifest.\n"
            "- Compute requested cross-experiment metrics/trends only when data is available.\n"
            "- Save outputs under output_dir: (1) data_processing_report.md, (2) optional plots PNG.\n"
            "- Be defensive: if some files are missing, continue with available data and explain limitations in report.\n"
            "- Prefer PyVista for reading OpenFOAM case data/fields when needed (do NOT depend on ParaView GUI).\n"
            "- Exit 0 on success.\n\n"
            "PyVista OpenFOAM loading snippet (use as a starting point):\n"
            "```python\n"
            "import pyvista as pv\n"
            "from pathlib import Path\n"
            "foam_output_dir = Path('/path/to/foam_output')\n"
            "# If no .foam exists, create one; PyVista uses it as a marker\n"
            "marker = foam_output_dir / f\"{foam_output_dir.name}.foam\"\n"
            "marker.touch(exist_ok=True)\n"
            "mesh = pv.read(str(marker))\n"
            "available_arrays = getattr(mesh, 'array_names', []) or []\n"
            "```\n"
        )
        user_tpl = (
            "Topic:\n{topic}\n\n"
            "Objectives:\n{objectives}\n\n"
            "Experiment manifest:\n{manifest}\n\n"
            "output_dir:\n{output_dir}\n\n"
            "Previous error:\n{previous_error}\n\n"
            "Previous script:\n{previous_script}\n\n"
            "Write a complete Python script."
        )

        last_error = ""
        last_script = ""
        script_path = proc_dir / "data_processing_script.py"
        print(
            f"[Analysis] cross_experiment: generating & running Python script (max {max_retries} attempts) -> {script_path}",
            flush=True,
        )
        for attempt in range(1, max_retries + 1):
            if verbose:
                print(f"[Analysis] cross_experiment: attempt {attempt}/{max_retries} (LLM script + execute)", flush=True)
            user = user_tpl.format(
                topic=topic,
                objectives=json.dumps(objectives, ensure_ascii=False),
                manifest=json.dumps(manifest, ensure_ascii=False),
                output_dir=str(proc_dir),
                previous_error=last_error or "(none)",
                previous_script=last_script or "(none)",
            )
            try:
                resp = self.llm.invoke([SystemMessage(content=script_system), HumanMessage(content=user)])
                script_text = getattr(resp, "content", str(resp))
            except Exception as e:
                last_error = f"LLM script generation failed: {e}"
                continue

            script_text = str(script_text).strip()
            if script_text.startswith("```"):
                script_text = script_text.strip("`")
                if script_text.lower().startswith("python"):
                    script_text = script_text[6:]
            script_path.write_text(script_text, encoding="utf-8")
            last_script = script_text
            if verbose:
                print(f"[Analysis] cross_experiment: wrote script ({len(script_text)} chars) -> {script_path}", flush=True)

            try:
                if verbose:
                    print(f"[Analysis] cross_experiment: running {sys.executable} {script_path} (cwd={out_dir})", flush=True)
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(out_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                rc = int(proc.returncode)
                if rc != 0:
                    stdout_tail = (proc.stdout or "")[-2000:]
                    stderr_tail = (proc.stderr or "")[-4000:]
                    last_error = (
                        f"Return code: {rc}\n"
                        f"STDOUT (tail):\n{stdout_tail}\n\n"
                        f"STDERR (tail):\n{stderr_tail}"
                    )

                    # Persist per-attempt artifacts for debugging + deterministic repair prompts.
                    try:
                        (proc_dir / f"attempt_{attempt:02d}_script.py").write_text(last_script or "", encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        (proc_dir / f"attempt_{attempt:02d}_error.txt").write_text(last_error, encoding="utf-8")
                    except Exception:
                        pass

                    print(f"[Analysis] cross_experiment: attempt {attempt} script failed rc={rc}", flush=True)
                    # Always show the error tail in terminal so user can see it immediately.
                    print(f"[Analysis] cross_experiment: attempt {attempt} error (tail):\n{last_error[-2000:]}", flush=True)
                    continue
            except Exception as e:
                last_error = f"Script runner exception: {e}"
                print(f"[Analysis] cross_experiment: attempt {attempt} runner exception: {e}", flush=True)
                continue

            report_path = proc_dir / "data_processing_report.md"
            if not report_path.exists():
                last_error = "Script succeeded but did not create data_processing_report.md"
                print(f"[Analysis] cross_experiment: attempt {attempt} missing data_processing_report.md after exit 0", flush=True)
                continue

            pngs = sorted(str(p) for p in proc_dir.glob("*.png") if p.is_file())
            print(
                f"[Analysis] cross_experiment: success after {attempt} attempt(s); report={report_path} pngs={len(pngs)}",
                flush=True,
            )
            if verbose and pngs:
                for p in pngs[:10]:
                    print(f"[Analysis] cross_experiment:   plot {p}", flush=True)
                if len(pngs) > 10:
                    print(f"[Analysis] cross_experiment:   ... and {len(pngs) - 10} more", flush=True)
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except Exception:
                report_text = ""
            interp = self._interpret_cross_experiment_outputs(
                topic=topic,
                proc_dir=proc_dir,
                objectives=objectives,
                script_text=last_script,
                report_text=report_text,
                image_paths=pngs,
                verbose=verbose,
            )
            return {
                "ok": True,
                "needs_processing": True,
                "objectives": objectives,
                "rationale": rationale,
                "script_path": str(script_path),
                "report_path": str(report_path),
                "images": pngs,
                "attempts": attempt,
                "error": "",
                "interpretation_text": interp.get("interpretation_text", ""),
                "interpretation_path": interp.get("interpretation_path"),
            }

        # Failure fallback report
        fallback_report = proc_dir / "data_processing_report.md"
        fallback_report.write_text(
            "# Cross-Experiment Data Processing\n\n"
            "Script-based post-processing could not be completed.\n\n"
            f"Last error:\n\n{last_error}\n",
            encoding="utf-8",
        )
        print(
            f"[Analysis] cross_experiment: FAILED after {max_retries} attempts; report={fallback_report}",
            flush=True,
        )
        if verbose and last_error:
            print(f"[Analysis] cross_experiment: last_error (tail):\n{last_error[-1500:]}", flush=True)
        try:
            fallback_text = fallback_report.read_text(encoding="utf-8")
        except Exception:
            fallback_text = ""
        interp = self._interpret_cross_experiment_outputs(
            topic=topic,
            proc_dir=proc_dir,
            objectives=objectives,
            script_text=last_script,
            report_text=fallback_text,
            image_paths=[],
            verbose=verbose,
        )
        return {
            "ok": False,
            "needs_processing": True,
            "objectives": objectives,
            "rationale": rationale,
            "script_path": str(script_path) if script_path.exists() else None,
            "report_path": str(fallback_report),
            "images": [],
            "attempts": max_retries,
            "error": last_error,
            "interpretation_text": interp.get("interpretation_text", ""),
            "interpretation_path": interp.get("interpretation_path"),
        }

    def run_full_analysis_pipeline(
        self,
        experiments: List[Dict[str, Any]],
        topic: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: (1) Decide what viz are needed for the paper, (2) Create those viz
        using interpreter script as reference in each experiment's analysis_viz folder,
        (3) Run analysis LLM with all images and user reqs. Returns analysis text and
        visualization bundle for the writer.
        experiments: list of {simulation_id, case_name, user_requirement, sim_dir, foam_output_dir}.
        """
        ex_by_id = {ex.get("simulation_id"): ex for ex in experiments if ex.get("simulation_id")}
        viz_spec = self.decide_visualizations(experiments, topic)
        if verbose:
            print("[Analysis] Creating visualizations for each experiment...", flush=True)
        viz_results = self.create_analysis_viz_for_experiments(experiments, viz_spec, verbose=verbose)
        experiments_with_images = []
        for r in viz_results:
            ex = ex_by_id.get(r["simulation_id"], {})
            experiments_with_images.append({
                "simulation_id": r["simulation_id"],
                "case_name": r["case_name"],
                "user_requirement": ex.get("user_requirement", ""),
                "experiment_idea_text": self._experiment_idea_text(ex),
                "image_paths": [Path(p) for p in r.get("images", [])],
            })
        analysis_text = self.run_analysis_with_images(
            experiments_with_images, topic, verbose=verbose
        )
        visualizations = []
        for r in viz_results:
            ex = ex_by_id.get(r["simulation_id"], {})
            visualizations.append({
                "simulation_id": r["simulation_id"],
                "case_name": r["case_name"],
                "description": ex.get("description", "") or ex.get("case_name", "") or r["case_name"],
                "visualization": r["visualization"],
            })
        if verbose:
            print("[Analysis] Done. Analysis text: %d chars, %d visualizations" % (
                len(analysis_text), len(visualizations)), flush=True)
        return {
            "analysis_text": analysis_text,
            "viz_spec": viz_spec,
            "visualizations": visualizations,
        }

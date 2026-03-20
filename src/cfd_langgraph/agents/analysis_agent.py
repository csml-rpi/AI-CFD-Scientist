from __future__ import annotations

import base64
import io
import json
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


class AnalysisAgent:
    """Max distinct visualization types to suggest per experiment (tweak as needed)."""
    MAX_EXP_VIZ = 10

    def __init__(self, model: str):
        self.model = model
        self.llm = create_langchain_llm(model=model, temperature=0.0)

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
            parts.append(
                f"Experiment {ex.get('simulation_id', '?')} ({ex.get('case_name', '')}):\n"
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
                paths = ex.get("image_paths") or []
                content_parts.append({
                    "type": "text",
                    "text": f"--- Experiment: {sim_id} ({case_name}) ---\nUser requirement: {user_req}\n\n",
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

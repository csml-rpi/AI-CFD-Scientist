"""
ResultsInterpreterAgent: for each experiment, (1) takes user requirement and case
structure (time folders, variables), (2) writes and runs a PyVista script to create
multiple visualizations (angles, variables at different times, mesh), (3) with
user requirement + images, decides if viz are good enough (retry up to 25 times),
then (4) decides if the simulation satisfied the user requirement, identifies issues,
and sets rerun_required with reasons.

Prompt context (no huge payloads): All LLM calls use only (a) short user_requirement
text, (b) last 20 lines of solver log in text-only fallback, or (c) user_requirement
+ base64 images in vision path. idea_json, experiment_spec, and experiment_results
are never sent to the model.
"""

from __future__ import annotations

import base64
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences
from cfd_langgraph.viz_creator import viz_creator

VIZ_MAX_RETRIES = 10


class ResultsInterpreterAgent:
    """Max distinct visualization types to suggest per experiment (tweak as needed)."""
    MAX_EXP_VIZ = 10

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("ResultsInterpreterAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.1)

    @staticmethod
    def _extract_output_dir(experiment_result: Dict[str, Any]) -> Optional[Path]:
        out = experiment_result.get("output_dir")
        if isinstance(out, str) and out.strip():
            p = Path(out)
            return p if p.exists() else None
        cmd = experiment_result.get("cmd")
        if isinstance(cmd, list):
            for i, tok in enumerate(cmd):
                if tok == "--output" and i + 1 < len(cmd):
                    p = Path(str(cmd[i + 1]))
                    return p if p.exists() else None
        return None

    @staticmethod
    def _locate_foam_dataset(foam_output_dir: Path) -> Optional[Path]:
        if not foam_output_dir.exists():
            return None
        foam_files = sorted(
            foam_output_dir.rglob("*.foam"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if foam_files:
            return foam_files[0]
        marker = foam_output_dir / "case.foam"
        try:
            marker.touch(exist_ok=True)
            return marker
        except Exception:
            return None

    @staticmethod
    def _get_case_structure(output_dir: Path, foam_path: Path) -> Dict[str, Any]:
        """Discover time folders and variables from the OpenFOAM case."""
        times: List[str] = []
        variables: List[str] = []
        try:
            for d in output_dir.iterdir():
                if not d.is_dir():
                    continue
                name = d.name
                if name == "constant" or name == "system":
                    continue
                try:
                    float(name)
                    times.append(name)
                except ValueError:
                    continue
            times = sorted(times, key=lambda x: float(x))
        except Exception:
            pass
        # Variables: from first time folder (e.g. 0) or constant
        for tdir in [output_dir / "0", output_dir / times[0] if times else None]:
            if tdir is None or not tdir.exists():
                continue
            for f in tdir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    variables.append(f.name)
            if variables:
                break
        # Fallback: try PyVista for arrays
        if not variables:
            try:
                import pyvista as pv
                mesh = pv.read(str(foam_path))
                variables = list(getattr(mesh, "array_names", []))
            except Exception:
                pass
        return {"times": times, "variables": variables}

    def _generate_pyvista_script(
        self,
        case_path: Path,
        case_structure: Dict[str, Any],
        out_dir: Path,
    ) -> str:
        """Generate a PyVista script: multiple angles, variables, and mesh outline."""
        times = case_structure.get("times", [])[:10] or ["0"]
        variables = list(case_structure.get("variables", [])[:8])
        if not variables:
            variables = ["U", "p"]
        case_str = str(case_path.resolve()).replace("\\", "\\\\")
        out_str = str(out_dir.resolve()).replace("\\", "\\\\")
        vars_repr = repr(variables)
        lines = [
            "import sys",
            "from pathlib import Path",
            "try:",
            "    import pyvista as pv",
            "except ImportError:",
            "    sys.exit(1)",
            f"out_dir = Path(r'{out_str}')",
            "out_dir.mkdir(parents=True, exist_ok=True)",
            "idx = 0",
            "def save(view_name, plotter):",
            "    global idx",
            "    p = out_dir / f'{idx:02d}_{view_name}.png'",
            "    plotter.screenshot(str(p))",
            "    plotter.close()",
            "    idx += 1",
            "try:",
            f"    mesh = pv.read(r'{case_str}')",
            "except Exception:",
            "    sys.exit(2)",
            "arrays = getattr(mesh, 'array_names', []) or []",
            f"for arr in {vars_repr}:",
            "    if arr not in arrays:",
            "        continue",
            "    try:",
            "        pl = pv.Plotter(off_screen=True)",
            "        pl.add_mesh(mesh, scalars=arr, cmap='viridis')",
            "        pl.add_axes()",
            "        save(f'scalar_{arr}', pl)",
            "    except Exception:",
            "        pass",
            "for angle, name in [((0,0,1), 'top'), ((1,0,0), 'side'), ((0,1,0), 'front')]:",
            "    try:",
            "        pl = pv.Plotter(off_screen=True)",
            "        pl.add_mesh(mesh.outline(), color='white')",
            "        pl.view_vector(angle)",
            "        pl.add_axes()",
            "        save(f'mesh_{name}', pl)",
            "    except Exception:",
            "        pass",
            "sys.exit(0)",
        ]
        return "\n".join(lines)

    def _run_viz_script(self, script_path: Path, cwd: Path) -> Tuple[bool, List[Path]]:
        """Execute the PyVista script; return (success, list of created image paths)."""
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return False, []
            out_dir = script_path.parent
            images = sorted(out_dir.glob("*.png"))
            return True, images
        except Exception:
            return False, []

    @staticmethod
    def _image_path_to_data_url(image_path: Path) -> Optional[str]:
        """Encode one image file as a data URL (base64), with MIME type from extension."""
        if not image_path.exists() or not image_path.is_file():
            return None
        try:
            b = image_path.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = image_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                return f"data:image/jpeg;base64,{b64}"
            if ext == ".png":
                return f"data:image/png;base64,{b64}"
            if ext == ".gif":
                return f"data:image/gif;base64,{b64}"
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    @staticmethod
    def _image_paths_to_content(image_paths: List[Path], max_images: int = 20) -> List[Dict[str, Any]]:
        """Build message content list: image_url blocks (base64) for vision LLM. MIME from extension."""
        content: List[Dict[str, Any]] = []
        for p in image_paths[:max_images]:
            url = ResultsInterpreterAgent._image_path_to_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def _invoke_vision_llm(
        self,
        user_requirement: str,
        image_paths: List[Path],
        system_prompt: str,
        user_prompt_template: str,
        max_retries: int = 10,
    ) -> str:
        """Invoke LLM with text + images (vision). Retries on throttling/transient errors."""
        text = user_prompt_template.format(user_requirement=user_requirement)
        image_blocks = self._image_paths_to_content(image_paths)
        if not image_blocks:
            text += "\n(No images provided.)"
        content: List[Any] = [{"type": "text", "text": text}]
        content.extend(image_blocks)

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                out = self.llm.invoke(messages)
                return getattr(out, "content", str(out)) if out else ""
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_retryable = (
                    "throttl" in err_str
                    or "too many requests" in err_str
                    or "rate" in err_str
                    or "validation error" in err_str
                )
                if attempt >= max_retries or not is_retryable:
                    raise
                delay = min(60.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.1)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        return ""

    @staticmethod
    def _user_requirement_text(idea_json: Dict[str, Any], experiment_spec: Dict[str, Any]) -> str:
        parts: List[str] = []
        if isinstance(idea_json, dict) and idea_json.get("description"):
            parts.append(str(idea_json["description"]).strip())
        if isinstance(experiment_spec, dict):
            if experiment_spec.get("description"):
                parts.append(str(experiment_spec["description"]).strip())
            if experiment_spec.get("case_name") and not parts:
                parts.append(str(experiment_spec["case_name"]).strip())
        return "\n".join(p for p in parts if p) or "No user requirement provided."

    def _plan_what_to_visualize(self, user_req: str, case_structure: Dict[str, Any]) -> str:
        """
        Decide, per experiment, what to visualize based on:
        - user requirement text
        - available time folders and variables in the case.

        Uses an LLM-based planner to produce a natural-language description
        for viz_creator. No heuristic fallback.
        """
        times = [str(t) for t in (case_structure.get("times", []) or [])]
        vars_ = [str(v) for v in (case_structure.get("variables", []) or [])]
    
        try:
            max_viz = getattr(self, "MAX_EXP_VIZ", 10)
            system = (
                "You are a CFD visualization planner for general CFD simulations. "
                "Given the user requirement, the available solution times, and field variables, "
                "you must describe in plain language what visualizations to generate. "
                "Focus on scientifically useful plots that help assess whether the simulation "
                "meets the requirement (e.g., mesh outline, key scalar/vector fields at important times, "
                "pressure or velocity contours, centerline/line profiles, etc.). "
                f"Suggest at most {max_viz} distinct visualization types per experiment. "
                "Do NOT write code. Return only a short paragraph or a few sentences describing "
                "what to visualize; this text will be given to a separate viz generator."
            )
            user_t = (
                "User requirement for this experiment:\n"
                "{user_requirement}\n\n"
                "Available time folders in the OpenFOAM case:\n"
                "{times}\n\n"
                "Available field variables (cell/point data):\n"
                "{variables}\n\n"
                f"Describe at most {max_viz} distinct visualization types to create "
                "to best evaluate whether the simulation satisfies the requirement. "
                "Mention which fields (e.g., U, p), which times (early/mid/late or specific values), "
                "and what types of plots (contours, slices, centerline profiles, etc.). "
                "Plain text only, no bullet lists, no JSON, no code."
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", user_t),
                ]
            )
            chain = prompt | self.llm
            content = chain.invoke(
                {
                    "user_requirement": user_req,
                    "times": ", ".join(times) if times else "(no time folders discovered)",
                    "variables": ", ".join(vars_) if vars_ else "(no variables discovered)",
                }
            ).content
            text = str(content or "").strip()
            if text:
                return text
            raise ValueError("LLM returned empty visualization plan")
        except Exception as e:
            import traceback
            print("[interpreter] _plan_what_to_visualize LLM failed:", e, flush=True)
            traceback.print_exc()
            raise

    def _text_only_interpret(
        self,
        user_req: str,
        solver_log_tail: str,
        experiment_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback when no foam case or PyVista fails: interpret from user req + solver log only."""
        system = self.prompts.get("system_prompt", "You are a CFD results interpreter.")
        user_t = self.prompts.get(
            "user_prompt",
            "USER REQUIREMENT:\n{user_requirement}\n\nSOLVER LOG (last 20 lines):\n{solver_log}\n\nReturn JSON with rerun_required, summary, reasons.",
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user_t)])
        chain = prompt | self.llm
        content = chain.invoke({"user_requirement": user_req, "solver_log": solver_log_tail}).content
        try:
            parsed = json.loads(strip_json_fences(content))
        except Exception:
            parsed = {"raw": content, "parse_error": True}
        rc = experiment_results.get("returncode")
        parsed.setdefault("rerun_required", rc != 0)
        parsed.setdefault("simulation_success", rc == 0)
        parsed.setdefault("requirement_met", False)
        parsed.setdefault("viz_ok", False)
        parsed.setdefault("viz_attempts", [])
        return parsed

    def interpret(
        self,
        idea_json: Dict[str, Any],
        experiment_spec: Dict[str, Any],
        experiment_results: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        # Context audit: we never send idea_json, experiment_spec, or experiment_results
        # to the LLM. We only use: (1) short user_req, (2) last 20 lines of solver log,
        # (3) images from our own PyVista script. No huge stdout/stderr or full run dumps.
        sim_id = experiment_spec.get("simulation_id", "?") if isinstance(experiment_spec, dict) else "?"
        if verbose:
            print("[Interpreter] Interpreting %s..." % sim_id, flush=True)
        user_req = self._user_requirement_text(idea_json, experiment_spec)
        output_dir = self._extract_output_dir(experiment_results)
        solver_log_payload = self._collect_solver_log_tails(experiment_results, n_lines=20)
        solver_log_tail = solver_log_payload.get("tail_text", "") or "(no solver log found)"

        if output_dir is None:
            return self._text_only_interpret(user_req, solver_log_tail, experiment_results)

        foam_path = self._locate_foam_dataset(output_dir)
        if foam_path is None:
            return self._text_only_interpret(user_req, solver_log_tail, experiment_results)

        case_structure = self._get_case_structure(output_dir, foam_path)
        viz_base = output_dir / "interpreter_viz"
        viz_base.mkdir(parents=True, exist_ok=True)

        # Use central viz_creator to generate and refine visualizations (no reference yet).
        if verbose:
            print("[Interpreter] Running viz_creator (PyVista)...", flush=True)
        viz_result = viz_creator(
            model=self.model,
            foam_output_dir=output_dir,
            viz_dir=viz_base,
            what_to_visualize=self._plan_what_to_visualize(user_req, case_structure),
            user_requirement=user_req,
            reference_viz_script=None,
            max_retries=VIZ_MAX_RETRIES,
        )

        image_paths: List[Path] = [Path(p) for p in viz_result.get("images", [])]
        viz_attempts: List[Dict[str, Any]] = []

        if not viz_result.get("ok") or not image_paths:
            # Fall back to text-only interpretation but record viz attempts.
            base = self._text_only_interpret(user_req, solver_log_tail, experiment_results)
            reason = viz_result.get("last_error", "viz_creator failed or produced no images")
            base.setdefault("viz_attempts", [])
            base["viz_attempts"].append(
                {
                    "attempt": viz_result.get("attempts", 0),
                    "viz_ok": False,
                    "reason": reason,
                }
            )
            base["viz_ok"] = False
            return base

        viz_attempts.append(
            {
                "attempt": viz_result.get("attempts", 1),
                "viz_ok": True,
                "reason": "viz_creator accepted visualizations",
            }
        )

        system_interp = self.prompts.get(
            "interpretation_system_prompt",
            "You are a CFD results interpreter. Given the user requirement and visualization images from a simulation, decide: (1) Did the simulation run successfully and satisfy the user requirement? (2) What issues exist, if any? (3) Should the run be redone (rerun_required)? Return ONLY valid JSON with keys: simulation_success (bool), requirement_met (bool), issues (string or list), rerun_required (bool), summary (string), reasons (string).",
        )
        user_interp = self.prompts.get(
            "interpretation_user_prompt",
            "User requirement:\n{user_requirement}\n\nBelow are visualizations from the run. Did the simulation satisfy the requirement? Any issues? check if flow field develops accuratley, expected flow features are well visible. Set rerun_required true if results are not acceptable. Return JSON only.",
        )
        if verbose:
            print("[Interpreter] Invoking vision LLM for interpretation...", flush=True)
        content = self._invoke_vision_llm(user_req, image_paths, system_interp, user_interp)
        try:
            parsed = json.loads(strip_json_fences(content))
        except Exception:
            parsed = {"raw": content, "parse_error": True}

        parsed.setdefault("rerun_required", False)
        parsed.setdefault("simulation_success", True)
        parsed.setdefault("requirement_met", False)
        parsed.setdefault("viz_ok", bool(viz_attempts and viz_attempts[-1].get("viz_ok")))
        parsed.setdefault("viz_attempts", viz_attempts)
        parsed.setdefault("case_structure", case_structure)
        if verbose:
            print("[Interpreter] Done: rerun_required=%s requirement_met=%s" % (
                parsed.get("rerun_required"), parsed.get("requirement_met")), flush=True)
        return parsed

    def _collect_solver_log_tails(
        self, experiment_result: Dict[str, Any], n_lines: int = 20
    ) -> Dict[str, Any]:
        output_dir = self._extract_output_dir(experiment_result)
        if output_dir is None:
            return {"output_dir": None, "files": [], "tail_text": ""}
        preferred = [
            "log.icoFoam", "log.pisoFoam", "log.pimpleFoam",
            "log.simpleFoam", "log.rhoPimpleFoam", "log.rhoSimpleFoam",
        ]
        files: List[Path] = []
        for name in preferred:
            p = output_dir / name
            if p.exists() and p.is_file():
                files.append(p)
        if not files:
            files = sorted(
                [p for p in output_dir.glob("log.*") if p.is_file() and "foam" in p.name.lower()]
            )
        chunks: List[str] = []
        for p in files:
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                tail = "\n".join(lines[-n_lines:])
                if tail.strip():
                    chunks.append(f"--- {p.name} (last {n_lines} lines) ---\n{tail}")
            except Exception:
                pass
        return {
            "output_dir": str(output_dir),
            "files": [str(p) for p in files],
            "tail_text": "\n\n".join(chunks),
        }

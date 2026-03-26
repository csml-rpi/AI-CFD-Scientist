from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from cfd_langgraph.agents.analysis_agent import AnalysisAgent
from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent
from cfd_langgraph.agents.interpreter_agent import ResultsInterpreterAgent
from cfd_langgraph.agents.rerun_analysis_agent import RerunAnalysisAgent
from cfd_langgraph.agents.writer_agent import WriterAgent
from cfd_langgraph.config import Settings
from cfd_langgraph.foam.runner import FoamAgentRunner
from cfd_langgraph.ideation import run_ideation
from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class WorkflowState(TypedDict, total=False):
    topic: str
    out_dir: Path
    execute: bool
    allow_non_executed_artifacts: bool
    verbose: bool

    ideation_result: Dict[str, Any]
    idea: Dict[str, Any]
    simulations: List[Dict[str, Any]]
    sim_index: int

    current_sim: Dict[str, Any]
    sim_dir: Path
    hyp: Dict[str, Any]
    req_text: str
    req_valid: bool
    attempt: int
    run_history: List[Dict[str, Any]]
    run_result: Dict[str, Any]
    interp: Dict[str, Any]
    requirement_updates: List[Dict[str, Any]]
    skip_run: bool

    pipeline_log: Dict[str, Any]

    # Rerun loop (batch mode, after interpreter_batch)
    rerun_queue: List[str]
    rerun_idx: int
    rerun_round: int


@dataclass
class CFDWorkflow:
    settings: Settings
    prompt_loader: PromptLoader
    _app: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.hypothesis = HypothesisAgent(self.settings.model, self.prompt_loader)
        self.interpreter = ResultsInterpreterAgent(
            self.settings.model, self.prompt_loader
        )
        self.rerun_analysis = RerunAnalysisAgent(self.settings.model, self.prompt_loader)
        self.analysis = AnalysisAgent(self.settings.model)
        self.writer = WriterAgent(self.settings.model, self.prompt_loader)
        self.foam = FoamAgentRunner(
            self.settings.foam_agent_main, self.settings.openfoam_path
        )

    # ------------------------------------------------------------------
    # Helper: build reference summary for rerun from a working case
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rerun_reference_summary(sim_entry: Dict[str, Any], out_dir: Path) -> str:
        """
        Build a compact text summary of a working case to guide rerun requirement repair.

        Only includes files from:
          - system/* (except anything under constant/polyMesh)
          - constant/* (excluding polyMesh)
          - 0/* (time-zero fields)
          - Allrun (if present)
        Never reads or mentions constant/polyMesh or time directories > 0.
        """
        sim_meta = sim_entry.get("simulation", {}) if isinstance(sim_entry.get("simulation", {}), dict) else {}
        sim_id = str(sim_meta.get("simulation_id", "") or "")
        case_name = str(sim_meta.get("case_name", sim_id) or "")
        sim_dir = out_dir / sim_id
        foam_dir = sim_dir / "foam_output"

        lines: List[str] = []
        lines.append(f"REFERENCE CASE ID: {sim_id}")
        if case_name:
            lines.append(f"REFERENCE CASE NAME: {case_name}")
        lines.append(f"REFERENCE ROOT DIR: {foam_dir}")

        # List candidate input files
        system_files: List[Path] = []
        constant_files: List[Path] = []
        zero_files: List[Path] = []
        allrun_path: Path | None = None

        if foam_dir.is_dir():
            sys_dir = foam_dir / "system"
            if sys_dir.is_dir():
                for p in sorted(sys_dir.iterdir()):
                    if p.is_file():
                        system_files.append(p)
            const_dir = foam_dir / "constant"
            if const_dir.is_dir():
                for p in sorted(const_dir.iterdir()):
                    # skip polyMesh entirely
                    if p.name == "polyMesh":
                        continue
                    if p.is_file():
                        constant_files.append(p)
            zero_dir = foam_dir / "0"
            if zero_dir.is_dir():
                for p in sorted(zero_dir.iterdir()):
                    if p.is_file():
                        zero_files.append(p)
            candidate_allrun = foam_dir / "Allrun"
            if candidate_allrun.is_file():
                allrun_path = candidate_allrun

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(foam_dir))
            except ValueError:
                return str(p)

        if system_files:
            lines.append("SYSTEM FILES:")
            for p in system_files:
                lines.append(f"  - { _rel(p) }")
        if constant_files:
            lines.append("CONSTANT FILES (excluding polyMesh):")
            for p in constant_files:
                lines.append(f"  - { _rel(p) }")
        if zero_files:
            lines.append("TIME 0 FILES (initial fields):")
            for p in zero_files:
                lines.append(f"  - { _rel(p) }")
        if allrun_path:
            lines.append(f"ALLRUN SCRIPT: { _rel(allrun_path) }")

        # Optionally embed a few high-signal file contents to help the LLM.
        def _read_if_exists(p: Path) -> str:
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                return ""

        # Mesh / numerics / BC snippets
        key_files = [
            foam_dir / "system" / "blockMeshDict",
            foam_dir / "system" / "fvSolution",
            foam_dir / "system" / "fvSchemes",
            foam_dir / "constant" / "turbulenceProperties",
            foam_dir / "0" / "U",
            foam_dir / "0" / "p",
            foam_dir / "0" / "k",
            foam_dir / "0" / "epsilon",
            foam_dir / "0" / "omega",
            foam_dir / "0" / "nuTilda",
            foam_dir / "0" / "nut",
        ]

        for p in key_files:
            if not p.is_file():
                continue
            content = _read_if_exists(p)
            if not content.strip():
                continue
            rel = _rel(p)
            # Keep each snippet bounded in size to avoid huge prompts.
            snippet = content.strip()
            if len(snippet) > 4000:
                snippet = snippet[:4000] + "\n... [truncated]"
            lines.append(f"\n--- BEGIN REFERENCE FILE {rel} ---\n{snippet}\n--- END REFERENCE FILE {rel} ---")

        return "\n".join(lines)

    @staticmethod
    def _read_text_file(path: Path, *, max_chars: int = 6000) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            txt = path.read_text(encoding="utf-8", errors="ignore")
            txt = txt.strip("\ufeff")
            if len(txt) <= max_chars:
                return txt
            # Keep head + tail so diffs still catch key changes.
            head = txt[: max_chars // 2]
            tail = txt[-(max_chars // 2) :]
            return head + "\n... [truncated] ...\n" + tail
        except Exception:
            return ""

    def _get_latest_requirement_text(
        self, sim_id: str, entry: Dict[str, Any], out_dir: Path
    ) -> str:
        """
        For rerun reference selection:
        Prefer the persisted `user_requirement.txt` for that case; otherwise fall back
        to pipeline-log stored hypothesis/requirement_updates.
        """
        sim_dir = out_dir / str(sim_id)
        req_path = sim_dir / "user_requirement.txt"
        if req_path.exists():
            txt = self._read_text_file(req_path, max_chars=12000)
            if txt.strip():
                return txt.strip()

        # Fallbacks (no persisted file)
        req = ""
        hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
        if isinstance(hyp.get("requirement"), str):
            req = hyp.get("requirement", "") or ""

        # If requirement_updates exists, take the last attempt requirement.
        updates = entry.get("requirement_updates", []) if isinstance(entry.get("requirement_updates", []), list) else []
        if updates:
            last = updates[-1]
            if isinstance(last, dict) and isinstance(last.get("requirement"), str):
                req = last["requirement"] or req

        return str(req or "").strip()

    def _collect_openfoam_inputs_bundle(
        self,
        sim_id: str,
        out_dir: Path,
        *,
        max_files: int = 18,
        max_chars_total: int = 12000,
        max_file_chars: int = 2500,
        skip_poly_mesh: bool = True,
    ) -> str:
        """
        Collect relevant OpenFOAM *inputs* from foam_output/system/, constant/, 0/
        (ignoring constant/polyMesh by default).
        """
        sim_dir = out_dir / str(sim_id)
        foam_dir = sim_dir / "foam_output"
        if not foam_dir.is_dir():
            return f"(no foam_output dir found at {foam_dir})"

        def _list_files(root: Path) -> List[Path]:
            if not root.is_dir():
                return []
            paths: List[Path] = []
            for p in sorted(root.iterdir()):
                if p.is_file():
                    paths.append(p)
                elif p.is_dir():
                    # Only include top-level dirs if they contain files; but we'll mostly avoid
                    # large generated mesh subtrees.
                    if p.name == "polyMesh" and skip_poly_mesh:
                        continue
                    for fp in sorted(p.rglob("*")):
                        if fp.is_file():
                            paths.append(fp)
            return paths

        system_dir = foam_dir / "system"
        constant_dir = foam_dir / "constant"
        zero_dir = foam_dir / "0"

        system_files = _list_files(system_dir)
        constant_files = _list_files(constant_dir)
        zero_files = _list_files(zero_dir)

        # Prioritize a small set of “high signal” known input files.
        priority_hints = {
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/blockMeshDict",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "constant/thermophysicalProperties",
            "constant/RASProperties",
            "0/U",
            "0/p",
            "0/k",
            "0/epsilon",
            "0/omega",
            "0/nuTilda",
            "0/nut",
        }

        def _rel(fp: Path) -> str:
            try:
                return str(fp.relative_to(foam_dir))
            except ValueError:
                return str(fp)

        all_files = system_files + constant_files + zero_files
        all_files = [p for p in all_files if p.is_file()]

        # Sort: priority first, then shortest relpath, then lexicographic.
        def _sort_key(p: Path):
            rel = _rel(p)
            pri = 0 if rel in priority_hints else 1
            return (pri, len(rel), rel)

        all_files.sort(key=_sort_key)

        total_chars = 0
        included: List[str] = []
        file_count = 0
        for p in all_files:
            if file_count >= max_files:
                break
            rel = _rel(p)
            if skip_poly_mesh and rel.startswith("constant/polyMesh/"):
                continue
            txt = self._read_text_file(p, max_chars=max_file_chars)
            if not txt.strip():
                continue
            if total_chars + len(txt) > max_chars_total:
                # Truncate to remaining budget.
                remaining = max_chars_total - total_chars
                if remaining <= 0:
                    break
                txt = txt[:remaining] + "\n...[truncated budget]..."
            included.append(f"--- BEGIN FILE {rel} ---\n{txt}\n--- END FILE {rel} ---")
            total_chars += len(txt)
            file_count += 1

        return "\n".join(
            [f"CASE_INPUT_BUNDLE sim_id={sim_id}", f"foam_output_dir={foam_dir}"] + included
        )

    def _select_closest_successful_case_llm(
        self,
        *,
        target_user_requirement: str,
        successful_entries: List[Dict[str, Any]],
        out_dir: Path,
        max_candidates: int = 6,
        verbose: bool = False,
    ) -> str:
        """
        LLM selects the closest-successful case based on user requirement similarity.
        Returns selected simulation_id.
        """
        # Build candidate list with their latest requirement text.
        candidates: List[Dict[str, Any]] = []
        for e in successful_entries:
            if not isinstance(e, dict):
                continue
            sim = e.get("simulation", {}) if isinstance(e.get("simulation", {}), dict) else {}
            sim_id = sim.get("simulation_id")
            if sim_id is None:
                continue
            sim_id_s = str(sim_id)
            req_txt = self._get_latest_requirement_text(sim_id_s, e, out_dir=out_dir)
            if not req_txt:
                continue
            candidates.append(
                {
                    "simulation_id": sim_id_s,
                    "case_name": sim.get("case_name", ""),
                    "requirement": req_txt[:4000],  # hard cap for selection prompt
                }
            )

        if not candidates:
            return ""

        # Cap candidates to keep prompt small: choose by closeness heuristic to reduce cost.
        target_base = str(target_user_requirement).split(".")[0][:120].strip()
        candidates.sort(
            key=lambda c: (0 if c.get("case_name", "").split("_")[0] in target_user_requirement else 1, c["simulation_id"])
        )
        candidates = candidates[:max_candidates]

        llm = create_langchain_llm(model=self.settings.model, temperature=0.0)
        system = (
            "You are a CFD workflow planner. You are given a target (non-working) Foam-Agent user requirement "
            "and several successful candidate cases from the same study. "
            "Choose which candidate is closest to the target in terms of intended setup, geometry, sweep parameter, "
            "solver usage, mesh/numerics, and boundary-condition pattern. "
            "Do not guess about missing files; use only the provided requirement texts. "
            "Return ONLY valid JSON."
        )
        user = {
            "target_user_requirement": target_user_requirement[:8000],
            "successful_candidates": candidates,
        }
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{payload}")])
        resp = llm.invoke(prompt.format_messages(payload=json.dumps(user, ensure_ascii=False)) )
        raw = getattr(resp, "content", str(resp))
        try:
            parsed = json.loads(strip_json_fences(raw))
        except Exception:
            if verbose:
                print(f"[rerun] LLM selection parse failed; raw={raw[:2000]}", flush=True)
            return candidates[0]["simulation_id"]
        sid = parsed.get("selected_simulation_id") if isinstance(parsed, dict) else None
        if sid is None:
            sid = parsed.get("simulation_id") if isinstance(parsed, dict) else None
        return str(sid or "") or candidates[0]["simulation_id"]

    def _diff_inputs_llm(
        self,
        *,
        working_sim_id: str,
        failing_sim_id: str,
        out_dir: Path,
        verbose: bool = False,
    ) -> str:
        """
        LLM creates a diff report (not a summary) between working vs failing OpenFOAM inputs,
        comparing only system/, constant/ (excluding polyMesh/), and 0/.
        """
        llm = create_langchain_llm(model=self.settings.model, temperature=0.0)

        working_bundle = self._collect_openfoam_inputs_bundle(working_sim_id, out_dir)
        failing_bundle = self._collect_openfoam_inputs_bundle(failing_sim_id, out_dir)

        system = (
            "You are an expert OpenFOAM case diff assistant. "
            "You will be given two case input bundles: a WORKING case and a failing (NON-WORKING) case. "
            "Both bundles include file contents from system/, constant/ (excluding constant/polyMesh), and 0/. "
            "Your job is to create a DIFF REPORT that explicitly describes what differs between the cases "
            "at the file level and the line/section level as much as possible. "
            "You must NOT produce a generic summary; the report should be diffs and actionable edits. "
            "Output ONLY valid JSON."
        )
        human = (
            "Return JSON with schema {"
            "\"diff_report\": string"
            "}. "
            "The diff_report must include: "
            "1) FILE PATHS that differ; "
            "2) For each differing file, show a unified-diff-like block (use '+'/'-' lines when possible); "
            "3) 'SUGGESTED USER-REQUIREMENT CHANGES': concrete guidance describing what parts of the Foam-Agent "
            "user requirement should be updated in the failing case to match the working case's inputs, "
            "while preserving the intended sweep concept. "
            "If the prompt truncates file contents, mention '<<TRUNCATED>>' inside the diff report for that file. \n\n"
            "WORKING_CASE:\n{working}\n\nFAILING_CASE:\n{failing}\n"
        )
        # Use ChatPromptTemplate for consistent formatting with other calls.
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        resp = llm.invoke(prompt.format_messages(working=working_bundle, failing=failing_bundle))
        raw = getattr(resp, "content", str(resp))
        try:
            parsed = json.loads(strip_json_fences(raw))
            diff_report = parsed.get("diff_report")
            if isinstance(diff_report, str) and diff_report.strip():
                return diff_report.strip()
        except Exception:
            if verbose:
                print(f"[rerun] LLM diff parse failed; raw={raw[:2000]}", flush=True)
        # Fallback: minimal report.
        return (
            f"<<DIFF REPORT UNAVAILABLE>>\n"
            f"working_sim_id={working_sim_id}\n"
            f"failing_sim_id={failing_sim_id}\n"
        )


    @staticmethod
    def _image_path_to_block(image_path: str) -> Optional[Dict[str, Any]]:
        p = Path(image_path)
        if not p.is_file():
            return None
        try:
            b = p.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        except Exception:
            return None

    def _vision_filter_viz_bundle(
        self,
        viz_bundle: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Vision-LLM figure quality gate.
        Keeps figures that are readable and show requested features clearly.
        Falls back to deterministic selection if any step fails.
        """
        ex_by_id: Dict[str, Dict[str, Any]] = {}
        for ex in experiments:
            sid = ex.get("simulation_id")
            if sid is not None:
                ex_by_id[str(sid)] = ex

        llm = create_langchain_llm(model=self.settings.model, temperature=0.0)
        out: List[Dict[str, Any]] = []

        system = (
            "You are a CFD figure quality selector for journal papers.\n"
            "Given a user requirement and candidate images from one experiment, choose images that are:\n"
            "1) readable for human readers (not tiny/illegible),\n"
            "2) framed appropriately (zoomed/cropped enough for requested local features when relevant),\n"
            "3) informative (show requested flow features or metrics).\n"
            "Reject images that are too zoomed-out, cluttered, blurry, or not useful.\n"
            "Return ONLY JSON with keys: keep_indices (list[int]), rejected_indices (list[int]), reason (string)."
        )

        for item in viz_bundle:
            if not isinstance(item, dict):
                out.append(item)
                continue

            sid = str(item.get("simulation_id", "") or "")
            ex = ex_by_id.get(sid, {})
            user_req = str(ex.get("user_requirement", "") or "")

            vis = item.get("visualization", {}) if isinstance(item.get("visualization"), dict) else {}
            images_raw = vis.get("images", []) if isinstance(vis.get("images"), list) else []
            images = [str(p) for p in images_raw if isinstance(p, str) and p.strip()]

            # Pure vision-based selection (no deterministic pre-filtering by filename).
            candidates = images
            # Keep payload bounded.
            candidates = candidates[:8]
            if not candidates:
                out.append(item)
                continue

            content: List[Any] = [{
                "type": "text",
                "text": (
                    f"Experiment: {sid}\n"
                    f"User requirement:\n{user_req}\n\n"
                    "Candidate images are provided in order. "
                    "Select the best subset for paper use."
                ),
            }]
            valid_candidates: List[str] = []
            for p in candidates:
                block = self._image_path_to_block(p)
                if block:
                    valid_candidates.append(p)
                    content.append({"type": "text", "text": f"IMAGE_INDEX={len(valid_candidates)-1} PATH={p}"})
                    content.append(block)

            if not valid_candidates:
                out.append(item)
                continue

            try:
                resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
                raw = getattr(resp, "content", str(resp))
                parsed = json.loads(strip_json_fences(raw))
                keep_idx = parsed.get("keep_indices", [])
                if not isinstance(keep_idx, list):
                    keep_idx = []
                keep_paths = []
                for i in keep_idx:
                    if isinstance(i, int) and 0 <= i < len(valid_candidates):
                        keep_paths.append(valid_candidates[i])
                # Fallback: if model rejects all, keep top 1 candidate.
                if not keep_paths:
                    keep_paths = valid_candidates[:1]
            except Exception as e:
                if verbose:
                    print(f"[CFD-WORKFLOW] vision filter failed for {sid}: {e}", flush=True)
                # Fallback behavior
                keep_paths = candidates[:1]

            item2 = {**item, "visualization": {**vis, "images": keep_paths}}
            out.append(item2)

        return out

    # ======================================================================
    # Restart helper: reuse completed Foam-Agent runs and re-run only
    # interpreter + analysis + writer stages for an existing out_dir.
    # This deliberately bypasses the LangGraph app and operates directly on
    # disk artifacts (pipeline_log.json, ideation_output.json, etc.).
    # ======================================================================

    def restart_from_foam(self, out_dir: Path, verbose: bool = True) -> Dict[str, Any]:
        """
        Restart pipeline AFTER Foam-Agent runs have completed.

        This:
        - Loads idea and simulations from existing ideation_output.json /
          hypothesis_output.json / pipeline_log.json.
        - For each simulation, loads the latest Foam-Agent run_result from
          pipeline_log and re-runs the interpreter agent.
        - Runs the analysis agent with images for all experiments.
        - Runs the writer agent (paper with literature + review loop).
        - Updates pipeline_log.json with new interpreter, analysis, and paper entries.

        Foam-Agent (Allrun) is NOT re-executed.
        """
        out_dir = out_dir.expanduser().resolve()
        if verbose:
            print(f"[CFD-WORKFLOW] RESTART from foam runs in {out_dir}", flush=True)

        # Load existing pipeline_log if present
        pipeline_path = out_dir / "pipeline_log.json"
        pipeline_log: Dict[str, Any] = {}
        if pipeline_path.is_file():
            try:
                pipeline_log = json.loads(pipeline_path.read_text(encoding="utf-8"))
            except Exception:
                pipeline_log = {}

        topic = str(pipeline_log.get("topic") or "")

        # Load idea from ideation_output.json if available
        idea: Dict[str, Any] = {}
        ideation_path = out_dir / "ideation_output.json"
        if ideation_path.is_file():
            try:
                ideation_payload = json.loads(ideation_path.read_text(encoding="utf-8"))
                ideation_bundle = ideation_payload.get("ideation", {}) or {}
                idea = ideation_bundle.get("idea", {}) if isinstance(ideation_bundle, dict) else {}
                if not topic:
                    topic = str(ideation_payload.get("topic") or "")
            except Exception:
                idea = {}

        if not topic:
            topic = str(pipeline_log.get("topic") or "")

        if verbose:
            print(f"[CFD-WORKFLOW] Restart topic: {topic!r}", flush=True)

        sims_entries: List[Dict[str, Any]] = []
        # Prefer simulations list from pipeline_log if present
        if isinstance(pipeline_log.get("simulations"), list):
            sims_entries = [s for s in pipeline_log.get("simulations", []) if isinstance(s, dict)]

        # Back-compat restart: rebuild simulations from hypothesis_output / foamagent_output / interpreter_output
        # when pipeline_log.json is missing or does not contain simulations.
        if not sims_entries:
            hyp_path = out_dir / "hypothesis_output.json"
            foam_path = out_dir / "foamagent_output.json"
            interp_path = out_dir / "interpreter_output.json"
            try:
                hyp_payload = json.loads(hyp_path.read_text(encoding="utf-8")) if hyp_path.is_file() else {}
            except Exception:
                hyp_payload = {}
            try:
                foam_payload = json.loads(foam_path.read_text(encoding="utf-8")) if foam_path.is_file() else {}
            except Exception:
                foam_payload = {}
            try:
                interp_payload = json.loads(interp_path.read_text(encoding="utf-8")) if interp_path.is_file() else {}
            except Exception:
                interp_payload = {}

            hyp_items = hyp_payload.get("items", []) if isinstance(hyp_payload, dict) else []
            foam_cases = foam_payload.get("cases", []) if isinstance(foam_payload, dict) else []
            interp_cases = interp_payload.get("cases", []) if isinstance(interp_payload, dict) else []

            foam_by_id = {}
            for c in foam_cases if isinstance(foam_cases, list) else []:
                if isinstance(c, dict) and c.get("simulation_id"):
                    foam_by_id[str(c.get("simulation_id"))] = c

            interp_by_id = {}
            for c in interp_cases if isinstance(interp_cases, list) else []:
                if isinstance(c, dict) and c.get("simulation_id"):
                    interp_by_id[str(c.get("simulation_id"))] = c

            rebuilt: List[Dict[str, Any]] = []
            for it in hyp_items if isinstance(hyp_items, list) else []:
                if not isinstance(it, dict):
                    continue
                sim_id = str(it.get("simulation_id", "") or "")
                if not sim_id:
                    continue
                exp = it.get("experiment", {}) if isinstance(it.get("experiment", {}), dict) else {}
                prompt_for_foam = str(it.get("prompt_for_foamagent", "") or "")
                ic = interp_by_id.get(sim_id, {}) or {}
                attempts_used = int(
                    (ic.get("foam_rerun_count") if isinstance(ic, dict) else None)
                    or (ic.get("attempts_used", 0) if isinstance(ic, dict) else 0)
                    or 0
                )
                latest_run = (foam_by_id.get(sim_id, {}) or {}).get("latest_run", {}) if isinstance(foam_by_id.get(sim_id, {}), dict) else {}

                rebuilt.append(
                    {
                        "simulation": exp or {"simulation_id": sim_id, "case_name": it.get("case_name", sim_id)},
                        "hypothesis": {"requirement": prompt_for_foam, "valid": bool(it.get("valid", True))},
                        "run_history": [latest_run] if isinstance(latest_run, dict) and latest_run else [],
                        "interpreter": (interp_by_id.get(sim_id, {}) or {}).get("interpreter", {}) if isinstance(interp_by_id.get(sim_id, {}), dict) else {},
                        "attempt": attempts_used,
                    }
                )

            sims_entries = rebuilt
            if sims_entries:
                pipeline_log["simulations"] = sims_entries
                pipeline_log["topic"] = topic or str(hyp_payload.get("topic") or "")
                # Persist a pipeline_log so future restarts/resumes are robust.
                self._write_json(pipeline_path, pipeline_log)

        if not sims_entries:
            if verbose:
                print("[CFD-WORKFLOW] No simulations found in existing artifacts; nothing to restart.", flush=True)
            return {"pipeline_log": pipeline_log}

        # Build experiments list for interpreter + analysis stages
        experiments: List[Dict[str, Any]] = []
        for entry in sims_entries:
            sim_meta = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            sim_id = str(sim_meta.get("simulation_id", "") or "")
            if not sim_id:
                continue
            sim_dir = out_dir / sim_id
            foam_output_dir = sim_dir / "foam_output"

            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            latest_run = run_history[-1] if run_history else {}

            hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
            user_req = str(hyp.get("requirement", "") or "")

            experiments.append(
                {
                    "simulation_id": sim_id,
                    "case_name": sim_meta.get("case_name", sim_id),
                    "description": sim_meta.get("description", ""),
                    "experiment_idea": sim_meta.get("case_data", None),
                    "user_requirement": user_req,
                    "sim_dir": sim_dir,
                    "foam_output_dir": foam_output_dir,
                    "latest_run_result": latest_run,
                    "simulation_meta": sim_meta,
                }
            )

        if verbose:
            print(f"[CFD-WORKFLOW] Restart: {len(experiments)} experiment(s) for interpreter/analysis/writer.", flush=True)

        max_reruns = int(self.settings.workflow_max_reruns_per_experiment)

        # Optional optimization: if interpreter_output.json exists, only
        # re-interpret cases that previously requested a rerun.
        interp_snapshot_path = out_dir / "interpreter_output.json"
        snapshot_by_sim_id: Dict[str, Dict[str, Any]] = {}
        if interp_snapshot_path.is_file():
            snap = self._read_json(interp_snapshot_path)
            snap_cases = snap.get("cases", []) if isinstance(snap, dict) else []
            if isinstance(snap_cases, list):
                for c in snap_cases:
                    if isinstance(c, dict) and c.get("simulation_id") is not None:
                        snapshot_by_sim_id[str(c.get("simulation_id"))] = c

        # Re-run interpreter for each experiment using latest Foam-Agent run_result.
        # For cases marked rerun_required=false in the snapshot, keep the existing
        # interpreter payload untouched to avoid unnecessary LLM calls.
        new_sims_entries: List[Dict[str, Any]] = []
        for entry, ex in zip(sims_entries, experiments):
            sim_meta = ex["simulation_meta"]
            sim_id = ex["simulation_id"]
            sim_id_s = str(sim_id)

            snap_case = snapshot_by_sim_id.get(sim_id_s)
            snap_rr = bool(snap_case.get("rerun_required", False)) if isinstance(snap_case, dict) else False
            snap_interp = snap_case.get("interpreter") if isinstance(snap_case, dict) else None
            snap_attempts_used = None
            if isinstance(snap_case, dict):
                snap_attempts_used = snap_case.get("attempts_used", snap_case.get("foam_rerun_count"))

            if snap_case and not snap_rr and isinstance(snap_interp, dict):
                entry2 = dict(entry)
                entry2["interpreter"] = snap_interp
                if snap_attempts_used is not None:
                    try:
                        entry2["attempt"] = int(snap_attempts_used)
                    except Exception:
                        pass
                if verbose:
                    print(f"[CFD-WORKFLOW] RESTART interpreter skip :: {sim_id_s} (snapshot rerun_required=false)", flush=True)
                new_sims_entries.append(entry2)
                continue

            if verbose:
                print(f"[CFD-WORKFLOW] RESTART interpreter :: {sim_id_s}", flush=True)
            interp = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=sim_meta,
                experiment_results=ex["latest_run_result"],
                verbose=verbose,
            )

            # Update entry with new interpreter results
            entry2 = dict(entry)
            entry2["interpreter"] = interp
            new_sims_entries.append(entry2)
            self._log_interpreter_rerun_required(sim_id_s, interp, verbose=verbose)
            self._log_rerun_skipped_at_cap(
                sim_id_s, interp, int(entry2.get("attempt", 0) or 0), max_reruns, verbose=verbose
            )

        pipeline_log["simulations"] = new_sims_entries
        self._write_interpreter_output_json(out_dir, new_sims_entries)

        # If some cases need rerun, perform rerun rounds here (restart should be resilient to workflow stops).
        while True:
            queue: List[str] = []
            for entry in new_sims_entries:
                sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
                sim_id = str(sim.get("simulation_id", "") or "")
                if not sim_id:
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                attempt_used = int(entry.get("attempt", 0) or 0)
                if rerun_required and attempt_used < max_reruns:
                    queue.append(sim_id)
                elif rerun_required:
                    self._log_rerun_skipped_at_cap(sim_id, interp, attempt_used, max_reruns, verbose=verbose)

            if not queue:
                break

            if verbose:
                print(f"[CFD-WORKFLOW] RESTART rerun-round queue={queue}", flush=True)

            for sim_id in queue:
                # Find entry
                entry = None
                for e in new_sims_entries:
                    sim = e.get("simulation", {}) if isinstance(e.get("simulation", {}), dict) else {}
                    if str(sim.get("simulation_id", "")) == str(sim_id):
                        entry = e
                        break
                if entry is None:
                    continue

                sim_dir = out_dir / str(sim_id)
                req_path = sim_dir / "user_requirement.txt"
                if req_path.exists():
                    try:
                        current_req = req_path.read_text(encoding="utf-8")
                    except Exception:
                        current_req = ""
                else:
                    hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
                    current_req = str(hyp.get("requirement", "") or hyp.get("prompt_for_foamagent", "") or "")

                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                if verbose:
                    print(f"[CFD-WORKFLOW] RESTART rerun :: {sim_id}", flush=True)

                sim_meta2 = cast(Dict[str, Any], entry.get("simulation", {}) or {})
                exp_idea = sim_meta2.get("case_data", None)

                # Choose closest successful case (by user requirement text),
                # then diff their OpenFOAM inputs (system/, constant/, 0/) ignoring polyMesh/.
                chosen_ref_sim_id = ""
                diff_report = ""
                try:
                    working_candidates: List[Dict[str, Any]] = []
                    for e2 in new_sims_entries:
                        if not isinstance(e2, dict):
                            continue
                        sim_e = e2.get("simulation", {}) if isinstance(e2.get("simulation", {}), dict) else {}
                        sim_id_e = sim_e.get("simulation_id", "")
                        if str(sim_id_e) == str(sim_id):
                            continue
                        interp_e = e2.get("interpreter", {}) if isinstance(e2.get("interpreter", {}), dict) else {}
                        if not interp_e:
                            continue
                        if not bool(interp_e.get("simulation_success", False)):
                            continue
                        if not bool(interp_e.get("requirement_met", False)):
                            continue
                        if not bool(interp_e.get("viz_ok", False)):
                            continue

                        latest_runs = e2.get("run_history", []) if isinstance(e2.get("run_history", []), list) else []
                        latest_run = latest_runs[-1] if latest_runs else {}
                        if isinstance(latest_run, dict) and latest_run.get("returncode") not in (0, None):
                            continue
                        working_candidates.append(e2)

                    if verbose:
                        print(
                            f"[CFD-WORKFLOW] RESTART rerun reference candidates :: target={sim_id} count={len(working_candidates)}",
                            flush=True,
                        )

                    chosen_ref_sim_id = self._select_closest_successful_case_llm(
                        target_user_requirement=current_req,
                        successful_entries=working_candidates,
                        out_dir=out_dir,
                        verbose=verbose,
                    )
                    if verbose and chosen_ref_sim_id:
                        print(
                            f"[CFD-WORKFLOW] RESTART rerun reference selected (LLM) :: target={sim_id} selected={chosen_ref_sim_id}",
                            flush=True,
                        )
                    if chosen_ref_sim_id:
                        diff_report = self._diff_inputs_llm(
                            working_sim_id=chosen_ref_sim_id,
                            failing_sim_id=str(sim_id),
                            out_dir=out_dir,
                            verbose=verbose,
                        )
                        if verbose:
                            print(
                                f"[CFD-WORKFLOW] RESTART rerun diff report :: target={sim_id} selected={chosen_ref_sim_id} chars={len(diff_report)}",
                                flush=True,
                            )
                except Exception:
                    chosen_ref_sim_id = ""
                    diff_report = ""

                revision = self.rerun_analysis.revise_requirement(
                    current_req,
                    cast(Dict[str, Any], interp or {}),
                    experiment_idea=exp_idea if isinstance(exp_idea, dict) else None,
                    reference_summary=None,
                    reference_diff_report=diff_report or None,
                    verbose=verbose,
                )
                if verbose:
                    print(
                        f"[CFD-WORKFLOW] RESTART rerun revise used_diff={bool(diff_report)} used_ref={bool(chosen_ref_sim_id)}",
                        flush=True,
                    )
                next_req = str(revision.get("requirement", "") or "")
                req_valid = bool(revision.get("valid", False))

                # Track requirement updates + attempts
                requirement_updates = entry.get("requirement_updates", []) if isinstance(entry.get("requirement_updates", []), list) else []
                attempt_used = int(entry.get("attempt", 0) or 0)
                requirement_updates.append(
                    {
                        "attempt": attempt_used,
                        "feedback": revision.get("feedback", []),
                        "valid": req_valid,
                        "requirement": next_req,
                    }
                )
                entry["requirement_updates"] = requirement_updates
                entry["attempt"] = attempt_used + 1

                if not req_valid or not next_req.strip():
                    entry["interpreter"] = {
                        **(cast(Dict[str, Any], interp or {})),
                        "rerun_required": True,
                        "rerun_reason": "rerun_analysis produced invalid/empty requirement; skipping rerun.",
                    }
                    continue

                self._write_requirement(req_path, next_req)
                run_result = self.foam.run(
                    user_requirement_path=req_path,
                    output_dir=sim_dir / "foam_output",
                    project_root=Path.cwd(),
                    execute=True,
                )

                run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
                run_history.append(run_result)
                entry["run_history"] = run_history

                # Re-interpret
                interp2 = self.interpreter.interpret(
                    idea_json=idea,
                    experiment_spec=cast(Dict[str, Any], entry.get("simulation", {}) or {}),
                    experiment_results=cast(Dict[str, Any], run_result if isinstance(run_result, dict) else {}),
                    verbose=verbose,
                )
                entry["interpreter"] = interp2
                self._log_interpreter_rerun_required(sim_id, interp2, verbose=verbose)
                self._log_rerun_skipped_at_cap(
                    sim_id,
                    interp2,
                    int(entry.get("attempt", 0) or 0),
                    max_reruns,
                    verbose=verbose,
                )

            pipeline_log["simulations"] = new_sims_entries
            self._write_json(pipeline_path, pipeline_log)
            self._write_interpreter_output_json(out_dir, new_sims_entries)

        # Run full analysis pipeline (viz + analysis text)
        if verbose:
            print("[CFD-WORKFLOW] RESTART analysis...", flush=True)
        # Strip helper-only keys before passing to analysis agent
        analysis_experiments: List[Dict[str, Any]] = []
        for ex in experiments:
            analysis_experiments.append(
                {
                    "simulation_id": ex["simulation_id"],
                    "case_name": ex["case_name"],
                    "description": ex["description"],
                    "experiment_idea": ex.get("experiment_idea", None),
                    "user_requirement": ex["user_requirement"],
                    "sim_dir": ex["sim_dir"],
                    "foam_output_dir": ex["foam_output_dir"],
                }
            )
        analysis_result = self.analysis.run_full_analysis_pipeline(analysis_experiments, topic, verbose=verbose)
        analysis_text = analysis_result.get("analysis_text", "")
        viz_bundle = analysis_result.get("visualizations", [])
        viz_bundle = self._vision_filter_viz_bundle(viz_bundle, analysis_experiments, verbose=verbose)
        data_proc = self.analysis.run_cross_experiment_data_processing(
            topic=topic,
            experiments=analysis_experiments,
            out_dir=out_dir,
            verbose=True,
        )
        data_proc_report_text = ""
        data_proc_report_path = str(data_proc.get("report_path", "") or "")
        if data_proc_report_path:
            rp = Path(data_proc_report_path)
            if rp.is_file():
                try:
                    data_proc_report_text = rp.read_text(encoding="utf-8")
                except Exception:
                    data_proc_report_text = ""
        data_proc_interp_text = str(data_proc.get("interpretation_text", "") or "")
        if not data_proc_interp_text:
            ip = Path(str(data_proc.get("interpretation_path", "") or ""))
            if ip.is_file():
                try:
                    data_proc_interp_text = ip.read_text(encoding="utf-8")
                except Exception:
                    data_proc_interp_text = ""
        # Make cross-experiment images visible to writer/reviewer path resolution.
        data_proc_images = [
            str(p) for p in (data_proc.get("images", []) or []) if isinstance(p, str) and p.strip()
        ]
        if data_proc_images:
            viz_bundle.append(
                {
                    "simulation_id": "cross_experiment",
                    "case_name": "cross_experiment_analysis",
                    "visualization": {"ok": True, "images": data_proc_images},
                }
            )

        analysis_path = out_dir / "analysis_report.md"
        self.analysis.save_analysis(analysis_path, analysis_text, topic=topic)
        pipeline_log["analysis"] = str(analysis_path)
        pipeline_log["cross_experiment_data_processing"] = data_proc

        # Map analysis visualizations back into pipeline_log simulations
        viz_by_id = {v["simulation_id"]: v.get("visualization") for v in viz_bundle if isinstance(v, dict)}
        for entry in new_sims_entries:
            sim_meta = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            sid = sim_meta.get("simulation_id")
            if sid and sid in viz_by_id:
                entry["analysis_visualization"] = viz_by_id[sid]

        # Run writer (paper + review loop)
        if verbose:
            print("[CFD-WORKFLOW] RESTART writer...", flush=True)
        exp_ideas_for_writer: List[Dict[str, Any]] = []
        for ex in analysis_experiments:
            exp_ideas_for_writer.append(
                {
                    "simulation_id": ex.get("simulation_id"),
                    "case_name": ex.get("case_name"),
                    "experiment_idea": ex.get("experiment_idea"),
                }
            )
        # Build section_context from idea + analysis text
        section_context = (
            json.dumps(idea, indent=2)
            + "\n\nPER-EXPERIMENT IDEAS/CONTEXT:\n"
            + json.dumps(exp_ideas_for_writer, indent=2, ensure_ascii=False)
            + "\n\n"
            + analysis_text
            + (
                "\n\nCROSS-EXPERIMENT DATA PROCESSING REPORT:\n"
                + data_proc_report_text
                if data_proc_report_text
                else ""
            )
            + (
                "\n\nCROSS-EXPERIMENT INTERPRETATION (LLM):\n"
                + data_proc_interp_text
                if data_proc_interp_text
                else ""
            )
        )
        paper_text, pdf_path, review_info = self.writer.write_paper_with_literature_and_review(
            topic=topic,
            section_context=section_context,
            out_dir=out_dir,
            work_dir=out_dir,
            ideation_literature_bundle=pipeline_log.get("ideation", {}),
            visualization_bundle=viz_bundle,
            verbose=verbose,
        )
        tex_path = out_dir / "paper_draft.tex"
        tex_path.write_text(paper_text, encoding="utf-8")
        pipeline_log["paper"] = {
            "tex": str(tex_path),
            "pdf": str(pdf_path) if pdf_path else None,
            "review": review_info,
        }

        # Persist updated pipeline_log
        pipeline_log["topic"] = topic
        self._write_json(pipeline_path, pipeline_log)

        if verbose:
            print("[CFD-WORKFLOW] RESTART completed.", flush=True)

        return {
            "pipeline_log": pipeline_log,
            "analysis": str(analysis_path),
            "paper": str(pdf_path) if pdf_path else None,
        }

    # ======================================================================
    # Resume helper: finish Foam-Agent runs that were not completed, then
    # reuse restart_from_foam to (re)run interpreter + analysis + writer.
    # ======================================================================

    def resume_after_runs(self, out_dir: Path, verbose: bool = True) -> Dict[str, Any]:
        """
        Resume pipeline in the Foam-Agent run stage and beyond.

        For each simulation in pipeline_log.json:
        - If run_history is empty OR latest run_result has no returncode,
          re-run Foam-Agent for that simulation using the last known
          validated requirement from the hypothesis stage.
        - Otherwise, leave its runs untouched.

        After ensuring all simulations have completed Foam-Agent runs,
        delegate to restart_from_foam to re-run interpreter + analysis + writer.
        """
        out_dir = out_dir.expanduser().resolve()
        if verbose:
            print(f"[CFD-WORKFLOW] RESUME after Foam-Agent runs in {out_dir}", flush=True)

        pipeline_path = out_dir / "pipeline_log.json"
        if not pipeline_path.is_file():
            # Back-compat: build pipeline_log.json from existing outputs if possible.
            if verbose:
                print("[CFD-WORKFLOW] No pipeline_log.json found; attempting to rebuild from outputs...", flush=True)
            rebuilt = self.restart_from_foam(out_dir=out_dir, verbose=verbose)
            # restart_from_foam will persist a pipeline_log.json if it can rebuild.
            if not pipeline_path.is_file():
                if verbose:
                    print("[CFD-WORKFLOW] Could not rebuild pipeline_log.json; nothing to resume.", flush=True)
                return rebuilt or {}

        try:
            pipeline_log = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except Exception:
            if verbose:
                print("[CFD-WORKFLOW] Could not parse pipeline_log.json; aborting resume.", flush=True)
            return {}

        sims_entries: List[Dict[str, Any]] = []
        if isinstance(pipeline_log.get("simulations"), list):
            sims_entries = [s for s in pipeline_log.get("simulations", []) if isinstance(s, dict)]

        if not sims_entries:
            if verbose:
                print("[CFD-WORKFLOW] No simulations in pipeline_log.json; nothing to resume.", flush=True)
            return {"pipeline_log": pipeline_log}

        # Ensure Foam-Agent runs exist for each simulation.
        for entry in sims_entries:
            sim_meta = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            sim_id = str(sim_meta.get("simulation_id", "") or "")
            if not sim_id:
                continue
            sim_dir = out_dir / sim_id
            sim_dir.mkdir(parents=True, exist_ok=True)

            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            latest_run = run_history[-1] if run_history else {}
            has_returncode = isinstance(latest_run, dict) and (latest_run.get("returncode") is not None)

            # If we already have a completed Foam-Agent run (success or failure), do not re-run here.
            if has_returncode:
                if verbose:
                    print(f"[CFD-WORKFLOW] RESUME: Foam run already completed for {sim_id} (returncode={latest_run.get('returncode')}); skipping.", flush=True)
                continue

            # Need to run Foam-Agent at least once for this simulation.
            hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
            req_text = str(hyp.get("requirement", "") or "")
            if not req_text.strip():
                if verbose:
                    print(f"[CFD-WORKFLOW] RESUME: No requirement text for {sim_id}; skipping Foam-Agent run.", flush=True)
                continue

            if verbose:
                print(f"[CFD-WORKFLOW] RESUME Foam-Agent :: {sim_id}", flush=True)
            req_path = sim_dir / "user_requirement.txt"
            self._write_requirement(req_path, req_text)

            run_result = self.foam.run(
                user_requirement_path=req_path,
                output_dir=sim_dir / "foam_output",
                project_root=Path.cwd(),
                execute=True,
            )
            run_history.append(run_result)
            entry["run_history"] = run_history

        # Update foamagent_output.json from updated run_history
        foam_cases: List[Dict[str, Any]] = []
        for entry in sims_entries:
            sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            latest_run = run_history[-1] if run_history else {}
            status = "unknown"
            if isinstance(latest_run, dict):
                if latest_run.get("skipped"):
                    status = "skipped"
                elif latest_run.get("planned"):
                    status = "planned"
                elif latest_run.get("returncode") == 0:
                    status = "success"
                elif latest_run.get("returncode") is not None:
                    status = "failed"
            foam_cases.append(
                {
                    "simulation_id": sim.get("simulation_id"),
                    "case_name": sim.get("case_name"),
                    "status": status,
                    "latest_run": latest_run,
                }
            )

        foam_output_path = out_dir / "foamagent_output.json"
        self._write_json(
            foam_output_path,
            {
                "total_cases": len(foam_cases),
                "success_count": sum(1 for c in foam_cases if c.get("status") == "success"),
                "failed_count": sum(1 for c in foam_cases if c.get("status") == "failed"),
                "skipped_count": sum(1 for c in foam_cases if c.get("status") == "skipped"),
                "planned_count": sum(1 for c in foam_cases if c.get("status") == "planned"),
                "cases": foam_cases,
            },
        )

        pipeline_log["simulations"] = sims_entries
        self._write_json(pipeline_path, pipeline_log)

        # Now delegate to restart_from_foam to (re)run interpreter + analysis + writer.
        return self.restart_from_foam(out_dir=out_dir, verbose=verbose)

    @staticmethod
    def _expand_simulations(
        idea: Dict[str, Any], max_total: int = 50
    ) -> List[Dict[str, Any]]:
        sims: List[Dict[str, Any]] = []

        # Preferred canonical schema: experiments[]
        experiments = idea.get("experiments", []) if isinstance(idea, dict) else []
        if isinstance(experiments, list) and experiments:
            for i, e in enumerate(experiments[:max_total], 1):
                if not isinstance(e, dict):
                    continue
                sims.append(
                    {
                        "simulation_id": e.get("experiment_id", f"sim_{i:03d}"),
                        "case_name": e.get("name", f"experiment_{i:03d}"),
                        "parameter_value": e.get("parameters", {}),
                        "description": e.get("notes", e.get("name", "")),
                        "visualization": "",
                        "case_data": e,
                    }
                )
            return sims[:max_total]

        # Generic fallback path: cases[] without explicit experiments[]
        cases = idea.get("cases", []) if isinstance(idea, dict) else []
        sim_id = 0
        for c in cases:
            if not isinstance(c, dict):
                continue
            sim_id += 1
            sims.append(
                {
                    "simulation_id": f"sim_{sim_id:03d}",
                    "case_name": c.get("name", f"case_{sim_id}"),
                    "parameter_value": c.get("parameters", {}),
                    "description": c.get("description", ""),
                    "visualization": "",
                    "case_data": c,
                }
            )
            if len(sims) >= max_total:
                return sims

        return sims[:max_total]

    @staticmethod
    def _locate_foam_dataset(foam_output_dir: Path) -> Path | None:
        """
        Use OpenFOAM-native loading path for PyVista:
        - Prefer an existing .foam file
        - If none exists, create one in the case folder and return it
        """
        if not foam_output_dir.exists():
            return None

        foam_files = sorted(foam_output_dir.rglob("*.foam"), key=lambda p: p.stat().st_mtime, reverse=True)
        if foam_files:
            return foam_files[0]

        # Create a marker .foam file so PyVista can load the OpenFOAM case directly.
        marker = foam_output_dir / "case.foam"
        try:
            marker.touch(exist_ok=True)
            return marker
        except Exception:
            return None

    @staticmethod
    def _write_requirement(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _interp_rerun_reason_for_display(interp: Dict[str, Any]) -> str:
        """Human-readable rerun motivation from interpreter output (several possible keys)."""
        if not isinstance(interp, dict):
            return ""
        r = interp.get("rerun_reason")
        if r is not None and str(r).strip():
            return str(r).strip()
        r2 = interp.get("reasons")
        if r2 is not None:
            if isinstance(r2, list):
                joined = "; ".join(str(x).strip() for x in r2 if str(x).strip())
                return joined
            s = str(r2).strip()
            if s:
                return s
        summ = interp.get("summary")
        if summ is not None and str(summ).strip():
            return str(summ).strip()
        issues = interp.get("issues")
        if issues is not None:
            if isinstance(issues, list):
                return "; ".join(str(x).strip() for x in issues if str(x).strip())
            s = str(issues).strip()
            if s:
                return s
        return ""

    @staticmethod
    def _log_interpreter_rerun_required(
        sim_id: str,
        interp: Dict[str, Any],
        *,
        verbose: bool,
        prefix: str = "[CFD-WORKFLOW]",
    ) -> None:
        if not verbose or not isinstance(interp, dict) or not bool(interp.get("rerun_required", False)):
            return
        reason = CFDWorkflow._interp_rerun_reason_for_display(interp)
        if reason:
            print(f"{prefix} Interpreter rerun_required :: {sim_id} — {reason}", flush=True)
        else:
            print(f"{prefix} Interpreter rerun_required :: {sim_id}", flush=True)

    @staticmethod
    def _log_rerun_skipped_at_cap(
        sim_id: str,
        interp: Dict[str, Any],
        attempt_used: int,
        max_reruns: int,
        *,
        verbose: bool,
    ) -> None:
        if (
            not verbose
            or not isinstance(interp, dict)
            or not bool(interp.get("rerun_required", False))
            or attempt_used < max_reruns
        ):
            return
        reason = CFDWorkflow._interp_rerun_reason_for_display(interp)
        base = (
            f"[CFD-WORKFLOW] Rerun not queued (foam_rerun_count={attempt_used} >= "
            f"workflow_max_reruns={max_reruns}) :: {sim_id}"
        )
        if reason:
            print(f"{base} — interpreter would rerun because: {reason}", flush=True)
        else:
            print(base, flush=True)

    @staticmethod
    def _build_interpreter_output_payload(
        sims_list: List[Dict[str, Any]],
        *,
        rerun_round: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Snapshot of per-case interpreter state for interpreter_output.json.
        foam_rerun_count / attempts_used mirror pipeline_log simulations[].attempt (lifetime corrective runs).
        """
        interp_cases: List[Dict[str, Any]] = []
        for entry in sims_list:
            if not isinstance(entry, dict):
                continue
            sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
            sim_id = sim.get("simulation_id", "unknown")
            rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            latest_run = run_history[-1] if run_history else {}
            latest_run = cast(Dict[str, Any], latest_run if isinstance(latest_run, dict) else {})
            attempt_val = int(entry.get("attempt", 0) or 0)
            rerun_reason_out = interp.get("rerun_reason") if isinstance(interp, dict) else None
            if rerun_reason_out is None or (isinstance(rerun_reason_out, str) and not str(rerun_reason_out).strip()):
                rerun_reason_out = (
                    CFDWorkflow._interp_rerun_reason_for_display(interp) if rerun_required else None
                )
            interp_cases.append(
                {
                    "simulation_id": sim.get("simulation_id"),
                    "case_name": sim.get("case_name"),
                    "rerun_required": rerun_required,
                    "rerun_reason": rerun_reason_out,
                    "status": ("rerun" if rerun_required else "ok"),
                    "attempts_used": attempt_val,
                    "foam_rerun_count": attempt_val,
                    "final_returncode": latest_run.get("returncode"),
                    "viz_final_ok": latest_run.get("viz_final_ok"),
                    "interpreter": interp,
                }
            )
        payload: Dict[str, Any] = {
            "total_cases": len(interp_cases),
            "rerun_required_count": sum(1 for c in interp_cases if c.get("rerun_required") is True),
            "ok_count": sum(1 for c in interp_cases if c.get("status") == "ok"),
            "cases": interp_cases,
        }
        if rerun_round is not None:
            payload["rerun_round"] = int(rerun_round)
        return payload

    def _write_interpreter_output_json(
        self,
        out_dir: Path,
        sims_list: List[Dict[str, Any]],
        *,
        rerun_round: Optional[int] = None,
    ) -> None:
        payload = self._build_interpreter_output_payload(sims_list, rerun_round=rerun_round)
        self._write_json(out_dir.expanduser().resolve() / "interpreter_output.json", payload)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _compact_experiment_idea_text(exp: Dict[str, Any]) -> str:
        idea = exp.get("case_data", None) or exp.get("experiment_idea", None)
        if isinstance(idea, dict):
            try:
                return json.dumps(idea, ensure_ascii=False)[:1200]
            except Exception:
                return str(idea)[:1200]
        if isinstance(idea, str):
            return idea[:1200]
        return str(exp.get("description", "") or "")[:800]

    def _all_experiment_ideas_summary(self, simulations: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for s in simulations or []:
            sid = str(s.get("simulation_id", "") or "")
            if not sid:
                continue
            cname = str(s.get("case_name", "") or "")
            lines.append(f"{sid} ({cname}): {self._compact_experiment_idea_text(s)}")
        return "\n".join(lines)[:8000]

    @staticmethod
    def _normalize_feedback_lines(interp: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for key in (
            "rerun_reason",
            "requirement_update",
            "requirement_updates",
            "recommended_requirement_update",
            "suggested_requirement_update",
            "next_requirement",
        ):
            val = interp.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(val.strip())
            elif isinstance(val, list):
                lines.extend(str(x).strip() for x in val if str(x).strip())

        health = interp.get("health")
        if isinstance(health, dict) and health.get("has_error_signals"):
            lines.append(
                "Previous run showed hard error signals. Tighten requirement to avoid invalid setup."
            )

        return lines


    @staticmethod
    def _viz_is_acceptable(interp: Dict[str, Any], run_result: Dict[str, Any]) -> bool:
        """Best-effort viz quality gate.

        Accepts viz if generation succeeded and interpreter does not explicitly request viz redo.
        """
        viz = run_result.get("viz") if isinstance(run_result, dict) else None
        viz_ok = isinstance(viz, dict) and bool(viz.get("ok", False))

        # Require at least one successfully generated diagnostic image.
        min_ok_images = 1
        ok_image_count = 0

        if isinstance(viz, dict):
            results = viz.get("results", [])
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict) or not bool(r.get("ok", False)):
                        continue
                    out = r.get("output")
                    if isinstance(out, str) and out.strip() and Path(out).exists():
                        ok_image_count += 1

        viz_ok = viz_ok and (ok_image_count >= min_ok_images)

        if not isinstance(interp, dict):
            return bool(viz_ok)

        explicit_bad = any(
            interp.get(k) is False
            for k in ("viz_ok", "viz_quality_ok", "visualization_ok", "viz_acceptable")
        )
        explicit_redo = any(
            bool(interp.get(k, False))
            for k in ("redo_viz", "viz_rerun_required", "regenerate_visualization")
        )

        return bool(viz_ok) and (not explicit_bad) and (not explicit_redo)

    @staticmethod
    def _viz_retry_reason(interp: Dict[str, Any], run_result: Dict[str, Any]) -> str:
        if isinstance(interp, dict):
            for k in (
                "viz_rerun_reason",
                "viz_reason",
                "visualization_issue",
                "visualization_feedback",
                "rerun_reason",
            ):
                v = interp.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        viz = run_result.get("viz") if isinstance(run_result, dict) else None
        if isinstance(viz, dict):
            err = viz.get("error")
            if isinstance(err, str) and err.strip():
                return err.strip()
        return "Visualization diagnostics were not acceptable; regenerating plots."

    def _revise_requirement_from_feedback(
        self,
        req: str,
        interp: Dict[str, Any],
        experiment_idea: Dict[str, Any] | None = None,
        run_topic: str = "",
        all_experiment_ideas: str = "",
        previous_requirement: str = "",
    ) -> Dict[str, Any]:
        feedback = self._normalize_feedback_lines(interp)
        if not feedback:
            verdict = self.hypothesis.llm_validate_requirement(req)
            return {
                "requirement": req,
                "valid": bool(verdict.get("valid", False)),
                "feedback": [],
            }

        guidance_lines: List[str] = [
            "Update the requirement to address interpreter-detected execution issues.",
            "Keep the requirement executable by Foam-Agent and do not include visualization instructions.",
        ]
        guidance_lines = self._append_experiment_idea_to_guidance(guidance_lines, experiment_idea)

        revised = self.hypothesis.repair_requirement(
            req,
            issues=[f"Interpreter feedback: {x}" for x in feedback],
            guidance=guidance_lines,
            run_topic=run_topic,
            all_experiment_ideas=all_experiment_ideas,
            current_experiment_idea=json.dumps(experiment_idea, ensure_ascii=False) if isinstance(experiment_idea, dict) else "",
            previous_requirement=previous_requirement,
        )
        revised = self.hypothesis._strip_visualization_mentions(revised)

        verdict = self.hypothesis.llm_validate_requirement(revised)
        if verdict.get("valid", False):
            return {"requirement": revised, "valid": True, "feedback": feedback}

        repair_guidance = verdict.get("repair_guidance", [])
        if isinstance(repair_guidance, list):
            repair_guidance = self._append_experiment_idea_to_guidance(repair_guidance, experiment_idea)

        repaired = self.hypothesis.repair_requirement(
            revised,
            issues=verdict.get("issues", []),
            guidance=repair_guidance,
            run_topic=run_topic,
            all_experiment_ideas=all_experiment_ideas,
            current_experiment_idea=json.dumps(experiment_idea, ensure_ascii=False) if isinstance(experiment_idea, dict) else "",
            previous_requirement=previous_requirement,
        )
        repaired = self.hypothesis._strip_visualization_mentions(repaired)
        repaired_verdict = self.hypothesis.llm_validate_requirement(repaired)
        return {
            "requirement": repaired,
            "valid": bool(repaired_verdict.get("valid", False)),
            "feedback": feedback,
            "validator": repaired_verdict,
        }

    def _append_experiment_idea_to_guidance(
        self, guidance: List[str], experiment_idea: Dict[str, Any] | None
    ) -> List[str]:
        if not isinstance(experiment_idea, dict) or not experiment_idea:
            return guidance
        idea_txt = json.dumps(experiment_idea, ensure_ascii=False, sort_keys=True)
        if len(idea_txt) > 1200:
            idea_txt = idea_txt[:1200] + "..."
        guidance.append(
            "Experiment concept you must preserve while repairing the requirement:\n"
            f"{idea_txt}"
        )
        return guidance

    def _build_langgraph_app(self):
        def log_stage(msg: str) -> None:
            print(f"[CFD-WORKFLOW] {msg}", flush=True)

        def ideate(state: WorkflowState) -> WorkflowState:
            log_stage("START ideation")
            out_dir = state["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            ideation_result = run_ideation(
                settings=self.settings,
                research_topic=state["topic"],
                verbose=bool(state.get("verbose", True)),
            )
            idea = cast(Dict[str, Any], ideation_result.get("idea", {}) or {})
            self._write_json(
                out_dir / "ideation_output.json",
                {
                    "topic": state["topic"],
                    "ideation": ideation_result,
                    "experiments": idea.get("experiments", []) if isinstance(idea, dict) else [],
                },
            )
            log_stage("END ideation")
            return {
                "ideation_result": ideation_result,
                "idea": idea,
            }

        def expand_and_init(state: WorkflowState) -> WorkflowState:
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            simulations = self._expand_simulations(
                idea,
                max_total=min(
                    self.settings.ideation_max_experiments,
                    self.settings.workflow_max_experiments_total,
                ),
            )

            # Batch hypothesis generation for ALL experiments before any Foam-Agent execution.
            log_stage(f"START hypothesis-batch :: total={len(simulations)}")
            hyp_items: List[Dict[str, Any]] = []
            updated_sims: List[Dict[str, Any]] = []
            all_ideas_summary = self._all_experiment_ideas_summary(simulations)
            prev_req_for_consistency = ""
            for sim in simulations:
                sim_id = str(sim.get("simulation_id", ""))
                log_stage(f"START hypothesis :: {sim_id}")
                hyp = self.hypothesis.generate_validated_requirement(
                    idea=idea,
                    simulation=sim,
                    run_topic=str(state.get("topic", "") or ""),
                    all_experiment_ideas=all_ideas_summary,
                    current_experiment_idea=self._compact_experiment_idea_text(sim),
                    previous_requirement=prev_req_for_consistency,
                    max_retries=3,
                )
                req_text = str(hyp.get("requirement", "") or "")
                req_valid = bool(hyp.get("valid", False))
                log_stage(f"END hypothesis :: {sim_id} (valid={req_valid})")
                prev_req_for_consistency = req_text or prev_req_for_consistency

                sim_enriched = dict(sim)
                sim_enriched["_hyp"] = hyp
                sim_enriched["_req_text"] = req_text
                sim_enriched["_req_valid"] = req_valid
                updated_sims.append(sim_enriched)

                hyp_items.append(
                    {
                        "simulation_id": sim_id,
                        "case_name": sim.get("case_name"),
                        "experiment": sim,
                        "prompt_for_foamagent": req_text,
                        "valid": req_valid,
                        "history": hyp.get("history", []),
                    }
                )

            self._write_json(
                state["out_dir"] / "hypothesis_output.json",
                {
                    "topic": state["topic"],
                    "count": len(hyp_items),
                    "items": hyp_items,
                },
            )
            log_stage("END hypothesis-batch")

            pipeline_log: Dict[str, Any] = {
                "topic": state["topic"],
                "ideation": state.get("ideation_result", {}),
                "simulations": [],
                "analysis": None,
                "paper": None,
            }
            return {
                "simulations": updated_sims,
                "sim_index": 0,
                "pipeline_log": pipeline_log,
            }

        def prepare_next_sim(state: WorkflowState) -> WorkflowState:
            idx = int(state.get("sim_index", 0) or 0)
            sims = state.get("simulations", []) or []
            if idx >= len(sims):
                return {}

            sim = sims[idx]
            log_stage(f"START case {idx + 1}/{len(sims)} :: {sim.get('simulation_id', 'unknown')}")
            sim_id = sim.get("simulation_id", f"sim_{idx:03d}")
            sim_dir = state["out_dir"] / str(sim_id)
            sim_dir.mkdir(parents=True, exist_ok=True)
            return {
                "current_sim": sim,
                "sim_dir": sim_dir,
                "hyp": cast(Dict[str, Any], sim.get("_hyp", {}) or {}),
                "req_text": str(sim.get("_req_text", "") or ""),
                "req_valid": bool(sim.get("_req_valid", False)),
                "attempt": 0,
                "run_history": [],
                "run_result": {},
                "interp": {},
                "requirement_updates": [],
                "skip_run": False,
            }

        def route_after_prepare(state: WorkflowState) -> str:
            idx = int(state.get("sim_index", 0) or 0)
            sims = state.get("simulations", []) or []
            return "precheck" if idx < len(sims) else "final_artifacts_gate"

        def generate_requirement(state: WorkflowState) -> WorkflowState:
            current_sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            log_stage(f"START hypothesis :: {current_sim.get('simulation_id', 'unknown')}")
            sims = cast(List[Dict[str, Any]], state.get("simulations", []) or [])
            idx = int(state.get("sim_index", 0) or 0)
            prev_req_for_consistency = ""
            if idx > 0 and idx - 1 < len(sims):
                prev_req_for_consistency = str((sims[idx - 1] or {}).get("_req_text", "") or "")
            hyp = self.hypothesis.generate_validated_requirement(
                idea=cast(Dict[str, Any], state.get("idea", {}) or {}),
                simulation=cast(Dict[str, Any], state.get("current_sim", {}) or {}),
                run_topic=str(state.get("topic", "") or ""),
                all_experiment_ideas=self._all_experiment_ideas_summary(sims),
                current_experiment_idea=self._compact_experiment_idea_text(current_sim),
                previous_requirement=prev_req_for_consistency,
                max_retries=3,
                verbose=bool(state.get("verbose", True)),
            )
            req_text = str(hyp.get("requirement", "") or "")
            req_valid = bool(hyp.get("valid", False))

            out_dir = state["out_dir"]
            current_sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = str(current_sim.get("simulation_id", ""))
            hypothesis_json_path = out_dir / "hypothesis_output.json"
            hypothesis_payload = self._read_json(hypothesis_json_path)
            items = hypothesis_payload.get("items", []) if isinstance(hypothesis_payload.get("items", []), list) else []
            item = {
                "simulation_id": sim_id,
                "case_name": current_sim.get("case_name"),
                "experiment": current_sim,
                "prompt_for_foamagent": req_text,
                "valid": req_valid,
                "history": hyp.get("history", []),
            }
            replaced = False
            for i, existing in enumerate(items):
                if isinstance(existing, dict) and str(existing.get("simulation_id", "")) == sim_id:
                    items[i] = item
                    replaced = True
                    break
            if not replaced:
                items.append(item)
            self._write_json(
                hypothesis_json_path,
                {
                    "topic": state["topic"],
                    "count": len(items),
                    "items": items,
                },
            )
            log_stage(f"END hypothesis :: {sim_id} (valid={req_valid})")

            return {
                "hyp": hyp,
                "req_text": req_text,
                "req_valid": req_valid,
                "attempt": 0,
            }

        def precheck(state: WorkflowState) -> WorkflowState:
            req_text = str(state.get("req_text", "") or "")
            req_valid = bool(state.get("req_valid", False))
            run_history = list(state.get("run_history", []) or [])

            if not req_text.strip():
                run_result = {
                    "planned": False,
                    "skipped": True,
                    "reason": "Empty user requirement; Foam run not attempted.",
                }
                run_history.append(run_result)
                interp = {
                    "rerun_required": False,
                    "rerun_reason": "Requirement is empty; cannot execute.",
                }
                return {
                    "skip_run": True,
                    "run_result": run_result,
                    "run_history": run_history,
                    "interp": interp,
                }

            if not req_valid:
                run_result = {
                    "planned": False,
                    "skipped": True,
                    "reason": "Hypothesis requirement failed validation gate; Foam run not attempted.",
                }
                run_history.append(run_result)
                interp = {
                    "rerun_required": False,
                    "rerun_reason": "Requirement invalid; fix hypothesis before execution.",
                }
                return {
                    "skip_run": True,
                    "run_result": run_result,
                    "run_history": run_history,
                    "interp": interp,
                }

            return {"skip_run": False}

        def route_after_precheck(state: WorkflowState) -> str:
            return "append_simulation_log" if state.get("skip_run") else "foam_run"

        def foam_run(state: WorkflowState) -> WorkflowState:
            sim_dir = state["sim_dir"]
            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = sim.get("simulation_id", "unknown")
            log_stage(f"START foamagent :: {sim_id}")
            req_path = sim_dir / "user_requirement.txt"
            self._write_requirement(req_path, str(state.get("req_text", "") or ""))

            run_result = self.foam.run(
                user_requirement_path=req_path,
                output_dir=sim_dir / "foam_output",
                project_root=Path.cwd(),
                execute=bool(state.get("execute", False)),
            )

            run_history = list(state.get("run_history", []) or [])
            run_history.append(run_result)
            rc = run_result.get("returncode") if isinstance(run_result, dict) else None
            log_stage(f"END foamagent :: {sim_id} (returncode={rc})")
            return {"run_result": run_result, "run_history": run_history}

        def interpret(state: WorkflowState) -> WorkflowState:
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = sim.get("simulation_id", "unknown")
            log_stage(f"START interpreter :: {sim_id}")

            run_result = cast(Dict[str, Any], state.get("run_result", {}) or {})
            interp: Dict[str, Any] = {}

            # Interpreter owns viz: it generates PyVista script, runs it, retries viz up to 25x,
            # then interprets with user_req + images and sets rerun_required.
            interp = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=sim,
                experiment_results=run_result,
                verbose=bool(state.get("verbose", True)),
            )
            run_result["viz_attempts"] = interp.get("viz_attempts", []) if isinstance(interp, dict) else []
            run_result["viz_final_ok"] = bool(interp.get("viz_ok", False)) if isinstance(interp, dict) else False
            run_result["viz"] = {
                "ok": run_result["viz_final_ok"],
                "from_interpreter": True,
            }

            log_stage(
                f"END interpreter :: {sim_id} (rerun_required={bool(interp.get('rerun_required', False)) if isinstance(interp, dict) else False})"
            )
            return {"interp": interp, "run_result": run_result}

        def route_after_interpret(state: WorkflowState) -> str:
            if not bool(state.get("execute", False)):
                return "append_simulation_log"

            rerun = bool((state.get("interp", {}) or {}).get("rerun_required", False))
            attempt = int(state.get("attempt", 0) or 0)
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            if rerun and attempt < max_reruns:
                return "revise_requirement"
            return "append_simulation_log"

        def revise_requirement(state: WorkflowState) -> WorkflowState:
            req_text = str(state.get("req_text", "") or "")
            interp = cast(Dict[str, Any], state.get("interp", {}) or {})
            current_sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            exp_idea = current_sim.get("case_data", None)
            sims = cast(List[Dict[str, Any]], state.get("simulations", []) or [])
            idx = int(state.get("sim_index", 0) or 0)
            prev_req_for_consistency = ""
            if idx > 0 and idx - 1 < len(sims):
                prev_req_for_consistency = str((sims[idx - 1] or {}).get("_req_text", "") or "")
            revision = self._revise_requirement_from_feedback(
                req_text,
                interp,
                experiment_idea=exp_idea if isinstance(exp_idea, dict) else None,
                run_topic=str(state.get("topic", "") or ""),
                all_experiment_ideas=self._all_experiment_ideas_summary(sims),
                previous_requirement=prev_req_for_consistency,
            )
            next_req = str(revision.get("requirement", "") or "")
            req_valid = bool(revision.get("valid", False))

            requirement_updates = list(state.get("requirement_updates", []) or [])
            attempt = int(state.get("attempt", 0) or 0)
            requirement_updates.append(
                {
                    "attempt": attempt,
                    "feedback": revision.get("feedback", []),
                    "valid": req_valid,
                    "requirement": next_req,
                }
            )

            return {
                "req_text": next_req,
                "req_valid": req_valid,
                "requirement_updates": requirement_updates,
                "attempt": attempt + 1,
            }

        def append_simulation_log(state: WorkflowState) -> WorkflowState:
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            sims_list.append(
                {
                    "simulation": state.get("current_sim", {}),
                    "hypothesis": state.get("hyp", {}),
                    "run_history": state.get("run_history", []),
                    "interpreter": state.get("interp", {}),
                    "requirement_updates": state.get("requirement_updates", []),
                    "attempt": int(state.get("attempt", 0) or 0),
                }
            )
            pipeline_log["simulations"] = sims_list

            out_dir = state["out_dir"]

            foam_cases: List[Dict[str, Any]] = []
            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                run_history = entry.get("run_history", []) if isinstance(entry, dict) else []
                latest_run = run_history[-1] if isinstance(run_history, list) and run_history else {}
                status = "unknown"
                if isinstance(latest_run, dict):
                    if latest_run.get("skipped"):
                        status = "skipped"
                    elif latest_run.get("planned"):
                        status = "planned"
                    elif latest_run.get("returncode") == 0:
                        status = "success"
                    elif latest_run.get("returncode") is not None:
                        status = "failed"
                foam_cases.append(
                    {
                        "simulation_id": sim.get("simulation_id"),
                        "case_name": sim.get("case_name"),
                        "status": status,
                        "latest_run": latest_run,
                    }
                )

            self._write_json(
                out_dir / "foamagent_output.json",
                {
                    "total_cases": len(foam_cases),
                    "success_count": sum(1 for c in foam_cases if c.get("status") == "success"),
                    "failed_count": sum(1 for c in foam_cases if c.get("status") == "failed"),
                    "skipped_count": sum(1 for c in foam_cases if c.get("status") == "skipped"),
                    "planned_count": sum(1 for c in foam_cases if c.get("status") == "planned"),
                    "cases": foam_cases,
                },
            )

            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            log_stage(f"END simulation :: {sim.get('simulation_id', 'unknown')}")

            return {"pipeline_log": pipeline_log, "sim_index": int(state.get("sim_index", 0) or 0) + 1}

        def final_artifacts_gate(state: WorkflowState) -> WorkflowState:
            # no-op; routing decides next node
            return {}

        def interpret_batch(state: WorkflowState) -> WorkflowState:
            if not bool(state.get("execute", False)):
                return {}

            log_stage("START interpreter-batch")
            out_dir = state["out_dir"]
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            verbose_ib = bool(state.get("verbose", True))
            max_reruns_ib = int(self.settings.workflow_max_reruns_per_experiment)

            # Optimization: if interpreter_output.json exists, only (re)interpret
            # cases that are currently marked rerun_required=true in that snapshot.
            snapshot_by_sim_id: Dict[str, Dict[str, Any]] = {}
            snap_path = out_dir / "interpreter_output.json"
            if snap_path.is_file():
                snap = self._read_json(snap_path)
                snap_cases = snap.get("cases", []) if isinstance(snap, dict) else []
                if isinstance(snap_cases, list):
                    for c in snap_cases:
                        if isinstance(c, dict) and c.get("simulation_id") is not None:
                            snapshot_by_sim_id[str(c.get("simulation_id"))] = c

            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                sim_id = sim.get("simulation_id", "unknown")
                log_stage(f"START interpreter :: {sim_id}")

                snap_case = snapshot_by_sim_id.get(str(sim_id))
                snap_rr = bool(snap_case.get("rerun_required", False)) if isinstance(snap_case, dict) else False
                snap_interp = snap_case.get("interpreter") if isinstance(snap_case, dict) else None
                snap_attempts_used = None
                if isinstance(snap_case, dict):
                    snap_attempts_used = snap_case.get("attempts_used", snap_case.get("foam_rerun_count"))

                if snap_case and not snap_rr and isinstance(snap_interp, dict):
                    entry["interpreter"] = snap_interp
                    if snap_attempts_used is not None:
                        try:
                            entry["attempt"] = max(int(entry.get("attempt", 0) or 0), int(snap_attempts_used))
                        except Exception:
                            pass
                    if verbose_ib:
                        print(
                            f"[CFD-WORKFLOW] interpreter-batch skip :: {sim_id} (snapshot rerun_required=false)",
                            flush=True,
                        )
                    log_stage(f"END interpreter :: {sim_id} (rerun_required=False; snapshot-skip)")
                    continue

                run_history = entry.get("run_history", []) if isinstance(entry, dict) else []
                run_result = run_history[-1] if isinstance(run_history, list) and run_history else {}
                run_result = cast(Dict[str, Any], run_result if isinstance(run_result, dict) else {})

                sim_dir = out_dir / str(sim_id)
                if not run_result.get("output_dir") and (sim_dir / "foam_output").exists():
                    run_result["output_dir"] = str(sim_dir / "foam_output")

                interp = self.interpreter.interpret(
                    idea_json=idea,
                    experiment_spec=cast(Dict[str, Any], sim or {}),
                    experiment_results=run_result,
                    verbose=verbose_ib,
                )
                run_result["viz_attempts"] = interp.get("viz_attempts", []) if isinstance(interp, dict) else []
                run_result["viz_final_ok"] = bool(interp.get("viz_ok", False)) if isinstance(interp, dict) else False
                run_result["viz"] = {
                    "ok": run_result["viz_final_ok"],
                    "from_interpreter": True,
                }

                entry["interpreter"] = interp
                if isinstance(run_history, list) and run_history:
                    run_history[-1] = run_result
                    entry["run_history"] = run_history

                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                self._log_interpreter_rerun_required(str(sim_id), interp, verbose=verbose_ib)
                self._log_rerun_skipped_at_cap(
                    str(sim_id),
                    interp,
                    int(entry.get("attempt", 0) or 0),
                    max_reruns_ib,
                    verbose=verbose_ib,
                )
                log_stage(f"END interpreter :: {sim_id} (rerun_required={rerun_required})")

            pipeline_log["simulations"] = sims_list
            self._write_interpreter_output_json(out_dir, sims_list)
            log_stage("END interpreter-batch")
            return {"pipeline_log": pipeline_log}

        def rerun_round_start(state: WorkflowState) -> WorkflowState:
            """Start a rerun round by building a queue of cases to rerun."""
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            queue: List[str] = []
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
                sim_id = str(sim.get("simulation_id", "") or "")
                if not sim_id:
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                attempt_used = int(entry.get("attempt", 0) or 0)
                if rerun_required and attempt_used < max_reruns:
                    queue.append(sim_id)
                elif rerun_required:
                    self._log_rerun_skipped_at_cap(
                        sim_id,
                        interp,
                        attempt_used,
                        max_reruns,
                        verbose=bool(state.get("verbose", True)),
                    )

            round_idx = int(state.get("rerun_round", 0) or 0) + 1
            log_stage(f"RERUN-ROUND-START round={round_idx} queue={queue}")
            return {"rerun_queue": queue, "rerun_idx": 0, "rerun_round": round_idx}

        def route_after_rerun_round_start(state: WorkflowState) -> str:
            queue = state.get("rerun_queue", []) or []
            idx = int(state.get("rerun_idx", 0) or 0)
            return "rerun_run_one" if idx < len(queue) else "analysis_and_writer"

        def _get_entry_by_sim_id(
            sims_list: List[Dict[str, Any]], sim_id: str
        ) -> Optional[Dict[str, Any]]:
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
                if str(sim.get("simulation_id", "")) == str(sim_id):
                    return entry
            return None

        def rerun_run_one(state: WorkflowState) -> WorkflowState:
            """
            For one queued case: revise requirement from interpreter report, run Foam-Agent, re-interpret.
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            queue = list(state.get("rerun_queue", []) or [])
            idx = int(state.get("rerun_idx", 0) or 0)
            if idx >= len(queue):
                return {"pipeline_log": pipeline_log, "rerun_idx": idx}

            sim_id = str(queue[idx])
            entry = _get_entry_by_sim_id(sims_list, sim_id)
            if entry is None:
                log_stage(f"RERUN-SKIP missing entry for sim_id={sim_id}")
                return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

            sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}

            # Determine last requirement text (prefer saved file, fallback to hypothesis).
            out_dir = state["out_dir"]
            sim_dir = out_dir / str(sim_id)
            req_path = sim_dir / "user_requirement.txt"
            if req_path.exists():
                try:
                    current_req = req_path.read_text(encoding="utf-8")
                except Exception:
                    current_req = ""
            else:
                hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
                current_req = str(hyp.get("requirement", "") or hyp.get("prompt_for_foamagent", "") or "")

            # LLM chooses the closest successful case (based on user requirement text),
            # then LLM diffs system/constant/0 inputs (excluding polyMesh) to produce
            # an actionable diff report for repairing the failing requirement.
            chosen_ref_sim_id = ""
            diff_report = ""
            try:
                working_candidates: List[Dict[str, Any]] = []
                for e in sims_list:
                    if not isinstance(e, dict):
                        continue
                    sim_e = e.get("simulation", {}) if isinstance(e.get("simulation", {}), dict) else {}
                    interp_e = e.get("interpreter", {}) if isinstance(e.get("interpreter", {}), dict) else {}
                    latest_runs = e.get("run_history", []) if isinstance(e.get("run_history", []), list) else []
                    latest_run = latest_runs[-1] if latest_runs else {}
                    if not isinstance(latest_run, dict):
                        continue
                    rc = latest_run.get("returncode")
                    if rc != 0:
                        continue
                    if not isinstance(interp_e, dict):
                        continue
                    if not bool(interp_e.get("simulation_success", False)):
                        continue
                    if not bool(interp_e.get("requirement_met", False)):
                        continue
                    if not bool(interp_e.get("viz_ok", False)):
                        continue
                    # Don't use the current failing case as its own reference.
                    if str(sim_e.get("simulation_id", "")) == str(sim_id):
                        continue
                    working_candidates.append(e)

                if bool(state.get("verbose", True)):
                    print(
                        f"[CFD-WORKFLOW] RERUN-REFERENCE candidates :: target={sim_id} count={len(working_candidates)}",
                        flush=True,
                    )

                chosen_ref_sim_id = self._select_closest_successful_case_llm(
                    target_user_requirement=current_req,
                    successful_entries=working_candidates,
                    out_dir=out_dir,
                    verbose=bool(state.get("verbose", True)),
                )
                if chosen_ref_sim_id:
                    if bool(state.get("verbose", True)):
                        print(
                            f"[CFD-WORKFLOW] RERUN-REFERENCE (LLM) target={sim_id} selected={chosen_ref_sim_id}",
                            flush=True,
                        )
                    diff_report = self._diff_inputs_llm(
                        working_sim_id=chosen_ref_sim_id,
                        failing_sim_id=str(sim_id),
                        out_dir=out_dir,
                        verbose=bool(state.get("verbose", True)),
                    )
                    if bool(state.get("verbose", True)):
                        print(
                            f"[CFD-WORKFLOW] RERUN-REFERENCE diff report :: target={sim_id} selected={chosen_ref_sim_id} chars={len(diff_report)}",
                            flush=True,
                        )
            except Exception:
                chosen_ref_sim_id = ""
                diff_report = ""

            if bool(state.get("verbose", True)):
                if chosen_ref_sim_id and diff_report:
                    print(
                        f"[CFD-WORKFLOW] RERUN-REFERENCE diff report generated for target={sim_id} selected={chosen_ref_sim_id}",
                        flush=True,
                    )
                elif chosen_ref_sim_id:
                    print(
                        f"[CFD-WORKFLOW] RERUN-REFERENCE diff report empty for target={sim_id} selected={chosen_ref_sim_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"[CFD-WORKFLOW] RERUN-REFERENCE target={sim_id} selected=NONE (no suitable working case)",
                        flush=True,
                    )

            log_stage(f"START rerun_analysis :: {sim_id}")
            exp_idea = sim.get("case_data", None) if isinstance(sim, dict) else None
            revision = self.rerun_analysis.revise_requirement(
                current_req,
                cast(Dict[str, Any], interp or {}),
                experiment_idea=exp_idea if isinstance(exp_idea, dict) else None,
                reference_summary=None,
                reference_diff_report=diff_report or None,
                verbose=bool(state.get("verbose", True)),
            )
            if bool(state.get("verbose", True)):
                print(
                    f"[CFD-WORKFLOW] RERUN-REFERENCE revise used_diff={bool(diff_report)} used_ref={bool(chosen_ref_sim_id)}",
                    flush=True,
                )
            next_req = str(revision.get("requirement", "") or "")
            req_valid = bool(revision.get("valid", False))
            log_stage(f"END rerun_analysis :: {sim_id} (valid={req_valid})")

            requirement_updates = entry.get("requirement_updates", []) if isinstance(entry.get("requirement_updates", []), list) else []
            attempt_used = int(entry.get("attempt", 0) or 0)
            requirement_updates.append(
                {
                    "attempt": attempt_used,
                    "feedback": revision.get("feedback", []),
                    "valid": req_valid,
                    "requirement": next_req,
                }
            )
            entry["requirement_updates"] = requirement_updates
            entry["attempt"] = attempt_used + 1

            # If invalid, don't run Foam-Agent; keep rerun_required True but record reason.
            if not req_valid or not next_req.strip():
                entry["interpreter"] = {
                    **(cast(Dict[str, Any], interp or {})),
                    "rerun_required": True,
                    "rerun_reason": "rerun_analysis produced invalid/empty requirement; skipping rerun.",
                }
                pipeline_log["simulations"] = sims_list
                return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

            # Run Foam-Agent for this case
            log_stage(f"START foamagent (rerun) :: {sim_id}")
            self._write_requirement(req_path, next_req)
            run_result = self.foam.run(
                user_requirement_path=req_path,
                output_dir=sim_dir / "foam_output",
                project_root=Path.cwd(),
                execute=True,
            )
            rc = run_result.get("returncode") if isinstance(run_result, dict) else None
            log_stage(f"END foamagent (rerun) :: {sim_id} (returncode={rc})")

            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            run_history.append(run_result)
            entry["run_history"] = run_history

            # Re-interpret only this case
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            run_result = cast(Dict[str, Any], run_result if isinstance(run_result, dict) else {})
            if not run_result.get("output_dir") and (sim_dir / "foam_output").exists():
                run_result["output_dir"] = str(sim_dir / "foam_output")

            log_stage(f"START interpreter (rerun) :: {sim_id}")
            interp2 = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=cast(Dict[str, Any], sim or {}),
                experiment_results=run_result,
                verbose=bool(state.get("verbose", True)),
            )
            log_stage(
                f"END interpreter (rerun) :: {sim_id} (rerun_required={bool(interp2.get('rerun_required', False)) if isinstance(interp2, dict) else False})"
            )
            entry["interpreter"] = interp2
            v = bool(state.get("verbose", True))
            max_rr = int(self.settings.workflow_max_reruns_per_experiment)
            self._log_interpreter_rerun_required(sim_id, interp2, verbose=v)
            self._log_rerun_skipped_at_cap(
                sim_id, interp2, int(entry.get("attempt", 0) or 0), max_rr, verbose=v
            )

            pipeline_log["simulations"] = sims_list
            return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

        def route_after_rerun_run_one(state: WorkflowState) -> str:
            queue = state.get("rerun_queue", []) or []
            idx = int(state.get("rerun_idx", 0) or 0)
            return "rerun_run_one" if idx < len(queue) else "rerun_finalize_round"

        def rerun_finalize_round(state: WorkflowState) -> WorkflowState:
            """
            Write an updated interpreter_output.json snapshot after rerun round.
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            out_dir = state["out_dir"]
            rr = int(state.get("rerun_round", 0) or 0)

            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                interp = entry.get("interpreter", {}) if isinstance(entry, dict) else {}
                sim_id = sim.get("simulation_id", "unknown")
                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                log_stage(f"RERUN-STATUS :: {sim_id} rerun_required={rerun_required}")

            self._write_interpreter_output_json(out_dir, sims_list, rerun_round=rr)
            pipeline_log["simulations"] = sims_list
            return {"pipeline_log": pipeline_log}

        def route_after_rerun_finalize_round(state: WorkflowState) -> str:
            """
            Continue rerun rounds until no case needs rerun (or attempts exhausted).
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                if not isinstance(interp, dict):
                    continue
                if bool(interp.get("rerun_required", False)) and int(entry.get("attempt", 0) or 0) < max_reruns:
                    return "rerun_round_start"
            return "analysis_and_writer"

        def route_after_interpret_batch(state: WorkflowState) -> str:
            """
            If any case needs rerun, enter rerun loop; otherwise proceed to analysis.
            """
            if not bool(state.get("execute", False)):
                return "analysis_and_writer"
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                if not isinstance(interp, dict):
                    continue
                if bool(interp.get("rerun_required", False)) and int(entry.get("attempt", 0) or 0) < max_reruns:
                    return "rerun_round_start"
            return "analysis_and_writer"

        def route_after_final_gate(state: WorkflowState) -> str:
            if bool(state.get("execute", False)):
                return "interpret_batch"
            allow_final = bool(state.get("allow_non_executed_artifacts", False))
            return "analysis_and_writer" if allow_final else "save_pipeline_log"

        def analysis_and_writer(state: WorkflowState) -> WorkflowState:
            log_stage("START analysis")
            out_dir = state["out_dir"]
            topic = state["topic"]
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})

            # Build experiments list for full pipeline: decide viz -> create with interpreter ref -> analyze with images
            experiments: List[Dict[str, Any]] = []
            for entry in pipeline_log.get("simulations", []) or []:
                sim_meta = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                sim_id = sim_meta.get("simulation_id")
                if not sim_id:
                    continue
                sim_dir = out_dir / str(sim_id)
                foam_output_dir = sim_dir / "foam_output"
                if CFDWorkflow._locate_foam_dataset(foam_output_dir) is None:
                    continue
                hyp = entry.get("hypothesis", {}) if isinstance(entry, dict) else {}
                user_req = (hyp.get("requirement", "") or "") if isinstance(hyp, dict) else ""
                if not user_req and (sim_dir / "user_requirement.txt").exists():
                    user_req = (sim_dir / "user_requirement.txt").read_text(encoding="utf-8")
                experiments.append({
                    "simulation_id": sim_id,
                    "case_name": sim_meta.get("case_name", sim_id),
                    "description": sim_meta.get("description", ""),
                    "experiment_idea": sim_meta.get("case_data", None),
                    "user_requirement": user_req,
                    "sim_dir": sim_dir,
                    "foam_output_dir": foam_output_dir,
                })

            if experiments:
                verbose = state.get("verbose", True)
                result = self.analysis.run_full_analysis_pipeline(experiments, topic, verbose=verbose)
                analysis_text = result["analysis_text"]
                viz_bundle = result["visualizations"]
                viz_bundle = self._vision_filter_viz_bundle(
                    viz_bundle,
                    experiments,
                    verbose=bool(state.get("verbose", True)),
                )
                data_proc = self.analysis.run_cross_experiment_data_processing(
                    topic=topic,
                    experiments=experiments,
                    out_dir=out_dir,
                    verbose=True,
                )
                data_proc_report_text = ""
                data_proc_report_path = str(data_proc.get("report_path", "") or "")
                if data_proc_report_path:
                    rp = Path(data_proc_report_path)
                    if rp.is_file():
                        try:
                            data_proc_report_text = rp.read_text(encoding="utf-8")
                        except Exception:
                            data_proc_report_text = ""
                data_proc_interp_text = str(data_proc.get("interpretation_text", "") or "")
                if not data_proc_interp_text:
                    ip = Path(str(data_proc.get("interpretation_path", "") or ""))
                    if ip.is_file():
                        try:
                            data_proc_interp_text = ip.read_text(encoding="utf-8")
                        except Exception:
                            data_proc_interp_text = ""
                # Include cross-experiment PNG outputs so writer can cite/add them.
                data_proc_images = [
                    str(p) for p in (data_proc.get("images", []) or []) if isinstance(p, str) and p.strip()
                ]
                if data_proc_images:
                    viz_bundle.append(
                        {
                            "simulation_id": "cross_experiment",
                            "case_name": "cross_experiment_analysis",
                            "visualization": {"ok": True, "images": data_proc_images},
                        }
                    )
                viz_by_id = {v["simulation_id"]: v.get("visualization") for v in viz_bundle}
                for entry in pipeline_log.get("simulations", []) or []:
                    sim_meta = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                    sid = sim_meta.get("simulation_id")
                    if sid and sid in viz_by_id:
                        entry["analysis_visualization"] = viz_by_id[sid]
            else:
                analysis_text = ""
                viz_bundle = []
                data_proc = {
                    "ok": True,
                    "needs_processing": False,
                    "report_path": None,
                    "images": [],
                    "attempts": 0,
                    "error": "",
                }
                data_proc_report_text = ""

            analysis_path = out_dir / "analysis_report.md"
            self.analysis.save_analysis(analysis_path, analysis_text, topic=topic)
            pipeline_log["analysis"] = str(analysis_path)
            pipeline_log["cross_experiment_data_processing"] = data_proc
            log_stage("END analysis")

            log_stage("START writer")
            exp_ideas_for_writer: List[Dict[str, Any]] = []
            for ex in experiments:
                exp_ideas_for_writer.append(
                    {
                        "simulation_id": ex.get("simulation_id"),
                        "case_name": ex.get("case_name"),
                        "experiment_idea": ex.get("experiment_idea"),
                    }
                )
            paper_context = (
                f"TOPIC:\n{topic}\n\n"
                f"IDEA:\n{json.dumps(state.get('idea', {}), indent=2)}\n\n"
                f"PER-EXPERIMENT IDEAS/CONTEXT:\n{json.dumps(exp_ideas_for_writer, indent=2, ensure_ascii=False)}\n\n"
                f"ANALYSIS:\n{analysis_text}\n\n"
                + (
                    f"CROSS-EXPERIMENT DATA PROCESSING REPORT:\n{data_proc_report_text}\n\n"
                    if data_proc_report_text
                    else ""
                )
                + (
                    f"CROSS-EXPERIMENT INTERPRETATION (LLM):\n{data_proc_interp_text}\n\n"
                    if data_proc_interp_text
                    else ""
                )
                +
                "Write paper draft using available artifacts."
            )
            work_dir = out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir
            paper_text, pdf_path, review_info = self.writer.write_paper_with_literature_and_review(
                topic=topic,
                section_context=paper_context,
                out_dir=out_dir,
                work_dir=work_dir,
                ideation_literature_bundle=(state.get("ideation_result", {}) or {}).get(
                    "literature_used", []
                ),
                visualization_bundle=viz_bundle,
                max_review_tries=10,
                verbose=bool(state.get("verbose", True)),
            )
            paper_path = out_dir / "paper_draft.tex"
            paper_path.write_text(paper_text, encoding="utf-8")
            pipeline_log["paper"] = str(paper_path)
            pipeline_log["paper_pdf"] = str(pdf_path) if pdf_path else None
            pipeline_log["review_tries"] = review_info.get("tries", 0)

            self._write_json(
                out_dir / "analysis_output.json",
                {
                    "topic": topic,
                    "analysis_report": str(analysis_path),
                    "paper_draft": str(paper_path),
                    "paper_pdf": str(pdf_path) if pdf_path else None,
                    "analysis_text": analysis_text,
                    "visualizations": viz_bundle,
                    "cross_experiment_data_processing": data_proc,
                },
            )
            log_stage("END writer")

            return {"pipeline_log": pipeline_log}

        def save_pipeline_log(state: WorkflowState) -> WorkflowState:
            out_dir = state["out_dir"]
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})

            allow_final = bool(state.get("execute", False)) or bool(
                state.get("allow_non_executed_artifacts", False)
            )
            if not allow_final:
                pipeline_log["analysis"] = None
                pipeline_log["paper"] = None
                pipeline_log["artifact_gate"] = {
                    "execute": bool(state.get("execute", False)),
                    "allow_non_executed_artifacts": bool(
                        state.get("allow_non_executed_artifacts", False)
                    ),
                    "skipped": ["analysis", "paper"],
                }
                self._write_json(
                    out_dir / "analysis_output.json",
                    {
                        "topic": str(state.get("topic", "")),
                        "analysis_report": None,
                        "paper_draft": None,
                        "status": "skipped",
                        "reason": "analysis_and_writer stage skipped by artifact gate",
                        "artifact_gate": pipeline_log.get("artifact_gate", {}),
                    },
                )

            (out_dir / "pipeline_log.json").write_text(
                json.dumps(pipeline_log, indent=2), encoding="utf-8"
            )
            return {"pipeline_log": pipeline_log}

        g = StateGraph(WorkflowState)
        g.add_node("ideate", ideate)
        g.add_node("expand_and_init", expand_and_init)
        g.add_node("prepare_next_sim", prepare_next_sim)
        g.add_node("generate_requirement", generate_requirement)
        g.add_node("precheck", precheck)
        g.add_node("foam_run", foam_run)
        g.add_node("interpret", interpret)
        g.add_node("revise_requirement", revise_requirement)
        g.add_node("append_simulation_log", append_simulation_log)
        g.add_node("final_artifacts_gate", final_artifacts_gate)
        g.add_node("interpret_batch", interpret_batch)
        g.add_node("rerun_round_start", rerun_round_start)
        g.add_node("rerun_run_one", rerun_run_one)
        g.add_node("rerun_finalize_round", rerun_finalize_round)
        g.add_node("analysis_and_writer", analysis_and_writer)
        g.add_node("save_pipeline_log", save_pipeline_log)

        g.add_edge(START, "ideate")
        g.add_edge("ideate", "expand_and_init")
        g.add_edge("expand_and_init", "prepare_next_sim")
        g.add_conditional_edges("prepare_next_sim", route_after_prepare)

        g.add_edge("generate_requirement", "precheck")
        g.add_conditional_edges("precheck", route_after_precheck)
        g.add_edge("foam_run", "append_simulation_log")
        g.add_edge("append_simulation_log", "prepare_next_sim")

        g.add_conditional_edges("final_artifacts_gate", route_after_final_gate)
        g.add_conditional_edges("interpret_batch", route_after_interpret_batch)
        g.add_conditional_edges("rerun_round_start", route_after_rerun_round_start)
        g.add_conditional_edges("rerun_run_one", route_after_rerun_run_one)
        g.add_conditional_edges("rerun_finalize_round", route_after_rerun_finalize_round)
        g.add_edge("analysis_and_writer", "save_pipeline_log")
        g.add_edge("save_pipeline_log", END)

        return g.compile()

    def _get_app(self):
        if self._app is None:
            self._app = self._build_langgraph_app()
        return self._app

    def run_topic(
        self,
        topic: str,
        out_dir: Path,
        execute: bool = False,
        allow_non_executed_artifacts: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        initial_state: WorkflowState = {
            "topic": topic,
            "out_dir": out_dir,
            "execute": execute,
            "allow_non_executed_artifacts": allow_non_executed_artifacts,
            "verbose": verbose,
        }
        final_state = cast(WorkflowState, self._get_app().invoke(initial_state))
        return cast(Dict[str, Any], final_state.get("pipeline_log", {}) or {})

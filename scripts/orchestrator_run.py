#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from timeline_logger import append_timeline_event

# Mesh-gate stability compares physics QoIs only (exclude mesh-size / volume |U| diagnostics).
_MESH_GATE_STABILITY_SKIP_KEYS = frozenset(
    {
        "mesh_n_cells",
        "mesh_n_points",
        "pyvista_time_used",
        "Umag_mean",
        "Umag_max",
    }
)

# Starter loose files with these extensions are read into LLM context (equations, code-mod notes).
_STARTER_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst", ".tex"})
_STARTER_TEXT_MAX_PER_FILE = 48_000
_STARTER_TEXT_MAX_TOTAL = 160_000
_STARTER_TEXT_MAX_FILES = 16


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


def _run_streaming(cmd: List[str], cwd: Path, env: Dict[str, str], stage: str) -> tuple[int, str, str]:
    """
    Run command with live stdout/stderr streaming while capturing full logs.
    """
    env_run = dict(env)
    env_run.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env_run,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    q: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _reader(pipe: Any, stream_name: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                q.put((stream_name, line))
        finally:
            q.put((stream_name, None))
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()
    done = {"stdout": False, "stderr": False}

    try:
        while not (done["stdout"] and done["stderr"]):
            stream_name, payload = q.get()
            if payload is None:
                done[stream_name] = True
                continue
            if stream_name == "stdout":
                stdout_chunks.append(payload)
                print(payload, end="")
            else:
                stderr_chunks.append(payload)
                print(payload, end="", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n[ORCH] Interrupt received. Terminating stage process tree: {stage}", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        raise

    rc = proc.wait()
    t_out.join(timeout=0.1)
    t_err.join(timeout=0.1)
    return rc, "".join(stdout_chunks), "".join(stderr_chunks)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _openfoam_application_from_control_dict(case_dir: Path) -> str:
    p = case_dir / "system" / "controlDict"
    if not p.is_file():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    m = re.search(r"^\s*application\s+([^;\s]+)\s*;", txt, re.M)
    return m.group(1).strip() if m else ""


def _infer_model_intent_from_requirement(text: str) -> str:
    t = (text or "").lower()
    if '"viscosity_model": "newtonian"' in t or "baseline newtonian" in t or "regular model" in t:
        return "regular"
    if "custom viscosity model" in t or "provided equation" in t or "newly introduced model" in t:
        return "custom"
    return "auto"


def _bootstrap_paths(repo_root: Path) -> None:
    foam_src = repo_root / "Foam-Agent" / "src"
    lang_src = repo_root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _discover_starter_assets(starter_dir: Path) -> Dict[str, List[Path]]:
    assets: Dict[str, List[Path]] = {
        "pdfs": [],
        "images": [],
        "cases": [],
        "text_files": [],
        "other_files": [],
    }
    if not starter_dir.exists() or not starter_dir.is_dir():
        return assets
    seen_cases = set()
    for p in starter_dir.rglob("controlDict"):
        case_dir = p.parent.parent.resolve()
        if (case_dir / "0").exists() and (case_dir / "constant").exists() and str(case_dir) not in seen_cases:
            assets["cases"].append(case_dir)
            seen_cases.add(str(case_dir))
    case_roots = [c.resolve() for c in assets["cases"]]
    for p in starter_dir.rglob("*"):
        if not p.is_file():
            continue
        rp = p.resolve()
        under_case = False
        for c in case_roots:
            try:
                rp.relative_to(c)
                under_case = True
                break
            except ValueError:
                continue
        if under_case:
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            assets["pdfs"].append(rp)
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            assets["images"].append(rp)
        elif ext in _STARTER_TEXT_EXTENSIONS:
            assets["text_files"].append(rp)
        else:
            assets["other_files"].append(rp)
    return assets


def _build_starter_context_text(repo_root: Path, starter_dir: Path) -> Dict[str, Any]:
    assets = _discover_starter_assets(starter_dir)
    case_summaries: List[Dict[str, Any]] = []
    for c in assets.get("cases", [])[:3]:
        summary: Dict[str, Any] = {"case_dir": str(c)}
        for rel in ("system/controlDict", "system/blockMeshDict", "constant/physicalProperties", "0/U", "0/p"):
            p = c / rel
            if p.exists():
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    summary[rel] = txt[:4000]
                except Exception:
                    summary[rel] = ""
        case_summaries.append(summary)

    pdf_samples: List[Dict[str, Any]] = []
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        PdfReader = None  # type: ignore
    for p in assets.get("pdfs", [])[:3]:
        chunk = ""
        if PdfReader is not None:
            try:
                rd = PdfReader(str(p))
                chunk = "\n".join((pg.extract_text() or "") for pg in rd.pages[:8])[:6000]
            except Exception:
                chunk = ""
        pdf_samples.append({"path": str(p), "text": chunk})

    image_summaries: List[Dict[str, Any]] = [{"path": str(p)} for p in assets.get("images", [])[:6]]

    starter_text_documents: List[Dict[str, Any]] = []
    text_budget = 0
    for p in sorted(assets.get("text_files", []), key=lambda x: str(x))[:_STARTER_TEXT_MAX_FILES]:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunk = raw[:_STARTER_TEXT_MAX_PER_FILE]
        need = len(chunk)
        if text_budget + need > _STARTER_TEXT_MAX_TOTAL:
            remain = _STARTER_TEXT_MAX_TOTAL - text_budget
            if remain < 500:
                break
            chunk = chunk[:remain]
            need = len(chunk)
        starter_text_documents.append(
            {
                "path": str(p),
                "text": chunk,
                "truncated": len(raw) > len(chunk),
                "bytes_omitted": max(0, len(raw) - len(chunk)),
            }
        )
        text_budget += need

    return {
        "starter_dir": str(starter_dir),
        "cases": [str(p) for p in assets.get("cases", [])],
        "pdfs": [str(p) for p in assets.get("pdfs", [])],
        "images": [str(p) for p in assets.get("images", [])],
        "text_files": [str(p) for p in assets.get("text_files", [])],
        "other_files": [str(p) for p in assets.get("other_files", [])[:30]],
        "starter_text_documents": starter_text_documents,
        "case_summaries": case_summaries,
        "pdf_samples": pdf_samples,
        "image_summaries": image_summaries,
    }


def _llm_starter_study_brief(
    repo_root: Path,
    topic: str,
    starter_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Multimodal starter analysis before literature:
    - reads topic + starter case snippets
    - reads extracted PDF text
    - reads PNG/JPG images directly via image_url content blocks
    Returns a concise study brief used to drive literature search.
    """
    _bootstrap_paths(repo_root)
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    from cfd_langgraph.utils import strip_json_fences  # type: ignore

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0)

    pdf_samples = starter_context.get("pdf_samples", []) if isinstance(starter_context, dict) else []
    pdf_text_chunks: List[str] = []
    if isinstance(pdf_samples, list):
        for s in pdf_samples[:3]:
            if not isinstance(s, dict):
                continue
            p = str(s.get("path", ""))
            t = str(s.get("text", ""))
            if t.strip():
                pdf_text_chunks.append(f"PDF: {p}\n{t[:6000]}")
    pdf_text_blob = "\n\n".join(pdf_text_chunks)[:18000]

    human_content: List[Any] = [
        {
            "type": "text",
            "text": (
                "Analyze this CFD starter kit and topic. "
                "Infer what study is intended and what exact operating setup is implied. "
                "Return strict JSON with keys: "
                "study_brief_lines (10-20 lines array), "
                "literature_query_topic (string — SHORT keyword query for Semantic Scholar API, "
                "max 5-6 content words, no filler words like 'in/of/for/and/the/using'. "
                "Example: 'non-Newtonian power-law OpenFOAM channel flow'), "
                "key_assumptions (array), "
                "known_setup (object)."
            ),
        },
        {"type": "text", "text": f"Topic:\n{topic}"},
        {"type": "text", "text": f"Starter context (cases/files/snippets):\n{json.dumps(starter_context, indent=2)[:30000]}"},
    ]
    if pdf_text_blob.strip():
        human_content.append({"type": "text", "text": f"Extracted PDF text:\n{pdf_text_blob}"})

    images = starter_context.get("images", []) if isinstance(starter_context, dict) else []
    if isinstance(images, list):
        for p in images[:4]:
            img_path = Path(str(p))
            if not img_path.exists():
                continue
            ext = img_path.suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                b = img_path.read_bytes()
                b64 = base64.b64encode(b).decode("utf-8")
                mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
                human_content.append({"type": "text", "text": f"Starter image: {str(img_path)}"})
                human_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception:
                continue

    sys_prompt = (
        "You are a CFD scientist. Build an accurate study intent summary from starter materials "
        "before literature search. Be concrete, avoid vague language, and do not invent facts."
    )
    raw = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_content)])
    content = getattr(raw, "content", "") if raw else ""
    cleaned = strip_json_fences(content if isinstance(content, str) else str(content))
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s : e + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Starter-study brief response is not valid JSON")
    return payload


def _llm_infer_understanding_and_plan(
    repo_root: Path,
    topic: str,
    starter_context: Dict[str, Any],
    literature_records: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    _bootstrap_paths(repo_root)
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    from cfd_langgraph.utils import strip_json_fences  # type: ignore

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    available_modules = [
        {"id": "literature", "required": True, "description": "Semantic Scholar literature collection via scripts/lit.py"},
        {"id": "benchmark_plan", "required": False, "description": "Benchmark/validation data planning via scripts/benchmark_data_prepare.py"},
        {"id": "reference_data_ingest", "required": False, "description": "Generic starter reference-data ingest + manifest via scripts/reference_data_ingest.py"},
        {"id": "hypothesis", "required": True, "description": "Hypothesis generation via scripts/hypothesis.py"},
        {"id": "requirements", "required": True, "description": "Experiment requirement generation via scripts/requirements.py"},
        {"id": "baseline_synthesis", "required": False, "description": "Synthesize canonical_base_case via Foam-Agent (RAG + scaffolding) when --no-starter or starter lacks a baseline; mesh-gate's seed."},
        {"id": "code_mod", "required": mode == "code_mod", "description": "OpenFOAM code-mod branch via scripts/code_mod_prepare.py + builder/apply_compile"},
        {"id": "mesh_gate", "required": True, "description": "Mesh sensitivity gate via mesh baseline/refined/analyze stages"},
        {"id": "experiments", "required": True, "description": "Case execution loop via scripts/foam_run.py + interpret loop"},
        {
            "id": "analysis",
            "required": True,
            "description": "Cross-case metrics via scripts/analyze.py; viz.py full only if --legacy-paper-pipeline (default uses paper_unified for figures).",
        },
        {
            "id": "paper_review",
            "required": True,
            "description": "Default: scripts/paper_unified.py (plan + PyVista paper figs + write/review loop). Legacy: paper_utils.py + reviewer.py via --legacy-paper-pipeline.",
        },
        {
            "id": "reference_verify",
            "required": False,
            "description": "Optional post-paper S2 reference verification (--reference-verify).",
        },
    ]
    lit_brief = []
    for r in (literature_records or [])[:20]:
        if not isinstance(r, dict):
            continue
        lit_brief.append(
            {
                "title": r.get("title", ""),
                "year": r.get("year", None),
                "doi": r.get("doi", ""),
                "citationCount": r.get("citationCount", 0),
            }
        )

    sys_prompt = (
        "You are a senior CFD scientist orchestrating an end-to-end study. "
        "First explain what the study is trying to achieve using topic + starter artifacts. "
        "Then produce an executable plan using only available modules."
    )
    user_prompt = (
        "Return strict JSON with keys:\n"
        "{\n"
        '  "study_understanding": [10 to 20 concise lines as array of strings],\n'
        '  "research_intent": "one paragraph",\n'
        '  "module_plan": [{"module":"...", "why":"...", "inputs":[...], "outputs":[...]}],\n'
        '  "notes": ["..."]\n'
        "}\n\n"
        f"Topic:\n{topic}\n\n"
        f"Mode:\n{mode}\n\n"
        f"Starter context:\n{json.dumps(starter_context, indent=2)[:35000]}\n\n"
        f"Literature brief (top items):\n{json.dumps(lit_brief, indent=2)[:18000]}\n\n"
        f"Available modules (must only use these ids):\n{json.dumps(available_modules, indent=2)}\n"
    )
    raw = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    content = getattr(raw, "content", "") if raw else ""
    cleaned = strip_json_fences(content if isinstance(content, str) else str(content))
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s : e + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM planning response is not a JSON object")
    payload["available_modules"] = available_modules
    return payload


def _validate_and_revise_module_plan(
    repo_root: Path,
    mode: str,
    plan_payload: Dict[str, Any],
) -> Dict[str, Any]:
    available = plan_payload.get("available_modules", [])
    allowed_ids = {str(m.get("id")) for m in available if isinstance(m, dict) and m.get("id")}
    required_ids = {str(m.get("id")) for m in available if isinstance(m, dict) and m.get("required") is True and m.get("id")}
    module_plan = plan_payload.get("module_plan", [])
    if not isinstance(module_plan, list):
        module_plan = []
    proposed_ids = []
    invalid_ids = []
    for item in module_plan:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("module", "")).strip()
        if not mid:
            continue
        proposed_ids.append(mid)
        if mid not in allowed_ids:
            invalid_ids.append(mid)

    missing_required = [m for m in required_ids if m not in proposed_ids]
    if not invalid_ids and not missing_required:
        return plan_payload

    _bootstrap_paths(repo_root)
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    from cfd_langgraph.utils import strip_json_fences  # type: ignore

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    sys_prompt = "Revise CFD workflow plan JSON to include only supported modules."
    user_prompt = (
        "Fix this JSON plan so that module_plan only uses allowed module ids and includes all required ids exactly once.\n"
        f"Allowed ids: {sorted(allowed_ids)}\n"
        f"Required ids: {sorted(required_ids)}\n"
        f"Invalid ids found: {sorted(set(invalid_ids))}\n"
        f"Missing required ids: {sorted(set(missing_required))}\n\n"
        f"Original JSON:\n{json.dumps(plan_payload, indent=2)[:30000]}\n\n"
        "Return full corrected JSON with same top-level keys."
    )
    raw = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    content = getattr(raw, "content", "") if raw else ""
    cleaned = strip_json_fences(content if isinstance(content, str) else str(content))
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s : e + 1]
    revised = json.loads(cleaned)
    if not isinstance(revised, dict):
        raise ValueError("Revised module plan is not valid JSON")
    revised["available_modules"] = available
    revised["plan_revision"] = {
        "invalid_ids_detected": sorted(set(invalid_ids)),
        "missing_required_detected": sorted(set(missing_required)),
        "revised": True,
    }
    return revised


def _module_plan_to_stage_order(mode: str, module_plan: Any) -> List[str]:
    default = [
        "literature",
        "benchmark_plan",
        "reference_data_ingest",
        "baseline_setup",
        "metric_setup",
        "hypothesis",
        "requirements",
        "baseline_synthesis",
        "code_mod",
        "mesh_gate",
        "experiments",
        "analysis",
        "paper_review",
        "reference_verify",
    ]
    if not isinstance(module_plan, list):
        return default
    stage_order: List[str] = []
    for item in module_plan:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("module", "")).strip()
        if mid and mid not in stage_order:
            stage_order.append(mid)
    # Keep only known stages and preserve LLM-proposed inclusion.
    precedence = [
        "literature",
        "benchmark_plan",
        "reference_data_ingest",
        "baseline_setup",
        "metric_setup",
        "hypothesis",
        "requirements",
        "baseline_synthesis",
        "code_mod",
        "mesh_gate",
        "experiments",
        "analysis",
        "paper_review",
        "reference_verify",
    ]
    stage_order = [s for s in precedence if s in stage_order]
    if not stage_order:
        return default
    # Minimal safety fallback for code_mod if LLM omitted code_mod by mistake.
    if mode == "code_mod" and "code_mod" not in stage_order:
        stage_order.append("code_mod")
        stage_order = [s for s in precedence if s in stage_order]
    # baseline_setup is a NEW stage the orchestrator's planner LLM doesn't
    # know about, so it gets filtered out of any LLM-proposed module_plan.
    # Always insert it (idempotent) — it's a no-op if the topic doesn't
    # request a baseline comparison (the stage's own LLM classifier decides).
    if "baseline_setup" not in stage_order:
        stage_order.append("baseline_setup")
        stage_order = [s for s in precedence if s in stage_order]
    # metric_setup is always inserted (idempotent) — runs only when both
    # baseline case + reference data are present, otherwise the OED loop's
    # legacy fallback path takes over.
    if "metric_setup" not in stage_order:
        stage_order.append("metric_setup")
        stage_order = [s for s in precedence if s in stage_order]
    return stage_order


def _stage_display_name(stage: str) -> str:
    if stage == "hypothesis":
        return "experiment_designer"
    return stage


def _discover_starter_case_dirs(repo_root: Path, starter_dir: Optional[Path] = None) -> List[Path]:
    starter = (starter_dir or (repo_root / "starter")).resolve()
    if not starter.is_dir():
        return []
    out: List[Path] = []
    seen = set()
    for p in starter.rglob("controlDict"):
        if p.name != "controlDict":
            continue
        case_dir = p.parent.parent.resolve()
        if (case_dir / "0").exists() and (case_dir / "constant").exists() and str(case_dir) not in seen:
            out.append(case_dir)
            seen.add(str(case_dir))
    return out


def _extract_first_float(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _infer_starter_case_context(repo_root: Path, starter_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Read the starter folder and ask an LLM to extract case configuration
    (Re, nu, Ub, dimension, formula, etc.) so we never ask the user for
    information already present in the provided files.
    PolyMesh and postProcessing directories are skipped to avoid binary bloat.
    """
    _SKIP_DIRS = {"polyMesh", "postProcessing", "dynamicCode", "processor0",
                  "processor1", "processor2", "processor3"}
    _TEXT_SUFFIXES = {
        ".txt", ".md", ".rst", ".C", ".H", ".py",
        ".cfg", ".yaml", ".yml", ".json",
        # OpenFOAM dictionary files (no extension or common names)
        "", ".foam",
    }
    _FOAM_DICT_NAMES = {
        "U", "p", "nuTilda", "nut", "k", "omega", "epsilon",
        "controlDict", "blockMeshDict", "fvSchemes", "fvSolution",
        "transportProperties", "physicalProperties", "momentumTransport",
        "turbulenceProperties", "decomposeParDict", "fvConstraints",
    }
    _MAX_FILE_CHARS = 3000
    _MAX_TOTAL_CHARS = 40000

    _resolved_starter = starter_dir or (repo_root / "starter")
    case_dirs = _discover_starter_case_dirs(repo_root, _resolved_starter)
    starter_dir = _resolved_starter if _resolved_starter.exists() else repo_root

    base_context: Dict[str, Any] = {
        "starter_found": bool(case_dirs),
        "starter_case_dir": str(case_dirs[0]) if case_dirs else "",
        "base_case_available": bool(case_dirs),
    }

    # Collect readable text from the starter folder.
    file_blobs: list[str] = []
    total_chars = 0
    scan_root = case_dirs[0].parent if case_dirs else starter_dir
    # Include both the case dir and any sibling text/script files.
    search_roots = list({scan_root, starter_dir})

    def _should_skip(p: Path) -> bool:
        return any(part in _SKIP_DIRS for part in p.parts)

    for sroot in search_roots:
        if not sroot.exists():
            continue
        for fp in sorted(sroot.rglob("*")):
            if not fp.is_file() or _should_skip(fp):
                continue
            if total_chars >= _MAX_TOTAL_CHARS:
                break
            sfx = fp.suffix.lower()
            is_foam_dict = fp.name in _FOAM_DICT_NAMES
            if sfx not in _TEXT_SUFFIXES and not is_foam_dict:
                continue
            try:
                raw = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunk = raw[:_MAX_FILE_CHARS]
            rel = fp.relative_to(sroot) if fp.is_relative_to(sroot) else fp
            blob = f"\n--- FILE: {rel} ---\n{chunk}"
            file_blobs.append(blob)
            total_chars += len(blob)

    if not file_blobs:
        return base_context

    files_text = "".join(file_blobs)[:_MAX_TOTAL_CHARS]

    system_prompt = (
        "You are a CFD case analyst. Given the contents of an OpenFOAM starter case "
        "and associated files, extract the key configuration parameters. "
        "Return ONLY valid JSON with these keys (use null if not found):\n"
        '{"Re": <number|null>, "nu": <number|null>, "Ub": <number|null>, '
        '"dimension": "2D"|"3D"|null, "notes": "<brief string>"}\n'
        "Do not include markdown fences or any other text."
    )
    user_prompt = (
        "Extract Re, nu, Ub and dimension from these starter case files.\n\n"
        + files_text
    )

    try:
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        txt = str(getattr(raw, "content", raw)).strip()
        # Strip optional markdown fences.
        if txt.startswith("```"):
            txt = txt.split("```")[1].lstrip("json").strip()
            txt = txt.rsplit("```", 1)[0].strip()
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            if parsed.get("Re") is not None:
                base_context["reynolds_inferred"] = float(parsed["Re"])
                base_context["reynolds_basis"] = {
                    "nu": parsed.get("nu"), "Ub": parsed.get("Ub"),
                    "source": "llm_starter_inference",
                }
            if parsed.get("dimension"):
                base_context["dimension"] = parsed["dimension"]
            if parsed.get("nu") is not None:
                base_context["nu_inferred"] = float(parsed["nu"])
            if parsed.get("Ub") is not None:
                base_context["Ub_inferred"] = float(parsed["Ub"])
            print(f"[ORCH] Starter context inferred by LLM: {parsed}")
    except Exception as exc:
        print(f"[ORCH] warning: LLM starter context inference failed: {exc}")

    return base_context


def _ensure_starter_seed_case(
    run_dir: Path,
    repo_root: Path,
    state_path: Path,
    timeline_path: Path,
    starter_dir: Optional[Path] = None,
) -> None:
    """
    Copy a starter/ OpenFOAM case (full tree including polyMesh) into the run directory so
    experiments can mutate a stable copy. Dictionary-level context for LLMs is handled separately
    in code_mod_prepare (0/system/constant without polyMesh text).
    """
    state = _read_json(state_path, {})
    if isinstance(state, dict) and state.get("starter_seed_case_dir"):
        p = Path(str(state["starter_seed_case_dir"]))
        if p.exists():
            return
    canon = run_dir / "canonical_base_case"
    if canon.exists() and (canon / "system" / "controlDict").exists():
        _update_state(state_path, {"starter_seed_case_dir": str(canon.resolve())})
        append_timeline_event(
            timeline_path,
            {
                "stage": "starter_seed",
                "event": "using_canonical_base_case",
                "path": str(canon.resolve()),
            },
        )
        return
    cases = _discover_starter_case_dirs(repo_root, starter_dir)
    if not cases:
        return
    dst = run_dir / "starter_case_seed"
    if dst.exists():
        _update_state(state_path, {"starter_seed_case_dir": str(dst.resolve())})
        return
    try:
        shutil.copytree(cases[0], dst, symlinks=False, ignore_dangling_symlinks=True)
    except Exception as exc:
        append_timeline_event(
            timeline_path,
            {"stage": "starter_seed", "event": "copy_failed", "error": str(exc)},
        )
        return
    _update_state(state_path, {"starter_seed_case_dir": str(dst.resolve())})
    append_timeline_event(
        timeline_path,
        {
            "stage": "starter_seed",
            "event": "copied",
            "source": str(cases[0]),
            "destination": str(dst.resolve()),
        },
    )


def _classify_topic(topic: str) -> str:
    t = topic.lower()
    code_mod_keys = [
        "viscosity",
        "turbulence model",
        "source term",
        "fvoption",
        "custom model",
        "modify openfoam",
    ]
    mesh_keys = ["mesh independence", "mesh sensitivity", "gci", "richardson", "grid convergence"]
    if any(k in t for k in code_mod_keys):
        return "code_mod"
    if any(k in t for k in mesh_keys):
        return "mesh_focus"
    return "standard"


def _llm_classify_topic_mode(
    *,
    repo_root: Path,
    topic: str,
    starter_context: Dict[str, Any],
) -> str:
    """
    LLM router for primary mode classification.
    Returns one of: code_mod | mesh_focus | pure_sweep | standard.
    Falls back to heuristic classifier on any error.

    Modes:
      code_mod   — topic asks to implement/modify source code (turbulence/viscosity/
                   source term/fvOption/custom model). Includes cases where the user
                   provides equations to implement, even if no OED budget is set.
      mesh_focus — primary intent is mesh independence / GCI / grid convergence.
      pure_sweep — parameter-only sweep, no model modification, no improvement
                   candidates. Use for studies like "vary Re, find correlation".
      standard   — regular CFD study without required source-code modification
                   (catch-all for anything that doesn't fit the others).
    """
    try:
        _bootstrap_paths(repo_root)
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        payload = {
            "topic": topic,
            "starter_context": starter_context if isinstance(starter_context, dict) else {},
            "allowed_modes": ["code_mod", "mesh_focus", "pure_sweep", "standard"],
            "definitions": {
                "code_mod": "Request asks to implement or modify turbulence/viscosity/source-term/fvOption/custom-model CODE. Includes cases where the user provides an equation or a new model formulation to implement and test.",
                "mesh_focus": "Primary intent is mesh independence / sensitivity / GCI / grid-convergence study.",
                "pure_sweep": "Pure parameter sweep / scan / correlation study. The user wants to vary a parameter (Re, viscosity, geometry size, ...), record outcomes, and possibly fit a correlation. NO model modification, NO improvement candidates, NO baseline-vs-improved framing.",
                "standard": "Regular CFD study that doesn't fit the others (e.g. validation against a benchmark, single-case investigation).",
            },
        }
        sys_msg = (
            "Classify CFD request mode.\n"
            "Return STRICT JSON only: {\"mode\": \"code_mod|mesh_focus|pure_sweep|standard\", \"reason\": \"...\"}.\n"
            "If the topic clearly asks to implement a new model or modify code, choose code_mod (regardless of whether a sweep is also implied).\n"
            "If the topic only sweeps a parameter and asks for a correlation or trend, choose pure_sweep.\n"
            "Use starter context to disambiguate when topic wording is broad."
        )
        raw = llm.invoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)[:42000]),
            ]
        )
        txt = str(getattr(raw, "content", raw))
        clean = strip_json_fences(txt)
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1 and e > s:
            clean = clean[s : e + 1]
        obj = json.loads(clean)
        if isinstance(obj, dict):
            mode = str(obj.get("mode", "")).strip()
            if mode in {"code_mod", "mesh_focus", "pure_sweep", "standard"}:
                return mode
    except Exception:
        pass
    return _classify_topic(topic)


def _is_open_discovery_request(topic: str) -> bool:
    t = (topic or "").lower()
    keys = [
        "open ended",
        "open-ended",
        "beat baseline",
        "beats baseline",
        "beat literature",
        "discover model",
        "find best model",
        "improve over baseline",
        "optimize model",
        "search over",
    ]
    return any(k in t for k in keys)


def _llm_decide_open_discovery(
    *,
    repo_root: Path,
    topic: str,
    starter_context: Dict[str, Any],
) -> bool:
    """
    LLM router for open-ended discovery intent.
    Uses topic + starter context; falls back to heuristic on any error.
    """
    try:
        _bootstrap_paths(repo_root)
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        payload = {
            "topic": topic,
            "starter_context": starter_context if isinstance(starter_context, dict) else {},
        }
        sys_msg = (
            "You are a routing classifier for a CFD orchestrator.\n"
            "Decide whether the request requires OPEN-ENDED DISCOVERY mode.\n"
            "OPEN-ENDED DISCOVERY means: autonomous model/design search to beat baseline/literature,\n"
            "iterative proposal-ranking and improvement loops, not just implementing a specific provided change.\n"
            "Return STRICT JSON only: {\"open_discovery\": true|false, \"reason\": \"...\"}."
        )
        raw = llm.invoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)[:40000]),
            ]
        )
        txt = str(getattr(raw, "content", raw))
        clean = strip_json_fences(txt)
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1 and e > s:
            clean = clean[s : e + 1]
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return bool(obj.get("open_discovery", False))
    except Exception:
        pass
    return _is_open_discovery_request(topic)


def _generate_open_discovery_hypotheses(
    *,
    repo_root: Path,
    topic: str,
    starter_context: Dict[str, Any],
    literature_records: List[Dict[str, Any]],
    run_dir: Path,
    max_experiments: int,
    improvement_threshold: float = 0.10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate open-ended discovery candidates + policy as hypothesis records.
    This is additive and only used when explicitly requested.
    """
    max_experiments = max(2, int(max_experiments))
    policy: Dict[str, Any] = {
        "mode": "open_discovery",
        "objective": "Beat baseline and comparable literature on common validation metrics.",
        "improvement_threshold": improvement_threshold,
        "budget": {"max_experiments": max_experiments, "max_compile_attempts_per_candidate": 10},
        "acceptance": {
            "must_beat_baseline": True,
            "must_beat_comparable_literature": True,
            "minimum_relative_improvement": improvement_threshold,
        },
    }

    def _fallback() -> List[Dict[str, Any]]:
        # Generic fallback when the LLM call fails: one baseline + N-1 variation placeholders.
        # No model-type or case-specific constants — FoamAgent will fill in appropriate
        # defaults from the starter case and topic description.
        n = max(1, min(max_experiments, 5))
        descs = [
            ("baseline", "Baseline run using starter case configuration as-is for reference."),
            ("variation_01", "First parameter variation targeting improved flow prediction."),
            ("variation_02", "Second parameter variation with adjusted model coefficients."),
            ("variation_03", "Third parameter variation exploring alternative correction term."),
            ("variation_04", "Fourth parameter variation for broader search coverage."),
        ]
        out: List[Dict[str, Any]] = []
        for i, (name, desc) in enumerate(descs[:n], 1):
            eid = f"exp_{i:03d}"
            out.append(
                {
                    "hypothesis_id": eid,
                    "experiment_id": eid,
                    "study_id": "open_discovery_cfd",
                    "description": desc,
                    "hypothesis_text": desc,
                    "parameter_value": {},
                    "valid": True,
                    "idea_experiment": {
                        "experiment_id": eid,
                        "name": name,
                        "topology": "2d",
                        "parameters": {},
                        "controls": {"end_time": 5000, "write_interval": 500},
                        "notes": desc,
                    },
                }
            )
        return out

    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        _bootstrap_paths(repo_root)
        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        lit_brief = [
            {
                "title": str(r.get("title", ""))[:220],
                "year": r.get("year"),
                "citations": r.get("citationCount"),
            }
            for r in (literature_records or [])[:20]
            if isinstance(r, dict)
        ]
        starter_cases = starter_context.get("cases", []) if isinstance(starter_context, dict) else []
        prompt_obj = {
            "topic": topic,
            "starter_cases": starter_cases[:6],
            "literature_brief": lit_brief,
            "max_experiments": max_experiments,
            "minimum_improvement_fraction": improvement_threshold,
        }
        sys_msg = (
            "You are designing an OPEN-ENDED CFD discovery study.\n"
            "Return STRICT JSON with keys:\n"
            "policy: {objective, primary_metrics, secondary_metrics, acceptance_rule, search_bounds, budget}\n"
            "experiments: list of up to max_experiments candidates.\n"
            "Each experiment item must contain: name, description, target_parameters(dict), rationale.\n"
            "Constraints:\n"
            "- Build from starter geometry/cases unless incompatible.\n"
            "- Infer the appropriate model class, parameter names, and search bounds from the topic and starter cases provided.\n"
            "- Include one baseline comparator candidate and multiple improvement candidates.\n"
            "- Acceptance should require >= minimum_improvement_fraction over baseline and comparable literature.\n"
            "- Search bounds should be physically plausible and local enough for iterative compile/run loops.\n"
            "- Do not return markdown; JSON only."
        )
        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=json.dumps(prompt_obj, ensure_ascii=False)[:42000])])
        txt = str(getattr(raw, "content", raw))
        clean = strip_json_fences(txt)
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1 and e > s:
            clean = clean[s : e + 1]
        obj = json.loads(clean)
        if isinstance(obj, dict):
            pol = obj.get("policy", {})
            if isinstance(pol, dict):
                policy.update(pol)
            experiments = obj.get("experiments", [])
            if isinstance(experiments, list) and experiments:
                hyp_out: List[Dict[str, Any]] = []
                for i, ex in enumerate(experiments[:max_experiments], 1):
                    if not isinstance(ex, dict):
                        continue
                    eid = f"exp_{i:03d}"
                    desc = str(ex.get("description") or ex.get("rationale") or f"Discovery candidate {i}")
                    params = ex.get("target_parameters", {})
                    if not isinstance(params, dict):
                        params = {}
                    name = str(ex.get("name") or f"candidate_{i}")
                    hyp_out.append(
                        {
                            "hypothesis_id": eid,
                            "experiment_id": eid,
                            "study_id": "open_discovery_cfd",
                            "description": desc,
                            "hypothesis_text": desc,
                            "parameter_value": params,
                            "valid": True,
                            "idea_experiment": {
                                "experiment_id": eid,
                                "name": name,
                                "topology": "2d",
                                "parameters": params,
                                "controls": {"end_time": 5000, "write_interval": 500},
                                "notes": str(ex.get("rationale") or desc),
                            },
                        }
                    )
                if hyp_out:
                    return hyp_out, policy
    except Exception:
        pass

    return _fallback(), policy


def _update_state(state_path: Path, mut: Dict[str, Any]) -> Dict[str, Any]:
    state = _read_json(state_path, {})
    if not isinstance(state, dict):
        state = {}
    state.update(mut)
    _write_json(state_path, state)
    return state


def _build_initial_plan(topic: str, mode: str, lit_records: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    paper_titles = [str(r.get("title", "")).strip() for r in lit_records[:20] if isinstance(r, dict)]
    plan: Dict[str, Any] = {
        "topic": topic,
        "mode": mode,
        "objective": "Produce publication-quality CFD study with validated setup, meaningful comparisons, and paper-ready conclusions.",
        "scientist_principles": [
            "Use literature-grounded, physically plausible assumptions.",
            "Prioritize verifiable comparisons against baselines/benchmarks.",
            "Preserve numerical stability and mesh credibility before claiming conclusions.",
            "Adapt experiment plan when evidence indicates poor setup or non-converged behavior.",
        ],
        "literature_context": {
            "paper_count": len(lit_records),
            "top_titles": paper_titles,
        },
        "actions": [],
        "status": "planned",
        "adaptive_updates": [],
        "artifacts": {
            "plan_path": str(run_dir / "plan.json"),
            "timeline_path": str(run_dir / "timeline.json"),
            "state_path": str(run_dir / "state.json"),
        },
    }
    actions = [
        {"id": "A1", "name": "literature_review", "reason": "ground study design in prior CFD work"},
        {"id": "A1b", "name": "benchmark_data_plan", "reason": "identify experimental/benchmark data for quantitative validation"},
        {"id": "A2", "name": "hypothesis_and_requirements", "reason": "convert topic to executable experiment set"},
        {"id": "A3", "name": "mesh_gate", "reason": "select mesh-insensitive setup for reliable conclusions"},
    ]
    if _is_open_discovery_request(topic):
        actions.insert(
            2,
            {
                "id": "A2x",
                "name": "open_discovery_candidate_design",
                "reason": "derive objective/search/acceptance policy and candidate variants within budget",
            },
        )
    if mode == "code_mod":
        actions.append(
            {
                "id": "A4",
                "name": "code_modification",
                "reason": "implement requested model/source changes and validate compile/load",
            }
        )
        actions.append(
            {
                "id": "A5",
                "name": "comparative_experiments",
                "reason": "compare modified model vs established baselines",
            }
        )
    else:
        actions.append({"id": "A4", "name": "experiments", "reason": "execute designed study matrix"})
    actions.extend(
        [
            {"id": "A6", "name": "interpreter_rerun_loop", "reason": "ensure each case is physically meaningful"},
            {"id": "A7", "name": "analysis", "reason": "derive conclusions and correlations across successful cases"},
            {"id": "A8", "name": "paper_write_review", "reason": "generate and review manuscript quality"},
        ]
    )
    plan["actions"] = actions
    return plan


def _record_plan_update(plan_path: Path, note: str, context: Dict[str, Any]) -> None:
    plan = _read_json(plan_path, {})
    if not isinstance(plan, dict):
        return
    updates = plan.get("adaptive_updates")
    if not isinstance(updates, list):
        updates = []
    updates.append({"note": note, "context": context})
    plan["adaptive_updates"] = updates
    _write_json(plan_path, plan)


def _needs_clarification(topic: str, mode: str) -> List[str]:
    questions: List[str] = []
    t = (topic or "").lower()
    if "re=" not in t and "reynolds" not in t:
        questions.append("What Reynolds number or operating range should be used?")
    if not any(k in t for k in ["2d", "3d"]):
        questions.append("Should this be treated as 2D or 3D?")
    if mode == "code_mod":
        if "base" not in t and "case" not in t and "tutorial" not in t:
            questions.append("Do you already have a preferred base case path, or should I choose automatically (local/tutorial/github/generated)?")
        if "equation" not in t and "model" in t:
            questions.append("Please provide the target equation/formula details (or confirm I should infer from provided PDFs/literature).")
    return questions


def _filter_clarification_questions_with_context(
    questions: List[str],
    topic: str,
    mode: str,
    starter_context: Dict[str, Any],
) -> List[str]:
    filtered: List[str] = []
    t = (topic or "").lower()
    base_case_signals = ["base case", "baseline", "starter", "provided case", "case provided", "starter folder"]
    for q in questions:
        ql = q.lower()
        if "reynolds" in ql:
            if "reynolds" in t or "re=" in t:
                continue
            if isinstance(starter_context, dict) and starter_context.get("reynolds_inferred") is not None:
                continue
        if "2d or 3d" in ql:
            if "2d" in t or "3d" in t:
                continue
            if isinstance(starter_context, dict) and starter_context.get("dimension") in {"2D", "3D"}:
                continue
        if mode == "code_mod" and "preferred base case path" in ql:
            if any(sig in t for sig in base_case_signals):
                continue
            if isinstance(starter_context, dict) and starter_context.get("base_case_available"):
                continue
        filtered.append(q)
    return filtered


def _collect_clarifications_interactive(questions: List[str]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    if not questions:
        return answers
    print("[ORCH] Clarification required before starting workflow.")
    for idx, q in enumerate(questions, 1):
        print(f"[ORCH][Q{idx}] {q}")
        try:
            ans = input("  Your answer (leave blank to skip): ").strip()
        except EOFError:
            ans = ""
        answers[q] = ans
    return answers


def _checkpoint(state_path: Path, run_dir: Path, checkpoint_name: str, extra: Dict[str, Any]) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json(state_path, {})
    if not isinstance(payload, dict):
        payload = {}
    payload["checkpoint"] = checkpoint_name
    payload["checkpoint_extra"] = extra
    _write_json(state_path, payload)
    _write_json(checkpoints_dir / f"{checkpoint_name}.json", payload)
    timeline_path = run_dir / "timeline.json"
    append_timeline_event(
        timeline_path,
        {
            "stage": "checkpoint",
            "event": "saved",
            "checkpoint": checkpoint_name,
            "extra": extra,
            "state_path": str(state_path),
        },
    )


def _should_skip_post_code_mod_hypothesis_requirements_revision(
    run_dir: Path,
    start_index: int,
    stage_order: List[str],
) -> Tuple[bool, str]:
    """
    After code-mod, hypothesis+requirements are revised once with code-mod context.
    Do not rewrite them again once experiments have started or completed, or when
    resuming at/after the experiments stage — keeps case dirs aligned with requirements.json.
    """
    cp_rev = run_dir / "checkpoints" / "hypothesis_requirements_revised.json"
    if cp_rev.is_file():
        return True, "post-code_mod revision already checkpointed"
    cp_exp = run_dir / "checkpoints" / "experiments_done.json"
    if cp_exp.is_file():
        return True, "experiments stage already completed (checkpoint)"
    if "experiments" in stage_order:
        ex_i = stage_order.index("experiments")
        if start_index >= ex_i:
            return True, f"start_index {start_index} >= experiments ({ex_i})"
    man_path = run_dir / "manifest.json"
    if man_path.is_file():
        data = _read_json(man_path, {})
        cases = data.get("cases") if isinstance(data, dict) else []
        if isinstance(cases, list):
            for c in cases:
                if not isinstance(c, dict):
                    continue
                raw = c.get("case_path") or ""
                if not raw:
                    continue
                p = Path(str(raw)).expanduser().resolve()
                if (p / "run_result.json").is_file():
                    return True, f"experiment started (found {p.name}/run_result.json)"
    return False, ""


def _run_reference_verify_post(
    *,
    repo_root: Path,
    run_dir: Path,
    paper_dir: Path,
    lit_path: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
    stage_order: List[str],
    skip_reference_verify: bool,
) -> None:
    if skip_reference_verify:
        append_timeline_event(
            timeline_path,
            {"stage": "reference_verify", "event": "skipped", "reason": "--skip-reference-verify"},
        )
        return
    if "reference_verify" not in stage_order:
        return
    if not paper_dir.is_dir() or not (paper_dir / "main.tex").is_file():
        append_timeline_event(
            timeline_path,
            {"stage": "reference_verify", "event": "skipped", "reason": "no_main_tex"},
        )
        return
    report_path = run_dir / "reference_verify_report.json"
    ix_refv = stage_order.index("reference_verify") + 1
    _append_stage_pointer(
        state_path=state_path,
        timeline_path=timeline_path,
        stage="reference_verify",
        phase="starting",
        index=ix_refv,
        total=len(stage_order),
        next_stage="finish",
        details={"paper_dir": str(paper_dir)},
    )
    _call_stage(
        [
            sys.executable,
            "scripts/reference_verify_post.py",
            "--paper-dir",
            str(paper_dir),
            "--literature",
            str(lit_path),
            "--output",
            str(report_path),
            "--apply-cleanup",
            "--recompile",
        ],
        "reference_verify",
        repo_root,
        env,
        timeline_path,
        state_path,
    )
    _checkpoint(
        state_path,
        run_dir,
        "reference_verify_done",
        {"report_path": str(report_path)},
    )
    _append_stage_pointer(
        state_path=state_path,
        timeline_path=timeline_path,
        stage="reference_verify",
        phase="completed",
        index=ix_refv,
        total=len(stage_order),
        next_stage="finish",
        details={"report_path": str(report_path)},
    )


def _call_stage(
    cmd: List[str],
    stage: str,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
) -> None:
    stage_label = _stage_display_name(stage)
    print(f"[ORCH] START {stage_label}")
    print(f"[ORCH] CMD   {' '.join(cmd)}")
    start_ts = time.time()
    append_timeline_event(
        timeline_path,
        {"stage": stage, "event": "start", "cmd": cmd, "cwd": str(repo_root)},
    )
    rc, stdout_text, stderr_text = _run_streaming(cmd, repo_root, env, stage)
    print(f"[ORCH] END   {stage_label} rc={rc}")
    append_timeline_event(
        timeline_path,
        {
            "stage": stage,
            "event": "finish",
            "returncode": rc,
            "duration_seconds": round(time.time() - start_ts, 3),
            "stdout_tail": (stdout_text or "")[-2000:],
            "stderr_tail": (stderr_text or "")[-2000:],
        },
    )
    if rc != 0:
        _update_state(
            state_path,
            {
                "status": "failed",
                "failed_stage": stage,
                "last_error": (stderr_text or stdout_text or "").strip()[-5000:],
            },
        )
        raise RuntimeError(f"Stage failed: {stage}")


def _append_stage_pointer(
    *,
    state_path: Path,
    timeline_path: Path,
    stage: str,
    phase: str,
    index: int,
    total: int,
    next_stage: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "current_stage": stage,
        "current_stage_phase": phase,
        "current_stage_index": index,
        "current_stage_total": total,
        "current_stage_progress": f"{index}/{total}",
        "next_stage": next_stage,
    }
    if isinstance(details, dict) and details:
        payload["current_stage_details"] = details
    _update_state(state_path, payload)
    rec: Dict[str, Any] = {
        "stage": "orchestrator_progress",
        "event": "stage_pointer",
        "pointer_stage": stage,
        "phase": phase,
        "index": index,
        "total": total,
        "next_stage": next_stage,
    }
    if isinstance(details, dict) and details:
        rec["details"] = details
    append_timeline_event(timeline_path, rec)
    stage_label = _stage_display_name(stage)
    next_label = _stage_display_name(next_stage) if next_stage else "-"
    print(f"[ORCH] PROGRESS {index}/{total} stage={stage_label} phase={phase} next={next_label}")


def _build_execution_plan(
    *,
    stage_order: List[str],
    start_index: int,
    mode: str,
    disable_mesh_gate: bool,
    no_starter: bool = False,
) -> List[Dict[str, Any]]:
    # Stages that consume the starter case (reference data, baseline simulation,
    # metric binding). When the topic is structurally independent of the starter
    # (e.g. fresh laminar jet vs. periodic-hill SA starter), skip these via
    # --no-starter to avoid wasting time on irrelevant artefacts.
    _STARTER_DEPENDENT = {"reference_data_ingest", "baseline_setup", "metric_setup"}
    plan: List[Dict[str, Any]] = []
    for idx, st in enumerate(stage_order, 1):
        reason = "enabled"
        will_run = idx - 1 >= start_index
        if st == "code_mod" and mode != "code_mod":
            will_run = False
            reason = "mode_not_code_mod"
        elif st == "mesh_gate" and disable_mesh_gate:
            will_run = False
            reason = "disabled_by_flag"
        elif no_starter and st in _STARTER_DEPENDENT:
            will_run = False
            reason = "no_starter_flag"
        elif idx - 1 < start_index:
            reason = "skipped_due_to_resume"
        plan.append({"index": idx, "stage": st, "will_run": will_run, "reason": reason})
    return plan


def _llm_verify_case(
    case_dir: Path,
    user_requirement: str,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
) -> Dict[str, Any]:
    """Run LLM-based verification+fix on a case via scripts/verify_params.py.
    Returns the verification result dict with keys: is_correct, reasoning, mismatches, etc."""
    req_file = case_dir / "_verify_requirement.txt"
    verify_output = case_dir / "verify_result.json"
    req_file.write_text(user_requirement, encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/verify_params.py",
        "--case", str(case_dir),
        "--requirement-file", str(req_file),
        "--output", str(verify_output),
        "--max-loops", "3",
    ]
    try:
        _call_stage(
            cmd,
            stage=f"verify_params:{case_dir.name}",
            repo_root=repo_root,
            env=env,
            timeline_path=timeline_path,
            state_path=case_dir.parent.parent / "state.json",
        )
    except RuntimeError:
        print(f"[ORCH] WARNING: verify_params for {case_dir.name} failed (timeout/crash)")
        return {"is_correct": False, "reasoning": "verification script failed", "mismatches": []}

    if verify_output.exists():
        return _read_json(verify_output, {"is_correct": False, "reasoning": "no output", "mismatches": []})
    return {"is_correct": False, "reasoning": "no output file", "mismatches": []}


def _copy_foam_dictionary_trees(src: Path, dst: Path) -> bool:
    """Copy 0/, constant/, system/ from src to dst (mirrors scripts/rerun_selector.py)."""
    copied_any = False
    for folder in ("0", "constant", "system"):
        sdir = src / folder
        if not sdir.is_dir():
            continue
        ddir = dst / folder
        if ddir.exists():
            shutil.rmtree(ddir)
        shutil.copytree(sdir, ddir, ignore=shutil.ignore_patterns("customModels"))
        copied_any = True
    return copied_any


def _compose_rerun_foam_requirement(
    base_foam_req: str,
    attempt: int,
    last_seed_meta: Dict[str, Any],
    last_interpreter_reason: str,
) -> str:
    """Single previous-attempt context block; requirement + donor files win on conflict."""
    reason = str(last_interpreter_reason).strip()
    if attempt <= 1 or not reason:
        return base_foam_req
    refresh = str(last_seed_meta.get("detail") or "Refresh status unknown.").strip()
    block = f"""---
PREVIOUS_ATTEMPT_CONTEXT (historical; current on-disk case may not match this description)
This experiment did not reach PROCEED in the immediately prior attempt. Summary from that attempt only:
{reason}

FILE_REFRESH:
{refresh}

RULES_FOR_THIS_ATTEMPT:
- The notes above describe the *previous* run, not necessarily the current 0/constant/system trees.
- Implement THIS EXPERIMENT requirement and any AUTHORITATIVE_TARGET_PARAMETERS / authoritative JSON exactly.
- Preserve the boundary-condition pattern, mesh topology, and stable numerics of the refreshed donor case unless the requirement explicitly overrides them.
- If anything above conflicts with the requirement or with the refreshed donor setup, follow the REQUIREMENT and DONOR files; ignore the conflicting observation.
---"""
    return base_foam_req + "\n\n" + block + "\n"


def _rerun_seed_from_nearest_success(
    *,
    run_dir: Path,
    case_id: str,
    case_dir: Path,
    req: Dict[str, Any],
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    attempt: int,
    trigger: str,
    baseline_case_dir: str = "",
) -> Dict[str, Any]:
    """Copy 0/, constant/, system/ from nearest manifest success, else baseline. Returns metadata for rerun prompt framing."""
    none_meta: Dict[str, Any] = {
        "seeded": False,
        "method": "none",
        "source_case_id": None,
        "source_case_path": None,
        "detail": (
            "No bulk dictionary refresh was applied: no successful case in manifest, selector failed, "
            "and baseline/mesh-gate path was missing or had nothing to copy."
        ),
    }

    def _seed_from_baseline() -> Dict[str, Any]:
        raw = (baseline_case_dir or "").strip()
        if not raw:
            return dict(none_meta)
        src = Path(raw).expanduser().resolve()
        if not (src / "system" / "controlDict").exists():
            return dict(none_meta)
        if not _copy_foam_dictionary_trees(src, case_dir):
            meta = dict(none_meta)
            meta["detail"] = f"Baseline path {src} had no 0/, constant/, or system/ directories to copy."
            return meta
        rec = {
            "source_case_id": None,
            "source_case_path": str(src),
            "distance": None,
            "files_copied": ["0", "constant", "system"],
            "seed_method": "baseline",
        }
        (case_dir / "rerun_selection.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return {
            "seeded": True,
            "method": "baseline",
            "source_case_id": None,
            "source_case_path": str(src),
            "detail": (
                "OpenFOAM dictionaries were refreshed by copying 0/, constant/, and system/ from the "
                f"mesh-gate / baseline seed case: {src}"
            ),
        }

    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path, {"cases": []})
    cases = manifest.get("cases", []) if isinstance(manifest, dict) else []
    has_success = isinstance(cases, list) and any(
        isinstance(c, dict) and c.get("status") == "success" for c in cases
    )

    if not has_success:
        out = _seed_from_baseline()
        print(f"[ORCH] RERUN SEED {case_id} ({trigger}): no manifest successes; baseline fallback -> {out['method']}")
        append_timeline_event(
            timeline_path,
            {
                "stage": f"rerun_selector:{case_id}:attempt_{attempt}",
                "event": "finish",
                "trigger": trigger,
                "seed_method": out.get("method"),
                "seeded": out.get("seeded"),
                "returncode": -1,
                "applied_to_case_dir": str(case_dir),
            },
        )
        if out.get("seeded"):
            print(f"[ORCH] RERUN SEED {case_id}: baseline seed copy applied")
        else:
            print(f"[ORCH] RERUN SEED {case_id}: skipped (no neighbor and no baseline)")
        return out

    tmp_manifest_path = run_dir / f"manifest_for_rerun_{case_id}.json"
    tmp_cases: List[Dict[str, Any]] = [c for c in cases if isinstance(c, dict)]
    if not any(Path(str(c.get("case_path", ""))).resolve() == case_dir for c in tmp_cases):
        tmp_cases.append(
            {
                "case_id": case_id,
                "case_path": str(case_dir),
                "status": "failed",
                "parameters": req.get("parameter_value", {})
                if isinstance(req.get("parameter_value", {}), dict)
                else {},
            }
        )
    _write_json(tmp_manifest_path, {"cases": tmp_cases})
    selector_cmd = [
        sys.executable,
        "scripts/rerun_selector.py",
        "--failed",
        str(case_dir),
        "--manifest",
        str(tmp_manifest_path),
        "--output",
        str(case_dir),
    ]
    print(
        f"[ORCH] RERUN SEED {case_id} ({trigger}): trying nearest successful case in manifest"
    )
    proc = _run(selector_cmd, repo_root, env)
    append_timeline_event(
        timeline_path,
        {
            "stage": f"rerun_selector:{case_id}:attempt_{attempt}",
            "event": "finish",
            "trigger": trigger,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "applied_to_case_dir": str(case_dir),
        },
    )
    if proc.returncode == 0:
        sel_path = case_dir / "rerun_selection.json"
        data = _read_json(sel_path, {}) if sel_path.exists() else {}
        sid = data.get("source_case_id")
        spath = data.get("source_case_path")
        print(f"[ORCH] RERUN SEED {case_id}: neighbor seed copy applied ({sid})")
        return {
            "seeded": True,
            "method": "neighbor",
            "source_case_id": sid,
            "source_case_path": spath,
            "detail": (
                "OpenFOAM dictionaries were refreshed by copying 0/, constant/, and system/ from the "
                f"nearest successful experiment case id={sid!r} at {spath}."
            ),
        }

    print(f"[ORCH] RERUN SEED {case_id}: selector failed; trying baseline fallback")
    out = _seed_from_baseline()
    append_timeline_event(
        timeline_path,
        {
            "stage": f"rerun_selector_baseline_fallback:{case_id}:attempt_{attempt}",
            "event": "finish",
            "trigger": trigger,
            "seed_method": out.get("method"),
            "seeded": out.get("seeded"),
            "applied_to_case_dir": str(case_dir),
        },
    )
    if out.get("seeded"):
        print(f"[ORCH] RERUN SEED {case_id}: baseline seed copy applied after selector failure")
    else:
        print(f"[ORCH] RERUN SEED {case_id}: baseline fallback unavailable")
    return out


def _frozen_target_params_signature(tp: Optional[Dict[str, Any]]) -> Optional[str]:
    """Stable string for comparing Target-parameters dicts across orchestrator runs."""
    if tp is None or not isinstance(tp, dict) or not tp:
        return None
    return json.dumps(tp, sort_keys=True, separators=(",", ":"), default=str)


def _should_skip_completed_experiment_case(
    *,
    case_dir: Path,
    target_params: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Allow skip only when PROCEED+success AND frozen target_parameters on disk match
    the current requirements row. Prevents silent mismatch between requirements.json
    and completed case dirs.
    """
    prev = _read_json(case_dir / "target_spec.json", {})
    prev_tp = prev.get("target_parameters") if isinstance(prev, dict) else None
    if not isinstance(prev_tp, dict):
        prev_tp = None
    sig_new = _frozen_target_params_signature(target_params)
    sig_old = _frozen_target_params_signature(prev_tp)
    if sig_new is None and sig_old is None:
        return True, "no_Target_parameters_json_to_compare_legacy_skip"
    if sig_new is None and sig_old is not None:
        return False, "current_row_missing_Target_parameters_json_but_case_has_frozen_spec"
    if sig_new is not None and sig_old is None:
        return True, "no_frozen_target_spec_on_disk_legacy_skip"
    if sig_new != sig_old:
        return False, "Target_parameters_json_drift_vs_case_target_spec.json"
    return True, "Target_parameters_match_frozen_spec"


def _run_experiment_fast(
    *,
    req: Dict[str, Any],
    case_id: str,
    case_dir: Path,
    run_dir: Path,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    artifact_seed: str,
    target_params: Optional[Dict[str, Any]],
    user_requirement: str,
) -> Dict[str, Any]:
    """Fast-path experiment runner used when the case is seeded from a
    working OED-discovered artifact. Skips the FoamAgent pre-check /
    reviewer / verify_params / viz / interpret LLM pipeline. Generic
    across topics and solvers — the seed's system/controlDict.application
    is preserved as-is.

    Steps:
      1. Copy artifact case → cases/<case_id>/  (drop stale comparator
         outputs, drop intermediate time folders; keep 0/ and the working
         dictionaries verbatim).
      2. Parse the requirement's `Target parameters` block. For each key
         that matches a real coefficient name in the case's runtime
         dictionary (constant/fvModels customSource block etc.), patch the
         scalar value. Unmatched keys are recorded but not propagated.
      3. Run via foam_run_simple.py (no FoamAgent reviewer).
      4. Score via the bound comparator.
      5. Write a synthetic decision.json (PROCEED/REVISE based on
         score-vs-baseline, direction-aware) + run_result.json.
    """
    import shutil
    sys_path_added = []
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
        sys_path_added.append(str(scripts_dir))
    try:
        from run_post_oed_experiments import (
            _copy_case as _bridge_copy_case,  # type: ignore
            _find_runtime_dict_block_name,    # type: ignore
            _parse_coefficients_from_dict_block,  # type: ignore
            _patch_coefficients_in_dict_block,    # type: ignore
        )
    finally:
        for p in sys_path_added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass

    src = Path(artifact_seed)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    _bridge_copy_case(src, case_dir)

    # Parse Target parameters JSON from the requirement text (best-effort).
    overrides_in: Dict[str, Any] = {}
    if isinstance(target_params, dict):
        overrides_in = dict(target_params)
    else:
        try:
            m = re.search(r"Target parameters:\s*(\{[^}]+\})", user_requirement or "", re.DOTALL)
            if m:
                tp = json.loads(m.group(1))
                if isinstance(tp, dict):
                    overrides_in = tp
        except Exception:
            overrides_in = {}

    patched_keys: List[str] = []
    skipped_keys: List[str] = []

    # Try to locate the runtime dict block (works for class_derivation cases
    # whose constant/momentumTransport carries a coefficient sub-dict, and
    # for runtime_source cases whose constant/fvModels carries a coded
    # block). Best-effort across both.
    candidate_dicts = [
        case_dir / "constant" / "fvModels",
        case_dir / "constant" / "fvOptions",
        case_dir / "constant" / "momentumTransport",
        case_dir / "constant" / "transportProperties",
    ]
    if overrides_in:
        for dict_path in candidate_dicts:
            if not dict_path.is_file():
                continue
            try:
                text = dict_path.read_text(encoding="utf-8")
            except Exception:
                continue
            block_name = _find_runtime_dict_block_name(text)
            if not block_name:
                continue
            existing = _parse_coefficients_from_dict_block(text, block_name)
            if not existing:
                continue
            apply: Dict[str, Any] = {
                k: v for k, v in overrides_in.items()
                if k in existing and isinstance(v, (int, float, str))
            }
            if apply:
                new_text, patched = _patch_coefficients_in_dict_block(text, block_name, apply)
                if patched:
                    dict_path.write_text(new_text, encoding="utf-8")
                    patched_keys.extend(patched)
        skipped_keys = [k for k in overrides_in.keys() if k not in patched_keys]

    plan = {
        "case_id": case_id,
        "fast_path": True,
        "artifact_seed": str(src),
        "patched_keys": patched_keys,
        "skipped_keys_no_match_in_dict": skipped_keys,
    }
    (case_dir / "_fast_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Run via foam_run_simple.py
    env_case = dict(env)
    env_case["CFD_TOKEN_LOG_PATH"] = str(case_dir / "llm_token_usage.json")
    run_result_path = case_dir / "run_result.json"
    cmd = [
        sys.executable, "scripts/foam_run_simple.py",
        "--base-case", str(case_dir),
        "--output-dir", str(case_dir),
        "--output", str(run_result_path),
        "--timeout", "21600",
    ]
    append_timeline_event(timeline_path, {
        "stage": "experiment",
        "event": "fast_path_start",
        "case_id": case_id,
        "patched_keys": patched_keys,
    })
    rc = subprocess.run(cmd, cwd=repo_root, env=env_case,
                        timeout=22000, check=False).returncode
    run_result = _read_json(run_result_path, {})
    run_ok = (rc == 0) and (str(run_result.get("status", "")).upper() == "OK")

    # Score via bound comparator (uses run_dir's objective_contract.json
    # written by OED). Deterministic, no LLM.
    primary: Optional[Dict[str, Any]] = None
    if run_ok:
        try:
            from open_ended_discovery import (  # type: ignore
                _run_bound_comparator as _oed_run_bound_comparator,
                _extract_error_metrics as _oed_extract_error_metrics,
                _choose_primary_score as _oed_choose_primary_score,
            )
            objective_contract = _read_json(
                run_dir / "open_ended_discovery" / "objective_contract.json", {})
            comp_out = _oed_run_bound_comparator(case_dir, objective_contract)
            if comp_out:
                extracted = _oed_extract_error_metrics(comp_out)
                primary = _oed_choose_primary_score(extracted)
        except Exception as ex:
            print(f"[ORCH][fast-path] comparator step failed for {case_id}: "
                  f"{type(ex).__name__}: {ex}")

    # Direction-aware PROCEED/REVISE from baseline.
    final_status = "FAILED"
    reason = ""
    baseline_metrics = _read_json(run_dir / "baseline_metrics.json", {})
    bs = baseline_metrics.get("primary_score") if isinstance(baseline_metrics, dict) else None
    bs_val = bs.get("value") if isinstance(bs, dict) else None
    bs_dir = (bs.get("direction", "min") if isinstance(bs, dict) else "min").strip().lower()
    if run_ok:
        if isinstance(primary, dict) and primary.get("value") is not None and bs_val is not None:
            v = float(primary["value"])
            beats = (v < float(bs_val)) if bs_dir != "max" else (v > float(bs_val))
            final_status = "PROCEED" if beats else "REVISE"
            cmp_word = "<" if (bs_dir != "max" and beats) else (">" if beats else ("≥" if bs_dir != "max" else "≤"))
            reason = (
                f"fast-path: comparator score {primary.get('metric','?')}="
                f"{v:.6g} {cmp_word} baseline {float(bs_val):.6g} "
                f"(direction={bs_dir})"
            )
        else:
            final_status = "UNKNOWN"
            reason = "fast-path: case ran but no comparator score available"
    else:
        final_status = "FAILED"
        reason = (run_result.get("error") or "fast-path: simpleFoam (or app) did not reach End")[:300]

    decision = {
        "status": final_status,
        "confidence": 0.8 if final_status == "PROCEED" else (0.3 if final_status == "REVISE" else 0.0),
        "reason": reason,
        "raw": {"score": primary, "baseline": bs_val, "direction": bs_dir,
                "patched_keys": patched_keys, "skipped_keys": skipped_keys},
        "fast_path": True,
    }
    (case_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    append_timeline_event(timeline_path, {
        "stage": "experiment",
        "event": "fast_path_finish",
        "case_id": case_id,
        "final_status": final_status,
        "score": primary,
    })

    return {
        "case_id": case_id,
        "final_status": final_status,
        "attempts": [{"attempt": 1, "status": final_status, "reason": reason,
                       "fast_path": True, "score": primary}],
        "case_dir": str(case_dir),
    }


def _run_experiment(
    req: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    max_reruns: int,
) -> Dict[str, Any]:
    case_id = str(req.get("case_id") or req.get("experiment_id") or "case")
    user_requirement = str(req.get("user_requirement_text") or "")
    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = case_dir / "figs"
    decision_path = case_dir / "decision.json"
    run_result_path = case_dir / "run_result.json"

    selected_mesh_spec = _read_json(run_dir / "selected_mesh_spec.json", {})
    mesh_note = ""
    mesh_gate_case_dir = ""
    if isinstance(selected_mesh_spec, dict) and selected_mesh_spec:
        ver = int(selected_mesh_spec.get("version") or 0)
        if ver >= 2:
            groups = selected_mesh_spec.get("groups") or {}
            cmap = selected_mesh_spec.get("case_to_group") or {}
            gid = str(cmap.get(case_id) or selected_mesh_spec.get("default_group") or "").strip()
            ginfo = groups.get(gid) if gid else None
            if isinstance(ginfo, dict):
                mesh_note = str(ginfo.get("requirement_suffix", "") or "")
                sel_level = str(ginfo.get("selected_level", "") or "").strip()
                if sel_level:
                    sel_p = Path(sel_level).expanduser().resolve()
                    if sel_p.exists() and (sel_p / "system" / "controlDict").exists():
                        mesh_gate_case_dir = str(sel_p)
        if not mesh_gate_case_dir:
            mesh_note = str(selected_mesh_spec.get("requirement_suffix", "") or "")
            sel_level = str(selected_mesh_spec.get("selected_level", "") or "").strip()
            if sel_level:
                sel_p = Path(sel_level).expanduser().resolve()
                if sel_p.exists() and (sel_p / "system" / "controlDict").exists():
                    mesh_gate_case_dir = str(sel_p)

    state = _read_json(run_dir / "state.json", {})
    seed_dir = ""
    if isinstance(state, dict):
        raw_seed = state.get("starter_seed_case_dir") or ""
        if isinstance(raw_seed, str) and raw_seed.strip():
            cand = Path(raw_seed.strip()).expanduser().resolve()
            if cand.exists() and (cand / "system" / "controlDict").exists():
                seed_dir = str(cand)

    # OED-artifact-aware seed selection. When the run produced a winning
    # code-modification artifact (OED or regular code_mod path) and that
    # artifact's working base case is on disk (with constant/, system/, 0/
    # all wired up — compiled customModels source + activated libs() +
    # selected model in the activation dict), use it as the experiment's
    # seed. This way each paper case starts from a case that ALREADY runs
    # cleanly with the discovered model loaded, and FoamAgent's reviewer
    # only needs to:
    #   - patch coefficient values per the requirement
    #   - swap the active model selection if the requirement asks for the
    #     parent / built-in variant for a reference case
    # Generic across modification families: works for runtime_source
    # (coded fvModels), class_derivation (compiled .so + customModels), and
    # dict_only patches. The mesh-gate-selected case still wins when set
    # (it may dictate a different mesh per physics group).
    artifact_seed = ""
    artifact_json = run_dir / "oed_artifact.json"
    if artifact_json.is_file():
        art = _read_json(artifact_json, {})
        if isinstance(art, dict) and art.get("status") == "ok":
            cand = (art.get("base_case_dir") or "").strip()
            if cand:
                p = Path(cand).expanduser().resolve()
                if p.is_dir() and (p / "system" / "controlDict").is_file() and (p / "constant").is_dir():
                    artifact_seed = str(p)

    # Precedence: mesh-gate (per-physics-group mesh) > OED artifact > starter seed
    effective_seed = mesh_gate_case_dir or artifact_seed or seed_dir

    # FAST PATH: when seeded from a working OED artifact case, skip the
    # FoamAgent pre-check / reviewer / verify_params / viz / interpret
    # pipeline. The artifact case already runs cleanly; we just need to
    # apply per-case parameter overrides (best-effort) and rerun the solver.
    # Generic across modification families and across solvers (the seed's
    # system/controlDict.application is whatever it was for the OED winner —
    # simpleFoam, pimpleFoam, foamRun, chtMultiRegionFoam, etc.).
    if effective_seed and artifact_seed and effective_seed == artifact_seed:
        try:
            return _run_experiment_fast(
                req=req,
                case_id=case_id,
                case_dir=case_dir,
                run_dir=run_dir,
                repo_root=repo_root,
                env=env,
                timeline_path=timeline_path,
                artifact_seed=artifact_seed,
                target_params=None,  # parsed inside helper
                user_requirement=user_requirement,
            )
        except Exception as ex:
            # Fall back to slow path on any unexpected failure rather than
            # losing the case entirely.
            print(f"[ORCH][fast-path] failed for {case_id} — falling back to "
                  f"FoamAgent path: {type(ex).__name__}: {ex}")

    seed_prefix = ""
    if effective_seed:
        seed_prefix = (
            "Baseline OpenFOAM case directory (full case including polyMesh on disk). "
            "Use this as the starting point: preserve the mesh unless the requirement explicitly "
            f"requires a new mesh; adapt 0/, system/, and constant/ as needed.\nPath: {effective_seed}\n\n---\n\n"
        )

    _bootstrap_paths(repo_root)
    from cfd_langgraph.experiment_requirement_canon import prepare_experiment_requirement_strings

    foam_req, verify_req, target_params = prepare_experiment_requirement_strings(
        user_requirement=user_requirement,
        seed_prefix=seed_prefix,
        mesh_note=mesh_note,
    )
    base_foam_req = foam_req

    spec_save: Dict[str, Any] = {"case_id": case_id, "target_parameters": target_params}
    if target_params is None:
        spec_save["note"] = (
            "No Target parameters:{...} JSON parsed; Foam/verify use snapshot-stripped text only."
        )

    # Skip already-completed experiments: PROCEED + successful run on disk (no LLM re-verify),
    # only if Target parameters still match frozen case/target_spec.json (avoid requirements drift).
    # To force a rerun: remove decision.json / run_result.json or the case dir.
    if decision_path.exists() and run_result_path.exists():
        existing_decision = _read_json(decision_path, {})
        existing_run_result = _read_json(run_result_path, {})
        if (
            str(existing_decision.get("status", "")).upper() == "PROCEED"
            and str(existing_run_result.get("status", "")).lower() == "success"
        ):
            may_skip, skip_detail = _should_skip_completed_experiment_case(
                case_dir=case_dir,
                target_params=target_params,
            )
            if may_skip:
                print(
                    f"[ORCH] EXPERIMENT {case_id} already PROCEED + run success on disk; skipping to next "
                    f"({skip_detail})."
                )
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "experiment",
                        "event": "skip_completed",
                        "case_id": case_id,
                        "detail": skip_detail,
                    },
                )
                return {
                    "case_id": case_id,
                    "case_dir": str(case_dir),
                    "final_status": "PROCEED",
                    "attempts": [
                        {
                            "attempt": 0,
                            "status": "PROCEED",
                            "reason": "already completed in previous run",
                            "run_status": "success",
                        }
                    ],
                }
            print(
                f"[ORCH] EXPERIMENT {case_id}: completed case on disk but NOT skipping — {skip_detail}. "
                "Re-running FoamAgent / verify so case matches current requirements."
            )
            append_timeline_event(
                timeline_path,
                {
                    "stage": "experiment",
                    "event": "skip_completed_blocked",
                    "case_id": case_id,
                    "detail": skip_detail,
                },
            )

    _write_json(case_dir / "target_spec.json", spec_save)

    attempts: List[Dict[str, Any]] = []
    last_seed_meta: Dict[str, Any] = {}
    last_interpreter_reason = ""
    baseline_for_seed = (effective_seed or "").strip()
    for attempt in range(1, max_reruns + 1):
        foam_req = _compose_rerun_foam_requirement(
            base_foam_req, attempt, last_seed_meta, last_interpreter_reason
        )
        print(f"[ORCH] EXPERIMENT {case_id} attempt {attempt}/{max_reruns}")
        append_timeline_event(
            timeline_path,
            {
                "stage": "experiment",
                "event": "attempt_start",
                "case_id": case_id,
                "attempt": attempt,
                "user_requirement_text": foam_req[:24000],
            },
        )
        # On rerun attempts, clear stale viz artifacts so the interpreter sees fresh
        # plots rather than re-evaluating the same bad images from the previous attempt.
        if attempt > 1:
            for _stale_dir in ("figs", "interpreter_viz"):
                _stale_path = case_dir / _stale_dir
                if _stale_path.exists():
                    shutil.rmtree(_stale_path, ignore_errors=True)
                    print(f"[ORCH] RERUN {case_id} attempt {attempt}: cleared {_stale_dir}/")

        env_case = dict(env)
        env_case["CFD_TOKEN_LOG_PATH"] = str(case_dir / "llm_token_usage.json")
        foam_cmd: List[str] = [
            sys.executable,
            "scripts/foam_run.py",
            "--requirement",
            foam_req,
            "--output-dir",
            str(case_dir),
            "--max-loop",
            "10",
            "--max-time-limit",
            "21600",
            "--precheck-max-loop",
            "5",
            "--timeline",
            str(timeline_path),
        ]
        if attempt == 1 and effective_seed:
            foam_cmd.extend(["--base-case-dir", effective_seed])
            run_mode = "copy_seed_case_then_edit"
        elif attempt > 1 and (case_dir / "system" / "controlDict").exists():
            foam_cmd.append("--reuse-existing-case")
            run_mode = "reuse_existing_case_then_edit"
        elif attempt > 1 and seed_dir:
            foam_cmd.extend(["--base-case-dir", seed_dir])
            run_mode = "reseed_case_then_edit"
        else:
            run_mode = "generate_or_mesh_then_write"
        append_timeline_event(
            timeline_path,
            {
                "stage": "experiment",
                "event": "run_strategy",
                "case_id": case_id,
                "attempt": attempt,
                "run_mode": run_mode,
                "seed_dir": seed_dir,
            },
        )
        attempt_failed = False
        try:
            _call_stage(
                foam_cmd,
                stage=f"foam_run:{case_id}:attempt_{attempt}",
                repo_root=repo_root,
                env=env_case,
                timeline_path=timeline_path,
                state_path=run_dir / "state.json",
            )
        except RuntimeError:
            print(f"[ORCH] WARNING: foam_run {case_id} attempt {attempt} failed (timeout/crash); will retry if attempts remain.")
            append_timeline_event(timeline_path, {"stage": "experiment", "event": "attempt_run_failed", "case_id": case_id, "attempt": attempt})
            attempt_failed = True

        # --- LLM verify: if it rewrites dictionaries after the solve, re-run foam so fields match ---
        _max_post_verify_foam = 3
        post_verify_foam_runs = 0
        if not attempt_failed:
            while True:
                print(f"[ORCH] VERIFY {case_id}: running LLM verification on case configuration...")
                verify_result = _llm_verify_case(
                    case_dir, verify_req, repo_root, env_case, timeline_path,
                )
                vr_path = case_dir / "verify_result.json"
                vr_merged = _read_json(vr_path, verify_result)
                fixes_applied = int(vr_merged.get("fixes_applied") or 0)
                is_ok = bool(vr_merged.get("is_correct", False))

                if not is_ok:
                    print(f"[ORCH] VERIFY {case_id}: LLM found issues — {vr_merged.get('reasoning', '?')}")
                    for mm in vr_merged.get("mismatches", []) or []:
                        if not isinstance(mm, dict):
                            continue
                        print(
                            f"[ORCH]   {mm.get('file','?')}: {mm.get('param','?')} — "
                            f"expected '{mm.get('expected','?')}', got '{mm.get('actual','?')}'"
                        )
                    append_timeline_event(
                        timeline_path,
                        {
                            "stage": "experiment",
                            "event": "param_verify_mismatch",
                            "case_id": case_id,
                            "attempt": attempt,
                            "mismatches": vr_merged.get("mismatches", []),
                            "fixes_applied": fixes_applied,
                        },
                    )
                    break

                print(f"[ORCH] VERIFY {case_id}: configuration confirmed correct by LLM")
                if fixes_applied <= 0:
                    break

                if post_verify_foam_runs >= _max_post_verify_foam:
                    print(
                        f"[ORCH] WARNING: VERIFY {case_id}: dictionaries still being rewritten after "
                        f"{_max_post_verify_foam} post-verify foam re-runs; stopping to avoid a loop."
                    )
                    append_timeline_event(
                        timeline_path,
                        {
                            "stage": "experiment",
                            "event": "post_verify_foam_cap",
                            "case_id": case_id,
                            "attempt": attempt,
                            "fixes_applied_reported": fixes_applied,
                        },
                    )
                    break

                if not (case_dir / "system" / "controlDict").exists():
                    print(
                        f"[ORCH] WARNING: VERIFY {case_id}: {fixes_applied} file(s) fixed but no controlDict; "
                        "skipping post-verify foam re-run."
                    )
                    break

                post_verify_foam_runs += 1
                print(
                    f"[ORCH] VERIFY {case_id}: {fixes_applied} dictionary file(s) were updated after the solve; "
                    f"re-running foam_run ({post_verify_foam_runs}/{_max_post_verify_foam}) so results match config."
                )
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "experiment",
                        "event": "post_verify_foam_rerun",
                        "case_id": case_id,
                        "attempt": attempt,
                        "post_verify_pass": post_verify_foam_runs,
                        "fixes_applied_reported": fixes_applied,
                    },
                )
                foam_cmd_post_verify: List[str] = [
                    sys.executable,
                    "scripts/foam_run.py",
                    "--requirement",
                    foam_req,
                    "--output-dir",
                    str(case_dir),
                    "--max-loop",
                    "10",
                    "--max-time-limit",
                    "21600",
                    "--precheck-max-loop",
                    "5",
                    "--timeline",
                    str(timeline_path),
                    "--reuse-existing-case",
                ]
                try:
                    _call_stage(
                        foam_cmd_post_verify,
                        stage=f"foam_run:{case_id}:attempt_{attempt}:post_verify_{post_verify_foam_runs}",
                        repo_root=repo_root,
                        env=env_case,
                        timeline_path=timeline_path,
                        state_path=run_dir / "state.json",
                    )
                except RuntimeError:
                    print(f"[ORCH] WARNING: post-verify foam_run {case_id} failed; stopping verify/solve loop.")
                    append_timeline_event(
                        timeline_path,
                        {
                            "stage": "experiment",
                            "event": "post_verify_foam_failed",
                            "case_id": case_id,
                            "attempt": attempt,
                        },
                    )
                    attempt_failed = True
                    break

        if not attempt_failed:
            try:
                _call_stage(
                    [
                        sys.executable,
                        "scripts/viz.py",
                        "--case",
                        str(case_dir),
                        "--mode",
                        "interpret",
                        "--output",
                        str(figs_dir),
                    ],
                    stage=f"viz:{case_id}:attempt_{attempt}",
                    repo_root=repo_root,
                    env=env_case,
                    timeline_path=timeline_path,
                    state_path=run_dir / "state.json",
                )
                _call_stage(
                    [
                        sys.executable,
                        "scripts/interpret.py",
                        "--case",
                        str(case_dir),
                        "--figs",
                        str(figs_dir),
                        "--output",
                        str(decision_path),
                        "--requirement",
                        verify_req,
                        "--timeline",
                        str(timeline_path),
                    ],
                    stage=f"interpret:{case_id}:attempt_{attempt}",
                    repo_root=repo_root,
                    env=env_case,
                    timeline_path=timeline_path,
                    state_path=run_dir / "state.json",
                )
            except RuntimeError:
                print(f"[ORCH] WARNING: viz/interpret {case_id} attempt {attempt} failed; treating as RERUN.")
                attempt_failed = True

        if attempt_failed:
            attempts.append({"attempt": attempt, "status": "RERUN", "reason": "stage crashed or timed out", "run_status": "failed"})
            append_timeline_event(timeline_path, {"stage": "experiment", "event": "attempt_finish", "case_id": case_id, "attempt": attempt, "status": "RERUN", "reason": "stage crashed"})
            # foam_run / viz / interpret crash: still reset dicts from a successful sibling so the
            # next --reuse-existing-case attempt does not keep broken system/0/constant.
            last_seed_meta = _rerun_seed_from_nearest_success(
                run_dir=run_dir,
                case_id=case_id,
                case_dir=case_dir,
                req=req,
                repo_root=repo_root,
                env=env,
                timeline_path=timeline_path,
                attempt=attempt,
                trigger="foam_or_viz_fail",
                baseline_case_dir=baseline_for_seed,
            )
            last_interpreter_reason = (
                "Previous attempt failed during OpenFOAM execution, visualization, or interpreter I/O "
                "(no PROCEED). Inspect run_result.json, log.* files, and orchestrator stage output."
            )
            continue

        decision = _read_json(decision_path, {})
        run_result = _read_json(run_result_path, {})
        status = str(decision.get("status") or "REVISE").upper()

        # Override: if interpreter says PROCEED but LLM verification found issues, force RERUN.
        # Re-read the verify_result.json written by the verify gate above.
        if status == "PROCEED":
            vr_path = case_dir / "verify_result.json"
            vr = _read_json(vr_path, {})
            if vr and not vr.get("is_correct", True):
                mismatches = vr.get("mismatches", [])
                mismatch_summary = "; ".join(
                    f"{m.get('param','?')}: expected {m.get('expected','?')}, got {m.get('actual','?')}"
                    for m in mismatches
                )
                print(
                    f"[ORCH] VERIFY OVERRIDE {case_id}: interpreter said PROCEED but "
                    f"LLM verification failed -> forcing RERUN"
                )
                if mismatch_summary:
                    print(f"[ORCH]   Mismatches: {mismatch_summary}")
                status = "RERUN"
                decision["status"] = "RERUN"
                decision["reason"] = (
                    f"LLM verification failed: {vr.get('reasoning', '')}. "
                    + (f"Mismatches: {mismatch_summary}. " if mismatch_summary else "")
                    + decision.get("reason", "")
                )
                decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

        print(
            f"[ORCH] EXPERIMENT {case_id} attempt {attempt} decision={status} "
            f"run_status={run_result.get('status', 'unknown')}"
        )
        attempt_rec = {
            "attempt": attempt,
            "status": status,
            "reason": decision.get("reason", ""),
            "run_status": run_result.get("status", "unknown"),
            "requirement_used": foam_req,
        }
        attempts.append(attempt_rec)
        append_timeline_event(
            timeline_path,
            {
                "stage": "experiment",
                "event": "attempt_finish",
                "case_id": case_id,
                "attempt": attempt,
                "status": status,
                "reason": decision.get("reason", ""),
            },
        )
        if status == "PROCEED":
            print(f"[ORCH] EXPERIMENT {case_id} completed with PROCEED")
            return {"case_id": case_id, "final_status": "PROCEED", "attempts": attempts, "case_dir": str(case_dir)}

        # Dynamic file-aware rerun prep: seed from similar successful case if available, else baseline.
        reason_text = str(decision.get("reason", "")).strip()
        last_seed_meta = _rerun_seed_from_nearest_success(
            run_dir=run_dir,
            case_id=case_id,
            case_dir=case_dir,
            req=req,
            repo_root=repo_root,
            env=env,
            timeline_path=timeline_path,
            attempt=attempt,
            trigger="interpreter_rerun",
            baseline_case_dir=baseline_for_seed,
        )
        # Only the immediately previous attempt's interpreter (or verify override) summary — no stacking.
        last_interpreter_reason = reason_text or "Re-run requested (REVISE/RERUN) without interpreter text."
        append_timeline_event(
            timeline_path,
            {
                "stage": "experiment",
                "event": "requirement_revised",
                "case_id": case_id,
                "attempt": attempt,
                "reason": reason_text,
                "seed_method": last_seed_meta.get("method"),
                "seeded": last_seed_meta.get("seeded"),
            },
        )

    return {"case_id": case_id, "final_status": "FAILED_MAX_RERUNS", "attempts": attempts, "case_dir": str(case_dir)}


def _collect_case_files_text(case_dir: Path, max_chars: int = 120000) -> str:
    """Read 0/, constant/, system/ (skip polyMesh) into a single text block for LLM context."""
    lines: list[str] = []
    total = 0
    for folder in ("0", "constant", "system"):
        root = case_dir / folder
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(case_dir)).replace("\\", "/")
            if "polyMesh" in rel:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunk = f"--- {rel} ---\n{txt}\n"
            if total + len(chunk) > max_chars:
                lines.append(f"--- (truncated; {max_chars} char limit reached) ---\n")
                return "\n".join(lines)
            lines.append(chunk)
            total += len(chunk)
    return "\n".join(lines)


def _collect_code_mod_context(run_dir: Path, state_path: Path, include_source: bool = False) -> str:
    """Summarise code-mod changes (customModels, code_mod_payload, case dictionaries).

    When *include_source* is True the actual C++/H source and case dictionary
    snippets are included so that downstream LLM calls (hypothesis revision,
    requirements revision) know the exact model name, parameters, and equation.
    """
    parts: list[str] = []
    st = _read_json(state_path, {})
    payload_path = run_dir / "code_mod_payload.json"
    if payload_path.is_file():
        try:
            pay = json.loads(payload_path.read_text(encoding="utf-8"))
            if isinstance(pay, dict):
                parts.append(f"Code-mod payload (model changes): {json.dumps(pay, indent=1)[:6000]}")
        except Exception:
            pass

    seed_dir = None
    if isinstance(st, dict):
        raw = st.get("starter_seed_case_dir") or ""
        if isinstance(raw, str) and raw.strip():
            seed_dir = Path(raw.strip())
    if seed_dir is None:
        canon = run_dir / "canonical_base_case"
        if canon.exists():
            seed_dir = canon

    if seed_dir is not None:
        cm = seed_dir / "customModels"
        if cm.is_dir():
            for md in sorted(cm.iterdir())[:8]:
                if not md.is_dir():
                    continue
                parts.append(f"customModels/{md.name}/")
                for ext in ("*.H", "*.C"):
                    for f in sorted(md.glob(ext))[:3]:
                        try:
                            src = f.read_text(encoding="utf-8", errors="ignore")
                            if include_source:
                                parts.append(f"--- {f.name} ---\n{src[:8000]}")
                            else:
                                parts.append(f"  {f.name} ({len(src)} chars)")
                        except Exception:
                            pass

        # Scan all dictionary files under constant/ (skip polyMesh)
        const_dir = seed_dir / "constant"
        if const_dir.is_dir():
            for p in sorted(const_dir.rglob("*")):
                if not p.is_file() or "polyMesh" in str(p.relative_to(seed_dir)):
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    rel = str(p.relative_to(seed_dir))
                    parts.append(f"--- {rel} ---\n{txt[:4000]}")
                except Exception:
                    pass

    if not parts:
        return ""
    return "CODE-MOD CONTEXT:\n" + "\n".join(parts)


def _llm_decide_analysis_metrics(
    *,
    repo_root: Path,
    topic: str,
    requirement_text: str,
    code_mod_context: str = "",
    max_metrics: int = 8,
) -> List[str]:
    """
    Let LLM choose analysis metrics for this study context.
    Returns a non-empty metric list.
    """
    _bootstrap_paths(repo_root)
    try:
        from cfd_langgraph.config import get_settings
        from cfd_langgraph.llm.factory import create_langchain_llm
        from cfd_langgraph.utils import strip_json_fences
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return [
            "centreline_Ux_mean",
            "centreline_Ux_max",
            "Umag_mean",
            "Umag_max",
            "wall_shear_mean",
            "reattachment_length",
        ]

    llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
    system_prompt = (
        "You are a CFD analysis planner. "
        "Choose metrics that are physically meaningful and extractable for the given case type. "
        "Prefer robust, field-derived metrics for internal/channel flows. "
        "Avoid body-force/aero coefficients (Cd/Cl) unless the requirement clearly includes an immersed body setup. "
        "Return ONLY valid JSON object: "
        '{"metrics": ["metric1", "metric2", "..."], "reason": "short rationale"}'
    )
    user_prompt = (
        f"Study topic:\n{topic}\n\n"
        f"Requirement text:\n{requirement_text[:12000]}\n\n"
        f"Code-mod context (if any):\n{code_mod_context[:12000]}\n\n"
        f"Select up to {max_metrics} metrics for cross-case analysis and mesh comparisons.\n"
        "Must include at least 4 metrics. Prefer keys likely derivable from PyVista field data "
        "(e.g., centreline_Ux_mean/max/min, Umag_mean/max, wall_shear_mean/max, "
        "pressure_drop_proxy, bulk_velocity_Ux, reattachment_length when separation exists, "
        "y_plus only if wall-treatment context exists).\n"
        "For mesh-independence studies, prefer quantities that stay well above numerical noise when the flow is "
        "correctly driven (e.g. pressure_drop_proxy, bulk_velocity_Ux); avoid relying solely on fields that may "
        "decay to ~0 if BCs are wrong."
    )
    try:
        out = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        raw = strip_json_fences(str(getattr(out, "content", out)).strip())
        payload = json.loads(raw)
        metrics = payload.get("metrics", []) if isinstance(payload, dict) else []
        metrics = [str(m).strip() for m in metrics if str(m).strip()]
        dedup: List[str] = []
        for m in metrics:
            if m not in dedup:
                dedup.append(m)
        if len(dedup) >= 1:
            return dedup[:max_metrics]
    except Exception:
        pass
    return [
        "centreline_Ux_mean",
        "centreline_Ux_max",
        "Umag_mean",
        "Umag_max",
        "wall_shear_mean",
        "reattachment_length",
    ]


def _merge_mesh_gate_metrics(current: List[str], suggested: List[str], max_metrics: int = 10) -> List[str]:
    out: List[str] = []
    for m in suggested + current:
        s = str(m).strip()
        if s and s not in out:
            out.append(s)
    return out[:max_metrics]


def _heuristic_mesh_gate_pair_fallback(
    q_a: Dict[str, Any],
    q_b: Dict[str, Any],
    pct_changes: Dict[str, float],
    *,
    magnitude_floor: float = 1e-9,
) -> Dict[str, Any]:
    """
    If the LLM is unavailable: never trust relative % when both sides are ~zero/noise.
    Only apply the legacy 5% rule to metrics whose magnitude exceeds a small floor.
    If none qualify, stop refinement (converged=True) to avoid an infinite spiral on garbage QoIs.
    """
    reliable_pct: Dict[str, float] = {}
    for k, p in pct_changes.items():
        va, vb = q_a.get(k), q_b.get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        fa, fb = float(va), float(vb)
        if max(abs(fa), abs(fb)) >= magnitude_floor:
            reliable_pct[k] = float(p)
    if not reliable_pct:
        return {
            "converged": True,
            "reason": (
                "Heuristic fallback: no paired QoI exceeded the magnitude floor; "
                "relative percent change is not meaningful — stopping refinement spiral."
            ),
            "qoi_reliability": "unreliable",
            "recommended_metrics_for_retry": [],
            "source": "heuristic_fallback",
        }
    conv = all(v <= 5.0 for v in reliable_pct.values())
    return {
        "converged": conv,
        "reason": (
            f"Heuristic fallback: applied 5% rule only to {list(reliable_pct.keys())} "
            f"(|Q|>={magnitude_floor})."
        ),
        "qoi_reliability": "mixed",
        "recommended_metrics_for_retry": [],
        "source": "heuristic_fallback",
    }


def _llm_mesh_gate_pair_convergence(
    llm: Any,
    *,
    parent_label: str,
    child_label: str,
    q_a: Dict[str, Any],
    q_b: Dict[str, Any],
    pct_changes: Dict[str, float],
    metrics_requested: List[str],
    topic_excerpt: str,
    requirement_excerpt: str,
    metric_attempt_index: int,
    max_metric_attempts: int,
) -> Dict[str, Any]:
    """
    LLM judges mesh sensitivity for one parent→child pair using raw QoIs + naive % deltas.
    May recommend different metrics for a re-run of analyze.py when values are noise-dominated.
    """
    from pydantic import BaseModel as _BM, Field as _F

    class _MeshGatePairDecision(_BM):
        converged: bool = _F(
            description=(
                "True if the coarser (parent) mesh is adequate: for QoIs you judge trustworthy, "
                "parent vs child differ by at most ~5% (standard mesh-independence target), "
                "OR QoIs are unreliable/noise-dominated and further refinement should stop."
            )
        )
        reason: str = _F(description="Short engineering rationale referencing magnitudes, not only %.")
        qoi_reliability: str = _F(
            description="One of: good, mixed, unreliable — are extracted QoIs trustworthy for a % comparison?"
        )
        recommended_metrics_for_retry: List[str] = _F(
            default_factory=list,
            description=(
                "If qoi_reliability is unreliable and a different metric set could help "
                "(e.g. early-time bulk U, pressure_drop_proxy, wall shear), suggest metric names "
                "for scripts/analyze.py. Empty if no retry or attempt budget exhausted."
            ),
        )

    payload_preview = {
        "parent": parent_label,
        "child": child_label,
        "metrics_requested": metrics_requested,
        "qoi_parent": {k: q_a.get(k) for k in sorted(q_a.keys()) if not str(k).startswith("_")},
        "qoi_child": {k: q_b.get(k) for k in sorted(q_b.keys()) if not str(k).startswith("_")},
        "naive_percent_change_parent_to_child": pct_changes,
        "metric_attempt_index": metric_attempt_index,
        "max_metric_attempts": max_metric_attempts,
    }
    system_prompt = (
        "You are a senior CFD engineer judging mesh sensitivity between two OpenFOAM runs "
        "(parent = coarser mesh, child = finer mesh).\n"
        "You receive raw QoIs from an automated PyVista/LLM extraction and naive relative percent changes "
        "computed as |v_child - v_parent| / max(|v_parent|, 1e-12) * 100.\n\n"
        "**Timestep policy:** Mesh-convergence QoIs are computed from the **final written timestep only** "
        "(last time in each case’s output), not by scanning or averaging over all intermediate write times. "
        "Assume both simulations were run to the **same endTime** and final output time unless the provided "
        "QoI JSON clearly shows different `pyvista_time_used` values — if they differ, note it in reason.\n\n"
        "Rules:\n"
        "- If expected velocities are O(1) m/s but reported QoIs are ~1e-12 or smaller, treat them as "
        "noise / wrong time window / missing driving force — NOT as mesh sensitivity. "
        "Percent changes between noise floors are meaningless (often 10^4–10^5 %).\n"
        "- Prefer metrics anchored in physics: sustained bulk or centreline speed, pressure drop proxy, "
        "wall shear, integrated flow rate — evaluated at that **final timestep** (developed steady state when applicable).\n"
        "- If current metrics are unreliable, set qoi_reliability=unreliable and suggest "
        "recommended_metrics_for_retry (e.g. bulk_velocity_Ux, pressure_drop_proxy, wall_shear_mean). "
        "Do **not** ask for multi-timestep sweeps for mesh convergence; stay at the final time.\n"
        "- If metric_attempt_index >= max_metric_attempts - 1, do NOT suggest retries; "
        "set converged=true if further refinement would only chase numerical noise, and explain.\n"
        "- **5% convergence rule (for trustworthy QoIs only):** When qoi_reliability is good or mixed, "
        "if all key physics QoIs you rely on change by **≤ 5%** from parent to child "
        "(use the provided naive_percent_change values when they apply, or compare raw qoi_parent vs qoi_child), "
        "that **is an acceptable mesh convergence outcome** — set **converged=true** and say so in reason.\n"
        "- If any **trustworthy** key QoI changes by **> 5%**, set **converged=false** so refinement can continue "
        "(unless the change is clearly numerical artifact — then explain).\n"
        "- converged=true means: stop refining (parent mesh is the selected level for this gate). "
        "converged=false means: trustworthy QoIs still differ by more than ~5% — continue refinement.\n"
    )
    user_prompt = (
        f"Study topic (excerpt):\n{topic_excerpt[:3000]}\n\n"
        f"Requirement (excerpt):\n{requirement_excerpt[:6000]}\n\n"
        f"Pair data (JSON):\n{json.dumps(payload_preview, indent=2)[:14000]}\n"
    )
    try:
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

        structured = llm.with_structured_output(_MeshGatePairDecision)
        out: _MeshGatePairDecision = structured.invoke([_SM(content=system_prompt), _HM(content=user_prompt)])
        return {
            "converged": bool(out.converged),
            "reason": str(out.reason or "").strip(),
            "qoi_reliability": str(out.qoi_reliability or "mixed").strip().lower(),
            "recommended_metrics_for_retry": [
                str(x).strip() for x in (out.recommended_metrics_for_retry or []) if str(x).strip()
            ],
            "source": "llm",
        }
    except Exception as exc:
        fb = _heuristic_mesh_gate_pair_fallback(q_a, q_b, pct_changes)
        fb["reason"] = f"LLM mesh-gate decision failed ({exc}); {fb.get('reason', '')}"
        fb["source"] = "heuristic_after_llm_error"
        return fb


def _llm_plan_analysis_stage(
    *,
    repo_root: Path,
    topic: str,
    requirement_text: str,
    case_dirs: List[str],
    code_mod_context: str = "",
    max_metrics: int = 8,
) -> Dict[str, Any]:
    """
    Build an analysis-stage plan with:
      - metrics: cross-case QoIs to compute
      - case_viz_spec_default: what full-mode viz should generate per case
      - case_viz_overrides: optional per-case viz specs
      - cross_case_objectives: requested cross-experiment comparisons/plots
    """
    default_metrics = _llm_decide_analysis_metrics(
        repo_root=repo_root,
        topic=topic,
        requirement_text=requirement_text,
        code_mod_context=code_mod_context,
        max_metrics=max_metrics,
    )
    plan: Dict[str, Any] = {
        "metrics": default_metrics,
        "case_viz_spec_default": (
            "Create publication-quality CFD figures for this case: velocity magnitude contour, "
            "pressure contour, streamlines, centerline and cross-stream velocity profiles, and one "
            "zoomed view around the most important feature (shear/recirculation/near-wall zone) "
            "relevant to the experiment intent."
        ),
        "case_viz_overrides": {},
        "cross_case_objectives": [
            "Compare selected QoIs across all cases and identify monotonic/non-monotonic trends.",
            "Generate cross-case comparison plots/tables suitable for paper Results section.",
            "Summarize similarities, differences, and likely physical causes across experiments.",
        ],
        "reason": "fallback_default",
    }

    _bootstrap_paths(repo_root)
    try:
        from cfd_langgraph.config import get_settings
        from cfd_langgraph.llm.factory import create_langchain_llm
        from cfd_langgraph.utils import strip_json_fences
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return plan

    llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
    system_prompt = (
        "You are a CFD analysis-stage planner for multi-experiment studies.\n"
        "Your goal: prepare a paper-oriented analysis plan that is cross-case, not single-case.\n"
        "Return ONLY JSON with keys:\n"
        "{\n"
        '  "metrics": [str],\n'
        '  "case_viz_spec_default": str,\n'
        '  "case_viz_overrides": { "<case_name>": "<viz spec>" },\n'
        '  "cross_case_objectives": [str],\n'
        '  "reason": str\n'
        "}\n"
        "Rules:\n"
        "- Choose physically meaningful metrics for the study topic/cases.\n"
        "- Avoid irrelevant metrics (e.g., Cd/Cl) unless clearly required by setup.\n"
        "- case_viz_spec_default must tell viz_creator what per-case flow figures to generate.\n"
        "- cross_case_objectives must include explicit across-experiment comparisons (trends/correlations/overlays).\n"
        "- Keep metrics <= 8."
    )
    user_prompt = (
        f"Topic:\n{topic}\n\n"
        f"Requirement text:\n{requirement_text[:15000]}\n\n"
        f"Case directories:\n{json.dumps(case_dirs, ensure_ascii=False)}\n\n"
        f"Code-mod context:\n{code_mod_context[:12000]}\n"
    )
    try:
        out = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        raw = strip_json_fences(str(getattr(out, "content", out)).strip())
        payload = json.loads(raw)
        if isinstance(payload, dict):
            metrics = [str(m).strip() for m in payload.get("metrics", []) if str(m).strip()]
            if metrics:
                dedup: List[str] = []
                for m in metrics:
                    if m not in dedup:
                        dedup.append(m)
                plan["metrics"] = dedup[:max_metrics]
            spec = str(payload.get("case_viz_spec_default", "")).strip()
            if spec:
                plan["case_viz_spec_default"] = spec
            overrides = payload.get("case_viz_overrides", {})
            if isinstance(overrides, dict):
                clean_overrides: Dict[str, str] = {}
                for k, v in overrides.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs:
                        clean_overrides[ks] = vs
                plan["case_viz_overrides"] = clean_overrides
            cobj = payload.get("cross_case_objectives", [])
            if isinstance(cobj, list):
                cobj_clean = [str(x).strip() for x in cobj if str(x).strip()]
                if cobj_clean:
                    plan["cross_case_objectives"] = cobj_clean[:12]
            reason = str(payload.get("reason", "")).strip()
            if reason:
                plan["reason"] = reason
    except Exception:
        pass
    return plan


def _mesh_gate_plan_experiments(
    llm: Any,
    topic: str,
    base_requirement: str,
    baseline_files_text: str,
    code_mod_context: str,
) -> list[dict]:
    """
    Ask an LLM to plan the mesh refinement experiments.
    Returns a list of dicts: [{name, role, description}, ...].
    role is one of: baseline, coarse, refined.
    """
    from pydantic import BaseModel as _BM, Field as _F

    class MeshExp(_BM):
        name: str = _F(description="Short directory name, e.g. coarse, baseline, refined, refined_2")
        role: str = _F(description="One of: coarse, baseline, refined")
        description: str = _F(description="One sentence describing what this experiment does relative to baseline")

    class MeshGatePlan(_BM):
        experiments: list[MeshExp] = _F(description="Ordered list of mesh experiments to run")

    system_prompt = (
        "You are a CFD mesh-independence study planner.\n"
        "Given the study topic, the baseline case files, and any code-mod context, "
        "decide the initial mesh experiments needed for a mesh refinement study.\n"
        "This plan applies to **one** physics-locked case family at a time (transport/viscosity already fixed); "
        "only mesh resolution may change between experiments.\n"
        "Rules:\n"
        "- Always include exactly one 'baseline' experiment (role=baseline) that runs the mesh as-is.\n"
        "- Include one 'coarse' experiment (role=coarse) that coarsens from baseline.\n"
        "- Include at least two 'refined' experiments (role=refined): each refines ~10% near-wall / ~5% away-from-wall "
        "relative to its parent (first refined from baseline, second from first refined, etc.).\n"
        "- Typical plan: [coarse, baseline, refined, refined_2]. This is the default unless the problem warrants more.\n"
        "- Name them: coarse, baseline, refined, refined_2, refined_3, etc.\n"
        "- Order: baseline first, then coarse, then refined levels in ascending order.\n"
        "- Do not include experiments that change physics/BCs/solver — only mesh resolution changes.\n"
    )
    user_prompt = (
        f"Study topic:\n{topic}\n\n"
        f"Base experiment requirement:\n{base_requirement[:8000]}\n\n"
        f"{code_mod_context[:6000]}\n\n"
        f"Baseline case files (0/, constant/, system/ without polyMesh):\n{baseline_files_text[:60000]}\n"
    )

    try:
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM
        resp = llm.invoke([_SM(content=system_prompt), _HM(content=user_prompt)])
        raw_text = getattr(resp, "content", str(resp))
        parsed = json.loads(re.search(r"\{.*\}", raw_text, re.DOTALL).group(0))  # type: ignore[union-attr]
        exps = parsed.get("experiments", [])
        if not exps:
            raise ValueError("empty experiments list")
        return [{"name": e["name"], "role": e["role"], "description": e.get("description", "")} for e in exps]
    except Exception:
        pass

    try:
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM
        structured = llm.with_structured_output(MeshGatePlan)
        plan: MeshGatePlan = structured.invoke([_SM(content=system_prompt), _HM(content=user_prompt)])
        return [{"name": e.name, "role": e.role, "description": e.description} for e in plan.experiments]
    except Exception:
        pass

    return [
        {"name": "baseline", "role": "baseline", "description": "Run baseline mesh as-is."},
        {"name": "coarse", "role": "coarse", "description": "Coarsen baseline mesh."},
        {"name": "refined", "role": "refined", "description": "Refine baseline mesh ~10%/5%."},
        {"name": "refined_2", "role": "refined", "description": "Refine from refined ~10%/5%."},
    ]


def _run_mesh_gate_group_impl(
    run_dir: Path,
    mesh_dir: Path,
    group_id: str,
    base_requirement: str,
    mesh_seed: str,
    topic: str,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
) -> Dict[str, Any]:
    """
    Mesh-gate workflow for one physics group:
      1. LLM plans experiments (coarse / baseline / refined / refined_2 …).
      2. Run baseline via foam_run (seed from mesh_seed).
      3. For each non-baseline experiment: copy baseline case, LLM edits files
         (mesh via --mesh-gate-role enforcement), run with foam reviewer.
      4. LLM PyVista batch loads all experiments, computes QoIs, decides converged level.
    """
    print(f"[ORCH] MESH GATE START group={group_id}")
    analysis_path = mesh_dir / "mesh_analysis.json"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_seed or not Path(mesh_seed).exists():
        print(f"[ORCH] MESH GATE group={group_id}: missing mesh_seed; returning empty selection")
        return {
            "group_id": group_id,
            "selected_level": "",
            "stable_name": "",
            "baseline_case": "",
            "analysis_path": str(analysis_path),
            "levels": [],
            "refined_chain": [],
            "plan": [],
            "requirement_suffix": "",
        }

    # ---- Step 1: LLM plans mesh experiments ----
    _bootstrap_paths(repo_root)
    from cfd_langgraph.llm.factory import create_langchain_llm
    from cfd_langgraph.config import get_settings
    llm = create_langchain_llm(model=get_settings().model, temperature=0.0)

    code_mod_ctx = _collect_code_mod_context(run_dir, state_path)
    baseline_files_text = ""
    if mesh_seed:
        baseline_files_text = _collect_case_files_text(Path(mesh_seed))

    topic_text = (topic or base_requirement or "")[:4000]
    experiments = _mesh_gate_plan_experiments(llm, topic_text, base_requirement, baseline_files_text, code_mod_ctx)
    _write_json(mesh_dir / "mesh_gate_plan.json", experiments)
    print(f"[ORCH] Mesh-gate plan: {[e['name'] for e in experiments]}")
    append_timeline_event(timeline_path, {"stage": "mesh_gate", "event": "plan", "experiments": experiments})
    mesh_metrics = _llm_decide_analysis_metrics(
        repo_root=repo_root,
        topic=topic_text,
        requirement_text=base_requirement,
        code_mod_context=code_mod_ctx,
        max_metrics=8,
    )
    print(f"[ORCH] Mesh-gate analysis metrics (LLM): {mesh_metrics}")
    append_timeline_event(
        timeline_path,
        {"stage": "mesh_gate", "event": "metrics_selected", "metrics": mesh_metrics},
    )

    has_baseline = any(e["role"] == "baseline" for e in experiments)
    if not has_baseline:
        experiments.insert(0, {"name": "baseline", "role": "baseline", "description": "Run baseline mesh as-is."})

    # ---- helpers ----
    def _touch_foam_markers(dirs: List[Path]) -> None:
        for cd in dirs:
            marker = cd / f"{cd.name}.foam"
            if not marker.exists():
                try:
                    marker.touch()
                except Exception:
                    pass

    def _run_pair_analysis(
        case_a: Path, case_b: Path, label: str, metrics: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any], Path]:
        """Run analyze on two cases; return (pct_changes, qoi_parent, qoi_child, output_path)."""
        pair_dirs = [case_a, case_b]
        _touch_foam_markers(pair_dirs)
        pair_out = mesh_dir / f"mesh_analysis_{label}.json"
        _call_stage(
            [
                sys.executable, "scripts/analyze.py",
                "--cases", str(case_a), str(case_b),
                "--metrics", ",".join(metrics),
                "--output", str(pair_out),
                "--qoi-source", "llm_pyvista",
            ],
            f"mesh_gate:analyze_{label}",
            repo_root, env, timeline_path, state_path,
        )
        pair_data = _read_json(pair_out, {})
        mb = pair_data.get("metrics", []) if isinstance(pair_data, dict) else []
        q_a = mb[0].get("qoi", {}) if isinstance(mb, list) and len(mb) > 0 and isinstance(mb[0], dict) else {}
        q_b = mb[1].get("qoi", {}) if isinstance(mb, list) and len(mb) > 1 and isinstance(mb[1], dict) else {}
        common = [
            k for k in q_a
            if k in q_b
            and k not in _MESH_GATE_STABILITY_SKIP_KEYS
            and isinstance(q_a.get(k), (int, float))
            and isinstance(q_b.get(k), (int, float))
        ]
        pct: Dict[str, float] = {}
        for k in common:
            va, vb = float(q_a[k]), float(q_b[k])
            pct[k] = abs(vb - va) / max(abs(va), 1e-12) * 100.0
        return pct, q_a, q_b, pair_out

    def _run_one_mesh_experiment(name: str, role: str, description: str, parent: Path) -> Path:
        exp_dir = mesh_dir / name
        parent_label = parent.name
        level_idx = sum(1 for d in refined_chain if d != baseline_dir)
        exp_requirement = (
            f"MESH-GATE EXPERIMENT: {name} (role={role})\n"
            f"Description: {description}\n\n"
            "MESH CHANGE ONLY — do not change physics, BCs, solver, or transport models.\n"
            "If code-mod / custom models were implemented, preserve them if the experiment requires them.\n\n"
            + base_requirement
        )
        if role == "coarse":
            exp_requirement += (
                "\n\nMesh change: COARSEN the baseline mesh in a controlled way.\n"
                "The runner will programmatically ask an LLM for a coarser blockMeshDict "
                "and write it to system/blockMeshDict.\n"
            )
        elif role == "refined":
            exp_requirement += (
                f"\n\nMesh change: REFINE from parent level '{parent_label}' — apply ~10% near-wall / ~5% away-from-wall "
                f"refinement relative to **that parent mesh only** (level {level_idx + 1} in chain).\n"
                "The runner will programmatically ask an LLM for the refined blockMeshDict "
                "and write it to system/blockMeshDict.\n"
            )
        cmd: List[str] = [
            sys.executable, "scripts/foam_run.py",
            "--requirement", exp_requirement,
            "--output-dir", str(exp_dir),
            "--max-loop", "10",
            "--max-time-limit", "21600",
            "--timeline", str(timeline_path),
            "--base-case-dir", str(parent),
            "--mesh-gate-role", role,
        ]
        _call_stage(cmd, f"mesh_gate:{name}", repo_root, env, timeline_path, state_path)
        return exp_dir

    # ---- Step 2: Run baseline ----
    baseline_dir = mesh_dir / "baseline"
    baseline_cmd: List[str] = [
        sys.executable, "scripts/foam_run.py",
        "--requirement", base_requirement,
        "--output-dir", str(baseline_dir),
        "--max-loop", "10",
        "--max-time-limit", "21600",
        "--timeline", str(timeline_path),
        "--mesh-gate-role", "baseline",
    ]
    if mesh_seed:
        baseline_cmd.extend(["--base-case-dir", mesh_seed])
    _call_stage(baseline_cmd, "mesh_gate:baseline", repo_root, env, timeline_path, state_path)

    all_case_dirs: List[Path] = [baseline_dir]
    refined_chain: List[Path] = [baseline_dir]

    # ---- Step 3: Sequential refine → analyze → decide loop ----
    max_refine_levels = 5
    mesh_pair_metric_attempts = 3
    converged = False
    selected_level = baseline_dir
    stable_name = "baseline"

    refine_failed: bool = False
    refine_failed_level: str = ""
    refine_failed_error: str = ""
    for level in range(1, max_refine_levels + 1):
        parent = refined_chain[-1]
        parent_label = parent.name
        ref_name = "refined" if level == 1 else f"refined_{level}"
        print(f"[ORCH] Mesh-gate: running {ref_name} (parent={parent_label})")

        try:
            ref_dir = _run_one_mesh_experiment(
                ref_name, "refined",
                f"Refine ~10%/5% from {parent_label}.",
                parent,
            )
        except Exception as exc:
            # Mesh-gate refinement failed (commonly: LLM-generated blockMeshDict
            # exceeds topic cell budget, or solver diverges on the new mesh).
            # Per mechanism.md: do NOT abort the whole run. Fall back to the
            # last successful refine level (or baseline) as the selected mesh,
            # log it, and continue to coarse + analyze + downstream experiments.
            refine_failed = True
            refine_failed_level = ref_name
            refine_failed_error = str(exc)[:600]
            print(
                f"[ORCH] Mesh-gate: {ref_name} FAILED ({refine_failed_error}); "
                f"falling back to last successful level: {parent_label}"
            )
            append_timeline_event(
                timeline_path,
                {
                    "stage": "mesh_gate",
                    "event": "refine_level_failed_fallback",
                    "level": ref_name,
                    "parent": parent_label,
                    "error_excerpt": refine_failed_error,
                    "fallback_selected": parent_label,
                },
            )
            break
        all_case_dirs.append(ref_dir)
        refined_chain.append(ref_dir)

        pair_label = f"{parent_label}_vs_{ref_name}"
        final_decision: Optional[Dict[str, Any]] = None
        pct_changes: Dict[str, float] = {}
        q_a: Dict[str, Any] = {}
        q_b: Dict[str, Any] = {}

        for metric_attempt in range(mesh_pair_metric_attempts):
            pct_changes, q_a, q_b, _pair_out = _run_pair_analysis(
                parent, ref_dir, pair_label, mesh_metrics
            )
            print(
                f"[ORCH] Mesh-gate naive QoI % change {parent_label} -> {ref_name} "
                f"(metrics attempt {metric_attempt + 1}/{mesh_pair_metric_attempts}): {pct_changes}"
            )
            decision = _llm_mesh_gate_pair_convergence(
                llm,
                parent_label=parent_label,
                child_label=ref_name,
                q_a=q_a,
                q_b=q_b,
                pct_changes=pct_changes,
                metrics_requested=list(mesh_metrics),
                topic_excerpt=topic_text,
                requirement_excerpt=base_requirement,
                metric_attempt_index=metric_attempt,
                max_metric_attempts=mesh_pair_metric_attempts,
            )
            rel = str(decision.get("qoi_reliability", "") or "").strip().lower()
            retry_metrics = decision.get("recommended_metrics_for_retry") or []
            if (
                rel == "unreliable"
                and retry_metrics
                and metric_attempt < mesh_pair_metric_attempts - 1
            ):
                merged = _merge_mesh_gate_metrics(mesh_metrics, retry_metrics, 10)
                if merged != mesh_metrics:
                    mesh_metrics = merged
                    print(
                        f"[ORCH] Mesh-gate: QoIs marked unreliable; re-analyzing pair with "
                        f"metrics={mesh_metrics}"
                    )
                    append_timeline_event(
                        timeline_path,
                        {
                            "stage": "mesh_gate",
                            "event": "metrics_retry",
                            "parent": parent_label,
                            "child": ref_name,
                            "attempt": metric_attempt + 1,
                            "updated_metrics": list(mesh_metrics),
                            "llm_reason_excerpt": (decision.get("reason") or "")[:800],
                        },
                    )
                    continue
            final_decision = decision
            break

        if final_decision is None:
            final_decision = _heuristic_mesh_gate_pair_fallback(q_a, q_b, pct_changes)

        converged_pair = bool(final_decision.get("converged"))
        numeric_all_5 = bool(pct_changes and all(v <= 5.0 for v in pct_changes.values()))
        print(
            f"[ORCH] Mesh-gate convergence (LLM): converged={converged_pair} "
            f"(naive all≤5%={numeric_all_5}) — {str(final_decision.get('reason', ''))[:220]}"
        )
        append_timeline_event(
            timeline_path,
            {
                "stage": "mesh_gate",
                "event": "pair_comparison",
                "parent": parent_label,
                "child": ref_name,
                "pct_changes": pct_changes,
                "numeric_all_within_5pct": numeric_all_5,
                "llm_converged": converged_pair,
                "llm_reason": final_decision.get("reason", ""),
                "qoi_reliability": final_decision.get("qoi_reliability", ""),
                "decision_source": final_decision.get("source", ""),
            },
        )

        if converged_pair:
            selected_level = parent
            stable_name = parent_label
            converged = True
            print(f"[ORCH] Mesh converged at level: {stable_name} (confirmed by {ref_name})")
            break

    if not converged:
        selected_level = refined_chain[-1]
        stable_name = refined_chain[-1].name
        if refine_failed:
            print(
                f"[ORCH] Mesh-gate: aborted after {refine_failed_level} failure; "
                f"falling back to {stable_name} (last successful level)"
            )
        else:
            print(f"[ORCH] Mesh-gate: not converged after {max_refine_levels} levels; using {stable_name}")

    # ---- Step 4: Run coarse for completeness (best-effort; non-fatal) ----
    try:
        coarse_dir = _run_one_mesh_experiment(
            "coarse", "coarse", "Coarsen baseline mesh.", baseline_dir,
        )
        all_case_dirs.insert(0, coarse_dir)
    except Exception as exc:
        print(f"[ORCH] Mesh-gate: coarse level FAILED (non-fatal, continuing): {exc}")
        append_timeline_event(
            timeline_path,
            {"stage": "mesh_gate", "event": "coarse_failed_skipped",
             "error_excerpt": str(exc)[:600]},
        )

    # ---- Step 5: Final full analysis across all levels (best-effort; non-fatal) ----
    _touch_foam_markers(all_case_dirs)
    try:
        _call_stage(
            [
                sys.executable, "scripts/analyze.py",
                "--cases", *[str(p) for p in all_case_dirs],
                "--metrics", ",".join(mesh_metrics),
                "--output", str(analysis_path),
                "--qoi-source", "llm_pyvista",
            ],
            "mesh_gate:analyze_final",
            repo_root, env, timeline_path, state_path,
        )
    except Exception as exc:
        print(f"[ORCH] Mesh-gate: analyze_final FAILED (non-fatal, continuing): {exc}")
        append_timeline_event(
            timeline_path,
            {"stage": "mesh_gate", "event": "analyze_final_failed_skipped",
             "error_excerpt": str(exc)[:600]},
        )

    selected = {
        "selected_level": str(selected_level),
        "stable_name": stable_name,
        "baseline_case": str(baseline_dir),
        "analysis_path": str(analysis_path),
        "levels": [str(p) for p in all_case_dirs],
        "refined_chain": [str(p) for p in refined_chain],
        "plan": experiments,
        "requirement_suffix": (
            "Use mesh-gate selected setup from the first stabilized mesh level for this physics group; "
            "keep same topology and numerics as in the mesh study."
        ),
        "group_id": group_id,
        "group_label": group_id,
    }
    print(
        f"[ORCH] MESH GATE SELECTED group={group_id} level={selected['selected_level']} ({stable_name})"
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "mesh_gate",
            "event": "selected",
            "group_id": group_id,
            "selected_level": stable_name,
            "selected_mesh_spec_path": str(run_dir / "selected_mesh_spec.json"),
        },
    )
    return selected


def _run_mesh_gate(
    run_dir: Path,
    base_requirement: str,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
    topic: str = "",
) -> Dict[str, Any]:
    """
    Plan mesh refinement **groups** (LLM: partition cases by physics), run one mesh-gate
    study per group, merge into ``selected_mesh_spec.json`` (version 2).
    """
    _bootstrap_paths(repo_root)
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.mesh_gate_groups import (
        normalize_group_plan,
        plan_mesh_refinement_groups_llm,
        resolve_mesh_seed_path,
    )

    topic_use = (topic or base_requirement or "").strip()
    req_path = run_dir / "requirements.json"
    requirements = _read_json(req_path, [])
    if not isinstance(requirements, list):
        requirements = []

    code_mod_ctx = _collect_code_mod_context(run_dir, state_path)
    all_case_ids = [
        str(r.get("case_id"))
        for r in requirements
        if isinstance(r, dict) and str(r.get("case_id", "")).strip()
    ]

    if requirements:
        plan = plan_mesh_refinement_groups_llm(
            model=get_settings().model,
            topic=topic_use,
            requirements=requirements,
            code_mod_context=code_mod_ctx,
        )
        plan = normalize_group_plan(plan, all_case_ids)
        _write_json(run_dir / "mesh_refinement_groups.json", plan.model_dump())
    else:
        from cfd_langgraph.mesh_gate_groups import MeshGateGroupPlan, MeshPhysicsGroup

        plan = MeshGateGroupPlan(
            groups=[
                MeshPhysicsGroup(
                    group_id="default",
                    label="requirements missing — single mesh study",
                    case_ids=[],
                    mesh_study_baseline_requirement=base_requirement,
                    mesh_seed_source="canonical_base_case",
                )
            ],
            reasoning="No requirements.json list; single default mesh gate.",
        )

    if not plan.groups:
        from cfd_langgraph.mesh_gate_groups import MeshGateGroupPlan as MGP, MeshPhysicsGroup as MPG

        plan = MGP(
            groups=[
                MPG(
                    group_id="default",
                    label="empty planner output",
                    case_ids=list(all_case_ids),
                    mesh_study_baseline_requirement=base_requirement,
                    mesh_seed_source="canonical_base_case",
                )
            ],
            reasoning="Planner produced zero groups; using default single mesh gate.",
        )

    mesh_top = run_dir / "mesh_gate"
    mesh_top.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {
        "version": 2,
        "groups": {},
        "case_to_group": {},
        "default_group": "",
        "reasoning": getattr(plan, "reasoning", ""),
    }

    for g in plan.groups:
        gid = g.group_id
        if not merged["default_group"]:
            merged["default_group"] = gid
        mseed = resolve_mesh_seed_path(
            mesh_seed_source=g.mesh_seed_source,
            run_dir=run_dir,
            state_path=state_path,
        )
        gdir = mesh_top / gid
        rec = _run_mesh_gate_group_impl(
            run_dir=run_dir,
            mesh_dir=gdir,
            group_id=gid,
            base_requirement=g.mesh_study_baseline_requirement,
            mesh_seed=mseed,
            topic=topic_use,
            repo_root=repo_root,
            env=env,
            timeline_path=timeline_path,
            state_path=state_path,
        )
        rec["group_label"] = g.label
        merged["groups"][gid] = rec
        for cid in g.case_ids or []:
            merged["case_to_group"][str(cid)] = gid

    for cid in all_case_ids:
        if cid not in merged["case_to_group"] and merged["default_group"]:
            merged["case_to_group"][cid] = merged["default_group"]

    dg = merged["default_group"]
    if dg and isinstance(merged["groups"].get(dg), dict):
        fg = merged["groups"][dg]
        merged["selected_level"] = fg.get("selected_level", "")
        merged["stable_name"] = fg.get("stable_name", "")
        merged["baseline_case"] = fg.get("baseline_case", "")
        merged["analysis_path"] = fg.get("analysis_path", "")
        merged["levels"] = fg.get("levels", [])
        merged["refined_chain"] = fg.get("refined_chain", [])
        merged["plan"] = fg.get("plan", [])
        merged["requirement_suffix"] = fg.get("requirement_suffix", "")

    _write_json(run_dir / "selected_mesh_spec.json", merged)
    append_timeline_event(
        timeline_path,
        {
            "stage": "mesh_gate",
            "event": "merged_selection",
            "version": 2,
            "groups": list((merged.get("groups") or {}).keys()),
            "case_to_group": merged.get("case_to_group", {}),
        },
    )
    return merged


def _su_flag(run_dir: Path) -> List[str]:
    """Return --starter-understanding <path> args if starter_understanding.json exists."""
    p = run_dir / "starter_understanding.json"
    return ["--starter-understanding", str(p)] if p.exists() else []


def _write_mesh_gate_resume(run_dir: Path) -> Optional[Path]:
    """
    After mesh gate selection, write mesh_gate_resume.json summarising per-group
    selected mesh info (blockMeshDict content, cell counts, QoI metrics).
    This is injected into hypothesis and requirements so experiments use the right mesh.
    Returns path to the written file, or None if nothing to write.
    """
    import re as _re

    spec_path = run_dir / "selected_mesh_spec.json"
    if not spec_path.is_file():
        return None
    spec = _read_json(spec_path, {})
    if not isinstance(spec, dict):
        return None

    ver = int(spec.get("version") or 0)
    groups_raw = spec.get("groups") if ver >= 2 else {}
    if not isinstance(groups_raw, dict):
        # v1 compat: wrap default group
        groups_raw = {"default": spec}

    resume: Dict[str, Any] = {
        "version": 1,
        "case_to_group": spec.get("case_to_group", {}),
        "default_group": spec.get("default_group", ""),
        "groups": {},
    }

    for gid, ginfo in groups_raw.items():
        if not isinstance(ginfo, dict):
            continue
        selected_level = str(ginfo.get("selected_level") or "")
        stable_name = str(ginfo.get("stable_name") or "")
        analysis_path = str(ginfo.get("analysis_path") or "")

        # Read blockMeshDict from selected level
        bmd_content = ""
        bmd_path = Path(selected_level) / "system" / "blockMeshDict" if selected_level else None
        if bmd_path and bmd_path.is_file():
            try:
                bmd_content = bmd_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        # Parse cell counts from blockMeshDict
        cell_counts: Dict[str, Any] = {}
        if bmd_content:
            hits = _re.findall(r'hex\s*\([^)]+\)\s*\((\d+)\s+(\d+)\s+(\d+)\)', bmd_content)
            if hits:
                total = sum(int(a) * int(b) * int(c) for a, b, c in hits)
                xs = sorted(set(int(a) for a, b, c in hits))
                ys = sorted(set(int(b) for a, b, c in hits))
                cell_counts = {
                    "total_cells": total,
                    "x_per_block": xs,
                    "y_per_block": ys,
                    "nx_total": sum(xs),
                    "ny_total": sum(ys),
                    "n_blocks": len(hits),
                }

        # Pull key QoI metrics from analysis JSON
        qoi_summary: Dict[str, Any] = {}
        if analysis_path and Path(analysis_path).is_file():
            try:
                analysis = _read_json(Path(analysis_path), {})
                metrics_list = analysis.get("metrics") or []
                # find the entry for the selected level
                for entry in metrics_list:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("case", "")).endswith(stable_name):
                        qoi = entry.get("qoi") or {}
                        # Generic QoI extraction: any scalar / list value the analysis
                        # stage produced. No hardcoded patch names or fields, since
                        # those vary per topic (BFS, channel, jet, mixer, …).
                        for k, v in qoi.items():
                            if isinstance(v, (int, float, str)):
                                qoi_summary[k] = v
                            elif isinstance(v, list) and len(v) <= 32:
                                qoi_summary[k] = v
                        break
            except Exception:
                pass

        # Brief prose summary for LLM context
        summary_lines = [
            f"Physics group: {gid}",
            f"Selected mesh level: {stable_name}  (path: {selected_level})",
        ]
        if cell_counts:
            summary_lines.append(
                f"Cell counts: nx={cell_counts.get('nx_total')} "
                f"ny={cell_counts.get('ny_total')} "
                f"total={cell_counts.get('total_cells')} "
                f"(x blocks: {cell_counts.get('x_per_block')}, "
                f"y blocks: {cell_counts.get('y_per_block')})"
            )
        if qoi_summary:
            summary_lines.append("QoIs from selected mesh:")
            for k, v in qoi_summary.items():
                if isinstance(v, float):
                    summary_lines.append(f"  - {k}: {v:.6g}")
                else:
                    summary_lines.append(f"  - {k}: {v}")
        summary_lines.append(
            "MESH POLICY: All downstream experiments for this physics group MUST use this "
            "mesh as the base. Do NOT specify mesh_cells_x / mesh_cells_y in experiment "
            "parameters — the mesh is authoritative from the mesh-gate study. "
            "If an experiment genuinely requires a different resolution, state it explicitly "
            "in natural language (e.g. 'refine near-wall by 10% from base mesh') and the "
            "mesh-gate LLM will apply the delta."
        )

        resume["groups"][gid] = {
            "group_id": gid,
            "group_label": ginfo.get("group_label", gid),
            "selected_level": selected_level,
            "stable_name": stable_name,
            "blockMeshDict_content": bmd_content,
            "cell_counts": cell_counts,
            "qoi_summary": qoi_summary,
            "summary": "\n".join(summary_lines),
        }

    out_path = run_dir / "mesh_gate_resume.json"
    _write_json(out_path, resume)
    print(f"[ORCH] Mesh-gate resume written: {out_path} ({len(resume['groups'])} group(s))")
    return out_path


def _build_mesh_independence_paper_bundle(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Collect mesh-gate outputs for the paper writer: selected mesh, per-level QoIs,
    analysis narrative, and any PNGs under mesh_gate/. Returns None if no mesh study on disk.
    """
    spec_path = run_dir / "selected_mesh_spec.json"
    if not spec_path.is_file():
        return None
    spec = _read_json(spec_path, {})
    if not isinstance(spec, dict):
        return None
    mesh_dir = run_dir / "mesh_gate"
    ver = int(spec.get("version") or 0)
    per_group_mesh: List[Dict[str, Any]] = []
    if ver >= 2 and isinstance(spec.get("groups"), dict):
        for gid, ginfo in spec["groups"].items():
            if not isinstance(ginfo, dict):
                continue
            per_group_mesh.append(
                {
                    "group_id": gid,
                    "label": ginfo.get("group_label") or ginfo.get("label") or gid,
                    "selected_level": ginfo.get("selected_level", ""),
                    "stable_name": ginfo.get("stable_name", ""),
                    "baseline_case": ginfo.get("baseline_case", ""),
                    "analysis_path": ginfo.get("analysis_path", ""),
                    "levels": ginfo.get("levels", []),
                }
            )

    analysis_path = Path(str(spec.get("analysis_path") or mesh_dir / "mesh_analysis.json"))
    if not analysis_path.is_file() and (mesh_dir / "mesh_analysis.json").is_file():
        analysis_path = mesh_dir / "mesh_analysis.json"
    ana: Dict[str, Any] = _read_json(analysis_path, {}) if analysis_path.is_file() else {}
    metrics_rows: List[Dict[str, Any]] = []
    if isinstance(ana.get("metrics"), list):
        for m in ana["metrics"]:
            if not isinstance(m, dict):
                continue
            case_p = Path(str(m.get("case", "")))
            qoi = m.get("qoi") if isinstance(m.get("qoi"), dict) else {}
            row: Dict[str, Any] = {
                "mesh_folder": case_p.name,
                "case_path": str(m.get("case", "")),
            }
            for k, v in qoi.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
            metrics_rows.append(row)
    analysis_md = str(ana.get("analysis") or "")[:40_000]
    if ver >= 2 and per_group_mesh:
        chunks: List[str] = []
        for row in per_group_mesh:
            ap = Path(str(row.get("analysis_path") or ""))
            if ap.is_file():
                sub = _read_json(ap, {})
                chunks.append(
                    f"### Mesh physics group {row.get('group_id')}\n"
                    + str(sub.get("analysis") or "")[:12_000]
                )
        if chunks:
            analysis_md = "\n\n".join(chunks)[:40_000]

    mesh_figs: List[str] = []
    if mesh_dir.is_dir():
        for p in sorted(mesh_dir.rglob("*.png")):
            if p.is_file():
                mesh_figs.append(str(p.resolve()))
    return {
        "source": "mesh_gate",
        "mesh_gate_version": ver,
        "per_physics_group_mesh": per_group_mesh,
        "case_to_mesh_group": spec.get("case_to_group", {}) if ver >= 2 else {},
        "selected_stable_name": spec.get("stable_name"),
        "selected_level_path": spec.get("selected_level"),
        "baseline_case": spec.get("baseline_case"),
        "mesh_levels": spec.get("levels", []),
        "refined_chain": spec.get("refined_chain", []),
        "mesh_gate_plan": spec.get("plan", []),
        "selection_note": spec.get("requirement_suffix", ""),
        "mesh_analysis_json_path": str(analysis_path) if analysis_path.is_file() else "",
        "metrics_by_mesh_level": metrics_rows,
        "cross_mesh_analysis_excerpt": analysis_md,
        "mesh_figure_paths": mesh_figs[:32],
        "reader_instructions": (
            "Use metrics_by_mesh_level to build a LaTeX table of cell counts and QoIs per mesh level; "
            "when mesh_gate_version>=2, each physics/model group has its own mesh study — use "
            "per_physics_group_mesh and case_to_mesh_group to explain which mesh each experiment used. "
            "State which level was selected per group and cite mesh independence (e.g. percent change in QoIs). "
            "Include mesh_figure_paths in \\includegraphics if non-empty."
        ),
    }


def _resolve_case_path_from_build_result(build_result: Path) -> Optional[Path]:
    data = _read_json(build_result, {})
    if not isinstance(data, dict):
        return None
    files = data.get("files_to_create", [])
    if not isinstance(files, list) or not files:
        return None
    p = str(files[0].get("path", ""))
    m = re.search(r"(.+)/customModels/", p)
    if not m:
        return None
    return Path(m.group(1)).expanduser().resolve()


def _resolve_case_path_from_payload(payload_path: Path) -> Optional[Path]:
    data = _read_json(payload_path, {})
    if not isinstance(data, dict):
        return None
    case_path = data.get("case_path")
    if isinstance(case_path, str) and case_path.strip():
        return Path(case_path).expanduser().resolve()
    return None


def _run_code_mod_validation_gate(
    run_dir: Path,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
    modified_case_dir: Path,
    control_case_dir: Optional[Path],
    *,
    continue_on_mismatch: bool = False,
) -> None:
    """
    Validate code-change in isolation:
    - run modified base case
    - interpret result
    - run control (pre-change) base case if available
    - compare outcomes to isolate code-change-caused failures
    """
    val_dir = run_dir / "code_mod_validation"
    mod_run = val_dir / "modified"
    ctrl_run = val_dir / "control"
    mod_figs = mod_run / "figs"
    ctrl_figs = ctrl_run / "figs"
    mod_dec = mod_run / "decision.json"
    ctrl_dec = ctrl_run / "decision.json"
    mod_req = (
        "Validation run for modified model using base case configuration. "
        "Check numerical stability, physically plausible fields, and expected trend consistency. "
        "CRITICAL: Do NOT rename dictionary keywords in any constant/ dictionaries where the "
        "custom model code expects exact keyword names as defined in the C++ source. "
        "Only fix solver numerics, BCs, or controlDict settings. "
        "Do NOT edit anything under customModels/ (Make/files, Make/options, *.C, *.H) — broken wmake "
        "produces stub libraries and unknown OpenFOAM runtime types / dlopen failures at runtime."
    )
    mod_run.mkdir(parents=True, exist_ok=True)
    mod_run_ok = True
    try:
        _call_stage(
            [
                sys.executable,
                "scripts/foam_run.py",
                "--requirement",
                mod_req,
                "--output-dir",
                str(mod_run),
                "--base-case-dir",
                str(modified_case_dir),
                "--max-loop",
                "10",
                "--max-time-limit",
                "21600",
                "--precheck-max-loop",
                "0",
                "--timeline",
                str(timeline_path),
            ],
            "code_mod:validation:run_modified",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
    except RuntimeError:
        print("[ORCH] WARNING: Code-mod validation run failed; continuing pipeline (model compiled OK, run issues may be fixed downstream).")
        append_timeline_event(timeline_path, {"stage": "code_mod:validation", "event": "run_modified_failed_continuing"})
        mod_run_ok = False

    if mod_run_ok:
        try:
            _call_stage(
                [sys.executable, "scripts/viz.py", "--case", str(mod_run), "--mode", "interpret", "--output", str(mod_figs)],
                "code_mod:validation:viz_modified",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
            _call_stage(
                [
                    sys.executable,
                    "scripts/interpret.py",
                    "--case",
                    str(mod_run),
                    "--figs",
                    str(mod_figs),
                    "--output",
                    str(mod_dec),
                    "--timeline",
                    str(timeline_path),
                ],
                "code_mod:validation:interpret_modified",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
        except RuntimeError:
            print("[ORCH] WARNING: Code-mod validation viz/interpret failed; continuing.")
            append_timeline_event(timeline_path, {"stage": "code_mod:validation", "event": "viz_interpret_failed_continuing"})

    ctrl_status = "not_available"
    if control_case_dir is not None and control_case_dir.exists():
        ctrl_run.mkdir(parents=True, exist_ok=True)
        _call_stage(
            [
                sys.executable,
                "scripts/foam_run.py",
                "--requirement",
                "Validation control run using pre-change base case configuration.",
                "--output-dir",
                str(ctrl_run),
                "--base-case-dir",
                str(control_case_dir),
                "--max-loop",
                "2",
                "--max-time-limit",
                "21600",
                "--precheck-max-loop",
                "0",
                "--timeline",
                str(timeline_path),
            ],
            "code_mod:validation:run_control",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        _call_stage(
            [sys.executable, "scripts/viz.py", "--case", str(ctrl_run), "--mode", "interpret", "--output", str(ctrl_figs)],
            "code_mod:validation:viz_control",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        _call_stage(
            [
                sys.executable,
                "scripts/interpret.py",
                "--case",
                str(ctrl_run),
                "--figs",
                str(ctrl_figs),
                "--output",
                str(ctrl_dec),
                "--timeline",
                str(timeline_path),
            ],
            "code_mod:validation:interpret_control",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        ctrl_status = str(_read_json(ctrl_dec, {}).get("status", "unknown")).upper()

    mod_obj = _read_json(mod_dec, {})
    mod_status = str(mod_obj.get("status", "unknown")).upper()
    likely_code_issue = mod_status in {"RERUN", "REVISE"} and ctrl_status == "PROCEED"
    append_timeline_event(
        timeline_path,
        {
            "stage": "code_mod_validation",
            "event": "comparison",
            "modified_status": mod_status,
            "control_status": ctrl_status,
            "likely_due_to_code_change": likely_code_issue,
        },
    )
    if likely_code_issue:
        reason_excerpt = str(mod_obj.get("reason", "") or "").strip()
        if len(reason_excerpt) > 1200:
            reason_excerpt = reason_excerpt[:1200] + "…"
        explain = (
            "Code-mod validation gate tripped: foam_run reported a completed solver run for the modified case, "
            "but interpret.py returned "
            f"{mod_status} for modified while control is {ctrl_status}.\n"
            "Those are different stages: foam_run checks log errors; the interpreter judges figures/residuals/physics.\n"
            f"Modified decision file: {mod_dec}\n"
            f"Interpreter reason (excerpt): {reason_excerpt or '(empty)'}"
        )
        print("[ORCH] =============================================")
        print("[ORCH] CODE-MOD VALIDATION MISMATCH (read this)")
        for ln in explain.split("\n"):
            print(f"[ORCH] {ln}")
        print("[ORCH] =============================================")
        append_timeline_event(
            timeline_path,
            {
                "stage": "code_mod_validation",
                "event": "mismatch_detail",
                "modified_decision": str(mod_dec),
                "modified_status": mod_status,
                "control_status": ctrl_status,
                "interpreter_reason_excerpt": reason_excerpt[:2000],
            },
        )
        if continue_on_mismatch:
            print(
                "[ORCH] WARNING: Continuing pipeline because "
                "--continue-on-code-mod-validation-mismatch was set (review decision.json above)."
            )
            append_timeline_event(
                timeline_path,
                {"stage": "code_mod_validation", "event": "mismatch_overridden_continue"},
            )
            return
        raise RuntimeError(
            "Code-mod validation: modified interpreter status is "
            f"{mod_status} but control is {ctrl_status}. See orchestrator log above and "
            f"{mod_dec}"
        )


def _run_code_mod_branch(
    run_dir: Path,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    state_path: Path,
    code_mod_payload: str,
    topic: str,
    lit_path: Path,
    base_case_dir: str,
    pdfs: List[str],
    equation_images: List[str],
    starter_dir: Optional[Path] = None,
    *,
    continue_on_code_mod_validation_mismatch: bool = False,
) -> None:
    print("[ORCH] CODE-MOD BRANCH START")
    if code_mod_payload.strip():
        payload = Path(code_mod_payload).expanduser().resolve()
        if not payload.exists():
            raise RuntimeError(f"Code-mod payload not found: {payload}")
    else:
        prepared_path = run_dir / "code_mod_prepared.json"
        # Shared recon cache at study root — reused by every downstream
        # code_mod call in this study (compile-retry loop, rerun attempts, etc).
        shared_recon_cache = run_dir / "discovered_paths.json"
        _call_stage(
            [
                sys.executable,
                "scripts/code_mod_prepare.py",
                "--topic",
                topic,
                "--run-dir",
                str(run_dir),
                "--literature",
                str(lit_path),
                "--base-case-dir",
                base_case_dir,
                "--output",
                str(prepared_path),
                "--recon-cache",
                str(shared_recon_cache),
                "--pdfs",
                *pdfs,
                "--equation-images",
                *equation_images,
                *(["--starter-dir", str(starter_dir)] if starter_dir and starter_dir.is_dir() else []),
                *(["--starter-understanding", str(run_dir / "starter_understanding.json")]
                  if (run_dir / "starter_understanding.json").exists() else []),
            ],
            "code_mod:prepare",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        prepared = _read_json(prepared_path, {})
        payload_raw = prepared.get("payload") if isinstance(prepared, dict) else {}
        if not isinstance(payload_raw, dict):
            raise RuntimeError("code_mod_prepare did not produce payload")
        payload = run_dir / "code_mod_payload.json"
        _write_json(payload, payload_raw)
        append_timeline_event(
            timeline_path,
            {
                "stage": "code_mod",
                "event": "prepare_selected_base_case",
                "meta": prepared.get("meta", {}) if isinstance(prepared, dict) else {},
                "payload_path": str(payload),
            },
        )
    # Keep pre-change snapshot for isolation validation.
    control_snapshot: Optional[Path] = None
    case_before = _resolve_case_path_from_payload(payload)
    if case_before is not None and case_before.exists():
        control_snapshot = run_dir / "code_mod_control_snapshot"
        if control_snapshot.exists():
            shutil.rmtree(control_snapshot)
        shutil.copytree(case_before, control_snapshot)
        append_timeline_event(
            timeline_path,
            {
                "stage": "code_mod",
                "event": "control_snapshot_created",
                "path": str(control_snapshot),
            },
        )
    build_result = run_dir / "code_mod_build_result.json"
    _call_stage(
        [
            sys.executable,
            "scripts/foam_code_builder.py",
            "--payload",
            str(payload),
            "--output",
            str(build_result),
        ],
        "code_mod:builder",
        repo_root,
        env,
        timeline_path,
        state_path,
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "code_mod",
            "event": "build_result_generated",
            "build_result_path": str(build_result),
            "note": "Apply files/patches and compile per build_result.json instructions.",
        },
    )
    apply_result = run_dir / "code_mod_apply_result.json"
    _call_stage(
        [
            sys.executable,
            "scripts/code_mod_apply_compile.py",
            "--build-result",
            str(build_result),
            "--output",
            str(apply_result),
            "--max-compile-attempts",
            "10",
        ],
        "code_mod:apply_compile",
        repo_root,
        env,
        timeline_path,
        state_path,
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "code_mod",
            "event": "apply_compile_done",
            "apply_result_path": str(apply_result),
        },
    )
    modified_case = _resolve_case_path_from_build_result(build_result)
    if modified_case is None:
        raise RuntimeError("Could not resolve modified case path from code_mod_build_result.json")
    _run_code_mod_validation_gate(
        run_dir=run_dir,
        repo_root=repo_root,
        env=env,
        timeline_path=timeline_path,
        state_path=state_path,
        modified_case_dir=modified_case,
        control_case_dir=control_snapshot,
        continue_on_mismatch=continue_on_code_mod_validation_mismatch,
    )
    print(f"[ORCH] CODE-MOD build result saved: {build_result}")


def _llm_canonical_baseline_requirement(
    *,
    topic: str,
    first_requirement: str,
    settings: Any,
) -> str:
    """Distill a SINGLE canonical baseline-case requirement from topic + a representative
    sweep requirement. The result is fed to Foam-Agent (RAG + scaffolding) to produce
    `<run_dir>/canonical_base_case/`.
    """
    from cfd_langgraph.llm.factory import create_langchain_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    sys_prompt = (
        "You are a senior CFD engineer. Produce ONE canonical OpenFOAM baseline case "
        "requirement that captures the topic's geometry, fluid, BCs, and solver class — "
        "NOT a sweep, NOT one of many cases. Just the simplest representative configuration "
        "that demonstrates the physics and runs cleanly. The case will be scaffolded by "
        "Foam-Agent (RAG over OpenFOAM tutorials + LLM scaffolding + precheck/run/fix loop) "
        "and executed once as a smoke test, then used as the seed for the mesh-independence "
        "study.\n\n"
        "Rules:\n"
        "- Use ONLY built-in OpenFOAM models (no custom code, no compiled libraries).\n"
        "- Pick a single representative parameter set, ideally near the middle of the "
        "intended sweep (e.g. mid-Re).\n"
        "- Use a simple structured blockMesh with a small cell count consistent with the "
        "topic's cell budget (default ~5000–9000 cells if the topic does not specify).\n"
        "- For transient physics (oscillation, vortex shedding, frequency extraction, etc.) "
        "use an unsteady solver. Otherwise prefer steady.\n"
        "- Specify: geometry, fluid properties, inlet/outlet/wall BCs, solver, schemes, "
        "initial conditions, end time / write interval, and any sampling probes the topic "
        "requires.\n"
        "- Output is a SINGLE PARAGRAPH or a short structured spec — no JSON, no markdown."
    )
    user_prompt = (
        f"Study topic:\n{topic}\n\n"
        f"Reference (one of the planned sweep requirements; extract the canonical setup "
        f"and DROP sweep-specific values like specific Re/U):\n{first_requirement[:6000]}\n\n"
        f"Produce the canonical baseline requirement now."
    )
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return str(content).strip()


def _synthesize_canonical_base_case(
    *,
    topic: str,
    run_dir: Path,
    state_path: Path,
    repo_root: Path,
    env: Dict[str, str],
    timeline_path: Path,
    settings: Any,
) -> bool:
    """Create `<run_dir>/canonical_base_case/` when no starter baseline exists.

    Per mechanism.md §2 + §3: when --no-starter is given (or starter has no
    baseline case), the orchestrator must still produce a canonical baseline
    so that mesh-gate has a seed. This stage:

      1. Distills a single canonical-baseline requirement from the topic +
         the first synthesised sweep requirement.
      2. Calls Foam-Agent (`scripts/foam_run.py`) which uses RAG over the
         OpenFOAM tutorials + LLM scaffolding + precheck/run/fix loop to
         produce a working case directory.
      3. Verifies `system/controlDict` exists, then records
         `canonical_base_case_dir` in state.json.

    Idempotent: returns True immediately if the case already exists or if
    state.starter_seed_case_dir already points to a real OpenFOAM case.
    Returns False on synthesis failure (caller decides whether to abort).
    """
    canon = run_dir / "canonical_base_case"
    if canon.exists() and (canon / "system" / "controlDict").is_file():
        print(f"[ORCH] baseline_synthesis: canonical_base_case already exists -> {canon}")
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "skipped_already_exists", "path": str(canon)},
        )
        return True

    state = _read_json(state_path, {})
    if isinstance(state, dict):
        seed_dir = str(state.get("starter_seed_case_dir") or "").strip()
        if seed_dir:
            p = Path(seed_dir)
            if p.exists() and (p / "system" / "controlDict").is_file():
                print(f"[ORCH] baseline_synthesis: starter seed already present -> {p}")
                append_timeline_event(
                    timeline_path,
                    {"stage": "baseline_synthesis", "event": "skipped_starter_seed_present",
                     "path": str(p)},
                )
                return True

    req_path = run_dir / "requirements.json"
    requirements = _read_json(req_path, [])
    first_req_text = ""
    if isinstance(requirements, list) and requirements and isinstance(requirements[0], dict):
        first_req_text = str(requirements[0].get("user_requirement_text", "")).strip()
    if not first_req_text:
        first_req_text = f"Study topic: {topic}"

    try:
        baseline_req = _llm_canonical_baseline_requirement(
            topic=topic,
            first_requirement=first_req_text,
            settings=settings,
        )
    except Exception as exc:
        print(f"[ORCH] baseline_synthesis: LLM requirement synthesis failed: {exc}")
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "llm_failed", "error": str(exc)},
        )
        return False

    if not baseline_req:
        print("[ORCH] baseline_synthesis: empty baseline requirement from LLM")
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "empty_requirement"},
        )
        return False

    (run_dir / "canonical_baseline_requirement.txt").write_text(baseline_req, encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {"stage": "baseline_synthesis", "event": "requirement_synthesized",
         "characters": len(baseline_req)},
    )
    print(f"[ORCH] baseline_synthesis: requirement built ({len(baseline_req)} chars) "
          f"-> canonical_baseline_requirement.txt")

    cmd: List[str] = [
        sys.executable, "scripts/foam_run.py",
        "--requirement", baseline_req,
        "--output-dir", str(canon),
        "--max-loop", "5",
        "--max-time-limit", "1800",
        "--precheck-max-loop", "3",
        "--timeline", str(timeline_path),
    ]
    try:
        _call_stage(cmd, "baseline_synthesis", repo_root, env, timeline_path, state_path)
    except Exception as exc:
        print(f"[ORCH] baseline_synthesis: foam_run failed: {exc}")
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "foam_run_failed", "error": str(exc)},
        )
        return False

    if not (canon / "system" / "controlDict").is_file():
        print(f"[ORCH] baseline_synthesis: canonical_base_case missing system/controlDict -> {canon}")
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "missing_controlDict", "path": str(canon)},
        )
        return False

    _update_state(state_path, {"canonical_base_case_dir": str(canon.resolve())})
    append_timeline_event(
        timeline_path,
        {"stage": "baseline_synthesis", "event": "completed",
         "canonical_base_case": str(canon.resolve())},
    )
    print(f"[ORCH] baseline_synthesis: canonical baseline ready -> {canon}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic CFD orchestrator runner with persistent state timeline.")
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--max-reruns", type=int, default=10)
    parser.add_argument("--max-experiments", type=int, default=10)
    parser.add_argument("--open-ended-budget", type=int, default=0,
                        help="Enable closed-loop open-ended discovery with this total budget "
                             "(code_mod=2 units, experiment=1 unit). When >0, replaces the "
                             "normal batch hypothesis sweep with a sequential decision loop.")
    # OED extensions — multi-metric + LLM-as-judge are always-on. The LLM
    # decides at startup what metrics to track based on topic + reference data
    # + baseline; per iteration it judges PROCEED/REVISE/RERUN holistically
    # from the full metric vector. Override with --oed-single-metric only if
    # you specifically need the legacy single-comparator path.
    parser.add_argument("--oed-single-metric", action="store_true",
                        help="Override: force OED to use the legacy single-comparator "
                             "path. Default is multi-metric + LLM-as-judge.")
    parser.add_argument("--oed-diversity-mode", default="off",
                        choices=["off", "hybrid", "aggressive"],
                        help="DEPRECATED, no-op: superseded by the search archive's "
                             "quality-based selection, always on in open_ended_discovery.py. "
                             "Kept parseable for backward compatibility.")
    parser.add_argument("--oed-diversity-far-ratio", type=float, default=0.3,
                        help="DEPRECATED, no-op — see --oed-diversity-mode.")
    parser.add_argument("--oed-saturation-window", type=int, default=None,
                        help="OED: recommend stopping once the search archive's best score "
                             "hasn't improved over this many real evaluations. "
                             "Default: max(3, budget // 4).")
    parser.add_argument("--oed-multi-flow", action="store_true",
                        help="OED Phase 3: validate candidates against multiple reference flows.")
    parser.add_argument("--oed-starter-dirs", nargs="+", default=[],
                        help="OED Phase 3: list of starter directories (one per flow). Implies --oed-multi-flow.")
    parser.add_argument("--oed-metric-aggregator", default="llm_judge",
                        choices=["llm_judge", "weighted_sum", "min_improvement", "pareto_rank"],
                        help="How to reduce metric vector to primary score. Default "
                             "'llm_judge' (LLM looks at vector+baseline+history holistically). "
                             "Fixed aggregators are legacy fallbacks.")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--provider", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--code-mod-payload", type=str, default="")
    parser.add_argument("--disable-mesh-gate", action="store_true")
    parser.add_argument(
        "--no-starter",
        action="store_true",
        help="Skip starter-dependent stages (reference_data_ingest, baseline_setup, "
             "metric_setup) when the topic is structurally independent of the starter "
             "case. Use for fresh studies where the starter geometry/physics doesn't "
             "apply (e.g. running a planar jet study with a periodic-hill starter).",
    )
    parser.add_argument("--paper-template", type=str, default="neurips", choices=["neurips", "icml", "iclr", "nature"])
    parser.add_argument(
        "--skip-reference-verify",
        action="store_true",
        help="Skip Semantic Scholar post-paper reference check if that stage is enabled.",
    )
    parser.add_argument(
        "--reference-verify",
        action="store_true",
        help="Enable post-paper reference verification + optional LLM cleanup (scripts/reference_verify_post.py). Off by default.",
    )
    parser.add_argument(
        "--legacy-paper-pipeline",
        action="store_true",
        help="Use viz.py --mode full + paper_utils + reviewer.py instead of unified paper pipeline (scripts/paper_unified.py).",
    )
    parser.add_argument(
        "--continue-on-code-mod-validation-mismatch",
        action="store_true",
        help=(
            "If code-mod validation: modified interpret.py status is RERUN/REVISE but control is PROCEED, "
            "log a clear warning and continue instead of aborting (default: abort)."
        ),
    )
    parser.add_argument("--base-case-dir", type=str, default="")
    parser.add_argument(
        "--starter-dir",
        type=str,
        default="",
        help="Path to the starter folder (reference data, equations, base case). "
             "Defaults to <repo-root>/starter if it exists.",
    )
    parser.add_argument("--pdfs", nargs="*", default=[])
    parser.add_argument("--equation-images", nargs="*", default=[])
    parser.add_argument("--ask-clarifications", action="store_true", default=True)
    parser.add_argument("--no-ask-clarifications", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        choices=[
            "",
            "literature",
            "benchmark_plan",
            "reference_data_ingest",
            "hypothesis",
            "requirements",
            "baseline_synthesis",
            "code_mod",
            "mesh_gate",
            "mesh_gate_resume",
            "experiments",
            "analysis",
            "analysis_without_viz_full",
            "paper_review",
            "reference_verify",
        ],
    )
    args = parser.parse_args()
    resume_skip_viz_full = args.resume_from == "analysis_without_viz_full"
    # mesh_gate_resume: skip the mesh gate run itself, just write the resume and
    # revise hypothesis/requirements, then continue to experiments.
    resume_skip_mesh_gate_run = args.resume_from == "mesh_gate_resume"
    resume_stage = (
        "analysis" if resume_skip_viz_full
        else "mesh_gate" if resume_skip_mesh_gate_run
        else args.resume_from
    )

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.out_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = run_dir / "timeline.json"
    state_path = run_dir / "state.json"
    os.environ["CFD_ORCH_TIMELINE_PATH"] = str(timeline_path)

    # Propagate provider/model into os.environ immediately so ALL in-process LLM
    # calls (routing, study brief, planning, mesh gate, analysis, etc.) use the
    # same provider as subprocesses. Must happen before any get_settings() call.
    if args.provider.strip():
        os.environ["CFD_SCIENTIST_LLM_PROVIDER"] = args.provider.strip()
        os.environ["FOAMAGENT_MODEL_PROVIDER"] = args.provider.strip()
    if args.model.strip():
        os.environ["CFD_SCIENTIST_MODEL"] = args.model.strip()
        os.environ["FOAMAGENT_MODEL_VERSION"] = args.model.strip()

    env = os.environ.copy()
    env["CFD_TOKEN_LOG_PATH"] = str(run_dir / "llm_token_usage.json")
    mode = _classify_topic(args.topic)
    ask_clarifications = args.ask_clarifications and (not args.no_ask_clarifications)
    verbose = args.verbose and (not args.no_verbose)
    if verbose:
        print("[ORCH] =============================================")
        print("[ORCH] Dynamic CFD orchestrator starting")
        print(f"[ORCH] Topic: {args.topic}")
        print(f"[ORCH] Mode: {mode}")
        print(f"[ORCH] Output dir: {run_dir}")
        print(f"[ORCH] Timeline: {timeline_path}")
        print(
            f"[ORCH] LLM provider/model: "
            f"{env.get('CFD_SCIENTIST_LLM_PROVIDER', '(auto)')} / "
            f"{env.get('CFD_SCIENTIST_MODEL', '(default)')}"
        )
        print("[ORCH] =============================================")
    prior_state = _read_json(state_path, {})
    if not isinstance(prior_state, dict):
        prior_state = {}
    prior_clarifications = prior_state.get("clarifications", {})
    if not isinstance(prior_clarifications, dict):
        prior_clarifications = {}
    _write_json(
        state_path,
        {
            "topic": args.topic,
            "mode": mode,
            "open_discovery": False,
            "status": "running",
            "run_dir": str(run_dir),
            "timeline_path": str(timeline_path),
            "provider": env.get("CFD_SCIENTIST_LLM_PROVIDER", ""),
            "model": env.get("CFD_SCIENTIST_MODEL", ""),
            "resume_from": args.resume_from,
            "clarifications": prior_clarifications,
        },
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "orchestrator",
            "event": "start",
            "topic": args.topic,
            "mode": mode,
        },
    )

    if args.starter_dir:
        starter_dir = Path(args.starter_dir).resolve()
    else:
        starter_dir = (repo_root / "starter").resolve()
    # Under --no-starter, do NOT build starter context. This prevents the
    # starter (e.g. periodic-hill) from leaking into experiment_designer's
    # `starter_cases` input and tainting the generated target_parameters
    # for unrelated studies (e.g. a fresh planar-jet sweep).
    if getattr(args, "no_starter", False):
        starter_context = {"cases": [], "pdfs": [], "images": [], "no_starter": True}
        if verbose:
            print("[ORCH] --no-starter: skipping starter context build")
    else:
        starter_context = _build_starter_context_text(repo_root, starter_dir)
    mode = _llm_classify_topic_mode(
        repo_root=repo_root,
        topic=args.topic,
        starter_context=starter_context if isinstance(starter_context, dict) else {},
    )
    _update_state(state_path, {"mode": mode})
    append_timeline_event(
        timeline_path,
        {
            "stage": "routing",
            "event": "primary_mode_decision",
            "mode": mode,
            "decision_source": "llm_with_heuristic_fallback",
        },
    )
    # When closed-loop open-ended budget is set, the discovery loop owns all code_mods
    # and experiments internally — force mode to standard so the regular code_mod stage
    # doesn't fire redundantly before the loop starts.
    if args.open_ended_budget > 0 and mode == "code_mod":
        mode = "standard"
        append_timeline_event(timeline_path, {
            "stage": "routing",
            "event": "mode_overridden_for_open_ended_discovery",
            "original_mode": "code_mod",
            "overridden_to": "standard",
            "reason": "open_ended_budget set; discovery loop handles code_mods internally",
        })
    if verbose:
        print(f"[ORCH] Routing decision (LLM): mode={mode}")
    open_discovery = _llm_decide_open_discovery(
        repo_root=repo_root,
        topic=args.topic,
        starter_context=starter_context if isinstance(starter_context, dict) else {},
    )
    # When budget is set, always treat as open discovery regardless of LLM decision.
    if args.open_ended_budget > 0:
        open_discovery = True
    _update_state(state_path, {"open_discovery": open_discovery})
    append_timeline_event(
        timeline_path,
        {
            "stage": "routing",
            "event": "open_discovery_decision",
            "open_discovery": open_discovery,
            "decision_source": "llm_with_heuristic_fallback",
            "closed_loop": args.open_ended_budget > 0,
        },
    )
    if verbose and open_discovery:
        mode_label = f"closed-loop (budget={args.open_ended_budget})" if args.open_ended_budget > 0 else "batch"
        print(f"[ORCH] Open discovery: enabled ({mode_label})")
    if not getattr(args, "no_starter", False):
        _write_json(run_dir / "starter_context.json", starter_context)
    append_timeline_event(
        timeline_path,
        {
            "stage": "planning",
            "event": "starter_context_collected",
            "starter_context_path": str(run_dir / "starter_context.json"),
            "starter_case_count": len(starter_context.get("cases", [])) if isinstance(starter_context, dict) else 0,
            "starter_pdf_count": len(starter_context.get("pdfs", [])) if isinstance(starter_context, dict) else 0,
            "starter_image_count": len(starter_context.get("images", [])) if isinstance(starter_context, dict) else 0,
        },
    )

    resume_mode = bool(args.resume_from.strip())
    reused_clarifications = resume_mode and bool(prior_clarifications)
    if reused_clarifications:
        clar_text = "\n".join(f"{k} -> {v}" for k, v in prior_clarifications.items() if v)
        if clar_text:
            args.topic = args.topic + "\n\nUser clarifications:\n" + clar_text
        append_timeline_event(
            timeline_path,
            {
                "stage": "clarification",
                "event": "reused_from_state_on_resume",
                "resume_from": args.resume_from,
                "clarification_count": len(prior_clarifications),
            },
        )
    if ask_clarifications and not reused_clarifications and not getattr(args, "no_starter", False):
        # Run the unified starter-folder LLM understanding once and cache it.
        su_path = run_dir / "starter_understanding.json"
        starter_understanding: Dict[str, Any] = {}
        if su_path.exists():
            starter_understanding = _read_json(su_path, {})
            print("[ORCH] Reusing cached starter_understanding.json")
        else:
            try:
                from starter_understand import understand_starter_folder  # type: ignore
                if starter_dir.is_dir():
                    starter_understanding = understand_starter_folder(starter_dir, args.topic)
                    _write_json(su_path, starter_understanding)
            except Exception as _su_err:
                print(f"[ORCH] warning: starter_understand failed: {_su_err}")

        # Build a clarification-filter context from the LLM understanding.
        fp = starter_understanding.get("flow_parameters", {}) if isinstance(starter_understanding, dict) else {}
        starter_case_context: Dict[str, Any] = {
            "starter_found": bool(starter_understanding.get("base_case_path")),
            "starter_case_dir": str(starter_dir / (starter_understanding.get("base_case_path") or "")),
            "base_case_available": bool(starter_understanding.get("base_case_path")),
            "dimension": fp.get("dimension") if isinstance(fp, dict) else None,
            "reynolds_inferred": fp.get("Re") if isinstance(fp, dict) else None,
        }
        append_timeline_event(
            timeline_path,
            {
                "stage": "clarification",
                "event": "starter_context_scanned",
                "starter_found": starter_case_context.get("starter_found"),
                "dimension": starter_case_context.get("dimension", ""),
                "reynolds_inferred": starter_case_context.get("reynolds_inferred"),
                "source": "llm_starter_understand",
            },
        )
        questions = _needs_clarification(args.topic, mode)
        questions = _filter_clarification_questions_with_context(questions, args.topic, mode, starter_case_context)
        answers = _collect_clarifications_interactive(questions)
        if answers:
            append_timeline_event(
                timeline_path,
                {
                    "stage": "clarification",
                    "event": "answers_collected",
                    "questions": questions,
                    "answers": answers,
                },
            )
            st = _read_json(state_path, {})
            if not isinstance(st, dict):
                st = {}
            st["clarifications"] = answers
            _write_json(state_path, st)
            clar_text = "\n".join(f"{k} -> {v}" for k, v in answers.items() if v)
            if clar_text:
                args.topic = args.topic + "\n\nUser clarifications:\n" + clar_text
    elif ask_clarifications and reused_clarifications and verbose:
        print("[ORCH] Clarification: reusing previously saved answers (resume mode).")

    lit_path = run_dir / "lit.json"
    hyp_path = run_dir / "hypotheses.json"
    req_path = run_dir / "requirements.json"
    plan_path = run_dir / "plan.json"

    stage_order = [
        "literature",
        "benchmark_plan",
        "reference_data_ingest",
        "metric_setup",
        "hypothesis",
        "requirements",
        "baseline_synthesis",
        "code_mod",
        "mesh_gate",
        "experiments",
        "analysis",
        "paper_review",
    ]
    start_index = stage_order.index(resume_stage) if resume_stage in stage_order else 0

    lit_topic = args.topic
    starter_brief_path = run_dir / "starter_study_brief.json"
    study_understanding_pre_lit_path = run_dir / "study_understanding_pre_lit.txt"
    plan_resume_mode = bool(args.resume_from.strip())
    if plan_resume_mode and starter_brief_path.exists():
        starter_brief = _read_json(starter_brief_path, {})
        if isinstance(starter_brief, dict):
            lit_query = str(starter_brief.get("literature_query_topic", "")).strip()
            if lit_query:
                lit_topic = lit_query
            append_timeline_event(
                timeline_path,
                {
                    "stage": "planning",
                    "event": "starter_brief_reused_from_disk",
                    "starter_brief_path": str(starter_brief_path),
                    "literature_query_topic": lit_topic,
                },
            )
    else:
        try:
            starter_brief = _llm_starter_study_brief(repo_root=repo_root, topic=args.topic, starter_context=starter_context)
            _write_json(starter_brief_path, starter_brief)
            lines = starter_brief.get("study_brief_lines", [])
            if isinstance(lines, list):
                study_understanding_pre_lit_path.write_text("\n".join(str(x) for x in lines[:20]), encoding="utf-8")
            lit_query = str(starter_brief.get("literature_query_topic", "")).strip()
            if lit_query:
                lit_topic = lit_query
            append_timeline_event(
                timeline_path,
                {
                    "stage": "planning",
                    "event": "starter_brief_created_pre_literature",
                    "starter_brief_path": str(starter_brief_path),
                    "study_understanding_path": str(study_understanding_pre_lit_path),
                    "literature_query_topic": lit_topic,
                },
            )
        except Exception as exc:
            append_timeline_event(
                timeline_path,
                {
                    "stage": "planning",
                    "event": "starter_brief_failed_using_raw_topic",
                    "error": str(exc),
                },
            )

    literature_completed = False
    if "literature" in stage_order and start_index <= stage_order.index("literature"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="literature",
            phase="starting",
            index=1,
            total=len(stage_order),
            next_stage="benchmark_plan",
        )
        try:
            _call_stage(
                [
                    sys.executable,
                    "scripts/lit.py",
                    "--topic",
                    lit_topic,
                    "--limit",
                    str(args.max_papers),
                    "--output",
                    str(lit_path),
                    "--timeline",
                    str(timeline_path),
                ],
                "literature",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
            literature_completed = True
        except RuntimeError:
            print("[ORCH] WARNING: Literature stage failed (API unreachable?); continuing without literature.")
            append_timeline_event(timeline_path, {"stage": "literature", "event": "failed_continuing_without"})
            if not lit_path.exists():
                lit_path.write_text("[]", encoding="utf-8")
            literature_completed = True
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="literature",
            phase="completed",
            index=1,
            total=len(stage_order),
            next_stage="benchmark_plan",
            details={"output": str(lit_path)},
        )
        _checkpoint(state_path, run_dir, "literature_done", {"lit_path": str(lit_path)})
    else:
        append_timeline_event(timeline_path, {"stage": "literature", "event": "skipped_due_to_resume"})
        if lit_path.exists():
            literature_completed = True
    lit_records = _read_json(lit_path, [])
    if not isinstance(lit_records, list):
        lit_records = []
    benchmark_path = run_dir / "benchmark_data.json"
    reference_manifest_path = run_dir / "reference_data_manifest.json"
    # LLM-based scientist understanding + module planning, after literature is available.
    llm_plan_path = run_dir / "llm_plan.json"
    llm_understanding_path = run_dir / "study_understanding.txt"
    try:
        llm_plan: Dict[str, Any]
        reused_plan = False
        if plan_resume_mode and llm_plan_path.exists():
            loaded = _read_json(llm_plan_path, {})
            if isinstance(loaded, dict) and isinstance(loaded.get("module_plan"), list):
                llm_plan = loaded
                reused_plan = True
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "planning",
                        "event": "llm_plan_reused_from_disk",
                        "llm_plan_path": str(llm_plan_path),
                    },
                )
            else:
                llm_plan = _llm_infer_understanding_and_plan(
                    repo_root=repo_root,
                    topic=args.topic,
                    starter_context=starter_context,
                    literature_records=lit_records if isinstance(lit_records, list) else [],
                    mode=mode,
                )
                llm_plan = _validate_and_revise_module_plan(repo_root=repo_root, mode=mode, plan_payload=llm_plan)
                _write_json(llm_plan_path, llm_plan)
        else:
            llm_plan = _llm_infer_understanding_and_plan(
                repo_root=repo_root,
                topic=args.topic,
                starter_context=starter_context,
                literature_records=lit_records if isinstance(lit_records, list) else [],
                mode=mode,
            )
            llm_plan = _validate_and_revise_module_plan(repo_root=repo_root, mode=mode, plan_payload=llm_plan)
            _write_json(llm_plan_path, llm_plan)
        lines = llm_plan.get("study_understanding", [])
        if isinstance(lines, list):
            txt = "\n".join(str(x) for x in lines[:20])
            llm_understanding_path.write_text(txt, encoding="utf-8")
        if verbose:
            print("[ORCH] LLM STUDY UNDERSTANDING (up to 20 lines)")
            if isinstance(lines, list) and lines:
                for idx, line in enumerate(lines[:20], 1):
                    print(f"[ORCH]   {idx:>2}. {line}")
            else:
                print("[ORCH]   (no study_understanding lines returned)")
            print("[ORCH] LLM MODULE PLAN (proposed)")
            mp = llm_plan.get("module_plan", [])
            if isinstance(mp, list) and mp:
                for idx, item in enumerate(mp, 1):
                    if not isinstance(item, dict):
                        continue
                    mid = str(item.get("module", ""))
                    why = str(item.get("why", ""))
                    print(f"[ORCH]   {idx:>2}. {mid} :: {why}")
            else:
                print("[ORCH]   (no module_plan returned)")
        stage_order = _module_plan_to_stage_order(mode, llm_plan.get("module_plan", []))
        if literature_completed and "literature" in stage_order:
            stage_order = [s for s in stage_order if s != "literature"]
        if verbose:
            print("[ORCH] VALIDATED STAGE ORDER")
            for idx, st in enumerate(stage_order, 1):
                print(f"[ORCH]   {idx:>2}. {_stage_display_name(st)}")
        append_timeline_event(
            timeline_path,
            {
                "stage": "planning",
                "event": "llm_plan_created",
                "llm_plan_path": str(llm_plan_path),
                "understanding_path": str(llm_understanding_path),
                "stage_order": stage_order,
                "reused_from_disk": reused_plan,
            },
        )
    except Exception as exc:
        append_timeline_event(
            timeline_path,
            {
                "stage": "planning",
                "event": "llm_plan_failed_fallback_default",
                "error": str(exc),
            },
        )
    if getattr(args, "reference_verify", False):
        stage_order.append("reference_verify")
    start_index = stage_order.index(resume_stage) if resume_stage in stage_order else 0
    execution_plan = _build_execution_plan(
        stage_order=stage_order,
        start_index=start_index,
        mode=mode,
        disable_mesh_gate=args.disable_mesh_gate,
        no_starter=getattr(args, "no_starter", False),
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "orchestrator",
            "event": "routing",
            "mode": mode,
            "resume_from": args.resume_from or "start",
            "start_index": start_index,
            "stage_order": stage_order,
            "mesh_gate_enabled": not args.disable_mesh_gate,
            "max_experiments": args.max_experiments,
            "max_reruns": args.max_reruns,
            "max_papers": args.max_papers,
            "execution_plan": execution_plan,
        },
    )
    if verbose:
        print("[ORCH] EXECUTION PLAN")
        for p in execution_plan:
            mark = "RUN" if p.get("will_run") else "SKIP"
            stage_name = _stage_display_name(str(p.get("stage")))
            print(f"[ORCH]   {p.get('index'):>2}. {stage_name} -> {mark} ({p.get('reason')})")

    benchmark_enabled = False
    if "benchmark_plan" in stage_order and start_index <= stage_order.index("benchmark_plan"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="benchmark_plan",
            phase="starting",
            index=2,
            total=len(stage_order),
            next_stage="hypothesis",
        )
        _call_stage(
            [
                sys.executable,
                "scripts/benchmark_data_prepare.py",
                "--topic",
                args.topic,
                "--literature",
                str(lit_path),
                "--output",
                str(benchmark_path),
            ],
            "benchmark_plan",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="benchmark_plan",
            phase="completed",
            index=2,
            total=len(stage_order),
            next_stage="hypothesis",
            details={"output": str(benchmark_path)},
        )
        _checkpoint(state_path, run_dir, "benchmark_plan_done", {"benchmark_path": str(benchmark_path)})
        bench = _read_json(benchmark_path, {})
        benchmark_enabled = bool(isinstance(bench, dict) and bench.get("benchmark_mode_enabled", False))
    elif "benchmark_plan" in stage_order:
        append_timeline_event(timeline_path, {"stage": "benchmark_plan", "event": "skipped_due_to_resume"})
    else:
        append_timeline_event(timeline_path, {"stage": "benchmark_plan", "event": "skipped_not_in_llm_plan"})

    if "reference_data_ingest" in stage_order and start_index <= stage_order.index("reference_data_ingest") and not getattr(args, "no_starter", False):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="reference_data_ingest",
            phase="starting",
            index=3,
            total=len(stage_order),
            next_stage="hypothesis",
        )
        # Pass --starter-dir only when the folder actually exists; the script
        # handles a missing/empty dir by writing an empty manifest and exiting cleanly.
        ref_ingest_cmd = [
            sys.executable,
            "scripts/reference_data_ingest.py",
            "--topic",
            args.topic,
            "--output",
            str(reference_manifest_path),
            "--timeline",
            str(timeline_path),
        ]
        if starter_dir.is_dir():
            ref_ingest_cmd += ["--starter-dir", str(starter_dir)]
        if (run_dir / "starter_understanding.json").exists():
            ref_ingest_cmd += ["--starter-understanding", str(run_dir / "starter_understanding.json")]
        _call_stage(
            ref_ingest_cmd,
            "reference_data_ingest",
            repo_root,
            env,
            timeline_path,
            state_path,
        )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="reference_data_ingest",
            phase="completed",
            index=3,
            total=len(stage_order),
            next_stage="hypothesis",
            details={"output": str(reference_manifest_path)},
        )
        _checkpoint(
            state_path,
            run_dir,
            "reference_data_ingest_done",
            {"reference_manifest_path": str(reference_manifest_path)},
        )
    elif "reference_data_ingest" in stage_order:
        append_timeline_event(timeline_path, {"stage": "reference_data_ingest", "event": "skipped_due_to_resume"})
    else:
        append_timeline_event(timeline_path, {"stage": "reference_data_ingest", "event": "skipped_not_in_llm_plan"})

    # ---- baseline_setup stage --------------------------------------------
    # Generic across CFD studies: parse topic for "compare against baseline X",
    # detect/generate the baseline run, write run_dir/baseline_metrics.json.
    # Downstream stages (OED planner / interpreter) read that file to gate
    # PROCEED on `variant < baseline`.
    baseline_metrics_path = run_dir / "baseline_metrics.json"
    if "baseline_setup" in stage_order and start_index <= stage_order.index("baseline_setup") and not getattr(args, "no_starter", False):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="baseline_setup",
            phase="starting",
            index=stage_order.index("baseline_setup") + 1,
            total=len(stage_order),
            next_stage="hypothesis",
        )
        objective_contract_path = run_dir / "open_ended_discovery" / "objective_contract.json"
        baseline_cmd = [
            sys.executable,
            "scripts/baseline_setup.py",
            "--run-dir", str(run_dir),
            "--topic", args.topic,
            "--output", str(baseline_metrics_path),
            "--timeout", "1800",
        ]
        if starter_dir.is_dir():
            baseline_cmd += ["--starter-dir", str(starter_dir)]
        if reference_manifest_path.is_file():
            baseline_cmd += ["--reference-data-manifest", str(reference_manifest_path)]
        if objective_contract_path.is_file():
            baseline_cmd += ["--objective-contract", str(objective_contract_path)]
        # baseline_setup is best-effort. Failure here MUST NOT block the rest
        # of the pipeline — the OED loop still works without baseline gating
        # (it just can't auto-judge variant-vs-baseline).
        try:
            _call_stage(
                baseline_cmd,
                "baseline_setup",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
        except SystemExit:
            append_timeline_event(timeline_path, {
                "stage": "baseline_setup",
                "event": "failed_non_fatal",
                "note": "baseline_setup raised SystemExit; OED will continue without baseline gating",
            })
        except Exception as exc:
            append_timeline_event(timeline_path, {
                "stage": "baseline_setup",
                "event": "failed_non_fatal",
                "note": f"baseline_setup raised: {exc}",
            })
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="baseline_setup",
            phase="completed",
            index=stage_order.index("baseline_setup") + 1,
            total=len(stage_order),
            next_stage="hypothesis",
            details={"output": str(baseline_metrics_path)},
        )
        _checkpoint(
            state_path,
            run_dir,
            "baseline_setup_done",
            {"baseline_metrics_path": str(baseline_metrics_path)},
        )
    elif "baseline_setup" in stage_order:
        append_timeline_event(timeline_path, {"stage": "baseline_setup", "event": "skipped_due_to_resume"})

    # ---- metric_setup stage --------------------------------------------------
    # Unified metric proposer + comparator author + verifier. Runs once after
    # baseline_setup; downstream OED loop reads `metric_specs.json` and skips
    # its own metric setup when this file is populated.
    metric_specs_path = run_dir / "metric_specs.json"
    comparator_out_dir = run_dir / "comparators"
    if "metric_setup" in stage_order and start_index <= stage_order.index("metric_setup") and not getattr(args, "no_starter", False):
        # Need a real baseline case dir to do anything useful.
        baseline_metrics_obj = _read_json(baseline_metrics_path, {}) if baseline_metrics_path.is_file() else {}
        bcd = ""
        if isinstance(baseline_metrics_obj, dict):
            bcd = str(baseline_metrics_obj.get("baseline_case_dir", "") or "")
        if bcd and Path(bcd).is_dir():
            _append_stage_pointer(
                state_path=state_path,
                timeline_path=timeline_path,
                stage="metric_setup",
                phase="starting",
                index=stage_order.index("metric_setup") + 1,
                total=len(stage_order),
                next_stage="hypothesis",
            )
            ms_cmd = [
                sys.executable,
                "scripts/metric_setup.py",
                "--run-dir", str(run_dir),
                "--topic", args.topic,
                "--baseline-case-dir", bcd,
                "--baseline-metrics", str(baseline_metrics_path),
                "--output", str(metric_specs_path),
                "--comparator-out", str(comparator_out_dir),
                "--timeline", str(timeline_path),
            ]
            if starter_dir.is_dir():
                ms_cmd += ["--starter-dir", str(starter_dir)]
            if reference_manifest_path.is_file():
                ms_cmd += ["--reference-data-manifest", str(reference_manifest_path)]
            try:
                _call_stage(ms_cmd, "metric_setup", repo_root, env, timeline_path, state_path)
            except SystemExit:
                append_timeline_event(timeline_path, {
                    "stage": "metric_setup",
                    "event": "failed_non_fatal",
                    "note": "metric_setup raised SystemExit; OED falls back to legacy metric path",
                })
            except Exception as exc:
                append_timeline_event(timeline_path, {
                    "stage": "metric_setup",
                    "event": "failed_non_fatal",
                    "note": f"metric_setup raised: {exc}",
                })
            _append_stage_pointer(
                state_path=state_path,
                timeline_path=timeline_path,
                stage="metric_setup",
                phase="completed",
                index=stage_order.index("metric_setup") + 1,
                total=len(stage_order),
                next_stage="hypothesis",
                details={"output": str(metric_specs_path)},
            )
            _checkpoint(
                state_path,
                run_dir,
                "metric_setup_done",
                {"metric_specs_path": str(metric_specs_path)},
            )
        else:
            append_timeline_event(timeline_path, {
                "stage": "metric_setup",
                "event": "skipped_no_baseline_case",
                "note": "baseline_setup produced no usable baseline_case_dir",
            })
    elif "metric_setup" in stage_order:
        append_timeline_event(timeline_path, {"stage": "metric_setup", "event": "skipped_due_to_resume"})

    plan = _build_initial_plan(args.topic, mode, lit_records, run_dir)
    _write_json(plan_path, plan)
    if benchmark_enabled:
        _record_plan_update(
            plan_path,
            "Benchmark/experimental comparison mode enabled from literature evidence.",
            {"benchmark_data_path": str(benchmark_path)},
        )
    append_timeline_event(
        timeline_path,
        {
            "stage": "planning",
            "event": "initial_plan_created",
            "plan_path": str(plan_path),
            "actions": plan.get("actions", []),
        },
    )
    if "hypothesis" in stage_order and start_index <= stage_order.index("hypothesis"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="hypothesis",
            phase="starting",
            index=3,
            total=len(stage_order),
            next_stage="requirements",
        )
        if open_discovery and args.open_ended_budget > 0:
            # === CLOSED-LOOP OPEN-ENDED DISCOVERY ===
            # Delegates entirely to open_ended_discovery.py which runs one action at a
            # time, feeds results back into the decision LLM, and repeats until budget.
            print(f"[ORCH] Open-ended discovery: closed-loop mode, budget={args.open_ended_budget}")
            append_timeline_event(timeline_path, {
                "stage": "hypothesis",
                "event": "open_ended_discovery_closed_loop_start",
                "budget": args.open_ended_budget,
            })
            oed_cmd = [
                sys.executable, "scripts/open_ended_discovery.py",
                "--run-dir", str(run_dir),
                "--topic", args.topic,
                "--budget", str(args.open_ended_budget),
                "--timeline", str(timeline_path),
                "--starter-dir", str(starter_dir),
            ]
            if (run_dir / "starter_understanding.json").exists():
                oed_cmd += ["--starter-understanding", str(run_dir / "starter_understanding.json")]
            if lit_path.is_file():
                oed_cmd += ["--literature", str(lit_path)]
            if args.base_case_dir.strip():
                oed_cmd += ["--base-case-dir", args.base_case_dir]
            if baseline_metrics_path.is_file():
                oed_cmd += ["--baseline-metrics", str(baseline_metrics_path)]
            # OED extensions pass-through. multi-metric + LLM-judge default ON.
            if getattr(args, "oed_single_metric", False):
                oed_cmd += ["--single-metric"]
            if getattr(args, "oed_diversity_mode", "off") and args.oed_diversity_mode != "off":
                oed_cmd += ["--diversity-mode", str(args.oed_diversity_mode),
                            "--diversity-far-ratio", str(args.oed_diversity_far_ratio)]
            if getattr(args, "oed_saturation_window", None) is not None:
                oed_cmd += ["--saturation-window", str(args.oed_saturation_window)]
            if getattr(args, "oed_multi_flow", False):
                oed_cmd += ["--multi-flow"]
            if getattr(args, "oed_starter_dirs", None):
                oed_cmd += ["--starter-dirs"] + list(args.oed_starter_dirs)
            if getattr(args, "oed_metric_aggregator", "llm_judge") and args.oed_metric_aggregator != "llm_judge":
                oed_cmd += ["--metric-aggregator", str(args.oed_metric_aggregator)]
            _call_stage(oed_cmd, "open_ended_discovery", repo_root, env, timeline_path, state_path)
            # Collect all PROCEED case dirs from the discovery summary for downstream analysis
            oed_summary = _read_json(run_dir / "open_ended_discovery" / "summary.json", {})
            oed_history = oed_summary.get("history", []) if isinstance(oed_summary, dict) else []
            proceed_cases = [
                h.get("case_dir", "") for h in oed_history
                if isinstance(h, dict) and h.get("status") == "PROCEED" and h.get("case_dir")
            ]
            best_case_dir = str(oed_summary.get("best_case_dir", "") or "") if isinstance(oed_summary, dict) else ""
            best_score = oed_summary.get("best_score") if isinstance(oed_summary, dict) else None
            has_valid_open_discovery_winner = bool(best_case_dir) and best_score is not None and len(proceed_cases) > 0
            # Write a stub requirements.json and hypotheses.json so downstream stages
            # (analysis, paper) have something to work with.
            stub_hyps = [
                {"hypothesis_id": f"oed_{i:03d}", "experiment_id": f"oed_{i:03d}",
                 "description": h.get("model_description", ""), "valid": True,
                 "idea_experiment": {"parameters": h.get("parameters", {})}}
                for i, h in enumerate(oed_history, 1)
            ]
            _write_json(hyp_path, stub_hyps)
            stub_reqs = [
                {"case_id": f"oed_{i:03d}", "experiment_id": f"oed_{i:03d}",
                 "user_requirement_text": h.get("model_description", ""),
                 "description": h.get("model_description", "")}
                for i, h in enumerate(oed_history, 1)
            ]
            _write_json(req_path, stub_reqs)
            append_timeline_event(timeline_path, {
                "stage": "hypothesis",
                "event": "open_ended_discovery_closed_loop_done",
                "budget_used": oed_summary.get("budget_used", 0),
                "proceed_count": len(proceed_cases),
                "has_valid_winner": has_valid_open_discovery_winner,
            })
            if not has_valid_open_discovery_winner:
                # Graceful degradation (generic): even without a formally-
                # "valid winning model" (which requires both PROCEED cases AND a
                # scored best_case_dir), if ANY iteration completed a compiled-
                # library experiment at all, proceed to analysis with whatever
                # data we have. Analysis and paper stages should describe what
                # was tried and what the metrics look like, rather than abort.
                # The previous behaviour — abort with no artifacts — was the
                # worst outcome for an already-expensive run.
                any_experiment_ran = any(
                    isinstance(h, dict)
                    and h.get("action_type") in ("code_mod", "experiment")
                    and (h.get("case_dir") or h.get("compiled_model_name"))
                    for h in oed_history
                )
                if any_experiment_ran:
                    append_timeline_event(timeline_path, {
                        "stage": "hypothesis",
                        "event": "open_ended_discovery_proceeding_without_formal_winner",
                        "budget_used": oed_summary.get("budget_used", 0),
                        "proceed_count": len(proceed_cases),
                        "best_case_dir": best_case_dir,
                        "note": ("No formally-valid PROCEED winner, but at least one "
                                 "experiment completed; descending into analysis with "
                                 "available data."),
                    })
                    _checkpoint(state_path, run_dir, "open_ended_discovery_done_no_formal_winner",
                                {"proceed_cases": proceed_cases,
                                 "budget_used": oed_summary.get("budget_used", 0),
                                 "best_case_dir": best_case_dir,
                                 "best_score": best_score})
                    print("[ORCH] Open-ended discovery has no formal PROCEED winner but "
                          f"{sum(1 for h in oed_history if isinstance(h,dict) and h.get('action_type') in ('code_mod','experiment'))} "
                          "experiment iterations completed. Continuing to analysis.")
                else:
                    append_timeline_event(timeline_path, {
                        "stage": "hypothesis",
                        "event": "open_ended_discovery_aborted_no_experiments_ran",
                        "budget_used": oed_summary.get("budget_used", 0),
                        "proceed_count": len(proceed_cases),
                        "best_case_dir": best_case_dir,
                    })
                    _checkpoint(
                        state_path,
                        run_dir,
                        "open_ended_discovery_no_experiments_ran",
                        {
                            "proceed_cases": proceed_cases,
                            "budget_used": oed_summary.get("budget_used", 0),
                            "best_case_dir": best_case_dir,
                            "best_score": best_score,
                        },
                    )
                    raise RuntimeError(
                        "Open-ended discovery ended without any experiment completing; "
                        "aborting downstream analysis (nothing to analyse)."
                    )
            # Skip remaining hypothesis/requirements/code_mod/experiments stages —
            # the discovery loop already ran everything.
            _checkpoint(state_path, run_dir, "open_ended_discovery_done",
                        {"proceed_cases": proceed_cases, "budget_used": oed_summary.get("budget_used", 0)})
        elif open_discovery:
            lit_records_for_discovery = _read_json(lit_path, [])
            if not isinstance(lit_records_for_discovery, list):
                lit_records_for_discovery = []
            discovery_hyp, discovery_policy = _generate_open_discovery_hypotheses(
                repo_root=repo_root,
                topic=args.topic,
                starter_context=starter_context if isinstance(starter_context, dict) else {},
                literature_records=lit_records_for_discovery,
                run_dir=run_dir,
                max_experiments=args.max_experiments,
                improvement_threshold=0.10,
            )
            _write_json(hyp_path, discovery_hyp)
            _write_json(run_dir / "discovery_policy.json", discovery_policy)
            append_timeline_event(
                timeline_path,
                {
                    "stage": "hypothesis",
                    "event": "open_discovery_candidates_generated",
                    "candidate_count": len(discovery_hyp),
                    "policy_path": str(run_dir / "discovery_policy.json"),
                },
            )
        else:
            _call_stage(
                [
                    sys.executable,
                    "scripts/hypothesis.py",
                    "--literature",
                    str(lit_path),
                    "--topic",
                    args.topic,
                    "--output",
                    str(hyp_path),
                    "--timeline",
                    str(timeline_path),
                    "--mode",
                    str(mode),
                    *_su_flag(run_dir),
                ],
                "hypothesis",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="hypothesis",
            phase="completed",
            index=3,
            total=len(stage_order),
            next_stage="requirements",
            details={"output": str(hyp_path)},
        )
        _checkpoint(state_path, run_dir, "hypothesis_done", {"hyp_path": str(hyp_path)})
    elif "hypothesis" in stage_order:
        append_timeline_event(timeline_path, {"stage": "hypothesis", "event": "skipped_due_to_resume"})
    else:
        append_timeline_event(timeline_path, {"stage": "hypothesis", "event": "skipped_not_in_llm_plan"})
    if "requirements" in stage_order and start_index <= stage_order.index("requirements"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="requirements",
            phase="starting",
            index=4,
            total=len(stage_order),
            next_stage="code_mod" if mode == "code_mod" else "mesh_gate",
        )
        _call_stage(
                    [
                        sys.executable,
                        "scripts/requirements.py",
                        "--hypotheses",
                        str(hyp_path),
                        "--topic",
                        args.topic,
                        "--output",
                        str(req_path),
                        "--timeline",
                        str(timeline_path),
                        *_su_flag(run_dir),
                    ],
                    "requirements",
                    repo_root,
                    env,
                    timeline_path,
                    state_path,
                )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="requirements",
            phase="completed",
            index=4,
            total=len(stage_order),
            next_stage="code_mod" if mode == "code_mod" else "mesh_gate",
            details={"output": str(req_path)},
        )
        _checkpoint(state_path, run_dir, "requirements_done", {"req_path": str(req_path)})
    elif "requirements" in stage_order:
        append_timeline_event(timeline_path, {"stage": "requirements", "event": "skipped_due_to_resume"})
    else:
        append_timeline_event(timeline_path, {"stage": "requirements", "event": "skipped_not_in_llm_plan"})

    # Per mechanism.md §2 + §3: synthesise a canonical baseline case when no
    # starter baseline exists, so mesh-gate (and code_mod, for code_mod runs)
    # has a real seed to start from. Idempotent: skips if a starter case is
    # already present or canonical_base_case already exists. Skipped entirely
    # when the user disables mesh-gate (no seed needed in that case).
    if (
        "baseline_synthesis" in stage_order
        and start_index <= stage_order.index("baseline_synthesis")
        and not args.disable_mesh_gate
    ):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="baseline_synthesis",
            phase="starting",
            index=stage_order.index("baseline_synthesis") + 1,
            total=len(stage_order),
            next_stage="code_mod" if mode == "code_mod" else "mesh_gate",
        )
        from cfd_langgraph.config import get_settings as _get_settings_for_baseline
        try:
            _settings_for_baseline = _get_settings_for_baseline()
        except Exception:
            _settings_for_baseline = None
        ok_baseline = False
        if _settings_for_baseline is not None:
            ok_baseline = _synthesize_canonical_base_case(
                topic=args.topic,
                run_dir=run_dir,
                state_path=state_path,
                repo_root=repo_root,
                env=env,
                timeline_path=timeline_path,
                settings=_settings_for_baseline,
            )
        else:
            print("[ORCH] baseline_synthesis: settings unavailable; skipping")
            append_timeline_event(
                timeline_path,
                {"stage": "baseline_synthesis", "event": "settings_unavailable"},
            )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="baseline_synthesis",
            phase="completed" if ok_baseline else "failed",
            index=stage_order.index("baseline_synthesis") + 1,
            total=len(stage_order),
            next_stage="code_mod" if mode == "code_mod" else "mesh_gate",
            details={"canonical_base_case": str((run_dir / "canonical_base_case").resolve())},
        )
        if ok_baseline:
            _checkpoint(
                state_path, run_dir, "baseline_synthesis_done",
                {"canonical_base_case": str((run_dir / "canonical_base_case").resolve())},
            )
        else:
            append_timeline_event(
                timeline_path,
                {"stage": "baseline_synthesis", "event": "did_not_produce_seed",
                 "consequence": "mesh_gate may stub out — investigate before re-run"},
            )
    elif "baseline_synthesis" in stage_order:
        append_timeline_event(
            timeline_path,
            {"stage": "baseline_synthesis", "event": "skipped_due_to_resume_or_disabled"},
        )

    if mode == "code_mod" and "code_mod" in stage_order and start_index <= stage_order.index("code_mod"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="code_mod",
            phase="starting",
            index=5,
            total=len(stage_order),
            next_stage="mesh_gate",
        )
        append_timeline_event(
            timeline_path,
            {
                "stage": "code_mod",
                "event": "protocol_check",
                "protocol_source": "embedded_v2",
                "protocol_applied": True,
            },
        )
        _run_code_mod_branch(
            run_dir=run_dir,
            repo_root=repo_root,
            env=env,
            timeline_path=timeline_path,
            state_path=state_path,
            code_mod_payload=args.code_mod_payload.strip(),
            topic=args.topic,
            lit_path=lit_path,
            base_case_dir=args.base_case_dir,
            pdfs=list(args.pdfs),
            equation_images=list(args.equation_images),
            starter_dir=starter_dir,
            continue_on_code_mod_validation_mismatch=args.continue_on_code_mod_validation_mismatch,
        )
        _checkpoint(state_path, run_dir, "code_mod_done", {})
        _record_plan_update(
            plan_path,
            "Code-mod branch executed before experiment sweep.",
            {"mode": mode},
        )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="code_mod",
            phase="completed",
            index=5,
            total=len(stage_order),
            next_stage="mesh_gate",
        )
    elif mode != "code_mod":
        append_timeline_event(timeline_path, {"stage": "code_mod", "event": "skipped_not_required_for_mode", "mode": mode})
    elif mode == "code_mod" and "code_mod" in stage_order:
        append_timeline_event(timeline_path, {"stage": "code_mod", "event": "skipped_due_to_resume"})
    elif mode == "code_mod":
        append_timeline_event(timeline_path, {"stage": "code_mod", "event": "skipped_not_in_llm_plan"})

    # ---- Post-code_mod revision of hypothesis + requirements ----
    # When code_mod ran (or was already done and we're resuming past it),
    # re-run hypothesis and requirements with the implemented model context
    # so experiments actually reference the custom model.
    _code_mod_done = (
        mode == "code_mod"
        and (run_dir / "canonical_base_case" / "customModels").is_dir()
    )
    if _code_mod_done and (not open_discovery):
        code_mod_ctx_text = _collect_code_mod_context(run_dir, state_path, include_source=True)
        if code_mod_ctx_text:
            skip_post_cm_rev, skip_cm_reason = _should_skip_post_code_mod_hypothesis_requirements_revision(
                run_dir, start_index, stage_order
            )
            if skip_post_cm_rev:
                print(f"[ORCH] SKIP post-code_mod hypothesis/requirements revision: {skip_cm_reason}")
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "hypothesis_revision",
                        "event": "skipped",
                        "reason": skip_cm_reason,
                    },
                )
            else:
                code_mod_ctx_file = run_dir / "code_mod_context.txt"
                code_mod_ctx_file.write_text(code_mod_ctx_text, encoding="utf-8")
                print(f"[ORCH] Code-mod context written ({len(code_mod_ctx_text)} chars) -> {code_mod_ctx_file}")

                mesh_gate_active = (not args.disable_mesh_gate) and ("mesh_gate" in stage_order)

                # Re-run hypothesis with code-mod awareness (once per run_dir before experiments)
                print("[ORCH] REVISING hypothesis with code-mod context")
                append_timeline_event(
                    timeline_path,
                    {"stage": "hypothesis_revision", "event": "starting", "reason": "code_mod_context_available"},
                )
                _call_stage(
                    [
                        sys.executable,
                        "scripts/hypothesis.py",
                        "--literature",
                        str(lit_path),
                        "--topic",
                        args.topic,
                        "--output",
                        str(hyp_path),
                        "--timeline",
                        str(timeline_path),
                        "--code-mod-context",
                        str(code_mod_ctx_file),
                        *_su_flag(run_dir),
                    ],
                    "hypothesis_revision",
                    repo_root,
                    env,
                    timeline_path,
                    state_path,
                )

                # Re-run requirements with code-mod context + mesh stripping
                print("[ORCH] REVISING requirements with code-mod context")
                req_cmd = [
                    sys.executable,
                    "scripts/requirements.py",
                    "--hypotheses",
                    str(hyp_path),
                    "--topic",
                    args.topic,
                    "--output",
                    str(req_path),
                    "--timeline",
                    str(timeline_path),
                    "--code-mod-context",
                    str(code_mod_ctx_file),
                    *_su_flag(run_dir),
                ]
                if mesh_gate_active:
                    req_cmd.append("--strip-mesh-params")
                _call_stage(
                    req_cmd,
                    "requirements_revision",
                    repo_root,
                    env,
                    timeline_path,
                    state_path,
                )
                _checkpoint(
                    state_path,
                    run_dir,
                    "hypothesis_requirements_revised",
                    {
                        "code_mod_context_file": str(code_mod_ctx_file),
                        "mesh_params_stripped": mesh_gate_active,
                    },
                )
                print("[ORCH] Hypothesis + requirements revised with code-mod context")
    elif _code_mod_done and open_discovery:
        append_timeline_event(
            timeline_path,
            {
                "stage": "hypothesis_revision",
                "event": "skipped_open_discovery_mode",
                "reason": "preserve discovery-generated candidate set and policy",
            },
        )

    if (not args.disable_mesh_gate) and ("mesh_gate" in stage_order) and start_index <= stage_order.index("mesh_gate"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="mesh_gate",
            phase="starting",
            index=6,
            total=len(stage_order),
            next_stage="experiments",
        )
        requirements_seed = _read_json(req_path, [])
        seed_req = ""
        if isinstance(requirements_seed, list) and requirements_seed and isinstance(requirements_seed[0], dict):
            seed_req = str(requirements_seed[0].get("user_requirement_text") or "")
        if not seed_req:
            seed_req = f"Baseline mesh-gate case for topic: {args.topic}"
        if resume_skip_mesh_gate_run:
            print("[ORCH] --resume-from mesh_gate_resume: skipping mesh gate run (already done), "
                  "going straight to resume + hypothesis/requirements revision")
            append_timeline_event(
                timeline_path,
                {"stage": "mesh_gate", "event": "skipped_resume_from_mesh_gate_resume"},
            )
        else:
            _run_mesh_gate(
                run_dir=run_dir,
                base_requirement=seed_req,
                repo_root=repo_root,
                env=env,
                timeline_path=timeline_path,
                state_path=state_path,
                topic=args.topic,
            )
        # Write per-group mesh resume and re-run hypothesis + requirements with mesh context.
        resume_path = _write_mesh_gate_resume(run_dir)
        if resume_path and resume_path.is_file():
            print("[ORCH] REVISING hypothesis + requirements with mesh-gate resume")
            append_timeline_event(
                timeline_path,
                {"stage": "mesh_gate_resume", "event": "starting", "resume_path": str(resume_path)},
            )
            # Re-run hypothesis with mesh context
            _call_stage(
                [
                    sys.executable,
                    "scripts/hypothesis.py",
                    "--literature", str(lit_path),
                    "--topic", args.topic,
                    "--output", str(hyp_path),
                    "--timeline", str(timeline_path),
                    "--mesh-gate-resume", str(resume_path),
                    *_su_flag(run_dir),
                ]
                + (["--code-mod-context", str(run_dir / "code_mod_context.txt")]
                   if (run_dir / "code_mod_context.txt").is_file() else []),
                "hypothesis_mesh_revision",
                repo_root, env, timeline_path, state_path,
            )
            # Re-run requirements with mesh context — strips mesh cell params, injects blockMeshDict
            req_cmd = [
                sys.executable,
                "scripts/requirements.py",
                "--hypotheses", str(hyp_path),
                "--topic", args.topic,
                "--output", str(req_path),
                "--timeline", str(timeline_path),
                "--mesh-gate-resume", str(resume_path),
                "--strip-mesh-params",
                *_su_flag(run_dir),
            ]
            if (run_dir / "code_mod_context.txt").is_file():
                req_cmd += ["--code-mod-context", str(run_dir / "code_mod_context.txt")]
            _call_stage(req_cmd, "requirements_mesh_revision",
                        repo_root, env, timeline_path, state_path)
            append_timeline_event(
                timeline_path,
                {"stage": "mesh_gate_resume", "event": "done",
                 "hypothesis_revised": True, "requirements_revised": True},
            )
            print("[ORCH] Hypothesis + requirements revised with mesh-gate resume")

        _checkpoint(state_path, run_dir, "mesh_gate_done", {"selected_mesh_spec": str(run_dir / "selected_mesh_spec.json")})
        _record_plan_update(
            plan_path,
            "Mesh gate completed; selected mesh spec applied to downstream experiments.",
            {"selected_mesh_spec": str(run_dir / "selected_mesh_spec.json")},
        )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="mesh_gate",
            phase="completed",
            index=6,
            total=len(stage_order),
            next_stage="experiments",
        )
    elif args.disable_mesh_gate:
        append_timeline_event(timeline_path, {"stage": "mesh_gate", "event": "disabled_by_flag"})
    elif "mesh_gate" in stage_order:
        append_timeline_event(timeline_path, {"stage": "mesh_gate", "event": "skipped_due_to_resume"})
    else:
        append_timeline_event(timeline_path, {"stage": "mesh_gate", "event": "skipped_not_in_llm_plan"})

    requirements = _read_json(req_path, [])
    if not isinstance(requirements, list):
        requirements = []
    requirements = requirements[: args.max_experiments]

    exp_results: List[Dict[str, Any]] = []
    manifest_cases: List[Dict[str, Any]] = []
    if "experiments" in stage_order and start_index <= stage_order.index("experiments"):
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="experiments",
            phase="starting",
            index=7,
            total=len(stage_order),
            next_stage="analysis",
            details={"max_experiments": args.max_experiments, "max_reruns": args.max_reruns},
        )
        # Resolve the active code-mod artifact (OED winner OR regular code_mod
        # output) and expose it to FoamAgent's writer/reviewer via
        # CFD_OED_ARTIFACT_JSON. This injects "DO NOT regenerate from scratch
        # — preserve THIS verbatim" guidance + the OF10 path facts into every
        # input_writer / reviewer LLM prompt during the experiments stage.
        # Generic across topics and across both OED / regular-code_mod
        # pathways (oed_artifact.py handles both). No-op when no artifact
        # exists (fresh runs without code mods).
        try:
            artifact_json_path = run_dir / "oed_artifact.json"
            ec = subprocess.run(
                [sys.executable, "scripts/oed_artifact.py",
                 "--run-dir", str(run_dir),
                 "--output", str(artifact_json_path)],
                cwd=repo_root, env=env, capture_output=True, text=True,
                timeout=60,
            )
            if ec.returncode == 0 and artifact_json_path.is_file():
                d = _read_json(artifact_json_path, {})
                if isinstance(d, dict) and d.get("status") == "ok":
                    env["CFD_OED_ARTIFACT_JSON"] = str(artifact_json_path)
                    if verbose:
                        print(f"[ORCH] OED artifact wired: category={d.get('category')} "
                              f"iter={d.get('source_iteration')} "
                              f"score={(d.get('best_score') or {}).get('value')}")
                    append_timeline_event(timeline_path, {
                        "stage": "experiments",
                        "event": "oed_artifact_wired",
                        "category": d.get("category"),
                        "source_iteration": d.get("source_iteration"),
                    })
        except Exception as ex:
            if verbose:
                print(f"[ORCH] OED artifact resolution skipped: {type(ex).__name__}: {ex}")
        _ensure_starter_seed_case(run_dir, repo_root, state_path, timeline_path, starter_dir)

        # When an OED artifact is wired AND this run produced a working
        # discovered model, delegate the experiments stage to the post-OED
        # bridge. The bridge auto-generates a parametric plan using the
        # artifact's REAL coefficient names (avoiding the LLM-renamed
        # vocabulary problem in requirements.json), seeds each case from
        # the working artifact case, patches coefficients in the right
        # dict block (handles both runtime_source and class_derivation),
        # runs via foam_run_simple.py (no FoamAgent reviewer overhead),
        # and scores via the bound comparator. Generic across topics and
        # modification families.
        # Gate the bridge to fire ONLY for real OED runs that actually
        # produced a winning iteration. Three conditions must all hold:
        #   (a) artifact file exists and reports status="ok"
        #   (b) state.open_discovery is True (this run elected the OED path)
        #   (c) artifact provenance is NOT "regular_code_mod" — that label is
        #       attached when oed_artifact.py synthesizes a pseudo-entry
        #       from a single code_mod case with no OED iteration history
        #   (d) best_iteration > 0 — a real OED winner, not iteration 0
        # This is generic across all code-mod families: regular code_mod
        # runs (--open-ended-budget=0 or no OED loop) fall through to the
        # standard _run_experiment loop on requirements.json, while real
        # OED winners still get the fast bridge.
        _artifact_data = _read_json(artifact_json_path, {}) if artifact_json_path.is_file() else {}
        if not isinstance(_artifact_data, dict):
            _artifact_data = {}
        try:
            _artifact_best_iter = int(_artifact_data.get("best_iteration") or 0)
        except (TypeError, ValueError):
            _artifact_best_iter = 0
        artifact_active = (
            _artifact_data.get("status") == "ok"
            and bool(open_discovery)
            and _artifact_data.get("provenance") != "regular_code_mod"
            and _artifact_best_iter > 0
        )
        bridge_summary_path = run_dir / "cases" / "_post_oed_summary.json"
        if artifact_active:
            print("[ORCH] experiments stage: delegating to post-OED bridge "
                  "(generates fresh parametric plan from artifact's real "
                  "coefficient names; bypasses FoamAgent reviewer).")
            append_timeline_event(timeline_path, {
                "stage": "experiments",
                "event": "delegate_to_bridge",
            })
            bridge_rc = subprocess.run(
                [sys.executable, "scripts/run_post_oed_experiments.py",
                 "--run-dir", str(run_dir),
                 "--cases-dir-name", "cases"],
                cwd=repo_root, env=env, check=False, timeout=86400,
            ).returncode
            print(f"[ORCH] bridge rc={bridge_rc}; reading {bridge_summary_path}")

            # Reconstruct exp_results + manifest from bridge's output.
            bridge_summary = _read_json(bridge_summary_path, {})
            bridge_results = (bridge_summary or {}).get("results", []) if isinstance(bridge_summary, dict) else []
            for r in bridge_results:
                if not isinstance(r, dict):
                    continue
                cd = r.get("case_dir", "")
                final_status = "PROCEED" if r.get("run_ok") else "FAILED"
                # baseline-vs-score gate already applied by bridge if it
                # produced a score; we mark PROCEED if run_ok regardless of
                # absolute score so the analysis stage sees all cases.
                exp_results.append({
                    "case_id": r.get("case_name", ""),
                    "final_status": final_status,
                    "attempts": [{"attempt": 1, "status": final_status,
                                   "score": r.get("score"),
                                   "fast_path": True}],
                    "case_dir": cd,
                })
                manifest_cases.append({
                    "case_id": r.get("case_name", ""),
                    "case_path": cd,
                    "status": "success" if r.get("run_ok") else "failed",
                    "parameters": r.get("overrides", {}),
                })
            _write_json(run_dir / "manifest.json", {"cases": manifest_cases})
            print(f"[ORCH] bridge: {len(exp_results)} cases recorded "
                  f"({sum(1 for r in exp_results if r['final_status'] == 'PROCEED')} PROCEED)")
        else:
            append_timeline_event(
                timeline_path,
                {
                    "stage": "experiments",
                    "event": "start_batch",
                    "requirements_count": len(requirements),
                },
            )
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                rec = _run_experiment(
                    req=req,
                    run_dir=run_dir,
                    repo_root=repo_root,
                    env=env,
                    timeline_path=timeline_path,
                    max_reruns=args.max_reruns,
                )
                exp_results.append(rec)
                if rec.get("final_status") != "PROCEED":
                    _record_plan_update(
                        plan_path,
                        "Experiment failed to reach PROCEED within rerun cap; downstream analysis will use successful subset only.",
                        {"case_id": rec.get("case_id"), "final_status": rec.get("final_status")},
                    )
                elif len(rec.get("attempts", [])) > 1:
                    _record_plan_update(
                        plan_path,
                        "Experiment required adaptive reruns before success.",
                        {"case_id": rec.get("case_id"), "attempts": len(rec.get("attempts", []))},
                    )
                if verbose:
                    print(
                        f"[ORCH] CASE {rec.get('case_id')} final_status={rec.get('final_status')} "
                        f"attempts={len(rec.get('attempts', []))}"
                    )
                case_status = "success" if rec.get("final_status") == "PROCEED" else "failed"
                manifest_cases.append(
                    {
                        "case_id": rec.get("case_id"),
                        "case_path": rec.get("case_dir"),
                        "status": case_status,
                        "parameters": {},
                    }
                )
                _write_json(run_dir / "manifest.json", {"cases": manifest_cases})
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "experiments",
                        "event": "manifest_updated",
                        "case_id": rec.get("case_id"),
                        "status": case_status,
                        "manifest_path": str(run_dir / "manifest.json"),
                    },
                )
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="experiments",
            phase="completed",
            index=7,
            total=len(stage_order),
            next_stage="analysis",
            details={"total_cases_processed": len(exp_results)},
        )
        _checkpoint(state_path, run_dir, "experiments_done", {"count": len(exp_results)})
    elif "experiments" in stage_order:
        append_timeline_event(timeline_path, {"stage": "experiments", "event": "skipped_due_to_resume"})
        m = _read_json(run_dir / "manifest.json", {"cases": []})
        mc = m.get("cases", []) if isinstance(m, dict) else []
        if isinstance(mc, list):
            for c in mc:
                if not isinstance(c, dict):
                    continue
                exp_results.append(
                    {
                        "case_id": c.get("case_id"),
                        "final_status": "PROCEED" if c.get("status") == "success" else "FAILED",
                        "attempts": [],
                        "case_dir": c.get("case_path", ""),
                    }
                )
    else:
        append_timeline_event(timeline_path, {"stage": "experiments", "event": "skipped_not_in_llm_plan"})

    success_dirs = [r["case_dir"] for r in exp_results if r.get("final_status") == "PROCEED"]
    analysis_path = run_dir / "analysis.json"
    figs_dir = run_dir / "figs"
    paper_dir = run_dir / "paper"
    review_path = run_dir / "review.json"
    ix_analysis = stage_order.index("analysis") + 1 if "analysis" in stage_order else 8
    ix_paper = stage_order.index("paper_review") + 1 if "paper_review" in stage_order else 9
    next_after_paper = (
        "reference_verify" if ("reference_verify" in stage_order) else "finish"
    )
    # Includes resume-from paper_review: re-run paper only (skip analyze/viz) when start_index > analysis.
    if success_dirs and ("analysis" in stage_order) and start_index <= stage_order.index("paper_review"):
        use_legacy_paper = bool(getattr(args, "legacy_paper_pipeline", False))
        if start_index <= stage_order.index("analysis"):
            req_text_for_metrics = "\n\n".join(
                str(r.get("user_requirement_text", ""))
                for r in requirements
                if isinstance(r, dict)
            )
            analysis_plan = _llm_plan_analysis_stage(
                repo_root=repo_root,
                topic=args.topic,
                requirement_text=req_text_for_metrics or args.topic,
                case_dirs=success_dirs,
                code_mod_context=_collect_code_mod_context(run_dir, state_path),
                max_metrics=8,
            )
            analysis_metrics = analysis_plan.get("metrics", []) if isinstance(analysis_plan, dict) else []
            if not isinstance(analysis_metrics, list) or not analysis_metrics:
                analysis_metrics = _llm_decide_analysis_metrics(
                    repo_root=repo_root,
                    topic=args.topic,
                    requirement_text=req_text_for_metrics or args.topic,
                    code_mod_context=_collect_code_mod_context(run_dir, state_path),
                    max_metrics=8,
                )
            analysis_plan_path = run_dir / "analysis_plan.json"
            _write_json(analysis_plan_path, analysis_plan if isinstance(analysis_plan, dict) else {})
            print(f"[ORCH] Analysis metrics (LLM): {analysis_metrics}")
            append_timeline_event(
                timeline_path,
                {
                    "stage": "analysis",
                    "event": "metrics_selected",
                    "metrics": analysis_metrics,
                    "analysis_plan_path": str(analysis_plan_path),
                },
            )
            _append_stage_pointer(
                state_path=state_path,
                timeline_path=timeline_path,
                stage="analysis",
                phase="starting",
                index=ix_analysis,
                total=len(stage_order),
                next_stage="paper_review",
                details={"successful_cases": len(success_dirs)},
            )
            if resume_skip_viz_full:
                print("[ORCH] Skipping viz_full; reusing existing figures directory.")
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "analysis",
                        "event": "viz_full_skipped",
                        "reason": "resume_from=analysis_without_viz_full",
                        "figs_dir": str(figs_dir),
                    },
                )
            elif use_legacy_paper:
                _call_stage(
                    [
                        sys.executable,
                        "scripts/viz.py",
                        "--cases",
                        *success_dirs,
                        "--mode",
                        "full",
                        "--output",
                        str(figs_dir),
                        "--viz-plan-json",
                        str(analysis_plan_path),
                    ],
                    "viz_full",
                    repo_root,
                    env,
                    timeline_path,
                    state_path,
                )
            else:
                append_timeline_event(
                    timeline_path,
                    {
                        "stage": "analysis",
                        "event": "viz_full_skipped",
                        "reason": "unified_paper_pipeline_scripts_paper_unified_py",
                        "figs_dir": str(figs_dir),
                    },
                )
            _record_plan_update(
                plan_path,
                "Analysis completed on successful experiment subset.",
                {"successful_cases": len(success_dirs)},
            )
            _checkpoint(state_path, run_dir, "analysis_done", {"analysis_path": str(analysis_path)})
            _call_stage(
                [
                    sys.executable,
                    "scripts/analyze.py",
                    "--cases",
                    *success_dirs,
                    "--metrics",
                    ",".join(analysis_metrics),
                    "--output",
                    str(analysis_path),
                    "--benchmark-data",
                    str(benchmark_path),
                    "--reference-manifest",
                    str(reference_manifest_path),
                    "--topic",
                    args.topic,
                    "--cross-objectives-json",
                    str(analysis_plan_path),
                ],
                "analysis",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
            _append_stage_pointer(
                state_path=state_path,
                timeline_path=timeline_path,
                stage="analysis",
                phase="completed",
                index=ix_analysis,
                total=len(stage_order),
                next_stage="paper_review",
                details={"analysis_output": str(analysis_path)},
            )
        else:
            append_timeline_event(
                timeline_path,
                {
                    "stage": "paper_unified",
                    "event": "resume_skip_analysis",
                    "reason": "--resume-from=paper_review (reusing analysis.json and prior artifacts)",
                    "analysis_path": str(analysis_path),
                },
            )

        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="paper_review",
            phase="starting",
            index=ix_paper,
            total=len(stage_order),
            next_stage=next_after_paper,
            details={"paper_template": args.paper_template},
        )
        mesh_ind_path = run_dir / "mesh_independence_context.json"
        mesh_bundle = _build_mesh_independence_paper_bundle(run_dir)
        if mesh_bundle:
            _write_json(mesh_ind_path, mesh_bundle)
            append_timeline_event(
                timeline_path,
                {
                    "stage": "paper_write",
                    "event": "mesh_independence_context_written",
                    "path": str(mesh_ind_path),
                    "mesh_levels": len(mesh_bundle.get("metrics_by_mesh_level") or []),
                },
            )
        if use_legacy_paper:
            paper_cmd: List[str] = [
                sys.executable,
                "scripts/paper_utils.py",
                "--analysis",
                str(analysis_path),
                "--figs",
                str(figs_dir),
                "--literature",
                str(lit_path),
                "--output",
                str(paper_dir),
                "--topic",
                args.topic,
                "--manifest",
                str(run_dir / "manifest.json"),
                "--review-output",
                str(review_path),
                "--max-review-loops",
                "10",
                "--template",
                args.paper_template,
            ]
            if mesh_bundle and mesh_ind_path.is_file():
                paper_cmd.extend(["--mesh-independence", str(mesh_ind_path)])
            paper_cmd.extend(_su_flag(run_dir))
            _call_stage(
                paper_cmd,
                "paper_write",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
            _call_stage(
                [
                    sys.executable,
                    "scripts/reviewer.py",
                    "--paper",
                    str(paper_dir),
                    "--output",
                    str(review_path),
                    "--topic",
                    args.topic,
                    "--analysis",
                    str(analysis_path),
                    "--figs",
                    str(figs_dir),
                    "--manifest",
                    str(run_dir / "manifest.json"),
                    "--pdf",
                    str(paper_dir / "main.pdf"),
                ],
                "paper_review",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
        else:
            unified_cmd: List[str] = [
                sys.executable,
                "scripts/paper_unified.py",
                "--repo-root",
                str(repo_root),
                "--run-dir",
                str(run_dir),
                "--topic",
                args.topic,
                "--paper-dir",
                str(paper_dir),
                "--analysis",
                str(analysis_path),
                "--manifest",
                str(run_dir / "manifest.json"),
                "--requirements",
                str(req_path),
                "--literature",
                str(lit_path),
                "--review-output",
                str(review_path),
                "--template",
                args.paper_template,
                "--max-review-loops",
                "10",
            ]
            if mesh_bundle and mesh_ind_path.is_file():
                unified_cmd.extend(["--mesh-independence", str(mesh_ind_path)])
            unified_cmd.extend(_su_flag(run_dir))
            _call_stage(
                unified_cmd,
                "paper_unified",
                repo_root,
                env,
                timeline_path,
                state_path,
            )
        _record_plan_update(
            plan_path,
            "Paper draft and review completed.",
            {"paper_dir": str(paper_dir), "review_path": str(review_path)},
        )
        _checkpoint(state_path, run_dir, "paper_review_done", {"paper_dir": str(paper_dir), "review_path": str(review_path)})
        _append_stage_pointer(
            state_path=state_path,
            timeline_path=timeline_path,
            stage="paper_review",
            phase="completed",
            index=ix_paper,
            total=len(stage_order),
            next_stage=next_after_paper,
            details={"paper_dir": str(paper_dir), "review_path": str(review_path)},
        )
        if "reference_verify" in stage_order and start_index <= stage_order.index("reference_verify"):
            _run_reference_verify_post(
                repo_root=repo_root,
                run_dir=run_dir,
                paper_dir=paper_dir,
                lit_path=lit_path,
                env=env,
                timeline_path=timeline_path,
                state_path=state_path,
                stage_order=stage_order,
                skip_reference_verify=args.skip_reference_verify,
            )
    elif (
        success_dirs
        and args.resume_from.strip() == "reference_verify"
        and ("reference_verify" in stage_order)
        and start_index <= stage_order.index("reference_verify")
    ):
        _run_reference_verify_post(
            repo_root=repo_root,
            run_dir=run_dir,
            paper_dir=paper_dir,
            lit_path=lit_path,
            env=env,
            timeline_path=timeline_path,
            state_path=state_path,
            stage_order=stage_order,
            skip_reference_verify=args.skip_reference_verify,
        )
    final_status = "completed" if success_dirs else "failed"
    plan_final = _read_json(plan_path, {})
    if isinstance(plan_final, dict):
        plan_final["status"] = "completed" if final_status == "completed" else "partially_completed_or_failed"
        _write_json(plan_path, plan_final)
    _update_state(
        state_path,
        {
            "status": final_status,
            "experiments_total": len(exp_results),
            "experiments_proceed": len(success_dirs),
            "analysis_path": str(analysis_path) if analysis_path.exists() else "",
            "benchmark_data_path": str(benchmark_path) if benchmark_path.exists() else "",
            "figs_dir": str(figs_dir) if figs_dir.exists() else "",
            "paper_dir": str(paper_dir) if paper_dir.exists() else "",
            "review_path": str(review_path) if review_path.exists() else "",
        },
    )
    append_timeline_event(
        timeline_path,
        {
            "stage": "orchestrator",
            "event": "finish",
            "status": final_status,
            "experiments_total": len(exp_results),
            "experiments_proceed": len(success_dirs),
        },
    )
    if verbose:
        print("[ORCH] =============================================")
        print(f"[ORCH] FINAL STATUS: {final_status}")
        print(f"[ORCH] Experiments total: {len(exp_results)}")
        print(f"[ORCH] Experiments proceed: {len(success_dirs)}")
        print(f"[ORCH] State file: {state_path}")
        print(f"[ORCH] Timeline file: {timeline_path}")
        print("[ORCH] =============================================")
    print(json.dumps(_read_json(state_path, {}), indent=2))
    return 0 if final_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


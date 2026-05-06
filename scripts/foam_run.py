#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from timeline_logger import append_timeline_event, resolve_timeline_path

# Real case-local OpenFOAM model .so files are much larger than empty link stubs (~15kB).
_CASE_LOCAL_SO_MIN_BYTES_PLAUSIBLE = 50_000


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def build_router_state(requirement: str, llm_service: Any, config: Any, case_dir: str) -> Dict[str, Any]:
    return {
        "user_requirement": requirement,
        "config": config,
        "case_dir": case_dir,
        "tutorial": "",
        "case_name": "",
        "subtasks": [],
        "current_subtask_index": 0,
        "error_command": None,
        "error_content": None,
        "loop_count": 0,
        "llm_service": llm_service,
        "case_stats": None,
        "tutorial_reference": None,
        "case_path_reference": None,
        "dir_structure_reference": None,
        "case_info": None,
        "allrun_reference": None,
        "dir_structure": None,
        "commands": None,
        "foamfiles": None,
        "error_logs": None,
        "history_text": None,
        "case_domain": None,
        "case_category": None,
        "case_solver": None,
        "mesh_info": None,
        "mesh_commands": None,
        "custom_mesh_used": None,
        "mesh_type": None,
        "custom_mesh_path": None,
        "review_analysis": None,
        "rewrite_plan": None,
        "input_writer_mode": None,
        "similar_case_advice": None,
        "requires_hpc": None,
        "requires_visualization": None,
        "job_id": None,
        "cluster_info": None,
        "slurm_script_path": None,
        "termination_reason": None,
    }


def _error_logs_to_text(error_logs: list[Any]) -> str:
    chunks: list[str] = []
    for it in (error_logs or []):
        if isinstance(it, str):
            chunks.append(it)
            continue
        if isinstance(it, dict):
            file_name = str(it.get("file", "") or "")
            content = str(it.get("error_content", "") or "")
            if file_name and content:
                chunks.append(f"[{file_name}] {content}")
            elif content:
                chunks.append(content)
            else:
                chunks.append(json.dumps(it, ensure_ascii=False))
            continue
        chunks.append(str(it))
    return "\n".join(chunks)


def _is_timeout_or_slow_progress(error_logs: list[Any]) -> bool:
    txt = _error_logs_to_text(error_logs).lower()
    markers = (
        "timed out",
        "time limit",
        "timeout",
        "execution timed out",
        "job exceeded",
        "walltime",
    )
    return any(m in txt for m in markers)


def _is_openfoam_case_dir(p: Path) -> bool:
    return (
        (p / "system" / "controlDict").exists()
        and (p / "constant").exists()
        and (p / "0").exists()
    )


def _case_has_poly_mesh(case_dir: Path) -> bool:
    pm = case_dir / "constant" / "polyMesh"
    if not pm.is_dir():
        return False
    return (pm / "points").exists() or (pm / "faces").exists() or (pm / "owner").exists()


def _read_poly_boundary_patch_types(case_dir: Path) -> Dict[str, str]:
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary_file.is_file():
        return {}
    txt = boundary_file.read_text(encoding="utf-8", errors="ignore")
    patch_types: Dict[str, str] = {}
    for m in re.finditer(r"(?ms)^\s*([A-Za-z_]\w*)\s*\{(.*?)\}", txt):
        patch_name = m.group(1)
        body = m.group(2)
        tm = re.search(r"(?m)^\s*type\s+([A-Za-z_]\w*)\s*;", body)
        if tm:
            patch_types[patch_name] = tm.group(1)
    return patch_types


def _force_patch_type_in_field(text: str, patch: str, patch_type: str) -> Tuple[str, bool]:
    pattern = re.compile(rf"(?ms)(^\s*{re.escape(patch)}\s*\{{)(.*?)(\}})")
    changed = False

    def _repl(m: re.Match[str]) -> str:
        nonlocal changed
        head, body, tail = m.group(1), m.group(2), m.group(3)
        tm = re.search(r"(?m)^(\s*type\s+)([A-Za-z_]\w*)(\s*;)", body)
        if tm:
            cur = tm.group(2)
            if cur != patch_type:
                changed = True
                body2 = re.sub(
                    r"(?m)^(\s*type\s+)([A-Za-z_]\w*)(\s*;)",
                    rf"\1{patch_type}\3",
                    body,
                    count=1,
                )
                return f"{head}{body2}{tail}"
            return m.group(0)
        changed = True
        insert = f"\n        type {patch_type};\n"
        return f"{head}{insert}{body}{tail}"

    out = pattern.sub(_repl, text, count=1)
    return out, changed


def _align_zero_patchfields_with_mesh(case_dir: Path) -> Dict[str, Any]:
    """
    Keep 0/* boundaryField patch types consistent with existing polyMesh boundary patch types.
    This prevents common runtime failures like 'patch type empty vs patchField cyclic'.
    """
    patch_types = _read_poly_boundary_patch_types(case_dir)
    if not patch_types:
        return {"changed": False, "files": []}
    zero_dir = case_dir / "0"
    if not zero_dir.is_dir():
        return {"changed": False, "files": []}

    enforce = {k: v for k, v in patch_types.items() if v in {"empty", "cyclic"}}
    if not enforce:
        return {"changed": False, "files": []}

    changed_files: list[str] = []
    for f in zero_dir.iterdir():
        if not f.is_file():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        if "boundaryField" not in txt:
            continue
        updated = txt
        changed = False
        for patch_name, patch_type in enforce.items():
            updated, c = _force_patch_type_in_field(updated, patch_name, patch_type)
            changed = changed or c
        if changed and updated != txt:
            f.write_text(updated, encoding="utf-8")
            changed_files.append(str(f))

    return {"changed": bool(changed_files), "files": changed_files}


def _prepare_output_dir_for_case_copy(out: Path) -> None:
    """Remove standard OpenFOAM subtrees so a baseline copy can replace them (keeps figs, logs, etc.)."""
    for sub in ("0", "system", "constant"):
        d = out / sub
        if d.is_dir():
            shutil.rmtree(d)
    for name in ("Allrun", "Allclean", "Allpost"):
        f = out / name
        if f.is_file():
            f.unlink()


def _sync_copy_openfoam_case(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            # Copy only case dictionaries/code assets; do not copy constant/polyMesh.
            if item.name in {"0", "system", "customModels"}:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target, symlinks=False, ignore_dangling_symlinks=True)
                continue
            if item.name == "constant":
                target.mkdir(parents=True, exist_ok=True)
                for child in item.iterdir():
                    if child.name == "polyMesh":
                        continue
                    child_target = target / child.name
                    if child.is_dir():
                        if child_target.exists():
                            shutil.rmtree(child_target)
                        shutil.copytree(child, child_target, symlinks=False, ignore_dangling_symlinks=True)
                    elif child.is_file():
                        shutil.copy2(child, child_target)
                continue
        elif item.is_file():
            shutil.copy2(item, target)


def _resolve_case_local_model_so(model_dir: Path) -> Optional[Path]:
    """Path to lib under customModels/<Class>/platforms/... if present."""
    plat = model_dir / "platforms"
    if not plat.is_dir():
        return None
    cls = model_dir.name
    for p in plat.rglob(f"lib{cls}.so"):
        if p.is_file():
            return p
    for p in sorted(plat.rglob("*.so")):
        if p.is_file():
            return p
    return None


def _case_local_so_looks_linked(so: Path, class_name: str) -> bool:
    """
    True if the shared object likely contains the custom model (not an empty link stub).

    Reviewer-corrupted ``Make/files`` (e.g. ``LIB=$(FOAM_CASE)/...``) can make wmake emit a
    tiny .so with almost no symbols; OpenFOAM then reports "Unknown ...ViscosityModel type".
    """
    if not so.is_file():
        return False
    try:
        blob = so.read_bytes()
    except OSError:
        return False
    if class_name.encode("utf-8") in blob:
        return True
    return so.stat().st_size >= _CASE_LOCAL_SO_MIN_BYTES_PLAUSIBLE


def _make_files_needs_case_local_repair(text: str, class_name: str) -> bool:
    """Detect wmake fragments that produce broken or empty custom libraries."""
    t = text.replace("\r\n", "\n")
    if "LIB_SRCS" in t:
        return True
    if "$(FOAM_CASE)" in t or "$(FOAM_USER_LIBBIN)" in t:
        return True
    if f"{class_name}.C" not in t:
        return True
    if not re.search(
        rf"^\s*LIB\s*=\s*\./platforms/\$\(WM_OPTIONS\)/lib/lib{re.escape(class_name)}\s*$",
        t,
        re.MULTILINE,
    ):
        return True
    return False


def _write_canonical_custom_model_make_files(model_dir: Path) -> None:
    """
    OpenFOAM wmake reads ``Make/files`` as a makefile include: list ``.C`` sources first,
    then ``LIB =``. There is no ``LIB_SRCS``; using it yields an empty link and a stub .so.
    """
    cls = model_dir.name
    body = f"{cls}.C\n\nLIB = ./platforms/$(WM_OPTIONS)/lib/lib{cls}\n"
    mf = model_dir / "Make" / "files"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(body, encoding="utf-8")


def _wclean_libso_in_model_dir(model_dir: Path) -> Any:
    md = str(model_dir.resolve())
    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    bashrc = Path(wm) / "etc" / "bashrc" if wm else None
    if bashrc and bashrc.is_file():
        inner = f'source "{bashrc}" && cd "{md}" && wclean libso'
        return subprocess.run(
            ["bash", "-lc", inner],
            capture_output=True,
            text=True,
        )
    return subprocess.run(
        "wclean libso",
        shell=True,
        cwd=md,
        capture_output=True,
        text=True,
    )


def _libs_token_for_model(case_dir: Path, model_dir: Path) -> str:
    """Quoted path token: case-relative .so if built, else short libName.so."""
    so = _resolve_case_local_model_so(model_dir)
    if so is not None and so.is_file():
        try:
            rel = so.resolve().relative_to(case_dir.resolve())
            return f'"{str(rel).replace(chr(92), "/")}"'
        except ValueError:
            return f'"{str(so.resolve()).replace(chr(92), "/")}"'
    return f'"lib{model_dir.name}.so"'


def _normalize_control_dict_custom_libs(case_dir: Path) -> List[str]:
    """
    Sanitize ``libs``: drop broken env/relative paths, dedupe, and prefer **case-relative**
    paths to built ``customModels/<Class>/platforms/.../lib/*.so`` entries.  Bare
    ``libClass.so`` often fails to load on OpenFOAM 10 for nested case-local libraries.
    """
    cd = case_dir / "system" / "controlDict"
    if not cd.is_file():
        return []
    txt = cd.read_text(encoding="utf-8", errors="ignore")
    mlibs = re.search(r"(?ms)^\s*libs[\s\n]*\([\s\S]*?\)\s*;", txt)
    if not mlibs:
        return []
    block = mlibs.group(0)
    cm = case_dir / "customModels"
    if not cm.is_dir():
        return []
    names = sorted(p.name for p in cm.iterdir() if p.is_dir() and (p / "Make" / "files").is_file())
    if not names:
        return []
    parts = " ".join(_libs_token_for_model(case_dir, cm / n) for n in names)
    replacement = f"libs\n(\n    {parts}\n);"
    messy = bool(
        re.search(r"FOAM_(CASE|USER_LIBBIN|SITE_LIBBIN)", block)
        or "$WM_OPTIONS" in block
        or "./customModels" in block
        or (block.count(".so") > len(names))
    )
    has_built = any(_resolve_case_local_model_so(cm / n) is not None for n in names)
    short_libs_only = has_built and not messy and all(
        (f'"lib{n}.so"' in block or f"'lib{n}.so'" in block) for n in names
    )
    if not messy and not short_libs_only:
        return []
    txt2 = re.sub(r"(?ms)^\s*libs[\s\n]*\([\s\S]*?\)\s*;", replacement, txt, count=1)
    if txt2 != txt:
        out_txt = txt2 if txt2.endswith("\n") else txt2 + "\n"
        cd.write_text(out_txt, encoding="utf-8")
        return names
    return []


def _wmake_libso_in_model_dir(model_dir: Path) -> Any:
    """
    Run wmake with OpenFOAM env when possible. Plain ``wmake`` from Python often
    inherits an incomplete environment (no WM_OPTIONS / compiler flags), producing
    no case-local .so even when the log looks quiet.
    """
    md = str(model_dir.resolve())
    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    bashrc = Path(wm) / "etc" / "bashrc" if wm else None
    if bashrc and bashrc.is_file():
        inner = f'set -e; source "{bashrc}" && cd "{md}" && wmake libso'
        return subprocess.run(
            ["bash", "-lc", inner],
            capture_output=True,
            text=True,
        )
    return subprocess.run(
        "wmake libso",
        shell=True,
        cwd=md,
        capture_output=True,
        text=True,
    )


def _ensure_custom_models_built(case_dir: Path) -> Tuple[bool, List[str]]:
    """
    Ensure each customModels/<Class>/ tree has a **real** linked ``lib<Class>.so``.

    Repairs reviewer-broken ``Make/files``, clears linker stubs, and runs ``wmake libso`` when needed.
    """
    cm = case_dir / "customModels"
    if not cm.is_dir():
        return True, []
    failures: List[str] = []
    for md in sorted(cm.iterdir()):
        if not md.is_dir() or not (md / "Make" / "files").is_file():
            continue
        cls = md.name
        try:
            mtxt = (md / "Make" / "files").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            mtxt = ""
        if mtxt and _make_files_needs_case_local_repair(mtxt, cls):
            _write_canonical_custom_model_make_files(md)
        so = _resolve_case_local_model_so(md)
        if so is not None and _case_local_so_looks_linked(so, cls):
            continue
        if so is not None:
            _wclean_libso_in_model_dir(md)
        proc = _wmake_libso_in_model_dir(md)
        so2 = _resolve_case_local_model_so(md)
        if proc.returncode != 0 or so2 is None or not _case_local_so_looks_linked(so2, cls):
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-1800:].strip()
            failures.append(f"{cls}: wmake rc={proc.returncode} :: {tail}")
    return (len(failures) == 0, failures)


def _resolve_base_case_flow(
    output_dir: Path,
    base_case_dir: str,
    reuse_existing_case: bool,
) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (edit_existing, skip_mesh_from_copy, detail_message).
    edit_existing: use Foam-Agent initial_write in edit mode (copied / in-place case).
    skip_mesh_from_copy: mesh already on disk — use copied_case Allrun path.
    """
    out = output_dir.resolve()
    reuse = bool(reuse_existing_case)
    base_raw = (base_case_dir or "").strip()
    base = Path(base_raw).expanduser().resolve() if base_raw else None

    if reuse:
        if not _is_openfoam_case_dir(out):
            raise ValueError(
                f"--reuse-existing-case requires output-dir to be a valid OpenFOAM case; got {out}"
            )
        has_mesh = _case_has_poly_mesh(out)
        return True, has_mesh, "reuse_existing_case"

    if base is None:
        return False, False, None

    if not _is_openfoam_case_dir(base):
        raise ValueError(f"--base-case-dir is not a valid OpenFOAM case: {base}")

    if base.resolve() != out.resolve():
        _prepare_output_dir_for_case_copy(out)
        _sync_copy_openfoam_case(base, out)

    has_mesh = _case_has_poly_mesh(out)
    return True, has_mesh, f"copied_from:{base}"


def _maybe_increase_timestep_cfl_safe(case_dir: str) -> Dict[str, Any]:
    """
    Conservative controlDict deltaT update for very slow/timeout cases.
    Keeps/forces adjustable timestep with a bounded maxCo so CFL remains controlled.
    """
    control_dict = Path(case_dir) / "system" / "controlDict"
    if not control_dict.exists():
        return {"changed": False, "reason": "controlDict not found"}

    text = control_dict.read_text(encoding="utf-8", errors="ignore")

    dt_match = re.search(r"(?m)^\s*deltaT\s+([0-9eE\+\-\.]+)\s*;", text)
    if not dt_match:
        return {"changed": False, "reason": "deltaT not found in controlDict"}

    try:
        current_dt = float(dt_match.group(1))
    except Exception:
        return {"changed": False, "reason": "could not parse deltaT"}

    # Small increase per attempt; do not jump aggressively.
    proposed_dt = current_dt * 1.2
    max_dt_cap = current_dt * 3.0
    new_dt = min(proposed_dt, max_dt_cap)
    if new_dt <= current_dt:
        return {"changed": False, "reason": "deltaT already at cap"}

    updated = re.sub(
        r"(?m)^(\s*deltaT\s+)([0-9eE\+\-\.]+)(\s*;)",
        lambda m: f"{m.group(1)}{new_dt:.10g}{m.group(3)}",
        text,
        count=1,
    )

    # Ensure CFL guardrails are present.
    if re.search(r"(?m)^\s*adjustTimeStep\s+", updated):
        updated = re.sub(
            r"(?m)^(\s*adjustTimeStep\s+)\w+(\s*;)",
            r"\1yes\2",
            updated,
            count=1,
        )
    else:
        updated += "\nadjustTimeStep  yes;\n"

    if re.search(r"(?m)^\s*maxCo\s+", updated):
        updated = re.sub(
            r"(?m)^(\s*maxCo\s+)([0-9eE\+\-\.]+)(\s*;)",
            lambda m: f"{m.group(1)}0.7{m.group(3)}",
            updated,
            count=1,
        )
    else:
        updated += "maxCo           0.7;\n"

    if re.search(r"(?m)^\s*maxDeltaT\s+", updated):
        updated = re.sub(
            r"(?m)^(\s*maxDeltaT\s+)([0-9eE\+\-\.]+)(\s*;)",
            lambda m: f"{m.group(1)}{new_dt:.10g}{m.group(3)}",
            updated,
            count=1,
        )
    else:
        updated += f"maxDeltaT       {new_dt:.10g};\n"

    control_dict.write_text(updated, encoding="utf-8")
    return {
        "changed": True,
        "old_deltaT": current_dt,
        "new_deltaT": new_dt,
        "maxCo": 0.7,
    }


def _read_solver_from_control_dict(case_dir: Path) -> str:
    p = case_dir / "system" / "controlDict"
    if not p.is_file():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?m)^\s*application\s+([^\s;]+)\s*;", txt)
    return m.group(1).strip() if m else ""


def _ensure_minimal_allrun(case_dir: Path, solver: str, mesh_exists: bool) -> None:
    allrun = case_dir / "Allrun"
    if allrun.exists():
        return
    solver_cmd = solver or "simpleFoam"
    mesh_cmd = "" if mesh_exists else "runApplication blockMesh\nrunApplication checkMesh\n"
    script = (
        "#!/bin/sh\n"
        "set -e\n"
        "cd \"$(dirname \"$0\")\" || exit 1\n"
        ". \"$WM_PROJECT_DIR/bin/tools/RunFunctions\"\n\n"
        f"{mesh_cmd}"
        f"runApplication {solver_cmd}\n"
    )
    allrun.write_text(script, encoding="utf-8")
    try:
        allrun.chmod(0o755)
    except Exception:
        pass


# Bedrock Converse + LangChain `with_structured_output` can return empty tool args ({}) for
# large mesh prompts; mesh-gate therefore uses plain completion + delimiter extraction only.
_MESH_GATE_BEGIN = "MESHGATE_BLOCKMESHDICT_BEGIN"
_MESH_GATE_END = "MESHGATE_BLOCKMESHDICT_END"


def _mesh_gate_diag_enabled(role: str) -> bool:
    """Refined always logs; coarse only if MESH_GATE_DIAG=1/true/yes/all."""
    v = (os.getenv("MESH_GATE_DIAG") or "").strip().lower()
    if v in {"1", "true", "yes", "all"}:
        return True
    return role == "refined"


def _mesh_gate_diag_print(label: str, fields: Dict[str, Any]) -> None:
    """Paste-friendly block on stderr (prefixed lines)."""
    print(f"[mesh-gate:diag] ========== {label} ==========", file=sys.stderr)
    for k in sorted(fields.keys()):
        print(f"[mesh-gate:diag] {k}: {fields[k]}", file=sys.stderr)
    print("[mesh-gate:diag] ========== end ==========\n", file=sys.stderr)


def _mesh_gate_llm_plain_text(resp: Any) -> str:
    """Normalize LangChain AIMessage.content (str or Bedrock content blocks) to a string."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        parts: List[str] = []
        for block in resp:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(resp)


def _extract_blockmeshdict_from_mesh_gate_response(raw: str) -> Optional[str]:
    """Parse mesh-gate plain LLM output: delimiters first, then JSON {\"content\":...}, then fenced/raw."""
    if not raw or not raw.strip():
        return None
    # Both markers present — ideal case.
    m = re.search(
        rf"(?s){re.escape(_MESH_GATE_BEGIN)}\s*\r?\n(.*)\r?\n\s*{re.escape(_MESH_GATE_END)}",
        raw,
    )
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # BEGIN marker present but END marker missing — extract everything after BEGIN.
    begin_idx = raw.find(_MESH_GATE_BEGIN)
    if begin_idx >= 0:
        after_begin = raw[begin_idx + len(_MESH_GATE_BEGIN):].lstrip("\r\n")
        # Strip a trailing END marker fragment if any
        end_idx = after_begin.find(_MESH_GATE_END)
        if end_idx >= 0:
            after_begin = after_begin[:end_idx]
        after_begin = after_begin.strip()
        if after_begin and _plausible_blockmeshdict_content(after_begin):
            return after_begin
    # JSON wrapper (model sometimes ignores delimiter instructions)
    try:
        jstart = raw.find("{")
        jend = raw.rfind("}")
        if jstart >= 0 and jend > jstart:
            obj = json.loads(raw[jstart : jend + 1])
            if isinstance(obj, dict):
                c = obj.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    except json.JSONDecodeError:
        pass
    # Last resort: strip markdown fence and check if the result looks like a valid dict.
    # Only accept if it starts with OpenFOAM header patterns — avoids accepting raw prose.
    stripped = _strip_markdown_fence(raw.strip())
    if _plausible_blockmeshdict_content(stripped) and _starts_like_foamfile(stripped):
        return stripped
    return None


def _starts_like_foamfile(text: str) -> bool:
    """True only if text begins with recognisable OpenFOAM file header (no prose preamble)."""
    head = text.lstrip()[:200]
    return head.startswith("/*") or head.startswith("FoamFile") or head.startswith("//")


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"^```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*)\n```\s*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*)\n```", t, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return t


def _plausible_blockmeshdict_content(text: str) -> bool:
    low = text.lower()
    if "foamfile" not in low and "convertToMeters" not in low:
        return False
    if "vertices" not in low or "blocks" not in low:
        return False
    return True


def _parse_blockmesh_cell_count(txt: str) -> int:
    """Total cell count across all hex blocks in blockMeshDict text (fallback parser)."""
    total = 0
    for m in re.finditer(r"hex\s*\([^)]+\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", txt, re.I | re.S):
        try:
            total += int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        except Exception:
            pass
    return total


def _checkmesh_cell_count(case_dir: "Path") -> int:
    """Run blockMesh + checkMesh and return true cell count. Returns 0 on failure."""
    import subprocess
    try:
        r = subprocess.run(
            "blockMesh && checkMesh -noFunctionObjects",
            shell=True, cwd=str(case_dir),
            capture_output=True, text=True, timeout=120,
        )
        for line in (r.stdout + r.stderr).splitlines():
            # checkMesh prints: "    cells:           12345"
            m = re.match(r"\s*cells:\s+(\d+)", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 0


def _apply_mesh_gate_blockmesh_from_requirement(requirement: str, case_dir: Path) -> bool:
    """
    If the requirement contains a MESH_GATE_BLOCKMESH_BEGIN/END block (injected by
    requirements.py after mesh gate), extract and write it to system/blockMeshDict
    before the LLM pre-check runs.  Returns True if a blockMeshDict was written.
    """
    begin_marker = "MESH_GATE_BLOCKMESH_BEGIN"
    end_marker = "MESH_GATE_BLOCKMESH_END"
    bi = requirement.find(begin_marker)
    if bi < 0:
        return False
    content_start = requirement.find("\n", bi)
    if content_start < 0:
        content_start = bi + len(begin_marker)
    else:
        content_start += 1
    ei = requirement.find(end_marker, content_start)
    if ei < 0:
        bmd_content = requirement[content_start:].strip()
    else:
        bmd_content = requirement[content_start:ei].strip()
    if not bmd_content or "FoamFile" not in bmd_content:
        return False
    bmd_path = case_dir / "system" / "blockMeshDict"
    bmd_path.parent.mkdir(parents=True, exist_ok=True)
    bmd_path.write_text(bmd_content, encoding="utf-8")
    print(f"[foam_run] Mesh-gate authoritative blockMeshDict written from requirement "
          f"({len(bmd_content)} chars) -> {bmd_path}")
    return True


def _apply_mesh_gate_blockmesh_llm(
    case_dir: Path,
    llm_service: Any,
    role: str,
    requirement: str,
    timeline_path: Path,
) -> bool:
    """
    Programmatic mesh-gate enforcement: snapshot baseline mesh, ask LLM for a new
    blockMeshDict, validate lightly, write it to system/blockMeshDict.
    """
    if role not in {"coarse", "refined"}:
        return True

    sys_d = case_dir / "system"
    active = sys_d / "blockMeshDict"
    baseline_snap = sys_d / "blockMeshDict_baseline"
    if not active.is_file():
        print("Mesh-gate LLM: system/blockMeshDict missing; skipping.", file=sys.stderr)
        return False

    if baseline_snap.is_file():
        base_text = baseline_snap.read_text(encoding="utf-8", errors="ignore")
    else:
        base_text = active.read_text(encoding="utf-8", errors="ignore")
        baseline_snap.write_text(base_text, encoding="utf-8")

    max_mesh_chars = 32000
    mesh_for_prompt = base_text
    truncated = False
    if len(mesh_for_prompt) > max_mesh_chars:
        mesh_for_prompt = mesh_for_prompt[:max_mesh_chars] + "\n// ... truncated ...\n"
        truncated = True

    direction = (
        "COARSEN the mesh (fewer cells / coarser grading) in a controlled way consistent with the requirement."
        if role == "coarse"
        else "REFINE the mesh (more cells / finer grading) in a controlled way consistent with the requirement."
    )
    system_prompt = (
        "You are an expert OpenFOAM blockMesh editor. "
        "Modify ONLY mesh discretization (hex divisions, simpleGrading, edge meshing if present). "
        "Preserve vertices, blocks topology, patch face lists, mergePatchPairs, and physical domain. "
        "For mesh-gate REFINED: the baseline blockMeshDict in the user message is the **immediate parent** mesh for this stage only; "
        "apply **one** incremental refinement step from it (typically ~10% finer near-wall, ~5% away-from-wall vs that parent), "
        "not a lumped refinement from an earlier study baseline. "
        "Output must be a valid blockMeshDict file. "
        "Respond with ONLY these two marker lines and the file between them (no markdown fences, no preamble): "
        f"a line containing exactly {_MESH_GATE_BEGIN!r}, then the full blockMeshDict, "
        f"then a line containing exactly {_MESH_GATE_END!r}."
    )

    # Get baseline cell count from checkMesh (authoritative); fall back to regex.
    n0_real = _checkmesh_cell_count(case_dir)
    n0 = n0_real if n0_real > 0 else _parse_blockmesh_cell_count(base_text)
    for attempt in (1, 2):
        extra = ""
        if attempt == 2:
            extra = (
                "\n\nYour previous attempt failed validation or cell-count sanity checks. "
                "Return a complete blockMeshDict again; keep topology identical to the baseline file."
            )
        tail_note = (
            "--- Note: baseline text was truncated in this prompt; use unchanged parts from context where needed. ---\n"
            if truncated
            else ""
        )
        user_prompt = (
            f"Mesh-gate role: {role.upper()}.\n"
            f"Task: {direction}\n"
            f"{extra}\n"
            f"Return the full modified blockMeshDict between {_MESH_GATE_BEGIN!r} and {_MESH_GATE_END!r} as instructed.\n\n"
            "--- User requirement (mesh-related instructions; obey fidelity locks if present) ---\n"
            f"{requirement[:16000]}\n\n"
            "--- Baseline blockMeshDict to modify ---\n"
            f"{mesh_for_prompt}\n"
            f"{tail_note}"
        )

        diag_on = _mesh_gate_diag_enabled(role)
        if diag_on:
            _mesh_gate_diag_print(
                f"pre_llm_invoke attempt={attempt}",
                {
                    "role": role,
                    "case_dir_name": case_dir.name,
                    "baseline_blockMeshDict_file_chars": len(base_text),
                    "mesh_text_in_prompt_chars": len(mesh_for_prompt),
                    "mesh_prompt_truncated": truncated,
                    "full_requirement_chars": len(requirement),
                    "requirement_slice_in_prompt_chars": min(16000, len(requirement)),
                    "system_prompt_chars": len(system_prompt),
                    "user_prompt_chars": len(user_prompt),
                    "user_plus_system_chars": len(system_prompt) + len(user_prompt),
                    "baseline_first_hex_cell_count_est": n0,
                    "llm_model_provider": getattr(llm_service, "model_provider", ""),
                    "llm_model_version": getattr(llm_service, "model_version", ""),
                    "bedrock_timeout_env": "BEDROCK_READ_TIMEOUT_SECONDS (default 900 in Foam-Agent tracking_aws)",
                    "note": "Read timeout = HTTP client gave up waiting for AWS (large prompt/output), not throttling.",
                },
            )

        t_llm0 = time.monotonic()
        try:
            # Plain completion avoids Bedrock/LC empty structured tool args ({}).
            raw_resp = llm_service.invoke(user_prompt, system_prompt, pydantic_obj=None)
            raw_text = _mesh_gate_llm_plain_text(raw_resp)
            llm_elapsed_s = round(time.monotonic() - t_llm0, 2)
            new_text = _extract_blockmeshdict_from_mesh_gate_response(raw_text)
            if not new_text:
                raise ValueError(
                    "Mesh-gate LLM response missing delimited blockMeshDict "
                    f"({_MESH_GATE_BEGIN} / {_MESH_GATE_END}) and no usable JSON or fenced fallback."
                )
            if diag_on:
                _mesh_gate_diag_print(
                    f"post_llm_invoke_ok attempt={attempt}",
                    {
                        "llm_wall_clock_s": llm_elapsed_s,
                        "raw_response_chars": len(raw_text),
                        "extracted_blockMeshDict_chars": len(new_text),
                    },
                )
        except Exception as exc:
            llm_elapsed_s = round(time.monotonic() - t_llm0, 2)
            if diag_on:
                _mesh_gate_diag_print(
                    f"post_llm_invoke_error attempt={attempt}",
                    {
                        "llm_wall_clock_s": llm_elapsed_s,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                    },
                )
            print(f"Mesh-gate LLM attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt == 2:
                return False
            continue

        if not _plausible_blockmeshdict_content(new_text):
            print(f"Mesh-gate LLM attempt {attempt}: implausible blockMeshDict content.", file=sys.stderr)
            if attempt == 2:
                return False
            continue

        # Write candidate dict, then verify cell count via checkMesh (authoritative).
        # Fall back to regex parser only if blockMesh/checkMesh fails.
        active.write_text(new_text, encoding="utf-8")
        n1_real = _checkmesh_cell_count(case_dir)
        n1 = n1_real if n1_real > 0 else _parse_blockmesh_cell_count(new_text)
        count_source = "checkMesh" if n1_real > 0 else "regex"
        if n0 > 0 and n1 > 0:
            if role == "coarse" and n1 >= n0:
                print(
                    f"Mesh-gate LLM attempt {attempt}: expected coarser mesh ({count_source}: {n1} >= baseline {n0}).",
                    file=sys.stderr,
                )
                if attempt == 2:
                    return False
                continue
            if role == "refined" and n1 <= n0:
                print(
                    f"Mesh-gate LLM attempt {attempt}: expected finer mesh ({count_source}: {n1} <= baseline {n0}).",
                    file=sys.stderr,
                )
                if attempt == 2:
                    return False
                continue

        new_cells_log = f"~{n1} ({count_source})" if n1 > 0 else "unparsed"
        print(
            f"Mesh-gate LLM: wrote system/blockMeshDict ({role}) "
            f"cells ~{n0} -> {new_cells_log}"
        )
        append_timeline_event(
            timeline_path,
            {
                "stage": "mesh_gate_blockmesh_llm",
                "role": role,
                "cells_baseline": n0,
                "cells_new": n1,
                "truncated_prompt": truncated,
                "attempt": attempt,
            },
        )
        return True

    return False


def _prepare_blockmesh_for_run(case_dir: Path) -> Dict[str, Any]:
    """
    Always prepare blockMesh inputs before each run attempt.
    Strict workflow: OpenFOAM must read mesh from system/blockMeshDict directly.
    """
    sys_dir = case_dir / "system"
    active = sys_dir / "blockMeshDict"
    if not sys_dir.is_dir():
        return {"changed": False, "reason": "system dir missing"}

    if not active.is_file():
        return {"changed": False, "reason": "active blockMeshDict missing"}

    active_txt = active.read_text(encoding="utf-8", errors="ignore")
    baseline_bak = sys_dir / "blockMeshDict_baseline"
    if not baseline_bak.exists():
        baseline_bak.write_text(active_txt, encoding="utf-8")
    return {
        "changed": False,
        "reason": "strict active blockMeshDict mode",
        "active": str(active),
        "cells": _parse_blockmesh_cell_count(active_txt),
    }


def _safe_read_snippet(path: Path, max_chars: int = 2000) -> str:
    if not path.is_file():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return txt[:max_chars]


def _active_viscosity_model(case_dir: Path) -> str:
    mt = case_dir / "constant" / "momentumTransport"
    if not mt.is_file():
        return ""
    txt = mt.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?m)^\s*viscosityModel\s+([^\s;]+)\s*;", txt)
    return m.group(1).strip() if m else ""


def _infer_model_intent(user_requirement: str) -> str:
    t = (user_requirement or "").lower()
    if (
        '"viscosity_model": "newtonian"' in t
        or "baseline newtonian" in t
        or "regular model" in t
        or "usual model" in t
    ):
        return "regular"
    if (
        "custom viscosity model" in t
        or "implemented model" in t
        or "provided equation" in t
        or "code_mod" in t
        or "newly introduced model" in t
    ):
        return "custom"
    return "auto"


def _build_local_custom_model_context(case_dir: Path) -> str:
    """
    Provide all FoamAgent components explicit awareness of local custom model code
    present in the case so they avoid replacing it with unrelated built-in models.
    """
    cm_root = case_dir / "customModels"
    if not cm_root.is_dir():
        return ""

    model_dirs = sorted([p for p in cm_root.iterdir() if p.is_dir()])
    if not model_dirs:
        return ""

    lines: list[str] = []
    active_visc = _active_viscosity_model(case_dir)
    lines.append("LOCAL CUSTOM-MODEL CONTEXT (mandatory):")
    lines.append("- This case includes local custom OpenFOAM model code under customModels/.")
    if active_visc:
        lines.append(f"- momentumTransport viscosityModel (if used): {active_visc}")
    lines.append("- Do not arbitrarily strip custom libraries or swap in unrelated built-in models while fixing runs.")
    lines.append(
        "- NEVER add customModels/** to rewrite targets and NEVER edit Make/files, Make/options, *.C, or *.H "
        "under customModels/ — broken wmake output produces empty .so files and runtime 'Unknown type' / dlopen errors."
    )
    lines.append("- If runtime issues occur, prefer numerical/BC/solver stabilization and targeted dictionary fixes.")
    lines.append("")
    lines.append("Detected custom model directories:")
    for md in model_dirs[:8]:
        lines.append(f"- {md.name}")
        for ext in ("*.H", "*.C"):
            for f in sorted(md.glob(ext))[:3]:
                try:
                    src = f.read_text(encoding="utf-8", errors="ignore")
                    lines.append(f"  --- {f.relative_to(case_dir)} ---")
                    lines.append(src[:6000])
                except Exception:
                    lines.append(f"  - {f.relative_to(case_dir)} (unreadable)")

    mt = case_dir / "constant" / "momentumTransport"
    cd = case_dir / "system" / "controlDict"
    mt_snip = _safe_read_snippet(mt, 2200)
    cd_snip = _safe_read_snippet(cd, 1200)
    if mt_snip:
        lines.append("")
        lines.append("Current activation dictionary snippet (constant/momentumTransport):")
        lines.append(mt_snip)
    if cd_snip:
        lines.append("")
        lines.append("Current controlDict snippet (system/controlDict):")
        lines.append(cd_snip)

    return "\n".join(lines).strip()


def _build_model_intent_lock(user_requirement: str, case_dir: Path, has_custom_ctx: bool) -> str:
    """
    Provide model-usage guidance per experiment requirement.

    When a custom model exists in the case tree, the LLM is given full context
    (model name, source files, activation dictionaries) and asked to decide
    which model configuration is appropriate for this specific experiment.
    """
    active_visc = _active_viscosity_model(case_dir)
    lines: list[str] = []

    if has_custom_ctx:
        lines.append("MODEL GUIDANCE (read carefully):")
        lines.append("A custom model has been implemented and compiled in this case tree.")
        if active_visc:
            lines.append(f"Currently active viscosityModel in momentumTransport: `{active_visc}`.")
        lines.append(
            "Based on the experiment requirement, decide whether this run should exercise the case-local "
            "custom code (library under customModels/) or a plain built-in setup. "
            "For baseline/reference comparisons, use built-ins; for validation of the change, keep the "
            "custom library loaded and the correct activation dictionaries (momentumTransport, "
            "turbulenceProperties, fvModels, fvSchemes, BCs, etc.). "
            "Keep controlDict libs, transport, and numerics mutually consistent."
        )
    else:
        intent = _infer_model_intent(user_requirement)
        lines.append("MODEL INTENT LOCK (mandatory):")
        lines.append(f"- Inferred intent: {intent}.")
        if intent == "regular":
            lines.append("- This run is for regular/baseline model behavior.")
            lines.append("- Keep solver/physics coherent with a built-in regular model for this experiment.")
        elif intent == "custom":
            lines.append("- Custom model intent requested, but no local custom model was detected in this case tree.")
        else:
            if active_visc:
                lines.append(
                    f"- Preserve currently active viscosityModel `{active_visc}` unless the requirement explicitly changes it."
                )
            else:
                lines.append(
                    "- Preserve current transport/turbulence/scheme/BC configuration unless the requirement explicitly changes it."
                )
    return "\n".join(lines)


class PrecheckFixFile(BaseModel):
    file_path: str = Field(description="Relative file path under case dir, e.g. system/fvSolution")
    content: str = Field(description="Full corrected file content")


class PrecheckResult(BaseModel):
    is_satisfactory: bool = Field(description="Whether current case files satisfy requirement")
    reasoning: str = Field(description="Short reason for decision")
    issues: list[str] = Field(default_factory=list, description="List of specific issues found")
    fixes: list[PrecheckFixFile] = Field(
        default_factory=list,
        description="If not satisfactory, provide corrected full contents for files that must change",
    )


def _collect_precheck_files(case_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for folder in ("0", "constant", "system"):
        root = case_dir / folder
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(case_dir)).replace("\\", "/")
            out[rel] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def _case_has_custom_models(case_dir: Path) -> bool:
    cm = case_dir / "customModels"
    if not cm.is_dir():
        return False
    return any(cm.iterdir())


def _build_precheck_auxiliary_context(case_dir: Path, requirement: str) -> str:
    """
    Extra instructions for LLM pre-check: authoritative params, prior verify_params,
    rerun seed metadata, and local custom OpenFOAM code (code-mod) — without dumping huge files twice.
    """
    blocks: list[str] = []

    if "AUTHORITATIVE_TARGET_PARAMETERS" in requirement:
        blocks.append(
            "=== CONFIGURATION AUTHORITY (mandatory) ===\n"
            "The requirement contains AUTHORITATIVE_TARGET_PARAMETERS. That JSON block is the single source of truth "
            "for every numeric and discrete control it lists. It overrides conflicting prose, legacy snapshots, "
            "mesh-gate donor defaults, or rerun-seed donor parameters.\n"
            "Your pre-check edits must make 0/, constant/, and system/ match that block exactly (exact numerics, "
            "correct viscosityModel name, correct generalisedNewtonian / libs() setup). "
            "Do not substitute 'close' values or revert to a simpler model unless the authoritative block explicitly allows it.\n"
        )

    vr_path = case_dir / "verify_result.json"
    if vr_path.is_file():
        try:
            vr = json.loads(vr_path.read_text(encoding="utf-8"))
            blocks.append(
                "=== RECENT verify_params (orchestrator / script) ===\n"
                f"- is_correct: {vr.get('is_correct')}\n"
                f"- fixes_applied (files written in that verification session): {vr.get('fixes_applied', 0)}\n"
                f"- loops: {vr.get('loops', '?')}\n"
                f"- reasoning (truncated): {str(vr.get('reasoning', ''))[:1200]}\n"
                "If verify already reported is_correct=true, prefer is_satisfactory=true unless dictionaries clearly "
                "contradict the authoritative requirement. Do not undo coefficient keywords that match the C++ model.\n"
            )
        except (json.JSONDecodeError, OSError):
            pass

    rs_path = case_dir / "rerun_selection.json"
    if rs_path.is_file():
        try:
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            sid = rs.get("source_case_id")
            sm = rs.get("seed_method", "")
            sp = rs.get("source_case_path", "")
            blocks.append(
                "=== RERUN SEED / CASE ORIGIN ===\n"
                f"Dictionaries may have been refreshed from another case (method={sm!r}, source_case_id={sid!r}, path={sp}). "
                "The experiment requirement and AUTHORITATIVE_TARGET_PARAMETERS still define the target physics; "
                "align this case to them, not to the donor's parameters.\n"
            )
        except (json.JSONDecodeError, OSError):
            pass

    note_path = case_dir / "precheck_context_note.txt"
    if note_path.is_file():
        try:
            raw = note_path.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                blocks.append(
                    "=== OPERATOR / PIPELINE NOTE (precheck_context_note.txt) ===\n"
                    f"{raw[:8000]}\n"
                )
        except OSError:
            pass

    has_cm = _case_has_custom_models(case_dir)
    model_lock = _build_model_intent_lock(requirement, case_dir, has_cm)
    if model_lock.strip():
        blocks.append("=== MODEL INTENT (requirement vs case) ===\n" + model_lock + "\n")

    custom_src = _build_local_custom_model_context(case_dir)
    if custom_src.strip():
        blocks.append(
            "=== LOCAL CODE-MOD (C++ / wmake) — read-only for pre-check ===\n"
            "The case may ship custom viscosity (or other) libraries under customModels/. "
            "Dictionary coefficient names and nesting MUST match what the C++ expects. "
            "Pre-check must NOT propose edits to *.C, *.H, Make/files, or Make/options under customModels/.\n"
            + custom_src
        )

    return "\n".join(blocks).strip()


def _apply_precheck_fixes(case_dir: Path, fixes: list[PrecheckFixFile]) -> int:
    changed = 0
    for fx in fixes:
        rel = (fx.file_path or "").strip().lstrip("./")
        if not rel:
            continue
        if not (rel.startswith("0/") or rel.startswith("constant/") or rel.startswith("system/")):
            continue
        target = (case_dir / rel).resolve()
        try:
            target.relative_to(case_dir.resolve())
        except ValueError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        old = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
        if old != fx.content:
            target.write_text(fx.content, encoding="utf-8")
            print(f"Pre-check updated file: {target}")
            changed += 1
    return changed


def _llm_precheck_case_files(
    *,
    case_dir: Path,
    requirement: str,
    llm_service: Any,
    max_loops: int = 10,
) -> Tuple[bool, Dict[str, Any]]:
    """
    LLM pre-check loop before any run:
    validate 0/, constant/, system/ against requirement and fix in-place until satisfactory.
    """
    if max_loops <= 0:
        return True, {
            "loops": 0,
            "changed_files": 0,
            "reasoning": "LLM pre-check disabled (max_loops<=0)",
            "issues": [],
        }
    loops = 0
    total_changed = 0
    last_reason = ""
    for i in range(1, max_loops + 1):
        loops = i
        files = _collect_precheck_files(case_dir)
        aux = _build_precheck_auxiliary_context(case_dir, requirement)
        system_prompt = (
            "You are an OpenFOAM pre-run validator. "
            "Check whether case dictionaries/files satisfy the requirement and are internally consistent. "
            "Only evaluate and fix files under folders 0, constant, and system. "
            "If not satisfactory, return minimal corrected full file contents for only the files that must change. "
            "Do not invent new physics outside requirement intent.\n\n"
            "STICK TO THE STATED CONFIGURATION:\n"
            "- Treat AUTHORITATIVE_TARGET_PARAMETERS (when present) and Target parameters JSON as hard constraints.\n"
            "- Do not replace the requested viscosity model, BC pattern (e.g. cyclic vs patches), or forcing (e.g. fvConstraints) "
            "with a different setup you find 'easier' unless the requirement explicitly allows alternatives.\n"
            "- If the auxiliary context describes a prior verify_params pass or code-mod custom library, respect it: "
            "avoid thrashing schemes/solutions already consistent with the requirement.\n\n"
            "PARAMETER CORRECTNESS (mandatory):\n"
            "The requirement may contain 'Target parameters: {...}' with exact values. You MUST ensure:\n"
            "1. Every parameter listed in Target parameters that corresponds to an OpenFOAM dictionary "
            "keyword must be set to the EXACT specified value in the appropriate case file.\n"
            "2. If Target parameters specify a custom/non-standard model name and the case has a compiled "
            "custom library under customModels/<Class>/platforms/.../lib/, ensure system/controlDict loads that "
            "library (short \\\"lib<ClassName>.so\\\" or a path relative to the case root; the runner will normalize). "
            "Do not put raw $FOAM_USER_LIBBIN, $FOAM_CASE, or bare $WM_OPTIONS tokens inside libs() — those break dlopen.\n"
            "3. If Target parameters specify a standard/built-in model, do NOT load custom libraries for "
            "that model type.\n"
            "4. Match the EXACT numeric values from Target parameters — do not round, change, or substitute them.\n\n"
            "CRITICAL RULE: If a custom model is loaded via libs() in controlDict and activated "
            "in a case dictionary, do NOT rename, restructure, or relocate "
            "its coefficient keywords or subdictionary.  The C++ source code expects exact keyword "
            "names and dictionary nesting — changing them causes runtime crashes.  Only fix solver "
            "numerics, BCs, fvSchemes, fvSolution, or controlDict run settings.\n\n"
            "OPENFOAM DICTIONARY SEMANTICS (apply these rules to every case, not just the current one):\n"
            "A. SOURCE / FORCING TERMS:\n"
            "   - Body forces, velocity constraints, and momentum sources that CONSTRAIN the solution\n"
            "     (e.g. meanVelocityForce, buoyancyForce, MRFZone, porousMedia, SHM refinement regions)\n"
            "     belong in system/fvConstraints (OpenFOAM v6+/v10).\n"
            "   - Volumetric source terms that ADD to equations but do not constrain\n"
            "     (e.g. Smagorinsky forcing, custom scalar sources, energy deposition)\n"
            "     belong in system/fvModels (v6+) or constant/fvOptions (older).\n"
            "   - Never place a velocity/momentum CONSTRAINT in fvModels, or a pure source in fvConstraints.\n"
            "   - If you see meanVelocityForce/pressureGradientExplicitSource in fvModels, move it to fvConstraints.\n"
            "B. TURBULENCE MODEL LOCATION:\n"
            "   - constant/momentumTransport (OpenFOAM 10) or constant/turbulenceProperties (older).\n"
            "   - Never put turbulence settings in system/.\n"
            "C. NORMALIZATION IN POST-PROCESSING:\n"
            "   - Any script that computes a non-dimensional coefficient (Cf, Cd, Nu, Cp, etc.) MUST\n"
            "     use the reference velocity/length/density from AUTHORITATIVE_TARGET_PARAMETERS\n"
            "     (or from the case transport/control dictionaries). Never hardcode reference = 1.\n"
            "D. SIGN CONVENTIONS:\n"
            "   - OpenFOAM wallShearStress, heatFlux, and similar wall function objects report the\n"
            "     quantity acting ON THE FLUID (not on the wall). If the reference data uses the\n"
            "     opposite convention, the post-processing script must negate the extracted value.\n"
            "   - Always verify sign by checking that physically expected attached-flow regions\n"
            "     give the correct sign (e.g. positive streamwise wall friction in attached flow)."
        )
        user_prompt_parts = []
        if aux:
            user_prompt_parts.append(aux)
            user_prompt_parts.append("")
        user_prompt_parts.append(f"Requirement:\n{requirement}\n")
        user_prompt_parts.append(
            "Current case files (relative path -> full content):\n"
            f"{json.dumps(files, ensure_ascii=False)[:180000]}"
        )
        user_prompt = "\n".join(user_prompt_parts)
        res: PrecheckResult = llm_service.invoke(user_prompt, system_prompt, pydantic_obj=PrecheckResult)
        last_reason = res.reasoning
        if res.is_satisfactory:
            return True, {
                "loops": loops,
                "changed_files": total_changed,
                "reasoning": last_reason,
                "issues": res.issues,
            }
        changed = _apply_precheck_fixes(case_dir, res.fixes)
        total_changed += changed
        print(
            f"Pre-check iteration {i}/{max_loops}: not satisfactory; "
            f"applied {changed} file update(s)."
        )
        if changed == 0:
            # Avoid spinning if model says not satisfactory but provides no actionable updates.
            break
    return False, {
        "loops": loops,
        "changed_files": total_changed,
        "reasoning": last_reason or "Pre-check did not converge",
    }


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Run full FoamAgent flow for one requirement.")
    parser.add_argument("--requirement", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--base-case-dir",
        default="",
        type=str,
        help="Copy this full OpenFOAM case into output-dir, then edit dictionaries and run (mesh preserved).",
    )
    parser.add_argument(
        "--reuse-existing-case",
        action="store_true",
        help="Do not copy; output-dir already contains a case (e.g. after a seeded rerun). Edit and run.",
    )
    parser.add_argument("--custom-mesh", default="", type=str)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--max-loop", default=3, type=int)
    parser.add_argument(
        "--precheck-max-loop",
        default=10,
        type=int,
        help="Max LLM pre-check iterations before Allrun (default: 10). Use 0 to skip pre-check entirely.",
    )
    parser.add_argument("--max-time-limit", default=21600, type=int)
    parser.add_argument("--timeline", default="", type=str)
    parser.add_argument(
        "--full-seed-rewrite",
        action="store_true",
        help="When a base/reused case exists, still run full RAG planner + initial writer.",
    )
    parser.add_argument(
        "--mesh-gate-role",
        default="",
        choices=["", "baseline", "coarse", "refined"],
        help="Mesh-gate: for coarse/refined, LLM rewrites system/blockMeshDict after pre-check; baseline is a no-op tag.",
    )
    args = parser.parse_args()
    timeline_path = resolve_timeline_path(args.timeline)

    from services.plan import generate_simulation_plan
    from services.input_writer import initial_write, build_allrun, rewrite_files
    from services.review import review_error_logs, generate_rewrite_plan
    from services.run_local import run_allrun_and_collect_errors
    from services.mesh import copy_custom_mesh, prepare_standard_mesh, handle_gmsh_mesh
    from services.run_hpc import (
        create_slurm_script,
        run_simulation_hpc,
        wait_for_job,
        check_logs_for_errors,
        extract_cluster_info_from_requirement,
    )
    from router_func import llm_requires_custom_mesh, llm_requires_hpc, llm_requires_visualization
    from services.visualization import (
        ensure_foam_file,
        generate_deterministic_pyvista_script,
        fix_pyvista_script,
        generate_pyvista_script,
        run_pyvista_script,
    )
    from services import global_llm_service
    from config import Config
    from utils import LLMService, scan_case_directory, read_case_foamfiles

    config = Config()
    out_path = Path(args.output_dir).resolve()
    token_log_path = str(out_path / "llm_token_usage.json")
    os.environ["CFD_TOKEN_LOG_PATH"] = token_log_path
    config.case_dir = str(out_path)
    config.max_loop = args.max_loop
    config.max_time_limit = args.max_time_limit
    llm_service = LLMService(config)
    llm_service.set_token_log_path(token_log_path)
    try:
        global_llm_service.set_token_log_path(token_log_path)
    except Exception:
        pass

    try:
        edit_existing, skip_mesh_from_copy, base_flow = _resolve_base_case_flow(
            out_path,
            args.base_case_dir,
            args.reuse_existing_case,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    built_ok, build_failures = _ensure_custom_models_built(out_path)
    if not built_ok:
        for msg in build_failures:
            print(f"[foam_run] customModels build failure: {msg}", file=sys.stderr)
        append_timeline_event(
            timeline_path,
            {
                "stage": "foam_run",
                "event": "custom_models_build_failed",
                "case_dir": str(out_path),
                "failures": build_failures,
            },
        )
        return 1
    norm_names = _normalize_control_dict_custom_libs(out_path)
    if norm_names:
        print(f"[foam_run] Normalized controlDict libs for custom model(s): {', '.join(norm_names)}")
        append_timeline_event(
            timeline_path,
            {
                "stage": "foam_run",
                "event": "controldict_libs_normalized",
                "class_names": norm_names,
            },
        )

    case_stats_path = Path(config.database_path) / "raw" / "openfoam_case_stats.json"
    if not case_stats_path.is_file():
        print(
            f"Missing OpenFOAM case stats: {case_stats_path}\n"
            "Ensure Foam-Agent database assets are present on disk.",
            file=sys.stderr,
        )
        return 1
    case_stats = json.loads(case_stats_path.read_text(encoding="utf-8"))

    local_model_ctx = _build_local_custom_model_context(out_path)
    effective_requirement = args.requirement
    model_lock = _build_model_intent_lock(args.requirement, out_path, bool(local_model_ctx))
    if model_lock:
        effective_requirement = f"{effective_requirement}\n\n{model_lock}"
    if local_model_ctx:
        effective_requirement = f"{args.requirement}\n\n{local_model_ctx}"
        if model_lock:
            effective_requirement = f"{effective_requirement}\n\n{model_lock}"

    mesh_gate_role_early = (args.mesh_gate_role or "").strip()
    if mesh_gate_role_early in {"coarse", "refined"}:
        sys_early = out_path / "system"
        active_early = sys_early / "blockMeshDict"
        baseline_early = sys_early / "blockMeshDict_baseline"
        if active_early.is_file():
            baseline_early.write_text(
                active_early.read_text(encoding="utf-8", errors="ignore"),
                encoding="utf-8",
            )

    seed_minimal_mode = edit_existing and not args.full_seed_rewrite
    if seed_minimal_mode:
        seed_solver = _read_solver_from_control_dict(out_path)
        plan = {
            "case_name": out_path.name,
            "case_solver": seed_solver or "simpleFoam",
            "case_dir": args.output_dir,
            "subtasks": [],
            "tutorial_reference": "",
            "allrun_reference": "",
            "similar_case_advice": None,
        }
        print("Step 1/10: Seed-case reuse mode (skip RAG planning)")
        print(
            f"Reusing existing case={plan['case_name']} solver={plan['case_solver']} "
            "and preserving existing dictionaries unless reviewer requests fixes."
        )
    else:
        print("Step 1/10: Planning with RAG + FAISS")
        plan = generate_simulation_plan(
            user_requirement=effective_requirement,
            case_stats=case_stats,
            case_dir=args.output_dir,
            searchdocs=config.searchdocs,
        )
        print(
            f"Planned case={plan['case_name']} solver={plan['case_solver']} "
            f"subtasks={len(plan['subtasks'])} tutorial_ref={'yes' if plan.get('tutorial_reference') else 'no'}"
        )
    append_timeline_event(
        timeline_path,
        {
            "stage": "foam_run_start",
            "case_dir": plan["case_dir"],
            "case_name": plan["case_name"],
            "case_solver": plan["case_solver"],
            "user_requirement_text": effective_requirement,
            "max_loop": int(args.max_loop),
            "max_time_limit": int(args.max_time_limit),
            "base_case_flow": base_flow,
            "edit_existing": edit_existing,
            "skip_mesh_from_copy": skip_mesh_from_copy,
        },
    )

    print("Step 2/10: Mesh routing")
    state = build_router_state(effective_requirement, llm_service, config, args.output_dir)
    mesh_type_int = llm_requires_custom_mesh(state)
    mesh_type = "standard_mesh"
    mesh_commands = []
    if skip_mesh_from_copy:
        mesh_type = "copied_case"
        mesh_commands = []
        mesh_result = {}
        print("Mesh type: copied_case (baseline mesh on disk; skipping mesh generation)")
    elif mesh_type_int == 1 and not args.custom_mesh:
        print(
            "Router requested custom_mesh but --custom-mesh was not provided. "
            "Provide a mesh file path or change the requirement.",
            file=sys.stderr,
        )
        return 1
    elif mesh_type_int == 1 and args.custom_mesh:
        mesh_result = copy_custom_mesh(args.custom_mesh, args.requirement, args.output_dir)
        mesh_type = "custom_mesh"
        mesh_commands = mesh_result.get("mesh_commands", [])
    elif mesh_type_int == 2:
        mesh_result = handle_gmsh_mesh(args.requirement, args.output_dir)
        mesh_type = "gmsh_mesh"
        mesh_commands = mesh_result.get("mesh_commands", [])
    else:
        mesh_result = prepare_standard_mesh(args.requirement, args.output_dir)
        mesh_type = "standard_mesh"
        mesh_commands = mesh_result.get("mesh_commands", [])
    if mesh_type != "copied_case":
        print(f"Mesh type: {mesh_type}")

    if seed_minimal_mode:
        print("Step 3/10: Reusing existing files (skip full writer)")
        dir_structure = scan_case_directory(plan["case_dir"])
        foamfiles = read_case_foamfiles(plan["case_dir"], dir_structure)
        print("Step 4/10: Ensure Allrun exists")
        _ensure_minimal_allrun(Path(plan["case_dir"]), str(plan.get("case_solver") or ""), mesh_type == "copied_case")
    else:
        print("Step 3/10: Initial writing")
        write_result = initial_write(
            case_dir=plan["case_dir"],
            subtasks=plan["subtasks"],
            user_requirement=effective_requirement,
            tutorial_reference=plan["tutorial_reference"],
            case_solver=plan["case_solver"],
            generation_mode=getattr(config, "input_writer_generation_mode", "sequential_dependency"),
            similar_case_advice=plan.get("similar_case_advice"),
            reuse_generated_dir=getattr(config, "reuse_generated_dir", ""),
            edit_existing=edit_existing,
        )
        dir_structure = write_result["dir_structure"]
        foamfiles = write_result["foamfiles"]

        print("Step 4/10: Building Allrun")
        build_allrun(
            case_dir=plan["case_dir"],
            database_path=str(config.database_path),
            searchdocs=config.searchdocs,
            dir_structure=dir_structure,
            case_info=f"case name: {plan['case_name']}\ncase solver: {plan['case_solver']}",
            allrun_reference=plan["allrun_reference"],
            mesh_type=mesh_type,
            mesh_commands=mesh_commands,
            user_requirement=effective_requirement,
        )

    # Step 4.45: If the requirement carries an authoritative mesh-gate blockMeshDict
    # (injected by requirements.py after mesh gate), write it to system/blockMeshDict
    # BEFORE the pre-check runs, so the pre-check validates but does not override it.
    _apply_mesh_gate_blockmesh_from_requirement(effective_requirement, Path(plan["case_dir"]))

    pre_loops = int(args.precheck_max_loop)
    if pre_loops <= 0:
        print("Step 4.5/10: LLM pre-check skipped (--precheck-max-loop 0)")
        pre_ok, pre_meta = _llm_precheck_case_files(
            case_dir=Path(plan["case_dir"]),
            requirement=effective_requirement,
            llm_service=llm_service,
            max_loops=0,
        )
    else:
        print("Step 4.5/10: LLM pre-check (0/ constant/ system)")
        pre_ok, pre_meta = _llm_precheck_case_files(
            case_dir=Path(plan["case_dir"]),
            requirement=effective_requirement,
            llm_service=llm_service,
            max_loops=pre_loops,
        )
    append_timeline_event(
        timeline_path,
        {
            "stage": "foam_run_precheck",
            "case_dir": plan["case_dir"],
            "status": "satisfied" if pre_ok else "failed",
            "details": pre_meta,
        },
    )
    if not pre_ok:
        print("Pre-check did not converge within max loops; continuing to Allrun as requested.")

    mesh_gate_role = (args.mesh_gate_role or "").strip()
    if mesh_gate_role in {"coarse", "refined"}:
        print(
            f"Step 4.6/10: Mesh-gate enforced blockMeshDict ({mesh_gate_role}) via LLM "
            "(overwrites system/blockMeshDict after pre-check)"
        )
        if not _apply_mesh_gate_blockmesh_llm(
            Path(plan["case_dir"]),
            llm_service,
            mesh_gate_role,
            effective_requirement,
            timeline_path,
        ):
            print("Mesh-gate blockMeshDict LLM enforcement failed.", file=sys.stderr)
            return 1

    # Pre-check may have modified files. Refresh foamfiles snapshot from disk for reviewer context.
    dir_structure = scan_case_directory(plan["case_dir"])
    foamfiles = read_case_foamfiles(plan["case_dir"], dir_structure)

    case_plan_path = Path(plan["case_dir"])
    post_ok, post_fail = _ensure_custom_models_built(case_plan_path)
    if not post_ok:
        for msg in post_fail:
            print(f"[foam_run] post-precheck customModels build failure: {msg}", file=sys.stderr)
        return 1
    post_norm = _normalize_control_dict_custom_libs(case_plan_path)
    if post_norm:
        print(f"[foam_run] Post-precheck: normalized controlDict libs for: {', '.join(post_norm)}")
        dir_structure = scan_case_directory(plan["case_dir"])
        foamfiles = read_case_foamfiles(plan["case_dir"], dir_structure)

    print("Step 5/10: Runner + reviewer loop")
    error_logs = []
    loop_count = 0
    history_text = []
    rewrite_history: list[str] = []
    while loop_count <= args.max_loop:
        # Do not modify Allrun: keep the standard Foam-Agent Allrun. Refresh libs + rebuild here
        # so reviewer edits to controlDict / sources still pick up case-local .so before each run.
        case_pre = Path(plan["case_dir"])
        nlib = _normalize_control_dict_custom_libs(case_pre)
        if nlib:
            print(f"[foam_run] Pre-Allrun: normalized controlDict libs for: {', '.join(nlib)}")
        b_ok, b_fail = _ensure_custom_models_built(case_pre)
        if not b_ok:
            for msg in b_fail:
                print(f"[foam_run] Pre-Allrun customModels build: {msg}", file=sys.stderr)
        _prepare_blockmesh_for_run(Path(plan["case_dir"]))
        align = _align_zero_patchfields_with_mesh(Path(plan["case_dir"]))
        if align.get("changed"):
            files = align.get("files") or []
            print(f"Aligned boundaryField patch types to mesh for {len(files)} file(s) before run.")
        use_hpc = args.hpc or llm_requires_hpc(state)
        if use_hpc:
            cluster_info = extract_cluster_info_from_requirement(args.requirement, plan["case_dir"])
            script_path = create_slurm_script(plan["case_dir"], cluster_info)
            run_out = run_simulation_hpc(script_path)
            if run_out.job_id:
                wait_for_job(run_out.job_id, max_wait_time=args.max_time_limit)
            error_logs = check_logs_for_errors(plan["case_dir"])
        else:
            error_logs = run_allrun_and_collect_errors(plan["case_dir"], args.max_time_limit)

        if not error_logs:
            print(f"Run successful after {loop_count} review loops.")
            break

        if _is_timeout_or_slow_progress(error_logs):
            tweak = _maybe_increase_timestep_cfl_safe(plan["case_dir"])
            append_timeline_event(
                timeline_path,
                {
                    "stage": "foam_run_slow_progress",
                    "case_dir": plan["case_dir"],
                    "loop_count": int(loop_count),
                    "timestep_update": tweak,
                },
            )
            if tweak.get("changed"):
                print(
                    "Detected slow/timeout behavior. Applied CFL-aware timestep increase: "
                    f"deltaT {tweak.get('old_deltaT')} -> {tweak.get('new_deltaT')} with maxCo={tweak.get('maxCo')}"
                )
            else:
                print(f"Detected slow/timeout behavior, but no timestep update was applied: {tweak.get('reason')}")

        print(f"Errors found. Starting reviewer loop {loop_count + 1}.")
        append_timeline_event(
            timeline_path,
            {
                "stage": "foam_run_reviewer_loop",
                "case_dir": plan["case_dir"],
                "loop_count": int(loop_count + 1),
                "error_logs": error_logs,
            },
        )
        # Always refresh current file contents so each loop sees latest state on disk.
        dir_structure = scan_case_directory(plan["case_dir"])
        foamfiles = read_case_foamfiles(plan["case_dir"], dir_structure)
        loop_req = effective_requirement
        if rewrite_history:
            loop_req = (
                f"{effective_requirement}\n\n"
                "<previous_rewrite_history>\n"
                + "\n".join(rewrite_history[-10:])
                + "\n</previous_rewrite_history>\n"
            )
        review_content, history_text = review_error_logs(
            tutorial_reference=plan["tutorial_reference"],
            foamfiles=foamfiles,
            error_logs=error_logs,
            user_requirement=loop_req,
            similar_case_advice=plan.get("similar_case_advice"),
            history_text=history_text,
            case_dir=plan["case_dir"],
        )
        rewrite_plan = generate_rewrite_plan(
            foamfiles=foamfiles,
            error_logs=error_logs,
            review_analysis=review_content,
            user_requirement=loop_req,
        )
        if rewrite_plan and isinstance(rewrite_plan, dict):
            tf = rewrite_plan.get("target_files", [])
            if isinstance(tf, list):
                filtered_tf: list[Any] = []
                for item in tf:
                    if not isinstance(item, dict):
                        continue
                    fp = (item.get("file") or "").replace("\\", "/").lstrip("./")
                    if fp.startswith("customModels/") or "/customModels/" in f"/{fp}/":
                        continue
                    filtered_tf.append(item)
                if len(filtered_tf) < len(tf):
                    print(
                        f"[foam_run] Dropped {len(tf) - len(filtered_tf)} rewrite target(s) under customModels/ "
                        "(protected)."
                    )
                if not filtered_tf:
                    print(
                        "[foam_run] Reviewer rewrite plan only listed customModels/ paths; "
                        "skipping rewrite this loop (foam_run will repair Make/files + rebuild before the next run)."
                    )
                    rewrite_history.append(
                        f"loop={loop_count + 1}; errors={len(error_logs)}; rewrite_skipped=customModels_only"
                    )
                    loop_count += 1
                    continue
                rewrite_plan = {**rewrite_plan, "target_files": filtered_tf}
        rewrite_result = rewrite_files(
            case_dir=plan["case_dir"],
            error_logs=error_logs,
            review_analysis=review_content,
            rewrite_plan=rewrite_plan,
            user_requirement=loop_req,
            foamfiles=foamfiles,
            dir_structure=dir_structure,
        )
        foamfiles = rewrite_result.get("foamfiles", foamfiles)
        case_loop = Path(plan["case_dir"])
        post_rw_norm = _normalize_control_dict_custom_libs(case_loop)
        if post_rw_norm:
            print(f"[foam_run] Post-rewrite: normalized controlDict libs for: {', '.join(post_rw_norm)}")
        rw_build_ok, rw_build_fail = _ensure_custom_models_built(case_loop)
        if not rw_build_ok:
            for msg in rw_build_fail:
                print(f"[foam_run] Post-rewrite customModels build: {msg}", file=sys.stderr)
        rewrite_history.append(
            f"loop={loop_count + 1}; errors={len(error_logs)}; rewrite_plan={json.dumps(rewrite_plan, ensure_ascii=False)[:2000]}"
        )
        loop_count += 1

    if error_logs and loop_count > args.max_loop:
        print("Max loops reached. Case may have errors.")

    visualization_summary = None

    def _guess_primary_field(user_requirement: str) -> str:
        """Ask the LLM which OpenFOAM field to visualise given the requirement text."""
        if not user_requirement:
            return "U"
        try:
            sys_p = (
                "You are a CFD expert. Given a simulation requirement, return the single most "
                "relevant OpenFOAM field name to visualise (e.g. U, p, T, k, omega, nut, "
                "nuTilda, epsilon, or any custom field mentioned). "
                "Reply with ONLY the field name, nothing else."
            )
            raw = llm_service.invoke(user_requirement, sys_p, pydantic_obj=None)
            field = str(getattr(raw, "content", raw)).strip().split()[0]
            return field if field else "U"
        except Exception:
            return "U"

    if not error_logs and llm_requires_visualization(state):
        print("Step 6/10: Visualization (explicitly requested in requirement)")
        case_dir_abs = str(Path(plan["case_dir"]).resolve())
        foam_file = ensure_foam_file(case_dir_abs)
        output_png_rel = "visualization.png"
        field_name = _guess_primary_field(args.requirement)
        timeout_s = 180
        viz_err_logs: list[str] = []

        deterministic_script = generate_deterministic_pyvista_script(
            foam_file=foam_file,
            output_png=output_png_rel,
            field_preference=field_name,
        )
        success, output_image, errs = run_pyvista_script(
            case_dir_abs,
            deterministic_script,
            filename="visualization.py",
            expected_png=output_png_rel,
            timeout_s=timeout_s,
        )
        if success and output_image:
            visualization_summary = {
                "success": True,
                "output_image": output_image,
                "used": "deterministic_template",
                "field_name": field_name,
            }
        else:
            viz_err_logs.extend(errs)
            current_loop = 0
            max_viz = getattr(config, "max_loop", 2)
            while current_loop < max_viz:
                current_loop += 1
                print(f"Visualization LLM attempt {current_loop} of {max_viz}")
                viz_script = generate_pyvista_script(
                    case_dir_abs, foam_file, args.requirement, viz_err_logs[-2:]
                )
                success, output_image, errs = run_pyvista_script(
                    case_dir_abs,
                    viz_script,
                    filename="visualization_llm.py",
                    expected_png=output_png_rel,
                    timeout_s=timeout_s,
                )
                if success and output_image:
                    visualization_summary = {
                        "success": True,
                        "output_image": output_image,
                        "used": "llm_script",
                        "field_name": field_name,
                    }
                    break
                viz_err_logs.extend(errs)
                if current_loop < max_viz:
                    fixed_script = fix_pyvista_script(foam_file, viz_script, viz_err_logs[-2:])
                    success, output_image, errs = run_pyvista_script(
                        case_dir_abs,
                        fixed_script,
                        filename="visualization_fixed.py",
                        expected_png=output_png_rel,
                        timeout_s=timeout_s,
                    )
                    if success and output_image:
                        visualization_summary = {
                            "success": True,
                            "output_image": output_image,
                            "used": "llm_fixed_script",
                            "field_name": field_name,
                        }
                        break
                    viz_err_logs.extend(errs)
            if visualization_summary is None:
                visualization_summary = {
                    "success": False,
                    "error": f"Visualization failed after {max_viz} LLM attempts",
                    "error_logs": viz_err_logs,
                }
    elif error_logs:
        print("Step 6/10: Skipping visualization (run did not complete cleanly)")
    else:
        print("Step 6/10: Skipping visualization (not requested)")

    print("Step 7/10: Writing run result")
    result = {
        "status": "success" if not error_logs else "failed",
        "case_dir": plan["case_dir"],
        "case_name": plan["case_name"],
        "case_solver": plan["case_solver"],
        "error_logs": error_logs,
        "loop_count": loop_count,
        "visualization": visualization_summary,
        "base_case_flow": base_flow,
        "edit_existing": edit_existing,
        "mesh_type": mesh_type,
    }
    output_result_path = Path(plan["case_dir"]) / "run_result.json"
    output_result_path.parent.mkdir(parents=True, exist_ok=True)
    output_result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {
            "stage": "foam_run_complete",
            "case_dir": plan["case_dir"],
            "case_name": plan["case_name"],
            "status": result["status"],
            "loop_count": int(loop_count),
            "error_logs": error_logs,
            "run_result_path": str(output_result_path),
        },
    )
    print(json.dumps(result, indent=2))
    sys.stdout.flush()
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
def _bootstrap_paths(repo_root: Path) -> None:
    foam_src = repo_root / "Foam-Agent" / "src"
    lang_src = repo_root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


EMBEDDED_CODE_MOD_PROTOCOL_V2 = """
SYSTEM WORKFLOW (EMBEDDED):
- Implement exactly one small in-family OpenFOAM 10 custom change per invocation.
- Supported modes: custom_source | custom_viscosity | custom_turbulence_model_modification | custom_case_library | compare_builtin_models.
- compare_builtin_models: NO custom compilation — run N cases, each selecting a different built-in
  model via `constant/<dict>` (e.g. momentumTransport's RASModel), same mesh/BCs/numerics.
  Skip `wmake libso` entirely. Generate N requirements, one per variant.
- custom_case_library: arbitrary case-local wmake lib (flux schemes, BCs, functionObjects, etc.); user/LLM wires activation dictionaries.
- No solver edits, no OpenFOAM installation edits, no boundary-framework redesign.
- All custom code must be case-local under: <case>/customModels/<ClassName>/
- Required outputs: <ClassName>.H, <ClassName>.C, Make/files, Make/options
- Build method: wmake libso only.
- Load through system/controlDict libs entry.
- Activate in:
  - custom_source -> constant/fvModels
  - custom_viscosity -> constant/momentumTransport (fallback turbulenceProperties)
  - turbulence modification -> constant/momentumTransport (fallback turbulenceProperties)
- Use only supported parent families:
  SpalartAllmaras, kEpsilon, RNGkEpsilon, realizableKE, kOmega, kOmegaSST.
- If symbols/units/parent/insertion site are ambiguous: stop and return NEEDS_INFO context.
- Never create new transported variables or new case fields.
"""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


_MAX_SNAPSHOT_FILE_BYTES = 800_000


def _collect_dictionary_texts(case_path: Path) -> Dict[str, str]:
    """
    Full OpenFOAM dictionary snapshot for LLM / payload context: 0/, system/, constant/
    excluding constant/polyMesh (mesh files stay on disk only).
    """
    out: Dict[str, str] = {}
    roots: Tuple[Tuple[str, Path], ...] = (
        ("0", case_path / "0"),
        ("system", case_path / "system"),
        ("constant", case_path / "constant"),
    )
    for prefix, root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if prefix == "constant" and rel.parts and rel.parts[0] == "polyMesh":
                continue
            rel_key = f"{prefix}/" + "/".join(rel.parts)
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz > _MAX_SNAPSHOT_FILE_BYTES:
                out[rel_key] = f"[OMITTED: file larger than {_MAX_SNAPSHOT_FILE_BYTES} bytes ({sz} bytes)]"
                continue
            txt = _read_text(p)
            if "\x00" in (txt[:4096] if txt else ""):
                out[rel_key] = "[OMITTED: binary or non-text file]"
                continue
            out[rel_key] = txt
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _copy_case_ignore(src: Path, dirpath: str, names: list[str]) -> set[str]:
    """
    Drop huge OpenFOAM time directories when cloning a starter case into canonical_base_case.
    Keeps ``0/`` and all of ``constant/``, ``system/``; strips ``500/``, ``1000/``, processors, etc.
    """
    ignored: set[str] = set()
    try:
        Path(dirpath).resolve().relative_to(src.resolve())
    except ValueError:
        return ignored
    if Path(dirpath).resolve() != src.resolve():
        return ignored
    for n in names:
        if n.isdigit() and n != "0":
            ignored.add(n)
        if n in {"postProcessing", "VTK", "dynamicCode", ".git"}:
            ignored.add(n)
        if n.startswith("processor"):
            ignored.add(n)
    return ignored


def _copy_case_for_working_dir(src: Path, dst: Path) -> None:
    """Full case tree including polyMesh (needed to run); dst is replaced if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        symlinks=False,
        ignore_dangling_symlinks=True,
        ignore=lambda d, names: sorted(_copy_case_ignore(src, d, names)),
    )


def _case_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _extract_paths_from_topic(topic: str, repo_root: Path) -> List[Path]:
    cands: List[Path] = []
    for m in re.findall(r"(/[\w\-\./]+)", topic or ""):
        p = Path(m).expanduser()
        if p.exists():
            cands.append(p.resolve())
    for m in re.findall(r"([\w\-/\.]+)", topic or ""):
        if "/" in m and not m.startswith("http"):
            p = (repo_root / m).resolve()
            if p.exists():
                cands.append(p)
    return cands


def _is_openfoam_case_dir(path: Path) -> bool:
    return (path / "system" / "controlDict").exists() and (path / "constant").exists() and (path / "0").exists()


def _discover_tutorial_cases(repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in repo_root.rglob("controlDict"):
        if p.name != "controlDict":
            continue
        case_dir = p.parent.parent
        if _is_openfoam_case_dir(case_dir):
            out.append(case_dir.resolve())
    return out


def _path_is_under_any_case(path: Path, case_dirs: List[Path]) -> bool:
    rp = path.resolve()
    for c in case_dirs:
        try:
            rp.relative_to(c.resolve())
            return True
        except ValueError:
            continue
    return False


# Plain-text notes in starter/ (not inside a case): equations, code-mod specs, etc.
_STARTER_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst", ".tex"})


def _discover_starter_assets(starter_dir: Path) -> Dict[str, List[Path]]:
    assets: Dict[str, List[Path]] = {"pdfs": [], "images": [], "cases": [], "text_files": []}
    if not starter_dir.exists() or not starter_dir.is_dir():
        return assets
    # Case dirs first so PDFs/images inside a case (e.g. postProcess) are not treated as equation inputs.
    seen_cases = set()
    for p in starter_dir.rglob("controlDict"):
        if p.name != "controlDict":
            continue
        case_dir = p.parent.parent.resolve()
        if _is_openfoam_case_dir(case_dir) and str(case_dir) not in seen_cases:
            assets["cases"].append(case_dir)
            seen_cases.add(str(case_dir))
    case_list = assets["cases"]
    for p in starter_dir.rglob("*"):
        if not p.is_file():
            continue
        if _path_is_under_any_case(p, case_list):
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            assets["pdfs"].append(p.resolve())
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            assets["images"].append(p.resolve())
        elif ext in _STARTER_TEXT_EXTENSIONS:
            assets["text_files"].append(p.resolve())
    return assets


def _case_dir_has_custom_model_sources(case_dir: Path) -> bool:
    """True if the case already contains case-local wmake sources (reference / pre-built code-mod)."""
    cm = case_dir / "customModels"
    if not cm.is_dir():
        return False
    for child in cm.iterdir():
        if child.is_dir() and (child / "Make" / "files").is_file():
            return True
    return False


def _score_case_for_topic(case_dir: Path, topic: str) -> int:
    t = (topic or "").lower()
    name = str(case_dir).lower()
    score = 0
    for k in ["backward", "step", "bfs", "channel", "cavity", "airfoil", "pipe"]:
        if k in t and k in name:
            score += 2
    if "tutorial" in name:
        score += 1
    has_custom_sources = _case_dir_has_custom_model_sources(case_dir)
    # New code-mod runs should prefer a baseline case without an existing customModels/ wmake tree,
    # so generated class names and Make/files are not confused with a shipped reference case.
    if has_custom_sources:
        score -= 8
    impl_hint = any(
        k in t
        for k in (
            "implement",
            "custom model",
            "custom viscos",
            "non-newtonian",
            "nonnewtonian",
            "power-law",
            "power law",
            "new class",
            "wmake",
            "code mod",
            "codemod",
            "source term",
            "fvoption",
            "turbulence",
            "flux",
            "scheme",
            "boundary",
            "compile",
            "openfoam",
            "case-local",
            "caselocal",
        )
    )
    if impl_hint and not has_custom_sources:
        score += 4
    return score


def _extract_github_urls(topic: str, literature_records: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    for u in re.findall(r"https?://[^\s]+", topic or ""):
        if "github.com" in u:
            urls.append(u.rstrip(").,"))
    for r in literature_records:
        if not isinstance(r, dict):
            continue
        for key in ("url", "source", "link"):
            u = str(r.get(key) or "")
            if "github.com" in u:
                urls.append(u)
    dedup: List[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    return dedup


def _clone_github_case(url: str, target_root: Path) -> Optional[Path]:
    target_root.mkdir(parents=True, exist_ok=True)
    repo_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.split("/")[-1] or "repo")
    dst = target_root / repo_name
    if dst.exists():
        return dst
    proc = subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return dst


def _collect_existing_fields(case_path: Path) -> List[str]:
    fields: List[str] = []
    zero_dir = case_path / "0"
    if zero_dir.exists():
        for f in zero_dir.iterdir():
            if f.is_file():
                fields.append(f.name)
    return sorted(set(fields))


def _extract_pdf_texts(pdf_paths: List[Path]) -> str:
    chunks: List[str] = []
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""
    for p in pdf_paths:
        try:
            rd = PdfReader(str(p))
            txt = "\n".join((pg.extract_text() or "") for pg in rd.pages[:20])
            chunks.append(txt)
        except Exception:
            continue
    return "\n".join(chunks)


def _extract_starter_text_files(text_paths: List[Path]) -> str:
    """
    Concatenate starter loose .txt/.md/... files for formula/mode extraction and builder context.
    Same role as PDF text + equation images: long equations live here instead of the CLI topic.
    """
    if not text_paths:
        return ""
    max_per_file = 48_000
    max_total = 120_000
    max_files = 16
    parts: List[str] = []
    budget = 0
    for p in sorted({x.resolve() for x in text_paths}, key=lambda x: str(x))[:max_files]:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunk = raw[:max_per_file]
        header = f"\n\n===== {p.name} ({p}) =====\n\n"
        room = max_total - budget - len(header)
        if room < 200:
            break
        if len(chunk) > room:
            chunk = chunk[:room]
        parts.append(header + chunk)
        budget += len(header) + len(chunk)
        if budget >= max_total:
            break
    return "".join(parts).strip()


def _extract_pdf_texts_llm(pdf_text: str, repo_root: Path) -> str:
    """
    Semantic extraction from PDF text using LLM.
    Keeps pypdf for deterministic text extraction, then uses LLM to extract
    equations/symbols/constants/units in a CFD-aware way.
    """
    if not (pdf_text or "").strip():
        return ""
    _bootstrap_paths(repo_root)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    except Exception:
        return ""
    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    text = pdf_text[:24000]
    sys_prompt = (
        "You extract CFD model equations and implementation-relevant details from paper text. "
        "Return compact plain text with: equations, symbol definitions, constants, units, "
        "target model family hints, and validation quantities."
    )
    user_prompt = (
        "Extract the most useful implementation details for OpenFOAM model change from this paper text:\n\n"
        f"{text}"
    )
    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        out = getattr(resp, "content", "") if resp else ""
        return out if isinstance(out, str) else str(out)
    except Exception:
        return ""


def _extract_image_texts(image_paths: List[Path], repo_root: Path) -> str:
    """
    Vision-LLM extraction for equation images (png/jpg/jpeg/webp).
    No OCR dependency.
    """
    if not image_paths:
        return ""
    _bootstrap_paths(repo_root)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    except Exception:
        return ""
    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0)
    out_chunks: List[str] = []
    sys_prompt = (
        "You are extracting mathematical equations from CFD model figures. "
        "Return only plain text equations and symbol definitions visible in the image. "
        "Do not add explanations or assumptions."
    )
    for p in image_paths:
        try:
            b = p.read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            ext = p.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content = [
                {"type": "text", "text": "Extract equation(s), symbols, constants, and units from this image."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=content)])
            txt = getattr(resp, "content", "") if resp else ""
            if isinstance(txt, str) and txt.strip():
                out_chunks.append(txt.strip())
        except Exception:
            continue
    return "\n".join(out_chunks)


def _pick_formula(topic: str, pdf_text: str, max_chars: int = 2000) -> str:
    """
    Extract the formula/equation block from combined topic + text sources.

    Skips file-header lines (lines whose stripped form matches ^=+.*=+$) so that
    the "===== filename.txt =====" banners added by _extract_starter_text_files are
    never mistaken for an equation.  Collects ALL qualifying lines (not just the
    first) and returns up to max_chars of content so multi-line equations are
    preserved intact.
    """
    _HEADER_RE = re.compile(r"^=+[^=].*[^=]=+$")
    _UNICODE_MAP = [
        ("ν", "nu"), ("𝜈", "nu"), ("η", "nu"),
        ("v∞", "nu_inf"), ("ν∞", "nu_inf"),
        ("γ̇", "gammaDot"), ("γ", "gamma"),
        ("−", "-"), ("–", "-"),
    ]

    def _normalise(s: str) -> str:
        for old, new in _UNICODE_MAP:
            s = s.replace(old, new)
        return s

    text = (topic or "") + "\n" + (pdf_text or "")
    eq_lines = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if _HEADER_RE.match(stripped):
            continue
        if "=" in stripped and 8 < len(stripped) < 400:
            eq_lines.append(_normalise(stripped))

    if eq_lines:
        result = "\n".join(eq_lines)
        return result[:max_chars]
    return "nu = nu0 * (1 + a * pow(gammaDot, n))"


def _guess_mode_and_parent(topic: str) -> Tuple[str, str]:
    t = (topic or "").lower()
    # compare_builtin_models: the task asks to COMPARE existing built-in
    # models (no custom code required). Detect common phrasings early so
    # comparison studies (like BFS "compare k-epsilon, k-omega SST, SA")
    # don't get routed into code_mod unnecessarily. Generic across any
    # physics family (turbulence, viscosity, radiation, combustion, ...).
    compare_signals = (
        "compare several",
        "compare existing",
        "compare built-in",
        "compare builtin",
        "compare the built-in",
        "sensitivity among",
        "model sensitivity",
        "model comparison",
        "compare different",
    )
    if any(k in t for k in compare_signals) and not any(
        k in t for k in ("custom", "novel", "new model", "new variant")
    ):
        return "compare_builtin_models", "unknown"
    if any(
        k in t
        for k in [
            "viscosity",
            "carreau",
            "bingham",
            "power law",
            "power-law",
            "non-newtonian",
            "nonnewtonian",
            "rheolog",
        ]
    ):
        return "custom_viscosity", "unknown"
    if any(k in t for k in ["komega", "k-omega", "sst", "komegasst"]):
        return "custom_turbulence_model_modification", "kOmegaSST"
    if any(k in t for k in ["kepsilon", "k-epsilon", "rng", "realizable"]):
        return "custom_turbulence_model_modification", "kEpsilon"
    if "spalart" in t or "sa model" in t:
        return "custom_turbulence_model_modification", "SpalartAllmaras"
    if any(
        k in t
        for k in [
            "fvmodels",
            "fv model",
            "fv models",
            "custom source",
            "source term",
            "actuator disk",
            "fvoption",
            "fv option",
        ]
    ):
        return "custom_source", "unknown"
    return "custom_case_library", "unknown"


_CODE_MOD_MODES = frozenset(
    {
        "custom_viscosity",
        "custom_turbulence_model_modification",
        "custom_source",
        "custom_case_library",
        "compare_builtin_models",
    }
)
_TURBULENCE_PARENTS = frozenset(
    {"SpalartAllmaras", "kEpsilon", "RNGkEpsilon", "realizableKE", "kOmega", "kOmegaSST"}
)


def _llm_classify_code_mod_mode(
    topic: str, merged_text: str, repo_root: Path, heuristic_mode: str, heuristic_parent: str
) -> Optional[Tuple[str, str]]:
    """
    LLM chooses code-mod mode + turbulence parent so routing stays accurate for arbitrary user wording.
    Returns None on any failure (caller keeps heuristics).
    """
    try:
        _bootstrap_paths(repo_root)
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
    except Exception:
        return None

    ctx = (merged_text or "")[:14000]
    sys_prompt = (
        "You route OpenFOAM 10 **case-local** code changes (under <case>/customModels/, wmake libso, libs in controlDict).\n"
        "Return STRICT JSON only, no markdown:\n"
        '{"mode":"<one>","parent_model_hint":"<one or unknown>","confidence":"high|medium|low","reason":"<short>"}\n'
        "mode must be exactly one of:\n"
        "  custom_viscosity — non-Newtonian / strain-rate viscosity / rheology\n"
        "  custom_turbulence_model_modification — RAS/LES closure tied to a named parent model\n"
        "  custom_source — fvModels / volumetric source / actuator-style term\n"
        "  custom_case_library — any other custom compiled library (flux scheme, BC, functionObject, …)\n"
        "  compare_builtin_models — NO custom code required; user is asking to run/compare\n"
        "                           existing built-in OpenFOAM models against each other. Pick this\n"
        "                           when the topic is 'compare <models A,B,C>' with no explicit\n"
        "                           hint of a novel implementation.\n"
        "parent_model_hint (turbulence only): one of SpalartAllmaras, kEpsilon, RNGkEpsilon, realizableKE, kOmega, kOmegaSST; else unknown.\n"
        "If the user text is ambiguous, prefer custom_case_library over guessing viscosity."
    )
    user_prompt = (
        f"Heuristic guess (may be wrong): mode={heuristic_mode!r}, parent_model_hint={heuristic_parent!r}\n\n"
        f"USER_TOPIC:\n{topic}\n\nCONTEXT:\n{ctx}"
    )
    try:
        settings = get_settings()
        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        raw = getattr(resp, "content", "") if resp else ""
        txt = strip_json_fences(raw if isinstance(raw, str) else str(raw))
        s, e = txt.find("{"), txt.rfind("}")
        if s == -1 or e <= s:
            return None
        obj = json.loads(txt[s : e + 1])
        if not isinstance(obj, dict):
            return None
        mode = str(obj.get("mode") or "").strip()
        parent = str(obj.get("parent_model_hint") or "unknown").strip()
        if mode not in _CODE_MOD_MODES:
            return None
        if mode != "custom_turbulence_model_modification":
            parent = "unknown"
        elif parent not in _TURBULENCE_PARENTS:
            parent = "unknown"
        return mode, parent
    except Exception:
        return None


def _extract_formula_symbols(formula_text: str, repo_root: Optional[Path] = None) -> List[str]:
    """
    Use an LLM to extract only genuine mathematical symbols and named constants
    from the formula text.  Falls back to a conservative regex if the LLM call fails.
    """
    if not (formula_text or "").strip():
        return []

    try:
        if repo_root is not None:
            _bootstrap_paths(repo_root)
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        sys_msg = (
            "You are a CFD equation analyst. "
            "Given a formula or model description, extract ONLY the genuine mathematical "
            "symbols, variable names, and named constants — things that would appear as "
            "C++ variable names in an implementation. "
            "Exclude: English prose words, prepositions, articles, verbs, adjectives, "
            "model family names (SpalartAllmaras, kEpsilon, etc.), and standard math "
            "function names (sqrt, pow, exp, log, sin, cos, abs, min, max, clamp). "
            "Return a JSON array of strings, nothing else. Example: "
            '[\"nuTilda\", \"Cb1\", \"Stilda\", \"beta\", \"Rref\", \"pMin\", \"pMax\", \"S\", \"Omega\"]'
        )
        raw = llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=f"Formula text:\n{formula_text[:4000]}"),
        ])
        txt = str(getattr(raw, "content", raw)).strip()
        # Strip markdown fences if present
        if txt.startswith("```"):
            txt = txt.split("```", 1)[1].lstrip("json").strip()
            if "```" in txt:
                txt = txt.rsplit("```", 1)[0].strip()
        s, e = txt.find("["), txt.rfind("]")
        if s != -1 and e != -1 and e > s:
            txt = txt[s: e + 1]
        result = json.loads(txt)
        if isinstance(result, list):
            symbols = [str(x) for x in result if isinstance(x, str) and x.strip()]
            print(f"[code_mod_prepare] LLM extracted {len(symbols)} formula symbols: {symbols}")
            return symbols
    except Exception as exc:
        print(f"[code_mod_prepare] LLM symbol extraction failed ({exc}); using regex fallback")

    # Regex fallback: conservative — only short identifiers likely to be math symbols
    _PROSE_WORDS = {
        "the", "a", "an", "and", "or", "not", "in", "on", "of", "to", "by",
        "for", "is", "are", "was", "be", "as", "at", "it", "its", "this",
        "that", "with", "from", "take", "use", "where", "only", "change",
        "new", "our", "so", "all", "other", "base", "remain", "unchanged",
        "added", "case", "term", "original", "standard", "custom", "model",
        "production", "terms", "constants", "such",
        "max", "min", "pow", "exp", "log", "sin", "cos", "tan", "sqrt", "abs",
        "clamp", "symm", "skew",
    }
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula_text)
    out: List[str] = []
    for t in tokens:
        if t.lower() in _PROSE_WORDS:
            continue
        if t not in out:
            out.append(t)
    return out


def _wm_project_dir() -> Optional[Path]:
    raw = os.environ.get("WM_PROJECT_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p if p.exists() else None


def _resolve_recon_cache_path(run_dir: Path, recon_cache_arg: str) -> Path:
    """Pick where discovered_paths.json lives.

    Priority: explicit --recon-cache > parent run dir (if run_dir looks like an
    iter_... sub-dir) > run_dir. Parent preference lets all OED iterations of
    the same study share one recon cache.
    """
    if recon_cache_arg:
        return Path(recon_cache_arg).expanduser().resolve()
    if run_dir.name.startswith("iter_") and run_dir.parent.is_dir():
        return (run_dir.parent / "discovered_paths.json").resolve()
    return (run_dir / "discovered_paths.json").resolve()


def _inject_recon_context(
    *,
    run_dir: Path,
    recon_cache_arg: str,
    topic: str,
    mode: str,
    parent: str,
    formula_text: str,
    repo_root: Path,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Load or create discovered_paths.json, return it trimmed for payload + meta."""
    cache_path = _resolve_recon_cache_path(run_dir, recon_cache_arg)
    history_path = cache_path.parent / "discovered_paths.history.json"
    meta: Dict[str, Any] = {"enabled": True, "cache_path": str(cache_path)}

    wm = _wm_project_dir()
    foam_src = (wm / "src") if (wm and (wm / "src").is_dir()) else None

    if foam_src is None:
        meta.update({"status": "no_foam_src"})
        return None, meta

    # Import source_recon only when needed; keep hard failure path clean.
    try:
        sys.path.insert(0, str(repo_root / "scripts"))
        import source_recon  # type: ignore
    except Exception as exc:
        meta.update({"status": "import_failed", "error": str(exc)[:300]})
        return None, meta

    task = {
        "topic": topic,
        "mode": mode,
        "parent_class": parent,
        "formula": (formula_text or "")[:4000],
    }

    if cache_path.is_file():
        print(f"[recon] cache HIT: {cache_path} (reused — no LLM calls)")
    else:
        print(f"[recon] cache MISS: {cache_path} — running slate search (one-time per study)")

    try:
        result = source_recon.run_slate_search(
            foam_src=foam_src,
            task=task,
            cache_path=cache_path,
            history_path=history_path,
            allow_cache_hit=True,
        )
    except Exception as exc:
        meta.update({"status": "recon_failed", "error": str(exc)[:400]})
        print(f"[recon] FAILED: {exc}")
        return None, meta

    cache_hit = bool(result.get("cache_hit"))
    meta.update({
        "status": "ok",
        "cache_hit": cache_hit,
        "rounds_run": result.get("rounds_run"),
        "stopped_reason": result.get("stopped_reason"),
        "selected_file_count": len(result.get("selected_files", [])),
        "verified_include_path_count": len(result.get("verified_include_paths", [])),
    })
    tag = "cache_hit" if cache_hit else "regenerated"
    n_files = len(result.get("selected_files", []))
    n_incl = len(result.get("verified_include_paths", []))
    print(f"[recon] {tag}: {n_files} selected files, {n_incl} verified include paths "
          f"(rounds={result.get('rounds_run')}, stopped={result.get('stopped_reason')})")

    # Compact trimmed copy to embed in payload (drop very large fields).
    trimmed = {
        "foam_src": result.get("foam_src"),
        "selected_files": result.get("selected_files", []),
        "verified_include_paths": result.get("verified_include_paths", []),
        "class_signatures": result.get("class_signatures", []),
        "cache_path": str(cache_path),
        "rounds_run": result.get("rounds_run"),
    }
    return trimmed, meta


def _collect_openfoam_reference_files(
    topic: str,
    mode: str,
    parent: str,
    formula_symbols: List[str],
) -> Dict[str, Any]:
    wm = _wm_project_dir()
    if wm is None:
        return {"wm_project_dir_found": False, "reference_files": []}

    topic_l = (topic or "").lower()
    refs: List[Path] = []
    base = wm / "src" / "MomentumTransportModels" / "momentumTransportModels" / "laminar" / "generalisedNewtonian"
    if mode == "custom_viscosity":
        candidates = [
            base / "generalisedNewtonianViscosityModels" / "generalisedNewtonianViscosityModel" / "generalisedNewtonianViscosityModel.H",
            base / "generalisedNewtonianViscosityModels" / "generalisedNewtonianViscosityModel" / "generalisedNewtonianViscosityModel.C",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "strainRateViscosityModel" / "strainRateViscosityModel.H",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "strainRateViscosityModel" / "strainRateViscosityModel.C",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "powerLaw" / "powerLaw.H",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "powerLaw" / "powerLaw.C",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "BirdCarreau" / "BirdCarreau.H",
            base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "BirdCarreau" / "BirdCarreau.C",
            wm / "src" / "MomentumTransportModels" / "momentumTransportModels" / "Make" / "options",
            wm / "src" / "physicalProperties" / "viscosity" / "viscosity.H",
        ]
        if any(k in topic_l for k in ["sisko", "herschel", "bulkley", "bingham", "yield"]):
            candidates.extend(
                [
                    base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "HerschelBulkley" / "HerschelBulkley.H",
                    base / "generalisedNewtonianViscosityModels" / "strainRateViscosityModels" / "HerschelBulkley" / "HerschelBulkley.C",
                ]
            )
    elif mode == "custom_turbulence_model_modification":
        mtm = wm / "src" / "MomentumTransportModels" / "momentumTransportModels"
        candidates = [
            mtm / "RAS" / "RASModel" / "RASModel.H",
            mtm / "RAS" / "RASModel" / "RASModel.C",
            mtm / "Make" / "options",
        ]
        if parent and parent != "unknown":
            # Select one concrete parent implementation if present.
            for p in mtm.rglob(f"{parent}.H"):
                candidates.append(p)
                c = p.with_suffix(".C")
                if c.exists():
                    candidates.append(c)
                break
    elif mode == "custom_source":
        candidates = [wm / "src" / "fvModels" / "fvModel.H", wm / "src" / "fvModels" / "Make" / "options"]
    elif mode == "custom_case_library":
        candidates = [
            wm / "src" / "finiteVolume" / "Make" / "options",
            wm / "src" / "fvModels" / "fvModel.H",
            wm / "src" / "OpenFOAM" / "Make" / "options",
            wm / "src" / "MomentumTransportModels" / "momentumTransportModels" / "Make" / "options",
        ]
        for sub in (
            wm / "src" / "finiteVolume" / "lnInclude",
            wm / "src" / "meshTools" / "lnInclude",
        ):
            if sub.is_dir():
                for p in sorted(sub.glob("*.H"))[:4]:
                    candidates.append(p)
    else:
        candidates = []

    seen = set()
    for c in candidates:
        if c.exists():
            rc = c.resolve()
            if str(rc) not in seen:
                refs.append(rc)
                seen.add(str(rc))

    out_refs: List[Dict[str, Any]] = []
    for r in refs[:16]:
        txt = _read_text(r)
        out_refs.append(
            {
                "path": str(r),
                "basename": r.name,
                "excerpt": txt[:12000],
            }
        )
    return {
        "wm_project_dir_found": True,
        "wm_project_dir": str(wm),
        "reference_files": out_refs,
        "selection_context": {
            "mode": mode,
            "parent_model_hint": parent,
            "topic": topic,
            "formula_symbols": formula_symbols,
        },
    }


def _build_payload(
    repo_root: Path,
    topic: str,
    case_path: Path,
    formula_text: str,
    mode: str,
    parent: str,
    protocol_text: str,
    starter_artifacts_text: str = "",
) -> Dict[str, Any]:
    dictionary_files = _collect_dictionary_texts(case_path)
    control_text = dictionary_files.get("system/controlDict") or _read_text(case_path / "system" / "controlDict")
    momentum_text = dictionary_files.get("constant/momentumTransport") or _read_text(
        case_path / "constant" / "momentumTransport"
    )
    turb_text = dictionary_files.get("constant/turbulenceProperties") or _read_text(
        case_path / "constant" / "turbulenceProperties"
    )
    fv_models_text = dictionary_files.get("constant/fvModels") or _read_text(case_path / "constant" / "fvModels")
    fields = _collect_existing_fields(case_path)
    if "U" not in fields:
        fields.append("U")

    symbol_table = [
        {"name": "U", "category": "existing_case_field", "units": "m/s", "availability": "available", "resolution_text": "velocity field from case"},
    ]
    constants: List[Dict[str, Any]] = []
    formula_symbols = _extract_formula_symbols(formula_text, repo_root=repo_root)
    lhs_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", formula_text or "")
    lhs_sym = lhs_match.group(1) if lhs_match else ""
    units_hint = {"gammaDot": "1/s", "U": "m/s"}
    # Generic mode: include all detected formula symbols so builder LLM can classify roles.
    # No model-specific assumptions here.
    for sym in formula_symbols:
        if any(isinstance(s, dict) and s.get("name") == sym for s in symbol_table):
            continue
        symbol_table.append(
            {
                "name": sym,
                "category": "formula_symbol",
                "units": units_hint.get(sym, "unknown"),
                "availability": "available",
                "resolution_text": (
                    "detected from user-provided formula; builder must classify as target/field/constant."
                ),
            }
        )
        if sym != lhs_sym:
            constants.append(
                {
                    "name": sym,
                    "value": "UNSPECIFIED_BY_PREPARE",
                    "units": units_hint.get(sym, "unknown"),
                }
            )
    term_specs: List[Dict[str, Any]] = []
    if mode == "custom_turbulence_model_modification":
        term_specs = [
            {
                "equation": "k",
                "kind": "closure_modification",
                "explicitness": "explicit",
                "insertion_site": "production",
                "formula_text": formula_text,
                "coefficient_name": None,
                "units": "various",
                "depends_on": ["k", "U"],
            }
        ]
        if parent in {"kEpsilon", "RNGkEpsilon", "realizableKE"}:
            for f in ["k", "epsilon"]:
                if f not in fields:
                    fields.append(f)
        if parent in {"kOmega", "kOmegaSST"}:
            for f in ["k", "omega"]:
                if f not in fields:
                    fields.append(f)
        if parent == "SpalartAllmaras" and "nuTilda" not in fields:
            fields.append("nuTilda")
    ref_ctx = _collect_openfoam_reference_files(topic, mode, parent, formula_symbols)

    if mode == "custom_viscosity":
        target_equations: List[str] = ["nu"]
    elif mode == "custom_source":
        target_equations = ["U"]
    elif mode == "custom_case_library":
        target_equations = []
    else:
        target_equations = ["k"]

    if mode == "custom_case_library":
        header_hints: List[Dict[str, str]] = [
            {
                "class_name": "generic_case_library",
                "header_path": "see openfoam_reference_context",
                "signature_text": (
                    "Derive base class, addToRunTimeSelectionTable target, and linked libs from the user topic "
                    "and reference_files (may be flux scheme, BC, turbulence, fvModel, functionObject, …)."
                ),
            }
        ]
    else:
        header_hints = [
            {
                "class_name": parent if parent != "unknown" else "viscosityModel",
                "header_path": "hint",
                "signature_text": "hint",
            }
        ]

    return {
        "openfoam_version": "10",
        "case_path": str(case_path),
        "solver": "simpleFoam",
        "case_snapshot": {
            "controlDict_text": control_text,
            "momentumTransport_text": momentum_text,
            "turbulenceProperties_text": turb_text,
            "fvModels_text": fv_models_text,
            "existing_fields": sorted(set(fields)),
            "existing_cellZones": [],
            "existing_custom_libs": [],
            "dictionary_files": dictionary_files,
            "dictionary_files_note": "Full 0/, system/, and constant/ text (constant/polyMesh excluded). Mesh remains on disk only.",
        },
        "solver_capabilities": {
            "supports_fvModels": True,
            "fvModels_supported_fields": ["U", "T", "h"],
        },
        "openfoam_api_context": {
            "header_hints": header_hints,
            "openfoam_reference_context": ref_ctx,
        },
        "request": {
            "raw_user_text": (topic + "\n\n[EMBEDDED CODE-MOD PROTOCOL]\n" + protocol_text[:4000]),
            "starter_artifacts_text": (starter_artifacts_text or "")[:100000],
            "formula_source": "text",
            "formula_text": formula_text,
            "formula_latex": "",
            "declared_mode_hint": mode,
            "target_equations": target_equations,
            "region": "all",
            "family_hint": "RAS",
            "parent_model_hint": parent,
            "symbol_table": symbol_table,
            "constants": constants,
            "term_specs": term_specs,
        },
        "build_policy": {"source_root": f"{case_path}/customModels", "naming_prefix": "Custom"},
    }


def _choose_base_case(
    topic: str,
    repo_root: Path,
    run_dir: Path,
    literature_records: List[Dict[str, Any]],
    explicit_base_dir: str,
    starter_cases: List[Path],
) -> Tuple[Path, str]:
    if explicit_base_dir.strip():
        p = Path(explicit_base_dir).expanduser().resolve()
        if _is_openfoam_case_dir(p):
            return p, "local_explicit"

    if starter_cases:
        starter_cases_sorted = sorted(starter_cases, key=lambda c: _score_case_for_topic(c, topic), reverse=True)
        return starter_cases_sorted[0], "starter_case"

    for p in _extract_paths_from_topic(topic, repo_root):
        if _is_openfoam_case_dir(p):
            return p, "local_topic_path"

    tutorials = _discover_tutorial_cases(repo_root)
    if tutorials:
        tutorials.sort(key=lambda c: _score_case_for_topic(c, topic), reverse=True)
        return tutorials[0], "local_tutorial"

    gh_urls = _extract_github_urls(topic, literature_records)
    for u in gh_urls:
        cloned = _clone_github_case(u, run_dir / "external_cases")
        if cloned is None:
            continue
        cands = [p.parent.parent for p in cloned.rglob("controlDict") if p.name == "controlDict"]
        for c in cands:
            if _is_openfoam_case_dir(c):
                return c.resolve(), "github_case"

    # Fallback: generate baseline with foam_run.
    gen_case = run_dir / "auto_base_case"
    gen_case.mkdir(parents=True, exist_ok=True)
    req = f"Create a baseline OpenFOAM case for topic: {topic}"
    proc = subprocess.run(
        [sys.executable, "scripts/foam_run.py", "--requirement", req, "--output-dir", str(gen_case), "--max-loop", "1", "--max-time-limit", "21600"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and _is_openfoam_case_dir(gen_case):
        return gen_case.resolve(), "generated_foamagent"
    return gen_case.resolve(), "generated_fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare code-mod payload from topic/PDFs/base-case options.")
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--literature", default="", type=str)
    parser.add_argument("--base-case-dir", default="", type=str)
    parser.add_argument("--pdfs", nargs="*", default=[])
    parser.add_argument("--equation-images", nargs="*", default=[])
    parser.add_argument("--starter-dir", default="starter", type=str)
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json from starter_understand.py; "
                             "formula_or_model_spec from this file takes priority over all "
                             "regex/heuristic extraction.")
    parser.add_argument("--recon-cache", default="", type=str,
                        help="Optional path to discovered_paths.json produced by "
                             "scripts/source_recon.py. If missing, recon is attempted "
                             "automatically and the cache is created at this path (or at "
                             "<run-dir>/discovered_paths.json / <run-dir>/../discovered_paths.json).")
    parser.add_argument("--skip-recon", action="store_true",
                        help="Disable source-recon integration entirely; keep only the legacy "
                             "hardcoded reference-file selection.")
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    protocol_text = EMBEDDED_CODE_MOD_PROTOCOL_V2
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lit_records = _read_json(Path(args.literature), []) if args.literature else []
    if not isinstance(lit_records, list):
        lit_records = []
    starter_dir = Path(args.starter_dir).expanduser()
    if not starter_dir.is_absolute():
        starter_dir = (repo_root / starter_dir).resolve()
    starter_assets = _discover_starter_assets(starter_dir)

    cli_pdfs = [Path(p).expanduser().resolve() for p in args.pdfs if Path(p).expanduser().exists()]
    cli_imgs = [
        Path(p).expanduser().resolve()
        for p in args.equation_images
        if Path(p).expanduser().exists()
    ]
    # Merge user-provided and starter assets.
    pdf_paths = sorted({*cli_pdfs, *starter_assets.get("pdfs", [])}, key=lambda p: str(p))
    eq_img_paths = sorted({*cli_imgs, *starter_assets.get("images", [])}, key=lambda p: str(p))

    base_case, source = _choose_base_case(
        args.topic,
        repo_root,
        run_dir,
        lit_records,
        args.base_case_dir,
        starter_assets.get("cases", []),
    )
    source_case_path = base_case
    working_case = base_case
    copied_for_isolation = False
    if not _case_is_under(base_case, run_dir):
        working_case = run_dir / "canonical_base_case"
        _copy_case_for_working_dir(base_case, working_case)
        copied_for_isolation = True

    pdf_text = _extract_pdf_texts(pdf_paths)
    pdf_text_llm = _extract_pdf_texts_llm(pdf_text, repo_root)
    img_text = _extract_image_texts(eq_img_paths, repo_root)
    starter_text = _extract_starter_text_files(starter_assets.get("text_files", []))
    merged_text = (pdf_text or "") + "\n" + (pdf_text_llm or "") + "\n" + (img_text or "") + "\n" + (starter_text or "")

    # Priority 1: starter_understanding.json produced by the unified LLM folder scan.
    # This contains the full formula/model spec as understood by an LLM that read
    # every file in the starter folder — most reliable source, no format assumptions.
    formula: str = ""
    if args.starter_understanding:
        su_path = Path(args.starter_understanding)
        if su_path.exists():
            try:
                su = json.loads(su_path.read_text(encoding="utf-8"))
                formula = su.get("formula_or_model_spec", "") or ""
                if formula.strip():
                    print(f"[code_mod_prepare] formula from starter_understanding.json "
                          f"({len(formula)} chars, file: {su.get('formula_file')})")
            except Exception as e:
                print(f"[code_mod_prepare] warning: could not read starter_understanding: {e}")

    # Priority 2: raw starter text files passed directly (full content, any format).
    if not formula.strip() and starter_text.strip():
        formula = starter_text[:8000]
        print(f"[code_mod_prepare] formula from raw starter text files ({len(formula)} chars)")

    # Priority 3: heuristic extraction from PDFs/images (last resort).
    if not formula.strip():
        formula = _pick_formula(args.topic, merged_text, max_chars=2000)
        print(f"[code_mod_prepare] formula from _pick_formula fallback ({len(formula)} chars)")
    mode_h, parent_h = _guess_mode_and_parent(args.topic + "\n" + merged_text)
    mode, parent = mode_h, parent_h
    mode_inference = "heuristic"
    llm_mode = _llm_classify_code_mod_mode(args.topic, merged_text, repo_root, mode_h, parent_h)
    if llm_mode:
        mode, parent = llm_mode
        mode_inference = "llm"
    payload = _build_payload(
        repo_root, args.topic, working_case, formula, mode, parent, protocol_text, starter_artifacts_text=starter_text
    )

    recon_meta: Dict[str, Any] = {"enabled": False}
    if not args.skip_recon:
        try:
            recon_ctx, recon_meta = _inject_recon_context(
                run_dir=run_dir,
                recon_cache_arg=args.recon_cache,
                topic=args.topic,
                mode=mode,
                parent=parent,
                formula_text=formula,
                repo_root=repo_root,
            )
        except Exception as exc:
            recon_ctx = None
            recon_meta = {"enabled": True, "status": "error", "error": str(exc)[:400]}
        if recon_ctx:
            api_ctx = payload.setdefault("openfoam_api_context", {})
            api_ctx["recon_reference_context"] = recon_ctx
    dict_files = payload.get("case_snapshot", {}).get("dictionary_files", {}) if isinstance(payload.get("case_snapshot"), dict) else {}
    if isinstance(dict_files, dict):
        dict_chars = sum(len(str(v)) for v in dict_files.values())
        dict_count = len(dict_files)
    else:
        dict_chars = 0
        dict_count = 0

    out = {
        "payload": payload,
        "meta": {
            "base_case_source": source,
            "base_case_path": str(source_case_path),
            "working_case_path": str(working_case),
            "case_copied_for_isolation": copied_for_isolation,
            "dictionary_snapshot_files": list(dict_files.keys()) if isinstance(dict_files, dict) else [],
            "dictionary_snapshot_file_count": dict_count,
            "dictionary_snapshot_total_chars": dict_chars,
            "pdfs_used": [str(p) for p in pdf_paths],
            "equation_images_used": [str(p) for p in eq_img_paths],
            "equation_image_llm_chars": len(img_text or ""),
            "starter_dir": str(starter_dir),
            "starter_case_candidates": [str(p) for p in starter_assets.get("cases", [])],
            "starter_text_files": [str(p) for p in starter_assets.get("text_files", [])],
            "starter_text_chars": len(starter_text or ""),
            "pdf_chars_pypdf": len(pdf_text or ""),
            "pdf_chars_llm_extract": len(pdf_text_llm or ""),
            "mode": mode,
            "mode_inference": mode_inference,
            "mode_heuristic": {"mode": mode_h, "parent_model_hint": parent_h},
            "parent_model_hint": parent,
            "formula_text": formula,
            "protocol_file": "embedded:openfoam_literature_change_agent_prompt_v2",
            "protocol_applied": True,
            "recon": recon_meta,
        },
    }
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["meta"], indent=2))
    return 0


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())


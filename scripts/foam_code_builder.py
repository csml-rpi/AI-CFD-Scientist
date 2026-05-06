#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


def fail(
    status: str,
    reason: str,
    items: List[str],
    *,
    mode_candidate: Optional[str] = None,
    parent_model_candidate: Optional[str] = None,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {"mode_candidate": mode_candidate, "parent_model_candidate": parent_model_candidate}
    return {
        "status": status,
        "reason": reason,
        "missing_or_blocking_items": items,
        "diagnostics": diag,
    }


def safe_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s or "")


def collect_formula_symbols(request: Dict[str, Any]) -> Set[str]:
    """
    Return the set of symbols to validate against the symbol_table.
    Uses only the names explicitly listed in symbol_table — never re-tokenizes
    the formula_text prose (which contains English words, C++ keywords, etc.).
    """
    symbols = request.get("symbol_table") or []
    if isinstance(symbols, list):
        return {s.get("name") for s in symbols if isinstance(s, dict) and s.get("name")}
    return set()


def keywords_builtin() -> Set[str]:
    return {"exp", "log", "sin", "cos", "tan", "pow", "sqrt", "min", "max", "mag", "symm", "grad"}


def _llm_suggest_activation_dictionary_path(
    payload: Dict[str, Any],
    *,
    mode: str,
    formula: str,
    raw_user: str,
) -> str:
    """
    For custom_case_library, pick one snapshot path where the user would typically wire the new .so.
    Returns "" if LLM unavailable or uncertain.
    """
    if mode != "custom_case_library":
        return ""
    snap = payload.get("case_snapshot") if isinstance(payload.get("case_snapshot"), dict) else {}
    dfiles = snap.get("dictionary_files") if isinstance(snap.get("dictionary_files"), dict) else {}
    keys = [k for k in sorted(dfiles.keys()) if isinstance(k, str) and k.strip()]
    if not keys:
        return ""
    try:
        bootstrap_paths()
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
    except Exception:
        return ""

    sys_prompt = (
        "You help wire OpenFOAM 10 case-local shared libraries (loaded via system/controlDict libs).\n"
        "Return STRICT JSON only, no markdown: {\"path\": \"<exact match from list>\"} or {\"path\": null}\n"
        "Choose the single dictionary file (relative path) where this kind of change is normally activated.\n"
        "Examples: constant/momentumTransport (transport/viscosity), constant/fvModels (sources), "
        "constant/turbulenceProperties (RAS/LES), system/fvSchemes (numerics/flux), constant/boundary conditions live under 0/ not here.\n"
        "If no path clearly fits, return null."
    )
    user_payload = {
        "allowed_paths_exact": keys,
        "user_request_excerpt": (raw_user or "")[:12000],
        "formula_text": (formula or "")[:4000],
    }
    try:
        settings = get_settings()
        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        resp = llm.invoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)[:50000]),
            ]
        )
        raw = getattr(resp, "content", "") if resp else ""
        txt = strip_json_fences(raw if isinstance(raw, str) else str(raw))
        s, e = txt.find("{"), txt.rfind("}")
        if s == -1 or e <= s:
            return ""
        obj = json.loads(txt[s : e + 1])
        if not isinstance(obj, dict):
            return ""
        p = obj.get("path")
        if p is None:
            return ""
        ps = str(p).strip().lstrip("./")
        return ps if ps in keys else ""
    except Exception:
        return ""


def validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (success_dict, failure_dict). Exactly one is non-None."""
    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    mode_raw = req.get("declared_mode_hint")
    mode = mode_raw if isinstance(mode_raw, str) else ""
    parent_raw = req.get("parent_model_hint")
    parent = parent_raw if isinstance(parent_raw, str) else None
    diagnostics_mode = mode if mode in {
        "custom_viscosity",
        "custom_turbulence_model_modification",
        "custom_source",
        "custom_case_library",
    } else None
    diagnostics_parent = parent if parent and parent != "unknown" else None

    case_path = payload.get("case_path")
    ov = payload.get("openfoam_version")
    if ov is None or str(ov).strip() == "":
        return None, fail(
            "NEEDS_INFO",
            "openfoam_version is required",
            ["openfoam_version"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )
    openfoam_version = str(ov)
    if openfoam_version != "10":
        return None, fail(
            "UNSUPPORTED",
            "Only OpenFOAM 10 payloads are supported",
            ["openfoam_version must be exactly \"10\""],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )

    if not case_path or not isinstance(case_path, str):
        return None, fail(
            "NEEDS_INFO",
            "Missing case_path",
            ["case_path"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )

    snap = payload.get("case_snapshot") if isinstance(payload.get("case_snapshot"), dict) else {}
    control_text = snap.get("controlDict_text")
    if not control_text or not str(control_text).strip():
        return None, fail(
            "NEEDS_INFO",
            "case_snapshot.controlDict_text is required",
            ["case_snapshot.controlDict_text"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )

    if mode not in {
        "custom_viscosity",
        "custom_turbulence_model_modification",
        "custom_source",
        "custom_case_library",
    }:
        return None, fail(
            "NEEDS_INFO",
            "declared_mode_hint must be a supported mode",
            ["request.declared_mode_hint"],
            mode_candidate=None,
            parent_model_candidate=diagnostics_parent,
        )

    formula = req.get("formula_text")
    if not formula or not str(formula).strip():
        return None, fail(
            "NEEDS_INFO",
            "formula_text is required",
            ["request.formula_text"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )
    formula = str(formula)
    raw_user = str(req.get("raw_user_text") or "")

    region = req.get("region", "all")
    if not isinstance(region, str):
        region = "all"
    if region == "unknown" or not region.strip():
        return None, fail(
            "NEEDS_INFO",
            "region must be explicit (e.g. 'all' or 'cellZone:<name>')",
            ["request.region"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )
    if region.startswith("cellZone:"):
        zone_name = region.split(":", 1)[1].strip()
        zones = snap.get("existing_cellZones") or []
        if zone_name not in zones:
            return None, fail(
                "NEEDS_INFO",
                "Requested cell zone is not listed in case snapshot",
                [f"cellZone:{zone_name}", "case_snapshot.existing_cellZones"],
                mode_candidate=diagnostics_mode,
                parent_model_candidate=diagnostics_parent,
            )

    existing_fields = list(snap.get("existing_fields") or [])
    momentum = str(snap.get("momentumTransport_text") or "")
    turbulence = str(snap.get("turbulenceProperties_text") or "")

    valid_parents = {"SpalartAllmaras", "kEpsilon", "RNGkEpsilon", "realizableKE", "kOmega", "kOmegaSST"}
    api = payload.get("openfoam_api_context") if isinstance(payload.get("openfoam_api_context"), dict) else {}
    header_hints = api.get("header_hints") or []

    caps = payload.get("solver_capabilities") if isinstance(payload.get("solver_capabilities"), dict) else {}

    if mode == "custom_source":
        if not caps.get("supports_fvModels"):
            return None, fail(
                "UNSUPPORTED_WITHOUT_SOLVER_EDIT",
                "Solver must expose fvModels for custom_source",
                ["solver_capabilities.supports_fvModels must be true"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        target_equations = req.get("target_equations") or []
        if not isinstance(target_equations, list):
            target_equations = []
        supported = set(caps.get("fvModels_supported_fields") or [])
        if len(target_equations) != 1:
            return None, fail(
                "NEEDS_INFO",
                "custom_source requires exactly one target equation for fvModels",
                ["request.target_equations"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        if target_equations[0] not in supported:
            return None, fail(
                "UNSUPPORTED_WITHOUT_SOLVER_EDIT",
                "Target equation is not exposed via fvModels for this solver",
                [str(target_equations[0]), "solver_capabilities.fvModels_supported_fields"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        activation_dictionary = "constant/fvModels"

    elif mode == "custom_viscosity":
        if "U" not in existing_fields:
            return None, fail(
                "NEEDS_INFO",
                "Strain-rate or velocity-based viscosity requires U in the case",
                ["case_snapshot.existing_fields must include U"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        if not momentum.strip() and not turbulence.strip():
            return None, fail(
                "NEEDS_INFO",
                "Momentum or turbulence transport dictionary text is required to activate viscosity model",
                ["case_snapshot.momentumTransport_text or turbulenceProperties_text"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        activation_dictionary = "constant/momentumTransport" if momentum.strip() else "constant/turbulenceProperties"
        if not header_hints:
            return None, fail(
                "NEEDS_INFO",
                "openfoam_api_context.header_hints is required to resolve the viscosity model base class",
                ["openfoam_api_context.header_hints"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )

    elif mode == "custom_case_library":
        activation_dictionary = _llm_suggest_activation_dictionary_path(
            payload,
            mode=mode,
            formula=formula,
            raw_user=raw_user,
        )

    else:  # custom_turbulence_model_modification
        if parent is None or parent == "unknown":
            return None, fail(
                "NEEDS_INFO",
                "parent_model_hint must name a supported turbulence parent model",
                ["request.parent_model_hint"],
                mode_candidate=mode,
                parent_model_candidate=None,
            )
        if parent not in valid_parents:
            return None, fail(
                "UNSUPPORTED",
                "parent_model_hint is not in the supported OpenFOAM 10 parent model list",
                ["request.parent_model_hint"],
                mode_candidate=mode,
                parent_model_candidate=parent,
            )
        if not momentum.strip() and not turbulence.strip():
            return None, fail(
                "NEEDS_INFO",
                "Momentum or turbulence transport dictionary is required",
                ["case_snapshot.momentumTransport_text or turbulenceProperties_text"],
                mode_candidate=mode,
                parent_model_candidate=parent,
            )
        term_specs = req.get("term_specs") or []
        if not isinstance(term_specs, list) or not term_specs:
            return None, fail(
                "NEEDS_INFO",
                "term_specs with insertion_site is required for turbulence modifications",
                ["request.term_specs"],
                mode_candidate=mode,
                parent_model_candidate=parent,
            )
        for i, ts in enumerate(term_specs):
            if not isinstance(ts, dict) or not str(ts.get("insertion_site") or "").strip():
                return None, fail(
                    "NEEDS_INFO",
                    "Each term_spec must include a non-empty insertion_site",
                    [f"request.term_specs[{i}].insertion_site"],
                    mode_candidate=mode,
                    parent_model_candidate=parent,
                )
            tform = ts.get("formula_text")
            if not tform or not str(tform).strip():
                return None, fail(
                    "NEEDS_INFO",
                    "Each term_spec must include formula_text",
                    [f"request.term_specs[{i}].formula_text"],
                    mode_candidate=mode,
                    parent_model_candidate=parent,
                )
        activation_dictionary = "constant/momentumTransport" if momentum.strip() else "constant/turbulenceProperties"
        hinted = False
        if isinstance(header_hints, list):
            for h in header_hints:
                if isinstance(h, dict) and h.get("class_name") == parent:
                    hinted = True
                    break
        if not hinted:
            return None, fail(
                "NEEDS_INFO",
                "openfoam_api_context.header_hints must include the parent model class for inheritance",
                [f"header hint for {parent}"],
                mode_candidate=mode,
                parent_model_candidate=parent,
            )

        def _fields_ok(_parent: str) -> bool:
            if _parent == "SpalartAllmaras":
                return "nuTilda" in existing_fields
            if _parent in {"kEpsilon", "RNGkEpsilon", "realizableKE"}:
                return "k" in existing_fields and "epsilon" in existing_fields
            if _parent in {"kOmega", "kOmegaSST"}:
                return "k" in existing_fields and "omega" in existing_fields
            return False

        if not _fields_ok(parent):
            return None, fail(
                "NEEDS_INFO",
                "Required turbulence fields for the parent model must appear in existing_fields",
                ["case_snapshot.existing_fields"],
                mode_candidate=mode,
                parent_model_candidate=parent,
            )

    symbols = req.get("symbol_table") or []
    if not isinstance(symbols, list):
        symbols = []
    symbol_map: Dict[str, Any] = {s.get("name"): s for s in symbols if isinstance(s, dict) and s.get("name")}
    formula_symbols = collect_formula_symbols(req)
    const_list = req.get("constants") or []
    if not isinstance(const_list, list):
        const_list = []
    const_names = {c.get("name") for c in const_list if isinstance(c, dict)}
    kw = keywords_builtin()
    unknown = sorted(s for s in formula_symbols if s not in symbol_map and s not in const_names and s not in kw)
    if unknown and mode != "custom_case_library":
        return None, fail(
            "NEEDS_INFO",
            "Symbols in formula_text / term_specs are not declared in symbol_table",
            unknown,
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )
    unavailable = sorted(
        n for n in formula_symbols if n in symbol_map and symbol_map[n].get("availability") != "available"
    )
    if unavailable:
        return None, fail(
            "NEEDS_INFO",
            "Some symbols are not available in this case",
            unavailable,
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )
    bad_constants = [c for c in const_list if not isinstance(c, dict) or not all(k in c for k in ("name", "value", "units"))]
    if bad_constants:
        return None, fail(
            "NEEDS_INFO",
            "Constants must include name, value, and units",
            ["request.constants"],
            mode_candidate=diagnostics_mode,
            parent_model_candidate=diagnostics_parent,
        )

    prefix = "Custom"
    bp = payload.get("build_policy") if isinstance(payload.get("build_policy"), dict) else {}
    if isinstance(bp.get("naming_prefix"), str) and bp["naming_prefix"].strip():
        prefix = bp["naming_prefix"].strip()
    mode_short = (
        "Src"
        if mode == "custom_source"
        else "Nu"
        if mode == "custom_viscosity"
        else "Lib"
        if mode == "custom_case_library"
        else (parent or "Model")
    )
    raw_user = str(req.get("raw_user_text") or "")
    words = re.findall(r"[A-Za-z0-9]+", raw_user)[:3]
    short_change = "".join(w.capitalize() for w in words) or "Change"
    class_name = f"{prefix}{mode_short}{short_change}"
    library_relpath = f"customModels/{class_name}/lib{class_name}.so"

    target_equations_out = req.get("target_equations")
    if not isinstance(target_equations_out, list):
        target_equations_out = []

    dict_patches: List[Dict[str, str]] = [
        {
            "path": "system/controlDict",
            "description": (
                f"Ensure libs loads the case-local shared object (short name lib{class_name}.so is OK; "
                "apply_compile normalizes to a path under customModels/.../platforms/.../lib/ if needed)."
            ),
        }
    ]
    verify_cmds: List[str] = ["foamDictionary system/controlDict -entry libs"]
    act_rel = str(activation_dictionary or "").strip()
    if act_rel:
        if mode == "custom_viscosity":
            act_desc = (
                f"In {act_rel}: laminar generalisedNewtonian with viscosityModel {class_name}; "
                f"put {class_name}Coeffs inside the laminar{{}} block (OpenFOAM 10); "
                "coefficient keywords must match the C++ read()."
            )
        elif mode == "custom_turbulence_model_modification":
            act_desc = (
                f"In {act_rel}: switch the turbulence model to {class_name} "
                f"(replacing parent {parent}); preserve sub-dictionaries the C++ expects."
            )
        elif mode == "custom_source":
            act_desc = f"In {act_rel}: add an fvModel entry activating {class_name} per OpenFOAM 10."
        else:
            act_desc = f"In {act_rel}: wire {class_name} per OpenFOAM rules for this change type."
        dict_patches.append({"path": act_rel, "description": act_desc})
        verify_cmds.append(f"foamDictionary {act_rel}")
    else:
        dict_patches.append(
            {
                "path": "(manual)",
                "description": (
                    "No fixed activation path for custom_case_library: edit the appropriate "
                    "constant/ or system/ dictionaries (fvModels, fvSchemes, momentumTransport, "
                    "boundary conditions, etc.) so the compiled library is used."
                ),
            }
        )

    ok: Dict[str, Any] = {
        "status": "OK",
        "mode": mode,
        "parent_model": parent if mode == "custom_turbulence_model_modification" else None,
        "class_name": class_name,
        "library_relpath": library_relpath,
        "activation_dictionary": activation_dictionary,
        "normalized_spec": {
            "summary": raw_user or formula,
            "region": region,
            "target_equations": target_equations_out,
            "symbols": symbols,
            "constants": const_list,
            "formula_text": formula,
            "term_specs": req.get("term_specs") or [],
        },
        "files_to_create": [
            {"path": f"{case_path}/customModels/{class_name}/{class_name}.H", "description": "C++ header — CC will write this"},
            {"path": f"{case_path}/customModels/{class_name}/{class_name}.C", "description": "C++ implementation — CC will write this"},
            {"path": f"{case_path}/customModels/{class_name}/Make/files", "description": "wmake files — CC will write this"},
            {"path": f"{case_path}/customModels/{class_name}/Make/options", "description": "wmake options — CC will write this"},
        ],
        "dictionary_patches": dict_patches,
        "build_commands": [f"cd {case_path}/customModels/{class_name}", "wmake libso"],
        "verification_commands": verify_cmds,
    }
    return ok, None


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Validate custom OpenFOAM model payload.")
    parser.add_argument("--payload", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    p = Path(args.payload)
    try:
        raw = p.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        result = fail("NEEDS_INFO", "Invalid or unreadable payload JSON", ["payload"])
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    if not isinstance(payload, dict):
        result = fail("NEEDS_INFO", "Payload must be a JSON object", ["payload"])
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    success, failure = validate_payload(payload)
    result = success if success is not None else failure
    assert result is not None
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

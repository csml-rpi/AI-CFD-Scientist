#!/usr/bin/env python3
"""
Runtime (no-wmake) code modifications for OpenFOAM.

Generic across modification kinds — works identically for turbulence model changes,
source-term add/delete, numerical method swaps, transport property models, thermal
property models, BC variants, multiphase coupling corrections, scalar transport.

Categories handled:
  - runtime_source : codedFvOption / scalarCodedSource / vectorCodedSource — adds a
                     source term to ANY transport equation (turbulence fields, U, T,
                     scalar fields, alpha, ...). Generated entry goes into
                     constant/fvOptions (or constant/fvModels if that's what the case
                     already uses). OpenFOAM JIT-compiles the snippet at solver start.
  - runtime_bc     : codedFixedValue / codedMixed — replaces a boundary condition
                     entry in a 0/<field> file with a JIT-compiled expression.
  - runtime_field  : coded function object — adds a derived field / sensor / QoI
                     calculator into system/controlDict.functions.
  - dict_only      : pure dictionary edits. Applies a list of (target_file,
                     unified_diff) or (target_file, key_path, new_value) operations
                     against the case. Covers numerical scheme swaps, solver
                     parameter changes, model swaps to existing built-in models,
                     transport / thermo property dictionary edits.

None of these touch Make/files, Make/options, wmake, or any C++ library. They are
order-of-magnitude faster than the class-derivation path and have essentially zero
build-environment failure surface.

Input action shape (subset; only fields relevant to the chosen category are read):

  {
    "modification_category":   "runtime_source"|"runtime_bc"|"runtime_field"|"dict_only",
    "model_description":       "human-readable label",
    "variant_name":            "short slug",
    "runtime_source": {
      "name":                  "<unique fvOption entry name>",
      "value_type":            "scalar"|"vector"|"tensor",
      "target_field":          "<field name, e.g. nuTilda, U, T, alpha.water>",
      "selection_mode":        "all"|"cellSet"|"cellZone"  (default "all"),
      "selection_arg":         "<cellSet/cellZone name when selection_mode != all>",
      "code_include":          "<extra #include lines (optional)>",
      "code_add_sup":          "<C++ body. Variable named 'eqn' is a fvMatrix; assign or +=/-=>",
      "code_add_sup_rho":      "<optional, for rho-aware solvers>",
      "code_constrain":        "<optional>",
      "code_correct":          "<optional, runs once per outer iteration>",
      "coefficients":          {"name": value, ...}    # exposed as scalars in code
    },
    "runtime_bc": {
      "field_file":            "0/<field>"  (or "0.orig/<field>"),
      "patch_name":            "<patch on which to install the coded BC>",
      "bc_type":               "codedFixedValue"|"codedMixed",
      "name":                  "<unique BC name>",
      "code_include":          "<#include lines (optional)>",
      "code":                  "<C++ body assigning to operator==(...) or refValue/refGrad/valueFraction>",
      "value_default":         "uniform 0"  (string; placed under 'value' for fixedValue)
    },
    "runtime_field": {
      "name":                  "<function-object name>",
      "code_include":          "<optional>",
      "code_execute":          "<C++ body>",
      "code_data":             "<optional persistent member declarations>",
      "libs":                  ["libutilityFunctionObjects.so"]   # default if absent
    },
    "dict_only": {
      "edits": [
        {
          "target":            "<relative path inside case, e.g. system/fvSchemes>",
          "unified_diff":      "<diff text against the current file content>",
          "key_path":          "<dot-or-slash path, optional alternative to diff>",
          "new_value":         "<value when key_path is used>"
        },
        ...
      ]
    }
  }

Output result JSON (compatible with the existing OED handoff):

  {
    "status":                  "OK"|"FAILED",
    "category":                "runtime_source"|...,
    "case_dir":                "<absolute path to applied case>",
    "class_name":              "runtime:<category>:<slug>",
    "compile_ok":              true,            # always true for runtime categories
    "compiled_model_name":     "runtime:<slug>",
    "applied_files":           [...],
    "error":                   "<message when status=FAILED>"
  }

CLI:
  python scripts/code_mod_runtime.py \
      --action     <action.json> \
      --base-case  <base_case_dir> \
      --iter-dir   <iter_dir> \
      --output     <result.json>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RUNTIME_CATEGORIES = (
    "runtime_source",
    "runtime_bc",
    "runtime_field",
    "dict_only",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _slug(s: str, n: int = 32) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(s or "")).strip("_")
    return s[:n] or "unnamed"


def _copy_case(src: Path, dst: Path) -> None:
    """Copy a case dir, skipping bulky postProcessing/log artifacts."""
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(directory: str, names: List[str]) -> List[str]:
        skip = []
        for n in names:
            if n in {"postProcessing", "processor*"}:
                skip.append(n)
            if n.startswith("log.") or n in {"Allrun.out", "Allrun.err"}:
                skip.append(n)
        return skip

    shutil.copytree(str(src), str(dst), ignore=ignore, dirs_exist_ok=False)


def _resolve_case_dir(case_dir: Path) -> Path:
    """If `case_dir` is a wrapper (no constant/), descend into the unique child case."""
    if (case_dir / "constant").is_dir() or (case_dir / "system").is_dir():
        return case_dir
    children = [p for p in case_dir.iterdir() if p.is_dir()]
    for c in children:
        if (c / "constant").is_dir() or (c / "system").is_dir():
            return c
    return case_dir


# ---------------------------------------------------------------------------
# OpenFOAM dictionary helpers
# ---------------------------------------------------------------------------

_FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
"""


def _foam_file_header(class_name: str, location: str, object_name: str) -> str:
    return (
        _FOAM_HEADER
        + "FoamFile\n{\n"
        + "    version     2.0;\n"
        + "    format      ascii;\n"
        + f"    class       {class_name};\n"
        + f'    location    "{location}";\n'
        + f"    object      {object_name};\n"
        + "}\n"
        + "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )


def _detect_openfoam_flavor() -> str:
    """
    Return "of-org" (OpenFOAM Foundation, e.g. OF-10/11) or "esi" (ESI, e.g. v2306).
    Detection priority:
      1) Environment: WM_PROJECT_VERSION or WM_PROJECT_DIR contents
      2) Default to "of-org" (current cfd-scientist target)

    The two flavors disagree on coded fvModel/fvOption syntax:
      of-org 10+:  type coded; field <name>;       (in constant/fvModels)
      esi:         type scalarCodedSource; fields (<name>);  (in constant/fvOptions)
    """
    import os
    wm_dir = os.environ.get("WM_PROJECT_DIR", "")
    ver = os.environ.get("WM_PROJECT_VERSION", "")
    # ESI versions look like "v2306", "v2312"; OF-org versions look like "10", "11"
    if ver.startswith("v"):
        return "esi"
    if ver and ver[0].isdigit():
        return "of-org"
    if wm_dir and "openfoam" in wm_dir.lower():
        # Heuristic: ESI installations typically have "OpenFOAM-v" in the path
        if "openfoam-v" in wm_dir.lower() or wm_dir.lower().endswith("-v2306"):
            return "esi"
    return "of-org"


def _resolve_fvoptions_path(case_dir: Path) -> Tuple[Path, str]:
    """
    Locate (or initialize) the right source-term dictionary for this case.
    Returns (path, kind) where kind is "fvModels" (OF-org 10+) or "fvOptions" (ESI).
    Picks an existing file if present; otherwise creates the right one for the
    detected flavor.
    """
    cand = [
        (case_dir / "constant" / "fvModels", "fvModels"),
        (case_dir / "constant" / "fvOptions", "fvOptions"),
        (case_dir / "system" / "fvOptions", "fvOptions"),
    ]
    for p, kind in cand:
        if p.is_file():
            return p, kind
    flavor = _detect_openfoam_flavor()
    if flavor == "of-org":
        target = case_dir / "constant" / "fvModels"
        kind = "fvModels"
    else:
        target = case_dir / "constant" / "fvOptions"
        kind = "fvOptions"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _foam_file_header("dictionary", "constant", kind)
        + "\n// (generated by code_mod_runtime)\n",
        encoding="utf-8",
    )
    return target, kind


def _ensure_function_objects_block(controldict: Path) -> None:
    """Make sure system/controlDict has a `functions { ... }` block."""
    txt = controldict.read_text(encoding="utf-8")
    if re.search(r"^\s*functions\s*\{", txt, re.MULTILINE):
        return
    txt = txt.rstrip() + "\n\nfunctions\n{\n}\n"
    controldict.write_text(txt, encoding="utf-8")


def _append_block_to_dictionary(path: Path, block: str) -> None:
    """Append a free-standing block at the top level of a dictionary file."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_foam_file_header("dictionary", path.parent.name, path.name) + "\n", encoding="utf-8")
    txt = path.read_text(encoding="utf-8").rstrip()
    path.write_text(txt + "\n\n" + block.rstrip() + "\n", encoding="utf-8")


def _insert_into_functions_block(controldict: Path, fo_name: str, fo_body: str) -> None:
    """Insert a function-object entry into system/controlDict.functions{...}."""
    _ensure_function_objects_block(controldict)
    txt = controldict.read_text(encoding="utf-8")
    # find the last '}' of the functions block
    m = re.search(r"(functions\s*\{)", txt)
    if not m:
        # safety: shouldn't happen because _ensure_function_objects_block ran
        controldict.write_text(txt.rstrip() + f"\n\nfunctions\n{{\n{fo_body}\n}}\n", encoding="utf-8")
        return
    start = m.end()
    depth = 1
    i = start
    while i < len(txt) and depth > 0:
        c = txt[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # i now sits at the closing '}' of functions block
    insert_at = i
    new_txt = txt[:insert_at] + ("    " + fo_body.replace("\n", "\n    ")).rstrip() + "\n" + txt[insert_at:]
    controldict.write_text(new_txt, encoding="utf-8")


def _find_field_file(case_dir: Path, field_path_or_name: str) -> Optional[Path]:
    """Find the canonical field file given a full relative path or just a name."""
    p = (case_dir / field_path_or_name).resolve()
    if p.is_file():
        return p
    # fall back: search 0/, 0.orig/
    for ts_dir in ("0", "0.orig"):
        cand = case_dir / ts_dir / Path(field_path_or_name).name
        if cand.is_file():
            return cand
    return None


def _replace_patch_block(text: str, patch_name: str, new_block: str) -> Optional[str]:
    """
    Replace the entry for `patch_name` inside the boundaryField{} block of an
    OpenFOAM field file with `new_block`. Generic — works for any field type.
    Returns the new text, or None if the patch entry wasn't found.
    """
    bf_match = re.search(r"boundaryField\s*\{", text)
    if not bf_match:
        return None
    bf_start = bf_match.end()
    # Locate end of boundaryField{}
    depth = 1
    i = bf_start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    bf_end = i
    bf_body = text[bf_start:bf_end]

    # Find patch_name { ... } inside bf_body
    pat = re.search(rf"\b{re.escape(patch_name)}\s*\{{", bf_body)
    if not pat:
        return None
    p_start = pat.start()
    open_brace = pat.end()
    depth = 1
    j = open_brace
    while j < len(bf_body) and depth > 0:
        c = bf_body[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    p_end = j + 1  # include closing }
    new_bf = bf_body[:p_start] + new_block.rstrip() + "\n" + bf_body[p_end:]
    return text[:bf_start] + new_bf + text[bf_end:]


# ---------------------------------------------------------------------------
# category-specific renderers
# ---------------------------------------------------------------------------

def _render_coded_source_block(spec: Dict[str, Any], flavor: Optional[str] = None) -> str:
    """Render a coded source-term entry. Flavor-aware:
      of-org (default for OF-10): `type coded; field <name>;` style — emitted into constant/fvModels.
      esi:                         `type scalarCodedSource; fields (<name>);` style — emitted into constant/fvOptions.
    """
    if flavor is None:
        flavor = _detect_openfoam_flavor()
    name = _slug(spec.get("name") or "customSource")
    value_type = (spec.get("value_type") or "scalar").strip().lower()
    if value_type not in ("scalar", "vector", "tensor"):
        value_type = "scalar"
    target_field = str(spec.get("target_field") or "").strip()
    if not target_field:
        raise ValueError("runtime_source: target_field is required")
    selection_mode = (spec.get("selection_mode") or "all").strip()
    selection_arg = (spec.get("selection_arg") or "").strip()
    code_include = spec.get("code_include") or ""
    code_add_sup = spec.get("code_add_sup") or ""
    code_add_sup_rho = spec.get("code_add_sup_rho") or ""
    code_constrain = spec.get("code_constrain") or ""
    code_correct = spec.get("code_correct") or ""
    coeffs = spec.get("coefficients") or {}

    # Coefficient passing: emit numeric coefficients DIRECTLY at the top level
    # of the fvModel/fvOption entry (siblings to `selectionMode`, `field`).
    # The LLM's code body reads them via OpenFOAM's standard pattern:
    #   coeffs().lookupOrDefault<scalar>("Csink", 1.0)
    # We do NOT use a `<name>Coeffs { ... }` subdict here because in OF-org 10
    # the codedFvModel framework sometimes interprets such a subdict as a
    # nested model config (looking inside it for selectionMode/field), which
    # crashes with "keyword X undefined in <name>Coeffs". Top-level coeffs
    # work for both OF-org and ESI flavors.
    coeffs_lines = ""
    if coeffs:
        items = []
        for k, v in coeffs.items():
            if not re.match(r"^[A-Za-z_]\w*$", str(k)):
                continue
            try:
                fv = float(v)
                items.append(f"    {k}    {fv};")
            except Exception:
                continue
        if items:
            coeffs_lines = "\n".join(items) + "\n"

    sel_line = f"    selectionMode   {selection_mode};\n"
    if selection_mode in ("cellSet", "cellZone") and selection_arg:
        if selection_mode == "cellSet":
            sel_line += f"    cellSet         {selection_arg};\n"
        else:
            sel_line += f"    cellZone        {selection_arg};\n"

    if flavor == "of-org":
        # OpenFOAM Foundation (OF-10/11) syntax. Goes into constant/fvModels.
        # Hooks: codeAddSup / codeAddRhoSup / codeAddAlphaRhoSup.
        # `field` is singular and unparenthesized.
        block = (
            f"{name}\n"
            f"{{\n"
            f"    type            coded;\n"
            f"{sel_line}"
            f"    field           {target_field};\n"
            f"\n"
            f"{coeffs_lines}"
            f"\n"
            f"    codeInclude\n"
            f"    #{{\n"
            f"{(code_include or '').rstrip()}\n"
            f"    #}};\n"
            f"\n"
            f"    codeAddSup\n"
            f"    #{{\n"
            f"{(code_add_sup or '').rstrip()}\n"
            f"    #}};\n"
            f"\n"
            f"    codeAddRhoSup\n"
            f"    #{{\n"
            f"{(code_add_sup_rho or '').rstrip()}\n"
            f"    #}};\n"
            f"\n"
            f"    codeAddAlphaRhoSup\n"
            f"    #{{\n"
            f"\n"
            f"    #}};\n"
            f"}}\n"
        )
        return block

    # ESI / older syntax. Goes into constant/fvOptions.
    block = (
        f"{name}\n"
        f"{{\n"
        f"    type            {value_type}CodedSource;\n"
        f"    active          true;\n"
        f"{sel_line}"
        f"    fields          ({target_field});\n"
        f"    name            {name};\n"
        f"\n"
        f"{coeffs_lines}"
        f"\n"
        f"    codeInclude\n"
        f"    #{{\n"
        f"{(code_include or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeCorrect\n"
        f"    #{{\n"
        f"{(code_correct or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeAddSup\n"
        f"    #{{\n"
        f"{(code_add_sup or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeAddSupRho\n"
        f"    #{{\n"
        f"{(code_add_sup_rho or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeConstrain\n"
        f"    #{{\n"
        f"{(code_constrain or '').rstrip()}\n"
        f"    #}};\n"
        f"}}\n"
    )
    return block


def _render_coded_bc_block(spec: Dict[str, Any]) -> str:
    """Render a coded BC body that goes inside `<patch> { ... }` of boundaryField."""
    name = _slug(spec.get("name") or "customBC")
    bc_type = (spec.get("bc_type") or "codedFixedValue").strip()
    if bc_type not in ("codedFixedValue", "codedMixed"):
        bc_type = "codedFixedValue"
    code_include = spec.get("code_include") or ""
    code = spec.get("code") or ""
    value_default = spec.get("value_default") or "uniform 0"
    body = (
        f"    {{\n"
        f"        type            {bc_type};\n"
        f"        value           {value_default};\n"
        f"        name            {name};\n"
        f"        codeInclude\n"
        f"        #{{\n"
        f"{(code_include or '').rstrip()}\n"
        f"        #}};\n"
        f"        code\n"
        f"        #{{\n"
        f"{(code or '').rstrip()}\n"
        f"        #}};\n"
        f"    }}\n"
    )
    return body


def _render_coded_function_object(spec: Dict[str, Any]) -> str:
    name = _slug(spec.get("name") or "customField")
    libs = spec.get("libs") or ["libutilityFunctionObjects.so"]
    libs_str = " ".join(f'"{l}"' for l in libs)
    code_include = spec.get("code_include") or ""
    code_execute = spec.get("code_execute") or ""
    code_data = spec.get("code_data") or ""
    body = (
        f"{name}\n"
        f"{{\n"
        f"    type            coded;\n"
        f"    libs            ({libs_str});\n"
        f"    name            {name};\n"
        f"\n"
        f"    codeInclude\n"
        f"    #{{\n"
        f"{(code_include or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeData\n"
        f"    #{{\n"
        f"{(code_data or '').rstrip()}\n"
        f"    #}};\n"
        f"\n"
        f"    codeExecute\n"
        f"    #{{\n"
        f"{(code_execute or '').rstrip()}\n"
        f"    #}};\n"
        f"}}\n"
    )
    return body


# ---------------------------------------------------------------------------
# unified-diff applier (for dict_only edits)
# ---------------------------------------------------------------------------

def _apply_unified_diff_to_file(target: Path, diff_text: str) -> Tuple[bool, str]:
    """
    Apply a unified diff against `target` using the system `patch` tool.
    Returns (success, message). Tolerant of fuzz/whitespace.
    """
    if not target.is_file():
        return False, f"target file not found: {target}"
    diff_path = target.with_suffix(target.suffix + ".rt.diff")
    diff_path.write_text(diff_text, encoding="utf-8")
    cwd = target.parent
    try:
        # patch -p1 first (handles diffs with 'a/<file> b/<file>' headers); if that
        # fails try -p0.
        for p in (1, 0):
            cmd = ["patch", f"-p{p}", "--fuzz=4", "--ignore-whitespace",
                   "-l", "-N", target.name]
            r = subprocess.run(
                cmd, cwd=str(cwd), input=diff_text, text=True,
                capture_output=True, check=False,
            )
            if r.returncode == 0:
                return True, f"patch -p{p} ok"
        return False, f"patch failed: {r.stdout}\n{r.stderr}"
    finally:
        try:
            diff_path.unlink()
        except Exception:
            pass


def _apply_dict_edit(case_dir: Path, edit: Dict[str, Any]) -> Tuple[bool, str, Optional[Path]]:
    target_rel = (edit.get("target") or "").strip()
    if not target_rel:
        return False, "edit missing 'target'", None
    target = (case_dir / target_rel).resolve()
    if case_dir.resolve() not in target.parents and target != case_dir.resolve():
        return False, f"target outside case dir: {target}", None
    if not target.is_file():
        # generic: dict_only is allowed to create new dictionary files (e.g. fvModels)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _foam_file_header("dictionary", target.parent.name, target.name),
            encoding="utf-8",
        )

    diff_text = edit.get("unified_diff") or ""
    if diff_text.strip():
        ok, msg = _apply_unified_diff_to_file(target, diff_text)
        return ok, msg, target

    # JSON-style key_path / new_value substitution (single key, leaf value).
    key_path = (edit.get("key_path") or "").strip()
    new_value = edit.get("new_value")
    if key_path and new_value is not None:
        return _apply_key_path_edit(target, key_path, str(new_value))

    # full-content replace as last resort
    new_content = edit.get("new_content")
    if isinstance(new_content, str) and new_content.strip():
        target.write_text(new_content, encoding="utf-8")
        return True, "full-content replaced", target

    return False, "edit has no unified_diff / key_path / new_content", target


def _apply_key_path_edit(target: Path, key_path: str, new_value: str) -> Tuple[bool, str, Path]:
    """
    Replace `key value;` by `key new_value;` inside an OpenFOAM dictionary.
    Supports flat keys and one-level nested keys (e.g. "ddtSchemes/default").
    """
    text = target.read_text(encoding="utf-8")
    parts = [p.strip() for p in re.split(r"[/.]", key_path) if p.strip()]
    if not parts:
        return False, "empty key_path", target

    if len(parts) == 1:
        key = parts[0]
        # Match `key  <value>;` allowing arbitrary spacing
        pat = re.compile(rf"(^|\n)(\s*){re.escape(key)}\s+([^;]+);", re.MULTILINE)
        replaced = [False]

        def _sub(m: re.Match[str]) -> str:
            replaced[0] = True
            return f"{m.group(1)}{m.group(2)}{key}    {new_value};"

        new_text = pat.sub(_sub, text, count=1)
        if not replaced[0]:
            # append
            new_text = text.rstrip() + f"\n{key}    {new_value};\n"
        target.write_text(new_text, encoding="utf-8")
        return True, f"key '{key}' set", target

    # nested: locate `parts[0] { ... parts[1] value; ... }` block
    parent = parts[0]
    leaf = parts[-1]
    block_pat = re.compile(rf"(^|\n)(\s*){re.escape(parent)}\s*\{{", re.MULTILINE)
    m = block_pat.search(text)
    if not m:
        return False, f"parent block '{parent}' not found", target
    block_start = m.end()
    depth = 1
    i = block_start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block_end = i
    inner = text[block_start:block_end]
    leaf_pat = re.compile(rf"(^|\n)(\s*){re.escape(leaf)}\s+([^;]+);", re.MULTILINE)
    new_inner, n = leaf_pat.subn(
        lambda mm: f"{mm.group(1)}{mm.group(2)}{leaf}    {new_value};",
        inner, count=1,
    )
    if n == 0:
        new_inner = inner.rstrip() + f"\n    {leaf}    {new_value};\n"
    new_text = text[:block_start] + new_inner + text[block_end:]
    target.write_text(new_text, encoding="utf-8")
    return True, f"nested key '{key_path}' set", target


# ---------------------------------------------------------------------------
# top-level apply
# ---------------------------------------------------------------------------

def apply(action: Dict[str, Any], base_case: Path, iter_dir: Path) -> Dict[str, Any]:
    """Apply a runtime-category modification. Returns result dict."""
    category = (action.get("modification_category") or "").strip().lower()
    if category not in RUNTIME_CATEGORIES:
        return {
            "status": "FAILED",
            "error": f"unsupported runtime category: {category!r}",
            "category": category,
        }

    iter_dir.mkdir(parents=True, exist_ok=True)
    work_case = iter_dir / "canonical_base_case"
    src = _resolve_case_dir(Path(base_case).resolve())
    _copy_case(src, work_case)
    case_dir = _resolve_case_dir(work_case)

    slug = _slug(action.get("variant_name") or action.get("model_description") or category)
    applied_files: List[str] = []

    try:
        if category == "runtime_source":
            spec = action.get("runtime_source") or {}
            block = _render_coded_source_block(spec)
            fvopts, _kind = _resolve_fvoptions_path(case_dir)
            _append_block_to_dictionary(fvopts, block)
            applied_files.append(str(fvopts))

        elif category == "runtime_bc":
            spec = action.get("runtime_bc") or {}
            field_file_rel = (spec.get("field_file") or "").strip()
            patch_name = (spec.get("patch_name") or "").strip()
            if not field_file_rel or not patch_name:
                return {
                    "status": "FAILED",
                    "error": "runtime_bc requires field_file and patch_name",
                    "category": category,
                }
            field_path = _find_field_file(case_dir, field_file_rel)
            if field_path is None:
                return {
                    "status": "FAILED",
                    "error": f"field file not found: {field_file_rel}",
                    "category": category,
                }
            new_block = _render_coded_bc_block(spec)
            text = field_path.read_text(encoding="utf-8")
            new_text = _replace_patch_block(text, patch_name, f"    {patch_name}\n" + new_block)
            if new_text is None:
                return {
                    "status": "FAILED",
                    "error": f"patch '{patch_name}' not found in {field_path}",
                    "category": category,
                }
            field_path.write_text(new_text, encoding="utf-8")
            applied_files.append(str(field_path))

        elif category == "runtime_field":
            spec = action.get("runtime_field") or {}
            controldict = case_dir / "system" / "controlDict"
            if not controldict.is_file():
                return {
                    "status": "FAILED",
                    "error": "system/controlDict not found",
                    "category": category,
                }
            fo_body = _render_coded_function_object(spec)
            _insert_into_functions_block(controldict, _slug(spec.get("name") or "customField"), fo_body)
            applied_files.append(str(controldict))

        elif category == "dict_only":
            spec = action.get("dict_only") or {}
            edits = spec.get("edits") or []
            if not isinstance(edits, list) or not edits:
                return {
                    "status": "FAILED",
                    "error": "dict_only requires non-empty edits[]",
                    "category": category,
                }
            errors: List[str] = []
            for idx, ed in enumerate(edits):
                if not isinstance(ed, dict):
                    errors.append(f"edit[{idx}]: not a dict")
                    continue
                ok, msg, t = _apply_dict_edit(case_dir, ed)
                if not ok:
                    errors.append(f"edit[{idx}]: {msg}")
                else:
                    if t is not None:
                        applied_files.append(str(t))
            if errors:
                return {
                    "status": "FAILED",
                    "error": "; ".join(errors)[:1000],
                    "category": category,
                    "applied_files": applied_files,
                }

    except Exception as exc:
        return {
            "status": "FAILED",
            "error": f"runtime apply exception: {exc}",
            "category": category,
            "applied_files": applied_files,
        }

    return {
        "status": "OK",
        "category": category,
        "case_dir": str(case_dir),
        "class_name": f"runtime:{category}:{slug}",
        "compile_ok": True,
        "compiled_model_name": f"runtime_{slug}",
        "compiled_model_description": str(action.get("model_description", ""))[:240],
        "applied_files": applied_files,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--action", required=True, type=str)
    parser.add_argument("--base-case", required=True, type=str)
    parser.add_argument("--iter-dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    action_path = Path(args.action).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    iter_dir = Path(args.iter_dir).expanduser().resolve()
    base_case = Path(args.base_case).expanduser().resolve()

    if not action_path.is_file():
        out = {"status": "FAILED", "error": f"action file not found: {action_path}"}
        _write_json(out_path, out)
        return 1
    if not base_case.exists():
        out = {"status": "FAILED", "error": f"base_case not found: {base_case}"}
        _write_json(out_path, out)
        return 1

    action = _read_json(action_path, {}) or {}
    if not isinstance(action, dict):
        out = {"status": "FAILED", "error": "action JSON must be an object"}
        _write_json(out_path, out)
        return 1

    result = apply(action, base_case=base_case, iter_dir=iter_dir)
    _write_json(out_path, result)
    print(json.dumps({"status": result.get("status"),
                      "category": result.get("category"),
                      "case_dir": result.get("case_dir"),
                      "class_name": result.get("class_name"),
                      "applied_files": result.get("applied_files", []),
                      "error": result.get("error", "")}, indent=2, default=str))
    return 0 if result.get("status") == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())

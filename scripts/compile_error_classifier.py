#!/usr/bin/env python3
"""
Classify wmake / gcc / linker error blobs into focused tiers and extract the
key actionable signal. The rest of the pipeline routes each tier to the
cheapest handler that can fix it (deterministic fixer, narrow LLM call, or
broader LLM call), instead of dumping 1500 lines of stderr at the model.

Tiers:
  L0 build_config : Make/files name vs class mismatch, missing
                    Make/options entry, EXE_INC pointing at a path that
                    doesn't exist, wmake invocation issue. Deterministic
                    fixers usually suffice.
  L1 parse        : Syntactic C++ errors (expected ';', unmatched braces).
                    Recoverable with a one-shot narrow LLM call.
  L2 lookup       : Unknown identifier / missing header / undeclared symbol.
                    LLM call with header-suggestion list.
  L3 type         : No matching function call, no operator overload,
                    `tmp<>` arithmetic mismatches, dimension errors,
                    template instantiation failure. LLM call with explicit
                    OpenFOAM-arithmetic coaching.
  L4 link         : Undefined reference / unresolved symbol. Deterministic
                    LIB_LIBS auto-add with LLM fallback.
  L5 environment  : OpenFOAM/wmake env not sourced, wmake not on PATH,
                    WM_PROJECT_DIR unset. Deterministic.
  L9 unknown      : Couldn't classify — fall back to legacy path.

Output is a structured dict that the compile-fix loop can consume.

Generic across CFD modification kinds — this module makes no assumptions
about the physics being implemented.

CLI:
  python scripts/compile_error_classifier.py --stderr <log_file>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


TIERS = ("L0", "L1", "L2", "L3", "L4", "L5", "L9")


# ---------------------------------------------------------------------------
# tier patterns — ordered: most specific first
# ---------------------------------------------------------------------------

_PAT_L5_ENV = [
    re.compile(r"wmake: command not found", re.IGNORECASE),
    re.compile(r"WM_PROJECT_DIR.*not set", re.IGNORECASE),
    re.compile(r"foamEtcFile: command not found", re.IGNORECASE),
    re.compile(r"OpenFOAM.*not sourced", re.IGNORECASE),
]

_PAT_L0_CONFIG = [
    re.compile(r"No such file or directory.*Make/(files|options)", re.IGNORECASE),
    re.compile(r"Could not find Make/(files|options)", re.IGNORECASE),
    re.compile(r"^make.*\*\*\* No rule to make target.*Stop\.", re.MULTILINE),
    re.compile(r"^make:.*Make/files.*Error", re.MULTILINE),
    re.compile(r"warning: ignoring command line option .*EXE_INC", re.IGNORECASE),
    # The most common LLM mistake — Make/files lists a .C file that doesn't exist
    re.compile(r"\.C: No such file or directory"),
]

_PAT_L4_LINK = [
    re.compile(r"undefined reference to"),
    re.compile(r"ld: cannot find -l"),
    re.compile(r"DSO missing from command line"),
    re.compile(r"unresolved symbol"),
    re.compile(r"collect2: error: ld returned"),
]

_PAT_L3_TYPE = [
    re.compile(r"no matching function for call to"),
    re.compile(r"no match for .operator"),
    re.compile(r"cannot convert .* to .* in (initialization|assignment|return)"),
    re.compile(r"invalid conversion from"),
    re.compile(r"no known conversion"),
    re.compile(r"invalid use of incomplete type"),
    re.compile(r"template argument deduction/substitution failed"),
    re.compile(r"ambiguous overload"),
    re.compile(r"dimensionedScalar.*not(?:\s|N)ot.*compatible", re.IGNORECASE),
    re.compile(r"different dimensions"),
]

_PAT_L2_LOOKUP = [
    re.compile(r"was not declared in this scope"),
    re.compile(r"is not a member of"),
    re.compile(r"has not been declared"),
    re.compile(r"there are no arguments to .* that depend on a template parameter"),
    re.compile(r"undefined declaration"),
    # missing headers — gcc says "fatal error: <header>: No such file or directory"
    # but only when the .h/.H/.C file is just a header lookup (NOT Make/files)
    re.compile(r"fatal error: [^:]+\.[hH]: No such file or directory"),
]

_PAT_L1_PARSE = [
    re.compile(r"expected .* before"),
    re.compile(r"expected .* at end of input"),
    re.compile(r"missing terminating .* character"),
    re.compile(r"stray .* in program"),
    re.compile(r"unterminated comment"),
]


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------

def _extract_missing_headers(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"fatal error: ([^:\s]+\.[hH]): No such file or directory", text):
        out.append(m.group(1))
    return list(dict.fromkeys(out))  # dedupe, preserve order


def _extract_undeclared_identifiers(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_:]*)\b\s*was not declared in this scope", text):
        out.append(m.group(1))
    for m in re.finditer(r"is not a member of\s*[‘'\"]?([A-Za-z_][A-Za-z0-9_:]*)", text):
        out.append(m.group(1))
    return list(dict.fromkeys(out))


def _extract_undefined_refs(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"undefined reference to\s+[‘'\"`]?([^’'\"`\n]+)", text):
        out.append(m.group(1).strip())
    return list(dict.fromkeys(out))


def _extract_missing_libs(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"ld: cannot find -l(\S+)", text):
        out.append(m.group(1))
    return list(dict.fromkeys(out))


def _extract_first_error_lines(text: str, max_lines: int = 10) -> List[str]:
    """Return the first N lines that contain 'error' or 'fatal'."""
    out: List[str] = []
    for line in text.splitlines():
        if re.search(r"\b(error|fatal)\b", line, re.IGNORECASE):
            out.append(line.strip())
            if len(out) >= max_lines:
                break
    return out


def _extract_make_files_csources(text: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_/.]*\.C):\s*No such file or directory", text):
        out.append(m.group(1))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# tier coaching — prompt fragments the build-engineer LLM gets
# ---------------------------------------------------------------------------

_COACHING: Dict[str, str] = {
    "L0": (
        "TIER L0 (build_config). The build pipeline is failing before gcc runs. "
        "Do NOT change the physics implementation. Likely fixes:\n"
        "  - Make/files lists a .C file with a name that does not exist on disk. "
        "Rename either the .C file OR the Make/files entry so they match.\n"
        "  - Make/files target path (LIB) must end in $(FOAM_USER_LIBBIN)/<libName> or "
        "$(FOAM_LIBBIN)/<libName> depending on case-local vs system install.\n"
        "  - Make/options EXE_INC paths must reference directories that exist. "
        "If a path doesn't exist, drop it.\n"
        "  - Output JSON in the schema requested. No prose."
    ),
    "L1": (
        "TIER L1 (parse). A C++ syntactic error. Fix only the offending line; do NOT "
        "rewrite surrounding code. Common cases: missing semicolon, mismatched braces, "
        "stray character. Return the smallest possible diff."
    ),
    "L2": (
        "TIER L2 (lookup). A symbol or header is missing. Fix actions:\n"
        "  - If a header file is missing, replace it with the verified header that "
        "exists in this OpenFOAM installation (check the recon list).\n"
        "  - If an identifier is undeclared, add the missing #include OR change to a "
        "fully-qualified name (e.g. Foam::tmp<volScalarField>).\n"
        "  - Do NOT add new physics. Smallest possible fix only."
    ),
    "L3": (
        "TIER L3 (type). OpenFOAM field-arithmetic type mismatch — the most subtle "
        "error class in OpenFOAM. Coaching:\n"
        "  - tmp<volScalarField>::ref() returns volScalarField& once ownership has "
        "been claimed; consecutive .ref() calls on the same tmp throw at runtime. "
        "Bind to a const ref before reusing.\n"
        "  - For implicit equation contributions use fvm:: operators (Sp, SuSp, "
        "ddt, div, laplacian); for explicit RHS use fvc::.\n"
        "  - Mixing volScalarField and dimensionedScalar requires consistent "
        "dimensions. If a coefficient is dimensionless in physics but enters a "
        "dimensioned equation, wrap in dimensionedScalar(\"name\", dimensions, value).\n"
        "  - operator==(...) on volScalarField is for fvMatrix RHS assignment, NOT "
        "field-equality.\n"
        "  - When an operator overload fails, prefer the explicit Sp/SuSp form over "
        "operator+= on a fvMatrix.\n"
        "Smallest possible fix only — do NOT change the physics."
    ),
    "L4": (
        "TIER L4 (link). The compiler succeeded; the linker failed. Fix actions:\n"
        "  - Add the missing library to LIB_LIBS in Make/options (typical: "
        "-lfiniteVolume, -lmeshTools, -lturbulenceModels, -lfvOptions).\n"
        "  - For undefined references on RAS/LES base-class symbols, ensure "
        "-lturbulenceModels or -lincompressibleMomentumTransportModels is linked.\n"
        "  - Do NOT reimplement missing functions — they are in OpenFOAM libs."
    ),
    "L5": (
        "TIER L5 (environment). OpenFOAM environment is not initialized. The "
        "build host needs `source $WM_PROJECT_DIR/etc/bashrc` (or the bundled "
        "OpenFOAM-10/etc/bashrc) before wmake can run. This is a host-level fix, "
        "not a code change."
    ),
    "L9": (
        "TIER L9 (unknown). Could not classify the build error. Read the error "
        "verbatim and apply the smallest possible fix. Do not change the physics."
    ),
}


# ---------------------------------------------------------------------------
# top-level classifier
# ---------------------------------------------------------------------------

def classify(stderr_text: str, stdout_text: str = "") -> Dict[str, Any]:
    """
    Classify a wmake build failure. `stderr_text` is the primary signal;
    `stdout_text` is consulted as a fallback (some toolchains write errors
    to stdout).
    """
    blob = (stderr_text or "") + "\n" + (stdout_text or "")
    if not blob.strip():
        return {
            "tier": "L9",
            "tier_label": "unknown",
            "key_messages": [],
            "missing_headers": [],
            "undeclared": [],
            "undefined_refs": [],
            "missing_libs": [],
            "make_files_csources": [],
            "coaching": _COACHING["L9"],
            "raw_excerpt": "",
        }

    tier = "L9"
    tier_label = "unknown"
    # Order: L5 first (env), L0 (config), L4 (link), L3 (type), L2 (lookup), L1 (parse).
    if any(p.search(blob) for p in _PAT_L5_ENV):
        tier, tier_label = "L5", "environment"
    elif any(p.search(blob) for p in _PAT_L0_CONFIG):
        tier, tier_label = "L0", "build_config"
    elif any(p.search(blob) for p in _PAT_L4_LINK):
        tier, tier_label = "L4", "link"
    elif any(p.search(blob) for p in _PAT_L3_TYPE):
        tier, tier_label = "L3", "type"
    elif any(p.search(blob) for p in _PAT_L2_LOOKUP):
        tier, tier_label = "L2", "lookup"
    elif any(p.search(blob) for p in _PAT_L1_PARSE):
        tier, tier_label = "L1", "parse"

    return {
        "tier": tier,
        "tier_label": tier_label,
        "key_messages": _extract_first_error_lines(blob, max_lines=12),
        "missing_headers": _extract_missing_headers(blob),
        "undeclared": _extract_undeclared_identifiers(blob),
        "undefined_refs": _extract_undefined_refs(blob),
        "missing_libs": _extract_missing_libs(blob),
        "make_files_csources": _extract_make_files_csources(blob),
        "coaching": _COACHING.get(tier, _COACHING["L9"]),
        "raw_excerpt": blob[-4000:],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a wmake build failure.")
    parser.add_argument("--stderr", required=True, type=str,
                        help="Path to a file containing the build stderr (or '-' for stdin).")
    parser.add_argument("--stdout", default="", type=str,
                        help="Optional path to build stdout.")
    args = parser.parse_args()

    stderr_path = args.stderr
    if stderr_path == "-":
        stderr_text = sys.stdin.read()
    else:
        stderr_text = Path(stderr_path).expanduser().read_text(encoding="utf-8", errors="replace")

    stdout_text = ""
    if args.stdout:
        stdout_text = Path(args.stdout).expanduser().read_text(encoding="utf-8", errors="replace")

    result = classify(stderr_text, stdout_text)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

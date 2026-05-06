"""
OpenFOAM environment grounding block for code-modification prompts.

Why this exists
---------------
LLMs frequently hallucinate OpenFOAM include paths and library names that look
plausible but don't exist on the actual install (e.g. `singlePhaseTransportModel.H`,
`-lincompressibleTransportModels` for OF-10 foundation). The fix is to inject
ground-truth facts about *this* OpenFOAM install into the prompt:

  * the real WM_PROJECT_DIR / WM_PROJECT_VERSION,
  * real subdirectories of `src/` (so the LLM sees actual family names),
  * real library basenames in $FOAM_LIBBIN (so it picks valid -l flags),
  * one real working `Make/options` file from the install (so it sees the
    exact `-I$(LIB_SRC)/...` shape that links cleanly on this tree).

Generic across CFD modification families — no turbulence/SA assumptions. The
block is purely descriptive of the install; the agent decides which family
(turbulence, transport, BC, source, scheme, ...) applies to its hypothesis.

CLI:
    python scripts/openfoam_grounding.py [--wm-project-dir PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional


# Generic family hints we drill one level deeper into when present. These are
# OpenFOAM-wide directory names (not specific to a study); listing them helps
# the LLM see real subfamily directory names that exist on disk.
_FAMILY_DRILLDOWN = [
    "MomentumTransportModels",
    "ThermophysicalTransportModels",
    "thermophysicalModels",
    "transportModels",
    "physicalProperties",
    "fvModels",
    "fvConstraints",
    "finiteVolume",
    "functionObjects",
    "radiationModels",
    "combustionModels",
    "lagrangian",
    "multiphaseModels",
    "specieTransfer",
]

# Make/options candidates, in priority order. We pick the first one that
# exists. Each is a real, shipped, working build description for an OpenFOAM
# library — perfect as a "this is what a valid Make/options looks like on
# THIS install" example.
_MAKE_OPTIONS_CANDIDATES = [
    "MomentumTransportModels/incompressible/Make/options",
    "MomentumTransportModels/compressible/Make/options",
    "fvModels/Make/options",
    "finiteVolume/Make/options",
]


def _detect_wm_project_dir(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    env = os.environ.get("WM_PROJECT_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    # Fallback: a couple of common install locations.
    for candidate in ("/opt/openfoam10", "/mnt/sda1/openfoam10",
                      "/usr/lib/openfoam/openfoam10"):
        p = Path(candidate)
        if p.is_dir():
            return p
    return None


def _detect_wm_version(wm: Path) -> str:
    env_v = os.environ.get("WM_PROJECT_VERSION")
    if env_v:
        return env_v
    # Best-effort guess from the install dir name.
    name = wm.name.lower()
    for tag in ("openfoam-", "openfoam"):
        if tag in name:
            return name.split(tag, 1)[-1] or "(unknown)"
    return "(unknown)"


def _detect_libbins() -> List[Path]:
    out: List[Path] = []
    for var in ("FOAM_LIBBIN", "FOAM_USER_LIBBIN"):
        v = os.environ.get(var)
        if v and Path(v).is_dir():
            out.append(Path(v))
    return out


def _list_dir(p: Path, max_entries: int = 80) -> List[str]:
    if not p.is_dir():
        return []
    try:
        names = sorted(x.name for x in p.iterdir() if not x.name.startswith("."))
    except OSError:
        return []
    if len(names) > max_entries:
        names = names[:max_entries] + [f"... (+{len(names) - max_entries} more)"]
    return names


def _list_so_basenames(libbin: Path, max_entries: int = 200) -> List[str]:
    if not libbin.is_dir():
        return []
    try:
        names = sorted(x.name for x in libbin.iterdir()
                       if x.suffix == ".so" and x.name.startswith("lib"))
    except OSError:
        return []
    if len(names) > max_entries:
        names = names[:max_entries] + [f"... (+{len(names) - max_entries} more)"]
    return names


def _read_first_make_options(wm: Path) -> Optional[tuple]:
    src = wm / "src"
    for rel in _MAKE_OPTIONS_CANDIDATES:
        p = src / rel
        if p.is_file():
            try:
                return rel, p.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
    return None


def build_grounding_block(wm_project_dir: Optional[Path] = None) -> str:
    """Return a markdown-style grounding block listing facts about THIS
    OpenFOAM install. Safe to call when WM_PROJECT_DIR is unset — falls back
    to a short notice block.

    The block is intentionally factual (no instructions). It is meant to be
    inserted near the top of an agent prompt so subsequent free-form
    generation has correct path/lib names to ground against.
    """
    wm = _detect_wm_project_dir(wm_project_dir)
    if wm is None:
        return (
            "OPENFOAM ENVIRONMENT FACTS\n"
            "==========================\n"
            "WM_PROJECT_DIR was not detected. The agent must read it from the\n"
            "environment or via `printenv WM_PROJECT_DIR` before generating any\n"
            "include paths or library names.\n"
        )

    version = _detect_wm_version(wm)
    src = wm / "src"
    libbins = _detect_libbins()

    lines: List[str] = []
    lines.append("OPENFOAM ENVIRONMENT FACTS (this install — do not invent paths)")
    lines.append("===============================================================")
    lines.append(f"WM_PROJECT_DIR    : {wm}")
    lines.append(f"WM_PROJECT_VERSION: {version}")
    if libbins:
        for lb in libbins:
            lines.append(f"libbin            : {lb}")
    lines.append("")
    lines.append("Use these EXACT paths and library basenames when writing")
    lines.append("Make/options, #include directives, and system/controlDict.libs.")
    lines.append("If you write a path or -l<name> that does not appear below, it")
    lines.append("almost certainly does not exist on this install.")
    lines.append("")

    # Top-level src/ directory listing
    top = _list_dir(src, max_entries=80)
    if top:
        lines.append(f"$WM_PROJECT_DIR/src/  (top level — real subsystem names)")
        lines.append("-------------------------------------------------------")
        for n in top:
            lines.append(f"  {n}")
        lines.append("")

    # Drill one level into common families when present
    for fam in _FAMILY_DRILLDOWN:
        sub = src / fam
        items = _list_dir(sub, max_entries=40)
        if not items:
            continue
        lines.append(f"$WM_PROJECT_DIR/src/{fam}/  (subfamilies that exist)")
        lines.append("-" * (len(fam) + 50))
        for n in items:
            lines.append(f"  {n}")
        lines.append("")

    # Real working Make/options example
    mk = _read_first_make_options(wm)
    if mk is not None:
        rel, body = mk
        lines.append(f"REAL Make/options shipped with this install ($WM_PROJECT_DIR/src/{rel})")
        lines.append("--------------------------------------------------------------------")
        lines.append("Use the same -I$(LIB_SRC)/... and -l<name> patterns as a template.")
        lines.append("```")
        lines.append(body)
        lines.append("```")
        lines.append("")

    # Available libraries
    for lb in libbins:
        names = _list_so_basenames(lb, max_entries=200)
        if not names:
            continue
        lines.append(f"Available .so libraries in {lb}")
        lines.append("-" * 60)
        for n in names:
            # strip lib prefix and .so suffix to give the LLM the linker-flag form
            stem = n[3:-3] if n.startswith("lib") and n.endswith(".so") else n
            lines.append(f"  {n}    (link as: -l{stem})")
        lines.append("")

    lines.append("END OF ENVIRONMENT FACTS")
    lines.append("")
    return "\n".join(lines)


def _scan_case_fields(case_dir: Path) -> List[tuple]:
    """Return a list of (field_name, class) tuples for every field file under
    `case_dir/0/` (or whichever start time exists). Reads the `class` line in
    the FoamFile header. Generic — any OpenFOAM case, any solver."""
    if not case_dir.is_dir():
        return []
    # pick start-time subdir. Prefer "0", fall back to lowest-numbered dir.
    start = case_dir / "0"
    if not start.is_dir():
        # find any pure-numeric subdir
        cands = [p for p in case_dir.iterdir() if p.is_dir()]
        cands = [p for p in cands if p.name.replace(".", "", 1).replace("-", "", 1).isdigit()]
        if not cands:
            return []
        start = sorted(cands, key=lambda p: float(p.name))[0]
    out: List[tuple] = []
    try:
        for f in sorted(start.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            try:
                head = f.read_text(encoding="utf-8", errors="replace")[:2000]
            except OSError:
                continue
            cls = ""
            for line in head.splitlines():
                s = line.strip()
                if s.startswith("class") and s.endswith(";"):
                    cls = s[len("class"):].strip().rstrip(";").strip()
                    break
            if cls:
                out.append((f.name, cls))
    except OSError:
        return []
    return out


def build_runtime_snippet_grounding(case_dir: Optional[Path] = None) -> str:
    """Grounding block for the runtime_source / runtime_bc / runtime_field
    planner LLM call. Lists the actual registered fields available on the
    case at solver startup, plus the OpenFOAM dimensioned-arithmetic rule and
    common name gotchas. Generic across CFD modification kinds.

    `case_dir` should point at a complete OpenFOAM case (with 0/ and
    constant/). If None or invalid, a generic block is returned.
    """
    lines: List[str] = []
    lines.append("OPENFOAM CODED-SNIPPET AUTHORING FACTS (do not invent field names or scalar+field arithmetic)")
    lines.append("=================================================================================")
    fields = _scan_case_fields(case_dir) if case_dir is not None else []
    if fields:
        lines.append(f"Registered fields on this case (read from {case_dir}/0/):")
        for name, cls in fields:
            lines.append(f"  {name:<24}  class {cls}    →  mesh().lookupObject<{cls}>(\"{name}\")")
        lines.append("")
        lines.append("If the field you need is NOT listed above, do NOT call lookupObject for it —")
        lines.append("it will throw at runtime (objectRegistry::lookupObject error).")
        lines.append("")
    lines.append("Wall-distance access (common gotcha):")
    lines.append("  there is NO field literally called \"y\". The wall-distance field is")
    lines.append("  registered as `yWall` (volScalarField). Access it either via")
    lines.append("    mesh().lookupObject<volScalarField>(\"yWall\")")
    lines.append("  or compute it via")
    lines.append("    #include \"wallDist.H\"")
    lines.append("    const volScalarField& y = wallDist::New(mesh()).y();")
    lines.append("")
    lines.append("Dimensioned arithmetic (HARD RULE — violations abort with FOAM FATAL):")
    lines.append("  OpenFOAM volScalarField / volVectorField carry dimensions. You may NOT add or")
    lines.append("  subtract a bare `scalar` to/from a dimensioned field; the dimension check fails:")
    lines.append("    volScalarField magU2(magSqr(U));   // [0 2 -2 0 0 0 0]")
    lines.append("    const scalar Ueps = 1e-12;")
    lines.append("    auto bad  = magU2 + Ueps;          // ABORT: m^2/s^2 + dimensionless")
    lines.append("  Use a `dimensionedScalar` with matching dimensions:")
    lines.append("    dimensionedScalar Ueps(\"Ueps\", magU2.dimensions(), 1e-12);")
    lines.append("    auto good = magU2 + Ueps;          // OK")
    lines.append("  Same rule applies to division, max, min, etc. — every operand of a binary")
    lines.append("  operator on a field must carry compatible dimensions.")
    lines.append("")
    lines.append("Coefficients from the dictionary:")
    lines.append("  numeric values listed under the entry are read with")
    lines.append("    coeffs().lookupOrDefault<scalar>(\"name\", default)")
    lines.append("  these are bare scalars; wrap them in `dimensionedScalar` before adding to a")
    lines.append("  field. For pure multiplications (scalar * field), bare scalars are fine.")
    lines.append("")
    lines.append("END OF CODED-SNIPPET FACTS")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print OpenFOAM grounding block")
    ap.add_argument("--wm-project-dir", type=Path, default=None,
                    help="Override WM_PROJECT_DIR (otherwise read from env)")
    args = ap.parse_args()
    print(build_grounding_block(args.wm_project_dir))


if __name__ == "__main__":
    main()

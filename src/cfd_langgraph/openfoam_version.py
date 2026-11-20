"""Which OpenFOAM a study runs on, and what that installation calls things.

The harness was written against OpenFOAM 10 from openfoam.org. A user may have
any of openfoam.org 9-14 or openfoam.com v2006-v2512 instead, and the case
files, solvers and utilities differ between them: openfoam.org 11 and later run
solver modules through ``foamRun -solver <module>`` in place of simpleFoam and
its siblings; openfoam.com keeps turbulenceProperties, transportProperties and
fvOptions where openfoam.org uses momentumTransport, physicalProperties and
fvModels; post-processing is ``foamPostProcess`` on one and ``postProcess`` on
the other.

Nothing here is a table of versions to keep up to date. Every fact is read from
the installation the study was given: its environment, the files its own
tutorials use, the wrapper scripts it ships for superseded solver names, and
the utilities on its PATH. The result is cached per installation. The compile
facts the code-modification agent needs (src families, library names, a real
Make/options) come from scripts/openfoam_grounding.py.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_VERSION = 2
_CACHE_DIR = Path(os.environ.get("CFD_SCIENTIST_CACHE", Path.home() / ".cache" / "cfd_scientist")) / "openfoam_profiles"

# The roles a case file plays, and the names the forks have used for each. Which
# name an installation uses is decided by counting its own tutorials.
_ROLE_NAMES = {
    "turbulence": ("momentumTransport", "turbulenceProperties"),
    "transport": ("physicalProperties", "transportProperties"),
    "thermo": ("physicalProperties", "thermophysicalProperties"),
    "sources": ("fvModels", "fvOptions"),
    "constraints": ("fvConstraints", "fvOptions"),
}
_UTILITIES = ("foamPostProcess", "postProcess", "foamDictionary", "foamListTimes", "foamRun",
              "checkMesh", "blockMesh", "snappyHexMesh", "decomposePar", "reconstructPar",
              "foamToVTK", "setFields", "topoSet", "createPatch", "wmake")
_STUB_RUN = re.compile(r"exec\s+env\s+foamRun\s+-solver\s+(\w+)")


def _env(openfoam_path: str) -> Dict[str, str]:
    from cfd_langgraph.foam_native.openfoam_env import resolve_openfoam_env

    return resolve_openfoam_env(str(openfoam_path))


def _which(name: str, env: Dict[str, str]) -> Optional[str]:
    for folder in env.get("PATH", "").split(":"):
        candidate = Path(folder) / name
        if folder and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _fork(env: Dict[str, str]) -> str:
    version = env.get("WM_PROJECT_VERSION", "")
    if env.get("WM_PROJECT", "").lower() == "foam" or "extend" in env.get("WM_FORK", "").lower():
        return "foam-extend"
    if version.startswith("v"):
        return "openfoam.com"
    return "openfoam.org"


def _tutorial_files(tutorials: Path, limit: int = 4000) -> Dict[str, Counter]:
    """How often each candidate name appears as a case file in the stock
    tutorials, and which solver or application their controlDicts name."""
    names: Counter = Counter()
    solvers: Counter = Counter()
    applications: Counter = Counter()
    wanted = {n for pair in _ROLE_NAMES.values() for n in pair}
    seen = 0
    for root, _dirs, files in os.walk(tutorials):
        base = os.path.basename(root)
        if base in ("constant", "system"):
            names.update(f for f in files if f in wanted)
        if base == "system" and "controlDict" in files:
            seen += 1
            try:
                text = Path(root, "controlDict").read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.strip().rstrip(";").split()
                if len(parts) == 2 and parts[0] == "solver":
                    solvers[parts[1]] += 1
                elif len(parts) == 2 and parts[0] == "application":
                    applications[parts[1]] += 1
            if seen >= limit:
                break
    return {"names": names, "solvers": solvers, "applications": applications}


def _superseded(project_dir: Path) -> Dict[str, str]:
    """Old solver names the installation still answers to, mapped to the
    solver module its own wrapper script runs instead (openfoam.org 11+)."""
    out: Dict[str, str] = {}
    bin_dir = project_dir / "bin"
    if not bin_dir.is_dir():
        return out
    for script in bin_dir.iterdir():
        if not script.name.endswith("Foam") or not script.is_file():
            continue
        try:
            match = _STUB_RUN.search(script.read_text(errors="replace")[:20000])
        except OSError:
            continue
        if match:
            out[script.name] = match.group(1)
    return dict(sorted(out.items()))


def build_profile(openfoam_path: str) -> Dict[str, Any]:
    """Read the installation at ``openfoam_path`` (its WM_PROJECT_DIR)."""
    env = _env(openfoam_path)
    project_dir = Path(env.get("WM_PROJECT_DIR") or openfoam_path)
    version = env.get("WM_PROJECT_VERSION", "")
    if not version:
        return {"ok": False, "error": f"could not source {openfoam_path}/etc/bashrc", "openfoam_path": openfoam_path}
    tutorials = Path(env.get("FOAM_TUTORIALS") or project_dir / "tutorials")
    empty = {"names": Counter(), "solvers": Counter(), "applications": Counter()}
    counts = _tutorial_files(tutorials) if tutorials.is_dir() else empty
    names = counts["names"]
    files: Dict[str, str] = {}
    for role, (newer, older) in _ROLE_NAMES.items():
        files[role] = newer if names[newer] >= names[older] else older
    if names["fvOptions"] > names["fvModels"]:
        files["sources"] = files["constraints"] = "fvOptions"
    utilities = {u: bool(_which(u, env)) for u in _UTILITIES}
    modules = bool(utilities["foamRun"]) and sum(counts["solvers"].values()) > sum(counts["applications"].values())
    app_bin = Path(env.get("FOAM_APPBIN", ""))
    applications = sorted(p.name for p in app_bin.iterdir()) if app_bin.is_dir() else []
    fork = _fork(env)
    return {
        "ok": True,
        "profile_version": PROFILE_VERSION,
        "fork": fork,
        "version": version,
        "label": f"OpenFOAM {version} ({fork})",
        "project_dir": str(project_dir),
        "bashrc": str(project_dir / "etc" / "bashrc"),
        "tutorials_dir": str(tutorials),
        "source_dir": str(project_dir / "src"),
        "user_libbin": env.get("FOAM_USER_LIBBIN", ""),
        "user_appbin": env.get("FOAM_USER_APPBIN", ""),
        "solver_style": "modules" if modules else "applications",
        "solver_modules": sorted(counts["solvers"]) if modules else [],
        # Standalone solvers its tutorials run, most used first.
        "solver_applications": [a for a, _ in counts["applications"].most_common()
                                if a.endswith("Foam") and a != "foamRun"],
        "superseded_solvers": _superseded(project_dir),
        "applications": applications,
        "files": files,
        "utilities": utilities,
        "post_process": "foamPostProcess" if utilities["foamPostProcess"] else "postProcess",
        "function_object_templates": str(project_dir / "etc" / "caseDicts" / "postProcessing"),
        "can_compile": bool(utilities["wmake"]) and (project_dir / "src").is_dir(),
    }


def _cache_path(openfoam_path: str, version: str) -> Path:
    key = re.sub(r"[^A-Za-z0-9]+", "_", f"{Path(openfoam_path).resolve()}_{version}").strip("_")
    return _CACHE_DIR / f"{key}.json"


def get_profile(openfoam_path: str = "", refresh: bool = False) -> Dict[str, Any]:
    """The profile of the installation at ``openfoam_path`` (default: the
    configured one), read once and cached."""
    if not openfoam_path:
        from cfd_langgraph.config import Settings

        openfoam_path = Settings().openfoam_path
    env = _env(openfoam_path)
    version = env.get("WM_PROJECT_VERSION", "")
    cache = _cache_path(openfoam_path, version or "unknown")
    if not refresh and version and cache.is_file():
        try:
            doc = json.loads(cache.read_text())
            if doc.get("profile_version") == PROFILE_VERSION and doc.get("version") == version:
                return doc
        except (OSError, ValueError):
            pass
    doc = build_profile(openfoam_path)
    if doc.get("ok"):
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(doc, indent=2))
        except OSError:
            pass
    return doc


def run_command(profile: Dict[str, Any], application: str) -> str:
    """How to run a solver this installation knows by ``application``: itself,
    or the foamRun module that superseded it."""
    module = (profile.get("superseded_solvers") or {}).get(application)
    if module and profile.get("solver_style") == "modules":
        return f"foamRun -solver {module}"
    return application


def describe(profile: Dict[str, Any]) -> str:
    """The facts a model writing or changing a case for this installation must
    hold, as prompt text."""
    if not profile.get("ok"):
        return "The OpenFOAM installation could not be read; follow the files of its own tutorials."
    f = profile["files"]
    lines = [
        f"This study runs {profile['label']}, installed at {profile['project_dir']}.",
        f"- Its own tutorials are the reference for every case file: {profile['tutorials_dir']}",
        f"- Turbulence settings: constant/{f['turbulence']}. Incompressible transport: constant/{f['transport']}. "
        f"Thermophysical properties: constant/{f['thermo']}.",
        f"- Source terms: {f['sources']}; constraints: {f['constraints']}.",
    ]
    if profile["solver_style"] == "modules":
        examples = ", ".join(f"{k} -> foamRun -solver {v}" for k, v in list(profile["superseded_solvers"].items())[:6])
        lines.append("- Solvers are modules run by foamRun: set `solver <module>;` in system/controlDict and run "
                     f"`foamRun`. Old solver names map to modules ({examples}).")
    else:
        lines.append("- Solvers are standalone applications named in system/controlDict `application` "
                     "(simpleFoam, pimpleFoam, ...).")
    lines.append(f"- Post-processing utility: {profile['post_process']}. Function-object templates: "
                 f"{profile['function_object_templates']}.")
    if profile.get("can_compile"):
        lines.append(f"- Headers for compiled code: {profile['source_dir']}. Build libraries with wmake into "
                     f"$FOAM_USER_LIBBIN ({profile['user_libbin']}) or a case-local folder.")
    from cfd_langgraph.solver_shelf import describe as shelf_describe

    shelf = shelf_describe(profile)
    if shelf:
        lines.append(shelf)
    return "\n".join(lines)


def case_version_evidence(case_dir: Path) -> Dict[str, Any]:
    """What a case's own files say about the version it was written for: the
    banner lines of its dictionaries, the case-file names it uses, and the
    solver or application its controlDict names."""
    case_dir = Path(case_dir)
    banners: Counter = Counter()
    present: List[str] = []
    control: Dict[str, str] = {}
    wanted = {n for pair in _ROLE_NAMES.values() for n in pair}
    for sub in ("system", "constant", "0", "0.orig"):
        folder = case_dir / sub
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            if p.name in wanted:
                present.append(f"{sub}/{p.name}")
            try:
                head = p.read_text(errors="replace")[:1200]
            except OSError:
                continue
            for line in head.splitlines():
                text = line.strip("|/\\ *").strip()
                if text.lower().startswith("version:") or "openfoam.org" in text or "openfoam.com" in text:
                    banners[" ".join(text.split())[:80]] += 1
    cd = case_dir / "system" / "controlDict"
    if cd.is_file():
        for line in cd.read_text(errors="replace").splitlines():
            parts = line.strip().rstrip(";").split()
            if len(parts) == 2 and parts[0] in ("solver", "application"):
                control[parts[0]] = parts[1]
    return {"banners": dict(banners.most_common(5)), "case_files": present, "control": control}


# --------------------------------------------------------------------------- #
# Shared lookups the rest of the harness uses instead of assuming OpenFOAM 10   #
# --------------------------------------------------------------------------- #

# Where package and source installs put OpenFOAM; CFD_SCIENTIST_OPENFOAM_ROOTS
# (colon-separated) adds others.
_INSTALL_ROOTS = ("/opt", "/usr/lib/openfoam", "/usr/local", str(Path.home() / "OpenFOAM"))


def find_installation(explicit: str = "") -> str:
    """The OpenFOAM installation to use: the one named, else the one sourced in
    this environment (WM_PROJECT_DIR), else the newest found in the usual
    places. Empty when there is none."""
    for path in (explicit, os.environ.get("WM_PROJECT_DIR", "")):
        if path and (Path(path) / "etc" / "bashrc").is_file():
            return str(Path(path))
    found: List[Path] = []
    extra = [r for r in os.environ.get("CFD_SCIENTIST_OPENFOAM_ROOTS", "").split(":") if r]
    for root in (*extra, *_INSTALL_ROOTS):
        base = Path(root)
        if not base.is_dir():
            continue
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        for child in children:
            name = child.name.lower()
            if ("openfoam" in name or name.startswith("foam-extend")) and (child / "etc" / "bashrc").is_file():
                found.append(child)
    return str(sorted(found, key=lambda p: p.stat().st_mtime)[-1]) if found else ""


def major(profile: Dict[str, Any]) -> int:
    """The release number: 13 for openfoam.org 13, 2512 for v2512, a large
    number for a development build."""
    version = str(profile.get("version", "")).lstrip("v")
    digits = "".join(ch for ch in version.split(".")[0] if ch.isdigit())
    return int(digits) if digits else 10_000


def _strip_comments(text: str) -> str:
    """The text without /* */ blocks and // line comments."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _dict_entries(path: Path) -> Dict[str, str]:
    """Top-level `key value;` entries of an OpenFOAM dictionary, plus those of
    its first-level sub-dictionaries as `sub/key`. Enough for the handful of
    settings read here; anything richer goes through foamDictionary."""
    out: Dict[str, str] = {}
    try:
        text = _strip_comments(path.read_text(errors="replace"))
    except OSError:
        return out
    stack: List[str] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("}"):
            if stack:
                stack.pop()
            pending = ""
            continue
        if line == "{":
            stack.append(pending or "?")
            pending = ""
            continue
        tokens = line.rstrip(";").split()
        if line.endswith("{"):
            stack.append(tokens[0] if tokens[0] != "{" else "?")
            pending = ""
            continue
        if len(tokens) == 1 and not line.endswith(";"):
            pending = tokens[0]
            continue
        pending = ""
        if len(tokens) >= 2 and line.endswith(";") and len(stack) <= 1:
            out.setdefault("/".join(stack + [tokens[0]]), " ".join(tokens[1:]))
    return out


def case_file(case_dir: Path, role: str) -> Optional[Path]:
    """The file of ``role`` (turbulence, transport, thermo, sources,
    constraints) this case actually has, whichever fork's name it uses."""
    for name in _ROLE_NAMES.get(role, ()):
        for sub in ("constant", "system"):
            p = Path(case_dir) / sub / name
            if p.is_file():
                return p
    return None


def turbulence_model(case_dir: Path) -> Dict[str, str]:
    """simulationType and model of a case, from momentumTransport or
    turbulenceProperties, with either the `model` or the `RASModel` /
    `LESModel` keyword."""
    path = case_file(case_dir, "turbulence")
    if path is None:
        return {}
    entries = _dict_entries(path)
    sim = entries.get("simulationType", "")
    model = ""
    for block in (sim, "RAS", "LES"):
        for key in ("model", f"{block}Model", "RASModel", "LESModel"):
            if entries.get(f"{block}/{key}"):
                model = entries[f"{block}/{key}"]
                break
        if model:
            break
    return {"file": str(path), "simulationType": sim, "model": model}


def case_solver(case_dir: Path, profile: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """What runs this case, from its own controlDict: the application, the
    solver module (openfoam.org 11+), the command to run it, and the log file
    an Allrun's runApplication writes."""
    entries = _dict_entries(Path(case_dir) / "system" / "controlDict")
    application = entries.get("application", "")
    module = entries.get("solver", "")
    profile = profile or {}
    if module:
        command, log = "foamRun", "log.foamRun"
        application = application or "foamRun"
    elif application:
        command = run_command(profile, application) if profile else application
        log = f"log.{application}"
        superseded = (profile.get("superseded_solvers") or {}).get(application)
        if superseded and profile.get("solver_style") == "modules":
            module = superseded
    else:
        command, log = "", ""
    return {"application": application, "module": module, "command": command, "log": log}


def post_process_command(profile: Dict[str, Any], solver: Dict[str, str], func: str, extra: str = "") -> str:
    """The command that evaluates function object ``func`` on a finished case
    with the solver's own models loaded: `foamPostProcess -solver <module>` on
    openfoam.org 11+, `<application> -postProcess` elsewhere."""
    tail = f" -func {func}" + (f" {extra}" if extra else "")
    if profile.get("solver_style") == "modules" or profile.get("utilities", {}).get("foamPostProcess"):
        module = solver.get("module")
        return f"foamPostProcess{' -solver ' + module if module else ''}{tail}"
    application = solver.get("application")
    if application and application != "foamRun":
        return f"{application} -postProcess{tail}"
    return f"{profile.get('post_process', 'postProcess')}{tail}"


def find_tutorials(profile: Dict[str, Any], solver: str, limit: int = 3) -> List[Path]:
    """Stock tutorials of this installation run by ``solver`` (an application
    name, or the module that superseded it), smallest first."""
    tutorials = Path(profile.get("tutorials_dir", ""))
    if not tutorials.is_dir() or not solver:
        return []
    names = {solver}
    module = (profile.get("superseded_solvers") or {}).get(solver)
    if module:
        names.add(module)
    hits: List[Path] = []
    for root, dirs, files in os.walk(tutorials):
        if os.path.basename(root) == "system" and "controlDict" in files:
            case = Path(root).parent
            info = case_solver(case)
            if info["application"] in names or info["module"] in names:
                hits.append(case)
            dirs[:] = []
    hits.sort(key=lambda c: sum(1 for _ in c.rglob("*")))
    return hits[:limit]


def solver_choices(profile: Dict[str, Any]) -> List[str]:
    """What a new case may name as its solver on this installation: the solver
    modules of openfoam.org 11+, else the standalone solvers its tutorials
    run, plus the programs of validated shelf solvers
    (cfd_langgraph.solver_shelf). Empty when the installation could not be
    read."""
    if not profile.get("ok"):
        return []
    from cfd_langgraph.solver_shelf import applications as shelf_applications

    own = profile.get("solver_modules") if profile.get("solver_style") == "modules" else profile.get("solver_applications")
    own = list(own or [])
    # Programs of validated third-party solvers on the shelf run as standalone
    # applications on any version.
    return own + [app for app in shelf_applications(profile) if app not in own]


def solver_command(profile: Dict[str, Any], solver: str) -> str:
    """The command that runs ``solver`` (a module or an application name)."""
    if profile.get("solver_style") == "modules" and solver in (profile.get("solver_modules") or []):
        return f"foamRun -solver {solver}"
    return run_command(profile, solver)


def log_name(command: str) -> str:
    """The log file RunFunctions' runApplication writes for ``command``."""
    first = (command or "").split()
    return f"log.{first[0]}" if first else ""

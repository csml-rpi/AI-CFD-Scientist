"""Third-party OpenFOAM solvers a study can use: a shelf of builds, each validated.

A request for a wave flume, a melt pool or a fluid-structure problem needs a
solver OpenFOAM does not ship (olaFlow, AdditiveFOAM, solids4foam, ...). Each
such solver is described once in the repository, under solvers/<name>/:

  solver.yaml   where it comes from (repository, pinned commit per OpenFOAM
                version), how it is built, what it provides, which domains it
                serves and how a case uses it;
  checks.py     the checks that decide whether a build may be used: tutorials
                to run, the numbers to read from them, the reference each is
                compared with and the tolerance. Written before any build and
                not changed by whatever builds or ports the solver.

A build lives outside the OpenFOAM installation, one per solver and installed
version, under the shelf root ($CFD_SCIENTIST_SHELF, default
~/.cfd_scientist/shelf):

  <root>/src/<name>/                         the solver's repository
  <root>/builds/<name>/<fork>_<version>/     a checkout at the pinned commit, its
                                             bin/ and lib/, the validation runs,
                                             and shelf.json, the record

shelf.json says what the build reached: built, runs (its tutorials ran but no
reference was available), validated (every check within tolerance) or failed,
with the numbers. Studies use validated builds only. scripts/shelf_install.py
builds, scripts/shelf_validate.py validates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = _REPO_ROOT / "solvers"
RECORD = "shelf.json"
USABLE = ("validated",)


def shelf_root() -> Path:
    return Path(os.environ.get("CFD_SCIENTIST_SHELF") or Path.home() / ".cfd_scientist" / "shelf")


# --------------------------------------------------------------------------- #
# The registry                                                                  #
# --------------------------------------------------------------------------- #

def registry() -> Dict[str, Dict[str, Any]]:
    """Every solver described under solvers/, by name."""
    import yaml

    out: Dict[str, Dict[str, Any]] = {}
    if not REGISTRY_DIR.is_dir():
        return out
    for path in sorted(REGISTRY_DIR.glob("*/solver.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, dict) and doc.get("name"):
            doc["_dir"] = str(path.parent)
            out[str(doc["name"])] = doc
    return out


def spec(name: str) -> Dict[str, Any]:
    reg = registry()
    found = reg.get(name) or {k.lower(): v for k, v in reg.items()}.get(name.lower())
    if found is None:
        raise KeyError(f"no solver named {name!r} in {REGISTRY_DIR}")
    return found


def _version_key(profile: Dict[str, Any]) -> str:
    raw = f"{profile.get('fork', 'openfoam')}_{profile.get('version', '')}"
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in raw)


def _words(text: str) -> set:
    """Lower-case words of three or more letters."""
    cleaned = "".join(ch.lower() if ch.isalpha() or ch == "-" else " " for ch in str(text or ""))
    return {w.strip("-") for w in cleaned.split() if len(w.strip("-")) >= 3}


def build_entry(solver: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """The build recipe for the installed version: one listing this exact
    version ({"match": "exact"}), else the one for the nearest version of the
    same fork ({"match": "nearest"}, which needs porting), else {}."""
    from cfd_langgraph.openfoam_version import major

    fork, version = profile.get("fork"), str(profile.get("version"))
    builds = [b for b in solver.get("builds") or [] if isinstance(b, dict)]
    for b in builds:
        if b.get("fork") == fork and version in [str(v) for v in b.get("versions") or []]:
            return {**b, "match": "exact"}
    # A solver that brings its own OpenFOAM (runtime kind own_openfoam) has one
    # build for any installed version.
    for b in builds:
        if b.get("fork") == "any":
            return {**b, "match": "exact"}
    same = [b for b in builds if b.get("fork") == fork] or builds
    if not same:
        return {}
    target = major(profile)

    def distance(b: Dict[str, Any]) -> int:
        return min(abs(major({"version": str(v)}) - target) for v in b.get("versions") or ["0"])

    return {**min(same, key=distance), "match": "nearest"}


# --------------------------------------------------------------------------- #
# Builds on the shelf                                                           #
# --------------------------------------------------------------------------- #

def source_dir(name: str) -> Path:
    return shelf_root() / "src" / name


def build_dir(name: str, profile: Dict[str, Any]) -> Path:
    """Where the solver's build for this installation lives: one folder per
    installed version, or the build's own ``key`` when one build serves every
    version (a solver with its own OpenFOAM)."""
    solver = registry().get(name)
    entry = build_entry(solver, profile) if solver else {}
    return shelf_root() / "builds" / name / str(entry.get("key") or _version_key(profile))


def record(name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads((build_dir(name, profile) / RECORD).read_text())
    except (OSError, ValueError):
        return {}


def write_record(name: str, profile: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    from datetime import datetime, timezone

    path = build_dir(name, profile) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = record(name, profile)
    doc.update(fields)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(doc, indent=2, default=str))
    return doc


def usable(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Solvers validated for the installed version, by name, with their records."""
    out = {}
    for name, solver in registry().items():
        rec = record(name, profile)
        if rec.get("status") in USABLE:
            out[name] = {**solver, "_record": rec, "_build": str(build_dir(name, profile))}
    return out


def env_additions(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    """Folders every usable shelf solver adds to PATH and LD_LIBRARY_PATH."""
    paths: List[str] = []
    libs: List[str] = []
    for solver in usable(profile).values():
        b = Path(solver["_build"])
        paths.append(str(b / "bin"))
        libs.append(str(b / "lib"))
        runtime = solver.get("runtime") or {}
        paths += [str(b / extra) for extra in runtime.get("path") or []]
        libs += [str(b / extra) for extra in runtime.get("ld_library_path") or []]
    return {"PATH": [p for p in paths if Path(p).is_dir()],
            "LD_LIBRARY_PATH": [p for p in libs if Path(p).is_dir()]}


def run_env(openfoam_path: str) -> Dict[str, str]:
    """The installation's environment with every usable shelf solver on it:
    what a case runs in."""
    from cfd_langgraph.foam_native.openfoam_env import resolve_openfoam_env
    from cfd_langgraph.openfoam_version import get_profile

    env = dict(resolve_openfoam_env(openfoam_path))
    # The resolved environment is cached and may predate the study marker the
    # CLI sets (cfd_langgraph.cli.repl.main); every case process carries it.
    if os.environ.get("CFD_SCIENTIST_INSIDE_STUDY"):
        env["CFD_SCIENTIST_INSIDE_STUDY"] = os.environ["CFD_SCIENTIST_INSIDE_STUDY"]
    profile = get_profile(openfoam_path)
    if not profile.get("ok"):
        return env
    for key, values in env_additions(profile).items():
        if values:
            env[key] = ":".join(values + [env.get(key, "")]).rstrip(":")
    return env


def shell_exports(profile: Dict[str, Any]) -> str:
    """The same additions as shell exports, for a command line."""
    parts = []
    for key, values in env_additions(profile).items():
        if values:
            parts.append(f'export {key}="{":".join(values)}:${key}"')
    return " && ".join(parts)


# --------------------------------------------------------------------------- #
# What a study is told                                                          #
# --------------------------------------------------------------------------- #

def tutorials_dir(solver: Dict[str, Any]) -> Path:
    return Path(solver["_build"]) / "src" / str(solver.get("tutorials") or "tutorials")


def describe(profile: Dict[str, Any]) -> str:
    """Prompt text: the shelf solvers this study may use, and how."""
    found = usable(profile)
    if not found:
        return ""
    lines = ["Third-party solvers built and validated for this installation (use one only when the "
             "physics needs it; write its cases from its own tutorials):"]
    for name, solver in found.items():
        apps = ", ".join(_case_programs(solver)) or "(library only)"
        lines.append(f"- {name}: {' '.join(str(solver.get('summary', '')).split())} Programs: {apps}. "
                     f"Tutorials: {tutorials_dir(solver)}. How a case uses it: "
                     f"{' '.join(str(solver.get('usage', '')).split())}")
    return "\n".join(lines)


def candidates(topic: str) -> List[Dict[str, Any]]:
    """Registry solvers whose domains share words with ``topic``, best first.
    The manager decides; this only narrows the list it reads."""
    words = _words(topic)
    scored = []
    for name, solver in registry().items():
        terms = {w for d in solver.get("domains") or [] for w in _words(d)}
        score = len(words & terms)
        if score:
            scored.append((score, name, solver))
    return [{"name": n, "summary": s.get("summary", ""), "domains": s.get("domains", []), "score": sc}
            for sc, n, s in sorted(scored, key=lambda t: (-t[0], t[1]))]


def shelf_tutorials(profile: Dict[str, Any]) -> List[Path]:
    """Tutorial folders of usable shelf solvers, for the case writer's index."""
    return [t for t in (tutorials_dir(s) for s in usable(profile).values()) if t.is_dir()]


def _own_programs(solver: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """For a solver that brings its own OpenFOAM, the programs a case's
    controlDict may name (runtime.programs), each with how it runs; {} for a
    solver built into the installed OpenFOAM."""
    runtime = solver.get("runtime") or {}
    if runtime.get("kind") != "own_openfoam":
        return {}
    return {str(name): dict(how or {}) for name, how in (runtime.get("programs") or {}).items()}


def _case_programs(solver: Dict[str, Any]) -> List[str]:
    return list(_own_programs(solver)) or list((solver.get("provides") or {}).get("applications") or [])


def applications(profile: Dict[str, Any]) -> List[str]:
    """Programs the usable shelf solvers provide: case solvers beside the
    installation's own."""
    return [app for solver in usable(profile).values() for app in _case_programs(solver)]


def own_runtime(case_dir: Path, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How a case runs when the program its controlDict names belongs to a
    usable solver that brings its own OpenFOAM: {"solver", "program",
    "wrapper" (path of the solver's wrapper, which loads that OpenFOAM in
    place of the installed one; every command of the case runs inside it),
    "before" (programs run after meshing, before the solve), "run" (the
    command that solves the case, when the program is not itself run) and
    "inputs" (files of the case beyond 0/, constant/ and system/),
    "function_objects" (False when nothing may add function objects to the
    case), "log_end" (the line a finished run's log ends with)}. {} for a
    case of the installed OpenFOAM."""
    from cfd_langgraph.openfoam_version import case_solver, get_profile

    if not (Path(case_dir) / "system" / "controlDict").is_file():
        return {}
    program = case_solver(Path(case_dir))["application"]
    if not program:
        return {}
    for name, solver in usable(profile or get_profile()).items():
        how = _own_programs(solver).get(program)
        if how is None:
            continue
        wrapper = Path(solver["_build"]) / "bin" / str((solver.get("runtime") or {}).get("wrapper") or "")
        if not wrapper.is_file():
            return {}
        return {"solver": name, "program": program, "wrapper": str(wrapper),
                "before": [str(p) for p in how.get("before") or []], "run": str(how.get("run") or ""),
                "inputs": [str(p) for p in how.get("inputs") or []],
                "function_objects": how.get("function_objects", True) is not False,
                "log_end": str(how.get("log_end") or "End")}
    return {}


def in_runtime(cmd: List[str], case_dir: Path) -> List[str]:
    """``cmd`` as it runs for this case: inside its solver's wrapper when the
    case runs on a solver's own OpenFOAM, else unchanged."""
    runtime = own_runtime(case_dir)
    return [runtime["wrapper"], *cmd] if runtime else list(cmd)


def api_index(solver: Dict[str, Any]) -> str:
    """The public methods of the Python classes a solver declares under
    ``api`` in its solver.yaml, as the build itself has them: one line per
    class, "dotted.name: method, method, ...". Read once through the solver's
    wrapper and kept beside the build (api_index.txt). A DAFoam run script
    written from memory called solver.getCellCenters, which pyDASolvers does
    not have (shelf_demo_dafoam_20261015_1318, case_003)."""
    classes = [str(c) for c in (solver.get("api") or [])]
    if not classes:
        return ""
    cache = Path(solver["_build"]) / "api_index.txt"
    if cache.is_file():
        return cache.read_text(errors="ignore").strip()
    wrapper = Path(solver["_build"]) / "bin" / str((solver.get("runtime") or {}).get("wrapper") or "")
    code = (
        "import importlib, sys\n"
        "for dotted in sys.argv[1:]:\n"
        "    mod, name = dotted.rsplit('.', 1)\n"
        "    try:\n"
        "        obj = getattr(importlib.import_module(mod), name)\n"
        "    except Exception as exc:\n"
        "        print(dotted + ': (could not be read: ' + type(exc).__name__ + ')'); continue\n"
        "    import inspect\n"
        "    def shown(m):\n"
        "        try:\n"
        "            return m + str(inspect.signature(getattr(obj, m)))\n"
        "        except (TypeError, ValueError):\n"
        "            return m\n"
        "    print(dotted + ': ' + ', '.join(shown(m) for m in dir(obj) if not m.startswith('_')))\n"
    )
    import subprocess

    cmd = ([str(wrapper)] if wrapper.is_file() else []) + ["python", "-c", code, *classes]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [line for line in proc.stdout.splitlines() if any(line.startswith(c + ":") for c in classes)]
    text = "\n".join(lines)
    if text:
        try:
            cache.write_text(text + "\n")
        except OSError:
            pass
    return text


def tutorial_cases(solver: Dict[str, Any]) -> List[Path]:
    """A usable solver's tutorial cases as they run on this installation: the
    ported copy where the tutorial had to be ported (tutorials_ported/, kept by
    scripts/shelf_validate.py), else the tutorial itself."""
    build = Path(solver["_build"])
    root = tutorials_dir(solver)
    if not root.is_dir():
        return []
    cases = []
    for control in sorted(root.rglob("system/controlDict")):
        case = control.parent.parent
        if any(part in ("processor0", "regressionTests") for part in case.parts):
            continue
        ported = build / "tutorials_ported" / case.relative_to(build / "src")
        record_file = ported.parent / f"{ported.name}.translation.json"
        try:
            port = json.loads(record_file.read_text()).get("status") if record_file.is_file() else ""
        except (OSError, ValueError):
            port = ""
        cases.append(ported if port in ("aligned", "translated") and ported.is_dir() else case)
    return cases


def solvers_for(requirement: str, case_solver: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Usable shelf solvers a case calls for: one whose program is the case's
    solver, or one the requirement names (a library such as turbinesFoam runs
    inside a stock solver, so only its name says it is wanted)."""
    words = _words(requirement)
    out = []
    for name, solver in usable(profile).items():
        if case_solver in _case_programs(solver) or name.lower() in words:
            out.append(solver)
    return out


def status(profile: Dict[str, Any], topic: str = "") -> List[Dict[str, Any]]:
    """Every registry solver with what its build reached on this installation
    (validated, built, runs, failed, needs_port, or not built), the ones whose
    domains match ``topic`` first."""
    reg = registry()
    order = [c["name"] for c in candidates(topic)] if topic else []
    names = order + [n for n in sorted(reg) if n not in order]
    out = []
    for name in names:
        solver = reg[name]
        rec = record(name, profile)
        entry = build_entry(solver, profile)
        out.append({
            "name": name,
            "summary": " ".join(str(solver.get("summary", "")).split()),
            "domains": solver.get("domains") or [],
            "matches_topic": name in order,
            "status": rec.get("status") or "not built",
            "usable": rec.get("status") in USABLE,
            "build_for_this_version": entry.get("match", "none") if entry else "none",
            "checks": [{"name": c.get("name"), "passed": c.get("passed")} for c in rec.get("checks") or []],
        })
    return out

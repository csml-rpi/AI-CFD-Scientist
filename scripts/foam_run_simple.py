#!/usr/bin/env python3
"""
Minimal OpenFOAM case runner: copy a complete case, source OpenFOAM, run
Allrun (or generate one if missing), capture logs, report rc.

Use this for CASES THAT ARE ALREADY COMPLETE — runtime / dict_only
modifications applied via OpenFOAM's coded* infrastructure or pure
dictionary edits. No LLM. No reviewer loop. No file rewriting. Generic
across topics — works for any application a controlDict names.

Why a separate runner:
  scripts/foam_run.py is FoamAgent-driven. Its reviewer loop assumes the
  case may need C++ class derivation and will rewrite constant/momentum
  Transport, Make/files, Make/options, etc. on Allrun failure. That is
  exactly wrong for runtime / dict_only modifications, where the case is
  intentionally complete and any failure is the user's spec bug, not
  something a class-deriving reviewer can fix.

CLI:
  python scripts/foam_run_simple.py \
      --base-case <complete OpenFOAM case dir> \
      --output-dir <where to copy the case and run> \
      --output     <result.json>

Result JSON shape:
  {
    "status":          "OK"|"FAILED",
    "rc":              <int>,
    "case_dir":        "<absolute path of run case>",
    "stdout_tail":     "<last 4000 chars of stdout>",
    "stderr_tail":     "<last 4000 chars of stderr>",
    "log_paths":       ["<paths to log.* files in case>"],
    "error":           "<message when status=FAILED>"
  }
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_application_from_controldict(controldict: Path) -> Optional[str]:
    txt = _read(controldict)
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("application"):
            parts = s.replace(";", "").split()
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _resolve_case_dir(case_dir: Path) -> Path:
    """If `case_dir` is a wrapper, descend into the unique child case."""
    if (case_dir / "constant").is_dir() or (case_dir / "system").is_dir():
        return case_dir
    children = [p for p in case_dir.iterdir() if p.is_dir()]
    for c in children:
        if (c / "constant").is_dir() or (c / "system").is_dir():
            return c
    return case_dir


def _copy_case(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(directory: str, names: List[str]) -> List[str]:
        skip = []
        for n in names:
            if n in {"postProcessing"} or n.startswith("processor") or n.startswith("log."):
                skip.append(n)
            if n in {"Allrun.out", "Allrun.err"}:
                skip.append(n)
        return skip

    shutil.copytree(str(src), str(dst), ignore=ignore)


def _ensure_allrun(case_dir: Path) -> Path:
    """If Allrun does not exist, write a generic one based on controlDict."""
    allrun = case_dir / "Allrun"
    if allrun.is_file():
        return allrun
    app = _read_application_from_controldict(case_dir / "system" / "controlDict") or "simpleFoam"
    has_blockmesh = (case_dir / "system" / "blockMeshDict").is_file()
    body = "#!/usr/bin/env bash\n"
    body += "set -e\n"
    # Use `dirname "$0"` (robust when invoked as `./Allrun`, `bash Allrun`,
    # or absolute path). The `${0%/*}` shorthand fails when $0 has no slash.
    body += 'cd "$(dirname "$(readlink -f "$0")")" || exit 1\n'
    body += '. "${WM_PROJECT_DIR:?WM_PROJECT_DIR not set — source OpenFOAM bashrc first}/bin/tools/RunFunctions"\n'
    if has_blockmesh:
        body += "runApplication blockMesh\n"
        body += "runApplication checkMesh -allTopology -allGeometry || true\n"
    body += f"runApplication {app}\n"
    allrun.write_text(body, encoding="utf-8")
    allrun.chmod(0o755)
    return allrun


def run(base_case: Path, output_dir: Path, openfoam_bashrc: Optional[Path] = None,
        timeout_seconds: int = 21600) -> Dict[str, Any]:
    """Run the case. Returns a result dict."""
    src = _resolve_case_dir(base_case)
    if not (src / "constant").is_dir() and not (src / "system").is_dir():
        return {
            "status": "FAILED",
            "rc": -1,
            "case_dir": str(src),
            "error": f"not an OpenFOAM case: {src}",
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    # If output_dir is itself the case dir (already populated), don't recopy.
    if (output_dir / "constant").is_dir() or (output_dir / "system").is_dir():
        case = _resolve_case_dir(output_dir)
    else:
        _copy_case(src, output_dir)
        case = _resolve_case_dir(output_dir)

    # Sanitize controlDict's libs entry so any case-local custom OpenFOAM
    # library compiled into <case>/customModels/<Class>/platforms/.../lib/*.so
    # is referenced via a case-relative path that survives the cwd switch
    # into the case dir. Mirrors the same normalization that foam_run.py
    # applies on the FoamAgent path so both runners give identical runtime
    # behavior. Generic across modification families (turbulence, viscosity,
    # source terms, ...). Best-effort: any failure is logged but does not
    # block the run.
    try:
        _foam_run_dir = str(Path(__file__).resolve().parent)
        if _foam_run_dir not in sys.path:
            sys.path.insert(0, _foam_run_dir)
        from foam_run import _normalize_control_dict_custom_libs as _norm_libs  # type: ignore
        _norm_libs(case)
    except Exception as _e:
        print(f"[foam_run_simple] libs normalization skipped: {type(_e).__name__}: {_e}")

    _ensure_allrun(case)

    # Build a shell command that sources OpenFOAM, then runs Allrun.
    # Be permissive about where OpenFOAM lives.
    bashrc_candidates = []
    if openfoam_bashrc:
        bashrc_candidates.append(str(openfoam_bashrc))
    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    if wm:
        bashrc_candidates.append(f"{wm}/etc/bashrc")
    bashrc_candidates += [
        "/mnt/sda1/openfoam10/etc/bashrc",
        "/opt/openfoam10/etc/bashrc",
        "/usr/lib/openfoam/openfoam2306/etc/bashrc",
    ]
    bashrc = next((p for p in bashrc_candidates if Path(p).is_file()), None)
    if bashrc is None:
        return {
            "status": "FAILED",
            "rc": -1,
            "case_dir": str(case),
            "error": (
                "Could not find OpenFOAM bashrc. Set WM_PROJECT_DIR or pass "
                "--openfoam-bashrc."
            ),
        }

    shell_cmd = (
        f". \"{bashrc}\" && cd \"{case}\" && ./Allrun"
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", shell_cmd],
            timeout=timeout_seconds,
            text=True,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "FAILED",
            "rc": -2,
            "case_dir": str(case),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": f"timeout after {timeout_seconds}s",
        }

    log_paths = [str(p) for p in sorted(case.glob("log.*"))]
    return {
        "status": "OK" if proc.returncode == 0 else "FAILED",
        "rc": int(proc.returncode),
        "case_dir": str(case),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "log_paths": log_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete OpenFOAM case via Allrun.")
    parser.add_argument("--base-case", required=True, type=str,
                        help="Source case directory (already prepared with whatever modification).")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Where to copy the case and run.")
    parser.add_argument("--output", required=True, type=str, help="Path to write result JSON.")
    parser.add_argument("--openfoam-bashrc", default="", type=str,
                        help="Override OpenFOAM bashrc path.")
    parser.add_argument("--timeout", default=21600, type=int)
    args = parser.parse_args()

    result = run(
        base_case=Path(args.base_case).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        openfoam_bashrc=Path(args.openfoam_bashrc).expanduser().resolve() if args.openfoam_bashrc else None,
        timeout_seconds=args.timeout,
    )
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "stdout_tail" and k != "stderr_tail"},
                     indent=2, default=str))
    return 0 if result.get("status") == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())

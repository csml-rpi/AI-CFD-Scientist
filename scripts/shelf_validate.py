#!/usr/bin/env python3
"""Validate a shelf build against its frozen checks.

  python scripts/shelf_validate.py <name> [--openfoam <WM_PROJECT_DIR>] [--check <name>]

Each check in solvers/<name>/checks.py runs one of the solver's own tutorials
(copied into the build's validation/ folder, or run in place when the tutorial
compiles code against the solver's tree), possibly changed first by the check's
prepare(), in a sandbox where only the build folder is writable. Its measure()
reads numbers from the run's own output, and each is compared with a reference
that does not come from the solver -- theory, data shipped in the repository,
an analytical solution -- within the check's tolerance.

The verdict is the framework's, not the solver's or any agent's: the build is
validated only if every check produced every number within tolerance. A check
with no reference can only show that the build runs. shelf.json records each
number, its reference, the tolerance, and the SHA-256 of checks.py, so a check
changed after a result is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cfd_langgraph import solver_shelf as shelf  # noqa: E402
from cfd_langgraph.openfoam_version import get_profile  # noqa: E402
from shelf_install import build_env  # noqa: E402


def load_checks(solver: Dict[str, Any]) -> tuple:
    path = Path(solver["_dir"]) / "checks.py"
    spec = importlib.util.spec_from_file_location(f"checks_{solver['name']}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "CHECKS", [])), hashlib.sha256(path.read_bytes()).hexdigest()


def runner(env: Dict[str, str], writable: Path) -> Callable[..., Dict[str, Any]]:
    """Run a shell command in a case folder, sandboxed: the machine read-only
    except the build folder and a private /tmp."""
    bwrap = shutil.which("bwrap")

    def run(cmd: str, cwd: Path, timeout: int = 3600) -> Dict[str, Any]:
        inner = ["bash", "-c", cmd]
        argv = ([bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--tmpfs", "/tmp",
                 "--bind", str(writable), str(writable), "--proc", "/proc", "--dev-bind", "/dev", "/dev",
                 "--chdir", str(cwd), *inner] if bwrap else inner)
        started = time.monotonic()
        try:
            proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = -9, str(exc.stdout or ""), f"timed out after {timeout}s"
        return {"rc": rc, "stdout": out[-4000:], "stderr": err[-4000:],
                "seconds": round(time.monotonic() - started, 1)}

    return run


def port_tutorial(case: Path, tutorial: Path, bdir: Path, profile: Dict[str, Any],
                  check: Dict[str, Any]) -> Path:
    """The tutorial, as it runs on the installed version: unchanged if it
    already does, else ported by scripts/translate_case.py (which may change
    only what the version requires, and must pass the same run check). The
    port is kept with the build, as the tutorial the case writer is shown, and
    reused."""
    rel = tutorial.relative_to(bdir / "src")
    ported = bdir / "tutorials_ported" / rel
    record = ported.parent / f"{ported.name}.translation.json"
    doc = json.loads(record.read_text()) if record.is_file() else {}
    if doc.get("status") not in ("aligned", "translated"):
        proc = subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "translate_case.py"), "--case", str(case),
             "--out", str(ported), "--output", str(record), "--openfoam", profile["project_dir"],
             "--extra-path", str(bdir / "bin"), "--extra-lib", str(bdir / "lib")],
            capture_output=True, text=True, timeout=4 * 3600)
        doc = json.loads(record.read_text()) if record.is_file() else {"status": "failed", "stderr": proc.stderr[-1500:]}
    print(f"    tutorial {rel}: {doc.get('status')}"
          + (f" ({' '.join(str(doc.get('agent_summary', '')).split())[:200]})" if doc.get("status") == "translated" else ""),
          flush=True)
    if doc.get("status") not in ("aligned", "translated"):
        raise RuntimeError(f"the tutorial {rel} does not run on {profile['label']} and could not be ported: "
                           f"{(doc.get('check') or {}).get('problems') or doc.get('stderr', '')}")
    shutil.rmtree(case)
    shutil.copytree(ported, case, symlinks=True)
    return case


def judge(values: Dict[str, Any], expect: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in expect:
        v = values.get(e["quantity"])
        ref = float(e["reference"])
        entry = {**e, "value": v}
        if not isinstance(v, (int, float)) or v != v:
            entry.update(passed=False, why="not measured")
        elif "abs_tol" in e:
            entry.update(error=abs(v - ref), passed=abs(v - ref) <= float(e["abs_tol"]))
        else:
            rel = abs(v - ref) / abs(ref) if ref else float("inf")
            entry.update(error=rel, passed=rel <= float(e["rel_tol"]))
        out.append(entry)
    return out


def validate(name: str, openfoam_path: str, only: str = "") -> Dict[str, Any]:
    profile = get_profile(openfoam_path)
    solver = shelf.spec(name)
    name = solver["name"]
    rec = shelf.record(name, profile)
    if rec.get("status") not in ("built", "runs", "validated", "failed") or not rec.get("ref"):
        raise SystemExit(f"{name} has no build for {profile.get('label')}; run shelf_install.py first")
    bdir = shelf.build_dir(name, profile)
    env = build_env(profile, bdir, 8, {}, recipe_dir=solver["_dir"])
    run = runner(env, bdir)
    checks, digest = load_checks(solver)
    results: List[Dict[str, Any]] = []
    for check in checks:
        if only and check["name"] != only:
            continue
        slug = "".join(ch if ch.isalnum() else "_" for ch in check["name"])[:60]
        tutorial = bdir / "src" / check["tutorial"]
        if check.get("in_place"):
            case = tutorial
        else:
            case = bdir / "validation" / slug
            if case.exists():
                shutil.rmtree(case)
            shutil.copytree(tutorial, case, symlinks=True)
        if check.get("translate", True) and not check.get("in_place"):
            case = port_tutorial(case, tutorial, bdir, profile, check)
        ctx = {"src": bdir / "src", "bdir": bdir, "profile": profile, "env": env, "run": run}
        entry: Dict[str, Any] = {"name": check["name"], "case": str(case),
                                 "reference_source": check.get("source", "")}
        try:
            if check.get("prepare"):
                check["prepare"](case, ctx)
            out = run(check["run"], case, int(check.get("timeout", 3600)))
            entry["run"] = {"rc": out["rc"], "seconds": out["seconds"], "stderr_tail": out["stderr"][-1500:]}
            values = check["measure"](case, ctx) if out["rc"] == 0 else {}
            entry["values"] = values
            entry["expect"] = judge(values, check.get("expect") or [])
            entry["ran"] = out["rc"] == 0
            entry["passed"] = entry["ran"] and bool(entry["expect"]) and all(e["passed"] for e in entry["expect"])
        except Exception as exc:  # noqa: BLE001
            entry.update(passed=False, ran=False, error=f"{type(exc).__name__}: {exc}")
        results.append(entry)
        print(f"[{'PASS' if entry.get('passed') else 'FAIL'}] {name}: {check['name']}", flush=True)
        for e in entry.get("expect") or []:
            print(f"    {e['quantity']}: {e.get('value')} vs {e['reference']} "
                  f"({'ok' if e.get('passed') else 'OUT'}; {e.get('why') or e.get('error')})", flush=True)
    if only:
        results = [c for c in rec.get("checks") or [] if c.get("name") != only] + results
    status, results = verdict(results, checks)
    return shelf.write_record(name, profile, status=status, checks=results, checks_sha256=digest)


def verdict(results: List[Dict[str, Any]], checks: List[Dict[str, Any]]) -> tuple:
    """The build's status from its check results: validated only if every
    check in checks.py has a result and passed. A check never run counts as
    failed, so one run alone validates nothing."""
    ran_names = {r.get("name") for r in results}
    results = list(results) + [{"name": c["name"], "passed": False, "ran": False, "error": "not run yet"}
                               for c in checks if c["name"] not in ran_names]
    every = bool(results) and all(r.get("passed") for r in results)
    ran_all = bool(results) and all(r.get("ran") for r in results)
    no_refs = not any(r.get("expect") for r in results)
    return ("validated" if every else ("runs" if ran_all and no_refs else "failed")), results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("--openfoam", default="")
    ap.add_argument("--check", default="", help="run only this check")
    a = ap.parse_args()
    doc = validate(a.name, a.openfoam, a.check)
    print(json.dumps({"status": doc.get("status"),
                      "checks": [(c["name"], c.get("passed")) for c in doc.get("checks", [])]}, indent=2))
    return 0 if doc.get("status") == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())

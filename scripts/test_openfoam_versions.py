"""The harness on more than one OpenFOAM: the version profile read from each
installation, the lookups built on it, the case runner's solver and log names,
the tutorial references, and the check that decides whether a starter case
needs translating.

Each installation is exercised in its own process, with its own bashrc sourced,
the way a study runs. Installations not present on this machine are skipped.

Run: python scripts/test_openfoam_versions.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLS = {
    "of10": "/mnt/sda1/openfoam10",
    "of13": "/mnt/sda1/openfoam_versions/of13/opt/openfoam13",
    "v2512": "/mnt/sda1/openfoam_versions/v2512/usr/lib/openfoam/openfoam2512",
}
OF10_CAVITY = "/mnt/sda1/openfoam10/tutorials/incompressible/icoFoam/cavity/cavity"

CHILD = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "src")
from cfd_langgraph import openfoam_version as ov
from cfd_langgraph.foam_native import allrun, rag
from cfd_langgraph.foam_native import loop
from cfd_langgraph.case_translation import check_case
p = ov.get_profile()
out = {"label": p["label"], "style": p["solver_style"], "files": p["files"], "post": p["post_process"],
       "choices": ov.solver_choices(p)[:5], "simpleFoam": ov.run_command(p, "simpleFoam")}
tut = ov.find_tutorials(p, "simpleFoam", limit=1)
out["tutorial"] = str(tut[0]) if tut else ""
if tut:
    s = ov.case_solver(tut[0], p)
    out["case_solver"] = s
    out["post_cmd"] = ov.post_process_command(p, s, "wallShearStress")
    out["allrun"] = allrun.build_allrun_script("blockMesh", case_solver=(s["module"] or s["application"]))
ref = rag._installation_tutorials("steady turbulent flow over a backward facing step", "simpleFoam")
out["ref_ok"] = ref.get("ok")
out["ref_has"] = {name: name in ref.get("tutorial_reference", "") for name in
                  ("momentumTransport", "turbulenceProperties", "transportProperties", "physicalProperties")}
with tempfile.TemporaryDirectory() as td:
    case = Path(td) / "c"; (case / "constant").mkdir(parents=True)
    (case / "constant" / "physicalProperties").write_text("x")
    (case / "constant" / "transportProperties").write_text("y")
    loop.drop_unread_transport_properties(case)
    out["kept_transportProperties"] = (case / "constant" / "transportProperties").is_file()
out["cavity_check"] = {k: v for k, v in check_case(sys.argv[1], timeout=300).items() if k in ("ok", "stage", "problems")}
sys.path.insert(0, "scripts")
import code_mod_prepare, code_mod_apply_compile, code_mod_runtime, open_ended_discovery
wm = Path(p["project_dir"])
out["class_files"] = [str(f.relative_to(wm / "src")) for f in code_mod_prepare._class_files(wm, ["RASModel", "kOmegaSST"])]
out["source_files"] = [str(f.relative_to(wm / "src")) for f in code_mod_prepare._class_files(wm, ["fvModel", "fvOption"])]
out["of10_rules"] = code_mod_apply_compile._of10_rules()
out["coded_note"] = open_ended_discovery._coded_source_note()
out["flavor"] = code_mod_runtime._detect_openfoam_flavor()
print("RESULT " + json.dumps(out))
'''

FAILURES: list[str] = []


def check(name: str, cond: object, detail: object = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {str(detail)[:900]}" if detail != "" else ""))


def run_on(install: str) -> dict:
    with tempfile.TemporaryDirectory() as cache:
        script = Path(cache) / "child.py"
        script.write_text(CHILD)
        cmd = (f'source "{install}/etc/bashrc" >/dev/null 2>&1; export CFD_SCIENTIST_CACHE="{cache}"; '
               f'cd "{ROOT}" && "{sys.executable}" "{script}" "{OF10_CAVITY}"')
        env = {k: v for k, v in os.environ.items() if not k.startswith(("WM_", "FOAM_"))}
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800, env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), "")
    if not line:
        return {"error": proc.stdout[-1500:] + proc.stderr[-1500:]}
    return json.loads(line[7:])


results = {}
for name, path in INSTALLS.items():
    if not (Path(path) / "etc" / "bashrc").is_file():
        print(f"(skipping {name}: not installed at {path})")
        continue
    print(f"== {name}")
    r = results[name] = run_on(path)
    check(f"{name}: the profile is read", "label" in r, r)
    if "label" not in r:
        continue
    check(f"{name}: a tutorial run by simpleFoam (or its module) is found", r["tutorial"], r)
    check(f"{name}: the reference comes from this installation's tutorials", r["ref_ok"], r)
    turbulence = r["files"]["turbulence"]
    check(f"{name}: the reference uses this version's turbulence file ({turbulence})",
          r["ref_has"].get(turbulence), r["ref_has"])

if "of10" in results and "label" in results["of10"]:
    r = results["of10"]
    check("of10: standalone solvers, momentumTransport, postProcess",
          r["style"] == "applications" and r["files"]["turbulence"] == "momentumTransport"
          and r["post"] == "postProcess" and r["simpleFoam"] == "simpleFoam", r)
    check("of10: the duplicate transportProperties is removed", not r["kept_transportProperties"], r)
    check("of10: its own cavity runs", r["cavity_check"]["ok"], r["cavity_check"])
if "of13" in results and "label" in results["of13"]:
    r = results["of13"]
    check("of13: solver modules, simpleFoam -> foamRun -solver incompressibleFluid",
          r["style"] == "modules" and r["simpleFoam"] == "foamRun -solver incompressibleFluid", r)
    check("of13: the Allrun runs foamRun for a module",
          "runApplication foamRun -solver incompressibleFluid" in r.get("allrun", ""), r.get("allrun"))
    check("of13: post-processing goes through foamPostProcess -solver",
          r.get("post_cmd", "").startswith("foamPostProcess -solver incompressibleFluid"), r.get("post_cmd"))
    check("of13: an OpenFOAM 10 cavity still runs", r["cavity_check"]["ok"], r["cavity_check"])
if "v2512" in results and "label" in results["v2512"]:
    r = results["v2512"]
    check("v2512: turbulenceProperties, transportProperties, fvOptions",
          r["files"] == {"turbulence": "turbulenceProperties", "transport": "transportProperties",
                         "thermo": "thermophysicalProperties", "sources": "fvOptions",
                         "constraints": "fvOptions"}, r["files"])
    check("v2512: transportProperties is kept beside physicalProperties", r["kept_transportProperties"], r)
    check("v2512: post-processing is <solver> -postProcess",
          r.get("post_cmd", "").startswith("simpleFoam -postProcess"), r.get("post_cmd"))
    check("v2512: an OpenFOAM 10 cavity does not run, so it would be translated",
          not r["cavity_check"]["ok"] and "transportProperties" in " ".join(r["cavity_check"]["problems"]),
          r["cavity_check"])

print("== discovery follows the installed version")
for name, expect in (("of10", dict(rules=True, tree="MomentumTransportModels", src="fvModel.H", flavor="of-org", note="fvModels")),
                     ("of13", dict(rules=False, tree="MomentumTransportModels", src="fvModel.H", flavor="of-org", note="fvModels")),
                     ("v2512", dict(rules=False, tree="TurbulenceModels", src="fvOption.H", flavor="esi", note="scalarCodedSource"))):
    r = results.get(name) or {}
    if "label" not in r:
        continue
    check(f"{name}: the OpenFOAM 10 compile rules apply only on OpenFOAM 10", r["of10_rules"] is expect["rules"], r["of10_rules"])
    check(f"{name}: parent classes are found in this installation's tree ({expect['tree']})",
          any(f.startswith(expect["tree"]) and f.endswith("RASModel.H") for f in r["class_files"])
          and any(f.endswith("kOmegaSST.H") for f in r["class_files"])
          and any(f.endswith("Make/options") for f in r["class_files"]), r["class_files"])
    check(f"{name}: the source-term base class is {expect['src']}",
          any(f.endswith(expect["src"]) for f in r["source_files"]), r["source_files"])
    check(f"{name}: the coded source form and fork match ({expect['note']}, {expect['flavor']})",
          expect["note"] in r["coded_note"] and r["flavor"] == expect["flavor"], (r["coded_note"], r["flavor"]))

print("== translated starter cases replace the originals")
sys.path.insert(0, str(ROOT / "src"))
from cfd_langgraph.starter_cases import resolve_starter_base_cases  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    starter = Path(td) / "starter"
    (starter / "case" / "system").mkdir(parents=True)
    (starter / "case" / "system" / "controlDict").write_text("application icoFoam;\n")
    copy = Path(td) / "study" / "starter_translated" / "case"
    (copy / "system").mkdir(parents=True)
    (copy / "system" / "controlDict").write_text("application icoFoam;\n")
    su = {"starter_dir": str(starter), "base_case_path": "case"}
    check("without a translation the starter's own case is used",
          resolve_starter_base_cases(su).dirs == [starter / "case"])
    su["translated_cases"] = {str((starter / "case").resolve()): {"status": "translated", "path": str(copy)}}
    cases = resolve_starter_base_cases(su)
    check("a translated case is used in its place",
          cases.dirs == [copy] and cases.originals == [starter / "case"], cases)
    su["translated_cases"][str((starter / "case").resolve())]["status"] = "failed"
    check("a failed translation leaves the original", resolve_starter_base_cases(su).dirs == [starter / "case"])

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

"""Reproduce the cavity_r2 failure shape and check every patched path."""
import importlib.util, shutil, sys, tempfile
from pathlib import Path

ROOT = Path("/home/somasn/Desktop/AI-CFD-Scientist")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

cc = load("case_clean", "src/cfd_langgraph/foam_native/case_clean.py")

CONTROL_FRESH = 'application simpleFoam;\nstartFrom startTime;\nstartTime 0;\nendTime 1000;\n'
CONTROL_LATEST = 'application simpleFoam;\nstartFrom latestTime;\nstartTime 0;\n'
CONTROL_RESTART = 'application simpleFoam;\nstartFrom startTime;\nstartTime 415;\n'
CONTROL_COMMENTED = 'application simpleFoam;\n// startFrom latestTime;\nstartFrom startTime;\nstartTime 0;\n'

def make_case(root, control=CONTROL_FRESH, times=("0","100","415","1000")):
    c = Path(root); (c/"system").mkdir(parents=True); (c/"constant"/"polyMesh").mkdir(parents=True)
    (c/"system"/"controlDict").write_text(control)
    (c/"constant"/"polyMesh"/"owner").write_text("nCells 1024;")
    (c/"constant"/"100").write_text("not a time dir, nested")   # must survive
    for t in times:
        (c/t).mkdir(); (c/t/"U").write_text(f"internalField at {t}")
    (c/"postProcessing").mkdir(); (c/"log.simpleFoam").write_text("End")
    (c/"processor0").mkdir()
    return c

fails = []
def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}\n        got={got}\n        want={want}" if not ok else f"  PASS  {label}")
    if not ok: fails.append(label)

print("== case_clean.remove_stale_time_dirs")
with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"fresh")
    check("fresh case drops non-zero times", sorted(cc.remove_stale_time_dirs(c)), ["100","1000","415"])
    check("time 0 kept", (c/"0").is_dir(), True)
    check("nested constant/100 survives", (c/"constant"/"100").is_file(), True)

with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"latest", control=CONTROL_LATEST)
    check("startFrom latestTime is left alone", cc.remove_stale_time_dirs(c), [])
    check("  its times still present", sorted(p.name for p in c.iterdir() if p.name in {"100","415","1000"}), ["100","1000","415"])

with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"restart", control=CONTROL_RESTART)
    check("non-zero startTime is left alone", cc.remove_stale_time_dirs(c), [])

with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"commented", control=CONTROL_COMMENTED)
    check("commented-out latestTime does not protect", sorted(cc.remove_stale_time_dirs(c)), ["100","1000","415"])

with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"forced", control=CONTROL_LATEST)
    check("force=True overrides restart", sorted(cc.remove_stale_time_dirs(c, force=True)), ["100","1000","415"])

with tempfile.TemporaryDirectory() as d:
    c = Path(d)/"nocontrol"; c.mkdir(); (c/"1000").mkdir()
    check("no controlDict -> never deletes", cc.remove_stale_time_dirs(c), [])

print("== scripts/foam_run_simple.py::_copy_case")
frs = load("frs", "scripts/foam_run_simple.py")
with tempfile.TemporaryDirectory() as d:
    src = make_case(Path(d)/"src"); dst = Path(d)/"dst"
    frs._copy_case(src, dst)
    check("copy drops non-zero times", sorted(p.name for p in dst.iterdir() if p.name.replace('.','').isdigit()), ["0"])
    check("copy keeps nested constant/100", (dst/"constant"/"100").is_file(), True)
    check("copy drops processor0", (dst/"processor0").exists(), False)

print("== scripts/code_mod_runtime.py::_copy_case")
cmr = load("cmr", "scripts/code_mod_runtime.py")
with tempfile.TemporaryDirectory() as d:
    src = make_case(Path(d)/"src"); dst = Path(d)/"dst"
    cmr._copy_case(src, dst)
    check("copy drops non-zero times", sorted(p.name for p in dst.iterdir() if p.name.replace('.','').isdigit()), ["0"])
    check("copy keeps nested constant/100", (dst/"constant"/"100").is_file(), True)
    check("processor* bug fixed", (dst/"processor0").exists(), False)

print("== scripts/foam_run.py::_prepare_output_dir_for_case_copy")
fr = load("fr", "scripts/foam_run.py")
with tempfile.TemporaryDirectory() as d:
    c = make_case(Path(d)/"out")
    (c/"figs").mkdir()
    fr._prepare_output_dir_for_case_copy(c)
    check("non-zero times removed", sorted(p.name for p in c.iterdir() if p.name.replace('.','').isdigit()), [])
    check("figs preserved", (c/"figs").is_dir(), True)

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)

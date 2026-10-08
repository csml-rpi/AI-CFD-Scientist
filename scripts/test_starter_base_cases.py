"""Resolving the OpenFOAM cases a starter ships, and matching one to its reference.

``base_case_path`` may be a list, a single path, or one string naming several.
Three things copy a starter case rather than authoring one -- requirement
context, the mesh-gate seed, and the function-object block generated cases
inherit -- and all read it through this module.

Covers: each spelling of the value; a declared case that does not exist;
a starter with no case; the prefix a derived directory carries; and picking
the right reference file for a case when a study declares several.

Run: python scripts/test_starter_base_cases.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.starter_cases import resolve_starter_base_cases  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def make_case(root: Path, name: str) -> Path:
    case = root / name
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application     simpleFoam;\n")
    return case


tmp = Path(tempfile.mkdtemp())
starter = tmp / "starter_two_cases"
starter.mkdir()
make_case(starter, "duct_Re1000")
make_case(starter, "duct_Re4000")
(starter / "reference_data").mkdir()


def resolve(base_case_path, starter_dir: Path = starter):
    return resolve_starter_base_cases(
        {"starter_dir": str(starter_dir), "base_case_path": base_case_path}
    )


print("== 1. one relative path, the shape that always worked")
r = resolve("duct_Re1000")
check("resolves", [d.name for d in r.dirs] == ["duct_Re1000"], str(r.dirs))
check("primary is that case", r.primary is not None and r.primary.name == "duct_Re1000")
check("nothing to report", r.problem() == "", r.problem())
check("not scanned", not r.scanned)


print("== 2. two cases named in one string")
r = resolve("duct_Re1000/; duct_Re4000/")
check("both cases resolve",
      [d.name for d in r.dirs] == ["duct_Re1000", "duct_Re4000"], str(r.dirs))
check("declaration order is kept", r.primary.name == "duct_Re1000")
check("not treated as unresolved", not r.unresolved)
check("no scan was needed", not r.scanned)
check("nothing to report", r.problem() == "", r.problem())

for sep in (", ", "\n", ";", " ; "):
    got = resolve(f"duct_Re1000{sep}duct_Re4000")
    check(f"separator {sep!r} works", len(got.dirs) == 2, str(got.dirs))


print("== 3. a JSON array, which is what the schema now asks for")
r = resolve(["duct_Re1000", "duct_Re4000"])
check("both resolve", [d.name for d in r.dirs] == ["duct_Re1000", "duct_Re4000"])
check("declared is readable in a message", "duct_Re4000" in r.declared, r.declared)

r = resolve(["duct_Re1000", "duct_Re1000"])
check("a repeated case is listed once", len(r.dirs) == 1, str(r.dirs))

r = resolve([str(starter / "duct_Re4000")])
check("an absolute path resolves", [d.name for d in r.dirs] == ["duct_Re4000"], str(r.dirs))


print("== 4. a declared case that is not there")
empty = tmp / "starter_no_case"
(empty / "reference_data").mkdir(parents=True)
r = resolve("no_such_case", starter_dir=empty)
check("resolves to nothing", r.dirs == [])
check("and says so", r.unresolved)
check("the message names the value and the folder",
      "no_such_case" in r.problem() and str(empty) in r.problem(), r.problem())

# A wrong name next to real cases is recoverable: scan, but say so.
r = resolve("no_such_case")
check("a scan recovers the real cases", len(r.dirs) == 2, str(r.dirs))
check("the recovery is flagged", r.scanned and "does not resolve" in r.problem(), r.problem())
check("but it is not an error", not r.unresolved)


print("== 5. a starter with no OpenFOAM case at all")
for value in (None, "", []):
    r = resolve(value, starter_dir=empty)
    check(f"{value!r}: no case", r.primary is None and r.dirs == [])
    check(f"{value!r}: no complaint", r.problem() == "" and not r.unresolved, r.problem())


print("== 6. a declaration of nothing, next to real cases")
r = resolve(None)
check("the cases are found anyway", len(r.dirs) == 2, str(r.dirs))
check("and that is stated", r.scanned and "names no base_case_path" in r.problem(), r.problem())


print("== 7. a case directory whose name contains a separator")
odd = tmp / "starter_odd"
odd.mkdir()
make_case(odd, "duct, Re=3418")
r = resolve("duct, Re=3418", starter_dir=odd)
check("the whole string is tried as a path first",
      [d.name for d in r.dirs] == ["duct, Re=3418"], str(r.dirs))
check("so it is not split", not r.scanned)


print("== 8. a directory that only looks like a case")
half = tmp / "starter_half"
(half / "looks_like_a_case" / "constant").mkdir(parents=True)
r = resolve(None, starter_dir=half)
check("no controlDict means no case", r.dirs == [], str(r.dirs))


print("== 9. the tag a derived directory carries")
from cfd_langgraph.starter_cases import seed_dir_tag  # noqa: E402

check("no seed means no tag", seed_dir_tag(None) == "")
check("a normal case name is kept",
      seed_dir_tag(Path("/s/duct_Re1000")) == "duct_Re1000__",
      seed_dir_tag(Path("/s/duct_Re1000")))
check("the condition token survives, which is the point",
      "Re1000" in seed_dir_tag(Path("/s/duct_Re1000")))
check("separators cannot escape the directory",
      "/" not in seed_dir_tag(Path("/s/a b/c")) and ".." not in seed_dir_tag(Path("/s/..")),
      seed_dir_tag(Path("/s/a b/c")))
check("spaces and punctuation are replaced, not dropped",
      seed_dir_tag(Path("/s/duct, Re=3418")) == "duct__Re_3418__",
      seed_dir_tag(Path("/s/duct, Re=3418")))
check("a very long name is bounded", len(seed_dir_tag(Path("/s/" + "x"*300))) <= 66)
check("a name of only punctuation falls back to untagged",
      seed_dir_tag(Path("/s/___")) == "", seed_dir_tag(Path("/s/___")))
for base in ("baseline", "refined", "refined_2"):
    name = seed_dir_tag(Path("/s/duct_Re4000")) + base
    check(f"{base!r} is still distinguishable", name.endswith(base) and "Re4000" in name, name)


print("== 10. matching a case to the reference file that belongs to it")
from cfd_langgraph.starter_cases import match_reference_to_case, identity_tokens  # noqa: E402

REFS = [Path("ref/dns_duct_Re1000_corner_bisector.csv"),
        Path("ref/dns_duct_Re4000_corner_bisector.csv")]

got = match_reference_to_case(["duct_Re1000__baseline"], REFS)
check("the gate level matches its own condition", got == REFS[0], str(got))
got = match_reference_to_case(["duct_Re4000__refined_2"], REFS)
check("and the other one matches the other", got == REFS[1], str(got))
got = match_reference_to_case(["baseline", "duct_Re4000"], REFS)
check("a neutral name still matches via its recorded seed", got == REFS[1], str(got))

check("a single candidate needs no matching",
      match_reference_to_case(["anything"], REFS[:1]) == REFS[0])
check("no candidates gives nothing", match_reference_to_case(["x"], []) is None)

got = match_reference_to_case(["baseline"], REFS)
check("a case that says nothing matches nothing", got is None, str(got))
got = match_reference_to_case(["duct"], REFS)
check("matching only on what the files SHARE is not a match", got is None, str(got))
got = match_reference_to_case(["duct_Re1000_and_Re4000"], REFS)
check("a name matching both is not a match", got is None, str(got))

check("tokens ignore case and punctuation",
      identity_tokens("Plane-Duct_Re1000") == {"plane", "duct", "re1000"},
      str(identity_tokens("Plane-Duct_Re1000")))

THREE = REFS + [Path("ref/dns_duct_Re9000_profile.csv")]
check("it still works with three conditions",
      match_reference_to_case(["duct_Re9000__baseline"], THREE) == THREE[2])


print("== the mesh gate recognises the same case under another group name")
import shutil  # noqa: E402

from cfd_langgraph.manager.tools import _case_fingerprint  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    a = Path(tmp) / "a" / "case"
    for sub, name, text in (("0", "U", "uniform (1 0 0)"), ("constant", "physicalProperties", "nu 1e-3;"),
                            ("system", "controlDict", "endTime 200;")):
        (a / sub).mkdir(parents=True, exist_ok=True)
        (a / sub / name).write_text(text)
    b = Path(tmp) / "b" / "elsewhere"
    shutil.copytree(a, b)
    (b / "postProcessing").mkdir()
    (b / "100").mkdir()
    check("same set-up, different folder and outputs: same fingerprint", _case_fingerprint(a) == _case_fingerprint(b))
    (b / "constant" / "physicalProperties").write_text("nu 2e-3;")
    check("a changed set-up file changes it", _case_fingerprint(a) != _case_fingerprint(b))

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

"""Mesh-gate refinement and level reuse: uniform block refinement done in code,
and what makes a solved level reusable.

Run: python scripts/test_mesh_refine.py   (section 2 needs OpenFOAM 10)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.foam_native.loop import (  # noqa: E402
    _foam_text, _foam_tokens, refine_block_counts, resolve_openfoam_env,
)
from cfd_langgraph.manager.tools import _has_time_averages, _setup_differences  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
convertToMeters 1;
"""


def two_block_2d(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + """
vertices
(
    (0 0 0) (1 0 0) (2 0 0) (0 1 0) (1 1 0) (2 1 0)
    (0 0 0.1) (1 0 0.1) (2 0 0.1) (0 1 0.1) (1 1 0.1) (2 1 0.1)
);
nx 41;
blocks
(
    hex (0 1 4 3 6 7 10 9) left ($nx 30 1) simpleGrading (1 ((0.5 0.5 4) (0.5 0.5 0.25)) 1)
    hex (1 2 5 4 7 8 11 10) ($nx 30 1) simpleGrading (2 ((0.5 0.5 4) (0.5 0.5 0.25)) 1)
);
boundary
(
    walls { type wall; faces ((0 6 9 3) (2 5 11 8) (0 1 7 6) (1 2 8 7) (3 9 10 4) (4 10 11 5)); }
    frontAndBack { type empty; faces ((0 3 4 1) (1 4 5 2) (6 7 10 9) (7 8 11 10)); }
);
""")


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)

    print("== 1. OpenFOAM list tokens")
    t = _foam_tokens("( hex (0 1 2 3 4 5 6 7) zoneA (10 20 30) simpleGrading ((0.5 0.5 2) (0.5 0.5 0.5) 1) )")
    check("nested lists parse", isinstance(t[0][1], list) and t[0][2] == "zoneA" and t[0][3] == ["10", "20", "30"], t)
    check("round trip keeps every token", _foam_tokens(_foam_text(t[0])) == t)
    for bad in ("( a (b )", "( a ) )"):
        try:
            _foam_tokens(bad)
            check(f"unbalanced {bad!r} is refused", False)
        except ValueError:
            check(f"unbalanced {bad!r} is refused", True)

    print("== 2. uniform refinement in code")
    env = resolve_openfoam_env("")
    case = tmp / "two_block"
    two_block_2d(case / "system" / "blockMeshDict")
    (case / "system" / "controlDict").write_text(
        HEADER.replace("blockMeshDict", "controlDict").replace("convertToMeters 1;\n", "")
        + "application icoFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\n"
          "deltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
    have_foam = subprocess.run(["bash", "-c", "command -v foamDictionary"], env=env,
                               capture_output=True).returncode == 0
    if not have_foam:
        print("[SKIP] OpenFOAM not available")
    else:
        def cells() -> int:
            p = subprocess.run(["bash", "-c", "blockMesh > log.bm 2>&1 && checkMesh 2>&1 | grep -m1 'cells:'"],
                               cwd=case, env=env, capture_output=True, text=True)
            return int(p.stdout.split()[-1]) if p.stdout.split() else -1
        before = cells()
        why = refine_block_counts(case / "system" / "blockMeshDict", env)
        check("refines without error", why is None, why)
        after = cells()
        check("the mesh builds and is finer within the budget", before > 0 and before < after <= 2.25 * before,
              f"{before} -> {after}")
        blocks = subprocess.run(["foamDictionary", "-entry", "blocks", "-value", "-expand",
                                 str(case / "system" / "blockMeshDict")], env=env,
                                capture_output=True, text=True).stdout
        b = _foam_tokens(blocks)[0]
        check("both blocks got the same count along their shared face", b[3] == ["61", "45", "1"]
              and b[8] == ["61", "45", "1"], b)
        check("the one-cell direction stays one", b[3][2] == "1")
        check("the zone name and multi-section grading survive", b[2] == "left" and isinstance(b[5][1], list), b)

        print("== 2b. point and line samplers")
        import numpy as np

        from cfd_langgraph.foam_load import sample_line, sample_points
        for t in ("0", "1"):
            (case / t).mkdir(exist_ok=True)
            (case / t / "U").write_text(
                HEADER.replace("dictionary", "volVectorField").replace("blockMeshDict", "U")
                .replace("convertToMeters 1;\n", "")
                + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 2 0);\nboundaryField\n{\n"
                  "    walls { type fixedValue; value uniform (1 2 0); }\n"
                  "    frontAndBack { type empty; }\n}\n")
        s = sample_points(case, [[0.5, 0.5, 7.0], [1.5, 0.25, 0.0], [3.0, 0.5, 0.05]])
        check("a point inside reads the field there, whatever z is given in 2-D",
              bool(s["inside"][0]) and np.allclose(s["U"][0], [1, 2, 0]), s)
        check("a point outside the domain is flagged and NaN",
              not s["inside"][2] and np.isnan(s["U"][2]).all(), s)
        line = sample_line(case, [0.1, 0.5, 0], [1.9, 0.5, 0], n=50)
        check("a line returns one value per point with positions",
              line["U"].shape == (50, 3) and line["position"].shape == (50, 3)
              and abs(line["distance"][-1] - 1.8) < 1e-9 and bool(line["inside"].all()), line["U"].shape)

    print("== 3. what makes a solved level reusable")
    parent, child = tmp / "p", tmp / "c"
    for d in (parent, child):
        for f, text in (("system/fvSolution", "solvers {}"), ("system/fvSchemes", "schemes"),
                        ("constant/physicalProperties", "nu 0.001;"),
                        ("system/blockMeshDict", f"mesh of {d.name}"), ("system/controlDict", f"end {d.name}")):
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text(text)
    (child / "constant" / "polyMesh").mkdir()
    (child / "constant" / "polyMesh" / "points").write_text("points")
    check("a mesh-only difference is the same set-up", _setup_differences(parent, child) == [],
          _setup_differences(parent, child))
    (child / "system" / "fvSolution").write_text("solvers { other }")
    check("changed numerics are a different set-up", _setup_differences(parent, child) == ["system/fvSolution"])
    (child / "constant" / "extra").write_text("x")
    check("a file only one level has counts", "constant/extra" in _setup_differences(parent, child))
    (parent / "100").mkdir()
    (parent / "100" / "U").write_text("u")
    check("no time-averaged field", not _has_time_averages(parent))
    (parent / "200").mkdir()
    (parent / "200" / "UMean").write_text("u")
    check("the latest time holds a time average", _has_time_averages(parent))

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

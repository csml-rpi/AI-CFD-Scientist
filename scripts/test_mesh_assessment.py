"""Mesh-gate review helpers that need no model: history statistics, transient
detection, time-averaging injection, and the refinement instruction.

Run: python scripts/test_mesh_assessment.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.mesh_assessment import (  # noqa: E402
    LevelAssessment, ensure_time_averaging, history_quantities, history_statistics, is_transient,
)

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def write_forces(case: Path, cl) -> None:
    d = case / "postProcessing" / "forceCoeffs" / "0"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["# Force coefficients", "# Time Cm Cd Cl"]
    for i in range(2000):
        t = 0.1 * (i + 1)
        lines.append(f"{t:g}\t0.0\t{1.2 + 0.05 * math.sin(2 * math.pi * 0.42 * t):.8e}\t{cl(t):.8e}")
    (d / "forceCoeffs.dat").write_text("\n".join(lines) + "\n")


def write_case(case: Path, ddt: str, extra_functions: str = "") -> None:
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "fvSchemes").write_text(
        f"ddtSchemes\n{{\n    default {ddt};\n}}\ngradSchemes\n{{\n    default Gauss linear;\n}}\n")
    (case / "system" / "controlDict").write_text(
        "application pimpleFoam;\nstartFrom latestTime; endTime 200; deltaT 0.01;\n"
        + (f"functions\n{{\n{extra_functions}}}\n" if extra_functions else ""))


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    print("== 1. history statistics")
    shed = tmp / "shed"
    write_forces(shed, lambda t: 0.6 * math.sin(2 * math.pi * 0.21 * t))
    stats = {s.column: s for s in history_statistics(shed)}
    check("every column after time is read", set(stats) == {"Cm", "Cd", "Cl"}, str(set(stats)))
    cl = stats.get("Cl")
    check("lift frequency found", cl and cl.dominant_frequency and abs(cl.dominant_frequency - 0.21) < 0.01,
          str(cl and cl.dominant_frequency))
    check("lift rms is the oscillation's", cl and abs(cl.fluctuation_rms - 0.6 / math.sqrt(2)) < 0.02,
          str(cl and cl.fluctuation_rms))
    check("drag mean over the settled window", abs(stats["Cd"].mean - 1.2) < 0.01, str(stats["Cd"].mean))
    check("a history settled from the start is used almost whole", cl and cl.window[0] < 20.0, str(cl and cl.window))

    ramp = tmp / "ramp"
    write_forces(ramp, lambda t: 0.6 * min(1.0, t / 80.0) * math.sin(2 * math.pi * 0.21 * t))
    rc = {s.column: s for s in history_statistics(ramp)}["Cl"]
    check("a start-up transient is left out of the window", 60.0 <= rc.window[0] <= 100.0, str(rc.window))
    check("and its statistics are the settled ones", abs(rc.fluctuation_rms - 0.6 / math.sqrt(2)) < 0.02,
          str(rc.fluctuation_rms))

    frozen = tmp / "frozen"
    write_forces(frozen, lambda t: 1e-9 * math.sin(2 * math.pi * 3.7 * t))
    fz = {s.column: s for s in history_statistics(frozen)}
    check("round-off gets no frequency", fz["Cl"].dominant_frequency is None, str(fz["Cl"].dominant_frequency))
    check("a constant column gets no frequency", fz["Cm"].dominant_frequency is None)

    q = history_quantities(history_statistics(shed))
    check("quantities are named by source and column",
          {"forceCoeffs.Cl.mean", "forceCoeffs.Cl.fluctuation_rms", "forceCoeffs.Cl.dominant_frequency"} <= set(q),
          str(sorted(q)))
    check("no frequency key where there is none",
          "forceCoeffs.Cl.dominant_frequency" not in history_quantities(history_statistics(frozen)))

    print("== 1b. a continued run's pieces are joined, and the window is split in halves")
    grow = tmp / "grow"
    d0 = grow / "postProcessing" / "forceCoeffs" / "0"
    d1 = grow / "postProcessing" / "forceCoeffs" / "150"
    d0.mkdir(parents=True); d1.mkdir(parents=True)
    head = "# Time Cm Cd Cl\n"
    def seg(t0, t1):
        rows = []
        t = t0
        while t < t1:
            amp = 0.01 * math.exp(t / 40.0)
            rows.append(f"{t:g}\t0\t1.0\t{amp * math.sin(2 * math.pi * 0.2 * t):.8e}")
            t += 0.1
        return head + "\n".join(rows) + "\n"
    (d0 / "forceCoeffs.dat").write_text(seg(0.1, 160))   # overlaps the restart at 150
    (d1 / "forceCoeffs.dat").write_text(seg(150.0, 300))
    g = {s.column: s for s in history_statistics(grow)}
    check("one series per column, not one per piece", len([s for s in history_statistics(grow) if s.column == "Cl"]) == 1)
    check("the joined history spans the whole run", g["Cl"].window[1] > 299, str(g["Cl"].window))
    check("growth shows as a larger second-half rms", g["Cl"].halves[1][1] > 1.1 * g["Cl"].halves[0][1], str(g["Cl"].halves))
    check("a history that never settles falls back to its last fifth", g["Cl"].window[0] >= 235.0, str(g["Cl"].window))
    check("a steady column has equal halves", abs(g["Cd"].halves[0][0] - g["Cd"].halves[1][0]) < 1e-9)
    check("quantities still named by function object", "forceCoeffs.Cl.mean" in history_quantities(history_statistics(grow)))

    print("== 2. transient detection")
    tr = tmp / "tr"; write_case(tr, "backward")
    st = tmp / "st"; write_case(st, "steadyState")
    eu = tmp / "eu"; write_case(eu, "Euler")
    check("backward is transient", is_transient(tr))
    check("Euler is transient", is_transient(eu))
    check("steadyState is not", not is_transient(st))
    check("no fvSchemes is not", not is_transient(tmp / "missing"))

    print("== 3. time averaging")
    check("added to a transient case", ensure_time_averaging(tr))
    text = (tr / "system" / "controlDict").read_text()
    check("starts halfway through", "timeStart 100;" in text, text)
    check("added once", not ensure_time_averaging(tr) and text.count("fieldAverage") == 1)
    check("steady case untouched", not ensure_time_averaging(st)
          and "fieldAverage" not in (st / "system" / "controlDict").read_text())
    fn = tmp / "fn"
    write_case(fn, "backward", "    forces\n    {\n        type forceCoeffs;\n    }\n")
    ensure_time_averaging(fn)
    t2 = (fn / "system" / "controlDict").read_text()
    check("goes inside an existing functions block",
          t2.count("functions") == 1 and "forceCoeffs" in t2 and "gateFieldAverage" in t2, t2)
    own = tmp / "own"
    write_case(own, "backward", "    avg\n    {\n        type fieldAverage;\n    }\n")
    check("an existing average is left alone", not ensure_time_averaging(own))

    print("== 4. refinement instruction")
    a = LevelAssessment(physically_plausible=False, numerically_adequate=False,
                        refine_where="In the wake behind the body.", refine_how="Halve the cross-stream spacing")
    ins = a.refinement_instruction()
    check("where and how both carried", "wake" in ins and "Halve" in ins, ins)
    check("resolution-only rule kept", "resolution change only" in ins, ins)
    check("no doubled full stops", ".." not in ins, ins)
    check("not acceptable", not a.acceptable)
    check("nothing to refine gives no instruction",
          LevelAssessment(True, True).refinement_instruction() == "")

    print("== 5. pair verdict, decided in code")
    from cfd_langgraph.mesh_assessment import (
        apply_quantity_review, is_history_dropped, pair_verdict, quantity_spec, relative_changes,
    )
    qa = {"u_min": -0.38, "p_max": 1.03, "forces.Cl.mean": 0.001, "forces.Cl.fluctuation_rms": 0.3}
    qb = {"u_min": -0.385, "p_max": 1.30, "forces.Cl.mean": 0.003, "forces.Cl.fluctuation_rms": 0.302}
    pct = relative_changes(qa, qb, list(qa))
    check("a zero-mean history is judged on its fluctuation", pct["forces.Cl.mean"] < 1.0, pct)
    check("a plain quantity is judged on its own size", abs(pct["p_max"] - 26.2) < 0.1, pct)
    v = pair_verdict(pct, qa, qb)
    check("one quantity over 5% blocks convergence", not v["converged"] and v["over_tolerance"] == ["p_max"], v)
    check("the reason names it with both values", "p_max" in v["reason"] and "1.03" in v["reason"], v["reason"])
    v2 = pair_verdict({k: p for k, p in pct.items() if k != "p_max"}, qa, qb)
    check("all within 5% converges", v2["converged"], v2)
    check("an empty comparison never converges", not pair_verdict({}, qa, qb)["converged"])

    print("== 6. quantity review applied")
    specs = [quantity_spec(n, f"definition of {n}", transient=False, from_inventory=True)
             for n in ("max_abs_u", "max_abs_p", "u_min_vertical_centreline")]
    hist = ["forces.Cd.mean", "forces.Cd.fluctuation_rms", "residuals.p.mean"]
    review = {"drop": [{"name": "max_abs_p", "reason": "sits in a corner"},
                       {"name": "residuals.p", "reason": "a residual"}],
              "add": [{"name": "vortex_centre_x", "definition": "x of the streamfunction minimum"}],
              "reason": "seen"}
    kept, hdrop, rec = apply_quantity_review(specs, hist, review, transient=False)
    names = [s["name"] for s in kept]
    check("the dropped field quantity is gone", "max_abs_p" not in names and "max_abs_u" in names, names)
    check("the added quantity is measured", "vortex_centre_x" in names, names)
    check("dropping an output's column drops its statistics", hdrop == ["residuals.p.mean"], hdrop)
    check("both drops are recorded with reasons", len(rec["dropped"]) == 2, rec)
    check("is_history_dropped matches by column",
          is_history_dropped("forces.Cl.mean", {"forces.Cl"}) and not is_history_dropped("forces.Cd.mean", {"forces.Cl"}))
    all_out = {"drop": [{"name": s["name"], "reason": "x"} for s in specs] + [{"name": h, "reason": "x"} for h in hist],
               "add": [], "reason": ""}
    k2, h2, r2 = apply_quantity_review(specs, hist, all_out, transient=False)
    check("a review that leaves too few is not applied", k2 == specs and h2 == [] and "not_applied" in r2, r2)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

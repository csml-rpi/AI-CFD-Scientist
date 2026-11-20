from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_rag_queries(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Answer several index questions in one subprocess.

    A subprocess, not an in-process call, so the ~2 GB embedding model is
    released the moment retrieval is done rather than being held for the whole
    multi-hour study. One subprocess for all three questions, not three: the
    index load dominates the cost, and it used to be paid once per question.

    Returns one payload per request, in order; a failed batch degrades every
    entry rather than raising, because the caller already has a documented
    fallback for "retrieval unavailable".
    """
    failed = lambda detail: [{"ok": False, "stderr": detail} for _ in requests]
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/rag_query.py", "--batch", json.dumps(requests), "--output", "-"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        return failed(str(exc))
    if proc.returncode != 0:
        return failed((proc.stderr or "")[-800:])
    try:
        items = json.loads(proc.stdout).get("batch", [])
    except Exception:
        return failed("unparseable rag_query.py output")
    if len(items) != len(requests):
        return failed(f"expected {len(requests)} results, got {len(items)}")
    for item in items:
        item["ok"] = True
    return items


def _results_to_text(results: List[Any]) -> str:
    if not results:
        return ""
    per_item_budget = max(500, 8000 // len(results))
    parts = []
    for i, r in enumerate(results, 1):
        text = json.dumps(r, ensure_ascii=False, default=str)[:per_item_budget]
        parts.append(f"<similar_case_{i}>\n{text}\n</similar_case_{i}>")
    return "\n".join(parts)


_INDEX_VERSION_FILE = "openfoam_version.json"


def _index_matches_installation() -> bool:
    """Whether the prebuilt tutorial index was built from this installation's
    version. An index carries openfoam_version.json naming the version it was
    built from; the prebuilt one has none and was built from openfoam.org 10."""
    from cfd_langgraph.foam_native.faiss_index import index_root
    from cfd_langgraph.openfoam_version import get_profile

    profile = get_profile()
    if not profile.get("ok"):
        return True
    built_from = {"fork": "openfoam.org", "version": "10"}
    root = index_root()
    if root is not None and (root / _INDEX_VERSION_FILE).is_file():
        try:
            built_from = json.loads((root / _INDEX_VERSION_FILE).read_text())
        except (OSError, ValueError):
            pass
    return (built_from.get("fork"), str(built_from.get("version"))) == (profile["fork"], str(profile["version"]))


def _build_installed_index() -> bool:
    """Build the tutorial indices for the installed version, once. A failed
    build is remembered beside where the index would go, so it is not retried
    on every case."""
    from cfd_langgraph.foam_native.faiss_index import versioned_index_dir
    from cfd_langgraph.openfoam_version import get_profile

    profile = get_profile()
    if not profile.get("ok"):
        return False
    target = versioned_index_dir(profile)
    failed = target.parent / f"{target.name}.build_failed"
    if failed.is_file():
        return False
    try:
        proc = subprocess.run([sys.executable, "scripts/build_tutorial_index.py", "--openfoam", profile["project_dir"]],
                              cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=1800)
        ok = proc.returncode == 0
        detail = (proc.stderr or proc.stdout)[-2000:]
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, str(exc)
    if not ok:
        try:
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(detail)
        except OSError:
            pass
        return False
    print(f"[foam] built the tutorial index for {profile['label']}", flush=True)
    return _index_matches_installation()


def _case_text(case: Path, budget: int) -> str:
    """A tutorial case's input files, each cut to its share of ``budget``."""
    files = [p for sub in ("system", "constant", "0", "0.orig") if (case / sub).is_dir()
             for p in sorted((case / sub).rglob("*"))
             if p.is_file() and "polyMesh" not in p.parts and p.stat().st_size < 200_000]
    if (case / "Allrun").is_file():
        files.append(case / "Allrun")
    from cfd_langgraph.openfoam_version import _strip_comments

    share = max(400, budget // max(1, len(files)))
    parts = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        text = _strip_comments(text).strip() if p.name != "Allrun" else text
        parts.append(f"<file path=\"{p.relative_to(case)}\">\n{text[:share]}\n</file>")
    return "\n".join(parts)[:budget]


def _installation_tutorials(user_requirement: str, case_solver: str, top: int = 2) -> Dict[str, Any]:
    """References from this installation's own tutorials: the cases run by
    ``case_solver`` whose names share most words with the requirement. Used
    when the prebuilt index was built from another OpenFOAM version, whose
    file names and solver settings would steer the writer wrong."""
    from cfd_langgraph.openfoam_version import find_tutorials, get_profile

    empty = {"ok": False, "tutorial_reference": "", "dir_structure": "", "allrun_reference": "", "fallback": True}
    profile = get_profile()
    cases = find_tutorials(profile, case_solver, limit=200)
    if not cases:
        return empty
    words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", user_requirement)}

    def score(case: Path) -> tuple:
        parts = {w.lower() for part in case.relative_to(profile["tutorials_dir"]).parts
                 for w in re.findall(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])", part)}
        return (-len(words & parts), sum(1 for _ in case.rglob("*")))

    chosen = sorted(cases, key=score)[:top]
    budget = 12000 // len(chosen)
    reference = "\n".join(
        f'<similar_case_{i} path="{c}">\n{_case_text(c, budget)}\n</similar_case_{i}>'
        for i, c in enumerate(chosen, 1))
    structure = "\n".join(
        f"<similar_case_{i}>\n" + "\n".join(str(p.relative_to(c)) for p in sorted(c.rglob("*"))
                                            if p.is_file() and "polyMesh" not in p.parts)
        + f"\n</similar_case_{i}>" for i, c in enumerate(chosen, 1))
    allrun = "\n".join((c / "Allrun").read_text(errors="ignore") for c in chosen if (c / "Allrun").is_file())
    return {"ok": True, "tutorial_reference": reference, "dir_structure": structure,
            "allrun_reference": allrun, "fallback": True, "source": "installation tutorials"}


def _shelf_tutorials(user_requirement: str, case_solver: str, top: int = 2) -> Dict[str, Any]:
    """References for a case that uses a third-party solver from the shelf
    (cfd_langgraph.solver_shelf): that solver's own tutorials, as ported to
    this installation, those sharing most words with the requirement first.
    Neither the prebuilt index nor the installation's tutorials hold them."""
    from cfd_langgraph import solver_shelf
    from cfd_langgraph.openfoam_version import get_profile

    empty = {"ok": False}
    profile = get_profile()
    if not profile.get("ok"):
        return empty
    chosen_solvers = solver_shelf.solvers_for(user_requirement, case_solver, profile)
    if not chosen_solvers:
        return empty
    words = solver_shelf._words(user_requirement)
    cases = []
    for solver in chosen_solvers:
        for case in solver_shelf.tutorial_cases(solver):
            name_words = {w for part in case.parts[-3:] for w in solver_shelf._words(
                "".join(" " + ch if ch.isupper() else ch for ch in part))}
            cases.append((-len(words & name_words), sum(1 for _ in case.rglob("*")), case))
    chosen = [c for _, _, c in sorted(cases)[:top]]
    if not chosen:
        return empty
    budget = 12000 // len(chosen)
    reference = "\n".join(
        f'<similar_case_{i} path="{c}">\n{_case_text(c, budget)}\n</similar_case_{i}>'
        for i, c in enumerate(chosen, 1))
    structure = "\n".join(
        f"<similar_case_{i}>\n" + "\n".join(str(p.relative_to(c)) for p in sorted(c.rglob("*"))
                                            if p.is_file() and "polyMesh" not in p.parts)
        + f"\n</similar_case_{i}>" for i, c in enumerate(chosen, 1))
    allrun = "\n".join((c / "Allrun").read_text(errors="ignore") for c in chosen if (c / "Allrun").is_file())
    return {"ok": True, "tutorial_reference": reference, "dir_structure": structure,
            "allrun_reference": allrun, "fallback": False,
            "source": "shelf tutorials: " + ", ".join(s["name"] for s in chosen_solvers)}


def _fallback_tutorial_listing(case_solver: str) -> Dict[str, Any]:
    """FoamAgent's own documented degraded path when the FAISS indices
    (see cfd_langgraph/foam_native/faiss_index.py) aren't present: read one real
    tutorial from $WM_PROJECT_DIR/tutorials/<solver>/ directly. Lower
    quality, but the writer stages still work — see
    cfd-skills/cfd-foamagent/SKILL.md Stage 2."""
    empty = {"ok": False, "tutorial_reference": "", "dir_structure": "", "allrun_reference": "", "fallback": True}
    wm_project_dir = os.environ.get("WM_PROJECT_DIR", "")
    if not wm_project_dir:
        return empty
    solver_dir = Path(wm_project_dir) / "tutorials" / case_solver
    if not solver_dir.is_dir():
        return empty
    case_dirs = sorted(p for p in solver_dir.rglob("*") if p.is_dir() and (p / "system").is_dir())[:1]
    if not case_dirs:
        return empty
    example = case_dirs[0]
    structure = "\n".join(str(p.relative_to(example)) for p in sorted(example.rglob("*")) if p.is_file())
    allrun_path = example / "Allrun"
    allrun_text = allrun_path.read_text(encoding="utf-8", errors="ignore") if allrun_path.is_file() else ""
    return {
        "ok": True,
        "tutorial_reference": f'<similar_case_1 path="{example}">\n{structure}\n</similar_case_1>',
        "dir_structure": structure,
        "allrun_reference": allrun_text,
        "fallback": True,
    }


def shelf_usage_notes(user_requirement: str, case_solver: str) -> str:
    """The usage notes of the shelf solvers a case runs on (solvers/<name>/
    solver.yaml), for the case's writer and reviewer. They held what had
    been learnt the hard way -- DAFoam's useRotations, its parallel crash on
    a cached gradient, that it decomposes the case itself -- and reached only
    the manager, never the agents that write the case."""
    from cfd_langgraph import solver_shelf
    from cfd_langgraph.openfoam_version import get_profile

    profile = get_profile()
    if not profile.get("ok"):
        return ""
    notes = []
    for s in solver_shelf.solvers_for(user_requirement, case_solver, profile):
        if s.get("usage"):
            notes.append(f"{s['name']}: {' '.join(str(s.get('usage') or '').split())}")
        api = solver_shelf.api_index(s)
        if api:
            notes.append(f"{s['name']} Python API, the public names its classes have in this build "
                         f"(call no other):\n{api}")
    return "\n".join(notes)


def shelf_example_scripts(user_requirement: str, case_solver: str, top: int = 4, budget: int = 32000) -> str:
    """A shelf solver's own example scripts (its regression tests and
    tutorials) that share the most distinctive words with the requirement,
    for writing a case its solver runs by a script.

    The retrieval above returns tutorial CASE folders. A DAFoam case is
    defined by its runScript.py, and the scripts showing DAFoam's API for a
    field input, a variance objective, gradients or an optimiser sit beside
    those cases as runRegTests_*.py; nothing showed them to the writer, which
    rewrote the run script from memory and failed on API details every time
    (shelf_demo_dafoam_20261015_1318, case_003: a field input without its
    'distributed' key, an attribute the builder does not have)."""
    import math
    from collections import Counter

    from cfd_langgraph import solver_shelf
    from cfd_langgraph.openfoam_version import get_profile

    profile = get_profile()
    if not profile.get("ok"):
        return ""
    texts: Dict[Path, str] = {}
    # Scripts the solver's own usage notes name are shown whatever the word
    # overlap: ranked by words alone, the only field-input example DAFoam's
    # notes point to came 7th of the top 4 for a field-inversion requirement.
    named: set = set()
    for solver in solver_shelf.solvers_for(user_requirement, case_solver, profile):
        for word in str(solver.get("usage") or "").split():
            word = word.strip(".,;:()[]`'\"")
            if word.endswith(".py") and "/" in word:  # a path, not "a runScript.py"
                named.add(word)
        root = solver_shelf.tutorials_dir(solver)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py"))[:500]:
            if path.is_file() and "processor0" not in path.parts:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                # A script that never names the solver is a helper (a test's
                # value checker), not an example of the solver's API.
                if str(solver.get("name") or "").lower() in text.lower():
                    texts[path] = text
    if not texts:
        return ""
    def counts(text: str) -> Counter:
        cleaned = "".join(ch.lower() if ch.isalpha() else " " for ch in text)
        return Counter(w for w in cleaned.split() if len(w) >= 3)

    tf = {p: counts(t) for p, t in texts.items()}
    seen_in = Counter(w for c in tf.values() for w in c)
    wanted = set(counts(user_requirement))
    n = len(texts)
    avg_len = sum(sum(c.values()) for c in tf.values()) / n

    def score(p: Path) -> float:
        # BM25: a word most scripts share says little, and a long script
        # must not win by holding many words.
        length = sum(tf[p].values())
        total = 0.0
        for w in wanted & set(tf[p]):
            idf = math.log(1 + (n - seen_in[w] + 0.5) / (seen_in[w] + 0.5))
            f = tf[p][w]
            total += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / avg_len))
        return total

    pinned = [p for p in sorted(texts) if any(str(p).endswith("/" + name) for name in named)]
    ranked = [p for p in sorted(texts, key=lambda p: (-score(p), len(texts[p]))) if p not in pinned]
    chosen = (pinned + ranked)[:max(top, len(pinned))]
    per = budget // len(chosen)
    return "\n".join(f'<example_script path="{p}">\n{texts[p][:per]}\n</example_script>' for p in chosen)


def retrieve_references(
    user_requirement: str, case_solver: str, case_domain: str = "", case_category: str = "", top: int = 3
) -> Dict[str, Any]:
    """FoamAgent Stage 2 (RAG retrieval). Queries the three FAISS indices via
    scripts/rag_query.py; on any failure (indices not built, embedding model
    missing — rc=2/3, matching the documented failure modes), falls back to
    reading a real tutorial from $WM_PROJECT_DIR/tutorials/ directly rather
    than blocking the whole case on the prebuilt indices being present.

    When the indices were built from another OpenFOAM version than the one
    installed, indices are built from the installation's own tutorials first
    (scripts/build_tutorial_index.py, once per version, about 15 s); if that
    cannot be done, its tutorials are read directly."""
    shelf = _shelf_tutorials(user_requirement, case_solver, top=min(top, 2))
    if shelf.get("ok"):
        return shelf
    if not _index_matches_installation() and not _build_installed_index():
        found = _installation_tutorials(user_requirement, case_solver, top=min(top, 2))
        if found.get("ok"):
            return found
    tutorials, structure, allrun = _run_rag_queries(
        [
            {"db": "openfoam_tutorials_details", "query": f"{case_solver} {user_requirement}", "top": top},
            {"db": "openfoam_tutorials_structure", "query": f"{case_solver} {case_domain} {case_category}", "top": top},
            {"db": "openfoam_allrun_scripts", "query": f"{case_solver} {user_requirement}", "top": top},
        ]
    )

    if not (tutorials.get("ok") and structure.get("ok") and allrun.get("ok")):
        found = _installation_tutorials(user_requirement, case_solver, top=min(top, 2))
        return found if found.get("ok") else _fallback_tutorial_listing(case_solver)

    return {
        "ok": True,
        "tutorial_reference": _results_to_text(tutorials.get("results", [])),
        "dir_structure": _results_to_text(structure.get("results", [])),
        "allrun_reference": _results_to_text(allrun.get("results", [])),
        "fallback": False,
    }

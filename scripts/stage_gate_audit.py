#!/usr/bin/env python3
"""Stage-gate audit — the executable enforcement layer for skill-mode runs.

Slices 7/8/9/10/11/12/13 added required checkpoints, ordering constraints, per-case
file-system gates, and paper-stage hard gates to the skill markdown. Agents
were able to bypass them because markdown is advisory. This script is the
gate: the orchestrator skill MUST invoke it before writing finish_done.json
and state.status="done". Non-zero exit means the run is not complete.

Slice 14 adds `--mode preflight --target-stage <name>` so each per-stage
skill can hard-check that all its REQUIRED predecessors are on disk
*before* it starts work — preventing the agent from skipping upstream
stages and reaching analysis or paper with a half-finished pipeline.

Slice 20 hardens the gates: a failed run_result.json now fails the audit
unless its rerun budget is documented as exhausted; FOAM FATAL ERROR in any
case or mesh-gate solver log hard-fails (H-FATAL); a success case must have a
solver log reaching the `End` line; OED iteration bookkeeping must be complete
(H33); a starter_mesh_locked self-exemption needs a live-verified DOI citation
(H31); near-duplicate / filler paragraphs (H21) and inconsistent reference
values (H34) are enforced; and state.json is reconciled against the
checkpoints on disk every run (the audit, not the skill chain, is the
authority on progress).

Usage:
    python scripts/stage_gate_audit.py --out-dir <run_dir> [--json]
    python scripts/stage_gate_audit.py --out-dir <run_dir> --mode preflight --target-stage <stage>

Exit codes:
    0  — all gates pass; safe to mark state.status="done"
    2  — missing required checkpoint(s) (or missing predecessor in preflight mode)
    3  — ordering violation (e.g., mesh_gate after experiments)
    4  — per-case file-system gate failed (figs empty / decision.json bad)
    5  — paper-stage gate failed (no paper / no cross-experiment figures /
         length/density/cross-figure flags false)
    6  — usage error (bad --out-dir, missing state.json, unknown mode,
         unknown --target-stage)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Mode → route number mapping. Mirrors skills/cfd-orchestrator/SKILL.md routing.
MODE_TO_ROUTE = {
    "research":       1,
    "code_mod":       2,
    "mesh_gate":      3,
    "analysis_only":  4,
    "paper_only":     5,
    "open_discovery": 6,
}

# Required checkpoints per route. Mirrors slice 10's updated table in
# skills/cfd-orchestrator/SKILL.md "Required checkpoints per route".
REQUIRED = {
    1: [
        "routing_done", "literature_done", "baseline_setup_done", "metric_setup_done",
        "hypothesis_done", "requirements_done", "mesh_gate_done", "experiments_done",
        "analysis_done", "paper_experiment_plan_done", "cross_experiment_analysis_done",
        "paper_review_done", "finish_done",
    ],
    2: [
        "routing_done", "literature_done", "baseline_setup_done", "metric_setup_done",
        "hypothesis_done", "requirements_done", "code_mod_done", "mesh_gate_done",
        "experiments_done", "analysis_done", "paper_experiment_plan_done",
        "cross_experiment_analysis_done", "paper_review_done", "finish_done",
    ],
    3: ["routing_done", "mesh_gate_done", "finish_done"],
    4: ["routing_done", "analysis_done", "finish_done"],
    5: [
        "routing_done", "paper_experiment_plan_done", "cross_experiment_analysis_done",
        "paper_review_done", "finish_done",
    ],
    6: [
        "routing_done", "literature_done", "baseline_setup_done", "metric_setup_done",
        "mesh_gate_done", "open_ended_discovery_done", "analysis_done",
        "paper_experiment_plan_done", "cross_experiment_analysis_done",
        "paper_review_done", "finish_done",
    ],
}

ROUTES_WITH_EXPERIMENTS = {1, 2, 6}
ROUTES_WITH_PAPER       = {1, 2, 5, 6}

# Stage names accepted by --target-stage. Map stage name -> "<stage>_done" key
# used inside REQUIRED. The orchestrator's sub-stage mapping table is the source
# of truth (skills/cfd-orchestrator/SKILL.md "Sub-stage skill mapping").
PREFLIGHT_STAGE_NAMES = {
    "literature":                  "literature_done",
    "baseline_setup":              "baseline_setup_done",
    "metric_setup":                "metric_setup_done",
    "hypothesis":                  "hypothesis_done",
    "requirements":                "requirements_done",
    "code_mod":                    "code_mod_done",
    "mesh_gate":                   "mesh_gate_done",
    "experiments":                 "experiments_done",
    "analysis":                    "analysis_done",
    "paper_experiment_plan":       "paper_experiment_plan_done",
    "cross_experiment_analysis":   "cross_experiment_analysis_done",
    "paper_review":                "paper_review_done",
    "open_ended_discovery":        "open_ended_discovery_done",
}

# ±TOLERANCE band applied to numeric floors/ceilings in the paper-stage audit.
# Values inside the band PASS with a WARN; values past it hard-FAIL.
# Structural gates (sections present, mesh-gate present, integer counts like
# citations≥10 / equations≥3) are NOT softened — they remain strict.
TOLERANCE = 0.05  # 5%


def _floor_status(val: float, floor: float, tol: float = TOLERANCE) -> str:
    """Return 'pass' (val >= floor), 'warn' (floor*(1-tol) <= val < floor),
    or 'fail' (val < floor*(1-tol)). For lower-bound checks."""
    if val >= floor:
        return "pass"
    if val >= floor * (1.0 - tol):
        return "warn"
    return "fail"


def _ceiling_status(val: float, ceiling: float, tol: float = TOLERANCE) -> str:
    """Mirror of _floor_status for upper bounds (val <= ceiling is pass)."""
    if val <= ceiling:
        return "pass"
    if val <= ceiling * (1.0 + tol):
        return "warn"
    return "fail"


def _range_status(val: float, lo: float, hi: float, tol: float = TOLERANCE) -> str:
    """For a range [lo, hi]: pass if inside, warn if within tol on either side,
    fail otherwise."""
    if lo <= val <= hi:
        return "pass"
    if lo * (1.0 - tol) <= val <= hi * (1.0 + tol):
        return "warn"
    return "fail"


def _worst_status(*statuses: str) -> str:
    """Combine multiple statuses: fail > warn > pass."""
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


NUMERICS_ONLY_PATTERNS = (
    "no separate.*interpretation", "no.*VLM", "no.*vision", "numerical checks only",
    "no separate.*VLM", "skill-mode validation used numerical",
)
VISION_KEYWORDS = (
    "pyvista", "contour", "field", "streamline", "recirculation", "mesh",
    "y\\+", "y-plus", "wall", "patch", "geometry", "render", "visualization",
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_doi(doi: str, timeout: float = 8.0) -> Tuple[Optional[bool], str]:
    """Item 9 — live DOI resolution. Returns:
      (True,  note) — DOI resolves (registered; doi.org redirects to a landing page)
      (False, note) — DOI does not resolve (404 / not registered / malformed)
      (None,  note) — auditor itself is offline / network unreachable (inconclusive)
    A run is never penalised for the auditor's own connectivity, so callers treat
    a None result as a warning, not a failure.
    """
    raw = (doi or "").strip()
    m = re.search(r"10\.\d{4,9}/[^\s\"'}<>]+", raw)
    if not m:
        return False, f"not a syntactically valid DOI: {raw!r}"
    clean = m.group(0).rstrip(").,;]")
    url = "https://doi.org/" + clean
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "cfd-stage-gate-audit/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
        if code and 200 <= code < 400:
            return True, f"resolved (HTTP {code})"
        if code == 404:
            return False, "HTTP 404 — DOI not registered with doi.org"
        return None, f"HTTP {code} — inconclusive"
    except Exception as e:
        return None, f"network unavailable ({type(e).__name__}: {e})"


def _check_starter_mesh_lock_citation(cd: Dict[str, Any]) -> List[str]:
    """Item 9 — a `starter_mesh_locked` self-exemption from the mesh-gate must be
    backed by a verifiable published citation, not a free-text rationale. The
    checkpoint must carry `mesh_source` and a `citation` with a DOI; the DOI is
    resolved live. Auditor-offline degrades to a stderr warning (not a failure).
    """
    failures: List[str] = []
    if not cd.get("mesh_source"):
        failures.append(
            "H31 starter_mesh_locked: 'mesh_source' field missing — name the "
            "starter blockMeshDict / polyMesh path the lock applies to."
        )
    citation = cd.get("citation")
    doi: Optional[str] = None
    if isinstance(citation, dict):
        doi = citation.get("doi") or citation.get("DOI")
    elif isinstance(citation, str):
        doi = citation
    if not doi:
        failures.append(
            "H31 starter_mesh_locked: no verifiable citation. A mesh-gate "
            "self-exemption requires a 'citation' field carrying the published "
            "DOI of the mesh-validation source the starter reproduces. A "
            "free-text 'rationale' is no longer sufficient."
        )
        return failures
    resolved, note = _resolve_doi(str(doi))
    if resolved is False:
        failures.append(
            f"H31 starter_mesh_locked: citation DOI did not resolve — {note}. "
            f"Provide the DOI of a published mesh-validation source."
        )
    elif resolved is None:
        print(
            f"WARN: H31 starter_mesh_locked citation DOI could not be verified "
            f"({note}); auditor appears offline — not penalising the run. "
            f"Re-run the audit with network access to confirm the citation.",
            file=sys.stderr,
        )
    return failures


def _check_literature_provenance(out_dir: Path) -> List[str]:
    """Slice 14: enforce that lit.json carries a Semantic Scholar record OR is an
    explicit empty array with a documented timeline event. Catches the failure mode
    where the agent writes lit.json with hand-typed records and skips S2.
    Returns a list of failure messages (empty = OK).
    """
    failures: List[str] = []
    lit_path = out_dir / "lit.json"
    if not lit_path.is_file():
        return failures  # absence handled by checkpoint check
    try:
        data = json.loads(lit_path.read_text())
    except Exception as e:
        return [f"lit.json present but unreadable: {e}"]
    if isinstance(data, dict):
        recs = data.get("records") or data.get("papers") or data.get("results") or []
    else:
        recs = data
    if not isinstance(recs, list):
        return ["lit.json has unexpected shape (not list or {records: [...]})"]
    if len(recs) == 0:
        # Empty is allowed but must be paired with a timeline event.
        tl = out_dir / "timeline.json"
        if tl.is_file():
            try:
                events = json.loads(tl.read_text())
                if isinstance(events, list) and any(
                    isinstance(e, dict) and e.get("event") in
                        ("literature_empty_result", "literature_skipped_existing")
                    for e in events
                ):
                    return failures
            except Exception:
                pass
        failures.append(
            "lit.json is [] but no 'literature_empty_result' timeline event was emitted "
            "— the agent must surface the empty-result explicitly; silent advance is forbidden."
        )
        return failures
    # Non-empty: at least one record must declare source == semanticscholar
    sources = [r.get("source") for r in recs if isinstance(r, dict)]
    has_s2 = any(s == "semanticscholar" for s in sources)
    has_any_provenance = all(isinstance(s, str) and s for s in sources)
    if not has_s2:
        failures.append(
            f"lit.json has {len(recs)} records but none have source=='semanticscholar'. "
            f"Semantic Scholar is mandatory (cfd-literature Hard rule #1). "
            f"Sources seen: {sorted(set(sources))!r}"
        )
    if not has_any_provenance:
        failures.append(
            "lit.json contains records without a 'source' field — provenance is required. "
            "Records lacking provenance are treated as fabricated."
        )
    return failures


def _check_preflight(out_dir: Path, route: int, target_stage: str) -> List[str]:
    """Predecessors-on-disk gate for a single stage.

    Returns a list of missing-predecessor messages. Empty list = OK to start
    the target stage. Slice-14 enforcement: each per-stage SKILL.md invokes
    this as its Step 0 so the agent cannot skip earlier stages and silently
    write a downstream checkpoint.
    """
    target_cp = PREFLIGHT_STAGE_NAMES.get(target_stage)
    if target_cp is None:
        return [f"preflight: unknown target stage '{target_stage}'. "
                f"Valid: {sorted(PREFLIGHT_STAGE_NAMES.keys())}"]
    seq = REQUIRED.get(route, [])
    if target_cp not in seq:
        # Route doesn't require this stage at all → nothing to enforce.
        return []
    idx = seq.index(target_cp)
    needed = [cp for cp in seq[:idx] if cp != "finish_done"]
    failures: List[str] = []
    cp_dir = out_dir / "checkpoints"
    for cp in needed:
        if not (cp_dir / f"{cp}.json").is_file():
            failures.append(
                f"preflight: '{target_stage}' cannot start — predecessor "
                f"checkpoint missing: {cp} "
                f"(run the corresponding skill first; see "
                f"skills/cfd-orchestrator/SKILL.md 'Required checkpoints per route')"
            )
    return failures


def _check_checkpoints(out_dir: Path, route: int) -> List[str]:
    failures: List[str] = []
    cp_dir = out_dir / "checkpoints"
    for stage in REQUIRED[route]:
        if stage == "finish_done":  # checked last by the orchestrator itself
            continue
        if not (cp_dir / f"{stage}.json").is_file():
            failures.append(f"missing required checkpoint: {stage}")
    return failures


def _check_mesh_gate_integrity(out_dir: Path, route: int) -> List[str]:
    """H31 (slice 19.1): mesh_gate_done.json must be backed by real artifacts
    (selected_mesh_spec.json OR mesh_independence_context.json on disk), OR
    explicitly declare itself as a 'starter mesh locked' stub. Catches the
    fake checkpoint pattern where the agent writes mesh_gate_done.json with
    only a comment and no artifacts. For routes that don't need mesh-gate
    (3 standalone is the only route that *requires* it), still enforce that
    if the checkpoint exists, it must be honest.
    """
    failures: List[str] = []
    cp = out_dir / "checkpoints" / "mesh_gate_done.json"
    if not cp.is_file():
        return failures  # missing reported elsewhere
    try:
        cd = json.loads(cp.read_text())
    except Exception:
        return ["H31 mesh_gate_done.json unreadable as JSON"]
    spec_exists = (out_dir / "selected_mesh_spec.json").is_file()
    ctx_exists  = (out_dir / "mesh_independence_context.json").is_file()
    mg_dir      = (out_dir / "mesh_gate").is_dir()
    # Explicit starter-locked declaration — must use the canonical
    # 'starter_mesh_locked' status to skip the artifact requirement.
    is_starter_locked_declared = (
        cd.get("status") == "starter_mesh_locked"
        or cd.get("starter_mesh_locked") is True
    )
    if is_starter_locked_declared:
        # Item 9 — a starter-mesh self-exemption is allowed ONLY with a
        # verifiable published citation, whether or not artifacts exist.
        failures.extend(_check_starter_mesh_lock_citation(cd))
    elif not (spec_exists or ctx_exists or mg_dir):
        failures.append(
            "H31 mesh_gate_done.json is present but no mesh-gate artifacts on disk "
            "(no selected_mesh_spec.json, no mesh_independence_context.json, no "
            "mesh_gate/ directory). The checkpoint was fabricated. If the starter's "
            "shipped mesh was intentionally locked (valid for OED runs where the "
            "mesh is a study control), re-emit the checkpoint with "
            "status='starter_mesh_locked', a 'mesh_source' field, and a 'citation' "
            "field carrying the published DOI of the mesh-validation source."
        )
    return failures


def _check_open_discovery_bridge(out_dir: Path, route: int) -> List[str]:
    """H27 (slice 18.3): for route 6 (open_discovery), enforce that the
    cfd-open-discovery Step 14 bridge actually ran. The bridge stamps
    `open_ended_discovery_done.json` with `bridge_signature` and produces
    `bridge.json` + `cases/case_*/` + `manifest.json`. Agents that fabricate
    the checkpoint without running the bridge body fail this gate.
    """
    if route != 6:
        return []
    failures: List[str] = []
    # Accept either checkpoint name (the skill writes the long form; some
    # script fast-paths write the short form). Both must carry the bridge
    # signature, both must have artifacts behind them.
    cp_long  = out_dir / "checkpoints" / "open_ended_discovery_done.json"
    cp_short = out_dir / "checkpoints" / "open_discovery_done.json"
    cp = cp_long if cp_long.is_file() else (cp_short if cp_short.is_file() else None)
    if cp is None:
        return failures  # _check_checkpoints will report this separately
    if cp_short.is_file() and not cp_long.is_file():
        failures.append(
            "H27 checkpoint name drift: checkpoints/open_discovery_done.json exists "
            "but the canonical name is open_ended_discovery_done.json. Rename it (the "
            "audit + bridge both expect the long form) and re-emit with the bridge "
            "signature."
        )
    try:
        ckdata = json.loads(cp.read_text())
    except Exception:
        return [f"{cp.name} unreadable as JSON"]
    sig = ckdata.get("bridge_signature")
    if sig != "cfd-open-discovery:bridge:v1":
        failures.append(
            f"H27 open_ended_discovery_done.json missing bridge_signature="
            f"'cfd-open-discovery:bridge:v1' (got {sig!r}). The bridge body in "
            f"cfd-open-discovery Step 14 was not executed — the agent skipped "
            f"the bridge and fabricated the checkpoint. Re-invoke "
            f"Skill cfd-skills/cfd-open-discovery (it resumes from the existing "
            f"history.json) so Step 14 actually runs."
        )
    # bridge.json must also exist with matching provenance.
    bj = out_dir / "bridge.json"
    if not bj.is_file():
        failures.append(
            "H27 bridge.json missing — the bridge step did not write the "
            "per-case decision summary. Re-run the bridge."
        )
    else:
        try:
            bdata = json.loads(bj.read_text())
            if bdata.get("provenance") != "cfd-open-discovery:bridge:v1":
                failures.append(
                    f"H27 bridge.json#provenance != 'cfd-open-discovery:bridge:v1' "
                    f"(got {bdata.get('provenance')!r}); bridge artifact is fabricated."
                )
        except Exception:
            failures.append("H27 bridge.json unreadable as JSON")
    # manifest.json and cases/ must exist with ≥ 2 cases (baseline + ≥1 candidate)
    mf = out_dir / "manifest.json"
    cases_dir = out_dir / "cases"
    if not mf.is_file():
        failures.append("H27 manifest.json missing — bridge did not build the cases/ layout")
    elif not cases_dir.is_dir():
        failures.append("H27 cases/ directory missing — bridge did not create per-case dirs")
    else:
        try:
            mdata = json.loads(mf.read_text())
            n_cases = len(mdata.get("cases", []))
            if n_cases < 2:
                failures.append(
                    f"H27 manifest.json#cases has only {n_cases} entries "
                    f"(need baseline + ≥1 OED candidate, even if REVISE'd). "
                    f"Bridge under-built — re-run Step 14."
                )
        except Exception:
            failures.append("H27 manifest.json unreadable as JSON")
    return failures


def _check_ordering(out_dir: Path, route: int) -> List[str]:
    if route not in ROUTES_WITH_EXPERIMENTS:
        return []
    cp_dir = out_dir / "checkpoints"
    mg = cp_dir / "mesh_gate_done.json"
    ex = cp_dir / "experiments_done.json"
    if not (mg.is_file() and ex.is_file()):
        return []  # missing checkpoint already reported by _check_checkpoints
    if mg.stat().st_mtime >= ex.stat().st_mtime:
        return [
            f"ordering violation: mesh_gate_done.json mtime ({mg.stat().st_mtime:.0f}) "
            f">= experiments_done.json mtime ({ex.stat().st_mtime:.0f}); "
            f"mesh-gate must run BEFORE experiments"
        ]
    return []


def _check_cases(out_dir: Path, route: int) -> List[str]:
    if route not in ROUTES_WITH_EXPERIMENTS:
        return []
    failures: List[str] = []
    cases_dir = out_dir / "cases"
    if not cases_dir.is_dir():
        return [f"cases/ directory missing under {out_dir}"]
    case_dirs = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        return [f"cases/ exists but is empty under {out_dir}"]

    for c in case_dirs:
        rr = c / "run_result.json"
        dj = c / "decision.json"
        figs_dir = c / "figs"

        if not rr.is_file():
            failures.append(f"{c.name}: run_result.json missing")
            continue
        rr_data = _read_json(rr)
        status = rr_data.get("status")
        if status != "success":
            # Slice 20 / item 1 — a non-success case is admissible only when
            # the failure is HONESTLY DOCUMENTED. The bar differs by route.
            dj_data = _read_json(dj) if dj.is_file() else {}
            if route == 6:
                # OED candidate: a failed / partial candidate is legitimate
                # data for the paper, but must carry an honest classification.
                honest = str(status or "").lower() in (
                    "failed", "partial", "scaffolded", "code_only",
                    "code-only", "timeout",
                )
                if not honest:
                    failures.append(
                        f"{c.name}: OED candidate run_result.json status="
                        f"{status!r} is not an honest classification (expected "
                        f"success / failed / partial / scaffolded / code_only / "
                        f"timeout)"
                    )
                continue
            # routes 1/2 — production experiment case (Decision 1): a failed
            # case passes only when the rerun budget was provably exhausted.
            if not rr_data.get("error_logs"):
                failures.append(
                    f"{c.name}: status={status!r} but no error_logs documented"
                )
                continue
            loop_count = rr_data.get("loop_count")
            max_loop = (rr_data.get("max_loop") or rr_data.get("loop_cap")
                        or rr_data.get("max_loops") or 10)
            dj_reason = str(dj_data.get("reason", "")).lower()
            retries_exhausted = (
                (isinstance(loop_count, (int, float)) and loop_count >= max_loop)
                or bool(rr_data.get("retries_exhausted"))
                or bool(dj_data.get("retries_exhausted"))
                or "max retries" in dj_reason
                or "max revisions" in dj_reason
                or "max reruns" in dj_reason
                or "retry budget" in dj_reason
            )
            if not retries_exhausted:
                failures.append(
                    f"{c.name}: status={status!r} but the rerun budget is not "
                    f"documented as exhausted (loop_count={loop_count}, "
                    f"max_loop={max_loop}; decision.json reason carries no "
                    f"max-retries marker). A failed production case passes the "
                    f"audit only when its retry budget was provably spent — "
                    f"otherwise rerun it (cfd-experiment CFL-aware retry)."
                )
            continue

        if not figs_dir.is_dir():
            failures.append(f"{c.name}: figs/ directory missing (cfd-viz was skipped)")
            continue
        png_count = len(list(figs_dir.glob("*.png")))
        if png_count < 1:
            failures.append(f"{c.name}: figs/ empty (cfd-viz was skipped)")
            continue

        if not dj.is_file():
            failures.append(f"{c.name}: decision.json missing")
            continue
        dj_data = _read_json(dj)
        if dj_data.get("status") == "PROCEED" and png_count < 1:
            failures.append(f"{c.name}: PROCEED without figs/")
        # numerics-only admission check
        haystack = " ".join(
            str(dj_data.get(k, "")) for k in ("note", "reason", "summary")
        ).lower()
        for pat in NUMERICS_ONLY_PATTERNS:
            if re.search(pat, haystack, re.IGNORECASE):
                failures.append(
                    f"{c.name}: decision.json admits numerics-only verdict "
                    f"(matched '{pat}'); cfd-interpret vision pass was skipped"
                )
                break
    return failures


FATAL_LOG_MARKERS = ("FOAM FATAL ERROR", "FOAM FATAL IO ERROR")

_MESH_LOG_TOKENS = (
    "blockmesh", "checkmesh", "snappy", "surfacefeature", "decompose",
    "renumber", "extrude", "postprocess", "toposet", "creatpatch",
)


def _scan_log_for_fatal(log_path: Path) -> bool:
    try:
        txt = log_path.read_text(errors="replace")
    except Exception:
        return False
    return any(mark in txt for mark in FATAL_LOG_MARKERS)


def _log_reached_end(log_path: Path) -> bool:
    """True if an OpenFOAM solver log ends cleanly (the 'End' line is present
    near the tail of the file)."""
    try:
        txt = log_path.read_text(errors="replace")
    except Exception:
        return False
    return bool(re.search(r"(?m)^\s*End\s*$", txt[-6000:]))


def _solver_logs_in(case_dir: Path) -> List[Path]:
    """All OpenFOAM log.* files for a case — handles both the <case>/log.*
    layout and the OED-bridge layout <case>/case/log.*."""
    logs: List[Path] = []
    for base in (case_dir, case_dir / "case"):
        try:
            if base.is_dir():
                logs += sorted(base.glob("log.*"))
        except OSError:
            continue
    return logs


def _check_solver_logs(out_dir: Path, route: int) -> List[str]:
    """Item 1 — FATAL-error and clean-completion gate over OpenFOAM solver logs.

      * Any FOAM FATAL ERROR in a mesh-gate log hard-fails — a mesh selected
        off a crashed run is not validated.
      * A production (route 1/2) case whose run_result.json claims success but
        has a FATAL log, no solver log, or no solver log reaching the 'End'
        line, hard-fails — the success claim is false.
    """
    failures: List[str] = []
    mg = out_dir / "mesh_gate"
    if mg.is_dir():
        for log in sorted(mg.rglob("log.*")):
            if _scan_log_for_fatal(log):
                failures.append(
                    f"mesh_gate: FOAM FATAL ERROR in "
                    f"{log.relative_to(out_dir)} — the mesh-gate run did not "
                    f"complete cleanly; the selected mesh is not validated."
                )
    if route not in ROUTES_WITH_EXPERIMENTS:
        return failures
    cases_dir = out_dir / "cases"
    if not cases_dir.is_dir():
        return failures
    for c in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
        rr = _read_json(c / "run_result.json")
        if rr.get("status") != "success":
            continue  # honest non-success handled by _check_cases
        logs = _solver_logs_in(c)
        fatal = [l for l in logs if _scan_log_for_fatal(l)]
        if fatal:
            failures.append(
                f"{c.name}: run_result.json says success but {fatal[0].name} "
                f"contains FOAM FATAL ERROR — the success status is false."
            )
            continue
        solver_logs = [
            l for l in logs
            if not any(tok in l.name.lower() for tok in _MESH_LOG_TOKENS)
        ]
        if not solver_logs:
            failures.append(
                f"{c.name}: run_result.json says success but no solver log "
                f"(log.<solver>) is present — the run cannot be confirmed."
            )
            continue
        if not any(_log_reached_end(l) for l in solver_logs):
            failures.append(
                f"{c.name}: run_result.json says success but no solver log "
                f"reaches the 'End' line — the solver did not finish cleanly."
            )
    return failures


def _check_oed_bookkeeping(out_dir: Path, route: int) -> List[str]:
    """Items 12 + 13 — OED iteration bookkeeping.

      * Every iter_* directory on disk is represented in history.json.
      * Each history record carries an integer iteration id; ids are unique.
      * Each record carries an honest status / decision classification.
      * No path named score.json is a directory (it must be a JSON file).
      * The declared budget bounds the iteration count, or the overrun is
        explicitly recorded.
    """
    if route != 6:
        return []
    failures: List[str] = []
    oed = out_dir / "open_ended_discovery"
    if not oed.is_dir():
        return failures  # absence handled by checkpoint / bridge gates
    iter_dirs = sorted(d for d in oed.glob("iter_*") if d.is_dir())
    hist_path = oed / "history.json"
    if not hist_path.is_file():
        if iter_dirs:
            failures.append(
                f"H33 OED ran {len(iter_dirs)} iter_* directories but "
                f"open_ended_discovery/history.json is missing — no iteration "
                f"is recorded."
            )
        return failures
    try:
        history = json.loads(hist_path.read_text())
    except Exception:
        return ["H33 open_ended_discovery/history.json is unreadable as JSON"]
    if not isinstance(history, list):
        return ["H33 open_ended_discovery/history.json is not a JSON array"]

    if len(history) < len(iter_dirs):
        failures.append(
            f"H33 OED bookkeeping incomplete: {len(iter_dirs)} iter_* "
            f"directories on disk but only {len(history)} history.json "
            f"records. Every iteration (including failed / scaffolded / "
            f"code-only) must be recorded — see cfd-open-discovery Step 11."
        )

    ids: List[int] = []
    for i, e in enumerate(history):
        if not isinstance(e, dict):
            failures.append(f"H33 history.json[{i}] is not an object")
            continue
        it = e.get("iter", e.get("iteration"))
        if not isinstance(it, int):
            failures.append(
                f"H33 history.json[{i}] has a non-integer iteration id "
                f"({it!r}) — iteration ids must be monotonic integers."
            )
        else:
            ids.append(it)
        cls = str(e.get("status") or e.get("decision") or e.get("result")
                  or e.get("action") or "").strip()
        if not cls:
            failures.append(
                f"H33 history.json record iter={it!r} carries no status / "
                f"decision classification — every iteration (success, partial, "
                f"failed, scaffolded, code-only) must say what it was."
            )
    if len(ids) != len(set(ids)):
        dups = sorted({x for x in ids if ids.count(x) > 1})
        failures.append(f"H33 history.json has duplicate iteration ids: {dups}")

    score_dirs = []
    try:
        score_dirs = [p for p in oed.rglob("score.json") if p.is_dir()]
    except OSError:
        pass
    if score_dirs:
        failures.append(
            f"H33 {len(score_dirs)} path(s) named score.json are directories, "
            f"not JSON files (e.g. {score_dirs[0].relative_to(out_dir)}). "
            f"Per-iteration scoring must write a score.json file."
        )

    budget: Optional[int] = None
    for src in (oed / "ext_state.json", out_dir / "oed_artifact.json"):
        d = _read_json(src)
        for k in ("budget_total", "budget", "open_ended_budget"):
            if isinstance(d.get(k), (int, float)):
                budget = int(d[k])
                break
        if budget is not None:
            break
    if budget is not None and ids:
        max_id = max(ids)
        artifact = _read_json(out_dir / "oed_artifact.json")
        ack = bool(artifact.get("budget_overrun") or artifact.get("stop_reason"))
        if max_id > budget and not ack:
            failures.append(
                f"H33 OED reached iteration {max_id} but the declared budget "
                f"is {budget} — record the overrun (oed_artifact.json "
                f"'budget_overrun' or 'stop_reason') or respect the budget."
            )
    return failures


def _reconcile_state_json(out_dir: Path, route: int,
                          state: Dict[str, Any]) -> Optional[str]:
    """Item 3 — make state.json reflect the checkpoints actually on disk. The
    audit is the authority on progress; state.json#current_stage is advisory
    and is frequently left stale by the skill chain. Returns a one-line note
    when the file had to be corrected, else None. Does NOT set status='done' —
    that happens only when the full audit passes (rc=0)."""
    seq = [s for s in REQUIRED.get(route, []) if s != "finish_done"]
    if not seq:
        return None
    cp_dir = out_dir / "checkpoints"
    furthest: Optional[str] = None
    for cp in seq:
        if (cp_dir / f"{cp}.json").is_file():
            furthest = cp
    if furthest is None:
        return None
    stage = furthest[:-5] if furthest.endswith("_done") else furthest
    old_stage = state.get("current_stage")
    old_status = state.get("status")
    if old_stage == stage and old_status in ("running", "done"):
        return None
    state["current_stage"] = stage
    if old_status != "done":
        state["status"] = "running"
    try:
        (out_dir / "state.json").write_text(json.dumps(state, indent=2))
    except Exception as e:
        return f"could not reconcile state.json: {e}"
    if old_stage != stage:
        return (f"current_stage was {old_stage!r}, reconciled to {stage!r} "
                f"(furthest checkpoint on disk)")
    return None


def _pdf_page_count(pdf: Path) -> int:
    """Use pdfinfo to count pages. Returns -1 if pdfinfo unavailable or fails."""
    import subprocess
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], stderr=subprocess.DEVNULL, timeout=10).decode()
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return -1


def _pdf_word_count(pdf: Path) -> int:
    """Use pdftotext to count words in the PDF body. Returns -1 if unavailable."""
    import subprocess
    try:
        text = subprocess.check_output(
            ["pdftotext", str(pdf), "-"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8", errors="replace")
        return len(re.findall(r"\b\w+\b", text))
    except Exception:
        return -1


def _tex_word_count_main_body(tex: Path) -> int:
    """Fallback when pdftotext unavailable: count words in main.tex excluding LaTeX commands and Appendix."""
    if not tex.is_file():
        return -1
    body = tex.read_text(errors="replace")
    # cut off everything from first \section{Appendix...} or \appendix
    body = re.split(r"\\appendix|\\section\{\s*Appendix", body, maxsplit=1)[0]
    body = re.sub(r"\\(?:label|cite|ref|includegraphics|begin|end)\{[^}]*\}", "", body)
    body = re.sub(r"\\[a-zA-Z]+\*?", "", body)
    body = re.sub(r"\$[^$]*\$", "", body)
    return len(re.findall(r"\b[A-Za-z][A-Za-z\-']{2,}\b", body))


PADDING_SECTION_PATTERNS = (
    r"^reproducibility", r"^re.run", r"^artifact", r"^inventory",
    r"notes$", r"^operational", r"^audit", r"^checklist", r"^traceability",
    r"^data products", r"^extended methodological",
    r"detail$", r"^additional", r"^practical recommendation",
    r"^final assessment",
)
CROSS_FIG_TYPE_KEYWORDS = {
    "overlay":     re.compile(r"overlay|comparison|comparator|cmp", re.IGNORECASE),
    "bar":         re.compile(r"\bbar\b|barchart|metric_bar", re.IGNORECASE),
    "trend":       re.compile(r"trend|sweep|parametric|vs_param|vs_temp|vs_re", re.IGNORECASE),
    "delta_field": re.compile(r"delta|diff|difference|residual", re.IGNORECASE),
    "error":       re.compile(r"error|rmse_dist|residual_dist|err_map", re.IGNORECASE),
    "contour":     re.compile(r"contour|panel|composite", re.IGNORECASE),
}


def _classify_section_name(name: str) -> str:
    n = name.strip().lower()
    for pat in PADDING_SECTION_PATTERNS:
        if re.search(pat, n):
            return "padding"
    return "main"


def _section_word_breakdown(tex: Path) -> Tuple[int, int, List[Tuple[str, str, int]]]:
    """Returns (main_body_words, padding_words, [(name, kind, word_count), ...]).
    Excludes Appendix sections from word counts."""
    if not tex.is_file():
        return (0, 0, [])
    text = tex.read_text(errors="replace")
    text = re.split(r"\\appendix|\\section\{\s*Appendix", text, maxsplit=1)[0]
    parts = re.split(r"\\section\{([^}]*)\}", text)
    breakdown: List[Tuple[str, str, int]] = []
    main_words = 0
    pad_words  = 0
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        body = re.sub(r"\\(?:label|cite|ref|includegraphics|begin|end)\{[^}]*\}", "", body)
        body = re.sub(r"\\[a-zA-Z]+\*?", "", body)
        body = re.sub(r"\$[^$]*\$", "", body)
        words = len(re.findall(r"\b[A-Za-z][A-Za-z\-']{2,}\b", body))
        kind = _classify_section_name(name)
        if kind == "padding":
            pad_words += words
        else:
            main_words += words
        breakdown.append((name, kind, words))
    return (main_words, pad_words, breakdown)


def _normalize_paragraph(p: str) -> str:
    """Whitespace + LaTeX-command normalization for duplicate-paragraph hashing."""
    p = re.sub(r"\\(?:label|cite[a-z]*|ref|includegraphics|begin|end)\{[^}]*\}", "", p)
    p = re.sub(r"\\[a-zA-Z]+\*?", "", p)
    p = re.sub(r"\s+", " ", p).strip().lower()
    return p


def _para_similarity(a: str, b: str) -> float:
    """difflib similarity ratio between two normalized paragraphs."""
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _find_ngram_loops(paragraph_norm: str, n: int = 5,
                      min_repeat: int = 3) -> Optional[str]:
    """Return a sample of an n-word phrase repeated >= min_repeat times inside a
    single paragraph — the filler / token-soup pattern. None if the paragraph
    is clean."""
    words = paragraph_norm.split()
    if len(words) < n * min_repeat:
        return None
    counts: Dict[str, int] = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
    return worst[0] if worst[1] >= min_repeat else None


def _find_duplicate_paragraphs(tex: Path) -> List[Tuple[str, int]]:
    """H21 (re-enabled, slice 20): detect (a) near-duplicate paragraphs and
    (b) intra-paragraph n-gram filler loops in the main.tex body.

    A 'paragraph' is a blank-line-separated body block (before \\appendix).
    Blocks shorter than 200 normalized chars are skipped — short transitions,
    table headers and captions legitimately recur. Two paragraphs whose
    difflib similarity ratio >= 0.85 count as near-duplicates, catching
    copy-paste with cosmetic edits (the BFS/Jet filler pattern), not only
    verbatim repeats. Returns [(sample, marker)] findings; empty == clean.
    """
    if not tex.is_file():
        return []
    text = tex.read_text(errors="replace")
    text = re.split(r"\\appendix|\\section\{\s*Appendix", text, maxsplit=1)[0]
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    # Strip environments that legitimately recur (equations, tables, figures).
    text = re.sub(
        r"\\begin\{(?:equation|align|figure|table|tabular|thebibliography|abstract)\*?\}.*?"
        r"\\end\{(?:equation|align|figure|table|tabular|thebibliography|abstract)\*?\}",
        "", text, flags=re.DOTALL,
    )
    paragraphs = [_normalize_paragraph(p) for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if len(p) >= 200]
    findings: List[Tuple[str, int]] = []
    # (a) near-duplicate paragraph pairs
    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):
            if _para_similarity(paragraphs[i], paragraphs[j]) >= 0.85:
                findings.append(
                    (f"near-duplicate paragraph: {paragraphs[i][:80]}", 2)
                )
                break
    # (b) intra-paragraph n-gram filler loops
    for p in paragraphs:
        loop = _find_ngram_loops(p)
        if loop:
            findings.append((f"repeated filler phrase: '{loop}'", 3))
    return findings


def _count_equation_blocks(tex: Path) -> int:
    """H20: count math display blocks: equation, align, gather, multline, and \\[...\\]."""
    if not tex.is_file():
        return 0
    text = tex.read_text(errors="replace")
    text = re.split(r"\\appendix|\\section\{\s*Appendix", text, maxsplit=1)[0]
    n = 0
    n += len(re.findall(r"\\begin\{equation\*?\}", text))
    n += len(re.findall(r"\\begin\{align\*?\}", text))
    n += len(re.findall(r"\\begin\{gather\*?\}", text))
    n += len(re.findall(r"\\begin\{multline\*?\}", text))
    # \[ ... \] display math (count \[ openings)
    n += len(re.findall(r"(?<!\\)\\\[", text))
    return n


def _count_refs_bib_entries(out_dir: Path) -> Tuple[bool, int, bool]:
    """H19: detect bibliography. Either:
      - external refs.bib with @-entries, OR
      - inline \\begin{thebibliography}...\\end{thebibliography} with \\bibitem entries.
    Returns (bibliography_present, n_entries, is_inline).
    """
    tex = out_dir / "paper" / "main.tex"
    # External refs.bib?
    for p in (out_dir / "paper" / "refs.bib", out_dir / "refs.bib"):
        if p.is_file():
            try:
                content = p.read_text(errors="replace")
                entries = len(re.findall(r"^\s*@\w+\s*\{", content, re.MULTILINE))
                if entries > 0:
                    return (True, entries, False)
            except Exception:
                pass
    # Inline thebibliography in main.tex?
    if tex.is_file():
        text = tex.read_text(errors="replace")
        if "\\begin{thebibliography}" in text:
            entries = len(re.findall(r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{", text))
            if entries > 0:
                return (True, entries, True)
    return (False, 0, False)


META_PARAGRAPH_PATTERNS = [
    r"\bin\s+(?:the\s+)?(?:introduction|related\s+work|methods?|results?|discussion|conclusion)\s+paragraph\s+\d+",
    r"\bparagraph\s+\d+\s+(?:of|in)\s+(?:the\s+)?(?:introduction|related\s+work|methods?|results?|discussion|conclusion)",
    r"\b(?:see|cf\.|refer\s+to)\s+paragraph\s+\d+",
    r"\b(?:previous|preceding|following|above|next)\s+paragraph\b",
    r"\bas\s+paragraph\s+\d+\s+(?:noted|stated|discussed)",
]


def _check_meta_paragraph_refs(tex: Path) -> List[Tuple[int, str]]:
    """H24: agents fill page count with paragraphs titled 'In introduction paragraph 1, ...' or
    'see paragraph 7'. Real papers do not reference paragraphs by number. Returns a list of
    (line_number, matched_snippet) tuples — empty == OK."""
    if not tex.is_file():
        return []
    hits: List[Tuple[int, str]] = []
    for i, line in enumerate(tex.read_text(errors="replace").splitlines(), 1):
        s = line.strip()
        if s.startswith("%"):
            continue
        for pat in META_PARAGRAPH_PATTERNS:
            m = re.search(pat, s, re.IGNORECASE)
            if m:
                hits.append((i, s[max(0, m.start()-10):m.end()+30]))
                break
    return hits


def _section_bodies(tex_text: str) -> Dict[str, str]:
    """Split main.tex into a {section_name: body_text} dict (excluding appendix)."""
    t = re.split(r"\\appendix|\\section\{\s*Appendix", tex_text, maxsplit=1)[0]
    parts = re.split(r"\\section\*?\{([^}]+)\}", t)
    out: Dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i+1] if i+1 < len(parts) else ""
        out[name] = body
    return out


def _check_citation_distribution(tex: Path) -> Tuple[bool, Dict[str, int]]:
    """H23: \\cite{} must appear in each of Introduction, Related Work, Methods, Discussion.
    A paper that dumps all citations at the end of one section is not citing — it's listing.
    Returns (ok, {section: cite_count})."""
    if not tex.is_file():
        return (False, {})
    text = tex.read_text(errors="replace")
    secs = _section_bodies(text)
    required = ["Introduction", "Related Work", "Methods", "Discussion"]
    counts: Dict[str, int] = {}
    for sec_name in required:
        body = None
        for k, v in secs.items():
            if k.strip().lower() == sec_name.lower():
                body = v
                break
        counts[sec_name] = len(re.findall(r"\\cite[a-z]*\{[^}]+\}", body or ""))
    ok = all(counts[s] >= 1 for s in required)
    # Additionally: no single section may hold > 60% of all citations.
    total = sum(counts.values()) or 1
    max_share = max(counts.values()) / total
    if max_share > 0.60:
        ok = False
    counts["_max_share_of_total"] = round(max_share, 3)
    return (ok, counts)


GOVERNING_EQ_MARKERS = [
    r"\\nabla\s*\\cdot",            # divergence
    r"\\partial.*\\partial\s*t",    # time derivative pair (∂u/∂t etc.)
    r"\\overline\{u",               # Reynolds-averaged velocity
    r"\\overline\{U",
    r"\\rho\s*\\bigl?\(?\s*\\partial",  # ρ ∂…/∂t (momentum)
    r"continuity",                  # textual flag
    r"momentum\s+equation",         # textual flag
    r"Navier.?Stokes",              # textual flag
]
CLOSURE_EQ_MARKERS = [
    # k-omega SST closure
    r"\\omega",
    r"\\beta\^?\*",
    # k-epsilon
    r"\\epsilon",
    r"\\varepsilon",
    # SA
    r"\\tilde\{\\nu\}", r"\\widetilde\{\\nu\}", r"\\nutilde", r"nuTilda",
    r"Spalart.?Allmaras",
    # eddy viscosity / Reynolds-stress
    r"\\nu_?t",
    r"\\mu_?t",
    # Bingham / Carreau / power-law
    r"Bingham", r"Carreau", r"power.?law",
]


def _check_governing_and_closure_equations(tex: Path) -> Tuple[bool, bool, List[str]]:
    """H25: at least one display-math block contains governing-equation markers (continuity /
    Reynolds-averaged momentum / Navier–Stokes) AND at least one contains turbulence-closure
    markers (k–ω, k–ε, SA transport, ν_t, …). Returns (governing_ok, closure_ok, found_markers).
    A paper with only Re/Cf/RMSE definitions is not enough."""
    if not tex.is_file():
        return (False, False, [])
    t = tex.read_text(errors="replace")
    t = re.split(r"\\appendix|\\section\{\s*Appendix", t, maxsplit=1)[0]
    eq_blocks = re.findall(
        r"\\begin\{(?:equation|align|gather|multline)\*?\}.*?"
        r"\\end\{(?:equation|align|gather|multline)\*?\}",
        t, flags=re.DOTALL,
    )
    # Also include \[ … \] display math
    eq_blocks += re.findall(r"\\\[(.*?)\\\]", t, flags=re.DOTALL)
    governing_hit: List[str] = []
    closure_hit: List[str] = []
    for blk in eq_blocks:
        for pat in GOVERNING_EQ_MARKERS:
            if re.search(pat, blk, re.IGNORECASE):
                governing_hit.append(pat)
        for pat in CLOSURE_EQ_MARKERS:
            if re.search(pat, blk, re.IGNORECASE):
                closure_hit.append(pat)
    return (
        bool(governing_hit),
        bool(closure_hit),
        sorted(set(governing_hit + closure_hit)),
    )


GEOMETRY_MARKERS = [
    r"step\s+height",            r"\bh\s*=\s*\d",
    r"expansion\s+ratio",        r"\bER\b",
    r"channel\s+height",         r"hill\s+height",
    r"domain\s+(?:length|extent|size)",
    r"inlet.{0,40}(?:length|location|x\s*/\s*h)",
    r"outlet.{0,40}(?:length|location|x\s*/\s*h)",
    r"Re_?h\s*=\s*\d",           r"Re_?\\tau\s*=\s*\d",
    r"x\s*/\s*h",                r"y\s*/\s*h",
    r"blockMesh",                r"\bbcs?\b.{0,40}(?:inlet|outlet|wall)",
]


def _check_geometry_description(tex: Path) -> Tuple[bool, List[str]]:
    """H26: Methods must describe the geometry (step height, expansion ratio, domain extents,
    Reynolds number formula, etc.). Returns (ok, matched_markers)."""
    if not tex.is_file():
        return (False, [])
    text = tex.read_text(errors="replace")
    secs = _section_bodies(text)
    methods_body = ""
    for k, v in secs.items():
        if "method" in k.strip().lower():
            methods_body += v + "\n"
    haystack = methods_body or text  # fall back to whole body if no Methods section
    hit: List[str] = []
    for pat in GEOMETRY_MARKERS:
        if re.search(pat, haystack, re.IGNORECASE):
            hit.append(pat)
    return (len(hit) >= 3, hit)


def _has_quantitative_results_table(tex: Path) -> Tuple[bool, int]:
    """H22: a results-quality CFD paper must show numerical comparisons in a table.
    Look for any tabular environment whose body contains ≥ 3 numeric tokens
    (integers or decimals). Returns (present, n_tabular_with_numbers).
    """
    if not tex.is_file():
        return (False, 0)
    text = tex.read_text(errors="replace")
    text = re.split(r"\\appendix|\\section\{\s*Appendix", text, maxsplit=1)[0]
    tabulars = re.findall(
        r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}",
        text, flags=re.DOTALL,
    )
    n_quant = 0
    for tab in tabulars:
        nums = re.findall(r"(?<![A-Za-z_])-?\d+\.\d+|(?<![A-Za-z_])-?\d{2,}", tab)
        if len(nums) >= 3:
            n_quant += 1
    return (n_quant > 0, n_quant)


PER_CASE_KINDS = {
    "u_contour", "p_contour", "velocity_magnitude_contour",
    "cf_profile", "pressure_profile", "streamlines",
}


def _figure_jobs_have_out_png(plan_data: Dict[str, Any], out_dir: Path) -> Tuple[bool, int, int, List[str]]:
    """H16 tightened (slice 18.1): each figure_jobs[i] must carry a non-null
    `out_png` string, AND:
      - all out_png values must be unique within the plan (no two jobs
        sharing the same file);
      - each out_png must point at a file that exists on disk;
      - per-case kinds (u_contour, p_contour, cf_profile, …) must:
          (a) live under paper_figs/ (not cross_experiment_analysis/), and
          (b) have the case_id substring in the filename — so the planner
              cannot label a cross-summary PNG as a per-case contour.

    Returns (all_have_and_valid, n_valid, n_total, reasons_for_invalid).
    """
    jobs = plan_data.get("figure_jobs", [])
    if not isinstance(jobs, list) or not jobs:
        return (False, 0, 0, ["figure_jobs missing or empty"])
    reasons: List[str] = []
    n_valid = 0
    seen_out_png: Dict[str, int] = {}
    for i, j in enumerate(jobs):
        if not isinstance(j, dict):
            reasons.append(f"figure_jobs[{i}] is not an object")
            continue
        out_png = j.get("out_png")
        if not (isinstance(out_png, str) and out_png.strip()):
            reasons.append(f"figure_jobs[{i}] (id={j.get('id')!r}) has no out_png")
            continue
        # Uniqueness — no two jobs may share the same out_png leaf name. This
        # blocks the "every per-case contour points at cf_overlay.png" trick.
        leaf = Path(out_png).name
        seen_out_png[leaf] = seen_out_png.get(leaf, 0) + 1
        # Existence on disk — out_png must resolve under out_dir/(paper_figs|cross_experiment_analysis|.).
        candidates = [
            out_dir / out_png,
            out_dir / "paper_figs" / leaf,
            out_dir / "cross_experiment_analysis" / leaf,
            out_dir / "paper" / leaf,
        ]
        if not any(c.is_file() for c in candidates):
            reasons.append(
                f"figure_jobs[{i}] (id={j.get('id')!r}) out_png={out_png!r} "
                f"does not exist on disk under paper_figs/, cross_experiment_analysis/, or paper/"
            )
            continue
        # Per-case kind rules
        kind = j.get("kind", "")
        if isinstance(kind, str) and kind.lower() in PER_CASE_KINDS:
            cid = j.get("case_id", "")
            if not (isinstance(cid, str) and cid):
                reasons.append(
                    f"figure_jobs[{i}] kind={kind!r} requires case_id; got {cid!r}"
                )
                continue
            # Must live under paper_figs/ — not the cross-summary directory.
            if not (out_dir / "paper_figs" / leaf).is_file():
                reasons.append(
                    f"figure_jobs[{i}] (case_id={cid!r}, kind={kind!r}) out_png={leaf!r} "
                    f"is not under paper_figs/ — per-case contours/profiles must be "
                    f"rendered into paper_figs/, not reused from cross_experiment_analysis/"
                )
                continue
            # Filename must reference the case_id (loose: contain it as substring).
            if cid not in leaf:
                reasons.append(
                    f"figure_jobs[{i}] kind={kind!r} declares case_id={cid!r} but "
                    f"out_png={leaf!r} does not contain that case_id — planner is "
                    f"mislabelling a generic PNG as a per-case render"
                )
                continue
        n_valid += 1
    # Add duplicates as reasons (after the loop, once we have full counts).
    for leaf, n in seen_out_png.items():
        if n > 1:
            reasons.append(
                f"out_png {leaf!r} is used by {n} figure_jobs — each job must "
                f"declare a unique out_png (cannot reuse one PNG for multiple per-case figures)"
            )
            # Subtract duplicated jobs from valid count (any duplicate poisons all uses)
            n_valid = max(0, n_valid - n)
    ok = (n_valid == len(jobs) and not reasons)
    return (ok, n_valid, len(jobs), reasons[:10])


def _includegraphics_covers_figure_jobs(tex_cite_paths: List[str], plan_data: Dict[str, Any]) -> Tuple[bool, int, int]:
    """H15 tightened: every plan figure_jobs[i].out_png must be referenced by an
    \\includegraphics in main.tex. Returns (all_covered, n_covered, n_jobs)."""
    jobs = plan_data.get("figure_jobs", [])
    if not isinstance(jobs, list) or not jobs:
        return (False, 0, 0)
    cites = [Path(p).name for p in tex_cite_paths]
    n_covered = 0
    for j in jobs:
        if not isinstance(j, dict): continue
        out_png = j.get("out_png")
        if not isinstance(out_png, str) or not out_png: continue
        leaf = Path(out_png).name
        if any(leaf in c or c == leaf for c in cites):
            n_covered += 1
    return (n_covered == len(jobs), n_covered, len(jobs))


def _compute_paper_audit(out_dir: Path, route: int = 0) -> Dict[str, Any]:
    """Deterministic paper-stage truth table. The reviewer LLM does NOT author these — this does."""
    paper = out_dir / "paper"
    pdf = paper / "main.pdf"
    tex = paper / "main.tex"
    cea = out_dir / "cross_experiment_analysis"
    # Slice 14: prefer lit_enriched.json (Stage 0.5 output) over lit.json
    lit_enriched = out_dir / "lit_enriched.json"
    lit_base = out_dir / "lit.json"
    lit = lit_enriched if lit_enriched.is_file() else lit_base
    review = out_dir / "review.json"

    audit: Dict[str, Any] = {
        "pdf_present": pdf.is_file(),
        "pdf_size_bytes": pdf.stat().st_size if pdf.is_file() else 0,
        "main_body_pages": -1,
        "main_body_words": -1,
        "words_per_page": 0.0,
        "length_ok": False,
        "word_density_ok": False,
        "cross_experiment_figures_ok": False,
        "cross_experiment_analysis_dir_present": cea.is_dir(),
        "cross_experiment_pngs": 0,
        "tex_cites_cross_experiment": False,
        "tex_cite_paths": [],
        # Slice-12 additions:
        "n_citations": 0,
        "n_lit_records": 0,
        "min_citations_required": 0,
        "citations_ok": False,
        "main_body_words_main": 0,
        "main_body_words_padding": 0,
        "padding_fraction": 0.0,
        "section_concentration_ok": False,
        "cross_experiment_figure_types": [],
        "n_distinct_cross_fig_types": 0,
        "cross_experiment_richness_ok": False,
        "figure_physics_ok": None,   # LLM-authored in review.json; we copy + verify presence
        # Slice-16 (paper-cycle decomposition) additions:
        "sections_ok": False,
        "sections_present": [],
        "sections_missing": [],
        "paper_figs_present": False,
        "paper_figs_n_png": 0,
        "paper_figs_avg_bytes": 0,
        "paper_figs_ok": False,
        # Slice-17 (schema-validation for fabricated artifacts):
        "paper_unified_plan_present": False,
        "paper_unified_plan_schema_ok": False,
        "review_schema_ok": False,
        # Slice-18 (H18–H22): hard gates against the OED/BFS fabrication patterns.
        "route": route,
        "lit_retrieval_ok": False,         # H18 — lit.json is non-empty for routes 1/2/5/6
        "refs_bib_present": False,         # H19 — refs.bib OR inline thebibliography found
        "refs_bib_entries": 0,
        "refs_bib_inline": False,
        "refs_bib_ok": False,
        "n_equation_blocks": 0,            # H20 — display math required for a CFD methods paper
        "equations_ok": False,
        "duplicate_paragraphs_count": 0,   # H21 — writer must not copy/paste paragraphs to inflate length
        "duplicate_paragraphs_samples": [],
        "duplicate_paragraphs_ok": False,
        "n_quant_tabulars": 0,             # H22 — quantitative comparisons in tables
        "quant_table_ok": False,
        "figure_jobs_have_out_png": False, # H16 tighter — each figure_job needs out_png
        "figure_jobs_with_out_png": 0,
        "figure_jobs_total": 0,
        "figure_jobs_invalid_reasons": [],
        "figure_jobs_covered_by_includegraphics": 0,
        "figure_jobs_includegraphics_ok": False,  # H15 tighter — writer must embed every planned PNG
        # Slice 18.2 (H23–H26): content-quality gates against the BFS-v2 trash patterns.
        "citation_distribution": {},        # H23 — \cite must appear in each major section
        "citation_distribution_ok": False,
        "meta_paragraph_refs": [],          # H24 — paragraphs must not reference each other by number
        "meta_paragraph_refs_ok": False,
        "governing_equations_ok": False,    # H25 — at least one equation must be a governing equation
        "closure_equations_ok": False,      # H25 — at least one equation must be the turbulence closure
        "equation_markers_found": [],
        "geometry_description_ok": False,   # H26 — Methods must describe the geometry
        "geometry_markers_found": [],
        # Slice 19.2 — H32 mesh-gate must be visible in paper Methods
        "mesh_gate_in_paper_ok": True,
        "mesh_gate_paper_evidence": [],
        # Slice 19 (H28, H29, H30): per-case coverage from manifest, PNG-size
        # thresholds (per-case + cross-summary), and saved-script integrity.
        "per_case_coverage_ok": True,
        "per_case_coverage_missing": [],
        "per_case_png_size_ok": True,
        "per_case_png_undersized": [],
        "cross_plots_size_ok": True,
        "cross_plots_undersized": [],
        "cross_script_present": False,
        "cross_script_bytes": 0,
        "cross_script_ok": True,
    }

    if pdf.is_file():
        total_pages = _pdf_page_count(pdf)
        total_words_pdf = _pdf_word_count(pdf)  # whole PDF for total reference

        # Main body = everything in main.tex BEFORE \appendix or first \section{Appendix...}
        main_words_tex = _tex_word_count_main_body(tex) if tex.is_file() else -1

        # Approximate main-body pages: scale total pages by main/total word ratio.
        # This catches the agent-dumps-everything-into-appendix failure mode.
        main_pages_est = -1
        if total_pages > 0 and main_words_tex > 0 and total_words_pdf > 0:
            main_fraction = main_words_tex / max(total_words_pdf, 1)
            main_pages_est = max(1, round(total_pages * main_fraction))

        audit["main_body_pages"] = main_pages_est
        audit["total_pdf_pages"] = total_pages
        audit["main_body_words"] = main_words_tex
        audit["total_pdf_words"] = total_words_pdf
        audit["main_to_total_word_ratio"] = round(main_words_tex / total_words_pdf, 3) if total_words_pdf > 0 else 0.0

        if main_words_tex > 0 and main_pages_est > 0:
            audit["words_per_page"] = round(main_words_tex / main_pages_est, 1)
        # H1: main-body pages in [8, 15] — appendix doesn't count
        audit["length_status"] = _range_status(main_pages_est, 8, 15)
        audit["length_ok"] = (audit["length_status"] != "fail")
        # H9: main-body density >= 350 words/page
        audit["word_density_status"] = _floor_status(audit["words_per_page"], 350.0)
        audit["word_density_ok"] = (audit["word_density_status"] != "fail")

    if cea.is_dir():
        pngs = list(cea.glob("*.png"))
        audit["cross_experiment_pngs"] = len(pngs)
        types_found = set()
        for p in pngs:
            for tname, pat in CROSS_FIG_TYPE_KEYWORDS.items():
                if pat.search(p.name):
                    types_found.add(tname)
        audit["cross_experiment_figure_types"] = sorted(types_found)
        audit["n_distinct_cross_fig_types"] = len(types_found)
        # H12: ≥ 2 distinct figure types
        audit["cross_experiment_richness_ok"] = (len(types_found) >= 2)

    if tex.is_file():
        tex_text = tex.read_text(errors="replace")
        cite_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text)
        audit["tex_cite_paths"] = cite_paths
        # H8: at least one \includegraphics must point at cross_experiment_analysis/
        audit["tex_cites_cross_experiment"] = any(
            "cross_experiment_analysis/" in p for p in cite_paths
        )
        audit["cross_experiment_figures_ok"] = (
            audit["tex_cites_cross_experiment"]
            and audit["cross_experiment_pngs"] >= 1
        )
        # H10: citation density (\cite{} count vs lit.json record count)
        n_cites = len(re.findall(r"\\cite\{[^}]+\}", tex_text))
        audit["n_citations"] = n_cites
        if lit.is_file():
            try:
                lit_data = json.loads(lit.read_text())
                if isinstance(lit_data, dict):
                    recs = lit_data.get("records") or lit_data.get("papers") or lit_data.get("results") or []
                else:
                    recs = lit_data
                if isinstance(recs, list):
                    audit["n_lit_records"] = len(recs)
            except Exception:
                pass
        # Slice 18 / H18: routes that include literature retrieval (1, 2, 5, 6) MUST
        # have a non-empty literature record. An empty lit.json no longer auto-lowers
        # the citation floor — the floor is fixed at 10 for these routes, which forces
        # the gate to fail loudly if literature was skipped, rather than rubber-stamping
        # a zero-citation paper.
        if route in ROUTES_WITH_PAPER:
            audit["min_citations_required"] = max(10, audit["n_lit_records"] // 4)
        elif audit["n_lit_records"] > 0:
            audit["min_citations_required"] = max(5, audit["n_lit_records"] // 4)
        else:
            audit["min_citations_required"] = 0
        audit["citations_ok"] = (n_cites >= audit["min_citations_required"])
        audit["lit_retrieval_ok"] = (
            route not in ROUTES_WITH_PAPER or audit["n_lit_records"] > 0
        )

        # H11: section concentration — both within-main-body padding AND appendix dominance
        main_w, pad_w, _bd = _section_word_breakdown(tex)
        audit["main_body_words_main"] = main_w
        audit["main_body_words_padding"] = pad_w
        denom = main_w + pad_w
        if denom > 0:
            audit["padding_fraction"] = round(pad_w / denom, 3)
        # appendix_fraction = appendix_words_pdf / total_words_pdf  (estimate via tex word ratio)
        if audit["total_pdf_words"] > 0 and audit["main_body_words"] > 0:
            audit["appendix_fraction"] = round(
                1.0 - audit["main_body_words"] / audit["total_pdf_words"], 3
            )
        else:
            audit["appendix_fraction"] = 0.0
        audit["section_concentration_status"] = _worst_status(
            _ceiling_status(audit["padding_fraction"], 0.30),
            _ceiling_status(audit["appendix_fraction"], 0.30),
        )
        audit["section_concentration_ok"] = (audit["section_concentration_status"] != "fail")

    # H13: figure physics correctness — LLM-authored in review.json#figure_physics_ok
    if review.is_file():
        try:
            rd = json.loads(review.read_text())
            v = rd.get("figure_physics_ok")
            if isinstance(v, bool):
                audit["figure_physics_ok"] = v
        except Exception:
            pass

    # Slice 16 — H14 mandatory journal section headers
    MANDATORY_SECTIONS = ["Introduction", "Related Work", "Methods", "Results", "Discussion", "Conclusion"]
    if tex.is_file():
        tex_text_l = tex.read_text(errors="replace")
        present, missing = [], []
        for sec in MANDATORY_SECTIONS:
            # accept exact section heading, case-insensitive; tolerate trailing space
            pat = re.compile(r"\\section\*?\{\s*" + re.escape(sec) + r"\s*\}", re.IGNORECASE)
            if pat.search(tex_text_l):
                present.append(sec)
            else:
                missing.append(sec)
        audit["sections_present"] = present
        audit["sections_missing"] = missing
        audit["sections_ok"] = (len(missing) == 0)

    # Slice 16 — H15 paper_figs/ population (catches gnuplot-only / batch-viz-skipped runs)
    paper_figs = out_dir / "paper_figs"
    if paper_figs.is_dir():
        png_list = [p for p in paper_figs.glob("*.png") if p.is_file()]
        audit["paper_figs_present"] = True
        audit["paper_figs_n_png"] = len(png_list)
        sizes = [p.stat().st_size for p in png_list]
        avg = int(sum(sizes) / max(1, len(sizes))) if sizes else 0
        audit["paper_figs_avg_bytes"] = avg
        # ≥ 2 PNGs and average size ≥ 20 KB (gnuplot placeholders are typically < 20 KB)
        # Count >=2 is structural (no tolerance); avg-bytes floor is softable.
        if len(png_list) < 2:
            audit["paper_figs_status"] = "fail"
        else:
            audit["paper_figs_status"] = _floor_status(avg, 20_000)
        audit["paper_figs_ok"] = (audit["paper_figs_status"] != "fail")

    # Slice 17 — paper_unified_plan.json schema (catches fabricated plans with wrong shape)
    # The script writes the schema: figure_jobs, unified_viz_brief, case_order_for_paper,
    # case_display_labels, omit_case_ids, analysis_narrative_hints, topic_paragraph_for_intro.
    # A fabricated plan has different keys (e.g. {title, sections, figures}).
    plan = out_dir / "paper_unified_plan.json"
    audit["paper_unified_plan_schema_ok"] = False
    audit["paper_unified_plan_present"] = plan.is_file()
    plan_data: Dict[str, Any] = {}
    if plan.is_file():
        try:
            plan_data = json.loads(plan.read_text())
            if not isinstance(plan_data, dict):
                plan_data = {}
            required = ["figure_jobs", "unified_viz_brief", "case_order_for_paper",
                        "case_display_labels", "omit_case_ids",
                        "analysis_narrative_hints", "topic_paragraph_for_intro"]
            audit["paper_unified_plan_schema_ok"] = (
                isinstance(plan_data, dict)
                and all(k in plan_data for k in required)
                and isinstance(plan_data.get("figure_jobs"), list)
                and len(plan_data.get("figure_jobs", [])) >= 1
            )
        except Exception:
            pass

    # Slice 18 — H16 tighter: each figure_jobs[i] must carry a non-null out_png string.
    # The OED run produced figure_jobs=[{name,path}] (wrong schema) and the BFS run
    # produced figure_jobs=[{case_id, what_to_visualize}] with no out_png. Both made
    # the writer guess, both lost the per-case contours.
    # Slice 18.1 — also: out_png must be unique, exist on disk, and per-case kinds
    # must live under paper_figs/ with the case_id in the filename (BFS v2 declared
    # 8 per-case contour jobs whose out_png all pointed at the same 3 cross-summary
    # PNGs — H16 passed by letter, fully gamed by intent).
    if plan_data:
        ok, n_have, n_total, jobs_reasons = _figure_jobs_have_out_png(plan_data, out_dir)
        audit["figure_jobs_have_out_png"] = ok
        audit["figure_jobs_with_out_png"] = n_have
        audit["figure_jobs_total"] = n_total
        audit["figure_jobs_invalid_reasons"] = jobs_reasons
        # Slice 18 — H15 tighter: every figure_jobs[i].out_png must be embedded as
        # an \includegraphics in main.tex (or appear in tex_cite_paths). Catches
        # orphan PNGs sitting on disk but never wired into the paper.
        cov_ok, n_cov, _ = _includegraphics_covers_figure_jobs(
            audit.get("tex_cite_paths", []), plan_data
        )
        audit["figure_jobs_covered_by_includegraphics"] = n_cov
        audit["figure_jobs_includegraphics_ok"] = cov_ok and ok

    # Slice 18 — H19 bibliography present (external refs.bib OR inline thebibliography)
    bib_present, bib_n, bib_inline = _count_refs_bib_entries(out_dir)
    audit["refs_bib_present"] = bib_present
    audit["refs_bib_entries"] = bib_n
    audit["refs_bib_inline"] = bib_inline
    # Bibliography entries must cover all \cite{} keys (heuristic: ≥ n_citations,
    # or ≥ 10 if route requires literature).
    min_bib_entries = (
        max(audit["n_citations"], 10) if route in ROUTES_WITH_PAPER
        else audit["n_citations"]
    )
    audit["refs_bib_ok"] = bib_present and bib_n >= min_bib_entries

    # Slice 18 — H20 equation blocks
    n_eqs = _count_equation_blocks(tex)
    audit["n_equation_blocks"] = n_eqs
    # routes 1/2/6 (full experimental papers): ≥ 3 display equations expected
    # route 5 (paper_only): ≥ 1 expected
    audit["equations_ok"] = (n_eqs >= (3 if route in (1, 2, 6) else 1 if route == 5 else 0))

    # Slice 18 — H21 duplicate paragraphs
    dups = _find_duplicate_paragraphs(tex)
    audit["duplicate_paragraphs_count"] = len(dups)
    audit["duplicate_paragraphs_samples"] = [
        {"snippet": s, "occurrences": n} for s, n in dups[:5]
    ]
    audit["duplicate_paragraphs_ok"] = (len(dups) == 0)

    # Slice 18 — H22 quantitative comparison table
    qt_present, n_qt = _has_quantitative_results_table(tex)
    audit["n_quant_tabulars"] = n_qt
    # routes 2 + 6 are model-modification / open-discovery — quantitative comparison
    # tables are mandatory. Routes 1 + 5: at least 1 recommended.
    audit["quant_table_ok"] = qt_present if route in ROUTES_WITH_PAPER else True

    # Slice 18.2 — H23 citation distribution across Introduction / Related Work / Methods / Discussion
    cit_ok, cit_dist = _check_citation_distribution(tex)
    audit["citation_distribution"] = cit_dist
    audit["citation_distribution_ok"] = cit_ok if route in ROUTES_WITH_PAPER else True

    # Slice 18.2 — H24 meta-paragraph references
    mpr_hits = _check_meta_paragraph_refs(tex)
    audit["meta_paragraph_refs"] = [
        {"line": ln, "snippet": sn[:100]} for ln, sn in mpr_hits[:10]
    ]
    audit["meta_paragraph_refs_ok"] = (len(mpr_hits) == 0)

    # Slice 18.2 — H25 governing + closure equations (content, not just count)
    gov_ok, clo_ok, markers = _check_governing_and_closure_equations(tex)
    audit["governing_equations_ok"] = gov_ok if route in ROUTES_WITH_PAPER else True
    audit["closure_equations_ok"] = clo_ok if route in ROUTES_WITH_PAPER else True
    audit["equation_markers_found"] = markers

    # Slice 18.2 — H26 geometry description in Methods
    geom_ok, geom_hit = _check_geometry_description(tex)
    audit["geometry_description_ok"] = geom_ok if route in ROUTES_WITH_PAPER else True
    audit["geometry_markers_found"] = geom_hit

    # Slice 19.2 — H32 mesh-gate evidence in paper Methods
    # Whichever path was used (real mesh-gate or starter_mesh_locked), the paper's
    # Methods MUST acknowledge it. A reader cannot assess CFD results without
    # knowing what mesh decision was made and why.
    audit["mesh_gate_in_paper_ok"] = True
    audit["mesh_gate_paper_evidence"] = []
    if route in ROUTES_WITH_PAPER and tex.is_file():
        text = tex.read_text(errors="replace")
        secs = _section_bodies(text)
        methods_body = ""
        for k, v in secs.items():
            if "method" in k.strip().lower():
                methods_body += v + "\n"
        # Patterns that signal mesh-gate evidence in Methods.
        mesh_patterns = [
            r"mesh\s+(?:independence|sensitivity|refinement|convergence)",
            r"grid\s+(?:independence|sensitivity|refinement|convergence)",
            r"baseline.{0,30}refined",
            r"refined.{0,30}baseline",
            r"GCI\b", r"Richardson\s+extrap",
            r"locked\s+mesh", r"mesh.{0,20}locked", r"starter\s+mesh",
            r"\bblockMesh", r"cell\s+count",
            r"y\^?\+", r"y\\?\^\\?\{?\\?\+",
        ]
        hits = [p for p in mesh_patterns if re.search(p, methods_body, re.IGNORECASE)]
        audit["mesh_gate_paper_evidence"] = hits
        # Require ≥ 2 markers (one for the decision, one for justification).
        audit["mesh_gate_in_paper_ok"] = (len(hits) >= 2)

    # Slice 19 — H28 per-case figure coverage from manifest.json
    # Catches the planner cherry-picking only baseline + winner and skipping the
    # other N-2 candidate cases. Reads manifest.json#cases as ground truth, then
    # verifies the plan has ≥1 u_contour + ≥1 p_contour per case AND that the
    # corresponding paper_figs/*.png exist at non-trivial size.
    audit["per_case_coverage_ok"] = True
    audit["per_case_coverage_missing"] = []
    mf_path = out_dir / "manifest.json"
    if route in ROUTES_WITH_PAPER and mf_path.is_file() and plan_data:
        try:
            mf = json.loads(mf_path.read_text())
            manifest_case_ids = [
                c.get("case_id") for c in mf.get("cases", [])
                if c.get("case_id") and c.get("status") == "success"
            ]
            jobs = plan_data.get("figure_jobs", []) or []
            missing = []
            for cid in manifest_case_ids:
                kinds_for_case = {
                    (j.get("kind") or "").lower()
                    for j in jobs
                    if isinstance(j, dict) and j.get("case_id") == cid
                }
                want = {"u_contour", "p_contour"}
                have = want & kinds_for_case
                if have != want:
                    missing.append({"case_id": cid, "missing_kinds": sorted(want - have)})
            audit["per_case_coverage_missing"] = missing
            audit["per_case_coverage_ok"] = (not missing)
        except Exception as e:
            audit["per_case_coverage_ok"] = False
            audit["per_case_coverage_missing"] = [{"error": str(e)}]

    # Slice 19 — H30 cross-experiment plot quality (catches stub bar/line plots)
    # Cross-experiment PNGs (cf_overlay, rmse_bar, reattachment_bar, etc.) authored
    # via matplotlib must use journal-quality styling per cfd-cross-analyze's
    # rcParams template. Threshold: ≥ 50 KB at 1600×900. Under-styled defaults
    # (grayscale, no labels, no annotations) produce 20-30 KB PNGs.
    audit["cross_plots_size_ok"] = True
    audit["cross_plots_status"] = "pass"
    audit["cross_plots_undersized"] = []
    audit["cross_plots_warn"] = []
    audit["cross_script_present"] = False
    audit["cross_script_bytes"] = 0
    audit["cross_script_ok"] = True
    audit["cross_script_status"] = "pass"
    if route in ROUTES_WITH_PAPER:
        cea_dir = out_dir / "cross_experiment_analysis"
        if cea_dir.is_dir():
            failed = []
            warned = []
            statuses = []
            for png in cea_dir.glob("*.png"):
                sz = png.stat().st_size
                s = _floor_status(sz, 50_000)
                statuses.append(s)
                if s == "fail":
                    failed.append({"file": png.name, "bytes": sz})
                elif s == "warn":
                    warned.append({"file": png.name, "bytes": sz})
            audit["cross_plots_undersized"] = failed
            audit["cross_plots_warn"] = warned
            audit["cross_plots_status"] = _worst_status(*statuses) if statuses else "pass"
            audit["cross_plots_size_ok"] = (audit["cross_plots_status"] != "fail")
            # Saved-script check: cfd-cross-analyze must write the actual Python
            # used to author the plots (≥ 2 KB of real code, not a comment stub).
            scripts_dir = cea_dir / "scripts"
            if scripts_dir.is_dir():
                py_scripts = sorted(scripts_dir.glob("*.py"))
                if py_scripts:
                    # Take the largest .py file (the final-attempt script).
                    largest = max(py_scripts, key=lambda p: p.stat().st_size)
                    audit["cross_script_present"] = True
                    audit["cross_script_bytes"] = largest.stat().st_size
                    # Count non-comment, non-blank lines as a stub-detector.
                    non_comment_lines = 0
                    try:
                        for line in largest.read_text(errors="replace").splitlines():
                            s = line.strip()
                            if not s or s.startswith("#"): continue
                            non_comment_lines += 1
                    except Exception:
                        pass
                    audit["cross_script_status"] = _worst_status(
                        _floor_status(largest.stat().st_size, 2_000),
                        _floor_status(non_comment_lines, 30),
                    )
                    audit["cross_script_ok"] = (audit["cross_script_status"] != "fail")

    # Slice 19 — H29 per-case PNG size threshold (catches matplotlib placeholders
    # masquerading as PyVista field renders). Real PyVista OpenFOAM contours at
    # ≥ 1200x600 weigh > 80 KB; matplotlib synthetic-data placeholders are < 50 KB.
    audit["per_case_png_size_ok"] = True
    audit["per_case_png_status"] = "pass"
    audit["per_case_png_undersized"] = []
    audit["per_case_png_warn"] = []
    if route in ROUTES_WITH_PAPER and plan_data:
        failed = []
        warned = []
        statuses = []
        for j in plan_data.get("figure_jobs", []) or []:
            if not isinstance(j, dict): continue
            kind = (j.get("kind") or "").lower()
            if kind not in PER_CASE_KINDS: continue
            out_png = j.get("out_png", "")
            if not out_png: continue
            leaf = Path(out_png).name
            pf = out_dir / "paper_figs" / leaf
            if pf.is_file():
                sz = pf.stat().st_size
                s = _floor_status(sz, 80_000)
                statuses.append(s)
                entry = {
                    "case_id": j.get("case_id"),
                    "kind": kind,
                    "out_png": leaf,
                    "bytes": sz,
                }
                if s == "fail":
                    failed.append(entry)
                elif s == "warn":
                    warned.append(entry)
        audit["per_case_png_undersized"] = failed
        audit["per_case_png_warn"] = warned
        audit["per_case_png_status"] = _worst_status(*statuses) if statuses else "pass"
        audit["per_case_png_size_ok"] = (audit["per_case_png_status"] != "fail")

    # Slice 17 — review.json schema (catches fabricated reviews missing recommendations[],
    # formatting_ok, figures_ok, content_ok, regenerate_batch_figures, etc.)
    audit["review_schema_ok"] = False
    if review.is_file():
        try:
            rd = json.loads(review.read_text())
            required = ["pass", "score", "recommendations", "formatting_ok",
                        "figures_ok", "content_ok", "summary"]
            audit["review_schema_ok"] = (
                isinstance(rd, dict)
                and all(k in rd for k in required)
                and isinstance(rd.get("recommendations"), list)
            )
        except Exception:
            pass

    return audit


def _check_paper_stage(out_dir: Path, route: int) -> Tuple[List[str], List[str]]:
    if route not in ROUTES_WITH_PAPER:
        return [], []
    failures: List[str] = []
    warnings: List[str] = []

    if not (out_dir / "paper_experiment_plan.json").is_file():
        failures.append("paper/Stage 0 not run: paper_experiment_plan.json missing")

    cea = out_dir / "cross_experiment_analysis"
    if not cea.is_dir():
        failures.append("cfd-cross-analyze not run: cross_experiment_analysis/ missing")
    else:
        png_count = len(list(cea.glob("*.png")))
        if png_count < 1:
            failures.append("cross_experiment_analysis/ has zero PNGs (no comparison figures)")
        if not (cea / "aggregate.csv").is_file():
            failures.append("cross_experiment_analysis/aggregate.csv missing")
        if not (cea / "cross_experiment_interpretation.md").is_file():
            failures.append("cross_experiment_analysis/cross_experiment_interpretation.md missing")

    pdf = out_dir / "paper" / "main.pdf"
    if not pdf.is_file():
        failures.append("paper/main.pdf missing")
    else:
        pdf_status = _floor_status(pdf.stat().st_size, 50_000)
        if pdf_status == "fail":
            failures.append(
                f"paper/main.pdf suspiciously small ({pdf.stat().st_size} bytes < "
                f"{int(50_000 * (1 - TOLERANCE))} bytes = 50KB - {int(TOLERANCE*100)}%)"
            )
        elif pdf_status == "warn":
            warnings.append(
                f"paper/main.pdf in tolerance band ({pdf.stat().st_size} bytes, "
                f"floor 50000) — passes but flagged"
            )

    # Deterministic paper-stage gates — replaces trusting reviewer LLM booleans.
    audit = _compute_paper_audit(out_dir, route=route)
    # Persist the deterministic audit unconditionally so cfd-paper can read it.
    try:
        (out_dir / "paper_audit.json").write_text(json.dumps(audit, indent=2))
    except Exception:
        pass

    if pdf.is_file():
        ls = audit.get("length_status", "pass" if audit["length_ok"] else "fail")
        if ls == "fail":
            failures.append(
                f"H1 length_ok=false: main_body_pages={audit['main_body_pages']} not in "
                f"[{8*(1-TOLERANCE):.1f}, {15*(1+TOLERANCE):.1f}] (±{int(TOLERANCE*100)}% band on [8,15]) "
                f"(total PDF pages={audit['total_pdf_pages']}, main-body word ratio={audit.get('main_to_total_word_ratio',0)}; "
                f"appendix dumping not counted toward main body)"
            )
        elif ls == "warn":
            warnings.append(
                f"H1 length in tolerance band: main_body_pages={audit['main_body_pages']} just outside "
                f"[8,15] but within ±{int(TOLERANCE*100)}% — passes"
            )
        wds = audit.get("word_density_status", "pass" if audit["word_density_ok"] else "fail")
        if wds == "fail":
            failures.append(
                f"H9 word_density_ok=false: main-body words/page={audit['words_per_page']} < "
                f"{350.0*(1-TOLERANCE):.1f} floor (350 - {int(TOLERANCE*100)}%) "
                f"(main_body_words={audit['main_body_words']}, main_body_pages_est={audit['main_body_pages']})"
            )
        elif wds == "warn":
            warnings.append(
                f"H9 word density in tolerance band: {audit['words_per_page']} words/page "
                f"(floor 350, within ±{int(TOLERANCE*100)}%) — passes"
            )
        if not audit["cross_experiment_figures_ok"]:
            failures.append(
                "H8 cross_experiment_figures_ok=false: paper/main.tex does NOT cite any figure under "
                "cross_experiment_analysis/. The reviewer LLM cannot override this — the script "
                "verified by grepping \\includegraphics paths."
            )
        # Slice 13 — H10 citation density
        if not audit["citations_ok"]:
            failures.append(
                f"H10 citations_ok=false: \\cite{{}} count={audit['n_citations']} < "
                f"required={audit['min_citations_required']} "
                f"(lit.json has {audit['n_lit_records']} records — use them)"
            )
        # Slice 13 — H11 section concentration (within-main-body padding + appendix dominance)
        scs = audit.get("section_concentration_status", "pass" if audit["section_concentration_ok"] else "fail")
        if scs == "fail":
            failures.append(
                f"H11 section_concentration_ok=false: "
                f"padding_fraction (within main body) = {audit['padding_fraction']*100:.1f}% "
                f"(ceiling 30% + {int(TOLERANCE*100)}% tolerance); "
                f"appendix_fraction (appendix vs total) = {audit['appendix_fraction']*100:.1f}% "
                f"(ceiling 30% + {int(TOLERANCE*100)}% tolerance). "
                f"The agent dumped content into \\appendix to satisfy length while "
                f"keeping main-body short — compress appendices, expand Methods / Results / Discussion."
            )
        elif scs == "warn":
            warnings.append(
                f"H11 section concentration in tolerance band: padding={audit['padding_fraction']*100:.1f}%, "
                f"appendix={audit['appendix_fraction']*100:.1f}% (ceilings 30%, within +{int(TOLERANCE*100)}%) — passes"
            )
        # Slice 13 — H12 cross-experiment richness
        if not audit["cross_experiment_richness_ok"]:
            failures.append(
                f"H12 cross_experiment_richness_ok=false: only "
                f"{audit['n_distinct_cross_fig_types']} distinct cross-experiment figure type(s) "
                f"under cross_experiment_analysis/ ({audit['cross_experiment_figure_types']}); "
                f"require ≥ 2 from {{overlay, bar, trend, delta_field, error, contour}}. "
                f"Invoke cfd-cross-analyze again to author the missing types."
            )
        # Slice 16 — H14 mandatory section headers
        if not audit["sections_ok"]:
            failures.append(
                f"H14 sections_ok=false: missing mandatory section headers in main.tex: "
                f"{audit['sections_missing']}. A journal-format CFD paper requires "
                f"Introduction, Related Work, Methods, Results, Discussion, Conclusion. "
                f"Sections like 'Objective', 'Audit Trail', 'Implementation Location' are "
                f"skill-mode boilerplate, not paper content — replace with the mandatory headers."
            )
        # Slice 16 — H15 paper_figs/ population (catches gnuplot-only / script-skipped runs)
        pfs = audit.get("paper_figs_status", "pass" if audit["paper_figs_ok"] else "fail")
        if pfs == "fail":
            failures.append(
                f"H15 paper_figs_ok=false: paper_figs/ has {audit['paper_figs_n_png']} PNG(s) "
                f"with average size {audit['paper_figs_avg_bytes']} bytes (count must be ≥ 2, "
                f"avg ≥ {int(20_000*(1-TOLERANCE))} bytes = 20KB - {int(TOLERANCE*100)}%). "
                f"Real PyVista renders average ≥ 20 KB; gnuplot placeholders or hand-fabricated "
                f"empty PNGs are smaller. scripts/paper_unified.py (invoked by cfd-paper) authors a batch "
                f"PyVista script with per-image vision QA and writes ≥ 2 × N_cases PNGs into paper_figs/."
            )
        elif pfs == "warn":
            warnings.append(
                f"H15 paper_figs avg in tolerance band: {audit['paper_figs_avg_bytes']} bytes "
                f"(floor 20000, within -{int(TOLERANCE*100)}%) — passes"
            )
        # Slice 17 — H16 paper_unified_plan.json schema (catches fabricated plans)
        if not audit["paper_unified_plan_present"]:
            failures.append(
                "H16 paper_unified_plan.json missing. The planner inside "
                "scripts/paper_unified.py writes this file; if it's absent, the script was not run."
            )
        elif not audit["paper_unified_plan_schema_ok"]:
            failures.append(
                "H16 paper_unified_plan_schema_ok=false: paper_unified_plan.json is present but does "
                "NOT match the script's schema (required keys: figure_jobs[], unified_viz_brief, "
                "case_order_for_paper, case_display_labels, omit_case_ids, analysis_narrative_hints, "
                "topic_paragraph_for_intro). A plan with shape {title, sections, figures} or similar "
                "is a hand-fabricated stub. Re-run scripts/paper_unified.py."
            )
        # Slice 17 — H17 review.json schema (catches fabricated reviews)
        if not audit["review_schema_ok"]:
            failures.append(
                "H17 review_schema_ok=false: review.json is missing required reviewer-prompt keys "
                "(pass, score, recommendations[], formatting_ok, figures_ok, content_ok, summary). "
                "A review with only {pages, min_pages, max_pages, findings} or similar is a "
                "hand-fabricated stub; the reviewer LLM was never invoked. "
                "Re-run scripts/paper_unified.py — its reviewer step writes the full schema."
            )
        # Slice 18 — H18 literature retrieval was not skipped
        if not audit["lit_retrieval_ok"]:
            failures.append(
                f"H18 lit_retrieval_ok=false: route {route} requires literature retrieval, "
                f"but lit.json has {audit['n_lit_records']} records. Empty literature is the "
                f"root cause of zero citations and zero refs.bib — the paper is unciteable. "
                f"Re-run cfd-research (Skill cfd-skills/cfd-literature) BEFORE re-running paper."
            )
        # Slice 18 — H19 bibliography (refs.bib or inline thebibliography with enough entries)
        if not audit["refs_bib_ok"]:
            if not audit["refs_bib_present"]:
                failures.append(
                    "H19 refs_bib_ok=false: no bibliography found. Either ship an external "
                    "paper/refs.bib with @-entries, OR embed a \\begin{thebibliography}…\\end{thebibliography} "
                    "block at the end of main.tex with \\bibitem entries (one per cited reference). "
                    "The OED run had zero of both — a paper without citations is not a paper."
                )
            else:
                failures.append(
                    f"H19 refs_bib_ok=false: bibliography present but only "
                    f"{audit['refs_bib_entries']} entries (need ≥ "
                    f"{max(audit['n_citations'], 10 if route in ROUTES_WITH_PAPER else audit['n_citations'])}). "
                    f"Either add more \\bibitem entries or remove unused \\cite{{}} keys."
                )
        # Slice 18 — H20 display-math equations
        min_eqs = 3 if route in (1, 2, 6) else 1 if route == 5 else 0
        if not audit["equations_ok"]:
            failures.append(
                f"H20 equations_ok=false: only {audit['n_equation_blocks']} display-math "
                f"block(s) (\\begin{{equation}}, \\begin{{align}}, \\[…\\], …); need ≥ {min_eqs} "
                f"for route {route}. A CFD methods paper must show the governing equations, "
                f"the closure (SA / k-ω / Bingham / …), and the modification or comparator formula. "
                f"Numbers in tables are not a substitute for the math."
            )
        # Slice 20 — H21 RE-ENABLED (item 15): near-duplicate paragraphs and
        # intra-paragraph n-gram filler loops. _find_duplicate_paragraphs now
        # uses a difflib similarity ratio (>= 0.85) plus an n-gram-loop scan,
        # so it catches copy-paste-with-cosmetic-edits and token-soup padding —
        # the BFS/Jet repeated-filler pattern — not just verbatim repeats.
        if not audit["duplicate_paragraphs_ok"]:
            samples = "; ".join(
                str(s) for s in audit.get("duplicate_paragraphs_samples", [])[:3]
            )
            failures.append(
                f"H21 duplicate_paragraphs_ok=false: "
                f"{audit['duplicate_paragraphs_count']} near-duplicate or "
                f"filler-loop paragraph issue(s) in the main body"
                + (f" — e.g. {samples}" if samples else "")
                + ". Rewrite the flagged paragraphs; each must carry one "
                "distinct message."
            )
        # Slice 18 — H22 quantitative comparison table (mandatory for routes 1/2/5/6)
        if not audit["quant_table_ok"]:
            failures.append(
                f"H22 quant_table_ok=false: no \\begin{{tabular}} environment in the main body "
                f"contains ≥ 3 numeric tokens. An open-ended-discovery or modification paper "
                f"MUST present a side-by-side numeric comparison (baseline vs. candidates vs. "
                f"reference) — RMSE, reattachment x_r/h, drag coefficient, peak Cf, "
                f"whatever the QoI is. Embed at least one results table in Methods or Results."
            )
        # Slice 18 — H16 tighter: figure_jobs[i] must carry out_png AND
        # Slice 18.1 — out_png must be unique, exist on disk, and per-case kinds
        # must live in paper_figs/ with case_id in the filename.
        if audit["paper_unified_plan_present"] and not audit["figure_jobs_have_out_png"]:
            reasons = audit.get("figure_jobs_invalid_reasons", []) or [
                "schema check failed (no specific reasons surfaced)"
            ]
            failures.append(
                f"H16 figure_jobs schema invalid: "
                f"{audit['figure_jobs_with_out_png']}/{audit['figure_jobs_total']} jobs passed. "
                f"Each figure_jobs[i] must have: (a) a unique out_png string, (b) the file "
                f"must already exist on disk, (c) for per-case kinds (u_contour, p_contour, "
                f"cf_profile, …) the out_png must live under paper_figs/ AND contain the "
                f"case_id in its filename. Specific violations:\n  - "
                + "\n  - ".join(reasons)
            )
        # Slice 18 — H15 tighter: includegraphics must cover every figure_jobs[i].out_png
        if (audit["paper_unified_plan_present"]
                and audit["figure_jobs_have_out_png"]
                and not audit["figure_jobs_includegraphics_ok"]):
            failures.append(
                f"H15 figure_jobs_includegraphics_ok=false: only "
                f"{audit['figure_jobs_covered_by_includegraphics']}/{audit['figure_jobs_total']} "
                f"planned figures are embedded via \\includegraphics in main.tex. "
                f"The writer skipped per-case contours despite the planner asking for them. "
                f"Re-run Phase D with the literal rule: emit one \\includegraphics per "
                f"figure_jobs[i].out_png."
            )
        # Slice 18.2 — H23 citation distribution
        if not audit["citation_distribution_ok"]:
            failures.append(
                f"H23 citation_distribution_ok=false: \\cite{{}} calls are not distributed "
                f"across the body. Per-section counts: {audit['citation_distribution']}. "
                f"Required: ≥ 1 \\cite{{}} in EACH of Introduction, Related Work, Methods, "
                f"Discussion; no single section may hold > 60% of all citations. The BFS-v2 "
                f"trash pattern was dumping all 10 \\cite{{}} calls at the end of a single "
                f"paragraph — that is not citing, it is listing. Weave citations into the "
                f"sentence where each claim is made."
            )
        # Slice 18.2 — H24 meta-paragraph cross-references: DEPRECATED in slice 19.
        # The cfd-paper-writer skill explicitly forbids meta-paragraph references
        # ("never write 'in introduction paragraph 1', 'see paragraph 7'") in
        # the Related Work and structural-discipline sections. The reviewer
        # flags these via prose_quality_issues[type="meta_paragraph_ref"].
        # Keeping the field computed for diagnostics but not blocking.
        if False and not audit["meta_paragraph_refs_ok"]:
            samples = audit["meta_paragraph_refs"][:5]
            failures.append(
                f"H24 meta_paragraph_refs_ok=false: paragraphs reference each other by "
                f"number. Samples: {samples}"
            )
        # Slice 18.2 — H25 governing & closure equations (content, not count)
        if not audit["governing_equations_ok"]:
            failures.append(
                f"H25 governing_equations_ok=false: no display-math block contains "
                f"continuity / RANS-momentum / Navier–Stokes markers. The 3 equations the "
                f"writer DID emit (Re definition, Cf definition, RMSE definition) are NOT "
                f"governing equations — those are just metric formulas. Add the actual "
                f"continuity + momentum equations that the solver is integrating, with "
                f"correct Reynolds-averaged notation. Markers found: "
                f"{audit['equation_markers_found']}"
            )
        if not audit["closure_equations_ok"]:
            failures.append(
                f"H25 closure_equations_ok=false: no display-math block contains a "
                f"turbulence-closure equation (k-ω, k-ε, Spalart–Allmaras transport, "
                f"ν_t / μ_t, Bingham, Carreau, …). For a turbulence-model comparison "
                f"paper the closure equations ARE the paper. Show each model's transport "
                f"or constitutive equation explicitly. Markers found: "
                f"{audit['equation_markers_found']}"
            )
        # Slice 19 — H28 per-case figure coverage from manifest.json
        if not audit["per_case_coverage_ok"]:
            missing = audit["per_case_coverage_missing"]
            failures.append(
                f"H28 per_case_coverage_ok=false: {len(missing)} case(s) in "
                f"manifest.json have no per-case u_contour + p_contour figure_jobs. "
                f"For routes 1/2/6 every accepted case must show its U and p fields, "
                f"not just the baseline + winner. The planner cherry-picked instead "
                f"of covering all cases. Re-run Phase A so every manifest case gets "
                f"≥1 u_contour AND ≥1 p_contour figure_job. Missing: "
                f"{missing[:6]}{'…' if len(missing) > 6 else ''}"
            )
        # Slice 19 — H30 cross-experiment plot quality + script integrity
        cps = audit.get("cross_plots_status", "pass" if audit["cross_plots_size_ok"] else "fail")
        if cps == "fail":
            under = audit["cross_plots_undersized"]
            failures.append(
                f"H30 cross_plots_size_ok=false: {len(under)} cross_experiment_analysis/ "
                f"PNG(s) are < {int(50_000*(1-TOLERANCE))} bytes (50 KB - {int(TOLERANCE*100)}%). "
                f"Default matplotlib styling (grayscale, no labels, no annotations, no DNS "
                f"reference overlay) produces 20-30 KB PNGs at 1600×900 — journal-quality "
                f"plots with the rcParams template in cfd-cross-analyze are 60-200 KB. "
                f"Re-run the cross-analysis script-author step using the journal-quality "
                f"matplotlib styling block, colored bars with value annotations, and a DNS "
                f"reference overlay on Cf plots. Undersized: {under[:6]}{'…' if len(under) > 6 else ''}"
            )
        elif cps == "warn":
            warned = audit.get("cross_plots_warn", [])
            warnings.append(
                f"H30 {len(warned)} cross-plot PNG(s) in tolerance band "
                f"(floor 50000, within -{int(TOLERANCE*100)}%) — passes"
            )
        css = audit.get("cross_script_status", "pass" if audit["cross_script_ok"] else "fail")
        if css == "fail":
            failures.append(
                f"H30 cross_script_ok=false: cross_experiment_analysis/scripts/ either "
                f"missing or contains only a comment-stub script "
                f"(largest .py is {audit['cross_script_bytes']} bytes; floors: "
                f"≥ {int(2_000*(1-TOLERANCE))} bytes AND ≥ {int(30*(1-TOLERANCE))} non-comment lines). "
                f"The actual Python that authored the cross-plots MUST be saved for "
                f"reproducibility. Replace any 'inline-authored' stub with the real script."
            )
        elif css == "warn":
            warnings.append(
                f"H30 cross_script in tolerance band: {audit['cross_script_bytes']} bytes "
                f"(floor 2000, within -{int(TOLERANCE*100)}%) — passes"
            )
        # Slice 19 — H29 per-case PNG size threshold
        pcs = audit.get("per_case_png_status", "pass" if audit["per_case_png_size_ok"] else "fail")
        if pcs == "fail":
            under = audit["per_case_png_undersized"]
            failures.append(
                f"H29 per_case_png_size_ok=false: {len(under)} per-case contour "
                f"PNG(s) are < {int(80_000*(1-TOLERANCE))} bytes (80 KB - {int(TOLERANCE*100)}%). "
                f"Real PyVista OpenFOAM field renders at 1200×600+ resolution weigh "
                f"200-500 KB; matplotlib placeholders with sparse data weigh < 50 KB. The "
                f"agent likely fell back to matplotlib synthetic-data plots instead of "
                f"loading the OpenFOAM case via PyVista. Re-run Phase B using xvfb-run + "
                f"PyVista's POpenFOAMReader on each case's .foam file (or write it first "
                f"if absent), select the latest time, and render U-magnitude and p "
                f"contours with proper colorbars. Undersized: "
                f"{under[:6]}{'…' if len(under) > 6 else ''}"
            )
        elif pcs == "warn":
            warned = audit.get("per_case_png_warn", [])
            warnings.append(
                f"H29 {len(warned)} per-case PNG(s) in tolerance band "
                f"(floor 80000, within -{int(TOLERANCE*100)}%) — passes"
            )
        # Slice 19.2 — H32 mesh-gate evidence in Methods
        if not audit["mesh_gate_in_paper_ok"]:
            failures.append(
                f"H32 mesh_gate_in_paper_ok=false: the paper's Methods section does "
                f"not contain mesh-gate evidence. Found markers: "
                f"{audit['mesh_gate_paper_evidence']} (need ≥ 2). Every CFD paper "
                f"must state how the mesh was chosen: either (a) a mesh-independence "
                f"subsection with the baseline vs refined comparison (cell counts, "
                f"monitored QoIs, selection criterion), OR (b) an explicit "
                f"'starter mesh was locked for the full study' paragraph naming the "
                f"starter and its validation source. Without this paragraph a "
                f"reviewer cannot assess the result — first reviewer question is "
                f"always about mesh sensitivity."
            )
        # Slice 18.2 — H26 geometry description
        if not audit["geometry_description_ok"]:
            failures.append(
                f"H26 geometry_description_ok=false: the Methods section does not describe "
                f"the geometry. Markers found: {audit['geometry_markers_found']} (need ≥ 3 "
                f"from: step height / h=, expansion ratio, channel/hill height, domain "
                f"length, inlet/outlet location, Re_h, x/h, y/h, blockMesh, BCs). A reader "
                f"who has never seen the case must learn the geometry from this paper alone."
            )
        # Slice 13 — H13 figure physics correctness (LLM-authored, must be present)
        if audit["figure_physics_ok"] is None:
            failures.append(
                "H13 figure_physics_ok missing from review.json: the reviewer LLM did not "
                "perform the per-figure physics-correctness vision pass (sign convention vs "
                "reference data, axis scale, expected physical features). See cfd-paper "
                "Step 7b for the required vision-LLM check."
            )
        elif audit["figure_physics_ok"] is False:
            failures.append(
                "H13 figure_physics_ok=false: the reviewer flagged a physical inconsistency "
                "in at least one comparison figure (sign convention mismatch with reference, "
                "axis-scale error, or missing expected feature). Fix the underlying figure "
                "(typically: negate the OpenFOAM signed-Cf to match DNS convention) and re-run."
            )

    review = out_dir / "review.json"
    if not review.is_file():
        failures.append("review.json missing (paper review never ran)")
    else:
        rd = _read_json(review)
        # Cross-check reviewer's claimed booleans against deterministic audit.
        # If they disagree, the reviewer is lying — flag it.
        for key in ("length_ok", "word_density_ok", "cross_experiment_figures_ok"):
            reviewer_claim = rd.get(key)
            ground_truth = audit.get(key)
            if reviewer_claim is True and ground_truth is False:
                failures.append(
                    f"review.json#{key}=true contradicts deterministic audit (={ground_truth}); "
                    f"reviewer LLM is fabricating the booleans"
                )
        score = rd.get("score", 0)
        thr = rd.get("score_threshold", 0.7)
        if isinstance(score, (int, float)) and isinstance(thr, (int, float)):
            if score < thr:
                failures.append(f"review.json#score = {score} < threshold {thr}")
        if rd.get("pass") is not True:
            failures.append("review.json#pass != true (paper not approved by reviewer)")

    # Slice 20 — item 15: cross-artifact numeric consistency.
    nc_failures, nc_warnings = _check_paper_numeric_consistency(out_dir, route)
    failures.extend(nc_failures)
    warnings.extend(nc_warnings)

    return failures, warnings


def _check_paper_numeric_consistency(out_dir: Path,
                                     route: int) -> Tuple[List[str], List[str]]:
    """Item 15 — cross-artifact numeric consistency for the paper.

      H34 (hard): a physical reference quantity (Reynolds number, reattachment
        x_r/h) is stated in main.tex with two materially-different values
        (> 5 % spread) — a single study must use one reference value.
      H35 (warning): a headline percentage claim in main.tex is not
        reconcilable with any number in analysis.json / bridge.json /
        manifest.json / the cross-experiment aggregate — verify it traces to a
        computed metric rather than an invented figure.
    """
    failures: List[str] = []
    warnings: List[str] = []
    tex = out_dir / "paper" / "main.tex"
    if not tex.is_file():
        return failures, warnings
    body = re.split(r"\\appendix", tex.read_text(errors="replace"), maxsplit=1)[0]

    def _num(tok: str) -> Optional[float]:
        t = tok.replace(",", "").replace("{", "").replace("}", "").strip()
        m = re.match(r"([0-9]*\.?[0-9]+)\s*(?:\\times|\\cdot|x|\*)\s*10\s*\^?\s*"
                     r"\{?(-?\d+)\}?", t)
        if m:
            try:
                return float(m.group(1)) * (10.0 ** int(m.group(2)))
            except Exception:
                return None
        try:
            return float(t)
        except Exception:
            return None

    # ---- H34: reference-value internal consistency ----
    num_tok = (r"([0-9][0-9,\.]*(?:\s*(?:\\times|\\cdot|x|\*)\s*10\s*\^?\s*"
               r"\{?-?\d+\}?)?)")
    assign = r"\s*(?:=|\\approx|\\simeq|\\sim|≈)\s*\$?\s*"
    refpats = {
        "Reynolds number": r"Re(?:[_^]?\{?\\?(?:h|tau|theta|D|x|L|c|b)?\}?)?",
        "reattachment x_r/h": r"x_?\{?r\}?\s*/\s*h",
    }
    for label, sym in refpats.items():
        vals: List[float] = []
        for m in re.finditer(sym + assign + num_tok, body):
            v = _num(m.group(1))
            if v is not None and v > 0:
                vals.append(v)
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            if lo > 0 and (hi - lo) / lo > 0.05:
                failures.append(
                    f"H34 main.tex states the {label} with inconsistent "
                    f"values {sorted(set(round(v, 4) for v in vals))} "
                    f"(spread {100 * (hi - lo) / lo:.0f}% > 5%). A single "
                    f"study must use one reference value."
                )

    # ---- H35: headline % claims vs computed artifacts ----
    artifact_numbers: List[float] = []
    for name in ("analysis.json", "bridge.json", "manifest.json"):
        p = out_dir / name
        if p.is_file():
            for m in re.finditer(r"-?\d+\.?\d*", p.read_text(errors="replace")):
                v = _num(m.group(0))
                if v is not None:
                    artifact_numbers.append(abs(v))
    agg = out_dir / "cross_experiment_analysis" / "aggregate.csv"
    if agg.is_file():
        for m in re.finditer(r"-?\d+\.?\d*", agg.read_text(errors="replace")):
            v = _num(m.group(0))
            if v is not None:
                artifact_numbers.append(abs(v))
    if artifact_numbers:
        claims = set()
        for m in re.finditer(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*\\?%", body):
            v = _num(m.group(1))
            if v is not None and 1.0 <= v <= 99.0:
                claims.add(round(v, 2))
        for c in sorted(claims):
            ok = any(
                abs(c - a) <= 0.5 or abs(c - a * 100.0) <= 0.5
                for a in artifact_numbers
            )
            if not ok:
                warnings.append(
                    f"H35 main.tex states a {c}% claim with no matching "
                    f"number in analysis.json / bridge.json / manifest.json / "
                    f"the cross-experiment aggregate — verify it traces to a "
                    f"computed metric, not an invented figure."
                )
    return failures, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Skill-mode stage-gate audit.")
    parser.add_argument("--out-dir", required=True, help="run directory")
    parser.add_argument("--mode",
                        choices=["all", "paper", "upstream-for-paper", "preflight"],
                        default="all",
                        help="all: full route audit. paper: only deterministic paper-stage truth table "
                        "(used by cfd-paper reviewer to override LLM booleans). upstream-for-paper: only "
                        "the prerequisites cfd-paper Stage 0 needs (literature/hypothesis/requirements/"
                        "metric_setup/baseline_setup/code_mod/mesh_gate/experiments/analysis + per-case figs). "
                        "preflight (slice 14): verify every REQUIRED predecessor checkpoint for "
                        "--target-stage is on disk before that stage starts.")
    parser.add_argument("--target-stage",
                        choices=sorted(PREFLIGHT_STAGE_NAMES.keys()),
                        help="for --mode preflight: the stage that is about to start. "
                        "The script checks that all earlier REQUIRED checkpoints exist.")
    parser.add_argument("--json", action="store_true", help="emit JSON report on stdout")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    state_path = out_dir / "state.json"
    if not state_path.is_file():
        print(f"ERROR: {state_path} missing", file=sys.stderr)
        return 6

    state = _read_json(state_path)
    mode = state.get("mode") or ""
    if state.get("open_discovery"):
        mode = "open_discovery"
    if mode not in MODE_TO_ROUTE:
        print(f"ERROR: unknown mode {mode!r} in {state_path}", file=sys.stderr)
        return 6
    route = MODE_TO_ROUTE[mode]

    if args.mode == "preflight":
        if not args.target_stage:
            print("ERROR: --mode preflight requires --target-stage <name>", file=sys.stderr)
            return 6
        failures = _check_preflight(out_dir, route, args.target_stage)
        rc = 0 if not failures else 2
        report = {
            "out_dir": str(out_dir),
            "mode": mode,
            "route": route,
            "audit_mode": "preflight",
            "target_stage": args.target_stage,
            "rc": rc,
            "n_failures": len(failures),
            "failures": failures,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            if rc == 0:
                print(f"PASS — preflight OK: '{args.target_stage}' can start "
                      f"(route {route}, {mode}).")
            else:
                print(f"FAIL — preflight blocked '{args.target_stage}' (route {route}, "
                      f"{mode}): {len(failures)} predecessor(s) missing:",
                      file=sys.stderr)
                for f in failures:
                    print(f"  - {f}", file=sys.stderr)
        return rc

    if args.mode == "paper":
        # Deterministic paper-stage truth table only. Used by cfd-paper reviewer step
        # to override fabricated LLM booleans. Always writes paper_audit.json.
        audit = _compute_paper_audit(out_dir, route=route)
        try:
            (out_dir / "paper_audit.json").write_text(json.dumps(audit, indent=2))
        except Exception as e:
            print(f"WARN: failed to write paper_audit.json: {e}", file=sys.stderr)
        all_pass = (
            audit["length_ok"]
            and audit["word_density_ok"]
            and audit["cross_experiment_figures_ok"]
            and audit["citations_ok"]
            and audit["lit_retrieval_ok"]
            and audit["refs_bib_ok"]
            and audit["equations_ok"]
            # H21 (duplicate_paragraphs_ok) and H24 (meta_paragraph_refs_ok)
            # deprecated in slice 19: cfd-paper-writer skill prevents these
            # structurally, reviewer flags them via prose_quality_issues.
            and audit["quant_table_ok"]
            and audit["figure_jobs_have_out_png"]
            and audit["figure_jobs_includegraphics_ok"]
            and audit["citation_distribution_ok"]
            and audit["governing_equations_ok"]
            and audit["closure_equations_ok"]
            and audit["geometry_description_ok"]
            and audit["mesh_gate_in_paper_ok"]
            and audit["per_case_coverage_ok"]
            and audit["per_case_png_size_ok"]
            and audit["cross_plots_size_ok"]
            and audit["cross_script_ok"]
        )
        rc = 0 if all_pass else 5
        if args.json:
            print(json.dumps({"rc": rc, "audit": audit}, indent=2))
        else:
            status = "PASS" if rc == 0 else "FAIL"
            print(f"{status} paper-mode audit:  pages={audit['main_body_pages']}  "
                  f"words/page={audit['words_per_page']}  "
                  f"length_ok={audit['length_ok']}  "
                  f"word_density_ok={audit['word_density_ok']}  "
                  f"cross_experiment_figures_ok={audit['cross_experiment_figures_ok']}",
                  file=(sys.stdout if rc == 0 else sys.stderr))
        return rc

    cp_failures: List[str] = []
    ord_failures: List[str] = []
    case_failures: List[str] = []
    paper_failures: List[str] = []
    paper_warnings: List[str] = []
    lit_failures: List[str] = []
    bridge_failures: List[str] = []

    if args.mode == "upstream-for-paper":
        # Only the prerequisites cfd-paper Stage 0 needs. Excludes paper_review_done /
        # cross_experiment_analysis_done / paper_experiment_plan_done / finish_done.
        upstream_required = [
            cp for cp in REQUIRED[route]
            if cp not in {"paper_review_done", "cross_experiment_analysis_done",
                          "paper_experiment_plan_done", "finish_done"}
        ]
        cp_dir = out_dir / "checkpoints"
        for stage in upstream_required:
            if not (cp_dir / f"{stage}.json").is_file():
                cp_failures.append(f"missing required upstream checkpoint: {stage}")
        ord_failures = _check_ordering(out_dir, route)
        case_failures = _check_cases(out_dir, route)
        lit_failures  = _check_literature_provenance(out_dir)
        bridge_failures = _check_open_discovery_bridge(out_dir, route)
    else:  # mode == "all"
        cp_failures = _check_checkpoints(out_dir, route)
        ord_failures = _check_ordering(out_dir, route)
        case_failures = _check_cases(out_dir, route)
        case_failures += _check_solver_logs(out_dir, route)        # item 1
        paper_failures, paper_warnings = _check_paper_stage(out_dir, route)
        bridge_failures = _check_open_discovery_bridge(out_dir, route)
        bridge_failures += _check_mesh_gate_integrity(out_dir, route)
        bridge_failures += _check_oed_bookkeeping(out_dir, route)   # items 12/13
        # Only enforce literature provenance on routes that require it.
        if "literature_done" in REQUIRED[route]:
            lit_failures = _check_literature_provenance(out_dir)
        # Item 3 — reconcile a stale state.json against the checkpoints on disk
        # before reporting. The audit is the authority on progress.
        _state_note = _reconcile_state_json(out_dir, route, state)
        if _state_note:
            paper_warnings = paper_warnings + [f"state.json: {_state_note}"]

    all_failures = cp_failures + ord_failures + case_failures + paper_failures + lit_failures + bridge_failures
    rc = 0
    if cp_failures:        rc = 2
    elif bridge_failures:  rc = 4  # bridge missing == per-stage gate
    elif ord_failures:     rc = 3
    elif case_failures:    rc = 4
    elif paper_failures:   rc = 5
    elif lit_failures:     rc = 2  # literature provenance counts as a missing-precondition

    report = {
        "out_dir": str(out_dir),
        "mode": mode,
        "route": route,
        "audit_mode": args.mode,
        "rc": rc,
        "n_failures": len(all_failures),
        "failures": all_failures,
        "n_warnings": len(paper_warnings),
        "warnings": paper_warnings,
        "tolerance": TOLERANCE,
        "checkpoint_failures": cp_failures,
        "ordering_failures": ord_failures,
        "case_failures": case_failures,
        "paper_failures": paper_failures,
        "paper_warnings": paper_warnings,
        "literature_failures": lit_failures,
        "bridge_failures": bridge_failures,
    }

    # On rc=0 in --mode=all, write the authoritative audit_passed.json. This is the
    # signal downstream tooling (cfd-paper Stage 0 in upstream-for-paper mode, manual
    # post-run verification) trusts — NOT state.json, which the agent can fabricate.
    if rc == 0 and args.mode == "all":
        # Tag with a signature key the agent doesn't know to forge.
        report["audit_signature"] = "stage_gate_audit.py:v1"
        try:
            (out_dir / "audit_passed.json").write_text(json.dumps(report, indent=2))
        except Exception:
            pass
        # Item 3 — a fully-passing audit is the only authority that may mark
        # state.json done. state.status / current_stage are otherwise advisory.
        try:
            state["status"] = "done"
            state["current_stage"] = "finish"
            (out_dir / "state.json").write_text(json.dumps(state, indent=2))
        except Exception:
            pass
    elif args.mode == "all":
        # Remove a stale audit_passed.json if it exists — either from a previous successful
        # run whose preconditions were undone, OR fabricated by the agent.
        try:
            (out_dir / "audit_passed.json").unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if rc == 0:
            print(f"PASS — all stage gates satisfied for route {route} ({mode}, mode={args.mode})")
        else:
            print(f"FAIL — {len(all_failures)} gate(s) failed (route {route}, {mode}, mode={args.mode}):",
                  file=sys.stderr)
            for f in all_failures:
                print(f"  - {f}", file=sys.stderr)
        if paper_warnings:
            stream = sys.stdout if rc == 0 else sys.stderr
            print(f"WARN — {len(paper_warnings)} gate(s) in ±{int(TOLERANCE*100)}% tolerance band "
                  f"(passing but flagged):", file=stream)
            for w in paper_warnings:
                print(f"  ~ {w}", file=stream)
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""The paper figures' contract: every figure job has an id, a kind and an out_png.

The final audit (scripts/stage_gate_audit.py, H15/H16/H28) reads
paper_unified_plan.json#figure_jobs and requires of every job a unique
`out_png` that exists on disk and is embedded by \\includegraphics, per-case
kinds under paper_figs/ with the case_id in the file name, and one u_contour
and one p_contour job per successful case. The planner used to emit only
{case_id, what_to_visualize, priority}, and the batch script named its PNGs
freely, so none of those checks could pass however good the figures were
(shelf_demo_dafoam_20261019_1057, shelf_demo_hystrath_20261022_1030). This
module fixes the names before the figure script runs, so the script, the
writer and the audit all work from one list.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Per-case kinds. The first six are stage_gate_audit.PER_CASE_KINDS (the audit
# holds those to paper_figs/<...case_id...>.png); scripts/test_paper_figure_contract.py
# checks the two lists agree.
AUDIT_PER_CASE_KINDS = (
    "u_contour",
    "p_contour",
    "velocity_magnitude_contour",
    "cf_profile",
    "pressure_profile",
    "streamlines",
)
# Other per-case figures a study asks for (a temperature or density contour, a
# stagnation-line profile, a surface distribution). Named the same way.
EXTRA_PER_CASE_KINDS = ("field_contour", "line_profile", "surface_distribution")
PER_CASE_KINDS = AUDIT_PER_CASE_KINDS + EXTRA_PER_CASE_KINDS
# Figures drawn from several cases at once.
CROSS_CASE_KINDS = ("case_comparison",)
FIGURE_KINDS = PER_CASE_KINDS + CROSS_CASE_KINDS

REQUIRED_PER_CASE_KINDS = ("u_contour", "p_contour")

# Names a planner writes for the kinds above.
_KIND_ALIASES = {
    "velocity_contour": "u_contour",
    "ux_contour": "u_contour",
    "u_field": "u_contour",
    "pressure_contour": "p_contour",
    "p_field": "p_contour",
    "umag_contour": "velocity_magnitude_contour",
    "velocity_magnitude": "velocity_magnitude_contour",
    "streamline": "streamlines",
    "skin_friction_profile": "cf_profile",
    "wall_pressure_profile": "pressure_profile",
    "contour": "field_contour",
    "profile": "line_profile",
    "comparison": "case_comparison",
    "overview": "case_comparison",
    "multi_case": "case_comparison",
}


# --------------------------------------------------------------------------
# Which velocity and pressure fields a case actually wrote
# --------------------------------------------------------------------------

def _as_time(name: str) -> Optional[float]:
    try:
        return float(name)
    except ValueError:
        return None


def _time_dirs(base: Path) -> List[Path]:
    try:
        dirs = [p for p in base.iterdir() if p.is_dir() and _as_time(p.name) is not None]
    except OSError:
        return []
    return sorted(dirs, key=lambda p: _as_time(p.name) or 0.0)


def _field_class(path: Path) -> Optional[str]:
    """The FoamFile header's class (volVectorField, ...), or None if unreadable.
    The header is ASCII even in binary-format fields."""
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as fh:
            head = fh.read(4096).decode("latin-1", errors="replace")
    except OSError:
        return None
    for line in head.splitlines():
        words = line.strip().split()
        if len(words) >= 2 and words[0] == "class":
            return words[1].rstrip(";")
    return None


def latest_written_fields(case_dir: Path) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """(time folder name, {field name: header class}) of the latest written
    solution time, at the case root or, for an unreconstructed parallel run,
    in processor0. Falls back to the initial conditions (time 0) when no later
    time was written."""
    case_dir = Path(case_dir)
    chosen: Optional[Path] = None
    for base in (case_dir, case_dir / "processor0"):
        later = [t for t in _time_dirs(base) if (_as_time(t.name) or 0.0) > 0.0]
        if later:
            chosen = later[-1]
            break
    if chosen is None:
        zero = [t for t in _time_dirs(case_dir) if _as_time(t.name) == 0.0]
        chosen = zero[0] if zero else None
    if chosen is None:
        return None, {}
    fields: Dict[str, Optional[str]] = {}
    try:
        entries = sorted(chosen.iterdir(), key=lambda p: p.name)
    except OSError:
        entries = []
    for p in entries:
        if not p.is_file() or p.name.startswith("."):
            continue
        name = p.name[:-3] if p.name.endswith(".gz") else p.name
        if name.endswith(".orig"):
            continue
        fields[name] = _field_class(p)
    return chosen.name, fields


def _first(names: Iterable[str]) -> Optional[str]:
    for n in names:
        return n
    return None


def pick_velocity_field(fields: Dict[str, Optional[str]]) -> Optional[str]:
    """U, else a species/phase velocity such as U_Ar (hyStrath's dsmcFoam+
    writes no U), else any other volVectorField whose name starts with U."""
    def vec(n: str) -> bool:
        return fields.get(n) in (None, "volVectorField")
    if "U" in fields and vec("U"):
        return "U"
    names = sorted(fields)
    hit = _first(n for n in names if n.startswith("U_") and vec(n))
    if hit:
        return hit
    return _first(n for n in names if n.startswith("U") and fields.get(n) == "volVectorField")


def pick_pressure_field(fields: Dict[str, Optional[str]]) -> Tuple[Optional[str], str]:
    """(field, note). p, else p_rgh, else a species pressure such as p_Ar, else
    another volScalarField whose name starts with p. With no pressure at all a
    density field stands in, and the note says so, so no caption calls it a
    pressure."""
    def sca(n: str) -> bool:
        return fields.get(n) in (None, "volScalarField")
    for n in ("p", "p_rgh"):
        if n in fields and sca(n):
            return n, ""
    names = sorted(fields)
    hit = _first(n for n in names if n.startswith("p_") and sca(n))
    if hit:
        return hit, ""
    hit = _first(n for n in names
                 if n.startswith("p") and n != "phi" and fields.get(n) == "volScalarField")
    if hit:
        return hit, ""
    for prefix in ("rho", "rhoM", "rhoN"):
        hit = _first(n for n in names if n.startswith(prefix) and sca(n))
        if hit:
            return hit, (f"the case wrote no pressure field; the pressure-like panel shows "
                         f"the density field {hit}, and its caption must say so")
    return None, "no pressure-like field found in the latest written time"


def case_field_summary(case_dir: Path) -> Dict[str, Any]:
    time_name, fields = latest_written_fields(case_dir)
    p_field, p_note = pick_pressure_field(fields)
    return {
        "time": time_name,
        "solution_written": bool(time_name) and (_as_time(time_name) or 0.0) > 0.0,
        "velocity_field": pick_velocity_field(fields),
        "pressure_field": p_field,
        "pressure_note": p_note,
        "fields": sorted(fields)[:60],
    }


# --------------------------------------------------------------------------
# Normalising the planner's figure jobs
# --------------------------------------------------------------------------

def normalize_kind(kind: Any, has_case: bool) -> str:
    k = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    k = _KIND_ALIASES.get(k, k)
    if k in PER_CASE_KINDS:
        return k if has_case else "case_comparison"
    if k in CROSS_CASE_KINDS:
        return k
    return "field_contour" if has_case else "case_comparison"


def _slug(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if (ch.isalnum() or ch in "_-") else "_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _default_what(kind: str, field: Optional[str], note: str) -> str:
    if kind == "u_contour":
        what = (f"Contour of the velocity field {field or 'U'} (magnitude or streamwise component, "
                f"say which) on the case's main plane at the latest written time, with a labelled colorbar.")
    else:
        what = (f"Contour of the pressure-like field {field or 'p'} on the case's main plane at the "
                f"latest written time, with a labelled colorbar.")
    return what + (f" Note: {note}." if note else "")


def normalize_figure_jobs(
    plan: Dict[str, Any],
    manifest_cases: List[Dict[str, Any]],
    fields_by_case: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Give every figure job an id, a kind from FIGURE_KINDS, a field for
    velocity/pressure contours and a unique out_png; add a u_contour and a
    p_contour job for each successful, included case the planner left without
    one. Returns the plan (modified in place); what was added or dropped is
    under plan['figure_contract']."""
    status = {str(c.get("case_id", "")): str(c.get("status", "")).lower()
              for c in manifest_cases if isinstance(c, dict) and c.get("case_id")}
    omit = {str(x) for x in (plan.get("omit_case_ids") or [])}
    raw_jobs = plan.get("figure_jobs") if isinstance(plan.get("figure_jobs"), list) else []

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    for j in raw_jobs:
        if not isinstance(j, dict):
            continue
        cid = str(j.get("case_id") or "").strip()
        if cid and cid not in status:
            # a case id the manifest does not know cannot be rendered or named
            dropped.append({"case_id": cid, "reason": "case_id not in manifest.json"})
            continue
        if cid and cid in omit:
            dropped.append({"case_id": cid, "reason": "case omitted by the plan"})
            continue
        summary = fields_by_case.get(cid, {}) if cid else {}
        if cid and status.get(cid) != "success" and not summary.get("solution_written"):
            dropped.append({"case_id": cid, "reason": "failed case with no written solution to plot"})
            continue
        job = dict(j)
        job["case_id"] = cid or None
        job["kind"] = normalize_kind(j.get("kind"), has_case=bool(cid))
        if job["kind"] == "u_contour" and summary.get("velocity_field"):
            if job.get("field") not in (summary.get("fields") or []):
                job["field"] = summary["velocity_field"]
        if job["kind"] == "p_contour" and summary.get("pressure_field"):
            if job.get("field") not in (summary.get("fields") or []):
                job["field"] = summary["pressure_field"]
            if summary.get("pressure_note"):
                job["field_note"] = summary["pressure_note"]
        job.setdefault("what_to_visualize", "")
        kept.append(job)

    order = [c for c in (plan.get("case_order_for_paper") or []) if c in status]
    order += [c for c in status if c not in order]
    required = [c for c in order if status.get(c) == "success" and c not in omit]
    added: List[str] = []
    for cid in required:
        have = {j["kind"] for j in kept if j.get("case_id") == cid}
        summary = fields_by_case.get(cid, {})
        for kind in REQUIRED_PER_CASE_KINDS:
            if kind in have:
                continue
            field = summary.get("velocity_field" if kind == "u_contour" else "pressure_field")
            note = summary.get("pressure_note", "") if kind == "p_contour" else ""
            job = {
                "case_id": cid,
                "kind": kind,
                "field": field,
                "what_to_visualize": _default_what(kind, field, note),
                "priority": 99,
            }
            if note:
                job["field_note"] = note
            kept.append(job)
            added.append(f"{cid}_{kind}")

    # ids and file names are the pipeline's, not the planner's: the audit needs
    # them unique and, for per-case kinds, carrying the case id.
    seen: set = set()
    n_cross = 0
    for j in kept:
        if j.get("case_id"):
            base = f"{j['case_id']}_{j['kind']}"
            if j["kind"] not in REQUIRED_PER_CASE_KINDS and j.get("field"):
                base += "_" + _slug(j["field"])
        else:
            n_cross += 1
            base = f"cross_{n_cross:02d}_{j['kind']}"
        # the case id is kept verbatim (it is a folder name): the audit looks
        # for it as a substring of the file name
        jid, n = base, 2
        while jid in seen:
            jid, n = f"{base}_{n}", n + 1
        seen.add(jid)
        j["id"] = jid
        j["out_png"] = f"paper_figs/{jid}.png"

    plan["figure_jobs"] = kept
    plan["figure_contract"] = {
        "kinds": list(FIGURE_KINDS),
        "required_per_case_kinds": list(REQUIRED_PER_CASE_KINDS),
        "required_cases": required,
        "added_jobs": added,
        "dropped_jobs": dropped,
        "fields_by_case": fields_by_case,
    }
    return plan


def figure_jobs_brief(jobs: List[Dict[str, Any]], run_dir: Path) -> str:
    """The list of PNGs the batch script must write, at their exact paths."""
    lines = [
        "REQUIRED OUTPUT FILES (mandatory): save each figure below at EXACTLY the absolute "
        "path given (also in cases_config.json -> figure_jobs[i].out_png_abs). Do not rename them. "
        "Extra PNGs are allowed, but every listed file must be written.",
    ]
    for j in jobs:
        if not isinstance(j, dict):
            continue
        path = Path(run_dir) / str(j.get("out_png", ""))
        bits = [f"- {path}", f"kind={j.get('kind')}"]
        if j.get("case_id"):
            bits.append(f"case={j['case_id']}")
        if j.get("field"):
            bits.append(f"field={j['field']}")
        lines.append("  ".join(bits))
        what = str(j.get("what_to_visualize") or "").strip()
        if what:
            lines.append(f"    {what[:600]}")
        if j.get("field_note"):
            lines.append(f"    NOTE: {j['field_note']}")
    return "\n".join(lines)


def split_rendered(jobs: List[Dict[str, Any]], run_dir: Path) -> Tuple[List[Path], List[str]]:
    """(absolute paths of out_png files that exist, out_png values that do not)."""
    have: List[Path] = []
    missing: List[str] = []
    for j in jobs:
        if not isinstance(j, dict) or not j.get("out_png"):
            continue
        p = (Path(run_dir) / str(j["out_png"])).resolve()
        if p.is_file():
            have.append(p)
        else:
            missing.append(str(j["out_png"]))
    return have, missing


# --------------------------------------------------------------------------
# Which figures a LaTeX source embeds
# --------------------------------------------------------------------------

def includegraphics_targets(tex: str) -> List[str]:
    """The file argument of every \\includegraphics[...]{...} (also
    \\includegraphics*), read by scanning, not by pattern matching."""
    out: List[str] = []
    key = "\\includegraphics"
    i = 0
    n = len(tex)
    while True:
        i = tex.find(key, i)
        if i < 0:
            return out
        k = i + len(key)
        if k < n and tex[k] == "*":
            k += 1
        while k < n and tex[k].isspace():
            k += 1
        if k < n and tex[k] == "[":
            close = tex.find("]", k)
            if close < 0:
                return out
            k = close + 1
            while k < n and tex[k].isspace():
                k += 1
        if k < n and tex[k] == "{":
            close = tex.find("}", k)
            if close < 0:
                return out
            out.append(tex[k + 1:close].strip())
            i = close + 1
        else:
            i = k


def missing_embeds(tex: str, required: List[Path]) -> List[str]:
    """Required PNG paths that no \\includegraphics names. Matched on the file
    name, the way stage_gate_audit._includegraphics_covers_figure_jobs does."""
    names = [Path(t).name for t in includegraphics_targets(tex)]
    gone: List[str] = []
    for p in required:
        leaf = Path(p).name
        if not any(leaf in c or c == leaf for c in names):
            gone.append(str(p))
    return gone


def required_figures_block(required: List[Path]) -> str:
    if not required:
        return ""
    lines = [
        "REQUIRED FIGURES (mandatory): each file below must be embedded in the paper with its own "
        "\\includegraphics, using the exact path shown (with the .png extension), inside a figure "
        "environment with a caption and a \\label that the text references. None may be left out."
    ]
    lines.extend(f"- {p}" for p in required)
    return "\n".join(lines)

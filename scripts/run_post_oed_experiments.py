"""
Post-OED experiments runner.

Once the OED has discovered a winning code modification, we want to run a
small parametric sensitivity study around it for the paper. This script
does that GENERICALLY:

  1. Resolve the OED's "active artifact" (`scripts/oed_artifact.py`).
  2. Build a parametric plan over the artifact's actual coefficient names
     (auto-derived if no plan is supplied).
  3. For each plan entry: copy the artifact's working base case, patch the
     coefficient values inside the runtime dictionary, run simpleFoam via
     `foam_run_simple.py` (skipping FoamAgent's reviewer entirely), score
     against the bound comparator.
  4. Write a paper-ready summary.

Why bypass FoamAgent's reviewer
-------------------------------
FoamAgent's input-writer + reviewer regenerate Make/files / Make/options /
constant/momentumTransport from LLM understanding of the topic. For an
OED-discovered modification this loses information: the LLM doesn't know
the exact coded snippet text, the exact coefficient names, or whether the
artifact is a runtime source vs a class derivation. The reviewer tends to
hallucinate Make/options paths (e.g. `transportModels/incompressible/` —
which doesn't exist on OF foundation 10) and try to compile a class that
was never written. The simpler, more reliable path is to copy the working
case verbatim and only adjust the coefficient values.

Generic across:
  * runtime_source / runtime_bc / runtime_field — patches scalar entries
    inside the coded dictionary block.
  * class_derivation — copies customModels/ + the compiled .so into the
    case (no rebuild) and re-runs simpleFoam. Coefficients can be patched
    via constant/momentumTransport keyword overrides if the variant
    declared them as model coefficients.
  * dict_only — just copies the full case and reruns.

CLI:
    python scripts/run_post_oed_experiments.py --run-dir <run_dir>

Optional --plan <json> gives a custom parametric plan; otherwise the script
auto-generates a sensible 1-reference + sweep-around-best plan from the
artifact's coefficient names.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# Make oed_artifact importable
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _copy_case(src: Path, dst: Path) -> None:
    """Deep-copy an OpenFOAM case dir, skipping prior simpleFoam outputs."""
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(
        "log.*", "Allrun.out", "Allrun.err",
        "postProcessing", "figs", "interpreter_viz",
        # framework-internal comparator/score output dirs from prior runs.
        # If these are copied along, the new comparator finds the prior
        # case's score and short-circuits — every patched case ends up with
        # the same score as the source. Generic across topics.
        "_baseline_comparison", "_oed_comparison_bound",
        "comparison_exactmatch", "comparison_*",
        "open_ended_case_assessment.json", "case_assessment.json",
        "decision.json", "run_result.json", "_plan.json",
        "*.foam",
        # numeric time folders other than 0 (decimal start-time `0` and
        # `0.xxxxxx` are kept; integer 1, 10, 100, ... are dropped).
        *[f"{i}*" for i in range(1, 100000)],
    )
    shutil.copytree(src, dst, ignore=ignore, symlinks=True, dirs_exist_ok=False)


def _parse_coefficients_from_dict_block(text: str, dict_block_name: str) -> Dict[str, str]:
    """Extract `name value;` scalar entries from a named OpenFOAM dict
    block (e.g. `customSource { ... }`). Returns {name: value-as-string}.
    Skips C++ code blocks (codeInclude, codeAddSup, ...). Generic OpenFOAM
    text parser.
    """
    # Find the block by name. Look for `<name>\s*{` then balanced braces.
    m = re.search(rf"\b{re.escape(dict_block_name)}\s*\{{", text)
    if not m:
        return {}
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = text[start:i]

    # Strip C++ code blocks (delimited by #{ ... #})
    block_no_code = re.sub(r"#\{.*?#\}", "", block, flags=re.DOTALL)

    out: Dict[str, str] = {}
    for line_match in re.finditer(
        r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+(-?\d+\.?\d*(?:e[+-]?\d+)?)\s*;",
        block_no_code, flags=re.MULTILINE):
        name = line_match.group(1)
        val = line_match.group(2)
        if name.lower() in {
            "version", "format", "class", "location", "object",
            "selectionmode", "field", "type", "name",
        }:
            continue
        out[name] = val
    return out


def _patch_coefficients_in_dict_block(text: str, dict_block_name: str,
                                      overrides: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Replace `name oldval;` with `name newval;` inside the named block.
    Returns (new_text, list_of_keys_actually_patched). Generic — works on
    any OpenFOAM dict block with `key value;` scalar entries.
    """
    if not overrides:
        return text, []
    m = re.search(rf"\b{re.escape(dict_block_name)}\s*\{{", text)
    if not m:
        return text, []
    block_start = m.end()
    depth = 1
    i = block_start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block_end = i  # index of the `}`
    head = text[:block_start]
    block = text[block_start:block_end]
    tail = text[block_end:]

    patched_keys: List[str] = []
    new_block = block
    for k, v in overrides.items():
        pat = re.compile(
            rf"(^\s+){re.escape(k)}(\s+)(-?\d+\.?\d*(?:e[+-]?\d+)?)(\s*;)",
            flags=re.MULTILINE,
        )
        # Avoid replacing inside C++ #{ ... #}: split block, patch only
        # outside the code blocks.
        parts = re.split(r"(#\{.*?#\})", new_block, flags=re.DOTALL)
        replaced_any = False
        for idx, part in enumerate(parts):
            if part.startswith("#{"):
                continue
            new_part, n = pat.subn(lambda mm: f"{mm.group(1)}{k}{mm.group(2)}{v}{mm.group(4)}",
                                   part)
            if n > 0:
                parts[idx] = new_part
                replaced_any = True
        new_block = "".join(parts)
        if replaced_any:
            patched_keys.append(k)
    return head + new_block + tail, patched_keys


def _find_runtime_dict_block_name(text: str) -> str:
    """Find the top-level dict-block name in a runtime fvModels file (the
    block that is NOT FoamFile)."""
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*\{", text, flags=re.MULTILINE):
        name = m.group(1)
        if name == "FoamFile":
            continue
        return name
    return ""


def auto_plan(coeff_names: List[str], best_values: Dict[str, str],
              *,
              parent_class_name: str = "",
              novel_coeff_names: Optional[List[str]] = None,
              n_per_coeff: int = 2) -> List[Dict[str, Any]]:
    """Build a default parametric plan: 1 reference + 1 best + sensitivity
    sweeps on novel coefficients. Generic across modification families.

    Parameters
    ----------
    coeff_names: all coefficient names available in the activation dict
        (includes inherited parent constants for class_derivation).
    best_values: current values from the artifact case (the OED winner).
    parent_class_name: when non-empty, the "reference" plan entry sets
        `use_parent_model=True` to swap the active model selection to the
        built-in parent (clean baseline-vs-variant comparison). When empty
        (e.g. runtime_source path), reference disables the term by zeroing
        non-regularization coefficients.
    novel_coeff_names: coefficients introduced by the subclass (those NOT
        inherited). When provided, sweeps run only over these, keeping the
        case count small. When empty, sweeps run over all coeff_names.
    """

    def _is_reg(name: str) -> bool:
        return bool(re.search(r"(eps|min|small|tol)$", name, flags=re.IGNORECASE))

    sweep_axes = list(novel_coeff_names) if novel_coeff_names else list(coeff_names)
    sweep_axes = [c for c in sweep_axes if not _is_reg(c)]

    plan: List[Dict[str, Any]] = []
    # 1. reference: parent class for class_derivation, else zeroed coeffs.
    if parent_class_name:
        plan.append({
            "name": "reference",
            "intent": (
                f"Reference baseline using the built-in parent class "
                f"{parent_class_name}. The active subclass is replaced by "
                f"its parent for a clean baseline-vs-variant comparison "
                f"on identical numerics and mesh."
            ),
            "overrides": {},
            "use_parent_model": True,
        })
    else:
        ref_overrides = {n: "0.0" for n in coeff_names if not _is_reg(n)}
        plan.append({
            "name": "reference",
            "intent": "Custom term disabled — clean baseline-vs-variant comparison on the same case.",
            "overrides": ref_overrides,
        })

    # 2. best-coefficients case (record what the OED found as best)
    plan.append({
        "name": "best",
        "intent": "OED's best-scoring coefficient set, re-run for paper figures.",
        "overrides": dict(best_values),
    })

    # 3. one-axis sweeps over the *novel* (subclass-introduced) coefficients.
    for k in sweep_axes:
        try:
            v0 = float(best_values.get(k, 0.0))
        except Exception:
            continue
        scales = [0.5, 2.0]
        for s in scales:
            label = "low" if s < 1.0 else "high"
            new_val = v0 * s
            plan.append({
                "name": f"{k}_{label}",
                "intent": f"Sensitivity sweep on {k} (×{s:g} relative to best).",
                "overrides": {k: f"{new_val:g}"},
            })
    return plan


def run_one_case(*, case_name: str, plan_entry: Dict[str, Any],
                 artifact: Dict[str, Any], cases_dir: Path,
                 timeout_s: int = 21600) -> Dict[str, Any]:
    """Seed a case from the artifact's base case, patch coefficients
    according to plan_entry.overrides, run via foam_run_simple.py, return
    a result dict with status + run paths + score (if comparator score
    is computable)."""
    case_dir = cases_dir / case_name
    src = Path(artifact["base_case_dir"])
    _copy_case(src, case_dir)

    # Apply overrides. Patcher target depends on category:
    #   runtime_source: edit a top-level coded block in fvModels (current
    #     `_find_runtime_dict_block_name` finds it).
    #   class_derivation: edit the <Class>Coeffs sub-block inside the
    #     activation dict (e.g. constant/momentumTransport.RAS.<X>Coeffs).
    # `coefficient_block_name` from the artifact tells us which block to
    # patch (set by the resolver). Generic across categories.
    overrides = plan_entry.get("overrides") or {}
    use_parent_model = bool(plan_entry.get("use_parent_model"))
    primary_dict = artifact.get("primary_dictionary") or ""
    patched_keys: List[str] = []
    skipped_keys: List[str] = []
    if primary_dict and (overrides or use_parent_model):
        rel = Path(primary_dict).relative_to(src)
        target = case_dir / rel
        if target.is_file():
            text = target.read_text(encoding="utf-8", errors="replace")
            block_name = (
                artifact.get("coefficient_block_name")
                or _find_runtime_dict_block_name(text)
            )
            if block_name and overrides:
                new_text, patched = _patch_coefficients_in_dict_block(text, block_name, overrides)
                if patched:
                    target.write_text(new_text, encoding="utf-8")
                    text = new_text
                    patched_keys = patched
                skipped_keys = [k for k in overrides.keys() if k not in patched]
            # Reference-case handling: swap the active model selection to
            # the parent class so this case is a clean built-in baseline.
            # We KEEP the libs() entry — having an unused .so loaded is
            # harmless and avoids an extra controlDict edit.
            if use_parent_model and artifact.get("parent_class_name"):
                parent = artifact["parent_class_name"]
                # Match `model <X>;` line and substitute, but only the FIRST
                # `model` line in the activation dict (the active selection).
                new_text, n = re.subn(
                    r"(\bmodel\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*;)",
                    rf"\g<1>{parent}\g<3>",
                    text,
                    count=1,
                )
                if n > 0:
                    target.write_text(new_text, encoding="utf-8")
                    patched_keys.append(f"model→{parent}")

    # Validation guard: re-read the patched file and confirm each
    # requested override now appears with the requested value. If a
    # patch silently no-op'd (e.g., key not in dict block), record it
    # as `verification_failed` so the case is treated as suspect rather
    # than a stealth-default-coefficients run. Generic across categories.
    verification_failed: List[str] = []
    if overrides and primary_dict:
        try:
            rel = Path(primary_dict).relative_to(src)
            target = case_dir / rel
            if target.is_file():
                final_text = target.read_text(encoding="utf-8", errors="replace")
                block_name = (
                    artifact.get("coefficient_block_name")
                    or _find_runtime_dict_block_name(final_text)
                )
                if block_name:
                    final_values = _parse_coefficients_from_dict_block(final_text, block_name)
                    for k, want in overrides.items():
                        try:
                            wf = float(str(want))
                            gf = float(str(final_values.get(k, "nan")))
                        except Exception:
                            verification_failed.append(k)
                            continue
                        if abs(wf - gf) > 1e-12 * (abs(wf) + 1.0):
                            verification_failed.append(k)
        except Exception:
            pass

    # Persist plan + provenance for the paper.
    _write_json(case_dir / "_plan.json", {
        "name": case_name,
        "plan": plan_entry,
        "patched_keys": patched_keys,
        "skipped_keys_not_in_dict": skipped_keys,
        "verification_failed_keys": verification_failed,
        "artifact_source_iteration": artifact.get("source_iteration"),
        "artifact_category": artifact.get("category"),
        "primary_dictionary": str(primary_dict),
    })

    # Run via foam_run_simple.py — no FoamAgent reviewer.
    run_result_path = case_dir / "run_result.json"
    cmd = [sys.executable, "scripts/foam_run_simple.py",
           "--base-case", str(case_dir),
           "--output-dir", str(case_dir),
           "--output", str(run_result_path),
           "--timeout", str(timeout_s)]
    print(f"[post-OED] {case_name}: running {' '.join(cmd[1:4])}", flush=True)
    try:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout_s + 60,
                            check=False).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    run_result = _read_json(run_result_path, {})
    run_ok = (rc == 0) and (str(run_result.get("status", "")).upper() == "OK")

    # Score via the bound comparator.
    score: Optional[Dict[str, Any]] = None
    score_reason = ""
    if run_ok:
        try:
            from open_ended_discovery import (  # type: ignore
                _run_bound_comparator,
                _extract_error_metrics,
                _choose_primary_score,
            )
            run_dir = Path(artifact["run_dir"])
            objective_contract = _read_json(
                run_dir / "open_ended_discovery" / "objective_contract.json",
                {},
            )
            comp_out = _run_bound_comparator(case_dir, objective_contract)
            if comp_out:
                extracted = _extract_error_metrics(comp_out)
                primary = _choose_primary_score(extracted)
                if primary is not None:
                    score = primary
                    score_reason = f"comparator score from {objective_contract.get('comparator_script','(bound)')}"
            else:
                score_reason = "comparator returned no output"
        except Exception as ex:
            score_reason = f"score lookup failed: {type(ex).__name__}: {ex}"

    return {
        "case_name": case_name,
        "case_dir": str(case_dir),
        "rc": rc,
        "run_ok": run_ok,
        "score": score,
        "score_reason": score_reason,
        "patched_keys": patched_keys,
        "skipped_keys": skipped_keys,
        "intent": plan_entry.get("intent", ""),
        "overrides": overrides,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run post-OED parametric experiments.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--plan", type=Path, default=None,
                    help="Optional JSON plan. List of {name, intent, overrides}. "
                         "If omitted, auto-generated from the artifact's coefficient names.")
    ap.add_argument("--cases-dir-name", default="cases",
                    help="Subdir name under run_dir for the post-OED cases (default: cases). "
                         "Matches the orchestrator's expected layout so --resume-from analysis "
                         "picks them up automatically. Pass --cases-dir-name paper_cases to keep "
                         "them isolated from the orchestrator-written cases/.")
    ap.add_argument("--timeout-s", type=int, default=21600)
    args = ap.parse_args()

    # 1. Resolve artifact
    from oed_artifact import resolve_oed_artifact  # type: ignore
    artifact = resolve_oed_artifact(args.run_dir)
    if artifact.get("status") != "ok":
        print(f"[post-OED] no usable artifact: {artifact.get('reason')}", file=sys.stderr)
        return 2

    print(f"[post-OED] artifact: category={artifact['category']} "
          f"iter={artifact['source_iteration']} "
          f"score={artifact.get('best_score')} "
          f"baseline={artifact.get('baseline_score')}", flush=True)
    print(f"[post-OED] base_case_dir: {artifact['base_case_dir']}", flush=True)
    print(f"[post-OED] coefficient_names: {artifact.get('coefficient_names')}", flush=True)

    # 2. Build plan
    if args.plan and args.plan.is_file():
        plan = _read_json(args.plan, [])
    else:
        # Best coefficient values come from the artifact's primary dictionary
        # (the values currently in the file are the OED's best). For
        # class_derivation, the artifact's `coefficient_block_name` (e.g.
        # `<Class>Coeffs`) names the nested sub-block. For runtime_source,
        # the top-level coded entry name in the file is used.
        best_values: Dict[str, str] = {}
        primary = artifact.get("primary_dictionary")
        if primary and Path(primary).is_file():
            text = Path(primary).read_text(encoding="utf-8", errors="replace")
            block = (
                artifact.get("coefficient_block_name")
                or _find_runtime_dict_block_name(text)
            )
            if block:
                best_values = _parse_coefficients_from_dict_block(text, block)
        coeff_names = artifact.get("coefficient_names") or list(best_values.keys())
        plan = auto_plan(
            coeff_names,
            best_values,
            parent_class_name=artifact.get("parent_class_name", ""),
            novel_coeff_names=artifact.get("novel_coefficient_names") or None,
        )

    print(f"[post-OED] plan: {len(plan)} cases — "
          f"{[p.get('name') for p in plan]}", flush=True)

    # 3. Run each case
    cases_dir = args.run_dir / args.cases_dir_name
    cases_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for i, entry in enumerate(plan, 1):
        name = entry.get("name") or f"case_{i:03d}"
        try:
            res = run_one_case(
                case_name=f"case_{i:03d}_{name}",
                plan_entry=entry,
                artifact=artifact,
                cases_dir=cases_dir,
                timeout_s=args.timeout_s,
            )
        except Exception as ex:
            res = {"case_name": f"case_{i:03d}_{name}", "rc": -1,
                   "run_ok": False, "error": f"{type(ex).__name__}: {ex}"}
        summary.append(res)
        s = res.get("score")
        sv = s.get("value") if isinstance(s, dict) else None
        print(f"[post-OED] {res['case_name']:<30} rc={res.get('rc')} "
              f"run_ok={res.get('run_ok')} score={sv}", flush=True)
        # incremental write
        _write_json(cases_dir / "_post_oed_summary.json", {
            "artifact": artifact,
            "plan": plan,
            "results": summary,
        })

    print(f"[post-OED] done. summary: {cases_dir / '_post_oed_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

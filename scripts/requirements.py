#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from timeline_logger import append_timeline_event, resolve_timeline_path


def bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


_MESH_PARAM_KEYS = frozenset({
    "mesh_nx", "mesh_ny", "mesh_nz", "mesh_n_cells",
    "mesh_cells_x", "mesh_cells_y", "mesh_cells_z",
    "ny_expansion_ratio", "nz_expansion_ratio", "nx_expansion_ratio",
    "grading", "mesh_grading", "cell_count", "n_cells",
})


def _strip_mesh_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _MESH_PARAM_KEYS}


def _synthesize_requirement_from_hypothesis(
    topic: str,
    item: Dict[str, Any],
    code_mod_context: str = "",
    strip_mesh: bool = False,
) -> str:
    desc = str(item.get("description") or item.get("hypothesis_text") or "").strip()
    params = item.get("parameter_value") or item.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    if strip_mesh:
        params = _strip_mesh_params(params)
    param_json = json.dumps(params, ensure_ascii=False)

    code_mod_note = ""
    if code_mod_context:
        code_mod_note = (
            "\n\nCUSTOM MODEL CONTEXT (mandatory — use this model, do not substitute built-in alternatives):\n"
            f"{code_mod_context[:6000]}\n"
        )

    mesh_note = ""
    if strip_mesh:
        mesh_note = (
            "\nMesh note: do NOT specify or change the mesh. "
            "The mesh is provided by the mesh-gate selected case and must be preserved as-is.\n"
        )

    return (
        f"Study topic: {topic}\n"
        f"Experiment intent: {desc or 'compare designed case against baseline'}\n"
        f"Target parameters: {param_json}\n"
        "Create and run a physically consistent OpenFOAM case satisfying this experiment intent, "
        "using stable numerics and producing fields/outputs needed for comparison."
        f"{code_mod_note}{mesh_note}"
    )


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Convert hypothesis output to Foam requirements.")
    parser.add_argument("--hypotheses", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--timeline", default="", type=str)
    parser.add_argument("--topic", default="", type=str)
    parser.add_argument("--code-mod-context", default="", type=str,
                        help="Path to text file with code-mod context to embed in each requirement")
    parser.add_argument("--strip-mesh-params", action="store_true", default=False,
                        help="Remove mesh-related parameters from requirements (mesh-gate provides the mesh)")
    parser.add_argument("--mesh-gate-resume", default="", type=str,
                        help="Path to mesh_gate_resume.json — injects selected mesh context per experiment")
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json — authoritative flow params override LLM guesses")
    args = parser.parse_args()
    timeline_path = resolve_timeline_path(args.timeline)

    in_path = Path(args.hypotheses)
    if not in_path.exists():
        print(f"Hypotheses file not found: {in_path}", file=sys.stderr)
        return 1
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Hypotheses JSON must be a list", file=sys.stderr)
        return 1

    code_mod_ctx = ""
    if args.code_mod_context:
        cm_path = Path(args.code_mod_context)
        if cm_path.is_file():
            code_mod_ctx = cm_path.read_text(encoding="utf-8", errors="ignore")
            print(f"[REQ] Loaded code-mod context ({len(code_mod_ctx)} chars)")
        else:
            print(f"[REQ] WARNING: code-mod-context path not found: {cm_path}")

    # Load mesh-gate resume: per-group selected mesh info + blockMeshDict
    mesh_resume: Dict[str, Any] = {}
    case_to_group: Dict[str, str] = {}
    default_group: str = ""
    if args.mesh_gate_resume:
        mg_path = Path(args.mesh_gate_resume)
        if mg_path.is_file():
            try:
                mesh_resume = json.loads(mg_path.read_text(encoding="utf-8"))
                case_to_group = mesh_resume.get("case_to_group") or {}
                default_group = str(mesh_resume.get("default_group") or "")
                n_groups = len(mesh_resume.get("groups") or {})
                print(f"[REQ] Loaded mesh-gate resume: {n_groups} group(s), "
                      f"{len(case_to_group)} case mappings, default_group={default_group!r}")
            except Exception as e:
                print(f"[REQ] WARNING: failed to read mesh-gate resume: {e}")
        else:
            print(f"[REQ] WARNING: mesh-gate-resume path not found: {mg_path}")

    if args.strip_mesh_params:
        print("[REQ] Mesh parameters (incl. mesh_cells_x/y) will be stripped from requirements")

    # Load starter understanding for authoritative flow parameters
    authoritative_params_block = ""
    authoritative_params: Dict[str, Any] = {}
    if args.starter_understanding:
        su_path = Path(args.starter_understanding)
        if su_path.is_file():
            try:
                su = json.loads(su_path.read_text(encoding="utf-8"))
                fp = su.get("flow_parameters") or {}
                if fp:
                    authoritative_params = {k: v for k, v in fp.items() if v is not None}
                ref = su.get("reference_data") or {}
                ref_lines = []
                if ref.get("description"):
                    ref_lines.append(f"Reference data: {ref['description']}")
                if ref.get("usage_guidance"):
                    ref_lines.append(f"How to compare: {ref['usage_guidance']}")
                param_str = json.dumps(authoritative_params, ensure_ascii=False) if authoritative_params else ""
                parts = []
                if param_str:
                    parts.append(
                        f"AUTHORITATIVE_FLOW_PARAMETERS (from base case — these override any LLM-guessed values): "
                        f"{param_str}"
                    )
                parts.extend(ref_lines)
                authoritative_params_block = "\n".join(parts)
                print(f"[REQ] Loaded starter_understanding: {param_str[:200]}")
            except Exception as e:
                print(f"[REQ] WARNING: failed to read starter-understanding: {e}")
        else:
            print(f"[REQ] WARNING: starter-understanding path not found: {su_path}")

    out: List[Dict[str, Any]] = []
    total = len(data)
    print(f"[REQ] Input hypothesis records: {total}")
    for i, item in enumerate(data, 1):
        exp_id = item.get("experiment_id") or item.get("hypothesis_id") or f"exp_{i:03d}"
        case_id = item.get("case_id") or f"case_{i:03d}"
        req = item.get("user_requirement_text") or item.get("requirement") or ""
        print(f"[REQ] Processing requirement {i}/{total} for experiment_id={exp_id}")

        # Overlay authoritative flow parameters from starter_understanding into the
        # hypothesis parameter_value block so synthesized requirements carry correct Re/nu/Ub.
        if authoritative_params and isinstance(item, dict):
            current_params = item.get("parameter_value") or item.get("parameters") or {}
            if not isinstance(current_params, dict):
                current_params = {}
            # Authoritative params fill in missing keys (don't clobber experiment-specific sweep values)
            for k, v in authoritative_params.items():
                if k not in current_params:
                    current_params[k] = v
            item = {**item, "parameter_value": current_params}

        if not req:
            req = _synthesize_requirement_from_hypothesis(
                args.topic,
                item if isinstance(item, dict) else {},
                code_mod_context=code_mod_ctx,
                strip_mesh=args.strip_mesh_params,
            )
            print("[REQ]   requirement source: synthesized from hypothesis")
        else:
            if code_mod_ctx and "CUSTOM MODEL CONTEXT" not in req:
                req += (
                    "\n\nCUSTOM MODEL CONTEXT (mandatory — use this model, do not substitute built-in alternatives):\n"
                    f"{code_mod_ctx[:6000]}\n"
                )
            print("[REQ]   requirement source: provided in hypothesis payload (code-mod context appended)"
                  if code_mod_ctx else "[REQ]   requirement source: provided in hypothesis payload")

        # Append authoritative params block so FoamAgent always sees the ground-truth values
        if authoritative_params_block and "AUTHORITATIVE_FLOW_PARAMETERS" not in req:
            req += f"\n\n{authoritative_params_block}"

        # Inject mesh-gate context: resolve which group this experiment belongs to,
        # then append the selected blockMeshDict and summary as authoritative mesh context.
        if mesh_resume:
            gid = case_to_group.get(str(case_id)) or default_group
            groups = mesh_resume.get("groups") or {}
            ginfo = groups.get(gid) or (next(iter(groups.values())) if groups else None)
            if ginfo:
                bmd = ginfo.get("blockMeshDict_content", "")
                summary = ginfo.get("summary", "")
                cell_counts = ginfo.get("cell_counts") or {}
                mesh_ctx_block = (
                    "\n\nMESH_GATE_CONTEXT — authoritative mesh from mesh-independence study:\n"
                    f"{summary}\n"
                )
                if cell_counts:
                    mesh_ctx_block += (
                        f"Selected mesh cell counts: "
                        f"nx={cell_counts.get('nx_total')} (x blocks: {cell_counts.get('x_per_block')}), "
                        f"ny={cell_counts.get('ny_total')} (y blocks: {cell_counts.get('y_per_block')}), "
                        f"total={cell_counts.get('total_cells')} cells.\n"
                    )
                if bmd:
                    mesh_ctx_block += (
                        "The blockMeshDict below is the authoritative mesh. "
                        "Copy it as-is to system/blockMeshDict. "
                        "Only deviate if this experiment explicitly requires a different resolution "
                        "(state it in natural language; the mesh-gate LLM will apply the delta).\n"
                        f"MESH_GATE_BLOCKMESH_BEGIN\n{bmd}\nMESH_GATE_BLOCKMESH_END\n"
                    )
                if "MESH_GATE_CONTEXT" not in req:
                    req += mesh_ctx_block
                    print(f"[REQ]   mesh-gate context injected (group={gid}, bmd={len(bmd)} chars)")

        print(f"[REQ]   requirement text: {str(req)[:500]}")
        out.append(
            {
                "case_id": item.get("case_id") or f"case_{i:03d}",
                "user_requirement_text": req,
                "experiment_id": exp_id,
                "description": item.get("description", f"Experiment {exp_id}"),
                "study_id": item.get("study_id", ""),
            }
        )
    print(f"[REQ] Total requirements generated: {len(out)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {
            "stage": "requirements",
            "requirement_count": len(out),
            "requirements": [
                {
                    "case_id": r.get("case_id"),
                    "experiment_id": r.get("experiment_id"),
                    "description": r.get("description", ""),
                    "user_requirement_text": r.get("user_requirement_text", ""),
                }
                for r in out
            ],
            "output_path": str(out_path),
        },
    )
    print(f"Requirements generated: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

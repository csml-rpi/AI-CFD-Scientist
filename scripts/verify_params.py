#!/usr/bin/env python3
"""LLM-based case configuration verification.

Reads all case dictionary files, compares against the experiment requirement
using an LLM, and optionally applies fixes. Generic — works for any
OpenFOAM parameter, model, or boundary condition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    return root


class ParamMismatch(BaseModel):
    param: str = Field(description="Parameter name or setting that is wrong")
    expected: str = Field(description="What the requirement specifies")
    actual: str = Field(description="What the case file actually has")
    file: str = Field(description="Relative path of the file containing the mismatch")


class FixFile(BaseModel):
    file_path: str = Field(description="Relative file path under case dir, e.g. constant/momentumTransport")
    content: str = Field(description="Full corrected file content")


class VerificationResult(BaseModel):
    is_correct: bool = Field(description="Whether the case configuration fully matches the requirement")
    reasoning: str = Field(description="Brief explanation of the verification result")
    mismatches: List[ParamMismatch] = Field(
        default_factory=list,
        description="List of specific parameter mismatches found",
    )
    fixes: List[FixFile] = Field(
        default_factory=list,
        description="If not correct, provide corrected full file contents for files that need changing",
    )


def _collect_case_files(case_dir: Path) -> Dict[str, str]:
    """Read all files under 0/, constant/, system/ (skip polyMesh binaries)."""
    out: Dict[str, str] = {}
    for folder in ("0", "constant", "system"):
        root = case_dir / folder
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(case_dir)).replace("\\", "/")
            if "polyMesh" in rel:
                continue
            try:
                out[rel] = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return out


def _collect_custom_model_context(case_dir: Path) -> str:
    """Summarise any custom model source code in customModels/."""
    parts: List[str] = []
    cm = case_dir / "customModels"
    if not cm.is_dir():
        return ""
    for subdir in sorted(cm.iterdir()):
        if not subdir.is_dir():
            continue
        parts.append(f"Custom model: {subdir.name}")
        for ext in ("*.H", "*.C"):
            for f in sorted(subdir.glob(ext))[:3]:
                try:
                    src = f.read_text(encoding="utf-8", errors="ignore")
                    parts.append(f"--- {f.name} (first 3000 chars) ---\n{src[:3000]}")
                except Exception:
                    pass
    return "\n".join(parts) if parts else ""


def _apply_fixes(case_dir: Path, fixes: List[FixFile]) -> int:
    """Write corrected files back to the case directory. Returns count of changed files."""
    changed = 0
    for fx in fixes:
        rel = (fx.file_path or "").strip().lstrip("./")
        if not rel:
            continue
        if not (rel.startswith("0/") or rel.startswith("constant/") or rel.startswith("system/")):
            continue
        target = (case_dir / rel).resolve()
        try:
            target.relative_to(case_dir.resolve())
        except ValueError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        old = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
        if old != fx.content:
            target.write_text(fx.content, encoding="utf-8")
            print(f"[VERIFY] Fixed: {rel}")
            changed += 1
    return changed


_SYSTEM_PROMPT = (
    "You are an OpenFOAM case configuration verifier. Your ONLY job is to check whether "
    "the case dictionary files match the experiment requirement.\n\n"
    "You will receive:\n"
    "1. The experiment requirement. If it begins with AUTHORITATIVE_TARGET_PARAMETERS, treat the "
    "JSON object in that section as the single source of truth for all numeric and discrete "
    "controls (coefficients, exponents, nu, mesh cell counts when listed there, bulk/inlet "
    "velocity targets, etc.). Downstream narrative or code-mod excerpts must NOT override those "
    "values.\n"
    "2. All case dictionary files under 0/, constant/, system/\n"
    "3. Any custom model source code from customModels/ (if present)\n\n"
    "Your task:\n"
    "- Check that EVERY parameter, model setting, boundary condition, and physics option "
    "specified in the requirement is correctly reflected in the case files.\n"
    "- If the requirement specifies a custom model and customModels/ source exists, "
    "verify the model is activated in the correct dictionary and its library is loaded in "
    "system/controlDict via libs().\n"
    "- If the requirement specifies a standard/built-in model, verify no custom library "
    "for that model type is loaded.\n"
    "- Check that numeric coefficient values match EXACTLY (no rounding, no substitution).\n"
    "- Check boundary conditions, solver settings, and any other specifics mentioned in the requirement.\n\n"
    "If anything is wrong, set is_correct=false, list every mismatch, and provide corrected "
    "full file contents for ONLY the files that need changing.\n\n"
    "CRITICAL RULES:\n"
    "- Do NOT rename, restructure, or relocate coefficient keywords for custom models. "
    "The C++ source expects exact keyword names.\n"
    "- Only fix files under 0/, constant/, system/.\n"
    "- When providing fixes, include the COMPLETE file content (not just the changed part).\n"
    "- Do NOT change the mesh (polyMesh), fvSchemes, or fvSolution unless the requirement "
    "explicitly says they are wrong.\n"
    "- Focus on model selection, model coefficients, boundary condition values, and library loading."
)


def verify_and_fix(
    case_dir: Path,
    requirement: str,
    max_loops: int = 3,
) -> Dict[str, Any]:
    """Run LLM-based verification loop. Returns verification result dict."""
    bootstrap_paths()
    from config import Config
    from utils import LLMService

    config = Config()
    config.case_dir = str(case_dir)
    llm_service = LLMService(config)

    custom_ctx = _collect_custom_model_context(case_dir)
    total_fixes_applied = 0

    for loop in range(1, max_loops + 1):
        files = _collect_case_files(case_dir)
        user_prompt = f"Experiment requirement:\n{requirement}\n\n"
        if custom_ctx:
            user_prompt += f"Custom model source code:\n{custom_ctx}\n\n"
        user_prompt += (
            "Case dictionary files (relative path -> full content):\n"
            f"{json.dumps(files, ensure_ascii=False)[:180000]}"
        )

        try:
            result: VerificationResult = llm_service.invoke(
                user_prompt, _SYSTEM_PROMPT, pydantic_obj=VerificationResult
            )
        except Exception as e:
            print(f"[VERIFY] LLM call failed (loop {loop}): {e}", file=sys.stderr)
            return {
                "is_correct": False,
                "reasoning": f"LLM verification failed: {e}",
                "mismatches": [],
                "fixes_applied": total_fixes_applied,
                "loops": loop,
            }

        if result.is_correct:
            print(f"[VERIFY] Configuration correct (loop {loop}): {result.reasoning}")
            return {
                "is_correct": True,
                "reasoning": result.reasoning,
                "mismatches": [],
                "fixes_applied": total_fixes_applied,
                "loops": loop,
            }

        mismatch_dicts = [m.model_dump() for m in result.mismatches]
        print(f"[VERIFY] Mismatches found (loop {loop}/{max_loops}): {len(result.mismatches)}")
        for m in result.mismatches:
            print(f"[VERIFY]   {m.file}: {m.param} — expected '{m.expected}', got '{m.actual}'")

        if result.fixes and loop < max_loops:
            n_fixed = _apply_fixes(case_dir, result.fixes)
            total_fixes_applied += n_fixed
            print(f"[VERIFY] Applied {n_fixed} fix(es), re-verifying...")
            if n_fixed == 0:
                return {
                    "is_correct": False,
                    "reasoning": result.reasoning,
                    "mismatches": mismatch_dicts,
                    "fixes_applied": total_fixes_applied,
                    "loops": loop,
                }
        else:
            n_fixed_last = 0
            if result.fixes:
                n_fixed_last = _apply_fixes(case_dir, result.fixes)
                total_fixes_applied += n_fixed_last
            return {
                "is_correct": False,
                "reasoning": result.reasoning,
                "mismatches": mismatch_dicts,
                "fixes_applied": total_fixes_applied,
                "loops": loop,
            }

    return {
        "is_correct": False,
        "reasoning": "Did not converge within max loops",
        "mismatches": [],
        "fixes_applied": total_fixes_applied,
        "loops": max_loops,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-based case configuration verification.")
    parser.add_argument("--case", required=True, type=str, help="Path to OpenFOAM case directory")
    parser.add_argument(
        "--requirement-file", required=True, type=str,
        help="Path to a text file containing the experiment requirement",
    )
    parser.add_argument("--output", required=True, type=str, help="Path to write verification JSON result")
    parser.add_argument("--max-loops", default=3, type=int, help="Max verify+fix loops (default 3)")
    args = parser.parse_args()

    case_dir = Path(args.case).resolve()
    req_path = Path(args.requirement_file).resolve()
    if not case_dir.exists():
        print(f"Case not found: {case_dir}", file=sys.stderr)
        return 1
    if not req_path.exists():
        print(f"Requirement file not found: {req_path}", file=sys.stderr)
        return 1

    requirement = req_path.read_text(encoding="utf-8")
    result = verify_and_fix(case_dir, requirement, max_loops=args.max_loops)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[VERIFY] Result: is_correct={result['is_correct']}, loops={result['loops']}")
    return 0 if result["is_correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


def vec(params: Dict[str, Any], keys: List[str]) -> List[float]:
    out = []
    for k in keys:
        v = params.get(k, 0.0)
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return out


def l2(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Select nearest successful case for rerun seeding.")
    parser.add_argument("--failed", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    failed = Path(args.failed).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = manifest if isinstance(manifest, list) else manifest.get("cases", [])
    if not entries:
        print("Manifest has no cases", file=sys.stderr)
        return 1

    failed_entry = next((e for e in entries if Path(e.get("case_path", "")).resolve() == failed), None)
    if not failed_entry:
        print("Failed case not found in manifest", file=sys.stderr)
        return 1
    successful = [e for e in entries if e.get("status") == "success"]
    if not successful:
        print("No successful cases found", file=sys.stderr)
        return 1

    f_params = failed_entry.get("parameters", {})
    keys = sorted({*f_params.keys(), *{k for e in successful for k in (e.get("parameters", {}) or {}).keys()}})
    f_vec = vec(f_params, keys)

    nearest: Tuple[Dict[str, Any], float] | None = None
    for s in successful:
        d = l2(f_vec, vec(s.get("parameters", {}), keys))
        if nearest is None or d < nearest[1]:
            nearest = (s, d)

    assert nearest is not None
    src = Path(nearest[0]["case_path"]).resolve()
    dst = Path(args.output).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for folder in ["0", "constant", "system"]:
        sdir = src / folder
        if not sdir.exists():
            continue
        if folder == "constant" and (sdir / "customModels").exists():
            pass
        ddir = dst / folder
        if ddir.exists():
            shutil.rmtree(ddir)
        shutil.copytree(sdir, ddir, ignore=shutil.ignore_patterns("customModels"))
        copied.append(folder)

    result = {
        "source_case_id": nearest[0].get("case_id"),
        "source_case_path": str(src),
        "distance": nearest[1],
        "files_copied": copied,
    }
    (dst / "rerun_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

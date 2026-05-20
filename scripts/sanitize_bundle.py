#!/usr/bin/env python3
"""Bundle sanitizer — item 14.

Prepare a CFD-scientist run directory (or a bundle of them) for release/sharing:

  1. Absolute machine paths in text/JSON/CSV/log files are rewritten to portable
     placeholders — `<BUNDLE>` (the bundle root), `<HOME>` (other home dirs),
     `<OPENFOAM>` (the OpenFOAM install). This removes the `/home/<user>/...`
     leaks the Sonnet skill-smoke audit found in manifest.json / state.json /
     audit_passed.json.
  2. Symlinks are catalogued. In-bundle absolute symlinks are converted to
     relative ones (so the bundle stays self-contained when moved); symlinks
     pointing outside the bundle are recorded as reproducibility risks.
  3. Compiled artifacts (.o / .so / .a) are KEPT in place and recorded in
     `reproducibility_manifest.json` with size, sha256, and a rebuild command
     when a sibling `customModels/<Class>/Make/` is found. (Decision: keep
     binaries, document them — do not strip.)

The manifest is always written. Path/symlink rewrites happen only with
`--apply`; without it the script is a dry-run report. `--apply` edits small
metadata files IN PLACE — run it on a copy or a zip-staging directory.

Usage:
    python scripts/sanitize_bundle.py --bundle runs/skill_smoke_claude
    python scripts/sanitize_bundle.py --bundle <dir> --apply

Exit codes:
    0 — completed (dry-run or apply); manifest written
    6 — usage error (bundle dir missing)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

TEXT_SUFFIXES = {".json", ".csv", ".txt", ".log", ".tex", ".md", ".dat",
                 ".yaml", ".yml", ".cfg", ".bib", ".out"}
BINARY_SUFFIXES = {".o", ".so", ".a"}
SKIP_DIRS = {".git", "__pycache__"}
MAX_TEXT_BYTES = 8 * 1024 * 1024  # don't rewrite files larger than this


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def _build_replacements(bundle: Path) -> List[tuple]:
    """Ordered (regex, placeholder) replacements — longest/most-specific first."""
    repls: List[tuple] = []
    bundle_abs = str(bundle)
    repls.append((re.escape(bundle_abs), "<BUNDLE>"))
    of_dir = os.environ.get("WM_PROJECT_DIR", "").rstrip("/")
    if of_dir:
        repls.append((re.escape(of_dir), "<OPENFOAM>"))
    repls.append((r"/(?:opt|usr/lib|usr/local)/openfoam[^\s\"':]*", "<OPENFOAM>"))
    # any other /home/<user> root (after the bundle-specific one above)
    repls.append((r"/home/[A-Za-z0-9_.-]+", "<HOME>"))
    repls.append((r"/Users/[A-Za-z0-9_.-]+", "<HOME>"))
    return repls


def _rewrite_text(text: str, repls: List[tuple]) -> str:
    for pat, placeholder in repls:
        text = re.sub(pat, placeholder, text)
    return text


def _rebuild_hint(so_path: Path) -> str:
    """If the .so sits under customModels/<Class>/, give the wmake rebuild line."""
    for parent in so_path.parents:
        if (parent / "Make" / "files").is_file():
            return f"cd {parent} && wmake libso"
    return "rebuild from the case's customModels/<Class>/ source via `wmake libso`"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanitize a CFD-scientist bundle.")
    ap.add_argument("--bundle", required=True,
                    help="run directory or bundle of run directories")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite paths + relativize in-bundle symlinks IN "
                         "PLACE (default: dry-run report only)")
    ap.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        print(f"ERROR: bundle directory not found: {bundle}", file=sys.stderr)
        return 6

    repls = _build_replacements(bundle)
    rewritten: List[str] = []
    binaries: List[Dict[str, Any]] = []
    symlinks: List[Dict[str, Any]] = []

    for root, dirs, files in os.walk(bundle, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rootp = Path(root)
        # symlinked subdirectories
        for d in list(dirs):
            dp = rootp / d
            if dp.is_symlink():
                dirs.remove(d)
                symlinks.append(_record_symlink(dp, bundle, args.apply))
        for name in files:
            p = rootp / name
            if p.is_symlink():
                symlinks.append(_record_symlink(p, bundle, args.apply))
                continue
            suffix = p.suffix.lower()
            if suffix in BINARY_SUFFIXES:
                binaries.append({
                    "path": str(p.relative_to(bundle)),
                    "size_bytes": p.stat().st_size if p.exists() else 0,
                    "sha256": _sha256(p),
                    "rebuild": _rebuild_hint(p),
                    "note": "kept in place; may contain embedded build paths",
                })
                continue
            if suffix in TEXT_SUFFIXES:
                try:
                    if p.stat().st_size > MAX_TEXT_BYTES:
                        continue
                    original = p.read_text(errors="replace")
                except Exception:
                    continue
                cleaned = _rewrite_text(original, repls)
                if cleaned != original:
                    rewritten.append(str(p.relative_to(bundle)))
                    if args.apply:
                        try:
                            p.write_text(cleaned)
                        except Exception as e:  # noqa: BLE001
                            print(f"  WARN: could not rewrite {p}: {e}",
                                  file=sys.stderr)

    external_links = [s for s in symlinks if s["class"] == "external"]
    manifest = {
        "bundle": str(bundle),
        "applied": bool(args.apply),
        "placeholders": {
            "<BUNDLE>": "this bundle's root directory",
            "<HOME>": "a user home directory (workstation path removed)",
            "<OPENFOAM>": "the OpenFOAM installation root ($WM_PROJECT_DIR)",
        },
        "n_text_files_with_paths": len(rewritten),
        "text_files_with_paths": sorted(rewritten),
        "n_symlinks": len(symlinks),
        "symlinks": symlinks,
        "n_external_symlinks": len(external_links),
        "n_compiled_artifacts": len(binaries),
        "compiled_artifacts": binaries,
        "reproducibility_notes": [
            "Compiled .o/.so/.a artifacts are kept; rebuild each from the listed "
            "`rebuild` command on the target OpenFOAM 10 install.",
            "External symlinks (class=external) point outside the bundle and "
            "WILL break when the bundle is moved — repackage their targets or "
            "rebuild them.",
        ],
    }
    (bundle / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        mode = "APPLIED" if args.apply else "DRY-RUN"
        print(f"[{mode}] sanitize {bundle}")
        print(f"  text files with workstation paths : {len(rewritten)}")
        print(f"  symlinks catalogued               : {len(symlinks)} "
              f"({len(external_links)} external / reproducibility risk)")
        print(f"  compiled artifacts (.o/.so/.a)     : {len(binaries)} "
              f"(kept; recorded in manifest)")
        print(f"  manifest: {bundle / 'reproducibility_manifest.json'}")
        if not args.apply and rewritten:
            print("  re-run with --apply to rewrite the paths in place.")
    return 0


def _record_symlink(link: Path, bundle: Path, apply: bool) -> Dict[str, Any]:
    try:
        target = os.readlink(link)
    except OSError:
        target = ""
    resolved = (link.parent / target).resolve() if target else link.resolve()
    try:
        inside = bundle in resolved.parents or resolved == bundle
    except Exception:
        inside = False
    rec: Dict[str, Any] = {
        "path": str(link.relative_to(bundle)) if bundle in link.parents
                else str(link),
        "target": target,
        "class": "internal" if inside else "external",
        "rewritten_relative": False,
    }
    # Convert an in-bundle ABSOLUTE symlink to a relative one so the bundle
    # stays self-contained when moved.
    if apply and inside and os.path.isabs(target):
        try:
            rel = os.path.relpath(resolved, link.parent)
            link.unlink()
            link.symlink_to(rel)
            rec["target"] = rel
            rec["rewritten_relative"] = True
        except Exception as e:  # noqa: BLE001
            rec["rewrite_error"] = str(e)
    return rec


if __name__ == "__main__":
    raise SystemExit(main())

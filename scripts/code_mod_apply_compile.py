#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_wmake_make_fragment(text: str) -> str:
    """
    wmake parses Make/files and Make/options as makefile fragments.
    C-style /* */ comments are invalid and cause 'missing separator' on line 1.
    Strip leading block comments repeatedly until none remain.
    """
    t = text.lstrip("\ufeff")
    while True:
        new = re.sub(r"^\s*/\*[\s\S]*?\*/\s*\n?", "", t, count=1)
        if new == t:
            break
        t = new
    return t.lstrip()


# Default *Coeffs* when none exist; keywords must match the fallback C++ template in this module.
_DEFAULT_CUSTOM_VISCOSITY_COEFF_LINES = (
    "        nu_inf      1e-06;\n"
    "        k           1e-03;\n"
    "        n           0.6;\n"
    "        gammaDotmin 1e-06;\n"
)


def _case_local_model_so(model_dir: Path) -> Optional[Path]:
    """Return path to built .so under customModels/<Class>/platforms/... if present."""
    plat = model_dir / "platforms"
    if not plat.is_dir():
        return None
    cls = model_dir.name
    for p in plat.rglob(f"lib{cls}.so"):
        if p.is_file():
            return p
    for p in sorted(plat.rglob("*.so")):
        if p.is_file():
            return p
    return None


def _so_looks_viable(so: Path) -> bool:
    """Best-effort viability check for built shared objects."""
    if not so.is_file():
        return False
    try:
        size = so.stat().st_size
        if size < 4096:
            return False
        with so.open("rb") as f:
            magic = f.read(4)
        # ELF shared object on Linux.
        return magic == b"\x7fELF"
    except OSError:
        return False


def _candidate_global_lib_dirs() -> List[Path]:
    """
    Potential non-case-local destinations used by OpenFOAM/wmake layouts.
    Keep this generic across installations and shells.
    """
    out: List[Path] = []
    for key in ("FOAM_USER_LIBBIN", "FOAM_SITE_LIBBIN", "FOAM_LIBBIN"):
        v = (os.environ.get(key) or "").strip()
        if v:
            out.append(Path(v).expanduser())
    return out


def _locate_built_shared_object(case_dir: Path, class_name: str) -> Optional[Path]:
    """
    Find a usable lib<Class>.so from either:
    - case-local customModels/.../platforms/...
    - OpenFOAM global lib bins (FOAM_USER_LIBBIN/FOAM_SITE_LIBBIN/FOAM_LIBBIN)
    """
    model_dir = case_dir / "customModels" / class_name
    local = _case_local_model_so(model_dir)
    if local is not None and _so_looks_viable(local):
        return local

    needle = f"lib{class_name}.so"
    for d in _candidate_global_lib_dirs():
        p = d / needle
        if _so_looks_viable(p):
            return p
    return None


def _normalize_make_files_lib_path(make_files_text: str, class_name: str) -> str:
    """
    Force wmake to emit the .so under <case>/platforms/... (not $FOAM_USER_LIBBIN).

    OpenFOAM loads ``libs ("libFoo.so")`` from
    ``$FOAM_CASE/customModels/<Class>/platforms/$(WM_OPTIONS)/lib/``.  If wmake
    installs to ``$(FOAM_USER_LIBBIN)`` instead, utilities still probe the case
    path first and fail with dlopen / unknown model type.  wmake runs in
    ``customModels/<ClassName>/``, so use ``./platforms/$(WM_OPTIONS)/lib/...``.
    """
    t = make_files_text
    # Standard mistaken pattern from templates / LLM output
    pat_user = re.compile(
        r"^(\s*)LIB\s*=\s*\$\(FOAM_USER_LIBBIN\)/lib(\w+)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def _repl(m: re.Match[str]) -> str:
        indent, libbase = m.group(1), m.group(2)
        return f"{indent}LIB = ./platforms/$(WM_OPTIONS)/lib/lib{libbase}"

    if pat_user.search(t):
        return pat_user.sub(_repl, t)

    # If LIB is missing entirely but we have sources, append a case-local rule
    if class_name and re.search(r"^\s*LIB\s*=", t, re.MULTILINE | re.IGNORECASE) is None:
        if t.strip() and not t.rstrip().endswith("\\"):
            t = t.rstrip() + "\n"
        t += f"LIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}\n"
    return t


def _fix_make_files_wmake_layout(make_files_text: str, class_name: str) -> str:
    """
    wmake's ``include $(FILES)`` builds ``SOURCE`` from ``.C`` lines in ``Make/files``.
    ``LIB_SRCS`` is not a wmake variable; it corrupts parsing and produces an empty .so.

    Also enforce that the listed ``.C`` source file matches the actual on-disk
    filename the builder wrote (``<class_name>.C``). LLM code-gen has been
    observed to invent its own "preferred" source-file name (e.g. the parent
    class identifier) and bake it into Make/files, while the builder writes
    the C/H pair under the canonical ``<class_name>`` stem — causing wmake
    to fail with 'No rule to make target <WrongName>.C.dep'.

    Generic rewrite (applies to all modes, all topics): if the Make/files
    references any ``.C`` line other than ``<class_name>.C``, or the LIB
    target basename disagrees with ``lib<class_name>``, replace the whole
    fragment with the canonical form. Safe because the builder always
    writes exactly one ``<class_name>.C`` under ``customModels/<class_name>/``.
    """
    t = make_files_text.replace("\r\n", "\n")
    if "LIB_SRCS" in t:
        return f"{class_name}.C\n\nLIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}\n"

    if not class_name:
        return t

    expected_c = f"{class_name}.C"
    expected_lib_basename = f"lib{class_name}"

    # Collect .C source lines (non-comment, non-empty, ending in .C)
    c_lines: List[str] = []
    for raw_line in t.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("/*"):
            continue
        # Skip any LIB= or similar assignment lines
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line):
            continue
        # Strip any trailing backslash-continuation
        stripped = line.rstrip("\\").strip()
        if stripped.endswith(".C"):
            c_lines.append(stripped)

    # Pull out the declared LIB basename, if any
    m_lib = re.search(r"^\s*LIB\s*=\s*(?:\./)?(?:platforms/[^/]+/lib/)?(lib\w+)",
                      t, re.MULTILINE | re.IGNORECASE)
    declared_lib = m_lib.group(1) if m_lib else ""

    # If what the LLM wrote already matches the canonical expectation, keep it.
    c_ok = len(c_lines) == 1 and Path(c_lines[0]).name == expected_c
    lib_ok = (not declared_lib) or declared_lib == expected_lib_basename
    if c_ok and lib_ok:
        return t

    # Otherwise rewrite to the canonical form. Topic/mode-agnostic — just
    # align with the files the builder actually wrote to disk.
    return f"{class_name}.C\n\nLIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}\n"


def _patch_control_dict(control_dict: Path, class_name: str) -> None:
    txt = control_dict.read_text(encoding="utf-8", errors="ignore") if control_dict.exists() else ""
    lib_line = f"\"lib{class_name}.so\""
    # If a top-level libs block already references our library, keep as-is.
    top_level_libs = re.search(r"(?ms)^\s*libs\s*\([\s\S]*?\)\s*;", txt)
    if top_level_libs and lib_line in top_level_libs.group(0):
        return
    if top_level_libs:
        txt = re.sub(
            r"(?ms)^\s*libs\s*\(([\s\S]*?)\)\s*;",
            lambda m: f"libs ({m.group(1)} {lib_line});",
            txt,
            count=1,
        )
    else:
        insertion = f"\nlibs ({lib_line});\n"
        # Prefer inserting near top-level controls instead of appending after functions{}.
        if "runTimeModifiable" in txt:
            txt = re.sub(
                r"(runTimeModifiable\s+\w+\s*;)",
                r"\1" + insertion,
                txt,
                count=1,
            )
        else:
            txt += insertion
    _write(control_dict, txt)


def _enforce_case_local_make_files(case_dir: Path, class_name: str) -> bool:
    """
    Keep Make/files local-only so libraries are emitted under case/customModels/.../platforms.
    Returns True if file changed.
    """
    mf = case_dir / "customModels" / class_name / "Make" / "files"
    if not mf.is_file():
        return False
    raw = mf.read_text(encoding="utf-8", errors="ignore")
    fixed = _fix_make_files_wmake_layout(raw, class_name)
    fixed = _normalize_make_files_lib_path(fixed, class_name)
    if fixed != raw:
        _safe_write_local(case_dir, class_name, mf, fixed)
        return True
    return False


def _patch_control_dict_case_relative_lib(case_dir: Path, class_name: str) -> None:
    """
    OpenFOAM 10 often does not resolve bare ``lib<Class>.so`` inside nested
    ``customModels/<Class>/platforms/$(WM_OPTIONS)/lib/``.  Point ``libs`` at the
    actual .so path **relative to the case root** so copies (e.g. validation)
    still work.
    """
    model_dir = case_dir / "customModels" / class_name
    so = _case_local_model_so(model_dir)
    if so is None or not so.is_file():
        return
    cd = case_dir / "system" / "controlDict"
    if not cd.is_file():
        return
    try:
        rel = so.resolve().relative_to(case_dir.resolve())
    except ValueError:
        tok = f'"{str(so.resolve()).replace(chr(92), "/")}"'
    else:
        tok = f'"{str(rel).replace(chr(92), "/")}"'
    replacement = f"libs ({tok});"
    txt = cd.read_text(encoding="utf-8", errors="ignore")
    txt2 = re.sub(r"(?ms)^\s*libs[\s\n]*\([\s\S]*?\)\s*;", replacement, txt, count=1)
    if txt2 == txt and "libs" not in txt.lower():
        txt2 = txt.rstrip() + "\n" + replacement + "\n"
    out = txt2 if txt2.endswith("\n") else txt2 + "\n"
    _write(cd, out)
    print(f"[CODEMOD] controlDict libs -> case-relative .so path: {tok}")


def _patch_activation_dict(path: Path, class_name: str, mode: str, build: Optional[Dict[str, Any]] = None) -> None:
    """
    Only apply **known-safe** automatic edits. Generic / unknown modes must not invent
    invalid dictionary keys (e.g. simulationType <Class>) — the LLM + user wire activation.
    """
    build = build if isinstance(build, dict) else {}
    if not path.is_file():
        return
    txt = path.read_text(encoding="utf-8", errors="ignore")

    if mode == "custom_viscosity":
        coeff_block_name = f"{class_name}Coeffs"
        if re.search(rf"\bviscosityModel\s+{re.escape(class_name)}\s*;", txt) and coeff_block_name in txt:
            _write(path, txt)
            return
        if re.search(r"\bviscosityModel\s+\w+\s*;", txt):
            txt = re.sub(r"\bviscosityModel\s+\w+\s*;", f"viscosityModel      {class_name};", txt, count=1)
        else:
            txt += f"\nviscosityModel      {class_name};\n"
        coeff_body = (
            f"\n\n    {coeff_block_name}\n"
            "    {\n"
            f"{_DEFAULT_CUSTOM_VISCOSITY_COEFF_LINES}"
            "    }"
        )
        if coeff_block_name not in txt:
            vm_re = re.compile(
                rf"(\bviscosityModel\s+{re.escape(class_name)}\s*;)",
                re.MULTILINE,
            )
            m = vm_re.search(txt)
            if m:
                insert_at = m.end()
                txt = txt[:insert_at] + coeff_body + txt[insert_at:]
            else:
                txt += coeff_body + "\n"
        _write(path, txt)
        return

    if mode == "custom_turbulence_model_modification":
        parent = str(build.get("parent_model") or "").strip()
        if parent and parent != "unknown":
            pat = re.compile(rf"\bmodel\s+{re.escape(parent)}\s*;", re.MULTILINE)
            if pat.search(txt):
                txt = pat.sub(f"model            {class_name};", txt, count=1)
                _write(path, txt)
        return

    # custom_source, custom_case_library, or anything else: no automatic activation patch.


def _bootstrap_paths(repo_root: Path) -> None:
    foam_src = repo_root / "Foam-Agent" / "src"
    lang_src = repo_root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


def _safe_write_local(case_path: Path, class_name: str, target: Path, text: str) -> None:
    root = (case_path / "customModels" / class_name).resolve()
    rt = target.resolve()
    try:
        rt.relative_to(root)
    except ValueError:
        raise RuntimeError(f"Refusing to write outside local customModels tree: {target}")
    _write(target, text)


def _load_payload_for_context(case_path: Path) -> Dict[str, Any]:
    run_dir = case_path.parent
    p = run_dir / "code_mod_payload.json"
    data = _read_json(p, {})
    return data if isinstance(data, dict) else {}


def _read_openfoam_refs_from_paths(paths: List[str]) -> List[Dict[str, str]]:
    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    if not wm:
        return []
    wm_root = Path(wm).expanduser().resolve()
    out: List[Dict[str, str]] = []
    seen = set()
    for raw in paths[:12]:
        p = Path(str(raw))
        if not p.is_absolute():
            p = (wm_root / p).resolve()
        else:
            p = p.resolve()
        if not p.exists() or not p.is_file():
            continue
        try:
            p.relative_to(wm_root)
        except ValueError:
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        out.append({"path": str(p), "excerpt": txt[:12000]})
    return out


# ---------------------------------------------------------------------------
# Generic helpers used by the three enhancement pieces (diff / static-checks /
# clone-parent). All are mode-agnostic — they work identically for turbulence,
# viscosity, phase-change, source-term, BC, flux-scheme, or any other
# OpenFOAM code-mod target.
# ---------------------------------------------------------------------------

def _apply_unified_diff_to_file(path: Path, diff_text: str) -> Tuple[bool, str]:
    """Apply a unified diff to `path`. Return (ok, error_message).

    Uses system `patch` via subprocess for maximum compatibility with LLM-
    emitted diff formats. Falls back to a tolerant Python-side applier if
    `patch` isn't available or rejects the diff.

    Generic — not specific to C++, not specific to OpenFOAM. Works for any
    text file and any properly-formed unified-diff hunks. If the diff lacks
    fuzz-friendly context, this applier is conservative: it refuses to write
    garbage. Returns an error message the LLM can use on retry.
    """
    diff = diff_text.strip()
    if not diff:
        return False, "empty diff"
    # Normalize line endings and ensure the file-header mentions only the
    # basename — avoids `patch -p` ambiguity with absolute paths.
    diff = diff.replace("\r\n", "\n")
    if not diff.endswith("\n"):
        diff += "\n"

    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tf:
            tf.write(diff)
            patch_path = tf.name
    except Exception as exc:
        return False, f"could not write temp patch file: {exc}"

    try:
        # Derive cwd for `patch`: for <class>.C / <class>.H (at root), use
        # their parent. For Make/files or Make/options, use the Make's
        # parent (i.e. the customModels/<class> dir). This way the diff's
        # `--- a/Make/options` / `--- a/<class>.C` headers both resolve to
        # real files when `patch` strips the `a/` via `-p1`.
        if path.parent.name == "Make":
            cwd = str(path.parent.parent)
        else:
            cwd = str(path.parent)
        # Prefer -p1 (standard diff format: a/foo → foo) over -p0.
        # --fuzz=3 tolerates context-line mismatches up to 3 lines.
        # --ignore-whitespace lets hunks match despite whitespace drift.
        # These two flags absorb the most common LLM diff-generation errors
        # (wrong line counts in @@ headers, slightly different context).
        for p_level in ("1", "0"):
            res = subprocess.run(
                ["patch", f"-p{p_level}", "--fuzz=3", "--ignore-whitespace", "-i", patch_path],
                cwd=cwd, capture_output=True, text=True, timeout=30,
            )
            if res.returncode == 0:
                return True, ""
        # Last-resort: feed the diff to patch with --merge (permissive).
        res = subprocess.run(
            ["patch", "--merge", "-p0", "--fuzz=5", "--ignore-whitespace", "-i", patch_path],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if res.returncode == 0:
            return True, ""
        err_tail = (res.stderr or res.stdout or "")[-500:]
        return False, f"patch rejected diff: {err_tail}"
    except FileNotFoundError:
        return False, "system `patch` binary not on PATH"
    except Exception as exc:
        return False, f"patch apply exception: {exc}"
    finally:
        try:
            Path(patch_path).unlink()
        except Exception:
            pass


def _precompile_static_checks(
    case_path: Path, class_name: str, recon_ctx: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Generic pre-compile invariants that apply to any OpenFOAM code-mod.

    Returns a list of violation strings. Empty list means all checks pass.
    The caller can either (a) abort and re-prompt the LLM with the violation
    list, or (b) proceed to wmake if the list is empty.

    Checks (all mode-agnostic):
      C1: Make/files references a single .C file that exists on disk
      C2: Class declared in .H matches class defined in .C
      C3: Every local `#include "X.H"` in .C/.H resolves to a file on disk
          (either in the same dir, $FOAM_SRC lnInclude dirs, or recon paths)
      C4: Every `-I<path>` in Make/options points to an existing directory
          (expanding $(LIB_SRC) and $(WM_PROJECT_DIR))
    """
    violations: List[str] = []
    root = case_path / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    files_path = root / "Make" / "files"
    opts_path = root / "Make" / "options"
    foam_src = os.environ.get("WM_PROJECT_DIR", "")
    lib_src = f"{foam_src}/src" if foam_src else ""

    # C1: Make/files points to a real .C
    if files_path.is_file():
        try:
            mf = files_path.read_text(encoding="utf-8", errors="ignore")
            listed = re.findall(r"^\s*([A-Za-z_][\w/.\-]*\.C)\s*$", mf, re.MULTILINE)
            for src in listed:
                src_path = (root / src).resolve()
                if not src_path.is_file():
                    violations.append(
                        f"static_check_C1: Make/files lists `{src}` but no such file at `{src_path}`"
                    )
        except Exception:
            pass
    else:
        violations.append("static_check_C1: Make/files missing")

    # C2: class declared vs defined match. Tolerant of templated syntax
    # like `ClassName<TemplArgs>::method(...)` which is standard OpenFOAM.
    if h_path.is_file() and c_path.is_file():
        try:
            h_txt = h_path.read_text(encoding="utf-8", errors="ignore")
            c_txt = c_path.read_text(encoding="utf-8", errors="ignore")
            decls = set(re.findall(r"\bclass\s+([A-Za-z_]\w*)\b", h_txt))
            # Match ClassName::foo, ClassName<T>::foo, ClassName<T,U,V>::foo
            defs = set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:<[^>]{0,200}>)?\s*::", c_txt))
            if class_name in decls and class_name not in defs:
                violations.append(
                    f"static_check_C2: class `{class_name}` declared in .H but no "
                    f"`{class_name}::` implementation in .C"
                )
        except Exception:
            pass

    # C3: local #includes resolve somewhere plausible
    recon_files = []
    if isinstance(recon_ctx, dict):
        for f in (recon_ctx.get("selected_files") or []):
            if isinstance(f, dict) and f.get("rel"):
                recon_files.append(f["rel"])
    foam_lninclude_search_dirs: List[Path] = []
    if lib_src:
        # Only probe common lnInclude subdirs; walking all of $LIB_SRC is slow.
        for sub in Path(lib_src).glob("*/lnInclude"):
            if sub.is_dir():
                foam_lninclude_search_dirs.append(sub)
            # Also probe one more level down
        for sub2 in Path(lib_src).glob("*/*/lnInclude"):
            if sub2.is_dir():
                foam_lninclude_search_dirs.append(sub2)

    for src_path in (h_path, c_path):
        if not src_path.is_file():
            continue
        try:
            txt = src_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r'#\s*include\s+"([\w./\-]+)"', txt):
            inc = m.group(1)
            # Check same dir, recon-selected files, then lnInclude dirs
            if (root / inc).is_file():
                continue
            if any(inc == Path(f).name for f in recon_files):
                continue
            found = False
            for d in foam_lninclude_search_dirs:
                if (d / Path(inc).name).is_file():
                    found = True; break
            if not found:
                violations.append(
                    f"static_check_C3: `#include \"{inc}\"` in {src_path.name} "
                    f"does not resolve to any known location"
                )

    # C4: Make/options -I paths exist
    if opts_path.is_file() and foam_src:
        try:
            opts = opts_path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"-I\s*([^\s\\]+)", opts):
                raw = m.group(1)
                expanded = (
                    raw.replace("$(LIB_SRC)", lib_src)
                       .replace("$(WM_PROJECT_DIR)", foam_src)
                )
                if expanded.startswith("/") and not Path(expanded).exists():
                    violations.append(
                        f"static_check_C4: include path `{raw}` -> `{expanded}` does not exist"
                    )
        except Exception:
            pass

    return violations


def _find_parent_source_pair(
    parent_class: str, recon_ctx: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[Path, Path]]:
    """Locate a parent class's (.H, .C) pair in $FOAM_SRC. Generic."""
    foam_src = os.environ.get("WM_PROJECT_DIR", "")
    if not foam_src or not parent_class:
        return None
    lib_src = Path(foam_src) / "src"
    if not lib_src.is_dir():
        return None
    # Prefer recon-selected files if any match
    if isinstance(recon_ctx, dict):
        for f in (recon_ctx.get("selected_files") or []):
            if not isinstance(f, dict):
                continue
            rel = f.get("rel", "")
            if rel.endswith(f"/{parent_class}.H"):
                h = (lib_src / rel).resolve()
                c = h.with_suffix(".C")
                if h.is_file() and c.is_file():
                    return h, c
    # Otherwise scan $LIB_SRC for matching files
    try:
        h_candidates = list(lib_src.rglob(f"{parent_class}.H"))
        h_candidates = [p for p in h_candidates if "/lnInclude/" not in str(p)]
        for h in h_candidates:
            c = h.with_suffix(".C")
            if c.is_file():
                return h.resolve(), c.resolve()
    except Exception:
        pass
    return None


def _clone_parent_skeleton(
    case_path: Path,
    class_name: str,
    parent_class: str,
    recon_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Clone a parent class's .H/.C into <case>/customModels/<class_name>/,
    rename identifiers, and return a summary dict. Returns None if no parent
    source can be located or if the clone fails. Generic — works for any
    parent class that has both a .H and .C in $FOAM_SRC.
    """
    pair = _find_parent_source_pair(parent_class, recon_ctx)
    if pair is None:
        return None
    parent_h, parent_c = pair
    dest = case_path / "customModels" / class_name
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "Make").mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return None

    try:
        h_txt = parent_h.read_text(encoding="utf-8", errors="ignore")
        c_txt = parent_c.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Rename class identifier occurrences. Conservative: only replace whole-
    # word matches of the exact parent class name. Preserves #include paths
    # that happen to contain the name as a substring — those are intentional
    # references back to the parent API.
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(parent_class)}(?![A-Za-z0-9_])")

    # But DON'T rename within #include "ParentClass.H" or <ParentClass.H> —
    # those pull the parent's type definitions; we want to keep them so the
    # new derived class still sees the parent.
    def _rename_preserving_includes(src: str) -> str:
        lines: List[str] = []
        for line in src.splitlines(keepends=True):
            if re.match(r'^\s*#\s*include\s+[<"]', line):
                lines.append(line)
            else:
                lines.append(pattern.sub(class_name, line))
        return "".join(lines)

    new_h = _rename_preserving_includes(h_txt)
    new_c = _rename_preserving_includes(c_txt)

    # Add an include to the parent header so the derived class actually has
    # access to its type (belt-and-suspenders; usually the cloned file
    # already includes it).
    parent_header_include = f'#include "{parent_class}.H"'
    if parent_header_include not in new_h:
        insert_pos = new_h.find("#include")
        if insert_pos != -1:
            new_h = new_h[:insert_pos] + parent_header_include + "\n" + new_h[insert_pos:]
        else:
            new_h = parent_header_include + "\n" + new_h

    h_dst = dest / f"{class_name}.H"
    c_dst = dest / f"{class_name}.C"
    h_dst.write_text(new_h, encoding="utf-8")
    c_dst.write_text(new_c, encoding="utf-8")

    # Write canonical Make/files
    (dest / "Make" / "files").write_text(
        f"{class_name}.C\n\nLIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}\n",
        encoding="utf-8",
    )
    return {
        "class_name": class_name,
        "parent_class": parent_class,
        "parent_h_source": str(parent_h),
        "parent_c_source": str(parent_c),
        "dest_h": str(h_dst),
        "dest_c": str(c_dst),
        "dest_make_files": str(dest / "Make" / "files"),
        "note": "Parent skeleton cloned. Only Make/options + equation-delta patch needed next.",
    }


# ---------------------------------------------------------------------------
# Deterministic compile-error pattern fixers (run BEFORE the LLM retry).
#
# Generic across all code-mod modes. Each rule inspects the compile log, the
# current on-disk files, and — if it recognizes the pattern — applies a
# minimal, targeted fix and returns a short tag describing what it did.
# If a rule doesn't match, it returns None and the next rule is tried.
#
# These are the kinds of one-line mistakes that LLM code-gen keeps making;
# catching them here avoids spending an LLM call (and tokens) on a trivially
# mechanical repair.
# ---------------------------------------------------------------------------

def _deterministic_compile_autofixes(
    case_path: Path, class_name: str, err_text: str, mode: str
) -> List[str]:
    """Apply mechanical fixes for common wmake compile errors. Returns list of tags."""
    applied: List[str] = []
    root = case_path / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    files_path = root / "Make" / "files"
    opts_path = root / "Make" / "options"

    def read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    # --- Rule A: "No rule to make target 'Make/.../<FooBar>.C.dep'" ---------
    # The LLM put the wrong .C filename in Make/files. Canonical sanitizer
    # already rewrites to <class_name>.C when it detects mismatch; re-run it
    # here to be safe.
    m_norule = re.search(
        r"No rule to make target '.*?/([A-Za-z_][A-Za-z0-9_]*)\.C\.dep'",
        err_text,
    )
    if m_norule:
        bad_stem = m_norule.group(1)
        if bad_stem != class_name and files_path.is_file():
            current = read(files_path)
            fixed = _fix_make_files_wmake_layout(current, class_name)
            if fixed != current:
                _safe_write_local(case_path, class_name, files_path, fixed)
                applied.append(f"rewrote_Make/files_bad_stem={bad_stem}")

    # --- Rule B: "#include "<Header>.H"" for a file that does not exist -----
    # Remove the broken include line from .C / .H so compilation can proceed.
    # Does NOT invent a replacement; the LLM can add a correct one on retry.
    m_missing = re.findall(
        r"fatal error:\s+([\w/.\-]+\.H):\s+No such file or directory",
        err_text,
    )
    if m_missing:
        missing_headers = {Path(h).name for h in m_missing}
        for src_path in (h_path, c_path):
            if not src_path.is_file():
                continue
            txt = read(src_path)
            new_lines: List[str] = []
            changed = False
            for line in txt.splitlines(keepends=True):
                stripped = line.strip()
                m_inc = re.match(r'#\s*include\s+[<"]([\w/.\-]+\.H)[>"]', stripped)
                if m_inc and Path(m_inc.group(1)).name in missing_headers:
                    changed = True
                    continue
                new_lines.append(line)
            if changed:
                _safe_write_local(case_path, class_name, src_path, "".join(new_lines))
                applied.append(f"stripped_missing_includes_from_{src_path.name}")

    # --- Rule C: "LIB_SRCS" (ESI-style) present in Make/files --------------
    # Existing _fix_make_files_wmake_layout already handles this, but only
    # runs inside _enforce_case_local_make_files. Run it here too so any
    # manual rewrites during LLM retry get normalized before wmake.
    if files_path.is_file():
        current = read(files_path)
        fixed = _fix_make_files_wmake_layout(current, class_name)
        fixed = _normalize_make_files_lib_path(fixed, class_name)
        if fixed != current:
            _safe_write_local(case_path, class_name, files_path, fixed)
            applied.append("normalized_Make/files_canonical")

    # --- Rule D: stray C-style block comments in Make/files or Make/options -
    # wmake's makefile parser rejects /* ... */ comments; LLM sometimes adds
    # them. Strip them. (Generic — any mode.)
    for mk_path in (files_path, opts_path):
        if not mk_path.is_file():
            continue
        current = read(mk_path)
        # Only act if /* or */ appears
        if "/*" in current or "*/" in current:
            stripped = re.sub(r"/\*.*?\*/", "", current, flags=re.DOTALL)
            if stripped != current:
                _safe_write_local(case_path, class_name, mk_path, stripped)
                applied.append(f"stripped_C_block_comments_from_{mk_path.name}")

    return applied


def _llm_compile_review_fix(
    case_path: Path,
    class_name: str,
    mode: str,
    build: Dict[str, Any],
    err_text: str,
    attempt_history: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    repo_root = Path(__file__).resolve().parent.parent
    _bootstrap_paths(repo_root)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
    except Exception:
        return []

    root = case_path / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    files_path = root / "Make" / "files"
    opts_path = root / "Make" / "options"
    cur = {
        "header": h_path.read_text(encoding="utf-8", errors="ignore") if h_path.exists() else "",
        "source": c_path.read_text(encoding="utf-8", errors="ignore") if c_path.exists() else "",
        "make_files": files_path.read_text(encoding="utf-8", errors="ignore") if files_path.exists() else "",
        "make_options": opts_path.read_text(encoding="utf-8", errors="ignore") if opts_path.exists() else "",
    }
    payload = _load_payload_for_context(case_path)
    ref_ctx = (((payload.get("openfoam_api_context") or {}) if isinstance(payload, dict) else {}).get("openfoam_reference_context"))
    if not isinstance(ref_ctx, dict):
        ref_ctx = {}

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0, effort="low")
    base_prompt = {
        "class_name": class_name,
        "mode": mode,
        "build_normalized_spec": build.get("normalized_spec", {}),
        "compile_error": err_text[-12000:],
        "current_files": cur,
        "reference_context": ref_ctx,
    }
    # Phase 1: ask if more reference files are needed.
    req_raw = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are reviewing OpenFOAM compile failures. "
                    "Return strict JSON with keys: needs_more_refs(bool), reference_paths(array), rationale(string). "
                    "reference_paths must be under $WM_PROJECT_DIR and are read-only references."
                )
            ),
            HumanMessage(content=json.dumps(base_prompt, indent=2)[:42000]),
        ]
    )
    req_txt = getattr(req_raw, "content", "") if req_raw else ""
    req_clean = strip_json_fences(req_txt if isinstance(req_txt, str) else str(req_txt))
    s, e = req_clean.find("{"), req_clean.rfind("}")
    if s != -1 and e != -1 and e > s:
        req_clean = req_clean[s : e + 1]
    extra_refs: List[Dict[str, str]] = []
    try:
        req_obj = json.loads(req_clean)
        if isinstance(req_obj, dict) and req_obj.get("needs_more_refs") is True:
            paths = req_obj.get("reference_paths", [])
            if isinstance(paths, list):
                extra_refs = _read_openfoam_refs_from_paths([str(p) for p in paths])
    except Exception:
        pass

    # Patch-only schema: LLM returns ONLY the files that need to change, as a
    # `files_to_edit` array of {path, new_contents}. This is the key
    # structural change — instead of regenerating all four files (which lets
    # the LLM introduce new bugs in unrelated files on every retry), we
    # explicitly ask it to patch surgically. Simulates what an agent session
    # does naturally, via plain chat-completions messages — works with any
    # provider (Anthropic / OpenAI / Bedrock / Codex-OAuth).
    #
    # attempt_history is the list of prior (error, fixes_tried) pairs so the
    # LLM can see the trajectory of failures and stop repeating mistakes.
    # filename_map accepts both the short bare form and the full relative /
    # absolute form that the LLM may emit — the key is just an identifier,
    # the target path is what actually gets written.
    model_rel = f"customModels/{class_name}"
    filename_map = {
        # Short forms (encourage LLM to use these in prompts)
        f"{class_name}.H": h_path,
        f"{class_name}.C": c_path,
        "Make/files": files_path,
        "Make/options": opts_path,
        # Relative-to-case forms (what the LLM often emits in practice)
        f"{model_rel}/{class_name}.H": h_path,
        f"{model_rel}/{class_name}.C": c_path,
        f"{model_rel}/Make/files": files_path,
        f"{model_rel}/Make/options": opts_path,
        # Absolute forms (belt-and-suspenders)
        str(h_path): h_path,
        str(c_path): c_path,
        str(files_path): files_path,
        str(opts_path): opts_path,
    }
    history_summary: List[Dict[str, Any]] = []
    for h in (attempt_history or [])[-4:]:  # cap: last 4 attempts in context
        if isinstance(h, dict):
            history_summary.append({
                "attempt": h.get("attempt"),
                "error_tail": str(h.get("error", ""))[-800:],
                "fixes_tried": h.get("fixes_tried", []),
            })

    # Tier-classify the build error so the LLM prompt carries focused per-tier
    # coaching rather than only the raw stderr blob. Generic across modification
    # kinds — the classifier reads compiler output, makes no physics assumptions.
    tier_block: Dict[str, Any] = {}
    try:
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import compile_error_classifier as _cec  # type: ignore
        tier_block = _cec.classify(err_text, "")
    except Exception:
        tier_block = {}

    fix_prompt = {
        "class_name": class_name,
        "mode": mode,
        "build_normalized_spec": build.get("normalized_spec", {}),
        "compile_error": err_text[-8000:],
        "current_files": cur,
        "reference_context": ref_ctx,
        "extra_references": extra_refs,
        "attempt_history": history_summary,
        "valid_patch_targets": list(filename_map.keys()),
        "build_error_tier": {
            "tier": tier_block.get("tier"),
            "tier_label": tier_block.get("tier_label"),
            "key_messages": tier_block.get("key_messages", []),
            "missing_headers": tier_block.get("missing_headers", []),
            "undeclared_identifiers": tier_block.get("undeclared", []),
            "undefined_refs": tier_block.get("undefined_refs", []),
            "missing_libs": tier_block.get("missing_libs", []),
            "make_files_csources": tier_block.get("make_files_csources", []),
            "tier_coaching": tier_block.get("coaching", ""),
        } if tier_block else {},
    }
    fix_raw = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a BUILD ENGINEER patching OpenFOAM custom-model files to fix a wmake "
                    "compile error. You are NOT a researcher — DO NOT change the physics, the "
                    "equation modification, or the underlying hypothesis. Your only job is to make "
                    "the existing physics implementation compile.\n\n"
                    "Use the per-tier coaching in `build_error_tier.tier_coaching` for the most\n"
                    "common cause of this error class. Apply the SMALLEST possible fix.\n\n"
                    "You see the current file contents on disk; patch surgically, do NOT regenerate "
                    "files that are not the direct source of the error. "
                    "\n\nPREFER returning unified-diff patches over full file contents. Diffs prevent "
                    "regression on lines unrelated to the error and make your patch much smaller. "
                    "\n\nReturn strict JSON with ONE of these two schemas:"
                    "\n\n  A) PREFERRED — diff form (minimum-viable patch):"
                    "\n  {\"patches\":[{\"path\":\"<one of valid_patch_targets>\","
                    "\"unified_diff\":\"<standard unified-diff hunk>\"}],"
                    "\"notes\":[\"<short reason>\"]}"
                    "\n\n  B) FALLBACK — full-file form (use only if a diff is not feasible):"
                    "\n  {\"files_to_edit\":[{\"path\":\"<one of valid_patch_targets>\","
                    "\"new_contents\":\"<full corrected file content>\"}],"
                    "\"notes\":[\"<short reason>\"]}"
                    "\n\nRules (apply to both schemas):"
                    "\n(1) Only include a file if its contents must change; leave untouched files out entirely."
                    "\n(2) Use ONLY the bare path forms listed in `valid_patch_targets` — "
                    "exactly `Make/options`, `Make/files`, `<class_name>.H`, `<class_name>.C`. "
                    "Do NOT prefix with `customModels/<class>/` or absolute paths."
                    "\n(3) `Make/files` and `Make/options` are wmake makefile fragments — no /* */ "
                    "block comments, no LIB_SRCS."
                    "\n(4) The `.C` source filename MUST match `<class_name>.C` exactly — do not rename it."
                    "\n(5) If prior attempts (see attempt_history) already tried the same fix, try a "
                    "DIFFERENT approach this time, not the same one."
                    "\n(6) MINIMUM-VIABLE-PATCH: return the smallest change that plausibly resolves the "
                    "error. Do not reorganize files. Do not refactor. Do not add comments. Do not touch "
                    "lines outside the direct vicinity of the error."
                    "\n(7) For diff form: use standard unified-diff syntax with `--- a/<path>` / `+++ b/<path>` "
                    "headers and `@@ -old,N +new,M @@` hunk markers. The `<path>` in the headers should "
                    "match the bare path form (e.g. `<class_name>.C`)."
                )
            ),
            HumanMessage(content=json.dumps(fix_prompt, indent=2)[:60000]),
        ]
    )
    fix_txt = getattr(fix_raw, "content", "") if fix_raw else ""
    fix_clean = strip_json_fences(fix_txt if isinstance(fix_txt, str) else str(fix_txt))
    s2, e2 = fix_clean.find("{"), fix_clean.rfind("}")
    if s2 != -1 and e2 != -1 and e2 > s2:
        fix_clean = fix_clean[s2 : e2 + 1]
    fix_clean = _fix_json_string_literals(fix_clean)
    try:
        obj = json.loads(fix_clean)
    except Exception:
        return ["llm_fix_parse_failed"]
    if not isinstance(obj, dict):
        return ["llm_fix_not_object"]
    notes: List[str] = []
    if isinstance(obj.get("notes"), list):
        notes = [str(x)[:300] for x in obj.get("notes", [])[:8]]
    # Schema A (preferred): unified-diff patches
    patches = obj.get("patches") or []
    if isinstance(patches, list):
        for patch_obj in patches:
            if not isinstance(patch_obj, dict):
                continue
            rel = str(patch_obj.get("path", "")).strip()
            diff_txt = str(patch_obj.get("unified_diff", "") or "")
            if not rel or not diff_txt.strip():
                continue
            # Resolve path with the same normalization as schema B
            target = filename_map.get(rel)
            if target is None:
                rel_norm = rel.lstrip("./").lstrip("/")
                target = filename_map.get(rel_norm)
            if target is None:
                base = rel.split("/")[-1] if "/" in rel else rel
                for key, t in filename_map.items():
                    if key.endswith(rel) or key.endswith(f"/{base}") or rel.endswith(f"/{key}"):
                        target = t
                        break
            if target is None:
                notes.append(f"llm_diff_rejected_unknown_path:{rel}")
                continue
            ok, err = _apply_unified_diff_to_file(target, diff_txt)
            if ok:
                notes.append(f"llm_diff_applied:{rel}")
            else:
                notes.append(f"llm_diff_apply_failed:{rel}:{err[:100]}")

    # Schema B (fallback): files_to_edit with full new_contents
    edits = obj.get("files_to_edit") or []
    if not isinstance(edits, list):
        edits = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        rel = str(edit.get("path", "")).strip()
        txt = str(edit.get("new_contents", "") or "")
        if not rel or not txt.strip():
            continue
        # Accept multiple path forms the LLM may emit. Try direct lookup,
        # then try stripping common prefixes, then try suffix-match against
        # the whitelist keys (covers any remaining variations).
        target = filename_map.get(rel)
        if target is None:
            rel_norm = rel.lstrip("./").lstrip("/")
            target = filename_map.get(rel_norm)
        if target is None:
            # Suffix-match: LLM might emit e.g. "CustomX/Make/options" — find
            # the whitelist key that ends with the same tail. Safe because
            # whitelist keys are all short canonical forms.
            base = rel.split("/")[-1] if "/" in rel else rel
            for key, t in filename_map.items():
                if key.endswith(rel) or key.endswith(f"/{base}") or rel.endswith(f"/{key}"):
                    target = t
                    break
        if target is None:
            notes.append(f"llm_rejected_unknown_path:{rel}")
            continue
        if rel in ("Make/files", "Make/options"):
            txt = _sanitize_wmake_make_fragment(txt)
            if rel == "Make/files":
                txt = _fix_make_files_wmake_layout(txt, class_name)
                txt = _normalize_make_files_lib_path(txt, class_name)
        _safe_write_local(case_path, class_name, target, txt)
        notes.append(f"llm_patched:{rel}")
    if extra_refs:
        notes.append(f"llm_read_extra_refs={len(extra_refs)}")
    return notes


def _fix_json_string_literals(txt: str) -> str:
    """Escape literal newlines/tabs inside JSON string values that the LLM forgot to escape."""
    result: List[str] = []
    in_string = False
    i = 0
    while i < len(txt):
        c = txt[i]
        if c == "\\" and in_string:
            result.append(c)
            i += 1
            if i < len(txt):
                result.append(txt[i])
                i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif c == "\n" and in_string:
            result.append("\\n")
        elif c == "\r" and in_string:
            result.append("\\r")
        elif c == "\t" and in_string:
            result.append("\\t")
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _llm_openfoam_custom_files_bundle(
    class_name: str,
    mode: str,
    build: Dict[str, Any],
    payload: Dict[str, Any],
    repo_root: Path,
    retry_hint: str = "",
) -> Tuple[str, str, str, str]:
    """
    Single LLM call: full header, source, Make/files, Make/options for case-local wmake libso.
    """
    _bootstrap_paths(repo_root)
    h_txt = c_txt = files_txt = opts_txt = ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore

        settings = get_settings()
        llm = create_langchain_llm(model=settings.model, temperature=0.0, effort="low")
        api_ctx = (payload.get("openfoam_api_context") or {}) if isinstance(payload, dict) else {}
        ref_ctx = api_ctx.get("openfoam_reference_context") if isinstance(api_ctx, dict) else {}
        recon_ctx = api_ctx.get("recon_reference_context") if isinstance(api_ctx, dict) else None
        prompt = {
            "class_name": class_name,
            "mode": mode,
            "normalized_spec": build.get("normalized_spec", {}),
            "activation_dictionary": build.get("activation_dictionary", ""),
            "reference_context": ref_ctx if isinstance(ref_ctx, dict) else {},
            "recon_reference_context": recon_ctx if isinstance(recon_ctx, dict) else None,
        }
        sys_extra = ""
        if mode == "custom_turbulence_model_modification":
            sys_extra = (
                " OpenFOAM 10 incompressible RAS turbulence libraries MUST: "
                "(1) use Make/options EXE_INC/LIB_LIBS with "
                "-I$(LIB_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude, "
                "-I$(LIB_SRC)/MomentumTransportModels/incompressible/lnInclude, "
                "-lincompressibleMomentumTransportModels, -lmomentumTransportModels "
                "(never use src/TurbulenceModels/turbulenceModels or -lturbulenceModels). "
                "(2) Do not #include a wrong header name like IncompressibleMomentumTransportModel.H "
                "(use patterns from $WM_PROJECT_DIR/src/MomentumTransportModels/.../SpalartAllmaras). "
                "(3) Do not put #include \"ClassName.C\" inside the .H under #ifdef NoRepository when "
                "Make/files compiles ClassName.C separately. "
                "(4) For SpalartAllmaras-derived models, the base constructor argument order is "
                "(alpha, rho, U, alphaRhoPhi, phi, viscosity, type) — not type first. "
                "(5) At the end of the .C file after namespace closing braces, register with "
                "#include \"makeIncompressibleMomentumTransportModel.H\" then makeRASModel(ClassName); "
                "— not makeIncompressibleMomentumTransportModelTypes / makeTemplatedIncompressibleMomentumTransportModel."
            )
        recon_clamp = ""
        if isinstance(recon_ctx, dict) and recon_ctx.get("verified_include_paths"):
            vpaths = recon_ctx.get("verified_include_paths") or []
            selected = recon_ctx.get("selected_files") or []
            rel_list = [f.get("rel", "") for f in selected if isinstance(f, dict)]
            recon_clamp = (
                " ANTI-HALLUCINATION CLAMP (generic, non-negotiable for all code-mod modes): "
                "A reconnaissance step has verified which OpenFOAM source files and include "
                "directories actually exist in this installation. "
                "Your Make/options EXE_INC MUST use ONLY include directories drawn from "
                "`recon_reference_context.verified_include_paths[].make_options_form`; do not "
                "invent additional `-I` entries beyond those and the handful of standard OpenFOAM "
                "core dirs needed (finiteVolume, meshTools, OpenFOAM, OSspecific/POSIX if relevant). "
                "Your header #includes MUST correspond to files that exist — prefer files named in "
                "`recon_reference_context.selected_files[].rel` and class names named in "
                "`recon_reference_context.class_signatures`. If the recon selected a Make/options "
                "sample from the parent library subtree, mirror its EXE_INC / LIB_LIBS structure. "
                f"Verified include paths from recon (use these, not ESI/foam-extend layouts): "
                f"{json.dumps(vpaths, default=str)[:3000]}. "
                f"Selected source files from recon: {json.dumps(rel_list)[:2000]}."
            )
        messages = [
            SystemMessage(
                content=(
                    "Generate OpenFOAM10 custom model files as strict JSON with keys: "
                    "header, source, make_files, make_options, optional needs_more_refs(array of paths). "
                    "No placeholders. Must compile with wmake libso in user case customModels. "
                    "make_files and make_options are wmake makefile fragments: valid makefile syntax only; "
                    "do not use C-style /* */ block comments (use # line comments only if needed). "
                    f"For make_files: put EXACTLY `{class_name}.C` on its own line (the builder writes "
                    f"the source as `<case>/customModels/{class_name}/{class_name}.C` — DO NOT name any "
                    f"other .C file here, not even your parent class or the declared `TypeName`; "
                    f"the filename must match the builder's on-disk stem), then a blank line, then "
                    f"`LIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}` — never use LIB_SRCS, "
                    f"never rename lib{class_name}, never list multiple .C files. "
                    f"Mode {mode!r}: pick correct base classes, includes, and LIB_LIBS for that mode "
                    "(viscosity, turbulence, fvModel, flux/BC, functionObject, …). "
                    "OPTIONAL: if the reference_context you were given is missing a header or class you "
                    "need to implement correctly, instead of guessing, return "
                    "{\"needs_more_refs\": [\"abs_or_LIB_SRC_relative_path_1\", ...]} with the paths to "
                    "additional OpenFOAM source files (under $WM_PROJECT_DIR/src) the orchestrator "
                    "should fetch for you. On the NEXT call you will see those file contents and can "
                    "emit the complete header/source/make_files/make_options. Paths listed in "
                    "recon_reference_context.selected_files are safe candidates; you may also name "
                    "any file under $WM_PROJECT_DIR/src. Do NOT request files outside that tree."
                    + recon_clamp
                    + sys_extra
                )
            ),
            HumanMessage(content=json.dumps(prompt, indent=2)[:42000]),
        ]
        if retry_hint.strip():
            messages.append(
                HumanMessage(
                    content=(
                        "CORRECTION REQUIRED:\n"
                        f"{retry_hint}\n"
                        "Re-emit the SAME JSON object with non-empty header, source, make_files, and make_options."
                    )
                )
            )
        resp = llm.invoke(messages)
        raw = getattr(resp, "content", "") if resp else ""
        print(f"[CODEMOD-DEBUG] LLM raw response length: {len(raw)}", flush=True)
        print(f"[CODEMOD-DEBUG] LLM raw response (first 500 chars): {raw[:500]!r}", flush=True)
        txt = strip_json_fences(raw if isinstance(raw, str) else str(raw))
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            txt = txt[s : e + 1]
        # Fix literal newlines/tabs inside JSON string values (LLM sometimes emits them unescaped)
        txt = _fix_json_string_literals(txt)
        obj = json.loads(txt)

        # Optional second pass: if the LLM asked for more reference files,
        # fetch them from disk and re-invoke with the extra context. Generic
        # across modes/topics; bounded to one extra round so we don't loop.
        if isinstance(obj, dict):
            needs = obj.get("needs_more_refs")
            if isinstance(needs, list) and needs and not all(obj.get(k, "").strip() for k in ("header", "source", "make_files", "make_options")):
                extra_refs = _read_openfoam_refs_from_paths(
                    [str(p) for p in needs][:8]
                )
                if extra_refs:
                    print(f"[CODEMOD] LLM requested {len(needs)} extra refs; fetched {len(extra_refs)}")
                    messages.append(
                        HumanMessage(
                            content=(
                                "You asked for additional reference files. Here are the contents:\n"
                                + json.dumps(extra_refs, indent=2)[:30000]
                                + "\n\nNow emit the complete header, source, make_files, make_options "
                                "as the JSON contract requires. Do not ask for more refs."
                            )
                        )
                    )
                    resp2 = llm.invoke(messages)
                    raw2 = getattr(resp2, "content", "") if resp2 else ""
                    txt2 = strip_json_fences(raw2 if isinstance(raw2, str) else str(raw2))
                    s2, e2 = txt2.find("{"), txt2.rfind("}")
                    if s2 != -1 and e2 != -1 and e2 > s2:
                        txt2 = txt2[s2 : e2 + 1]
                    txt2 = _fix_json_string_literals(txt2)
                    try:
                        obj = json.loads(txt2)
                    except Exception:
                        pass  # fall through; we'll use the original (empty) obj

        if isinstance(obj, dict):
            h_txt = str(obj.get("header", ""))
            c_txt = str(obj.get("source", ""))
            files_txt = _sanitize_wmake_make_fragment(str(obj.get("make_files", "")))
            opts_txt = _sanitize_wmake_make_fragment(str(obj.get("make_options", "")))
    except Exception as _exc:
        print(f"[CODEMOD-DEBUG] LLM bundle exception: {type(_exc).__name__}: {_exc}", flush=True)
    return h_txt, c_txt, files_txt, opts_txt


def _generate_files(case_path: Path, class_name: str, mode: str, build: Dict[str, Any]) -> Dict[str, str]:
    root = case_path / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    make_files = root / "Make" / "files"
    make_opts = root / "Make" / "options"
    payload = _load_payload_for_context(case_path)
    repo_root = Path(__file__).resolve().parent.parent

    h_txt, c_txt, files_txt, opts_txt = _llm_openfoam_custom_files_bundle(
        class_name, mode, build, payload, repo_root, retry_hint=""
    )
    incomplete = not (h_txt.strip() and c_txt.strip() and files_txt.strip() and opts_txt.strip())
    if incomplete:
        hint = (
            "First response omitted one or more of: header, source, make_files, make_options. "
            "All four must be complete compilable file bodies."
        )
        h2, c2, f2, o2 = _llm_openfoam_custom_files_bundle(
            class_name, mode, build, payload, repo_root, retry_hint=hint
        )
        if h2.strip() and c2.strip() and f2.strip() and o2.strip():
            h_txt, c_txt, files_txt, opts_txt = h2, c2, f2, o2
            incomplete = False

    incomplete = not (h_txt.strip() and c_txt.strip() and files_txt.strip() and opts_txt.strip())
    if incomplete:
        # Reuse existing local files if present; avoids hard fail on transient LLM empty responses.
        existing = {
            "header": h_path.read_text(encoding="utf-8", errors="ignore") if h_path.exists() else "",
            "source": c_path.read_text(encoding="utf-8", errors="ignore") if c_path.exists() else "",
            "make_files": make_files.read_text(encoding="utf-8", errors="ignore") if make_files.exists() else "",
            "make_options": make_opts.read_text(encoding="utf-8", errors="ignore") if make_opts.exists() else "",
        }
        if all(existing[k].strip() for k in ("header", "source", "make_files", "make_options")):
            h_txt = existing["header"]
            c_txt = existing["source"]
            files_txt = existing["make_files"]
            opts_txt = existing["make_options"]
            incomplete = False
    # Deterministic fallback exists only for generalisedNewtonian viscosity — never invent wrong C++ for other modes.
    if incomplete and mode != "custom_viscosity":
        return {
            "_generation_incomplete": True,
            "reason": (
                f"Incomplete LLM output for mode={mode!r}; no automatic C++/Make fallback "
                "(only custom_viscosity has a safe template)."
            ),
            "header": str(h_path),
            "source": str(c_path),
            "make_files": str(make_files),
            "make_options": str(make_opts),
            "mode": mode,
        }

    if incomplete:
        h_txt = f"""#ifndef {class_name}_H
#define {class_name}_H
#include "strainRateViscosityModel.H"
namespace Foam {{
namespace laminarModels {{
namespace generalisedNewtonianViscosityModels {{
class {class_name} : public strainRateViscosityModel {{
    dimensionedScalar nuInf_, k_, n_, gammaDotMin_;
public:
    TypeName("{class_name}");
    {class_name}(const dictionary&, const Foam::viscosity&, const volVectorField&);
    virtual bool read(const dictionary&);
    virtual tmp<volScalarField> nu(const volScalarField&, const volScalarField&) const;
}};
}}}}
#endif
"""
        c_txt = f"""#include "{class_name}.H"
#include "addToRunTimeSelectionTable.H"
namespace Foam {{ namespace laminarModels {{ namespace generalisedNewtonianViscosityModels {{
defineTypeNameAndDebug({class_name}, 0);
addToRunTimeSelectionTable(generalisedNewtonianViscosityModel, {class_name}, dictionary);
}}}}
Foam::laminarModels::generalisedNewtonianViscosityModels::{class_name}::{class_name}
(const dictionary& d, const Foam::viscosity& v, const volVectorField& U)
: strainRateViscosityModel(d, v, U), nuInf_("nuInf", dimViscosity, 0.0), k_("k", dimViscosity, 0.0), n_("n", dimless, 1.0), gammaDotMin_("gammaDotMin", dimless/dimTime, SMALL)
{{ read(d); correct(); }}
bool Foam::laminarModels::generalisedNewtonianViscosityModels::{class_name}::read(const dictionary& d)
{{ strainRateViscosityModel::read(d); const dictionary& c = d.optionalSubDict(typeName + "Coeffs"); nuInf_.read(c); k_.read(c); n_.read(c); gammaDotMin_.read(c); return true; }}
Foam::tmp<Foam::volScalarField> Foam::laminarModels::generalisedNewtonianViscosityModels::{class_name}::nu
(const volScalarField&, const volScalarField& s) const
{{ return nuInf_ + k_*pow(max(s, gammaDotMin_), n_.value() - scalar(1)); }}
"""
        files_txt = _sanitize_wmake_make_fragment(
            f"""{class_name}.C

LIB = ./platforms/$(WM_OPTIONS)/lib/lib{class_name}
"""
        )
        opts_txt = _sanitize_wmake_make_fragment(
            """EXE_INC = \\
    -I$(LIB_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude \\
    -I$(LIB_SRC)/physicalProperties/viscosity/lnInclude \\
    -I$(LIB_SRC)/finiteVolume/lnInclude \\
    -I$(LIB_SRC)/OpenFOAM/lnInclude
LIB_LIBS = \\
    -lmomentumTransportModels \\
    -lfiniteVolume \\
    -lOpenFOAM
"""
        )
    files_txt = _fix_make_files_wmake_layout(files_txt, class_name)
    files_txt = _normalize_make_files_lib_path(files_txt, class_name)
    _safe_write_local(case_path, class_name, h_path, h_txt)
    _safe_write_local(case_path, class_name, c_path, c_txt)
    _safe_write_local(case_path, class_name, make_files, files_txt)
    _safe_write_local(case_path, class_name, make_opts, opts_txt)
    return {
        "header": str(h_path),
        "source": str(c_path),
        "make_files": str(make_files),
        "make_options": str(make_opts),
        "mode": mode,
        "_generation_incomplete": False,
    }


def _run_cmd(cmd: str, cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def _append_if_missing(path: Path, text: str) -> None:
    src = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    if text not in src:
        _write(path, src + ("\n" if src and not src.endswith("\n") else "") + text + "\n")


# OpenFOAM 10: incompressible RAS models link against these (see tutorials / lib sources).
_MAKE_OPTIONS_OF10_INCOMPRESSIBLE_RAS = """EXE_INC = \\
    -I$(LIB_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude \\
    -I$(LIB_SRC)/MomentumTransportModels/incompressible/lnInclude \\
    -I$(LIB_SRC)/finiteVolume/lnInclude \\
    -I$(LIB_SRC)/meshTools/lnInclude \\
    -I$(LIB_SRC)/physicalProperties/lnInclude \\
    -I$(LIB_SRC)/OpenFOAM/lnInclude

LIB_LIBS = \\
    -lincompressibleMomentumTransportModels \\
    -lmomentumTransportModels \\
    -lfiniteVolume \\
    -lmeshTools \\
    -lOpenFOAM
"""


def _make_options_needs_of10_reset(text: str) -> bool:
    """True if LLM/deterministic fixes left invalid OF10 paths (pre-10 layouts) or empty file."""
    if not text.strip():
        return True
    bad = (
        "TurbulenceModels/turbulenceModels",
        "-lturbulenceModels",
        "/src/transportModels/",
    )
    return any(b in text for b in bad)


def _strip_bad_make_options_lines(text: str) -> str:
    """Remove known-bad include paths and orphan flags from Make/options."""
    out_lines: List[str] = []
    for ln in text.splitlines():
        if "TurbulenceModels/turbulenceModels" in ln:
            continue
        if ln.strip() == "-lturbulenceModels" or ln.rstrip().endswith(" -lturbulenceModels"):
            continue
        if "/src/transportModels/" in ln and "MomentumTransportModels" not in ln:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).rstrip() + ("\n" if out_lines else "")


def _sanitize_custom_turbulence_model_of10(
    case_dir: Path,
    class_name: str,
    parent_model: str,
) -> List[str]:
    """
    Deterministic OpenFOAM 10 fixes for LLM-generated custom *incompressible* RAS models.
    Fixes recurring failures: wrong header names, NoRepository .C include, wrong ctor order,
    and invalid registration macros for makeRASModel.
    """
    notes: List[str] = []
    root = case_dir / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    opts_path = root / "Make" / "options"
    if not h_path.is_file() or not c_path.is_file():
        return notes

    parent = (parent_model or "").strip()
    if parent not in ("SpalartAllmaras", "kEpsilon", "RNGkEpsilon", "realizableKE", "kOmega", "kOmegaSST"):
        return notes

    h_txt = h_path.read_text(encoding="utf-8", errors="ignore")
    h2 = re.sub(
        rf"\n#ifdef\s+NoRepository\s*\n\s*#\s*include\s+\"{re.escape(class_name)}\.C\"\s*\n#endif\s*",
        "\n",
        h_txt,
        flags=re.MULTILINE,
    )
    if h2 != h_txt:
        _write(h_path, h2)
        notes.append("removed_NoRepository_include_from_header")

    c_txt = c_path.read_text(encoding="utf-8", errors="ignore")
    c2 = c_txt
    stripped_any = False
    for bad in (
        '#include "IncompressibleMomentumTransportModel.H"',
        "#include \"IncompressibleMomentumTransportModel.H\"",
        '#include "transportModel.H"',
        '#include "makeRASModel.H"',
    ):
        if bad in c2:
            c2 = c2.replace(bad, "")
            stripped_any = True
    if stripped_any:
        notes.append("stripped_invalid_transport_includes_from_source")

    # SpalartAllmaras base ctor: OF10 order is (alpha, rho, U, alphaRhoPhi, phi, viscosity, type)
    if parent == "SpalartAllmaras":
        wrong_ctor = re.compile(
            r"(SpalartAllmaras\s*<\s*[^>]+>\s*\()\s*\n\s*type\s*,\s*\n\s*alpha\s*,\s*\n\s*rho\s*,\s*\n\s*U\s*,\s*\n\s*alphaRhoPhi\s*,\s*\n\s*phi\s*,\s*\n\s*viscosity\s*\)",
            re.MULTILINE,
        )
        if wrong_ctor.search(c2):
            c2 = wrong_ctor.sub(
                r"\1\n        alpha,\n        rho,\n        U,\n        alphaRhoPhi,\n        phi,\n        viscosity,\n        type",
                c2,
                count=1,
            )
            notes.append("fixed_SpalartAllmaras_base_ctor_argument_order")

    macro_tail = (
        '\n#include "makeIncompressibleMomentumTransportModel.H"\n\n'
        f"makeRASModel({class_name});\n"
    )
    # Remove LLM hallucinated registration macros (invalid on OF10)
    c2_before = c2
    c2 = re.sub(
        r"\n#include\s+\"makeIncompressibleMomentumTransportModel\.H\"\s*\n\n"
        r"makeIncompressibleMomentumTransportModelTypes\s*\([^)]*\)\s*;\s*\n\n"
        r"makeTemplatedIncompressibleMomentumTransportModel\s*\([^)]*\)\s*;\s*",
        "\n",
        c2,
        flags=re.DOTALL,
    )
    if c2 != c2_before:
        notes.append("removed_bogus_momentum_transport_registration_macros")

    if f"makeRASModel({class_name})" not in c2:
        c2 = c2.rstrip() + macro_tail
        notes.append("appended_makeRASModel_registration")

    if c2 != c_txt:
        _write(c_path, c2)

    if opts_path.is_file():
        raw = opts_path.read_text(encoding="utf-8", errors="ignore")
        stripped = _strip_bad_make_options_lines(raw)
        if _make_options_needs_of10_reset(stripped):
            _write(opts_path, _sanitize_wmake_make_fragment(_MAKE_OPTIONS_OF10_INCOMPRESSIBLE_RAS))
            notes.append("reset_Make/options_to_OF10_incompressible_RAS_template")
        elif stripped != raw:
            _write(opts_path, _sanitize_wmake_make_fragment(stripped))
            notes.append("stripped_invalid_entries_from_Make/options")

    return notes


def _rewrite_tmp_tensor_v_access(c_path: Path) -> List[str]:
    """
    Fix common OF10 compile error:
      tmp<...> has no member named 'v'
    caused by generated expressions like symm(fvc::grad(U)()).v()
    """
    if not c_path.is_file():
        return []
    txt = c_path.read_text(encoding="utf-8", errors="ignore")
    new = txt
    # Handle both symm(...) and skew(...) patterns on tmp gradients.
    replacements = (
        (r"symm\(\s*fvc::grad\(([^)]+)\)\(\)\s*\)\.v\(\)", r"symm(fvc::grad(\1)())"),
        (r"skew\(\s*fvc::grad\(([^)]+)\)\(\)\s*\)\.v\(\)", r"skew(fvc::grad(\1)())"),
    )
    changed = False
    for pat, repl in replacements:
        upd = re.sub(pat, repl, new)
        if upd != new:
            changed = True
            new = upd
    if changed:
        _write(c_path, new)
        return ["rewrote_tmp_tensor_v_access_in_source"]
    return []


def _apply_compile_fixes(case_dir: Path, class_name: str, err_text: str, mode: str = "") -> List[str]:
    """
    Deterministic compile-fix reviewer loop (non-LLM).
    Applies common OpenFOAM C++ build fixes and returns fix notes.
    """
    fixes: List[str] = []
    root = case_dir / "customModels" / class_name
    h_path = root / f"{class_name}.H"
    c_path = root / f"{class_name}.C"
    opts_path = root / "Make" / "options"
    err = (err_text or "").lower()

    # Missing include-style symbol errors.
    if "dimensionedscalar" in err:
        _append_if_missing(h_path, '#include "dimensionedScalar.H"')
        fixes.append("added include dimensionedScalar.H")
    if "addtoruntimeselectiontable" in err:
        _append_if_missing(c_path, '#include "addToRunTimeSelectionTable.H"')
        fixes.append("added include addToRunTimeSelectionTable.H")
    if "fvc::grad" in err or "symm" in err:
        _append_if_missing(c_path, '#include "fvcGrad.H"')
        fixes.append("added include fvcGrad.H")
    if "viscositymodel" in err:
        _append_if_missing(h_path, '#include "viscosityModel.H"')
        fixes.append("added include viscosityModel.H")
    if "fvmodel" in err:
        _append_if_missing(h_path, '#include "fvModel.H"')
        fixes.append("added include fvModel.H")

    # Namespace resolution issues.
    if "was not declared in this scope" in err or "unknown type name" in err:
        c_txt = c_path.read_text(encoding="utf-8", errors="ignore") if c_path.exists() else ""
        if "using namespace Foam;" not in c_txt:
            _write(c_path, "using namespace Foam;\n" + c_txt)
            fixes.append("added using namespace Foam; in source")

    # Missing headers: never append bare -I/-l lines to Make/options (breaks wmake; OF10 uses
    # MomentumTransportModels, not src/TurbulenceModels/turbulenceModels). For turbulence libs,
    # reset to a known-good OF10 incompressible RAS template when needed.
    if any(k in err for k in ["fatal error:", "no such file or directory", "cannot find"]):
        # OF10 compatibility: some generated sources still include makeRASModel.H,
        # but OF10 registration uses makeIncompressibleMomentumTransportModel.H.
        if "makerasmodel.h" in err and c_path.is_file():
            raw_c = c_path.read_text(encoding="utf-8", errors="ignore")
            fixed_c = raw_c.replace('#include "makeRASModel.H"\n', "")
            if '#include "makeIncompressibleMomentumTransportModel.H"' not in fixed_c:
                fixed_c = (
                    '#include "makeIncompressibleMomentumTransportModel.H"\n'
                    + fixed_c
                )
            if fixed_c != raw_c:
                _write(c_path, fixed_c)
                fixes.append("replaced_makeRASModel_header_with_OF10_registration_header")
        if mode == "custom_turbulence_model_modification" and opts_path.is_file():
            raw = opts_path.read_text(encoding="utf-8", errors="ignore")
            stripped = _strip_bad_make_options_lines(raw)
            if _make_options_needs_of10_reset(stripped):
                _write(opts_path, _sanitize_wmake_make_fragment(_MAKE_OPTIONS_OF10_INCOMPRESSIBLE_RAS))
                fixes.append("reset_Make/options_to_OF10_incompressible_RAS_template")
            elif stripped != raw:
                _write(opts_path, _sanitize_wmake_make_fragment(stripped))
                fixes.append("stripped_invalid_Make/options_lines")
            parent = ""  # re-run C++ OF10 sanitization after bad LLM Make/options
            try:
                br_path = case_dir.parent / "code_mod_build_result.json"
                br = _read_json(br_path, {})
                if isinstance(br, dict):
                    parent = str(br.get("parent_model") or "").strip()
            except Exception:
                parent = ""
            sf = _sanitize_custom_turbulence_model_of10(case_dir, class_name, parent)
            fixes.extend(sf)
        # Non-turbulence: do not append orphan -I/-l lines to Make/options (invalid makefile syntax).

    if mode == "custom_turbulence_model_modification" and (
        "no member named" in err or ".v())" in err or ".v()" in err
    ):
        fixes.extend(_rewrite_tmp_tensor_v_access(c_path))

    # wmake rejects C-style /* */ at start of Make/files or Make/options.
    files_path = root / "Make" / "files"
    if "missing separator" in err:
        for mp in (opts_path, files_path):
            if not mp.exists():
                continue
            raw = mp.read_text(encoding="utf-8", errors="ignore")
            cleaned = _sanitize_wmake_make_fragment(raw)
            if cleaned != raw:
                _write(mp, cleaned)
                fixes.append(f"sanitized_wmake_fragment:{mp.name}")

    return fixes


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply code-mod build result, generate files, patch dictionaries, compile/verify.")
    parser.add_argument("--build-result", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--max-compile-attempts", type=int, default=10)
    args = parser.parse_args()

    build = _read_json(Path(args.build_result), {})
    out_path = Path(args.output).expanduser().resolve()
    if not isinstance(build, dict) or build.get("status") != "OK":
        out = {"status": "FAILED", "reason": "build_result missing or status != OK", "build_result": build}
        _write(out_path, json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 1

    class_name = str(build.get("class_name") or "")
    mode = str(build.get("mode") or "")
    activation_dict = str(build.get("activation_dictionary") or "")
    file_specs = build.get("files_to_create") or []
    case_path = ""
    if isinstance(file_specs, list) and file_specs:
        p = str(file_specs[0].get("path", ""))
        m = re.search(r"(.+)/customModels/", p)
        if m:
            case_path = m.group(1)
    if not case_path:
        out = {"status": "FAILED", "reason": "Could not resolve case_path from build_result"}
        _write(out_path, json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 1

    case_dir = Path(case_path).expanduser().resolve()
    print(f"[CODEMOD] Resolved case dir: {case_dir}")
    print(f"[CODEMOD] Class name: {class_name}")
    print(f"[CODEMOD] Mode: {mode}")
    print(f"[CODEMOD] Activation dictionary: {activation_dict}")
    generated = _generate_files(case_dir, class_name, mode, build)
    if generated.get("_generation_incomplete"):
        reason = str(generated.get("reason") or "incomplete_generation")
        print(f"[CODEMOD] FAILED: {reason}", file=sys.stderr)
        out = {
            "status": "FAILED",
            "reason": reason,
            "class_name": class_name,
            "case_path": str(case_dir),
            "compile_ok": False,
            "generated_files": {k: v for k, v in generated.items() if not str(k).startswith("_")},
        }
        _write(out_path, json.dumps(out, indent=2))
        print(json.dumps({"status": "FAILED", "compile_ok": False, "class_name": class_name}, indent=2))
        return 1
    generated.pop("_generation_incomplete", None)
    if mode == "custom_turbulence_model_modification":
        parent = str(build.get("parent_model") or "").strip()
        pre_notes = _sanitize_custom_turbulence_model_of10(case_dir, class_name, parent)
        if pre_notes:
            print(f"[CODEMOD] OF10 turbulence sanitization (pre-compile): {', '.join(pre_notes)}")
    print("[CODEMOD] Generated/updated local files:")
    for k, v in generated.items():
        if k == "mode":
            continue
        print(f"[CODEMOD]   {k}: {v}")
    _patch_control_dict(case_dir / "system" / "controlDict", class_name)
    act_rel = activation_dict.strip()
    if act_rel:
        act_path = case_dir / act_rel
        if act_path.is_file():
            _patch_activation_dict(act_path, class_name, mode, build)
            print(f"[CODEMOD] Patched activation dictionary: {act_rel}")
        else:
            print(f"[CODEMOD] Activation path missing ({act_rel}); skipping automatic activation patch.")
    else:
        print("[CODEMOD] No activation_dictionary in build result; skipping automatic activation patch.")
    print("[CODEMOD] Patched system/controlDict libs (if needed).")

    compile_logs: List[Dict[str, Any]] = []
    compile_ok = False
    attempt_history: List[Dict[str, Any]] = []
    # Load recon context once (used by static checks + clone-parent)
    _payload = _load_payload_for_context(case_dir) if case_dir.is_dir() else {}
    _recon_ctx = (
        (((_payload.get("openfoam_api_context") or {}) if isinstance(_payload, dict) else {})
         .get("recon_reference_context"))
    )

    for attempt in range(1, args.max_compile_attempts + 1):
        if _enforce_case_local_make_files(case_dir, class_name):
            print("[CODEMOD] Enforced case-local Make/files LIB target.")

        # Pre-compile static checks (generic, no LLM cost). If violations are
        # found, log them and let them surface in the attempt_history so the
        # next LLM retry sees them as specific guidance. We don't block wmake
        # because some "violations" (e.g. transitive includes that resolve
        # only through lnInclude discovery) are false positives — the real
        # compiler is still the oracle.
        sc_violations = _precompile_static_checks(case_dir, class_name, _recon_ctx)
        if sc_violations:
            print(f"[CODEMOD] Pre-compile static checks: {len(sc_violations)} violation(s)")
            for v in sc_violations[:5]:
                print(f"[CODEMOD]   - {v}")

        print(f"[CODEMOD] Compile attempt {attempt}/{args.max_compile_attempts}")
        res = _run_cmd(f"cd {case_dir}/customModels/{class_name} && wmake libso", case_dir)
        res["static_check_violations"] = sc_violations
        res["attempt"] = attempt
        print(f"[CODEMOD]   cmd: {res.get('cmd')}")
        print(f"[CODEMOD]   returncode: {res.get('returncode')}")
        if (res.get("stderr") or "").strip():
            err_lines = [ln for ln in str(res.get("stderr", "")).splitlines() if ln.strip()]
            print("[CODEMOD]   stderr tail:")
            for ln in err_lines[-8:]:
                print(f"[CODEMOD]     {ln}")
        elif (res.get("stdout") or "").strip():
            out_lines = [ln for ln in str(res.get("stdout", "")).splitlines() if ln.strip()]
            print("[CODEMOD]   stdout tail:")
            for ln in out_lines[-6:]:
                print(f"[CODEMOD]     {ln}")
        if res["returncode"] != 0:
            err_text = (res.get("stderr", "") + "\n" + res.get("stdout", ""))
            # Phase 1: deterministic pattern fixers — cheap, no LLM tokens.
            # Catches the common mechanical mistakes (bad Make/files stem,
            # missing-header includes, LIB_SRCS, C-style block comments in
            # makefile fragments). Generic across all code-mod modes.
            det_fixes = _deterministic_compile_autofixes(
                case_dir, class_name, err_text, mode=mode
            )
            if det_fixes:
                print(f"[CODEMOD]   deterministic fixes: {', '.join(det_fixes)}")

            # Phase 2: LLM patch call. Only include prior history when we're
            # past attempt 1 so the model can see the trajectory. The patch
            # prompt asks ONLY for files that need to change, not a full
            # regeneration, and validates the LLM's target paths.
            llm_fixes: List[str] = []
            try:
                llm_fixes = _llm_compile_review_fix(
                    case_path=case_dir,
                    class_name=class_name,
                    mode=mode,
                    build=build,
                    err_text=err_text,
                    attempt_history=attempt_history,
                )
            except Exception as exc:
                llm_fixes = [f"llm_fix_exception:{type(exc).__name__}"]

            # Phase 3: legacy _apply_compile_fixes catch-all (kept for
            # backwards compatibility — fires harmlessly if already fixed).
            legacy_fixes = _apply_compile_fixes(
                case_dir, class_name, err_text, mode=mode,
            )
            res["auto_fixes_applied"] = det_fixes + llm_fixes + legacy_fixes
            if res["auto_fixes_applied"]:
                print(f"[CODEMOD]   fixes applied: {', '.join(res['auto_fixes_applied'])}")
            else:
                print("[CODEMOD]   no automatic fixes applied this attempt")
            # Keep library emission local even after LLM rewrites.
            if _enforce_case_local_make_files(case_dir, class_name):
                print("[CODEMOD]   re-normalized Make/files to case-local LIB target.")
            # Record this attempt for the next iteration's LLM context.
            attempt_history.append({
                "attempt": attempt,
                "error": err_text[-3000:],
                "fixes_tried": res["auto_fixes_applied"],
            })
        compile_logs.append(res)
        if res["returncode"] == 0:
            model_dir = case_dir / "customModels" / class_name
            so_path = _locate_built_shared_object(case_dir, class_name)
            if so_path is not None:
                print(f"[CODEMOD] Compile succeeded (shared object: {so_path}).")
                compile_ok = True
                break
            print(
                "[CODEMOD] wmake returned 0 but no viable shared object found under "
                f"{model_dir}/platforms or FOAM_*_LIBBIN; re-normalizing Make/files and retrying."
            )
            mf = model_dir / "Make" / "files"
            if mf.is_file():
                raw = mf.read_text(encoding="utf-8", errors="ignore")
                fixed = _fix_make_files_wmake_layout(raw, class_name)
                fixed = _normalize_make_files_lib_path(fixed, class_name)
                if fixed != raw:
                    _safe_write_local(case_dir, class_name, mf, fixed)
                    print("[CODEMOD]   Make/files updated for case-local LIB= target.")
            continue
    if not compile_ok:
        print("[CODEMOD] Compile failed after max attempts.")
    else:
        _patch_control_dict_case_relative_lib(case_dir, class_name)

    verify_logs: List[Dict[str, Any]] = []
    for cmd in build.get("verification_commands", []):
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        print(f"[CODEMOD] Verification command: {cmd}")
        vr = _run_cmd(cmd, case_dir)
        verify_logs.append(vr)
        print(f"[CODEMOD]   returncode: {vr.get('returncode')}")
        if (vr.get("stdout") or "").strip():
            vout = [ln for ln in str(vr.get("stdout", "")).splitlines() if ln.strip()]
            for ln in vout[-6:]:
                print(f"[CODEMOD]   {ln}")
        if (vr.get("stderr") or "").strip():
            verr = [ln for ln in str(vr.get("stderr", "")).splitlines() if ln.strip()]
            for ln in verr[-6:]:
                print(f"[CODEMOD]   ERR {ln}")

    out = {
        "status": "OK" if compile_ok else "FAILED",
        "class_name": class_name,
        "case_path": str(case_dir),
        "generated_files": generated,
        "compile_ok": compile_ok,
        "compile_logs": compile_logs,
        "verification_logs": verify_logs,
    }
    _write(out_path, json.dumps(out, indent=2))
    print(json.dumps({"status": out["status"], "compile_ok": compile_ok, "class_name": class_name}, indent=2))
    return 0 if compile_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


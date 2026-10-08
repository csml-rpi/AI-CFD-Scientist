#!/usr/bin/env python3
"""Make a renamed, case-local copy of a stock OpenFOAM class that compiles
before anything in it is changed.

    python3 foam_scaffold.py --name <class or type name> --new-name <name> --dest <dir>

Everything is read from the OpenFOAM installation itself, not from a list of
known models:
  * the class      -- by header name, or by the TypeName a case uses for it;
  * its source     -- the header's own directory;
  * how it builds  -- the Make/options of the stock library that compiles it,
                      with relative include paths made absolute;
  * how it is registered -- the registration statement that names the class,
                      from its own .C or from the library's registration file,
                      with every other registration in that file left out.

The copy is renamed (class name and type name), compiled with wmake into
<dest>/lib, and the result is printed as JSON: what was copied, where the
library is, and how a case selects it. Nothing outside <dest> is written.
Needs the OpenFOAM environment loaded (FOAM_SRC, wmake on PATH).
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


def fail(message: str, **extra) -> int:
    print(json.dumps({"ok": False, "error": message, **extra}, indent=2))
    return 1


def sources() -> list[Path]:
    root = Path(os.environ.get("FOAM_SRC", ""))
    if not root.is_dir():
        raise SystemExit(fail("FOAM_SRC is not set: load the OpenFOAM environment first."))
    return [p for p in root.rglob("*") if p.suffix in (".H", ".C") and "lnInclude" not in p.parts]


def find_class(name: str, files: list[Path]) -> tuple[Path | None, list[str]]:
    """The header that declares ``name``, by file name or by TypeName."""
    by_file = [p for p in files if p.suffix == ".H" and p.stem == name]
    if len(by_file) == 1:
        return by_file[0], []
    pattern = re.compile(r'TypeName\(\s*"' + re.escape(name) + r'"\s*\)')
    by_type = [p for p in files if p.suffix == ".H" and pattern.search(p.read_text(errors="ignore"))]
    found = by_file or by_type
    if len(found) == 1:
        return found[0], []
    return None, [str(p) for p in found]


def statements(text: str) -> list[tuple[str, str]]:
    """Top-level pieces of a registration file: ("pp", line) for preprocessor
    lines, ("call", text) for a macro call with its arguments (across lines),
    ("other", text) for anything else."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append(("other", text[i:j])); i = j; continue
        if text.startswith("//", i):
            j = text.find("\n", i); j = n if j < 0 else j
            out.append(("other", text[i:j])); i = j; continue
        if text.startswith("/*", i):
            j = text.find("*/", i); j = n if j < 0 else j + 2
            out.append(("other", text[i:j])); i = j; continue
        if text[i] == "#":
            j = text.find("\n", i); j = n if j < 0 else j
            out.append(("pp", text[i:j])); i = j; continue
        m = re.match(r"[A-Za-z_]\w*\s*\(", text[i:])
        if m:
            depth, j = 0, i + m.end() - 1
            while j < n:
                depth += text[j] == "("
                depth -= text[j] == ")"
                j += 1
                if depth == 0:
                    break
            while j < n and text[j] in " \t;":
                j += 1
            out.append(("call", text[i:j])); i = j; continue
        j = i + 1
        out.append(("other", text[i:j])); i = j
    return out


def names_class(call: str, cls: str) -> bool:
    return re.search(r"\b" + re.escape(cls) + r"\b", call) is not None


def registration_in(path: Path, cls: str) -> bool:
    text = path.read_text(errors="ignore")
    return any(kind == "call" and names_class(body, cls) and re.match(
        r"(make\w*|addToRunTimeSelectionTable|addNamedToRunTimeSelectionTable)\s*\(", body)
        for kind, body in statements(text))


def library_dir(path: Path) -> Path | None:
    for parent in path.parents:
        if (parent / "Make" / "files").is_file():
            return parent
    return None


def options_of(lib: Path, extra_inc: list[str]) -> tuple[str, str]:
    """EXE_INC and LIB_LIBS of a stock library, include paths made absolute."""
    text = (lib / "Make" / "options").read_text(errors="ignore") if (lib / "Make" / "options").is_file() else ""
    def block(key: str) -> list[str]:
        m = re.search(key + r"\s*=\s*((?:.*\\\n)*.*)", text)
        return [t for t in re.split(r"\s+", m.group(1).replace("\\\n", " ")) if t] if m else []
    inc = []
    for tok in block("EXE_INC"):
        if tok.startswith("-I") and not tok[2:].startswith(("$", "/")):
            tok = "-I" + str((lib / tok[2:]).resolve())
        inc.append(tok)
    libs = block("LIB_LIBS")
    files = (lib / "Make" / "files").read_text(errors="ignore")
    m = re.search(r"LIB\s*=\s*\S*/lib(\w+)", files)
    if m and f"-l{m.group(1)}" not in libs:
        libs.append(f"-l{m.group(1)}")
    return " \\\n    ".join(extra_inc + inc), " \\\n    ".join(libs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="class name or the type name a case selects")
    ap.add_argument("--new-name", required=True, help="type name the copy is selected by")
    ap.add_argument("--dest", required=True, help="new folder for the copy (inside the case)")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    new_type = a.new_name.strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", new_type):
        return fail("--new-name must be a C++ identifier.")
    dest = Path(a.dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        return fail(f"{dest} already exists and is not empty; choose a new --dest.")

    files = sources()
    header, ambiguous = find_class(a.name, files)
    if header is None:
        return fail(f"no single class found for {a.name!r}.", candidates=ambiguous[:20])
    cls = header.stem
    old_type = a.name if a.name != cls else cls
    new_cls = new_type + cls[len(old_type):] if cls.startswith(old_type) else new_type
    src_dir = header.parent
    own = sorted(p for p in src_dir.iterdir() if p.is_file() and p.stem == cls and p.suffix in (".H", ".C"))
    template = f'#include "{cls}.C"' in header.read_text(errors="ignore")

    reg_file = None
    for p in own:
        if p.suffix == ".C" and registration_in(p, cls):
            reg_file = p
    if reg_file is None:
        hits = [p for p in files if p.suffix == ".C" and p not in own and registration_in(p, cls)]
        hits.sort(key=lambda p: len(p.parts))
        if not hits:
            return fail(f"found {header} but no registration statement naming {cls}.")
        reg_file = hits[0]

    def rename(text: str) -> str:
        text = re.sub(r"\b" + re.escape(cls) + r"\b", new_cls, text)
        if old_type != cls:
            text = re.sub(r'TypeName\(\s*"' + re.escape(old_type) + r'"\s*\)', f'TypeName("{new_type}")', text)
        return text

    dest.mkdir(parents=True, exist_ok=True)
    copied, compile_units = [], []
    for p in own:
        target = dest / (new_cls + p.suffix)
        target.write_text(rename(p.read_text(errors="ignore")))
        copied.append(str(target))
        if p.suffix == ".C" and not template:
            compile_units.append(target.name)

    if reg_file not in own:
        kept, dropped = [], set()
        pieces = statements(reg_file.read_text(errors="ignore"))
        for kind, body in pieces:
            if kind == "call" and not names_class(body, cls):
                dropped.update(re.findall(r"\b(\w+)\b", body))
        for kind, body in pieces:
            if kind == "call" and not names_class(body, cls):
                continue
            if kind == "pp":
                inc = re.match(r'#include\s+"(\w+)\.H"', body)
                if inc and inc.group(1) in dropped and inc.group(1) != cls:
                    continue
            kept.append(body)
        reg_target = dest / f"{new_cls}Registration.C"
        reg_target.write_text(rename("".join(kept)))
        copied.append(str(reg_target))
        compile_units.append(reg_target.name)

    lib = library_dir(reg_file)
    if lib is None:
        return fail(f"no Make/files found above {reg_file}.")
    exe_inc, lib_libs = options_of(lib, [f"-I{dest}", f"-I{lib}/lnInclude", f"-I{src_dir}"])
    (dest / "Make").mkdir(exist_ok=True)
    (dest / "Make" / "files").write_text("\n".join(compile_units) + f"\n\nLIB = {dest}/lib/lib{new_type}\n")
    (dest / "Make" / "options").write_text(f"EXE_INC = \\\n    {exe_inc}\n\nLIB_LIBS = \\\n    {lib_libs}\n")

    result = {
        "ok": True, "stock_class": cls, "stock_type_name": old_type, "source_dir": str(src_dir),
        "template": template, "registration_from": str(reg_file), "built_like": str(lib),
        "new_class": new_cls, "new_type_name": new_type, "files": copied + [str(dest / "Make")],
    }
    if a.no_compile:
        print(json.dumps(result, indent=2)); return 0
    if shutil.which("wmake") is None:
        return fail("wmake is not on PATH: load the OpenFOAM environment first.", **result)
    proc = subprocess.run(["wmake", "libso"], cwd=dest, capture_output=True, text=True)
    so = dest / "lib" / f"lib{new_type}.so"
    result.update({
        "compiled": proc.returncode == 0 and so.is_file(),
        "library": str(so) if so.is_file() else "",
        "compile_log_tail": (proc.stdout + proc.stderr)[-3000:] if proc.returncode != 0 else "",
        "how_to_use": (
            f'Load it from the case with libs ("{so}"); in system/controlDict, and select it with the '
            f"type name {new_type!r} wherever the case currently selects {old_type!r}. Run the case once "
            "like that, unchanged, before editing the copy."),
    })
    result["ok"] = result["compiled"]
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

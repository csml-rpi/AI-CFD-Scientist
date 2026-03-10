"""Utilities for LaTeX paper compilation and PDF generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


def extract_pdflatex_errors(log_content: str, max_errors: int = 5) -> str:
    """
    Extract the key error messages from a pdflatex log for the reviewer.
    Returns a concise summary: "! ... Error:" lines and the "l.XXX" line references.
    """
    if not log_content or not log_content.strip():
        return "No log content available."
    lines = log_content.split("\n")
    errors: list[str] = []
    i = 0
    while i < len(lines) and len(errors) < max_errors:
        line = lines[i]
        # Match "! Package X Error: ..." or "! Undefined control sequence" etc.
        if line.strip().startswith("!"):
            err_block = [line.strip()]
            # Often the next line is "l.XXX" or "..." or "See the X package documentation"
            j = i + 1
            while j < len(lines) and j < i + 5:
                next_line = lines[j].strip()
                if next_line.startswith("l."):
                    err_block.append(next_line)
                    break
                if next_line and not next_line.startswith("!"):
                    err_block.append(next_line[:80])
                j += 1
            errors.append("\n  ".join(err_block))
            i = j if j < len(lines) else i + 1
        i += 1
    if not errors:
        # Fallback: return last 800 chars (often contains the fatal error)
        return "Key error not found in standard format. Last part of log:\n" + log_content.strip()[-800:]
    return "PRIMARY ERRORS (fix these first):\n\n" + "\n\n---\n\n".join(errors)


def compile_tex_to_pdf(
    tex_path: Path,
    work_dir: Path | None = None,
    runs: int = 2,
) -> Tuple[bool, Path | None, str]:
    """
    Compile a .tex file to PDF using pdflatex.

    Args:
        tex_path: Path to the .tex file.
        work_dir: Working directory for pdflatex (for resolving \\includegraphics paths).
                  If None, uses tex_path.parent.
        runs: Number of pdflatex runs (2 for cross-references).

    Returns:
        (success, pdf_path, stderr_or_error)
        pdf_path is None if compilation failed.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        return False, None, f"File not found: {tex_path}"

    out_dir = tex_path.parent
    cwd = Path(work_dir).resolve() if work_dir else out_dir

    # pdflatex -output-directory puts aux/log/pdf in out_dir
    # Run from cwd so \includegraphics paths (e.g. runs/...) are relative to cwd
    try:
        tex_arg = str(tex_path.relative_to(cwd)) if cwd in tex_path.parents or cwd == tex_path.parent else str(tex_path)
    except ValueError:
        tex_arg = str(tex_path)
    if cwd == out_dir:
        tex_arg = tex_path.name

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={out_dir}",
        tex_arg,
    ]

    err_parts: list[str] = []
    for _ in range(runs):
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            # pdflatex usually puts error details in stdout (log); include both for reviewer
            err_parts.append(proc.stdout or "")
            err_parts.append(proc.stderr or "")
            return False, None, "\n".join(err_parts).strip() or f"pdflatex exit {proc.returncode}"
        if proc.stderr:
            err_parts.append(proc.stderr)

    pdf_path = out_dir / tex_path.with_suffix(".pdf").name
    if pdf_path.is_file():
        return True, pdf_path, ""
    return False, None, "\n".join(err_parts).strip() or "PDF not produced"

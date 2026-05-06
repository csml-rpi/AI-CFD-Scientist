#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Review paper using PaperReviewerAgent.")
    parser.add_argument("--paper", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--topic", default="", type=str)
    parser.add_argument("--analysis", default="", type=str)
    parser.add_argument("--figs", default="", type=str)
    parser.add_argument("--manifest", default="", type=str)
    parser.add_argument("--pdf", default="", type=str)
    args = parser.parse_args()

    from cfd_langgraph.agents.paper_reviewer_agent import PaperReviewerAgent
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.prompts.loader import PromptLoader

    paper_dir = Path(args.paper).resolve()
    tex_files = sorted(paper_dir.rglob("*.tex"))
    if not tex_files:
        print(f"No .tex files found in {paper_dir}", file=sys.stderr)
        return 1
    tex_content = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in tex_files)
    context_blocks = []
    if args.topic.strip():
        context_blocks.append(f"TOPIC:\n{args.topic.strip()}")
    if args.analysis:
        ap = Path(args.analysis).resolve()
        if ap.exists():
            context_blocks.append(f"ANALYSIS JSON (truncated):\n{ap.read_text(encoding='utf-8', errors='ignore')[:30000]}")
    if args.manifest:
        mp = Path(args.manifest).resolve()
        if mp.exists():
            context_blocks.append(f"EXPERIMENT MANIFEST:\n{mp.read_text(encoding='utf-8', errors='ignore')[:12000]}")
    if args.figs:
        fp = Path(args.figs).resolve()
        if fp.exists():
            imgs = sorted(
                [
                    str(p.relative_to(fp))
                    for p in fp.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                ]
            )
            context_blocks.append("AVAILABLE FIGURES:\n" + "\n".join(imgs[:500]))
    pdf_path = None
    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
    else:
        cand = paper_dir / "main.pdf"
        pdf_path = cand if cand.exists() else None
    if pdf_path and pdf_path.exists():
        try:
            from pypdf import PdfReader  # type: ignore

            rdr = PdfReader(str(pdf_path))
            pages = []
            for pg in rdr.pages[:20]:
                txt = pg.extract_text() or ""
                if txt.strip():
                    pages.append(txt[:4000])
            if pages:
                context_blocks.append("PDF TEXT PREVIEW:\n" + "\n\n".join(pages)[:60000])
        except Exception:
            pass
    if context_blocks:
        tex_content = tex_content + "\n\n% --- REVIEW CONTEXT ---\n" + "\n\n".join(context_blocks)

    settings = get_settings()
    prompts = PromptLoader(settings.prompts_path)
    reviewer = PaperReviewerAgent(model=settings.model, prompt_loader=prompts)
    review = reviewer.review(tex_content=tex_content, compile_ok=True, compile_error="")
    out = {
        "overall_score": review.get("score", 0.0),
        "major_issues": review.get("recommendations", []),
        "minor_issues": [],
        "recommendations": review.get("recommendations", []),
        "raw": review,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Score={out['overall_score']} major_issues={len(out['major_issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

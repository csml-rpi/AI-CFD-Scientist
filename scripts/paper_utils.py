#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


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
    parser = argparse.ArgumentParser(description="Generate paper artifacts using WriterAgent.")
    parser.add_argument("--analysis", required=True, type=str)
    parser.add_argument("--figs", required=True, type=str)
    parser.add_argument("--literature", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--topic", default="CFD study manuscript", type=str)
    parser.add_argument("--manifest", default="", type=str)
    parser.add_argument(
        "--mesh-independence",
        default="",
        type=str,
        help="Optional JSON from mesh gate (mesh_independence_context.json) for Methods/table + figures.",
    )
    parser.add_argument("--review-output", default="", type=str)
    parser.add_argument("--max-review-loops", default=10, type=int)
    parser.add_argument(
        "--max-compile-recovery",
        default=10,
        type=int,
        help="After review loops, if pdflatex still fails, run this many compile-only LLM fixes (no publishability review).",
    )
    parser.add_argument("--template", choices=["neurips", "icml", "iclr", "nature"], default="neurips")
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json for flow params, formula, reference data context")
    args = parser.parse_args()

    from cfd_langgraph.agents.writer_agent import WriterAgent
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.prompts.loader import PromptLoader

    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    literature = json.loads(Path(args.literature).read_text(encoding="utf-8"))
    figs_dir = Path(args.figs).resolve()
    figs = [str(p) for p in figs_dir.rglob("*.png")]

    mesh_independence: Dict[str, Any] = {}
    mip = Path(args.mesh_independence).expanduser().resolve() if args.mesh_independence.strip() else None
    if mip and mip.is_file():
        try:
            mesh_independence = json.loads(mip.read_text(encoding="utf-8"))
        except Exception:
            mesh_independence = {}
    if isinstance(mesh_independence, dict):
        extra = mesh_independence.get("mesh_figure_paths") or []
        if isinstance(extra, list):
            for p in extra:
                if isinstance(p, str) and Path(p).is_file():
                    figs.append(str(Path(p).resolve()))

    settings = get_settings()
    prompts = PromptLoader(settings.prompts_path)
    writer = WriterAgent(model=settings.model, prompt_loader=prompts)

    manifest: Dict[str, Any] = {}
    if args.manifest:
        mp = Path(args.manifest)
        if mp.exists():
            try:
                manifest = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

    starter_understanding: Dict[str, Any] = {}
    if args.starter_understanding:
        sup = Path(args.starter_understanding)
        if sup.is_file():
            try:
                starter_understanding = json.loads(sup.read_text(encoding="utf-8"))
                print(f"[PAPER] Loaded starter_understanding for paper context ({sup})")
            except Exception as _e:
                print(f"[PAPER] WARNING: could not load starter_understanding: {_e}")

    section_context = json.dumps(
        {
            "topic": args.topic,
            "analysis": analysis,
            "template": args.template,
            "figures": figs,
            "manifest": manifest,
            "mesh_independence": mesh_independence if isinstance(mesh_independence, dict) else {},
            "starter_understanding": starter_understanding,
        },
        indent=2,
    )
    tex, pdf_path, review_info = writer.write_paper_with_literature_and_review(
        topic=args.topic or "CFD study manuscript",
        section_context=section_context,
        out_dir=Path(args.output).resolve(),
        work_dir=Path(args.output).resolve(),
        ideation_literature_bundle=literature if isinstance(literature, list) else [literature],
        visualization_bundle=[{"simulation_id": "all", "visualization": {"images": figs}}],
        max_review_tries=max(1, int(args.max_review_loops)),
        max_compile_recovery_tries=max(0, int(args.max_compile_recovery)),
        verbose=True,
    )

    out = Path(args.output).resolve()
    (out / "sections").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    def _fig_dest_name(src: Path) -> str:
        base = src.name
        if base not in used_names:
            used_names.add(base)
            return base
        stem, suf = src.stem, src.suffix
        parent = src.parent.name.replace(" ", "_")
        cand = f"meshgate_{parent}_{stem}{suf}"
        n = 0
        while cand in used_names:
            n += 1
            cand = f"meshgate_{parent}_{stem}_{n}{suf}"
        used_names.add(cand)
        return cand

    for f in figs:
        src = Path(f)
        if src.exists():
            dest = out / "figures" / _fig_dest_name(src)
            shutil.copy2(src, dest)
    (out / "main.tex").write_text(tex, encoding="utf-8")
    (out / "sections" / "body.tex").write_text(tex, encoding="utf-8")
    (out / "references.bib").write_text("", encoding="utf-8")
    if pdf_path and Path(pdf_path).exists():
        shutil.copy2(Path(pdf_path), out / "main.pdf")
    if args.review_output:
        ro = Path(args.review_output).resolve()
        ro.parent.mkdir(parents=True, exist_ok=True)
        last_review = {}
        if isinstance(review_info, dict):
            revs = review_info.get("reviews", [])
            if isinstance(revs, list) and revs:
                last_review = revs[-1]
        out_review = {
            "overall_score": float(last_review.get("score", 0.0)) if isinstance(last_review, dict) else 0.0,
            "major_issues": (last_review.get("recommendations", []) if isinstance(last_review, dict) else []),
            "minor_issues": [],
            "recommendations": (last_review.get("recommendations", []) if isinstance(last_review, dict) else []),
            "raw": {
                "last_review": last_review,
                "review_info": review_info,
                "pdf_path": str(out / "main.pdf") if (out / "main.pdf").exists() else "",
            },
        }
        ro.write_text(json.dumps(out_review, indent=2), encoding="utf-8")
    print(f"Created: {out / 'main.tex'}")
    print(f"Created: {out / 'sections' / 'body.tex'}")
    print(f"Created: {out / 'references.bib'}")
    if (out / "main.pdf").exists():
        print(f"Created: {out / 'main.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

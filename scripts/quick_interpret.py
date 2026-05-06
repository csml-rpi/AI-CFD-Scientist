#!/usr/bin/env python3
"""Lightweight one-shot interpreter for the VLM ablation.

The full ResultsInterpreterAgent.interpret() pipeline regenerates figures
through viz_creator (with up to 10 vision-LLM-graded retries) and then does
ANOTHER vision call for the actual judgment. That's 2-11 vision calls per
case, easily hitting the 900s wrapper timeout when one upstream call stalls.

For the ablation we already have validated figures in --figs. We just need to
send them to the vision LLM ONCE with the embedded interpretation_system_prompt
+ interpretation_user_prompt (verbatim from prompts/prompts.yaml), parse the
JSON verdict, and write decision.json. ~30-90s per case instead of 10-15 min.

Usage:
    python scripts/quick_interpret.py \
        --case <case_dir> \
        --figs <figs_dir> \
        --requirement "<full requirement text>" \
        --output <case_dir>/decision.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent


def _bootstrap_paths() -> None:
    src = REPO / "src"
    if str(src) not in sys.path and src.is_dir():
        sys.path.insert(0, str(src))


VISION_IMAGE_MAX_SIDE = 1280
LLM_INVOKE_TIMEOUT_S = 300   # 5 min per call (single shot, not 10)


def _image_to_data_url(image_path: Path, max_side: int = VISION_IMAGE_MAX_SIDE) -> str:
    """Resize + base64-encode one image. PIL is required."""
    from PIL import Image
    with Image.open(image_path) as im:
        im.load()
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def _build_image_blocks(figs_dir: Path, max_images: int = 12) -> List[Dict[str, Any]]:
    paths = sorted(figs_dir.glob("*.png"))[:max_images]
    blocks: List[Dict[str, Any]] = []
    for p in paths:
        try:
            url = _image_to_data_url(p)
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception as exc:
            print(f"  [warn] failed to encode {p.name}: {exc}", file=sys.stderr)
    return blocks


def _load_prompts():
    """Load interpretation_system_prompt + interpretation_user_prompt verbatim from prompts.yaml."""
    import yaml
    with open(REPO / "prompts" / "prompts.yaml") as f:
        data = yaml.safe_load(f)
    section = data.get("ResultsInterpreterAgent", {})
    return (
        section.get("interpretation_system_prompt", ""),
        section.get("interpretation_user_prompt", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot VLM interpreter (no viz regeneration).")
    parser.add_argument("--case", required=True)
    parser.add_argument("--figs", required=True)
    parser.add_argument("--requirement", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=LLM_INVOKE_TIMEOUT_S)
    args = parser.parse_args()

    case_dir = Path(args.case).resolve()
    figs_dir = Path(args.figs).resolve()
    out_path = Path(args.output).resolve()

    if not figs_dir.is_dir():
        print(f"figs_dir not found: {figs_dir}", file=sys.stderr)
        return 2

    img_blocks = _build_image_blocks(figs_dir, max_images=args.max_images)
    if not img_blocks:
        print(f"no .png images in {figs_dir}", file=sys.stderr)
        return 2
    print(f"  encoded {len(img_blocks)} images", file=sys.stderr)

    sys_prompt, user_prompt_template = _load_prompts()
    user_msg_text = user_prompt_template.replace("{user_requirement}", args.requirement.strip() or "(no requirement provided)")

    _bootstrap_paths()
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    try:
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
    except Exception:
        def strip_json_fences(s: str) -> str:
            return re.sub(r"^```(?:json)?\s*|\s*```$", "", str(s).strip(), flags=re.MULTILINE | re.DOTALL)

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.1)

    user_content: List[Any] = [{"type": "text", "text": user_msg_text}]
    user_content.extend(img_blocks)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_content),
    ]

    t0 = time.time()
    print(f"  invoking vision LLM (model={settings.model}) ...", file=sys.stderr)
    try:
        # Use concurrent.futures to enforce the timeout; many providers don't honor a
        # native timeout kwarg on .invoke().
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(llm.invoke, messages)
            try:
                resp = fut.result(timeout=args.timeout)
            except concurrent.futures.TimeoutError:
                print(f"  TIMEOUT after {args.timeout}s", file=sys.stderr)
                return 3
    except Exception as exc:
        print(f"  LLM call failed: {exc!r}", file=sys.stderr)
        return 4
    dt = time.time() - t0
    print(f"  LLM returned in {dt:.1f}s", file=sys.stderr)

    raw = resp.content if hasattr(resp, "content") else str(resp)

    # Parse
    try:
        cleaned = strip_json_fences(raw) if isinstance(raw, str) else str(raw)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}
    except Exception:
        parsed = {"raw": str(raw)[:2000], "parse_error": True}

    # Map to {status, reason, suggested_changes, ...} schema (same as the heavy interpreter)
    rerun_required = bool(parsed.get("rerun_required", False))
    requirement_met = parsed.get("requirement_met", None)
    sim_success = parsed.get("simulation_success", None)
    if rerun_required:
        # Distinguish numerical RERUN from physics REVISE based on issues content.
        issues_str = json.dumps(parsed.get("issues", "")) + " " + str(parsed.get("reasons", ""))
        if any(kw in issues_str.lower() for kw in ("geometry", "scenario", "mismatch", "wrong", "unphysical", "boundary")):
            status = "REVISE"
        else:
            status = "RERUN"
    else:
        status = "PROCEED"

    decision = {
        "status": status,
        "confidence": 0.8 if not parsed.get("parse_error") else 0.4,
        "reason": parsed.get("summary", "") or parsed.get("raw", "")[:200],
        "suggested_changes": parsed.get("issues", ""),
        "raw": parsed,
        "elapsed_s": round(dt, 1),
        "model": settings.model,
        "n_images": len(img_blocks),
        "interpreter_variant": "quick_interpret_v1",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2))
    print(f"  decision: {status} (conf={decision['confidence']:.1f}, {dt:.0f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

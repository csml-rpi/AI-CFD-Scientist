#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))


def _load_prompt_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a dumped rewrite prompt directly against Bedrock/LLMService."
    )
    p.add_argument("--prompt-file", required=True, help="Path to prompt JSON dump.")
    p.add_argument(
        "--mode",
        choices=("structured", "plain", "both"),
        default="both",
        help="Invoke with structured FoamPydantic, plain completion, or both.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Optional directory to write raw outputs as files.",
    )
    return p.parse_args()


def _normalize_raw_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for blk in content:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                out.append(blk["text"])
            elif isinstance(blk, str):
                out.append(blk)
        return "\n".join(out)
    return str(content)


def main() -> int:
    args = _parse_args()
    _bootstrap_paths()

    from config import Config
    from utils import LLMService, FoamPydantic

    payload = _load_prompt_file(Path(args.prompt_file))
    system_prompt = str(payload.get("system_prompt", ""))
    user_prompt = str(payload.get("user_prompt", ""))
    print(
        f"[replay] prompt chars: system={len(system_prompt)} user={len(user_prompt)} total={len(system_prompt)+len(user_prompt)}"
    )

    llm = LLMService(Config())
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"structured", "both"}:
        print("[replay] running structured FoamPydantic call...")
        try:
            structured = llm.invoke(user_prompt, system_prompt, pydantic_obj=FoamPydantic)
            n = len(getattr(structured, "list_foamfile", []) or [])
            print(f"[replay] structured success: list_foamfile count={n}")
            if out_dir:
                sp = out_dir / "structured_result.json"
                with open(sp, "w", encoding="utf-8") as f:
                    json.dump(structured.model_dump(), f, ensure_ascii=False, indent=2)
                print(f"[replay] wrote {sp}")
        except Exception as e:
            print(f"[replay] structured failed: {type(e).__name__}: {e}")

    if args.mode in {"plain", "both"}:
        print("[replay] running plain completion call...")
        try:
            plain = llm.invoke(user_prompt, system_prompt, pydantic_obj=None)
            text = _normalize_raw_text(plain)
            print(f"[replay] plain success: raw chars={len(text)}")
            preview = text[:600].replace("\n", "\\n")
            print(f"[replay] plain preview: {preview}")
            if out_dir:
                pp = out_dir / "plain_result.txt"
                with open(pp, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[replay] wrote {pp}")
        except Exception as e:
            print(f"[replay] plain failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


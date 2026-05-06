#!/usr/bin/env python3
"""
Standalone reproducer for the interpreter-hang issue.

Mirrors what scripts/interpret.py → ResultsInterpreterAgent._invoke_vision_llm
does at the network layer:
  - same `create_langchain_llm` factory call (whatever provider env says)
  - same message shape: SystemMessage + HumanMessage(content=[text, image_url, ...])
  - same data-URL base64 image blocks
  - same `temperature=0.1`

Adds:
  - per-call wall-clock timing
  - per-call hard timeout (so we can DETECT hangs, not just suffer them)
  - sequential-N runs so intermittent hangs become visible
  - clean summary at the end (success count, hang count, mean/median duration,
    per-call status table)

Run examples:
    # Smoke run with synthetic dummy images, default model from env, 5 calls,
    # 120s per-call timeout:
    python scripts/test_interpret_hang.py --n 5 --per-call-timeout 120

    # Use real figures from a finished case (the same ones interpret.py would see):
    python scripts/test_interpret_hang.py \
        --case-figs /path/to/runs/.../iter_NNN_code_mod_X/experiment/figs \
        --n 10 --per-call-timeout 600 --model gpt-5.4

    # Compare gpt-5.4 vs gpt-5.5 hang rate:
    python scripts/test_interpret_hang.py --model gpt-5.4 --n 8
    python scripts/test_interpret_hang.py --model gpt-5.5 --n 8

The provider is whatever your env / CFD_SCIENTIST_LLM_PROVIDER says — same as
the live run. No mock, no monkey-patch — just exactly what interpret.py does.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "Foam-Agent" / "src", _REPO_ROOT / "scripts"):
    sp = str(_p)
    if _p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a CFD results interpreter. Look at the figures and the experiment "
    "description. Decide whether the simulation result meets the requirement "
    "and whether it converged cleanly. Return STRICT JSON with keys: "
    "{\"requirement_met\": bool, \"rerun_required\": bool, "
    "\"summary\": string, \"issues\": [string]}. No prose outside the JSON."
)

_DEFAULT_USER_PROMPT_TEMPLATE = (
    "Experiment description:\n{user_requirement}\n\n"
    "Look at the attached figures. Output the strict JSON described in the "
    "system prompt. No markdown."
)

_DEFAULT_USER_REQUIREMENT = (
    "Periodic-hill flow at Re=5600. Spalart-Allmaras turbulence model with a "
    "candidate near-wall sink modification. Solver simpleFoam reached End at "
    "t=5000 s. Compare bottom-wall Cf curve against DNS reference. Was the "
    "modification effective and did the solver converge cleanly?"
)


def _make_dummy_image(width: int = 320, height: int = 240, label: str = "dummy") -> bytes:
    """Generate a small valid PNG without depending on PIL/matplotlib being
    available everywhere. Uses pyplot if it imports, else falls back to a
    minimal handwritten PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        x = np.linspace(0, 9, 200)
        ax.plot(x, np.sin(x) * 0.005, label=f"{label} sim")
        ax.plot(x, np.sin(x) * 0.005 + 0.001, "--", label="ref")
        ax.set_title(label)
        ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        # Hand-rolled tiny 1x1 PNG fallback (good enough as a payload)
        # 1x1 transparent PNG:
        return base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
            b"DUlEQVR42mP8/x8AAusB9eMtj1MAAAAASUVORK5CYII="
        )


def _png_bytes_to_data_url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _gather_image_data_urls(case_figs: Optional[Path], n_dummy: int) -> List[str]:
    out: List[str] = []
    if case_figs and case_figs.is_dir():
        for p in sorted(case_figs.glob("*.png"))[:8]:
            try:
                b = p.read_bytes()
            except Exception:
                continue
            out.append(_png_bytes_to_data_url(b))
    if not out:
        for i in range(max(1, n_dummy)):
            out.append(_png_bytes_to_data_url(_make_dummy_image(label=f"dummy_{i}")))
    return out


# ---------------------------------------------------------------------------
# the core invocation (same shape as ResultsInterpreterAgent._invoke_vision_llm)
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, user_prompt: str,
                   image_data_urls: List[str]) -> Tuple[Any, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return SystemMessage(content=system_prompt), HumanMessage(content=content)


def _do_invoke(llm: Any, messages: List[Any]) -> str:
    out = llm.invoke(messages)
    return getattr(out, "content", str(out)) if out else ""


def invoke_with_timeout(llm: Any, messages: List[Any], timeout_s: float
                        ) -> Dict[str, Any]:
    """Run llm.invoke(...) with a hard wall-clock cap. Returns a status dict."""
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_do_invoke, llm, messages)
        try:
            result = future.result(timeout=timeout_s)
            elapsed = time.time() - started
            return {
                "status": "ok",
                "elapsed_s": elapsed,
                "response_chars": len(result or ""),
                "preview": (result or "")[:200],
            }
        except concurrent.futures.TimeoutError:
            elapsed = time.time() - started
            # The thread is still alive. We can't safely kill it (no signal in
            # Python threads). The future will linger until the underlying
            # socket eventually gives up. We just stop waiting.
            return {
                "status": "hung",
                "elapsed_s": elapsed,
                "response_chars": 0,
                "preview": "",
                "note": "future still running; we stopped waiting",
            }
        except Exception as exc:
            elapsed = time.time() - started
            return {
                "status": "error",
                "elapsed_s": elapsed,
                "response_chars": 0,
                "preview": "",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default="", type=str,
                        help="Model name. Defaults to env CFD_SCIENTIST_MODEL / "
                             "FOAMAGENT_MODEL_VERSION / gpt-5.4")
    parser.add_argument("--n", default=5, type=int, help="Number of sequential invocations.")
    parser.add_argument("--per-call-timeout", default=300, type=float,
                        help="Hard wall-clock cap per invoke, in seconds.")
    parser.add_argument("--case-figs", default="", type=str,
                        help="Optional dir with real PNG figures from a converged case "
                             "to use as the image payload. If empty, synthetic dummies are used.")
    parser.add_argument("--n-dummy-images", default=2, type=int)
    parser.add_argument("--user-requirement", default=_DEFAULT_USER_REQUIREMENT, type=str)
    parser.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT, type=str)
    parser.add_argument("--user-prompt-template", default=_DEFAULT_USER_PROMPT_TEMPLATE, type=str)
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="Same as ResultsInterpreterAgent default (0.1).")
    parser.add_argument("--output-jsonl", default="", type=str,
                        help="If set, append per-call result lines as JSONL.")
    args = parser.parse_args()

    # Resolve model
    model = (
        args.model.strip()
        or os.environ.get("CFD_SCIENTIST_MODEL", "").strip()
        or os.environ.get("FOAMAGENT_MODEL_VERSION", "").strip()
        or "gpt-5.4"
    )
    provider = (
        os.environ.get("CFD_SCIENTIST_LLM_PROVIDER", "")
        or os.environ.get("FOAMAGENT_MODEL_PROVIDER", "")
        or "(env-default)"
    )

    print(f"== test_interpret_hang ==")
    print(f"  provider:           {provider}")
    print(f"  model:              {model}")
    print(f"  N invocations:      {args.n}")
    print(f"  per-call timeout:   {args.per_call_timeout} s")
    print(f"  case_figs:          {args.case_figs or '(none — synthetic dummies)'}")
    print(f"  temperature:        {args.temperature}")
    print()

    # Build payload
    image_data_urls = _gather_image_data_urls(
        Path(args.case_figs).expanduser().resolve() if args.case_figs else None,
        args.n_dummy_images,
    )
    print(f"  image blocks:       {len(image_data_urls)}")
    user_prompt = args.user_prompt_template.format(user_requirement=args.user_requirement)
    print(f"  user_prompt chars:  {len(user_prompt)}")
    print(f"  approx total bytes: {len(user_prompt) + sum(len(u) for u in image_data_urls)} "
          f"(image data URLs are base64; the wire payload is similar)")
    print()

    # Build LLM
    try:
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        llm = create_langchain_llm(model=model, temperature=args.temperature)
    except Exception as exc:
        print(f"FATAL: could not build LLM: {exc}", file=sys.stderr)
        return 1

    # Build messages once — same content on every call (so per-call variance
    # is server-side / network-side, not prompt-side).
    sys_m, user_m = build_messages(args.system_prompt, user_prompt, image_data_urls)
    messages = [sys_m, user_m]

    out_jsonl: Optional[Any] = None
    if args.output_jsonl:
        out_jsonl = open(args.output_jsonl, "a", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for i in range(1, args.n + 1):
        print(f"-- call {i}/{args.n} --", flush=True)
        r = invoke_with_timeout(llm, messages, timeout_s=args.per_call_timeout)
        r["call_index"] = i
        r["model"] = model
        r["provider"] = provider
        rows.append(r)
        if r["status"] == "ok":
            print(f"  OK   elapsed={r['elapsed_s']:.1f}s  resp_chars={r['response_chars']}")
            if r.get("preview"):
                print(f"  preview: {r['preview'][:160]}")
        elif r["status"] == "hung":
            print(f"  HUNG elapsed≥{r['elapsed_s']:.1f}s (timeout fired)  -- {r.get('note','')}")
        else:
            print(f"  ERR  elapsed={r['elapsed_s']:.1f}s  {r.get('error','?')}")
        if out_jsonl is not None:
            out_jsonl.write(json.dumps(r, default=str) + "\n")
            out_jsonl.flush()

    if out_jsonl is not None:
        out_jsonl.close()

    # Summary
    oks = [r for r in rows if r["status"] == "ok"]
    hungs = [r for r in rows if r["status"] == "hung"]
    errs = [r for r in rows if r["status"] == "error"]
    durations = [r["elapsed_s"] for r in oks] if oks else []
    print()
    print("== SUMMARY ==")
    print(f"  total:    {len(rows)}")
    print(f"  ok:       {len(oks)}")
    print(f"  hung:     {len(hungs)}  (per-call cap = {args.per_call_timeout}s)")
    print(f"  errored:  {len(errs)}")
    if durations:
        print(f"  ok mean:  {statistics.mean(durations):.1f}s")
        print(f"  ok median:{statistics.median(durations):.1f}s")
        print(f"  ok min:   {min(durations):.1f}s")
        print(f"  ok max:   {max(durations):.1f}s")
    if hungs:
        print()
        print("HANG CALLS (per-call cap fired):")
        for r in hungs:
            print(f"  call {r['call_index']}: elapsed={r['elapsed_s']:.1f}s")
        print()
        print("If you observe hangs here, the bottleneck is the LLM provider /")
        print("network, not interpret.py. The hang IS the streaming socket sitting")
        print("in poll() waiting for bytes from the model endpoint. Same root cause")
        print("we observed in the live OED runs.")
    return 0 if not hungs else 2


if __name__ == "__main__":
    raise SystemExit(main())

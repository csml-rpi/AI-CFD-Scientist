#!/usr/bin/env python3
"""Live check: the logged tokens equal what each real provider reported, for plain, streamed and
structured calls. Spends a few small requests per provider.

Ground truth is captured independently of the logging path:
  codex    -- the raw HTTP stream: every `response.completed` event's id and usage,
              read off the wire before any of CFD Scientist's parsing
  gemini   -- the usage the Google client parsed from each response (its _generate result)
  qwen     -- the usage the OpenAI-compatible client parsed from each response

Usage: check_token_accounting_live.py [codex] [gemini] [qwen]    (default: all three)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pydantic import BaseModel  # noqa: E402

QWEN_URL = ("https://191566261639970816.us-central1-1071456886687.prediction.vertexai.goog/v1/"
            "projects/gen-lang-client-0168697037/locations/us-central1/endpoints/191566261639970816")
fails: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"\n        got={got}\n        want={want}"))
    if not ok:
        fails.append(label)


class Answer(BaseModel):
    answer: int


def run(provider: str) -> None:
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "CFD_SCIENTIST_EFFORT", "CFD_SCIENTIST_VERTEX_ENDPOINT_URL"):
        os.environ.pop(k, None)
    truth: list[dict] = []
    if provider == "codex":
        os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai-codex", CFD_SCIENTIST_MODEL="gpt-5.6-sol",
                          CFD_SCIENTIST_EFFORT="none")
        from cfd_langgraph.llm import codex_oauth
        real_post = codex_oauth.requests.post

        def tee_post(*a, **kw):
            resp = real_post(*a, **kw)
            orig = resp.iter_lines

            def iter_lines(*ia, **ikw):
                for line in orig(*ia, **ikw):
                    s = line.decode() if isinstance(line, bytes) else line
                    if s and s.startswith("data:") and '"response.completed"' in s:
                        body = json.loads(s[5:]).get("response", {})
                        u = body.get("usage") or {}
                        truth.append({"id": body.get("id"), "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                                      "cached": (u.get("input_tokens_details") or {}).get("cached_tokens")})
                    yield line
            resp.iter_lines = iter_lines
            return resp
        codex_oauth.requests.post = tee_post
        model_id = "gpt-5.6-sol"
    elif provider == "gemini":
        os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="gemini", CFD_SCIENTIST_MODEL="gemini-3.8-flash",
                          GOOGLE_CLOUD_PROJECT="gen-lang-client-0168697037", GOOGLE_CLOUD_LOCATION="global")
        from langchain_google_genai import ChatGoogleGenerativeAI
        orig_gen = ChatGoogleGenerativeAI._generate

        def rec_gen(self, *a, **kw):
            res = orig_gen(self, *a, **kw)
            um = res.generations[0].message.usage_metadata or {}
            truth.append({"in": um.get("input_tokens"), "out": um.get("output_tokens"),
                          "cached": (um.get("input_token_details") or {}).get("cache_read")})
            return res
        ChatGoogleGenerativeAI._generate = rec_gen
        model_id = "gemini-3.8-flash"
    else:
        os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="vertex-endpoint", CFD_SCIENTIST_MODEL="Qwen/Qwen3.8-27B-FP8",
                          CFD_SCIENTIST_EFFORT="none", CFD_SCIENTIST_VERTEX_ENDPOINT_URL=QWEN_URL,
                          CFD_SCIENTIST_VERTEX_TIMEOUT="600", GOOGLE_CLOUD_PROJECT="gen-lang-client-0168697037")
        from langchain_openai import ChatOpenAI
        orig_gen = ChatOpenAI._generate

        def rec_gen(self, *a, **kw):
            res = orig_gen(self, *a, **kw)
            um = res.generations[0].message.usage_metadata or {}
            truth.append({"in": um.get("input_tokens"), "out": um.get("output_tokens"),
                          "cached": (um.get("input_token_details") or {}).get("cache_read")})
            return res
        ChatOpenAI._generate = rec_gen
        orig_stream = ChatOpenAI._stream

        def rec_stream(self, *a, **kw):
            um = {}
            for chunk in orig_stream(self, *a, **kw):
                um = getattr(chunk.message, "usage_metadata", None) or um
                yield chunk
            truth.append({"in": um.get("input_tokens"), "out": um.get("output_tokens"),
                          "cached": (um.get("input_token_details") or {}).get("cache_read")})
        ChatOpenAI._stream = rec_stream
        model_id = "Qwen/Qwen3.8-27B-FP8"

    log = Path(tempfile.mkdtemp())
    os.environ["CFD_TOKEN_LOG_DIR"] = str(log)
    os.environ["CFD_SCIENTIST_STAGE"] = f"live_{provider}"
    from cfd_langgraph.llm.factory import create_langchain_llm
    from cfd_langgraph.utils import structured_output
    llm = create_langchain_llm(model_id, temperature=0.0, agent="live-check")

    print(f"== {provider}")
    llm.invoke("Reply with exactly: OK")
    res = structured_output(llm, Answer).invoke("What is 2+3? Answer as JSON.")
    check(f"{provider}: structured answer parsed", getattr(res, "answer", None), 5)
    # Streamed, as every oed_extensions._llm_invoke call is. On an OpenAI-compatible
    # endpoint this reports usage only if the client asked for it.
    streamed = "".join(str(c.content) for c in llm.stream("Reply with exactly: OK"))
    check(f"{provider}: streamed reply arrived", bool(streamed.strip()), True)

    rows = [json.loads(l) for l in (log / "llm_token_usage_calls.jsonl").open()]
    check(f"{provider}: one logged row per provider response", len(rows), len(truth))
    check(f"{provider}: logged input == provider input", sum(r["input_tokens"] for r in rows), sum(t["in"] or 0 for t in truth))
    check(f"{provider}: logged output == provider output", sum(r["output_tokens"] for r in rows), sum(t["out"] or 0 for t in truth))
    check(f"{provider}: logged cached == provider cached", sum(r.get("cached_input_tokens", 0) for r in rows), sum(t["cached"] or 0 for t in truth))
    check(f"{provider}: every row is real provider usage", {r["token_source"] for r in rows}, {"provider_usage"})
    check(f"{provider}: every row attributed", {(r["agent"], r["stage"]) for r in rows}, {("live-check", f"live_{provider}")})
    if provider == "codex":
        check("codex: logged response ids == ids on the wire",
              sorted(i for r in rows for i in r.get("response_ids", [])), sorted(t["id"] for t in truth))
    for r, t in zip(rows, truth):
        print(f"      row in={r['input_tokens']:>6} out={r['output_tokens']:>4} cached={r.get('cached_input_tokens',0):>6}"
              f"   provider in={t['in']:>6} out={t['out']:>4} cached={t['cached'] or 0:>6}")


if __name__ == "__main__":
    import subprocess
    targets = sys.argv[1:] or ["codex", "gemini", "qwen"]
    if len(targets) > 1:  # one fresh process per provider: the patches above are process-wide
        rc = 0
        for t in targets:
            rc |= subprocess.run([sys.executable, __file__, t]).returncode
        sys.exit(rc)
    run(targets[0])
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)

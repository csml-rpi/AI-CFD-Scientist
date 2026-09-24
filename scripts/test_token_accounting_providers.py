"""Token counts must survive every provider's reply shape, streamed or not.

Found on the Qwen studies: every call made through oed_extensions._llm_invoke
streams (for its heartbeat), and ChatOpenAI asks for usage on a streamed reply
only when talking to api.openai.com. Behind any other URL -- the self-hosted
Qwen endpoint, Vertex's OpenAI route, OpenRouter -- the stream ended with no
counts and the call was logged as zero tokens under "provider_usage".

The OpenAI-compatible server here is a mock transport that records what it
billed for each request, so the log is checked against the wire, including
through the real forked _llm_invoke path.
"""
import json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        fails.append(label)

def rows(d):
    p = Path(d) / "llm_token_usage_calls.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


# --- a mock OpenAI-compatible server that records what it billed -------------
WIRE = None  # path of the ground-truth file; set per test

def _usage_for(n):
    return {"prompt_tokens": 1000 + 7 * n, "completion_tokens": 20 + n,
            "total_tokens": 1020 + 8 * n,
            "prompt_tokens_details": {"cached_tokens": 300 + n},
            "completion_tokens_details": {"reasoning_tokens": 3 + n}}

def _serve(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.read() or b"{}")
    with open(WIRE, "a+", encoding="utf-8") as f:
        f.seek(0)
        n = sum(1 for _ in f)
        usage = _usage_for(n)
        wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        billed = usage if (not body.get("stream") or wants_usage) else None
        # What the provider charged, whether or not the client asked to be told.
        f.write(json.dumps({"stream": bool(body.get("stream")), "usage": usage,
                            "reported": billed is not None}) + "\n")
    base = {"id": f"chatcmpl-{n}", "created": 0, "model": "fake/qwen"}
    if not body.get("stream"):
        return httpx.Response(200, json={
            **base, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hello there"}}],
            "usage": usage})
    events = []
    for piece in ("hello", " there"):
        events.append({**base, "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece},
                                    "finish_reason": None}]})
    events.append({**base, "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if billed:
        events.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
    sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())


class _NoAuth(httpx.Auth):
    def auth_flow(self, request):
        yield request

from cfd_langgraph.llm import vertex_openai
vertex_openai._ADCBearer = lambda *a, **k: _NoAuth()
vertex_openai.httpx.HTTPTransport = lambda *a, **k: httpx.MockTransport(_serve)

os.environ.update(
    CFD_SCIENTIST_LLM_PROVIDER="vertex-endpoint",
    CFD_SCIENTIST_MODEL="fake/qwen",
    CFD_SCIENTIST_VERTEX_ENDPOINT_URL="https://fake.example/v1/projects/p/locations/l/endpoints/e",
    CFD_SCIENTIST_STAGE="manager",
)
os.environ.pop("CFD_TOKEN_LOG_PATH", None)
os.environ.pop("CFD_SCIENTIST_EFFORT", None)

from cfd_langgraph.llm.factory import create_langchain_llm


def wire(d):
    return [json.loads(l) for l in open(Path(d) / "wire.jsonl")]

def logged_matches_wire(label, d):
    w, r = wire(d), rows(d)
    check(f"{label}: one row per request", len(r), len(w))
    for key, wkey in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        check(f"{label}: {key} sum equals what was billed",
              sum(x[key] for x in r), sum(x["usage"][wkey] for x in w))
    check(f"{label}: cached sum equals what was billed",
          sum(x["cached_input_tokens"] for x in r),
          sum(x["usage"]["prompt_tokens_details"]["cached_tokens"] for x in w))
    check(f"{label}: reasoning sum equals what was billed",
          sum(x["reasoning_output_tokens"] for x in r),
          sum(x["usage"]["completion_tokens_details"]["reasoning_tokens"] for x in w))
    check(f"{label}: every row is provider usage", {x["token_source"] for x in r}, {"provider_usage"})


print("== OpenAI-compatible endpoint: plain and streamed calls")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    WIRE = str(Path(d) / "wire.jsonl")
    llm = create_langchain_llm("fake/qwen", temperature=0.0, agent="case-runner")
    llm.invoke([HumanMessage(content="hi")])
    out = None
    for chunk in llm.stream([HumanMessage(content="hi")]):
        out = chunk if out is None else out + chunk
    check("streamed text arrives intact", out.content, "hello there")
    logged_matches_wire("endpoint", d)
    check("every streamed request asked for usage", all(x["reported"] for x in wire(d)), True)
    check("agent label kept", {x["agent"] for x in rows(d)}, {"case-runner"})

print("== the real _llm_invoke path (forked child, streaming heartbeat)")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    WIRE = str(Path(d) / "wire.jsonl")
    import oed_extensions
    text = oed_extensions._llm_invoke([("system", "be brief"), ("user", "hi")], timeout_s=60)
    check("_llm_invoke returns the text", text, "hello there")
    check("_llm_invoke really streamed", [x["stream"] for x in wire(d)], [True])
    logged_matches_wire("_llm_invoke", d)

print("== negative control: a stream that reports nothing is logged as missing")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    WIRE = str(Path(d) / "wire.jsonl")
    llm = create_langchain_llm("fake/qwen", temperature=0.0)
    llm.stream_usage = False  # the old behaviour on every non-OpenAI URL
    for _ in llm.stream([HumanMessage(content="hi")]):
        pass
    r = rows(d)
    check("the call is still logged", len(r), 1)
    check("labelled missing, not provider_usage", r[0]["token_source"], "missing")
    check("server did bill it", wire(d)[0]["reported"], False)


from cfd_langgraph.llm.token_stats import TokenStatsCallbackHandler

def one(result):
    with tempfile.TemporaryDirectory() as d:
        os.environ["CFD_TOKEN_LOG_DIR"] = d
        TokenStatsCallbackHandler("x").on_llm_end(result)
        return rows(d)[0]

print("== Anthropic: raw input_tokens excludes cache; the standard field does not")
r = one(LLMResult(
    generations=[[ChatGeneration(message=AIMessage(content="x", usage_metadata={
        "input_tokens": 1100, "output_tokens": 50, "total_tokens": 1150,
        "input_token_details": {"cache_read": 900, "cache_creation": 100}}))]],
    # What langchain_anthropic puts in llm_output: the API's own usage block.
    llm_output={"usage": {"input_tokens": 100, "output_tokens": 50,
                          "cache_read_input_tokens": 900, "cache_creation_input_tokens": 100}}))
check("input includes cached share", r["input_tokens"], 1100)
check("cached recorded", r["cached_input_tokens"], 900)

print("== ChatOpenAI shape with no usage_metadata: details still read")
r = one(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                  llm_output={"token_usage": _usage_for(0), "model_name": "m"}))
check("input", r["input_tokens"], 1000)
check("cached from prompt_tokens_details", r["cached_input_tokens"], 300)
check("reasoning from completion_tokens_details", r["reasoning_output_tokens"], 3)

print("== Codex estimate keeps its label")
r = one(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                  llm_output={"token_usage": {"prompt_tokens": 40, "completion_tokens": 2},
                              "token_source": "estimate"}))
check("estimate label kept", r["token_source"], "estimate")

print("== a streamed reply's merged model name does not replace the model asked for")
import uuid
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    h, rid = TokenStatsCallbackHandler("x"), uuid.uuid4()
    h.on_chat_model_start({}, [[]], run_id=rid, invocation_params={"model": "qwen/qwen3.8-flash"})
    h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="x", usage_metadata={"input_tokens": 9, "output_tokens": 1, "total_tokens": 10},
        response_metadata={"model_name": "qwen/qwen3.8-flashqwen/qwen3.8-flash"}))]]), run_id=rid)
    check("model is the one asked for", rows(d)[0]["model"], "qwen/qwen3.8-flash")
    check("nothing left behind per run", h._runs, {})

print("== a repaired narrated tool call keeps its token counts")
from cfd_langgraph.llm.tool_call_repair import repair_if_narrated

class _FakeRepairLLM:
    def invoke(self, msgs):
        return AIMessage(content='{"is_tool_call": true, "name": "run_case", "args": {"x": 1}}')

um = {"input_tokens": 5000, "output_tokens": 40, "total_tokens": 5040}
narrated = AIMessage(content="I will now call run_case(x=1)", usage_metadata=um)
fixed = repair_if_narrated(_FakeRepairLLM(), narrated, ["run_case"])
check("repair happened", bool(fixed.tool_calls), True)
check("usage survives the repair", fixed.usage_metadata, um)

print("== a log from before the sidecar keeps all its rows when resumed")
from cfd_langgraph.llm import token_usage_logger as tul
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    legacy = [{"ts": f"t{i}", "input_tokens": i, "output_tokens": 0} for i in range(tul._CALLS_TAIL + 100)]
    (Path(d) / "llm_token_usage.json").write_text(json.dumps({
        "schema_version": "v1", "totals": {"input_tokens": sum(range(len(legacy))), "output_tokens": 0,
                                           "total_tokens": sum(range(len(legacy))), "calls": len(legacy),
                                           "by_model": {}},
        "calls": legacy}))
    tul.append_usage_call(provider="p", model="m", input_tokens=7, output_tokens=0, token_source="t")
    r = rows(d)
    check("sidecar holds every row", len(r), len(legacy) + 1)
    check("oldest row first", r[0]["ts"], "t0")
    check("new row last", r[-1]["input_tokens"], 7)
    check("json keeps only the tail", len(json.loads((Path(d) / "llm_token_usage.json").read_text())["calls"]),
          tul._CALLS_TAIL)

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)

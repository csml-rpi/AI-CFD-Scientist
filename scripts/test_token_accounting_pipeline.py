"""Token accounting across real process boundaries, one property at a time.

A local OpenAI-compatible HTTP server stands in for the provider and records
every request it serves and what it billed. Because it is a real server,
subprocesses -- which no in-process patch can reach -- talk to it exactly as a
study's stages talk to a real provider. Each section below checks one thing:

  1. the study's --out-dir is where accounting lands, whatever the shell had set
  2. with no --out-dir, the prelude's calls still end up in the study's log
  3. a stage launched by the manager, and a script that stage launches in
     turn, are each recorded under their own stage, agent, script and model
  4. inside the manager, each tool's own calls are filed under that tool, even
     with tools running concurrently, and a labelled agent keeps its label;
     in the real manager graph each subagent's calls are filed under it
  5. every call is appended once, under concurrent writers from many processes
  6. a log file cut off mid-write is rebuilt from the per-call rows, not reset
  7. the running totals equal a recomputation from the per-call rows, on every
     breakdown (stage, model, stage x model, stage x agent, token source)
"""
import json, os, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        fails.append(label)


# ---------------------------------------------------------------- fake provider
WIRE = Path(tempfile.mkdtemp()) / "wire.jsonl"
MANAGER_SCRIPT: list = []  # subagents the fake manager delegates to, in order (section 4b)
_WIRE_LOCK = threading.Lock()

def _usage_for(n):
    return {"prompt_tokens": 1000 + 7 * n, "completion_tokens": 20 + n, "total_tokens": 1020 + 8 * n,
            "prompt_tokens_details": {"cached_tokens": 300 + n},
            "completion_tokens_details": {"reasoning_tokens": 3 + n}}

def _fill(schema):
    """A minimal object satisfying a JSON schema's properties."""
    out = {}
    for name, prop in (schema.get("properties") or {}).items():
        kinds = {prop.get("type")} | {a.get("type") for a in prop.get("anyOf", [])}
        out[name] = 5 if "integer" in kinds else (None if "null" in kinds else "x")
    return out

class _Provider(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        stream = bool(body.get("stream"))
        wants = bool((body.get("stream_options") or {}).get("include_usage"))
        with _WIRE_LOCK:
            n = sum(1 for _ in open(WIRE)) if WIRE.exists() else 0
            usage = _usage_for(n)
            with open(WIRE, "a") as f:
                f.write(json.dumps({"n": n, "model": body.get("model"), "stream": stream,
                                    "reported": (not stream) or wants, "usage": usage}) + "\n")
        message = {"role": "assistant", "content": "hello there"}
        rf = body.get("response_format") or {}
        tool_names = [t.get("function", {}).get("name") for t in body.get("tools") or []]
        if MANAGER_SCRIPT and "task" in tool_names:
            # The manager: delegate to general-purpose, then to case-runner, then stop.
            done = sum(1 for m in body.get("messages") or [] if m.get("role") == "tool")
            if done < len(MANAGER_SCRIPT):
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_{n}", "type": "function", "function": {"name": "task", "arguments": json.dumps(
                        {"description": "Say hello and stop.", "subagent_type": MANAGER_SCRIPT[done]})}}]}
            else:
                message = {"role": "assistant", "content": "finished"}
        elif MANAGER_SCRIPT:
            message = {"role": "assistant", "content": "done"}  # a subagent's whole answer
        elif rf.get("type") == "json_schema":
            message["content"] = json.dumps(_fill((rf.get("json_schema") or {}).get("schema") or {}))
        elif body.get("tools") and body.get("tool_choice"):
            fn = body["tools"][0]["function"]
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{n}", "type": "function",
                "function": {"name": fn["name"], "arguments": json.dumps(_fill(fn.get("parameters") or {}))}}]}
        base = {"id": f"chatcmpl-{n}", "created": 0, "model": body.get("model")}
        if not stream:
            payload = json.dumps({**base, "object": "chat.completion", "usage": usage,
                                  "choices": [{"index": 0, "finish_reason": "stop", "message": message}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode())
            return
        events = [{**base, "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"role": "assistant", "content": message.get("content") or ""},
                                "finish_reason": None}]},
                  {**base, "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        if wants:
            events.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode())

server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
threading.Thread(target=server.serve_forever, daemon=True).start()

for k in ("CFD_TOKEN_LOG_PATH", "CFD_TOKEN_LOG_DIR", "CFD_SCIENTIST_STAGE", "CFD_SCIENTIST_STAGE_OWNER",
          "CFD_SCIENTIST_EFFORT", "OPENAI_API_BASE"):
    os.environ.pop(k, None)
os.environ.update(
    CFD_SCIENTIST_LLM_PROVIDER="openai",
    CFD_SCIENTIST_MODEL="fake/model-a",
    OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
    OPENAI_API_KEY="fake",
)

from cfd_langgraph.llm import token_usage_logger as tul  # noqa: E402
from cfd_langgraph.llm.factory import create_langchain_llm  # noqa: E402


def wire():
    return [json.loads(l) for l in open(WIRE)] if WIRE.exists() else []

def rows(d):
    return tul.read_rows(Path(d) / "llm_token_usage_calls.jsonl")

def totals(d):
    return json.loads((Path(d) / "llm_token_usage.json").read_text())["totals"]

def consistent(label, d):
    """Section 7, applied after every section: running totals == recomputation."""
    t, r = totals(d), rows(d)
    check(f"{label}: totals equal a recomputation from the rows", t, tul.rebuild_totals(r))

def wire_since(mark):
    return wire()[mark:]

def matches_wire(label, d, mark):
    w, r = wire_since(mark), rows(d)
    check(f"{label}: one row per request served", len(r), len(w))
    check(f"{label}: input tokens equal what was billed",
          sum(x["input_tokens"] for x in r), sum(x["usage"]["prompt_tokens"] for x in w))
    check(f"{label}: output tokens equal what was billed",
          sum(x["output_tokens"] for x in r), sum(x["usage"]["completion_tokens"] for x in w))
    check(f"{label}: cached tokens equal what was billed",
          sum(x["cached_input_tokens"] for x in r),
          sum(x["usage"]["prompt_tokens_details"]["cached_tokens"] for x in w))
    check(f"{label}: every request reported usage", all(x["reported"] for x in w), True)
    check(f"{label}: no row is missing usage", [x for x in r if x["token_source"] != "provider_usage"], [])
    for model in sorted({x["model"] for x in w}):
        check(f"{label}: {model} billed tokens land under {model}",
              sum(x["input_tokens"] for x in r if x["model"] == model),
              sum(x["usage"]["prompt_tokens"] for x in w if x["model"] == model))


# ------------------------------------------------------------------- section 1
print("== 1. accounting lands in the study's --out-dir, whatever the shell had set")
from cfd_langgraph.cli import repl  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    study, stray_dir, stray_file = Path(tmp) / "study", Path(tmp) / "stray_dir", Path(tmp) / "stray" / "log.json"
    os.environ["CFD_TOKEN_LOG_DIR"] = str(stray_dir)
    os.environ["CFD_TOKEN_LOG_PATH"] = str(stray_file)
    repl._bind_token_log(study)
    mark = len(wire())
    create_langchain_llm("fake/model-a", temperature=0.0).invoke("hi")
    check("log written inside the study folder", (study / "llm_token_usage.json").is_file(), True)
    check("per-call rows written inside the study folder", (study / "llm_token_usage_calls.jsonl").is_file(), True)
    check("nothing written to the stray directory", stray_dir.exists(), False)
    check("nothing written to the stray file", stray_file.exists(), False)
    matches_wire("out-dir", study, mark)
    consistent("out-dir", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


# ------------------------------------------------------------------- section 2
print("== 2. with no --out-dir, the prelude's calls end up in the study's own log")
with tempfile.TemporaryDirectory() as tmp:
    topic = Path(tmp) / "topic.md"
    topic.write_text("Study the lid-driven cavity at Re 100.")
    started = {}
    real_start, cwd = repl._start_session, os.getcwd()
    repl._start_session = lambda settings, out_dir, payload: started.setdefault("out_dir", Path(out_dir))
    os.chdir(tmp)
    try:
        mark = len(wire())
        repl.cmd_run(repl.build_parser().parse_args(["run", "--topic-file", str(topic)]))
    finally:
        repl._start_session = real_start
        os.chdir(cwd)
    study = (Path(tmp) / started["out_dir"]).resolve()
    r = rows(study)
    check("the prelude made a model call", len(wire_since(mark)) >= 1, True)
    check("its rows are in the study folder", len(r), len(wire_since(mark)))
    check("filed under stage cli_prelude", {x["stage"] for x in r}, {"cli_prelude"})
    check("nothing left in the working directory", (Path(tmp) / "llm_token_usage.json").exists(), False)
    consistent("prelude", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


# ------------------------------------------------------------------- section 3
print("== 3. a manager-launched stage, and the script it launches, each recorded as themselves")
PROBE = r'''
import os, subprocess, sys
sys.path.insert(0, {src!r}); sys.path.insert(0, {scripts!r})
from pydantic import BaseModel
from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.utils import structured_output
class Answer(BaseModel):
    answer: int
llm = create_langchain_llm(os.environ["CFD_SCIENTIST_MODEL"], temperature=0.0)
llm.invoke("plain")
assert structured_output(llm, Answer).invoke("2+3?").answer == 5
for _ in llm.stream("streamed"):
    pass
import oed_extensions
oed_extensions._llm_invoke([("user", "forked and streamed")], timeout_s=60)
if {nested!r}:
    subprocess.run([sys.executable, {nested!r}], check=True,
                   env={{**os.environ, "CFD_SCIENTIST_MODEL": "fake/model-b"}})
'''
from cfd_langgraph.manager import tools as mtools  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    os.environ["CFD_TOKEN_LOG_DIR"] = str(study)
    nested = Path(tmp) / "probe_nested.py"
    nested.write_text(PROBE.format(src=str(ROOT / "src"), scripts=str(ROOT / "scripts"), nested=""))
    stage = Path(tmp) / "probe_stage.py"
    stage.write_text(PROBE.format(src=str(ROOT / "src"), scripts=str(ROOT / "scripts"), nested=str(nested)))
    mark = len(wire())
    proc = mtools._run_script([str(stage)], cwd=Path(tmp), timeout=300)
    check("the stage script ran cleanly", proc.returncode, 0)
    if proc.returncode:
        print(proc.stdout[-2000:], proc.stderr[-3000:])
    r = rows(study)
    matches_wire("stage + nested", study, mark)
    by = lambda key: sorted({(x["stage"], x[key]) for x in r})
    check("stages", sorted({x["stage"] for x in r}), ["probe_stage", "probe_stage/probe_nested"])
    check("agent defaults to the stage's own name, never 'manager'", by("agent"),
          [("probe_stage", "probe_stage"), ("probe_stage/probe_nested", "probe_stage/probe_nested")])
    check("script recorded per row", by("script"),
          [("probe_stage", "probe_stage"), ("probe_stage/probe_nested", "probe_nested")])
    check("model recorded per stage", by("model"),
          [("probe_stage", "fake/model-a"), ("probe_stage/probe_nested", "fake/model-b")])
    check("plain, structured, streamed and forked calls all logged in the stage",
          sum(1 for x in r if x["stage"] == "probe_stage"), 4)
    check("the forked child is logged under the same stage",
          len({x["pid"] for x in r if x["stage"] == "probe_stage"}) >= 2, True)
    t = totals(study)["by_stage"]
    check("by_stage x by_model: nested stage used only model-b",
          sorted(t["probe_stage/probe_nested"]["by_model"]), ["fake/model-b"])
    check("by_stage x by_model: parent stage used only model-a",
          sorted(t["probe_stage"]["by_model"]), ["fake/model-a"])
    consistent("stage + nested", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


# ------------------------------------------------------------------- section 4
print("== 4. inside the manager: each tool's calls filed under that tool, concurrently")
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
import oed_extensions  # noqa: E402

def tool_alpha() -> str:
    """Makes three unlabelled calls, one of them through the forked helper."""
    llm = create_langchain_llm("fake/model-a", temperature=0.0)
    for _ in range(2):
        llm.invoke("alpha")
        time.sleep(0.05)
    oed_extensions._llm_invoke([("user", "alpha forked")], timeout_s=60)
    return "ok"

def tool_beta() -> str:
    """Makes three unlabelled calls and one through a labelled agent model."""
    llm = create_langchain_llm("fake/model-a", temperature=0.0)
    for _ in range(3):
        llm.invoke("beta")
        time.sleep(0.05)
    create_langchain_llm("fake/model-a", temperature=0.0, agent="case-runner").invoke("labelled")
    return "ok"

with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    os.environ["CFD_TOKEN_LOG_DIR"] = str(study)
    with tul.stage_scope("manager"):
        # The manager's tools run in LangGraph's ToolNode, which executes a
        # turn's tool calls concurrently in a thread pool: run them the same way.
        from langgraph.graph import END, START, MessagesState, StateGraph
        g = StateGraph(MessagesState)
        g.add_node("tools", ToolNode([StructuredTool.from_function(mtools._with_progress(f))
                                      for f in (tool_alpha, tool_beta)]))
        g.add_edge(START, "tools")
        g.add_edge("tools", END)
        mark = len(wire())
        g.compile().invoke({"messages": [AIMessage(content="", tool_calls=[
            {"name": "tool_alpha", "args": {}, "id": "a", "type": "tool_call"},
            {"name": "tool_beta", "args": {}, "id": "b", "type": "tool_call"}])]})
        create_langchain_llm("fake/model-a", temperature=0.0, agent="manager").invoke("manager's own turn")
    r = rows(study)
    matches_wire("tools", study, mark)
    count = lambda agent: sum(1 for x in r if x["agent"] == agent)
    check("tool_alpha's own calls, forked one included", count("tool:tool_alpha"), 3)
    check("tool_beta's own calls", count("tool:tool_beta"), 3)
    check("a labelled agent inside a tool keeps its label", count("case-runner"), 1)
    check("the manager's own turn stays 'manager'", count("manager"), 1)
    check("everything in the manager stage", {x["stage"] for x in r}, {"manager"})
    check("by_agent breakdown matches", {a: v["calls"] for a, v in totals(study)["by_stage"]["manager"]["by_agent"].items()},
          {"tool:tool_alpha": 3, "tool:tool_beta": 3, "case-runner": 1, "manager": 1})
    consistent("tools", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


# ------------------------------------------------------------------ section 4b
print("== 4b. the real manager graph: each subagent's calls filed under that subagent")
# deepagents adds a general-purpose subagent of its own and hands it the
# manager's model; case-runner has a model of its own.
from cfd_langgraph.config import get_settings  # noqa: E402
from cfd_langgraph.manager import build_manager  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    study.mkdir()
    os.environ["CFD_TOKEN_LOG_DIR"] = str(study)
    MANAGER_SCRIPT[:] = ["general-purpose", "case-runner"]
    try:
        with tul.stage_scope("manager"):
            graph, stack = build_manager(get_settings(), study)
            mark = len(wire())
            graph.invoke({"messages": [{"role": "user", "content": "Delegate twice, then stop."}]},
                         config={"configurable": {"thread_id": "tok-pipeline"}, "recursion_limit": 40})
            stack.close()
    finally:
        MANAGER_SCRIPT[:] = []
    r = rows(study)
    matches_wire("manager graph", study, mark)
    agents = {}
    for x in r:
        agents[x["agent"]] = agents.get(x["agent"], 0) + 1
    check("manager turns filed as manager (delegate, delegate, stop)", agents.get("manager"), 3)
    check("the general-purpose subagent, on the manager's model, filed as itself", agents.get("general-purpose"), 1)
    check("the case-runner subagent filed as itself", agents.get("case-runner"), 1)
    check("no other agents", sorted(agents), ["case-runner", "general-purpose", "manager"])
    consistent("manager graph", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


# ------------------------------------------------------------------- section 5
print("== 5. every call appended exactly once, with many processes writing at once")
WRITER = r'''
import sys; sys.path.insert(0, {src!r})
from cfd_langgraph.llm import token_usage_logger as tul
for i in range(60):
    tul.append_usage_call(provider="p", model="m%d" % ({k} % 2), input_tokens=i + 1, output_tokens=1,
                          token_source="provider_usage", cached_input_tokens=1)
'''
with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    env = {**os.environ, "CFD_TOKEN_LOG_DIR": str(study), "CFD_SCIENTIST_STAGE": "writers"}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER.format(src=str(ROOT / "src"), k=k)], env=env)
             for k in range(8)]
    check("all writers exited cleanly", [p.wait() for p in procs], [0] * 8)
    r = rows(study)
    check("480 rows, none lost or doubled", len(r), 480)
    t = totals(study)
    check("totals.calls", t["calls"], 480)
    check("totals.input_tokens exact", t["input_tokens"], 8 * sum(range(1, 61)))
    check("by_model split exact", {m: v["calls"] for m, v in t["by_model"].items()}, {"m0": 240, "m1": 240})
    check("json still valid and holds the tail", len(json.loads((study / "llm_token_usage.json").read_text())["calls"]),
          min(480, tul._CALLS_TAIL))
    consistent("concurrent writers", study)


# ------------------------------------------------------------------- section 6
print("== 6. a log cut off mid-write is rebuilt from the per-call rows, not reset")
with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    os.environ["CFD_TOKEN_LOG_DIR"] = str(study)
    for i in range(5):
        tul.append_usage_call(provider="p", model="m", input_tokens=100, output_tokens=10, token_source="provider_usage")
    log = study / "llm_token_usage.json"
    log.write_text(log.read_text()[:57])  # what a kill mid-write used to leave
    tul.append_usage_call(provider="p", model="m", input_tokens=100, output_tokens=10, token_source="provider_usage")
    t = totals(study)
    check("calls survive the corruption", t["calls"], 6)
    check("tokens survive the corruption", t["input_tokens"], 600)
    check("the broken file is kept for inspection", len(list(study.glob("llm_token_usage.json.corrupt-*"))), 1)
    check("no temp files left behind", list(study.glob(".*.tmp")), [])
    consistent("rebuild", study)
os.environ.pop("CFD_TOKEN_LOG_DIR", None)


print("== stage labels: set, nested and restored")
os.environ.pop("CFD_SCIENTIST_STAGE", None)
with tul.stage_scope("outer"):
    with tul.stage_scope("inner"):
        check("inner scope wins", tul.current_stage(), "inner")
    check("outer restored", tul.current_stage(), "outer")
check("unset after both scopes", os.environ.get("CFD_SCIENTIST_STAGE"), None)
check("owner unset after both scopes", os.environ.get("CFD_SCIENTIST_STAGE_OWNER"), None)
out = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r});"
                      "from cfd_langgraph.llm import token_usage_logger as t; print(t.current_stage())"],
                     env={**os.environ, "CFD_SCIENTIST_STAGE": "s", "CFD_SCIENTIST_STAGE_OWNER": "other"},
                     capture_output=True, text=True).stdout.strip()
check("python -c (no script name) keeps the inherited stage", out, "s")

server.shutdown()
print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)

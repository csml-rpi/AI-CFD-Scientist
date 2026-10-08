"""Loop guards and flagged requirements, as seen on the Qwen cavity and hill runs.

  1. the same call returning the same result 3 times in a row gets a warning,
     and from the 6th it is not run (a case-runner wrote the same script
     10,000 times on qwen cavity_r8, warnings and all)
  2. a call failing identically 12 times is handed back to the model as
     blocked; the study is not paused for a person (qwen periodic_hill sat 9h44m)
  3. the pause message no longer claims Ctrl-C was pressed
  4. a requirement the checker flagged keeps the checker's issues, run_case_native
     returns them, and accepts a clarification for that case only
"""
import json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai", CFD_SCIENTIST_MODEL="fake/model",
                  OPENAI_API_KEY="fake", OPENAI_BASE_URL="http://127.0.0.1:9/v1")

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        fails.append(label)

from cfd_langgraph.manager import tools as mt  # noqa: E402
from cfd_langgraph.manager.control import GLOBAL_INTERRUPT, build_interrupt_on  # noqa: E402

print("== 1. the same call with the same result, repeated")
def write_text_file(path: str, content: str) -> dict:
    """Write text to a file."""
    return {"path": path, "bytes_written": len(content)}
w = mt._with_progress(write_text_file)
out = [w("/x/run.sh", "same") for _ in range(8)]
check("no warning before the 3rd identical call", any("loop_warning" in r for r in out[:2]), False)
check("warning on the 3rd", "loop_warning" in out[2], True)
check("the original result is kept with the warning", out[2]["bytes_written"], 4)
check("not run from the 6th", all(r.get("error", "").startswith("Not run") for r in out[5:]), True)
check("the count is reported", out[7].get("repeated_identical_calls"), 8)
check("a different argument starts a fresh count", "loop_warning" in w("/x/other.sh", "same"), False)

counter = {"n": 0}
def list_directory(path: str) -> dict:
    counter["n"] += 1
    return {"entries": counter["n"]}  # the result changes every call
l = mt._with_progress(list_directory)
check("a changing result never warns", any("loop_warning" in l("/x") for _ in range(12)), False)

def read_text_file(path: str) -> str:
    return "same text"
r = mt._with_progress(read_text_file)
texts = [r("/x/f") for _ in range(3)]
check("a text result gets the warning appended", "exact call 3 times" in texts[2], True)

def oed_candidate_status(candidate_dir: str) -> dict:
    return {"state": "running"}
st = mt._with_progress(oed_candidate_status)
check("status polling is never flagged", any("loop_warning" in st("/c") for _ in range(20)), False)

print("== 2. an identical failure, repeated, is handed back, not paused")
GLOBAL_INTERRUPT.clear()
def oed_propose_candidates(num_candidates: int) -> dict:
    return {"error": "No candidate survived validation this round."}
f = mt._with_progress(oed_propose_candidates)
res = [f(1) for _ in range(mt._REPEAT_FAILURE_ABORT)]
check("the study is not paused", GLOBAL_INTERRUPT.is_set(), False)
check("the model is told the call is blocked", "BLOCKED" in res[-1]["error"], True)
check("the original error is still there", res[-1]["error"].startswith("No candidate survived"), True)
check("not blocked before the threshold", "BLOCKED" in res[-2]["error"], False)

print("== 2b. a blocked call is answered from its error, not run again")
mt._BLOCKED.clear(); mt._REPEAT_FAILURES.clear()
ran = {"n": 0}
def oed_setup_search(topic: str) -> dict:
    ran["n"] += 1
    return {"error": "Blocked: this physics group has no converged mesh-gate selection."}
g = mt._with_progress(oed_setup_search)
for _ in range(mt._REPEAT_FAILURE_ABORT):
    g("t")
check("the tool really ran up to the threshold", ran["n"], mt._REPEAT_FAILURE_ABORT)
after = [g("t") for _ in range(20)]
check("further identical calls are not run at all", ran["n"], mt._REPEAT_FAILURE_ABORT)
check("they still answer with the same error", "BLOCKED" in after[-1]["error"], True)
check("and say the call is not being run", "without being run" in after[-1]["error"], True)
check("a different argument is still run", (g("other"), ran["n"])[1], mt._REPEAT_FAILURE_ABORT + 1)
mt._BLOCKED.clear()
check("the block lifts so a fixed precondition recovers", (g("t"), ran["n"])[1], mt._REPEAT_FAILURE_ABORT + 2)

print("== 2c. a success clears the block")
mt._BLOCKED.clear(); mt._REPEAT_FAILURES.clear()
state = {"fail": True, "runs": 0}
def run_case_native(case_id: str) -> dict:
    state["runs"] += 1
    return {"error": "Blocked: no converged mesh-gate selection."} if state["fail"] else {"status": "success"}
rc = mt._with_progress(run_case_native)
for _ in range(mt._REPEAT_FAILURE_ABORT):
    rc("case_001")
rc("case_001")
runs_while_blocked = state["runs"]
mt._BLOCKED.clear()          # the manager fixed the mesh gate; block window elapsed
state["fail"] = False
check("blocked calls did not reach the tool", runs_while_blocked, mt._REPEAT_FAILURE_ABORT)
check("it runs again once unblocked", rc("case_001").get("status"), "success")
check("and stays unblocked after succeeding", rc("case_001").get("status"), "success")

print("== 2d. subagents get a step limit, and report back when they hit it")
from cfd_langgraph.manager import subagents as sa
import deepagents.middleware.subagents as dsa
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


def _looping_graph(tool_text):
    """agent -> tools -> agent -> ... forever; every tool result is ``tool_text``."""
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    steps = {"n": 0}
    def agent(state):
        steps["n"] += 1
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": f"c{steps['n']}"}])]}
    def tools(state):
        return {"messages": [ToolMessage(content=tool_text, tool_call_id=f"c{steps['n']}")]}
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_edge("agent", "tools")
    g.add_edge("tools", "agent")
    return g.compile(), steps


def _capped_with(graph, precheck=None):
    real_create = dsa.create_sub_agent
    dsa.create_sub_agent = lambda spec, **kw: graph
    try:
        return sa._capped({"name": "case-runner", "description": "d"}, precheck=precheck)
    finally:
        dsa.create_sub_agent = real_create


start = {"messages": [HumanMessage(content="go")]}
check("a limit is set, well under the framework default", 0 < sa._SUBAGENT_STEP_LIMIT < 9999, True)
graph, steps = _looping_graph('{"ok": true}')
capped = _capped_with(graph)
check("the subagent is compiled, not a raw spec", "runnable" in capped, True)
# The task tool passes the manager's own limit; it must not override this one.
out = capped["runnable"].invoke(start, {"recursion_limit": 9999})
check("the limit holds when the caller passes a larger one", steps["n"] <= sa._SUBAGENT_STEP_LIMIT, True)
text = out["messages"][0].content
check("hitting the limit answers instead of raising", "was stopped after" in text, True)
check("and says the work is not lost", "Nothing it did is lost" in text, True)
check("and tells the manager what to do next", "mesh gate" in text, True)

graph, steps = _looping_graph('{"error": "Not run: this exact call has been made 9 times in a row"}')
out = _capped_with(graph)["runnable"].invoke(start, {"recursion_limit": 9999})
check("a run of refused calls stops it early", steps["n"] <= sa._REFUSAL_STOP + 1, True)
check("and says why", "refused calls among its recent ones" in out["messages"][0].content, True)

graph, steps = _looping_graph('{"ok": true}')
out = _capped_with(graph, precheck=lambda: "Not started: no converged gate")["runnable"].invoke(start)
check("a precheck can refuse to start it", (steps["n"], "Not started" in out["messages"][0].content), (0, True))
with tempfile.TemporaryDirectory() as tmp:
    check("no converged mesh gate stops case-runners", bool(sa._no_converged_mesh_gate(Path(tmp))), True)
    spec = Path(tmp) / "mesh_gate" / "g" / "selected_mesh_spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"converged": true}')
    check("a converged one lets them start", sa._no_converged_mesh_gate(Path(tmp)), None)

print("== 2e. repeated rounds are collapsed, thinking turns on when stuck, refusals stop the agent")
from cfd_langgraph.llm import caching as cm

def _round(i, name, args, result):
    return [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"r{i}"}]),
            ToolMessage(content=result, tool_call_id=f"r{i}")]

REF = '{"error": "Not run: this exact call has been made 7 times in a row"}'
msgs = [HumanMessage(content="go")] + _round(0, "read", {"p": "T"}, "T body")
for i in range(1, 6):
    msgs += _round(i, "read", {"p": "T"}, REF.replace("7", str(5 + i)))
msgs += _round(9, "read", {"p": "U"}, "U body")
out = cm.collapse_repeated_rounds(msgs)
kept_ids = [m.tool_call_id for m in out if getattr(m, "type", "") == "tool"]
check("the first real read is kept", kept_ids[0], "r0")
check("five identical refusals collapse to the first and the latest", kept_ids[1:3], ["r1", "r5"])
check("a different call is untouched", kept_ids[-1], "r9")
check("the latest repeat says how many there were",
      "made 5 times" in next(m for m in out if getattr(m, "tool_call_id", "") == "r5").content, True)
check("a conversation without repeats is returned as is", cm.collapse_repeated_rounds(msgs[:4]) is msgs[:4] or
      cm.collapse_repeated_rounds(msgs[:4]) == msgs[:4], True)
alt = [HumanMessage(content="go")]
for i in range(8):
    alt += _round(i, "write", {"f": "ab"[i % 2]}, REF)
ids = [m.tool_call_id for m in cm.collapse_repeated_rounds(alt) if getattr(m, "type", "") == "tool"]
check("two alternating refused calls each keep their first and latest", ids, ["r0", "r1", "r6", "r7"])

alternating = [HumanMessage(content="go")]
for i in range(40):
    ok = '{"ok": true, "n": %d}' % i if i % 5 == 4 else REF
    alternating += _round(i, "write" if i % 5 != 4 else "run", {"f": "a"}, ok)
check("a loop of refusals broken by an occasional call that runs is caught", cm.stuck(alternating, 30) > 0, True)
check("a few refusals are not", cm.stuck(msgs, 30), 0)
check("the last round was refused", cm._last_round_refused(msgs[:-2]), True)
check("the last round was not refused", cm._last_round_refused(msgs), False)
class _M: extra_body = {"chat_template_kwargs": {"enable_thinking": False}}, 
_M.extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
on = cm._thinking_on(_M)
check("thinking is switched on through the same template switch",
      (on["chat_template_kwargs"]["enable_thinking"], on["top_k"]), (True, 20))
class _N: extra_body = None
check("a model without that switch is left alone", cm._thinking_on(_N), None)

# End to end: an agent whose model keeps making the same call is stopped.
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

seen_lengths = []
class _Loop(BaseChatModel):
    @property
    def _llm_type(self): return "loop"
    def bind_tools(self, tools, **kw): return self
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        seen_lengths.append(len(messages))
        n = len(seen_lengths)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"name": "same_call", "args": {"x": 1}, "id": f"c{n}"}]))])

def same_call(x: int) -> dict:
    """Always the same answer."""
    return {"ok": True, "value": 1}
mt._REPEAT_SAME.clear()
agent = create_agent(_Loop(), tools=[mt._with_progress(same_call)],
                     middleware=[cm.build_loop_control_middleware(refusal_stop=5)])
final = agent.invoke({"messages": [HumanMessage(content="go")]}, {"recursion_limit": 200})
last = final["messages"][-1].content
check("an agent stuck on refused calls is stopped", last.startswith("Stopped:"), True)
check("after the set number of refusals", len(seen_lengths) <= mt._REPEAT_SAME_REFUSE + 5, True)
check("and the model never saw the pile of repeats", max(seen_lengths) <= 12, True)

print("== 2e2. an empty reply is asked again, not taken as the end")
replies = iter([AIMessage(content=""), AIMessage(content="All done: the study is finished.")])
asked: list = []

class _Empty(_Loop):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        asked.append(messages[-1].content if messages else "")
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

agent = create_agent(_Empty(), tools=[mt._with_progress(same_call)],
                     middleware=[cm.build_loop_control_middleware(refusal_stop=5)])
final = agent.invoke({"messages": [HumanMessage(content="go")]}, {"recursion_limit": 20})
check("the run continues past an empty reply", final["messages"][-1].content, "All done: the study is finished.")
check("and the model is told its reply was empty", "reply was empty" in str(asked[-1]), True)

print("== 2f. a study is offered only its own tools once its kind is known")
from cfd_langgraph.manager.deep_agent import build_study_mode_middleware
from langchain_core.messages import SystemMessage

class _Req:
    def __init__(self, tools, system):
        self.tools, self.system_message = tools, system
    def override(self, **kw):
        r = _Req(kw.get("tools", self.tools), kw.get("system_message", self.system_message))
        return r
class _T:
    def __init__(self, name): self.name = name
names = ["read_text_file", "impl_run", "run_mesh_gate", "oed_setup_search", "task", "write_paper"]
with tempfile.TemporaryDirectory() as tmp:
    mw = build_study_mode_middleware(Path(tmp))
    got = {}
    mw.wrap_model_call(_Req([_T(n) for n in names], SystemMessage(content="PROMPT")), lambda r: got.setdefault("r", r))
    check("before the kind is known, every tool is offered", [t.name for t in got["r"].tools], names)
    Path(tmp, "study_mode.json").write_text('{"mode": "implementation"}')
    got = {}
    mw.wrap_model_call(_Req([_T(n) for n in names], SystemMessage(content="PROMPT")), lambda r: got.setdefault("r", r))
    check("an implementation study gets its own tools only", [t.name for t in got["r"].tools], ["read_text_file", "impl_run"])
    check("and is told its kind first", got["r"].system_message.content.startswith("STUDY TYPE: implementation"), True)
    Path(tmp, "study_mode.json").write_text('{"mode": "solver"}')
    got = {}
    mw.wrap_model_call(_Req([_T(n) for n in names], SystemMessage(content="PROMPT")), lambda r: got.setdefault("r", r))
    check("a solver study loses only the implementation tools",
          [t.name for t in got["r"].tools], ["read_text_file", "run_mesh_gate", "oed_setup_search", "task", "write_paper"])

# The real thing still compiles, with deepagents' own entrypoint.
from cfd_langgraph.llm.factory import create_langchain_llm
real_spec = sa.build_case_runner_subagent([write_text_file], create_langchain_llm("fake/model", 0.0), Path("/tmp"))
check("the real case-runner compiles too", "runnable" in real_spec, True)

print("== 3. the pause message")
desc = build_interrupt_on([oed_propose_candidates])["oed_propose_candidates"]["description"]
check("does not claim Ctrl-C was pressed", "Ctrl-C was pressed" in desc, False)

print("== 4. a flagged requirement: issues kept, returned, and clarifiable")
from cfd_langgraph.config import get_settings  # noqa: E402
import cfd_langgraph.agents.hypothesis_agent as ha  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    (out_dir / "hypotheses_approved.json").write_text(json.dumps({"approved_hypotheses": [{
        "candidate_id": "cand_01", "idea": {"study_id": "s1", "experiments": [
            {"experiment_id": "exp_001", "name": "a"}, {"experiment_id": "exp_002", "name": "b"}]}}]}))
    (out_dir / "hypotheses_ranked.json").write_text(json.dumps({"research_topic": "t"}))
    tools = {fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out_dir)["manager_tools"]}
    runner = {fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out_dir)["case_runner_tools"]}

    original = ha.HypothesisAgent.generate_validated_requirement
    def fake(self, idea, simulation, run_topic="", **kw):
        if simulation["simulation_id"] == "exp_001":
            return {"requirement": "Run the cavity at Re = 100, 400 and 1000.", "valid": False,
                    "final_verdict": {"valid": False, "issues": ["Lists three Reynolds numbers for one case."]}}
        return {"requirement": "Run the cavity at Re = 400.", "valid": True}
    ha.HypothesisAgent.generate_validated_requirement = fake
    try:
        gen = tools["generate_case_requirements"]()
    finally:
        ha.HypothesisAgent.generate_validated_requirement = original
    reqs = {r["case_id"]: r for r in json.loads((out_dir / "requirements.json").read_text())}
    check("the flagged requirement keeps the checker's issues",
          reqs["case_001"].get("requirement_issues"), ["Lists three Reynolds numbers for one case."])
    check("a passing requirement carries none", "requirement_issues" in reqs["case_002"], False)
    check("the manager is told each flagged case's issues",
          gen.get("unvalidated_issues"), {"case_001": ["Lists three Reynolds numbers for one case."]})

    seed = out_dir / "mesh_gate" / "g" / "baseline"
    seed.mkdir(parents=True)
    (out_dir / "mesh_gate" / "g" / "selected_mesh_spec.json").write_text(
        json.dumps({"converged": True, "selected_level": str(seed)}))
    seen = []
    real_run = mt.foam_native.run_foam_case
    mt.foam_native.run_foam_case = lambda llm, case_dir, requirement, **kw: (
        seen.append(requirement) or {"status": "success", "success": True})
    try:
        run = runner["run_case_native"]
        flagged_text = reqs["case_001"]["user_requirement_text"]
        r1 = run("case_001", flagged_text, physics_group="g")
        check("the run returns the checker's issues", r1.get("requirement_checker_issues"),
              ["Lists three Reynolds numbers for one case."])
        check("and says what to do about them", "clarification=" in r1.get("requirement_checker_note", ""), True)
        check("the writer got the requirement unchanged", seen[-1], flagged_text)
        r2 = run("case_001", flagged_text, physics_group="g", clarification="This case is Re = 400, nu = 0.0025.")
        check("a clarification reaches the writer", "This case is Re = 400" in seen[-1], True)
        check("after the verbatim requirement", seen[-1].startswith(flagged_text), True)
        check("and is recorded in the result", r2.get("clarification"), "This case is Re = 400, nu = 0.0025.")
        ok_text = reqs["case_002"]["user_requirement_text"]
        r3 = run("case_002", ok_text, physics_group="g", clarification="anything")
        check("a clarification on a passing requirement is refused", "only accepted" in r3.get("error", ""), True)
        r4 = run("case_002", ok_text, physics_group="g")
        check("a passing requirement reports no issues", "requirement_checker_issues" in r4, False)
        run("case_002", "a paraphrase of the requirement", physics_group="g")
        check("a paraphrased requirement_text is not what runs: the approved text for the case_id is",
              seen[-1], ok_text)
        check("an unknown case_id is refused with the valid ones listed",
              "Valid case_ids" in run("case_999", "", physics_group="g").get("error", ""), True)
    finally:
        mt.foam_native.run_foam_case = real_run

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)

"""Token accounting must be complete: every call lands in the study's own log,
attributed to a stage and an agent. A gap here is only discovered after a run
is finished and the cost question can no longer be answered."""
import json, os, sys, tempfile, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.llm import token_usage_logger as tul
from cfd_langgraph.llm.token_stats import TokenStatsCallbackHandler

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        fails.append(label)

def read(d):
    return json.loads((Path(d) / "llm_token_usage.json").read_text())

def rows(d):
    p = Path(d) / "llm_token_usage_calls.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []

print("== every call carries a stage and an agent")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    os.environ.pop("CFD_TOKEN_LOG_PATH", None)
    os.environ["CFD_SCIENTIST_STAGE"] = "manager"
    tul.append_usage_call(provider="p", model="m", input_tokens=10, output_tokens=5,
                          token_source="t", agent="case-runner")
    tul.append_usage_call(provider="p", model="m", input_tokens=20, output_tokens=7,
                          token_source="t", agent="oed-candidate-runner")
    tul.append_usage_call(provider="p", model="m", input_tokens=1, output_tokens=1,
                          token_source="t")  # no agent -> default
    r = rows(d)
    check("all rows have a stage", all(x.get("stage") for x in r), True)
    check("all rows have an agent", all(x.get("agent") for x in r), True)
    check("no row is unattributed", [x for x in r if x.get("stage") == "unattributed"], [])
    check("default agent is 'manager'", r[-1]["agent"], "manager")

    t = read(d)["totals"]
    check("totals.calls matches rows", t["calls"], len(r))
    check("totals.total_tokens exact", t["total_tokens"], 10+5+20+7+1+1)
    ag = t["by_stage"]["manager"]["by_agent"]
    check("by_agent has all three roles", sorted(ag), ["case-runner", "manager", "oed-candidate-runner"])
    check("case-runner tokens", ag["case-runner"]["total_tokens"], 15)
    check("oed-candidate-runner tokens", ag["oed-candidate-runner"]["total_tokens"], 27)
    check("by_agent sums to stage total",
          sum(a["total_tokens"] for a in ag.values()),
          t["by_stage"]["manager"]["total_tokens"])

print("== stage_scope attributes correctly and restores")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    os.environ["CFD_SCIENTIST_STAGE"] = "manager"
    with tul.stage_scope("cli_prelude"):
        tul.append_usage_call(provider="p", model="m", input_tokens=3, output_tokens=2, token_source="t")
    tul.append_usage_call(provider="p", model="m", input_tokens=4, output_tokens=1, token_source="t")
    r = rows(d)
    check("scoped call attributed to the scope", r[0]["stage"], "cli_prelude")
    check("stage restored after the scope", r[1]["stage"], "manager")

print("== the agent label rides on the handler, not on global state")
h_case = TokenStatsCallbackHandler("case-runner")
h_oed = TokenStatsCallbackHandler("oed-candidate-runner")
check("handlers keep distinct labels", (h_case.agent, h_oed.agent), ("case-runner", "oed-candidate-runner"))

print("== concurrent agents do not cross-attribute")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    os.environ["CFD_SCIENTIST_STAGE"] = "manager"
    def worker(agent, n):
        for _ in range(n):
            tul.append_usage_call(provider="p", model="m", input_tokens=1, output_tokens=0,
                                  token_source="t", agent=agent)
    ts = [threading.Thread(target=worker, args=("case-runner", 60)),
          threading.Thread(target=worker, args=("oed-candidate-runner", 60))]
    [t.start() for t in ts]; [t.join() for t in ts]
    ag = read(d)["totals"]["by_stage"]["manager"]["by_agent"]
    check("case-runner count under concurrency", ag["case-runner"]["calls"], 60)
    check("oed-candidate-runner count under concurrency", ag["oed-candidate-runner"]["calls"], 60)
    check("no rows lost under concurrency", len(rows(d)), 120)

print("== a study directory never falls back to the working directory")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    os.environ.pop("CFD_TOKEN_LOG_PATH", None)
    check("resolves inside the study dir", tul._resolve_path().parent, Path(d).resolve())
    os.environ.pop("CFD_TOKEN_LOG_DIR")
    check("with nothing set it falls back to cwd (the case we must avoid)",
          tul._resolve_path().parent, Path.cwd().resolve())

print("== the agent label reaches the handler through the real factory, per provider")
# Built through create_langchain_llm, not the logger directly: the Codex and Claude Code
# providers construct their model in a helper, and a label that stopped at the factory
# silently filed every Codex call under the default agent. Constructing a model makes no
# API call.
from cfd_langgraph.llm.factory import create_langchain_llm
def _label(model):
    cbs = getattr(model, "callbacks", None) or []
    return [getattr(c, "agent", None) for c in cbs]
for provider, model_id in [("openai-codex", "gpt-5.6-sol"), ("gemini", "gemini-3.8-flash")]:
    os.environ["CFD_SCIENTIST_LLM_PROVIDER"] = provider
    os.environ.pop("CFD_SCIENTIST_EFFORT", None)
    try:
        m = create_langchain_llm(model_id, temperature=0.0, agent="case-runner")
    except Exception as exc:  # credentials unavailable on this machine
        print(f"  SKIP  {provider}: {type(exc).__name__}")
        continue
    check(f"{provider}: handler carries the agent label", "case-runner" in _label(m), True)
os.environ.pop("CFD_SCIENTIST_LLM_PROVIDER", None)

print("== a provider estimate is labelled as one, and cached/reasoning are recorded")
with tempfile.TemporaryDirectory() as d:
    os.environ["CFD_TOKEN_LOG_DIR"] = d
    os.environ["CFD_SCIENTIST_STAGE"] = "manager"
    tul.append_usage_call(provider="p", model="m", input_tokens=100, output_tokens=10,
                          cached_input_tokens=80, reasoning_output_tokens=4,
                          token_source="provider_usage", agent="manager")
    tul.append_usage_call(provider="p", model="m", input_tokens=7, output_tokens=3,
                          token_source="estimate", agent="manager")
    t = read(d)["totals"]
    check("cached input recorded", t.get("cached_input_tokens"), 80)
    check("reasoning output recorded", t.get("reasoning_output_tokens"), 4)
    check("calls split by token source", t.get("calls_by_token_source"), {"provider_usage": 1, "estimate": 1})

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)

"""Build-agent repeat handling: a command asked for again with nothing changed
is answered from its last run, a re-read of an unchanged file is flagged, and a
session that keeps repeating itself ends early saying what it repeated.

Run: python scripts/test_build_agent_repeats.py   (needs bubblewrap)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import code_mod_agentic as cma  # noqa: E402
from code_mod_agentic import Sandbox  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    run_dir, starter = root / "run", root / "starter"
    run_dir.mkdir()
    starter.mkdir()
    (run_dir / "a.txt").write_text("alpha\n")
    (run_dir / "b.txt").write_text("beta\n")
    sb = Sandbox(run_dir=run_dir, starter_case=starter, wm_project_dir=None)

    print("== 1. a command asked for again with nothing changed")
    first = sb.run_bash("cat a.txt", cwd=str(run_dir))
    check("first run executes", first.get("rc") == 0 and "alpha" in first.get("stdout", "")
          and not sb.last_call_repeated, first)
    sb.run_bash("cat b.txt", cwd=str(run_dir))
    again = sb.run_bash("cat a.txt", cwd=str(run_dir))
    check("alternating with another read does not hide the repeat",
          "not_rerun" in again and sb.last_call_repeated and "alpha" in again.get("stdout", ""), again)
    sb.run_bash("printf gamma > a.txt", cwd=str(run_dir))
    check("a command that writes a file counts as a change", sb.last_call_changed)
    after = sb.run_bash("cat a.txt", cwd=str(run_dir))
    check("after a change the command runs again",
          "not_rerun" not in after and "gamma" in after.get("stdout", "") and not sb.last_call_repeated, after)

    print("== 2. re-reading an unchanged file")
    r1 = sb.read_file(str(run_dir / "b.txt"))
    r2 = sb.read_file(str(run_dir / "b.txt"))
    check("a second read of the same unchanged part is flagged", "unchanged" in r2 and sb.last_call_repeated
          and "unchanged" not in r1, r2)
    sb.write_file(str(run_dir / "b.txt"), "delta\n")
    check("write_file counts as a change", sb.last_call_changed and not sb.last_call_repeated)
    r3 = sb.read_file(str(run_dir / "b.txt"))
    check("after the file changed the read is not flagged", "unchanged" not in r3 and "delta" in r3["content"], r3)

    print("== 3. a session that keeps repeating itself ends early")

    class _Fake:
        """Alternates two read-only commands forever, like slau_r7's build agent."""

        def __init__(self) -> None:
            self.n = 0

        def bind_tools(self, *a, **k):
            return self

        def bind(self, *a, **k):
            return self

        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            self.n += 1
            cmd = "cat a.txt" if self.n % 2 else "cat b.txt"
            return AIMessage(content=json.dumps({"tool": "run_bash", "args": {"cmd": cmd}}))

    import cfd_langgraph.llm.factory as factory
    fake = _Fake()
    factory.create_langchain_llm = lambda *a, **k: fake
    os.environ["CFD_SCIENTIST_AGENT_TOOL_PROTOCOL"] = "text"
    loop_dir = root / "loop"
    loop_dir.mkdir()
    (loop_dir / "a.txt").write_text("alpha\n")
    (loop_dir / "b.txt").write_text("beta\n")
    res = cma.run_agent_loop(
        repo_root=ROOT, hypothesis="h", variant_name="v", run_dir=loop_dir, starter_case=starter,
        topic="t", model="fake", max_turns=60, timeout_s=600,
        system_message="sys", initial_prompt="go",
    )
    reason = str(res.get("aborted_reason", ""))
    check("the session stopped early as stuck", reason.startswith("stuck:") and fake.n < 30,
          f"calls={fake.n} reason={reason[:200]}")
    check("the reason names what it repeated", "cat a.txt" in reason and "cat b.txt" in reason, reason)

    print("== 4. a continuation is told what the last session did")
    from cfd_langgraph.manager.tools import _impl_last_actions
    log = root / "agentic_trajectory.log"
    log.write_text("\n".join([
        "# agentic loop start variant=implementation", "--- turn 1 ---",
        'AI tool_call: {"tool": "run_bash", "args": {"cmd": "cat old_session.txt"}}',
        'TOOL [run_bash] -> {"ok": true, "rc": 0}',
        "# agentic loop start variant=implementation", "--- turn 1 ---",
        'AI tool_call: {"tool": "run_bash", "args": {"cmd": "wmake solver"}}',
        'TOOL [run_bash] -> {"ok": true, "rc": 2, "error_summary": "ld returned 1 exit status"}',
        "--- turn 2 ---",
        'AI tool_call: {"tool": "write_file", "args": {"path": "/x/solver.C", "content": "...',
        'TOOL [write_file] -> {"ok": true}',
    ]) + "\n")
    acts = _impl_last_actions(log)
    check("only the latest session is summarised", len(acts) == 2 and "old_session" not in " ".join(acts), acts)
    check("each action carries its outcome", "rc=2" in acts[0] and "ld returned 1" in acts[0], acts)
    check("a call cut short in the log is still listed", "write_file" in acts[1] and "-> ok" in acts[1], acts)
    import implementation_agentic as ia
    brief = ia.build_prompt(topic="t", run_dir=root, starter_root=starter, timeout_s=0, guidance="Fix X first.")
    check("the manager's guidance is marked as unrun advice",
          "has not run anything" in brief and "trust what you observe" in brief and "Fix X first." in brief)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

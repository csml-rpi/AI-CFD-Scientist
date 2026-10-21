#!/usr/bin/env python3
"""No single tool result can grow the manager's conversation past what the
provider accepts.

On shelf_demo_dafoam_20261015_1318 (2026-10-18) the manager grepped *.json for
a pattern that matched a JSON file holding a 157k-cell field on one line. The
match came back whole, 29.7 MB; the provider takes at most 10 MB per item, so
every later request was refused and the study stopped after five
continuations. Checked here: grep_files cuts long lines, the middleware caps
any tool result, and a result already in a saved conversation is cut when the
request is built.

Run: python scripts/test_tool_result_caps.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import ToolMessage  # noqa: E402

import cfd_langgraph.manager.tools as tools  # noqa: E402
from cfd_langgraph.config import Settings  # noqa: E402
from cfd_langgraph.llm import caching, tool_bridge  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


print("== 1. grep_files cuts a match that is a whole one-line file")
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    (out / "beta.json").write_text(json.dumps({"beta": [1.0] * 400000}))
    (out / "small.json").write_text('{"beta": 1}\n')
    built = tools.build_manager_tools(Settings(), out)
    grep = [f for group in built.values() if isinstance(group, list) for f in group
            if getattr(f, "__name__", "") == "grep_files"][0]
    res = grep(pattern='"beta"', root=str(out), glob="*.json")
    lines = res.get("matches") or []
    check("both files match", len(lines), 2)
    longest = max((len(x) for x in lines), default=0)
    check("no returned line is longer than the cap plus its note", longest < tools._GREP_LINE_CHARS + 200)
    check("a cut line says how long it was", any("characters; cut]" in x for x in lines))
    check("a short line is returned whole", any(x.endswith('{"beta": 1}') for x in lines))

print("== 2. the middleware caps any tool result, not only errors")
trim = caching._tool_error_trim_middleware()
big = ToolMessage(content="x" * (caching._TOOL_RESULT_CHARS * 3), tool_call_id="c1")
small = ToolMessage(content="ok", tool_call_id="c2")
cut = trim.wrap_tool_call(None, lambda _r: big)
check("a huge result is cut to about the cap", len(cut.content) < caching._TOOL_RESULT_CHARS + 200)
check("and says how much was left out", "characters omitted" in cut.content)
check("a small result is untouched", trim.wrap_tool_call(None, lambda _r: small).content, "ok")
err = ToolMessage(content="e" * 10000, tool_call_id="c3", status="error")
check("an error keeps its tighter cap", len(trim.wrap_tool_call(None, lambda _r: err).content)
      < caching._TOOL_ERROR_CHARS + 200)

print("== 3. a huge result already in a saved conversation is cut when the request is built")
items = tool_bridge.messages_to_responses_input([ToolMessage(content="y" * 29_679_338, tool_call_id="c4")])
output = items[0]["output"]
check("the request item is under the provider's 10 MB limit", len(output) < 10_485_760)
check("and near the cap", len(output) < tool_bridge._TOOL_RESULT_CHARS + 200)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

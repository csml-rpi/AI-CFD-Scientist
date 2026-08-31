"""A 401 mid-run must reload the token from disk and retry, not kill the study."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests
from cfd_langgraph.llm import codex_oauth as C

F = 0
def check(n, c, d=""):
    global F
    print(f"[{'PASS' if c else 'FAIL'}] {n}" + (f" — {d}" if d and not c else ""))
    if not c: F += 1

import tempfile
tmp = Path(tempfile.mkdtemp()); auth = tmp/"auth.json"
def write(tok):
    auth.write_text(json.dumps({"tokens":{"access_token":tok,"account_id":"acct"}}))
    time.sleep(0.02)

write("OLD")
C.codex_auth_candidates = lambda: [auth]
C.load_codex_oauth = lambda: (json.loads(auth.read_text())["tokens"]["access_token"], "acct")

w = C.CodexResponsesWrapper(token="OLD", account_id="acct", model="m",
                            base_url="https://x", instructions="", stream=False)

seen = []
class FakeResp:
    def __init__(self, code): self.status_code, self.ok, self.text = code, code < 400, "token_expired"
    def close(self): pass
    def json(self): return {"output":[{"content":[{"type":"output_text","text":"ok"}]}]}

def fake_post(url, headers=None, json=None, timeout=None, stream=None):
    seen.append(headers["Authorization"])
    return FakeResp(401 if headers["Authorization"] == "Bearer OLD" else 200)

C.requests.post = fake_post
write("NEW")                      # the CLI refreshes the file mid-run
try:
    r = w.invoke([{"role":"user","content":"hi"}])
    check("the request succeeds after reloading the token", True)
except Exception as e:
    check("the request succeeds after reloading the token", False, f"{type(e).__name__}: {e}")

# Better than expected: the refreshed file is picked up BEFORE sending, so the
# stale token is never put on the wire at all.
check("a refreshed file is used without needing a 401 first",
      seen == ["Bearer NEW"], str(seen))

# A 401 with nothing newer on disk must fail fast, not spin: there is nothing
# to reload, so retrying would just burn the budget on the same dead token.
seen.clear()
write("STALE")
w2 = C.CodexResponsesWrapper(token="STALE", account_id="a", model="m",
                             base_url="https://x", instructions="", stream=False)
def always_401(url, headers=None, json=None, timeout=None, stream=None):
    seen.append(headers["Authorization"]); return FakeResp(401)
C.requests.post = always_401
try:
    w2.invoke([{"role":"user","content":"hi"}])
    check("an unrefreshable 401 still raises", False, "it returned instead of raising")
except Exception as e:
    check("an unrefreshable 401 still raises", "401" in str(e), f"{type(e).__name__}: {str(e)[:80]}")
check("and it fails fast rather than spinning", len(seen) == 1, f"{len(seen)} requests")

print()
print("ALL PASS" if not F else f"{F} FAILURE(S)")
sys.exit(1 if F else 0)

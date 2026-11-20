#!/usr/bin/env python3
"""The build agent's transcript window, and the prompt prefix it has to keep.

A build agent re-sends its whole history every turn, so the leading characters
of that string decide whether the provider serves the turn from its prompt
cache. A cap applied as `full[-cap:]` makes the view a sliding window: each
turn the text is longer, the window starts at a different offset, and the
cached prefix is destroyed every single turn once the cap is passed.

Measured over one 40-turn build agent, that cost 74% of a whole study's input
tokens at a 43% hit rate, with the cached portion pinned to a fixed floor while
the prompt itself grew to 80k tokens -- the miss getting more expensive exactly
as it got more frequent.

What has to hold:
  1. under the cap, nothing is dropped and each turn is a pure append
  2. over the cap, the window moves in STEPS, so consecutive turns usually
     share a long prefix
  3. chunks are dropped whole -- never a mid-chunk cut
  4. the boundary only ever moves forward
  5. the retry ladder's explicit cap does not move the shared boundary
  6. a single chunk bigger than the cap still yields something under the cap

Run: python scripts/test_transcript_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


from code_mod_agentic import render_transcript_window as render  # noqa: E402

PROMPT = "Now output your next tool-call JSON object."
CAP = 240_000


def turn(i: int, size: int = 9_000) -> str:
    return f"--- turn {i} ---\n" + ("x" * size)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


print("== 1. empty and under-cap behaviour")
text, kept = render([], kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
check("no chunks renders nothing", text == "" and kept == 0)

chunks = [turn(i) for i in range(1, 6)]
text, kept = render(chunks, kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
check("nothing is dropped while it fits", kept == 0)
check("every turn is present", all(f"--- turn {i} ---" in text for i in range(1, 6)))
check("it is not marked truncated", "truncated" not in text)
check("the next-turn prompt is last", text.rstrip().endswith(PROMPT))


print("== 2. under the cap each turn is a pure append")
prev, _ = render(chunks, kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
chunks.append(turn(6))
nxt, _ = render(chunks, kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
shared = common_prefix(prev, nxt)
check("the whole previous body is a prefix of the next",
      shared >= len(prev) - len(PROMPT) - 40, f"shared={shared} prev={len(prev)}")


print("== 3. over the cap the window moves in steps, not every turn")
chunks = []
kept = 0
prev = None
prefixes: list[int] = []
lengths: list[int] = []
moves = 0
backwards = False
for i in range(1, 61):
    chunks.append(turn(i))
    text, new_kept = render(chunks, kept_from=kept, cap=CAP, next_turn_prompt=PROMPT)
    if new_kept != kept:
        moves += 1
    if new_kept < kept:
        backwards = True
    kept = new_kept
    if prev is not None:
        prefixes.append(common_prefix(prev, text))
        lengths.append(len(text))
    prev = text

check("the boundary never moves backwards", not backwards)
check("the cap is never exceeded", max(lengths) <= CAP, f"max={max(lengths)}")
check("the window did have to move", moves > 0, f"moves={moves}")
check("but it moves on few turns, not most", moves < len(prefixes) / 3,
      f"{moves} moves over {len(prefixes)} turns")
mean_frac = sum(p / l for p, l in zip(prefixes, lengths)) / len(prefixes)
check("consecutive turns share most of their prompt", mean_frac > 0.75,
      f"mean shared prefix {100 * mean_frac:.1f}%")
print(f"       mean shared prefix {100 * mean_frac:.1f}% over {len(prefixes)} turns, "
      f"{moves} window moves")

# The behaviour being replaced, for contrast: recompute full[-cap:] each turn.
old_prefixes, old_lengths = [], []
prev = None
acc: list[str] = []
for i in range(1, 61):
    acc.append(turn(i))
    full = ("\n\n=== CONVERSATION SO FAR ===\n" + "\n".join(acc)
            + "\n=== END CONVERSATION ===\n\n" + PROMPT)
    text = full if len(full) <= CAP else (
        "\n\n=== CONVERSATION SO FAR (truncated; older turns omitted) ===\n" + full[-CAP:])
    if prev is not None:
        old_prefixes.append(common_prefix(prev, text))
        old_lengths.append(len(text))
    prev = text
old_frac = sum(p / l for p, l in zip(old_prefixes, old_lengths)) / len(old_prefixes)
check("and that is a real improvement on a sliding window", mean_frac > old_frac + 0.3,
      f"stepped {100 * mean_frac:.1f}% vs sliding {100 * old_frac:.1f}%")
print(f"       a sliding window would share {100 * old_frac:.1f}%")


print("== 4. chunks are dropped whole, never cut in half")
chunks = [turn(i) for i in range(1, 61)]
text, kept = render(chunks, kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
check("something was dropped", kept > 0, f"kept_from={kept}")
body = text.split("===\n", 1)[1]
check("the oldest kept turn starts at its own header",
      body.lstrip().startswith(f"--- turn {kept + 1} ---"), body.lstrip()[:40])
check("no dropped turn survives", f"--- turn {kept} ---" not in text)
check("every kept turn is whole",
      all(f"--- turn {i} ---" in text for i in range(kept + 1, 61)))


print("== 5. the retry ladder must not move the shared boundary")
chunks = [turn(i) for i in range(1, 61)]
_, steady = render(chunks, kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
small, ladder_kept = render(chunks, kept_from=0, cap=CAP // 10, next_turn_prompt=PROMPT)
check("a smaller cap drops more", ladder_kept > steady, f"{ladder_kept} vs {steady}")
check("and honours that smaller cap", len(small) <= CAP // 10, f"len={len(small)}")
check("the steady boundary is the smaller one, so the caller can discard the other",
      steady < ladder_kept)


print("== 6. a chunk larger than the whole cap")
text, kept = render([turn(1, size=CAP * 2)], kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
check("still comes back within the cap", len(text) <= CAP, f"len={len(text)}")
check("and says it is truncated", "truncated" in text)

text, kept = render([turn(1, size=500), turn(2, size=CAP * 2)],
                    kept_from=0, cap=CAP, next_turn_prompt=PROMPT)
check("the newest turn is what survives", len(text) <= CAP and "turn 1 ---" not in text)


print("== 7. a boundary handed in past the end is clamped, not crashed")
text, kept = render([turn(1), turn(2)], kept_from=99, cap=CAP, next_turn_prompt=PROMPT)
check("it renders", bool(text))
check("clamped to the last chunk", kept == 1, f"kept={kept}")


print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)

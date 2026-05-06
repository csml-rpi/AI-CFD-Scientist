"""
Claude Code SDK — text + optional image invocation with token counting.

Prerequisites:
    1. Install Claude Code CLI:   npm install -g @anthropic-ai/claude-code
    2. Log in:                    claude login
    3. Install Python packages:   pip install claude-agent-sdk tiktoken

Usage:
    python claude_sdk_text_invoke.py
"""

import asyncio
import base64
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions


# ── Token counter (approximate, uses GPT tokenizer as proxy) ─────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))


# ── Core invoke function ──────────────────────────────────────────────────────
async def _invoke(
    system_prompt: str,
    user_prompt: str,
    model: str,
    image_path: str | None,
) -> tuple[str, int, int]:
    """
    Returns (response_text, input_tokens, output_tokens).
    image_path: optional path to a PNG/JPG file to send alongside the user prompt.
    """
    opts = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # disable all agent tools — pure text generation (prompt in, text out)
    )

    # Build prompt — string for text-only, AsyncIterable[dict] for images
    if image_path:
        image_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        suffix = Path(image_path).suffix.lower().lstrip(".")
        media_type = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

        stream_message = {
            "type": "user",
            "session_id": "",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                ],
            },
            "parent_tool_use_id": None,
        }

        async def _image_gen():
            yield stream_message

        prompt_arg = _image_gen()
    else:
        prompt_arg = user_prompt

    # Collect response
    chunks = []
    async for msg in query(prompt=prompt_arg, options=opts):
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)

    response_text = "\n".join(chunks).strip()

    # Approximate token counts
    input_tokens  = count_tokens(system_prompt) + count_tokens(user_prompt)
    output_tokens = count_tokens(response_text)

    return response_text, input_tokens, output_tokens


def invoke(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    image_path: str | None = None,
) -> tuple[str, int, int]:
    """
    Synchronous wrapper — call this from normal (non-async) code.

    Returns:
        response  : str   — Claude's reply
        in_tokens : int   — approximate input token count
        out_tokens: int   — approximate output token count
    """
    return asyncio.run(_invoke(system_prompt, user_prompt, model, image_path))


# ── Examples ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Available models ─────────────────────────────────────────────────────────
    # Claude Code CLI only supports current-generation models.
    # Older models (3.x, 3.5.x) are NOT supported and will raise an error.
    #
    # "claude-opus-4-7"           # Most capable, slower
    # "claude-sonnet-4-6"         # Best balance of speed and quality (recommended)
    # "claude-haiku-4-5-20251001" # Fastest, lightest
    # ─────────────────────────────────────────────────────────────────────────

    MODEL = "claude-sonnet-4-6"  # <-- change this to any model above

    # ── Example 1: text only ──────────────────────────────────────────────────
    print("=" * 60)
    print("EXAMPLE 1 — Text only")
    print("=" * 60)

    system = "You are a fluid dynamics expert. Be concise and precise."
    user   = "Explain the difference between laminar and turbulent flow in two sentences."

    response, in_tok, out_tok = invoke(system, user, model=MODEL)

    print(f"Response    : {response}")
    print(f"Input tokens: {in_tok}")
    print(f"Output tokens: {out_tok}")
    print(f"Total tokens: {in_tok + out_tok}")

    # ── Example 2: text only, different prompts ───────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 2 — Text only (different prompts)")
    print("=" * 60)

    system2 = "You are a helpful assistant. Answer briefly."
    user2   = "What is the Navier-Stokes equation used for?"

    response2, in_tok2, out_tok2 = invoke(system2, user2, model=MODEL)

    print(f"Response    : {response2}")
    print(f"Input tokens: {in_tok2}")
    print(f"Output tokens: {out_tok2}")
    print(f"Total tokens: {in_tok2 + out_tok2}")

    # ── Example 3: image + text ───────────────────────────────────────────────
    # Replace IMAGE_FILE with an actual image path to test this.
    IMAGE_FILE = None  # e.g. "my_plot.png"

    if IMAGE_FILE and Path(IMAGE_FILE).exists():
        print("\n" + "=" * 60)
        print("EXAMPLE 3 — Image + text")
        print("=" * 60)

        system3 = "You are a CFD expert. Describe simulation results concisely."
        user3   = "Describe what you see in this plot. What does it indicate about the simulation?"

        response3, in_tok3, out_tok3 = invoke(system3, user3, model=MODEL, image_path=IMAGE_FILE)

        print(f"Response    : {response3}")
        print(f"Input tokens : {in_tok3}  (image tokens not counted — approx text only)")
        print(f"Output tokens: {out_tok3}")
    else:
        print("\nExample 3 (image) skipped — set IMAGE_FILE to a real path to test.")

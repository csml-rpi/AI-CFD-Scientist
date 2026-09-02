from __future__ import annotations

import importlib.util
import asyncio
import os
import re
import shutil
import time
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from cfd_langgraph.llm.token_stats import TOKEN_STATS_HANDLER


def _extract_gemini_text(content: Any) -> str:
    """Normalize Gemini content to a plain string.

    Gemini 3+ models return content as a list of typed blocks
    e.g. [{'type': 'text', 'text': '...', 'extras': {...}}].
    Older models return a plain string. Handle both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def _is_pure_text_block_list(content: Any) -> bool:
    """True only if ``content`` is a list where every block is a plain text
    block — the one shape ``_extract_gemini_text`` is safe to flatten.

    Gemini 3's "thinking" models attach non-text blocks (thought-signature /
    reasoning metadata) to responses in multi-turn, tool-calling
    conversations, and require that exact block to be echoed back unchanged
    on the next turn or the API rejects the request with "Invalid thought
    signature". Flattening those away (as this class used to, unconditionally)
    silently breaks multi-turn tool use on Gemini 3 thinking models — it just
    never showed up before because nothing here previously exercised a
    multi-turn, tool-calling conversation against one. Only collapse to a
    string when there is nothing but text to lose.
    """
    if not isinstance(content, list):
        return False
    return bool(content) and all(isinstance(b, dict) and b.get("type") == "text" for b in content)


class GeminiChatModel(BaseChatModel):
    """Thin wrapper over ChatGoogleGenerativeAI that normalizes content to str.

    Gemini 3+ may return content as a list of typed blocks; all downstream
    framework code expects resp.content to be a plain string.
    """

    model_name: str
    temperature: float = 0.1
    _inner: Any = PrivateAttr()

    def __init__(self, *, inner: Any, model_name: str, temperature: float = 0.1,
                 callbacks: Any = None, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, temperature=temperature,
                         callbacks=callbacks, **kwargs)
        self._inner = inner

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        result = self._inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # Normalize content to str ONLY when it's safe to (pure text blocks,
        # nothing else) — see _is_pure_text_block_list for why: Gemini 3
        # thinking models attach a thought-signature block that must survive
        # unchanged into the next turn's history, and collapsing it away
        # breaks multi-turn tool-calling conversations with "Invalid thought
        # signature". Every other field on the message is carried through
        # untouched regardless — tool_calls above all, since a bare
        # AIMessage(content=...) rebuild used to silently drop those too,
        # breaking tool-calling agents without an error message.
        normalized: list[ChatGeneration] = []
        for gen in result.generations:
            original_content = gen.message.content
            content: Any = (
                _extract_gemini_text(original_content)
                if _is_pure_text_block_list(original_content)
                else original_content
            )
            msg = AIMessage(
                content=content,
                tool_calls=getattr(gen.message, "tool_calls", None) or [],
                invalid_tool_calls=getattr(gen.message, "invalid_tool_calls", None) or [],
                usage_metadata=getattr(gen.message, "usage_metadata", None),
                response_metadata=getattr(gen.message, "response_metadata", None) or {},
                # langchain_google_genai stashes a tool_call_id -> thought-signature
                # map here (its own `_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY`,
                # "__gemini_function_call_thought_signatures__") — dropping this
                # was the real remaining cause of "Invalid thought signature" on
                # multi-turn tool-calling conversations even after content/tool_calls
                # were fixed: the library's own dummy-signature fallback only
                # covers a *missing* signature, not one that existed and got
                # discarded by a wrapper rebuilding the message from scratch.
                additional_kwargs=getattr(gen.message, "additional_kwargs", None) or {},
                id=getattr(gen.message, "id", None),
            )
            normalized.append(ChatGeneration(message=msg, generation_info=gen.generation_info))
        return ChatResult(generations=normalized, llm_output=result.llm_output)

    def get_num_tokens(self, text: str) -> int:
        try:
            return self._inner.get_num_tokens(text)
        except Exception:
            return max(1, len((text or "").split()))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self._inner.with_structured_output(schema, **kwargs)

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """Delegate tool-schema formatting to the real Gemini client, but
        bind the result onto *this* wrapper (not the inner client) so every
        subsequent call still goes through our `_generate` override above —
        otherwise tool calls would bypass the content-normalization fix
        entirely. The base `BaseChatModel.bind_tools` raises
        NotImplementedError by default; every LangChain/deepagents tool-
        calling agent calls this before every model turn, so leaving it
        unimplemented breaks the agent loop immediately, not just when a
        tool happens to be used.
        """
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        bound_inner = self._inner.bind_tools(tools, **kwargs)
        bound_kwargs = getattr(bound_inner, "kwargs", {}) or {}
        return self.bind(**bound_kwargs)


def _project_root() -> Path:
    # factory.py -> llm -> cfd_langgraph -> src -> repo root
    return Path(__file__).resolve().parents[3]





def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def _lc_messages_to_dicts(messages: Any) -> List[dict]:
    """Convert whatever LangChain hands `.invoke()` into role/content dicts.

    A bare string is the single most common way `.invoke()` is called —
    ``llm.with_structured_output(Schema).invoke("some prompt")`` — and six call
    sites in this repo do exactly that, including OED candidate proposal and
    every stage of native case writing. A ``str`` is also a ``Sequence``, so
    iterating it yielded one message per *character*: the model received
    hundreds of one-letter turns and answered, coherently, that it had been
    given nothing. Nothing raised; the prompt simply never arrived.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, BaseMessage):
        messages = [messages]
    elif hasattr(messages, "to_messages"):
        # ChatPromptValue / StringPromptValue, what a `prompt | llm` chain passes.
        messages = messages.to_messages()
    elif isinstance(messages, dict):
        messages = [messages]

    out: List[dict] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role") or "user"
            content = _message_content_to_text(m.get("content", ""))
            out.append({"role": role, "content": content})
            continue
        t = getattr(m, "type", None)
        if t == "system":
            role = "system"
        elif t == "human":
            role = "user"
        elif t == "ai":
            role = "assistant"
        elif t == "tool":
            role = "tool"
        else:
            role = "user"
        raw_content = getattr(m, "content", "")
        if isinstance(raw_content, list):
            content = raw_content
        else:
            content = _message_content_to_text(raw_content)
        out.append({"role": role, "content": content})
    return out


_EMPTY_TURN_ATTEMPTS = 4


def _retry_empty_turn(produce, provider: str):
    """Call ``produce`` until it yields a non-empty assistant turn.

    ``produce`` returns ``(text, tool_calls)``.

    A turn with neither text nor a tool call is never a valid answer from an
    agent: LangGraph reads "no tool calls" as the agent choosing to stop, so
    an empty completion does not surface as an error — it silently ends the
    study. Measured on run closure_20260824_codex: the first model call came
    back as a 317-byte AIMessage with no content and no tool calls, the graph
    ended, and the CLI printed "Done (or waiting on the next stage)" having
    done nothing at all. The same prompt returned 761 characters on retry, so
    the emptiness was transient, not a property of the request.

    Retried here rather than in the graph because both providers can produce
    it and the graph cannot tell an empty turn from a deliberate stop.
    """
    import time as _time

    last_text, last_calls = "", []
    for attempt in range(1, _EMPTY_TURN_ATTEMPTS + 1):
        text, tool_calls = produce()
        if (text or "").strip() or tool_calls:
            return text, tool_calls
        last_text, last_calls = text, tool_calls
        if attempt < _EMPTY_TURN_ATTEMPTS:
            delay = 2 ** attempt
            print(
                f"[{provider}] empty assistant turn (no text, no tool calls) on attempt "
                f"{attempt}/{_EMPTY_TURN_ATTEMPTS}; retrying in {delay}s",
                flush=True,
            )
            _time.sleep(delay)
    # Raising rather than returning the empty turn: the caller's error path
    # keeps the session alive and lets the user retry the step, whereas an
    # empty turn ends the run while reporting success.
    raise RuntimeError(
        f"{provider} returned an empty assistant turn "
        f"{_EMPTY_TURN_ATTEMPTS} times in a row (no text and no tool calls)."
    )


class CodexOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by the Codex OAuth cache + ChatGPT Codex Responses API."""

    model_name: str
    _codex: Any = PrivateAttr()

    def __init__(
        self,
        *,
        codex_wrapper: Any,
        model_name: str,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, callbacks=callbacks, **kwargs)
        self._codex = codex_wrapper

    @property
    def _llm_type(self) -> str:
        return "openai-codex-oauth"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """Expose tools to the ChatGPT/Codex Responses backend natively.

        The bound specs travel to ``_generate`` as a keyword argument (the same
        mechanism ChatOpenAI uses), where they become the request's ``tools``
        array. The backend answers with real ``function_call`` items, so
        nothing has to be parsed out of prose.
        """
        from .tool_bridge import openai_tool_specs, responses_tool_specs

        specs = responses_tool_specs(openai_tool_specs(tools))
        return self.bind(codex_tools=specs, codex_tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        from .tool_bridge import messages_to_responses_input

        tools = kwargs.get("codex_tools")

        def _produce() -> tuple[str, list[dict]]:
            if tools:
                response = self._codex.invoke(
                    None,
                    items=messages_to_responses_input(messages),
                    tools=tools,
                    tool_choice=kwargs.get("codex_tool_choice"),
                )
                return (
                    getattr(response, "content", "") or "",
                    list(getattr(response, "tool_calls", []) or []),
                )
            response = self._codex.invoke(_lc_messages_to_dicts(messages))
            return getattr(response, "content", "") or "", []

        text, tool_calls = _retry_empty_turn(_produce, "codex")
        msg = AIMessage(
            content=text,
            tool_calls=[
                {"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"}
                for c in tool_calls
                if c.get("name")
            ],
        )
        gen = ChatGeneration(message=msg)
        prompt_tokens = 0
        completion_tokens = 0
        try:
            for m in messages:
                prompt_tokens += int(self._codex.get_num_tokens(_message_content_to_text(m.content)))
            completion_tokens = int(self._codex.get_num_tokens(text))
        except Exception:
            prompt_tokens = completion_tokens = 0
        return ChatResult(
            generations=[gen],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "model_name": self.model_name,
            },
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Delegate to CodexResponsesWrapper (expects dict messages); bridge LangChain messages."""
        inner = self._codex.with_structured_output(schema)

        class _StructuredBridge:
            def get_num_tokens(self, text: str) -> int:
                return inner.get_num_tokens(text)

            def invoke(self, messages: Any, config: Any = None, **kw: Any) -> Any:
                dicts = _lc_messages_to_dicts(messages)
                return inner.invoke(dicts)

        return _StructuredBridge()


_NEVER_WRITE_TOOL_CALLS = (
    "\n\nTOOL USE: never write a tool call as text, XML or JSON in your reply. "
    "Writing one out executes nothing. Use the tool interface. Never state or "
    "assume the result of a tool you have not actually called."
)


def _resolve_claude_cli() -> Optional[str]:
    """Path to the Claude Code CLI the SDK should run, or None to let it choose.

    claude-agent-sdk's ``_find_cli`` returns its own bundled binary
    unconditionally and only falls back to an installed one if the bundle is
    missing. That bundle is pinned at the SDK release and goes stale, and a
    stale CLI fails in a way nothing upstream reports: it exits 1 mid-stream
    writing NOTHING to stderr (confirmed by redirecting the child's stderr to
    a real file), so the SDK raises ProcessError with the hardcoded text
    "Check stderr output for details" and no cause is recoverable.

    Measured on the same prompt, same account, back to back:
        bundled 2.1.112 (Apr 16)   7 ok / 3 fail of 10
        installed 2.1.241          10 ok / 0 fail of 10
    Every candidate in run oed_20260823_opus_low died of this.

    Preferring the installed CLI when there is one; CFD_SCIENTIST_CLAUDE_CLI
    overrides, and an absent/unusable path falls through to the SDK's own
    choice rather than failing the call.
    """
    override = (os.environ.get("CFD_SCIENTIST_CLAUDE_CLI") or "").strip()
    if override:
        return override if os.access(override, os.X_OK) else None
    found = shutil.which("claude")
    return found if found and os.access(found, os.X_OK) else None


class _ClaudeCodeAgentWrapper:
    """Synchronous wrapper over claude-agent-sdk query().

    Text-only calls use the simple string prompt path.
    Multimodal calls (messages with image_url blocks) use the AsyncIterable[dict]
    streaming path, converting OpenAI-style image_url blocks to Anthropic-native
    image source blocks so images travel through the CLI's existing auth — no
    separate ANTHROPIC_API_KEY required.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, model: str, temperature: float = 0.0, *, max_turns: int = 1, effort: str | None = None):
        self._model = (model or "").strip()
        self._temperature = temperature
        self._max_turns = max_turns
        self._effort = effort
        try:
            import tiktoken  # type: ignore
            self._enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._enc = None
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # type: ignore
        except Exception as e:
            raise ImportError(
                "claude-agent-sdk-python is required for provider='claude-code'. "
                "Install with: pip install claude-agent-sdk"
            ) from e
        self._query = query
        self._ClaudeAgentOptions = ClaudeAgentOptions

    def get_num_tokens(self, text: str) -> int:
        if self._enc is None:
            return max(1, len((text or "").split()))
        return len(self._enc.encode(text or ""))

    # The Agent SDK's subprocess transport fails intermittently, surfacing as
    # "Fatal error in message reader: Command failed with exit code 1" with the
    # CLI writing nothing to stderr. It is not deterministic in the payload:
    # measured on run oed_20260823_opus_low, the exact prompt that killed a
    # candidate succeeds on replay, and a 20KB prompt ran 12/12 clean.
    #
    # Without a retry here the blip is fatal. Every one of that run's seven
    # candidates died this way, at turns 1, 4, 4, 4, 8 and 8, and the whole
    # search stalled at 10/100 budget having produced nothing. Codex survives
    # the same conditions only because CodexResponsesWrapper retries; this path
    # had no retry at all.
    _MAX_ATTEMPTS = 4
    _TRANSPORT_FAILURES = (
        "command failed with exit code",
        "fatal error in message reader",
        "control request timeout",
        "cli process ended",
        "process exited with code",
        "broken pipe",
    )

    @classmethod
    def _is_transport_failure(cls, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(p in msg for p in cls._TRANSPORT_FAILURES)

    def _with_retry(self, run: Any, what: str) -> Any:
        last: BaseException | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return run()
            except Exception as exc:
                if not self._is_transport_failure(exc):
                    raise
                last = exc
            if attempt < self._MAX_ATTEMPTS:
                delay = 2 ** attempt
                print(
                    f"[claude-code] transport failure on {what} attempt "
                    f"{attempt}/{self._MAX_ATTEMPTS} ({str(last)[:80]}); retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"claude-code {what} failed after {self._MAX_ATTEMPTS} attempts: {last!r}"
        )

    @staticmethod
    def _extract_json_object(text: str) -> str:
        if not text:
            raise ValueError("Empty response; expected JSON")
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```\s*$", "", s).strip()
        if s.startswith("{") and s.endswith("}"):
            return s
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            raise ValueError(f"Could not find a JSON object in response: {s[:200]}")
        return m.group(0)

    @staticmethod
    def _content_to_anthropic_blocks(content: Any) -> List[dict]:
        """Convert LangChain message content to Anthropic API content blocks.

        Handles:
          - str → text block
          - list of dicts with type="text" → text block
          - list of dicts with type="image_url" (data URL or http URL) → image block
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]
        blocks: List[dict] = []
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            btype = item.get("type", "")
            if btype == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif btype == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    # data:<media_type>;base64,<data>
                    header, data = url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    })
                elif url:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            elif btype == "image":
                # Already Anthropic-format; pass through
                blocks.append(item)
        return blocks or [{"type": "text", "text": ""}]

    @staticmethod
    def _content_has_images(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "image_url"
            for b in content
        )

    def _messages_have_images(self, messages: List[dict]) -> bool:
        return any(self._content_has_images(m.get("content", "")) for m in messages)

    def _messages_to_prompt(self, messages: List[dict]) -> tuple[str, str]:
        system_parts: List[str] = []
        convo_parts: List[str] = []
        for m in messages:
            role = str(m.get("role") or "user").strip().lower()
            content = _message_content_to_text(m.get("content", ""))
            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
            else:
                convo_parts.append(f"{role.upper()}:\n{content.strip()}")
        prompt = "\n\n".join(convo_parts).strip() or "USER:\n(Empty prompt)"
        return prompt, "\n\n".join(system_parts).strip()

    def _build_stream_dicts(self, messages: List[dict]) -> List[dict]:
        """Convert messages to AsyncIterable[dict] stream-json format with full content blocks."""
        result = []
        for m in messages:
            role = str(m.get("role") or "user").strip().lower()
            if role == "system":
                continue  # system goes to options.system_prompt
            blocks = self._content_to_anthropic_blocks(m.get("content", ""))
            result.append({
                "type": "user",
                "session_id": "",
                "message": {"role": "user" if role == "user" else "assistant", "content": blocks},
                "parent_tool_use_id": None,
            })
        return result or [{"type": "user", "session_id": "", "message": {"role": "user", "content": [{"type": "text", "text": "(empty)"}]}, "parent_tool_use_id": None}]

    def _make_options(self, system_prompt: str) -> Any:
        opts: dict[str, Any] = {
            "max_turns": int(self._max_turns or 1),
            # Disable all CLI agent tools so every call is pure text generation,
            # identical to a direct API call (Bedrock/Anthropic behaviour).
            # Without this, the CLI agent tries to run bash/write files and
            # crashes when those commands fail (e.g. wmake in compile-repair loops).
            "tools": [],
        }
        if self._model:
            opts["model"] = self._model
        if system_prompt:
            opts["system_prompt"] = system_prompt
        if self._effort is not None:
            opts["effort"] = self._effort
        cli = _resolve_claude_cli()
        if cli:
            opts["cli_path"] = cli
        # ClaudeAgentOptions does not expose temperature; omit it.
        return self._ClaudeAgentOptions(**opts)

    def invoke_with_tools(self, messages: Any, tool_specs: List[dict]) -> tuple[str, List[dict]]:
        """One assistant turn that may return native tool calls.

        Tools reach Claude as an in-process SDK MCP server, which is what makes
        the returned calls real ``ToolUseBlock``s rather than JSON scraped out
        of prose. The registered handlers are never invoked: LangGraph owns
        tool execution, so this reads the assistant turn and stops there.

        The conversation is rendered to prompt text (``render_conversation``)
        because the SDK takes a prompt, not a structured message list. Nothing
        is parsed back out of that text, so rendering costs no reliability —
        the one direction that must be machine-readable, the model's own tool
        call, stays native.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            tool as sdk_tool,
        )

        from .tool_bridge import (
            MCP_SERVER_NAME,
            looks_like_a_written_tool_call,
            mcp_tool_name,
            render_conversation,
            run_coroutine_blocking,
            sdk_tool_triples,
            split_system,
            strip_mcp_prefix,
        )

        triples = sdk_tool_triples(tool_specs)

        def _stub(_name: str):
            async def handler(_args: Any) -> dict:
                # Unreachable in normal operation (the loop below breaks on the
                # assistant turn, before the SDK would execute anything). Kept
                # explicit so that if it ever *is* reached the result is an
                # obvious error rather than a plausible-looking empty answer.
                return {"content": [{"type": "text", "text": f"{_name} is executed by the caller, not here"}]}

            return handler

        sdk_tools = [
            sdk_tool(t["name"], t["description"], t["parameters"])(_stub(t["name"]))
            for t in triples
        ]
        server = create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=sdk_tools)

        system_prompt, rest = split_system(messages)
        prompt_text = render_conversation(rest)
        # Reinforced in the system prompt as well as the transcript preamble.
        # A model that starts transcribing tool calls instead of making them
        # produces a reply that reads like completed work while nothing ran,
        # so it is worth saying twice.
        system_prompt = (system_prompt + _NEVER_WRITE_TOOL_CALLS).strip()

        opts: dict[str, Any] = {
            "mcp_servers": {MCP_SERVER_NAME: server},
            "allowed_tools": [mcp_tool_name(t["name"]) for t in triples],
            # No built-in CLI tools: the agent must not go off and run bash or
            # edit files on its own — every tool call belongs to LangGraph.
            "tools": [],
        }
        if self._model:
            opts["model"] = self._model
        if system_prompt:
            opts["system_prompt"] = system_prompt
        if self._effort is not None:
            opts["effort"] = self._effort
        cli = _resolve_claude_cli()
        if cli:
            opts["cli_path"] = cli
        options = self._ClaudeAgentOptions(**opts)
        query = self._query

        async def run() -> tuple[str, List[dict]]:
            async def prompt_stream():
                yield {
                    "type": "user",
                    # Read at call time, not closure-creation time, so the
                    # corrective retry below actually sends the corrected text.
                    "message": {"role": "user", "content": prompt_text},
                    "parent_tool_use_id": None,
                    "session_id": "default",
                }

            texts: List[str] = []
            calls: List[dict] = []
            stream = query(prompt=prompt_stream(), options=options)
            try:
                async for msg in stream:
                    if isinstance(msg, ResultMessage):
                        break
                    if not isinstance(msg, AssistantMessage):
                        continue
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            calls.append(
                                {
                                    "name": strip_mcp_prefix(block.name),
                                    "args": block.input if isinstance(block.input, dict) else {},
                                    "id": block.id,
                                }
                            )
                        elif isinstance(block, TextBlock):
                            texts.append(block.text)
                    # Stop ONLY on a tool call, never on text.
                    #
                    # One logical assistant turn arrives as several
                    # AssistantMessages: extended thinking is its own message
                    # holding a ThinkingBlock, a spoken preamble ("I'll read
                    # the starter folder.") is another, and the tool_use is a
                    # third. Two earlier stop rules each cut the turn short and
                    # returned an AIMessage with no tool_calls — which
                    # LangGraph reads as "the agent is finished", so a study
                    # ended after one model turn having done nothing. Both
                    # looked like a clean completion rather than a failure.
                    #
                    # Breaking here, on the tool call itself, is also what
                    # keeps the SDK from executing the tool and continuing its
                    # own agent loop past the point where LangGraph
                    # checkpoints, interrupts and schedules work. A turn that
                    # legitimately has no tool call ends at the ResultMessage
                    # above.
                    if calls:
                        break
            finally:
                await stream.aclose()
            return "".join(texts).strip(), calls

        text, calls = self._with_retry(lambda: run_coroutine_blocking(run), "invoke_with_tools")
        if not calls and looks_like_a_written_tool_call(text):
            # One corrective retry. Returning this as-is hands LangGraph an
            # AIMessage with no tool calls, which ends the agent's turn — so a
            # transcribed tool call silently becomes "the agent decided to
            # stop", with prose that claims otherwise.
            print(
                "[claude-code] the model wrote a tool call instead of making one; retrying",
                flush=True,
            )
            prompt_text = (
                prompt_text
                + "\n\n<correction>\nYour previous reply contained tool-call syntax written "
                "as text. That executed nothing. Make the call through the tool interface "
                "instead, or say plainly that you are not calling a tool.\n</correction>"
            )
            text, calls = self._with_retry(lambda: run_coroutine_blocking(run), "invoke_with_tools")
        return text, calls

    @staticmethod
    def _collect_chunks(msg: Any) -> List[str]:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            return []
        return [
            block.text
            for block in content
            if isinstance(getattr(block, "text", None), str) and block.text
        ]

    def invoke(self, messages: List[dict]):
        _, system_prompt = self._messages_to_prompt(messages)
        has_images = self._messages_have_images(messages)

        if has_images:
            stream_dicts = self._build_stream_dicts(messages)
            options = self._make_options(system_prompt)

            async def _run_multimodal() -> str:
                async def _gen():
                    for d in stream_dicts:
                        yield d

                chunks: List[str] = []
                async for msg in self._query(prompt=_gen(), options=options):
                    chunks.extend(self._collect_chunks(msg))
                return "\n".join(chunks).strip()

            run = _run_multimodal
        else:
            prompt, _ = self._messages_to_prompt(messages)
            options = self._make_options(system_prompt)

            async def _run_text() -> str:
                chunks: List[str] = []
                async for msg in self._query(prompt=prompt, options=options):
                    chunks.extend(self._collect_chunks(msg))
                return "\n".join(chunks).strip()

            run = _run_text

        def _once():
            try:
                return asyncio.run(run())
            except RuntimeError as exc:
                if self._is_transport_failure(exc):
                    raise
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(run())
                finally:
                    loop.close()

        out = self._with_retry(_once, "invoke")
        return self._Resp(out)

    def with_structured_output(self, pydantic_obj: Any):
        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )
                patched = list(messages)
                patched.insert(0, {"role": "system", "content": schema_hint})
                resp = parent.invoke(patched)
                raw = getattr(resp, "content", "")
                json_text = parent._extract_json_object(raw)
                return pydantic_obj.model_validate_json(json_text)

        return _StructuredWrapper()


class ClaudeCodeChatModel(BaseChatModel):
    """LangChain chat model backed by claude-agent-sdk (Claude Code).

    Text calls go via query(prompt=str).
    Multimodal calls (image_url content blocks) go via query(prompt=AsyncIterable[dict])
    using Anthropic-native image source blocks — same CLI auth, no separate API key.
    """

    model_name: str
    temperature: float = 0.2
    _claude: Any = PrivateAttr()

    def __init__(
        self,
        *,
        claude_wrapper: Any,
        model_name: str,
        temperature: float = 0.2,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, callbacks=callbacks, **kwargs)
        self._claude = claude_wrapper

    @property
    def _llm_type(self) -> str:
        return "claude-code-sdk"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """Expose tools to Claude through the Agent SDK's MCP mechanism.

        ``tool_choice`` is accepted for interface compatibility and ignored:
        the SDK has no equivalent knob, and silently pretending to honour a
        forced choice would be worse than not offering one.
        """
        from .tool_bridge import openai_tool_specs

        return self.bind(claude_tools=openai_tool_specs(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_specs = kwargs.get("claude_tools")

        def _produce() -> tuple[str, list[dict]]:
            if tool_specs:
                return self._claude.invoke_with_tools(messages, tool_specs)
            response = self._claude.invoke(_lc_messages_to_dicts(messages))
            return getattr(response, "content", "") or "", []

        text, tool_calls = _retry_empty_turn(_produce, "claude-code")
        msg = AIMessage(
            content=text,
            tool_calls=[
                {"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"}
                for c in tool_calls
                if c.get("name")
            ],
        )
        gen = ChatGeneration(message=msg)
        prompt_tokens = 0
        completion_tokens = 0
        try:
            for m in messages:
                prompt_tokens += int(self._claude.get_num_tokens(_message_content_to_text(m.content)))
            completion_tokens = int(self._claude.get_num_tokens(text))
        except Exception:
            prompt_tokens = completion_tokens = 0
        return ChatResult(
            generations=[gen],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "model_name": self.model_name,
            },
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        inner = self._claude.with_structured_output(schema)

        class _StructuredBridge:
            def get_num_tokens(self, text: str) -> int:
                return inner.get_num_tokens(text)

            def invoke(self, messages: Any, config: Any = None, **kw: Any) -> Any:
                dicts = _lc_messages_to_dicts(messages)
                return inner.invoke(dicts)

        return _StructuredBridge()


def _create_codex_oauth_chat_model(
    model_name: str, temperature: float, effort: str | None = None
) -> CodexOAuthChatModel:
    from .codex_oauth import CodexResponsesWrapper, default_instructions, load_codex_oauth

    token, account_id = load_codex_oauth()
    codex = CodexResponsesWrapper(
        token=token,
        account_id=account_id,
        model=model_name,
        temperature=temperature,
        base_url="https://chatgpt.com/backend-api/codex",
        instructions=default_instructions(),
        stream=True,
        effort=effort,
    )
    return CodexOAuthChatModel(
        codex_wrapper=codex,
        model_name=model_name,
        callbacks=[TOKEN_STATS_HANDLER],
    )


def _create_claude_code_chat_model(model_name: str, temperature: float, effort: str | None = None) -> ClaudeCodeChatModel:
    claude = _ClaudeCodeAgentWrapper(
        model=model_name,
        temperature=temperature,
        max_turns=1,
        effort=effort,
    )
    return ClaudeCodeChatModel(
        claude_wrapper=claude,
        model_name=model_name,
        temperature=temperature,
        callbacks=[TOKEN_STATS_HANDLER],
    )


def smoke_test_codex_oauth(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    prompt: str = "Reply with exactly one word: pong",
) -> str:
    """One ChatGPT Codex round-trip using the same OAuth client as ``openai-codex``.

    Returns the assistant message text. Raises if the token is missing or the API errors.
    """
    from .codex_oauth import default_codex_model

    m = (model or os.environ.get("CFD_SCIENTIST_MODEL", "")).strip() or default_codex_model()
    llm = _create_codex_oauth_chat_model(m, temperature, effort=_default_effort())
    out = llm.invoke([HumanMessage(content=prompt)])
    return (getattr(out, "content", None) or str(out)).strip()


# Longer read timeout for Bedrock (vision/large payloads can take 60s+). Env: BEDROCK_READ_TIMEOUT (default 300).
def _bedrock_read_timeout() -> int:
    return int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))


# The union across providers. Each backend accepts a subset and is checked
# where it is built, because the sets genuinely differ: the ChatGPT/Codex
# Responses API answered with its own list — none, minimal, low, medium, high,
# xhigh — while the Claude Agent SDK takes low, medium, high, max.
_VALID_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_CODEX_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
_CLAUDE_EFFORTS = ("low", "medium", "high", "max")


def _default_effort() -> str | None:
    """Reasoning effort from ``CFD_SCIENTIST_EFFORT``, or None.

    Applied here rather than at the ~20 ``create_langchain_llm`` call sites,
    none of which passed ``effort`` — so the parameter existed and was
    unreachable, and there was no way to ask for high effort at all. Setting
    it centrally means every stage of a study runs at the effort the user
    asked for, not just whichever call site someone remembered to thread it
    through.
    """
    value = (os.environ.get("CFD_SCIENTIST_EFFORT") or "").strip().lower()
    if not value:
        return None
    if value not in _VALID_EFFORTS:
        # Loud, not silent: a typo'd effort that quietly ran at the default
        # would be invisible in the output and only show up on the bill.
        raise ValueError(
            f"CFD_SCIENTIST_EFFORT={value!r} is not one of {', '.join(_VALID_EFFORTS)}."
        )
    return value


def create_langchain_llm(model: str, temperature: float = 0.2, effort: str | None = None) -> Any:
    m = (model or "").strip()
    effort = effort or _default_effort()

    # Optional explicit provider override.
    # Supported providers:
    # - bedrock, openai, openai-codex, anthropic, claude-code, gemini
    # If unset/empty: infer provider from model string (existing behavior).
    provider = (
        os.environ.get("CFD_SCIENTIST_LLM_PROVIDER")
        or os.environ.get("CFD_SCIEINTIST_LLM_PROVIDER")
        or ""
    ).strip().lower()

    def _bedrock_model_id(mm: str) -> str:
        return mm.split("bedrock/", 1)[1].strip() if mm.startswith("bedrock/") else mm

    if provider:
        if provider == "bedrock":
            model_id = _bedrock_model_id(m)
            try:
                from botocore.config import Config
                from boto3 import client as boto_client

                config = Config(read_timeout=_bedrock_read_timeout(), connect_timeout=30)
                bedrock_client = boto_client("bedrock-runtime", config=config)
                return ChatBedrockConverse(
                    client=bedrock_client,
                    model=model_id,
                    temperature=temperature,
                    callbacks=[TOKEN_STATS_HANDLER],
                )
            except Exception:
                return ChatBedrockConverse(
                    model=model_id,
                    temperature=temperature,
                    callbacks=[TOKEN_STATS_HANDLER],
                )

        if provider == "openai":
            return ChatOpenAI(model=m, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

        if provider == "openai-codex":
            from .codex_oauth import default_codex_model

            if m.startswith("codex/"):
                m = m.split("codex/", 1)[1].strip() or default_codex_model()
            elif m in ("", "codex"):
                m = default_codex_model()
            if effort and effort not in _CODEX_EFFORTS:
                raise ValueError(
                    f"CFD_SCIENTIST_EFFORT={effort!r} is not supported by provider 'openai-codex' "
                    f"(accepts {', '.join(_CODEX_EFFORTS)})."
                )
            return _create_codex_oauth_chat_model(m, temperature, effort=effort)

        if provider == "claude-code":
            if effort and effort not in _CLAUDE_EFFORTS:
                raise ValueError(
                    f"CFD_SCIENTIST_EFFORT={effort!r} is not supported by provider 'claude-code' "
                    f"(accepts {', '.join(_CLAUDE_EFFORTS)}). Silently running at a different "
                    "effort than asked for would be invisible except on the bill."
                )
            return _create_claude_code_chat_model(m, temperature, effort=effort)

        if provider == "anthropic":
            return ChatAnthropic(model=m, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

        if provider in {"gemini", "google"}:
            from langchain_google_genai import ChatGoogleGenerativeAI
            inner = ChatGoogleGenerativeAI(model=m, temperature=temperature)
            return GeminiChatModel(inner=inner, model_name=m, temperature=temperature,
                                   callbacks=[TOKEN_STATS_HANDLER])

        if provider in {"vertex-openai", "glm"}:
            # Vertex serves third-party MaaS models through an OpenAI-shaped
            # route, so ChatOpenAI drives them unmodified. Auth is ADC with
            # automatic refresh -- see vertex_openai for why that matters over
            # a multi-hour study.
            from .vertex_openai import create_vertex_openai_chat_model

            if effort:
                # The endpoint accepts `reasoning_effort` and ignores it:
                # measured output tokens went 2922 / 2374 / 1163 for
                # none / low / high on one prompt -- non-monotonic, and with
                # no reasoning-token accounting in output_token_details. So
                # honouring the request is not possible, and accepting it
                # silently would show up only on the bill.
                raise ValueError(
                    f"CFD_SCIENTIST_EFFORT={effort!r} is not supported by provider "
                    f"{provider!r}. The Vertex OpenAI-compatible route accepts the field "
                    "but does not act on it. Unset CFD_SCIENTIST_EFFORT for this provider."
                )
            if "/" not in m:
                raise ValueError(
                    f"CFD_SCIENTIST_MODEL={m!r} is not a Vertex MaaS publisher model id. "
                    "These are fully qualified, e.g. 'zai-org/glm-5.2-maas'. Passing a bare "
                    "name would 404 at request time, several minutes into a study."
                )
            return create_vertex_openai_chat_model(
                m,
                temperature,
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", ""),
                callbacks=[TOKEN_STATS_HANDLER],
            )

    # Back-compat inference from model string.
    if (
        m.startswith("arn:aws:bedrock")
        or m.startswith("bedrock/")
        or m.startswith("us.anthropic.")
        or m.startswith("anthropic.")
    ):
        model_id = m.split("bedrock/")[-1] if m.startswith("bedrock/") else m
        try:
            from botocore.config import Config
            from boto3 import client as boto_client
            config = Config(read_timeout=_bedrock_read_timeout(), connect_timeout=30)
            bedrock_client = boto_client("bedrock-runtime", config=config)
            return ChatBedrockConverse(
                client=bedrock_client,
                model=model_id,
                temperature=temperature,
                callbacks=[TOKEN_STATS_HANDLER],
            )
        except Exception:
            return ChatBedrockConverse(
                model=model_id,
                temperature=temperature,
                callbacks=[TOKEN_STATS_HANDLER],
            )
    if m.startswith("claude-"):
        return ChatAnthropic(model=m, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

    # Codex/OpenAI models use API-key auth via OPENAI_API_KEY.
    # Examples:
    #   CFD_SCIENTIST_MODEL=codex/gpt-5-codex
    #   CFD_SCIENTIST_MODEL=gpt-5-codex
    if m.startswith("codex/"):
        openai_model = m.split("codex/", 1)[1].strip() or "gpt-5-codex"
        return ChatOpenAI(model=openai_model, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

    if m == "codex":
        return ChatOpenAI(model="gpt-5-codex", temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return ChatOpenAI(model=m, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])
    if "gemini" in m:
        from langchain_google_genai import ChatGoogleGenerativeAI
        inner = ChatGoogleGenerativeAI(model=m, temperature=temperature)
        return GeminiChatModel(inner=inner, model_name=m, temperature=temperature,
                               callbacks=[TOKEN_STATS_HANDLER])
    raise ValueError(f"Unsupported model: {model}")

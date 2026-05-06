from __future__ import annotations

import importlib.util
import asyncio
import os
import re
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
        # Normalize every generation's message content to str
        normalized: list[ChatGeneration] = []
        for gen in result.generations:
            text = _extract_gemini_text(gen.message.content)
            normalized.append(ChatGeneration(message=AIMessage(content=text)))
        return ChatResult(generations=normalized, llm_output=result.llm_output)

    def get_num_tokens(self, text: str) -> int:
        try:
            return self._inner.get_num_tokens(text)
        except Exception:
            return max(1, len((text or "").split()))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self._inner.with_structured_output(schema, **kwargs)


def _project_root() -> Path:
    # factory.py -> llm -> cfd_langgraph -> src -> repo root
    return Path(__file__).resolve().parents[3]


_foam_agent_utils_mod: Any = None


def _load_foam_agent_utils():
    """Load Foam-Agent `utils` without importing a top-level `utils` package.

    Note: `Foam-Agent/src/utils.py` runs `load_faiss_dbs()` at import time; the first
    Codex-OAuth use in a process will initialize those indices (same as Foam-Agent).
    """
    global _foam_agent_utils_mod
    if _foam_agent_utils_mod is not None:
        return _foam_agent_utils_mod
    foam_src = _project_root() / "Foam-Agent" / "src"
    foam_src_str = str(foam_src)
    # utils.py imports tracking_aws, config, token_usage_logger from this directory.
    if foam_src_str not in sys.path:
        sys.path.insert(0, foam_src_str)
    utils_path = foam_src / "utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(
            f"Foam-Agent utils not found at {utils_path}; cannot use Codex OAuth bridge."
        )
    spec = importlib.util.spec_from_file_location("foam_agent_utils_dynamic", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {utils_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _foam_agent_utils_mod = mod
    return mod


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


def _lc_messages_to_dicts(messages: Sequence[Union[BaseMessage, dict]]) -> List[dict]:
    """Convert LangChain messages to dicts expected by Foam-Agent `_CodexResponsesWrapper`."""
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


class CodexOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by Foam-Agent Codex OAuth + ChatGPT Codex Responses API."""

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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        dicts = _lc_messages_to_dicts(messages)
        resp = self._codex.invoke(dicts)
        text = getattr(resp, "content", "") or ""
        msg = AIMessage(content=text)
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
        """Delegate to Foam-Agent wrapper (expects dict messages); bridge LangChain messages."""
        inner = self._codex.with_structured_output(schema)

        class _StructuredBridge:
            def get_num_tokens(self, text: str) -> int:
                return inner.get_num_tokens(text)

            def invoke(self, messages: Any, config: Any = None, **kw: Any) -> Any:
                dicts = _lc_messages_to_dicts(messages)
                return inner.invoke(dicts)

        return _StructuredBridge()


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
        # ClaudeAgentOptions does not expose temperature; omit it.
        return self._ClaudeAgentOptions(**opts)

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

        try:
            out = asyncio.run(run())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                out = loop.run_until_complete(run())
            finally:
                loop.close()
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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        dicts = _lc_messages_to_dicts(messages)
        resp = self._claude.invoke(dicts)
        text = getattr(resp, "content", "") or ""
        msg = AIMessage(content=text)
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


def _create_codex_oauth_chat_model(model_name: str, temperature: float) -> CodexOAuthChatModel:
    mod = _load_foam_agent_utils()
    CodexResponsesWrapper = getattr(mod, "_CodexResponsesWrapper", None)
    LLMService = getattr(mod, "LLMService", None)
    if CodexResponsesWrapper is None or LLMService is None:
        raise ImportError("Foam-Agent utils missing _CodexResponsesWrapper or LLMService")

    token, account_id = LLMService._load_codex_oauth(LLMService.__new__(LLMService))

    instructions_path = _project_root() / "Foam-Agent" / "src" / "codex_instructions_default.txt"
    try:
        instructions = instructions_path.read_text(encoding="utf-8")
    except Exception:
        instructions = (
            "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."
        )

    codex = CodexResponsesWrapper(
        token=token,
        account_id=account_id,
        model=model_name,
        temperature=temperature,
        base_url="https://chatgpt.com/backend-api/codex",
        instructions=instructions,
        stream=True,
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
    m = (model or os.environ.get("CFD_SCIENTIST_MODEL", "gpt-5-codex")).strip() or "gpt-5-codex"
    llm = _create_codex_oauth_chat_model(m, temperature)
    out = llm.invoke([HumanMessage(content=prompt)])
    return (getattr(out, "content", None) or str(out)).strip()


# Longer read timeout for Bedrock (vision/large payloads can take 60s+). Env: BEDROCK_READ_TIMEOUT (default 300).
def _bedrock_read_timeout() -> int:
    return int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))


def create_langchain_llm(model: str, temperature: float = 0.2, effort: str | None = None) -> Any:
    m = (model or "").strip()

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
            if m.startswith("codex/"):
                m = m.split("codex/", 1)[1].strip() or "gpt-5-codex"
            elif m == "codex":
                m = "gpt-5-codex"
            return _create_codex_oauth_chat_model(m, temperature)

        if provider == "claude-code":
            return _create_claude_code_chat_model(m, temperature, effort=effort)

        if provider == "anthropic":
            return ChatAnthropic(model=m, temperature=temperature, callbacks=[TOKEN_STATS_HANDLER])

        if provider in {"gemini", "google"}:
            from langchain_google_genai import ChatGoogleGenerativeAI
            inner = ChatGoogleGenerativeAI(model=m, temperature=temperature)
            return GeminiChatModel(inner=inner, model_name=m, temperature=temperature,
                                   callbacks=[TOKEN_STATS_HANDLER])

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

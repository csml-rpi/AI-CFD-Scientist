from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def build_caching_middleware(model: Any) -> List[Any]:
    """Provider-appropriate prompt-caching middleware for ``model``.

    Detected from the model instance's own type — works whether the provider
    was set explicitly (``CFD_SCIENTIST_LLM_PROVIDER``) or inferred from the
    model string inside ``create_langchain_llm`` — rather than re-deriving
    provider from env vars a second time here.

    This harness's manager and case-runner subagent both resend a large,
    mostly-static prefix on every turn (system prompt + ~15-20 tool
    definitions) — exactly the shape prompt caching is for.

    **Scope.** These are LangChain *agent* middlewares: they apply to the
    model instance driving an agent graph, and only there — never to a bare
    ``llm.invoke(...)``. The FoamAgent stages
    (``foam_native/{parser,decomposer,writer,allrun,review}``) run on the
    separate ``foam_llm`` instance from ``manager/tools.py`` and call
    ``invoke`` directly, so they are covered instead by
    :func:`cacheable_human_message` below, which places the same kind of
    breakpoint inside the message itself. Between the two, both the agent's
    turns and the bulk of the per-case token spend are cached.

    Still uncached: the subprocess runners (``code_mod_agentic.py``,
    ``foam_run_simple.py``), which build their own models in their own
    processes and hold no conversation to reuse.

    - **Bedrock** (this repo's default, ``us.anthropic.claude-sonnet-4-6``):
      the official ``BedrockPromptCachingMiddleware`` from ``langchain-aws``.
    - **Direct Anthropic API**: the official ``AnthropicPromptCachingMiddleware``
      from ``langchain-anthropic``. Both tag the system prompt's last content
      block and the tool-definitions block with a cache breakpoint.
    - **OpenAI**: automatic server-side caching for prompts over ~1024
      tokens — no client-side action needed, nothing to add here.
    - **Gemini/Vertex**: most current models cache repeated prefixes
      implicitly server-side too; the separate *explicit* caching API is
      heavier (its own create/reference lifecycle) and not wired here.
    - **claude-code / openai-codex** (CLI-session-backed wrappers in
      ``llm/factory.py``): whatever caching those CLIs do internally happens
      in their own session handling, not reachable through a LangChain
      middleware attached to our chat-model wrapper.

    Always returns a list (empty if nothing applies), so callers can do
    ``middleware=build_caching_middleware(model) + [...]`` unconditionally.
    """
    try:
        from langchain_aws import ChatBedrockConverse

        if isinstance(model, ChatBedrockConverse):
            from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware

            return [BedrockPromptCachingMiddleware()]
    except ImportError:
        pass

    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            from langchain_anthropic.middleware.prompt_caching import AnthropicPromptCachingMiddleware

            return [AnthropicPromptCachingMiddleware()]
    except ImportError:
        pass

    logger.debug("No prompt-caching middleware available for model type %s", type(model).__name__)
    return []


# A cache breakpoint only pays for itself above the provider's minimum
# cacheable prefix (1024 tokens for Claude Sonnet on both Anthropic and
# Bedrock). ~3.5 chars/token is a deliberately conservative estimate for
# English prose mixed with OpenFOAM dictionaries, so this errs toward not
# placing a breakpoint that could never be honoured.
_MIN_CACHEABLE_PREFIX_CHARS = 4096


def message_cache_dialect(model: Any) -> Optional[str]:
    """Which in-message cache-breakpoint syntax ``model`` understands.

    Returns ``"bedrock"``, ``"anthropic"``, or ``None``. Separate from
    :func:`build_caching_middleware` because they solve different halves of
    the problem: the middleware caches an *agent's* turns, while this covers
    bare ``llm.invoke(...)`` calls, which no middleware ever sees.
    """
    try:
        from langchain_aws import ChatBedrockConverse

        if isinstance(model, ChatBedrockConverse):
            return "bedrock"
    except ImportError:
        pass
    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            return "anthropic"
    except ImportError:
        pass
    return None


def cacheable_human_message(model: Any, stable_prefix: str, variable_tail: str) -> HumanMessage:
    """One ``HumanMessage`` whose text is ``stable_prefix + variable_tail``,
    with a provider-native cache breakpoint between the two where supported.

    The point of splitting into content blocks rather than rewriting the
    prompt: the blocks are concatenated by the API, so the model sees exactly
    the same characters in exactly the same order as the single formatted
    string this replaces. Nothing about the prompt's meaning, wording or
    ordering changes — the only difference is that the provider is told where
    the reusable part ends.

    ``stable_prefix`` must be genuinely identical between calls (e.g. a case's
    tutorial reference and user requirement, which every file-write and every
    review round in that case repeat verbatim). A prefix that varies is not
    wrong, merely useless — it will simply never hit.

    Note that a breakpoint here can only be honoured if *everything* before it
    in the request is also identical, system prompt included — which is why
    the FoamAgent write prompts carry no per-file text in their system
    message (see ``foam_native/prompts.py``).
    """
    dialect = message_cache_dialect(model)
    if dialect is None or len(stable_prefix) < _MIN_CACHEABLE_PREFIX_CHARS:
        return HumanMessage(content=stable_prefix + variable_tail)

    blocks: List[Dict[str, Any]]
    if dialect == "anthropic":
        # Anthropic marks the breakpoint ON the block that ends the cached
        # prefix; Bedrock's Converse API uses a standalone cachePoint block
        # between the two.
        blocks = [
            {"type": "text", "text": stable_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": variable_tail},
        ]
    else:
        blocks = [
            {"type": "text", "text": stable_prefix},
            {"cachePoint": {"type": "default"}},
            {"type": "text", "text": variable_tail},
        ]
    return HumanMessage(content=blocks)

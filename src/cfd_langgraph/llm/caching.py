from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:  # langchain-core without the typed error

    class ContextOverflowError(Exception):  # type: ignore[no-redef]
        """Never raised; keeps the emergency-clip handler below inert."""


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


# --- Context management -------------------------------------------------
#
# A long study is not one long thought; it is thousands of small tool calls,
# and every one of them stays in the conversation forever. Measured across
# three concurrent periodic-hill runs on 2026-09-04:
#
#     gemini   4,258 tool calls   22,661 checkpoints   7.03 GB
#     glm        363 tool calls    3,645 checkpoints   1.43 GB
#     codex      336 tool calls    1,770 checkpoints   0.46 GB
#
# Storage grows quadratically because each step checkpoints the whole message
# list: gemini's checkpoints went 3 KB -> 274 KB -> 606 KB across the run. By
# the end it had stopped searching and was grepping the harness's own source
# for a bare line number, its actual task buried under thousands of stale
# grep results. It never compacted once.
#
# The reason it never compacted is worth recording, because nothing looked
# broken: deepagents DOES install SummarizationMiddleware by default, but the
# threshold is read from the model's profile, and LangChain has no profile for
# `gemini-3.8-flash`. It fell back to trigger=170,000 tokens / keep=6
# messages, and the run sat just under that ceiling for hours.
#
# Two layers, following Anthropic's context-engineering cookbook, which
# measured the same shape of problem (~96% of its baseline context was
# file-read results) and reports a 48% peak reduction from clearing alone:
#
#   1. CLEAR TOOL RESULTS first. Cheap, no inference, and lossless for a
#      re-callable tool -- the model can always read the file again. This is
#      the right primary lever here precisely because this harness's context
#      is dominated by read_text_file/grep_files/list_directory output.
#   2. SUMMARISE what remains. Runs a model call and is lossy, so it is the
#      second line, for the reasoning and results that clearing cannot touch.
#
# Both thresholds are set explicitly rather than inferred, so a provider
# LangChain has no profile for behaves the same as one it knows.


def _context_trigger_tokens() -> int:
    """Token count at which stale tool results are cleared.

    Refused at or below zero. ``ClearToolUsesEdit`` is a plain dataclass and
    validates nothing, and its test is ``if tokens <= self.trigger: return`` --
    so a trigger of 0 or less means "always", from the first turn. Measured: at
    both -1 and 0 it cleared 2 of 10 tool results on a 1,029-token history,
    destroying results the model still needed with nothing to reclaim.
    """
    tokens = int(os.environ.get("CFD_SCIENTIST_CLEAR_TOOLS_TOKENS", "100000"))
    if tokens <= 0:
        msg = f"CFD_SCIENTIST_CLEAR_TOOLS_TOKENS must be greater than 0, got {tokens}."
        raise ValueError(msg)
    return tokens


def _context_keep_tool_uses() -> int:
    """How many recent tool results survive clearing.

    Refused below zero rather than passed through. ``ClearToolUsesEdit`` slices
    ``candidates[:-keep]``, so a negative value silently inverts the knob: -1
    means "clear the OLDEST result and keep the other nineteen" -- measured,
    1 of 20 cleared. LangChain validates the two *summarization* thresholds
    (``trigger``/``keep`` are refused at or below zero by
    ``_validate_context_size``) but nothing on ``ClearToolUsesEdit``, which is a
    plain dataclass -- so both of this layer's knobs are checked here.
    """
    keep = int(os.environ.get("CFD_SCIENTIST_CLEAR_TOOLS_KEEP", "6"))
    if keep < 0:
        msg = f"CFD_SCIENTIST_CLEAR_TOOLS_KEEP must be 0 or more, got {keep}."
        raise ValueError(msg)
    return keep


def _summarize_trigger_tokens() -> int:
    """Token count at which the conversation is summarised."""
    return int(os.environ.get("CFD_SCIENTIST_SUMMARIZE_TOKENS", "150000"))


def _summarize_trigger_messages() -> int:
    """Message count at which the conversation is summarised.

    Needed because the two layers do NOT fix the same problem, and a
    token-only trigger lets one hide the other.

    ContextEditingMiddleware implements only ``wrap_model_call`` and returns
    ``handler(request.override(messages=...))`` -- it edits what this turn
    sends and never writes back to state. That is exactly right for keeping
    the model's attention on recent work, and does nothing at all for the
    checkpoint, which still stores every full tool result forever. Only
    summarization rewrites state.

    So a token-only trigger is self-defeating: clearing holds the request
    under the summarization threshold, summarization never fires, state grows
    unbounded, and the 7 GB checkpoint that started this comes back while the
    model looks fine. Counting messages measures the thing clearing cannot
    touch, and the two triggers are OR-ed so whichever is breached first wins.
    """
    return int(os.environ.get("CFD_SCIENTIST_SUMMARIZE_MESSAGES", "300"))


def _summarize_trim_tokens() -> Optional[int]:
    """How much of the evicted window the summariser is allowed to read.

    120k tokens, and only because the trim it feeds is this file's own (see
    ``_build_bounded_summarizer``). Handed to the stock trim, any numeric value
    silently produces no summary at all in a loop like this one; ``none``/``0``
    /``off`` still turns the bound off entirely.

    LangChain trims with ``trim_messages(..., strategy="last",
    start_on="human")``. That is right for a chat app, where human turns are
    frequent and one is always near the end. An autonomous study has exactly
    one human message -- the topic, at index 0 -- and then hundreds of
    assistant/tool exchanges, so scanning back from the end never finds a
    human turn to start on and the trim returns an empty list. ``_create_summary``
    then writes "Previous conversation was too long to summarize." and the
    messages are removed anyway: measured, 781 messages deleted and replaced by
    that one sentence, at every tool-result size from 200 to 8000 characters,
    for both 4,000 and 100,000 token budgets.

    With ``None`` the trim is skipped and the summariser reads the whole
    evicted window. That is only safe because the triggers bound the window --
    summarization fires at the lower of 150k tokens or 300 messages, so the
    window is at most about 150k tokens, which fits every provider this
    harness runs on. Raise the trigger and this needs rethinking.
    """
    raw = os.environ.get("CFD_SCIENTIST_SUMMARIZE_TRIM", "").strip()
    if raw:
        if raw.lower() in {"none", "0", "off"}:
            return None
        budget = int(raw)
        # Refused rather than passed through, for the same reason as the two
        # clearing knobs: nothing downstream validates it and the failure is
        # silent and total. Measured at -1: every message costs more than the
        # budget, so `_truncated_to_budget` clips the last one to a single
        # character and the summariser is handed 1 message / 26 tokens
        # ("[\n\n[... truncated for summarization ...]") -- while the
        # compaction behind it still deleted 100 of 121 messages.
        if budget <= 0:
            msg = (
                "CFD_SCIENTIST_SUMMARIZE_TRIM must be greater than 0 (or "
                f"none/0/off to disable the bound), got {budget}."
            )
            raise ValueError(msg)
        return budget
    return 120000


_SUMMARY_BUDGET = threading.local()
"""The summary window for the compaction running on THIS thread.

One middleware instance serves every concurrent subagent, so the retreat
cannot lower the budget by assigning to the instance: two interleaved retries
leave it stranded at whichever value the last one captured as its "original".
Measured -- a failing compaction overlapping a fresh one left the budget at
60,000 permanently, halving every later compaction in the process for the rest
of the run, silently. A thread-local is per-compaction by construction.

Thread-local rather than a `ContextVar` even though asyncio tasks share a
thread, because there is no suspension point between the write and the read:
`_acreate_summary` sets the value, then awaits the base class's coroutine,
which trims (reading it) before its first `await`. Measured, not assumed --
twelve `asyncio` tasks retreating concurrently through one shared middleware
instance each saw exactly [120000, 60000, 30000, 15000], with none left set
afterwards. If a future LangChain ever awaits before trimming, that stops
being true and this must become a `ContextVar`.
"""


# How many times the summariser retreats to a smaller window before it gives
# up. Three halvings take 120k -> 15k, which any provider can read.
_SUMMARY_SHRINK_ATTEMPTS = 4


class _SummaryUnavailable(RuntimeError):
    """Every summarization attempt and every retreat failed.

    Raised instead of returning a placeholder string, because the base class
    treats any returned string as a summary and deletes the history behind it.
    ``before_model`` catches this and abandons the compaction; nothing else in
    the stack calls ``_create_summary``.
    """


# A compaction must evict at least this share (1/4) of the conversation's
# tokens to be worth its own model call. See `_determine_cutoff_index`.
_MIN_EVICTION_SHARE_DIVISOR = 4


# Above this the first human turn is a pasted document, not a research topic,
# and pinning it across every compaction would cost more than it saves. Live
# topics measured on this harness run 200-800 characters; 8,000 tokens is
# roughly 26 KB of text, thirty times the largest of them.
_MAX_PINNED_OBJECTIVE_TOKENS = 8000


def _truncated_to_budget(message: Any, budget: int, count_tokens: Any) -> Any:
    """A copy of ``message`` whose content fits in ``budget`` tokens.

    Measured rather than assumed: the counter charges different rates per
    provider (3.3 chars/token for Anthropic, 4.0 elsewhere) and images a flat
    85, so a chars-per-token guess would be wrong for exactly the messages
    this is for. One proportional estimate, then halve until it actually fits.
    """
    marker = "\n\n[... truncated for summarization ...]"
    if isinstance(message.content, str):
        content = message.content
    else:
        # A list of content blocks stringifies to its Python repr, whose HEAD is
        # whichever block came first -- for a reasoning provider that is the
        # thinking trace, so a head-truncation kept the block the summary needs
        # least and cut the assistant's own text off the end. Measured on an
        # AIMessage carrying a 900,000-character thinking block followed by
        # "here is the plan": the plan did not survive, and what did survive
        # was `[{'type': 'thinking', 'thinking': 'BBBB...`. `.text` splices the
        # text blocks in order, which is what the summary prompt would have
        # rendered anyway; the repr stays as the fallback for content with no
        # text blocks at all (images), where it is at least a description.
        # `str(...)` because `.text` is a `TextAccessor` -- a `str` subclass
        # that holds a reference back to the message it came from.
        content = str(getattr(message, "text", "")) or str(message.content)
    # Estimate against the message that is actually going to be sent, not the
    # original. Pulling the text out of a list of blocks can shrink it by
    # orders of magnitude -- a 900,000-character thinking block beside a
    # 16-character plan -- and scaling the ORIGINAL's cost by the extracted
    # length then asked for 1 character: measured, that copy came back at 37
    # tokens against a 4,000-token budget, throwing away 99% of what fitted.
    trimmed = message.model_copy(update={"content": content})
    cost = max(1, count_tokens([trimmed]))
    if cost <= budget:
        return trimmed
    chars = max(1, int(len(content) * budget / cost * 0.9))
    for _ in range(24):
        candidate = message.model_copy(update={"content": content[:chars] + marker})
        if count_tokens([candidate]) <= budget or chars <= 1:
            return candidate
        chars //= 2
    return message.model_copy(update={"content": content[:1] + marker})


def _build_bounded_summarizer(base_cls: Any) -> Any:
    """`SummarizationMiddleware` with a trim that works in an agent loop.

    The stock trim is ``trim_messages(..., strategy="last",
    start_on="human")``. In a chat app that is right; here it returns an empty
    list every time, because the only human message is the topic at index 0
    and there is never another one near the end to start on. The summary then
    becomes the string "Previous conversation was too long to summarize." and
    the messages are deleted anyway -- measured across tool-result sizes from
    200 to 8000 characters and budgets from 4,000 to 100,000 tokens: 0 of 781
    messages read, every time.

    Turning the trim off entirely fixes that but removes the only bound on
    what the summariser is asked to read. On a fresh run the triggers keep the
    window near 150k tokens, which is fine; on a RESUMED run carrying an
    existing long thread -- the case this whole change exists for -- the first
    compaction faces the lot. Measured on realistic histories: 305k, 793k and
    1.56M tokens. Those exceed a provider's context, the call fails, and the
    result is the same placeholder and the same lost history.

    So take the last N tokens and keep whatever pairing is there, without
    requiring a human turn to start on. Older material is genuinely dropped,
    which a summary cannot avoid; what matters is that the recent window is
    actually read and a real summary is written.
    """

    class _BoundedSummarizationMiddleware(base_cls):  # type: ignore[misc, valid-type]
        @property
        def name(self) -> str:
            """Report the base class's name so this REPLACES the default.

            deepagents merges custom middleware into its base stack by `.name`,
            and `AgentMiddleware.name` defaults to the class name. A subclass
            therefore stops matching and gets appended instead of substituted,
            leaving deepagents' own summarizer in place -- two summarizers,
            one of them still on the unprofiled 170k/6 fallback this change
            exists to replace. Verified: the stack showed both before this.
            """
            return base_cls.__name__

        def before_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
            """Compact only if a summary was actually written.

            The base class deletes the history the moment ``_create_summary``
            returns, whatever it returned. With its retreat exhausted that
            meant one apologetic sentence replacing the run's entire
            memory: measured, 801 messages -> 21 after 12 failed provider
            calls, with nothing summarised. Dropping the compaction instead
            costs a larger request for one turn and retries on the next,
            which is strictly the cheaper mistake.
            """
            try:
                update = super().before_model(state, runtime)
            except _SummaryUnavailable:
                logger.error(
                    "summarization produced no summary; leaving the "
                    "conversation uncompacted rather than deleting it"
                )
                return None
            return self._keep_objective(state, update)

        async def abefore_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
            """Async twin of :meth:`before_model`; same reason."""
            try:
                update = await super().abefore_model(state, runtime)
            except _SummaryUnavailable:
                logger.error(
                    "summarization produced no summary; leaving the "
                    "conversation uncompacted rather than deleting it"
                )
                return None
            return self._keep_objective(state, update)

        def _keep_objective(
            self, state: Any, update: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            """Carry the run's own topic through the compaction, verbatim.

            The cutoff is ``len(messages) - keep``, so message 0 is always in
            the evicted half -- and in this harness message 0 is the study's
            objective, delivered as ``{"messages": [{"role": "user", "content":
            topic}]}`` by the CLI and repeated nowhere else in context (the
            manager system prompt is built from ``out_dir`` alone). Measured
            end to end through a real deepagents graph: after the first
            compaction the topic text is gone from state and survives only
            inside whatever the summariser wrote, which the next compaction
            then summarises again.

            That is the drift this repo has already paid for once elsewhere:
            "an 86-character paraphrase reached oed_setup_search in place of a
            775-character prompt, dropping the scoring contract and the 'beat
            it by 10%' target" (``cli/repl.py``). Tools were fixed by reading
            ``user_prompt.txt`` from disk; the manager's own reasoning has no
            such fallback, so pin the message. It costs one message of a few
            hundred tokens and it is re-pinned, not re-appended, on every later
            compaction because it is message 0 again by then.

            A summary message is never pinned -- otherwise a thread compacted
            before this existed would keep an old summary alive forever -- and
            neither is an implausibly large first turn, which would be a pasted
            document rather than an objective.
            """
            if not update:
                return update
            messages = state["messages"]
            if not messages:
                return update
            first = messages[0]
            if getattr(first, "type", "") != "human":
                return update
            if getattr(first, "additional_kwargs", {}).get("lc_source") == "summarization":
                return update
            if self.token_counter([first]) > _MAX_PINNED_OBJECTIVE_TOKENS:
                return update
            out = list(update.get("messages") or ())
            # out is [RemoveMessage(REMOVE_ALL_MESSAGES), summary, *preserved];
            # slot the objective in behind the removal so the reducer re-adds it
            # under its own id, ahead of the summary that describes it. If the
            # base class ever stops leading with the removal the update is an
            # append, not a replacement, and nothing here needs pinning -- so
            # check rather than assume the position.
            if not out or type(out[0]).__name__ != "RemoveMessage":
                return update
            if any(getattr(m, "id", None) == getattr(first, "id", None) for m in out[1:]):
                return update
            out.insert(1, first)
            return {**update, "messages": out}

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            """Survive a provider that says the request is too big.

            `langchain_google_genai` raises `GoogleContextOverflowError` (a
            `ContextOverflowError`) for "input exceeds the maximum number of
            tokens allowed", in its own words "so that upstream middleware can
            catch it and fall back to context compaction" -- and gemini and
            glm both reach it, since `GeminiChatModel` delegates straight to
            that library's `_generate`. deepagents' summarizer caught it; this
            one replaces that summarizer, and LangChain's implements
            `before_model` only. Measured on a model that refuses an oversized
            request: deepagents' stack ran to completion, ours died on the
            first refusal.

            The trigger cannot prevent this on its own, because the reason we
            are here is that the estimate was wrong -- 4 chars/token is
            generous for OpenFOAM dictionaries and source. So believe the
            provider, send the window this middleware would have kept anyway,
            and let the next turn's `before_model` compact state properly.
            """
            try:
                return handler(request)
            except ContextOverflowError:
                for messages in self._emergency_windows(request):
                    try:
                        return handler(request.override(messages=messages))
                    except ContextOverflowError:
                        continue
                raise

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            """Async twin of :meth:`wrap_model_call`; same reason."""
            try:
                return await handler(request)
            except ContextOverflowError:
                for messages in self._emergency_windows(request):
                    try:
                        return await handler(request.override(messages=messages))
                    except ContextOverflowError:
                        continue
                raise

        def _emergency_windows(self, request: Any) -> List[List[Any]]:
            """Progressively smaller tails of the request, pair-safe.

            Halving rather than one fixed size because a single oversized tool
            result inside the keep window would otherwise make the one retry
            fail for the same reason as the original call. Each window starts
            on a message the provider will accept as an opening -- never on a
            tool result whose call was cut away.

            Measured: every window in practice opens on the AIMessage that made
            the call, not on a human turn, which is one thing LangChain's own
            eviction never does (it always prepends a summary HumanMessage).
            Left as it is deliberately: only `langchain_google_genai` and the
            OpenAI-compatible glm endpoint raise `ContextOverflowError` at all,
            and both accept an assistant-first history -- checked through
            `_parse_chat_history`, which yields role `model` then `user`. A
            provider that both raises this error and requires a user turn first
            would need a synthetic one prepended here.
            """
            messages = list(request.messages)
            keep = self.keep[1] if self.keep[0] == "messages" else _summarize_keep_messages()
            # Never larger than half of what was just refused. `keep` is 20 and
            # a request can be refused well below that -- a handful of large
            # tool results is enough -- and `_find_safe_cutoff` returns 0 (no
            # cut at all) whenever the window is not actually smaller, which
            # made the first version of this retry nothing.
            keep = min(keep, max(1, len(messages) // 2))
            windows: List[List[Any]] = []
            seen: set = set()
            while keep >= 1:
                cutoff = self._find_safe_cutoff(messages, keep)
                # `_find_safe_cutoff_point` walks FORWARD past a run of tool
                # results whose calling AIMessage it cannot find, and that walk
                # can reach the end of the list. Measured on a history whose
                # tail is eight ToolMessages with no matching call: it returned
                # one window of zero messages, and the warning below duly
                # announced "retrying with the last 0". No provider accepts a
                # request with no messages, so that retry could only turn a
                # recoverable overflow into a hard rejection. A cutoff at or
                # past the end cannot get smaller as `keep` halves, so stop.
                if not 0 < cutoff < len(messages):
                    break
                if cutoff not in seen:
                    seen.add(cutoff)
                    windows.append(messages[cutoff:])
                keep //= 2
            if windows:
                logger.warning(
                    "provider rejected a %d-message request as too large; retrying "
                    "with the last %d", len(messages), len(windows[0]),
                )
            return windows

        def _determine_cutoff_index(self, messages: List[Any]) -> int:
            """Don't pay for a compaction that cannot get under the trigger.

            ``keep`` is 20 *messages*, so nothing stops those 20 from being
            worth more than the 150k-token trigger on their own -- ten tool
            results of 60 KB reach it, and the largest single results in the
            three live runs are 53, 64 and 71 KB. When that happens the trigger is
            still met immediately after compacting, so the next turn compacts
            again, and the next: measured, five compactions on five
            consecutive turns, each one an extra full-window model call, and
            each one re-summarising the previous summary. Five generations of
            telephone in five turns is how the study's stated intent quietly
            drifts.

            Deferring is safe because the state is already as small as this
            configuration can make it -- the request is the same size either
            way. What is NOT safe is deferring a compaction the message-count
            trigger asked for: that trigger is the only bound on the
            checkpoint (see ``_summarize_trigger_messages``), so it always
            wins.
            """
            cutoff = super()._determine_cutoff_index(messages)
            if cutoff <= 0:
                return cutoff
            clauses = getattr(self, "_trigger_clauses", None)
            if clauses is None or any(
                len(messages) >= clause["messages"]
                for clause in clauses
                if set(clause) == {"messages"}
            ):
                # An unrecognisable trigger (a LangChain that renamed this)
                # falls through to compacting. Doing unnecessary work is the
                # recoverable half of this decision.
                return cutoff
            evicted = self.token_counter(messages[:cutoff])
            total = self.token_counter(messages)
            if evicted * _MIN_EVICTION_SHARE_DIVISOR < total:
                logger.debug(
                    "skipping summarization: evicting %d of %d tokens would not "
                    "get under the trigger", evicted, total,
                )
                return 0
            return cutoff

        def _create_summary(self, messages_to_summarize: List[Any]) -> str:
            """Summarise, halving the window rather than giving up on failure.

            A failure here is expensive: the base class emits its
            `RemoveMessage(REMOVE_ALL_MESSAGES)` whatever this returns, which
            is why `before_model` above throws the compaction away when the
            retreat is exhausted rather than letting the history go with it.

            The likeliest cause is the summary call itself overflowing the
            model's context, and that is recoverable: ask again with less.
            LangChain has no profile for any provider this harness runs
            (`_get_profile_limits()` returns None for all four), so the budget
            cannot be chosen from a known limit and has to be found by
            retreating. Transient faults benefit too, since each retry is a
            fresh request.
            """
            budget = _summarize_trim_tokens()
            try:
                for attempt in range(_SUMMARY_SHRINK_ATTEMPTS):
                    try:
                        _SUMMARY_BUDGET.value = budget
                        summary = super()._create_summary(messages_to_summarize)
                        if summary and "too long to summarize" not in summary:
                            return summary
                    except Exception as exc:  # noqa: BLE001 - any provider error
                        logger.warning(
                            "summarization attempt %d/%d failed (budget %s): %s",
                            attempt + 1, _SUMMARY_SHRINK_ATTEMPTS, budget, exc,
                        )
                    if budget is None:
                        budget = _summarize_trim_tokens() or 60000
                    budget = max(4000, budget // 2)
            finally:
                _SUMMARY_BUDGET.value = None
            raise _SummaryUnavailable(
                f"no summary after {_SUMMARY_SHRINK_ATTEMPTS} attempts"
            )

        async def _acreate_summary(self, messages_to_summarize: List[Any]) -> str:
            """The async path, with the same retreat as the sync one.

            LangGraph picks the hook by how the graph is driven, and this
            harness drives it synchronously today -- but the base class's
            async summariser would silently inherit whatever budget the last
            sync retry happened to leave behind, and give up on the first
            failure. Cheap to make the two paths behave identically.
            """
            budget = _summarize_trim_tokens()
            try:
                for attempt in range(_SUMMARY_SHRINK_ATTEMPTS):
                    try:
                        _SUMMARY_BUDGET.value = budget
                        summary = await super()._acreate_summary(messages_to_summarize)
                        if summary and "too long to summarize" not in summary:
                            return summary
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "async summarization attempt %d/%d failed (budget %s): %s",
                            attempt + 1, _SUMMARY_SHRINK_ATTEMPTS, budget, exc,
                        )
                    if budget is None:
                        budget = _summarize_trim_tokens() or 60000
                    budget = max(4000, budget // 2)
            finally:
                _SUMMARY_BUDGET.value = None
            raise _SummaryUnavailable(
                f"no summary after {_SUMMARY_SHRINK_ATTEMPTS} async attempts"
            )

        def _trim_messages_for_summary(self, messages: List[Any]) -> List[Any]:
            # This thread's retreat budget when a compaction is in progress,
            # otherwise the configured default. Reading the env unconditionally
            # made every retry send the identical payload -- four attempts all
            # at 120,630 tokens against a 40k ceiling -- so the retreat never
            # actually retreated.
            budget = getattr(_SUMMARY_BUDGET, "value", None)
            if budget is None:
                budget = self.trim_tokens_to_summarize
            if budget is None or not messages:
                return list(messages)
            # Count each message once and accumulate. Re-counting the whole
            # kept list on every step is quadratic, and the cost is real
            # rather than theoretical: measured 11.5s per compaction on a
            # history of many small messages, where the budget is never hit
            # early enough to stop the loop. Counting one message at a time
            # UNDER-estimates by up to 25%, because `count_tokens_approximately`
            # applies its usage-metadata scaling only to a list of more than
            # one message; the budget is a self-imposed ceiling well under any
            # provider's, so 120k that is really 150k still fits.
            kept: List[Any] = []
            total = 0
            for message in reversed(messages):
                cost = self.token_counter([message])
                if not kept and cost > budget:
                    # One message bigger than the entire budget. Sending it
                    # whole is what the retreat above cannot recover from:
                    # measured, a 2 MB read_text_file result came back as a
                    # 500,015-token payload at EVERY budget from 120k down to
                    # 4k, so all four attempts failed identically and the
                    # history was lost. `max_chars` is the model's to choose
                    # (live runs already show 70,736-character reads), so this
                    # is reachable. Read its head instead of failing on all
                    # of it.
                    kept.append(_truncated_to_budget(message, budget, self.token_counter))
                    break
                if kept and total + cost > budget:
                    break
                kept.append(message)
                total += cost
            kept.reverse()
            # Never OPEN on an orphaned tool result -- a ToolMessage whose
            # AIMessage was trimmed away reads as a reply to nothing -- but
            # drop the leading run only while something that is not a tool
            # result survives it. Popping `while len(kept) > 1` instead threw
            # away everything but the last message whenever the whole window
            # was a single parallel tool batch, which is this harness's normal
            # shape: the manager launches "one task call per case ... in a
            # single message". Measured on 30 concurrent `task` results of
            # 20 KB each, 23 messages / 115,138 tokens fitted the 120,000-token
            # budget and the summariser was handed 1 message / 5,006 tokens --
            # 4% of it. The 22 discarded messages were subagent reports being
            # evicted from state in the same breath, which is the most
            # expensive thing in this conversation to lose.
            first_kept = next(
                (i for i, m in enumerate(kept) if getattr(m, "type", "") != "tool"),
                None,
            )
            if first_kept is not None:
                del kept[:first_kept]
            return kept

    return _BoundedSummarizationMiddleware


def _summarize_keep_messages() -> int:
    """How many recent messages survive summarisation."""
    return int(os.environ.get("CFD_SCIENTIST_SUMMARIZE_KEEP", "20"))


# Clearing a tool result is safe only when calling the tool again reproduces
# it cheaply. That is a small minority of what this harness offers, so the
# safe set is enumerated and everything else is exempt -- the opposite of the
# denylist this started as, which named seven tools and would have cleared
# `oed_run_code_mod_candidate`. Re-calling that does not re-read a file; it
# recompiles a turbulence model and re-solves the case, costing hours and real
# solver budget, and the model would have had no way to know it had already
# run. `task` is the same hazard one level up: it is how a subagent reports
# back, and re-calling it re-runs the subagent.
#
# Enumerated as an allowlist and inverted against the live tool list rather
# than maintained by hand, so a tool added to that list later is exempt by
# default and a mistake costs a wasted re-read instead of a wasted candidate.
# Note the inversion only covers tools this harness passes in: a tool
# deepagents injects and we never see (`write_todos`, its built-in
# `read_file`/`grep`) is absent from the list and therefore clearable. That is
# the right answer for those particular tools -- all cheap re-reads, and
# `read_file` results were indeed cleared in a replay of a real run's
# checkpoint -- but it is not a general guarantee, which is why `task` is
# named explicitly below rather than being trusted to appear.
_CHEAPLY_REPEATABLE_TOOLS = frozenset({
    "read_text_file",
    "grep_files",
    "list_directory",
    "find_files",
    "directory_tree",
})


def _tools_to_exempt(tools: List[Any]) -> tuple:
    """Every tool whose result must survive clearing.

    Derived from the tools actually bound to this agent. ``task`` is added
    unconditionally: deepagents injects it rather than passing it here, and it
    carries subagent reports.

    There is no safe answer without a tool list. ``exclude_tools`` is an exact
    name match, not a pattern, so a sentinel like ``"*"`` excludes nothing and
    clears everything -- measured, it wiped 194 of 200 candidate reports. The
    caller therefore omits the clearing layer entirely in that case; see
    ``build_context_middleware``.
    """
    names = {getattr(t, "__name__", getattr(t, "name", "")) for t in tools}
    names.discard("")
    return tuple(sorted((names - _CHEAPLY_REPEATABLE_TOOLS) | {"task"}))


def _token_counter_kwargs(model: Any) -> Dict[str, Any]:
    """A Claude-tuned token counter when the provider is Claude in disguise.

    LangChain picks 3.3 chars/token for Claude and 4.0 for everything else,
    keyed on ``_llm_type.startswith("anthropic-chat")``. This harness's
    subscription wrapper reports ``claude-code-sdk``, so it silently gets the
    4.0 rate for a model that tokenizes at 3.3: measured on one 101-message
    conversation, 52,649 estimated against 63,771 real -- 21% low, which puts
    the 150k trigger at about 182k actual tokens against Sonnet's 200k window.
    Unlike gemini and glm, this provider raises no ``ContextOverflowError``
    the emergency clip could catch, so the estimate is the only defence.
    """
    llm_type = str(getattr(model, "_llm_type", "")).lower()
    if "claude" not in llm_type and "anthropic" not in llm_type:
        return {}
    from functools import partial

    from langchain_core.messages.utils import count_tokens_approximately

    return {
        "token_counter": partial(
            count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3
        )
    }


# Longest tool-error message the model is shown in full.
_TOOL_ERROR_CHARS = int(os.getenv("CFD_SCIENTIST_TOOL_ERROR_CHARS") or 2000)


def _tool_error_trim_middleware() -> Any:
    """Cut an oversized tool-error message down before it enters the conversation.

    When a tool's arguments fail validation, the tool node's error text repeats
    every argument back ("Error invoking tool ... with kwargs {...} with
    error"). A model stuck retrying with a growing argument therefore grows the
    conversation twice per turn: on experiments_for_paper/qwen_27b_nothink/palmo
    one such message was 13,226 characters, in a run that ended when the
    conversation passed the endpoint's 217,718-token limit. The head and the
    tail are kept, and the tail is where the error itself is.
    """
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    def _trim(message: Any) -> Any:
        content = getattr(message, "content", None)
        if (
            isinstance(message, ToolMessage)
            and getattr(message, "status", None) == "error"
            and isinstance(content, str)
            and len(content) > _TOOL_ERROR_CHARS
        ):
            half = _TOOL_ERROR_CHARS // 2
            omitted = len(content) - 2 * half
            return message.model_copy(update={
                "content": content[:half] + f"\n…[{omitted} characters omitted]…\n" + content[-half:]
            })
        return message

    class TrimToolErrors(AgentMiddleware):
        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            return _trim(handler(request))

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            return _trim(await handler(request))

    return TrimToolErrors()


def build_context_middleware(
    model: Any, tools: Optional[List[Any]] = None
) -> List[Any]:
    """Bound the conversation two ways: clear stale tool results, and summarise.

    They fix different problems, which is why both are here.

    Clearing implements only ``wrap_model_call`` and rewrites the outgoing
    request -- it decides what this turn SENDS, and never touches state. That
    is what stops the model drowning in its own old file reads. Summarising
    returns ``RemoveMessage(REMOVE_ALL_MESSAGES)`` plus a summary, rewriting
    state -- that is what stops the checkpoint growing without bound.

    Note the stack order is not the order written here: deepagents replaces
    the summarizer in its base position and appends clearing after the core
    entries, so summarisation actually wraps clearing and sees the untrimmed
    request. That is why summarisation carries a message-count trigger as well
    as a token one; see ``_summarize_trigger_messages``.
    """
    middleware: List[Any] = []
    try:
        from langchain.agents.middleware import (
            ClearToolUsesEdit,
            ContextEditingMiddleware,
            SummarizationMiddleware,
        )
    except ImportError:
        # An older langchain without context editing: the agent still runs,
        # just without compaction, exactly as it did before this existed.
        return middleware

    middleware.append(_tool_error_trim_middleware())

    # Clearing needs the tool list to know what is safe to clear. Without it,
    # summarization alone still bounds the conversation -- slower and lossier,
    # but it never throws away a result that cost hours to produce.
    if tools:
        middleware.append(
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=_context_trigger_tokens(),
                        keep=_context_keep_tool_uses(),
                        # Clear the result, keep the call. The model still sees
                        # that it read a file and which one -- it just no longer
                        # carries the contents -- so it can decide to read it
                        # again rather than being confused about what it did.
                        clear_tool_inputs=False,
                        exclude_tools=_tools_to_exempt(tools),
                        placeholder=(
                            "[tool result cleared to free context - call this tool "
                            "again if you still need its output]"
                        ),
                    )
                ],
            )
        )
    middleware.append(
        _build_bounded_summarizer(SummarizationMiddleware)(
            model=model,
            **_token_counter_kwargs(model),
            trigger=[
                ("tokens", _summarize_trigger_tokens()),
                ("messages", _summarize_trigger_messages()),
            ],
            # 20 messages, not the 6 the unprofiled fallback would have used.
            # A round here spans propose -> launch -> score -> record, and
            # cutting to six lands mid-round with the batch half described.
            keep=("messages", _summarize_keep_messages()),
            trim_tokens_to_summarize=_summarize_trim_tokens(),
        )
    )
    return middleware

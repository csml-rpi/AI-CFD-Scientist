from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
import os
from cfd_langgraph.llm.token_usage_logger import append_usage_call, current_caller


@dataclass
class TokenStats:
    """Simple global accumulator for LLM token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


_GLOBAL_STATS = TokenStats()


class TokenStatsCallbackHandler(BaseCallbackHandler):
    """LangChain callback that records token usage from LLM responses.

    It looks for common usage fields across providers (OpenAI, Anthropic, Bedrock):
    - token_usage: {prompt_tokens, completion_tokens}
    - usage: {input_tokens, output_tokens}

    ``agent`` labels every call this handler sees. It is carried on the
    handler — and so on the model instance it is attached to — rather than in
    a global or an env var, because the manager runs its tool calls in a
    thread pool: a process-global label would be read by whichever thread
    logged next and would mis-attribute concurrent subagents to each other.
    Binding it to the model instance makes the attribution correct by
    construction, with no locking.
    """

    def __init__(self, agent: str = "", graph_name: str = "") -> None:
        super().__init__()
        self.agent = (agent or "").strip()
        # The agent graph this model was built for, when its name differs from
        # the label (the manager's graph is "cfd-scientist-manager").
        self.graph_name = (graph_name or "").strip()
        # run_id -> (model the call asked for, agent graph running it). Keyed
        # by run, so one handler shared across threads never mixes calls up.
        self._runs: Dict[Any, tuple] = {}

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any = None,
                            metadata: Any = None, **kwargs: Any) -> None:
        params = kwargs.get("invocation_params") or {}
        asked = params.get("model") or params.get("model_name") or params.get("model_id") or ""
        running = str((metadata or {}).get("lc_agent_name") or "")
        if run_id is not None:
            self._runs[run_id] = (str(asked), running)

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        self._runs.pop(run_id, None)

    def _agent_for(self, running: str) -> str:
        if self.agent:
            # A model built for one agent but run by another agent's graph is
            # that other agent's work. deepagents hands the manager's own model
            # to the general-purpose subagent it adds by default, so without
            # this everything that subagent did was filed as "manager".
            if running and running not in (self.agent, self.graph_name):
                return running
            return self.agent
        # An unlabelled model: the manager tool it was called from, else the
        # agent graph running it; else the logger files it under the stage.
        return current_caller() or running

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:  # type: ignore[override]
        llm_output: Dict[str, Any] = getattr(response, "llm_output", {}) or {}
        # A provider that estimates rather than reports says so; everything else is
        # the provider's own accounting. Labelling every call "provider_usage" had made
        # Codex's tokenizer estimates indistinguishable from real reported usage.
        token_source = str(llm_output.get("token_source") or "provider_usage")
        response_ids = list(llm_output.get("response_ids") or [])
        attempts = int(llm_output.get("attempts") or 1)
        # A streamed reply has no llm_output, and its response_metadata is the
        # merge of every chunk, where a name sent twice is concatenated: the
        # repo-root log holds "qwen/qwen3.8-flashqwen/qwen3.8-flash" from
        # OpenRouter streams, a model that exists nowhere. The model the call
        # asked for is used before that merged name.
        asked, running = self._runs.pop(run_id, ("", ""))
        model = str(llm_output.get("model_name") or llm_output.get("model") or asked or "")

        # LangChain's own `usage_metadata` on the message comes first. Every
        # integration fills it the same way -- input includes the cached share,
        # output includes reasoning -- while each provider's raw usage in
        # llm_output is its own dialect. Reading the raw one first undercounted
        # Anthropic, whose `input_tokens` leaves out cache reads and writes, so
        # a cached Claude agent turn was logged at a fraction of its input. It
        # also lost ChatOpenAI's cached and reasoning counts, which sit under
        # prompt_tokens_details / completion_tokens_details. The Vertex/Gemini
        # transport puts nothing at all in llm_output, only here.
        prompt = completion = cached = reasoning = 0
        for batch in (getattr(response, "generations", None) or []):
            for gen in (batch or []):
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if not model:
                    rm = getattr(msg, "response_metadata", None)
                    if isinstance(rm, dict):
                        model = str(rm.get("model_name") or rm.get("model") or "")
                if isinstance(um, dict) and (um.get("input_tokens") or um.get("output_tokens")):
                    prompt = int(um.get("input_tokens") or 0)
                    completion = int(um.get("output_tokens") or 0)
                    cached = int((um.get("input_token_details") or {}).get("cache_read") or 0)
                    reasoning = int((um.get("output_token_details") or {}).get("reasoning") or 0)
                    break
            if prompt or completion:
                break

        if not prompt and not completion:
            usage: Dict[str, Any] = (
                llm_output.get("token_usage")
                or llm_output.get("usage")
                or {}
            )
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            cached = int(
                usage.get("cached_input_tokens")
                or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                or (usage.get("input_tokens_details") or {}).get("cached_tokens")
                or 0
            )
            reasoning = int(
                usage.get("reasoning_output_tokens")
                or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
                or 0
            )

        if not prompt and not completion:
            # The call happened and was paid for, but nothing reported its size.
            # Logged as "missing" rather than as a zero under "provider_usage":
            # measured on the Qwen studies, where every streamed call from
            # study_mode, metric_setup, surrogate_setup and OED came back with
            # no usage and sat in the log as a real-looking zero.
            token_source = "missing"

        if not model:
            # The Vertex/Gemini route reports model_name as an empty string,
            # so nothing in the response identifies which model answered and
            # every GLM call was filed under "unknown". The model that was
            # asked is the model that replied, so fall back to the configured
            # one. A name the response DOES supply always wins, which keeps
            # this correct if stages are ever routed to different models and
            # only leaves the fallback for providers that report nothing.
            model = (
                os.getenv("CFD_SCIENTIST_MODEL")
                or os.getenv("CFD_SCIENITST_MODEL")
                or ""
            ).strip()

        try:
            _GLOBAL_STATS.prompt_tokens += int(prompt or 0)
            _GLOBAL_STATS.completion_tokens += int(completion or 0)
            provider = (
                os.getenv("CFD_SCIENTIST_LLM_PROVIDER")
                or os.getenv("CFD_SCIEINTIST_LLM_PROVIDER")
                or ""
            )
            append_usage_call(
                agent=self._agent_for(running),
                provider=provider,
                model=model,
                input_tokens=int(prompt or 0),
                output_tokens=int(completion or 0),
                cached_input_tokens=cached,
                reasoning_output_tokens=reasoning,
                token_source=token_source,
                response_ids=response_ids,
                attempts=attempts,
            )
        except Exception as exc:
            # Never let accounting break a study, but never lose a call silently
            # either: a dropped row is only noticed when the cost question is
            # asked, long after it could have been fixed.
            import sys
            print(f"[token accounting] call NOT logged ({type(exc).__name__}: {exc})",
                  file=sys.stderr, flush=True)
            return


TOKEN_STATS_HANDLER = TokenStatsCallbackHandler()


def get_token_stats() -> TokenStats:
    """Return a copy of the current global token stats."""
    return TokenStats(
        prompt_tokens=_GLOBAL_STATS.prompt_tokens,
        completion_tokens=_GLOBAL_STATS.completion_tokens,
    )


def estimate_sonnet_46_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, float]:
    """Estimate USD cost for Claude Sonnet 4.6 given token counts.

    Pricing (per user spec):
      - $3.00 per 1M input (prompt) tokens
      - $15.00 per 1M output (completion) tokens
    """
    in_millions = prompt_tokens / 1_000_000.0
    out_millions = completion_tokens / 1_000_000.0

    prompt_cost = 3.0 * in_millions
    completion_cost = 15.0 * out_millions
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": total_cost,
    }


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler


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
    """

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        llm_output: Dict[str, Any] = getattr(response, "llm_output", {}) or {}
        usage: Dict[str, Any] = (
            llm_output.get("token_usage")
            or llm_output.get("usage")
            or {}
        )

        prompt = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        completion = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )

        try:
            _GLOBAL_STATS.prompt_tokens += int(prompt or 0)
            _GLOBAL_STATS.completion_tokens += int(completion or 0)
        except Exception:
            # Swallow any casting issues; stats are best-effort only.
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


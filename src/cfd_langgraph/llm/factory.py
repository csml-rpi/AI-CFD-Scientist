from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse


# Longer read timeout for Bedrock (vision/large payloads can take 60s+). Env: BEDROCK_READ_TIMEOUT (default 300).
def _bedrock_read_timeout() -> int:
    return int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))


def create_langchain_llm(model: str, temperature: float = 0.2) -> Any:
    m = model.strip()
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
            return ChatBedrockConverse(client=bedrock_client, model=model_id, temperature=temperature)
        except Exception:
            return ChatBedrockConverse(model=model_id, temperature=temperature)
    if m.startswith("claude-"):
        return ChatAnthropic(model=m, temperature=temperature)

    # Codex/OpenAI models use API-key auth via OPENAI_API_KEY.
    # Examples:
    #   CFD_SCIENTIST_MODEL=codex/gpt-5-codex
    #   CFD_SCIENTIST_MODEL=gpt-5-codex
    if m.startswith("codex/"):
        openai_model = m.split("codex/", 1)[1].strip() or "gpt-5-codex"
        return ChatOpenAI(model=openai_model, temperature=temperature)

    if m == "codex":
        return ChatOpenAI(model="gpt-5-codex", temperature=temperature)

    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return ChatOpenAI(model=m, temperature=temperature)
    if "gemini" in m:
        # assuming OpenAI-compatible endpoint if user configured env externally
        return ChatOpenAI(model=m, temperature=temperature)
    raise ValueError(f"Unsupported model: {model}")

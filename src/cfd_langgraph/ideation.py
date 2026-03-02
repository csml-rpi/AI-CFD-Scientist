from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from difflib import SequenceMatcher

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Settings
from .llm.factory import create_langchain_llm
from .literature import LiteratureClient
from .utils import strip_json_fences


def load_prompts(prompts_path: Path) -> Dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_literature_context(items: list[dict]) -> str:
    if not items:
        return "No external literature retrieved (missing API keys or no results)."

    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"[{i}] {it.get('title', '')}"
            f" | year={it.get('year', 'n/a')} | venue={it.get('venue', 'n/a')}"
            f" | source={it.get('source', '')}\n"
            f"URL: {it.get('url', '')}\n"
            f"Summary: {it.get('snippet', '')[:300]}"
        )
    return "\n\n".join(lines)


def _idea_to_text(idea_json: Dict[str, Any]) -> str:
    return json.dumps(idea_json, ensure_ascii=False, sort_keys=True).lower()


def _literature_texts(lit_items: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for x in lit_items:
        t = f"{x.get('title', '')} {x.get('snippet', '')}".strip().lower()
        if t:
            texts.append(t)
    return texts


def novelty_score(idea_json: Dict[str, Any], lit_items: List[Dict[str, Any]]) -> float:
    """
    Returns maximum string similarity between proposed idea and any prior-study text.
    Lower is better (more novel). Uses SequenceMatcher ratio as a conservative filter.
    """
    itext = _idea_to_text(idea_json)
    refs = _literature_texts(lit_items)
    if not refs:
        return 0.0
    return max(SequenceMatcher(None, itext, r).ratio() for r in refs)


def _extract_experiment_count(idea_json: Dict[str, Any]) -> int | None:
    """
    Best-effort count using common fields in this project schema.
    """
    if not isinstance(idea_json, dict):
        return None

    # If model returns explicit field, trust it
    if isinstance(idea_json.get("total_experiments"), int):
        return int(idea_json["total_experiments"])

    cases = idea_json.get("cases")
    if not isinstance(cases, list):
        return None

    total = 0
    for c in cases:
        if not isinstance(c, dict):
            continue
        fs = c.get("fuel_speed_list")
        bs = c.get("box_size_list")
        if isinstance(fs, list) and isinstance(bs, list):
            total += max(1, len(fs) * len(bs))
        elif isinstance(fs, list):
            total += max(1, len(fs))
        elif isinstance(bs, list):
            total += max(1, len(bs))
        else:
            total += 1
    return total


def run_ideation(settings: Settings, research_topic: str) -> Dict[str, Any]:
    prompts = load_prompts(settings.prompts_path)
    ideation_prompts = prompts["IdeationAgent"]

    lit_items = []
    if settings.ideation_enable_literature:
        client = LiteratureClient(settings.s2_api_key, settings.brave_api_key)
        lit_items = client.to_json_ready(
            client.collect(
                query=research_topic,
                max_papers=settings.ideation_max_papers,
                max_web_results=settings.ideation_max_web_results,
            )
        )

    literature_context = build_literature_context(lit_items)

    system_prompt = ideation_prompts["initial_idea_prompt"]
    user_prompt = ideation_prompts.get(
        "literature_aware_user_prompt",
        "Research topic: {research_topic}\n\nPrior studies:\n{literature_context}\nMax experiments: {max_experiments}",
    ).format(
        research_topic=research_topic,
        literature_context=literature_context,
        max_experiments=settings.ideation_max_experiments,
    )

    llm = create_langchain_llm(model=settings.model, temperature=0.2)

    retries = max(0, settings.ideation_novelty_max_retries)
    threshold = settings.ideation_novelty_threshold

    novelty_val = None
    count_val = None
    last_raw = ""
    idea_json: Dict[str, Any] = {}

    for attempt in range(retries + 1):
        resp = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        last_raw = content

        try:
            idea_json = json.loads(strip_json_fences(content))
        except Exception:
            idea_json = {"raw_output": content, "parse_error": True}
            novelty_val = 1.0
            count_val = None
        else:
            novelty_val = novelty_score(idea_json, lit_items)
            count_val = _extract_experiment_count(idea_json)

        too_similar = novelty_val is not None and novelty_val >= threshold
        too_many = (
            count_val is not None and count_val > settings.ideation_max_experiments
        )

        if not too_similar and not too_many and not idea_json.get("parse_error"):
            break

        if attempt < retries:
            retry_tpl = ideation_prompts.get(
                "novelty_retry_user_prompt",
                "Previous idea too similar or exceeds limits. Regenerate with better novelty and max experiments <= {max_experiments}.\n\nPrior studies:\n{literature_context}\n\nPrevious idea:\n{previous_idea}",
            )
            user_prompt = retry_tpl.format(
                literature_context=literature_context,
                previous_idea=json.dumps(idea_json, ensure_ascii=False),
                max_experiments=settings.ideation_max_experiments,
            )

    return {
        "research_topic": research_topic,
        "literature_used": lit_items,
        "idea": idea_json,
        "novelty": {
            "max_similarity_to_prior": novelty_val,
            "threshold": threshold,
            "passed": (novelty_val is None or novelty_val < threshold),
            "max_retries": retries,
        },
        "experiment_count": {
            "estimated_total": count_val,
            "max_allowed": settings.ideation_max_experiments,
            "passed": (
                count_val is None or count_val <= settings.ideation_max_experiments
            ),
        },
        "raw_output": last_raw,
    }

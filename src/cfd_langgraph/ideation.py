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


def novelty_score_llm(
    llm: Any,
    idea_json: Dict[str, Any],
    lit_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LLM-based novelty evaluator.
    Returns JSON-like dict:
      {
        "max_similarity_to_prior": float in [0,1],
        "judgement": "novel"|"too_similar",
        "reason": str
      }
    """
    literature_context = build_literature_context(lit_items)
    system = (
        "You are a strict CFD novelty evaluator.\n"
        "Compare ONE proposed study idea against prior studies.\n"
        "Decide if the idea is too similar to existing work.\n"
        "Output ONLY JSON with keys:\n"
        '- "max_similarity_to_prior": number in [0,1] (higher means more similar)\n'
        '- "judgement": "novel" or "too_similar"\n'
        '- "reason": short string\n'
        "Do not output markdown or extra text."
    )
    user = (
        "Prior studies:\n"
        f"{literature_context}\n\n"
        "Proposed idea JSON:\n"
        f"{json.dumps(idea_json, ensure_ascii=False)}\n\n"
        "Evaluate novelty now."
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    parsed = json.loads(strip_json_fences(content))
    return parsed if isinstance(parsed, dict) else {}




def _dedupe_and_cap_experiments(experiments: List[Dict[str, Any]], max_experiments: int) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for e in experiments:
        if not isinstance(e, dict):
            continue
        key_obj = {
            "name": e.get("name"),
            "topology": e.get("topology"),
            "dimensions": e.get("dimensions"),
            "parameters": e.get("parameters", {}),
            "controls": e.get("controls", {}),
        }
        key = json.dumps(key_obj, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    # ensure IDs are normalized and cap applied
    out = []
    for i, e in enumerate(uniq[:max_experiments], 1):
        x = dict(e)
        x["experiment_id"] = f"exp_{i:03d}"
        out.append(x)
    return out


def _normalize_to_experiments_schema(idea_json: Dict[str, Any], max_experiments: int) -> Dict[str, Any]:
    """
    Normalize older or partial idea JSONs to the canonical experiments[] schema.
    New code should populate experiments[] directly; this helper keeps a minimal,
    generic path for legacy ideas without embedding fuel-specific concepts.
    """
    if not isinstance(idea_json, dict):
        return idea_json

    # Preferred path: experiments already provided
    if isinstance(idea_json.get("experiments"), list):
        idea_json["experiments"] = _dedupe_and_cap_experiments(idea_json.get("experiments", []), max_experiments)
        return idea_json

    # Generic legacy path: convert cases[*].parameters -> experiments without fuel-specific fields
    cases = idea_json.get("cases", [])
    experiments: List[Dict[str, Any]] = []
    if isinstance(cases, list):
        for c in cases:
            if not isinstance(c, dict):
                continue
            params = c.get("parameters") if isinstance(c.get("parameters"), dict) else {}
            experiments.append(
                {
                    "name": c.get("name", "experiment"),
                    "topology": c.get("topology", "2d"),
                    "dimensions": c.get("dimensions", [1.0, 1.0, 0.1]),
                    "parameters": params,
                    "controls": {"target_CFL": idea_json.get("target_CFL", 0.5)},
                    "notes": c.get("description", ""),
                }
            )

    idea_json["experiments"] = _dedupe_and_cap_experiments(experiments, max_experiments)
    return idea_json
def _extract_experiment_count(idea_json: Dict[str, Any]) -> int | None:
    """
    Best-effort count using common fields in this project schema.
    """
    if not isinstance(idea_json, dict):
        return None

    # If explicit field exists, trust it
    if isinstance(idea_json.get("total_experiments"), int):
        return int(idea_json["total_experiments"])

    experiments = idea_json.get("experiments")
    if isinstance(experiments, list):
        return len(experiments)

    cases = idea_json.get("cases")
    if isinstance(cases, list):
        # Fallback: treat each case as at least one experiment
        return sum(1 for c in cases if isinstance(c, dict))
    return None


def run_ideation(settings: Settings, research_topic: str, verbose: bool = True) -> Dict[str, Any]:
    prompts = load_prompts(settings.prompts_path)
    ideation_prompts = prompts["IdeationAgent"]

    if verbose:
        print("[Ideation] Starting literature-aware ideation...", flush=True)

    lit_items = []
    if settings.ideation_enable_literature:
        client = LiteratureClient(settings.s2_api_key, settings.brave_api_key)
        if verbose:
            print("[Ideation] Fetching literature (Semantic Scholar, web)...", flush=True)
        lit_items = client.to_json_ready(
            client.collect(
                query=research_topic,
                max_papers=settings.ideation_max_papers,
                max_web_results=settings.ideation_max_web_results,
            )
        )
        if verbose:
            print("[Ideation] Literature: %d items" % len(lit_items), flush=True)
    else:
        if verbose:
            print("[Ideation] Literature disabled (ideation_enable_literature=False)", flush=True)

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

    if verbose:
        print("[Ideation] Generating idea (LLM)...", flush=True)

    retries = max(0, settings.ideation_novelty_max_retries)
    threshold = settings.ideation_novelty_threshold

    novelty_val = None
    novelty_reason = ""
    novelty_method = "llm"
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
            idea_json = _normalize_to_experiments_schema(idea_json, settings.ideation_max_experiments)
            try:
                novelty_eval = novelty_score_llm(llm, idea_json, lit_items)
                novelty_val = float(novelty_eval.get("max_similarity_to_prior", 1.0))
                novelty_reason = str(novelty_eval.get("reason", "") or "")
                if novelty_val < 0.0:
                    novelty_val = 0.0
                elif novelty_val > 1.0:
                    novelty_val = 1.0
            except Exception:
                # Safety fallback: keep pipeline running if novelty LLM output is malformed.
                novelty_method = "llm_fallback_heuristic"
                novelty_val = novelty_score(idea_json, lit_items)
                novelty_reason = "LLM novelty parse/eval failed; used heuristic fallback."
            count_val = _extract_experiment_count(idea_json)

        too_similar = novelty_val is not None and novelty_val >= threshold
        too_many = (
            count_val is not None and count_val > settings.ideation_max_experiments
        )

        if not too_similar and not too_many and not idea_json.get("parse_error"):
            if verbose:
                print("[Ideation] Idea accepted (novelty=%.3f, count=%s)" % (
                    novelty_val or 0, count_val), flush=True)
            break

        if verbose and (too_similar or too_many):
            print("[Ideation] Retry %d/%d: too_similar=%s too_many=%s" % (
                attempt + 1, retries, too_similar, too_many), flush=True)

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

    if verbose:
        exp_count = _extract_experiment_count(idea_json)
        print("[Ideation] Done. Experiments: %s" % exp_count, flush=True)

    return {
        "research_topic": research_topic,
        "literature_used": lit_items,
        "idea": idea_json,
        "novelty": {
            "max_similarity_to_prior": novelty_val,
            "threshold": threshold,
            "passed": (novelty_val is None or novelty_val < threshold),
            "max_retries": retries,
            "method": novelty_method,
            "reason": novelty_reason,
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

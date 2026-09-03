from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Settings
from .llm.factory import create_langchain_llm
from .literature import LiteratureClient
from .utils import extract_json_object, strip_json_fences


def load_prompts(prompts_path: Path) -> Dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_literature_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize the two literature schemas used by this repository.

    ``scripts/lit.py`` persists Semantic Scholar records with ``abstract``
    while :class:`LiteratureClient` emits ``snippet``/``source`` records.
    The deep-agent path fetches with the former and ideation historically
    expected the latter, which meant the supposedly literature-grounded
    hypothesis prompt received blank summaries and then fetched a second,
    unrelated paper set.  Keep one canonical in-memory shape instead.
    """
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "") or "").strip()
        url = str(raw.get("url", "") or "").strip()
        doi = str(raw.get("doi", "") or "").strip()
        key = (doi or url or title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        snippet = str(raw.get("snippet") or raw.get("abstract") or "").strip()
        normalized.append(
            {
                **raw,
                "source": str(raw.get("source") or "semantic_scholar"),
                "title": title,
                "url": url,
                "doi": doi or None,
                "snippet": snippet[:1200],
            }
        )
    return normalized


def build_literature_context(items: list[dict]) -> str:
    items = normalize_literature_records(items)
    if not items:
        return "No external literature retrieved (missing API keys or no results)."

    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"[{i}] {it.get('title', '')}"
            f" | year={it.get('year', 'n/a')} | venue={it.get('venue', 'n/a')}"
            f" | source={it.get('source', '')}\n"
            f"URL: {it.get('url', '')}\n"
            f"DOI: {it.get('doi') or 'n/a'}\n"
            f"Summary: {str(it.get('snippet') or it.get('abstract') or '')[:600]}"
        )
    return "\n\n".join(lines)


def _idea_to_text(idea_json: Dict[str, Any]) -> str:
    return json.dumps(idea_json, ensure_ascii=False, sort_keys=True).lower()


def _literature_texts(lit_items: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for x in lit_items:
        t = f"{x.get('title', '')} {x.get('snippet') or x.get('abstract') or ''}".strip().lower()
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
    parsed = json.loads(extract_json_object(content))
    if not isinstance(parsed, dict):
        raise ValueError("novelty evaluator did not return a JSON object")
    value = float(parsed.get("max_similarity_to_prior"))
    if not math.isfinite(value):
        raise ValueError("novelty similarity is not finite")
    judgement = str(parsed.get("judgement", "")).strip().lower()
    if judgement not in {"novel", "too_similar"}:
        raise ValueError(f"invalid novelty judgement: {judgement!r}")
    return {
        "max_similarity_to_prior": min(1.0, max(0.0, value)),
        "judgement": judgement,
        "reason": str(parsed.get("reason", "") or ""),
    }


def _idea_distinguishing_text(idea_json: Dict[str, Any]) -> str:
    """Text used to reject duplicate candidates within one proposal batch.

    Excludes the shared JSON scaffolding (IDs, generic controls) that makes
    whole-document SequenceMatcher scores misleadingly high.
    """
    parts = [
        str(idea_json.get("description", "") or ""),
        str(idea_json.get("objective", "") or ""),
        str(idea_json.get("solver", "") or ""),
        str((idea_json.get("post") or {}).get("objective", ""))
        if isinstance(idea_json.get("post"), dict) else "",
    ]
    for exp in idea_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        parts.extend(
            [
                str(exp.get("name", "") or ""),
                json.dumps(exp.get("parameters", {}), sort_keys=True, default=str),
                str(exp.get("notes", "") or ""),
            ]
        )
    return " ".join(" ".join(parts).lower().split())


def candidate_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    left = _idea_distinguishing_text(a)
    right = _idea_distinguishing_text(b)
    if not left or not right:
        return 1.0 if left == right else 0.0
    return SequenceMatcher(None, left, right).ratio()




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


def _fetch_literature(settings: Settings, research_topic: str, verbose: bool = True) -> List[Dict[str, Any]]:
    if not settings.ideation_enable_literature:
        if verbose:
            print("[Ideation] Literature disabled (ideation_enable_literature=False)", flush=True)
        return []
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
    return lit_items


def _judged_too_similar(judgement: str, similarity, threshold: float) -> bool:
    """Whether the novelty judge rejected this idea.

    The judge returns a verdict AND a self-reported similarity. The verdict
    decides; the number is a fallback for when there is no verdict to read,
    because it scores "same research area" rather than "same idea" and a study
    pinned to one case has a similarity floor set by its own topic.

    This lives in one place because it was written in two -- the retry loop's
    accept test and the `passed` flag recorded for the pipeline -- and fixing
    only the first produced candidates logged as "Idea accepted
    (judgement=novel)" and then rejected downstream with "Failed novelty gate
    before critique ran". Three of three, on runs/ideation_probe8.
    """
    if judgement == "too_similar":
        return True
    return not judgement and similarity is not None and similarity >= threshold


def _generate_one_idea(
    llm: Any,
    ideation_prompts: Dict[str, Any],
    research_topic: str,
    literature_context: str,
    lit_items: List[Dict[str, Any]],
    settings: Settings,
    verbose: bool = True,
    previous_ideas: Optional[List[Dict[str, Any]]] = None,
    candidate_similarity_threshold: float = 0.92,
    case_context: str = "",
) -> Dict[str, Any]:
    """One novelty-checked idea, with retries. Extracted from ``run_ideation``'s
    original loop body so both a single idea and a batch of candidates
    (:func:`run_ideation_batch`) can share the exact same generation +
    novelty-gate logic."""
    system_prompt = str(ideation_prompts["initial_idea_prompt"]).replace(
        "{max_experiments}", str(settings.ideation_max_experiments)
    )
    if case_context:
        # The base prompt asks for "an impactful CFD research idea" and "a
        # non-overlapping set of experiments" -- i.e. a study design. Given a
        # fixed case that is the wrong deliverable, and saying so only in the
        # user turn does not survive: measured on runs/ideation_probe, with the
        # fixed-case block moved to character 0 of the user prompt, 3 of 4
        # candidates still came back with pimpleFoam/IDDES/3D and 0 of 4 passed
        # critique. The same run at effort=none produced 4 of 4 3D LES ideas, so
        # it is not a reasoning-depth problem either -- the model is answering
        # the question the system prompt asked. Redefining the deliverable has
        # to happen here, in the same turn that sets the task.
        system_prompt = (
            "THIS STUDY RUNS ON A FIXED CASE. You are not designing a simulation "
            "campaign. The setup given below under FIXED CASE SETUP is already "
            "decided, and nothing you propose changes it.\n\n"
            "Propose variation ONLY in what the research topic asks to be varied. "
            "Whatever that is, your experiments are variants of it evaluated by "
            "re-running the one existing case — not different simulations. An "
            "idea that re-specifies any part of the fixed setup is off-topic by "
            "construction and will be rejected, however strong it would be as an "
            "independent study.\n\n"
            "Use the prior studies for the mechanisms they identify, expressed "
            "within what the topic varies; they are not study designs to copy.\n\n"
        ) + system_prompt
    previous_ideas = previous_ideas or []
    user_prompt = ideation_prompts.get(
        "literature_aware_user_prompt",
        "Research topic: {research_topic}\n\nPrior studies:\n{literature_context}\nMax experiments: {max_experiments}",
    ).format(
        research_topic=research_topic,
        literature_context=literature_context,
        max_experiments=settings.ideation_max_experiments,
    )
    if case_context:
        # Without this the ideator has no idea a concrete case already exists,
        # and proposes studies of a *different* configuration — observed on a
        # real run: every candidate was a setup-sensitivity study (confinement,
        # spanwise size, grid-scheme interaction), half of them at Re_H=10595
        # when the starter case is Re_H=5600.
        #
        # It leads the prompt rather than trailing it. Appended last it sat
        # behind the literature block -- 807 characters of constraint after
        # 11,544 characters of papers on run ph_codex_20260902_1402, 20,354 on
        # oed_20260822_1626_codex_high -- and lost. Measured pass rate fell with
        # the size of that block: 20/25 with no literature, 37/61 at ten papers,
        # 14/32 at twenty.
        #
        # The field list is the other half. The required schema asks for
        # `solver`, `topology` and `dimensions`, so a model with no instruction
        # to the contrary invents them, and the critic then rejects the idea for
        # exactly those fields: 45 of 64 rejections across every run cite a
        # setup violation (mesh 33, LES 33, dimension 31, solver 30, 3D 27).
        # Three consecutive rounds scored 0 of 5 this way. Naming the fields as
        # copied removes the choice instead of forbidding its consequences.
        user_prompt = (
            "FIXED CASE SETUP — this study runs on an existing case. Geometry, "
            "mesh, boundary conditions, solver, numerics and flow parameters below are "
            "GIVEN and must not be changed or re-proposed. Your hypothesis is about what "
            "to CHANGE IN THE MODEL relative to this case, evaluated on this case:\n"
            f"{case_context}\n"
            "\nThe schema's `solver`, `topology`, `dimensions` and `controls` fields "
            "DESCRIBE this fixed case — copy them from above. They are not choices to "
            "make. The only field your hypothesis decides is `parameters`: the model "
            "change itself. Read the prior studies below for mechanisms you can express "
            "as such a change, not for a study design to adopt.\n\n"
        ) + user_prompt
    if previous_ideas:
        prior_summaries = [
            _idea_distinguishing_text(x)[:1200] for x in previous_ideas[-8:]
        ]
        user_prompt += (
            "\n\nOTHER CANDIDATES ALREADY PROPOSED IN THIS BATCH:\n- "
            + "\n- ".join(prior_summaries)
            + "\nYour candidate must be materially different from all of them."
        )

    if verbose:
        print("[Ideation] Generating idea (LLM)...", flush=True)

    retries = max(0, settings.ideation_novelty_max_retries)
    threshold = settings.ideation_novelty_threshold

    novelty_val = None
    novelty_reason = ""
    novelty_judgement = ""
    novelty_method = "llm"
    max_candidate_similarity = 0.0
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
            idea_json = json.loads(extract_json_object(content))
        except Exception:
            idea_json = {"raw_output": content, "parse_error": True}
            novelty_val = 1.0
            count_val = None
        else:
            idea_json = _normalize_to_experiments_schema(idea_json, settings.ideation_max_experiments)
            try:
                novelty_eval = novelty_score_llm(llm, idea_json, lit_items)
                novelty_val = float(novelty_eval.get("max_similarity_to_prior", 1.0))
                novelty_judgement = str(novelty_eval.get("judgement", "")).lower()
                novelty_reason = str(novelty_eval.get("reason", "") or "")
                if novelty_val < 0.0:
                    novelty_val = 0.0
                elif novelty_val > 1.0:
                    novelty_val = 1.0
            except Exception:
                if lit_items:
                    # Lexical SequenceMatcher is not a semantic novelty test;
                    # treating its usually-low score as approval lets a
                    # malformed evaluator response bypass literature review.
                    # Retry, then fail closed if the evaluator never returns
                    # a valid judgement.
                    novelty_method = "llm_failed_closed"
                    novelty_val = 1.0
                    novelty_judgement = "too_similar"
                    novelty_reason = "LLM novelty evaluation failed; novelty was not verified."
                else:
                    novelty_method = "heuristic_no_literature"
                    novelty_val = novelty_score(idea_json, lit_items)
                    novelty_judgement = "novel" if novelty_val < threshold else "too_similar"
                    novelty_reason = "No literature was available; used lexical fallback."
            count_val = _extract_experiment_count(idea_json)
            max_candidate_similarity = max(
                (candidate_similarity(idea_json, prev) for prev in previous_ideas),
                default=0.0,
            )

        # The judge returns BOTH a verdict and a self-reported similarity, and
        # the number used to be able to veto the verdict. It should not: the
        # number scores "same research area", not "same idea", and on a study
        # pinned to one case that floor is set by the topic itself. Measured
        # across probes 5 and 6, every one of six rejected candidates carried
        # judgement "novel" -- four were discarded on the number alone, one at
        # exactly the threshold, with reasons like "none listed introduces ...
        # the application and calibration setting overlap strongly". That is a
        # correct similarity and an irrelevant one. The number now decides only
        # when there is no verdict to read; the failed-closed path still sets
        # judgement itself, and the within-batch duplicate check is unaffected.
        too_similar = _judged_too_similar(novelty_judgement, novelty_val, threshold)
        too_similar_to_batch = max_candidate_similarity >= candidate_similarity_threshold
        invalid_count = (
            count_val is None
            or count_val < 1
            or count_val > settings.ideation_max_experiments
        )

        if not too_similar and not too_similar_to_batch and not invalid_count and not idea_json.get("parse_error"):
            if verbose:
                print("[Ideation] Idea accepted (similarity=%.3f/%.2f, judgement=%s, count=%s)" % (
                    novelty_val or 0, threshold, novelty_judgement or "-", count_val), flush=True)
            break

        if verbose and (too_similar or too_similar_to_batch or invalid_count):
            print("[Ideation] Attempt %d/%d rejected: too_similar=%s duplicate_candidate=%s invalid_experiment_count=%s" % (
                attempt + 1, retries + 1, too_similar, too_similar_to_batch, invalid_count), flush=True)

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
            if previous_ideas:
                user_prompt += (
                    "\n\nAlso avoid duplicating these already-proposed candidates:\n- "
                    + "\n- ".join(_idea_distinguishing_text(x)[:1200] for x in previous_ideas[-8:])
                )

    if verbose:
        exp_count = _extract_experiment_count(idea_json)
        print("[Ideation] Done. Experiments: %s" % exp_count, flush=True)

    return {
        "idea": idea_json,
        "novelty": {
            "max_similarity_to_prior": novelty_val,
            "judgement": novelty_judgement,
            "threshold": threshold,
            "passed": (
                not _judged_too_similar(novelty_judgement, novelty_val, threshold)
                and max_candidate_similarity < candidate_similarity_threshold
            ),
            "max_similarity_to_batch": max_candidate_similarity,
            "batch_similarity_threshold": candidate_similarity_threshold,
            "max_retries": retries,
            "method": novelty_method,
            "reason": novelty_reason,
        },
        "experiment_count": {
            "estimated_total": count_val,
            "max_allowed": settings.ideation_max_experiments,
            "passed": (
                count_val is not None
                and 1 <= count_val <= settings.ideation_max_experiments
            ),
        },
        "raw_output": last_raw,
    }


def run_ideation(settings: Settings, research_topic: str, verbose: bool = True) -> Dict[str, Any]:
    prompts = load_prompts(settings.prompts_path)
    ideation_prompts = prompts["IdeationAgent"]

    if verbose:
        print("[Ideation] Starting literature-aware ideation...", flush=True)

    lit_items = _fetch_literature(settings, research_topic, verbose=verbose)
    literature_context = build_literature_context(lit_items)
    llm = create_langchain_llm(model=settings.model, temperature=0.0)

    result = _generate_one_idea(
        llm, ideation_prompts, research_topic, literature_context, lit_items, settings, verbose=verbose
    )
    return {
        "research_topic": research_topic,
        "literature_used": lit_items,
        **result,
    }


def run_ideation_batch(
    settings: Settings,
    research_topic: str,
    num_candidates: int = 6,
    verbose: bool = True,
    literature_items: Optional[List[Dict[str, Any]]] = None,
    require_literature: bool = False,
    case_context: str = "",
) -> Dict[str, Any]:
    """Propose step of the propose -> critique -> rank hypothesis pipeline.

    Fetches literature once, then generates ``num_candidates`` independent,
    novelty-checked ideas against it (same generation + novelty-gate logic as
    :func:`run_ideation`, run ``num_candidates`` times instead of once). Each
    candidate gets a ``candidate_id`` so downstream critique/rank steps can
    refer back to it. See ``src/cfd_langgraph/hypothesis_pipeline.py`` for the
    critique and rank steps that consume this output.
    """
    prompts = load_prompts(settings.prompts_path)
    ideation_prompts = prompts["IdeationAgent"]

    if verbose:
        print(f"[Ideation] Proposing {num_candidates} candidate ideas...", flush=True)

    if literature_items is None:
        lit_items = _fetch_literature(settings, research_topic, verbose=verbose)
    else:
        lit_items = normalize_literature_records(literature_items)
    if require_literature and not lit_items:
        raise ValueError(
            "Literature-grounded hypothesis generation requires a non-empty literature set."
        )
    literature_context = build_literature_context(lit_items)
    # Was 0.55, on the reasoning that candidates should differ from each other
    # and not just from the literature. Now 0.0 like every other call site: a
    # study is only reproducible if its verdicts are, and the measured cost of
    # sampling was a validator that returned 13 issues and then "valid" on
    # identical text. Diversity between candidates comes from the prompt
    # instead -- each generation is shown the ones already proposed in this
    # batch and told to differ from them.
    llm = create_langchain_llm(model=settings.model, temperature=0.0)

    candidates: List[Dict[str, Any]] = []
    prior_ideas: List[Dict[str, Any]] = []
    for i in range(max(1, num_candidates)):
        one = _generate_one_idea(
            llm, ideation_prompts, research_topic, literature_context, lit_items,
            settings, verbose=verbose, previous_ideas=prior_ideas,
            case_context=case_context,
        )
        one["candidate_id"] = f"cand_{i + 1:02d}"
        candidates.append(one)
        if isinstance(one.get("idea"), dict) and not one["idea"].get("parse_error"):
            prior_ideas.append(one["idea"])

    return {
        "research_topic": research_topic,
        "literature_used": lit_items,
        "candidates": candidates,
    }

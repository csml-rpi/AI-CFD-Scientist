from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agents.hypothesis_critique_agent import HypothesisCritiqueAgent
from .agents.hypothesis_rank_agent import HypothesisRankAgent
from .config import Settings
from .ideation import run_ideation_batch
from .prompts.loader import PromptLoader


def run_propose_critique_rank(
    settings: Settings,
    research_topic: str,
    num_candidates: int = 6,
    verbose: bool = True,
    literature_records: Optional[List[Dict[str, Any]]] = None,
    require_literature: bool = True,
    case_context: str = "",
) -> Dict[str, Any]:
    """Propose -> Critique -> Rank: the Mechanism 2 upgrade to hypothesis generation.

    1. Propose  — ``run_ideation_batch`` generates ``num_candidates`` independent,
                  novelty-checked candidate ideas (SciMON-style grounding against
                  the retrieved literature already happens per-candidate here).
    2. Critique — each novelty-surviving candidate is checked for physical
                  plausibility and FoamAgent feasibility (Google DeepMind AI
                  co-scientist-style reflection, arXiv 2502.18864).
    3. Rank     — surviving candidates are compared against each other and
                  ordered strongest-first.

    Returns a dict shaped for the CLI's hypothesis-approval interrupt: every
    candidate is present with its stage-by-stage verdicts, so a rejected idea
    is visible (and overridable) rather than silently dropped.
    """
    overlay_path = None
    knowledge_bundle_dir = getattr(settings, "knowledge_bundle_dir", None)
    if knowledge_bundle_dir:
        overlay_path = knowledge_bundle_dir / "active_prompts.yaml"
    prompt_loader = PromptLoader(settings.prompts_path, overlay_path=overlay_path)

    batch = run_ideation_batch(
        settings,
        research_topic,
        num_candidates=num_candidates,
        verbose=verbose,
        literature_items=literature_records,
        require_literature=require_literature,
        case_context=case_context,
    )
    lit_items = batch["literature_used"]
    candidates = batch["candidates"]

    critique_agent = HypothesisCritiqueAgent(settings.model, prompt_loader)
    rank_agent = HypothesisRankAgent(settings.model, prompt_loader)

    survivors = []
    rejected = []
    for c in candidates:
        novelty_passed = c.get("novelty", {}).get("passed") is True
        experiment_count_passed = c.get("experiment_count", {}).get("passed") is True
        if not novelty_passed or not experiment_count_passed:
            issues = []
            if not novelty_passed:
                issues.append("Failed novelty gate before critique ran.")
            if not experiment_count_passed:
                issues.append("Candidate must contain between one and the configured maximum experiments.")
            c["critique"] = {"verdict": "reject", "issues": issues}
            rejected.append(c)
            continue
        if verbose:
            print(f"[Hypothesis] Critiquing {c['candidate_id']}...", flush=True)
        critique = critique_agent.critique(
            c.get("idea", {}),
            lit_items,
            research_topic=research_topic,
            case_context=case_context,
        )
        c["critique"] = critique
        if critique.get("verdict") == "pass":
            survivors.append(c)
        else:
            rejected.append(c)

    if verbose:
        print(f"[Hypothesis] {len(survivors)}/{len(candidates)} candidates passed critique. Ranking...", flush=True)

    ranked = rank_agent.rank(survivors, research_topic=research_topic) if survivors else []

    return {
        "research_topic": research_topic,
        "literature_used": lit_items,
        "literature_count": len(lit_items),
        "literature_grounded": bool(lit_items),
        "num_proposed": len(candidates),
        "num_passed_critique": len(survivors),
        "ranked_hypotheses": ranked,
        "rejected": rejected,
        # Filled in by the CLI's approve/edit/reject loop before this study proceeds.
        "human_review": {
            "status": "pending",
            "decision": None,
            "notes": None,
        },
    }

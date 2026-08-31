from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from cfd_langgraph.knowledge_bundle import KnowledgeBundle
from cfd_langgraph.llm.factory import create_langchain_llm


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


ReplayFn = Callable[[str, Dict[str, Any]], bool]
"""``replay_fn(prompt_text, validation_case_manifest) -> passed``.

What "replay" means is stage-specific (e.g. for the hypothesis critique
prompt: re-run the critique call against the recorded literature/idea inputs
and check the verdict still matches what the audited study actually did).
Building that replay harness per stage is deliberately left to the caller —
it depends on real audited studies existing, and at the time this module
ships there are none yet. See ``knowledge_bundle/`` in the design doc.
"""


class PromptEvolver:
    """OPRO-style: propose one small prompt tweak, test it, keep it only if it wins.

    Gated on :meth:`KnowledgeBundle.is_bootstrapped` — with fewer than
    ``min_validation_studies`` audited studies on record, there is nothing
    honest to test a tweak against, so :meth:`propose_variant` is a no-op.
    There is no pre-seeded validation corpus; a from-scratch installation
    earns its way into self-evolution one completed, audited study at a time.

    Every proposal is recorded (proposed/promoted/rejected) via
    ``KnowledgeBundle.save_variant`` — nothing is silently discarded, and a
    promoted variant only ever lands in the ``active_prompts.yaml`` overlay,
    never in ``prompts/prompts.yaml`` itself.
    """

    def __init__(
        self,
        model: str,
        bundle: KnowledgeBundle,
        *,
        min_validation_studies: int = 3,
    ):
        self.bundle = bundle
        self.min_validation_studies = min_validation_studies
        self.llm = create_langchain_llm(model=model, temperature=0.4)

    def is_active(self) -> bool:
        return self.bundle.is_bootstrapped(self.min_validation_studies)

    def propose_variant(
        self, stage: str, prompt_key: str, current_prompt: str
    ) -> Optional[Dict[str, Any]]:
        if not self.is_active():
            return None

        lessons_entries = self.bundle.recent_lessons(n=20)
        lessons_text = "\n".join(
            f"- {note}" for entry in lessons_entries for note in entry.get("lessons", [])
        )

        system = (
            "You improve a single prompt used inside a CFD research-automation pipeline. "
            "Propose ONE small, targeted rewording that addresses a real, recurring problem "
            "visible in the lessons below. Do not change what the prompt asks for "
            "structurally (its inputs, its output format) — only how clearly and robustly "
            "it asks for it. Return ONLY the full replacement prompt text, nothing else."
        )
        user = (
            f"Stage: {stage} / {prompt_key}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Recent lessons from completed, audited studies:\n{lessons_text or '(none yet)'}\n\n"
            "Propose the improved prompt."
        )
        candidate = self.llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        ).content.strip()

        if not candidate or candidate == current_prompt:
            return None

        variant: Dict[str, Any] = {
            "stage": stage,
            "prompt_key": prompt_key,
            "baseline_prompt": current_prompt,
            "candidate_prompt": candidate,
            "proposed_at": _now_iso(),
            "status": "proposed",
        }
        self.bundle.save_variant(variant)
        return variant

    def evaluate_and_promote(
        self, variant: Dict[str, Any], replay_fn: ReplayFn
    ) -> Dict[str, Any]:
        """Score baseline vs. candidate over every recorded validation case.

        A variant is promoted only if it matches or beats the baseline on
        every case and wins at least one — anything else is rejected, and
        the next study never sees it. This is the PSV-style "hard,
        unforgeable check gates the change" rule from the design doc, using
        the audit-backed validation suite as the check.
        """
        cases = self.bundle.list_validation_cases()
        base_pass = sum(1 for c in cases if replay_fn(variant["baseline_prompt"], c))
        cand_pass = sum(1 for c in cases if replay_fn(variant["candidate_prompt"], c))

        variant["baseline_score"] = base_pass
        variant["candidate_score"] = cand_pass
        variant["n_validation_cases"] = len(cases)
        variant["evaluated_at"] = _now_iso()

        if cases and cand_pass >= base_pass and cand_pass > 0:
            variant["status"] = "promoted"
            self.bundle.promote_variant(variant)
        else:
            variant["status"] = "rejected"

        self.bundle.save_variant(variant)
        return variant

    def promotable_skills(self, out_dir_lessons: List[str]) -> List[str]:
        """Lightweight heuristic pointer, not an automatic promotion.

        Flags lesson lines that read like a reusable recipe (mesh tuning,
        solver settings) so a human — or a later, more careful skill-mining
        pass — can decide whether to call ``KnowledgeBundle.promote_skill``.
        Intentionally does not promote anything itself: skills are code the
        pipeline will reuse, and that's too consequential to auto-promote
        from a text heuristic alone.
        """
        keywords = ("adjustTimeStep", "maxCo", "mesh", "customModels", "solver")
        return [line for line in out_dir_lessons if any(k in line for k in keywords)]

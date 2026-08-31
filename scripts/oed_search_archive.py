#!/usr/bin/env python3
"""
Family-niched quality-diversity search archive for open_ended_discovery.py.

Replaces the old flat single-chain history + fixed-period "diversity mode"
nudge (scripts/oed_extensions.py: decide_search_mode/render_diversity_constraint)
with:

  - an explicit archive of the best-known variant per model family (the niche
    descriptor comes from oed_extensions.classify_family, reused as-is — it's
    already a cheap, discrete categorical classifier, so no CVT/clustering is
    needed to define niches);
  - a UCB-style selection rule over that archive (PUCT-flavored: exploit a
    niche that's scoring well, but don't starve one that's under-visited; an
    explicit "try a brand-new family" option competes on equal footing via a
    neutral prior so it can win once every known niche is well-explored);
  - an archive-wide saturation check, for a real plateau-based stopping
    signal instead of only "LLM says stop" or "budget exhausted".

Deliberately NOT included (see the design rationale in
scripts/open_ended_discovery.py's history): separate islands with migration,
and CVT-MAP-Elites continuous-space clustering. Both exist in the literature
this borrows from (CodeEvolve/FunSearch/AlphaEvolve) to extract value from
hundreds-to-thousands of cheap evaluations; a real CFD study runs on the
order of 10-30 total evaluations, so that machinery has no population large
enough to pay for itself here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_NEW_FAMILY = None  # sentinel: `family=None` in a selection result means "propose a new one"


def _classify_family_safe(
    model_description: str, model_class: str = "", *, use_llm: bool = False
) -> Tuple[str, str]:
    """Lazily import oed_extensions.classify_family, degrading to a single
    'unknown' bucket if that import ever fails — the archive still functions
    (as one big niche) rather than crashing the whole discovery run over a
    sys.path issue in an unrelated module.

    ``use_llm`` defaults to False because this is reached from ``replay()``,
    which runs over the entire history on every proposal call. Asking a model
    there would make resuming cost one call per unlabelled entry AND make the
    archive non-reproducible — the same history could rebuild into different
    niches on different runs. Live classification of a NEW proposal passes
    use_llm=True; replay stays deterministic and free."""
    try:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import oed_extensions  # type: ignore

        return oed_extensions.classify_family(
            model_description, model_class, use_llm=use_llm
        )
    except Exception:
        return "unknown", "unknown"


# Coarse strategy buckets. Deliberately few: the archive niches on the
# (mechanism, strategy) pair, so the niche count is the product of the two.
# Free-text strategy labels would give every candidate its own niche, which is
# pure exploration with no exploitation left — the failure mode quality-
# diversity search exists to avoid. Four buckets keep the second dimension
# informative while leaving each niche enough visits to develop an elite.
STRATEGIES: Tuple[str, ...] = (
    "analytic",       # a model form reasoned from physics, coefficients chosen by hand
    "sweep",          # reuse an already-compiled model, vary its coefficients
    "solver_fit",     # coefficients fitted by optimising the SCORED objective through the solver
    "offline_fit",    # a model fitted to stored high-fidelity data before it ever meets the solver
)
_DEFAULT_STRATEGY = "analytic"

# What each strategy actually asks the candidate agent to DO. The proposer is
# handed the line for whichever strategy the search assigned, because "propose
# an offline_fit" is not actionable on its own -- a model told only the label
# writes an analytic plan and calls it a fit, which is what happened to
# sst_a1_limiter_025 and nonlinear_stress_crot_neg010_cstrain_neg010 on run
# closure_20260826_codex.
STRATEGY_GUIDANCE: Dict[str, str] = {
    "analytic": (
        "derive the form and choose every coefficient by reasoning from physics, "
        "then compile once and evaluate"
    ),
    "sweep": (
        "reuse an already-compiled model and vary its runtime coefficients across "
        "runs, keeping the best -- no recompile"
    ),
    "solver_fit": (
        "choose the coefficients by RUNNING the solver inside an optimisation loop "
        "on a subset of cases and optimising the scored objective (a-posteriori / "
        "solver-in-the-loop), then evaluate the fitted model on the full set. The "
        "plan must name the optimiser and the subset"
    ),
    "offline_fit": (
        "fit the correction to the stored high-fidelity fields BEFORE it meets the "
        "solver -- regression, symbolic regression, or a small network mapping local "
        "invariants to the correction -- then compile the fitted form and evaluate. "
        "The plan must name which stored fields are the target and what is fitted"
    ),
}

_STRATEGY_HINTS: Dict[str, Tuple[str, ...]] = {
    "solver_fit": (
        "solver-in-the-loop", "solver in the loop", "a-posteriori", "a posteriori",
        "propagated", "through the solver", "model-consistent", "model consistent",
        "cfd-driven", "cfd driven", "differential evolution", "cma-es", "nelder",
        "optimise the score", "optimize the score",
    ),
    "offline_fit": (
        "offline", "regress", "regression", "a-priori", "a priori", "frozen",
        "field inversion", "symbolic regression", "neural", "random forest",
        "gradient boost", "sklearn", "torch", "train on", "fitted to dns",
        "fit to dns", "fit to les",
    ),
    "sweep": (
        "sweep", "coefficient scan", "vary the coefficient", "retune", "recalibrat",
        "parameter study", "reuse the compiled", "already-compiled", "already compiled",
    ),
    "analytic": (
        "analytic", "derive", "closed form", "closed-form", "hand-chosen",
        "physically motivated",
    ),
}


def classify_strategy(text: str) -> str:
    """Map a free-text strategy description onto one of ``STRATEGIES``.

    The proposer is free to describe *how* it intends to find a modification in
    its own words; this bins that description so the archive has a small,
    comparable second dimension. Substring matching only — no model call — so
    replaying a history is deterministic and free, exactly as family
    classification is during replay.

    Order matters: a plan that fits through the solver often also mentions
    regression, and the distinction that matters for the search is whether the
    SCORED objective was in the loop. So solver_fit is tested first.
    """
    blob = str(text or "").strip().lower()
    if not blob:
        return _DEFAULT_STRATEGY
    for strategy in ("solver_fit", "offline_fit", "sweep", "analytic"):
        for hint in _STRATEGY_HINTS[strategy]:
            if hint in blob:
                return strategy
    return _DEFAULT_STRATEGY


_FITTING_STRATEGIES = frozenset({"solver_fit", "offline_fit"})


def _llm_classify_strategy_safe(
    plan: str, hypothesis: str = "", declared: str = ""
) -> Optional[str]:
    """``oed_extensions.llm_classify_strategy``, or None if it is unreachable.

    Same lazy-import shape as ``_classify_family_safe``: a sys.path problem in
    an unrelated module must not take the archive down.
    """
    try:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import oed_extensions  # type: ignore

        return oed_extensions.llm_classify_strategy(plan, hypothesis, declared)
    except Exception:
        return None


def normalize_strategy(
    value: Any,
    plan: Any = None,
    *,
    hypothesis: Any = "",
    use_llm: bool = False,
) -> str:
    """A declared strategy, coerced into a known bucket and checked against the plan.

    The declaration is a statement of intent; the plan is what the candidate
    agent actually carries out. When the two disagree, the plan wins — this
    axis exists to measure how much fitting the search has really done, and a
    label taken on trust measures nothing.

    ``use_llm=True`` asks the model to read the plan, which is the only thing
    that can make this judgement correctly, and is what live classification of
    a NEW proposal passes. Two measured failures, both on run
    closure_20260826_codex, define the requirement:

      * A candidate declared ``solver_fit`` while its plan read "compile it
        once, then run and score exactly all 32 supplied cases" — no optimiser
        and no reference to the stored high-fidelity fields.
      * Candidate ``sst_a1_limiter_025`` declared ``solver_fit`` while its plan
        read "...may be read offline only for sign/branch diagnostics ...
        Implement the fixed a1=0.25 runtime closure exactly as specified, with
        no fitted or case-dependent parameters." A hand-chosen constant.

    The keyword table catches the first and cannot catch the second: the plan
    contains the word "offline", so ``classify_strategy`` returns
    ``offline_fit``, which is itself a fitting label, and the false claim
    passes. Whether reading data constitutes fitting depends on what is done
    with it, which no keyword can see.

    ``use_llm`` defaults to False so ``replay()`` stays deterministic and free,
    for the same reason family classification does: replay runs over the whole
    history on every proposal call, and a model call there would make the same
    history rebuild into different niches on different runs. Replayed entries
    carry a strategy already decided when they were proposed.

    Without the model, the keyword table is the fallback and only fit-claims
    are downgraded. Nothing is ever upgraded by keywords: a candidate that
    quietly fits while calling itself analytic understates rather than
    inflates, and guessing intent from prose in that direction would be worse
    than the error it fixes. The model, having read the plan, may correct in
    either direction.
    """
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    declared = text if text in STRATEGIES else classify_strategy(value)

    plan_text = str(plan or "").strip()
    if not plan_text:
        return declared

    if use_llm:
        decided = _llm_classify_strategy_safe(
            plan_text, str(hypothesis or ""), declared
        )
        if decided in STRATEGIES:
            return decided

    if declared in _FITTING_STRATEGIES:
        supported = classify_strategy(plan_text)
        if supported not in _FITTING_STRATEGIES:
            return supported
    return declared


class SearchArchive:
    """One elite (best-scoring history entry) per model family, plus visit
    counts for UCB-style selection and a chronological best-score trace for
    saturation detection."""

    def __init__(
        self,
        exploration_c: float = 0.3,
        stale_halflife: int = 3,
        exploration_floor: int = 8,
        strategy_transfer: float = 1.0,
        strategy_prior_visits: int = 0,
    ) -> None:
        # q is min-max normalized into [0, 1] (see select_niche), so
        # exploration_c must be calibrated against THAT range, not raw score
        # magnitude. At real study budgets (tens of evaluations, not
        # hundreds), sqrt(ln(total_visits)/1) for a zero-visit "new family"
        # option is already close to 2 by ~30 total visits — with c=1.4 that
        # alone (2.8) swamps any real niche's q (capped at 1.0) regardless
        # of how well-established or excellent it is, so exploitation could
        # never win within a real study's lifetime. 0.3 keeps "try something
        # new" attractive while a niche is thin (a handful of visits) but
        # lets a strong, reasonably-visited niche's q actually win once it's
        # earned it — verified in scripts/test_oed_search_archive.py.
        self.exploration_c = exploration_c
        # Visits-without-improvement at which a family's exploitation value
        # is halved. 3 is deliberately short: by the third consecutive
        # non-improving evaluation a coefficient sweep has already mapped
        # its optimum, and further visits buy nothing.
        self.stale_halflife = stale_halflife
        # Hard floor on exploration: force a brand-new family after this many
        # consecutive evaluations that all went to families already on the
        # board.
        #
        # The PUCT terms alone do not deliver this. Measured on run
        # oed_20260822_1626_codex_high: the "new family" option scores a fixed
        # q=0.5 plus its exploration bonus (0.76-0.92 across the whole run),
        # while any incumbent visited even *once* is min-max normalized to
        # q~0.95, carries no stale penalty yet, and still holds most of its
        # own exploration bonus (1.05-1.28). A single lucky sample therefore
        # outranks "try something genuinely new" permanently, and no setting
        # of exploration_c or stale_halflife closes a structural 0.3 gap.
        # That run explored 16 families in its first 36 evaluations and then
        # zero in its last 25 -- 41% of the budget spent entirely on families
        # already known, while the eventual winner had come from a family
        # first tried at evaluation 24.
        self.exploration_floor = exploration_floor
        # How much of a mechanism's proven score carries over to a strategy
        # never tried on it, in [0, 1]: 0 treats an empty cell as knowing
        # nothing (neutral q, like a brand-new family), 1 treats "this
        # mechanism works" as full evidence that it will work when determined
        # a different way.
        #
        # 1.0 is wrong and was measured to be: an empty cell then carries the
        # family's q AND the maximal zero-visit exploration bonus, so it beats
        # the very niche it inherited from, every time, until visited. The
        # archive stopped being able to exploit a decisively-best niche at all
        # -- two selection tests that assert exactly that began failing.
        # 0.5 keeps empty cells genuinely competitive without letting them
        # pre-empt a proven one, which is the balance this axis needs: a
        # different way of fitting the SAME mechanism is a real lead, not a
        # certainty.
        self.strategy_transfer = strategy_transfer
        # An untried strategy on a PROVEN mechanism is a smaller leap than a
        # brand-new mechanism, so it does not deserve the same maximal
        # zero-visit exploration bonus. Giving it one made empty cells beat
        # every option including a barely-tried rival mechanism -- the
        # strategy axis would have been opened by starving the mechanism axis.
        # A pseudo-count shrinks the bonus without special-casing the formula.
        self.strategy_prior_visits = max(0, int(strategy_prior_visits))
        # Evaluations recorded since a family was last seen for the first
        # time. Rebuilt correctly on resume because replay() feeds update()
        # in chronological order.
        self._visits_since_new_family = 0
        self.niches: Dict[str, Dict[str, Any]] = {}
        # Chronological trace of the archive-wide best normalized score after
        # every real (scored) evaluation — used by is_saturated().
        self._best_trace: List[Optional[float]] = []
        # The one objective direction this archive ranks on; set by the
        # first scored evaluation (see update()).
        self._direction: Optional[str] = None

    @staticmethod
    def classify(
        model_description: str, model_class: str = "", *, use_llm: bool = False
    ) -> str:
        family, _eq = _classify_family_safe(
            model_description, model_class, use_llm=use_llm
        )
        return family

    @staticmethod
    def _normalize(value: float, direction: str) -> float:
        """Internally everything is 'lower is better' so selection/saturation
        logic doesn't need to branch on direction."""
        return value if direction != "max" else -value

    def _new_niche(self) -> Dict[str, Any]:
        return {
            "elite_score": None,
            "elite_norm_score": None,
            "elite_iteration": None,
            "elite_history_entry": None,
            "visits": 0,
            # Visits to this family since its elite last improved. A family
            # whose sweep is mined out keeps scoring well without learning
            # anything new, and PUCT alone cannot tell those apart — see
            # `stale_penalty` in select_niche.
            "stale_visits": 0,
        }

    def replay(self, history: List[Dict[str, Any]], baseline_direction: str = "min") -> None:
        """Rebuild archive state from a resumed history.json — so resuming a
        paused/crashed run doesn't reset exploration progress to empty."""
        for h in history:
            if not isinstance(h, dict):
                continue
            if h.get("action_type") not in ("code_mod", "experiment"):
                continue
            family = h.get("family") or self.classify(
                h.get("model_description", "") or h.get("compiled_model_description", ""),
                h.get("compiled_model_name", ""),
            )
            score = h.get("score")
            direction = baseline_direction
            if isinstance(score, dict):
                d = str(score.get("direction", "")).strip().lower()
                if d in ("min", "max"):
                    direction = d
            try:
                iteration = int(h.get("iteration", 0) or 0)
            except (TypeError, ValueError):
                # A malformed iteration must not abort a resume — the entry is
                # still a real visit and still carries an elite score.
                iteration = 0
            self.update(
                family, iteration, score, direction, h,
                strategy=h.get("strategy") or h.get("strategy_label") or "",
            )

    def update(
        self,
        family: str,
        iteration: int,
        score: Optional[Dict[str, Any]],
        direction: str,
        history_entry: Dict[str, Any],
        strategy: str = "",
    ) -> None:
        """Record one real evaluation against its family's niche. Call this
        once per code_mod/experiment iteration, even if score is None (a
        failed attempt still counts as a visit, which is what keeps the
        selection policy from hammering a family that keeps failing)."""
        # A candidate whose modification provably changed nothing (see the
        # bit-exact baseline check in oed_score_candidate) is not evidence
        # about its family: it never tested the family at all. Counting it
        # would both consume a visit and record a baseline-equal score,
        # which is exactly how run oed_20260822_1626_codex_high wrote off
        # the reverse-flow diffusivity family on a single no-op.
        if isinstance(history_entry, dict) and history_entry.get("no_op"):
            return

        key = (family, normalize_strategy(strategy))
        if key[0] not in {k[0] for k in self.niches}:
            # A mechanism never seen before resets the exploration floor. The
            # floor exists to stop the search circling known PHYSICS; trying a
            # third strategy on a mechanism already in the archive is not the
            # new ground it is meant to force.
            self._visits_since_new_family = 0
        else:
            self._visits_since_new_family += 1
        if key not in self.niches:
            self.niches[key] = self._new_niche()
        niche = self.niches[key]
        niche["visits"] += 1

        # Every niche must be on one comparable scale. _normalize negates for
        # "max", so an archive holding both a min-direction family (0.001) and
        # a max-direction one (0.95 -> -0.95) would rank the max family as
        # better by pure sign. A study has exactly one objective direction;
        # entries claiming otherwise are a bug upstream, not a signal to fold
        # into the ranking.
        if self._direction is None:
            self._direction = direction
        elif direction != self._direction:
            score = None

        norm: Optional[float] = None
        val: Optional[float] = None
        if isinstance(score, dict):
            try:
                val = float(score.get("value"))
                norm = self._normalize(val, direction) if math.isfinite(val) else None
            except Exception:
                norm = None

        if norm is not None and (niche["elite_norm_score"] is None or norm < niche["elite_norm_score"]):
            niche["elite_score"] = val
            niche["elite_norm_score"] = norm
            niche["elite_iteration"] = iteration
            niche["elite_history_entry"] = history_entry
            niche["stale_visits"] = 0
        else:
            niche["stale_visits"] += 1

        # Failed/unscored attempts still consume a visit and therefore reduce
        # this family's exploration bonus, but they are not score samples and
        # must not advance the score-saturation window.
        if norm is not None:
            overall_best = min(
                (n["elite_norm_score"] for n in self.niches.values() if n["elite_norm_score"] is not None),
                default=None,
            )
            self._best_trace.append(overall_best)

    def families(self) -> Dict[str, Dict[str, Any]]:
        """A mechanism-level view of the archive, strategies merged.

        The archive niches on (mechanism, strategy) so the search can learn
        which strategies pay off, but plenty of callers only care about the
        physics: "which mechanisms have been tried", "what is this mechanism's
        best result". Merging here keeps that question cheap to ask without
        those callers having to know the key is a pair.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for (family, _strategy), niche in self.niches.items():
            row = merged.get(family)
            if row is None:
                merged[family] = dict(niche)
                continue
            row["visits"] += niche["visits"]
            norm = niche["elite_norm_score"]
            if norm is not None and (
                row["elite_norm_score"] is None or norm < row["elite_norm_score"]
            ):
                for field in (
                    "elite_score", "elite_norm_score", "elite_iteration", "elite_history_entry",
                ):
                    row[field] = niche[field]
        return merged

    def niche_for(self, family: str, strategy: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """One niche by mechanism, optionally pinned to a strategy.

        With no strategy, returns that mechanism's best-scoring niche — the
        answer to "how well has this mechanism done", regardless of how it
        was reached.
        """
        if strategy is not None:
            return self.niches.get((family, normalize_strategy(strategy)))
        rows = [n for (f, _s), n in self.niches.items() if f == family]
        if not rows:
            return None
        scored = [n for n in rows if n["elite_norm_score"] is not None]
        if not scored:
            return rows[0]
        return min(scored, key=lambda n: n["elite_norm_score"])

    def select_niche(
        self, budget_remaining: int, budget_total: int, force_new_family: bool = False
    ) -> Dict[str, Any]:
        """Pick which niche the next proposal should be conditioned on.

        Returns {"family": str|None, "is_new": bool, "elite": dict|None}.
        `family=None, is_new=True` means "propose something in a family never
        tried yet"; `elite` is that niche's current-best history entry
        (carries formula/case_dir to mutate from), or None for a new family.
        """
        if budget_remaining <= 0:
            return {
                "family": None,
                "strategy": None,
                "is_new": False,
                "elite": None,
                "budget_exhausted": True,
            }
        if force_new_family or not self.niches:
            return {"family": None, "strategy": None, "is_new": True, "elite": None}

        # Exploration floor. Only fires while enough budget remains to
        # actually develop whatever the new family turns up -- opening a new
        # mechanism with two evaluations left just wastes them, which is the
        # legitimate concern the horizon term was reaching for.
        new_family_reserve = max(4, int(0.10 * max(1, budget_total)))
        if (
            self.exploration_floor > 0
            and self._visits_since_new_family >= self.exploration_floor
            and budget_remaining >= new_family_reserve
        ):
            return {
                "family": None,
                "strategy": None,
                "is_new": True,
                "elite": None,
                "forced_by_exploration_floor": True,
            }

        total_visits = sum(n["visits"] for n in self.niches.values()) or 1
        horizon_fraction = min(1.0, max(0.05, budget_remaining / max(1, budget_total)))
        horizon_scale = math.sqrt(horizon_fraction)

        # Q must be on a comparable scale to the exploration term, which is
        # O(1) — but raw scores are arbitrary (a Cf RMSE of 0.001-0.01, an
        # L2 error of 1e-6, whatever the study's metric happens to be). Using
        # the raw (negated) score as q directly made the exploration term
        # dominate unconditionally regardless of visit count — a heavily-
        # visited excellent niche could never outweigh "try something new"
        # because 0.001 vs 0.01 is invisible next to an O(1) bonus. Min-max
        # normalize across the niches actually being compared instead, so
        # q sits in [0, 1] (1 = best niche seen) and can actually compete.
        scored_values = [n["elite_norm_score"] for n in self.niches.values() if n["elite_norm_score"] is not None]
        lo = min(scored_values) if scored_values else 0.0
        hi = max(scored_values) if scored_values else 0.0
        span = hi - lo

        def normalized_q(norm_q: Optional[float]) -> float:
            if norm_q is None or span <= 0:
                # No signal yet, or every scored niche is tied — neutral,
                # same footing as a new, never-tried family.
                return 0.5
            return 1.0 - (norm_q - lo) / span

        def stale_penalty(stale_visits: int) -> float:
            """Damp q for a family that keeps being visited without improving.

            Min-max normalization makes the leading niche's q exactly 1.0 no
            matter how slim its lead, so a family that is marginally ahead
            outranks "try something new" indefinitely. Measured on a real
            run: one family took 21 of 24 evaluations and stopped improving
            after the 5th, while three rival families got one shot each and
            were never revisited.

            Being ahead is not the same as still having something to give.
            Consecutive visits that fail to move the family's own elite are
            the evidence that its seam is worked out, so they progressively
            discount its exploitation value while leaving its elite (and thus
            the reported best result) untouched.
            """
            return 1.0 / (1.0 + stale_visits / float(max(1, self.stale_halflife)))

        def puct_raw(q: float, visits: int) -> float:
            """PUCT for a q that is already on the [0, 1] scale."""
            return q + (
                self.exploration_c
                * horizon_scale
                * math.sqrt(math.log(total_visits + 1) / (1 + visits))
            )

        def puct(norm_q: Optional[float], visits: int, stale_visits: int = 0) -> float:
            q = normalized_q(norm_q) * stale_penalty(stale_visits)
            exploration = (
                self.exploration_c
                * horizon_scale
                * math.sqrt(math.log(total_visits + 1) / (1 + visits))
            )
            return q + exploration

        def transferred_q(family_norm_q: Optional[float]) -> float:
            """q for a strategy never tried on a mechanism that has scored.

            Interpolates between neutral (0.5 -- what a brand-new family gets,
            i.e. "nothing is known") and the mechanism's own q, by
            ``strategy_transfer``. The mechanism working IS evidence about the
            cell; it is not proof that a different way of determining it will
            work.
            """
            weight = min(1.0, max(0.0, float(self.strategy_transfer)))
            return 0.5 + weight * (normalized_q(family_norm_q) - 0.5)

        candidates: List[Tuple[Optional[Tuple[str, str]], float, Optional[Dict[str, Any]]]] = []
        for key, niche in self.niches.items():
            candidates.append(
                (key, puct(niche["elite_norm_score"], niche["visits"], niche.get("stale_visits", 0)), niche)
            )

        # Empty cells of the (mechanism, strategy) grid compete as well.
        #
        # Without this the grid only ever fills one column. Every existing
        # niche on run closure_20260826_codex was (something, "analytic"),
        # because that is all that had been scored, so an untried strategy was
        # never in the candidate set and PUCT could not give it an exploration
        # bonus. The prompt told the proposer "a strategy earns budget by
        # scoring" while the search made it impossible for a strategy to be
        # selected and therefore to ever score -- a closed loop that 20
        # launched candidates never escaped. Filling empty cells is the whole
        # point of a quality-diversity archive; this one was only doing it
        # along one axis.
        #
        # q is inherited from the mechanism's own best result rather than the
        # neutral 0.5, because that is what is actually known: the mechanism
        # has evidence, the way of determining it does not. So a strong family
        # gets its untried strategies explored before a weak family's, and the
        # exploration bonus is the ordinary zero-visit one -- no new constant.
        #
        # Only for families with a scored elite. Offering a novel strategy on
        # a mechanism that has never worked is two unknowns at once, and the
        # result teaches nothing about either.
        # Only while there is budget left to develop what a new strategy turns
        # up -- the same reserve the new-family option uses, for the same
        # reason, and it is what keeps the horizon honest.
        #
        # This gate replaces damping the cell's score, which was measured to be
        # unworkable: damping strong enough to let a proven niche win at the
        # END of a budget was also strong enough to stop empty cells EVER
        # winning at the start, because early on every niche has one visit and
        # so the same exploration bonus, leaving only q to separate them. On
        # the real archive of run closure_20260826_codex that produced twelve
        # empty cells and zero selections of any of them. The two regimes
        # cannot both be served by one constant; only a knife-edge value
        # satisfied both, which is not a calibration, it is a coincidence.
        offer_empty_cells = budget_remaining >= new_family_reserve
        scored_families: Dict[str, Optional[float]] = {}
        for (family, _strategy), niche in self.niches.items():
            if niche["elite_norm_score"] is None:
                continue
            best = scored_families.get(family)
            if best is None or niche["elite_norm_score"] < best:
                scored_families[family] = niche["elite_norm_score"]
        for family, family_q in (scored_families.items() if offer_empty_cells else ()):
            transferred = transferred_q(family_q)
            for strategy in STRATEGIES:
                if (family, strategy) in self.niches:
                    continue
                candidates.append(
                    ((family, strategy),
                     puct_raw(transferred, self.strategy_prior_visits),
                     None)
                )

        # "Propose a new family" always competes too, with a neutral q=0.5
        # (same as an unscored niche) and zero visits (maximal exploration
        # bonus).
        candidates.append((None, puct(None, 0), None))

        best_key, _best_score, best_niche = max(candidates, key=lambda c: c[1])
        if best_key is None:
            return {"family": None, "strategy": None, "is_new": True, "elite": None}
        if best_key not in self.niches:
            # An empty cell: the mechanism is known, this way of arriving at it
            # is not. No elite to mutate from -- the elite of the mechanism's
            # other strategies describes a model built a different way, and
            # handing it over invites the proposer to re-describe that model
            # and relabel it.
            return {
                "family": best_key[0],
                "strategy": best_key[1],
                "is_new": False,
                "is_new_strategy": True,
                "elite": None,
            }
        return {
            "family": best_key[0],
            # The strategy that produced this niche's elite is a suggestion,
            # not a constraint: the proposer may judge that a mechanism which
            # scored well under one strategy is worth revisiting under
            # another. Reporting it is what lets the search LEARN which
            # strategies pay off, because the elite of every (mechanism,
            # strategy) pair is tracked separately.
            "strategy": best_key[1],
            "is_new": False,
            "is_new_strategy": False,
            "elite": (best_niche or {}).get("elite_history_entry"),
        }

    def is_saturated(self, window: int) -> bool:
        """True if the archive-wide best (normalized) score hasn't improved
        over the last `window` real evaluations. Requires at least
        `window + 1` scored evaluations to have a baseline to compare against."""
        if window <= 0 or len(self._best_trace) < window + 1:
            return False
        before = self._best_trace[-window - 1]
        cur = self._best_trace[-1]
        if before is None or cur is None:
            return False
        # Relative epsilon, not exact equality — re-running the same family
        # can jitter by noise-level amounts without that counting as
        # "improvement". Relative because the metric's magnitude is arbitrary:
        # a fixed 1e-9 is a sane noise floor for a Cf RMSE around 1e-3, but a
        # 0.1% threshold for an L2 error around 1e-6, which would call real
        # sub-0.1% progress a plateau and stop the search early.
        tolerance = max(1e-12, abs(before) * 1e-6)
        return not (before - cur > tolerance)

    def render_summary(
        self, baseline_score: Optional[float] = None, baseline_direction: str = "min"
    ) -> str:
        if not self.niches:
            return "SEARCH ARCHIVE: empty — no model family has been scored yet."
        lines = ["SEARCH ARCHIVE (best known variant per model family tried so far):"]

        def sort_key(item: Tuple[Tuple[str, str], Dict[str, Any]]) -> Tuple[bool, float]:
            _key, niche = item
            s = niche["elite_norm_score"]
            return (s is None, s if s is not None else 0.0)

        for key, niche in sorted(self.niches.items(), key=sort_key):
            family, strategy = key
            score_str = f"{niche['elite_score']:.6g}" if niche["elite_score"] is not None else "n/a"
            delta_str = ""
            if niche["elite_score"] is not None and baseline_score is not None:
                delta = niche["elite_score"] - baseline_score
                beats = delta < 0 if baseline_direction != "max" else delta > 0
                delta_str = f", Δ vs baseline={delta:+.4g} ({'beats' if beats else 'behind'})"
            lines.append(
                f"  - {family} [via {strategy}]: best score={score_str}{delta_str}, "
                f"visits={niche['visits']}, from iteration {niche['elite_iteration']}"
            )
        lines.append("")
        lines.append(self.render_strategy_summary(baseline_score, baseline_direction))
        return "\n".join(lines)

    def render_strategy_summary(
        self, baseline_score: Optional[float] = None, baseline_direction: str = "min"
    ) -> str:
        """What each STRATEGY has actually bought, aggregated over mechanisms.

        This is the point of the second archive dimension. The proposer is not
        told which strategy to use; it is shown what the strategies have
        returned so far and left to draw the conclusion. A strategy that keeps
        producing the best elite across several mechanisms earns more of the
        budget because the elites say so, not because a prompt asserted it.
        """
        if not self.niches:
            return "STRATEGY SCOREBOARD: empty."
        rolled: Dict[str, Dict[str, Any]] = {}
        for (_family, strategy), niche in self.niches.items():
            row = rolled.setdefault(strategy, {"visits": 0, "best": None, "mechanisms": 0})
            row["visits"] += niche["visits"]
            row["mechanisms"] += 1
            norm = niche["elite_norm_score"]
            if norm is not None and (row["best"] is None or norm < row["best"][0]):
                row["best"] = (norm, niche["elite_score"])
        lines = [
            "STRATEGY SCOREBOARD (what each way of producing a candidate has returned):"
        ]
        for strategy, row in sorted(
            rolled.items(), key=lambda kv: (kv[1]["best"] is None, kv[1]["best"][0] if kv[1]["best"] else 0.0)
        ):
            if row["best"] is None:
                best_str = "no scored result yet"
            else:
                best_str = f"best score={row['best'][1]:.6g}"
                if baseline_score is not None:
                    delta = row["best"][1] - baseline_score
                    beats = delta < 0 if baseline_direction != "max" else delta > 0
                    best_str += f" ({'beats' if beats else 'behind'} baseline by {abs(delta):.4g})"
            lines.append(
                f"  - {strategy}: {best_str}, {row['visits']} evaluation(s) "
                f"across {row['mechanisms']} mechanism(s)"
            )
        untried = [x for x in STRATEGIES if x not in rolled]
        if untried:
            lines.append(f"  - never tried: {', '.join(untried)}")
        return "\n".join(lines)

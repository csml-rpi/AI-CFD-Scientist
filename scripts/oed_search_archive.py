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
        population_size: int = 3,
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
        # How many scored candidates each cell keeps, best-first. Two or three
        # holds "the tuned version" and "the structurally new but untuned one"
        # at once -- the pair strict elitism collapses into a single winner.
        self.population_size = max(1, int(population_size))
        # iteration -> history entry, so a chain can be walked back through
        # parent_iteration without rescanning history on every call.
        self._by_iteration: Dict[int, Dict[str, Any]] = {}
        # Arm statistics for select_action. Start at the prior (0/0 -> Beta(1,1),
        # a coin flip) so no action is favoured before there is evidence.
        self._widen_wins = 0
        self._widen_losses = 0
        self._newfam_wins = 0
        self._newfam_losses = 0
        self._deepen_wins = 0
        self._deepen_losses = 0
        # Pseudo-counts for a lineage's prior, from its score rank in the
        # archive, and the weight given to observed gain magnitude. The prior
        # has to be strong enough that a well-scoring root outranks a poor one
        # before any refinement, and weak enough that two real gains overturn
        # it.
        # Chain bookkeeping kept OUTSIDE the per-cell population, which is
        # capped. The cap exists to bound how many candidate artifacts an
        # archive holds; it must not silently bound how much of a chain's
        # HISTORY the allocator can see. A six-step chain in one cell was
        # reporting a three-point trace, so its momentum evidence was clipped
        # to the cap and the trace shown to the proposer was wrong.
        # These are a few floats per candidate, so there is no reason to cap.
        self._chain_scores: Dict[int, List[Dict[str, Any]]] = {}
        # Attempts that produced no score at all. A chain whose last two builds
        # failed to compile is not as promising as one that did not fail, and
        # without this it looked identical: same depth, same trace, same gain.
        self._chain_failures: Dict[int, int] = {}
        # A scoreless attempt weighs like one failed trial. Calibrated against
        # the gain scale: at _gain_weight 60 a typical 1.7% refinement earns
        # about one pseudo-count, so one failure costing one is symmetric.
        self._failure_weight = 1.0
        self._lineage_prior = 3.0
        # Applied to RELATIVE gains, so a chain improving ~1.7% per step earns
        # about one pseudo-count per step -- enough to compete with the score
        # prior after two good steps, not enough to override it on one.
        self._gain_weight = 60.0
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

    # -- lineage bookkeeping -------------------------------------------------
    #
    # A "lineage" is a chain of refinements: a root candidate and everything
    # descended from it by mutation. The archive needs these because the
    # allocation question -- refine what we have, or start something new -- is
    # a question about chains, not about individual candidates.
    #
    # Reconstructed from `parent_iteration`, which the proposer already records
    # whenever a pick carried an elite to mutate from. Entries with no parent
    # are roots of their own lineage.

    def _lineage_id(self, history_entry: Dict[str, Any], iteration: int) -> int:
        """Which chain this candidate belongs to, as the iteration of its root."""
        if not isinstance(history_entry, dict):
            return iteration
        seen = set()
        cur, cur_iter = history_entry, iteration
        while True:
            parent = cur.get("parent_iteration")
            if parent is None or parent in seen:
                return cur_iter
            seen.add(parent)
            nxt = self._by_iteration.get(parent)
            if nxt is None:
                # Parent not in the archive (a failed or unscored ancestor).
                # The chain is still real; root it at the highest ancestor we
                # can actually see rather than dropping the relationship.
                return parent
            cur, cur_iter = nxt, parent

    def _depth_of(self, history_entry: Dict[str, Any]) -> int:
        """How many refinement steps from the root. A root is depth 0."""
        if not isinstance(history_entry, dict):
            return 0
        depth, seen = 0, set()
        cur = history_entry
        while True:
            parent = cur.get("parent_iteration")
            if parent is None or parent in seen:
                return depth
            seen.add(parent)
            depth += 1
            nxt = self._by_iteration.get(parent)
            if nxt is None:
                return depth
            cur = nxt

    def lineages(self) -> Dict[int, Dict[str, Any]]:
        """Every chain in the archive, with the trace the allocator needs.

        A lineage's value is not its best score but whether it is still
        *moving*: a chain improving 1.7% per step is worth another pull even
        while it trails a flat chain that is already better. That is the case
        strict elitism cannot express, and it is why this returns the score
        trace rather than a single number.
        """
        # Artifacts come from the (capped) per-cell populations, because the
        # tip has to carry a history_entry the proposer can mutate from. The
        # SCORE HISTORY comes from the uncapped chain record, so a long chain
        # reports all of its steps rather than the last `population_size`.
        # MEMBERSHIP comes from the uncapped chain record, not from the capped
        # populations. Deriving it from the populations deleted whole lineages:
        # a chain whose members all rank below the top `population_size` of
        # their cell vanished from the allocator completely -- not truncated,
        # gone -- while _chain_scores still held it, so nothing looked broken.
        #
        # Reproduced: a cell holding three flat candidates at 0.090/0.091/0.092
        # plus a chain running 0.20 -> 0.15 -> 0.12 (+25% per step) offered the
        # allocator only [1, 2, 3]. The fastest-improving chain in the archive
        # was unreachable forever -- which is precisely the case this method
        # exists to serve. Invisible on a wide archive where most cells are
        # visited once; certain on the deep focused study the design is for.
        chains: Dict[int, List[Dict[str, Any]]] = {}
        for niche in self.niches.values():
            for member in niche.get("population", []):
                chains.setdefault(member["lineage_id"], []).append(member)
        # Any chain with scores but no surviving population member is still a
        # real lineage. Rebuild a minimal member for it so it can be selected;
        # its history_entry comes from the iteration index.
        for root, scored in self._chain_scores.items():
            if root in chains or not scored:
                continue
            best = min(scored, key=lambda m: m["norm_score"])
            entry = self._by_iteration.get(best["iteration"]) or {}
            chains[root] = [{
                "norm_score": best["norm_score"],
                "score": best["score"],
                "iteration": best["iteration"],
                "history_entry": entry,
                "lineage_id": root,
                "depth": self._depth_of(entry),
            }]
        out: Dict[int, Dict[str, Any]] = {}
        for root, members in chains.items():
            members = sorted(members, key=lambda m: m["iteration"])
            # The cap never drops the best member (the population is sorted
            # best-first before truncation), so the tip is always present.
            tip = min(members, key=lambda m: m["norm_score"])
            history = sorted(
                self._chain_scores.get(root, []), key=lambda m: m["iteration"]
            ) or members
            trace = [m["score"] for m in history]
            last_gain = 0.0
            if len(history) >= 2:
                last_gain = history[-2]["norm_score"] - history[-1]["norm_score"]
            out[root] = {
                "root_iteration": root,
                "members": members,
                "history": history,
                "score_trace": trace,
                "failures": self._chain_failures.get(root, 0),
                "depth": max(m["depth"] for m in members),
                "tip": tip,
                "best_norm_score": tip["norm_score"],
                "last_gain": last_gain,
            }
        return out

    def _new_niche(self) -> Dict[str, Any]:
        return {
            "elite_score": None,
            "elite_norm_score": None,
            "elite_iteration": None,
            "elite_history_entry": None,
            # Runners-up, best-first, capped at `population_size`. The elite
            # fields above stay as they were -- everything reading this archive
            # expects them -- and this is strictly additional.
            #
            # Why keep losers at all: a structurally new variant arrives with
            # untuned coefficients. Under strict elitism it is compared once
            # against a cell whose elite has already been tuned, loses by a
            # little, and is discarded with its compiled model and its case
            # directory. There is then nothing to tune next round. Measured on
            # run closure_20260826_codex, every one of the 5 refinements that
            # did happen beat its parent (median 0.1089 vs 0.1135 for fresh
            # starts, 5/5 vs 28/39 beating baseline) -- refinement is the move
            # that works, and it had 7% of the budget.
            "population": [],
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
        prior_best: Optional[float] = None
        # Index everything up front. History is not guaranteed ordered, and an
        # entry may name a parent that appears later in the list; walking a
        # chain must not depend on the order entries happen to be written in.
        for h in history:
            if isinstance(h, dict) and h.get("iteration") is not None:
                try:
                    self._by_iteration[int(h["iteration"])] = h
                except (TypeError, ValueError):
                    pass
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
            # Replay the allocator's arms too, or a resumed study forgets which
            # kind of move has been paying off and restarts from the priors.
            # "Improved" means it beat the best score the archive held BEFORE
            # this entry -- the question the arm is actually being asked.
            action = str(h.get("search_action") or "").strip()
            if action:
                value = None
                if isinstance(score, dict):
                    try:
                        value = float(score.get("value"))
                    except (TypeError, ValueError):
                        value = None
                if value is None:
                    # A candidate that produced no score is a loss for the arm
                    # that chose it. Skipping these taught the arms nothing
                    # from 11 of 55 candidates on the real archive -- 581 of
                    # 2770 solver runs, 21% of the budget -- so an arm that
                    # reliably yields uncompilable candidates was never charged
                    # for it and kept getting bought.
                    self.record_action_outcome(action, False)
                else:
                    norm = self._normalize(value, direction)
                    # An empty archive has no incumbent to beat, so the first
                    # scored candidate would otherwise always register a win
                    # against inf. It is a baseline, not evidence about the arm.
                    if prior_best is not None:
                        self.record_action_outcome(action, norm < prior_best)
            prior_best = min(
                (n["elite_norm_score"] for n in self.niches.values()
                 if n["elite_norm_score"] is not None),
                default=prior_best,
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

        # Indexed before any lineage walk, so a chain can be followed back
        # through parent_iteration. Without this every candidate roots at its
        # own parent and depth never exceeds 1.
        if isinstance(history_entry, dict):
            self._by_iteration[iteration] = history_entry

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

        # Chain bookkeeping, before the elite/population logic, and for failed
        # candidates too -- a failure is evidence about the chain even though
        # it can never be an elite.
        if isinstance(history_entry, dict):
            chain = self._lineage_id(history_entry, iteration)
            if norm is None:
                self._chain_failures[chain] = self._chain_failures.get(chain, 0) + 1
            else:
                self._chain_scores.setdefault(chain, []).append(
                    {"iteration": iteration, "norm_score": norm, "score": val}
                )

        if norm is not None and (niche["elite_norm_score"] is None or norm < niche["elite_norm_score"]):
            niche["elite_score"] = val
            niche["elite_norm_score"] = norm
            niche["elite_iteration"] = iteration
            niche["elite_history_entry"] = history_entry
            niche["stale_visits"] = 0
        else:
            niche["stale_visits"] += 1

        # Population, kept independently of the elite so that admitting a
        # runner-up cannot change which entry is the elite. Anything scored
        # competes; the cap keeps the archive small and the summary legible.
        if norm is not None:
            pop = niche.setdefault("population", [])
            pop.append({
                "norm_score": norm,
                "score": val,
                "iteration": iteration,
                "history_entry": history_entry,
                "lineage_id": self._lineage_id(history_entry, iteration),
                "depth": self._depth_of(history_entry),
            })
            pop.sort(key=lambda m: m["norm_score"])
            del pop[self.population_size:]

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

    def select_action(
        self,
        budget_remaining: int,
        budget_total: int,
        rng: Optional[Any] = None,
        exclude_lineages: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Decide the KIND of move to make next, then which lineage to make it on.

        The archive's older question was "which cell should the next proposal
        target". That is a breadth question and it only ever has breadth
        answers, which is why run closure_20260826_codex opened 33 mechanism
        families over 55 evaluations -- 1.1 evaluations per cell, 46 of 50
        cells visited exactly once, and 50 of 55 candidates with no parent at
        all. There was no depth to allocate.

        This asks the allocation question instead, over three actions:

          deepen      refine the tip of an existing lineage
          widen       start a new lineage inside a family already in the archive
          new_family  open a mechanism not yet tried

        Chosen by Thompson sampling, following Xin26 (BaSE), who found it beat
        UCB and EXP3.P on this exact depth-versus-breadth allocation, and Mis25
        (AB-MCTS), whose GEN node is the same "or make something new" arm. Both
        are validated from ~8 evaluations upward, which is the regime we are
        in; a fixed depth/breadth schedule is not, because the optimal split is
        task-dependent and unknown in advance.

        The reward a lineage is sampled against is its most recent IMPROVEMENT,
        not its score. A chain still moving 1.7% per step deserves another pull
        even while it trails a flat chain that is already better -- that is the
        case the elite-only archive cannot express, and on our own data every
        refinement that happened beat its parent (5 of 5).

        Returns {"action", "lineage_id", "family", "strategy", "elite",
        "is_new", "rationale"}. `elite` is the entry to mutate from, so a
        caller that only understands select_niche's contract still works.
        """
        import random as _random
        rng = rng or _random.Random()

        if budget_remaining <= 0:
            return {"action": "stop", "family": None, "strategy": None,
                    "is_new": False, "elite": None, "lineage_id": None,
                    "budget_exhausted": True,
                    "rationale": "no budget remains"}

        lineages = self.lineages()
        # Lineages already picked earlier in this batch. Proposing "refine
        # lineage 17" twice in one message wastes a slot -- the duplicate is
        # usually killed downstream, shrinking the batch -- and measured on the
        # real archive 44% of four-candidate batches contained a repeat. The
        # visit bump the caller does between picks cannot prevent it, because
        # the deepen path never reads `visits`.
        if exclude_lineages:
            lineages = {
                root: lin for root, lin in lineages.items()
                if root not in set(exclude_lineages)
            }
        if not lineages:
            return {"action": "new_family", "family": None, "strategy": None,
                    "is_new": True, "elite": None, "lineage_id": None,
                    "rationale": "archive is empty; nothing to refine yet"}

        # Sampled value of deepening each lineage. Beta over "did the last step
        # improve", so a chain with a run of gains is pulled more often, and a
        # chain that has stalled decays toward its prior WITHOUT being deleted
        # -- it can still win a draw later, which is the whole point of keeping
        # it. Depth is not penalised: there is no evidence a deep chain is
        # exhausted, and the one depth-2 lineage we have is the second-best
        # model in the study.
        #
        # Two-stage on purpose, and the second stage does NOT reuse the first
        # stage's draw. Sampling every lineage and taking the max decides the
        # action by arithmetic rather than merit: the maximum of N Beta(1,1)
        # draws concentrates at N/(N+1), which is 0.976 for the 39 lineages in
        # our own archive, so "deepen" wins ~98% of draws however badly those
        # chains are doing. Handing that winning draw forward unchanged does
        # not fix it -- the number is still an order statistic over N.
        #
        # So: sampling picks WHICH lineage (that is Thompson sampling doing its
        # job, exploring among chains), and the chosen lineage's posterior MEAN
        # -- one number, not a maximum -- competes against the other two arms.
        # A single fresh draw against that mean keeps the action choice
        # stochastic without the order-statistic inflation.
        # Scores are normalised across the CURRENT lineage set, not by an
        # absolute formula. Two reasons, both measured:
        #
        #   - An absolute quality term spans whatever the metric's magnitude
        #     happens to be. On our archive 1/(1+score) spread only 18% across
        #     every lineage, far too little to outweigh the sampling below.
        #   - `max(0.0, norm)` made quality exactly 1.0 for EVERY lineage when
        #     the objective direction is "max", because _normalize negates and
        #     the clamp erased the whole ranking. Direction-dependent silence.
        #
        # norm_score is already direction-corrected (lower is better for both
        # min and max objectives), so min-max normalising it here gives a real
        # [0, 1] ranking that behaves identically in either direction.
        # Percentile rank, not raw min-max.
        #
        # Min-max is hostage to its own extremes, and in CFD the extremes are
        # not rare: a closure that destabilises the solver returns an enormous
        # error, and that single outlier sets `hi` so every real lineage
        # compresses toward 1.0. Measured: scores [0.09, 0.10, 0.11] give a
        # decisive 878/59/1 split; adding one diverged candidate at 5.0 turns
        # it into 339/355/363 -- a three-way coin toss. A rank is immune to how
        # far away the worst candidate is.
        #
        # And min-max is undefined when every lineage ties or there is only
        # one: `(hi - lo) or 1.0` then handed 0.0 to EVERY lineage, so the
        # allocator refused to deepen 90% of the time -- which is the state of
        # every campaign for its first few rounds, exactly when the one chain
        # it has is the only thing worth refining. select_niche already guards
        # this (`if span <= 0: return 0.5`); select_action did not.
        norms = sorted(l["best_norm_score"] for l in lineages.values())

        def _quality(norm: float) -> float:
            if len(norms) < 2 or norms[0] == norms[-1]:
                return 0.5
            # Fraction of lineages this one is at least as good as (lower
            # norm_score is better for both objective directions).
            worse = sum(1 for n in norms if n > norm)
            ties = sum(1 for n in norms if n == norm)
            return (worse + 0.5 * (ties - 1)) / (len(norms) - 1)

        # One entry per DISTINCT model, not per lineage.
        #
        # Re-implementations of the same physics land as separate roots with
        # the same tip score, and each then draws its own sample -- so a model
        # implemented three times gets three times the chance of being picked,
        # on no extra evidence. Measured on run closure_20260826_codex: the
        # score 0.113601 existed as EIGHT lineages (the no-op candidates, all
        # scoring exactly baseline) and collected 13% of every deepen. The
        # duplicates were outvoting the archive.
        #
        # Grouped on the tip score to a relative tolerance, keeping the deepest
        # member as the representative -- if the same model was reached twice,
        # the chain that got further is the one worth continuing.
        deduped: List[Dict[str, Any]] = []
        for lin in sorted(lineages.values(), key=lambda l: (-l["depth"], l["root_iteration"])):
            twin = next(
                (d for d in deduped
                 if abs(d["best_norm_score"] - lin["best_norm_score"])
                 <= 1e-9 * max(1.0, abs(lin["best_norm_score"]))),
                None,
            )
            if twin is None:
                deduped.append(lin)

        best: Optional[Tuple[float, Dict[str, Any]]] = None
        best_sample = 0.0
        for lin in deduped:
            gains = [
                lin["members"][i - 1]["norm_score"] - lin["members"][i]["norm_score"]
                for i in range(1, len(lin["members"]))
            ]
            # Evidence is the SIZE of each step, not its sign. A binary
            # improved/did-not throws away the thing that distinguishes a
            # chain worth pursuing from one crawling: under it a chain sitting
            # at 0.1090 ranks below a worse chain at 0.1110 that happened to
            # tick the right way. Gains are expressed as a fraction of the
            # archive's own score spread so the scale is metric-independent.
            # RELATIVE to the chain's own score, not to the archive spread.
            # A refinement step is small next to the gap between the best and
            # worst model in the archive: our real refinements gained 0.46% to
            # 1.69% of their parent's score, which is 0.2%-0.7% of the spread.
            # Dividing by the spread made the whole momentum term worth ~1% of
            # the signal -- present in the code, absent from the decision.
            # Relative improvement is also scale-free, so it behaves the same
            # whether the objective is 0.1 or 10000.
            history = lin.get("history") or lin["members"]
            rel = []
            for i in range(1, len(history)):
                prev = history[i - 1]["norm_score"]
                cur = history[i]["norm_score"]
                denom = abs(prev) or 1.0
                rel.append((prev - cur) / denom)
            gained = sum(max(0.0, g) for g in rel)
            lost = sum(max(0.0, -g) for g in rel)

            # An attempt that produced no score is evidence against the chain.
            # Without this a chain whose last two builds failed to compile read
            # exactly like one that never failed -- same depth, same trace,
            # same gain -- and the allocator would keep spending refinements on
            # something that cannot produce a candidate. Weighted like a
            # middling regression: enough that repeated failure moves the
            # chain down the order, not so much that one bad build buries an
            # otherwise productive line.


            # An unrefined root is not a coin flip. Beta(1,1) says "no idea",
            # and with 35 such roots in a 39-lineage archive the maximum of
            # their draws lands near 0.97 -- so the uninformed crowd wins the
            # selection on count alone and the one proven chain is reached
            # about 12% of the time, barely above the 10% it would get from
            # picking at random. We do know something about an unrefined root:
            # its score. Folding that in as pseudo-counts means a good root
            # starts ahead of a bad one, and real gains then accumulate on top.
            quality = _quality(lin["best_norm_score"])
            # Quality is what the chain is WORTH; buildability is the chance a
            # refinement of it produces anything at all. The prior is the
            # product, because a chain that cannot be built has no expected
            # value however good its best member scores -- and score alone is
            # static, so without this a chain stayed top of the order through
            # any number of consecutive failed builds.
            scored_n = len(lin.get("history") or lin["members"])
            failures = lin.get("failures", 0)
            buildable = scored_n / float(scored_n + failures) if scored_n else 0.0
            effective_quality = quality * buildable

            alpha = 1.0 + self._lineage_prior * effective_quality + self._gain_weight * gained
            # Failures are counted in PSEUDO-COUNTS, not in relative-gain
            # units. A scoreless attempt is a failed attempt at deepening this
            # chain, so it should weigh like one failed trial -- the same as a
            # successful step weighs like one won trial. Expressed as a gain
            # fraction it was worth 0.6% of a trial and moved nothing.
            beta = (
                1.0
                + self._lineage_prior * (1.0 - effective_quality)
                + self._gain_weight * lost
                + self._failure_weight * failures
            )
            sample = rng.betavariate(alpha, beta)
            if best is None or sample > best[0]:
                best = (sample, lin)
                # Posterior MEAN of the chosen lineage, which is what competes
                # against the other two arms -- never its draw, which is an
                # order statistic over however many lineages happen to exist.
                best_sample = alpha / (alpha + beta)

        # The two "make something new" arms, sampled the same way from how
        # often each has historically paid off, so the split between refining
        # and exploring is learned rather than fixed.
        widen_value = rng.betavariate(1.0 + self._widen_wins, 1.0 + self._widen_losses)
        new_value = rng.betavariate(1.0 + self._newfam_wins, 1.0 + self._newfam_losses)

        # Exploration floor still applies: it exists to stop the search
        # circling known physics, and deepening is circling by construction.
        new_family_reserve = max(4, int(0.10 * max(1, budget_total)))
        if (
            self.exploration_floor > 0
            and self._visits_since_new_family >= self.exploration_floor
            and budget_remaining >= new_family_reserve
        ):
            return {"action": "new_family", "family": None, "strategy": None,
                    "is_new": True, "elite": None, "lineage_id": None,
                    "forced_by_exploration_floor": True,
                    "rationale": (
                        f"{self._visits_since_new_family} visits since the last new "
                        f"mechanism; the exploration floor forces one"
                    )}
        if budget_remaining < new_family_reserve:
            new_value = 0.0  # too little left to develop a new mechanism

        assert best is not None
        _, lin = best
        # One fresh draw for the deepen arm, on the same footing as the other
        # two: the chosen lineage's posterior mean supplies the centre, and the
        # study-level record of whether deepening has actually been paying off
        # supplies the same win/loss counts the other arms carry.
        #
        # Before this, deepen was judged purely per-lineage while widen and
        # new_family were judged on global win rates -- three numbers on two
        # different scales, compared as though they were one. record_action_
        # outcome silently ignored "deepen" entirely, so a study where every
        # refinement failed learned nothing from it.
        deepen_draw = rng.betavariate(
            max(1e-6, 2.0 * best_sample + self._deepen_wins),
            max(1e-6, 2.0 * (1.0 - best_sample) + self._deepen_losses),
        )
        chosen = max(
            (deepen_draw, "deepen"), (widen_value, "widen"), (new_value, "new_family")
        )[1]

        if chosen == "deepen":
            tip = lin["tip"]
            entry = tip["history_entry"] or {}
            return {
                "action": "deepen",
                "lineage_id": lin["root_iteration"],
                "family": entry.get("family"),
                "strategy": entry.get("strategy"),
                "elite": entry,
                "is_new": False,
                "depth": lin["depth"],
                "score_trace": lin["score_trace"],
                "rationale": (
                    f"lineage rooted at iteration {lin['root_iteration']} is at depth "
                    f"{lin['depth']} with trace {[round(x, 6) for x in lin['score_trace']]}; "
                    f"refining its best member"
                ),
            }
        if chosen == "widen":
            # A new lineage inside a family we already know something about --
            # the middle ground between refining one chain and opening an
            # untried mechanism.
            # allow_new_family=False, because select_action has already
            # decided this is NOT a new-family move -- possibly having just
            # zeroed that arm because the budget reserve forbids one. Without
            # the switch, select_niche appended its own unconditional new-family
            # option with no budget gate, and widen could hand back
            # {is_new: True, family: None}: the reserve was circumvented, the
            # proposer was told "target a NEW model family", and the outcome was
            # credited to the widen arm. Reproduced with budget_remaining=50
            # against a reserve of 400 -- 420 of 2000 widen picks came back new.
            pick = self.select_niche(
                budget_remaining, budget_total, allow_new_family=False
            )
            if pick.get("family") is None:
                # No existing family could be offered, so there is nothing to
                # widen into. Say so honestly rather than silently mutating
                # into a different action.
                return {"action": "new_family", "family": None, "strategy": None,
                        "is_new": True, "elite": None, "lineage_id": None,
                        "rationale": ("no existing family was available to widen "
                                      "into; opening a new mechanism instead")}
            pick["action"] = "widen"
            pick["lineage_id"] = None
            pick["elite"] = None  # a new chain, not a refinement of the elite
            pick.setdefault(
                "rationale",
                "starting a new lineage in a family already in the archive",
            )
            return pick
        return {"action": "new_family", "family": None, "strategy": None,
                "is_new": True, "elite": None, "lineage_id": None,
                "rationale": "opening a mechanism not yet in the archive"}

    def record_action_outcome(self, action: str, improved: bool) -> None:
        """Tell the allocator whether a move paid off, so its arms can learn.

        Called once per recorded candidate, for every action including
        "deepen" -- which this used to drop on the floor, leaving the deepen
        arm with no study-level evidence at all while the other two accumulated
        it. Without it the split between refining and exploring never adapts to
        the problem, which is the fixed schedule Xin26 shows is suboptimal.
        """
        if action == "deepen":
            if improved:
                self._deepen_wins += 1
            else:
                self._deepen_losses += 1
        elif action == "widen":
            if improved:
                self._widen_wins += 1
            else:
                self._widen_losses += 1
        elif action == "new_family":
            if improved:
                self._newfam_wins += 1
            else:
                self._newfam_losses += 1

    def select_niche(
        self, budget_remaining: int, budget_total: int, force_new_family: bool = False,
        allow_new_family: bool = True,
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

        # "Propose a new family" competes too, with a neutral q=0.5 (same as an
        # unscored niche) and zero visits (maximal exploration bonus) -- unless
        # the caller has already ruled that move out. select_action's `widen`
        # arm does exactly that: it has decided this is not a new-family move,
        # sometimes because the budget reserve forbids one, and an unconditional
        # option here silently overrode that decision.
        if allow_new_family:
            candidates.append((None, puct(None, 0), None))
        if not candidates:
            return {"family": None, "strategy": None, "is_new": False, "elite": None}

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

    @staticmethod
    def _per_case(niche: Dict[str, Any]) -> Dict[str, float]:
        """The elite's per-case scores, if it recorded any usable ones.

        Degenerate sets are treated as absent. A study whose comparator scores
        the whole set in one invocation writes the same aggregate into all N
        keys -- run closure_20260826_codex did exactly that, 32 identical
        values per candidate -- and printing 32 copies of one number as though
        it were a breakdown is worse than printing nothing.
        """
        entry = niche.get("elite_history_entry") or {}
        raw = entry.get("per_case_scores") if isinstance(entry, dict) else None
        if not isinstance(raw, dict) or len(raw) < 2:
            return {}
        clean = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
        if len(clean) < 2 or len(set(clean.values())) < 2:
            return {}
        return clean

    def render_case_difficulty(
        self,
        baseline_per_case: Optional[Dict[str, float]] = None,
        baseline_direction: str = "min",
        worst_n: int = 6,
    ) -> str:
        """Where the error actually concentrates, across every elite in the archive.

        The archive used to show one scalar per niche and nothing else, so a
        proposer could see THAT a family scored 0.09 but never WHERE those 0.09
        came from. The per-case scores were computed, written to
        candidate_record.json, and read by nothing -- one writer, no readers.

        That is the difference between "my model is mediocre" and "my model is
        excellent on separated flows and no better than baseline on the ducts",
        and only the second tells you which mechanism to reach for next. On run
        closure_20260826_codex the entire gap to the leaderboard sat in three
        duct cases while the hills were already competitive; nothing in the
        search could see that, and the mechanism that addresses it was not
        proposed until candidate ~56.

        Aggregated over elites rather than shown for one, because a case that
        every family struggles with is a property of the problem, while a case
        one family alone fails is a property of that family.
        """
        per_case_sets = [pc for pc in (self._per_case(n) for n in self.niches.values()) if pc]
        if not per_case_sets:
            return ""
        totals: Dict[str, List[float]] = {}
        for pc in per_case_sets:
            for case, value in pc.items():
                totals.setdefault(case, []).append(value)
        means = {c: sum(v) / len(v) for c, v in totals.items()}
        worse_is = (lambda a, b: a > b) if baseline_direction != "max" else (lambda a, b: a < b)
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=baseline_direction != "max")

        lines = [
            "PER-CASE DIFFICULTY (mean over the %d elite(s) that recorded per-case scores)."
            % len(per_case_sets),
            "  This is where the score is actually being lost. A mechanism that only",
            "  helps cases already near baseline cannot move the overall mean much.",
        ]
        for case, value in ranked[:worst_n]:
            base = (baseline_per_case or {}).get(case)
            delta = ""
            if isinstance(base, (int, float)):
                gap = value - base
                if gap == 0:
                    # Equal is not "better by +0": on this benchmark a case
                    # sitting exactly on baseline is the signature of a model
                    # that does nothing there, which is worth seeing as such.
                    delta = "  (baseline %.6g, unchanged)" % base
                else:
                    delta = "  (baseline %.6g, %s baseline by %+.4g)" % (
                        base, "worse than" if worse_is(value, base) else "better than", gap
                    )
            lines.append("    %-28s %.6g%s" % (case, value, delta))
        if len(ranked) > worst_n:
            best = ranked[-1]
            lines.append("    ... %d more; best case is %s at %.6g"
                         % (len(ranked) - worst_n, best[0], best[1]))
        return "\n".join(lines)

    def render_summary(
        self, baseline_score: Optional[float] = None, baseline_direction: str = "min",
        baseline_per_case: Optional[Dict[str, float]] = None,
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
            # This elite's own worst cases. The aggregate below says where the
            # problem is in general; this says where THIS family is losing,
            # which is what decides whether refining it is worth a visit.
            per_case = self._per_case(niche)
            if per_case:
                worst = sorted(per_case.items(), key=lambda kv: kv[1],
                               reverse=baseline_direction != "max")[:3]
                lines.append("      worst cases: "
                             + ", ".join(f"{c} {v:.4g}" for c, v in worst))
        lines.append("")
        lines.append(self.render_strategy_summary(baseline_score, baseline_direction))
        difficulty = self.render_case_difficulty(baseline_per_case, baseline_direction)
        if difficulty:
            lines.append("")
            lines.append(difficulty)
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

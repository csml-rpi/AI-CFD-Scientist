#!/usr/bin/env python3
"""
Offline test for scripts/oed_search_archive.py — the family-niched
quality-diversity archive that replaced the old flat single-chain search
policy in open_ended_discovery.py. No OpenFOAM, no LLM calls: drives
SearchArchive directly with a scripted sequence of synthetic scores across
three fake families.

Follows this repo's existing standalone-script test convention (see
scripts/test_metric_setup.py) rather than pytest — run directly:

    python3 scripts/test_oed_search_archive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oed_search_archive import SearchArchive  # noqa: E402

FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES += 1


def score(value: float, direction: str = "min") -> dict:
    return {"metric": "cf_rmse", "value": value, "direction": direction}


def test_classify_degrades_gracefully() -> None:
    fam = SearchArchive.classify("some description of a Spalart-Allmaras rotation correction", "SA_RC_v1")
    check("classify() returns a family string for SA/RC text", isinstance(fam, str) and len(fam) > 0)
    fam_unknown = SearchArchive.classify("", "")
    check("classify() degrades to a string (not a crash) on empty input", isinstance(fam_unknown, str))


def test_update_tracks_elite_per_family() -> None:
    archive = SearchArchive()
    archive.update("SA-APG", 1, score(0.0043), "min", {"iteration": 1})
    archive.update("SA-APG", 2, score(0.0041), "min", {"iteration": 2})  # better — becomes new elite
    archive.update("SA-APG", 3, score(0.0050), "min", {"iteration": 3})  # worse — elite unchanged
    niche = archive.niche_for("SA-APG")
    check("elite tracks the best (lowest) score seen", niche["elite_score"] == 0.0041)
    check("elite_iteration points at the iteration that produced it", niche["elite_iteration"] == 2)
    check("visits counts every update, not just improvements", niche["visits"] == 3)


def test_update_handles_missing_score() -> None:
    archive = SearchArchive()
    archive.update("SA-RC", 1, None, "min", {"iteration": 1})  # failed run, no score
    niche = archive.niche_for("SA-RC")
    check("a failed (unscored) attempt still counts as a visit", niche["visits"] == 1)
    check("a failed attempt does not fabricate an elite", niche["elite_score"] is None)


def test_invalid_scores_do_not_become_elites() -> None:
    archive = SearchArchive()
    archive.update("SA-RC", 1, score(float("nan")), "min", {"iteration": 1})
    archive.update("SA-RC", 2, score(float("inf")), "min", {"iteration": 2})
    check("NaN/inf scores never become elites", archive.niche_for("SA-RC")["elite_score"] is None)


def test_selection_prefers_promising_over_stale() -> None:
    archive = SearchArchive()
    # SA-APG: one very good result, one visit.
    archive.update("SA-APG", 1, score(0.001), "min", {"iteration": 1, "formula": "apg-corrected"})
    # SA-RC: repeatedly mediocre, many visits — should look less attractive
    # than SA-APG despite more attempts.
    for i in range(2, 8):
        archive.update("SA-RC", i, score(0.01), "min", {"iteration": i})

    sel = archive.select_niche(budget_remaining=10, budget_total=20)
    check(
        "select_niche favors the strong, under-visited niche over a heavily-visited mediocre one",
        sel["family"] in ("SA-APG", None),  # SA-APG or "try something new" are both defensible; SA-RC is not
        detail=f"got {sel}",
    )
    check("select_niche is never SA-RC here (repeatedly-tried and mediocre)", sel["family"] != "SA-RC")


def test_selection_offers_new_family_when_archive_empty() -> None:
    archive = SearchArchive()
    sel = archive.select_niche(budget_remaining=10, budget_total=10)
    check("an empty archive always proposes a new family", sel["is_new"] is True and sel["family"] is None)


def test_selection_conditions_on_elite_history_entry() -> None:
    archive = SearchArchive()
    entry = {"iteration": 5, "formula": "some formula", "case_dir": "/tmp/fake"}
    archive.update("SA-Production", 5, score(0.002), "min", entry)
    # A single niche means span == 0, so every q collapses to the neutral 0.5
    # and the zero-visit "new family" option wins on its exploration bonus —
    # the old version of this test guarded its only assertion behind
    # `if sel["family"] == "SA-Production"`, which is unreachable here, so it
    # executed zero checks. Give the archive a second, clearly worse family so
    # there is a real score spread and the elite path is actually taken.
    archive.update("SA-RC", 6, score(0.05), "min", {"iteration": 6})
    for i in range(7, 12):  # make the weak niche well-visited so it can't win
        archive.update("SA-RC", i, score(0.05), "min", {"iteration": i})
    sel = archive.select_niche(budget_remaining=5, budget_total=10)
    check(
        "a decisively-best niche is selected over a well-visited weak one and a new family",
        sel["family"] == "SA-Production",
        detail=f"got {sel}",
    )

    # The first picks are the empty (SA-Production, strategy) cells, which
    # deliberately carry no elite -- the mechanism's elite describes a model
    # built a different way, and handing it over invites the proposer to
    # re-describe that model under a new label. So drive until an EXISTING
    # niche is selected and assert the identity there. `checked` guards
    # against this passing vacuously, which an earlier version of the
    # mined-out test did for exactly this kind of reason.
    checked = False
    for _ in range(20):
        sel = archive.select_niche(budget_remaining=5, budget_total=10)
        if sel["family"] == "SA-Production" and not sel.get("is_new_strategy"):
            check("selected elite is the exact history entry stored on update",
                  sel["elite"] is entry, detail=f"got {sel}")
            checked = True
            break
        family, strategy = sel.get("family"), sel.get("strategy") or "analytic"
        if family:
            archive.niches.setdefault((family, strategy), archive._new_niche())
            archive.niches[(family, strategy)]["visits"] += 1
    check("an existing-niche pick was actually reached, so the check above ran", checked)


def test_q_is_normalized_not_raw_score() -> None:
    """Regression test for the original calibration bug.

    Raw scores are arbitrary in magnitude (a Cf RMSE of ~1e-3, an L2 error of
    ~1e-6). Comparing them directly against an O(1) exploration bonus made
    exploration win unconditionally, no matter how good or well-established a
    niche was. The fix min-max normalizes q into [0, 1]. This test pins that:
    it must pass for scores at 1e-3 AND at 1e-9, which a raw-score policy
    cannot do — with raw q, both niches sit within 1e-9 of zero and the
    exploration term decides everything.
    """
    for magnitude in (1e-3, 1e-9):
        archive = SearchArchive()
        good = {"iteration": 1, "formula": "good"}
        archive.update("SA-Production", 1, score(1.0 * magnitude), "min", good)
        for i in range(2, 9):
            archive.update("SA-RC", i, score(50.0 * magnitude), "min", {"iteration": i})
        sel = archive.select_niche(budget_remaining=8, budget_total=10)
        check(
            f"score magnitude {magnitude:g}: the better niche wins on q, not on raw magnitude",
            sel["family"] == "SA-Production",
            detail=f"got {sel}",
        )


def test_exploration_constant_is_calibrated_for_normalized_q() -> None:
    """Pins exploration_c against the [0, 1] q range it is calibrated for.

    With the pre-fix c=1.4, sqrt(log(total+1)/1) for the zero-visit new-family
    option reaches ~2 by ~30 total visits, so 2.8 swamps any real niche's q
    (capped at 1.0) and exploitation can never win inside a real study. This
    asserts the opposite: an established, clearly-best niche beats "try
    something new" once it has earned it.
    """
    archive = SearchArchive()
    archive.update("SA-Production", 1, score(0.001), "min", {"iteration": 1})
    for i in range(2, 7):
        archive.update("SA-RC", i, score(0.02), "min", {"iteration": i})
    sel = archive.select_niche(budget_remaining=6, budget_total=12)
    check(
        "a proven niche can outweigh the new-family option (exploration_c is calibrated)",
        sel["family"] == "SA-Production" and sel["is_new"] is False,
        detail=f"got {sel}, exploration_c={archive.exploration_c}",
    )


def test_budget_horizon_shifts_toward_exploitation() -> None:
    """The horizon term must decay exploration as budget runs out, not grow it.

    Pins direction, so an inverted or deleted horizon term fails here.
    """
    def build() -> SearchArchive:
        # Tuned so the horizon factor is the deciding term, and nothing else:
        # 'best' is the top scorer but heavily visited (small bonus), 'mid'
        # is close behind and barely visited (large bonus), 'weak' only sets
        # the low end of the min-max span.
        #
        # 'best' improves on every visit on purpose. A niche that is visited
        # 40 times without improving is the mined-out case the staleness
        # damping exists to demote (see
        # test_a_mined_out_family_stops_monopolising_the_search), and this
        # test is about the horizon term, not that one — a flat fixture here
        # would be testing the wrong thing and would fail for the right
        # reason.
        a = SearchArchive()
        for i in range(1, 41):
            a.update("best", i, score(0.0011 - i * 1e-6), "min", {"iteration": i})
        a.update("mid", 41, score(0.004), "min", {"iteration": 41})
        a.update("weak", 42, score(0.011), "min", {"iteration": 42})
        return a

    early = build().select_niche(budget_remaining=20, budget_total=20)
    late = build().select_niche(budget_remaining=1, budget_total=20)
    # Asserts the PROPERTY (exploration early, exploitation late), not which
    # exploratory option wins. It used to pin early["family"] == "mid",
    # because before empty (mechanism, strategy) cells were selectable, the
    # barely-visited rival mechanism was the only exploratory option that
    # could win. Now an untried strategy on the top mechanism competes too,
    # and early legitimately picks ('best', sweep) -- still exploration, and a
    # smaller, better-founded leap than a mediocre rival mechanism. Pinning
    # the family name would test the size of the option set, not the horizon.
    early_is_exploratory = bool(
        early.get("is_new") or early.get("is_new_strategy") or early["family"] != "best"
    )
    check(
        "with the full budget ahead, the policy explores rather than exploits",
        early_is_exploratory,
        detail=f"got {early}",
    )
    check(
        "with the budget nearly gone, the policy exploits the proven best niche instead",
        late["family"] == "best"
        and not late.get("is_new")
        and not late.get("is_new_strategy"),
        detail=f"got {late}",
    )


def _drive(archive, picks, start_iteration=100):
    """Run `picks` selections, feeding each one back as a real, non-improving
    evaluation.

    Feeding results back through update() rather than bumping ``visits`` by
    hand matters: only update() advances ``stale_visits``, and staleness
    damping is what lets any option other than the current leader ever win. A
    hand-bumped fixture leaves the leader's q pinned at 1.0 forever and tests
    a policy the archive never actually runs.
    """
    seen, kinds, iteration = set(), [], start_iteration
    for n in range(picks):
        sel = archive.select_niche(budget_remaining=900, budget_total=1000)
        kinds.append(
            "new_family" if sel.get("is_new")
            else ("new_strategy" if sel.get("is_new_strategy") else "existing")
        )
        if sel.get("is_new_strategy"):
            seen.add(sel.get("strategy"))
        family = sel.get("family") or f"invented_{n}"
        strategy = sel.get("strategy") or "analytic"
        archive.update(family, iteration, score(0.0109), "min",
                       {"iteration": iteration}, strategy=strategy)
        iteration += 1
    return seen, kinds


def test_empty_strategy_cells_are_selectable() -> None:
    """The (mechanism, strategy) grid must be fillable along BOTH axes.

    Measured on run closure_20260826_codex: every niche in the archive was
    (something, "analytic"), because select_niche enumerated only niches that
    already existed. An untried strategy was therefore never in the candidate
    set, could never be selected, and so could never score -- while the
    proposer prompt told the model "a strategy earns budget by scoring". 20
    candidates were launched and not one of them fitted anything.
    """
    archive = SearchArchive()
    archive.update("SST-crossdiff", 1, score(0.0110), "min", {"iteration": 1})
    archive.update("SST-crossdiff", 2, score(0.0108), "min", {"iteration": 2})
    archive.update("SST-production", 3, score(0.0180), "min", {"iteration": 3})

    seen, _kinds = _drive(archive, 12)
    check(
        "a strategy never tried on a scored mechanism does get selected",
        bool(seen),
        detail="no empty (mechanism, strategy) cell was ever picked",
    )
    check(
        "the fitting strategies specifically are reachable",
        {"solver_fit", "offline_fit"} <= seen,
        detail=f"strategies reached: {sorted(seen)}",
    )


def test_an_empty_cell_carries_no_elite_to_mutate_from() -> None:
    """The mechanism's elite describes a model built a DIFFERENT way.

    Handing it to the proposer alongside "now use offline_fit" invites it to
    re-describe that same model and relabel it, which is the mislabelling this
    axis exists to measure honestly.
    """
    archive = SearchArchive()
    archive.update("SST-crossdiff", 1, score(0.0110), "min", {"iteration": 1})
    archive.update("SST-crossdiff", 2, score(0.0108), "min", {"iteration": 2})
    archive.update("SST-production", 3, score(0.0180), "min", {"iteration": 3})

    iteration, checked = 100, False
    for _ in range(12):
        sel = archive.select_niche(budget_remaining=900, budget_total=1000)
        if sel.get("is_new_strategy"):
            checked = True
            check("an empty cell carries no elite", sel.get("elite") is None,
                  detail=f"got {sel}")
        archive.update(sel.get("family") or "x", iteration, score(0.0109), "min",
                       {"iteration": iteration}, strategy=sel.get("strategy") or "analytic")
        iteration += 1
    check("an empty cell was actually reached, so the check above ran", checked)


def test_a_mechanism_that_never_scored_gets_no_strategy_cells() -> None:
    """Offering a novel strategy on a mechanism that has never worked is two
    unknowns at once, and the result teaches nothing about either."""
    archive = SearchArchive()
    archive.update("broken", 1, None, "min", {"iteration": 1})  # failed, no score
    iteration = 100
    for _ in range(6):
        sel = archive.select_niche(budget_remaining=900, budget_total=1000)
        check(
            "no empty strategy cell is offered for an unscored mechanism",
            not (sel.get("is_new_strategy") and sel.get("family") == "broken"),
            detail=f"got {sel}",
        )
        archive.update(sel.get("family") or "invented", iteration, None, "min",
                       {"iteration": iteration}, strategy=sel.get("strategy") or "analytic")
        iteration += 1


def test_strategy_axis_does_not_starve_the_mechanism_axis() -> None:
    """Opening the strategy axis must not consume every pick.

    An empty cell inherits only part of its mechanism's score
    (`strategy_transfer`) and starts with a pseudo-count instead of the
    maximal zero-visit bonus (`strategy_prior_visits`), precisely so that
    trying a proven mechanism a new way cannot pre-empt exploiting it or
    exploring a genuinely new mechanism. With a full zero-visit bonus and full
    score transfer, measured, empty cells beat every other option every time
    until visited.
    """
    archive = SearchArchive()
    for i in range(1, 6):
        archive.update("A", i, score(0.010 - i * 1e-4), "min", {"iteration": i})
    archive.update("B", 6, score(0.020), "min", {"iteration": 6})

    _seen, kinds = _drive(archive, 30)
    check(
        "empty strategy cells do not take every pick",
        0 < kinds.count("new_strategy") < len(kinds),
        detail=f"{kinds.count('new_strategy')}/{len(kinds)} were new-strategy",
    )
    check(
        "established niches are still exploited",
        "existing" in kinds,
        detail=f"kinds seen: {sorted(set(kinds))}",
    )
    check(
        "and genuinely new mechanisms are still explored",
        "new_family" in kinds,
        detail=f"kinds seen: {sorted(set(kinds))}",
    )


def test_classify_family_does_not_confuse_source_with_rotation_curvature() -> None:
    """Regression test: family IS the niche identity.

    `"rc" in text` matched "sou*rc*e" and "fo*rc*e", so every SA source-term
    variant was filed as rotation-curvature and collapsed into that niche,
    while a real rotation-curvature model that said only "SA" (no
    "spalart"/"sa-"/"sa_") missed the SA gate entirely and became "unknown".
    """
    cases = [
        ("Spalart-Allmaras with an extra source term in the transport equation", "SA"),
        ("A source-term fvOption applied to the SA model", "SA"),
        ("SA model with a body force term", "SA"),
        ("SA with rotation-curvature correction", "SA-RC"),
        ("SA rotation curvature correction (SARC)", "SA-RC"),
        ("Modified SA destruction term with fw recalibration", "SA-Destruction"),
        ("SA-based adverse pressure gradient production modification", "SA-APG"),
        ("k-omega SST with a stress limiter", "k-omega-SST"),
    ]
    for description, expected in cases:
        got = SearchArchive.classify(description, "")
        check(f"classify({description[:44]!r}...) == {expected}", got == expected, detail=f"got {got}")

    # 5, not 8: the three source/force descriptions legitimately share the
    # generic "SA" family (the equation they touch is carried separately, in
    # classify_family's second return value). What must never happen is those
    # three landing in SA-RC and shadowing a real rotation-curvature elite.
    distinct = {SearchArchive.classify(d, "") for d, _ in cases}
    check(
        "distinct mechanisms do not collapse into one niche",
        len(distinct) >= 5,
        detail=f"only {len(distinct)} distinct families: {sorted(distinct)}",
    )


def test_saturation_tolerance_scales_with_the_metric() -> None:
    """A fixed absolute epsilon is a sane noise floor at Cf-RMSE magnitudes
    (~1e-3) but a 0.1% threshold at L2-error magnitudes (~1e-6), where it
    would call real progress a plateau and stop the search early."""
    # Magnitude chosen so a real 1% step (~1e-11) falls BELOW a fixed 1e-9
    # epsilon: the old absolute threshold declared this converged, a relative
    # one does not.
    archive = SearchArchive()
    value = 1e-9
    for i in range(1, 8):
        value *= 0.99  # a genuine 1% improvement every iteration
        archive.update("SA-Production", i, score(value), "min", {"iteration": i})
    check(
        "steady real improvement at tiny metric magnitudes is not read as a plateau",
        archive.is_saturated(window=3) is False,
        detail=f"trace ended at {value:g}",
    )

    flat = SearchArchive()
    for i in range(1, 8):
        flat.update("SA-Production", i, score(1e-9), "min", {"iteration": i})
    check("a genuinely flat archive still saturates at the same magnitude", flat.is_saturated(window=3))


def test_mixed_metric_directions_do_not_corrupt_the_ranking() -> None:
    """_normalize negates for "max", so mixing directions in one archive would
    rank a max-direction family above every min-direction one by sign alone."""
    archive = SearchArchive()
    archive.update("SA-Production", 1, score(0.001, "min"), "min", {"iteration": 1})
    archive.update("Bogus-Max", 2, score(0.95, "max"), "max", {"iteration": 2})
    check(
        "an entry contradicting the archive's direction is not admitted as an elite",
        archive.niche_for("Bogus-Max")["elite_score"] is None,
        detail=f"got {archive.niche_for('Bogus-Max')}",
    )
    check("the contradicting entry still counts as a visit", archive.niche_for("Bogus-Max")["visits"] == 1)
    check(
        "the legitimate family remains the archive's best",
        archive.select_niche(budget_remaining=2, budget_total=10)["family"] != "Bogus-Max",
    )


def test_replay_survives_a_malformed_iteration() -> None:
    archive = SearchArchive()
    archive.replay([
        {"action_type": "code_mod", "family": "SA-RC", "iteration": None, "score": score(0.01)},
        {"action_type": "code_mod", "family": "SA-RC", "iteration": "n/a", "score": score(0.008)},
    ])
    check("a malformed iteration does not abort a resume", archive.niche_for("SA-RC")["visits"] == 2)
    check("the elite is still recovered from a malformed entry", archive.niche_for("SA-RC")["elite_score"] == 0.008)


def test_saturation_detects_plateau() -> None:
    archive = SearchArchive()
    # Big improvement early, then a long flat plateau.
    archive.update("SA-APG", 1, score(0.01), "min", {"iteration": 1})
    archive.update("SA-APG", 2, score(0.004), "min", {"iteration": 2})  # improvement
    for i in range(3, 9):
        archive.update("SA-APG", i, score(0.004), "min", {"iteration": i})  # flat

    check("not saturated with too few evaluations for the window", not archive.is_saturated(window=10))
    check("saturated once the best score has been flat for `window` real evals", archive.is_saturated(window=5))


def test_saturation_false_when_still_improving() -> None:
    archive = SearchArchive()
    vals = [0.02, 0.015, 0.012, 0.009, 0.006, 0.004]
    for i, v in enumerate(vals, start=1):
        archive.update("SA-APG", i, score(v), "min", {"iteration": i})
    check("not saturated while the score keeps improving every step", not archive.is_saturated(window=3))


def test_failed_attempts_do_not_fake_score_saturation() -> None:
    archive = SearchArchive()
    archive.update("SA-APG", 1, score(0.01), "min", {"iteration": 1})
    for i in range(2, 10):
        archive.update("SA-APG", i, None, "min", {"iteration": i})
    check("unscored failures do not advance the score-saturation window", not archive.is_saturated(window=3))


def test_max_direction_and_budget_exhaustion() -> None:
    archive = SearchArchive()
    archive.update("family", 1, score(0.7, "max"), "max", {"iteration": 1})
    archive.update("family", 2, score(0.9, "max"), "max", {"iteration": 2})
    check("max-direction archive retains the larger score", archive.niche_for("family")["elite_score"] == 0.9)
    selection = archive.select_niche(budget_remaining=0, budget_total=2)
    check("selection reports exhausted budget instead of proposing work", selection.get("budget_exhausted") is True)


def test_replay_reconstructs_state_from_history() -> None:
    history = [
        {"iteration": 1, "action_type": "code_mod", "family": "SA-APG",
         "score": score(0.005), "compiled_model_name": "SA_APG_v1"},
        {"iteration": 2, "action_type": "code_mod", "family": "SA-APG",
         "score": score(0.0038), "compiled_model_name": "SA_APG_v2"},
        {"iteration": 3, "action_type": "python_script"},  # not a real evaluation — must be skipped
        {"iteration": 4, "action_type": "experiment", "family": "SA-RC",
         "score": score(0.006), "compiled_model_name": "SA_RC_v1"},
    ]
    archive = SearchArchive()
    archive.replay(history, baseline_direction="min")
    check("replay reconstructs both families seen in history", set(archive.families()) == {"SA-APG", "SA-RC"})
    check("replay finds the correct elite for a resumed niche", archive.niche_for("SA-APG")["elite_score"] == 0.0038)
    check("replay does not count python_script entries as visits", archive.niche_for("SA-APG")["visits"] == 2)


def test_render_summary_is_bounded_by_niche_count_not_iteration_count() -> None:
    archive = SearchArchive()
    for i in range(1, 51):  # 50 iterations, but only 3 distinct families
        fam = ["SA-APG", "SA-RC", "SA-Production"][i % 3]
        archive.update(fam, i, score(0.01 - i * 0.0001), "min", {"iteration": i})
    summary = archive.render_summary(baseline_score=0.0043, baseline_direction="min")
    line_count = summary.count("\n") + 1
    check(
        "render_summary has one line per family (+header, +strategy scoreboard), "
        "not per iteration",
        line_count <= 10,
        detail=f"got {line_count} lines for 50 iterations / 3 families",
    )
    check("render_summary includes a baseline delta when given one", "Δ vs baseline" in summary)


def test_a_mined_out_family_stops_monopolising_the_search() -> None:
    """Regression from a real run: one family took 21 of 24 evaluations.

    Min-max normalisation puts the leading niche's q at exactly 1.0 however
    slim its lead, so a marginally-best family outranked "try something new"
    forever. Three rival families got one evaluation each and were never
    revisited, while the leader was visited 16 more times after it had
    stopped improving.
    """
    # A family that improves for a while, then flatlines — a coefficient
    # sweep that has found its optimum.
    history = []
    for i, value in enumerate([0.00430, 0.00426, 0.004223], start=1):
        history.append({"action_type": "experiment", "family": "SA-Cross-Diffusion", "iteration": i,
                        "score": {"value": value, "direction": "min"}, "cost": 1})
    for i in range(4, 14):  # ten more visits, no improvement
        history.append({"action_type": "experiment", "family": "SA-Cross-Diffusion", "iteration": i,
                        "score": {"value": 0.00424, "direction": "min"}, "cost": 1})
    for family, value in (("SA-RC", 0.00440), ("Kato-Launder", 0.00437)):
        history.append({"action_type": "code_mod", "family": family, "iteration": 20,
                        "score": {"value": value, "direction": "min"}, "cost": 2})

    archive = SearchArchive()
    archive.replay(history, baseline_direction="min")
    check("staleness is counted", archive.niche_for("SA-Cross-Diffusion")["stale_visits"] >= 10,
          detail=str(archive.niche_for("SA-Cross-Diffusion")["stale_visits"]))
    check("a family that keeps improving is not marked stale",
          archive.niche_for("SA-RC")["stale_visits"] == 0)

    # niches are keyed by (family, strategy), not family. This used to index
    # archive.niches[family] -- a KeyError that never fired, because every one
    # of these four picks returned family=None ("propose a new family") before
    # empty (mechanism, strategy) cells were selectable. picks was
    # [None, None, None, None], so the assertion below compared nothing and
    # the test passed without ever exercising the damping it names.
    picks = []
    for _ in range(4):
        sel = archive.select_niche(budget_remaining=70, budget_total=100)
        picks.append(sel.get("family"))
        family = sel.get("family")
        if family:
            key = (family, sel.get("strategy") or "analytic")
            archive.niches.setdefault(key, archive._new_niche())
            archive.niches[key]["visits"] += 1
    check("the mined-out family no longer wins every slot",
          picks.count("SA-Cross-Diffusion") < len(picks), detail=str(picks))

    # ...but the elite it found is still the archive's best result.
    check("its best result is preserved, not discarded",
          archive.niche_for("SA-Cross-Diffusion")["elite_score"] == 0.004223)


def test_a_still_improving_family_keeps_being_exploited() -> None:
    """The damping must not punish a family that is still paying out —
    otherwise it just trades one failure mode for the opposite one."""
    history = [
        {"action_type": "experiment", "family": "Improving", "iteration": i,
         "score": {"value": 0.0050 - i * 0.0002, "direction": "min"}, "cost": 1}
        for i in range(1, 7)
    ]
    history += [
        {"action_type": "code_mod", "family": "Flat", "iteration": 10,
         "score": {"value": 0.0060, "direction": "min"}, "cost": 2}
    ]
    archive = SearchArchive()
    archive.replay(history, baseline_direction="min")
    check("an improving family carries no staleness",
          archive.niche_for("Improving")["stale_visits"] == 0)
    sel = archive.select_niche(budget_remaining=70, budget_total=100)
    check("it is still selected for exploitation", sel.get("family") == "Improving",
          detail=str(sel.get("family")))


def test_a_no_op_candidate_does_not_count_against_its_family() -> None:
    """A modification that provably changed nothing is not evidence.

    Regression for run oed_20260822_1626_codex_high, where candidate
    m051_diff_revflow compiled, ran, converged, and returned a score
    bit-identical to the baseline. It was recorded as an ordinary REVISE,
    so it consumed its family's only visit and wrote a baseline-equal
    "result" into the archive — and the reverse-flow diffusivity family was
    never revisited on the strength of a measurement that tested nothing.
    """
    arch = SearchArchive()
    entry = {"action_type": "code_mod", "family": "SA diffusivity", "no_op": True}
    arch.update("SA diffusivity", 1, score(0.00432102), "min", entry)
    check("a no-op creates no niche", "SA diffusivity" not in arch.families())

    arch.update("SA diffusivity", 2, score(0.0041), "min", {"action_type": "code_mod"})
    check(
        "a real evaluation of the same family still lands",
        (arch.niche_for("SA diffusivity") or {}).get("visits") == 1,
    )
    check(
        "and the no-op did not pollute the family's elite",
        arch.niche_for("SA diffusivity")["elite_score"] == 0.0041,
    )


def test_replay_skips_no_ops() -> None:
    history = [
        {"action_type": "code_mod", "family": "F1", "score": score(0.0043), "no_op": True, "iteration": 1},
        {"action_type": "code_mod", "family": "F2", "score": score(0.0041), "iteration": 2},
    ]
    arch = SearchArchive()
    arch.replay(history)
    check("replay admits only the real evaluation", list(arch.families()) == ["F2"])


def test_exploration_floor_breaks_a_family_monopoly() -> None:
    """PUCT alone never re-opens exploration once incumbents exist.

    Measured on run oed_20260822_1626_codex_high: the "new family" option
    scores a fixed q=0.5 plus its exploration bonus (0.76-0.92 across the
    whole run), while an incumbent visited even once is min-max normalized
    to q~0.95 with no staleness yet (1.05-1.28). The gap is structural, so
    the archive explored 16 families in 36 evaluations and then none at all
    in its last 25 — 41% of the budget — even though the eventual winner
    came from a family first tried at evaluation 24.
    """
    arch = SearchArchive(exploration_floor=8)
    for i in range(1, 21):
        arch.update("SA production multiplier", i, score(0.0043 - i * 1e-6), "min", {"action_type": "code_mod"})

    sel = arch.select_niche(budget_remaining=60, budget_total=100)
    check(
        "the floor forces a brand-new family after a long monopoly",
        sel["is_new"] and sel.get("forced_by_exploration_floor"),
        f"got {sel}",
    )

    arch2 = SearchArchive(exploration_floor=8)
    for i in range(1, 21):
        arch2.update("SA production multiplier", i, score(0.0043 - i * 1e-6), "min", {"action_type": "code_mod"})
    sel = arch2.select_niche(budget_remaining=3, budget_total=100)
    check(
        "but not when too little budget remains to develop one",
        not sel.get("forced_by_exploration_floor"),
        f"got {sel}",
    )


def test_exploration_floor_resets_when_a_new_family_appears() -> None:
    arch = SearchArchive(exploration_floor=4)
    for i in range(1, 6):
        arch.update("F1", i, score(0.004), "min", {"action_type": "code_mod"})
    check("counter accumulates within one family", arch._visits_since_new_family >= 4)
    arch.update("F2", 6, score(0.004), "min", {"action_type": "code_mod"})
    check("a genuinely new family resets the counter", arch._visits_since_new_family == 0)
    check(
        "so the floor does not fire again immediately",
        not arch.select_niche(budget_remaining=60, budget_total=100).get("forced_by_exploration_floor"),
    )


# --- strategy as a second archive dimension ---------------------------------
# The search used to pick WHICH physics to modify but never HOW to determine
# the modification: the action space was a two-value enum (code_mod /
# experiment), so "fit this through the solver" was not something the proposer
# could choose. Measured over runs oed_20260822_1626_codex_high and
# oed_20260823_opus_low: 0 of 86 candidates involved fitting of any kind, and
# 0 of 90 candidate trajectories imported sklearn, a scipy optimiser or torch,
# with all three installed throughout.
#
# Strategy is now a declared, coarsely-binned second niche dimension, so the
# archive keeps a separate elite per (mechanism, strategy) and the search can
# measure which strategies pay off instead of a prompt asserting it.

def test_strategy_binning_is_coarse_and_deterministic() -> None:
    from oed_search_archive import classify_strategy, normalize_strategy, STRATEGIES

    expectations = {
        "fit the coefficients by optimising the scored objective through the solver": "solver_fit",
        "a-posteriori model-consistent fit using differential evolution": "solver_fit",
        "regress b_ij on DNS velocity gradients offline with sklearn": "offline_fit",
        "frozen-RANS field inversion followed by symbolic regression": "offline_fit",
        "sweep the coefficient c_b1 of the already-compiled model": "sweep",
        "derive the form from physics and pick coefficients by hand": "analytic",
        "": "analytic",
    }
    for text, expected in expectations.items():
        check(
            f"strategy binning: {text[:44]!r} -> {expected}",
            classify_strategy(text) == expected,
            detail=f"got {classify_strategy(text)}",
        )
    check("an explicit valid label passes through", normalize_strategy("solver_fit") == "solver_fit")
    check("a hyphenated label normalises", normalize_strategy("solver-fit") == "solver_fit")
    check(
        "the vocabulary stays small enough to leave niches exploitable",
        len(STRATEGIES) <= 5,
        detail=f"{len(STRATEGIES)} strategies would give a mechanism count x {len(STRATEGIES)} niches",
    )


def test_archive_niches_on_mechanism_and_strategy() -> None:
    arch = SearchArchive()
    arch.update("SA-production", 1, score(0.0043), "min", {}, strategy="analytic")
    arch.update("SA-production", 2, score(0.0039), "min", {}, strategy="solver_fit")

    check("one mechanism under two strategies makes two niches", len(arch.niches) == 2)
    check("but the mechanism view still shows one family", list(arch.families()) == ["SA-production"])
    check(
        "the mechanism view reports its best elite across strategies",
        arch.niche_for("SA-production")["elite_score"] == 0.0039,
    )
    check(
        "and a niche can still be pinned to one strategy",
        arch.niche_for("SA-production", "analytic")["elite_score"] == 0.0043,
    )


def test_strategy_scoreboard_reports_what_each_approach_returned() -> None:
    arch = SearchArchive()
    arch.update("SA-production", 1, score(0.0043), "min", {}, strategy="analytic")
    arch.update("SA-diffusion", 2, score(0.0039), "min", {}, strategy="solver_fit")
    board = arch.render_strategy_summary(baseline_score=0.0043, baseline_direction="min")
    check(
        "the better-scoring strategy is ranked first",
        board.index("solver_fit") < board.index("analytic"),
        detail=board,
    )
    check("strategies never tried are named as such", "never tried" in board and "offline_fit" in board)


def test_exploration_floor_counts_mechanisms_not_strategies() -> None:
    """The floor exists to force NEW PHYSICS, not a third way of doing old physics."""
    arch = SearchArchive(exploration_floor=3)
    for i in range(1, 7):
        arch.update("F1", i, score(0.004), "min", {}, strategy="analytic")
    before = arch._visits_since_new_family
    arch.update("F1", 7, score(0.004), "min", {}, strategy="solver_fit")
    check(
        "a new strategy on a known mechanism does not reset the floor",
        arch._visits_since_new_family == before + 1,
        detail=f"{before} -> {arch._visits_since_new_family}",
    )
    arch.update("F2", 8, score(0.004), "min", {}, strategy="analytic")
    check("a genuinely new mechanism does reset it", arch._visits_since_new_family == 0)


def test_replay_reconstructs_the_strategy_dimension() -> None:
    arch = SearchArchive()
    arch.replay([
        {"action_type": "code_mod", "family": "F", "score": score(0.004), "iteration": 1,
         "strategy": "solver_fit"},
        {"action_type": "code_mod", "family": "F", "score": score(0.005), "iteration": 2,
         "strategy": "analytic"},
    ])
    check("replay rebuilds both (mechanism, strategy) niches", len(arch.niches) == 2,
          detail=f"got {sorted(arch.niches)}")



def test_a_fit_claim_must_be_supported_by_its_plan() -> None:
    """The declared strategy is intent; the plan is what actually runs.

    Regression for run closure_20260826_codex, where a candidate declared
    `solver_fit` while its plan read "compile it once, then run and score
    exactly all 32 supplied cases" — no optimiser, no coefficient search, no
    use of the stored high-fidelity fields. Taken at its word, it made the
    strategy scoreboard report fitting that never happened, in the one place
    the second dimension exists to measure honestly.
    """
    from oed_search_archive import normalize_strategy

    check(
        "a fit claim contradicted by its plan is downgraded",
        normalize_strategy(
            "solver_fit",
            plan="Compile it once, then run and score exactly all 32 supplied cases.",
        ) == "analytic",
    )
    check(
        "a genuine solver-in-the-loop fit is kept",
        normalize_strategy(
            "solver_fit",
            plan="Run a Nelder-Mead search over (a, b); each evaluation re-runs the "
                 "solver and re-scores.",
        ) == "solver_fit",
    )
    check(
        "a genuine offline fit is kept",
        normalize_strategy(
            "offline_fit",
            plan="Regress bijDelta on the stored DNS gradients with sklearn before compiling.",
        ) == "offline_fit",
    )
    check(
        "nothing is ever upgraded — understating is safer than inflating",
        normalize_strategy("analytic", plan="fit coefficients through the solver") == "analytic",
    )
    check(
        "a declared fit with no plan to check is trusted",
        normalize_strategy("solver_fit", plan="") == "solver_fit",
    )
    check(
        "non-fit declarations are untouched",
        normalize_strategy("sweep", plan="compile once and run") == "sweep",
    )


_A1_LIMITER_PLAN = (
    "Strategy: solver_fit. U_LES, k_LES, and tauij_LES training fields may be "
    "read offline only for sign/branch diagnostics; do not inspect validation "
    "labels for fitting or selection. Implement the fixed a1=0.25 runtime "
    "closure exactly as specified, with no fitted or case-dependent "
    "parameters. Exact compare_velocity_mae.py scoring is required downstream "
    "for each locked evaluation case."
)


def test_keywords_cannot_judge_a_fit_claim_that_merely_reads_data() -> None:
    """The keyword table's blind spot, pinned so it is never mistaken for a fix.

    Candidate `sst_a1_limiter_025` on run closure_20260826_codex declared
    `solver_fit` and fitted nothing: a1=0.25 is hand-chosen, and the LES fields
    are read "for sign/branch diagnostics". The plan contains the word
    "offline", so `classify_strategy` returns `offline_fit` — itself a fitting
    label — and the false claim passes validation.

    Whether reading data is fitting depends on what is done with it, which no
    keyword can see. This asserts the limitation rather than papering over it;
    `normalize_strategy(..., use_llm=True)` is what actually decides, and the
    keyword table stays only as the offline fallback.
    """
    from oed_search_archive import classify_strategy, normalize_strategy

    check(
        "keywords read 'offline' as a fit, whatever the plan does with it",
        classify_strategy(_A1_LIMITER_PLAN) == "offline_fit",
    )
    check(
        "so the keyword-only path cannot catch this false claim",
        normalize_strategy("solver_fit", plan=_A1_LIMITER_PLAN) == "solver_fit",
    )


def test_llm_decides_strategy_and_may_correct_either_way() -> None:
    """With the model reachable, the plan is read, not pattern-matched.

    Also pins the two properties the keyword fallback does not have: the model
    is consulted only when a plan exists to read, and it may correct in either
    direction — a candidate that quietly fits while calling itself analytic is
    upgraded, which keywords deliberately never do.
    """
    import oed_search_archive as arch

    calls = []

    def fake(plan, hypothesis="", declared=""):
        calls.append((plan, hypothesis, declared))
        return "analytic" if "no fitted" in plan else "solver_fit"

    original = arch._llm_classify_strategy_safe
    arch._llm_classify_strategy_safe = fake
    try:
        check(
            "the model overrides a false solver_fit claim the keywords accept",
            arch.normalize_strategy(
                "solver_fit", plan=_A1_LIMITER_PLAN, use_llm=True
            ) == "analytic",
        )
        check(
            "the model may also upgrade, which keywords never do",
            arch.normalize_strategy(
                "analytic",
                plan="Optimise (a, b) by re-running the solver each evaluation.",
                use_llm=True,
            ) == "solver_fit",
        )
        seen = len(calls)
        check(
            "no plan means nothing to read, so no model call is made",
            arch.normalize_strategy("solver_fit", plan="", use_llm=True) == "solver_fit"
            and len(calls) == seen,
        )
        check(
            "the declared label is passed so the model can see the claim it judges",
            calls[0][2] == "solver_fit",
        )

        arch._llm_classify_strategy_safe = lambda *a, **k: None
        check(
            "an unreachable model falls back to the keyword table, not a crash",
            arch.normalize_strategy(
                "solver_fit",
                plan="Compile it once, then run and score exactly all 32 cases.",
                use_llm=True,
            ) == "analytic",
        )
    finally:
        arch._llm_classify_strategy_safe = original


def test_replay_never_calls_the_model() -> None:
    """Replay runs over the whole history on every proposal call.

    A model call there would cost one request per entry and, worse, make the
    same history rebuild into different niches on different runs — the archive
    would stop being reproducible. Replayed entries carry a strategy already
    decided when they were proposed.
    """
    import oed_search_archive as arch

    called = []
    original = arch._llm_classify_strategy_safe
    arch._llm_classify_strategy_safe = lambda *a, **k: called.append(1) or "sweep"
    try:
        archive = arch.SearchArchive()
        archive.replay(
            [
                {
                    "action_type": "code_mod",
                    "family": "SST omega cross-diffusion scaling",
                    "strategy": "analytic",
                    "plan": "Fit the coefficients through the solver.",
                    "score": {"value": 0.11, "direction": "min"},
                }
            ]
        )
        check("replay made no model call", not called)
    finally:
        arch._llm_classify_strategy_safe = original



def _pc_entry(name, family, score, per_case, iteration=1, strategy="analytic"):
    return {"action_type": "code_mod", "variant_name": name, "family": family,
            "strategy": strategy, "iteration": iteration,
            "score": {"metric": "m", "value": score, "direction": "min"},
            "per_case_scores": per_case}


def test_archive_shows_where_the_score_is_being_lost() -> None:
    """per_case_scores had one writer and no readers.

    The proposer saw one scalar per niche, so it could tell THAT a family
    scored 0.09 but never WHERE those 0.09 came from -- the difference between
    "my model is mediocre" and "my model is excellent on separated flows and no
    better than baseline on the ducts". Only the second says which mechanism to
    reach for next.
    """
    a = SearchArchive()
    a.replay([
        _pc_entry("m1", "famA", 0.09,
                  {"DUCT_1": 0.30, "DUCT_2": 0.28, "HILL_1": 0.02, "HILL_2": 0.01}),
        _pc_entry("m2", "famB", 0.11,
                  {"DUCT_1": 0.32, "DUCT_2": 0.29, "HILL_1": 0.03, "HILL_2": 0.02}, 2),
    ], baseline_direction="min")
    out = a.render_summary(
        baseline_score=0.12, baseline_direction="min",
        baseline_per_case={"DUCT_1": 0.31, "DUCT_2": 0.30, "HILL_1": 0.05, "HILL_2": 0.04})

    check("difficulty block appears", "PER-CASE DIFFICULTY" in out)
    check("worst cases lead it", out.index("DUCT_1") < out.index("HILL_1"))
    check("aggregated over both elites", "over the 2 elite(s)" in out)
    check("each niche shows its own worst cases", out.count("worst cases:") == 2)
    check("per-case baseline shown for comparison", "baseline 0.31" in out)
    check("a case exactly on baseline is called unchanged, not 'better by +0'",
          "unchanged" in out)
    check("a case better than baseline is named", "better than baseline" in out)

    # With more cases than fit the list, the easiest one is named so the
    # reader can see the floor without every row being printed.
    many = SearchArchive()
    many.replay([_pc_entry("big", "famA", 0.1,
                           {f"c{i}": 0.30 - i * 0.02 for i in range(10)})],
                baseline_direction="min")
    big = many.render_case_difficulty()
    check("a long case list is truncated", "more; best case is" in big)
    check("and the easiest case is still named", "c9" in big)


def test_an_aggregate_comparator_does_not_fake_a_breakdown() -> None:
    """A comparator that scores the whole set in one invocation writes the same
    aggregate into every key -- run closure_20260826_codex produced 32 identical
    values per candidate. Printing 32 copies of one number as a breakdown is
    worse than printing nothing."""
    d = SearchArchive()
    d.replay([_pc_entry("agg", "famA", 0.11, {f"case_{i}": 0.11 for i in range(32)})],
             baseline_direction="min")
    out = d.render_summary(baseline_score=0.12)
    check("identical per-case values are suppressed", "PER-CASE DIFFICULTY" not in out)
    check("and no per-niche worst-case line either", "worst cases:" not in out)


def test_missing_per_case_data_is_harmless() -> None:
    n = SearchArchive()
    n.replay([{"action_type": "code_mod", "variant_name": "x", "family": "famA",
               "strategy": "analytic", "iteration": 1,
               "score": {"metric": "m", "value": 0.1, "direction": "min"}}],
             baseline_direction="min")
    out = n.render_summary(baseline_score=0.12)
    check("no per-case data -> no block, summary still renders",
          "PER-CASE DIFFICULTY" not in out and "famA" in out)
    check("render_case_difficulty alone returns empty", n.render_case_difficulty() == "")

    b = SearchArchive()
    b.replay([_pc_entry("mixed", "famA", 0.1, {"a": 0.3, "b": None, "c": 0.1})],
             baseline_direction="min")
    check("non-numeric per-case values are skipped, not raised",
          "a" in b.render_case_difficulty())


def test_case_difficulty_respects_metric_direction() -> None:
    m = SearchArchive()
    m.replay([_pc_entry("mx", "famA", 0.9, {"good": 0.95, "bad": 0.10})],
             baseline_direction="max")
    out = m.render_case_difficulty(baseline_direction="max")
    check("with direction=max the LOWEST score is the worst case",
          out.index("bad") < out.index("good"))


def _ln(i, fam, score, parent=None, action=None, strategy="analytic"):
    e = {"action_type": "code_mod", "variant_name": f"v{i}", "family": fam,
         "strategy": strategy, "iteration": i,
         "score": {"metric": "m", "value": score, "direction": "min"}}
    if parent is not None:
        e["parent_iteration"] = parent
    if action is not None:
        e["search_action"] = action
    return e


def test_a_cell_keeps_more_than_its_winner() -> None:
    """Strict elitism discards a structurally new variant that arrives untuned.

    It is compared once against an already-tuned elite, loses by a little, and
    goes -- with its compiled model and its case directory -- so there is
    nothing to tune next round.
    """
    a = SearchArchive()
    a.replay([_ln(1, "F", 0.10), _ln(2, "F", 0.11), _ln(3, "F", 0.12)],
             baseline_direction="min")
    niche = list(a.niches.values())[0]
    check("the elite is still the best entry", niche["elite_score"] == 0.10)
    check("runners-up are kept too", len(niche["population"]) == 3)
    check("population is ordered best-first",
          [m["score"] for m in niche["population"]] == [0.10, 0.11, 0.12])

    b = SearchArchive(population_size=2)
    b.replay([_ln(i, "F", 0.10 + 0.01 * i) for i in range(1, 6)], baseline_direction="min")
    check("the population is capped", len(list(b.niches.values())[0]["population"]) == 2)


def test_lineages_are_reconstructed_from_parents() -> None:
    a = SearchArchive()
    a.replay([_ln(1, "F", 0.10), _ln(2, "F", 0.09, parent=1), _ln(3, "F", 0.08, parent=2),
              _ln(4, "G", 0.20)], baseline_direction="min")
    lin = a.lineages()
    check("one chain per root", len(lin) == 2)
    chain = lin[1]
    check("depth counts refinement steps", chain["depth"] == 2, chain["depth"])
    check("the trace is in order", chain["score_trace"] == [0.10, 0.09, 0.08])
    check("the tip is the best member", chain["tip"]["score"] == 0.08)
    check("a still-improving chain reports a positive last gain", chain["last_gain"] > 0)

    # A chain whose ancestor never scored must not be dropped.
    c = SearchArchive()
    c.replay([_ln(9, "F", 0.09, parent=7)], baseline_direction="min")
    check("a chain with an unscored ancestor still forms a lineage", len(c.lineages()) == 1)


def test_allocator_asks_the_allocation_question() -> None:
    import random
    a = SearchArchive()
    check("an empty archive can only open a new family",
          a.select_action(100, 100)["action"] == "new_family")

    a.replay([_ln(1, "F", 0.10), _ln(2, "F", 0.09, parent=1)], baseline_direction="min")
    seen = {a.select_action(500, 1000, rng=random.Random(s))["action"] for s in range(60)}
    check("all three actions are reachable",
          seen == {"deepen", "widen", "new_family"}, seen)

    d = [a.select_action(500, 1000, rng=random.Random(s)) for s in range(200)]
    deep = [x for x in d if x["action"] == "deepen"]
    check("a deepen decision names the lineage it will refine",
          all(x.get("lineage_id") is not None for x in deep))
    check("a deepen decision carries the elite to mutate from",
          all(x.get("elite") for x in deep))
    check("a deepen decision explains itself",
          all("still improving" in x["rationale"] or "refining" in x["rationale"] for x in deep))
    check("no budget means no action", a.select_action(0, 100)["action"] == "stop")


def test_allocation_follows_the_evidence() -> None:
    """The split between refining and exploring must be learned, not fixed.

    Xin26's result is that the optimal depth/breadth split is task-dependent
    and unknown in advance, so a fixed schedule is the wrong shape.
    """
    import random
    from collections import Counter

    def share(hist, new_wins, new_losses, n=400):
        a = SearchArchive()
        a.replay(hist, baseline_direction="min")
        a._newfam_wins, a._newfam_losses = new_wins, new_losses
        rng = random.Random(11)
        c = Counter(a.select_action(500, 1000, rng=rng)["action"] for _ in range(n))
        return {k: c[k] / n for k in ("deepen", "widen", "new_family")}

    improving = [_ln(1, "F", 0.10)] + [_ln(i, "F", 0.10 - 0.005 * i, parent=i - 1)
                                       for i in range(2, 6)]
    flat = [_ln(1, "F", 0.10)] + [_ln(i, "F", 0.10, parent=i - 1) for i in range(2, 6)]

    good_chain = share(improving, 1, 12)   # chain works, new families do not
    dead_chain = share(flat, 12, 1)        # chain stalled, new families work

    check("a productive chain is deepened more than a stalled one",
          good_chain["deepen"] > dead_chain["deepen"],
          f"{good_chain['deepen']:.2f} vs {dead_chain['deepen']:.2f}")
    check("new families are opened more when they have been paying off",
          dead_chain["new_family"] > good_chain["new_family"],
          f"{dead_chain['new_family']:.2f} vs {good_chain['new_family']:.2f}")


def test_deepen_does_not_win_by_arithmetic() -> None:
    """The maximum of N Beta(1,1) draws concentrates at N/(N+1).

    Sampling every lineage and taking the max would pick "deepen" ~98% of the
    time with 39 lineages -- as it did on our own archive -- however badly
    those chains were doing. That is the count winning, not the evidence.
    """
    import random
    from collections import Counter
    many = SearchArchive()
    many.replay([_ln(i, f"F{i}", 0.10) for i in range(1, 40)], baseline_direction="min")
    check("39 unrefined roots exist", len(many.lineages()) == 39)
    rng = random.Random(3)
    c = Counter(many.select_action(500, 1000, rng=rng)["action"] for _ in range(400))
    deepen = c["deepen"] / 400
    check("deepen does not dominate merely because there are many lineages",
          deepen < 0.60, f"deepen share {deepen:.2f} over 39 unrefined roots")


def test_replay_relearns_the_arms() -> None:
    a = SearchArchive()
    a.replay([_ln(1, "F", 0.10, action="new_family"),
              _ln(2, "F", 0.09, parent=1, action="deepen"),
              _ln(3, "G", 0.20, action="new_family")], baseline_direction="min")
    check("a resumed study remembers which arms paid off",
          (a._newfam_wins, a._newfam_wins + a._newfam_losses) == (1, 2),
          f"{a._newfam_wins}/{a._newfam_wins + a._newfam_losses}")

    b = SearchArchive()
    b.replay([_ln(1, "F", 0.10), _ln(2, "F", 0.09, parent=1)], baseline_direction="min")
    check("history without search_action leaves the arms at their priors",
          b._newfam_wins == 0 and b._widen_wins == 0)


def test_widen_is_not_mistaken_for_a_refinement() -> None:
    """`widen` clears the elite on purpose -- it starts a NEW chain.

    The proposer's prompt builder had no branch for that, so a widen pick fell
    through to the "build on the elite" text with an empty elite and produced
    "Build on family 'X' (best result so far, from iteration ?): ." -- a
    literal question mark, no formula, no runtime coefficients -- and then
    asked for a base_case_dir that had never been shown. A candidate answering
    with action_type=experiment was dropped for having no valid base case.

    Invisible on a wide archive, where widen is served by the empty-strategy-
    cell branch, and live as soon as a family's four strategy cells fill up --
    which is what a small focused study looks like.
    """
    import random
    from pathlib import Path as _P

    def e(i, fam, strat, v):
        return {"action_type": "code_mod", "variant_name": f"v{i}", "family": fam,
                "strategy": strat, "iteration": i, "model_description": f"model {i}",
                "case_dir": f"/tmp/c{i}",
                "score": {"metric": "m", "value": v, "direction": "min"}}

    # All four strategy cells filled, so there is no empty cell to offer and
    # select_niche must return a real niche -- the shape that used to break.
    a = SearchArchive()
    a.replay([e(1, "A", "analytic", 0.10), e(2, "A", "sweep", 0.11),
              e(3, "A", "solver_fit", 0.12), e(4, "A", "offline_fit", 0.13)],
             baseline_direction="min")

    broken_shape = 0
    for seed in range(400):
        d = a.select_action(500, 1000, rng=random.Random(seed))
        if d["action"] != "widen":
            continue
        check_once = (not d.get("is_new")) and (not d.get("is_new_strategy"))
        if check_once:
            broken_shape += 1
            check("a widen pick carries no elite to refine", d.get("elite") in (None, {}))
            check("a widen pick still names its family", bool(d.get("family")))
            break
    check("the previously-broken widen shape is reachable at all", broken_shape > 0)

    # The prompt builder must special-case it rather than falling through.
    src = _P("src/cfd_langgraph/manager/tools.py").read_text()
    check("the proposer prompt has a widen branch", "WIDEN family" in src)
    check("widen tells the proposer not to reuse a parent model",
          "do NOT use action_type=experiment" in src)
    check("widen asks for a genuinely different formulation",
          "attack the same" in src and "different way" in src)


def _ch(i, fam, v, parent=None, strat="analytic"):
    d = {"action_type": "code_mod", "variant_name": f"v{i}", "family": fam,
         "strategy": strat, "iteration": i,
         "score": {"metric": "m", "value": v, "direction": "min"}}
    if parent is not None:
        d["parent_iteration"] = parent
    return d


def _chfail(i, fam, parent, strat="analytic"):
    return {"action_type": "code_mod", "variant_name": f"f{i}", "family": fam,
            "strategy": strat, "iteration": i, "parent_iteration": parent,
            "score": None}


def _deepen_share(hist, n=1200):
    import random
    from collections import Counter
    a = SearchArchive()
    a.replay(hist, baseline_direction="min")
    c = Counter()
    for seed in range(n):
        d = a.select_action(500, 1000, rng=random.Random(seed))
        if d["action"] == "deepen":
            c[d["lineage_id"]] += 1
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def test_a_chains_history_is_not_capped_by_the_population() -> None:
    """The per-cell cap bounds stored artifacts, not what the allocator knows.

    A six-step chain living in one cell reported a three-point trace, so its
    momentum evidence was clipped to population_size and the trace shown to the
    proposer was simply wrong.
    """
    a = SearchArchive(population_size=3)
    a.replay([_ch(1, "A", 0.120)] + [_ch(i, "A", 0.120 - 0.004 * i, i - 1) for i in range(2, 7)],
             baseline_direction="min")
    lin = list(a.lineages().values())[0]
    check("a 6-step chain reports all 6 scores", len(lin["score_trace"]) == 6,
          len(lin["score_trace"]))
    check("depth is still right", lin["depth"] == 5, lin["depth"])
    check("the population itself stays capped",
          len(list(a.niches.values())[0]["population"]) == 3)


def test_failed_attempts_count_against_their_chain() -> None:
    """A chain whose builds keep failing is not as promising as one that works.

    Only scored candidates entered the population, so a chain that had failed
    to compile three times running was indistinguishable from one that never
    failed -- same depth, same trace, same last_gain -- and kept collecting
    refinements it could not use.
    """
    a = SearchArchive()
    a.replay([_ch(1, "A", 0.10), _ch(2, "A", 0.09, 1)] + [_chfail(10 + k, "A", 2) for k in range(3)],
             baseline_direction="min")
    lin = list(a.lineages().values())[0]
    check("failures are recorded against the chain", lin["failures"] == 3, lin["failures"])
    check("they do not corrupt the score trace", lin["score_trace"] == [0.10, 0.09])

    base = [_ch(1, "A", 0.1000), _ch(2, "A", 0.0985, 1), _ch(3, "A", 0.0970, 2),
            _ch(10, "B", 0.0971)]
    clean = _deepen_share(base).get(1, 0.0)
    failing = _deepen_share(base + [_chfail(20 + k, "A", 3) for k in range(6)]).get(1, 0.0)
    check("repeated build failures reduce a chain's share of refinements",
          failing < clean - 0.15, f"clean {clean:.2f} -> failing {failing:.2f}")


def test_duplicate_models_do_not_multiply_their_odds() -> None:
    """Re-implementations of one model are one piece of evidence, not N.

    Each landed as its own root with the same tip score and drew its own
    sample, so a model implemented eight times was eight times as likely to be
    picked on no extra evidence.
    """
    import random
    from collections import Counter
    # Eight copies of a mediocre model against one genuinely better model.
    # Before deduplication the copies won on count alone; the better model has
    # to win, or the search refines whatever happened to be re-implemented most.
    dupes = [_ch(i, f"F{i}", 0.100) for i in range(1, 9)]
    better = [_ch(20, "G", 0.090)]
    a = SearchArchive()
    a.replay(dupes + better, baseline_direction="min")
    check("all nine start as separate lineages", len(a.lineages()) == 9)

    c = Counter()
    for seed in range(1200):
        d = a.select_action(500, 1000, rng=random.Random(seed))
        if d["action"] == "deepen":
            c[d["lineage_id"]] += 1
    total = sum(c.values()) or 1
    dup_roots = set(range(1, 9))
    picked = {k for k, v in c.items() if v}
    check("only one representative of the duplicate set is ever offered",
          len(picked & dup_roots) <= 1, sorted(picked & dup_roots))
    check("eight copies cannot outvote one better model",
          c[20] / total > sum(c[r] for r in dup_roots) / total,
          f"better {c[20] / total:.2f} vs copies {sum(c[r] for r in dup_roots) / total:.2f}")


def test_every_arm_carries_the_same_kind_of_evidence() -> None:
    """deepen was judged per-lineage while the others were judged globally.

    record_action_outcome dropped "deepen" on the floor, so a study in which
    every refinement failed learned nothing from it while widen and new_family
    accumulated win rates -- two scales compared as one.
    """
    a = SearchArchive()
    for action, improved in [("deepen", True), ("deepen", False), ("widen", True),
                             ("new_family", False)]:
        a.record_action_outcome(action, improved)
    check("deepen outcomes are recorded", (a._deepen_wins, a._deepen_losses) == (1, 1),
          (a._deepen_wins, a._deepen_losses))
    check("widen outcomes are recorded", (a._widen_wins, a._widen_losses) == (1, 0))
    check("new_family outcomes are recorded", (a._newfam_wins, a._newfam_losses) == (0, 1))

    b = SearchArchive()
    b.replay([dict(_ch(1, "A", 0.10), search_action="new_family"),
              dict(_ch(2, "A", 0.09, 1), search_action="deepen"),
              dict(_ch(3, "B", 0.20), search_action="new_family")],
             baseline_direction="min")
    check("a resumed study relearns the deepen arm too",
          b._deepen_wins + b._deepen_losses == 1,
          (b._deepen_wins, b._deepen_losses))


def test_a_lineage_is_never_deleted_by_the_population_cap() -> None:
    """The cap bounds stored artifacts; it must not bound WHO can be chosen.

    Deriving lineage membership from the capped per-cell populations deleted
    whole chains: one whose members all rank below the top `population_size`
    of their cell vanished from the allocator entirely -- not truncated, gone
    -- while _chain_scores still held it, so nothing looked broken. The chain
    most worth continuing is often exactly the one still climbing from a poor
    start, which is the one this dropped.
    """
    import random
    from collections import Counter
    a = SearchArchive(population_size=3)
    a.replay([_ch(1, "A", 0.090), _ch(2, "A", 0.091), _ch(3, "A", 0.092),
              _ch(4, "A", 0.20), _ch(5, "A", 0.15, 4), _ch(6, "A", 0.12, 5)],
             baseline_direction="min")
    lin = a.lineages()
    check("the buried chain is still a lineage", 4 in lin, sorted(lin))
    check("its history is intact", lin[4]["score_trace"] == [0.20, 0.15, 0.12])
    check("the cell population is still capped",
          len(list(a.niches.values())[0]["population"]) == 3)

    c = Counter()
    for seed in range(1200):
        d = a.select_action(500, 1000, rng=random.Random(seed))
        if d["action"] == "deepen":
            c[d["lineage_id"]] += 1
    check("and it can actually be selected", c[4] > 0, dict(c))
    check("its momentum earns it a real share", c[4] / max(1, sum(c.values())) > 0.15,
          c[4] / max(1, sum(c.values())))


def test_quality_survives_a_degenerate_or_outlier_archive() -> None:
    """CFD produces both cases routinely, and min-max broke on each.

    A closure that destabilises the solver returns an enormous error and, as
    the single maximum, compressed every real lineage toward the same quality.
    And with one lineage or all-equal scores, `(hi - lo) or 1.0` handed 0.0 to
    everything, so the allocator refused to deepen the only chain it had --
    the state of every campaign's opening rounds.
    """
    import random
    from collections import Counter

    def actions(hist, n=900):
        a = SearchArchive()
        a.replay(hist, baseline_direction="min")
        return Counter(a.select_action(500, 1000, rng=random.Random(s))["action"]
                       for s in range(n))

    one = actions([_ch(1, "A", 0.09)])
    check("a single-lineage archive still deepens", one["deepen"] / 900 > 0.20,
          one["deepen"] / 900)
    tied = actions([_ch(1, "A", 0.09), _ch(2, "B", 0.09), _ch(3, "C", 0.09)])
    check("an all-equal archive still deepens", tied["deepen"] / 900 > 0.20,
          tied["deepen"] / 900)

    def top_share(hist, n=900):
        a = SearchArchive()
        a.replay(hist, baseline_direction="min")
        c = Counter()
        for s in range(n):
            d = a.select_action(500, 1000, rng=random.Random(s))
            if d["action"] == "deepen":
                c[round(a.lineages()[d["lineage_id"]]["tip"]["score"], 3)] += 1
        return c[0.09] / max(1, sum(c.values()))

    clean = top_share([_ch(1, "A", 0.09), _ch(2, "B", 0.10), _ch(3, "C", 0.11)])
    withdiv = top_share([_ch(1, "A", 0.09), _ch(2, "B", 0.10), _ch(3, "C", 0.11),
                         _ch(4, "D", 5.0)])
    check("one diverged candidate does not flatten the ranking",
          withdiv > 0.5 * clean, f"clean {clean:.2f} -> with outlier {withdiv:.2f}")


def main() -> int:
    test_classify_degrades_gracefully()
    test_update_tracks_elite_per_family()
    test_update_handles_missing_score()
    test_invalid_scores_do_not_become_elites()
    test_selection_prefers_promising_over_stale()
    test_selection_offers_new_family_when_archive_empty()
    test_selection_conditions_on_elite_history_entry()
    test_q_is_normalized_not_raw_score()
    test_exploration_constant_is_calibrated_for_normalized_q()
    test_budget_horizon_shifts_toward_exploitation()
    test_empty_strategy_cells_are_selectable()
    test_an_empty_cell_carries_no_elite_to_mutate_from()
    test_a_mechanism_that_never_scored_gets_no_strategy_cells()
    test_strategy_axis_does_not_starve_the_mechanism_axis()
    test_classify_family_does_not_confuse_source_with_rotation_curvature()
    test_saturation_tolerance_scales_with_the_metric()
    test_mixed_metric_directions_do_not_corrupt_the_ranking()
    test_replay_survives_a_malformed_iteration()
    test_saturation_detects_plateau()
    test_saturation_false_when_still_improving()
    test_failed_attempts_do_not_fake_score_saturation()
    test_max_direction_and_budget_exhaustion()
    test_replay_reconstructs_state_from_history()
    test_render_summary_is_bounded_by_niche_count_not_iteration_count()
    test_a_mined_out_family_stops_monopolising_the_search()
    test_a_still_improving_family_keeps_being_exploited()
    test_a_no_op_candidate_does_not_count_against_its_family()
    test_replay_skips_no_ops()
    test_exploration_floor_breaks_a_family_monopoly()
    test_exploration_floor_resets_when_a_new_family_appears()
    test_strategy_binning_is_coarse_and_deterministic()
    test_archive_niches_on_mechanism_and_strategy()
    test_strategy_scoreboard_reports_what_each_approach_returned()
    test_exploration_floor_counts_mechanisms_not_strategies()
    test_replay_reconstructs_the_strategy_dimension()
    test_a_fit_claim_must_be_supported_by_its_plan()
    test_keywords_cannot_judge_a_fit_claim_that_merely_reads_data()
    test_llm_decides_strategy_and_may_correct_either_way()
    test_replay_never_calls_the_model()
    test_archive_shows_where_the_score_is_being_lost()
    test_an_aggregate_comparator_does_not_fake_a_breakdown()
    test_missing_per_case_data_is_harmless()
    test_case_difficulty_respects_metric_direction()
    test_widen_is_not_mistaken_for_a_refinement()
    test_a_chains_history_is_not_capped_by_the_population()
    test_failed_attempts_count_against_their_chain()
    test_duplicate_models_do_not_multiply_their_odds()
    test_every_arm_carries_the_same_kind_of_evidence()
    test_a_lineage_is_never_deleted_by_the_population_cap()
    test_quality_survives_a_degenerate_or_outlier_archive()
    test_a_cell_keeps_more_than_its_winner()
    test_lineages_are_reconstructed_from_parents()
    test_allocator_asks_the_allocation_question()
    test_allocation_follows_the_evidence()
    test_deepen_does_not_win_by_arithmetic()
    test_replay_relearns_the_arms()

    print(f"\n{'ALL PASS' if FAILURES == 0 else f'{FAILURES} FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())


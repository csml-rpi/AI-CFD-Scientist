# Search strategy under an expensive-evaluation budget — scope

Branch `search_strategy`. What we would change to adopt two ideas from the
literature, sized against the loop we actually have.

## The constraint that shapes everything

One candidate costs ~50 solver invocations and 30–90 min. The whole 4000-unit
budget buys **~79 candidates**. Measured on `closure_20260826_codex`:

| | |
|---|---|
| candidates recorded | 55 |
| mean cost | 50 solver invocations |
| distinct (family, strategy) cells | 50 |
| **mean evaluations per cell** | **1.1** |
| cells visited once / twice / three times | 46 / 3 / 1 |
| candidates with a recorded parent | **5 of 55** |

The search opens a new cell almost every evaluation and essentially never
refines. There is no depth to preserve yet.

`Sai21` does CFD-in-the-loop symbolic identification of Reynolds-stress models
— our exact problem — with **~5000 CFD solves**. We have 79. We are ~60x short,
so the algorithm has to buy sample-efficiency, not just spend better.

`Wu18` bounds what is possible: Reynolds-stress errors below 0.5% produce mean
velocity errors up to 35.1%, and a-priori fit quality is explicitly a poor
predictor of a-posteriori accuracy. **No cheap surrogate can replace the real
evaluation in this domain.** A proxy may only ever order the queue.

## Current loop, and the two places it is wrong for this budget

```
oed_propose_candidates(num_candidates=2..4)
    -> select_niche() x N   (PUCT over (family, strategy) cells)
    -> LLM writes N candidate specs
manager launches ALL N concurrently        <-- 1 proposal = 1 expensive evaluation
    -> build, run 32 cases, score
oed_record_candidate_results
    -> archive.update() keeps ONE elite per cell, drops the rest
```

Two structural problems:

1. **Proposal and evaluation are 1:1.** There is no triage. Every idea the
   proposer writes costs a full evaluation, so the proposer must be right first
   time, every time.
2. **One elite per cell, and lineage is not tracked.** `update()` replaces the
   elite only on strict improvement and discards the losing entry;
   `_new_niche()` has a single `elite_history_entry` slot. `parent_iteration`
   is set only when a pick carries an elite, so 50 of 55 candidates recorded no
   parent. There are no lineages to allocate compute between.

---

## Idea 1 — proxy ranking, never proxy deciding (`Liu26`, Janus)

Janus runs at 20–250 real evaluations, the same regime as ours, and reports
**59.1% fewer real evaluations** to reach 99% of terminal performance.

The shape that makes it safe under `Wu18`: the proxy chooses *which* candidate
is evaluated, never *whether* a candidate is good. Nothing enters the archive
without a real evaluation, so a wrong proxy wastes a slot and cannot corrupt
the search.

### What changes

**Propose a pool, evaluate a few.** `oed_propose_candidates` currently returns
`num_candidates` specs and the manager runs all of them. It would instead
return a pool (say 10–15) at no CFD cost, of which only the top 2–3 are
promoted. Proposal becomes cheap; only promotion is expensive.

**A new proxy object.** An LLM-written Python scoring program, not a Gaussian
process — our candidates are symbolic PDE modifications with different numbers
of coefficients and no distance metric, which is exactly why Janus uses code.
It reads a candidate spec (hypothesis, plan, strategy, target family) and
returns `(predicted_score, validity_probability, uncertainty)`.

Inputs it can legitimately use without running CFD:
- the mechanism and where it acts (transport equation vs direct stress
  substitution — `Wu18` says this predicts conditioning, and our own data
  agrees: transport-equation corrections are our two best models, stress
  substitutions our five worst)
- whether the candidate is a no-op or a duplicate of something already tried
- coefficient magnitudes against realizability bounds (`Sai21` pre-screens this)
- per-case difficulty from the archive — which cases the current elites lose on

**Calibration and credit.** Refit the proxy's parameters whenever new real
outcomes land. Score proxies on whether their top-k ranking recovers the best
archive programs, not on global prediction error. Track credit per region and
demote a proxy that promotes a candidate which then scores badly.

**Exploration gaps.** Periodically promote an unranked candidate directly, so a
shared proxy bias cannot silently suppress a whole direction.

### New / changed surface

| | |
|---|---|
| `oed_propose_candidates` | return a pool; add `pool_size`, do not imply evaluation |
| new `oed_score_candidates_by_proxy` | rank the pool, return promotions + rationale |
| new `oed_evolve_proxy` | LLM writes/mutates the scoring program, refit on archive |
| `proxy_population.json`, `proxy_credit.json` | proxy programs and their regional credit |
| manager prompt step b/c | propose pool -> proxy rank -> launch only promoted |
| `history.json` | record `proxy_predicted` next to the real score, to measure proxy skill |

### What it buys, and the risk

If it matches Janus, ~59% fewer real evaluations — effectively turning 79
candidates into ~190. The risk is that the proxy has nothing real to learn
from: with 55 archive points and 33 families, a code-based proxy may be fitting
noise. Mitigated by the fact that it can only reorder a queue, and by the
exploration gap. **Measurable before committing to it**: replay the 55 recorded
candidates, have a proxy rank them, and check whether its top-k recovers the
known winners. If it cannot rank history, it will not rank the future.

---

## Idea 2 — depth vs breadth as a bandit (`Xin26` BaSE, `Mis25` AB-MCTS)

`Xin26` frames budget as `C = T x N` and shows the optimal split is
task-dependent; their BaSE runs K trajectories as bandit arms, pulls one to
extend it, reward = fitness of the new candidate. Thompson sampling won:
**+12.3% fitness, ~40% fewer generations**, swept from 8 to 512 calls.
`Mis25` does the same decision inside a tree via a GEN node and beats baselines
**from 8 evaluations upward**; its AB-MCTS-A variant uses conjugate priors, so
no MCMC.

### What changes

**Prerequisite: keep lineages.** This cannot be built on the current archive.
Two changes first:

- `_new_niche()` keeps a small population (2–3) instead of one
  `elite_history_entry`, so a structurally new variant that lands slightly
  worse is still available to refine.
- Every candidate records its parent, always — not only when the pick carried
  an elite. Today that is 5 of 55.

**Then the allocation rule.** Replace "pick a cell by PUCT, propose one
candidate" with "pick an *action* by Thompson sampling":

- **deepen** lineage *i* — refine its current tip
- **widen** — start a new lineage in an existing family
- **new family** — open a mechanism not yet tried

Arms are lineages plus a GEN arm for "something new" (`Mis25`'s construction).
Reward is the improvement the pulled arm produced. A lineage that keeps
improving keeps getting pulled; one that stalls decays without being deleted —
which is your depth-4 case, and the thing strict elitism cannot express.

### New / changed surface

| | |
|---|---|
| `_new_niche` | `population: List[entry]` (cap 2–3) replacing single `elite_history_entry` |
| `update()` | admit into population by rank, keep lineage id + parent + depth |
| new `select_action()` | Thompson sampling over {deepen_i, widen, new_family} |
| `select_niche()` | kept as the fallback / cold-start path |
| `render_summary` | show lineage depth and trend, not just each cell's best |

### Interaction with idea 1

They compose cleanly and in one direction: the bandit decides *what kind* of
move to make, the proxy decides *which* of several drafts of that move is worth
the CFD. Build the bandit on top of the proxy, not beside it.

---

## What we are not doing

**3 (Sai21's cost tricks) is mostly not available.** Sensitivity analysis to
cut the coefficient space assumes a fixed parameterisation; ours is open-ended
symbolic. Sparse training data (4–8% of mesh points) conflicts with a benchmark
that grades the full field with a fixed comparator. Two pieces *are* portable
and are cheap:

- **Resample rather than discard a diverging candidate** — Sai21 perturbs the
  coefficients downward and retries instead of scoring it a failure. We
  currently throw it away.
- **Warm start.** `Mcc23` (turbo-RANS, by the benchmark's own author) seeds
  every probe from the converged default-coefficient fields and reports it
  "greatly reduces the computational cost". We deliberately delete non-zero
  time directories and solve from the initial condition. That is a defensible
  scientific choice, but it is a large speedup we are declining and it should
  be a deliberate decision, not a default.

**Racing over the 32 cases** is the one cheap-evaluation trick `Wu18` does not
rule out, and it is not in scope here. It needs evidence that a subset of cases
predicts the full-set ranking, which we can test from the per-case scores we
already store.

---

## Order of work

1. **Measure first, on data we already have.** Replay the 55 recorded
   candidates: (a) can a proxy rank them? (b) does any lineage show the
   rank-reversal pattern — poor early, better after refinement? The deep search
   found **no controlled rank-reversal analysis in the literature**, so this is
   both a prerequisite and publishable if the effect is real.
2. **Lineage + population** in the archive. Small, self-contained, and a
   prerequisite for the bandit.
3. **Proxy ranking** (idea 1). Largest expected saving.
4. **Bandit allocation** (idea 2). Needs 2 and benefits from 3.

Step 1 is cheap and decides whether 3 and 4 are worth building.

## References

- `Liu26` Janus: Algorithm-Evaluator Co-Evolution under Expensive Evaluation
  Budgets — arxiv.org/abs/2608.08189
- `Xin26` Compute Allocation in Evolutionary Search: From Depth-Breadth to
  Multi-Armed Bandits — doi.org/10.48550/arXiv.2605.29268
- `Mis25` Wider or Deeper? Adaptive Branching Tree Search — doi.org/10.48550/arXiv.2503.04412
- `Wu18` RANS with explicit data-driven Reynolds stress closure can be
  ill-conditioned — doi.org/10.1017/jfm.2019.205
- `Sai21` CFD-driven symbolic identification of algebraic Reynolds-stress
  models — doi.org/10.1016/j.jcp.2022.111037
- `Mcc23` turbo-RANS: Bayesian optimization of turbulence model coefficients —
  doi.org/10.1108/HFF-12-2023-0726
- `Lan25` ShinkaEvolve — doi.org/10.48550/arXiv.2509.19349
- `Rom23` FunSearch, Nature — doi.org/10.1038/s41586-023-06924-6

Literature workspace:
https://app.undermind.ai/projects/85309384-e156-4918-bb6b-85c4c12b8309

# Search-strategy ablation plan (2026-11-02, not started)

On hold: Codex quota is nearly used up. Nothing below has been built yet.

## Fixed in every run
- Case: SD7003, started from the checked 60k mesh and its verified baseline
  (5.42%), so no mesh gate. Source: trial 3's selected level,
  framework_comparisons/sd7003_trial_3/.../mesh_gate/<group>/sd7003_lm__refined_2.
- Same prompt, model, candidate agent, repair loop, scorer, and literature and
  hypothesis stage.
- Budget: 150 solver runs or 16 h, stated in the prompt and enforced from
  outside. 16 cores per study.
- A fresh frozen code copy for the ablation; the live trials' copy stays untouched.
- The only change between arms is the picker in `oed_propose_candidates`
  (`manager/tools.py`): which candidate comes next. Every arm reuses the
  existing proposer instruction texts (new family / deepen / widen), so a
  difference comes from the choice, not from wording.

## Part A: real CFD, 4 arms x 3 repeats = 12 studies
1. **Ours**
   - `SearchArchive.select_action`: Thompson sampling over
     deepen / widen / new_family, with a family x fit-method archive.
   - Forced new family after 8 evaluations (`exploration_floor`).
   - At least 3 tries per chain (`min_chain_tries`).
2. **Breadth only**
   - `SearchArchive.select_niche`, the previous picker: PUCT over families
     plus a new-family option.
   - Never deepens a chain, which shows what depth buys.
3. **AIDE-style greedy**
   - The first 5 candidates are new families (AIDE's draft count).
   - After that, every slot deepens the best-scoring lineages, best first,
     with no exploration bonus.
   - AIDE's debug step is already in every arm through the repair loop.
4. **Islands (FunSearch / OpenEvolve style)**
   - 3 islands, seeded by the first new families.
   - Each slot takes the next island in turn, samples a parent weighted by
     score rank within that island, and deepens it.
   - Every 9 scored candidates, the worst island is wiped and reseeded from
     the best island's best.
   - No new families after seeding.
   - Island count is small because 150 runs is about 18-20 candidates
     (trial 2 averaged about 8 runs per candidate).

Measured per arm:
- final best score;
- best score against solver runs used;
- families opened and chain depth;
- runs lost to failed builds;
- held-out bubble on the final model, scored off-machine like the framework
  comparison.

Three repeats is thin, so report mean and range and make no claim of
significance.

## Part B: setting ablations on a simulated search, no CFD
- Simulated study:
  - families with a hidden true quality;
  - refinements whose gains shrink each step, plus noise;
  - a set build-failure rate;
  - per-candidate cost and failure rate taken from real runs (SD7003 trials
    1-3 and the closure runs).
- 300 simulated studies per setting, budget 150, one setting changed at a time
  from the defaults:
  - `exploration_c` 0.1 / 0.3 / 1.0
  - `stale_halflife` 1 / 3 / 6
  - `exploration_floor` off / 4 / 8 / 16
  - `min_chain_tries` 1 / 3 / 5
  - `population_size` 1 / 3
  - fit-method axis (`strategy_transfer`) off / on
  - batch size (`num_candidates`) 1 / 3 / 4 / 8
- Run the 4 Part A arms in the simulator too. Trust the setting results only if
  the simulator ranks the arms the way real CFD does.
- A setting with a big effect may get one real CFD arm later, but only after
  asking the user.

## Not in this batch (real CFD only, ask first)
- Proposer without the score history and worst-case feedback.
- No repair loop.
- No literature stage.
- Best-of-N (AI Scientist v1 style); the user did not pick it.
- AI Scientist v2's best-first tree is close to arm 3.

## Build list
- A `CFD_SCIENTIST_SEARCH_METHOD` switch = ours | breadth | greedy | islands,
  with tests (scripts/test_oed_search_archive.py).
- Start a study from a given checked mesh and baseline, skipping the mesh gate.
- The simulator. The "200 synthetic campaigns" mentioned in
  oed_search_archive.py comments were never saved to the repo, so it must be
  rebuilt.
- An `experiments_for_paper/search_ablation/` folder next to
  `framework_comparisons/`.

## Scheduling
- About 5 studies fit on the free cores while the SD7003 trials run; 12
  studies is about 3 rounds, roughly 2 days.
- Once the trials finish, 8 fit at once.

# Search-knob ablation on SD7003 (plan, 2026-11-12; built and piloting)

Built in /mnt/sda1/somasn_cfd_scientist_experiments/experiments_for_paper/search_ablation.
Changes from the first draft, found while building:
- Start point: trial 3 ran up to 13 candidates at once, so knob-driven picks began
  before candidate 7 was recorded. The clean start is instead the 13 OPENING candidates
  (bg_001..bg_013: the first three proposal calls, all made with an empty archive),
  all scored, 108 solver runs, best 5.159. Manager conversation forked at checkpoint
  1f1dc3bc-b31a-6dc9-8049-e257ab4f1360 (right after oed_setup_search, 30 messages).
- Search budget: search_config total_budget = 108 + 150 = 258; outside ledger cap 150
  (trial 3's ledger and the search's own count agreed exactly, 383 = 383). The
  conversation's text, including "500 solver runs", is left as trial 3 had it.
- Batch size is not a fixed setting in the code: the manager picks it per call (1-5 in
  trial 3). Default = manager's choice; B1/B3/B6 force it (CFD_SEARCH_BATCH).
- Not all at once: 8 studies at a time, 16 logical CPUs each (trial 3's share), queue
  of 117 ordered seed 1 of every setting first. Same total time, no oversubscription.
- Disk: shared base/ (1.6 GB) hard-linked into every study (379 MB own each);
  check_base.py verifies no study wrote into a shared file.
- Patch verified: with no override, patched and original search made identical
  choices in 1800/1800 replays of trial 3's history.


Replaces Part B (simulator) of search_ablation_plan.md, which the user rejected.

## Starting point (shared by every study)
- SD7003 kOmegaSSTLM trial 3, CFD Scientist arm, GPT-5.6-sol at effort none:
  framework_comparisons/sd7003lm_trial_3/arms/cfd_scientist/workspace
- Start after the 13 opening candidates (see the header): all new families from an
  empty archive, all scored (best 5.159 from baseline 7.475), 108 solver runs used.
- Literature, hypotheses, mesh gate (60k level), baseline and metric are kept
  as they are. Nothing before the fork is re-run.
- The search has made no choice yet at that point: with an empty archive the
  only move is "new idea", so no knob has acted.

## Code
- New frozen copy of cfd_scientist_code_20261031 (the code trial 3 ran; its
  search code is identical to the repo today). Two changes only:
  1. Knob override: CFD_SEARCH_KNOBS='{"name": value}' read by SearchArchive.
     The hidden constants (K=2, a=1, r_min=4, f=0.10) become named attributes.
  2. Seeded sampler: CFD_SEARCH_SEED; select_action gets
     random.Random(hash(seed, iteration)). Today it is unseeded.
- Test: with no override, a replay of trial 3's history gives the same
  decision distribution as the unpatched code.

## Studies: one knob changed per study, 3 repeats (seeds 1-3)
Defaults in brackets.
| knob | values |
|---|---|
| m  min_chain_tries | 1, [3], 5 |
| c  _max_step_gain | 0.03, [0.10], no cap |
| P  _lineage_prior | 0, 1, [3], 10 |
| W  _gain_weight | 0, 20, [60], 180 |
| F  _failure_weight | 0, 1, [2.5], 5 |
| S  _stall_weight | 0, [1], 3 |
| K  deepen-draw weight | 1, [2], 5 |
| a  widen/new prior | 0.5, [1], 3 |
| E  exploration_floor | off, 4, [8], 16 |
| f  new-family reserve | off, 5%, [10%], 20% |
| C  exploration_c | 0.1, [0.3], 1.0 |
| h  stale_halflife | 1, [3], 6, off |
| T  strategy_transfer | 0, [0.5], 1 |
| V  strategy_prior_visits | [1], 3 |
| population_size | 1, [3], 5 |
| num_candidates | 1, [3], 6 |
- 38 changed settings + defaults = 39 settings x 3 seeds = 117 studies.

## Fork procedure (per study)
- Copy the workspace: checkpoints and artifacts up to the search, the selected
  mesh level and start case, and the 13 opening candidate folders.
- history.json keeps the 13 opening entries only. proposals.json is filtered to them;
  search_status.json and later candidate folders are dropped.
- Absolute paths inside the copied JSON files are rewritten to the new study
  folder by walking the parsed JSON (no text substitution).
- The run restarts at the search stage. To confirm before building: how
  cfd_cli.py resumes from checkpoints/*_done.json; add a resume entry if needed.

## Budget and enforcement
- 150 solver runs after the fork, counted by the trial harness's ledger daemon
  (its own ledger per study, cap 150). Safety stops only: 48 h wall clock, or 3 h
  after the 150th run if the search has not closed.
- Same prompt, model, effort, agents, scorer for all. No nudges.

## Measured per study
- Best score and best-so-far against solver runs.
- Moves taken (deepen / widen / new idea), chains opened, chain depth.
- Failed builds and runs lost.
- First iteration where the study's path differs from the default's with the
  same seed.
- Final model on 44k and the held-out bubble, scored off-machine afterwards.

## Cost and risks
- About 117 x 150 runs x 6.5 min x 7 cores = 13,000 core-hours: about 4 days
  on 128 cores, however the launches are staged.
- 117 Codex sessions on one login at once may hit rate limits; the launcher
  should stagger starts and the run must tolerate retries.
- Codex answers still differ run to run; the 3 seeds are for that. Report mean
  and range, no significance claims.
- Disk: 117 copies of the start; check size before launching.

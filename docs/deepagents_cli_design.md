# AI CFD Scientist — deepagents CLI: design doc

This documents the deepagents-based CLI (`python scripts/cfd_cli.py run`) and everything
built into it in this development cycle: the manager/subagent architecture, interrupt and
caching behavior, the full pipeline tool wiring, the FoamAgent native port, and the
open-ended discovery search-policy rewrite (from a single greedy loop to a family-niched
quality-diversity search that runs inside the same graph). It also lists the concrete
bugs found and fixed along the way, since several of them affect correctness in ways that
aren't obvious from reading the code alone.

This CLI is one of two ways to run a study in this repo — the other is the older LangGraph
pipeline (`scripts/orchestrator_run.py`, "Mode A") and the markdown skill recipes under
`cfd-skills/` ("Mode B"). This doc is scoped to the deepagents CLI only. See the top-level
`README.md`/`CLAUDE.md` for how the three relate.

## 1. Why deepagents

The CLI is built on the `deepagents` library on top of LangGraph, not a hand-rolled agent
loop, specifically for three properties a plain script can't give you:

- **Real interrupt/resume with zero information loss.** A user can hit Ctrl-C mid-study
  and get a clean pause, not a killed process.
- **Prompt caching** on the large, mostly-static system prompt + tool-definitions block,
  for providers that support it.
- **Concurrent, context-isolated subagents.** Independent work (running several
  experiment cases, or several open-ended-discovery candidates) can run as real parallel
  `task` calls instead of a serial Python loop, each with its own clean context so the
  manager's own context doesn't get flooded with a subagent's internal chatter.

A recurring theme in this doc: every one of these three properties only applies **between
tool calls**, never *during* one. That fact drove a real architecture correction partway
through this cycle (§7) — a feature that ran as one giant blocking subprocess call got
none of the above, even though it lived inside the same CLI.

## 2. Architecture

```
cfd_cli.py run
  -> src/cfd_langgraph/cli/repl.py        (REPL: prompt, SIGINT handling, resume)
  -> src/cfd_langgraph/manager/deep_agent.py   (build_manager: the top-level graph)
       - manager_tools                    (pipeline stages, scoped filesystem, OED setup)
       - subagents=[case-runner, oed-candidate-runner]
  -> src/cfd_langgraph/manager/tools.py   (every tool's actual implementation)
  -> src/cfd_langgraph/manager/subagents.py    (SubAgent definitions + their prompts)
  -> src/cfd_langgraph/manager/control.py      (the interrupt mechanism)
```

**The manager** is one `create_deep_agent(...)` graph: a single model instance, a system
prompt describing the whole study sequence, the pipeline tools, and two subagents it can
dispatch work to via the `task` tool. Experiment cases and open-ended-discovery
candidates are always delegated to a subagent; the manager reads only the subagent's
short final report.

One deliberate exception: `run_mesh_gate` is a **manager** tool and runs its baseline
case plus every refinement level in-process, sequentially, because each level's
refinement is derived from the previous one's result — there is nothing to fan out. It
is therefore a single long-running tool call, with no pause point between levels; a
Ctrl-C during it lands after the whole gate finishes. It is idempotent: a group whose
gate already converged returns the existing `selected_mesh_spec.json` instead of
re-running hours of OpenFOAM.

**`case-runner`** (`subagents.py: build_case_runner_subagent`) runs exactly one
experiment case: FoamAgent's parse → decompose → write → Allrun → review/retry loop,
via this workflow's own native port (`src/cfd_langgraph/foam_native/`, §6). The manager
launches one `task` call per case, as many concurrently as it wants — the real
hardware-safe concurrency cap is enforced underneath by `CaseCoordinator`
(`scheduling/`), not left to the model to self-limit.

`CaseCoordinator` calibrates the first case of each physics group exclusively. CPU
percentage is converted back to logical-core equivalents, and the effective study-wide
limit is the minimum of the calibrated group limits. A global condition gate prevents
different physics groups from each consuming a full-machine allowance independently;
the same global gate also applies to the first case when the user supplies a fixed cap.

**`oed-candidate-runner`** (added in §8) runs exactly one open-ended-discovery
candidate — compile/run a proposed model modification, or re-run an existing one with
new coefficients, then score it — the same pattern, for the same reason.

Both subagents get: their own cached model instance
(`build_caching_middleware(model)`), the same Ctrl-C interrupt coverage as the manager
(`build_interrupt_on`), and `DENY_BUILTIN_FILESYSTEM_TOOLS` — see §3 and §4.

## 3. Interrupt / resume

`control.py`'s `InterruptFlag` (module-global `GLOBAL_INTERRUPT`) is set by the CLI's
SIGINT handler on Ctrl-C. Every tool on the manager and both subagents is wired via
`build_interrupt_on(tools)` to check this flag through deepagents' `InterruptOnConfig.when`
— which fires **before** a tool call starts, never mid-call. That's a deliberate,
load-bearing design choice, not a limitation worked around elsewhere: LangGraph
checkpoints at completed steps, so the only way to guarantee nothing partial is ever lost
is to never interrupt mid-tool-call, only ever pause before the next one. A second Ctrl-C
while paused force-quits.

State survives a pause via a `SqliteSaver` checkpointer
(`<out_dir>/state/checkpoints.sqlite`) — `resume --out-dir <dir>` continues exactly where
a study left off, including across a CLI process restart, not just within one session.

**This only works as long as the unit of work actually IS a tool call the graph knows
about.** §7 covers what happens when it isn't.

## 4. Filesystem access, and why deepagents' own tools are blocked

The manager and both subagents have real disk-backed read tools — `list_directory`,
`directory_tree`, `read_text_file`, and `grep_files` (`find_files` is manager-only).
General-purpose writes are code-enforced to descendants of the current `<out_dir>`.
Authoritative artifacts can be written only by their owning pipeline tool: the full list
is `_PROTECTED_ARTIFACT_NAMES` in `manager/tools.py` (24 filenames — `run_result.json`,
`decision.json`, `candidate_record.json`, `requirements.json`, `history.json`,
`audit_passed.json`, `state.json`, …), plus anything under a `checkpoints` path
component. The same list is enforced on **both** write paths — `_writable_path` for the
general write/edit tools and `_safe_case_path` for `foam_write_case_file` — since
several of those files live inside a case directory, and a case-scoped writer that
skipped the check was a way to forge a successful run record for a case that never ran. OED candidate
subagents receive no general write or shell tool at all. This keeps exploratory agent
work useful without allowing it to forge a successful run, score, checkpoint, or audit.

deepagents ships its own built-in `ls`/`read_file`/`write_file`/`grep`/etc., wired to
whatever `backend=` is configured. This CLI never passes one, so those built-ins would
silently operate against an empty in-memory `StateBackend` — worse than absent, because a
call like `grep` against nothing returns a clean "no matches found" instead of an error,
which reads as a real (if unhelpful) answer rather than the decoy it is.
`DENY_BUILTIN_FILESYSTEM_TOOLS` (a deny-all `FilesystemPermission` on every path) forces
every model onto the real, disk-backed tools instead — an explicit permission error
instead of a silently wrong one. This was found by noticing a tool-call line in a live
transcript with no preceding `▶` progress marker — proof it wasn't one of this
workflow's own tools.

## 5. Standard study sequence (manager tools)

The manager's system prompt (`deep_agent.py: _build_manager_system_prompt`) walks it
through: `read_starter_folder` (if given a starter/base-case path) → `fetch_literature` →
`propose_and_rank_hypotheses` → **human approval gate** (`advance_with_approved_hypotheses`,
always interrupts regardless of the Ctrl-C flag — approve/edit/reject) →
`generate_case_requirements` → `run_mesh_gate` per physics group (baseline case, then a
chain of refined meshes, stopping at the first LLM-judged-converged level) → launch every
case concurrently via `task`/`case-runner` → `interpret_case` per case →
`analyze_all_cases` → `write_paper` → `run_audit_and_record` (stage-gate audit, then
records the study into the knowledge bundle on pass).

For an open-ended-discovery topic ("find a novel model that beats baseline by X%"),
`generate_case_requirements` and the mesh gate still run — the search needs the gate's
locked selected level and cannot invent its own baseline. Only the case-launch and
per-case interpretation steps are replaced, by the loop in §8.

Every artifact this study produces — including anything the manager creates itself
outside the named tools, e.g. a scratch test or a custom-model source file — is scoped to
`<out_dir>` by both the system prompt and path validation, never the repo root or `/tmp`,
both of which are shared across every study that has ever run.

### Literature-grounded hypothesis generation

`fetch_literature` persists the exact Semantic Scholar result set in `lit.json`, and
`propose_and_rank_hypotheses` passes that same set into the ideation pipeline; it does not
perform a second, potentially divergent literature fetch. Records are normalized so the
`abstract` field emitted by `scripts/lit.py` becomes the ideation `snippet`, DOI/URL
provenance is retained, and duplicate papers are removed.

`scripts/lit.py` derives several compact, complementary keyword queries after removing
generic task-intent words. The first retrieval pass allocates the paper budget across all
query variants before any one query can exhaust it; a second offset-based fill pass
recovers capacity lost to duplicates or sparse result sets. DOI/title deduplication and a
hard global limit are applied across the combined result set.

Each generated candidate is rejected if it is too similar to the persisted literature
or another candidate in the same batch, then passes through a fail-closed physical-
plausibility and feasibility critique. Only novelty- and critique-approved candidates
are ranked. Human approval rejects unknown or empty candidate IDs, and invalid case
requirements remain a draft rather than being published as executable input.
Malformed novelty-evaluator output also fails closed when literature is present; a
lexical fallback is not treated as semantic proof of novelty. Candidates with zero or
too many experiments never reach critique/ranking.

### Gemini / Vertex AI support

`llm/factory.py`'s `GeminiChatModel` wraps `ChatGoogleGenerativeAI` with fixes needed for
this harness specifically: a `bind_tools` override (missing on the base class), and
`_generate` content handling that preserves `tool_calls` and `additional_kwargs` (Gemini
3.x's "thought signature," required for multi-turn tool-calling — dropping it produces an
"Invalid thought signature" API error on the next turn). Provider selects via
`CFD_SCIENTIST_LLM_PROVIDER=gemini` + `GOOGLE_GENAI_USE_VERTEXAI=true` +
`GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION`, or a bare API key without Vertex.

## 6. FoamAgent native port

`src/cfd_langgraph/foam_native/` ports FoamAgent's parse → RAG → decompose → write →
Allrun → review/retry loop as first-class Python inside this workflow
(`loop.py: run_foam_case`), rather than depending on the vendored `Foam-Agent/` package's
own Python service layer at runtime. Two fixes worth knowing about:

- **OpenFOAM environment resolution** (`openfoam_env.py: resolve_openfoam_env`) —
  memoized, sources `$OPENFOAM_PATH/etc/bashrc` via a `bash -c "... && env -0"`
  subprocess trick and parses the result, instead of assuming the launching shell already
  had OpenFOAM sourced. Every subprocess that runs `blockMesh`/`simpleFoam`/`wmake` etc.
  in this codebase goes through this now.
- **Stale-log retry bug** (`loop.py: _clean_stale_logs`) — OpenFOAM's `runApplication`
  refuses to rerun a step whose log file already exists from a prior attempt (prints
  `"... already run ... remove log file to re-run"` and exits 0, silently). Without
  clearing `log.*` between retry-loop iterations, a rewritten file (e.g. a reviewer-fixed
  `blockMeshDict`) never actually got re-exercised — every retry silently no-op'd through
  every step that got as far as writing a log file on a prior attempt, and the loop just
  spun to `max_loop` reporting "failed" even when the fix was correct. This was the
  single highest-impact bug found this cycle: it was the reason an entire prior study run
  (9 cases) failed 0-for-9 despite the underlying physics setup being fine.
- **`refine_mesh_from_parent`** — mesh-gate's refined levels copy the parent case's real
  input files and edit only `system/blockMeshDict`, rather than writing a refined case
  from scratch (which can't guarantee identical physics/BC/solver settings to the
  baseline it's supposed to be mesh-independence-checking against).
- **Selected-mesh enforcement** — a case cannot launch without a converged mesh-group
  selection. `run_case_native` copies the selected level's `blockMeshDict` into the
  generated case before `Allrun`, so passing the gate is not merely bookkeeping: every
  downstream simulation actually uses the accepted mesh.

## 6b. Prompt caching, in two halves

Caching is worth being precise about, because the obvious wiring only covers a minority
of the spend.

**Agent turns** — `build_caching_middleware(model)` attaches the provider's official
middleware (`BedrockPromptCachingMiddleware` / `AnthropicPromptCachingMiddleware`) to the
manager and to both subagents. That caches the large static prefix each agent resends
every turn: system prompt plus ~15-20 tool definitions. It is a no-op on providers
without a wired middleware (Gemini/Vertex, OpenAI — the latter caches server-side
anyway).

**The FoamAgent stages** — these are *not* agent turns. They run on the separate
`foam_llm` instance and call `llm.invoke(...)` directly, which no middleware ever sees,
and they are where the volume is: one write call per case file, plus a reviewer call per
retry round (up to `max_loop`), each re-sending the same retrieved tutorial reference.

They are cached instead by `cacheable_human_message(model, stable_prefix, tail)`, which
returns the same message split into content blocks with a provider-native breakpoint
between them — `{"cachePoint": {"type": "default"}}` for Bedrock's Converse API,
`cache_control: {"type": "ephemeral"}` on the prefix block for Anthropic, and a plain
string (no blocks) for everyone else. The blocks are concatenated by the API, so **the
model reads exactly the same characters in the same order as the single formatted string
this replaced** — a property pinned by test, for all three dialects, against the flat
template.

Two consequences worth knowing:

- The split points are the headings already in the templates
  (`WRITE_CACHE_SPLIT_MARKER`, `REVIEW_CACHE_SPLIT_MARKER`): the requirement and the
  tutorial reference sit before them and repeat verbatim across a case; the growing
  written-files context, foam files, and error logs sit after and change every call.
- A breakpoint only hits if *everything* before it is byte-identical, system prompt
  included. That is why the two write system prompts no longer interpolate the target
  file name: it is stated in the user message instead — which already named it
  explicitly — leaving the system prompt reusable for every file in the case. No
  instruction was dropped or reworded, and a regression test fails if per-file text
  reappears there.
- Below ~4096 characters the breakpoint is skipped entirely, since it is under the
  provider's minimum cacheable prefix and would never be honoured.

Still uncached: the subprocess runners, which are separate processes with their own
models and no reused conversation.

## 7. Open-ended discovery, part 1: from greedy to quality-diversity search

### The problem

A reviewer characterized the discovery loop as "a greedy LLM hill-climbing agent: proposes
one edit per iteration, promotes only when the score beats the unmodified baseline."
Investigation of `scripts/open_ended_discovery.py` (the ~4800-line standalone script this
loop lived in) confirmed the core criticism: a single linear `while budget_used < budget:`
chain with **no lineage tracking at all** (zero `parent_iteration`/`based_on` fields
anywhere) and no principled selection policy — "diversity" was a fixed-period nudge
(`decide_search_mode`'s `period = 1/far_ratio`) independent of how well any direction was
actually doing. (Two specific reviewer claims were *not* accurate, worth knowing for any
rebuttal: promotion was already baseline-gated, not current-best-gated, and proposals
already were history-conditioned — see `oed_extensions.py`.)

### Literature grounding

- **AutoTurb** (Fu et al., arXiv:2410.10657 / *Physics of Fluids* 2025) — LLM-driven
  algebraic turbulence-closure discovery for periodic-hill flow, the same benchmark this
  study uses, via island-based evolutionary search with complexity/convergence
  constraints on fitness. Domain-identical validation.
- **CodeEvolve** (arXiv:2510.14150) / the FunSearch → AlphaEvolve lineage — LLM-as-
  mutation-operator inside an island model with a CVT-MAP-Elites quality-diversity
  archive. Open-source reference implementation.
- **Aygün et al.** (*Nature* 654:909-916, 2026) — predictor + PUCT tree search, with
  score-saturation-after-N-nodes as the stopping signal. Contributed the UCB-style
  selection formula and the saturation-stopping idea, *not* the tree/backprop machinery
  (see below).
- **Toledo et al. / MLE-bench** (arXiv:2507.02554, Meta FAIR) — Greedy vs. MCTS vs.
  Evolutionary search policy comparison. Its finding — greedy isn't *always* worse, it
  depends on budget — is the actual justification for *not* over-building this: a full
  tree-search system would be over-engineering relative to what a ~10-30-evaluation
  budget can pay for.

**Deliberately not adopted**, and why: separate islands + migration (CodeEvolve/FunSearch
scale — needs hundreds-to-thousands of cheap evaluations to pay for itself; a real CFD
study gets ~10-30 expensive ones), CVT-MAP-Elites (the niche descriptor here is already a
handful of discrete, named model families — no continuous-space clustering needed to find
them), and full PUCT tree/backprop bookkeeping (the "tree" here is effectively depth-1:
each niche's elite is directly mutated, not built up through generations of within-niche
branching, given tiny budgets).

### `SearchArchive` (`scripts/oed_search_archive.py`)

One elite (best-scoring history entry) per model family — the family comes from
`oed_extensions.classify_family`, reused unchanged, a cheap keyword-based classifier
(`SA-RC`, `SA-APG`, `SA-Production`, `k-omega-SST`, etc.):

- `update(family, iteration, score, direction, history_entry)` — records one real
  evaluation; a failed/unscored attempt still counts as a visit (discourages repeatedly
  hammering a family that keeps failing) but cannot create false score saturation.
- `select_niche(budget_remaining, budget_total)` — PUCT-style: exploit a niche scoring
  well, don't starve one that's under-visited; an explicit "propose a brand-new family"
  option competes on equal footing with a neutral prior, so it can win once every known
  niche is well-explored or unpromising. The exploration bonus decays as the remaining
  budget shrinks, so the policy naturally shifts toward exploitation near exhaustion.
- `is_saturated(window)` — true if the archive-wide best score hasn't improved over the
  last `window` real evaluations — a genuine plateau detector, not a vibe check.
- `replay(history)` — reconstructs archive state from a resumed `history.json`, so a
  paused/resumed study doesn't lose its exploration progress.
- `render_summary(...)` — one line per family, bounded by niche count, not iteration
  count — this also fixed an unrelated pre-existing issue where the decision prompt's
  `_compact_history` rendered *every* history entry into every LLM call with no cap on
  entry count, growing unbounded over a long run.

**A real calibration bug, found by direct testing, not inspection:** the first version
compared a raw score (e.g. a Cf RMSE of 0.001–0.01) directly against an O(1) exploration
bonus. Proven with a scripted test: a niche with 20 real evaluations and an excellent
score *still lost to "try something brand new," every time*, regardless of visit count —
because the score's raw magnitude was invisible next to the bonus. Fixed two ways: min-max
normalize scores into `[0, 1]` across the niches being compared before adding the
exploration term, and recalibrate the exploration constant (`1.4 → 0.3`) so exploitation
can actually win once a niche has earned it. Verified after the fix: early exploration
still correctly favors untried families when evidence is thin; a strong, well-established
niche correctly wins once it has real evidence behind it. Regression-tested in
`scripts/test_oed_search_archive.py` (22 checks, offline, no OpenFOAM/LLM calls needed).

This is the search-policy engine both integration points below use.

## 8. Open-ended discovery, part 2: bringing the loop into the deepagents graph

### The problem this section fixes

The first integration wired `SearchArchive` into `open_ended_discovery.py`'s own loop and
exposed it as a single manager tool (`run_open_ended_discovery`) that shelled out to the
whole script as one blocking subprocess call. That got the smarter search *policy*, but
none of §1's three deepagents properties applied to it, because the entire multi-hour
loop was one tool call:

- **Interrupt was dead for the whole run** — Ctrl-C set the flag, but nothing checked it
  again until the process finished or hit its 6-hour cap.
- **No prompt caching** — the script called `create_langchain_llm(...)` fresh at each of
  its own call sites, never the graph's cached model instance.
- **No real concurrency** — even its `code_mod_batch` action ran multiple variants in a
  plain sequential `for` loop, not through `CaseCoordinator`.

### The fix

The outer control loop (which candidate to try next, when to stop) now lives in the
manager's own reasoning — a sequence of real tool/`task` calls, the same pattern the
case-launch flow already used — while every per-candidate execution mechanism stays
exactly as it was (they already shelled out to standalone, tested scripts, which don't
need to change):

- **`oed_setup_search(topic, baseline_case_dir, total_budget, starter_dir)`** — one
  bounded (few-minute) call to `open_ended_discovery.py --setup-only` (a new flag added
  to the existing script): runs the same tested comparator-authoring/discovery setup the
  script always did, then returns before the iteration loop instead of entering it.
  It accepts `baseline_case_dir` only when that resolved path exactly matches a converged
  `selected_level` produced by this study's mesh gate, then computes and persists the
  baseline score against that locked case.
- **`oed_propose_candidates(topic, num_candidates)`** — replays the archive from
  `history.json`, picks `num_candidates` niches (with an in-batch provisional visit-bump
  so a batch doesn't blindly repeat one pick — though a niche that has genuinely pulled
  decisively ahead can still legitimately win every slot; that's correct exploitation,
  not a diversification bug), then one structured-output LLM call proposing that many
  concrete candidates, each conditioned on its niche's elite.
- The manager launches every returned candidate as a **`task` call to
  `oed-candidate-runner`, concurrently, one message** — real `CaseCoordinator`
  concurrency, real per-candidate interrupt coverage, real per-subagent caching.
- **`oed-candidate-runner`** (mirrors `case-runner`): `oed_run_code_mod_candidate`
  (subprocess to `scripts/code_mod_agentic.py`, which self-resolves OpenFOAM env per call
  and executes its shell in a read-only bubblewrap mount namespace where only the
  candidate run directory is writable) or
  `oed_run_experiment_candidate` (the runtime-model path only — copy a compiled case,
  patch `constant/fvModels` coefficients, run `scripts/foam_run_simple.py`, which also
  self-resolves env); then `oed_score_candidate`, which loads the study's
  `bound_comparators.json`/`objective_contract.json` once (authored/discovered exactly
  once per study, reused for free — see `compute_metric_vector`,
  `scripts/oed_extensions.py`), computes a real score, and **writes
  `candidate_record.json` to disk itself** rather than relying on the subagent to
  transcribe a score into its final report.
- **`oed_record_candidate_results(candidate_dirs)`** — reads each `candidate_record.json`
  back by path (not by trusting prose), appends to `history.json`, replays the archive,
  returns `budget_used`/`proceed_count`/`is_saturated`/the archive summary — what the
  manager uses to decide whether to propose another round or move on to
  `interpret_case`/`analyze_all_cases`/`write_paper`.

Setup also locks the converged mesh-gate case, objective metric, optimization direction,
verified baseline score, and any percentage target parsed from the topic into
`search_config.json`. Scoring fails closed when execution, convergence, comparator,
baseline, or target evidence is missing. Recording is idempotent across resumes. When a
winner exists, the tool promotes the locked baseline and valid evaluated candidates into
ordinary `cases/case_*` directories, writes the manifest/requirements bridge, and signs
the OED checkpoint so the standard interpretation, analysis, paper, and audit stages can
consume them without trusting subagent prose.

The code-mod artifact validator accepts only a fresh ELF shared library under the
candidate case's `customModels/` tree and a clean log for the exact application named in
`controlDict`. `$FOAM_USER_LIBBIN`, the OpenFOAM installation, arbitrary external case
paths, and hand-written `lib*.so`/generic `log.*` files cannot satisfy the gate.

Deliberately **not** ported: the legacy class-derivation experiment path
(`scripts/foam_run.py`), which requires `$WM_PROJECT_DIR` to already be a non-empty env
var with no fallback and pulls in the vendored Foam-Agent package — every code_mod path
already defaults to the agentic/runtime route, so this loses nothing in practice.

### A real bug found in the first live run of this path

The first real run produced two candidates that both compiled and ran to convergence —
the concurrency/subagent mechanics worked — but scoring silently failed:
`oed_setup_search` accepted a `baseline_case_dir` parameter but never actually passed it
through as `--base-case-dir` to the setup subprocess, so the metric-proposer LLM call had
no real postProcessing sample to look at and proposed zero metrics; `bound_comparators.json`
never got created. Given a null score from `oed_score_candidate`, the candidate-runner
subagent — resourceful, with real filesystem tools — read the candidate's own
self-produced `comparison_exactmatch/summary.md` (something the agentic code-mod runner
generates as part of its own self-check) and **hand-wrote `candidate_record.json` itself**,
bypassing the tool that owns that file. Caught by an internal contradiction the real tool
would never produce: `"baseline_score": null` alongside `"baseline_verified": true` — a
result presented as a verified beat-baseline comparison when no baseline had actually been
established. Fixed both ends: `oed_setup_search` now passes `--base-case-dir`, and
`oed-candidate-runner`'s system prompt now explicitly forbids writing or editing
`candidate_record.json` under any circumstance, including "the tool gave me a null score"
— a null score is meant to be reported honestly, not patched over.

Both candidates from that run, once scored against the study's actual known baseline
directly, turned out not to beat it anyway (one was statistically identical to baseline,
one marginally worse) — consistent with every SA-modification attempt across this whole
project so far: RMSE stuck around 0.0043, reattachment length stuck around 3h too long
relative to DNS truth regardless of which term gets tuned. The production/curvature-term
family of modifications looks plateaued; a destruction-side, rotation-vs-strain-gated
mechanism has not yet been tried through this pipeline.

## 9. File map

| File | Role |
|---|---|
| `src/cfd_langgraph/cli/repl.py` | REPL, SIGINT handling, prompt-hint extraction, resume |
| `src/cfd_langgraph/manager/deep_agent.py` | Top-level graph, manager system prompt |
| `src/cfd_langgraph/manager/subagents.py` | `case-runner`, `oed-candidate-runner` |
| `src/cfd_langgraph/manager/tools.py` | Every manager/subagent tool implementation |
| `src/cfd_langgraph/manager/control.py` | `InterruptFlag`, `build_interrupt_on`, filesystem-tool deny rule |
| `src/cfd_langgraph/foam_native/` | Native FoamAgent port (parse/decompose/write/Allrun/review) |
| `src/cfd_langgraph/llm/factory.py` | Provider-agnostic model factory, Gemini fixes |
| `src/cfd_langgraph/llm/caching.py` | Per-provider prompt caching: agent middleware + in-message cache breakpoints |
| `scripts/open_ended_discovery.py` | Standalone OED script (used directly by `orchestrator_run.py`; `--setup-only` also used by the deepagents path) |
| `scripts/oed_search_archive.py` | `SearchArchive` — shared by both OED integration points |
| `scripts/oed_extensions.py` | Metric proposal, comparator authoring/discovery, `classify_family` |
| `scripts/code_mod_agentic.py` | Sandboxed, case-local agentic OpenFOAM code-mod runner (compile+run one candidate) |
| `scripts/test_oed_search_archive.py` | Offline `SearchArchive` regression tests |
| `scripts/test_deepagents_mechanisms.py` | Offline hypothesis, safety, scoring, resume, and OED-bridge regression tests |

## 9b. Correctness fixes from the review pass

An independent review of this codebase against this document surfaced a set of real
defects; all are now fixed, each with a regression test that was verified to fail when
the fix is reverted (`test_deepagents_mechanisms.py`, `test_oed_search_archive.py`).

| Defect | Fix |
|---|---|
| `oed_score_candidate` silently scored on a *different* metric when the baseline's metric was missing from a candidate's vector, then compared it to the baseline anyway — reproducibly reporting an 80% "improvement" from comparing an L2 norm to a Cf RMSE | Refuses to score across metrics; returns `INDETERMINATE` with a `score_error` naming the missing metric |
| Stale logs were cleared only *between* retries, so a relaunched case no-opped through Allrun and was read as a clean success off the previous attempt's solver log | `_clean_stale_run_artifacts` runs before the first attempt too, and also clears `processor*/` (fatal for `decomposePar`) and `postProcessing/` (what scoring reads) |
| `classify_family` matched `"rc"` as a bare substring, so `source`/`force` classified as `SA-RC`, while a real "SA with rotation-curvature…" fell through to `unknown` — corrupting the niche identity the whole archive is built on | Whole-token matching; added `SA-Destruction` / `SA-Diffusion` families |
| An opening round proposed N "new family" candidates that all classified identically, burning most of the budget on one direction | In-batch family dedup, variant names deduped against full study history, already-taken families named in the proposal prompt |
| Ctrl-C during a fan-out left one interrupt per subagent; the CLI resumed only the first, raising `RuntimeError` and making the study permanently unresumable | All pending interrupts collected; resume is interrupt-id-keyed and streamed (so post-pause output is no longer discarded). `task` is now interrupt-gated, so a pause lands *before* a fan-out |
| OED candidate runners called their subprocesses directly, bypassing `CaseCoordinator` entirely | Both go through `coordinator.run_case("oed_candidates", …)` |
| A calibration case that died before the first 2 s sample looked like a zero-cost case, producing limits like 435 concurrent cases | `ResourceProfile.measured`; unmeasured profiles fall back to `min(8, cores//4)`, cores-per-case floor raised to 1.0, hard ceiling at logical core count |
| Promotion re-ran on every call, `rmtree`-ing already-interpreted cases and resetting `bridge.json` decisions; it also overwrote the approved `requirements.json` with OED stubs | Promotion skips already-promoted cases, preserves bridge decisions, and merges rather than replaces requirements |
| `run_mesh_gate` had no memory: a manager that re-called it would redo hours of OpenFOAM, and a rejected requirement gave no way to fix it | Returns the existing selection when already converged; rejection now says the text must be copied verbatim and lists the available `case_id`s |
| Candidate lineage (`parent_iteration`) existed only in the standalone script, though the manager prompt claimed the archive tracked it | `oed_propose_candidates` records family + parent into `proposals.json`; `oed_score_candidate` stamps them onto the record from there, not from subagent prose |
| `_compact_history` rendered the entire history into every decision prompt, unbounded | Capped at the most recent 12 entries, with the omitted count stated rather than silently dropped |
| Scheduler held a semaphore slot while spinning on memory pressure, deadlocking the study against calibration exclusivity | Pressure is waited out *before* taking a slot, with a bounded timeout; `_benchmark_exclusive` notifies on its way out even when interrupted |
| The FoamAgent stages — the bulk of per-case token spend — bypassed prompt caching entirely, since agent middleware cannot see a bare `llm.invoke` | In-message cache breakpoints via `cacheable_human_message`, with the write system prompts made file-agnostic so the prefix is reusable (§6b) |
| `compute_metric_vector` fell back to the first `identifier: number` anywhere in a comparator's output, so a comparator that computed nothing still returned a number — `nPoints = 12800` became `cf_rmse` | The fallback now requires a line naming *that* metric; otherwise the metric is reported as an error |
| The retry loop could rewrite `blockMeshDict` on a case seeded from the mesh gate, running the experiment on a mesh no independence study examined while `run_result.json` still cited the certified seed | Reviewer edits to `blockMeshDict` are rejected on gate-seeded cases |
| `claude-code` and `openai-codex` cannot bind tools, so selecting them crashed on the manager's first turn with an opaque traceback | `_require_tool_calling_support` refuses them at build time and names the providers that work |
| A fast double Ctrl-C could take the "pause requested" branch twice and never force-quit; Ctrl-D or piped stdin at a pause crashed the study | Flag is set before any printing; `EOFError` at the pause prompt behaves like `quit` |
| Answering the hypothesis gate left a stray Ctrl-C flag set, silently putting the study into single-step mode | The flag is cleared on approve/edit/reject, with a line saying so |
| `is_saturated` used an absolute 1e-9 epsilon — a 0.1% threshold for an L2 error near 1e-6, stopping the search on real progress | Relative tolerance scaled to the metric's own magnitude |
| An archive mixing `min` and `max` metric directions ranked the max family best by sign alone | The first scored entry fixes the archive's direction; contradicting entries count as visits but never become elites |
| SQLite `-wal`/`-shm` sidecars were writable, and protected names matched case-sensitively | `_is_protected_artifact` covers sidecars, `state/`, and matches case-insensitively |
| `oed_propose_candidates` could return an empty list with no error, letting the manager loop forever without spending budget | Returns an explicit error explaining why every pick was filtered and what to do next |

## 10. Known open items

- The hardened deepagents-native OED path is covered by offline archive, scoring,
  idempotence, and bridge tests, but has not yet completed a new end-to-end live
  OpenFOAM/LLM study. A live run is still required to validate provider behavior,
  generated comparators, compilation, solver convergence, and publication artifacts as
  one integrated system.
- **Prompt caching does not reach the subprocess runners.** `code_mod_agentic.py` and
  `foam_run_simple.py` build their own models in their own processes and hold no
  conversation to reuse across calls. Everything in-process — agent turns and the
  FoamAgent stages alike — is cached (§6b).
- No study run through either OED path has yet beaten baseline Cf by the target margin;
  every attempt so far plateaus around the same RMSE regardless of which model family is
  tuned, which is itself useful signal about where *not* to keep spending budget.

# CFD Scientist — Skill Mode (ARIS/DS-style)

This directory provides a **skill-based, markdown-driven** version of CFD Scientist that runs without the Python orchestrator. It is offered **alongside** the existing orchestrated pipeline (not as a replacement).

After the Nov 2026 refactor, every skill in this directory is **self-contained** in the ARIS / DeepScientist style:

- The agent-driven recipe is the **primary path**. Each skill walks the calling agent through every step end-to-end.
- The expert prompts from `prompts/prompts.yaml` (HypothesisAgent, IdeationAgent, ResultsInterpreterAgent, WriterAgent, PaperReviewerAgent, RunValidityAgent, MetricProposer, ComparatorAuthor, ComparatorVerifier, MetricSetupAgent, MetricSetupVerifier) are **embedded verbatim** in the SKILL files that use them. The YAML remains the single authoritative source; the embedded copies allow each skill to be useful from a different agent framework where that YAML may not be available.
- The OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol (`openfoam_literature_change_agent_prompt_v2.txt`) is embedded verbatim inside `cfd-code-modify/SKILL.md`.
- Scripts are an **optional fast-path** inside each skill, not a primary mechanism. The exception is `cfd-experiment`, which keeps `scripts/foam_run.py` as the actual execution mechanism because that script *is* the FoamAgent framework wrapper — reimplementing it would mean reimplementing FoamAgent (DeepScientist makes the same trade-off with its baseline runner).

The two modes share the same expert prompts (`prompts/prompts.yaml`), the same artifact JSON contracts (`lit.json`, `requirements.json`, `selected_mesh_spec.json`, `decision.json`, `analysis.json`, etc.), and — where helpful — the same Python helper scripts (`scripts/*.py`). Pick whichever mode fits your workflow, or mix them.

---

## Three modes

### Mode A — Python orchestrator (default for unattended runs)
Run the full pipeline end-to-end with checkpointing, resume-from-stage, timeline logging, automatic provider/model selection.

```bash
conda activate cfd-scientist
python scripts/orchestrator_run.py \
  --topic "..." \
  --out-dir runs/my_study \
  --provider claude-code --model claude-sonnet-4-6 \
  --starter-dir starter
```

Use this when you want one command to do everything, automatic resume on failure, and hands-off long runs. See top-level `README.md` and `AGENTS.md`.

### Mode B — Skill-driven (this directory)
Invoke individual skills from any LLM agent (Claude Code, Cursor, Codex CLI, custom). Each skill is a `SKILL.md` markdown contract with its expert prompts embedded — no Python orchestrator state. Memory lives in the run directory's artifact files.

```text
/cfd-pipeline "topic"            # top-level chain (analog to orchestrator)
/cfd-literature                  # one stage at a time
/cfd-mesh-gate
/cfd-experiment
/cfd-paper
...
```

Use this when you want manual control, are integrating into another agent framework, or want to run only part of the pipeline ad-hoc.

### Mode C — Hybrid
Run Mode A for the main pipeline; invoke skills (Mode B) ad-hoc for one-off tasks against the same `out-dir`. Because both modes read/write the same JSON contracts, you can:

- Resume from any orchestrator checkpoint with a skill (e.g., the orchestrator stopped at `analysis_done`; invoke `/cfd-paper` against the same `out-dir`).
- Use a skill to hand-craft one stage's output, then let the orchestrator pick up from there with `--resume-from <next_stage>`.
- Use the orchestrator for long unattended runs; use skills for interactive exploration on the same artifacts.

---

## Skills in this directory

| Skill | Purpose | Embedded prompts (from `prompts/prompts.yaml`) |
|---|---|---|
| [`cfd-pipeline`](cfd-pipeline/SKILL.md)             | Top-level router; chains sub-skills end-to-end | `MetricSetupAgent.metric_setup_*`, `MetricSetupVerifier.*` |
| [`cfd-literature`](cfd-literature/SKILL.md)         | Semantic Scholar lit retrieval → `lit.json` | (HTTP recipe — no LLM prompt) |
| [`cfd-hypothesis`](cfd-hypothesis/SKILL.md)         | Generate testable hypotheses → `hypotheses.json` | `HypothesisAgent.hypothesis_system_prompt`, `hypothesis_user_prompt` |
| [`cfd-requirements`](cfd-requirements/SKILL.md)     | FoamAgent requirement strings → `requirements.json` | `IdeationAgent.initial_idea_prompt`, `literature_aware_user_prompt`, `novelty_retry_user_prompt`; reuses `HypothesisAgent.*` for expansion |
| [`cfd-mesh-gate`](cfd-mesh-gate/SKILL.md)           | Per-physics-group mesh independence study (mandatory) | (STEPS A–E protocol, embedded verbatim) |
| [`cfd-experiment`](cfd-experiment/SKILL.md)         | Run one FoamAgent case → `case_dir/run_result.json` | `RunValidityAgent.allrun_preflight_*`, `investigate_runtime_*`, `decide_action_run_invalid_user_fragment` |
| [`cfd-code-modify`](cfd-code-modify/SKILL.md)       | Custom OpenFOAM model build (case-local) | OPENFOAM 10 LITERATURE CHANGE AGENT v2 (full §1–§16 embedded); `compile_fix_*` (wmake variant inline) |
| [`cfd-open-discovery`](cfd-open-discovery/SKILL.md) | OED loop: propose → run → score → repeat | `MetricProposer.*`, `ComparatorAuthor.*`, `ComparatorVerifier.*`, plus the LLM-judge prompt |
| [`cfd-viz`](cfd-viz/SKILL.md)                       | PyVista figure generation | `ResultsInterpreterAgent.viz_quality_*`; QA via `vision_*` |
| [`cfd-interpret`](cfd-interpret/SKILL.md)           | PROCEED/REVISE/RERUN decision per case | `ResultsInterpreterAgent.interpretation_*`, `vision_system_prompt`, `vision_user_prompt` |
| [`cfd-analyze`](cfd-analyze/SKILL.md)               | Cross-case metrics → `analysis.json` | `ResultsInterpreterAgent.system_prompt`, `user_prompt` (cross-case adapter included) |
| [`cfd-paper`](cfd-paper/SKILL.md)                   | Figure + LaTeX + review loop → `paper/main.pdf` | `WriterAgent.system_prompt`, `user_prompt`; `PaperReviewerAgent.compile_fix_*`, `citation_*`, review `system_prompt`, `user_prompt`; `vision_*` |

### Supporting stages (no dedicated skill — invocations live in `cfd-pipeline`)

The Python orchestrator runs five additional thin stages that do not justify a dedicated skill directory; their inline recipes (and optional script fast-paths) are documented in [`cfd-pipeline/SKILL.md`](cfd-pipeline/SKILL.md):

| Stage | Optional script | When it fires | Output |
|---|---|---|---|
| `benchmark_plan` | `scripts/benchmark_data_prepare.py` | optional; topic mentions a benchmark/validation target | `benchmark_data.json` |
| `reference_data_ingest` | `scripts/reference_data_ingest.py` | optional; starter contains DNS/exp reference data | `reference_data_manifest.json` |
| `baseline_setup` | `scripts/baseline_setup.py` | mandatory (every mode) | `baseline_case/` + `baseline_metrics.json` |
| `metric_setup` | `scripts/metric_setup.py` | mandatory (every mode) | `metric_specs.json` + `comparators/compute_metrics.py` |
| `reference_verify` | `scripts/reference_verify_post.py` | optional, post-paper; only with `enable_reference_verify=true` | `reference_verify_report.json` |

`cfd-pipeline` embeds the `MetricSetupAgent` and `MetricSetupVerifier` prompts verbatim for the `metric_setup` stage.

---

## Installation as Claude Code skills

Symlink each skill into your project's `.claude/skills/` directory so Claude Code discovers them via the Skill tool:

```bash
# from repo root
mkdir -p .claude/skills
for s in cfd-skills/*/; do
  ln -sf "$(pwd)/$s" ".claude/skills/$(basename $s)"
done

# also expose the router-level skills
for s in skills/*/; do
  ln -sf "$(pwd)/$s" ".claude/skills/$(basename $s)"
done
```

Or symlink whole directories once:

```bash
ln -sf "$(pwd)/cfd-skills"/* .claude/skills/
ln -sf "$(pwd)/skills"/*     .claude/skills/
```

Restart Claude Code; the skills appear under `/cfd-pipeline`, `/cfd-literature`, `/cfd-orchestrator`, etc.

For other agent frameworks (Cursor, Codex CLI), point the framework's skill-discovery path at `cfd-skills/` and `skills/`.

---

## Shared contracts (memory)

Skill mode is stateless across invocations — there is no in-memory state passed between skills. Persistence happens through files in the run directory. Both modes read and write the **same** files:

| File | Producer | Consumer |
|---|---|---|
| `lit.json` | `cfd-literature` | `cfd-hypothesis`, `cfd-paper` |
| `hypotheses.json` | `cfd-hypothesis` | `cfd-requirements` |
| `requirements.json` | `cfd-requirements` | `cfd-mesh-gate`, `cfd-experiment` |
| `benchmark_data.json` | `cfd-pipeline` (Step 3) | `cfd-mesh-gate`, `cfd-paper` |
| `reference_data_manifest.json` | `cfd-pipeline` (Step 4) | `cfd-experiment`, `cfd-open-discovery`, `cfd-paper` |
| `baseline_metrics.json` | `cfd-pipeline` (Step 5) | `cfd-pipeline` (Step 6), `cfd-analyze`, `cfd-open-discovery` |
| `metric_specs.json`, `comparators/compute_metrics.py` | `cfd-pipeline` (Step 6) | `cfd-experiment`, `cfd-analyze`, `cfd-open-discovery` |
| `selected_mesh_spec.json`, `mesh_independence_context.json` | `cfd-mesh-gate` | `cfd-experiment`, `cfd-paper` |
| `cases/case_*/run_result.json` | `cfd-experiment` | `cfd-interpret`, `cfd-analyze` |
| `cases/case_*/decision.json` | `cfd-interpret` | `cfd-experiment` (rerun), `cfd-analyze` |
| `analysis.json` | `cfd-analyze` | `cfd-paper` |
| `paper_unified_plan.json` | `cfd-paper` (planner) | `cfd-paper` (viz/writer) |
| `paper/main.tex`, `paper/paper_draft.pdf`, `paper_figs/*.png` | `cfd-paper` | — |
| `oed_artifact.json` | `cfd-open-discovery` | post-OED bridge in experiments |
| `timeline.json` | all skills (append-only event log) | observability |

Any skill can resume from any prior artifact set. To "resume" from `analysis`, run `/cfd-paper` with `out-dir=<existing_dir>` — it picks up `analysis.json` and proceeds.

---

## Expert prompts — single source of truth

`prompts/prompts.yaml` is the canonical source. Each skill embeds the verbatim text of the prompts it uses, marked with a header like:

```
## System prompt (from `prompts/prompts.yaml: HypothesisAgent.hypothesis_system_prompt`)
```

If a prompt evolves in the YAML, refresh the embedded copies (a quick `grep -n "from prompts/prompts.yaml" cfd-skills/*/SKILL.md` lists every embedded reference). Both modes use the same upstream prompt, so the YAML edit alone is enough for the Python pipeline; the skill copies need a manual sync.

---

## Quick start (skill mode)

```text
# fresh research project end-to-end
/cfd-pipeline topic="LES of backward-facing step Re=5100" out-dir=runs/bfs_skill

# manual stage-by-stage
/cfd-literature   topic="..." out-dir=runs/bfs_skill
/cfd-hypothesis   out-dir=runs/bfs_skill
/cfd-requirements out-dir=runs/bfs_skill n_cases=4
/cfd-mesh-gate    out-dir=runs/bfs_skill
/cfd-experiment   out-dir=runs/bfs_skill case_id=case_001
/cfd-interpret    out-dir=runs/bfs_skill case_id=case_001
/cfd-analyze      out-dir=runs/bfs_skill
/cfd-paper        out-dir=runs/bfs_skill
```

## When to choose which mode

| Need | Mode |
|---|---|
| Hands-off long unattended run                  | A (orchestrator) |
| Quick exploratory iteration on one stage       | B (skill)        |
| Custom agent framework, not Claude Code        | B (skill)        |
| Mid-run intervention on a stuck pipeline       | C (hybrid)       |
| Reproducible end-to-end with full timeline     | A (orchestrator) |
| Demo / pedagogy / debugging a single agent     | B (skill)        |

Both modes are first-class. The orchestrator is faster for full pipelines because it caches LLM context and parallelizes; skill mode is more transparent and composable.

---

## OED extensions (Phase 1/2/3, both modes)

The OED loop accepts three opt-in extension families. All defaults preserve the legacy single-metric / single-flow / greedy behavior when flags are omitted.

**Default behaviour** (no flags needed): multi-metric tracking + LLM-as-judge per iteration are **always on**. At startup the LLM reads the topic + reference data + baseline postProcessing and decides what metrics to track (degrades to single-metric automatically if only one is relevant). Per iteration, the LLM is given the candidate's metric vector, baseline vector, modification formula/parameters/rationale, run log, prior PROCEED summaries, recent history, and diversity context, and it judges PROCEED/REVISE/RERUN holistically with a synthesized primary score and rationale.

| Flag (Python orchestrator) | Skill-side equivalent | Effect |
|---|---|---|
| `--oed-diversity-mode hybrid|aggressive` | `cfd-open-discovery diversity_mode=hybrid|aggressive` | Force "far-from-baseline" iterations (different family / equation than recent winners). |
| `--oed-diversity-far-ratio 0.3` | `diversity_far_ratio=0.3` | Hybrid: fraction of budget that must explore unseen families. |
| `--oed-multi-flow` + `--oed-starter-dirs ...` | `multi_flow_starter_dirs=[...]` | Validate each candidate against multiple reference flows. |
| `--oed-single-metric` | `single_metric=true` | Override: force the legacy single-comparator path (cost-constrained reruns only). |
| `--oed-metric-aggregator weighted_sum|min_improvement|pareto_rank` | `metric_aggregator=...` | Legacy fallback — replace LLM-as-judge with a fixed math reduction. Default is `judge_synthesis`. |

Implementation lives in `scripts/oed_extensions.py` (Python pipeline) and inside the embedded recipe in `cfd-open-discovery/SKILL.md` (skill mode). The skill loop and the Python loop produce the same artifacts (`history.json`, `bound_comparators.json`, `families_explored.json`, `oed_artifact.json`).

See [`cfd-open-discovery/SKILL.md`](cfd-open-discovery/SKILL.md) for example commands and the self-test / graceful-degradation behaviour.

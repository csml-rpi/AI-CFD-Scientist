---
name: cfd-orchestrator
description: Top-level CFD scientist orchestrator. Reads a user CFD topic, picks one route, initializes a run directory, and invokes the first skill of a fixed chain. From that point on, every per-stage skill's `## Next` block names the next skill, and the agent walks the chain to a finished paper. No user shell commands between stages.
allowed-tools: Bash, Read, Write
---

# cfd-orchestrator

## What you do

The user gave you a CFD topic. You will execute a complete study, end-to-end, in one agent session. You do **not** ask permission between stages. You do **not** skip stages. You walk a fixed chain of `cfd-skills/cfd-<name>/SKILL.md` invocations, in order, until a paper PDF lands on disk and the audit passes.

If you find yourself thinking "this is a simple code modification, I'll just write the C++ and compile" — **STOP**. That is the exact failure mode this skill exists to prevent. The chain ALWAYS starts at `cfd-literature`. There are no shortcuts to compiling, running cases, or writing a paper.

## Step 1 — Pick the route (one decision; record it; never revisit)

Read the topic and pick exactly one route:

| If the topic contains... | Route | `mode` |
|---|---|---|
| implement, modify, viscosity model, turbulence model change, Bingham, power-law, Carreau, custom model, source term, fvOption, SA modification, k-ω correction | code-modification | `code_mod` |
| open-ended, discover a novel model, beat baseline, find the best model | open-ended discovery | `open_discovery` |
| mesh independence, GCI, grid convergence, Richardson (standalone) | mesh-gate-only | `mesh_gate` |
| analyze, plot, post-process, visualize results (run dir already exists with cases) | analysis-only | `analysis_only` |
| write paper, LaTeX, manuscript, draft the paper (analysis.json already exists) | paper-only | `paper_only` |
| anything else (study, sensitivity, turbulence, channel, backward step, experiment) | standard research | `research` |

If genuinely ambiguous, ask **one** concise clarifying question, then commit. After this step you do not revisit routing.

## Step 2 — Initialize the run directory

```bash
OUT=runs/<short_study_slug>
mkdir -p "$OUT/checkpoints"
python3 - <<PY
import json, datetime, os
out = os.environ.get("OUT", "$OUT")
state = {
  "topic": "<verbatim user topic>",
  "mode": "<route from Step 1>",
  "status": "running",
  "current_stage": "literature",
  "started_at": datetime.datetime.utcnow().isoformat() + "Z"
}
open(f"{out}/state.json", "w").write(json.dumps(state, indent=2))
open(f"{out}/timeline.json", "w").write("[]")
open(f"{out}/checkpoints/routing_done.json", "w").write(json.dumps({"stage": "routing", "ts": state["started_at"]}, indent=2))
PY
```

`mode` is the field every downstream skill reads from `state.json` to pick its `## Next` branch. Write it correctly and never overwrite it.

## Step 3 — Activate environment

```bash
conda activate cfd-scientist
```

If activation fails because the env does not exist, stop and surface the one-time setup command from `README.md`. Do not continue.

`S2_API_KEY` should be set in the shell (Semantic Scholar). If it is not set, surface a one-line warning to the user — `cfd-literature` will still work on the public endpoint but is rate-limited.

## Step 4 — Walk the deterministic skill sequence

You invoke skills in the explicit ordered list below — not by chasing recursive `## Next` links through 8-12 markdown files. The ordered list survives long-context drift; recursive `## Next` chains do not, and **mesh-gate** (the most-skipped stage) gets buried as a 3-4 hop deep instruction. The list below is the authoritative skill order. Each row is a mandatory invocation when its `enter_if` condition is true.

Each per-stage skill's own `## Next` block still exists as a fallback / chain continuation hint. The list here is the canonical contract.

### Step 4.A — `mode ∈ {research, code_mod, open_discovery}`: full chain

Walk this top-to-bottom. Do **not** skip mesh-gate. Do **not** fabricate any `checkpoints/*_done.json` — the per-stage skill writes its own checkpoint after it really runs, and the audit verifies signatures.

| Step | Skill | `enter_if` | Writes (must exist after) |
|---|---|---|---|
| 4.1 | `cfd-literature` | always | `lit.json`, `checkpoints/literature_done.json` |
| 4.2 | `cfd-pipeline` (Steps 5 + 6 only) | always | `baseline_metrics.json`, `metric_definitions.json`, `checkpoints/baseline_setup_done.json`, `checkpoints/metric_setup_done.json` |
| **4.3** | **`cfd-mesh-gate`** | **always (HARD — non-skippable)** | **`selected_mesh_spec.json`, `mesh_independence_context.json` OR a `checkpoints/mesh_gate_done.json` with `status: "starter_mesh_locked"`, a `mesh_source` field, and a `citation` (published DOI)** |
| 4.4 | `cfd-hypothesis` | `mode ∈ {research, code_mod}` | `hypotheses.json`, `checkpoints/hypothesis_done.json` |
| 4.5 | `cfd-requirements` | `mode ∈ {research, code_mod}` | `requirements.json`, `checkpoints/requirements_done.json` |
| 4.6 | `cfd-code-modify` | `mode == code_mod` | `code_mod/` artifacts, `checkpoints/code_mod_done.json` |
| 4.7 | `cfd-experiment` per case (loop) | `mode ∈ {research, code_mod}` | `cases/case_*/run_result.json`, `cases/case_*/decision.json`, `cases/case_*/figs/*.png`, `checkpoints/experiments_done.json` |
| 4.8 | `cfd-open-discovery` | `mode == open_discovery` | `oed_artifact.json`, `bridge.json`, `manifest.json`, `cases/case_*/...`, `checkpoints/open_ended_discovery_done.json` with `bridge_signature: "cfd-open-discovery:bridge:v1"` |
| 4.9 | `cfd-analyze` | always | `analysis.json`, `checkpoints/analysis_done.json` |
| 4.10 | `cfd-cross-analyze` | always | `cross_experiment_analysis/{aggregate.csv, *.png, cross_experiment_interpretation.md, scripts/*.py}`, `checkpoints/cross_experiment_analysis_done.json` |
| 4.11 | `cfd-paper` (full Phase 0 → A → B → C → D → E → F cycle) | always | `paper_unified_plan.json`, `paper_figs/*.png`, `paper/main.tex`, `paper/refs.bib`, `paper/main.pdf`, `review.json`, `checkpoints/paper_review_done.json` |

**Mesh-gate is Step 4.3 and is non-skippable.** Even for open-ended discovery on a starter case, you must either (a) invoke `cfd-mesh-gate` for real (baseline vs refined comparison) OR (b) emit `mesh_gate_done.json` with `status: "starter_mesh_locked"`, a `mesh_source` field, and a `citation` object carrying the **published DOI** of the mesh-validation source the starter reproduces. The audit's H31 gate resolves that DOI live — a free-text `rationale` is no longer accepted, and fabricated stub checkpoints fail.

### Step 4.X — keep `state.json` honest (after every stage)

The moment a skill writes its `checkpoints/<stage>_done.json`, advance `state.json` so it never lags reality — the stale-`state.json` defect where a finished run still reported `current_stage: "literature"`:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("$OUT/state.json")
s = json.loads(p.read_text())
s["current_stage"] = "<stage just completed>"   # literature, mesh_gate, experiments, analysis, paper_review, …
s["status"] = "running"
p.write_text(json.dumps(s, indent=2))
PY
```

The `checkpoints/` files remain the source of truth; the Step 5 audit reconciles `state.json` against them either way. Bumping it here keeps a crashed run resumable from the right stage. Only the passing audit sets `status: "done"`.

After Step 4.11 (or after `cfd-analyze` for routes 3/4), call the final audit per Step 5 below.

### Step 4.B — `mode == analysis_only`

| Step | Skill | Writes |
|---|---|---|
| 4.1 | `cfd-analyze` | `analysis.json`, `checkpoints/analysis_done.json` |

Chain ends here. Run Step 5 audit.

### Step 4.C — `mode == paper_only`

| Step | Skill | Writes |
|---|---|---|
| 4.1 | `cfd-paper` | `paper/main.pdf`, `review.json`, full paper-stage artifacts |

Chain ends here. Run Step 5 audit.

### Step 4.D — `mode == mesh_gate`

| Step | Skill | Writes |
|---|---|---|
| 4.1 | `cfd-mesh-gate` | `selected_mesh_spec.json`, `mesh_independence_context.json` |

Chain ends here. Run Step 5 audit.

You do **not** stop between stages to summarize, ask, or wait. Walk every numbered step in the table for your mode before reaching Step 5.

## Step 5 — Final audit loop (mandatory; non-negotiable; binding)

The run is **not complete** until `stage_gate_audit.py` exits `rc == 0` and has written a signed `audit_passed.json`. This is a **loop**, not a single call. You do not stop while `rc != 0`:

```bash
while true; do
  python scripts/stage_gate_audit.py --out-dir "$OUT" && break
  # rc != 0 — the script printed every gap on stderr. Fix each, then loop:
  #   * "missing required checkpoint: <stage>"  → invoke Skill cfd-skills/cfd-<stage>
  #   * a case failure (failed run_result.json without an exhausted-rerun
  #     record, FOAM FATAL ERROR in a solver log, or no clean 'End' line)
  #                                             → rerun that case via cfd-experiment
  #   * H33 OED bookkeeping incomplete          → re-run cfd-open-discovery Step 14 (bridge)
  #   * H31 mesh-gate citation                  → fix mesh_gate_done.json / run cfd-mesh-gate
  #   * a paper gate (H1, H8–H35)               → invoke the relevant cfd-paper phase
done
```

There is no scenario in which you skip this, and no scenario in which you report success while `rc != 0`. Slice-20 hardening means a failed `run_result.json` fails the audit unless its rerun budget is documented as exhausted, and a `FOAM FATAL ERROR` in any solver log fails unconditionally — so "the paper looks finished" is **not** completion. **Completion = a signed `audit_passed.json` on disk.**

- `rc == 0` → the audit wrote `<out-dir>/audit_passed.json` (`audit_signature: "stage_gate_audit.py:v1"`, schema: `status`, `ts`, `rc`, `n_failures`, `failures: []`, `checkpoint_failures`, `case_failures`, `paper_failures`, `route`, `mode`, `audit_mode`) and reconciled `state.json` to `status: "done"`. Report the paper path to the user.
- `rc != 0` → never report success. Fix every printed gap and re-audit.

**Do not write `audit_passed.json` yourself under any circumstance.** The audit script is the only writer. A fabricated `audit_passed.json` (e.g. one with a `"note": "skipped script"` field, or any schema that doesn't match the script's report dict) is detected as fake by the audit on its next run and removed. Writing it yourself is a bug, not a shortcut.

`audit_passed.json` written by the audit script is the only authoritative completion signal. `state.json#status` is agent-writable and not authoritative.

### Literature is mandatory before the paper stage (routes 1, 2, 5, 6)

For modes `research`, `code_mod`, `paper_only`, and `open_discovery`, `lit.json` (or `lit_enriched.json`) **must be present and non-empty** before `Skill cfd-skills/cfd-paper` is invoked. The audit enforces this via H18 (`lit_retrieval_ok`), H10 (`citations_ok`, floor = 10), and H19 (`refs_bib_ok`). Empty literature → unciteable paper → audit fails. If the chain reaches the paper stage without literature on disk, **re-invoke `Skill cfd-skills/cfd-literature`** before proceeding. Do not let the chain author a paper on top of `lit.json = []`.

### What the paper-stage audit checks (Slice 18, H18–H22)

In addition to the existing H1–H17 gates, the audit now enforces:

- **H18** lit.json non-empty for routes 1/2/5/6
- **H19** bibliography (external `paper/refs.bib` OR inline `\begin{thebibliography}`) with ≥ N entries covering every `\cite{}` key
- **H20** ≥ 3 display-math equations for routes 1/2/6 (≥ 1 for route 5) — governing equations, closure, QoI/modification
- **H21** zero duplicate paragraphs in the main body (writer must not copy/paste to inflate length)
- **H22** ≥ 1 `\begin{tabular}` environment containing ≥ 3 numeric tokens (quantitative case × QoI comparison)
- **H16 tighter** every `figure_jobs[i]` in `paper_unified_plan.json` carries a non-null `out_png` and `kind`, and every included case has at least one `u_contour` AND one `p_contour` job
- **H15 tighter** every `figure_jobs[i].out_png` is embedded via `\includegraphics` in `main.tex` (no orphan PNGs in `paper_figs/`)

## Step 6 — Bundle audit & release hygiene (multi-task batches only)

A single orchestrator run produces one task directory and ends at Step 5. When several tasks share one parent directory — a smoke-test batch, a paper's experiment set — two batch tools verify and clean the whole bundle:

```bash
# Every task in the bundle must carry a valid signed audit_passed.json.
# A missing record means the chain stopped before Step 5 — a process failure,
# even if the task's solver runs and paper look complete.
python scripts/audit_bundle.py --bundle runs/<bundle>
python scripts/audit_bundle.py --bundle runs/<bundle> --rerun-audit   # also re-audit gaps

# Before sharing/zipping a bundle: rewrite workstation paths to portable
# placeholders, catalogue symlinks, and record .o/.so compiled artifacts in
# reproducibility_manifest.json (binaries are kept, not stripped).
python scripts/sanitize_bundle.py --bundle runs/<bundle>            # dry-run report
python scripts/sanitize_bundle.py --bundle runs/<bundle> --apply    # rewrite in place
```

`audit_bundle.py` is the gate that catches the "successful runs, no audit record" pattern. `sanitize_bundle.py` keeps compiled binaries but documents them with a sha256 + rebuild command — run it on a copy or a zip-staging directory, never on a live run you still need.

## On Python scripts in skill mode

Skills can — and should — invoke per-stage Python scripts. `scripts/lit.py`, `scripts/viz.py`, `scripts/interpret.py`, `scripts/analyze.py`, `scripts/paper_unified.py`, `scripts/foam_run.py`, `scripts/stage_gate_audit.py`, `scripts/audit_bundle.py`, `scripts/sanitize_bundle.py`, `scripts/rag_query.py`, `scripts/code_mod_agentic.py` — each is a per-stage or batch-level CLI wrapper that does one thing and writes one artifact. They use the same prompts and produce the same outputs as the inline agent recipes. Skills can import any library (PyVista, numpy, matplotlib, pandas, etc.) inside Python they author and run.

**The only script skill mode does NOT call** is `scripts/orchestrator_run.py` — that one drives the whole LangGraph pipeline end-to-end and bypasses skill chaining. Every other script under `scripts/` is fair game.

## Non-negotiable constraints (apply to every stage)

1. Do **NOT** edit OpenFOAM installation directories (`$WM_PROJECT_DIR`, `$FOAM_SRC`, anything under the installation tree).
2. The starter directory passed as `--starter-dir` is **READ-ONLY**. Copy into `<out-dir>/` before any edit. No `sed -i`, `mv`, `rm`, or redirect-overwrites inside the starter.
3. Code modifications go only inside `<case_path>/customModels/<ClassName>/`. Compile case-locally with `wmake libso`.
4. Mesh-gate always precedes experiments (enforced by `cfd-mesh-gate`'s `## Next` and by the preflight at the top of `cfd-experiment`).
5. Long CFD runs are normal. Treat 3–4 hour solver wall-clock as expected. Do not declare a run failed on elapsed time alone — apply the CFL-aware timestep policy in `cfd-skills/cfd-experiment/SKILL.md` first.

## What this skill does NOT do

This skill picks a route, sets up state, and hands off to `cfd-literature` (or the route's first skill). It does NOT itself author requirements, run mesh-gate, run cases, or write the paper. Those live in `cfd-skills/`. After the handoff, **the per-stage `## Next` chain drives the run.** This file should not be re-read between stages.

## Resume after a crash

If you arrive in a run directory that already has `state.json`, read `current_stage` and invoke `Skill cfd-skills/cfd-<current_stage>`. Its Step 0 preflight verifies that upstream checkpoints are on disk; if any are missing, fix the gap first.

For manual stage-by-stage progression with external validation (e.g. CI, smoke tests), `scripts/skill_bootstrap.py --init / --advance / --status` exists as a fallback driver. It is NOT the primary path — the agent-driven chain above is.

## Cross-link

- Per-stage skill recipes: `cfd-skills/cfd-<name>/SKILL.md`
- Pipeline reference (orchestrator-agnostic stage spec): `AGENTS.md`
- Project setup + install: `README.md` (top-level)
- LangGraph mode (Python pipeline, unattended): `python scripts/orchestrator_run.py --topic … --mode <route>`. Same artifact contracts. Use only when you need a fully-enforced run with no agent decisions.

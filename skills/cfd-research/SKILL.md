---
name: cfd-research
description: Top-level CFD research entry — alias for cfd-skills/cfd-pipeline (general path). Use when the user gives a CFD research topic and wants the full literature → hypothesis → mesh-gate → experiments → analysis → paper flow. The full self-contained recipe lives in cfd-skills/cfd-pipeline/SKILL.md.
allowed-tools: Bash, Read, Write
---

# cfd-research (alias)

This skill is a **thin alias** for `cfd-skills/cfd-pipeline/SKILL.md`. The full self-contained, ARIS/DS-style end-to-end recipe (literature, hypothesis, requirements, mesh-gate, experiments, interpret, analyze, paper) — including all expert prompts embedded verbatim — is documented there.

## Why an alias?
- Historically, `skills/` housed the router + research/code-mod/mesh-independence skills. `cfd-skills/` was added later as the per-stage recipe layer.
- After the Nov 2026 ARIS/DS-style refactor, the full recipes live under `cfd-skills/`. This file remains so existing routing in `CLAUDE.md` and external agent configurations keep working.

## Use this skill if you came here via:
- The `CLAUDE.md` "study/sensitivity/turbulence/backward step/channel/experiment/run" routing line.
- An external agent config that was set up before the refactor.

## Do this
1. Open `cfd-skills/cfd-pipeline/SKILL.md` and follow the **general path** chain (Steps 1–13).
2. For each sub-stage, follow the corresponding `cfd-skills/cfd-<stage>/SKILL.md` (each is fully self-contained with its expert prompts embedded).

## Behavioral contracts preserved here
- **Mandatory mesh-independence gate** before any experiment sweep — see `cfd-skills/cfd-mesh-gate/SKILL.md`.
- **Foam-Agent runtime** for case execution — see `skills/cfd-foamagent-runtime/SKILL.md` (still the canonical Foam-Agent execution contract).
- **CFL-aware retry policy** for slow/failing runs — see `cfd-skills/cfd-experiment/SKILL.md`.
- **Long-run policy** (multi-hour OK; no premature timeout) — applies everywhere.
- **Expert prompts from `prompts/prompts.yaml`** are embedded verbatim in every relevant `cfd-skills/cfd-<stage>/SKILL.md`. The YAML remains the authoritative source.

## Migration note
If you have agent code that previously invoked `/cfd-research`, replace it with `/cfd-pipeline` for new work; both work identically today, but new development happens under `cfd-skills/`.

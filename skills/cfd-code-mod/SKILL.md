---
name: cfd-code-mod
description: Custom OpenFOAM model implementation entry — alias for cfd-skills/cfd-code-modify. Use when the user wants to implement, modify, or add a custom viscosity / turbulence / source term / fvOption inside a case-local customModels directory. The full self-contained recipe (with the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol embedded verbatim) lives in cfd-skills/cfd-code-modify/SKILL.md.
allowed-tools: Bash, Read, Write
---

# cfd-code-mod (alias)

This skill is a **thin alias** for `cfd-skills/cfd-code-modify/SKILL.md`. The full ARIS/DS-style self-contained recipe — including the OPENFOAM 10 LITERATURE CHANGE AGENT v2 protocol embedded verbatim, the wmake compile-fix loop, and the smoke-test recipe — lives there.

## Why an alias?
After the Nov 2026 ARIS/DS-style refactor, all recipe content moved to `cfd-skills/`. `CLAUDE.md` routing for "implement / modify / viscosity model / Bingham / power-law / Carreau / source term / fvOption" historically pointed here, so this file remains for backward compatibility.

## Do this
1. Open `cfd-skills/cfd-code-modify/SKILL.md`.
2. Follow Steps 1–7 there: build payload → resolve OpenFOAM API grounding → run the LITERATURE CHANGE AGENT (LLM call with the full v2 protocol embedded) → compile → compile-fix loop → smoke-test → write `build_result.json`.

## Hard rules (preserved from the original `skills/cfd-code-mod`)
- **Never** edit `$WM_PROJECT_DIR`, `$FOAM_SRC`, or any OpenFOAM installation/source directory.
- All custom code lives in `<case_path>/customModels/<ClassName>/`.
- Compile case-locally with `wmake libso`. Link via `controlDict.libs` and one activation dictionary (`constant/momentumTransport`, `constant/fvModels`, or fallback `constant/turbulenceProperties`).
- One coherent in-family modification per invocation. Out-of-scope requests (new transported variables, solver edits, BC framework changes, multi-region, multi-physics outside kinematic viscosity) → return `UNSUPPORTED`.

## Cross-link
- Standard CFD research after a code mod → continue from Stage 2 of `cfd-skills/cfd-pipeline/SKILL.md` (general path), with the mesh gate then running against the modified physics group.
- For OED of custom turbulence models, the code-mod is invoked **inside** `cfd-skills/cfd-open-discovery/SKILL.md`'s discovery loop, once per candidate.

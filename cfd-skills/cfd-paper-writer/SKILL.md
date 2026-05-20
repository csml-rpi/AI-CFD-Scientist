---
name: cfd-paper-writer
description: General-purpose academic paper writer skill for CFD research, adapted from Master-cai/Research-Paper-Writing-Skills with CFD-domain overlays. Embeds section-by-section structural guidance (Abstract / Introduction / Related Work / Methods / Experiments / Discussion / Conclusion), Master-cai's reviewer rubric, and a claim-evidence-citation discipline. Used by cfd-paper Phase D (writer / reviser) and Phase F (reviewer). Replaces the previous ad-hoc HARD-rule list with a structural writer skill — the audit then enforces only objectively-checkable properties.
allowed-tools: Bash, Read, Write
---

# cfd-paper-writer

Adapted from Master-cai/Research-Paper-Writing-Skills (MIT) and Prof. Peng Sida's writing notes. Augmented with CFD-domain requirements (governing equations, turbulence closures, per-case contours, mesh independence, comparator definitions).

This skill is invoked **from inside `cfd-paper`** — by Phase D (writer / reviser) and by Phase F (reviewer). It never runs standalone. It receives a fully-built context bundle (analysis, manifest, literature, plan, figure paths, prior reviewer feedback) and emits either:
- a complete LaTeX manuscript, or
- a structured reviewer verdict.

## Core philosophy (apply to every section)

1. **One paragraph for one message.** The first sentence states what the paragraph will do. Every subsequent sentence advances that one message (cause, contrast, consequence, refinement) — nothing else.
2. **Claim → evidence → citation.** Every non-trivial claim is followed by a number from a table or figure (evidence) and a `\cite{}` to the literature that supports the claim or that you are distinguishing from. No claim sits alone.
3. **Visual quality is content, not decoration.** Figures, captions, tables, and equation typography are part of the substantive contribution. Treat them as core writing — not afterthoughts.
4. **Reverse-outline as you write.** After each section, you should be able to write a 3–7 bullet outline of what it actually says. If you can't, the section is a token-soup draft.
5. **Adversarial self-review before handing off.** Read your own text as a hostile reviewer would. Flag every unsupported claim, every figure that doesn't carry its caption's promise, every term that drifts in meaning between sentences.

These rules eliminate the failure patterns that plagued earlier CFD paper drafts: n-gram word loops within paragraphs, "in introduction paragraph 1" meta-cross-references, citation clusters dumped at section ends, equations that are merely Re/Cf/RMSE definitions, missing geometry descriptions, and orphan figures.

---

## Section-by-section writing guides

Each guide gives you: (a) pre-writing questions, (b) one or more concrete paragraph templates with sentence skeletons, (c) a quality checklist.

You pick the template that best fits the work and customize the sentence skeletons to your topic. **Templates are guides, not prison cells** — adapt them, but do not abandon their structural discipline.

### 1. Abstract

#### Pre-writing questions

1. What technical / physical problem do we solve, and why is there no well-established CFD solution?
2. What is our technical contribution? (a sensitivity study, a model modification, an open-ended discovery, a quantitative correlation, …)
3. Why does our method work in essence?
4. What new insight does the experiment produce? (e.g., "k-ω SST under-predicts reattachment by 20% on adverse-pressure-gradient flows", "a wall-distance-gated SA destruction reduction does not survive aggressive grids", …)

#### Templates

**Version A (Challenge → Contribution):** for studies with a single dominant contribution.
1. One sentence: the CFD task and physical setting.
2. One to two sentences: the technical challenge for previous closures / approaches.
3. One to two sentences: our technical contribution that addresses the challenge.
4. One sentence: the benefit (improvement vs baseline / reference, in concrete numbers).
5. One sentence: experiment summary (sweep / cases / metric outcomes).

**Version B (Challenge → Insight → Contribution):** for model-modification or OED papers where the insight is more interesting than the implementation.
1. Task + setting.
2. Technical challenge for previous methods.
3. One sentence: the insight (e.g., "near-wall production over-prediction on the recovery shoulder is the dominant error source for k-ω SST on periodic hill at Re=5600").
4. One to two sentences: the technical contribution implementing the insight.
5. One sentence: benefits + experiment summary.

**Version C (Multi-contribution):** for sweeps / sensitivities where each row of the design matrix is a contribution.
1. Task + setting.
2. If needed, one contrast sentence about prior approaches.
3. Contribution 1 + advantage.
4. Contribution 2 + advantage.
5. Contribution 3 + advantage.
6. Experiment summary.

#### Hard rules

- ≤ 250 words. No case IDs (`case_001`). No `\cite{}` calls. No equations beyond a single inline Reynolds-number formula if essential. No reference to section numbers or internal labels.
- Every claim in the Abstract must be re-derivable from a number in the Results table.

#### CFD specifics

- For sensitivity sweeps: state the parameter range, the QoI(s), and the strongest quantitative outcome (e.g., "the reattachment x_r/h shifted by 40% between k-ε and SA").
- For model-modification: state which closure was modified and the headline metric delta vs baseline.
- For OED: state how many families were tried, whether a winner was found, and either the winner's improvement or the no-winner family-coverage statement.

#### Quality checklist

1. Can a reader identify task, challenge, insight/contribution, and headline result in one pass?
2. Are all major claims supported by experimental numbers?
3. Are technical names self-contained (no jargon that requires reading another paper)?
4. Is there any sentence that mixes two or more messages?

---

### 2. Introduction

#### Logic map (think this through before writing)

```
What CFD task / flow problem
       ↓
What QoI(s) measure success
       ↓
SOTA methods fail to meet QoI target
       ↓
Root technical issue behind that failure
       ↓
Our technical solution + study design
       ↓
Why the solution / study works
       ↓
Additional contributions and impact
```

#### Backward reasoning (answer before drafting)

1. What CFD problem do we solve, and why is there no well-established solution? (most important)
2. What are the contributions of our study? (a new flow regime characterized? a new closure-modification family? a new correlation? a falsification of a published claim?)
3. What are the benefits of our contributions, why can they answer this challenge, and what new insight do they bring?
4. How do we use prior CFD work to lead readers to our challenge?

#### Forward story (write in this order)

1. Introduce the CFD task / flow problem and applications (heat transfer, drag prediction, mixing efficiency, separation control, …).
2. Use prior CFD work to lead to the technical challenge we solve.
3. Present our contribution(s).
4. Explain the technical advantage and the new insight explicitly.

#### Section skeleton

```latex
\section{Introduction}
% Task and application
% Technical challenge for previous methods (limitation + technical reason)
% Our pipeline / study design
% Experiment summary
% Contributions
```

#### Opening templates

- **Version 1** (task is niche → define task, then applications).
- **Version 2** (task is familiar → open with applications).
- **Version 3** (general → specific: state general task, then narrow to this paper's setting).
- **Version 4** (familiar task → open with applications, then expose the unresolved failure in the very first paragraph via a prior method).

#### Technical-challenge templates

- **Version 1** (existing task, existing methods): general challenge → traditional methods + limitation → recent methods + limitation → exact challenge we solve.
- **Version 2** (existing task with insight historically grounded): mainstream limitation → classical line with related insight → why that classical line is still insufficient → modern methods unresolved → bridge to our method.
- **Version 3** (novel task / OED): state the goal, decompose into N independent challenge points using First/Second/Finally.

#### Pipeline / study-design templates

- **Version 1** (one contribution, one teaser figure): framework + key novelty + teaser pointer + implementation + multi-advantage list.
- **Version 2** (two contributions, one teaser figure): framework + contribution 1 + advantage 1 + remaining challenge + contribution 2.
- **Version 3** (new module on existing pipeline): prior pipeline setup + new module + observation that motivates it + mechanism + comparison to alternatives.
- **Version 4** (observation-driven): innovation first + intuitive observation as motivation + implementation + technical advantage + empirical gain.

#### Hard rules

- ≥ 4 paragraphs, each with a single message.
- ≥ 3 `\cite{}` calls in the Introduction (claim → reference, not citation cluster at end).
- The first paragraph must already gesture at what problem the paper solves — even if the technical-challenge paragraph comes later.
- No hiding concrete contribution in abstract "novelty" prose. Be concrete.

#### CFD specifics

- Cite prior CFD studies on the same geometry / closure / flow regime — not just generic turbulence references.
- If using DNS/experimental reference data, cite the canonical source (Driver-Seegmiller for BFS, Frohlich et al. for periodic hill, etc.).
- For sensitivity studies, state the range of Re / parameter explored.
- For OED / code-modification, name the closure being modified in the second paragraph at the latest.

#### Quality checklist

1. Does the first sentence of each paragraph state its message?
2. Does each paragraph carry one message only?
3. Are the technical challenge, the technical reason, and the solution mechanism all explicit?
4. Are claims in the Introduction aligned with the Results table?
5. Is terminology stable across all sections (e.g., "skin friction coefficient" not switching between "Cf", "C_f", "wall shear coefficient")?

---

### 3. Related Work

#### Goal

Position your work against the most relevant lines of CFD research. Make your novelty / specific contribution easy for a reviewer to verify.

#### Workflow

1. List directly competing and recent papers first (last 3–5 years preferred).
2. Group literature by **technical topic**, not by publication year.
3. For each topic: summarize the common paradigm in 1–2 sentences, then the key limitation tied to your study's QoI.
4. End each topic with a transition sentence that distinguishes your work.

#### Suggested topic groupings for CFD

For a turbulence-model sensitivity study: (a) RANS closures for separated flows; (b) DNS / LES reference data; (c) prior sensitivity studies on similar geometries.

For a viscosity / rheology study: (a) constitutive law variants; (b) prior solver implementations; (c) experimental validation studies.

For an OED / model-modification study: (a) original closure and canonical extensions; (b) prior model-modification attempts and their failure modes; (c) discovery-loop methodology (if relevant).

For a sweep / correlation study: (a) prior empirical correlations on this geometry; (b) prior simulation studies bracketing the parameter range; (c) the experimental literature that grounds the QoI.

#### Paragraph template (per topic)

1. **Topic sentence**: define the scope of this topic in one sentence.
2. **Representative methods**: one compact summary of the dominant paradigm, with citations.
3. **Limitation**: tied to your target technical challenge / QoI / parameter range.
4. **Transition sentence**: leads to your method / study.

#### Hard rules

- 2–4 focused topics. Not more (becomes a citation dump). Not fewer (insufficient coverage).
- ≥ 1 `\cite{}` call per paragraph, woven into the sentence — not appended at the end.
- Compare mechanisms, assumptions, and failure modes — not just method names.
- Do not hide the strongest baseline. If a recent paper directly competes with yours, cite it in the first or second topic.
- Never write meta-references ("in this section we…", "as paragraph 3 noted…", "in introduction paragraph 1…"). Every sentence stands on its own.

#### Quality checklist

1. Are the strongest / most recent competitors covered?
2. Is each topic connected to your problem setting?
3. Is your difference explained in *technical* terms, not marketing terms?
4. Is citation coverage complete for all major prior-work claims?

---

### 4. Methods

#### Goal

Describe what was actually simulated (geometry, BCs, governing equations, closures, numerics, mesh, comparator, QoI extraction) with enough detail that a competent CFD reader could reproduce the run.

#### Pre-writing questions (per module / per subsection)

For every distinct part of the methodology (geometry / governing equations / closure / numerics / mesh / case design / comparator):

1. **How does it work?** (Forward process: input → step → output.)
2. **Why is it needed?** (What gap or design constraint motivates this choice?)
3. **Why does it work?** (Soundness or empirical justification.)

#### Three-element subsection structure

For each Methods subsection:

1. **Design**: describe the data structures, parameters, equations, or forward process — "input → step 1 → step 2 → output". Concrete and unambiguous.
2. **Motivation**: problem-driven logic: "because X is true / required, we use Y".
3. **Technical advantage**: explain why this choice beats alternatives, with measurable / observable outcomes.

#### Section organization

```latex
\section{Methods}
\subsection{Flow configuration and geometry}
\subsection{Governing equations}
\subsection{Turbulence closure / constitutive model}
\subsection{Numerical setup}
\subsection{Mesh and mesh-independence}
\subsection{Case design and parameter sweep}
\subsection{Comparator and QoI extraction}
```

#### CFD-specific HARD rules (these are non-negotiable)

The audit script enforces every rule in this list. A draft missing any of these will fail at the audit gate.

##### a. Geometry description

Methods must concretely describe the geometry. Include at least three of:
- Step height *h*, expansion ratio, hill height, channel height
- Domain length / extent (in step heights or characteristic lengths)
- Inlet x/h, outlet x/h, lateral / spanwise extents
- Reynolds number *Re_h*, *Re_τ*, *Re_D*, …
- x/h, y/h coordinate convention
- Boundary conditions for inlet, outlet, walls, sides

A reader who has never seen the case must be able to reconstruct the geometry from your Methods section alone.

##### b. Governing equations

At least one display equation must show the **governing equations** the solver integrates. Use Reynolds-averaged notation when applicable:

```latex
\begin{equation}
\nabla \cdot \overline{\mathbf{U}} = 0
\label{eq:continuity}
\end{equation}

\begin{equation}
\frac{\partial \overline{\mathbf{U}}}{\partial t}
+ \nabla \cdot (\overline{\mathbf{U}}\,\overline{\mathbf{U}})
= -\frac{1}{\rho}\nabla \overline{p}
+ \nabla \cdot \left[(\nu + \nu_t)\,(\nabla\overline{\mathbf{U}}
+ \nabla\overline{\mathbf{U}}^{\mathsf T})\right]
\label{eq:rans-momentum}
\end{equation}
```

Adjust for compressible / multiphase / non-Newtonian as appropriate. The audit recognizes `\nabla \cdot`, `\partial / \partial t`, `\overline{u}` / `\overline{U}`, `Navier-Stokes`, `continuity`, `momentum equation` as governing-equation markers.

##### c. Turbulence-closure / constitutive equations

At least one display equation must be the **closure / constitutive equation** for the model(s) under study. For a turbulence-model comparison paper, this is the entire point of the paper. Examples:

- k-ω SST: the k and ω transport equations, with the SST blending function.
- k-ε: the k and ε transport equations.
- Spalart-Allmaras: the \tilde{\nu} transport equation with c_b1, c_b2, c_w1, σ, κ.
- Bingham: τ = τ_y + μ_p γ̇ (with the regularization used).
- Carreau: η(γ̇) = η_∞ + (η_0 - η_∞)(1 + (λ γ̇)^2)^{(n-1)/2}.

The audit recognizes `\omega`, `\beta^*`, `\epsilon`, `\varepsilon`, `\tilde{\nu}`, `\widetilde{\nu}`, `nuTilda`, `Spalart-Allmaras`, `\nu_t`, `\mu_t`, `Bingham`, `Carreau`, `power-law` as closure markers.

For OED / code-modification: the modification equation itself is one of these closure equations — show it explicitly with the modified coefficient or term highlighted.

##### d. Mesh and mesh-independence (HARD — H32 enforces this)

The Methods MUST contain a mesh subsection. There are exactly two valid forms — pick whichever matches the run's `checkpoints/mesh_gate_done.json`:

**Form 1 — Real mesh-independence study** (when `mesh_independence_context.json` exists on disk and contains `metrics_by_mesh_level`). The subsection MUST include:

1. A `\begin{tabular}` listing every mesh level compared (coarse, baseline, refined, …) with cell count and monitored QoI values. Use ONLY numbers from `metrics_by_mesh_level`; no invented values.
2. The selection criterion stated explicitly (≤ 5% / ≤ 1% QoI change, GCI < threshold, Richardson extrapolation, …) and the selected mesh.
3. A one-sentence summary from `cross_mesh_analysis_excerpt`.
4. A `\includegraphics` of any mesh plot in `mesh_figure_paths` when present.

Example template:

```latex
\subsection{Mesh independence}
\begin{table}[h]\centering
\begin{tabular}{lrrr}\toprule
Level & Cells & $C_f$ RMSE & $x_r/h$ \\\midrule
Coarse  & 32{,}000 & 0.00488 & 7.6 \\
Baseline& 48{,}000 & 0.00430 & 7.7 \\
Refined & 96{,}000 & 0.00427 & 7.7 \\\bottomrule
\end{tabular}
\caption{Mesh independence: refined-to-baseline change in $C_f$ RMSE is 0.7\,\%, below the 5\,\% selection threshold.}
\end{table}
The baseline mesh was selected for all subsequent runs since the refined level produced $<5\%$ change in the primary QoIs and a $7\times$ longer wall-clock per case.
```

**Form 2 — Starter mesh locked** (when `mesh_gate_done.json#status == "starter_mesh_locked"`). The subsection MUST:

1. Name the starter (`<out-dir>/state.json#starter_dir`).
2. Cite the starter's validation source (typically the starter's README, or the published paper that the starter mesh reproduces).
3. State the lock rationale explicitly (e.g. "the mesh is a study control for the OED sweep; refining the mesh per candidate would confound scoring by mixing mesh and model effects").
4. Acknowledge it as a limitation in the Conclusion / Limitations paragraph.

Example template:

```latex
\subsection{Mesh}
This study locked the mesh shipped with the starter case
\texttt{starter\_oed\_turbulence/periodic\_hill\_sa} (validated against the
Fr\"ohlich et al.\ (2005) DNS, see starter README). The mesh was kept
identical across every OED candidate so that the only difference between
runs was the SA-closure modification under study; refining the mesh per
candidate would have confounded the scoring by mixing mesh-resolution
effects with model effects. We acknowledge mesh-sensitivity as a remaining
limitation in the Conclusion.
```

If neither form applies (no `mesh_gate_done.json` of any kind): do NOT fabricate. Invoke `cfd-mesh-gate` first.

**The audit's H32 gate parses the Methods section for ≥ 2 mesh-related markers (mesh independence, GCI, locked mesh, blockMesh, y+, cell count, baseline-vs-refined, …). A paper without any of these is rejected.**

##### e. Comparator and QoI extraction

For every QoI used to score cases, define exactly how it is computed. Examples:

- Cf RMSE: state the binning convention, interpolation onto reference x-stations, sign convention for τ_w, exclusion of upstream regions.
- Reattachment x_r/h: state the criterion (Cf sign crossing, ψ = 0, time-averaged separation streamline).
- Strouhal number: state the probe location, the velocity component, the time-window for FFT.

For OED runs with bound comparators, name the two extractors that must agree (text + PyVista, typically) and state the relative-disagreement tolerance.

#### Hard structural rules

- ≥ 3 `\cite{}` calls woven across Methods (for the closure model, the comparator, the reference data, the geometry).
- One sentence per equation explaining what it represents and where the symbols are defined.
- No bare `\begin{equation}` blocks dropped in mid-prose without introduction.
- Subsection-by-subsection: every subsection is a self-contained 1–3 paragraph unit.

#### Quality checklist

1. Could a competent CFD reader reproduce the case from Methods alone (with the cited solver version)?
2. Are governing equations present?
3. Are closure / constitutive equations present?
4. Is the geometry described concretely with ≥ 3 markers?
5. Is mesh-independence reported (or its absence justified)?
6. Are comparator and QoI definitions unambiguous?

---

### 5. Results / Experiments

#### Goal

Convince reviewers with complete evidence on **what happened, why, and how it generalizes**. For CFD: the headline number AND the field-level picture (contours) AND the cross-case story (overlays).

#### Three core questions

1. **Is the result better than (or different from) the strong baselines / reference?**
   - Report standard metrics on the canonical reference (DNS / experiment / published correlation).
   - Include the strongest available baseline closure as a comparator, not only weak ones.
   - Keep protocol fair across cases (same mesh, same numerics, same comparator).

2. **Which design choices / parameters drive the observed differences?**
   - For sensitivity sweeps: regress / plot the QoI vs the swept parameter; report the slope or correlation.
   - For OED / code-mod: ablate / vary the modification's coefficient; show monotonic or non-monotonic response.

3. **How far does the conclusion generalize?**
   - For a single-geometry study, state the bounding range of validity explicitly.
   - For OED with `no_winner`: state which families were explored, which were ruled out, and what would extend the search.

#### Section organization

```latex
\section{Results}
\subsection{Baseline validation}
\subsection{Cross-case comparison}     % main story
\subsection{Per-case field visualization} % contours per case
\subsection{Parametric trends}          % regression / correlation if applicable
\subsection{Failure modes}              % cases that went wrong and why
```

#### CFD-specific HARD rules (audit-enforced)

##### a. Per-case figures

For every accepted case in `manifest.json`, at minimum:
- One U (velocity magnitude or streamwise) **PyVista contour** rendered into `paper_figs/<case_id>_u_contour.png`.
- One p (pressure) **PyVista contour** rendered into `paper_figs/<case_id>_p_contour.png`.

Plus, where relevant:
- A Cf / wall-pressure / probe profile per case.

The planner's `figure_jobs[]` MUST declare these with the canonical schema:

```json
{
  "id": "case_001_u_contour",
  "case_id": "case_001",
  "kind": "u_contour",
  "out_png": "paper_figs/case_001_u_contour.png",
  "caption_hint": "Streamwise velocity for case 1"
}
```

Hard rules on the planner output (audit gate H16 enforces these):
- Every `out_png` is **unique** within the plan.
- Every `out_png` actually exists on disk.
- For per-case `kind` values (u_contour, p_contour, cf_profile, …) the `out_png` filename MUST contain the `case_id` substring.
- Reusing a cross-summary PNG for a per-case slot is rejected.

##### b. Cross-experiment figures

At least one figure under `cross_experiment_analysis/` must be embedded via `\includegraphics`, and at least two distinct types must exist (overlay + bar, overlay + trend, …). Audit gates H8 + H12 enforce this.

##### c. Quantitative comparison table

At least one `\begin{tabular}` environment in Results (or Methods) must contain ≥ 3 numeric tokens. For multi-case studies, present a case × QoI table (`booktabs` style) listing the headline numbers. Audit gate H22 enforces this.

#### Table writing rules (CFD adaptation of Master-cai's rules)

1. Use `booktabs`: `\toprule`, `\midrule`, `\bottomrule`. No vertical bars.
2. Label each metric column with units AND direction: `Cf RMSE ↓ (–)`, `x_r/h →` (closer-to-DNS is better), `Re_h (–)`.
3. Bold or highlight the row closest to reference, or the headline result row.
4. Consistent numeric precision within a column.
5. Caption above the table.
6. One table, one message.

#### Hard structural rules

- ≥ 2 paragraphs of Results prose for every case-figure block. Do NOT just paste figures.
- Every per-case figure has a caption that names: (a) the case (e.g., "Re = 200 case"), (b) the field shown, (c) what to look at.
- Every cross-summary figure has a caption that names the comparison axis and the headline conclusion.
- Numbers in the text MUST come from the QoI table or the analysis bundle. Do not invent.

#### Quality checklist

1. Is there at least one per-case figure for every accepted case?
2. Is there a quant comparison table with the headline numbers?
3. Are baselines / references included?
4. Is the parametric trend (if any) plotted, not just narrated?
5. Are failure modes documented when relevant (especially for `no_winner` OED runs)?
6. Are all numbers in the text traceable to the table / analysis bundle?

---

### 6. Discussion

#### Goal

Interpret the results: what do they imply about the closure / physics / parameter regime? Connect back to the Introduction's challenge. Identify failure modes and their causes.

#### Structure (suggested)

1. **Headline interpretation paragraph**: restate the strongest observed difference, frame it in physical terms.
2. **Mechanism paragraph**: explain why the closure / modification / parameter range produced that result. Cite prior CFD work that supports or contradicts the mechanism.
3. **Family / regime coverage paragraph** (for OED / sensitivity studies): state which model families or parameter regions you ruled in / ruled out and the technical reason.
4. **Limitations paragraph**: 2D vs 3D, RANS vs LES, mesh resolution boundaries, time-averaging windows, comparator sensitivity, …
5. **Next-step direction paragraph**: concrete experiment / model variant / dataset that would extend the work.

#### Hard rules

- ≥ 3 `\cite{}` calls in Discussion, woven into mechanism / regime / limitations paragraphs.
- Every mechanistic claim must reference either a number from Results or a prior published result (with citation).
- No "in summary, …" repetition of Results numbers. Discussion adds interpretation; it does not list.

#### Quality checklist

1. Does the Discussion connect back to the Introduction's stated challenge?
2. Are mechanism claims supported by Results numbers or cited prior work?
3. Are limitations honest, specific, and tied to scope (not vague disclaimers)?
4. Is the next-step direction concrete?

---

### 7. Conclusion

#### Structure

1. Restate the problem solved and the core technical / scientific idea.
2. Summarize the strongest evidence from Results in one or two sentences (with numbers).
3. State the practical impact or new insight.
4. Limitations (scope-bounded, not implementation-defect).
5. End with one concrete future direction.

#### Hard rules

- **Single cohesive paragraph of continuous prose.** Never format the Conclusion as bullet points or numbered lists.
- ≤ 250 words.
- No `\cite{}` calls (Conclusion summarizes, does not re-debate prior work).

#### Limitation guidance

Prefer scope-bounded limitations:
- "Validated only on the 2D periodic-hill geometry at Re=5600; transfer to 3D and higher Re is open."
- "Comparator agreement was checked between text and PyVista extractors but not against an independent third extractor."
- "The mesh independence study used the wall-distance and streamwise-density independence criteria; spanwise resolution sensitivity in 3D extensions remains untested."

Avoid limitations framed as fixable implementation defects.

#### Quality checklist

1. Single paragraph?
2. Headline result re-stated with concrete numbers?
3. Scope-bounded limitation present?
4. One concrete future direction?

---

## References & bibliography (HARD rules — audit enforces these)

CFD papers stand on their citations. A paper that cites poorly is judged poorly.

### Source of truth

**`<out-dir>/lit.json`** (or `lit_enriched.json` if present) is the only source for references. Every entry must have:
- `title`, `authors`, `year`, `venue`, `paperId` or `doi`, `abstract` (truncated).
- `source` field — for entries used in the paper, `source == "semanticscholar"` is required (audit H18 enforces this for routes 1/2/5/6).

You may NOT invent citations. You may NOT cite papers absent from `lit.json`.

### Bibliography file (preferred)

Author-time decision: external `paper/refs.bib` (BibTeX) is preferred over an inline `\begin{thebibliography}` block. BibTeX gives:
- One canonical entry per paper, reusable across drafts.
- Better author-year formatting and venue handling.
- `bibtex` compile step in Phase E that catches dangling keys.

Inline `thebibliography` is acceptable as a fallback when bibtex is not available in the build environment. The audit accepts either.

### refs.bib structure

For every `\cite{<key>}` in `main.tex`, there must be a matching `@type{<key>, …}` entry in `paper/refs.bib`. Build entries directly from `lit.json`:

```bibtex
@article{driver_seegmiller_1985,
  author  = {Driver, D. M. and Seegmiller, H. L.},
  title   = {Features of a Reattaching Turbulent Shear Layer in Divergent Channel Flow},
  journal = {AIAA Journal},
  year    = {1985},
  volume  = {23},
  number  = {2},
  pages   = {163--171},
  doi     = {10.2514/3.8890}
}

@article{spalart_allmaras_1992,
  author  = {Spalart, P. R. and Allmaras, S. R.},
  title   = {A One-Equation Turbulence Model for Aerodynamic Flows},
  journal = {AIAA Paper},
  year    = {1992},
  number  = {92-0439}
}
```

### LaTeX preamble for refs.bib + bibtex

```latex
\documentclass{article}
\usepackage{amsmath,amssymb,amsfonts,graphicx,subcaption,booktabs,xcolor,hyperref}
\bibliographystyle{plainnat}    % or unsrtnat / abbrvnat / IEEEtran
\usepackage[numbers]{natbib}    % or remove [numbers] for author-year

% ... document ...

\bibliography{refs}
```

### Inline thebibliography fallback

If you cannot use bibtex (no `bibtex` binary in the compile environment, or the user explicitly requested inline), embed `\begin{thebibliography}` at end of `main.tex`:

```latex
\begin{thebibliography}{99}
\bibitem{driver_seegmiller_1985}
D. M. Driver and H. L. Seegmiller.
\emph{Features of a Reattaching Turbulent Shear Layer in Divergent Channel Flow}.
AIAA Journal, 23(2):163--171, 1985.

\bibitem{spalart_allmaras_1992}
P. R. Spalart and S. R. Allmaras.
\emph{A One-Equation Turbulence Model for Aerodynamic Flows}.
AIAA Paper 92-0439, 1992.
\end{thebibliography}
```

### Hard rules

1. **Coverage (H10)**: ≥ 10 distinct `\cite{<key>}` calls across the body.
2. **Distribution (H23)**: ≥ 1 `\cite{}` in EACH of Introduction, Related Work, Methods, Discussion. No single section may hold > 60% of all citations.
3. **No dangling cites (H19)**: every `\cite{<key>}` has a matching `\bibitem{<key>}` or `@type{<key>, …}` entry.
4. **No unused entries**: every `\bibitem` / `@type` entry is cited at least once in `main.tex`.
5. **Provenance (H18)**: `lit.json` for routes 1/2/5/6 must contain ≥ 5 records with `source: "semanticscholar"`. Empty `lit.json` is auto-rejected — re-invoke `cfd-skills/cfd-literature` first.
6. **Citation weaving**: place `\cite{}` inline at the point where each specific claim is made, NOT clustered at the end of a paragraph as `\cite{a} \cite{b} \cite{c} …`. The audit's H23 distribution check catches end-of-section citation dumps.
7. **Cite by content, not by name dropping**: when you cite, the surrounding sentence must say WHAT the cited work showed / claimed / measured / used, not just that it exists.

### How to author refs.bib from lit.json

Build refs.bib procedurally. The agent's Phase D Python helper should:

```python
import json, re
from pathlib import Path

OUT = Path("<out-dir>")
lit = json.loads((OUT / "lit.json").read_text())
records = lit if isinstance(lit, list) else (lit.get("records") or lit.get("papers") or [])

def make_key(rec):
    first_author = (rec.get("authors") or [{}])[0].get("name", "anon").split()[-1].lower()
    year = rec.get("year", "nd")
    # title prefix for disambiguation
    title_word = re.findall(r"[A-Za-z]+", (rec.get("title") or "x"))[0:1] or ["x"]
    return f"{first_author}_{year}_{title_word[0].lower()[:8]}"

def fmt_authors(rec):
    return " and ".join(a.get("name","Anon") for a in (rec.get("authors") or []))

def make_entry(rec):
    key = make_key(rec)
    title = (rec.get("title") or "Untitled").replace("{","").replace("}","")
    return f"""@article{{{key},
  author  = {{{fmt_authors(rec)}}},
  title   = {{{{{title}}}}},
  journal = {{{rec.get("venue", "")}}},
  year    = {{{rec.get("year","")}}},
  doi     = {{{rec.get("doi", (rec.get("externalIds") or {}).get("DOI", ""))}}}
}}"""

refs_bib = "\n\n".join(make_entry(r) for r in records)
(OUT / "paper" / "refs.bib").write_text(refs_bib)
```

Then `\cite{<key>}` calls in the body refer to these generated keys. **The key the writer must remember**: the key string from `make_key` is deterministic from the lit.json entry — pre-emit a `key_lookup.json` mapping `paperId → key` so the writer can resolve.

---

## Adversarial self-review checklist (run before handing off)

Adapted from Master-cai's 25-question rubric, augmented for CFD.

### Contribution

1. Is the technical / physical contribution clearly stated in Abstract and Introduction?
2. Does the paper solve a real CFD problem (not a synthetic toy variant)?
3. Is the contribution one a reviewer can verify from the experiments shown?

### Clarity

4. Could a competent CFD reader reproduce every case from Methods?
5. Are governing equations present and correctly typeset?
6. Are closure / constitutive equations present for every model under study?
7. Is the geometry described concretely (≥ 3 markers)?
8. Is the QoI extraction recipe unambiguous?
9. Does each paragraph carry one message?
10. Is terminology stable across sections?

### Empirical strength

11. Is at least one strong / canonical baseline included (DNS, published correlation, established closure)?
12. Are headline numbers in a `\begin{tabular}` with ≥ 3 numeric tokens?
13. Is the parametric trend (if any) plotted, not just narrated?
14. Are per-case U and p contours embedded for every accepted case?
15. Are cross-summary figures embedded with at least 2 distinct types?

### Evaluation completeness

16. Is mesh independence reported (or absence justified)?
17. Is comparator agreement documented (text vs PyVista, two-extractor)?
18. For OED with no_winner: are all attempted families enumerated with their failure modes?
19. Are failure cases (numerical divergence, comparator disagreement) acknowledged?

### Soundness

20. Is every claim re-derivable from a number in a Results table?
21. Are any claims overstated relative to the data?
22. Is the limitation discussion specific and scope-bounded (not vague)?
23. Are citations distributed across Introduction, Related Work, Methods, Discussion?
24. Are any `\cite{}` keys dangling (no matching bib entry)?
25. Are any bib entries unused (cited zero times)?

A draft that cannot answer YES to all 25 is not ready. Iterate.

---

## Reviewer rubric (used by cfd-paper Phase F)

Below is the reviewer's evaluation protocol. The reviewer LLM ingests the current `main.tex` + figures + analysis bundle + lit.json and emits `<out-dir>/review.json` matching this schema:

```json
{
  "pass": true|false,
  "score": 0.0-1.0,
  "score_threshold": 0.7,
  "recommendations": ["<actionable item 1>", "<actionable item 2>", ...],
  "formatting_ok": true|false,
  "figures_ok": true|false,
  "content_ok": true|false,
  "length_ok": true|false,
  "word_density_ok": true|false,
  "cross_experiment_figures_ok": true|false,
  "figure_physics_ok": true|false,
  "regenerate_batch_figures": true|false,
  "summary": "<1-2 sentence overall verdict>",
  "prose_quality_issues": [
    {"type": "ngram_loop|token_soup|meta_paragraph_ref|citation_cluster|orphan_figure|missing_governing_eq|missing_closure_eq|missing_geometry|invented_number|...", "section": "<section name>", "evidence": "<short verbatim quote>", "fix": "<what to change>"}
  ]
}
```

### Five-category rejection-risk evaluation (Master-cai)

Mark `pass: false` if the paper falls into any of these:

1. **Insufficient contribution** — a sensitivity sweep with too few cases, a model modification with a single trivial parameter change, an OED with one family that didn't work and no broader coverage statement.
2. **Unclear writing** — missing governing equations / closure equations / geometry, ambiguous QoI extraction, ill-defined comparator, n-gram word loops within paragraphs, meta-paragraph references, citation clusters.
3. **Weak empirical effect** — sensitivity claim not supported by trend plot, model-modification claim not supported by a numerical improvement, OED claim of "search exhausted" not supported by family-coverage table.
4. **Incomplete evaluation** — missing per-case contours, missing cross-experiment overlay, missing mesh-independence statement when data is available, missing reference comparison.
5. **Problematic method design** — unrealistic numerics (CFL > 5, wall y+ > 50 for RANS that needs y+ < 1), implausible parameter ranges, fabricated geometry.

### Verdict logic

- All five categories pass + ≥ 20 of the 25 adversarial-checklist items pass + audit's H1–H27 gates pass → `pass: true`, score = `n_passing_checks / 25`.
- Otherwise → `pass: false`, score = best estimate, `recommendations` lists the top 5–8 most actionable fixes, ordered by impact.

### Reviewer must flag (in `prose_quality_issues`)

1. **N-gram word loops** within any paragraph: a 4–8-word phrase repeated ≥ 3× in the same paragraph. The writer was fluffing length; demand a rewrite of that paragraph.
2. **Meta-paragraph references**: "in introduction paragraph 1", "see paragraph 7", "the previous paragraph". Demand removal — every sentence stands on its own.
3. **Citation clusters**: `\cite{a} \cite{b} \cite{c} \cite{d} …` at end of paragraph. Demand redistribution: weave each cite into the sentence where the cited work's contribution is invoked.
4. **Orphan figures**: a PNG in `paper_figs/` not embedded via `\includegraphics` (or vice versa).
5. **Missing governing / closure equations**: zero display equations matching the markers in the Methods-section HARD rules.
6. **Missing geometry**: < 3 geometry markers in Methods.
7. **Invented numbers**: any number in the text that does NOT trace back to `analysis.json`, `manifest.json`, `bridge.json`, or a `metric_vector`.

### Decision flag

`regenerate_batch_figures: true|false` — set true if any figure has a content problem (axis mismatch, sign-convention error, missing labels, wrong field) that can only be fixed by re-running Phase B. Set false if only the text needs revision.

---

## PyVista loading + figure generation patterns (HARD — use these, not matplotlib placeholders)

Per-case U and p contours, lower-wall Cf line plots, and QoI extraction (reattachment, separation) are produced by **loading each case's OpenFOAM data via PyVista**, not by matplotlib synthetic-data placeholders. The audit gates H28 (per-case coverage) and H29 (PNG size threshold ≥ 80 KB) catch placeholder substitution and fail the run.

### Why this matters

OpenFOAM case directories are first-class data sources. PyVista's `POpenFOAMReader` (and the underlying VTK reader) load the case mesh, the boundary patches, and every field at every saved time. The "contour" you produce must show the **actual solved field**, not a stylized stand-in. Reviewers can tell the difference instantly. The audit can tell from file size (~30 KB matplotlib placeholder vs ~250 KB PyVista field render).

### Canonical case-loading idiom

```python
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
import pyvista as pv
from pathlib import Path

pv.OFF_SCREEN = True

def load_case(case_dir: Path):
    """Load the latest time of an OpenFOAM case via PyVista. Returns the
    internalMesh UnstructuredGrid block. Auto-creates the .foam stub if absent."""
    foam_files = list(case_dir.glob("*.foam"))
    if not foam_files:
        stub = case_dir / "case.foam"
        stub.touch()
        foam_files = [stub]
    reader = pv.POpenFOAMReader(str(foam_files[0]))
    # Latest non-zero time. If there is only "0/", we still get something.
    times = list(reader.time_values)
    target_time = times[-1] if times else 0.0
    reader.set_active_time_value(target_time)
    mesh = reader.read()
    internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh[0]
    return internal, target_time, reader
```

### U-magnitude contour (per case)

```python
def render_u_contour(case_dir: Path, out_png: Path,
                    title: str = "", window_size=(1600, 800)):
    """Render velocity-magnitude contour from the latest time of the OpenFOAM case.
    Writes a journal-quality PNG with colorbar, axes hidden, full domain in frame."""
    internal, t, _ = load_case(case_dir)
    if "U" not in internal.array_names:
        raise RuntimeError(f"U field missing at t={t} in {case_dir}")
    # Magnitude
    import numpy as np
    U = internal["U"]
    internal["Umag"] = np.linalg.norm(U, axis=1)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.add_mesh(
        internal, scalars="Umag", cmap="viridis",
        show_edges=False,
        scalar_bar_args=dict(
            title="|U|  [m/s]", n_labels=5, fmt="%.2f",
            position_x=0.78, position_y=0.10, width=0.18, height=0.55,
        ),
    )
    p.view_xy()
    p.camera.parallel_projection = True
    p.add_text(title, position="upper_left", font_size=10)
    p.screenshot(str(out_png), transparent_background=False)
    p.close()
```

### Pressure contour (per case)

Same pattern as U-magnitude, with `scalars="p"` and `cmap="coolwarm"`. Center the colorbar at the median pressure when the range spans both signs.

### Lower-wall Cf line plot (per case AND cross-case overlay)

OpenFOAM stores wall shear stress in `wallShearStress` (built by `simpleFoam`/`pimpleFoam` post-processing) or you compute it from `nut * dU/dn` at the bottomWall patch. PyVista's MultiBlock reader exposes boundary patches:

```python
def extract_lower_wall_cf(case_dir: Path, U_ref: float, rho: float = 1.0):
    """Return (x_over_h, Cf) arrays on the lower wall at the latest time.
    Cf = tau_w_x / (0.5 rho U_ref^2). Sign convention: positive Cf in the
    main flow direction.
    """
    foam_files = list(case_dir.glob("*.foam"))
    reader = pv.POpenFOAMReader(str(foam_files[0]))
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    # Patches live under boundary/*. Names depend on the case; check the case constant/polyMesh/boundary.
    if "boundary" in mesh.keys():
        boundary = mesh["boundary"]
        # bottomWall, lowerWall, walls — name varies. Iterate and pick.
        target = None
        for name in boundary.keys():
            if any(w in name.lower() for w in ("bottom", "lower", "wall")):
                target = boundary[name]
                break
        if target is None:
            raise RuntimeError(f"no lower-wall patch under boundary/ in {case_dir}")
    else:
        raise RuntimeError(f"no boundary block in {case_dir} — case not decomposed?")
    if "wallShearStress" not in target.array_names:
        raise RuntimeError("wallShearStress not on lower wall — run "
                           "postProcess -func wallShearStress before extraction")
    import numpy as np
    centers = target.cell_centers().points  # (Nc, 3)
    wss = target["wallShearStress"]         # (Nc, 3)
    # x-component (streamwise) of wall stress. Sign = main flow direction.
    tau_x = wss[:, 0]
    Cf = tau_x / (0.5 * rho * U_ref**2)
    # Sort by x
    order = np.argsort(centers[:, 0])
    return centers[order, 0], Cf[order]
```

For the cross-case overlay (`cross_experiment_analysis/cf_overlay.png`), call this once per case, plot each curve, add a DNS reference if present in `reference_data/`, save via matplotlib with proper labels (`x/h`, `C_f`, units, legend, gridlines, colorbar-or-color-per-case).

### Reattachment x_r/h extraction

```python
def extract_reattachment_x(case_dir: Path, U_ref: float, x_window=(1.0, 30.0)):
    """Locate the first lower-wall Cf sign crossing downstream of x_window[0]
    that returns Cf to positive. Returns x_r/h or NaN if no crossing."""
    import numpy as np
    x, cf = extract_lower_wall_cf(case_dir, U_ref)
    mask = (x >= x_window[0]) & (x <= x_window[1])
    x, cf = x[mask], cf[mask]
    # Find first index where cf transitions from negative to positive
    for i in range(1, len(cf)):
        if cf[i-1] < 0 and cf[i] >= 0:
            # Linear interpolation for the crossing
            return x[i-1] - cf[i-1] * (x[i] - x[i-1]) / (cf[i] - cf[i-1])
    return float("nan")
```

### Hard rules for PyVista figures

1. **Always set `off_screen=True` / `PYVISTA_OFF_SCREEN=true` + `MPLBACKEND=Agg`** before opening any Plotter (CI / headless servers).
2. **Wrap your script in `xvfb-run -a python …`** so a virtual X server is available for GL contexts.
3. **Use a real OpenFOAM `.foam` file** in the case directory. If absent, create an empty one (`Path("case.foam").touch()`) before passing to `POpenFOAMReader`.
4. **Select the latest non-zero time** (`reader.time_values[-1]`). Skip 0/ for converged steady or developed transient cases.
5. **Render at ≥ 1200 × 600 pixels**. Periodic-hill / BFS / channel geometries are wide; use a 2:1 or 3:1 aspect ratio.
6. **Use `parallel_projection`** to avoid the default perspective squish.
7. **Fit the geometry to the frame** — the mesh must FILL the rendered PNG, not float in a sea of background. After `view_xy()` and `reset_camera()`, tighten the camera so there is at most a thin margin of empty space around the mesh:

   ```python
   p = pv.Plotter(off_screen=True, window_size=(1600, 800))
   p.add_mesh(internal, scalars=scalars, cmap=cmap, show_edges=False, scalar_bar_args=...)
   p.view_xy()
   p.camera.parallel_projection = True
   p.reset_camera()
   p.camera.tight(padding=0.02, view="xy")   # PyVista ≥ 0.43: tight fit to mesh bounds
   # Compatibility fallback for older PyVista:
   try:
       p.camera.tight(padding=0.02, view="xy")
   except (AttributeError, TypeError):
       # Manual tight-fit: set parallel scale from the mesh y-extent
       (xmin, xmax, ymin, ymax, _, _) = internal.bounds
       cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
       p.camera.SetFocalPoint(cx, cy, 0.0)
       p.camera.SetPosition(cx, cy, 1.0)
       p.camera.SetViewUp(0.0, 1.0, 0.0)
       p.camera.SetParallelScale(0.5*(ymax-ymin)*1.02)  # 2% margin
   p.screenshot(out_png, transparent_background=False)
   p.close()
   ```

   The mesh's domain (inlet, outlet, walls, hill / step / obstacle, recirculation region) must occupy ≥ 85% of the rendered frame area. A reviewer who opens the figure should see the geometry filling the image, not a postage stamp on a white field. If the rendered PNG shows large empty borders, the camera was not tightened — re-render with the `tight()` call above.
8. **Colorbar inside the frame on the right** at ~78% width, height 55% — never let it cover the geometry.
9. **Never fall back to matplotlib synthetic data** when PyVista fails. If PyVista fails, surface the error and re-render from the real case — do not write a placeholder. The audit's H29 catches placeholder substitution by file size.
10. **One render per case per field**. Do not reuse a baseline's PNG for a candidate.

### Hard rules for QoI extraction (text + PyVista agreement)

For every QoI a paper claims (Cf RMSE, x_r/h, peak Cf, ...), extract it **twice**: once from OpenFOAM's `postProcess -func` output (text), once from PyVista's reader pipeline. Compare with relative tolerance ≤ 1e-3. Disagreement larger than that means one extractor is wrong — fix it before writing any number into the paper.

The OED loop's `bound_comparators.json` already implements this for its primary objective. Apply the same discipline for every QoI surfaced in the paper's Results table.

## Calling convention (how `cfd-paper` invokes this skill)

The `cfd-paper` skill's Phase D (writer / reviser) and Phase F (reviewer) invoke this skill by **embedding its current SKILL.md as the system prompt** for the relevant LLM call, then appending the per-call context (section_context JSON, figure paths, prior reviewer feedback, etc.) as the user message.

This skill is content — not a code module. The agent does not invoke it via `Skill cfd-skills/cfd-paper-writer` directly; instead, `cfd-paper` reads this file and uses its content as the writer / reviewer system prompt for those specific LLM calls. The same content thus drives both the writer and the reviewer, ensuring symmetry.

## Next

This skill has no `## Next`. It is a content skill — invoked by `cfd-paper`. After Phase D or Phase F completes the LLM call, control returns to the cfd-paper cycle.

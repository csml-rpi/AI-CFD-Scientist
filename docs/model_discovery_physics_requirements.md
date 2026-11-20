# Physics requirements for model-discovery prompts (draft, 2026-11-20)

Wired in (2026-11-20) from `src/cfd_langgraph/model_requirements.py`, solver studies only:
- the open-ended discovery proposer (`manager/tools.py`, `oed_propose_candidates`); each candidate
  now carries a `requirements_check`, saved in `proposals.json`;
- the build agent (`scripts/code_mod_agentic.py`), which implements the hypothesis as written and
  reports any MUST HOLD item it breaks rather than changing the model;
- the `nn-coupler` subagent, for a network's inputs, outputs and bounds.
All three sections are shown; the reader applies the one that fits. Fitted-model (surrogate) and
implementation studies do not get them.

Two tiers, following Spalart (2023): **must hold** (a candidate that breaks one is not a model) and
**break only with a stated reason** (the candidate's plan must say which one it breaks and why).
Each candidate states, in its plan, how it meets the must-hold list.

---

## A. Turbulence closures (RANS / transition / added stress terms)

Must hold
1. Dimensionally consistent; every added coefficient is dimensionless. No dimensional floors
   hidden in max(k, 1e-14) style limiters without saying so.
2. Independent of the coordinate axes: no fixed direction vectors (e.g. a hard-coded streamwise
   (1 0 0)), no "normal vs shear" stress components. Tensors built from invariant bases
   (strain, rotation, Pope's integrity basis); scalar inputs are invariants.
3. Galilean invariant: no mean velocity itself, no quantities that change under a uniform
   velocity shift.
4. Well-posed: no negative effective diffusion, no term that makes short waves grow; result must
   hold up under mesh refinement.
5. Realizable / bounded: turbulence variables stay non-negative (k, ε, ω, ν̃, ν_t ≥ 0); an
   added Reynolds stress keeps the stress tensor positive semi-definite.
6. No properties of this particular flow: no freestream velocity, Reynolds number, chord, hill
   height, case-specific coordinates or zones.
7. Keeps the baseline's calibrated behaviour where the baseline is right: law of the wall and
   flat-plate skin friction, free shear layers / wakes, decaying turbulence, and sensible
   behaviour at the edge of a turbulent region (turbulence must not grow into or collapse
   before the free stream). A correction should switch itself off in those flows.

Break only with a stated reason
- Pressure gradient or acceleration as an input (not Galilean invariant with moving walls).
- Quantities along the flow direction (streamline curvature, helicity, d/ds).
- Wall quantities at a field point (u_τ, wall-normal vector, wall distance beyond the usual d).
- Boundary-layer quantities (thickness, edge velocity), von Kármán length.
- Molecular viscosity away from walls (needs a clear infinite-Re limit).
- Non-smooth functions (min, max, abs, hard switches) and high derivatives.
- Uncontrolled growth of turbulence at stagnation points.
- Large operation count (e.g. a big network evaluated every cell every iteration).

## B. Numerical flux functions / Riemann solvers / schemes

Must hold
1. Consistency: F(u, u; n) equals the physical flux f(u)·n.
2. Conservation: one flux per face, F(uL, uR; n) = -F(uR, uL; -n).
3. Rotational invariance and mirror symmetry: computed in the face-normal frame; swapping left
   and right with the normal flipped gives the mirrored flux.
4. Galilean invariance where the equations have it (same flux structure in a moving frame).
5. Entropy condition: no expansion shocks (entropy fix at sonic points where needed).
6. Positivity: density and pressure stay positive for states where the exact solution does.
7. Zero mass flux through a solid wall; uniform flow stays uniform (freestream preservation).
8. Lipschitz continuous in the states; no case-specific constants (Mach, freestream, geometry).

Break only with a stated reason
- Shock stability: no carbuncle / odd-even decoupling on grid-aligned strong shocks.
- Contact and shear resolution: stationary contacts and shear layers kept sharp (low dissipation).
- All-speed behaviour: numerical dissipation scales correctly as Mach -> 0 (pressure ~ M²).
- Differentiability (needed for implicit solvers and adjoints).
- Order of accuracy and monotonicity of the reconstruction it is paired with.

## C. Any other solver model (wall functions, source terms, BCs, NN inside the solver)

Must hold
1. Dimensionally consistent; frame and Galilean invariant inputs and outputs.
2. Only local, solver-available quantities; nothing tied to this case's geometry or conditions.
3. Conserves what the equations conserve; keeps bounded variables in bounds.
4. Reduces to the validated baseline in the flows the baseline already gets right.
5. Stable and mesh-convergent.

Break only with a stated reason
- Inputs outside the range the model was fitted on (state how it is clipped or extrapolated).
- Non-smooth switches; large per-cell cost.
- For learned models: unnormalised or non-invariant inputs, unbounded outputs.

---

## Sources
- Spalart, "An Old-Fashioned Framework for Machine Learning in Turbulence Modeling",
  arXiv:2308.00837 (2023): the Hard / Soft lists A is built on.
- Spalart & Speziale, "A note on constraints in turbulence modelling", JFM 391 (1999):
  why acceleration / pressure gradient are not admissible inputs.
- Cazalbou, Spalart & Bradshaw, Phys. Fluids 6 (1994): edge of a turbulent region.
- Pope, JFM 72 (1975): integrity basis for invariant stress models.
- Schumann, Phys. Fluids 20 (1977): realizability.
- Ling, Kurzawski & Templeton, JFM 807 (2016): invariant inputs for learned closures.
- Duraisamy, Iaccarino & Xiao, Annu. Rev. Fluid Mech. 51 (2019): review of data-driven closures.
- Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: flux consistency, conservation,
  entropy.
- Einfeldt, Munz, Roe & Sjögreen, JCP 92 (1991): positivity near low densities.
- Quirk, IJNMF 18 (1994): carbuncle and shock instability.
- Guillard & Viozat, Comput. Fluids 28 (1999); Shima & Kitamura, AIAA J 49 (2011): low-Mach
  dissipation scaling.
- Zhang et al., "Learning the Exact Flux: Neural Riemann Solvers with Hard Constraints",
  arXiv:2603.30007: positivity, consistency, mirror symmetry, Galilean and scaling invariance
  for learned fluxes.

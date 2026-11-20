"""What a new solver model must satisfy to count as a model, whatever it scores.

Shown to the open-ended-discovery proposer and to the build agent in solver
studies. Without it the search optimises the score alone, and a score can be
bought with things no general model may use: the periodic-hill winner
(ph_codex_20260902_1806, iteration 58) reads a hard-coded streamwise direction
(1 0 0) and the pressure gradient, both on Spalart's list of hard constraints.

All three sections are shown and the reader applies the one that fits what the
candidate modifies. Picking a section in code would mean classifying the
hypothesis by its words, and a misclassified candidate would be told the wrong
rules with no sign anything went wrong.

Sources: Spalart, arXiv:2308.00837 (2023), hard and soft constraints; Spalart &
Speziale, JFM 391 (1999); Cazalbou, Spalart & Bradshaw, Phys. Fluids 6 (1994);
Pope, JFM 72 (1975); Schumann, Phys. Fluids 20 (1977); Ling, Kurzawski &
Templeton, JFM 807 (2016); Toro, Riemann Solvers and Numerical Methods for
Fluid Dynamics; Einfeldt et al., JCP 92 (1991); Quirk, IJNMF 18 (1994);
Guillard & Viozat, Comput. Fluids 28 (1999); Zhang et al., arXiv:2603.30007.
See docs/model_discovery_physics_requirements.md.
"""

MODEL_REQUIREMENTS = """\
WHAT MAKES IT A MODEL, NOT A FIT TO THIS CASE
Use the section that matches what the candidate modifies. MUST HOLD: a candidate that
breaks one of these is not a usable model, however well it scores. STATED REASON ONLY:
allowed, but say which one the candidate uses and why it is needed.

A. Turbulence closures (RANS equations, transition, eddy viscosity, added stresses)
  MUST HOLD
  1. Consistent units; every new coefficient is dimensionless.
  2. Independent of the coordinate axes: no fixed direction vectors such as a hard-coded
     streamwise (1 0 0), no "normal" vs "shear" stress components. Build tensors from
     strain and rotation (an invariant tensor basis) and use invariant scalar inputs.
  3. Galilean invariant: no mean velocity itself, nothing that changes under a uniform
     shift of velocity.
  4. Well-posed: no negative effective diffusion, no term that makes short waves grow;
     the result must hold under mesh refinement.
  5. Realizable and bounded: k, epsilon, omega, nuTilda and nut stay non-negative; an
     added Reynolds stress keeps the stress tensor positive semi-definite.
  6. Nothing about this particular flow: no freestream velocity, Reynolds number, body
     size, case coordinates or zones.
  7. Keeps what the baseline already gets right: the law of the wall and flat-plate skin
     friction, wakes and free shear layers, decaying turbulence, and a turbulent region
     that neither grows into nor collapses before the free stream. A correction should
     switch itself off in those flows.
  STATED REASON ONLY
  - Pressure gradient or acceleration as an input (not invariant once walls move).
  - Quantities along the flow direction: streamline curvature, helicity, d/ds.
  - Wall quantities at a field point (friction velocity, wall-normal vector), boundary-
    layer thickness or edge velocity, the von Karman length.
  - Molecular viscosity away from walls (state the infinite-Reynolds-number limit).
  - Non-smooth functions (min, max, abs, hard switches) or high derivatives.
  - Turbulence growth near stagnation points left uncontrolled.
  - A large per-cell cost, such as a big network evaluated every iteration.

B. Numerical fluxes, Riemann solvers and discretisation schemes
  MUST HOLD
  1. Consistent: F(u, u; n) equals the physical flux f(u).n.
  2. Conservative: one flux per face, F(uL, uR; n) = -F(uR, uL; -n).
  3. Rotationally invariant and mirror symmetric: computed in the face-normal frame.
  4. Galilean invariant where the equations are.
  5. Entropy condition: no expansion shocks (entropy fix at sonic points if needed).
  6. Density and pressure stay positive wherever the exact solution keeps them positive.
  7. Zero mass flux through a solid wall; a uniform flow stays uniform.
  8. Continuous in the left and right states; no constants tied to this case's Mach
     number, freestream or geometry.
  STATED REASON ONLY
  - Shock instability (carbuncle, odd-even decoupling) on grid-aligned strong shocks.
  - Smeared stationary contacts or shear layers.
  - Numerical dissipation that does not scale correctly as Mach number goes to zero.
  - Non-differentiable switches (matter for implicit solvers and adjoints).

C. Any other solver model (wall functions, source terms, boundary conditions, a network
   inside the solver)
  MUST HOLD
  1. Consistent units; inputs and outputs independent of the coordinate axes and of a
     uniform velocity shift.
  2. Only local quantities the solver has; nothing tied to this case's geometry or
     conditions.
  3. Conserves what the equations conserve; bounded variables stay in bounds.
  4. Reduces to the validated baseline in flows the baseline already gets right.
  5. Stable and mesh-convergent.
  STATED REASON ONLY
  - Inputs outside the range the model was fitted on (state how they are clipped).
  - Non-smooth switches or a large per-cell cost.
  - For a learned model: unnormalised or non-invariant inputs, unbounded outputs.
"""


def model_requirements_block() -> str:
    """The requirements, framed as a prompt section."""
    return (
        "============================================================\n"
        + MODEL_REQUIREMENTS
        + "============================================================\n"
    )

Open-ended discovery: find a novel Spalart-Allmaras (SA) turbulence model
modification for periodic-hill flow at Re_h = 5600 that improves lower-wall
skin-friction (Cf) prediction over the baseline SA model.

Starter case: starter_oed_turbulence/periodic_hill_sa
  Working OpenFOAM 10 case (simpleFoam, SA, periodic hill Re_h=5600), already
  meshed and solved. Its converged baseline scores Cf RMSE = 0.004297 against
  the DNS reference.

Reference data and comparator:
  starter_oed_turbulence/reference_data/compare_exactmatch_cf.py is the
      authoritative scoring method. It produced the 0.004297 baseline. Read it
      and follow it exactly.
  starter_oed_turbulence/reference_data/reference_exactmatch_cf.csv
      columns x_over_h, cf_dns_exactmatch — DNS lower-wall Cf at 766 stations.
      The case's bottomWall patch has 99 faces, so the simulated Cf curve must
      be INTERPOLATED onto the 766 reference x_over_h stations before the RMSE
      is taken. The two grids do not match; do not pair them face-by-face.
  Cf per wall face = -2.0 * wallShearStress_x / Ub^2, with Ub read from the
      case's constant/transportProperties (0.028 here) and h = 1.0. Note the
      minus sign: dropping it inverts the curve.
  starter_oed_turbulence/periodic_hill_sa/reference_exactmatch_cf.csv is a copy
      of the same 766-row reference.

Objective: minimise RMSE between the simulated lower-wall Cf and the DNS Cf
over x/h. Target at least 30% improvement over the baseline, i.e. Cf RMSE
<= 0.003008.

Constraints:
  - Modify only the turbulence closure. Do not change the mesh, the numerics,
    the boundary conditions, or the reference data.
  - Never edit the OpenFOAM installation. Build case-local libraries only,
    under the candidate case's own customModels/.
  - Expose every new coefficient through the model's runtime coeffDict, so a
    later candidate can vary it without a rebuild.
  - starter_oed_turbulence/other_re/ is HELD-OUT test data for Re=2800 and
    Re=10595. Do not read it, score against it, or let it influence any
    candidate during the search. It is evaluated once, by hand, after the
    study has finished.

Before running any candidate, verify the scoring method reproduces Cf RMSE
= 0.004297 on the already-solved starter case. If it does not, the extraction
is wrong — fix it before spending a solver run.

Deliverable: when the search finishes, write the work up as a LaTeX paper and
compile it to a PDF inside the run directory — problem and motivation, methods
and case setup, results with figures and a quantitative comparison table,
discussion, conclusions.

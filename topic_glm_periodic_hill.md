Open-ended discovery: find a novel Spalart-Allmaras (SA) turbulence model
modification for periodic-hill flow at Re_h = 5600 that improves lower-wall
skin-friction (Cf) prediction over the baseline SA model.

Starter case: starter_oed_turbulence/periodic_hill_sa
  Working OpenFOAM 10 case (simpleFoam, SA, periodic hill Re_h=5600), already
  meshed and solved. Its converged baseline scores Cf RMSE = 0.004297 against
  the DNS reference.

Reference data and comparator:
  starter_oed_turbulence/periodic_hill_sa/reference_exactmatch_cf.csv
      columns x_over_h, cf_dns_exactmatch — DNS lower-wall Cf, already sampled
      at the case's own wall face centres, so no interpolation is needed.
  starter_oed_turbulence/reference_data/ — the same reference plus a
      comparison script.

Objective: minimise RMSE between the simulated lower-wall Cf and the DNS Cf
over x/h. Target at least 30% improvement over the baseline.

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

Deliverable: when the search finishes, write the work up as a LaTeX paper and
compile it to a PDF inside the run directory — problem and motivation, methods
and case setup, results with figures and a quantitative comparison table,
discussion, conclusions.

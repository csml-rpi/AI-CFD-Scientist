# Periodic Hill `case_1p0` SA Baseline, Procedural 2D

This is a clean OpenFOAM v10 steady RANS baseline built from the archived
`case_1p0` benchmark-style SA setup, but with a fully procedural 2D mesh so
the case can be rebuilt from `blockMeshDict` and run through `Allrun`.

Case choices:

- Solver: `simpleFoam`
- Turbulence model: `SpalartAllmaras`
- Geometry: `alpha = 1.0`, `Lx/h = 9`, `Ly/h = 3.036`
- Driving method: `meanVelocityForce`
- Reynolds number: `Re_h = U_b h / nu = 5600`

Scaling used here:

- Mesh coordinates are nondimensional with `h = 1`
- `nu = 5.0e-06`
- Bulk velocity `U_b = 0.028`
- Volume-averaged forcing target `Ubar = 0.020188`
- The benchmark relation is `U_b = Ubar / 0.721`

Source meshes and references:

- Mesh source: `xiaoh/para-database-for-PIML`, `pehill-5-cases-OpenFOAM/case_1p0`
- Matching DNS: `reference_dns/alph10-9-3036`
- Matching experiment at the same standard geometry and `Re = 5600`:
  `reference_experiment/rapp_manhart_Re5600/profiles`

Important note:

- This folder preserves the archived benchmark-style scaling and forcing:
  `h = 1`, `U_b = 0.028`, `Ubar = 0.020188`, `nu = 5.0e-06`.
- The procedural mesh uses the archived patch naming:
  `bottomWall`, `topWall`, `inlet`, `outlet`, `defaultFaces`.

## What this handoff folder contains

- a fully procedural OpenFOAM v10 SA case
- `blockMeshDict`
- parallel `Allrun` and `Allclean`
- a local exact-match benchmark wall-friction reference:
  `reference_exactmatch_cf.csv`
- a local comparison helper:
  `compare_exactmatch_cf.py`
- a convenience wrapper:
  `Allcompare`

## Recommended use

1. `source /opt/openfoam10/etc/bashrc`
2. `./Allclean`
3. `./Allrun`
4. `./Allcompare 5000`

This will regenerate:

- `comparison_exactmatch/5000/01_cf_vs_exactmatch_reference.png`
- `comparison_exactmatch/5000/summary.md`

## Reference notes

This handoff folder compares against the local benchmark file:

- `reference_exactmatch_cf.csv`

That file was exported from the repo's archived periodic-hill validation pack
so that this folder stays self-contained.

For full provenance, see:

- `REFERENCE_PROVENANCE.md`

## Literature behavior to expect

For the standard periodic-hill `Re_h = 5600` benchmark, baseline SA typically:

- predicts a longer separated region than DNS / experiment
- delays recovery on the downstream hill
- therefore stays lower than the benchmark through much of the `x/h > 8`
  recovery shoulder
- can then peak later and more broadly than the benchmark

That means the trusted benchmark comparison is **not** "DNS must always stay
above SA for all `x/h > 8`". A more accurate expectation is:

- DNS/reference rises earlier and is larger over most of the shoulder
- SA can overtake locally only after the benchmark peak has already passed

Useful external references for the benchmark family:

1. Krank, Kronbichler, Wall (2018), TUM publication page:
   `https://portal.fis.tum.de/en/publications/direct-numerical-simulation-of-flow-over-periodic-hills-up-to-re-/`
2. Direct wall-data DOI listed in the local KKW wall-reference file:
   `http://doi.org/10.14459/2018mp1415670`
3. Xie et al. (2021) periodic-hill data-driven SA comparison:
   `https://arxiv.org/abs/2101.10528`

These references are provided for physical context. The file actually used by
`Allcompare` is still the local `reference_exactmatch_cf.csv` in this folder.

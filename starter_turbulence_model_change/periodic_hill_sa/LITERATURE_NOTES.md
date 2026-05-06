# Periodic-Hill Literature Notes

This note is included so the handoff package documents the expected qualitative
behavior of the baseline Spalart--Allmaras model on the standard periodic-hill
benchmark.

## Benchmark family

- Geometry: standard periodic hill, `alpha = 1.0`
- Reynolds number: `Re_h = 5600`
- Local experiment folder used elsewhere in this repo:
  `/work/openfoam-10/run/periodic_hill_paper/reference_experiment/rapp_manhart_Re5600`
- Local DNS family used elsewhere in this repo:
  `/work/openfoam-10/run/periodic_hill_paper/reference_dns/alph10-9-3036`
- Direct wall-data reference file used elsewhere in this repo:
  `/work/openfoam-10/run/periodic_hill_paper/newObjective/reference/KKW_DNS_Periodic_Hill_Re5600_cf_cp_bottom.dat`

## External reference sources

1. Krank, Kronbichler, Wall (2018), TUM page:
   `https://portal.fis.tum.de/en/publications/direct-numerical-simulation-of-flow-over-periodic-hills-up-to-re-/`
2. DOI listed in the local direct wall-data file:
   `http://doi.org/10.14459/2018mp1415670`
3. Xie et al. (2021), data-driven SA comparison on periodic hills:
   `https://arxiv.org/abs/2101.10528`

## Expected baseline-SA behavior

For this benchmark, baseline SA is expected to:

- separate reasonably near the first sign change
- reattach too late relative to DNS / experiment
- recover too slowly on the downstream hill

In wall-friction terms this usually means:

- over much of the `x/h > 8` recovery shoulder, DNS/reference is larger than SA
- SA can still become locally larger after the reference peak because the SA
  recovery peak is delayed and broadened

This note is only for physical interpretation. The actual comparison in this
handoff package is defined by:

- `reference_exactmatch_cf.csv`
- `compare_exactmatch_cf.py`

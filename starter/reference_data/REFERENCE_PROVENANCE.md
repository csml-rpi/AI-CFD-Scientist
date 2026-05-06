# Reference Provenance

This folder is self-contained for rerunning the case and regenerating the
benchmark wall-friction comparison.

## Actual reference file used by `Allcompare`

- File used directly by this handoff package: `reference_exactmatch_cf.csv`
- Script that uses it: `compare_exactmatch_cf.py`
- Output location after running: `comparison_exactmatch/<time>/`

This local CSV is the comparison target for the handoff case. It is the file
your friend will actually use when they run `./Allcompare 5000`.

## What `reference_exactmatch_cf.csv` contains

The CSV contains two columns:

- `x_over_h`
- `cf_dns_exactmatch`

These values are the benchmark-style periodic-hill wall-friction reference used
for the `case_1p0` SA validation workflow in this repo.

## Exact local provenance chain

The local CSV in this folder was exported from:

- `/work/openfoam-10/run/periodic_hill_paper/finalizedModelMarch/plots/validations/periodic_hill_family_plus_training_20260323/results/summary.json`

Specifically from:

- row with `case_key = training_case`

That validation summary was created by:

- `/work/openfoam-10/run/periodic_hill_paper/finalizedModelMarch/scripts/build_periodic_hill_validation_pack.py`

For the training case, that script uses:

- SA case: `/work/openfoam-10/run/periodic_hill_paper/periodic_hill_case1p0_sa_v10`
- DNS directory: `/work/openfoam-10/run/periodic_hill_paper/reference_dns/alph10-9-3036`

The validation-pack script calls:

- `/work/openfoam-10/run/periodic_hill_paper/compare_periodic_hill_cf_models.py`

and for the training case that script:

1. loads DNS mean-field data from:
   - `reference_dns/alph10-9-3036/mean_files.dat`
2. reconstructs DNS wall-friction along the curved bottom wall via
   `compute_dns_cf(...)`
3. stores the resulting benchmark `Cf` curve in the validation summary
4. from that summary, `reference_exactmatch_cf.csv` was exported for this
   handoff folder

So the handoff-package comparison is:

- **not** using the old KKW direct wall file directly
- **not** using the noisy cloned-case comparison path
- **yes** using the exact-match benchmark curve already used by the archived
  `case_1p0` SA validation workflow inside this repo

## Published benchmark family behind the local DNS data

The periodic-hill `Re_h = 5600` benchmark used in this repo belongs to the
standard periodic-hill DNS / experiment family used widely in the literature.

Useful external references for that benchmark family are:

1. Krank, Kronbichler, Wall (2018), direct DNS wall data source used elsewhere
   in this repo:
   - DOI listed in the local file header:
     `http://doi.org/10.14459/2018mp1415670`
   - local file:
     `/work/openfoam-10/run/periodic_hill_paper/newObjective/reference/KKW_DNS_Periodic_Hill_Re5600_cf_cp_bottom.dat`
   - TUM publication page:
     `https://portal.fis.tum.de/en/publications/direct-numerical-simulation-of-flow-over-periodic-hills-up-to-re-/`

2. Rapp and Manhart experimental profile dataset used in this repo for the same
   benchmark geometry / Reynolds number:
   - local folder:
     `/work/openfoam-10/run/periodic_hill_paper/reference_experiment/rapp_manhart_Re5600`

## Important distinction

The comparison in this handoff folder uses:

- local exact-match benchmark CSV:
  `reference_exactmatch_cf.csv`

That CSV traces back to the repo's archived training-case validation workflow
and local DNS folder:

- `reference_dns/alph10-9-3036`

I did **not** find a direct download URL stored alongside `alph10-9-3036`
inside this repo. So the external published benchmark family is documented
above, but the exact handoff CSV itself comes from the local validation chain
described here.

## Expected commands

1. `source /opt/openfoam10/etc/bashrc`
2. `./Allclean`
3. `./Allrun`
4. `./Allcompare 5000`

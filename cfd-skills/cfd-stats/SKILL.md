---
name: cfd-stats
description: Numerical analysis patterns for OpenFOAM run outputs — log parsing, D²-law fits, slope/RMSE/Richardson extrapolation, residual-plateau checks, drift checks. Self-contained — the agent writes 30-line Python snippets via Bash using these patterns. No Python wrapper.
---

# cfd-stats

A reference book of small, well-tested numerical patterns. Each pattern is a self-contained recipe for the agent to drop into a Python snippet and run via Bash. No Python framework code, no helper imports beyond `numpy`/`scipy`.

When invoked, the agent picks the patterns it needs, composes them into a single Python script, runs it, and writes the result as JSON in the case or run directory. This skill is consulted by `cfd-interpret`, `cfd-analyze`, `cfd-mesh-gate`, `cfd-open-discovery`, and `cfd-paper`.

## Inputs (per pattern)
Each pattern declares its own inputs. Most patterns take one or more of:
- A path to `log.<solver>` for time-series extraction
- A path to a `postProcessing/<functionObject>/<time>/<file>` table
- A path to `cloudOutputProperties` or other Lagrangian outputs
- Reference data CSV/JSON paths

## Outputs
Each pattern returns a small JSON dict (numbers, units, fit quality). The agent persists it to `<case_dir>/<pattern>_result.json` or `<run_dir>/<pattern>_summary.json`.

---

## Pattern 1 — Droplet K from log mass-history (D²-law)

**Use when:** measuring evaporation-rate constant K [mm²/s] of a single suspended Lagrangian droplet vs a topic-supplied reference correlation `K_ref(T_inf)` (linear in T, tabulated, or otherwise). **Do NOT read `cloud:rhoTrans_*` Eulerian source fields or `cloudOutputProperties.massInjected` — both are constant by design at write times** (see anti-patterns at the bottom of this skill).

**Why log-parsing:** `log.<solver>` writes `Time = X` followed by `Current mass in system = Y` at every solver timestep (typically 50–200 samples/s). This is the live droplet mass — captured at every solver step, not just at write times. Works on any solver that runs an OpenFOAM Lagrangian cloud (e.g., `reactingFoam`, `sprayFoam`).

**Snippet template:**
```python
import re, json
from pathlib import Path

CASE  = Path("<case_dir>")
SOLVER_LOG = "<log.reactingFoam | log.sprayFoam | ...>"   # whatever solver controlDict named
log = (CASE / SOLVER_LOG).read_text(errors="replace")

rows, t = [], None
for line in log.splitlines():
    m = re.match(r"Time =\s*([-+0-9.eE]+)", line)
    if m:
        t = float(m.group(1)); continue
    m = re.search(r"Current mass in system\s*=\s*([-+0-9.eE]+)", line)
    if m and t is not None:
        rows.append((t, float(m.group(1))))

if len(rows) < 5:
    raise SystemExit("too few mass samples — solver may have crashed early")

# D²-law: D² = D₀² - K·t. With m = ρ·(π/6)·D³, D ∝ m^(1/3), so D²/D₀² = (m/m₀)^(2/3).
# Fit slope of y = (m/m₀)^(2/3) vs t in the linear window [0.05, 0.95].
positive = [(t_, m_) for t_, m_ in rows if m_ > 0]
m0 = max(m_ for _, m_ in positive)
fit = [(t_, (m_/m0)**(2/3)) for t_, m_ in positive if 0.05 <= (m_/m0)**(2/3) <= 0.95]
if len(fit) < 3:
    fit = [(t_, (m_/m0)**(2/3)) for t_, m_ in positive]

n = len(fit)
sx  = sum(t_ for t_, _ in fit)
sy  = sum(y_ for _, y_ in fit)
sxx = sum(t_*t_ for t_, _ in fit)
sxy = sum(t_*y_ for t_, y_ in fit)
slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)

# K = -slope * D₀² where D₀ is the initial droplet diameter.
# Read D₀ from constant/cloudProperties (typically the uniformDistribution.maxValue
# under injectionModels.<modelN>.sizeDistribution; the value is in metres).
def read_d0_mm(case_dir):
    txt = (case_dir / "constant/cloudProperties").read_text(errors="replace")
    m = re.search(r"(?:maxValue|fixedValue|d0)\s+([0-9.eE+-]+)\s*;", txt)
    if not m:
        raise SystemExit("could not read D₀ from constant/cloudProperties — set D0_mm by hand")
    return float(m.group(1)) * 1000.0   # m → mm

D0_mm = read_d0_mm(CASE)
K_mm2_per_s = -slope * (D0_mm ** 2)

result = {
    "K_mm2_per_s": K_mm2_per_s,
    "D0_mm": D0_mm,
    "n_samples": len(rows),
    "n_fit_points": n,
    "t_end": rows[-1][0],
    "final_mass_fraction": rows[-1][1] / m0,
}
(CASE / "K_droplet_result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
```

**Compare to reference:** `ratio = K_measured / K_ref(condition)`, where the condition (typically `T_inf`) is read from the case (`0/T`, `constant/<dict>`, etc.) and `K_ref` is the topic-supplied correlation or DNS table.

**Anti-failure-mode notes:**
- If `Current mass in system` lines are absent, the solver isn't a Lagrangian solver or wasn't compiled with cloud reporting — fall back to integrated species mass: read the gas-phase fuel field across all timesteps, integrate `ρ·Y_<species>` over the domain, take the time derivative.
- If fewer than 5 samples and `t_end < expected_lifetime`, the solver crashed early — don't attempt the fit; flag `solver_crashed: true`.
- Always parameterize `D₀` from `constant/cloudProperties`. Hardcoding `D₀² = 0.25 mm²` (i.e. `D₀ = 0.5 mm`) is a common copy-paste bug across studies with different droplet sizes.

---

## Pattern 2 — Wall-shear-stress / Cf extraction

**Use when:** computing skin-friction coefficient `Cf = τ_w / (½·ρ·U_∞²)` along a no-slip wall.

**Source (good):** `<case_dir>/postProcessing/wallShearStress/<time>/wallShearStress.dat` (function-object output, table format).

**Source (bad):** Eulerian volume fields at cell centers — boundary `wallShearStress` field at the wall patch is the right read but only via PyVista's `OpenFOAMReader` with `Read All Patch Arrays` enabled.

**Snippet template (function-object):**
```python
import numpy as np, json
from pathlib import Path

case = Path("<case_dir>")
ppdir = case / "postProcessing/wallShearStress"
times = sorted([float(p.name) for p in ppdir.iterdir() if p.is_dir()])
latest = times[-1]
data = np.loadtxt(ppdir / f"{latest}" / "wallShearStress.dat", comments="#")
# Columns depend on functionObject config; common: t, x, y, z, tau_x, tau_y, tau_z
# For Cf: |tau_w| / (0.5 * rho * U_inf^2)
```

If the function-object isn't configured, switch to PyVista on the wall patch (cfd-viz pattern). Don't try to integrate from cell-center fields with finite-difference — accuracy is poor.

---

## Pattern 3 — Drag / lift via forceCoeffs

**Use when:** scoring force coefficients on a body (cylinder, airfoil, etc.).

**Source:** `<case_dir>/postProcessing/forceCoeffs/<time>/forceCoeffs.dat` — header lists `Time Cd Cs Cl CmRoll CmPitch CmYaw Cd(f) Cd(r) Cs(f) ...`.

**Snippet:**
```python
import numpy as np, json
from pathlib import Path

case = Path("<case_dir>")
fc_dir = case / "postProcessing/forceCoeffs"
times = sorted([float(p.name) for p in fc_dir.iterdir() if p.is_dir()])
data = []
for t in times:
    f = fc_dir / f"{t}" / "forceCoeffs.dat"
    if f.is_file():
        data.append(np.loadtxt(f, comments="#"))
data = np.vstack(data)
t, Cd, _, Cl = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Time-average over the last 30% (post-transient)
mask = t >= t.max() * 0.7
result = {
    "Cd_mean":  float(np.mean(Cd[mask])),
    "Cd_std":   float(np.std (Cd[mask])),
    "Cl_mean":  float(np.mean(Cl[mask])),
    "Cl_std":   float(np.std (Cl[mask])),
    "Strouhal": None,  # set if FFT of Cl is meaningful
}
(case / "drag_result.json").write_text(json.dumps(result, indent=2))
```

---

## Pattern 4 — Residual plateau check

**Use when:** judging whether the simulation reached steady-state (or quasi-steady periodicity for transient cases).

**Source:** `log.<solver>` lines like `DILUPBiCGStab:  Solving for Ux, Initial residual = 0.0123, Final residual = 1.2e-9`.

**Snippet:**
```python
import re, json
from pathlib import Path

log = Path("<case_dir>/log.<solver>").read_text(errors="replace")
fields = ["Ux", "Uy", "p", "k", "epsilon", "omega"]
hist = {f: [] for f in fields}
for line in log.splitlines():
    m = re.search(r"Solving for (\w+),.*Final residual = ([-+0-9.eE]+)", line)
    if m and m.group(1) in hist:
        hist[m.group(1)].append(float(m.group(2)))

# Plateau test: last 10% of samples have CV < 5% of max in last 10%
plateau = {}
for f, vals in hist.items():
    if len(vals) < 50:
        plateau[f] = {"converged": False, "reason": "too few samples"}
        continue
    tail = vals[-max(50, len(vals)//10):]
    plateau[f] = {
        "converged": (max(tail) - min(tail)) / (max(tail) + 1e-30) < 0.05,
        "final": vals[-1],
        "tail_max": max(tail),
        "tail_min": min(tail),
    }

(Path("<case_dir>") / "residual_plateau.json").write_text(json.dumps(plateau, indent=2))
```

---

## Pattern 5 — QoI drift check (transient cases)

**Use when:** verifying that endTime was sufficient — last 10% of QoI history shouldn't be drifting more than tolerance.

**Snippet:**
```python
import numpy as np, json

t, q = np.loadtxt("<probe_or_qoi_csv>", unpack=True)
N = len(t)
last_10 = q[int(0.9*N):]
drift = (last_10[-1] - last_10[0]) / (np.mean(np.abs(last_10)) + 1e-30)
result = {"drift_last_10pct": float(drift), "converged": abs(drift) < 0.05}
```

**Threshold guidance:** `|drift| < 5%` is the default mesh-gate convention. Stricter (1%) for steady-state convergence judgments.

---

## Pattern 6 — Mesh-independence / Richardson extrapolation (GCI)

**Use when:** triggered by `cfd-mesh-gate` STEP D when the simple 5%-threshold test fails.

**Inputs:** at least 3 mesh refinements with QoI values `phi_1` (coarsest), `phi_2`, `phi_3` (finest), and refinement ratios `r_21 = h_2/h_1`, `r_32 = h_3/h_2`. Typically uniform: `r_21 = r_32 = r`.

**Snippet:**
```python
import math, json

# Inputs
phi_1, phi_2, phi_3 = 0.42, 0.405, 0.401  # example
r_21, r_32 = 2.0, 2.0                     # refinement ratios

# Apparent order
e_21 = phi_2 - phi_1
e_32 = phi_3 - phi_2
p = math.log(abs(e_32 / e_21)) / math.log(r_32)

# Extrapolated value (Richardson)
phi_ext = phi_3 + (phi_3 - phi_2) / (r_32**p - 1)

# GCI on finest mesh (factor of safety = 1.25 for 3-mesh studies)
e_a = abs((phi_3 - phi_2) / phi_3)
GCI_fine = 1.25 * e_a / (r_32**p - 1)

result = {
    "apparent_order": p,
    "phi_extrapolated": phi_ext,
    "GCI_fine_pct": GCI_fine * 100,
    "passes": GCI_fine < 0.05,  # 5% threshold
}
print(json.dumps(result, indent=2))
```

**Reference:** ASME V&V 20-2009 / Roache 1994 / Celik et al. 2008.

---

## Pattern 7 — Two-extractor agreement check

**Use when:** verifying a comparator's output against an independent data path. **This is the verification pattern from `cfd-open-discovery`.**

**Recipe:**
1. Comparator-A reads via path A (e.g. `log.<solver>` text parsing).
2. Comparator-B reads via path B (e.g. PyVista on Eulerian fields, or function-object output).
3. If `|A - B| / max(|A|, |B|, ε) > tolerance` (default 0.5), neither passes selftest. The two extractors must agree to within 50% relative error before the comparator is bound.

**Why the high tolerance:** A and B are unlikely to agree better than ~10% in practice (different time integration, different averaging windows). 50% catches gross blind-spot failures (one returns 0, other returns 0.13) without flagging legitimate methodological differences.

**Snippet:**
```python
import json, math

A = json.load(open("<comparator_A_output>.json"))["value"]
B = json.load(open("<comparator_B_output>.json"))["value"]
denom = max(abs(A), abs(B), 1e-30)
rel = abs(A - B) / denom
result = {
    "A": A, "B": B, "relative_disagreement": rel,
    "agreement_ok": rel < 0.5,
}
```

---

## Pattern 8 — Two-sided ratio metric

**Use when:** scoring "how close is `K_pred` to `K_ref`?" — common for code-mod studies that want to minimize mean over-prediction.

**Form:**
```python
ratio = K_pred / K_ref          # > 1 means over-prediction; < 1 means under-prediction
abs_err = abs(ratio - 1)        # 0 is perfect; standard ranking loss
```

**Reporting:** publish both `mean_ratio` (so reviewers see direction of bias) and `mean_abs_ratio_error` (the loss to minimize). A sub-1.10 `mean_ratio` paired with a sub-0.20 `mean_abs_ratio_error` is typical of a well-tuned correction.

---

## Anti-patterns (do NOT use these)

These are general failure modes, not study-specific. Each came from a real silent-zero result somewhere; the fix is the same: don't read fields that are constant by design or written sparsely.

- **`cloudOutputProperties.<modelN>.massInjected`** — cumulative injection bookkeeping. Constant by design once injection completes (= total mass injected, irrespective of how much has evaporated). Ratio metrics computed from this field will be ~0 or ~1 for every case. Use the live droplet mass from `log.<solver>` instead (Pattern 1).
- **`cloud:rhoTrans_<species>`** Eulerian volume field — transient source term written at solver write times. The saved value reflects only the last sub-step before each write; for slow-evaporating cases or coarse `writeInterval`, the field is sparse or zero almost everywhere even when evaporation is occurring. Use the live droplet mass or integrated species mass instead.
- **Cell-center finite-differencing for `wallShearStress`** — accuracy is poor; use the `wallShearStress` function-object output (Pattern 2) instead.
- **Reading `0/<field>` and assuming it's the latest** — `0/` is the initial condition; read the largest non-zero time directory.
- **Reading any cumulative bookkeeping field and treating it as a per-step measurement** — the general principle. If a field's value at the final time equals the value at all preceding times, it's a cumulative count, not a state variable. Verify with a constant-field smell test (`cfd-stats` Pattern 7's two-extractor agreement check is the strongest defense).

## Anti-hallucination

- Every pattern outputs a JSON file with explicit units and `n_samples` count. If `n_samples` is below the pattern's threshold (typically 5), the result is invalid; don't downstream-rank on it.
- If the agent's snippet raises `SystemExit` or returns NaN/inf, write a `<pattern>_failed.json` with the error message, not a fake number.
- For comparator binding (Pattern 7), `agreement_ok=false` → reject the comparator regardless of either extractor's internal selftest.

---

## Optional script fast-path (LangGraph mode only)

The LangGraph workflow in `scripts/orchestrator_run.py` and `scripts/oed_extensions.py` invokes equivalent extractors automatically as part of the OED comparator-author + verifier loop. Skill-mode users do NOT call those scripts — write the pattern as a Bash-launched Python snippet and persist the JSON yourself.

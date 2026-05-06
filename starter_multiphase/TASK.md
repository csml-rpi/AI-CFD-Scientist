# Research Task: Droplet Evaporation Model Improvement

## Problem Statement

The provided OpenFOAM v10 case simulates a single n-heptane droplet evaporating in hot quiescent nitrogen. The built-in evaporation model significantly **over-predicts the evaporation rate** compared to published experimental measurements.

Your task: investigate why the built-in model fails, research improved approaches in the literature, implement a custom model as a compiled OpenFOAM library (.C/.H files), and demonstrate improved agreement with experiment.

## Base Case

Located in `base_case/`. A working `reactingFoam` case:
- **Droplet**: n-heptane (C₇H₁₆), d₀ = 500 µm, T₀ = 300 K, quiescent (no external flow)
- **Environment**: hot nitrogen at T∞ = 673 K, atmospheric pressure
- **Solver**: `reactingFoam` with Lagrangian parcels
- **Mesh**: 2000 cells (verified mesh-independent)
- **Runtime**: ~2 minutes per simulation

Run with: blockMesh && reactingFoam`

The key output is the droplet mass history. Parse `Current mass in system` from the solver log to obtain the evaporation rate constant K:
```
K [mm²/s] = |d(d²)/dt| × d₀²    where  (d/d₀)² = (m/m₀)^(2/3)
```

## Experimental Reference

**Verwey, C. & Birouk, M. (2023)**, PDF provided as `verwey_birouk_2023.pdf`.

This paper reports evaporation rate constants K for real n-heptane droplets at temperatures from 473 K to 973 K. They used an ultra-thin 14 µm cross-fiber technique that does not interfere with the evaporation process. Their fiber-free (idealized) evaporation rate in microgravity is given by:

```
K₀(T) = 3.6552 × 10⁻⁴ × T[K] − 0.1078    [mm²/s]
```

At T = 673 K: **K_experiment ≈ 0.138 mm²/s**

## Baseline Result

The built-in model gives **K ≈ 0.40 mm²/s** at 673 K — roughly **2.9× higher** than the experimental value. This over-prediction is consistent across all temperatures:

| T [K] | K_built-in | K_experiment | Over-prediction |
|-------|-----------|--------------|-----------------|
| 473   | 0.15      | 0.065        | 2.3×            |
| 573   | 0.27      | 0.102        | 2.7×            |
| 673   | 0.40      | 0.138        | 2.9×            |
| 873   | 0.40      | 0.211        | 1.9×            |
| 973   | 0.40      | 0.248        | 1.6×            |

## What to Investigate

The built-in model uses standard Ranz-Marshall correlations for both heat transfer and mass transfer. These correlations were developed for conditions where **the vapor mass flux from the droplet surface is small** relative to the diffusive transport. At elevated temperatures, the vapor outflow from the evaporating surface is NOT small — it modifies the transport processes near the droplet in ways the standard correlations do not capture.

Consider:
- How does the outward flow of vapor from the droplet surface affect the species concentration profile in the surrounding gas?
- Does the assumption of a linear concentration gradient remain valid when the evaporation rate is high?
- Are both heat and mass transfer affected, or only one? (Hint: at quasi-steady state, the evaporation rate is determined by the energy balance — think about what limits the rate.)
- Search for how spray combustion and droplet vaporization literature handles this regime.

## Implementation Guide

The custom model infrastructure is in `base_case/src/`:

- `src/reference/LiquidEvaporation.C/.H` — the built-in PhaseChangeModel (study the `calculate()` method and the `Sh()` method)
- `src/reference/RanzMarshall.C/.H` — the built-in HeatTransferModel (study the `Nu()` method)
- `src/customModels/Make/` — compilation infrastructure
- `src/customModels/makeTemplate.C.example` — template for registering custom models

**Two model types can be customized:**
1. `PhaseChangeModel` — controls mass transfer (the `calculate()` method computes mass change per parcel per timestep)
2. `HeatTransferModel` — controls heat transfer (the `Nu(Re, Pr)` method returns the Nusselt number)

Compilation: `cd src/customModels && wmake libso`
Activation: add `libs ("libCustomDropletModels.so");` to `system/controlDict`, change model types in `constant/cloudProperties`.

## Deliverable

Run the base case and your improved case at T∞ = 473, 573, 673, 873, 973 K. Produce a comparison showing K vs temperature for: built-in, your custom model, and the experimental correlation. Report the mean over-prediction ratio for each.

## Notes

- m₀ = ρ_l × π/6 × d₀³ where ρ_l = 684 kg/m³
- To change T∞: edit `0/T` (both internalField and inlet value)
- The `BirdCorrection` flag in `RanzMarshallCoeffs` is a built-in option — you may experiment with it, but understand what it does physically before relying on it

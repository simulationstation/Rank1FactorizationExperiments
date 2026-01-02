# LHCb Pentaquark Rank-1 Bottleneck Test

## Executive Summary
**Verdict: MODEL MISMATCH**

The fit-health gates failed for both channels (χ²/dof > 3). This indicates that the simplified model (3 Breit-Wigners + polynomial background) is insufficient to describe the complex m(J/ψ p) spectrum, which contains contributions from Λ* reflections and other effects not captured by our model.

**No conclusions about rank-1 factorization can be drawn until the model adequately describes the data.**

## Data Provenance

| Item | Value |
|------|-------|
| HEPData Record | [89271](https://www.hepdata.net/record/ins1728691) |
| INSPIRE ID | 1728691 |
| Publication | PRL 122, 222001 (2019) |
| DOI | [10.1103/PhysRevLett.122.222001](https://doi.org/10.1103/PhysRevLett.122.222001) |

### Channel Definitions

- **Channel A**: Table 1 - Full m(J/ψ p) invariant mass spectrum (unweighted)
- **Channel B**: Table 3 - cos θ_Pc weighted m(J/ψ p) spectrum

Both spectra from Λ_b → J/ψ p K⁻ decay at √s = 7, 8, 13 TeV.

## Pentaquark Family

Testing the 2019 LHCb narrow pentaquark triple:

| State | Mass [MeV] | Width [MeV] |
|-------|------------|-------------|
| Pc(4312)⁺ | 4311.9 ± 0.7 | 9.8 ± 2.7 |
| Pc(4440)⁺ | 4440.3 ± 1.3 | 20.6 ± 4.9 |
| Pc(4457)⁺ | 4457.3 ± 0.6 | 6.4 ± 2.0 |

## Fit Quality

### Fit Health Gates (threshold: < 3.0)

| Channel | χ²/dof | Deviance/dof | Pass |
|---------|--------|--------------|------|
| A (unweighted) | 3.647 | 3.650 | ✗ |
| B (weighted) | 6.042 | 6.072 | ✗ |

**Gates overall: FAIL**

The poor fit quality is expected because:
1. The actual LHCb analysis uses a full amplitude model with ~100 parameters
2. Our simplified model ignores Λ* → pK⁻ reflections in the m(J/ψ p) spectrum
3. Interference patterns are complex and not fully captured

## Amplitude Ratio Results

Testing ratio R = c(Pc4457)/c(Pc4440)

### Unconstrained Fits

| Channel | |R| | arg(R) [°] |
|---------|-----|-----------|
| A | 0.2353 | 95.0 |
| B | 0.2538 | 38.4 |

### Constrained Fit (Rank-1)

| Parameter | Value |
|-----------|-------|
| |R_shared| | 0.2382 |

## Likelihood Ratio Test

| Quantity | Value |
|----------|-------|
| NLL (unconstrained A+B) | -2554216.38 |
| NLL (constrained) | -2554229.70 |
| **Λ = -2ΔlnL** | **26.653** |

## Bootstrap p-value

- Parametric bootstrap from constrained fit
- 300 valid replicates out of 300
- Parallelized across 31 CPU cores

| Statistic | Value |
|-----------|-------|
| Λ_obs | 26.653 |
| Λ_boot (median) | 46.554 |
| Λ_boot (95th percentile) | 155.170 |
| **p-value** | **0.73** |

*Note: The high p-value suggests the rank-1 constraint is not rejected by the data, but this result cannot be trusted due to MODEL MISMATCH.*

## Conclusion

**MODEL MISMATCH**

The simplified coherent Breit-Wigner model fails the fit-health gates (χ²/dof > 3 for both channels). This is the correct conservative outcome - we cannot draw conclusions about the rank-1 factorization hypothesis until we have a model that adequately describes the underlying physics.

### Recommendations for Future Work

1. **Use full amplitude model**: The LHCb analysis includes ~6 K* and Λ* resonances with complex interference patterns
2. **Account for reflections**: The m(J/ψ p) spectrum is contaminated by Λ* → pK⁻ reflections
3. **Consider amplitude-level data**: If available, use the actual amplitude fits rather than projections

## Files Generated

- `out/fit_A.png` - Channel A fit plot
- `out/fit_B.png` - Channel B fit plot
- `out/results.json` - Complete numerical results
- `logs/analysis.log` - Full analysis log
- `logs/COMMANDS.txt` - Commands used

---
*Analysis performed using Poisson NLL with multi-start optimization (30 restarts)*  
*Bootstrap: 300 replicates with 31 parallel workers*  
*Data: HEPData record 89271 (LHCb PRL 122, 222001)*

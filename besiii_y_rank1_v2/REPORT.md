# BESIII Y-Sector Rank-1 Bottleneck Test v3

## Summary

**Test Status:** MODEL MISMATCH (inconclusive)

The rank-1 bottleneck test cannot be reliably evaluated because the 2-resonance model fails to adequately describe either data channel. The optimizer bug has been fixed, but the physical model needs additional resonances.

---

## Data Sources

| Channel | Process | Source | Points |
|---------|---------|--------|--------|
| A | e+e- → π+π- J/ψ | PRD 106, 072001 (2022) | 25 |
| B | e+e- → π+π- h_c | PRL 135, 071901 (2025) | 25 |

Energy range: 4.08 - 4.55 GeV (overlapping subset)

---

## Results (v3 Fixed)

### Optimizer Bug Fixed
```
v2 (buggy):   NLL_con = 114.38 < NLL_unc = 156.68  ✗ (impossible)
v3 (fixed):   NLL_unc = 189.93 < NLL_con = 233.34  ✓ (correct)
```

The bug was caused by the unconstrained optimizer getting stuck in a local minimum. Fixed by seeding unconstrained optimization from the constrained solution.

### Fit Parameters
```
Resonance 1: M = 4243.7 MeV, Γ = 150.0 MeV
Resonance 2: M = 4280.0 MeV, Γ = 300.0 MeV

R_A (unconstrained) = 1.425 exp(i 2.036)
R_B (unconstrained) = 1.645 exp(i 2.279)
R_shared            = 1.453 exp(i 2.031)
```

### Statistical Test
```
Λ = 2 × (NLL_con - NLL_unc) = 86.81
p-value = 0.000 (500 bootstrap replicates)
```

### Fit Health (FAILED)
```
Channel A: χ²/dof = 12.88  [gate: 0.5-3.0] → FAIL
Channel B: χ²/dof = 10.45  [gate: 0.5-3.0] → FAIL
```

---

## Interpretation

### Why the Model Fails

The 2-resonance Breit-Wigner model is insufficient because:

1. **ππJ/ψ channel** shows structure consistent with multiple Y states:
   - Y(4160) at ~4.19 GeV
   - Y(4230)/Y(4260) at ~4.22-4.26 GeV
   - Y(4360) at ~4.36 GeV
   - Possibly Y(4660) at higher energies

2. **ππh_c channel** also requires multiple resonances for an adequate fit

3. The high χ²/dof values indicate the model systematically deviates from the data, making any statistical inference unreliable.

### What the Results *Would* Mean (If Valid)

If the model were valid:
- Λ = 86.81 with p = 0.000 would **strongly disfavor** rank-1 factorization
- R_A ≈ 1.43e^(i2.04) and R_B ≈ 1.65e^(i2.28) differ in both magnitude (~15%) and phase (~0.24 rad)

However, this conclusion is **not reliable** due to model mismatch.

---

## Next Steps

To obtain a valid rank-1 test:

1. **Expand to 3-4 resonances** for ππJ/ψ
2. **Use 2-3 resonances** for ππh_c
3. **Match resonance parameters** to PDG values as priors
4. **Re-run with improved model** and verify χ²/dof < 3.0 for both channels

The rank-1 hypothesis can only be properly tested when the underlying amplitude model adequately describes the data.

---

## Files

- `src/y_rank1_v3_fixed.py` - Analysis code with optimizer bug fix
- `out/fits_v3.png` - Fit comparison plots
- `out/bootstrap_v3.png` - Bootstrap Λ distribution
- `logs/v3_analysis.log` - Full analysis log

---

*Generated: 2025-12-30*

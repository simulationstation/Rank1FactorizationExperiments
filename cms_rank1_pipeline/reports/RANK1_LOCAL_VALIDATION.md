# CMS Rank-1 Bottleneck Test Results

## Overview

This report documents the rank-1 bottleneck test applied to mock CMS 4-muon / di-J/ψ mass spectra.

**Test Purpose**: Determine if the complex coupling ratio R = c₂/c₁ is shared between two decay channels, which would indicate a common intermediate state or production mechanism.

**Note**: This is a **demonstration run** using mock data. For valid physics conclusions, run with actual CMSSW-produced spectra from FourMu/Di-J/ψ MC or ParkingDoubleMuonLowMass real data.

## Summary

| Metric | Value |
|--------|-------|
| **Verdict** | MODEL MISMATCH |
| Reason | chi2/dof: A=22.34, B=18.40 |
| NLL (constrained) | 919.94 |
| NLL (unconstrained) | 896.29 |
| Lambda | 47.31 |
| p-value | 0.490 |
| chi2/dof (A) | 22.34 |
| chi2/dof (B) | 18.40 |

## Coupling Ratios

```
R_shared = 0.584 exp(i 2.222)
R_A      = 0.409 exp(i 2.185)
R_B      = 0.247 exp(i 1.907)
```

## Input Files

- Channel A: `outputs/rank1_inputs/channelA.csv`
- Channel B: `outputs/rank1_inputs/channelB.csv`
- Bootstrap replicates: 200

## Interpretation

### Why "MODEL MISMATCH"?

The verdict is **MODEL MISMATCH** because the simple two-resonance Breit-Wigner model used in this demo does not adequately describe the mock Gaussian+exponential data:

- χ²/dof = 22.34 (Channel A) and 18.40 (Channel B)
- Fit health gate requires 0.5 < χ²/dof < 5.0

This is **expected behavior** for a demonstration. The mock data was generated with Gaussian peaks (not Breit-Wigner resonances), so the BW model naturally fails.

### What Would a Valid Test Show?

With properly simulated CMSSW data:

1. **COMPATIBLE**: R_A ≈ R_B → shared intermediate state (e.g., same Y resonance)
2. **DISFAVORED**: R_A ≠ R_B → different production mechanisms or multiple resonances

## Limitations

1. **Mock Data**: Generated with simple Gaussian+exponential, not realistic physics simulation
2. **Simple Model**: Two-resonance BW amplitude; real di-J/ψ spectra may require more complex models
3. **No Detector Effects**: Resolution, acceptance, efficiency not modeled

## Next Steps for Production Use

1. Generate actual FourMu and Di-J/ψ samples using CMSSW on lxplus
2. Extract mass spectra using the provided analysis configs
3. Re-run this test with real spectra:
   ```bash
   python3 cms_rank1_test.py \
     --channel-a outputs/local_mc/dijpsi_hist_dijpsi.csv \
     --channel-b outputs/local_mc/fourmu_hist_4mu.csv \
     --output reports/RANK1_PRODUCTION.md \
     --bootstrap 500
   ```

## Files Generated

| File | Description |
|------|-------------|
| `outputs/rank1_inputs/channelA.csv` | Mock Channel A spectrum |
| `outputs/rank1_inputs/channelB.csv` | Mock Channel B spectrum |
| `reports/RANK1_LOCAL_VALIDATION.md` | This report |

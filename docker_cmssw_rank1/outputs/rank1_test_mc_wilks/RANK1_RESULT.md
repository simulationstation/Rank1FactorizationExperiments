# CMS Rank-1 Bottleneck Test Results

**Version**: Publication Grade v2.0
**Date**: 2025-12-31 08:07:30

## Summary

| Metric | Value |
|--------|-------|
| **Verdict** | INCONCLUSIVE |
| Reason | Underconstrained fits: A=chi2/dof=0.12 < 0.5, B=chi2/dof=0.48 < 0.5 |
| NLL (constrained) | 27.53 |
| NLL (unconstrained) | 27.53 |
| Lambda | 0.00 |
| dof_diff | 2 |
| chi2(2) 95% threshold | 5.99 |

## P-values

| Method | p-value | Notes |
|--------|---------|-------|
| Wilks (reference) | 1.0000 | chi2(2) approximation |

## Fit Health

| Channel | chi2/dof | deviance/dof | Status |
|---------|----------|--------------|--------|
| A | 0.12 | 1.56 | UNDERCONSTRAINED |
| B | 0.48 | 4.89 | UNDERCONSTRAINED |

*Thresholds: UNDERCONSTRAINED if chi2/dof < 0.5, MODEL_MISMATCH if chi2/dof > 3.0 or dev/dof > 3.0*

## Coupling Ratios

```
R_shared = 0.9886 * exp(i * 3.0918)
R_A      = 0.0129 * exp(i * 3.1007)
R_B      = 0.9852 * exp(i * 3.0778)
```

## Input Files

- Channel A: `/home/primary/DarkBItParticleColiderPredictions/docker_cmssw_rank1/outputs/dijpsi_hist_dijpsi.csv`
- Channel B: `/home/primary/DarkBItParticleColiderPredictions/docker_cmssw_rank1/outputs/fourmu_hist_4mu.csv`
- Bootstrap replicates: 500
- Optimizer starts: 30

## Interpretation Guide

| Verdict | Meaning |
|---------|--------|
| NOT_REJECTED | Data consistent with shared R (rank-1) |
| DISFAVORED | Evidence against shared R |
| INCONCLUSIVE | Cannot draw conclusion (fit issues) |
| MODEL_MISMATCH | Model does not describe data |
| OPTIMIZER_FAILURE | Numerical issues in optimization |

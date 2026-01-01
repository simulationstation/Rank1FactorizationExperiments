# MEGA MECHANISM SWEEP v3 - Issues Found in Previous Run

## Summary of Issues

| Test | Issue Type | Details | Action Required |
|------|------------|---------|-----------------|
| LHCb Pair1 quad | **Λ < 0** | Λ=-15.66 (optimizer failure) | Mark OPTIMIZER_FAILURE |
| LHCb Pair2 quad | **Λ < 0** | Λ=-5.94 (optimizer failure) | Mark OPTIMIZER_FAILURE |
| LHCb Pair1 quad | **PROXY mislabeled** | Marked M0_SUPPORTED but is PROXY_ONLY | Correct to PROXY_ONLY |
| LHCb Pair2 quad | **PROXY mislabeled** | Marked M0_SUPPORTED but is PROXY_ONLY | Correct to PROXY_ONLY |
| ATLAS v5 | **Missing chi2 values** | chi2_A/B shown as UNKNOWN in table | Extract from source |
| Y-states Belle/BaBar | **Underconstrained** | Channel B chi2/dof=0.36 < 0.5 | Mark NO_VERDICT |
| BESIII Y-states | **Poor fit** | Channel B chi2/dof=8.82 > 3.0 | Mark NO_VERDICT |
| BaBar K*K | **Underconstrained** | Both channels chi2/dof << 0.5 | Mark NO_VERDICT |
| BaBar phi-eta | **Underconstrained** | Both channels chi2/dof << 0.5 | Mark NO_VERDICT |

---

## Detailed Issue Descriptions

### 1. LHCb Pentaquark Tests: Λ < 0 (Critical)

**Source**: `lhcb_rank1_test_v2/out/results.json`

The pair1_quad and pair2_quad variants show negative Λ values:
- pair1_quad: `lambda_obs = -15.660395232758674`
- pair2_quad: `lambda_obs = -5.936433902511453`

**Root Cause**: By definition, Λ = 2*(NLL_constrained - NLL_unconstrained). Since the constrained
model is nested within the unconstrained model, we must have NLL_con ≥ NLL_unc, hence Λ ≥ 0.

A negative Λ indicates the optimizer found a better (lower NLL) solution for the constrained
problem than for the unconstrained problem, which is mathematically impossible if both are
optimized correctly.

**Resolution**: Mark as OPTIMIZER_FAILURE. These tests cannot contribute to mechanism ranking
until properly re-optimized with more starts or better initialization.

### 2. LHCb Tests: PROXY Mislabeled as M0_SUPPORTED

The LHCb pentaquark tests are yield-ratio based (PROXY tests), not amplitude-level tests.
Per the validity rules, PROXY_ONLY tests cannot declare M0 as overall winner - they can only
show "not rejected" as weak/auxiliary evidence.

**Resolution**: Relabel as PROXY_ONLY with verdict "NOT_REJECTED" (if Λ were valid) or
OPTIMIZER_FAILURE (given the Λ < 0 issue).

### 3. ATLAS v5: Missing chi2 Values in Table

The MEGA_TABLE showed "UNKNOWN" for chi2_A and chi2_B, but the source data has valid values:
- chi2_dof_4mu = 1.39 (PASS)
- chi2_dof_4mu2pi = 9.74 (POOR FIT)

**Resolution**: Correctly extract and display these values. Final verdict remains MODEL_MISMATCH
because 4mu2pi channel exceeds health gate threshold.

### 4. Y-states Belle/BaBar: Underconstrained

**Source**: `y_rank1_test_v2/out/results.json`
- Channel A: chi2/dof = 0.70 (PASS)
- Channel B: chi2/dof = 0.36 (UNDERCONSTRAINED)

Per validity rules: chi2/dof < 0.5 → UNDERCONSTRAINED → NO_VERDICT

### 5. BESIII Y-states: Poor Fit

**Source**: `besiii_rank1_test_v2/out/results.json`
- Channel A: chi2/dof = 1.66 (PASS)
- Channel B: chi2/dof = 8.82 (POOR FIT >> 3.0)

Per validity rules: chi2/dof > 3.0 → MODEL_MISMATCH → NO_VERDICT

### 6. BaBar K*K and phi-eta: Severely Underconstrained

**Source**: `model_sweep/runs/*/results.json`
- BaBar K*K: chi2/dof_A = 0.01, chi2/dof_B = 0.01
- BaBar phi-eta: chi2/dof_A = 0.02, chi2/dof_B = 0.02

These chi2/dof values are far below 0.5, indicating the 2-BW coherent model has too many
degrees of freedom relative to the constraining power of the data. The data cannot distinguish
between different mechanism hypotheses.

---

## Validity Rules Applied

1. **Λ ≥ 0**: All constrained vs unconstrained tests must have Λ ≥ 0 (tolerance -1e-6)
2. **Fit Health Gates**:
   - chi2/dof > 3.0 OR deviance/dof > 3.0 → MODEL_MISMATCH
   - chi2/dof < 0.5 → UNDERCONSTRAINED
3. **PROXY handling**: Yield-ratio tests labeled PROXY_ONLY, cannot declare M0 winner
4. **Winner Definition**:
   - Requires VALID test (passes all gates)
   - If p < 0.05: M0 rejected → M1 wins
   - If p ≥ 0.05: Compare AIC/BIC for final winner

---

*Generated for mega_mechanism_sweep_v3*

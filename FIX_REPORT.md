# FIX REPORT: Rank-1 Discovery Mine Nested Model Invariant

**Date**: 2026-01-02
**Candidate**: `lhcb_pc_4312_extensions` (LHCb Pc(4440)/Pc(4457))

---

## Executive Summary

Critical bugs were identified and fixed in the rank-1 test harness that were silently producing false "NOT_REJECTED" verdicts by masking optimizer failures.

---

## What Was Wrong

### 1. Nested Model Invariant Violation (CRITICAL)

**Bug**: The unconstrained fit was returning `nll_unc > nll_con`, which is mathematically impossible for nested models.

For a likelihood ratio test between nested models:
- Constrained model: shared coupling ratio `R` across channels
- Unconstrained model: independent `R_A`, `R_B` per channel

The unconstrained model has **more** free parameters, so by definition:
```
nll_unconstrained <= nll_constrained
```

Any violation indicates **optimizer failure**, not a physics result.

**Impact**: When this invariant was violated, the code was silently clamping `Lambda = max(0, 2*(nll_con - nll_unc))`, converting optimizer failures into `Lambda = 0` and false "NOT_REJECTED" verdicts.

### 2. Lambda Definition/Sign Error

**Bug**: The sign convention and clamping logic made it impossible to detect when the optimizer failed.

**Correct definition**:
```python
Lambda_raw = 2 * (nll_con - nll_unc)  # Raw value, can be negative if invariant violated
Lambda = max(0, Lambda_raw)            # Only clamp AFTER verifying invariant holds
```

### 3. Bootstrap P-Value Correctness

**Bug**: Bootstrap samples with invariant violations were being silently discarded without tracking, potentially biasing p-values.

**Correct approach**: Track both valid samples and failures separately, use conservative estimator:
```python
p_boot = (1 + n_exceedances) / (1 + n_valid)  # Conservative Bayesian estimator
```

### 4. Report Generation Placeholders

**Bug**: Reports had placeholder text like "[TBD]" for coupling ratios and missing sanity check sections.

---

## What Was Changed

### File: `rank1_discovery_mine/harness/rank1_core.py`

**Complete rewrite** with the following changes:

1. **Nested Model Invariant Checking**
   ```python
   NESTED_MODEL_TOL = 1e-4  # Absolute tolerance
   NESTED_MODEL_REL_TOL = 1e-6  # Relative tolerance

   def check_nested_invariant(nll_con: float, nll_unc: float) -> Tuple[bool, float]:
       """Check if nested model invariant holds: nll_unc <= nll_con (within tolerance)."""
       tol = max(NESTED_MODEL_TOL, NESTED_MODEL_REL_TOL * abs(nll_con))
       violation = nll_unc - nll_con
       if violation > tol:
           return False, violation
       return True, violation
   ```

2. **Lambda Diagnostics with Lambda_raw**
   ```python
   def compute_lambda_with_diagnostics(nll_con: float, nll_unc: float) -> Dict[str, Any]:
       """Compute Lambda with full diagnostics including Lambda_raw."""
       invariant_holds, violation = check_nested_invariant(nll_con, nll_unc)
       Lambda_raw = 2 * (nll_con - nll_unc)
       if not invariant_holds:
           return {'Lambda_raw': float(Lambda_raw), 'Lambda': None,
                   'invariant_holds': False, 'violation': float(violation)}
       Lambda = max(0.0, Lambda_raw)
       return {'Lambda_raw': float(Lambda_raw), 'Lambda': float(Lambda),
               'invariant_holds': True, 'violation': float(violation)}
   ```

3. **Retry Logic for Unconstrained Optimizer**
   - If initial unconstrained fit violates invariant, retry with initialization from constrained solution
   - Added `init_from_constrained` parameter to `fit_joint_unconstrained()`

4. **Bootstrap with Invariant Enforcement**
   - Track violations separately from valid samples
   - Return dict with `n_valid`, `n_failed`, and violation counts
   - Use conservative p-value estimator

5. **New `quick_check()` Diagnostic Function**
   - Runs minimal fit without bootstrap to verify invariant holds
   - Returns detailed diagnostics for debugging

6. **New Verdict: OPTIMIZER_FAILURE**
   - When invariant cannot be satisfied even after retry
   - Distinct from INCONCLUSIVE (fit health issues)

### File: `rank1_discovery_mine/harness/run_test.py`

1. **Report Generation Updated**
   - Added Sanity Checks section showing invariant status
   - Added complete Coupling Ratios table (Individual, Unconstrained, Constrained)
   - Added Bootstrap Distribution statistics (mean, std, median)
   - Removed all placeholder text
   - Added Lambda_raw to Test Statistics table

---

## Verification Results

Ran `verify_fix.py` with `n_boot=50` on `lhcb_pc_4312_extensions`:

### Pair 1: Table 1 vs Table 2

| Metric | Value |
|--------|-------|
| NLL constrained | 67.5731 |
| NLL unconstrained | 64.7329 |
| Lambda_raw | 5.6805 |
| Lambda (clamped) | 5.6805 |
| Invariant holds | **YES** |
| p_boot | 0.0784 |
| Verdict | NOT_REJECTED |

### Pair 2: Table 2 vs Table 3

| Metric | Value |
|--------|-------|
| NLL constrained | 75.5071 |
| NLL unconstrained | 74.5412 |
| Lambda_raw | 1.9318 |
| Lambda (clamped) | 1.9318 |
| Invariant holds | **YES** |
| p_boot | 0.5294 |
| Verdict | NOT_REJECTED |

### Checks Passed

- [x] Nested invariant enforced (nll_unc <= nll_con)
- [x] Lambda_raw = 2*(nll_con - nll_unc) computed correctly
- [x] Bootstrap: 50 valid, 0 failed for both pairs
- [x] Report contains Sanity Checks section
- [x] Report contains Coupling Ratios table
- [x] Report contains Lambda_raw values
- [x] No placeholder text in report

---

## Physics Interpretation

The LHCb Pc(4440)/Pc(4457) pentaquark doublet passes the rank-1 test with both data pairs showing p > 0.05. This is **consistent with** the two states sharing a common production mechanism (rank-1 factorization), as expected for a genuine doublet originating from heavy quark spin symmetry.

**Note**: This does NOT prove the states are a genuine doublet - it merely confirms the data is consistent with rank-1 behavior and there is no evidence against it.

---

## Files Modified

```
rank1_discovery_mine/harness/rank1_core.py  # Complete rewrite
rank1_discovery_mine/harness/run_test.py    # Report generation fixes
verify_fix.py                                # New verification script
```

---

*Generated by Claude Code during rank-1 harness fix session*

# Di-charmonium Calibration Failure - Detailed Analysis

## Executive Summary
The rank-1 bottleneck hypothesis test has inflated Type I error (~16% vs 5% target). The parametric bootstrap p-values are systematically too small because the bootstrap Lambda distribution doesn't match the true null distribution.

---

## Statistical Context

### What We're Testing
- **Null hypothesis (H0)**: Two channels share a common complex amplitude ratio R (rank-1 factorization)
- **Test statistic**: Lambda = 2 * (NLL_constrained - NLL_unconstrained)
- **Under H0**: Lambda should follow approximately chi2(df=2) distribution
- **P-value**: Computed via parametric bootstrap - P(Lambda_boot >= Lambda_obs)

### What Should Happen Under Correct Calibration
- If null is true, p_boot should be Uniform(0,1)
- Type I error at alpha=0.05 should be ~5%
- KS test of p_boot against Uniform should pass

### What Actually Happens
- Type I error is ~16% (3x too high)
- Many trials have p_boot=0 (Lambda_obs exceeds ALL bootstrap samples)
- KS test fails badly

---

## Detailed Observations

### Trial-by-Trial Results (15 trials, seed=i*100)
```
Trial  0: Lambda=8.2455, p_boot=0.000  <- OUTLIER
Trial  1: Lambda=1.6089, p_boot=0.520
Trial  2: Lambda=0.5456, p_boot=0.740
Trial  3: Lambda=0.0403, p_boot=1.000
(diagnostic stopped early)
```

### Optimization Check (Trial 0)
Lambda does NOT decrease with more optimizer starts:
```
Starts= 80: NLL_unc=-13039.4192, NLL_con=-13035.2964, Lambda=8.2455
Starts=150: NLL_unc=-13039.4192, NLL_con=-13035.2956, Lambda=8.2472
Starts=300: NLL_unc=-13039.4192, NLL_con=-13035.2964, Lambda=8.2455
Starts=500: NLL_unc=-13039.4192, NLL_con=-13035.2964, Lambda=8.2455
```
**Conclusion**: This is NOT an optimization failure. Both fits converged to global optima.

---

## Root Cause Analysis

### The Bootstrap Procedure (sim_fit_v3.py lines 536-613)

```python
def bootstrap_pvalue_calibrated(...):
    # Get predicted values from CONSTRAINED fit
    y_pred_a = fit_con.get('y_pred_a')  # <- smooth predictions
    y_pred_b = fit_con.get('y_pred_b')

    for b in range(n_bootstrap):
        # Generate bootstrap data from constrained predictions
        boot_a = generate_bootstrap_data_poisson(ch_a, y_pred_a, rng)
        boot_b = generate_bootstrap_data_poisson(ch_b, y_pred_b, rng)

        # Fit both models to bootstrap data
        fit_unc_b = fit_joint_unconstrained(boot_dataset, ...)
        fit_con_b = fit_joint_constrained(boot_dataset, ...)

        lambda_b = 2 * (fit_con_b['nll'] - fit_unc_b['nll'])
```

### Poisson Bootstrap Generation (lines 512-524)
```python
def generate_bootstrap_data_poisson(ch_data, y_pred, rng):
    y_pred_safe = np.maximum(y_pred, 0.1)
    y_boot = rng.poisson(y_pred_safe)  # Sample from smooth rate
    sigma_boot = np.sqrt(np.maximum(y_boot, 1))
    return {'y': y_boot, 'sigma': sigma_boot, 'type': 'poisson', ...}
```

### The Problem

**Hypothesis**: The bootstrap samples are generated from the constrained fit's smooth predictions, which "bakes in" the rank-1 structure. When we add Poisson noise to these smooth predictions, the resulting datasets are "easy" for the constrained model to fit, producing small Lambda values.

But the ORIGINAL data was generated with noise that may create patterns that look like rank-1 violations, even though the true model is rank-1. These noise patterns cause high Lambda_obs values that the bootstrap doesn't replicate.

**Expected behavior**:
- Lambda_obs ~ chi2(2) with mean=2, 95th percentile ~6.0
- Lambda_boot should have same distribution

**Actual behavior**:
- Lambda_obs sometimes reaches 8+ (reasonable under chi2(2), happens ~2% of time)
- Lambda_boot distribution is too narrow, max rarely exceeds 4-5
- This causes p_boot=0 for high Lambda_obs values

---

## Potential Fixes to Consider

### Option 1: Use Asymptotic Distribution
Instead of bootstrap, use Wilks' theorem: p_wilks = 1 - chi2.cdf(Lambda_obs, df=2)
- Pro: No bootstrap variance issues
- Con: Asymptotics may not hold for this sample size

### Option 2: Increase Bootstrap Samples
Use n_bootstrap=500-1000 to better capture tail
- Pro: Simple
- Con: Computationally expensive, may not solve root cause

### Option 3: Double Bootstrap or Bootstrap Calibration
Adjust bootstrap p-values to correct for bias
- Pro: Principled correction
- Con: Complex, computationally expensive

### Option 4: Generate Bootstrap from Unconstrained Fit
Use y_pred from unconstrained fit instead of constrained
- Pro: May better capture true noise structure
- Con: Violates standard parametric bootstrap theory

### Option 5: Residual Resampling Bootstrap
Resample residuals from observed data instead of parametric generation
- Pro: Preserves noise structure
- Con: May not work well for Poisson data

---

## Data Characteristics (Di-charmonium)

```
Type: Poisson (both channels)
Channel A (J/psi J/psi): 90 bins, mean counts ~34
Channel B (J/psi psi(2S)): 90 bins, mean counts ~17
Total DOF: 180 bins
Constraint DOF: 2 (complex R = magnitude + phase)
```

---

## Files in This Folder

| File | Description |
|------|-------------|
| sim_fit_v3.py | Core fitting code - bootstrap at lines 536-613 |
| sim_generate.py | Data generation under null (M0) and alternatives |
| tests_top3.json | Test configuration - Di-charmonium is tests[2] |
| check_outlier.py | Script to verify optimization sufficiency |
| check_outlier.log | Output showing Lambda stable with more starts |
| check_bootstrap.py | Script to analyze bootstrap distribution (incomplete) |
| diag_dicharmonium.log | Partial diagnostic output |

---

## Reproduction

```python
import sys
sys.path.insert(0, '.')
from sim_generate import generate_dataset
from sim_fit_v3 import run_calibration_trial
import json

with open('tests_top3.json') as f:
    config = json.load(f)
test = config['tests'][2]  # Di-charmonium

# Run 10 trials to see the problem
for i in range(10):
    dataset = generate_dataset(test, 'M0', scale_factor=1.0, seed=i*100)
    result = run_calibration_trial(dataset, n_bootstrap=50, n_starts=80)
    print(f"Trial {i}: Lambda={result['lambda_obs']:.4f}, p_boot={result['p_boot']:.3f}")

# Expected output: Some trials will have p_boot=0 with Lambda>5
```

---

## Success Criteria

After fix:
1. Type I error should be 5% ± 3% (Wilson CI)
2. KS test of p_boot against Uniform(0,1) should have p > 0.05
3. No trials should have p_boot=0 unless Lambda_obs is truly extreme (>15)

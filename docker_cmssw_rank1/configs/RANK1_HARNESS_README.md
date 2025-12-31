# Rank-1 Bottleneck Test Harness

## Overview

The rank-1 bottleneck test determines whether a complex coupling ratio R = c2/c1 is shared
between two measurement channels. This is publication-grade statistical inference tool with:

- **Correct degrees of freedom**: dof=2 for complex R constraint (magnitude + phase)
- **Bootstrap p-values**: Primary inference method (Wilks approximation as reference)
- **Fit-health gates**: Detect both underconstrained and model-mismatch conditions
- **Multi-start optimization**: Robust fitting with L-BFGS-B + Powell/Nelder-Mead fallbacks
- **Injection/recovery mode**: End-to-end validation of the statistical procedure

## Files

```
configs/
├── cms_rank1_test.py       # Main inference harness
├── rank1_injection.py      # Spectrum simulator for injection tests
└── RANK1_HARNESS_README.md # This file
```

## Quick Start

### Basic Analysis

```bash
# Run on two channel CSV files
python3 cms_rank1_test.py \
  --channel-a dijpsi_hist_dijpsi.csv \
  --channel-b fourmu_hist_4mu.csv \
  --bootstrap 500 \
  --starts 50 \
  --outdir outputs/rank1_analysis
```

### Injection/Recovery Test

```bash
# Run 50 trials each for rank-1 true and false scenarios
python3 cms_rank1_test.py \
  --inject \
  --inject-trials 50 \
  --bootstrap 300 \
  --starts 50 \
  --outdir outputs/injection_test
```

### Quick Wilks-Only Mode

```bash
# Skip bootstrap for quick exploration
python3 cms_rank1_test.py \
  --channel-a A.csv \
  --channel-b B.csv \
  --use-wilks-only
```

## CSV Input Format

```csv
mass_GeV,counts,stat_err
6.0500,0,1.0000
6.1500,1,1.0000
6.2500,2,1.4142
...
```

- `mass_GeV`: Bin center in GeV
- `counts`: Event counts in bin
- `stat_err`: Statistical uncertainty (typically sqrt(counts) or 1 for empty bins)

## Command Line Options

### Input/Output
- `--channel-a FILE`: Channel A CSV file
- `--channel-b FILE`: Channel B CSV file
- `--output FILE`: Output report path (default: RANK1_RESULT.md in outdir)
- `--outdir DIR`: Output directory (default: current)

### Bootstrap Options
- `--bootstrap N`: Number of bootstrap replicates (default: 500)
- `--bootstrap-seed N`: Random seed (default: 42)
- `--bootstrap-workers N`: Parallel workers (default: auto)
- `--use-wilks-only`: Skip bootstrap, use Wilks approximation only

### Optimizer Options
- `--starts N`: Number of multi-start initializations (default: 50)
- `--use-poisson`: Use Poisson likelihood instead of Gaussian

### Statistical Options
- `--dof N`: Degrees of freedom for constraint (default: 2 for complex R)

### Injection Mode
- `--inject`: Run injection/recovery test
- `--inject-trials N`: Trials per scenario (default: 50)
- `--inject-config FILE`: Custom injection configuration JSON

## Verdicts

| Verdict | Meaning |
|---------|---------|
| `NOT_REJECTED` | Data consistent with shared R (rank-1 hypothesis not rejected) |
| `DISFAVORED` | Evidence against shared R (p < 0.05) |
| `INCONCLUSIVE` | Cannot draw conclusion due to fit issues or optimizer instability |
| `MODEL_MISMATCH` | The two-resonance model does not describe the data |
| `OPTIMIZER_FAILURE` | Numerical optimization failed |

## Fit Health Gates

The test includes automatic quality checks:

### Chi-squared per DOF
- chi2/dof < 0.5: **UNDERCONSTRAINED** (data errors too large or model too flexible)
- chi2/dof > 3.0: **MODEL_MISMATCH** (model doesn't fit data)
- 0.5 <= chi2/dof <= 3.0: **HEALTHY**

### Poisson Deviance
- deviance/dof > 3.0: **MODEL_MISMATCH**

## Statistical Details

### Test Statistic

Lambda = 2 * (NLL_constrained - NLL_unconstrained)

where:
- **Constrained model**: R_A = R_B (shared complex coupling)
- **Unconstrained model**: R_A and R_B are independent

### Degrees of Freedom

The complex coupling ratio R = r * exp(i*phi) has two real parameters.
When comparing constrained vs unconstrained models:
- **dof_diff = 2**: Testing that both |R| and arg(R) are shared

The chi2(2) 95% threshold is 5.99 (vs 3.84 for chi2(1)).

### P-value Computation

1. **Bootstrap (primary)**: Generate pseudo-data from constrained best-fit,
   refit both models, count exceedances of observed Lambda
2. **Wilks (reference)**: Assume Lambda ~ chi2(dof_diff) under null hypothesis

## Injection/Recovery Test

The injection test validates the statistical procedure:

### Rank-1 TRUE Scenario
- Generate data where R_A = R_B (true shared coupling)
- Expected result: Mostly NOT_REJECTED
- Type-I error rate should be ~0.05

### Rank-1 FALSE Scenario
- Generate data where R_A != R_B (different couplings)
- Expected result: Mostly DISFAVORED
- Power should be high (>0.80) for useful test

### Interpretation
```
Type-I Error Rate: Fraction of false rejections when null is true
                   Should be ~0.05 for well-calibrated test

Power:            Fraction of correct rejections when alternative is true
                   Higher is better (ideally >0.80)
```

## Output Files

### RANK1_RESULT.md
Main results report including:
- Verdict and reason
- Lambda statistic and threshold
- Bootstrap and Wilks p-values
- Fit health assessment
- Recovered coupling ratios

### RANK1_OPTIMIZER_AUDIT.md
Optimizer diagnostics:
- Convergence statistics
- Methods used
- Bootstrap Lambda distribution

### INJECTION_REPORT.md (injection mode)
Injection/recovery summary:
- Type-I error rate
- Power
- Lambda distributions by scenario

## Physical Model

The test uses a two-resonance Breit-Wigner interference model:

```
I(m) = |c1 * (BW1(m) + R * BW2(m))|^2 * scale

where:
  BW(m; M, Gamma) = M * Gamma / (M^2 - m^2 + i*M*Gamma)
  R = r * exp(i * phi)   # Complex coupling ratio
```

Parameters per channel:
- M1, Gamma1: First resonance mass and width (shared)
- M2, Gamma2: Second resonance mass and width (shared)
- c1: Overall coupling strength (per channel)
- R = (r, phi): Complex coupling ratio (shared in constrained, independent in unconstrained)
- scale: Normalization (per channel)

## Version History

- **v2.0.0** (Publication Grade): Correct dof, bootstrap default, fit-health gates,
  multi-start optimizer, injection/recovery mode
- **v1.0.0**: Initial implementation with single-start optimizer

## References

- Wilks' theorem: S. S. Wilks (1938), Ann. Math. Statist. 9(1): 60-62
- Bootstrap methods: Efron & Tibshirani (1993), An Introduction to the Bootstrap

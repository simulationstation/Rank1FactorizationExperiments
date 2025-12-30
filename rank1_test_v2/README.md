# CMS Rank-1 Bottleneck Test v2/v3

## Overview

Publication-quality implementation of the rank-1 factorization constraint test using CMS data. This version includes proper statistical treatment with Poisson likelihood, vector PDF extraction, and bootstrap p-value computation.

## Physics Motivation

The rank-1 factorization constraint tests whether tetraquark states X(6900) and X(7100) are produced via a factorizable mechanism. If true, the complex coupling ratio:

```
R = g_{X7100} / g_{X6900} = r × exp(iφ)
```

must be identical in both the di-J/ψ and J/ψ+ψ(2S) decay channels.

## Data Sources

- **Channel A**: CMS-BPH-21-003 via HEPData (di-J/ψ, official data points)
- **Channel B**: CMS-PAS-BPH-22-004 via vector PDF extraction (J/ψ+ψ(2S))

## Method

### Signal Model
```
I(m) = |c₁·BW(m₆₉₀₀) + c₂·BW(m₇₁₀₀)|² × phase_space + background
```

### Statistical Framework
- **Likelihood**: Poisson negative log-likelihood
- **Optimization**: Differential evolution + local refinement
- **Uncertainty**: Bootstrap resampling (300 replicates)
- **Test statistic**: Likelihood ratio Λ = 2×ΔNLL

## Results (v3)

| Channel | r | φ (deg) | χ²/dof |
|---------|---|---------|--------|
| A (di-J/ψ) | 0.633 | -55.8 | 1.02 |
| B (J/ψ+ψ(2S)) | 0.923 | -98.4 | 2.49 |
| **Shared** | **0.596** | **-49.9** | - |

### Likelihood Ratio Test
| Quantity | Value |
|----------|-------|
| Λ | 1.82 |
| Bootstrap p-value | **0.403** |
| Verdict | **RANK-1 CONSTRAINT SUPPORTED** |

## Interpretation

The bootstrap p-value of 0.40 indicates that the observed difference between channels is consistent with statistical fluctuations. The rank-1 factorization constraint is **not rejected** at the 5% significance level.

## Files

```
rank1_test_v2/
├── src/
│   ├── rank1_bottleneck_v2.py      # v2 implementation
│   └── rank1_bottleneck_v3.py      # v3 with vector extraction
├── data/
│   ├── hepdata/                    # Official HEPData for Channel A
│   └── derived/                    # Extracted spectra
└── out/
    ├── REPORT.md                   # v2 report
    ├── REPORT_v3.md                # v3 report
    ├── rank1_test_v3_summary.json  # Numerical results
    └── *.png                       # Fit and diagnostic plots
```

## Version History

- **v1**: Basic digitization (see `rank1_test/`)
- **v2**: Added HEPData for Channel A, improved fitting
- **v3**: Vector PDF extraction for Channel B, Poisson likelihood, bootstrap p-value

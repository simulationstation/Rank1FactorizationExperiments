# ATLAS Rank-1 Bottleneck Test v1

## Overview

Replication of the rank-1 factorization constraint test using ATLAS public data for J/ψ+ψ(2S) in the 4μ and 4μ+2π decay channels.

**Status:** Superseded by v4 (see `atlas_rank1_test_v4/`)

## Physics Motivation

Tests whether the complex coupling ratio R = g_{X7200}/g_{X6900} is identical between the 4μ and 4μ+2π final states, as predicted by rank-1 factorization.

## Data Sources

- **arXiv:2509.13101**: ATLAS J/ψ+ψ(2S) analysis
- **arXiv:2304.08962**: ATLAS di-J/ψ spectrum (background reference)
- Vector extraction from ATLAS public PDFs (BPHY-2023-01)

## Results

| Channel | r | φ (deg) | χ²/dof |
|---------|---|---------|--------|
| 4μ | 0.749 | -101.2 | 0.03 |
| 4μ+2π | 0.064 | +75.9 | 0.01 |
| **Shared** | **0.674** | **-107.0** | - |

### Likelihood Ratio Test
| Quantity | Value |
|----------|-------|
| Λ | 0.12 |
| Bootstrap p-value | **0.913** |
| Verdict | **RANK-1 CONSTRAINT SUPPORTED** |

## Known Issues

The extremely low χ²/dof values (0.03 and 0.01) indicate that uncertainties were **massively overestimated** due to per-bin digitization sigma. This means the test had essentially no discriminating power.

**This analysis is superseded by v4**, which uses:
- Pure Poisson likelihood (no per-bin sigma)
- Correlated nuisance parameters
- Profile likelihood contours

## Files

```
atlas_rank1_test/
├── src/
│   ├── extract_atlas_figures.py
│   └── fit_atlas_spectra.py
├── data/
│   ├── raw/                        # Downloaded PDFs
│   └── derived/                    # Extracted spectra
└── out/
    ├── REPORT_ATLAS.md
    ├── ATLAS_summary.json
    └── *.png
```

## See Also

- `atlas_rank1_test_v4/` - Publication-grade analysis with proper statistics

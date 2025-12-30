# Tetraquark Rank-1 Factorization Tests

Testing the rank-1 factorization constraint for exotic tetraquark states X(6900) and X(7100/7200) using LHC data from CMS and ATLAS.

## Physics Background

Tetraquark states (cccc) have been observed at the LHC in di-charmonium final states. The **rank-1 factorization constraint** tests whether these states are produced via a factorizable mechanism (e.g., gluon-gluon fusion through a single intermediate state).

If factorization holds, the complex coupling ratio:

```
R = g_{X7100} / g_{X6900} = r × exp(iφ)
```

must be **identical** across all decay channels. This provides a powerful test of the production mechanism.

## Summary of Results

| Analysis | Experiment | Channels | r_shared | φ_shared | p-value | Verdict |
|----------|------------|----------|----------|----------|---------|---------|
| CMS v3 | CMS | di-J/ψ vs J/ψ+ψ(2S) | 0.596 | -49.9° | 0.403 | SUPPORTED |
| ATLAS v4 | ATLAS | 4μ vs 4μ+2π | 0.214 | -14.6° | 0.893 | SUPPORTED |

Both experiments support the rank-1 factorization hypothesis at the 5% significance level.

## Analyses

### CMS Tests

| Directory | Description | Status |
|-----------|-------------|--------|
| `rank1_test/` | Initial implementation (v1) | Superseded |
| `rank1_test_v2/` | Publication-quality with HEPData + bootstrap (v2/v3) | **Current** |

### ATLAS Tests

| Directory | Description | Status |
|-----------|-------------|--------|
| `atlas_rank1_test/` | Initial implementation (v1) - had statistical issues | Superseded |
| `atlas_rank1_test_v4/` | Publication-grade with profile likelihood contours | **Current** |

## Methods

### Data Sources
- **CMS**: HEPData (CMS-BPH-21-003) + PDF vector extraction (CMS-PAS-BPH-22-004)
- **ATLAS**: PDF vector extraction from public figures (BPHY-2023-01)

### Statistical Framework
- Poisson negative log-likelihood
- Correlated nuisance parameters for systematic uncertainties
- Profile likelihood confidence contours
- Bootstrap p-value computation (300+ replicates)

### Signal Model
Coherent sum of two interfering Breit-Wigner amplitudes:
```
I(m) = |c₁·BW(m₆₉₀₀) + c₂·BW(m₇₁₀₀)|² × phase_space + background
```

## Requirements

- Python 3.8+
- numpy, scipy, pandas, matplotlib
- PyMuPDF (for PDF extraction)

## References

- CMS Collaboration, "Observation of new structure in the J/ψJ/ψ mass spectrum" (CMS-BPH-21-003)
- CMS Collaboration, "Study of J/ψψ(2S) production" (CMS-PAS-BPH-22-004)
- ATLAS Collaboration, "Study of J/ψ+ψ(2S) production" (arXiv:2509.13101)

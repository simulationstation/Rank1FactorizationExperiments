# Exotic Hadron Rank-1 Factorization Tests

**Testing whether exotic tetraquark states share a common production mechanism across decay channels**

---

## Latest Result: CMS X(6900)/X(7100)

### Verdict: **NOT_REJECTED** (p = 0.40)

The complex coupling ratio R = g(X7100)/g(X6900) is **consistent with being shared** between the J/ψJ/ψ and J/ψψ(2S) channels.

| Metric | Value |
|--------|-------|
| Lambda (test statistic) | 0.50 |
| Bootstrap p-value | 0.3987 (319/800 exceedances) |
| chi2(2) 95% threshold | 5.99 |
| dof | 2 (complex R constraint) |

### Coupling Ratios

| Channel | \|R\| | arg(R) |
|---------|-------|--------|
| **Shared** | **7.51** | **1.51 rad (87°)** |
| J/ψJ/ψ (A) | 8.01 | 1.51 rad |
| J/ψψ(2S) (B) | 7.59 | 1.55 rad |

### Fit Health

| Channel | chi2/dof | Status |
|---------|----------|--------|
| A (J/ψJ/ψ) | 1.21 | HEALTHY |
| B (J/ψψ(2S)) | 1.91 | HEALTHY |

---

## Visualizations

### Bootstrap Lambda Distribution

![Bootstrap Lambda Distribution](cms_x6900_rank1_v4/out/bootstrap_hist.png)

*The observed Lambda (0.50, red dashed) is well below the chi2(2) rejection threshold (5.99, orange dotted). 40% of bootstrap replicates exceeded the observed value.*

### Channel B Extraction Verification

![Channel B Extraction](cms_x6900_rank1_v4/out/debug_channelB_overlay.png)

*Left: Original CMS-PAS-BPH-22-004 Figure 2. Right: Extracted J/ψψ(2S) spectrum with X(6900) and X(7100) positions marked.*

---

## Validation Status: Almost Complete

**Di-charmonium (X(6900)/X(7100)) Statistical Validation**

The rank-1 test harness is undergoing rigorous Monte Carlo validation to ensure correct Type-I error rates and statistical power.

### Power Analysis Results (Complete)

| Statistics Level | Type-I Error (M0) | Power (M1) |
|-----------------|-------------------|------------|
| 0.5x | 2.7% | 71.3% |
| 1.0x | 7.3% | 98.7% |
| 2.0x | 7.3% | 100% |

- **M0**: Rank-1 true (shared R) - measures false positive rate
- **M1**: Rank-1 false (different R per channel) - measures detection power
- Target Type-I: 5% (within tolerance)

### M4 Grid Analysis (In Progress)

Testing detectability across parameter space (dr, dphi deviations from rank-1).

### Single Trial Verification

```
Test: Di-charmonium (J/ψJ/ψ vs J/ψψ(2S))
Lambda_obs: 0.3334
p_boot: 0.89
Gates: PASS
Verdict: Rank-1 NOT rejected (correct under M0)
```

---

## Physics Background

### The Rank-1 Factorization Test

Tetraquark states like X(6900) and X(7100) have been observed at the LHC in multiple decay channels. The **rank-1 factorization hypothesis** tests whether these states are produced via a common mechanism.

If the production factorizes, then the complex amplitude ratio:

```
R = g(X7100) / g(X6900) = r × exp(iφ)
```

must be **identical** across all decay channels. This is a powerful probe of the underlying QCD dynamics.

### Test Statistic

We use a likelihood ratio test:

```
Lambda = 2 × (NLL_constrained - NLL_unconstrained)
```

Where:
- **Constrained**: R_A = R_B (shared complex coupling)
- **Unconstrained**: R_A and R_B independent

Under the null hypothesis (rank-1 holds), Lambda ~ chi2(2) since R has 2 real parameters (magnitude and phase).

---

## How to Reproduce

### Prerequisites

```bash
pip install numpy scipy matplotlib
pip install --user pymupdf  # For PDF extraction
```

### Step 1: Download Data

**Channel A (J/ψJ/ψ)** - from HEPData:
```bash
curl -L "https://www.hepdata.net/download/table/ins2668013/Figure%201/1/csv" \
  -o cms_x6900_rank1_v4/data/hepdata/figure1_spectrum.csv
```

**Channel B (J/ψψ(2S))** - from CDS:
```bash
curl -L "https://cds.cern.ch/record/2929529/files/Figure_002.pdf" \
  -o cms_x6900_rank1_v4/data/cds/Figure_002.pdf
```

### Step 2: Prepare Channel CSVs

Convert HEPData format and extract Channel B:
```bash
cd cms_x6900_rank1_v4/src
python3 convert_hepdata.py
python3 extract_channelB_v2.py
```

### Step 3: Run the Rank-1 Test

```bash
python3 docker_cmssw_rank1/configs/cms_rank1_test.py \
  --channel-a cms_x6900_rank1_v4/extracted/channelA_trimmed.csv \
  --channel-b cms_x6900_rank1_v4/extracted/channelB_jpsi_psi2S_bins.csv \
  --bootstrap 800 \
  --starts 300 \
  --outdir cms_x6900_rank1_v4/out/run
```

### Step 4: View Results

```bash
cat cms_x6900_rank1_v4/out/run/RANK1_RESULT.md
```

---

## Harness Features (v2.0)

The rank-1 test harness includes publication-grade statistical machinery:

| Feature | Description |
|---------|-------------|
| **dof_diff = 2** | Correct degrees of freedom for complex R |
| **Bootstrap p-values** | Primary inference method (800 replicates default) |
| **Fit-health gates** | 0.5 < chi2/dof < 3.0 prevents false conclusions |
| **Multi-start optimizer** | 300 starts with L-BFGS-B + Powell fallback |
| **Verdict system** | NOT_REJECTED / DISFAVORED / INCONCLUSIVE / MODEL_MISMATCH |

### Harness Location

```
docker_cmssw_rank1/configs/cms_rank1_test.py
docker_cmssw_rank1/configs/rank1_injection.py  # For validation
docker_cmssw_rank1/configs/RANK1_HARNESS_README.md
```

---

## Data Sources

### CMS X(6900)/X(7100) Analysis

| Channel | Source | DOI/Record |
|---------|--------|------------|
| J/ψJ/ψ | HEPData | [10.17182/hepdata.141028](https://doi.org/10.17182/hepdata.141028) |
| J/ψψ(2S) | CDS | [2929529](https://cds.cern.ch/record/2929529) |

**Publications**:
- CMS Collaboration, "Observation of new structure in the J/ψJ/ψ mass spectrum", PRL 132 (2024) 111901
- CMS-PAS-BPH-22-004, "Study of J/ψψ(2S) and J/ψJ/ψ production"

---

## Directory Structure

```
DarkBItParticleColiderPredictions/
├── cms_x6900_rank1_v4/          # Current CMS analysis
│   ├── data/
│   │   ├── hepdata/             # Official HEPData downloads
│   │   └── cds/                 # CDS figure files
│   ├── extracted/               # Processed CSV files
│   ├── out/                     # Results and plots
│   │   ├── REPORT.md            # Full analysis report
│   │   ├── RANK1_RESULT.md      # Harness output
│   │   ├── bootstrap_hist.png   # Lambda distribution
│   │   └── debug_channelB_overlay.png
│   ├── src/                     # Extraction scripts
│   └── logs/                    # Command history
│
├── docker_cmssw_rank1/configs/  # Rank-1 test harness (v2.0)
│   ├── cms_rank1_test.py        # Main harness
│   ├── rank1_injection.py       # Validation framework
│   └── RANK1_HARNESS_README.md  # Documentation
│
└── [other analyses...]          # Previous/parallel tests
```

---

## Coming Soon: Other Exotic Families

The rank-1 factorization test will be extended to other exotic hadron families:

| Family | States | Channels | Status |
|--------|--------|----------|--------|
| **X(6900)/X(7100)** | cccc tetraquark | J/ψJ/ψ, J/ψψ(2S) | **Complete** |
| Pc pentaquarks | Pc(4312), Pc(4440), Pc(4457) | J/ψp | Planned |
| Zc states | Zc(3900), Zc(4020) | J/ψπ, hcπ | Planned |
| Y states | Y(4260), Y(4360) | J/ψππ, ψ(2S)ππ | Planned |
| X(3872) | ccqq | J/ψππ, D*D | Planned |

Each analysis will use the same publication-grade harness with:
- Official HEPData where available
- Vector extraction from figures as fallback
- Bootstrap p-values with fit-health validation

---

## Interpretation Guide

| Verdict | Meaning |
|---------|---------|
| **NOT_REJECTED** | Data consistent with shared R (supports factorization) |
| **DISFAVORED** | Evidence against shared R (p < 0.05) |
| **INCONCLUSIVE** | Cannot draw conclusion (fit issues) |
| **MODEL_MISMATCH** | Two-resonance model doesn't fit data |

**Important**: NOT_REJECTED does not prove identical couplings. It means we cannot reject equality at the 5% level given the available statistics.

---

## References

1. CMS Collaboration, "Observation of new structure in the J/ψJ/ψ mass spectrum in proton-proton collisions at √s = 13 TeV", Phys. Rev. Lett. 132 (2024) 111901
2. CMS-PAS-BPH-22-004, "Observation of new structures in the J/ψψ(2S) mass spectrum in proton-proton collisions at √s = 13 TeV"
3. LHCb Collaboration, "Observation of J/ψp resonances...", Phys. Rev. Lett. 115 (2015) 072001
4. BESIII Collaboration, "Observation of Zc(3900)", Phys. Rev. Lett. 110 (2013) 252001

---

## License

This analysis code is provided for scientific research purposes.

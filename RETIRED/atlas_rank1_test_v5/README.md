# ATLAS Rank-1 Bottleneck Test v5

Publication-grade rank-1 factorization test for the ATLAS di-charmonium (X(6900)/X(7200)) resonance family.

## Verdict: MODEL MISMATCH

The 4μ+2π channel fails fit-health gates (χ²/dof = 9.74 > 3), preventing a valid rank-1 constraint verdict.

## Physics Goal

Test whether the complex amplitude ratio R = c₂/c₁ (coupling of X(7200) relative to X(6900)) is channel-invariant across:
- **Channel A**: J/ψ J/ψ → 4μ
- **Channel B**: J/ψ J/ψ → 4μ + 2π

## Data Sources

| Channel | Source | Method |
|---------|--------|--------|
| 4μ | ATLAS fig_01b.pdf | Vector extraction |
| 4μ+2π | ATLAS fig_01c.pdf | Vector extraction |

**ATLAS Papers:**
- arXiv:2509.13101 - "Observation of structures in J/ψ+ψ(2S) mass spectrum"
- arXiv:2304.08962 - "Observation of an excess of di-charmonium events" (PRL 131, 151902)

## Model

3-resonance coherent Breit-Wigner interference:
```
I(m) = |A_thresh·BW_thresh + A_6900·BW_6900 + A_7200·BW_7200|² × PS(m) + BG(m)
```

| Resonance | Mass (GeV) | Width (GeV) |
|-----------|------------|-------------|
| Threshold | 6.40 | 0.40 |
| X(6900) | 6.905 | 0.150 |
| X(7200) | 7.22 | 0.100 |

## Key Results

| Channel | r = |c₂/c₁| | φ (deg) | χ²/dof | Health |
|---------|---------|--------|---------|--------|
| 4μ | 0.402 | -82.6 | 1.39 | PASS |
| 4μ+2π | 0.303 | -96.8 | 9.74 | FAIL |

| Metric | Value |
|--------|-------|
| Λ (likelihood ratio) | 1.54 |
| Shared R in both 95% CL | Yes |
| Bootstrap p-value | Not computed (gates failed) |

## Directory Structure

```
atlas_rank1_test_v5/
├── README.md
├── src/
│   └── fit_atlas_v5.py          # Main analysis script
├── data/
│   ├── arxiv/
│   │   ├── atlas_2304.08962.pdf
│   │   └── atlas_2509.13101.pdf
│   └── derived/
│       ├── 4mu_bins.csv         # 31 bins, 703 counts
│       └── 4mu+2pi_bins.csv     # 33 bins, 1058 counts
├── out/
│   ├── REPORT_v5.md             # Full analysis report
│   ├── ATLAS_v5_summary.json    # Machine-readable results
│   ├── ATLAS_model_notes.md     # Model documentation
│   ├── fit_plots_v5.png         # Fit overlays
│   ├── contours_v5.png          # Profile likelihood contours
│   ├── optimizer_stability.json
│   ├── optimizer_stability.md
│   └── profile_tables/          # NLL grid data
│       ├── r_grid.csv
│       ├── phi_grid.csv
│       ├── nll_surface_4mu.csv
│       └── nll_surface_4mu2pi.csv
└── logs/
    ├── COMMANDS.txt
    └── fit_v5.log
```

## Methodology

1. **Multi-start optimization**: 30 random starts × 3 optimizers (L-BFGS-B, Powell, DE)
2. **Fit-health gates**: χ²/dof < 3 AND D/dof < 3 required for both channels
3. **Profile likelihood contours**: 75×73 grid over (r, φ)
4. **Confidence levels**: 68% (Δχ² < 2.30), 95% (Δχ² < 5.99)

## Comparison with v4

| Metric | v4 | v5 |
|--------|----|----|
| Model | 2-BW + polynomial | 3-BW (ATLAS-like) |
| Fit gate | None | χ²/dof < 3 |
| Multi-start | 1 DE | 30 × 3 optimizers |
| 4μ χ²/dof | 2.41 | 1.39 |
| 4μ+2π χ²/dof | 7.63 | 9.74 |
| Verdict | SUPPORTED | MODEL MISMATCH |

The v5 analysis correctly identifies that v4's "SUPPORTED" verdict was unreliable due to poor 4μ+2π fit quality.

## Running

```bash
cd atlas_rank1_test_v5
nohup python3 -u src/fit_atlas_v5.py > logs/fit_v5.log 2>&1 &
```

## Interpretation

The 4μ+2π channel shows χ²/dof = 9.74, indicating severe model-data disagreement. Possible causes:
- Featureless spectrum (nearly uniform 30-37 counts/bin)
- Higher combinatorial background overwhelming signal
- Figure extraction limitations
- Model inadequacy for this channel

The 4μ channel achieves good fit quality (χ²/dof = 1.39), suggesting either different physics or different data quality between channels.

---
*Analysis date: 2024-12-29*

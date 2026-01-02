# Zc Rank-1 Bottleneck Test v3: PROVENANCE & SANITY

**Generated**: 2025-12-30
**Status**: PROVENANCE & SANITY Run
**Purpose**: Verify data provenance and test sensitivity to analysis choices

---

## 1. Executive Summary

| Pair | States | Verdict | Reason |
|------|--------|---------|--------|
| **A** | Zc(3900)/Zc(3885) | **DISFAVORED** | p=0.000 < 0.05, tension with shared resonance |
| **B** | Zc(4020)/Zc(4025) | **DISFAVORED** | p=0.000 < 0.05, tension with shared resonance |

---

## 2. Data Provenance

| Channel | Source Type | arXiv | Method | File |
|---------|-------------|-------|--------|------|
| Zc(3900) π J/ψ | RECONSTRUCTED | 1303.5949 | Publication parameter reconstruction with Poisson ... | /home/primary/DarkBItParticleColiderPredictions/zc_rank1_test_v3_provenance/data/reconstructed/zc3900_piJpsi_bins.csv |
| Zc(3885) D D* | RECONSTRUCTED | 1310.1163 | Publication parameter reconstruction with Poisson ... | /home/primary/DarkBItParticleColiderPredictions/zc_rank1_test_v3_provenance/data/reconstructed/zc3885_ddstar_bins.csv |
| Zc(4020) π h_c | RECONSTRUCTED | 1309.1896 | Publication parameter reconstruction with Poisson ... | /home/primary/DarkBItParticleColiderPredictions/zc_rank1_test_v3_provenance/data/reconstructed/zc4020_pihc_bins.csv |
| Zc(4025) D* D* | RECONSTRUCTED | 1308.2760 | Publication parameter reconstruction with Poisson ... | /home/primary/DarkBItParticleColiderPredictions/zc_rank1_test_v3_provenance/data/reconstructed/zc4025_dstardstar_bins.csv |

---

## 3. Anti-Clone Metrics

These metrics verify that paired spectra are not trivially scaled copies of each other.

### Pair A: Zc(3900) vs Zc(3885)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² (linear fit) | 0.0394 | >0.995 would indicate clone |
| Residual Autocorr | 0.6246 | Structured residuals if |r| > 0.3 |
| Jensen-Shannon Div | 0.5988 | Higher = more different shapes |
| Peak Mass Δ | 2.6 MeV | Published: 15 MeV difference |
| Peak Width Δ | 10.0 MeV | Published: 21 MeV difference |
| **Clone-like?** | **False** | Spectra have distinct shapes |

### Pair B: Zc(4020) vs Zc(4025)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² (linear fit) | 0.4369 | >0.995 would indicate clone |
| Residual Autocorr | 0.3036 | Structured residuals if |r| > 0.3 |
| Jensen-Shannon Div | 0.3306 | Higher = more different shapes |
| Peak Mass Δ | 2.8 MeV | Published: 3 MeV difference |
| Peak Width Δ | 1.2 MeV | Published: 17 MeV difference |
| **Clone-like?** | **False** | Spectra have distinct shapes |

---

## 4. Individual Channel Fits

| Channel | M (GeV) | Γ (MeV) | χ²/dof | Dev/dof | Health |
|---------|---------|---------|--------|---------|--------|
| Zc(3900) π J/ψ | 3.8993 | 46.2 | 0.67 | 0.69 | PASS |
| Zc(3885) D D* | 3.8839 | 24.8 | 1.45 | 1.39 | PASS |
| Zc(4020) π h_c | 4.0228 | 7.5 | 0.63 | 0.61 | PASS |
| Zc(4025) D* D* | 4.0262 | 25.2 | 1.14 | 1.12 | PASS |

Health gates: 0.5 < χ²/dof < 3.0 AND deviance/dof < 3.0

---

## 5. Joint Fit Results (Mode 1: Shared M, Γ)

### Pair A: Zc(3900)/Zc(3885)

| Parameter | Value |
|-----------|-------|
| Shared M | 3.8918 GeV |
| Shared Γ | 43.5 MeV |
| Λ = 2ΔlnL | 4700.71 |
| Bootstrap p | 0.000 |
| Replicates | 300 |

### Pair B: Zc(4020)/Zc(4025)

| Parameter | Value |
|-----------|-------|
| Shared M | 4.0259 GeV |
| Shared Γ | 19.1 MeV |
| Λ = 2ΔlnL | 1294.37 |
| Bootstrap p | 0.000 |
| Replicates | 300 |

---

## 6. Sensitivity Analysis

### Pair A

| Variant | Λ |
|---------|---|
| Baseline (linear bg) | 4700.71 |
| Quadratic background | 4694.01 |
| Window 90% | 4704.50 |
| Window 110% | 4695.44 |

### Pair B

| Variant | Λ |
|---------|---|
| Baseline (linear bg) | 1294.37 |
| Quadratic background | 1285.63 |
| Window 90% | 1295.98 |
| Window 110% | 1292.76 |

---

## 7. Verdict Criteria

| Criterion | Description |
|-----------|-------------|
| SUPPORTED | p > 0.05, data consistent with shared resonance |
| DISFAVORED | p < 0.05, tension with shared resonance hypothesis |
| INCONCLUSIVE | Anti-clone check failed (R² > 0.995) |
| MODEL MISMATCH | Fit health check failed (χ²/dof or deviance/dof outside bounds) |
| OPTIMIZER FAILURE | Λ < 0 after multiple restarts |
| UNSTABLE | Verdict changes significantly across sensitivity variants |

---

## 8. Final Verdicts

### Pair A: Zc(3900)/Zc(3885)
**DISFAVORED**

p=0.000 < 0.05, tension with shared resonance

### Pair B: Zc(4020)/Zc(4025)
**DISFAVORED**

p=0.000 < 0.05, tension with shared resonance

---

## 9. Output Files

| File | Description |
|------|-------------|
| `data/figures/*_page.png` | High-res (600 DPI) figure extractions |
| `data/figures/*_page.pdf` | Page-only PDFs |
| `data/reconstructed/*_bins.csv` | Reconstructed bin data |
| `out/debug_*_overlay.png` | Extraction overlay visualizations |
| `out/bootstrap_distributions.png` | Bootstrap Λ distributions |
| `out/reconstruction_method_*.md` | Reconstruction documentation |

---

## 10. Interpretation Notes

1. **RECONSTRUCTED spectra**: These are generated from published resonance parameters
   with Poisson sampling. They demonstrate the methodology but are not true extractions.

2. **Anti-clone verification**: The low R² values and non-zero JS divergence confirm
   the spectra are not trivially scaled copies of each other.

3. **For publication-grade results**: True extraction from PDFs or HEPData would be
   required. This analysis demonstrates the statistical framework.

---

*Report generated by Zc rank-1 bottleneck test pipeline v3 (PROVENANCE & SANITY)*

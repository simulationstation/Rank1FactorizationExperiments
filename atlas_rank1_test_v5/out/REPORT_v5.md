# ATLAS Rank-1 Bottleneck Test v5: Publication-Grade Analysis

## Executive Summary

This analysis tests the **rank-1 factorization constraint** using ATLAS public data with a publication-grade statistical framework including hard fit-health gates, multi-start optimization, and proper model validation.

### Final Verdict: **MODEL MISMATCH**

The analysis cannot provide a valid rank-1 constraint verdict because the 4μ+2π channel fails the fit-health gate (χ²/dof = 9.74 > 3).

---

## 1. Key Results

| Quantity | 4μ Channel | 4μ+2π Channel |
|----------|------------|---------------|
| r | 0.402 | 0.303 |
| φ (deg) | -82.6 | -96.8 |
| χ²/dof | **1.39** (PASS) | **9.74** (FAIL) |
| D/dof | **1.34** (PASS) | **8.96** (FAIL) |
| Gate | PASS | FAIL |

| Joint Fit | Value |
|-----------|-------|
| r_shared | 0.326 |
| φ_shared | -95.1° |
| Λ | 1.54 |
| Optimizer stable | Yes (Λ ≥ 0) |

### Contour Check
| Check | Result |
|-------|--------|
| Shared in 4μ 68% | ✓ |
| Shared in 4μ 95% | ✓ |
| Shared in 4μ+2π 68% | ✓ |
| Shared in 4μ+2π 95% | ✓ |
| In BOTH 95% | ✓ |

**Bootstrap p-value:** NOT COMPUTED (gates failed)

---

## 2. Data Provenance

### Source Files
| File | Description | Extraction Method |
|------|-------------|-------------------|
| `4mu_bins.csv` | 31 bins, 703 counts | Vector extraction from ATLAS fig_01b.pdf |
| `4mu+2pi_bins.csv` | 33 bins, 1058 counts | Vector extraction from ATLAS fig_01c.pdf |

### ATLAS Papers
- **arXiv:2509.13101**: "Observation of structures in the J/ψ+ψ(2S) mass spectrum"
- **arXiv:2304.08962**: "Observation of an excess of di-charmonium events" (Phys. Rev. Lett. 131, 151902)

---

## 3. ATLAS Model Implementation

### Signal Model (3-resonance interference)
```
I(m) = |A_thresh·BW_thresh + A_6900·BW_6900 + A_7200·BW_7200|² × PS(m) + BG(m)
```

### Resonance Parameters (ATLAS-inspired)
| Resonance | Mass (GeV) | Width (GeV) |
|-----------|------------|-------------|
| Threshold | 6.40 | 0.40 |
| X(6900) | 6.905 | 0.150 |
| X(7200) | 7.22 | 0.100 |

### Background Model
- Threshold turn-on: √(m - m_thresh)
- Chebyshev polynomial modulation: c₀T₀ + c₁T₁ + c₂T₂

### Nuisance Parameters
| Parameter | Description | Prior σ |
|-----------|-------------|---------|
| s_x | Mass scale | 1% |
| b_x | Mass shift | 20 MeV |
| s_y | Intensity scale | 2% |

---

## 4. Fit Health Metrics

### Definition
- **Pearson χ²**: Σᵢ (yᵢ - μᵢ)² / max(μᵢ, 10⁻⁹)
- **Poisson Deviance D**: 2 Σᵢ [yᵢ ln(yᵢ/μᵢ) - (yᵢ - μᵢ)]
- **dof**: N_bins - N_params = N_bins - 11

### Gate Rule
Both channels must satisfy: χ²/dof < 3 AND D/dof < 3

### Results
| Channel | N_bins | dof | χ² | χ²/dof | D | D/dof | Gate |
|---------|--------|-----|-------|--------|-------|-------|------|
| 4μ | 31 | 20 | 27.9 | 1.39 | 26.9 | 1.34 | PASS |
| 4μ+2π | 33 | 22 | 214.3 | **9.74** | 197.1 | **8.96** | FAIL |

**Overall: FAIL** (4μ+2π channel exceeds threshold)

---

## 5. Multi-Start Optimizer Stability

### Settings
- Random initializations: 30 per fit
- Optimizers: L-BFGS-B, Powell, Differential Evolution
- Best-of selection across all starts and optimizers

### Results
| Fit | Successful Starts | Best NLL | Worst NLL | Stable |
|-----|-------------------|----------|-----------|--------|
| 4μ | 31 | -1484.7 | 1704.4 | Yes |
| 4μ+2π | 31 | -2501.6 | -1476.0 | Yes |
| Joint | 31 | -3985.5 | -2112.1 | Yes |

### Λ Check
- NLL unconstrained: -3986.30
- NLL constrained: -3985.53
- **Λ = 1.54 ≥ 0** → Optimizer stable

---

## 6. Profile Likelihood Contours

### Grid
- r ∈ [0.02, 1.50], step 0.02 (75 points)
- φ ∈ [-180°, 180°], step 5° (73 points)

### Confidence Levels (2D, 2 dof)
- 68% CL: Δ(-2 ln L) < 2.30
- 95% CL: Δ(-2 ln L) < 5.99

### Contour Check
The shared point (r=0.326, φ=-95.1°) lies within:
- 4μ 68% contour: ✓
- 4μ 95% contour: ✓
- 4μ+2π 68% contour: ✓
- 4μ+2π 95% contour: ✓

Despite the model mismatch in 4μ+2π, the contours are broad enough to contain the shared point.

---

## 7. Bootstrap Analysis

**NOT PERFORMED**

Bootstrap p-value was not computed because the fit-health gate failed for the 4μ+2π channel. Computing bootstrap statistics with a misspecified model would produce meaningless results.

---

## 8. Interpretation of Model Mismatch

### Why Does 4μ+2π Fail?

The 4μ+2π channel has χ²/dof = 9.74, indicating severe model-data disagreement. Possible causes:

1. **Featureless Spectrum**: The 4μ+2π data shows nearly uniform counts (30-37 per bin) with no pronounced resonance peaks visible in the figure-derived bins.

2. **Higher Background**: The 4μ+2π channel may have substantially higher combinatorial background, overwhelming the signal.

3. **Different Efficiency**: Channel-dependent efficiency curves could distort the apparent mass spectrum.

4. **Figure Extraction Limitations**: Vector extraction from ATLAS PDFs may not capture fine structure needed to fit resonance models.

5. **Model Inadequacy**: The 3-BW + polynomial background model may be oversimplified for this channel.

### Comparison with 4μ Channel

The 4μ channel achieves χ²/dof = 1.39, indicating acceptable model fit. The difference between channels suggests:
- Either the physics differs substantially between channels
- Or the data quality/extraction differs

---

## 9. Comparison with Previous Versions

| Metric | v4 | v5 |
|--------|----|----|
| Model | 2-BW + polynomial | 3-BW (ATLAS-like) |
| Fit gate | None | χ²/dof < 3, D/dof < 3 |
| Multi-start | 1 DE + polish | 30 starts × 3 optimizers |
| 4μ χ²/dof | 2.41 | 1.39 |
| 4μ+2π χ²/dof | 7.63 | 9.74 |
| Λ artifact | -0.13 (unstable) | 1.54 (stable) |
| Verdict | SUPPORTED | MODEL MISMATCH |

The v5 analysis correctly identifies that the previous "SUPPORTED" verdict was unreliable due to the poor 4μ+2π fit.

---

## 10. Conclusions

### Summary Table
| Criterion | 4μ | 4μ+2π | Assessment |
|-----------|-------|-------|------------|
| χ²/dof < 3 | 1.39 ✓ | 9.74 ✗ | 4μ+2π FAILS |
| D/dof < 3 | 1.34 ✓ | 8.96 ✗ | 4μ+2π FAILS |
| Λ ≥ 0 | - | - | 1.54 ✓ |
| Shared in 95% | ✓ | ✓ | PASS |

### Final Verdict: **MODEL MISMATCH**

The rank-1 factorization test cannot produce a valid verdict because the 3-resonance signal model does not adequately describe the 4μ+2π channel data. The fit-health gate (χ²/dof < 3) is not satisfied.

### Recommendations for Future Work

1. **Alternative Background Models**: Test different background parameterizations (exponential, physics-motivated shapes)

2. **Higher-Statistics Data**: The figure-derived bins may have insufficient resolution; official ATLAS data would be preferable

3. **Channel-Specific Models**: Allow different signal models per channel if physics permits

4. **HEPData Access**: Check if ATLAS has released official bin-by-bin data through HEPData

---

## 11. Output Files

| File | Description |
|------|-------------|
| `ATLAS_v5_summary.json` | Complete numerical results |
| `ATLAS_model_notes.md` | Signal model documentation |
| `fit_plots_v5.png` | Fit overlays for both channels |
| `contours_v5.png` | Profile likelihood contours |
| `optimizer_stability.json` | Multi-start optimization details |
| `optimizer_stability.md` | Stability summary |
| `profile_tables/` | Grid data for contours |

---

## 12. Summary

```
============================================================
ATLAS RANK-1 TEST v5 RESULTS
============================================================

Fit Health Gates:
  4μ:      χ²/dof = 1.39 (PASS), D/dof = 1.34 (PASS)
  4μ+2π:   χ²/dof = 9.74 (FAIL), D/dof = 8.96 (FAIL)

Unconstrained MLEs:
  4μ:      r = 0.402, φ = -82.6°
  4μ+2π:   r = 0.303, φ = -96.8°

Constrained (shared):
  r_shared = 0.326, φ_shared = -95.1°

Likelihood Ratio:
  Λ = 1.54 (≥ 0, optimizer stable)

Contour Check:
  Shared in BOTH 95% contours: YES

Bootstrap p-value:
  NOT COMPUTED (gates failed)

VERDICT: MODEL MISMATCH
============================================================
```

---

*Generated: 2024-12-29*
*Analysis: atlas_rank1_test_v5/src/fit_atlas_v5.py*
*Model: 3-resonance Breit-Wigner with coherent interference*

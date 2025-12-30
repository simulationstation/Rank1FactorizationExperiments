# ATLAS Signal Model Documentation

## Source Papers

1. **arXiv:2509.13101** - "Observation of structures in the J/ψ+ψ(2S) mass spectrum with the ATLAS detector"
2. **arXiv:2304.08962** - "Observation of an excess of di-charmonium events in the four-muon final state" (Phys. Rev. Lett. 131, 151902 (2023))

## ATLAS Model Structure

### Signal Model

ATLAS uses **three interfering Breit-Wigner resonances**:
- A threshold structure (~6.2-6.6 GeV)
- X(6900) - the main resonance
- X(7200) - upper structure (observed as upper limit in J/ψ+ψ(2S))

The model includes coherent interference between all resonances.

### Functional Form (as implemented in v5)

We implement a relativistic Breit-Wigner amplitude:

```
BW(m; M₀, Γ₀) = √(M₀Γ₀) / (M₀² - m² - iM₀Γ(m))
```

With mass-dependent width:
```
Γ(m) = Γ₀ × (q/q₀)^(2L+1) × (M₀/m) × B_L(q,q₀)
```

Where:
- q = breakup momentum at mass m
- q₀ = breakup momentum at M₀
- L = orbital angular momentum (assume L=0 for S-wave)
- B_L = Blatt-Weisskopf barrier factor

### Phase Space Factor

```
PS(m) = √((m² - (M₁+M₂)²)(m² - (M₁-M₂)²)) / (2m)
```

Where M₁ = M(J/ψ) = 3.097 GeV, M₂ = M(ψ(2S)) = 3.686 GeV.

Threshold: m_thresh = 6.783 GeV

### Background Model

ATLAS uses a "hybrid approach involving Monte Carlo simulations and data-driven methods."

For our figure-derived analysis, we implement:
- Smooth threshold turn-on: (m - m_thresh)^α where α ~ 0.5
- Chebyshev polynomial modulation: T₀ + c₁T₁(x) + c₂T₂(x)
- where x = 2(m - m_min)/(m_max - m_min) - 1

### Interference Structure

The intensity is given by:
```
I(m) = |A_thresh·BW_thresh + A_6900·BW_6900 + A_7200·BW_7200|² × PS(m) + BG(m)
```

Where each amplitude A_i = |A_i|·exp(iφ_i) is complex.

For the rank-1 test, we parameterize:
- A_6900 = c_norm (real, positive reference)
- A_7200 = c_norm × r × exp(iφ)
- A_thresh = c_thresh × exp(iφ_thresh) (independent complex coefficient)

The rank-1 constraint requires R = A_7200/A_6900 = r·exp(iφ) to be identical across channels.

## ATLAS Fit Models (from paper)

ATLAS reports multiple models (A, B, α, β) for different assumptions:
- Model A/B: Different interference assumptions in di-J/ψ
- Model α/β: Different assumptions for J/ψ+ψ(2S) channel

The masses and widths from ATLAS fits:
- X(6900): M ≈ 6.90 GeV, Γ ≈ 0.15-0.20 GeV
- X(7200): M ≈ 7.2 GeV, Γ ≈ 0.10 GeV (upper limit only)
- Threshold: M ≈ 6.2-6.6 GeV, Γ ≈ 0.3-0.5 GeV (broad)

## Implementation Notes for v5

1. **Three-resonance model**: Include threshold, X(6900), and X(7200)
2. **Fixed masses/widths**: Use ATLAS central values with tight priors
3. **Coherent interference**: Full complex amplitude sum
4. **Phase space**: Proper two-body threshold factor
5. **Background**: Threshold × Chebyshev polynomial
6. **Nuisance parameters**: Mass scale (1%), y-scale (2%), shift (20 MeV)

## References

- ATLAS Collaboration, Phys. Rev. Lett. 131, 151902 (2023)
- ATLAS public results: https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/BPHY-2022-01/

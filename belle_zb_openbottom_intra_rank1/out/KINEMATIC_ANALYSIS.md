# Kinematic Constraints for Intra-Open-Bottom Rank-1 Test

## Threshold Masses

| Threshold | Mass (MeV) | Formula |
|-----------|------------|---------|
| BB* | ~10604 | M_B + M_B* ≈ 5279 + 5325 |
| B*B* | ~10650 | 2 × M_B* ≈ 2 × 5325 |

## Zb Masses

| State | Mass (MeV) | Width (MeV) |
|-------|------------|-------------|
| Zb(10610) | 10607.2 ± 2.0 | 18.4 ± 2.4 |
| Zb(10650) | 10652.2 ± 1.5 | 11.5 ± 2.2 |

## Kinematic Accessibility

| Decay | Zb(10610) | Zb(10650) |
|-------|-----------|-----------|
| → BB*π | ✓ Allowed (above threshold by ~3 MeV) | ✓ Allowed (above threshold by ~48 MeV) |
| → B*B*π | ✗ Forbidden (below threshold by ~43 MeV) | ✓ Allowed (above threshold by ~2 MeV) |
| → BBπ | ✗ Forbidden (both below BB threshold ~10558 MeV) | ✗ Forbidden |

## Implications for Rank-1 Test

### The Problem

A proper rank-1 test requires extracting the complex ratio:
```
R = g(Zb10650) / g(Zb10610)
```
from multiple channels and comparing them.

**In BB*π**: Both Zb states are visible → R can be extracted ✓
**In B*B*π**: Only Zb(10650) is visible → R cannot be extracted ✗

### What We CAN Test

1. **Verify Zb(10610) absence in B*B*π**: Fit B*B*π with two-BW model and confirm Zb(10610) amplitude ≈ 0

2. **Compare Zb(10650) yields (PROXY_ONLY)**: Extract Zb(10650) signal from both channels and compare relative strengths

3. **Phase consistency check**: If we can extract Zb(10650) phase from both channels, check if they're consistent

### Conclusion

**A true intra-open-bottom rank-1 test is NOT POSSIBLE** due to kinematic constraints.

The B*B*π channel cannot probe Zb(10610), so we cannot compare the coupling ratio R between open-bottom channels.

We will instead perform:
- A kinematic validation (confirming Zb(10610) absence in B*B*π)
- A PROXY yield comparison for Zb(10650) between channels

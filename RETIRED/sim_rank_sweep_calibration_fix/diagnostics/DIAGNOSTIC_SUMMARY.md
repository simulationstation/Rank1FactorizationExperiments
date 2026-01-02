# Calibration Diagnostic Summary

Generated: 2025-12-31 13:53:34

Configuration:
- Trials per test: 50
- Stats level: 1.0x
- Bootstrap: 200
- Optimizer restarts: 80
- dof_diff: 2

## Results

| Test | Valid/Total | Type I | Target | Lambda Mean | Lambda Std |
|------|-------------|--------|--------|-------------|------------|
| Y-states | 50/50 | 0.000 | 0.05 | 0.159 | 0.140 |
| Zc-like | 50/50 | 0.160 | 0.05 | 2.329 | 2.611 |
| Di-charmonium | 50/50 | 0.220 | 0.05 | 3.862 | 3.382 |

## Interpretation

- Type I error should be ~0.05 (within 2-8% for 50 trials)
- Lambda distribution under H0 should follow chi2(2)
- Expected chi2(2) mean = 2.0, median = 1.386

## Calibration Status

- **Y-states**: TOO CONSERVATIVE (Type I = 0.000 < 0.02)
- **Zc-like**: INFLATED (Type I = 0.160 > 0.10)
- **Di-charmonium**: INFLATED (Type I = 0.220 > 0.10)

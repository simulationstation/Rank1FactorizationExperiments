# Calibration Diagnostic Results

Generated: 2025-12-31 16:43:35

## Summary Table

| Test | Pass Rate | Type I | Type I SE | Lambda Mean | Lambda Median | Lambda Boot Mean | KS Stat | KS p-val | Status |
|------|-----------|--------|-----------|-------------|---------------|------------------|---------|----------|--------|
| Y-states | 0.995 | 0.171 | 0.027 | 0.190 | 0.117 | 0.101 | 0.232 | 0.0000 | FAIL |
| Zc-like | 0.995 | 0.060 | 0.017 | 1.962 | 1.441 | 1.837 | 0.077 | 0.1750 | PASS |
| Di-charmonium | 1.000 | 0.160 | 0.026 | 3.588 | 1.897 | 2.719 | 0.145 | 0.0004 | FAIL |

## Criteria

- Type I error: [0.02, 0.08]
- Pass rate: >= 0.8
- KS test: Approximate check against Uniform(0,1)

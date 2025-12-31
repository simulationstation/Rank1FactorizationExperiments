# Rank-1 Injection/Recovery Report

**Date**: 2025-12-31 09:31:53

## Configuration

- Trials per scenario: 30
- Bootstrap replicates: 200
- Optimizer starts: 30

## Summary Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| **Type-I Error Rate** | 0.000 | False rejections when rank-1 TRUE (0/30) |
| **Power** | 0.000 | Correct rejections when rank-1 FALSE (0/30) |
| Inconclusive (TRUE) | 30/30 | Fit/optimizer issues |
| Inconclusive (FALSE) | 30/30 | Fit/optimizer issues |

## Interpretation

- **Type-I error** should be ~0.05 for a well-calibrated test
- **Power** should be high (>0.80) for the test to be useful
- High inconclusive rates indicate fit stability issues

## Rank-1 TRUE Scenario Details

True parameters: R_A = R_B (shared complex coupling)

| Verdict | Count | Fraction |
|---------|-------|----------|
| MODEL_MISMATCH | 30 | 1.000 |

## Rank-1 FALSE Scenario Details

True parameters: R_A != R_B (different complex couplings)

| Verdict | Count | Fraction |
|---------|-------|----------|
| MODEL_MISMATCH | 30 | 1.000 |

## Lambda Distributions

| Scenario | Mean | Std | Median | 95th pct |
|----------|------|-----|--------|----------|
| Rank-1 TRUE | 60.87 | 83.03 | 6.97 | 222.87 |
| Rank-1 FALSE | 33.39 | 60.48 | 7.52 | 156.52 |

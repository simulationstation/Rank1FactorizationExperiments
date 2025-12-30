# CMS Rank-1 Bottleneck Test v1

## Overview

Initial implementation of the rank-1 factorization constraint test using CMS data for tetraquark states X(6900) and X(7100).

**Status:** Superseded by v2/v3 (see `rank1_test_v2/`)

## Physics Motivation

The rank-1 factorization constraint predicts that for a tetraquark produced via gluon-gluon fusion and decaying to different final states, the complex coupling ratio:

```
R = g_{X7100} / g_{X6900} = r × exp(iφ)
```

should be **identical** across all decay channels if the production mechanism factorizes.

## Data Sources

- **Channel A**: CMS-BPH-21-003 (di-J/ψ spectrum)
- **Channel B**: CMS-PAS-BPH-22-004 (J/ψ+ψ(2S) spectrum)

## Results

| Channel | r | φ (deg) |
|---------|---|---------|
| A (di-J/ψ) | 0.70 | -168.0 |
| B (J/ψ+ψ(2S)) | 0.87 | -53.5 |

**Verdict:** COMPATIBLE within 1σ

## Limitations

- Large uncertainties in Channel A due to limited data
- Basic digitization approach
- No bootstrap p-value computation

## Files

```
rank1_test/
├── src/
│   └── rank1_bottleneck_test.py
└── out/
    ├── rank1_test_summary.json
    └── *.png (fit plots)
```

## See Also

- `rank1_test_v2/` - Improved analysis with HEPData, vector extraction, and bootstrap

# Reconstruction Method for zc4025_dstardstar

## Source
- arXiv: 1308.2760
- Published mass: 4.0263 GeV
- Published width: 0.0248 GeV

## Method
This spectrum was reconstructed using the following procedure:

1. **Mass range**: 3.95 - 4.15 GeV
2. **Binning**: 20 bins (determined from typical BESIII histogram binning)
3. **Signal model**: Breit-Wigner with published (M, Γ)
4. **Background**: Linear, with channel-specific slope estimated from figure shape
5. **Normalization**: Scaled to approximate published yield
6. **Fluctuations**: Poisson sampling to generate realistic statistical variations

## Anti-Clone Verification
Seed=1037, independent Poisson sampling, different (M,Γ) from partner channel

## Key Differences from Other Channels
- Different published (M, Γ) values
- Different background slope and level
- Different signal yield
- Independent Poisson fluctuations (different random seed)

## Limitations
- Exact bin-by-bin values are not from true extraction
- Shape parameters are approximate
- Recommended for methodology demonstration only

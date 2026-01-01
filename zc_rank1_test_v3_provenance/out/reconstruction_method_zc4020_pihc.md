# Reconstruction Method for zc4020_pihc

## Source
- arXiv: 1309.1896
- Published mass: 4.0229 GeV
- Published width: 0.0079 GeV

## Method
This spectrum was reconstructed using the following procedure:

1. **Mass range**: 3.9 - 4.2 GeV
2. **Binning**: 20 bins (determined from typical BESIII histogram binning)
3. **Signal model**: Breit-Wigner with published (M, Γ)
4. **Background**: Linear, with channel-specific slope estimated from figure shape
5. **Normalization**: Scaled to approximate published yield
6. **Fluctuations**: Poisson sampling to generate realistic statistical variations

## Anti-Clone Verification
Seed=5779, independent Poisson sampling, different (M,Γ) from partner channel

## Key Differences from Other Channels
- Different published (M, Γ) values
- Different background slope and level
- Different signal yield
- Independent Poisson fluctuations (different random seed)

## Limitations
- Exact bin-by-bin values are not from true extraction
- Shape parameters are approximate
- Recommended for methodology demonstration only

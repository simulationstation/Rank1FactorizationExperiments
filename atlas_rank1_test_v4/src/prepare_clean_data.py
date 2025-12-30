#!/usr/bin/env python3
"""
Prepare clean bin data for ATLAS v4 analysis.
Extracts only counts (rounded to int) from existing extraction, no per-bin sigma.
"""

import pandas as pd
import numpy as np

def prepare_clean_data():
    """Read existing extraction and output clean integer counts only."""

    # Source files from v1 extraction
    src_4mu = "/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test/data/derived/4mu_bins.csv"
    src_4mu2pi = "/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test/data/derived/4mu+2pi_bins.csv"

    # Output files
    out_4mu = "/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/data/derived/4mu_bins.csv"
    out_4mu2pi = "/home/primary/DarkBItParticleColiderPredictions/atlas_rank1_test_v4/data/derived/4mu+2pi_bins.csv"

    for src, out, name in [(src_4mu, out_4mu, "4mu"), (src_4mu2pi, out_4mu2pi, "4mu+2pi")]:
        df = pd.read_csv(src)

        # Keep only bins with actual data
        df_clean = df[df['has_data'] == True].copy()

        # Round counts to integers for proper Poisson
        df_clean['count'] = df_clean['count'].round().astype(int)

        # Keep only essential columns
        df_out = df_clean[['m_low', 'm_high', 'm_center', 'count']].copy()

        # Save
        df_out.to_csv(out, index=False)

        print(f"{name}:")
        print(f"  Bins: {len(df_out)}")
        print(f"  Total counts: {df_out['count'].sum()}")
        print(f"  Mass range: {df_out['m_center'].min():.3f} - {df_out['m_center'].max():.3f} GeV")
        print(f"  Mean count/bin: {df_out['count'].mean():.1f}")
        print()

if __name__ == "__main__":
    prepare_clean_data()

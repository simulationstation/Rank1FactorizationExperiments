#!/usr/bin/env python3
"""
Prepare Channel A data from HEPData CSV.
CMS-BPH-21-003: J/ψJ/ψ invariant mass spectrum.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

def main():
    # Read HEPData CSV (skip comment lines)
    csv_path = os.path.join(DATA_DIR, 'hepdata', 'figure1_spectrum.csv')

    # Read with proper header handling
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Find header line (starts with $M)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('$M'):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header line")

    # Read from header
    df = pd.read_csv(csv_path, skiprows=header_idx)

    # Clean column names
    df.columns = ['m_center', 'm_low', 'm_high', 'count', 'stat_up', 'stat_down']

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna()

    # Compute symmetric error as average of asymmetric
    df['stat_err'] = (np.abs(df['stat_up']) + np.abs(df['stat_down'])) / 2

    # For bins with zero counts, use error of 1 (Poisson)
    df.loc[df['stat_err'] == 0, 'stat_err'] = 1.0

    # Save clean CSV
    output_path = os.path.join(DATA_DIR, 'hepdata', 'channel_A_mass_spectrum.csv')
    df[['m_center', 'm_low', 'm_high', 'count', 'stat_err']].to_csv(output_path, index=False)

    print(f"Channel A spectrum saved: {output_path}")
    print(f"Total bins: {len(df)}")
    print(f"Mass range: {df['m_low'].min():.3f} - {df['m_high'].max():.3f} GeV")
    print(f"Total counts: {df['count'].sum():.0f}")

    # Print summary for fit region
    fit_mask = (df['m_center'] >= 6.6) & (df['m_center'] <= 7.4)
    df_fit = df[fit_mask]
    print(f"\nFit region (6.6-7.4 GeV):")
    print(f"  Bins: {len(df_fit)}")
    print(f"  Counts: {df_fit['count'].sum():.0f}")

    # Show some data points
    print(f"\nSample data (around peaks):")
    sample_mask = (df['m_center'] >= 6.7) & (df['m_center'] <= 7.3)
    print(df[sample_mask][['m_center', 'count', 'stat_err']].head(20).to_string(index=False))

    return df

if __name__ == "__main__":
    main()

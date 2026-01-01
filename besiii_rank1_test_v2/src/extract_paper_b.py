#!/usr/bin/env python3
"""
Extract BESIII π+π- ψ(3686) cross-section data from Paper B (arXiv:2107.09210)
PRD 104, 052012 (2021) - Table I

Output: CSV with columns [sqrt_s_gev, sigma_pb, stat_err_pb, syst_err_pb]
"""

import csv
import os
import re

# Data extracted from Table I in PRD 104, 052012 (2021)
# Format: Ecms (GeV), σB (pb) with first error stat, second error syst
# Combined Born cross sections from pages 4-5

# Note: asymmetric errors converted to symmetric using average
paper_b_data = """
4.0076  0.4+0.5-0.3  0.0
4.0854  5.0+2.5-1.9  0.2
4.1285  7.1  1.0  0.3
4.1574  9.6  1.1  0.4
4.1784  10.2  0.4  0.5
4.1888  10.1  0.9  0.5
4.1989  14.1  1.2  0.6
4.2091  13.6  1.1  0.6
4.2185  19.0  1.3  0.9
4.2263  20.0  1.0  1.0
4.2357  17.7  1.3  0.9
4.2436  16.6  1.2  0.8
4.2580  18.1  1.0  0.9
4.2668  21.1  1.5  1.0
4.2777  30.3  3.1  1.5
4.2879  28.5  1.8  1.4
4.3079  31.7  5.6  1.4
4.3121  38.1  1.9  1.6
4.3374  52.0  2.2  2.2
4.3583  57.5  2.1  2.5
4.3774  59.4  2.2  2.6
4.3874  55.7  6.4  2.4
4.3964  56.2  2.2  2.6
4.4156  42.7  1.3  2.0
4.4362  33.4  1.6  1.5
4.4671  9.0+2.3-2.0  0.4
4.5271  7.9+2.1-1.8  0.3
4.5745  8.1+3.6-3.0  0.4
4.5995  12.9  1.0  0.6
4.6121  14.4  2.7  0.6
4.6279  20.0  1.4  0.9
4.6409  21.7  1.4  1.0
4.6613  24.0  1.5  1.1
4.6812  22.1  0.8  1.0
4.6984  18.9  1.3  0.8
"""

def parse_value_with_errors(line):
    """Parse a line with sqrt_s, sigma with errors, and syst error."""
    parts = line.strip().split()
    if not parts:
        return None

    sqrt_s = float(parts[0])

    # Check if sigma has asymmetric errors (format: X+Y-Z)
    sigma_part = parts[1]

    if '+' in sigma_part and '-' in sigma_part:
        # Asymmetric error: X+Y-Z  syst
        m = re.match(r'(-?[\d.]+)\+([\d.]+)-([\d.]+)', sigma_part)
        if m:
            sigma = float(m.group(1))
            err_up = float(m.group(2))
            err_down = float(m.group(3))
            stat_err = (err_up + err_down) / 2.0
            syst_err = float(parts[2])
    else:
        # Symmetric error: sigma  stat  syst
        sigma = float(parts[1])
        stat_err = float(parts[2])
        syst_err = float(parts[3])

    return (sqrt_s, sigma, stat_err, syst_err)

def main():
    results = []

    for line in paper_b_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        parsed = parse_value_with_errors(line)
        if parsed:
            results.append(parsed)

    print(f"Paper B (π+π- ψ(3686)): {len(results)} points")

    # Sort by energy
    results.sort(key=lambda x: x[0])

    print(f"Energy range: {results[0][0]:.4f} - {results[-1][0]:.4f} GeV")

    # Write CSV
    outdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(outdir, 'data', 'besiii_pipipsi3686.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sqrt_s_gev', 'sigma_pb', 'stat_err_pb', 'syst_err_pb'])
        for row in results:
            writer.writerow([f'{row[0]:.4f}', f'{row[1]:.1f}', f'{row[2]:.1f}', f'{row[3]:.1f}'])

    print(f"Written to: {outpath}")

    # Summary stats
    energies = [p[0] for p in results]
    sigmas = [p[1] for p in results]
    print(f"\nσ range: {min(sigmas):.1f} - {max(sigmas):.1f} pb")

    # Count points in overlap region (4.0 - 4.6 GeV for Y-states)
    overlap_points = [p for p in results if 4.0 <= p[0] <= 4.6]
    print(f"Points in 4.0-4.6 GeV range: {len(overlap_points)}")

if __name__ == '__main__':
    main()

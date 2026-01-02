#!/usr/bin/env python3
"""
Extract BESIII π+π- J/ψ cross-section data from Paper A (arXiv:1611.01317)
LaTeX source: draft_prl_v4.tex

Output: CSV with columns [sqrt_s_gev, sigma_pb, stat_err_pb, syst_err_pb]
"""

import re
import csv
import os

# Data extracted from LaTeX tables in draft_prl_v4.tex

# Table III (XYZ data) - lines 866-884
# Format: sqrt(s) GeV, L pb^-1, N_sig, epsilon, 1+delta, sigma (pb) with stat±syst
xyz_data = """
3.7730  2931.8  3093.3±61.5   0.423  0.732  28.5±0.6±1.7
3.8077  50.5    34.7±6.9      0.396  0.871  16.7±3.3±1.0
3.8962  52.6    36.1±7.1      0.393  0.856  17.1±3.4±1.0
4.0076  482.0   325.8±21.7    0.392  0.901  16.0±1.1±1.0
4.0855  52.6    33.9±6.9      0.374  0.961  15.0±3.1±0.9
4.1886  43.1    26.9±6.5      0.394  0.858  15.5±3.8±0.9
4.2077  54.6    114.9±11.6    0.446  0.740  53.4±5.4±3.1
4.2171  54.1    130.5±12.2    0.458  0.731  60.3±5.7±3.5
4.2263  1091.7  3853.1±68.1   0.465  0.748  85.1±1.5±4.9
4.2417  55.6    203.5±15.1    0.453  0.802  84.4±6.3±4.9
4.2580  825.7   2220.9±53.7   0.444  0.853  59.5±1.4±3.4
4.3079  44.9    101.7±11.2    0.398  0.917  52.0±5.7±3.0
4.3583  539.8   621.5±28.8    0.372  1.022  25.4±1.2±1.5
4.3874  55.2    50.5±8.1      0.331  1.155  20.0±3.2±1.2
4.4156  1073.6  574.5±28.3    0.302  1.227  12.1±0.6±0.7
4.4671  109.9   63.4±9.8      0.293  1.240  13.3±2.1±0.8
4.5271  110.0   50.0±8.8      0.293  1.223  10.6±1.9±0.6
4.5745  47.7    26.1±6.1      0.281  1.213  13.4±3.2±0.8
4.5995  566.9   143.4±15.9    0.274  1.205  6.4±0.7±0.4
"""

# Table IV (Scan data) - lines 899-924
# Format: sqrt(s) GeV, sigma (pb) with asymmetric stat ± syst
# Note: Asymmetric errors like X^{+Y}_{-Z} - we'll use symmetric approximation (Y+Z)/2
scan_data_raw = """
3.8874  9.7+13.1-9.1  0.6
3.8924  14.3+13.8-9.8  0.8
3.8974  20.7+13.4-9.3  1.2
3.9024  18.5+14.2-10.2  1.1
3.9074  16.0+12.8-8.5  0.9
3.9124  12.2+13.4-9.2  0.7
3.9174  3.6+11.4-6.6  0.2
3.9224  26.9+17.1-12.6  1.6
3.9274  24.2+15.6-11.1  1.4
3.9324  6.8+12.4-8.1  0.4
3.9374  13.5+12.7-8.5  0.8
3.9424  17.1+12.6-8.6  1.0
3.9474  22.2+14.8-11.0  1.3
3.9524  18.0+13.0-9.3  1.0
3.9574  21.0+13.9-10.4  1.2
3.9624  15.5+12.3-8.4  0.9
3.9674  14.4+13.2-9.1  0.8
3.9724  9.9+12.7-9.0  0.6
3.9774  9.2+11.7-7.9  0.5
3.9824  25.2+14.5-10.8  1.5
3.9874  10.0+12.1-8.4  0.6
3.9924  1.0+10.5-6.6  0.1
3.9974  18.5+12.7-8.9  1.1
4.0024  21.2+14.9-11.0  1.2
4.0074  21.0+14.3-10.2  1.2
4.0094  10.4+13.3-8.9  0.6
4.0114  25.0+15.3-10.9  1.4
4.0134  13.3+13.8-9.2  0.8
4.0154  14.8+13.6-9.3  0.9
4.0174  36.5+17.2-13.0  2.1
4.0224  32.7+16.6-12.2  1.9
4.0274  9.1+7.7-6.0  0.5
4.0324  22.3+15.2-10.9  1.3
4.0374  -2.4+11.7-7.0  0.1
4.0474  -1.2+12.6-8.0  0.1
4.0524  24.8+14.4-10.2  1.4
4.0574  14.7+13.9-9.2  0.9
4.0624  13.3+13.4-9.2  0.8
4.0674  10.7+12.3-8.2  0.6
4.0774  19.1+13.6-9.9  1.1
4.0874  12.2+12.9-9.1  0.7
4.0974  7.5+11.7-7.6  0.4
4.1074  9.9+12.6-8.5  0.6
4.1174  7.2+11.2-7.3  0.4
4.1274  10.0+12.7-8.5  0.6
4.1374  29.8+15.1-11.1  1.7
4.1424  12.4+12.5-8.6  0.7
4.1474  9.5+11.4-7.3  0.6
4.1574  29.4+15.5-11.8  1.7
4.1674  6.8+6.5-4.8  0.4
4.1774  26.0+14.4-10.2  1.5
4.1874  4.4+11.2-7.0  0.3
4.1924  27.7+14.6-10.6  1.6
4.1974  35.3+15.5-11.5  2.0
4.2004  49.1+19.9-15.6  2.8
4.2034  26.4+15.9-11.9  1.5
4.2074  29.7+15.1-11.1  1.7
4.2124  69.2+19.8-16.1  4.0
4.2174  64.3+19.5-15.9  3.7
4.2224  83.7+20.0-16.6  4.9
4.2274  124.5+22.9-19.7  7.2
4.2324  69.4+18.2-15.0  4.0
4.2374  99.4+21.4-18.0  5.8
4.2404  74.7+18.3-15.2  4.3
4.2424  47.0+15.5-12.3  2.7
4.2454  60.5+16.5-13.5  3.5
4.2474  66.3+16.6-13.5  3.8
4.2524  45.7+14.7-11.7  2.7
4.2574  75.9+17.1-14.3  4.4
4.2624  58.2+15.9-12.9  3.4
4.2674  75.6+17.2-14.3  4.4
4.2724  53.0+16.0-13.0  3.1
4.2774  38.4+14.1-11.0  2.2
4.2824  60.5+16.6-13.6  3.5
4.2874  60.1+15.7-12.8  3.5
4.2974  32.4+14.3-11.1  1.9
4.3074  64.0+16.4-13.3  3.7
4.3174  39.1+13.3-10.4  2.3
4.3274  27.9+13.2-10.0  1.6
4.3374  31.0+13.3-10.2  1.8
4.3474  14.0+11.4-8.2  0.8
4.3574  37.5+14.8-11.6  2.2
4.3674  34.8+13.7-10.6  2.0
4.3774  17.1+12.2-8.9  1.0
4.3874  20.5+13.2-9.6  1.2
4.3924  23.8+13.2-9.5  1.4
4.3974  17.5+12.1-8.2  1.0
4.4074  4.7+11.0-6.2  0.3
4.4174  16.9+12.3-8.6  1.0
4.4224  19.1+12.4-8.6  1.1
4.4274  9.9+11.9-7.6  0.6
4.4374  18.7+12.1-8.4  1.1
4.4474  3.0+10.2-6.4  0.2
4.4574  6.9+9.4-6.1  0.4
4.4774  12.2+11.2-7.7  0.7
4.4974  1.0+8.3-4.3  0.1
4.5174  12.7+10.2-6.7  0.7
4.5374  13.6+10.6-7.5  0.8
4.5474  14.7+10.8-7.4  0.9
4.5574  4.9+10.0-6.2  0.3
4.5674  7.8+10.6-6.8  0.5
4.5774  8.7+11.1-7.5  0.5
4.5874  2.0+8.7-4.4  0.1
"""

def parse_xyz_data():
    """Parse XYZ data with symmetric errors."""
    results = []
    for line in xyz_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Parse: sqrt_s  L  N±err  eps  1+delta  sigma±stat±syst
        parts = line.split()
        sqrt_s = float(parts[0])
        # Last part is sigma±stat±syst
        sigma_str = parts[-1]
        # Parse X±Y±Z format
        m = re.match(r'([\d.]+)±([\d.]+)±([\d.]+)', sigma_str)
        if m:
            sigma = float(m.group(1))
            stat_err = float(m.group(2))
            syst_err = float(m.group(3))
            results.append((sqrt_s, sigma, stat_err, syst_err))
    return results

def parse_scan_data():
    """Parse scan data with asymmetric errors (convert to symmetric)."""
    results = []
    for line in scan_data_raw.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        sqrt_s = float(parts[0])
        # Parse sigma+up-down format
        sigma_str = parts[1]
        syst_err = float(parts[2])

        # Parse X+Y-Z format (or negative values like -2.4+11.7-7.0)
        m = re.match(r'(-?[\d.]+)\+([\d.]+)-([\d.]+)', sigma_str)
        if m:
            sigma = float(m.group(1))
            err_up = float(m.group(2))
            err_down = float(m.group(3))
            # Symmetric approximation
            stat_err = (err_up + err_down) / 2.0
            results.append((sqrt_s, sigma, stat_err, syst_err))
    return results

def main():
    # Parse both tables
    xyz_points = parse_xyz_data()
    scan_points = parse_scan_data()

    print(f"XYZ data: {len(xyz_points)} points")
    print(f"Scan data: {len(scan_points)} points")

    # Combine all points
    all_points = xyz_points + scan_points

    # Sort by energy
    all_points.sort(key=lambda x: x[0])

    print(f"Total: {len(all_points)} points")
    print(f"Energy range: {all_points[0][0]:.4f} - {all_points[-1][0]:.4f} GeV")

    # Write CSV
    outdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(outdir, 'data', 'besiii_pipijpsi.csv')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sqrt_s_gev', 'sigma_pb', 'stat_err_pb', 'syst_err_pb'])
        for row in all_points:
            writer.writerow([f'{row[0]:.4f}', f'{row[1]:.1f}', f'{row[2]:.1f}', f'{row[3]:.1f}'])

    print(f"Written to: {outpath}")

    # Print summary stats
    energies = [p[0] for p in all_points]
    sigmas = [p[1] for p in all_points]
    print(f"\nEnergy range: {min(energies):.4f} - {max(energies):.4f} GeV")
    print(f"σ range: {min(sigmas):.1f} - {max(sigmas):.1f} pb")

    # Count points in overlap region (4.0 - 4.6 GeV for Y-states)
    overlap_points = [p for p in all_points if 4.0 <= p[0] <= 4.6]
    print(f"Points in 4.0-4.6 GeV range: {len(overlap_points)}")

if __name__ == '__main__':
    main()

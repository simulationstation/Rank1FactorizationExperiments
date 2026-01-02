#!/usr/bin/env python3
"""
Extract Channel B data from CMS-PAS-BPH-22-004 Figure 2 using VECTOR paths.
No pixel digitization - extract directly from PDF vector markers.

The PDF contains data points + error bars. We identify data points as the
higher-count marker at each mass value.
"""

import os
import json
import numpy as np
import pandas as pd
import fitz
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')


class PDFAxisMapper:
    """Map PDF coordinates to physics coordinates using tick marks."""

    def __init__(self, mass_ticks, count_ticks):
        self.mass_ticks = sorted(mass_ticks, key=lambda x: x['value'])
        self.count_ticks = sorted(count_ticks, key=lambda x: x['value'])

        # Fit linear mapping for mass
        ys = np.array([t['pdf_y'] for t in self.mass_ticks])
        ms = np.array([t['value'] for t in self.mass_ticks])
        A = np.vstack([ys, np.ones(len(ys))]).T
        self.mass_slope, self.mass_intercept = np.linalg.lstsq(A, ms, rcond=None)[0]

        # Fit linear mapping for counts
        xs = np.array([t['pdf_x'] for t in self.count_ticks])
        cs = np.array([t['value'] for t in self.count_ticks])
        A = np.vstack([xs, np.ones(len(xs))]).T
        self.count_slope, self.count_intercept = np.linalg.lstsq(A, cs, rcond=None)[0]

        print(f"Mass mapping: mass = {self.mass_slope:.6f} * pdf_y + {self.mass_intercept:.4f}")
        print(f"Count mapping: count = {self.count_slope:.6f} * pdf_x + {self.count_intercept:.4f}")

        # Store mapping quality metrics
        mass_pred = self.mass_slope * ys + self.mass_intercept
        count_pred = self.count_slope * xs + self.count_intercept
        self.mass_residual = np.sqrt(np.mean((mass_pred - ms)**2))
        self.count_residual = np.sqrt(np.mean((count_pred - cs)**2))
        print(f"Mapping residuals: mass={self.mass_residual:.4f} GeV, count={self.count_residual:.2f}")

    def pdf_to_physics(self, pdf_x, pdf_y):
        mass = self.mass_slope * pdf_y + self.mass_intercept
        count = self.count_slope * pdf_x + self.count_intercept
        return mass, count

    def physics_to_pdf(self, mass, count):
        pdf_y = (mass - self.mass_intercept) / self.mass_slope
        pdf_x = (count - self.count_intercept) / self.count_slope
        return pdf_x, pdf_y

    def perturb(self, scale_frac=0.005, shift_pts=1.0):
        """Return perturbed mapper for systematic estimation."""
        perturbed = PDFAxisMapper.__new__(PDFAxisMapper)
        perturbed.mass_ticks = self.mass_ticks
        perturbed.count_ticks = self.count_ticks
        perturbed.mass_slope = self.mass_slope * (1 + np.random.uniform(-scale_frac, scale_frac))
        perturbed.mass_intercept = self.mass_intercept + np.random.uniform(-shift_pts, shift_pts) * abs(self.mass_slope)
        perturbed.count_slope = self.count_slope * (1 + np.random.uniform(-scale_frac, scale_frac))
        perturbed.count_intercept = self.count_intercept + np.random.uniform(-shift_pts, shift_pts) * abs(self.count_slope)
        return perturbed


def extract_markers_from_pdf(pdf_path):
    """Extract circular data markers from PDF vector paths."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    paths = page.get_drawings()

    markers = []
    seen = set()

    for path in paths:
        items = path.get('items', [])
        rect = path.get('rect')
        fill = path.get('fill')

        has_curves = any(item[0] == 'c' for item in items)

        if has_curves and rect:
            w, h = rect.width, rect.height
            if 2 < w < 15 and 2 < h < 15 and 0.3 < w/h < 3:
                cx = (rect.x0 + rect.x1) / 2
                cy = (rect.y0 + rect.y1) / 2

                key = (round(cx, 1), round(cy, 1))
                if key not in seen:
                    seen.add(key)
                    markers.append({
                        'pdf_x': cx,
                        'pdf_y': cy,
                        'width': w,
                        'height': h,
                        'fill': fill
                    })

    # Extract text for axis ticks
    text_dict = page.get_text("dict")
    blocks = text_dict.get('blocks', [])

    text_items = []
    for block in blocks:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if text:
                        bbox = span['bbox']
                        text_items.append({
                            'text': text,
                            'x': (bbox[0] + bbox[2]) / 2,
                            'y': (bbox[1] + bbox[3]) / 2
                        })

    # Find mass tick labels (7.0, 7.5, 8.0, 8.5, 9.0)
    mass_ticks = []
    for t in text_items:
        try:
            val = float(t['text'])
            # Mass ticks: 7, 7.5, 8, 8.5, 9
            if val in [7.0, 7.5, 8.0, 8.5, 9.0]:
                mass_ticks.append({'value': val, 'pdf_y': t['y'], 'pdf_x': t['x']})
        except:
            pass

    # Find count tick labels (0, 5, 10, 15, 20, 25) - must be on left side
    count_ticks = []
    valid_counts = {0, 5, 10, 15, 20, 25, 30}
    for t in text_items:
        try:
            val = float(t['text'])
            if val in valid_counts and t['x'] < 300:
                count_ticks.append({'value': val, 'pdf_x': t['x'], 'pdf_y': t['y']})
        except:
            pass

    # Deduplicate - keep first occurrence of each value
    unique_mass = {}
    for mt in mass_ticks:
        v = mt['value']
        if v not in unique_mass:
            unique_mass[v] = mt

    unique_count = {}
    for ct in count_ticks:
        v = ct['value']
        if v not in unique_count:
            unique_count[v] = ct

    doc.close()

    return {
        'markers': markers,
        'mass_ticks': list(unique_mass.values()),
        'count_ticks': list(unique_count.values())
    }


def identify_data_points(markers, mapper, bin_width=0.040, mass_range=(7.0, 9.0)):
    """
    Identify which markers are data points vs error bar endpoints.
    At each mass value, there are typically 2 markers: data point (higher) and lower error endpoint.
    We take the higher count marker as the data point.
    """
    # Convert all markers to physics
    phys_markers = []
    for m in markers:
        mass, count = mapper.pdf_to_physics(m['pdf_x'], m['pdf_y'])
        if mass_range[0] - 0.02 <= mass <= mass_range[1] + 0.02:
            phys_markers.append({
                **m,
                'mass_GeV': mass,
                'count': count
            })

    # Group by mass (within 0.5 * bin_width)
    groups = defaultdict(list)
    for m in phys_markers:
        # Round to nearest half-bin
        bin_idx = round((m['mass_GeV'] - mass_range[0]) / bin_width)
        groups[bin_idx].append(m)

    data_points = []
    error_bars = []

    for bin_idx, group in groups.items():
        if len(group) == 0:
            continue

        # Sort by count (descending)
        group.sort(key=lambda x: x['count'], reverse=True)

        # Highest count = data point
        data_point = group[0]

        # If there are multiple markers, second highest might be lower error bar endpoint
        if len(group) >= 2:
            lower_endpoint = group[-1]  # Lowest count marker
            error_bar = data_point['count'] - lower_endpoint['count']
        else:
            error_bar = np.sqrt(max(data_point['count'], 1))

        data_points.append({
            'mass_GeV': data_point['mass_GeV'],
            'count': max(0, data_point['count']),
            'pdf_x': data_point['pdf_x'],
            'pdf_y': data_point['pdf_y'],
            'error_from_bars': error_bar
        })
        error_bars.append(error_bar)

    return data_points, error_bars


def bin_data_points(data_points, bin_width=0.040, mass_range=(7.0, 9.0)):
    """Bin data points into histogram bins."""
    n_bins = int((mass_range[1] - mass_range[0]) / bin_width)
    bins = []

    for i in range(n_bins):
        m_low = mass_range[0] + i * bin_width
        m_high = m_low + bin_width
        m_center = (m_low + m_high) / 2

        in_bin = [d for d in data_points if m_low <= d['mass_GeV'] < m_high]

        if in_bin:
            # Should be exactly one data point per bin after grouping
            count = in_bin[0]['count']
            error_bar = in_bin[0].get('error_from_bars', np.sqrt(max(count, 1)))
            pdf_x = in_bin[0]['pdf_x']
            pdf_y = in_bin[0]['pdf_y']
        else:
            count = 0.0
            error_bar = 1.0
            pdf_x = None
            pdf_y = None

        bins.append({
            'm_low': m_low,
            'm_high': m_high,
            'm_center': m_center,
            'count': count,
            'error_bar': error_bar,
            'pdf_x': pdf_x,
            'pdf_y': pdf_y,
            'has_data': len(in_bin) > 0
        })

    return bins


def compute_digitization_systematic(pdf_path, n_perturbations=200):
    """Estimate digitization systematic via perturbation analysis."""
    extraction = extract_markers_from_pdf(pdf_path)
    markers = extraction['markers']
    mass_ticks = extraction['mass_ticks']
    count_ticks = extraction['count_ticks']

    base_mapper = PDFAxisMapper(mass_ticks, count_ticks)
    base_data, _ = identify_data_points(markers, base_mapper)
    base_bins = bin_data_points(base_data)
    n_bins = len(base_bins)

    perturbed_counts = np.zeros((n_perturbations, n_bins))

    for i in range(n_perturbations):
        pert_mapper = base_mapper.perturb(scale_frac=0.005, shift_pts=1.0)
        pert_data, _ = identify_data_points(markers, pert_mapper)
        pert_bins = bin_data_points(pert_data)

        for j, b in enumerate(pert_bins):
            perturbed_counts[i, j] = b['count']

    sigma_digit = np.std(perturbed_counts, axis=0)
    print(f"Digitization systematic: mean={np.mean(sigma_digit):.3f}, max={np.max(sigma_digit):.3f}")

    return sigma_digit, base_bins, perturbed_counts


def main():
    print("="*60)
    print("Channel B Vector Extraction (v2 - data point identification)")
    print("="*60)

    pdf_path = os.path.join(DATA_DIR, 'cds', 'fig_B.pdf')

    # Extract markers and axis ticks
    print("\n1. Extracting vector paths from PDF...")
    extraction = extract_markers_from_pdf(pdf_path)

    markers = extraction['markers']
    mass_ticks = extraction['mass_ticks']
    count_ticks = extraction['count_ticks']

    print(f"   Found {len(markers)} unique markers")
    print(f"   Found {len(mass_ticks)} mass tick labels: {[mt['value'] for mt in mass_ticks]}")
    print(f"   Found {len(count_ticks)} count tick labels: {[ct['value'] for ct in count_ticks]}")

    # Create axis mapper
    print("\n2. Building axis mapping...")
    mapper = PDFAxisMapper(mass_ticks, count_ticks)

    # Identify data points vs error bar endpoints
    print("\n3. Identifying data points...")
    data_points, error_bars = identify_data_points(markers, mapper)
    print(f"   Found {len(data_points)} data points")

    # Bin the data
    print("\n4. Binning data (40 MeV bins)...")
    bins = bin_data_points(data_points)

    # Compute digitization systematic
    print("\n5. Computing digitization systematic (200 perturbations)...")
    sigma_digit, _, perturbed_counts = compute_digitization_systematic(pdf_path, n_perturbations=200)

    # Build final dataframe
    df = pd.DataFrame(bins)
    df['sigma_poisson'] = np.sqrt(np.maximum(df['count'], 1))
    df['sigma_digit'] = sigma_digit
    df['sigma_total'] = np.sqrt(df['sigma_poisson']**2 + df['sigma_digit']**2)

    # Filter to bins with data
    df_data = df[df['has_data']].copy()

    # Save CSV (all bins for fitting context)
    output_path = os.path.join(DATA_DIR, 'derived', 'channel_B_vector_bins.csv')
    df[['m_low', 'm_high', 'm_center', 'count', 'sigma_poisson', 'sigma_digit', 'sigma_total', 'has_data']].to_csv(
        output_path, index=False
    )
    print(f"\n6. Saved: {output_path}")

    # Also save perturbed counts for bootstrap
    np.save(os.path.join(DATA_DIR, 'derived', 'channel_B_perturbed_counts.npy'), perturbed_counts)

    # Create debug overlay
    print("\n7. Creating debug overlay...")
    create_debug_overlay(pdf_path, data_points, mapper)

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total bins: {len(df)}")
    print(f"Bins with data: {len(df_data)}")
    print(f"Mass range: {df_data['m_center'].min():.3f} - {df_data['m_center'].max():.3f} GeV")
    print(f"Total counts: {df_data['count'].sum():.1f}")
    print(f"Mean count/bin: {df_data['count'].mean():.1f}")

    print("\n   Data points:")
    for _, row in df_data.iterrows():
        print(f"      {row['m_center']:.3f} GeV: {row['count']:.1f} ± {row['sigma_total']:.2f}")

    return df


def create_debug_overlay(pdf_path, data_points, mapper):
    """Create debug visualization."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    doc.close()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img)

    # Plot extracted data points
    for d in data_points:
        px = d['pdf_x'] * zoom
        py = d['pdf_y'] * zoom
        ax.plot(px, py, 'go', markersize=10, markerfacecolor='none', linewidth=2)
        ax.annotate(f"{d['count']:.1f}", (px + 8, py - 5), fontsize=7, color='green',
                   fontweight='bold')

    ax.set_title("Channel B Vector Extraction - Data Points Identified")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'debug_B_vector_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: out/debug_B_vector_overlay.png")


if __name__ == "__main__":
    main()

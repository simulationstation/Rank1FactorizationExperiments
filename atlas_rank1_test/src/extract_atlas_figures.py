#!/usr/bin/env python3
"""
Extract spectrum data from ATLAS PDF figures using vector paths.
No pixel centroid digitization - extract directly from PDF vector markers.
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


class ATLASAxisMapper:
    """Map ATLAS PDF coordinates to physics coordinates."""

    def __init__(self, mass_ticks, count_ticks, name=""):
        """
        ATLAS figures have rotated axes:
        - Mass on vertical axis (PDF y)
        - Counts on horizontal axis (PDF x)
        """
        self.name = name
        self.mass_ticks = sorted(mass_ticks, key=lambda x: x['value'])
        self.count_ticks = sorted(count_ticks, key=lambda x: x['value'])

        # Fit linear mapping for mass: mass = a * pdf_y + b
        ys = np.array([t['pdf_y'] for t in self.mass_ticks])
        ms = np.array([t['value'] for t in self.mass_ticks])
        A = np.vstack([ys, np.ones(len(ys))]).T
        self.mass_slope, self.mass_intercept = np.linalg.lstsq(A, ms, rcond=None)[0]

        # Fit linear mapping for counts: count = a * pdf_x + b
        xs = np.array([t['pdf_x'] for t in self.count_ticks])
        cs = np.array([t['value'] for t in self.count_ticks])
        A = np.vstack([xs, np.ones(len(xs))]).T
        self.count_slope, self.count_intercept = np.linalg.lstsq(A, cs, rcond=None)[0]

        print(f"[{name}] Mass mapping: mass = {self.mass_slope:.6f} * pdf_y + {self.mass_intercept:.4f}")
        print(f"[{name}] Count mapping: count = {self.count_slope:.6f} * pdf_x + {self.count_intercept:.4f}")

    def pdf_to_physics(self, pdf_x, pdf_y):
        """Convert PDF coordinates to (mass_GeV, count)."""
        mass = self.mass_slope * pdf_y + self.mass_intercept
        count = self.count_slope * pdf_x + self.count_intercept
        return mass, count

    def perturb(self, scale_frac=0.005, shift_pts=1.0):
        """Return a perturbed copy for systematic estimation."""
        perturbed = ATLASAxisMapper.__new__(ATLASAxisMapper)
        perturbed.name = self.name
        perturbed.mass_ticks = self.mass_ticks
        perturbed.count_ticks = self.count_ticks
        perturbed.mass_slope = self.mass_slope * (1 + np.random.uniform(-scale_frac, scale_frac))
        perturbed.mass_intercept = self.mass_intercept + np.random.uniform(-shift_pts, shift_pts) * abs(self.mass_slope)
        perturbed.count_slope = self.count_slope * (1 + np.random.uniform(-scale_frac, scale_frac))
        perturbed.count_intercept = self.count_intercept + np.random.uniform(-shift_pts, shift_pts) * abs(self.count_slope)
        return perturbed


def extract_markers_from_pdf(pdf_path):
    """Extract data markers and axis ticks from ATLAS PDF."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    paths = page.get_drawings()

    # Find curved paths (data markers)
    markers = []
    seen = set()

    for path in paths:
        items = path.get('items', [])
        rect = path.get('rect')
        fill = path.get('fill')

        has_curve = any(item[0] == 'c' for item in items)

        if has_curve and rect:
            w, h = rect.width, rect.height
            # ATLAS markers are ~7.1 x 5.3 pixels
            if 4 < w < 12 and 3 < h < 10:
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

    # Extract text for axis labels
    text_dict = page.get_text("dict")
    blocks = text_dict.get('blocks', [])

    text_items = []
    for block in blocks:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip().replace('−', '-')
                    if text:
                        bbox = span['bbox']
                        text_items.append({
                            'text': text,
                            'x': (bbox[0] + bbox[2]) / 2,
                            'y': (bbox[1] + bbox[3]) / 2
                        })

    doc.close()

    return markers, text_items


def identify_axis_ticks(text_items, fig_type):
    """Identify mass and count ticks from text items."""
    mass_ticks = []
    count_ticks = []

    # Valid mass values (GeV)
    valid_mass = {6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5}
    # Valid count values
    valid_count = {0, 20, 40, 50, 60, 100, 150, 200, 250}

    for t in text_items:
        try:
            val = float(t['text'])

            # Mass ticks: on right side of figure (high x), at specific values
            if val in valid_mass and t['x'] > 450:
                mass_ticks.append({'value': val, 'pdf_y': t['y'], 'pdf_x': t['x']})

            # Count ticks: at bottom of figure (high y), at specific values
            if val in valid_count and t['y'] > 500:
                count_ticks.append({'value': val, 'pdf_x': t['x'], 'pdf_y': t['y']})

        except:
            pass

    # Deduplicate
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

    return list(unique_mass.values()), list(unique_count.values())


def identify_data_points(markers, mapper, mass_range, bin_width=0.04):
    """
    Identify data points from markers.
    Group markers by mass bin and take highest count as data point.
    """
    phys_markers = []
    for m in markers:
        mass, count = mapper.pdf_to_physics(m['pdf_x'], m['pdf_y'])
        if mass_range[0] - 0.1 <= mass <= mass_range[1] + 0.1:
            phys_markers.append({
                **m,
                'mass_GeV': mass,
                'count': count
            })

    # Group by mass bin
    groups = defaultdict(list)
    for m in phys_markers:
        bin_idx = round((m['mass_GeV'] - mass_range[0]) / bin_width)
        groups[bin_idx].append(m)

    data_points = []
    for bin_idx, group in groups.items():
        if len(group) == 0:
            continue

        # Sort by count (descending) - highest = data point
        group.sort(key=lambda x: x['count'], reverse=True)
        data_point = group[0]

        # Error bar from second marker if present
        if len(group) >= 2:
            error_bar = data_point['count'] - group[-1]['count']
        else:
            error_bar = np.sqrt(max(data_point['count'], 1))

        data_points.append({
            'mass_GeV': data_point['mass_GeV'],
            'count': max(0, data_point['count']),
            'error_bar': abs(error_bar),
            'pdf_x': data_point['pdf_x'],
            'pdf_y': data_point['pdf_y']
        })

    return data_points


def bin_data_points(data_points, bin_width, mass_range):
    """Bin data points into histogram bins."""
    n_bins = int((mass_range[1] - mass_range[0]) / bin_width)
    bins = []

    for i in range(n_bins):
        m_low = mass_range[0] + i * bin_width
        m_high = m_low + bin_width
        m_center = (m_low + m_high) / 2

        in_bin = [d for d in data_points if m_low <= d['mass_GeV'] < m_high]

        if in_bin:
            count = in_bin[0]['count']
            error_bar = in_bin[0]['error_bar']
            pdf_x = in_bin[0]['pdf_x']
            pdf_y = in_bin[0]['pdf_y']
            has_data = True
        else:
            count = 0.0
            error_bar = 1.0
            pdf_x = None
            pdf_y = None
            has_data = False

        bins.append({
            'm_low': m_low,
            'm_high': m_high,
            'm_center': m_center,
            'count': count,
            'error_bar': error_bar,
            'pdf_x': pdf_x,
            'pdf_y': pdf_y,
            'has_data': has_data
        })

    return bins


def compute_digitization_systematic(pdf_path, fig_type, mass_range, bin_width, n_perturbations=200):
    """Compute digitization systematic via perturbation analysis."""
    markers, text_items = extract_markers_from_pdf(pdf_path)
    mass_ticks, count_ticks = identify_axis_ticks(text_items, fig_type)

    base_mapper = ATLASAxisMapper(mass_ticks, count_ticks, fig_type)
    base_data = identify_data_points(markers, base_mapper, mass_range, bin_width)
    base_bins = bin_data_points(base_data, bin_width, mass_range)
    n_bins = len(base_bins)

    perturbed_counts = np.zeros((n_perturbations, n_bins))

    for i in range(n_perturbations):
        pert_mapper = base_mapper.perturb(scale_frac=0.005, shift_pts=1.0)
        pert_data = identify_data_points(markers, pert_mapper, mass_range, bin_width)
        pert_bins = bin_data_points(pert_data, bin_width, mass_range)

        for j, b in enumerate(pert_bins):
            perturbed_counts[i, j] = b['count']

    sigma_digit = np.std(perturbed_counts, axis=0)
    print(f"[{fig_type}] Digitization systematic: mean={np.mean(sigma_digit):.2f}, max={np.max(sigma_digit):.2f}")

    return sigma_digit, base_bins, perturbed_counts


def extract_figure(pdf_path, fig_type, mass_range, bin_width):
    """Full extraction pipeline for one ATLAS figure."""
    print(f"\n{'='*60}")
    print(f"Extracting: {fig_type}")
    print(f"{'='*60}")

    # Extract markers and text
    markers, text_items = extract_markers_from_pdf(pdf_path)
    print(f"Found {len(markers)} markers, {len(text_items)} text items")

    # Identify axis ticks
    mass_ticks, count_ticks = identify_axis_ticks(text_items, fig_type)
    print(f"Mass ticks: {[mt['value'] for mt in mass_ticks]}")
    print(f"Count ticks: {[ct['value'] for ct in count_ticks]}")

    if len(mass_ticks) < 2 or len(count_ticks) < 2:
        print(f"ERROR: Not enough axis ticks for {fig_type}")
        return None

    # Create mapper
    mapper = ATLASAxisMapper(mass_ticks, count_ticks, fig_type)

    # Identify data points
    data_points = identify_data_points(markers, mapper, mass_range, bin_width)
    print(f"Identified {len(data_points)} data points")

    # Bin the data
    bins = bin_data_points(data_points, bin_width, mass_range)

    # Compute digitization systematic
    sigma_digit, _, perturbed_counts = compute_digitization_systematic(
        pdf_path, fig_type, mass_range, bin_width, n_perturbations=200
    )

    # Build dataframe
    df = pd.DataFrame(bins)
    df['sigma_poisson'] = np.sqrt(np.maximum(df['count'], 1))
    df['sigma_digit'] = sigma_digit
    df['sigma_total'] = np.sqrt(df['sigma_poisson']**2 + df['sigma_digit']**2)

    # Save
    output_name = fig_type.replace(' ', '_').replace('/', '_').lower()
    df[['m_low', 'm_high', 'm_center', 'count', 'sigma_poisson', 'sigma_digit', 'sigma_total', 'has_data']].to_csv(
        os.path.join(DATA_DIR, 'derived', f'{output_name}_bins.csv'), index=False
    )

    # Save perturbed counts for bootstrap
    np.save(os.path.join(DATA_DIR, 'derived', f'{output_name}_perturbed.npy'), perturbed_counts)

    # Create debug overlay
    create_debug_overlay(pdf_path, data_points, mapper, fig_type)

    # Summary
    df_data = df[df['has_data']]
    print(f"\nSummary for {fig_type}:")
    print(f"  Total bins: {len(df)}, with data: {len(df_data)}")
    print(f"  Mass range: {df_data['m_center'].min():.2f} - {df_data['m_center'].max():.2f} GeV")
    print(f"  Total counts: {df_data['count'].sum():.1f}")

    return df


def create_debug_overlay(pdf_path, data_points, mapper, fig_type):
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

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)

    for d in data_points:
        px = d['pdf_x'] * zoom
        py = d['pdf_y'] * zoom
        ax.plot(px, py, 'go', markersize=10, markerfacecolor='none', linewidth=2)
        ax.annotate(f"{d['count']:.0f}", (px + 8, py - 5), fontsize=7, color='green', fontweight='bold')

    output_name = fig_type.replace(' ', '_').replace('/', '_').lower()
    ax.set_title(f"ATLAS {fig_type} - Vector Extraction")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'debug_{output_name}_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved debug overlay: out/debug_{output_name}_overlay.png")


def main():
    print("="*70)
    print("ATLAS Figure Extraction")
    print("="*70)

    # Figure configurations
    figures = [
        {
            'pdf': 'fig_01a.pdf',
            'type': 'di-Jpsi',
            'mass_range': (6.2, 9.0),
            'bin_width': 0.05,
        },
        {
            'pdf': 'fig_01b.pdf',
            'type': '4mu',
            'mass_range': (6.8, 9.5),
            'bin_width': 0.05,
        },
        {
            'pdf': 'fig_01c.pdf',
            'type': '4mu+2pi',
            'mass_range': (6.8, 9.5),
            'bin_width': 0.05,
        },
    ]

    results = {}
    for fig in figures:
        pdf_path = os.path.join(DATA_DIR, 'atlas_figs', fig['pdf'])
        if os.path.exists(pdf_path):
            df = extract_figure(pdf_path, fig['type'], fig['mass_range'], fig['bin_width'])
            if df is not None:
                results[fig['type']] = df
        else:
            print(f"File not found: {pdf_path}")

    print("\n" + "="*70)
    print("Extraction complete")
    print("="*70)

    return results


if __name__ == "__main__":
    main()

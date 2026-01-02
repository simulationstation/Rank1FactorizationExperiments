#!/usr/bin/env python3
"""
Final digitization script with calibrated plot regions.
Uses detected axis positions and known physical ranges.
"""

import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Calibrated plot regions (from line detection analysis)
# These are the pixel coordinates of the plot axes
PLOT_REGIONS = {
    'A': {
        'left': 1072,
        'right': 2662,
        'top': 760,
        'bottom': 1688,
    },
    'B': {
        'left': 1145,
        'right': 2888,
        'top': 767,
        'bottom': 1751,
    }
}

# Physical axis ranges from PAS documents
AXIS_CONFIG = {
    'A': {
        'x_min': 6.5, 'x_max': 9.0,  # GeV
        'y_min': 0, 'y_max': 600,     # Candidates / 25 MeV
        'bin_width': 0.025,           # 25 MeV bins
        'fit_window': (6.6, 7.5),
    },
    'B': {
        'x_min': 7.0, 'x_max': 9.0,  # GeV
        'y_min': 0, 'y_max': 30,      # Candidates / 40 MeV
        'bin_width': 0.040,           # 40 MeV bins
        'fit_window': (6.9, 7.6),
    }
}

def load_image(path):
    """Load image."""
    img_color = cv2.imread(path)
    if img_color is None:
        raise ValueError(f"Could not load image: {path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def detect_data_markers(img_gray, region, channel):
    """
    Detect data point markers within the plot region.
    CMS uses filled black circles with error bars.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Extract plot region with padding to avoid axis lines
    pad = 15
    roi = img_gray[top+pad:bottom-pad, left+pad:right-pad].copy()
    roi_h, roi_w = roi.shape

    print(f"  ROI dimensions: {roi_w} x {roi_h}")

    # Threshold to find dark objects (data points and error bars are black)
    _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  Found {len(contours)} contours")

    # Filter contours to find data point markers
    markers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Data markers are small-medium sized blobs
        if area < 50 or area > 3000:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip objects too close to edges
        if x < 20 or x > roi_w - 20:
            continue
        if y < 15 or y > roi_h - 15:
            continue

        # Calculate centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calculate circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Calculate aspect ratio
            aspect = w / max(h, 1)

            # Data points should be roughly circular and compact
            # Error bars are elongated vertically
            if circularity > 0.15 and 0.1 < aspect < 10:
                markers.append({
                    'x_roi': cx,
                    'y_roi': cy,
                    'x_abs': cx + left + pad,
                    'y_abs': cy + top + pad,
                    'area': area,
                    'circularity': circularity,
                    'aspect': aspect
                })

    print(f"  Filtered to {len(markers)} candidate markers")
    return markers, (roi_w, roi_h), pad

def cluster_markers_by_x(markers, bin_width_px):
    """
    Group markers by x-coordinate into histogram bins.
    """
    if not markers:
        return []

    # Sort by x
    markers = sorted(markers, key=lambda m: m['x_roi'])

    # Cluster by x proximity
    clusters = []
    current_cluster = [markers[0]]

    for m in markers[1:]:
        if m['x_roi'] - current_cluster[-1]['x_roi'] < bin_width_px * 0.7:
            current_cluster.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = [m]
    clusters.append(current_cluster)

    # For each cluster, find the best representative
    # Prefer markers that are:
    # 1. Higher circularity (more likely to be the data point, not error bar)
    # 2. Reasonable y position (not at extreme top or bottom)
    result = []
    for cluster in clusters:
        if len(cluster) == 1:
            result.append(cluster[0])
        else:
            # Sort by circularity
            sorted_cluster = sorted(cluster, key=lambda m: -m['circularity'])
            # Take the most circular marker
            best = sorted_cluster[0]
            result.append(best)

    return result

def pixels_to_physical(markers, region, config, roi_size, pad):
    """
    Convert pixel coordinates to physical (mass, count) values.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # The actual data region (excluding padding)
    plot_w = right - left - 2 * pad
    plot_h = bottom - top - 2 * pad

    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']

    data = []
    for m in markers:
        # Relative position (0 to 1)
        rel_x = m['x_roi'] / plot_w
        rel_y = m['y_roi'] / plot_h

        # Physical values
        mass = x_min + rel_x * (x_max - x_min)
        count = y_max - rel_y * (y_max - y_min)  # y is inverted

        if count > 0 and x_min <= mass <= x_max:
            data.append((mass, count))

    return data

def bin_data(data, config):
    """
    Bin data points into histogram format.
    """
    bin_width = config['bin_width']
    x_min, x_max = config['x_min'], config['x_max']

    bins = defaultdict(list)

    for mass, count in data:
        bin_idx = int((mass - x_min) / bin_width)
        bin_center = x_min + (bin_idx + 0.5) * bin_width
        bins[bin_center].append(count)

    # Average counts in each bin
    result = []
    for bin_center in sorted(bins.keys()):
        if x_min <= bin_center <= x_max:
            counts = bins[bin_center]
            avg_count = np.median(counts)  # Use median to be robust
            result.append((bin_center, avg_count))

    return result

def process_channel(channel, img_path):
    """Process one channel."""
    print(f"\n{'='*60}")
    print(f"CHANNEL {channel}")
    print('='*60)

    config = AXIS_CONFIG[channel]
    region = PLOT_REGIONS[channel]

    # Load image
    img_color, img_gray = load_image(img_path)
    print(f"Image: {img_path}")
    print(f"Size: {img_gray.shape[1]} x {img_gray.shape[0]}")
    print(f"Plot region: L={region['left']}, R={region['right']}, T={region['top']}, B={region['bottom']}")

    # Detect markers
    markers, roi_size, pad = detect_data_markers(img_gray, region, channel)

    # Estimate bin width in pixels
    plot_w = region['right'] - region['left'] - 2 * pad
    x_range = config['x_max'] - config['x_min']
    bin_width_px = (config['bin_width'] / x_range) * plot_w
    print(f"  Bin width: {bin_width_px:.1f} pixels")

    # Cluster by x to get one point per bin
    clustered = cluster_markers_by_x(markers, bin_width_px)
    print(f"  Clustered to {len(clustered)} points")

    # Convert to physical values
    data = pixels_to_physical(clustered, region, config, roi_size, pad)

    # Bin the data
    binned = bin_data(data, config)

    # Create debug image
    debug_img = img_color.copy()
    cv2.rectangle(debug_img,
                  (region['left'], region['top']),
                  (region['right'], region['bottom']),
                  (0, 255, 0), 3)

    # Mark all detected markers in blue
    for m in markers:
        cv2.circle(debug_img, (m['x_abs'], m['y_abs']), 4, (255, 0, 0), -1)

    # Mark clustered (selected) markers in red
    for m in clustered:
        cv2.circle(debug_img, (m['x_abs'], m['y_abs']), 7, (0, 0, 255), 2)

    # Save outputs
    df = pd.DataFrame(binned, columns=['mass_GeV', 'count'])

    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    debug_path = os.path.join(OUT_DIR, f'debug_{channel}_overlay.png')

    df.to_csv(csv_path, index=False)
    cv2.imwrite(debug_path, debug_img)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {debug_path}")

    # Show data summary
    print(f"\nData summary:")
    print(f"  Total bins: {len(df)}")
    if len(df) > 0:
        print(f"  Mass range: {df['mass_GeV'].min():.3f} - {df['mass_GeV'].max():.3f} GeV")
        print(f"  Count range: {df['count'].min():.1f} - {df['count'].max():.1f}")

        # Filter to fit window
        fit_min, fit_max = config['fit_window']
        df_fit = df[(df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)]
        print(f"  Points in fit window ({fit_min}-{fit_max} GeV): {len(df_fit)}")

        print(f"\nFirst 15 data points:")
        print(df.head(15).to_string(index=False))

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig_a_path = os.path.join(DATA_DIR, 'fig_A.png')
    fig_b_path = os.path.join(DATA_DIR, 'fig_B.png')

    df_a = process_channel('A', fig_a_path)
    df_b = process_channel('B', fig_b_path)

    print("\n" + "="*60)
    print("DIGITIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

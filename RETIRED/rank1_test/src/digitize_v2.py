#!/usr/bin/env python3
"""
Simplified but robust digitization for CMS histogram data points.
Uses a combination of edge detection and blob analysis.
"""

import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Axis configurations - verified from PDFs
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

def find_plot_region_manual(img_gray, channel):
    """
    Find plot region using line detection with fallback to manual estimation.
    For CMS PAS figures, the plots are in specific locations.
    """
    h, w = img_gray.shape

    # Use Canny edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=200, maxLineGap=20)

    if lines is None:
        # Manual fallback based on typical CMS figure layout
        if channel == 'A':
            return {'left': 480, 'right': 1650, 'top': 255, 'bottom': 820}
        else:
            return {'left': 510, 'right': 1280, 'top': 255, 'bottom': 925}

    # Find horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if angle < 5:  # Horizontal
            h_lines.append({'y': (y1+y2)/2, 'x1': min(x1,x2), 'x2': max(x1,x2), 'len': length})
        elif angle > 85:  # Vertical
            v_lines.append({'x': (x1+x2)/2, 'y1': min(y1,y2), 'y2': max(y1,y2), 'len': length})

    # Sort by length
    h_lines.sort(key=lambda l: -l['len'])
    v_lines.sort(key=lambda l: -l['len'])

    # Find x-axis (long horizontal in upper part of image - since this is full page)
    x_axis_y = None
    for hl in h_lines:
        if h * 0.1 < hl['y'] < h * 0.5 and hl['len'] > w * 0.2:
            x_axis_y = int(hl['y'])
            break

    # Find y-axis (long vertical on left side)
    y_axis_x = None
    for vl in v_lines:
        if vl['x'] < w * 0.5 and vl['len'] > h * 0.05:
            y_axis_x = int(vl['x'])
            break

    if x_axis_y is None or y_axis_x is None:
        # Fallback
        print(f"Warning: Could not detect axes for channel {channel}, using fallback")
        if channel == 'A':
            return {'left': 480, 'right': 1650, 'top': 255, 'bottom': 820}
        else:
            return {'left': 510, 'right': 1280, 'top': 255, 'bottom': 925}

    # Estimate plot boundaries
    # Top is typically y_axis start
    top_y = None
    for vl in v_lines:
        if abs(vl['x'] - y_axis_x) < 50:
            top_y = int(vl['y1'])
            break

    # Right edge
    right_x = None
    for hl in h_lines:
        if abs(hl['y'] - x_axis_y) < 50:
            right_x = int(hl['x2'])
            break

    if top_y is None:
        top_y = x_axis_y - int(h * 0.15)
    if right_x is None:
        right_x = y_axis_x + int(w * 0.3)

    return {
        'left': y_axis_x,
        'right': right_x,
        'top': top_y,
        'bottom': x_axis_y
    }

def detect_data_markers(img_gray, region, channel):
    """
    Detect data point markers using contour analysis.
    CMS uses filled black circles as data markers.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Add small padding to avoid axes
    pad = 10
    roi = img_gray[top+pad:bottom-pad, left+pad:right-pad].copy()
    roi_h, roi_w = roi.shape

    # Threshold to get dark objects
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleaning - remove thin lines (axes, grid)
    # First close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by area - data markers are typically small filled circles
        if area < 30 or area > 5000:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by aspect ratio (should be roughly circular)
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Skip if too close to edges (probably axis labels)
        if x < 30 or x > roi_w - 30:
            continue
        if y < 20 or y > roi_h - 20:
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

            # Only keep reasonably circular blobs
            if circularity > 0.2:
                markers.append({
                    'x': cx + left + pad,
                    'y': cy + top + pad,
                    'area': area,
                    'circularity': circularity
                })

    print(f"  Found {len(markers)} candidate markers")
    return markers

def aggregate_to_histogram(markers, region, config):
    """
    Aggregate detected markers into histogram bins.
    Returns (mass, count) pairs.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    plot_w = right - left
    plot_h = bottom - top

    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']
    bin_width = config['bin_width']

    # Convert markers to physical coordinates
    physical_points = []
    for m in markers:
        # Relative position
        rel_x = (m['x'] - left) / plot_w
        rel_y = (m['y'] - top) / plot_h

        # Physical values
        mass = x_min + rel_x * (x_max - x_min)
        count = y_max - rel_y * (y_max - y_min)

        if count > 0 and x_min <= mass <= x_max:
            physical_points.append((mass, count, m['circularity']))

    # Bin by mass
    bins = defaultdict(list)
    n_bins = int((x_max - x_min) / bin_width)

    for mass, count, circ in physical_points:
        bin_idx = int((mass - x_min) / bin_width)
        bin_idx = max(0, min(bin_idx, n_bins - 1))
        bin_center = x_min + (bin_idx + 0.5) * bin_width
        bins[bin_center].append((count, circ))

    # For each bin, select the best marker (highest circularity = most likely data point)
    result = []
    for bin_center in sorted(bins.keys()):
        points = bins[bin_center]
        if points:
            # Take the point with highest circularity
            best = max(points, key=lambda p: p[1])
            result.append((bin_center, best[0]))

    return result

def process_channel(channel, img_path):
    """Process a single channel's figure."""
    print(f"\n{'='*60}")
    print(f"Channel {channel}: {os.path.basename(img_path)}")
    print('='*60)

    config = AXIS_CONFIG[channel]

    # Load image
    img_color, img_gray = load_image(img_path)
    h, w = img_gray.shape
    print(f"Image dimensions: {w} x {h}")

    # Find plot region
    region = find_plot_region_manual(img_gray, channel)
    print(f"Plot region: L={region['left']}, R={region['right']}, T={region['top']}, B={region['bottom']}")

    plot_w = region['right'] - region['left']
    plot_h = region['bottom'] - region['top']
    print(f"Plot size: {plot_w} x {plot_h} pixels")

    # Detect markers
    markers = detect_data_markers(img_gray, region, channel)

    # Aggregate to histogram
    histogram = aggregate_to_histogram(markers, region, config)
    print(f"Histogram bins: {len(histogram)}")

    # Create debug image
    debug_img = img_color.copy()
    cv2.rectangle(debug_img,
                  (region['left'], region['top']),
                  (region['right'], region['bottom']),
                  (0, 255, 0), 3)

    for m in markers:
        cv2.circle(debug_img, (m['x'], m['y']), 5, (0, 0, 255), -1)

    # Save outputs
    df = pd.DataFrame(histogram, columns=['mass_GeV', 'count'])

    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    debug_path = os.path.join(OUT_DIR, f'debug_{channel}_overlay.png')

    df.to_csv(csv_path, index=False)
    cv2.imwrite(debug_path, debug_img)

    print(f"Saved: {csv_path}")
    print(f"Saved: {debug_path}")

    # Print first few rows
    print(f"\nFirst 10 data points:")
    print(df.head(10).to_string(index=False))

    # Filter to fit window
    fit_min, fit_max = config['fit_window']
    df_fit = df[(df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)]
    print(f"\nPoints in fit window ({fit_min}-{fit_max} GeV): {len(df_fit)}")

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

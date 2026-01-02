#!/usr/bin/env python3
"""
Robust digitization that specifically targets data point markers.
Uses blob detection and shape analysis to distinguish data points from fit curves.
"""

import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Calibrated plot regions
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
        'top': 770,
        'bottom': 1748,
    }
}

# Physical axis ranges
AXIS_CONFIG = {
    'A': {
        'x_min': 6.5, 'x_max': 9.0,
        'y_min': 0, 'y_max': 600,
        'bin_width': 0.025,
        'fit_window': (6.6, 7.5),
    },
    'B': {
        'x_min': 7.0, 'x_max': 9.0,
        'y_min': 0, 'y_max': 30,
        'bin_width': 0.040,
        'fit_window': (6.9, 7.6),
    }
}

def load_image(path):
    img_color = cv2.imread(path)
    if img_color is None:
        raise ValueError(f"Could not load image: {path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def detect_data_points_blob(img_gray, region, channel):
    """
    Use SimpleBlobDetector to find data point markers.
    CMS data points are small, dark, filled circles.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    pad = 20
    roi = img_gray[top+pad:bottom-pad, left+pad:right-pad].copy()
    roi_h, roi_w = roi.shape

    # Setup blob detector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0  # Dark blobs

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 1500

    params.filterByCircularity = True
    params.minCircularity = 0.4

    params.filterByConvexity = True
    params.minConvexity = 0.6

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    detector = cv2.SimpleBlobDetector_create(params)

    # Invert for blob detection
    roi_inv = 255 - roi
    keypoints = detector.detect(roi_inv)

    markers = []
    for kp in keypoints:
        x, y = kp.pt
        if 30 < x < roi_w - 30 and 20 < y < roi_h - 20:
            markers.append({
                'x_roi': int(x),
                'y_roi': int(y),
                'x_abs': int(x) + left + pad,
                'y_abs': int(y) + top + pad,
                'size': kp.size
            })

    return markers, (roi_w, roi_h), pad

def detect_data_points_contour(img_gray, region, channel):
    """
    Alternative detection using contour analysis.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    pad = 20
    roi = img_gray[top+pad:bottom-pad, left+pad:right-pad].copy()
    roi_h, roi_w = roi.shape

    # Binary threshold
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to separate connected blobs
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Skip edge markers
        if x < 30 or x > roi_w - 30:
            continue
        if y < 20 or y > roi_h - 20:
            continue

        # Calculate shape metrics
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0

        aspect = w / max(h, 1)

        # Data points: circular, compact, not too elongated
        if circularity > 0.25 and 0.3 < aspect < 3.0:
            markers.append({
                'x_roi': cx,
                'y_roi': cy,
                'x_abs': cx + left + pad,
                'y_abs': cy + top + pad,
                'area': area,
                'circularity': circularity,
                'aspect': aspect
            })

    return markers, (roi_w, roi_h), pad

def merge_detections(blob_markers, contour_markers):
    """
    Merge detections from both methods.
    """
    all_markers = []

    # Add blob markers
    for m in blob_markers:
        all_markers.append({
            'x_roi': m['x_roi'],
            'y_roi': m['y_roi'],
            'x_abs': m['x_abs'],
            'y_abs': m['y_abs'],
            'score': 1.0  # High score for blob detection
        })

    # Add contour markers if not duplicate
    for cm in contour_markers:
        is_duplicate = False
        for am in all_markers:
            dist = np.sqrt((cm['x_roi'] - am['x_roi'])**2 + (cm['y_roi'] - am['y_roi'])**2)
            if dist < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            all_markers.append({
                'x_roi': cm['x_roi'],
                'y_roi': cm['y_roi'],
                'x_abs': cm['x_abs'],
                'y_abs': cm['y_abs'],
                'score': cm.get('circularity', 0.5)
            })

    return all_markers

def cluster_by_x(markers, bin_width_px):
    """
    Cluster markers by x position.
    """
    if not markers:
        return []

    markers = sorted(markers, key=lambda m: m['x_roi'])

    clusters = []
    current_cluster = [markers[0]]

    for m in markers[1:]:
        if m['x_roi'] - current_cluster[-1]['x_roi'] < bin_width_px * 0.6:
            current_cluster.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = [m]
    clusters.append(current_cluster)

    result = []
    for cluster in clusters:
        best = max(cluster, key=lambda m: m['score'])
        result.append(best)

    return result

def pixels_to_physical(markers, region, config, roi_size, pad):
    """
    Convert to physical coordinates.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    plot_w = right - left - 2 * pad
    plot_h = bottom - top - 2 * pad

    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']

    data = []
    for m in markers:
        rel_x = m['x_roi'] / plot_w
        rel_y = m['y_roi'] / plot_h

        mass = x_min + rel_x * (x_max - x_min)
        count = y_max - rel_y * (y_max - y_min)

        if count > 0 and x_min <= mass <= x_max:
            data.append((mass, count))

    return data

def bin_data(data, config):
    """
    Bin data into histogram format.
    """
    bin_width = config['bin_width']
    x_min, x_max = config['x_min'], config['x_max']

    bins = defaultdict(list)
    for mass, count in data:
        bin_idx = int((mass - x_min) / bin_width)
        bin_center = x_min + (bin_idx + 0.5) * bin_width
        bins[bin_center].append(count)

    result = []
    for bin_center in sorted(bins.keys()):
        if x_min <= bin_center <= x_max:
            counts = bins[bin_center]
            result.append((bin_center, np.median(counts)))

    return result

def process_channel(channel, img_path):
    """Process a channel."""
    print(f"\n{'='*60}")
    print(f"CHANNEL {channel}")
    print('='*60)

    config = AXIS_CONFIG[channel]
    region = PLOT_REGIONS[channel]

    img_color, img_gray = load_image(img_path)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Plot region: L={region['left']}, R={region['right']}, T={region['top']}, B={region['bottom']}")

    # Both detection methods
    blob_markers, roi_size, pad = detect_data_points_blob(img_gray, region, channel)
    print(f"  Blob detector found: {len(blob_markers)} points")

    contour_markers, _, _ = detect_data_points_contour(img_gray, region, channel)
    print(f"  Contour method found: {len(contour_markers)} points")

    # Merge
    all_markers = merge_detections(blob_markers, contour_markers)
    print(f"  Merged: {len(all_markers)} unique points")

    # Cluster
    plot_w = region['right'] - region['left'] - 2 * pad
    x_range = config['x_max'] - config['x_min']
    bin_width_px = (config['bin_width'] / x_range) * plot_w
    print(f"  Bin width: {bin_width_px:.1f} pixels")

    clustered = cluster_by_x(all_markers, bin_width_px)
    print(f"  Clustered: {len(clustered)} bins")

    # Convert to physical
    data = pixels_to_physical(clustered, region, config, roi_size, pad)
    binned = bin_data(data, config)

    # Debug image
    debug_img = img_color.copy()
    cv2.rectangle(debug_img,
                  (region['left'], region['top']),
                  (region['right'], region['bottom']),
                  (0, 255, 0), 3)

    for m in blob_markers:
        cv2.circle(debug_img, (m['x_abs'], m['y_abs']), 6, (255, 0, 0), 2)

    for m in contour_markers:
        cv2.circle(debug_img, (m['x_abs'], m['y_abs']), 4, (0, 255, 255), -1)

    for m in clustered:
        cv2.circle(debug_img, (m['x_abs'], m['y_abs']), 8, (0, 0, 255), 3)

    # Save
    df = pd.DataFrame(binned, columns=['mass_GeV', 'count'])

    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    debug_path = os.path.join(OUT_DIR, f'debug_{channel}_overlay.png')

    df.to_csv(csv_path, index=False)
    cv2.imwrite(debug_path, debug_img)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {debug_path}")

    if len(df) > 0:
        print(f"\nData summary:")
        print(f"  Bins: {len(df)}")
        print(f"  Mass range: {df['mass_GeV'].min():.3f} - {df['mass_GeV'].max():.3f} GeV")
        print(f"  Count range: {df['count'].min():.1f} - {df['count'].max():.1f}")

        fit_min, fit_max = config['fit_window']
        df_fit = df[(df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)]
        print(f"  Points in fit window ({fit_min}-{fit_max} GeV): {len(df_fit)}")

        print(f"\nFirst 20 data points:")
        print(df.head(20).to_string(index=False))

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

#!/usr/bin/env python3
"""
Improved digitization using blob detection for CMS histogram data points.
Focus on extracting actual histogram marker centroids.
"""

import cv2
import numpy as np
import pandas as pd
import os
from scipy import ndimage
from skimage import measure
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Known axis ranges - verified from PDFs
AXIS_CONFIG = {
    'A': {
        'x_min': 6.5, 'x_max': 9.0,  # GeV - Figure 3 shows this range
        'y_min': 0, 'y_max': 600,     # Candidates / 25 MeV
        'bin_width': 0.025,           # 25 MeV bins
        'fit_window': (6.6, 7.5),     # Region of interest for fit
    },
    'B': {
        'x_min': 7.0, 'x_max': 9.0,  # GeV - Figure 2 shows this range
        'y_min': 0, 'y_max': 30,      # Candidates / 40 MeV
        'bin_width': 0.040,           # 40 MeV bins
        'fit_window': (6.9, 7.6),     # Region of interest
    }
}

def load_image(path):
    """Load image in color and grayscale."""
    img_color = cv2.imread(path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def find_plot_box_by_white_region(img_gray, channel):
    """
    Find plot region by detecting the white rectangular plot area.
    CMS figures have a white background for the plot area.
    """
    h, w = img_gray.shape

    # Threshold for white pixels
    _, white_mask = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = np.ones((5,5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular white region in the upper part of the image
    best_rect = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # Must be in upper half of image (plot area)
        if y > h * 0.6:
            continue

        # Must be substantial size
        if area < h * w * 0.02:
            continue

        # Prefer regions that are more rectangular and in the upper portion
        # Score based on area and position
        score = area * (1 - y / h)

        if score > best_score:
            best_score = score
            best_rect = (x, y, x + cw, y + ch)

    if best_rect:
        # Add margin inside for the actual plot area
        margin = 10
        return {
            'left': best_rect[0] + margin,
            'right': best_rect[2] - margin,
            'top': best_rect[1] + margin,
            'bottom': best_rect[3] - margin
        }

    # Fallback
    return {
        'left': int(w * 0.15),
        'right': int(w * 0.85),
        'top': int(h * 0.10),
        'bottom': int(h * 0.40)
    }

def detect_axis_lines(img_gray, initial_region):
    """
    Refine the plot region by finding actual axis lines.
    """
    left, right, top, bottom = initial_region['left'], initial_region['right'], initial_region['top'], initial_region['bottom']
    h, w = img_gray.shape

    # Expand search area
    search_margin = 100
    left_search = max(0, left - search_margin)
    right_search = min(w, right + search_margin)
    top_search = max(0, top - search_margin)
    bottom_search = min(h, bottom + search_margin)

    # Find horizontal line (x-axis) - look for continuous dark pixels
    best_x_axis = bottom
    max_score = 0

    for y in range(top_search, bottom_search):
        row = img_gray[y, left_search:right_search]
        # Count dark pixels in a row
        dark = np.sum(row < 100)
        line_length = right_search - left_search
        if dark > line_length * 0.5:
            # Also check for continuous dark streak
            dark_mask = row < 100
            labeled = measure.label(dark_mask)
            props = measure.regionprops(labeled)
            if props:
                max_len = max(p.area for p in props)
                score = dark * max_len
                if score > max_score:
                    max_score = score
                    best_x_axis = y

    # Find vertical line (y-axis)
    best_y_axis = left
    max_score = 0

    for x in range(left_search, min(left_search + 200, right_search)):
        col = img_gray[top_search:bottom_search, x]
        dark = np.sum(col < 100)
        col_height = bottom_search - top_search
        if dark > col_height * 0.3:
            dark_mask = col < 100
            labeled = measure.label(dark_mask)
            props = measure.regionprops(labeled)
            if props:
                max_len = max(p.area for p in props)
                score = dark * max_len
                if score > max_score:
                    max_score = score
                    best_y_axis = x

    # Find top boundary
    best_top = top
    for y in range(top_search, best_x_axis):
        row = img_gray[y, best_y_axis:right_search]
        dark = np.sum(row < 100)
        if dark > (right_search - best_y_axis) * 0.3:
            best_top = y
            break

    # Find right boundary
    best_right = right
    for x in range(right_search - 1, best_y_axis, -1):
        col = img_gray[best_top:best_x_axis, x]
        dark = np.sum(col < 100)
        if dark > (best_x_axis - best_top) * 0.3:
            best_right = x
            break

    return {
        'left': best_y_axis,
        'right': best_right,
        'top': best_top,
        'bottom': best_x_axis
    }

def detect_blobs_simple(img_gray, region, channel):
    """
    Detect data point markers using SimpleBlobDetector.
    CMS uses filled black circles for data points.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Extract ROI
    roi = img_gray[top:bottom, left:right].copy()

    # Set up blob detector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by color (we want dark blobs)
    params.filterByColor = True
    params.blobColor = 0  # Dark blobs

    # Filter by area
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 2000

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    detector = cv2.SimpleBlobDetector_create(params)

    # Invert the image for blob detection (blobs should be white)
    roi_inv = 255 - roi
    keypoints = detector.detect(roi_inv)

    # Convert keypoints to list of (x, y) in image coordinates
    points = []
    for kp in keypoints:
        x, y = kp.pt
        points.append((int(x) + left, int(y) + top, kp.size))

    return points

def detect_markers_contour(img_gray, region, channel):
    """
    Detect data markers using contour analysis.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Extract ROI with margin
    margin = 5
    roi = img_gray[top+margin:bottom-margin, left+margin:right-margin].copy()
    roi_h, roi_w = roi.shape

    # Adaptive threshold to handle varying background
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 5)

    # Morphological cleaning
    kernel_small = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 3000:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by aspect ratio (markers are roughly circular)
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 3.0:
            continue

        # Filter by position (not too close to edges - avoid axis labels)
        if x < 20 or x > roi_w - 20:
            continue
        if y < 10 or y > roi_h - 10:
            continue

        # Calculate centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Perimeter for circularity
            perim = cv2.arcLength(cnt, True)
            if perim > 0:
                circ = 4 * np.pi * area / (perim * perim)
            else:
                circ = 0

            # Store marker with quality score
            markers.append({
                'x': cx + left + margin,
                'y': cy + top + margin,
                'area': area,
                'circularity': circ,
                'bbox': (x, y, w, h)
            })

    return markers

def cluster_by_x(markers, bin_width_px):
    """
    Cluster markers by x-coordinate to identify histogram bins.
    """
    if not markers:
        return []

    # Sort by x
    markers = sorted(markers, key=lambda m: m['x'])

    # Group by x proximity
    clusters = []
    current_cluster = [markers[0]]

    for m in markers[1:]:
        if m['x'] - current_cluster[-1]['x'] < bin_width_px:
            current_cluster.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = [m]
    clusters.append(current_cluster)

    # For each cluster, find the marker with highest circularity (most likely data point)
    result = []
    for cluster in clusters:
        # Sort by circularity
        best = max(cluster, key=lambda m: m['circularity'])
        result.append(best)

    return result

def pixels_to_data(markers, region, config):
    """
    Convert pixel coordinates to physical values.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    plot_w = right - left
    plot_h = bottom - top

    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']

    data = []
    for m in markers:
        # Relative position in plot
        rel_x = (m['x'] - left) / plot_w
        rel_y = (m['y'] - top) / plot_h

        # Convert to physical
        x_val = x_min + rel_x * (x_max - x_min)
        y_val = y_max - rel_y * (y_max - y_min)  # y is inverted

        if y_val > 0:  # Only keep positive counts
            data.append((x_val, y_val))

    return data

def process_channel(channel, img_path):
    """Process a single channel."""
    print(f"\n{'='*60}")
    print(f"Processing Channel {channel}: {os.path.basename(img_path)}")
    print('='*60)

    config = AXIS_CONFIG[channel]

    # Load image
    img_color, img_gray = load_image(img_path)
    h, w = img_gray.shape
    print(f"Image size: {w} x {h}")

    # Find plot region
    initial_region = find_plot_box_by_white_region(img_gray, channel)
    print(f"Initial white region: {initial_region}")

    # Refine with axis detection
    region = detect_axis_lines(img_gray, initial_region)
    print(f"Refined region: left={region['left']}, right={region['right']}, "
          f"top={region['top']}, bottom={region['bottom']}")

    plot_w = region['right'] - region['left']
    plot_h = region['bottom'] - region['top']
    print(f"Plot dimensions: {plot_w} x {plot_h} pixels")

    # Estimate bin width in pixels
    x_range = config['x_max'] - config['x_min']
    bin_width_px = (config['bin_width'] / x_range) * plot_w
    print(f"Estimated bin width: {bin_width_px:.1f} pixels")

    # Detect markers using multiple methods
    blobs = detect_blobs_simple(img_gray, region, channel)
    print(f"SimpleBlobDetector found: {len(blobs)} blobs")

    contour_markers = detect_markers_contour(img_gray, region, channel)
    print(f"Contour method found: {len(contour_markers)} markers")

    # Use contour markers (usually more reliable)
    markers = contour_markers

    # Cluster by x to get one point per bin
    clustered = cluster_by_x(markers, bin_width_px * 0.8)
    print(f"After clustering: {len(clustered)} bins")

    # Convert to physical values
    data = pixels_to_data(clustered, region, config)

    # Sort by x
    data = sorted(data, key=lambda d: d[0])

    # Create debug image
    debug_img = img_color.copy()
    cv2.rectangle(debug_img,
                  (region['left'], region['top']),
                  (region['right'], region['bottom']),
                  (0, 255, 0), 3)

    for m in clustered:
        cv2.circle(debug_img, (m['x'], m['y']), 8, (0, 0, 255), -1)

    # Also mark blob detections in blue
    for b in blobs:
        cv2.circle(debug_img, (b[0], b[1]), 6, (255, 0, 0), 2)

    # Save outputs
    df = pd.DataFrame(data, columns=['mass_GeV', 'count'])

    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    debug_path = os.path.join(OUT_DIR, f'debug_{channel}_overlay.png')

    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} data points)")

    cv2.imwrite(debug_path, debug_img)
    print(f"Saved: {debug_path}")

    # Print summary
    print(f"\nData summary for Channel {channel}:")
    print(f"  Mass range: {df['mass_GeV'].min():.3f} - {df['mass_GeV'].max():.3f} GeV")
    print(f"  Count range: {df['count'].min():.1f} - {df['count'].max():.1f}")

    # Filter to fit window
    fit_min, fit_max = config['fit_window']
    df_fit = df[(df['mass_GeV'] >= fit_min) & (df['mass_GeV'] <= fit_max)]
    print(f"  Points in fit window ({fit_min}-{fit_max} GeV): {len(df_fit)}")

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig_a_path = os.path.join(DATA_DIR, 'fig_A.png')
    fig_b_path = os.path.join(DATA_DIR, 'fig_B.png')

    df_a = process_channel('A', fig_a_path)
    df_b = process_channel('B', fig_b_path)

    print("\n" + "="*60)
    print("DIGITIZATION SUMMARY")
    print("="*60)

    print("\nChannel A (J/ψ J/ψ) - first 15 points:")
    print(df_a.head(15).to_string(index=False))

    print("\nChannel B (J/ψ ψ(2S)) - first 15 points:")
    print(df_b.head(15).to_string(index=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Digitize data points from CMS figure images using computer vision.
Automatically detect plot region and extract data markers.
"""

import cv2
import numpy as np
import pandas as pd
import os
from scipy import ndimage
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Known axis ranges from the PAS documents
AXIS_CONFIG = {
    'A': {
        'x_min': 6.5, 'x_max': 9.0,  # GeV
        'y_min': 0, 'y_max': 600,     # Candidates / 25 MeV
        'bin_width': 0.025,           # 25 MeV bins
    },
    'B': {
        'x_min': 7.0, 'x_max': 9.0,  # GeV
        'y_min': 0, 'y_max': 30,      # Candidates / 40 MeV
        'bin_width': 0.040,           # 40 MeV bins
    }
}

def load_image(path):
    """Load image in color and grayscale."""
    img_color = cv2.imread(path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def find_plot_region(img_gray, channel):
    """
    Find the main plotting region by detecting axis lines.
    Uses Hough transform to find long horizontal and vertical lines.
    """
    h, w = img_gray.shape

    # Threshold to get near-black pixels (axis lines)
    _, binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Use edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        print(f"Warning: No lines detected for channel {channel}")
        # Fall back to manual estimation based on typical CMS figure layout
        # Figures typically have plot in upper portion of page
        return estimate_plot_region(img_gray, channel)

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((y1, y2, x1, x2, length))
        elif 80 < angle < 100:  # Vertical
            v_lines.append((x1, x2, y1, y2, length))

    # Find the longest lines in each direction
    if len(h_lines) > 0 and len(v_lines) > 0:
        # Sort by length
        h_lines.sort(key=lambda x: -x[4])
        v_lines.sort(key=lambda x: -x[4])

        # Look for x-axis (long horizontal line, not at very top or bottom)
        x_axis_y = None
        for hl in h_lines:
            y_pos = (hl[0] + hl[1]) / 2
            if h * 0.2 < y_pos < h * 0.8:  # Middle region
                x_axis_y = int(y_pos)
                x_axis_x_range = (min(hl[2], hl[3]), max(hl[2], hl[3]))
                break

        # Look for y-axis (long vertical line on left side)
        y_axis_x = None
        for vl in v_lines:
            x_pos = (vl[0] + vl[1]) / 2
            if x_pos < w * 0.5:  # Left half
                y_axis_x = int(x_pos)
                y_axis_y_range = (min(vl[2], vl[3]), max(vl[2], vl[3]))
                break

        if x_axis_y and y_axis_x:
            # Find plot top (look for horizontal line above x-axis)
            top_y = None
            for hl in h_lines:
                y_pos = (hl[0] + hl[1]) / 2
                if y_pos < x_axis_y - 100 and y_pos > h * 0.05:
                    if top_y is None or y_pos < top_y:
                        top_y = int(y_pos)

            # Find plot right edge
            right_x = None
            for vl in v_lines:
                x_pos = (vl[0] + vl[1]) / 2
                if x_pos > y_axis_x + 100:
                    if right_x is None or x_pos > right_x:
                        right_x = int(x_pos)

            if top_y is None:
                top_y = y_axis_y_range[0] if y_axis_y_range else int(h * 0.15)
            if right_x is None:
                right_x = x_axis_x_range[1] if x_axis_x_range else int(w * 0.85)

            return {
                'left': y_axis_x,
                'right': right_x,
                'top': top_y,
                'bottom': x_axis_y
            }

    return estimate_plot_region(img_gray, channel)

def estimate_plot_region(img_gray, channel):
    """
    Estimate plot region using intensity profile analysis.
    """
    h, w = img_gray.shape

    # For CMS PAS figures, the plot is typically in the upper portion
    # Look for regions with more variation (the data plot area)

    # Scan for white background region (the plot area has white background)
    _, white_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours of white regions
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest white rectangular region (likely the plot area)
    best_rect = None
    best_area = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # Must be substantial size and somewhat rectangular
        if area > h * w * 0.05 and cw > w * 0.3 and ch > h * 0.1:
            if area > best_area:
                best_area = area
                best_rect = (x, y, x + cw, y + ch)

    if best_rect:
        return {
            'left': best_rect[0],
            'right': best_rect[2],
            'top': best_rect[1],
            'bottom': best_rect[3]
        }

    # Fallback to typical CMS figure proportions
    return {
        'left': int(w * 0.12),
        'right': int(w * 0.88),
        'top': int(h * 0.08),
        'bottom': int(h * 0.42)
    }

def refine_plot_region(img_gray, region, channel):
    """
    Refine the plot region by looking for actual axis lines near the estimate.
    """
    h, w = img_gray.shape
    margin = 50

    # Extract region around estimated borders
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Look for dark horizontal line (x-axis) near bottom
    y_search_start = max(0, bottom - margin)
    y_search_end = min(h, bottom + margin)

    best_x_axis = bottom
    max_dark_pixels = 0

    for y in range(y_search_start, y_search_end):
        row = img_gray[y, left:right]
        dark_pixels = np.sum(row < 100)
        if dark_pixels > max_dark_pixels and dark_pixels > (right - left) * 0.3:
            max_dark_pixels = dark_pixels
            best_x_axis = y

    # Look for dark vertical line (y-axis) near left
    x_search_start = max(0, left - margin)
    x_search_end = min(w, left + margin)

    best_y_axis = left
    max_dark_pixels = 0

    for x in range(x_search_start, x_search_end):
        col = img_gray[top:bottom, x]
        dark_pixels = np.sum(col < 100)
        if dark_pixels > max_dark_pixels and dark_pixels > (bottom - top) * 0.3:
            max_dark_pixels = dark_pixels
            best_y_axis = x

    # Update region
    refined = {
        'left': best_y_axis,
        'right': right,
        'top': top,
        'bottom': best_x_axis
    }

    print(f"Channel {channel}: Refined region: left={refined['left']}, right={refined['right']}, "
          f"top={refined['top']}, bottom={refined['bottom']}")

    return refined

def detect_data_points(img_color, region, channel):
    """
    Detect data points (markers) within the plot region.
    CMS uses filled black circles for data points.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Add small margin inside the axes
    margin = 5
    left += margin
    top += margin
    right -= margin
    bottom -= margin

    # Extract plot region
    plot_roi = img_color[top:bottom, left:right].copy()
    plot_gray = cv2.cvtColor(plot_roi, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark objects (data points are black)
    _, binary = cv2.threshold(plot_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find data point markers
    # Data points are small, roughly circular blobs
    points = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area < 5:  # Too small (noise)
            continue

        # Circularity check
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0

        # Get bounding rect for aspect ratio
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / max(h_box, 1)

        # Data point markers: moderate size, roughly circular or square
        # Typical data markers are 5-50 pixels in area at this resolution
        if 10 < area < 5000 and circularity > 0.3 and 0.3 < aspect_ratio < 3.0:
            # Get centroid
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Exclude points too close to edges (likely axis labels)
                plot_h, plot_w = plot_roi.shape[:2]
                if 10 < cx < plot_w - 10 and 10 < cy < plot_h - 10:
                    points.append((cx, cy, area, circularity))

    # Sort by x coordinate
    points.sort(key=lambda p: p[0])

    # Convert pixel coordinates to plot-relative coordinates
    plot_h, plot_w = plot_roi.shape[:2]

    # Create debug image
    debug_img = img_color.copy()
    cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)

    pixel_points = []
    for (cx, cy, area, circ) in points:
        # Convert to absolute image coordinates
        abs_x = cx + left
        abs_y = cy + top
        pixel_points.append((abs_x, abs_y, cx, cy))
        cv2.circle(debug_img, (abs_x, abs_y), 5, (0, 0, 255), -1)

    print(f"Channel {channel}: Detected {len(pixel_points)} potential data points")

    return pixel_points, debug_img, (plot_w, plot_h)

def pixel_to_data(pixel_points, plot_size, region, config):
    """
    Convert pixel coordinates to physical data values.
    """
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']
    plot_w, plot_h = plot_size

    x_min, x_max = config['x_min'], config['x_max']
    y_min, y_max = config['y_min'], config['y_max']

    data_points = []

    for (abs_x, abs_y, rel_x, rel_y) in pixel_points:
        # x: left edge is x_min, right edge is x_max
        x_frac = rel_x / plot_w
        x_val = x_min + x_frac * (x_max - x_min)

        # y: top edge is y_max, bottom edge is y_min (inverted)
        y_frac = 1.0 - (rel_y / plot_h)
        y_val = y_min + y_frac * (y_max - y_min)

        data_points.append((x_val, y_val))

    return data_points

def bin_data_points(data_points, config):
    """
    Aggregate data points into histogram bins.
    """
    bin_width = config['bin_width']
    x_min, x_max = config['x_min'], config['x_max']

    # Create bins
    bins = defaultdict(list)

    for x, y in data_points:
        if x_min <= x <= x_max:
            bin_idx = int((x - x_min) / bin_width)
            bin_center = x_min + (bin_idx + 0.5) * bin_width
            bins[bin_center].append(y)

    # Average y values in each bin
    binned_data = []
    for bin_center in sorted(bins.keys()):
        y_values = bins[bin_center]
        # Take the median or mean
        y_avg = np.median(y_values)
        binned_data.append((bin_center, y_avg))

    return binned_data

def process_channel(channel, img_path):
    """Process a single channel's figure."""
    print(f"\n{'='*50}")
    print(f"Processing Channel {channel}: {img_path}")
    print('='*50)

    config = AXIS_CONFIG[channel]

    # Load image
    img_color, img_gray = load_image(img_path)
    print(f"Image size: {img_gray.shape[1]} x {img_gray.shape[0]}")

    # Find plot region
    region = find_plot_region(img_gray, channel)
    print(f"Initial region estimate: {region}")

    # Refine region
    region = refine_plot_region(img_gray, region, channel)

    # Detect data points
    pixel_points, debug_img, plot_size = detect_data_points(img_color, region, channel)

    # Convert to physical values
    data_points = pixel_to_data(pixel_points, plot_size, region, config)

    # Bin the data
    binned_data = bin_data_points(data_points, config)

    print(f"Binned data points: {len(binned_data)}")

    # Save outputs
    csv_path = os.path.join(OUT_DIR, f'digitized_{channel}.csv')
    debug_path = os.path.join(OUT_DIR, f'debug_{channel}_overlay.png')

    # Save CSV
    df = pd.DataFrame(binned_data, columns=['mass_GeV', 'count'])
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save debug image
    cv2.imwrite(debug_path, debug_img)
    print(f"Saved: {debug_path}")

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Process both channels
    fig_a_path = os.path.join(DATA_DIR, 'fig_A.png')
    fig_b_path = os.path.join(DATA_DIR, 'fig_B.png')

    df_a = process_channel('A', fig_a_path)
    df_b = process_channel('B', fig_b_path)

    print("\n" + "="*50)
    print("DIGITIZATION COMPLETE")
    print("="*50)
    print(f"\nChannel A (J/ψ J/ψ): {len(df_a)} bins")
    print(df_a.head(10))
    print(f"\nChannel B (J/ψ ψ(2S)): {len(df_b)} bins")
    print(df_b.head(10))

if __name__ == "__main__":
    main()

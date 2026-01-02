#!/usr/bin/env python3
"""
Extract Channel B data from CMS-PAS-BPH-22-004 Figure 2.
Tries vector PDF extraction first, then falls back to pixel digitization.
"""

import os
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Known axis ranges from PAS (Figure 2 shows combined Run2+Run3)
# x: 7.0 to 9.0 GeV
# y: -3 to 30 (approximately, with pull panel below)
AXIS_CONFIG = {
    'x_min': 7.0,
    'x_max': 9.0,
    'y_min': 0,
    'y_max': 30,
    'bin_width': 0.040,  # 40 MeV bins
}

def try_pdf_vector_extraction():
    """
    Attempt to extract data points directly from PDF vector paths.
    """
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not available, skipping PDF vector extraction")
        return None

    pdf_path = os.path.join(DATA_DIR, 'cds', 'fig_B.pdf')

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]

            # Get page dimensions
            width = page.width
            height = page.height
            print(f"PDF page size: {width} x {height}")

            # Try to extract curves/paths
            curves = page.curves if hasattr(page, 'curves') else []
            lines = page.lines if hasattr(page, 'lines') else []

            print(f"Found {len(curves)} curves, {len(lines)} lines")

            # Look for point-like objects (small circles or markers)
            chars = page.chars
            rects = page.rects if hasattr(page, 'rects') else []

            print(f"Found {len(chars)} chars, {len(rects)} rects")

            # PDF vector extraction is complex for CMS plots
            # They typically use embedded images or complex paths
            # Return None to fall back to pixel extraction
            return None

    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return None

def pixel_digitization(jitter_x=0, jitter_y=0):
    """
    Extract data points from PNG using computer vision.
    jitter_x, jitter_y: pixel offsets for systematic studies
    """
    png_path = os.path.join(DATA_DIR, 'cds', 'fig_B.png')

    img = cv2.imread(png_path)
    if img is None:
        raise ValueError(f"Could not load image: {png_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print(f"Image size: {w} x {h}")

    # Detect plot region using edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    # Identify horizontal and vertical lines
    h_lines = []
    v_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if angle < 5:  # Horizontal
                h_lines.append({'y': (y1+y2)/2, 'x1': min(x1,x2), 'x2': max(x1,x2), 'len': length})
            elif angle > 85:  # Vertical
                v_lines.append({'x': (x1+x2)/2, 'y1': min(y1,y2), 'y2': max(y1,y2), 'len': length})

    # Sort by position
    h_lines.sort(key=lambda l: l['y'])
    v_lines.sort(key=lambda l: l['x'])

    # Find main plot axes
    # Top of main plot: first significant horizontal line
    # Bottom of main plot (x-axis): before the pull panel
    # Left axis: leftmost significant vertical line

    # Filter for long lines
    h_lines_long = [l for l in h_lines if l['len'] > w * 0.3]
    v_lines_long = [l for l in v_lines if l['len'] > h * 0.1]

    print(f"Long horizontal lines: {len(h_lines_long)}")
    print(f"Long vertical lines: {len(v_lines_long)}")

    # Estimate plot region
    # For Figure 2, the main plot is typically in the upper portion
    if len(h_lines_long) >= 2 and len(v_lines_long) >= 1:
        # Find lines that form the plot box
        # Sort by y position
        h_sorted = sorted(h_lines_long, key=lambda l: l['y'])

        # The x-axis is typically a prominent horizontal line
        # Look for one around y ~ 0.4-0.6 of image height
        x_axis_candidates = [l for l in h_sorted if h*0.3 < l['y'] < h*0.7]
        if x_axis_candidates:
            x_axis = x_axis_candidates[0]
        else:
            x_axis = h_sorted[len(h_sorted)//2]

        # Top of plot
        top_candidates = [l for l in h_sorted if l['y'] < x_axis['y'] - 50]
        if top_candidates:
            top_line = top_candidates[0]
        else:
            top_line = {'y': x_axis['y'] - h * 0.3}

        # Left axis
        left_axis = v_lines_long[0]

        # Right edge
        right_candidates = [l for l in v_lines_long if l['x'] > left_axis['x'] + 100]
        if right_candidates:
            right_line = right_candidates[-1]
        else:
            right_line = {'x': left_axis['x'] + w * 0.7}

        plot_region = {
            'left': int(left_axis['x']) + jitter_x,
            'right': int(right_line['x']) + jitter_x,
            'top': int(top_line['y']) + jitter_y,
            'bottom': int(x_axis['y']) + jitter_y
        }
    else:
        # Fallback to manual estimate
        plot_region = {
            'left': int(w * 0.15) + jitter_x,
            'right': int(w * 0.92) + jitter_x,
            'top': int(h * 0.08) + jitter_y,
            'bottom': int(h * 0.45) + jitter_y
        }

    print(f"Plot region: {plot_region}")

    # Extract data points using blob detection
    left, right, top, bottom = plot_region['left'], plot_region['right'], plot_region['top'], plot_region['bottom']

    # Ensure valid region
    left = max(0, min(left, w-10))
    right = max(left+10, min(right, w))
    top = max(0, min(top, h-10))
    bottom = max(top+10, min(bottom, h))

    pad = 10
    roi = gray[top+pad:bottom-pad, left+pad:right-pad].copy()
    roi_h, roi_w = roi.shape

    if roi_h < 10 or roi_w < 10:
        print("Warning: ROI too small")
        return None, None

    # Threshold to find dark objects (data markers)
    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 2000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Skip edge markers
        if x < 20 or x > roi_w - 20:
            continue
        if y < 10 or y > roi_h - 10:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

            if circularity > 0.2:
                markers.append({
                    'x': cx,
                    'y': cy,
                    'area': area,
                    'circ': circularity
                })

    print(f"Found {len(markers)} markers")

    # Convert to physical coordinates
    plot_w = right - left - 2*pad
    plot_h = bottom - top - 2*pad

    x_min, x_max = AXIS_CONFIG['x_min'], AXIS_CONFIG['x_max']
    y_min, y_max = AXIS_CONFIG['y_min'], AXIS_CONFIG['y_max']

    data_points = []
    for m in markers:
        rel_x = m['x'] / plot_w
        rel_y = m['y'] / plot_h

        mass = x_min + rel_x * (x_max - x_min)
        count = y_max - rel_y * (y_max - y_min)

        if count > 0 and x_min <= mass <= x_max:
            data_points.append((mass, count, m['circ']))

    # Bin the data
    bin_width = AXIS_CONFIG['bin_width']
    bins = defaultdict(list)

    for mass, count, circ in data_points:
        bin_idx = int((mass - x_min) / bin_width)
        bin_center = x_min + (bin_idx + 0.5) * bin_width
        bins[bin_center].append((count, circ))

    # Take best marker per bin
    result = []
    for bin_center in sorted(bins.keys()):
        points = bins[bin_center]
        if points:
            # Take highest circularity point
            best = max(points, key=lambda p: p[1])
            count = best[0]
            # Poisson error
            sigma = np.sqrt(max(count, 1))
            result.append((bin_center, count, sigma))

    # Create debug image
    debug_img = img.copy()
    cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)
    for m in markers:
        abs_x = m['x'] + left + pad
        abs_y = m['y'] + top + pad
        cv2.circle(debug_img, (abs_x, abs_y), 5, (0, 0, 255), -1)

    return result, debug_img

def main():
    print("="*60)
    print("Channel B Data Extraction")
    print("="*60)

    # Try PDF vector extraction first
    print("\n--- Attempting PDF vector extraction ---")
    pdf_data = try_pdf_vector_extraction()

    if pdf_data is not None:
        print("PDF vector extraction successful")
        data = pdf_data
    else:
        print("PDF extraction failed, using pixel digitization")
        data, debug_img = pixel_digitization()

        if data is None:
            print("ERROR: Could not extract data")
            return

        # Save debug image
        debug_path = os.path.join(OUT_DIR, 'debug_B_overlay.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved: {debug_path}")

    # Create DataFrame
    df = pd.DataFrame(data, columns=['mass_center_GeV', 'count', 'sigma_count'])

    # Save
    output_path = os.path.join(DATA_DIR, 'derived', 'channel_B_digitized.csv')
    df.to_csv(output_path, index=False)

    print(f"\nChannel B spectrum saved: {output_path}")
    print(f"Total bins: {len(df)}")
    print(f"Mass range: {df['mass_center_GeV'].min():.3f} - {df['mass_center_GeV'].max():.3f} GeV")
    print(f"Total counts: {df['count'].sum():.1f}")

    # Show data
    print(f"\nExtracted data:")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract Channel B data from CMS-PAS-BPH-22-004 Figure 2.
Renders PDF at high DPI and performs pixel digitization.
"""

import os
import numpy as np
import pandas as pd
import cv2
import fitz  # PyMuPDF
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

# Known axis ranges from PAS Figure 2
# The combined plot shows 7.0-9.0 GeV on x-axis, 0-30 on y-axis
AXIS_CONFIG = {
    'x_min': 7.0,
    'x_max': 9.0,
    'y_min': 0,
    'y_max': 30,
    'bin_width': 0.040,  # 40 MeV bins
}

def render_pdf_high_res(pdf_path, dpi=600):
    """Render PDF page at high DPI."""
    doc = fitz.open(pdf_path)

    # Check pages
    print(f"PDF has {len(doc)} pages")

    # Render first page (Figure 2)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Convert to numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Convert RGBA to BGR for OpenCV
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()

    print(f"Rendered at {dpi} DPI: {img.shape[1]} x {img.shape[0]} pixels")

    # Save rendered image
    render_path = os.path.join(DATA_DIR, 'cds', 'fig_B_highres.png')
    cv2.imwrite(render_path, img)
    print(f"Saved: {render_path}")

    return img

def find_plot_region(gray):
    """Find the main plot region using line detection."""
    h, w = gray.shape

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                            minLineLength=w//5, maxLineGap=20)

    if lines is None:
        print("No lines detected, using default region")
        return {
            'left': int(w * 0.12),
            'right': int(w * 0.95),
            'top': int(h * 0.05),
            'bottom': int(h * 0.70)
        }

    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if angle < 5:
            h_lines.append({'y': (y1+y2)/2, 'x1': min(x1,x2), 'x2': max(x1,x2), 'len': length})
        elif angle > 85:
            v_lines.append({'x': (x1+x2)/2, 'y1': min(y1,y2), 'y2': max(y1,y2), 'len': length})

    print(f"Found {len(h_lines)} horizontal, {len(v_lines)} vertical lines")

    # Sort by position
    h_lines.sort(key=lambda l: l['y'])
    v_lines.sort(key=lambda l: l['x'])

    # Find main plot boundaries
    # Look for the x-axis (major horizontal line in upper half)
    x_axis_y = None
    for hl in h_lines:
        if h * 0.3 < hl['y'] < h * 0.8 and hl['len'] > w * 0.5:
            x_axis_y = int(hl['y'])
            break

    # Look for y-axis (leftmost long vertical)
    y_axis_x = None
    for vl in v_lines:
        if vl['len'] > h * 0.2 and vl['x'] < w * 0.3:
            y_axis_x = int(vl['x'])
            break

    # Find top boundary
    top_y = None
    for hl in h_lines:
        if hl['y'] < (x_axis_y or h*0.5) - 50 and hl['len'] > w * 0.3:
            top_y = int(hl['y'])
            break

    # Find right boundary
    right_x = None
    for vl in reversed(v_lines):
        if vl['x'] > (y_axis_x or 0) + 100 and vl['len'] > h * 0.1:
            right_x = int(vl['x'])
            break

    # Defaults if not found
    if x_axis_y is None:
        x_axis_y = int(h * 0.65)
    if y_axis_x is None:
        y_axis_x = int(w * 0.12)
    if top_y is None:
        top_y = int(h * 0.05)
    if right_x is None:
        right_x = int(w * 0.95)

    region = {
        'left': y_axis_x,
        'right': right_x,
        'top': top_y,
        'bottom': x_axis_y
    }

    print(f"Plot region: left={region['left']}, right={region['right']}, "
          f"top={region['top']}, bottom={region['bottom']}")

    return region

def detect_data_markers(gray, region, min_area=50, max_area=3000):
    """Detect data point markers using blob detection."""
    left, right, top, bottom = region['left'], region['right'], region['top'], region['bottom']

    # Add padding to avoid axes
    pad = 15
    left = left + pad
    right = right - pad
    top = top + pad
    bottom = bottom - pad

    roi = gray[top:bottom, left:right].copy()
    roi_h, roi_w = roi.shape

    print(f"ROI size: {roi_w} x {roi_h}")

    # Threshold for dark objects
    _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Skip edge objects
        if x < 20 or x > roi_w - 20:
            continue
        if y < 15 or y > roi_h - 15:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

            # Aspect ratio
            aspect = cw / max(ch, 1)

            # Data markers: compact, roughly circular
            if circ > 0.15 and 0.2 < aspect < 5:
                markers.append({
                    'x': cx,
                    'y': cy,
                    'x_abs': cx + left,
                    'y_abs': cy + top,
                    'area': area,
                    'circ': circ
                })

    print(f"Found {len(markers)} candidate markers")

    return markers, (roi_w, roi_h), (left, top)

def markers_to_data(markers, roi_size, offset, region):
    """Convert pixel markers to physical (mass, count) values."""
    roi_w, roi_h = roi_size
    offset_x, offset_y = offset

    x_min, x_max = AXIS_CONFIG['x_min'], AXIS_CONFIG['x_max']
    y_min, y_max = AXIS_CONFIG['y_min'], AXIS_CONFIG['y_max']

    data = []
    for m in markers:
        rel_x = m['x'] / roi_w
        rel_y = m['y'] / roi_h

        mass = x_min + rel_x * (x_max - x_min)
        count = y_max - rel_y * (y_max - y_min)

        if count > 0 and x_min <= mass <= x_max:
            data.append({
                'mass': mass,
                'count': count,
                'circ': m['circ'],
                'x_abs': m['x_abs'],
                'y_abs': m['y_abs']
            })

    return data

def bin_data(data, bin_width=0.040):
    """Aggregate data into histogram bins."""
    x_min = AXIS_CONFIG['x_min']

    bins = defaultdict(list)
    for d in data:
        bin_idx = int((d['mass'] - x_min) / bin_width)
        bin_center = x_min + (bin_idx + 0.5) * bin_width
        bins[bin_center].append(d)

    result = []
    for bin_center in sorted(bins.keys()):
        points = bins[bin_center]
        # Take highest circularity marker as best estimate
        best = max(points, key=lambda p: p['circ'])
        count = best['count']
        sigma = np.sqrt(max(count, 1))
        result.append({
            'mass_center_GeV': bin_center,
            'count': count,
            'sigma_count': sigma,
            'x_abs': best['x_abs'],
            'y_abs': best['y_abs']
        })

    return result

def main():
    print("="*60)
    print("Channel B Data Extraction (High-Resolution)")
    print("="*60)

    # Render PDF at high resolution
    pdf_path = os.path.join(DATA_DIR, 'cds', 'fig_B.pdf')
    img = render_pdf_high_res(pdf_path, dpi=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find plot region
    region = find_plot_region(gray)

    # Detect markers
    markers, roi_size, offset = detect_data_markers(gray, region)

    if len(markers) == 0:
        print("ERROR: No markers found!")
        return None

    # Convert to physical values
    data = markers_to_data(markers, roi_size, offset, region)

    # Bin the data
    binned = bin_data(data)

    # Create DataFrame
    df = pd.DataFrame(binned)

    # Create debug image
    debug_img = img.copy()
    cv2.rectangle(debug_img,
                  (region['left'], region['top']),
                  (region['right'], region['bottom']),
                  (0, 255, 0), 3)

    for row in binned:
        cv2.circle(debug_img, (int(row['x_abs']), int(row['y_abs'])), 8, (0, 0, 255), 2)

    debug_path = os.path.join(OUT_DIR, 'debug_B_overlay.png')
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image saved: {debug_path}")

    # Save data
    output_path = os.path.join(DATA_DIR, 'derived', 'channel_B_digitized.csv')
    df[['mass_center_GeV', 'count', 'sigma_count']].to_csv(output_path, index=False)

    print(f"\nChannel B spectrum saved: {output_path}")
    print(f"Total bins: {len(df)}")
    print(f"Mass range: {df['mass_center_GeV'].min():.3f} - {df['mass_center_GeV'].max():.3f} GeV")
    print(f"Total counts: {df['count'].sum():.1f}")

    print(f"\nExtracted data:")
    print(df[['mass_center_GeV', 'count', 'sigma_count']].to_string(index=False))

    return df

if __name__ == "__main__":
    main()

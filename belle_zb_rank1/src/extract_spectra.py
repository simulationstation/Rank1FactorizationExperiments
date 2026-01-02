#!/usr/bin/env python3
"""
Vector-first extraction of Belle Zb mass spectra from arXiv:1110.2251.

Strategy:
1. Try vector extraction via PyMuPDF drawing primitives
2. Fall back to pixel-based extraction if vector fails
3. Use known axis ranges from paper to calibrate

Target spectra (from Figure 2 and Figure 3, page 4):
- Υ(1S)π: M range [10.1, 10.8] GeV, panels (a)
- Υ(2S)π: M range [10.4, 10.75] GeV, panel (c)
- Υ(3S)π: M range [10.58, 10.74] GeV, panel (e)
- hb(1P)π: Mmiss range [10.4, 10.7] GeV, Figure 3(a)
- hb(2P)π: Mmiss range [10.4, 10.7] GeV, Figure 3(b)
"""

import fitz
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json

# Known axis ranges from the paper (in GeV)
CHANNEL_INFO = {
    'upsilon1s': {
        'name': 'Υ(1S)π',
        'x_range': (10.1, 10.8),
        'y_label': 'Events/10 MeV',
        'bin_width': 0.010,  # GeV
        'fig': '2a'
    },
    'upsilon2s': {
        'name': 'Υ(2S)π',
        'x_range': (10.4, 10.75),
        'y_label': 'Events/5 MeV',
        'bin_width': 0.005,
        'fig': '2c'
    },
    'upsilon3s': {
        'name': 'Υ(3S)π',
        'x_range': (10.58, 10.74),
        'y_label': 'Events/4 MeV',
        'bin_width': 0.004,
        'fig': '2e'
    },
    'hb1p': {
        'name': 'hb(1P)π',
        'x_range': (10.4, 10.7),
        'y_label': 'Events/10 MeV',
        'bin_width': 0.010,
        'fig': '3a'
    },
    'hb2p': {
        'name': 'hb(2P)π',
        'x_range': (10.4, 10.7),
        'y_label': 'Events/10 MeV',
        'bin_width': 0.010,
        'fig': '3b'
    }
}

# Approximate figure panel bounding boxes on page 4 (in points, 72 dpi)
# These are estimated from the layout - will refine based on drawings
PANEL_BOXES = {
    'upsilon1s': (36, 72, 200, 200),     # Fig 2a - top left
    'upsilon2s': (36, 230, 200, 360),    # Fig 2c - middle left
    'upsilon3s': (36, 390, 200, 520),    # Fig 2e - bottom left
    'hb1p': (310, 72, 480, 260),         # Fig 3a - top right area
    'hb2p': (310, 280, 480, 460),        # Fig 3b - bottom right area
}


def extract_drawings_in_region(page, bbox):
    """Extract drawing elements within a bounding box."""
    x0, y0, x1, y1 = bbox
    drawings = page.get_drawings()

    elements = []
    for d in drawings:
        # Check if drawing overlaps with bbox
        drect = d.get('rect')
        if drect:
            if (drect.x0 < x1 and drect.x1 > x0 and
                drect.y0 < y1 and drect.y1 > y0):
                elements.append(d)
    return elements


def find_data_points_from_drawings(drawings, bbox, x_range, y_range_est=(0, 150)):
    """
    Extract data points from vector drawing elements.
    Look for circles, rectangles, or line segments that represent data points.
    """
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0

    points = []
    errors = []

    for d in drawings:
        items = d.get('items', [])
        rect = d.get('rect')

        if not rect:
            continue

        # Check for small rectangles or circles (data points)
        w = rect.width
        h = rect.height

        # Data points are typically small markers
        if 2 < w < 15 and 2 < h < 15:
            cx = (rect.x0 + rect.x1) / 2
            cy = (rect.y0 + rect.y1) / 2

            # Convert to data coordinates
            x_frac = (cx - x0) / box_width
            y_frac = 1 - (cy - y0) / box_height  # Invert y

            if 0 <= x_frac <= 1 and 0 <= y_frac <= 1:
                x_data = x_range[0] + x_frac * (x_range[1] - x_range[0])
                y_data = y_range_est[0] + y_frac * (y_range_est[1] - y_range_est[0])
                points.append((x_data, y_data, cx, cy))

        # Check for vertical lines (error bars)
        for item in items:
            if item[0] == 'l':  # line
                p1, p2 = item[1], item[2]
                # Vertical line check
                if abs(p1.x - p2.x) < 2:  # Nearly vertical
                    cx = (p1.x + p2.x) / 2
                    err_size = abs(p2.y - p1.y)
                    errors.append((cx, err_size))

    return points, errors


def extract_from_pixel_image(img_path, channel, crop_box=None):
    """
    Pixel-based extraction as fallback.
    Uses image processing to find data points.
    """
    from PIL import Image
    import cv2

    img = cv2.imread(img_path)
    if crop_box:
        # Scale crop box from 72 dpi to 600 dpi
        scale = 600 / 72
        x0, y0, x1, y1 = [int(v * scale) for v in crop_box]
        img = img[y0:y1, x0:x1]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find dark points (data markers are typically black)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:  # Data point size range
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                points.append((cx, cy))

    return points


def digitize_from_table_values():
    """
    Use Table I values from the paper to create synthetic spectra
    for fitting, as the exact bin-by-bin data is not available.

    This is a last-resort approach using the published fit parameters.
    """
    # From Table I in the paper
    table_data = {
        'upsilon1s': {
            'M1': 10.611, 'G1': 0.0223, 'M2': 10.657, 'G2': 0.0163,
            'rel_norm': 0.57, 'rel_phase': 58,  # degrees
            'n_events_approx': 350
        },
        'upsilon2s': {
            'M1': 10.609, 'G1': 0.0242, 'M2': 10.651, 'G2': 0.0133,
            'rel_norm': 0.86, 'rel_phase': -13,
            'n_events_approx': 500
        },
        'upsilon3s': {
            'M1': 10.608, 'G1': 0.0176, 'M2': 10.652, 'G2': 0.0084,
            'rel_norm': 0.96, 'rel_phase': -9,
            'n_events_approx': 550
        },
        'hb1p': {
            'M1': 10.605, 'G1': 0.0114, 'M2': 10.654, 'G2': 0.0209,
            'rel_norm': 1.39, 'rel_phase': 187,
            'n_events_approx': 8000  # Much higher yield
        },
        'hb2p': {
            'M1': 10.599, 'G1': 0.013, 'M2': 10.651, 'G2': 0.019,
            'rel_norm': 1.6, 'rel_phase': 181,
            'n_events_approx': 12000
        }
    }
    return table_data


def read_points_from_figure_manually():
    """
    Manually digitized points from the figures.
    Read from visual inspection of the 600 DPI renders.

    This function will be populated after visual inspection.
    """
    # These values need to be extracted from the figures
    # For now, return placeholder indicating manual digitization needed
    return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, 'data/papers/belle_zb_1110.2251.pdf')
    fig_dir = os.path.join(base_dir, 'data/figures')
    extracted_dir = os.path.join(base_dir, 'extracted')
    out_dir = os.path.join(base_dir, 'out')

    os.makedirs(extracted_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[3]  # Page 4 (0-indexed)

    print("=" * 60)
    print("Belle Zb Spectrum Extraction")
    print("=" * 60)

    # Get all drawings on the page
    all_drawings = page.get_drawings()
    print(f"\nTotal drawings on page 4: {len(all_drawings)}")

    # Analyze drawing types
    drawing_types = defaultdict(int)
    for d in all_drawings:
        for item in d.get('items', []):
            drawing_types[item[0]] += 1

    print("\nDrawing element types:")
    for dtype, count in sorted(drawing_types.items()):
        print(f"  {dtype}: {count}")

    # Try to extract bounding boxes of figure regions by finding
    # clusters of drawings
    rects = []
    for d in all_drawings:
        r = d.get('rect')
        if r:
            rects.append((r.x0, r.y0, r.x1, r.y1))

    if rects:
        rects = np.array(rects)
        print(f"\nDrawing bounds: x=[{rects[:,0].min():.1f}, {rects[:,2].max():.1f}], "
              f"y=[{rects[:,1].min():.1f}, {rects[:,3].max():.1f}]")

    # Since vector extraction is complex for this PDF,
    # let's use the published Table I parameters to understand the spectra
    print("\n" + "=" * 60)
    print("Using Table I parameters for spectrum reconstruction")
    print("=" * 60)

    table_data = digitize_from_table_values()

    for channel, params in table_data.items():
        info = CHANNEL_INFO[channel]
        print(f"\n{info['name']} ({info['fig']}):")
        print(f"  Zb(10610): M={params['M1']:.3f} GeV, Γ={params['G1']*1000:.1f} MeV")
        print(f"  Zb(10650): M={params['M2']:.3f} GeV, Γ={params['G2']*1000:.1f} MeV")
        print(f"  Rel. norm (aZ2/aZ1): {params['rel_norm']:.2f}")
        print(f"  Rel. phase: {params['rel_phase']}°")

    doc.close()

    # Save extraction metadata
    extraction_info = {
        'method': 'table_parameters',
        'source': 'arXiv:1110.2251 Table I',
        'channels': list(CHANNEL_INFO.keys()),
        'table_data': table_data,
        'note': 'Bin-by-bin data requires manual digitization from figures'
    }

    with open(os.path.join(out_dir, 'extraction_info.json'), 'w') as f:
        json.dump(extraction_info, f, indent=2)

    print(f"\nExtraction info saved to: {out_dir}/extraction_info.json")
    print("\nNext step: Manual digitization of figure points or use Table I parameters for analysis")


if __name__ == '__main__':
    main()

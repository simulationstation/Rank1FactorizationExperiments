#!/usr/bin/env python3
"""
Deeper exploration of PDF vector content - analyze lines for histogram structure.
"""

import os
import fitz
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def analyze_lines(pdf_path):
    """Analyze line segments to find histogram bars."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    print(f"Page rect: {page.rect}")

    paths = page.get_drawings()
    print(f"Total paths: {len(paths)}")

    # Collect all line segments
    all_lines = []
    for path in paths:
        items = path.get('items', [])
        color = path.get('color')
        fill = path.get('fill')
        width = path.get('width', 1)

        for item in items:
            if item[0] == 'l':  # line
                p1, p2 = item[1], item[2]
                all_lines.append({
                    'x1': p1.x, 'y1': p1.y,
                    'x2': p2.x, 'y2': p2.y,
                    'color': color,
                    'fill': fill,
                    'width': width
                })

    print(f"Total line segments: {len(all_lines)}")

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    for L in all_lines:
        dx = abs(L['x2'] - L['x1'])
        dy = abs(L['y2'] - L['y1'])
        if dy < 1 and dx > 1:  # horizontal
            h_lines.append(L)
        elif dx < 1 and dy > 1:  # vertical
            v_lines.append(L)

    print(f"Horizontal lines: {len(h_lines)}")
    print(f"Vertical lines: {len(v_lines)}")

    # Look for histogram structure in different page regions
    # The axis labels suggest:
    # - y-axis counts 0-25 around x=41-276 (left side)
    # - x-axis mass 7-9 GeV around y=7-395

    # Get text for axis mapping
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
                            'y': (bbox[1] + bbox[3]) / 2,
                            'bbox': bbox
                        })

    # Find x-axis tick positions (mass values 7, 7.5, 8, 8.5, 9)
    mass_ticks = []
    for t in text_items:
        try:
            val = float(t['text'])
            if 6.5 <= val <= 9.5:  # mass range
                mass_ticks.append({'value': val, 'x': t['x'], 'y': t['y']})
        except:
            pass

    # Find y-axis tick positions (count values 0, 5, 10, 15, 20, 25)
    count_ticks = []
    for t in text_items:
        try:
            val = float(t['text'])
            if val in [0, 5, 10, 15, 20, 25, 30]:
                # These should be on the left side
                if t['x'] < 300:  # left half of page
                    count_ticks.append({'value': val, 'x': t['x'], 'y': t['y']})
        except:
            pass

    print(f"\nMass ticks found:")
    for mt in sorted(mass_ticks, key=lambda x: x['value']):
        print(f"  {mt['value']} GeV at ({mt['x']:.1f}, {mt['y']:.1f})")

    print(f"\nCount ticks found:")
    for ct in sorted(count_ticks, key=lambda x: x['value']):
        print(f"  {ct['value']} at ({ct['x']:.1f}, {ct['y']:.1f})")

    # Analyze horizontal lines in plot region
    # Look for lines that could be histogram bar tops
    # Group horizontal lines by y position
    y_groups = defaultdict(list)
    for L in h_lines:
        y_mid = (L['y1'] + L['y2']) / 2
        y_groups[round(y_mid, 1)].append(L)

    print(f"\nHorizontal lines grouped by y position: {len(y_groups)} groups")

    # Look at vertical lines - these should be bin edges
    # Group by x position
    x_groups = defaultdict(list)
    for L in v_lines:
        x_mid = (L['x1'] + L['x2']) / 2
        x_groups[round(x_mid, 1)].append(L)

    print(f"Vertical lines grouped by x position: {len(x_groups)} groups")

    # Find the plot area by looking for a rectangular frame
    # Look for long horizontal and vertical lines
    long_h = [L for L in h_lines if abs(L['x2'] - L['x1']) > 100]
    long_v = [L for L in v_lines if abs(L['y2'] - L['y1']) > 100]

    print(f"\nLong horizontal lines (>100pts): {len(long_h)}")
    for L in long_h[:5]:
        print(f"  y={L['y1']:.1f}, x={L['x1']:.1f}->{L['x2']:.1f}")

    print(f"\nLong vertical lines (>100pts): {len(long_v)}")
    for L in long_v[:5]:
        print(f"  x={L['x1']:.1f}, y={L['y1']:.1f}->{L['y2']:.1f}")

    # Find data points - look for markers (circles, squares)
    # These might be in the "other" paths category
    print("\n=== Analyzing other paths (curves/markers) ===")

    circles = []
    for path in paths:
        items = path.get('items', [])
        rect = path.get('rect')
        fill = path.get('fill')

        # Circles/ellipses often have 'c' (curve) items
        has_curves = any(item[0] == 'c' for item in items)

        if has_curves and rect:
            w, h = rect.width, rect.height
            # Small, roughly square = marker
            if 2 < w < 20 and 2 < h < 20 and 0.5 < w/h < 2:
                cx = (rect.x0 + rect.x1) / 2
                cy = (rect.y0 + rect.y1) / 2
                circles.append({
                    'cx': cx, 'cy': cy,
                    'w': w, 'h': h,
                    'fill': fill
                })

    print(f"Found {len(circles)} potential circular markers")

    if circles:
        # Sort by x position
        circles.sort(key=lambda c: c['cx'])
        print("\nMarkers (sorted by x):")
        for c in circles[:30]:
            print(f"  ({c['cx']:.1f}, {c['cy']:.1f}), size={c['w']:.1f}x{c['h']:.1f}, fill={c['fill']}")

    doc.close()

    return {
        'mass_ticks': mass_ticks,
        'count_ticks': count_ticks,
        'circles': circles,
        'h_lines': h_lines,
        'v_lines': v_lines
    }


if __name__ == "__main__":
    pdf_path = os.path.join(DATA_DIR, 'cds', 'fig_B.pdf')
    result = analyze_lines(pdf_path)

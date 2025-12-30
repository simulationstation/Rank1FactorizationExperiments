#!/usr/bin/env python3
"""
Explore ATLAS PDF figures to understand vector structure.
"""

import os
import fitz  # PyMuPDF
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

def explore_pdf(pdf_path, name):
    """Explore vector content of an ATLAS figure PDF."""
    print(f"\n{'='*60}")
    print(f"Exploring: {name}")
    print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    page = doc[0]

    print(f"Page size: {page.rect.width:.1f} x {page.rect.height:.1f} pts")

    # Get all drawings (vector paths)
    paths = page.get_drawings()
    print(f"Total paths: {len(paths)}")

    # Categorize paths
    rects = []
    lines_h = []
    lines_v = []
    curves = []

    for path in paths:
        items = path.get('items', [])
        rect = path.get('rect')
        fill = path.get('fill')
        stroke = path.get('color')

        has_curve = any(item[0] == 'c' for item in items)

        if rect:
            w, h = rect.width, rect.height

            # Histogram bars are typically narrow rectangles
            if h > 2 and w > 0.5 and w < 20:
                rects.append({
                    'x0': rect.x0, 'y0': rect.y0,
                    'x1': rect.x1, 'y1': rect.y1,
                    'width': w, 'height': h,
                    'fill': fill, 'stroke': stroke
                })

        # Analyze lines
        for item in items:
            if item[0] == 'l':
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                if dy < 1 and dx > 5:
                    lines_h.append({'y': (p1.y + p2.y)/2, 'x1': min(p1.x, p2.x), 'x2': max(p1.x, p2.x)})
                elif dx < 1 and dy > 5:
                    lines_v.append({'x': (p1.x + p2.x)/2, 'y1': min(p1.y, p2.y), 'y2': max(p1.y, p2.y)})

        if has_curve:
            if rect and 2 < rect.width < 15 and 2 < rect.height < 15:
                curves.append({
                    'cx': (rect.x0 + rect.x1) / 2,
                    'cy': (rect.y0 + rect.y1) / 2,
                    'w': rect.width, 'h': rect.height,
                    'fill': fill
                })

    print(f"Rectangles (potential histogram bars): {len(rects)}")
    print(f"Horizontal lines: {len(lines_h)}")
    print(f"Vertical lines: {len(lines_v)}")
    print(f"Curved paths (markers): {len(curves)}")

    # Show sample rectangles
    if rects:
        rects.sort(key=lambda r: r['x0'])
        print(f"\nSample rectangles (sorted by x):")
        for r in rects[:15]:
            print(f"  x0={r['x0']:.1f}, y0={r['y0']:.1f}, w={r['width']:.1f}, h={r['height']:.1f}, fill={r['fill']}")

    # Show sample curves (data points)
    if curves:
        curves.sort(key=lambda c: c['cx'])
        print(f"\nSample data markers (sorted by x):")
        for c in curves[:15]:
            print(f"  ({c['cx']:.1f}, {c['cy']:.1f}), size={c['w']:.1f}x{c['h']:.1f}, fill={c['fill']}")

    # Extract text for axis labels
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
                            'y': (bbox[1] + bbox[3]) / 2
                        })

    # Find numeric labels
    numbers = []
    for t in text_items:
        try:
            val = float(t['text'].replace('−', '-'))
            numbers.append({**t, 'value': val})
        except:
            pass

    print(f"\nText items: {len(text_items)}")
    print(f"Numeric labels: {len(numbers)}")

    # Show numbers sorted by position
    numbers.sort(key=lambda n: (n['y'], n['x']))
    print(f"\nNumeric labels (top to bottom):")
    for n in numbers[:20]:
        print(f"  '{n['text']}' = {n['value']} at ({n['x']:.1f}, {n['y']:.1f})")

    doc.close()

    return {
        'rects': rects,
        'curves': curves,
        'numbers': numbers,
        'lines_h': lines_h,
        'lines_v': lines_v
    }


if __name__ == "__main__":
    figures = [
        ('fig_01a.pdf', 'di-J/psi spectrum'),
        ('fig_01b.pdf', 'J/psi+psi(2S) 4mu'),
        ('fig_01c.pdf', 'J/psi+psi(2S) 4mu+2pi'),
    ]

    results = {}
    for fname, desc in figures:
        pdf_path = os.path.join(DATA_DIR, 'atlas_figs', fname)
        if os.path.exists(pdf_path):
            results[fname] = explore_pdf(pdf_path, desc)
        else:
            print(f"File not found: {pdf_path}")

    # Save exploration results
    with open(os.path.join(OUT_DIR, 'atlas_pdf_exploration.json'), 'w') as f:
        json.dump({
            k: {
                'n_rects': len(v['rects']),
                'n_curves': len(v['curves']),
                'n_numbers': len(v['numbers']),
            } for k, v in results.items()
        }, f, indent=2)

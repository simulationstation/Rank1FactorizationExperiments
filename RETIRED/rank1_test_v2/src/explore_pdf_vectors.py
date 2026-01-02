#!/usr/bin/env python3
"""
Explore vector content of Channel B PDF to understand structure.
"""

import os
import fitz  # PyMuPDF
import json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def explore_pdf_vectors(pdf_path):
    """Extract and analyze vector content from PDF."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    print(f"Page size: {page.rect.width} x {page.rect.height} pts")
    print(f"MediaBox: {page.mediabox}")

    # Get all drawings (vector paths)
    paths = page.get_drawings()
    print(f"\nTotal paths: {len(paths)}")

    # Categorize paths
    rects = []
    lines = []
    curves = []
    other = []

    for i, path in enumerate(paths):
        items = path.get('items', [])
        fill = path.get('fill')
        stroke = path.get('color')
        rect = path.get('rect')

        # Check what type of path
        is_rect = False
        is_line = False

        if len(items) >= 4:
            # Check if it's a rectangle (4 lines forming closed path)
            ops = [item[0] for item in items]
            if ops.count('l') >= 3:  # lines
                is_rect = True

        if len(items) == 1 and items[0][0] == 'l':
            is_line = True

        if is_rect and rect:
            w = rect.width
            h = rect.height
            # Histogram bars are tall narrow rectangles
            if h > 5 and w > 1 and w < 50:
                rects.append({
                    'idx': i,
                    'rect': rect,
                    'fill': fill,
                    'stroke': stroke,
                    'width': w,
                    'height': h,
                    'x0': rect.x0,
                    'y0': rect.y0,
                    'x1': rect.x1,
                    'y1': rect.y1
                })
        elif is_line:
            lines.append({'idx': i, 'items': items, 'stroke': stroke})
        else:
            other.append({'idx': i, 'items': items[:3], 'fill': fill, 'stroke': stroke})

    print(f"\nRectangles (potential histogram bars): {len(rects)}")
    print(f"Lines: {len(lines)}")
    print(f"Other paths: {len(other)}")

    # Show some rectangles
    if rects:
        print("\n=== Sample rectangles ===")
        # Sort by x position
        rects.sort(key=lambda r: r['x0'])
        for r in rects[:20]:
            print(f"  x0={r['x0']:.1f}, y0={r['y0']:.1f}, w={r['width']:.1f}, h={r['height']:.1f}, fill={r['fill']}")

    # Look for text (axis labels)
    print("\n=== Text elements ===")
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
                            'x': bbox[0],
                            'y': bbox[1],
                            'x1': bbox[2],
                            'y1': bbox[3]
                        })

    # Find axis tick labels (numbers)
    numbers = []
    for t in text_items:
        try:
            val = float(t['text'])
            numbers.append({**t, 'value': val})
        except:
            pass

    print(f"Total text items: {len(text_items)}")
    print(f"Numeric labels: {len(numbers)}")

    # Sort numbers by position
    numbers.sort(key=lambda n: (n['y'], n['x']))
    print("\nNumeric labels (potential axis ticks):")
    for n in numbers[:30]:
        print(f"  '{n['text']}' at ({n['x']:.1f}, {n['y']:.1f}) = {n['value']}")

    doc.close()

    return {
        'rects': rects,
        'lines': lines,
        'numbers': numbers,
        'text_items': text_items
    }

if __name__ == "__main__":
    pdf_path = os.path.join(DATA_DIR, 'cds', 'fig_B.pdf')
    result = explore_pdf_vectors(pdf_path)

    # Save for analysis
    output = {
        'n_rects': len(result['rects']),
        'n_numbers': len(result['numbers']),
        'rects': [{'x0': r['x0'], 'y0': r['y0'], 'x1': r['x1'], 'y1': r['y1'],
                   'width': r['width'], 'height': r['height']} for r in result['rects']],
        'numbers': result['numbers']
    }

    with open(os.path.join(BASE_DIR, 'out', 'pdf_vector_analysis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\nSaved analysis to out/pdf_vector_analysis.json")

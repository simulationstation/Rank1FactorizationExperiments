#!/usr/bin/env python3
"""
Render Belle Zb paper figures at 600 DPI for extraction.
Target: Page 4 containing Figure 2 and Figure 3.
"""

import fitz  # PyMuPDF
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, 'data/papers/belle_zb_1110.2251.pdf')
    fig_dir = os.path.join(base_dir, 'data/figures')

    doc = fitz.open(pdf_path)

    # Page 4 (0-indexed as page 3) contains Figure 2 and Figure 3
    target_pages = [3]  # Page 4 in 1-indexed

    dpi = 600
    zoom = dpi / 72  # 72 is default PDF resolution
    mat = fitz.Matrix(zoom, zoom)

    for page_num in target_pages:
        page = doc[page_num]

        # Render full page
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(fig_dir, f'page{page_num+1}_600dpi.png')
        pix.save(out_path)
        print(f"Saved: {out_path} ({pix.width}x{pix.height} px)")

        # Get page dimensions for cropping reference
        rect = page.rect
        print(f"Page {page_num+1} dimensions: {rect.width:.1f} x {rect.height:.1f} pts")

        # Extract images/drawings info
        print(f"\nPage {page_num+1} drawings/paths:")
        drawings = page.get_drawings()
        print(f"  Found {len(drawings)} drawing elements")

        # Look for text to help identify figure boundaries
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])
        print(f"  Found {len(blocks)} text/image blocks")

    doc.close()

    # Also render page 5 (Table I with parameters)
    doc = fitz.open(pdf_path)
    page = doc[4]  # Page 5
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(fig_dir, 'page5_600dpi.png')
    pix.save(out_path)
    print(f"Saved: {out_path}")
    doc.close()

    print("\nRendering complete!")

if __name__ == '__main__':
    main()

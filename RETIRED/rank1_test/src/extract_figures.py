#!/usr/bin/env python3
"""
Extract specific figures from CMS PAS PDFs.
- Figure 3 from BPH-24-003-pas.pdf (J/psi J/psi spectrum)
- Figure 2 from BPH-22-004-pas.pdf (J/psi psi(2S) spectrum)
"""

import fitz  # PyMuPDF
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def find_page_with_text(doc, search_text):
    """Find the page index containing the given text."""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if search_text in text:
            return page_num
    return None

def render_page_to_png(doc, page_num, output_path, dpi=450):
    """Render a page to PNG at given DPI."""
    page = doc[page_num]
    # Calculate zoom factor for desired DPI (default PDF is 72 dpi)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    print(f"Saved: {output_path} (page {page_num+1}, {pix.width}x{pix.height} px, {dpi} DPI)")
    return pix.width, pix.height

def main():
    # Process BPH-24-003 (Channel A - J/psi J/psi)
    pdf_a_path = os.path.join(DATA_DIR, 'BPH-24-003-pas.pdf')
    doc_a = fitz.open(pdf_a_path)

    # Search for "Figure 3:" which is the J/psi J/psi invariant mass spectrum
    page_a = find_page_with_text(doc_a, "Figure 3:")
    if page_a is None:
        # Try alternative search
        page_a = find_page_with_text(doc_a, "Figure 3")

    if page_a is not None:
        print(f"BPH-24-003: Found Figure 3 on page {page_a+1}")
        render_page_to_png(doc_a, page_a, os.path.join(DATA_DIR, 'fig_A.png'), dpi=450)
    else:
        print("ERROR: Could not find Figure 3 in BPH-24-003")
        # List all pages with figures
        for i in range(len(doc_a)):
            text = doc_a[i].get_text()
            if 'Figure' in text:
                figures = [line for line in text.split('\n') if 'Figure' in line][:2]
                print(f"  Page {i+1}: {figures}")

    doc_a.close()

    # Process BPH-22-004 (Channel B - J/psi psi(2S))
    pdf_b_path = os.path.join(DATA_DIR, 'BPH-22-004-pas.pdf')
    doc_b = fitz.open(pdf_b_path)

    # Search for "Figure 2:" which is the J/psi psi(2S) spectrum
    page_b = find_page_with_text(doc_b, "Figure 2:")
    if page_b is None:
        page_b = find_page_with_text(doc_b, "Figure 2")

    if page_b is not None:
        print(f"BPH-22-004: Found Figure 2 on page {page_b+1}")
        render_page_to_png(doc_b, page_b, os.path.join(DATA_DIR, 'fig_B.png'), dpi=450)
    else:
        print("ERROR: Could not find Figure 2 in BPH-22-004")
        for i in range(len(doc_b)):
            text = doc_b[i].get_text()
            if 'Figure' in text:
                figures = [line for line in text.split('\n') if 'Figure' in line][:2]
                print(f"  Page {i+1}: {figures}")

    doc_b.close()

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()

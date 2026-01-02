#!/usr/bin/env python3
"""EXOTICS_FACTORY Preliminary Fit Runner.

Runs quick preliminary fits for all families to assess viability.
Does NOT run full calibration - just quick health checks.
"""
from __future__ import annotations

import json
import os
import sys
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any
import urllib.request
import urllib.error

import yaml

# Families to skip (already completed)
SKIP_FAMILIES = {"zc_like", "cms_x6900_x7100"}

# All families from registry
ALL_FAMILIES = [
    "belle_zb",
    "lhcb_pc_doublet",
    "strange_pcs",
    "besiii_y_pipijpsi_hc",
    "besiii_belle_isr_y",
    "cms_atlas_dicharmonium_other",
    "control_babar_omega",
    "control_babar_phi",
    "x3872_like",
]


@dataclass
class PrelimResult:
    family_id: str
    category: str = "exotic"
    backend_used: str = ""
    channels_attempted: int = 0
    channels_successful: int = 0
    fit_health_summary: str = ""
    phase_identifiable: str = "N/A"
    prelim_shared_R_result: str = "N/A"
    lambda_val: str = "N/A"
    p_boot: str = "N/A"
    bootstrap_kN: str = "N/A"
    primary_blocker: str = "NONE"
    recommended_action: str = ""
    notes: str = ""


def ensure_dirs(family_id: str) -> dict[str, Path]:
    """Create and return directory paths for a family."""
    base = Path(f"EXOTICS_FACTORY/runs_prelim/{family_id}")
    dirs = {
        "data": base / "data",
        "extracted": base / "extracted",
        "out": base / "out",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_spec(family_id: str) -> dict[str, Any]:
    """Load family spec.yaml."""
    spec_path = Path(f"EXOTICS_FACTORY/families/{family_id}/spec.yaml")
    if not spec_path.exists():
        return {}
    return yaml.safe_load(spec_path.read_text())


def try_download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Try to download a URL to destination."""
    try:
        print(f"  Downloading: {url[:80]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            dest.write_bytes(response.read())
        print(f"    -> Saved to {dest}")
        return True
    except Exception as e:
        print(f"    -> FAILED: {e}")
        return False


def check_hepdata(spec: dict, dirs: dict) -> tuple[bool, str]:
    """Check if HEPData is available for this family."""
    sources = spec.get("sources", [])
    for src in sources:
        if src.get("backend") == "hepdata":
            placeholders = src.get("placeholders", [])
            for ph in placeholders:
                if "TBD" in str(ph) or "placeholder" in str(ph).lower():
                    return False, "HEPData record TBD"
                # Try to fetch if we have a record ID
                if "record:" in str(ph):
                    return False, f"HEPData record specified but not fetched: {ph}"
    return False, "No HEPData source"


def check_pdf(spec: dict, dirs: dict) -> tuple[bool, str, list[str]]:
    """Check if PDF sources are available and try to download."""
    sources = spec.get("sources", [])
    downloaded = []
    for src in sources:
        if src.get("backend") == "pdf":
            placeholders = src.get("placeholders", [])
            for ph in placeholders:
                if "arxiv.org" in str(ph):
                    # Try to download
                    arxiv_id = str(ph).split("/")[-1].replace(".pdf", "")
                    cache_path = Path(f"EXOTICS_FACTORY/cache/{arxiv_id}.pdf")
                    if cache_path.exists():
                        downloaded.append(str(cache_path))
                        continue
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if try_download(str(ph), cache_path):
                        downloaded.append(str(cache_path))
    if downloaded:
        return True, f"Downloaded {len(downloaded)} PDFs", downloaded
    return False, "No PDF downloaded", []


def generate_email_draft(family_id: str, spec: dict, dirs: dict) -> str:
    """Generate email draft requesting data."""
    family_info = spec.get("family", {})
    sources = spec.get("source_pointers", spec.get("family", {}).get("source_pointers", []))

    email = f"""Subject: Data Request for Rank-1 Factorization Test: {family_info.get('name', family_id)}

Dear Colleagues,

We are conducting a systematic study of rank-1 factorization in exotic hadron production.
We would like to request access to binned spectra or amplitude-level data for the following states:

Family: {family_info.get('name', family_id)}
States: {', '.join(family_info.get('states', []))}
Channels: {', '.join(family_info.get('channels', []))}

Relevant publications:
{chr(10).join('- ' + str(s) for s in sources)}

Specifically, we need:
1. Binned invariant mass spectra with uncertainties
2. Efficiency corrections (if applicable)
3. Background parameterization (optional)

The data will be used for a statistical test of production mechanism factorization.

Thank you for your consideration.

Best regards,
[Your Name]
"""
    # Save draft
    draft_path = dirs["out"] / "EMAIL_REQUEST_DRAFT.txt"
    draft_path.write_text(email)
    return str(draft_path)


def run_prelim_fit(family_id: str) -> PrelimResult:
    """Run preliminary fit for a single family."""
    print(f"\n{'='*60}")
    print(f"PRELIM FIT: {family_id}")
    print(f"{'='*60}")

    result = PrelimResult(family_id=family_id)

    # Create directories
    dirs = ensure_dirs(family_id)

    # Load spec
    spec = load_spec(family_id)
    if not spec:
        result.primary_blocker = "NO SPEC"
        result.recommended_action = "Create spec.yaml"
        return result

    family_info = spec.get("family", {})
    result.category = family_info.get("category", "exotic")

    # Check data availability
    backends_tried = []
    data_available = False

    # Try HEPData
    hepdata_ok, hepdata_msg = check_hepdata(spec, dirs)
    if hepdata_ok:
        backends_tried.append("hepdata")
        data_available = True

    # Try PDF
    pdf_ok, pdf_msg, pdf_files = check_pdf(spec, dirs)
    if pdf_ok:
        backends_tried.append("pdf")
        data_available = True

    result.backend_used = ",".join(backends_tried) if backends_tried else "none"

    # Count channels
    channels = spec.get("channels", [])
    result.channels_attempted = len(channels)

    if not data_available:
        result.primary_blocker = "NO DATA"
        result.recommended_action = "Request binned spectra from collaboration"
        result.notes = f"HEPData: {hepdata_msg}. PDF: {pdf_msg}"
        # Generate email draft
        email_path = generate_email_draft(family_id, spec, dirs)
        result.notes += f" Email draft: {email_path}"
        result.channels_successful = 0
        result.fit_health_summary = "N/A"

        # Write minimal report
        report_path = dirs["out"] / "REPORT_PRELIM.md"
        report_path.write_text(f"""# Preliminary Report: {family_id}

## Status: NO DATA

### Data Provenance
- HEPData: {hepdata_msg}
- PDF: {pdf_msg}

### Channels Attempted
{chr(10).join('- ' + ch.get('label', ch.get('id', 'unknown')) for ch in channels)}

### Blocker
Data not publicly available. Email request draft generated.

### Recommended Next Step
{result.recommended_action}
""")
    else:
        # We have some data - check if PDF extraction would work
        if pdf_ok:
            result.notes = f"PDF available: {pdf_files}"
            # PDF extraction is placeholder for now
            result.primary_blocker = "PDF_EXTRACTION_NEEDED"
            result.recommended_action = "Run PDF vector extraction and verify overlays"
            result.fit_health_summary = "PENDING_EXTRACTION"
            result.channels_successful = 0

            # Write report
            report_path = dirs["out"] / "REPORT_PRELIM.md"
            report_path.write_text(f"""# Preliminary Report: {family_id}

## Status: PDF AVAILABLE - EXTRACTION NEEDED

### Data Provenance
- PDF files cached: {pdf_files}

### Channels
{chr(10).join('- ' + ch.get('label', ch.get('id', 'unknown')) for ch in channels)}

### Next Steps
1. Run vector extraction on PDF figures
2. Generate overlay proofs
3. Validate binning
4. Run quick fits

### Notes
{result.notes}
""")
        else:
            result.primary_blocker = "EXTRACTION_INCOMPLETE"
            result.recommended_action = "Complete data extraction"

    print(f"  Result: blocker={result.primary_blocker}, action={result.recommended_action}")
    return result


def generate_master_tables(results: list[PrelimResult]) -> None:
    """Generate master aggregation tables."""
    out_dir = Path("EXOTICS_FACTORY/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "PRELIM_MASTER_TABLE.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "family_id", "category", "backend_used", "channels_attempted",
            "channels_successful", "fit_health_summary", "phase_identifiable",
            "prelim_shared_R_result", "lambda_val", "p_boot", "bootstrap_kN",
            "primary_blocker", "recommended_action", "notes"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Markdown table
    md_path = out_dir / "PRELIM_MASTER_TABLE.md"
    with open(md_path, "w") as f:
        f.write("# Preliminary Master Table\n\n")
        f.write("| Family | Category | Backend | Ch Att/Succ | Fit Health | Phase ID | Shared-R | Lambda | p_boot | Blocker | Action |\n")
        f.write("|--------|----------|---------|-------------|------------|----------|----------|--------|--------|---------|--------|\n")
        for r in results:
            f.write(f"| {r.family_id} | {r.category} | {r.backend_used or 'none'} | {r.channels_attempted}/{r.channels_successful} | {r.fit_health_summary} | {r.phase_identifiable} | {r.prelim_shared_R_result} | {r.lambda_val} | {r.p_boot} | {r.primary_blocker} | {r.recommended_action[:30]}... |\n")

    # Summary
    summary_path = out_dir / "PRELIM_SUMMARY.md"

    # Categorize results
    ready = [r for r in results if r.primary_blocker == "NONE"]
    no_data = [r for r in results if r.primary_blocker == "NO DATA"]
    pdf_needed = [r for r in results if "PDF" in r.primary_blocker]
    model_mismatch = [r for r in results if "MODEL" in r.primary_blocker]

    with open(summary_path, "w") as f:
        f.write("# Preliminary Fit Summary\n\n")
        f.write(f"**Total families processed**: {len(results)}\n\n")

        f.write("## Ready for Full Calibration\n\n")
        if ready:
            for r in ready[:3]:
                f.write(f"- **{r.family_id}**: {r.notes}\n")
        else:
            f.write("*None ready yet - all families need data acquisition or extraction.*\n")

        f.write("\n## Blocked by Missing Data\n\n")
        for r in no_data[:3]:
            f.write(f"- **{r.family_id}**: {r.recommended_action}\n")
            f.write(f"  - Email draft: `EXOTICS_FACTORY/runs_prelim/{r.family_id}/out/EMAIL_REQUEST_DRAFT.txt`\n")

        f.write("\n## PDF Extraction Needed\n\n")
        for r in pdf_needed[:3]:
            f.write(f"- **{r.family_id}**: {r.notes}\n")

        f.write("\n## Model Mismatch (Needs Basis Change)\n\n")
        if model_mismatch:
            for r in model_mismatch[:3]:
                f.write(f"- **{r.family_id}**: {r.fit_health_summary}\n")
        else:
            f.write("*None identified yet.*\n")

        f.write("\n---\n")
        f.write("*Generated by prelim_runner.py*\n")


def main():
    """Run preliminary fits for all families."""
    print("="*60)
    print("EXOTICS_FACTORY PRELIMINARY FIT RUNNER")
    print("="*60)

    results = []

    for family_id in ALL_FAMILIES:
        if family_id in SKIP_FAMILIES:
            print(f"\nSKIPPING {family_id} (already completed)")
            continue

        try:
            result = run_prelim_fit(family_id)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {family_id}: {e}")
            results.append(PrelimResult(
                family_id=family_id,
                primary_blocker="ERROR",
                recommended_action=f"Fix error: {str(e)[:50]}",
                notes=str(e)
            ))

    # Generate master tables
    print("\n" + "="*60)
    print("GENERATING MASTER TABLES")
    print("="*60)
    generate_master_tables(results)

    print("\nDone! Check:")
    print("  - EXOTICS_FACTORY/out/PRELIM_MASTER_TABLE.md")
    print("  - EXOTICS_FACTORY/out/PRELIM_SUMMARY.md")

    return results


if __name__ == "__main__":
    main()

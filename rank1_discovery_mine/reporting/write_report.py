"""
Report generation module.

Generates REPORT.md and related documents for each candidate,
including "blocked but useful" reports when data is unavailable.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def write_candidate_report(
    candidate_dir: Path,
    candidate: "Candidate",
    status: str,
    search_results: Optional[Dict] = None,
    extraction_results: Optional[Dict] = None,
    rank1_results: Optional[Dict] = None,
) -> Path:
    """
    Write REPORT.md for a candidate.

    Generates useful output even when blocked.
    """
    candidate_dir = Path(candidate_dir)
    out_dir = candidate_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "REPORT.md"

    # Load meta.json if available
    meta_path = candidate_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    # Generate report content
    lines = []

    # Header
    lines.append(f"# Rank-1 Test Report: {candidate.title}")
    lines.append("")
    lines.append(f"**Slug:** `{candidate.slug}`")
    lines.append(f"**Status:** {status}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # States and channels
    lines.append("## Target States")
    lines.append("")
    for state in candidate.states:
        lines.append(f"- {state}")
    lines.append("")

    lines.append("## Channels")
    lines.append("")
    for ch in candidate.channels:
        lines.append(f"- **{ch.id}**: {ch.label}")
        if ch.notes:
            lines.append(f"  - Notes: {ch.notes}")
    lines.append("")

    # Data search section
    lines.append("## Data Search")
    lines.append("")

    planned_queries = meta.get("planned_queries", [])
    if planned_queries:
        lines.append("### Planned Queries")
        lines.append("")
        lines.append("| Source | Query |")
        lines.append("|--------|-------|")
        for pq in planned_queries:
            lines.append(f"| {pq.get('source', 'N/A')} | `{pq.get('query', 'N/A')[:80]}...` |")
        lines.append("")

    discovered_urls = meta.get("discovered_urls", [])
    if discovered_urls:
        lines.append("### Discovered Data Sources")
        lines.append("")
        for url_info in discovered_urls:
            if isinstance(url_info, dict):
                lines.append(f"- [{url_info.get('type', 'link')}]({url_info.get('url', '')})")
                if url_info.get('title'):
                    lines.append(f"  - Title: {url_info['title']}")
            else:
                lines.append(f"- {url_info}")
        lines.append("")
    else:
        lines.append("*No data sources discovered.*")
        lines.append("")

    # Extraction section
    lines.append("## Data Extraction")
    lines.append("")

    extracted_dir = candidate_dir / "extracted"
    if extracted_dir.exists():
        extracted_files = list(extracted_dir.glob("*.csv"))
        if extracted_files:
            lines.append("### Extracted Files")
            lines.append("")
            for ef in extracted_files:
                lines.append(f"- `{ef.name}`")
            lines.append("")
        else:
            lines.append("*No CSV files extracted.*")
            lines.append("")
    else:
        lines.append("*Extraction not performed.*")
        lines.append("")

    # Rank-1 results section
    lines.append("## Rank-1 Test Results")
    lines.append("")

    if rank1_results:
        lines.append(f"**Verdict:** {rank1_results.get('verdict', 'N/A')}")
        lines.append(f"**p-value (bootstrap):** {rank1_results.get('p_boot', 'N/A')}")
        lines.append(f"**Test statistic Λ:** {rank1_results.get('Lambda', 'N/A')}")
        lines.append("")
    else:
        lines.append("*Rank-1 test not performed.*")
        lines.append("")

    # Blocker classification
    if status in ["NO_DATA", "BLOCKED", "ERROR"]:
        lines.append("## Blocker Analysis")
        lines.append("")

        blocker = _classify_blocker(candidate_dir, meta)
        lines.append(f"**Classification:** {blocker['classification']}")
        lines.append("")
        lines.append("**Details:**")
        lines.append("")
        lines.append(blocker['details'])
        lines.append("")

        # Recommendations
        lines.append("### Recommendations")
        lines.append("")
        for rec in blocker['recommendations']:
            lines.append(f"- {rec}")
        lines.append("")

    # Reproduction commands
    lines.append("## Reproduction Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# Re-run discovery for this candidate")
    lines.append(f"python -m rank1_discovery_mine run --one {candidate.slug} --execute")
    lines.append("")
    lines.append("# View status")
    lines.append("python -m rank1_discovery_mine status")
    lines.append("```")
    lines.append("")

    # Write report
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    return report_path


def _classify_blocker(
    candidate_dir: Path,
    meta: Dict,
) -> Dict[str, Any]:
    """
    Classify why a candidate is blocked.

    Returns dict with: classification, details, recommendations
    """
    discovered_urls = meta.get("discovered_urls", [])
    downloaded_files = meta.get("downloaded_files", [])

    raw_dir = candidate_dir / "raw"
    extracted_dir = candidate_dir / "extracted"

    has_pdfs = list(raw_dir.glob("*.pdf")) if raw_dir.exists() else []
    has_csvs = list(extracted_dir.glob("*.csv")) if extracted_dir.exists() else []

    # Determine classification
    if not discovered_urls:
        return {
            "classification": "NO_DATA_PUBLIC",
            "details": (
                "No public data sources were found for this candidate. "
                "The data may not be published, or may be in a format not "
                "covered by the search."
            ),
            "recommendations": [
                "Check if the collaboration has released supplementary material",
                "Search HEPData directly for related papers",
                "Contact the collaboration to request data tables",
                "Consider manual PDF digitization if figures are available",
            ],
        }

    if has_pdfs and not has_csvs:
        return {
            "classification": "PDF_ONLY_NEEDS_EXTRACTION",
            "details": (
                f"Found {len(has_pdfs)} PDF file(s) but could not extract "
                "tabular data. The data may be in figure form only."
            ),
            "recommendations": [
                "Use WebPlotDigitizer to manually extract data from figures",
                "Check if tables are present but poorly formatted",
                "Look for supplementary data files from the paper",
            ],
        }

    if discovered_urls and not downloaded_files:
        return {
            "classification": "DOWNLOAD_FAILED",
            "details": (
                f"Found {len(discovered_urls)} potential data source(s) but "
                "download failed. This may be a network or permission issue."
            ),
            "recommendations": [
                "Check network connectivity",
                "Verify URLs are still valid",
                "Try manual download and place files in raw/ directory",
            ],
        }

    if has_csvs:
        return {
            "classification": "TABLE_FOUND_NEEDS_MAPPING",
            "details": (
                f"Extracted {len(has_csvs)} CSV file(s) but they could not "
                "be mapped to the rank-1 harness format."
            ),
            "recommendations": [
                "Review extracted CSV columns manually",
                "Create custom adapter for this data format",
                "Verify extracted data contains required information",
            ],
        }

    return {
        "classification": "UNKNOWN",
        "details": "Unable to determine specific blocker reason.",
        "recommendations": [
            "Review logs for error messages",
            "Manually investigate data availability",
        ],
    }


def write_email_request(
    candidate_dir: Path,
    candidate: "Candidate",
) -> Path:
    """
    Generate EMAIL_REQUEST.txt with a template for requesting data
    from the collaboration.
    """
    candidate_dir = Path(candidate_dir)
    out_dir = candidate_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    email_path = out_dir / "EMAIL_REQUEST.txt"

    collab = candidate.collaboration or "[Collaboration Name]"

    content = f"""Subject: Request for published data tables - {candidate.title}

Dear {collab} Collaboration,

I am conducting a systematic study of exotic hadron candidates using
rank-1 factorization tests, which probe whether observed resonances
are consistent with having a single underlying pole structure.

For the analysis of {', '.join(candidate.states)}, I would greatly
appreciate access to the following data:

REQUESTED DATA:
1. Binned invariant mass spectrum (or efficiency-corrected yields)
   - Channels: {', '.join(ch.label for ch in candidate.channels)}
   - Format: CSV or any tabular format
   - Required columns: mass bin centers/edges, yields, statistical errors

2. If available:
   - Covariance matrix for statistical uncertainties
   - Systematic uncertainty breakdown
   - Efficiency corrections applied

DATA FORMAT:
The ideal format is a CSV file with columns:
  mass_GeV, counts, stat_err

Or for cross-section measurements:
  sqrt_s_GeV, sigma_pb, stat_err, sys_err

NOTES:
{candidate.notes}

If this data is already publicly available (e.g., on HEPData or as
supplementary material), I would appreciate a pointer to the location.

Thank you for your consideration.

Best regards,
[Your Name]
[Your Institution]

---
Generated by Rank1 Discovery Mine
Candidate: {candidate.slug}
Date: {datetime.now(timezone.utc).isoformat()}
"""

    with open(email_path, 'w') as f:
        f.write(content)

    return email_path


def write_rank1_result(
    candidate_dir: Path,
    candidate: "Candidate",
    results: Dict[str, Any],
) -> Path:
    """
    Write RANK1_RESULT.md with the test verdict.
    """
    candidate_dir = Path(candidate_dir)
    out_dir = candidate_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "RANK1_RESULT.md"

    verdict = results.get("verdict", "UNKNOWN")
    p_boot = results.get("p_boot", "N/A")
    lambda_val = results.get("Lambda", "N/A")

    # Emoji based on verdict
    emoji = {
        "NOT_REJECTED": "✅",
        "DISFAVORED": "❌",
        "INCONCLUSIVE": "⚠️",
        "MODEL_MISMATCH": "🔧",
        "OPTIMIZER_FAILURE": "💥",
    }.get(verdict, "❓")

    content = f"""# Rank-1 Test Result: {candidate.title}

## Verdict: {emoji} {verdict}

| Metric | Value |
|--------|-------|
| Bootstrap p-value | {p_boot} |
| Test statistic Λ | {lambda_val} |
| Wilks p-value (ref) | {results.get('p_wilks', 'N/A')} |

## Interpretation

"""

    if verdict == "NOT_REJECTED":
        content += (
            "The rank-1 hypothesis is **not rejected** at the 5% significance level. "
            "The data are consistent with a shared complex coupling ratio R across "
            "the tested channels."
        )
    elif verdict == "DISFAVORED":
        content += (
            "The rank-1 hypothesis is **disfavored** (p < 0.05). "
            "There is evidence that the coupling ratio R differs between channels, "
            "which may indicate multiple underlying pole structures."
        )
    elif verdict == "INCONCLUSIVE":
        content += (
            "The test is **inconclusive** due to fit quality issues or insufficient data. "
            "Additional data or model refinements may be needed."
        )
    else:
        content += f"Verdict: {verdict}"

    content += f"""

## Details

- States tested: {', '.join(candidate.states)}
- Channels: {', '.join(ch.label for ch in candidate.channels)}
- Rank-1 mode: {candidate.rank1_mode}

---
Generated: {datetime.now(timezone.utc).isoformat()}
"""

    with open(result_path, 'w') as f:
        f.write(content)

    return result_path

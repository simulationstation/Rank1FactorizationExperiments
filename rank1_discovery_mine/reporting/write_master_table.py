"""
Master table generation module.

Maintains the global summary table of all candidate attempts.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List


MASTER_TABLE_COLUMNS = [
    "slug",
    "title",
    "status",
    "data_sources_found",
    "channels_extracted",
    "p_value",
    "verdict",
    "blocker_reason",
    "timestamp",
]


def initialize_master_table(registry_path: Path) -> Path:
    """
    Initialize empty master table files.

    Returns path to CSV file.
    """
    registry_path = Path(registry_path)
    registry_path.mkdir(parents=True, exist_ok=True)

    csv_path = registry_path / "MASTER_TABLE.csv"
    md_path = registry_path / "MASTER_TABLE.md"

    # Create CSV with header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_TABLE_COLUMNS)
        writer.writeheader()

    # Create markdown header
    _write_master_md(md_path, [])

    return csv_path


def append_row(
    registry_path: Path,
    row: Dict[str, Any],
) -> None:
    """
    Append a row to the master table.
    """
    registry_path = Path(registry_path)
    csv_path = registry_path / "MASTER_TABLE.csv"
    md_path = registry_path / "MASTER_TABLE.md"

    # Ensure all columns present
    for col in MASTER_TABLE_COLUMNS:
        if col not in row:
            row[col] = ""

    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Append to CSV
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_TABLE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Regenerate markdown
    _regenerate_master_md(csv_path, md_path)


def _regenerate_master_md(csv_path: Path, md_path: Path) -> None:
    """
    Regenerate MASTER_TABLE.md from CSV.
    """
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    _write_master_md(md_path, rows)


def _write_master_md(md_path: Path, rows: List[Dict]) -> None:
    """
    Write MASTER_TABLE.md with current data.
    """
    lines = []

    lines.append("# Rank1 Discovery Mine - Master Results Table")
    lines.append("")
    lines.append(f"Last updated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Summary statistics
    if rows:
        total = len(rows)
        done = sum(1 for r in rows if r.get("status") == "DONE")
        no_data = sum(1 for r in rows if r.get("status") == "NO_DATA")
        blocked = sum(1 for r in rows if r.get("status") == "BLOCKED")
        errors = sum(1 for r in rows if r.get("status") == "ERROR")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total candidates: {total}")
        lines.append(f"- Completed (DONE): {done}")
        lines.append(f"- No public data (NO_DATA): {no_data}")
        lines.append(f"- Blocked: {blocked}")
        lines.append(f"- Errors: {errors}")
        lines.append("")

    # Table
    lines.append("## Results")
    lines.append("")

    if rows:
        # Header
        lines.append("| " + " | ".join(MASTER_TABLE_COLUMNS) + " |")
        lines.append("| " + " | ".join(["---"] * len(MASTER_TABLE_COLUMNS)) + " |")

        # Rows
        for row in rows:
            values = [str(row.get(col, ""))[:30] for col in MASTER_TABLE_COLUMNS]
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("*No candidates processed yet.*")

    lines.append("")

    # Legend
    lines.append("## Status Legend")
    lines.append("")
    lines.append("| Status | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| PENDING | Not yet processed |")
    lines.append("| PLANNED | Dry-run completed, ready for execution |")
    lines.append("| IN_PROGRESS | Currently being processed |")
    lines.append("| DONE | Successfully completed rank-1 test |")
    lines.append("| NO_DATA | No public data available |")
    lines.append("| BLOCKED | Data found but extraction/test failed |")
    lines.append("| ERROR | Processing error occurred |")
    lines.append("")

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))


def generate_summary_stats(registry_path: Path) -> Dict[str, Any]:
    """
    Generate summary statistics from master table.
    """
    csv_path = Path(registry_path) / "MASTER_TABLE.csv"

    if not csv_path.exists():
        return {"total": 0, "message": "No data yet"}

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"total": 0, "message": "No data yet"}

    # Count by status
    status_counts = {}
    for row in rows:
        status = row.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Count verdicts
    verdict_counts = {}
    for row in rows:
        verdict = row.get("verdict", "")
        if verdict:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    # Blocker breakdown
    blocker_counts = {}
    for row in rows:
        blocker = row.get("blocker_reason", "")
        if blocker:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

    return {
        "total": len(rows),
        "status_counts": status_counts,
        "verdict_counts": verdict_counts,
        "blocker_counts": blocker_counts,
    }
